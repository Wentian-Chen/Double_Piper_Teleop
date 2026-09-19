"""VLA-JEPA × Piper 真机客户端（完整示例）。

骨架与 `example/dream-adapter/dream-adapter-piper_client.py` 一致（同一套
`InferenceClient` + `PiperSingleRobot` + `VlaZmqClient`），但**默认值按 VLA-JEPA 项目约定**：

    state_type="joint"          # 发 joint(6)+gripper(1)，与训练 state 语义一致
    action_type="joint"
    absolute_action=True        # 服务端返回绝对关节角 + 夹爪，原样下发
    enable_gripper_transform=False   # 关掉客户端"再下压 0.3"的逻辑
    execute_chunk_steps=7       # = 模型 chunk 长度 (future_action_window_size + 1)

依据：VLA-JEPA-Alex/doc/robot/deployment.md §4.2（真机默认配置）与
      VLA-JEPA-Alex/doc/robot/piper_server_plan.md §5.4（联合配置表）。

服务端在 GPU 机器上跑（VLA 环境）：
    cd <VLA-JEPA-Alex>
    python examples/real-robot/piper_zmq_server.py --ckpt_path <run_dir>/final_model/pytorch_model.pt \
        --port 5555 --use-bf16 --default-instruction "0"
"""

from dataclasses import dataclass
import json
import logging
import time
import typing as t
from pathlib import Path
import sys

import draccus
import numpy as np

from vla_infer.src.inference.client import InferenceClient
from vla_infer.src.robots.piper_single import PiperSingleRobot
from vla_infer.src.zmq.zmq_client import VlaZmqClient
from vla_infer.src.process.utils import (
    adaptive_resize_image,
    check_uint8_rgb,
    smooth_action_chunk,
    delta_action_chunk_to_absolute,
    interpolate_action_chunk,
)

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]          # …/Double_Piper_Teleop
sys.path.append(str(REPO_ROOT))


@dataclass
class VLAJepaInferenceConfig:
    """VLA-JEPA 真机运行配置（默认值即本项目约定）。"""

    # ── 通信 ────────────────────────────────────────────────
    server_ip: str = "127.0.0.1"      # GPU 机器 IP
    port: int = 5555                  # 与 server --port 一致
    timeout_ms: int = 5000            # 超时=急停：VLA-JEPA 单次推理 = VLM 前向 + 4 步去噪，先给宽一点
    jpeg_quality: int = 80

    # ── 任务 ────────────────────────────────────────────────
    # 必须是该 checkpoint 训练时使用的语言条件原文：dataloader 会把 parquet 里的
    # task_index 用 meta/tasks.jsonl 解析成字符串（starVLA/dataloader/gr00t_lerobot/datasets.py:1027），
    # 因此这里要填任务原文，不能填 task_index 数字。
    # 当前 iclr_adjust_cup（adjust_cup_0409_1_offset_state_v2_1）task_index=0 原文如下；
    # 换 checkpoint 时必须替换为对应数据集 meta/tasks.jsonl 的原文。
    task_instruction: str = "Put the cup the right way up on the table."
    max_steps: int = 1000
    stop_on_timeout: bool = True      # 超时即停（安全）

    # ── 动作/状态语义（勿随意修改）────────────────────────────
    state_type: str = "joint"         # joint = joint(6)+gripper(1)；qpos 会送出错误的 state
    action_type: str = "joint"
    absolute_action: bool = True      # True = 服务端给的是绝对关节角
    action_key: str = "action"

    # ── 执行 ────────────────────────────────────────────────
    execute_chunk_steps: int = 7      # = future_action_window_size + 1
    control_interval_s: float = 0.04

    # ── 夹爪 ────────────────────────────────────────────────
    enable_binary_gripper: bool = False
    binary_gripper_threshold: float = 0.4
    gripper_open_value: float = 0.5
    gripper_closed_value: float = 0.2
    enable_gripper_transform: bool = False   # 默认关闭：0/1 指令不再被下压成 -0.3
    gripper_transform_threshold: float = 0.55
    gripper_transform_delta: float = 0.3

    # ── 动作后处理（默认关，保持与训练动作分布一致）──────────────
    use_smoothing: bool = False
    enable_action_interpolation: bool = False
    interpolation_method: str = "linear"
    interpolation_target_steps: int = 0

    # ── 调试 ────────────────────────────────────────────────
    show_output_track: bool = False
    track_file: str = "tem.json"
    log_level: str = "INFO"


class VLAJepaPiperClient(InferenceClient):
    """把 PiperSingleRobot 与 VLA-JEPA ZMQ server 接起来的最小客户端。

    `get_observation()` 返回的观测契约（服务端按此解析）::

        {
          "state":       np.ndarray(7,) float32,   # 6 关节角 + 夹爪，未归一化
          "image":       np.ndarray(H, W, 3) uint8 # 头部相机
          "wrist_image": np.ndarray(H, W, 3) uint8 # 腕部相机
        }
    服务端返回 ``{"action": np.ndarray(T, 7) float32}``（绝对关节角 + 夹爪）。
    """

    def __init__(
        self,
        cfg: VLAJepaInferenceConfig,
        robot: t.Optional[PiperSingleRobot] = None,
        client: t.Optional[VlaZmqClient] = None,
    ) -> None:
        self.cfg = cfg
        logging.basicConfig(
            level=getattr(logging, cfg.log_level.upper(), logging.INFO),
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self._warn_if_non_default_config()

        self.robot = robot if robot is not None else PiperSingleRobot()
        time.sleep(2)
        self.robot.reset()
        time.sleep(2)
        self.zmq_client = (
            client
            if client is not None
            else VlaZmqClient(
                server_ip=cfg.server_ip,
                port=cfg.port,
                timeout_ms=cfg.timeout_ms,
                jpeg_quality=cfg.jpeg_quality,  # 否则该字段是死配置，请求图像始终按库默认 80 编码
            )
        )
        self.obs: t.Dict[str, t.Any] = {}

    # ------------------------------------------------------------------ #
    def _warn_if_non_default_config(self) -> None:
        if self.cfg.state_type != "joint":
            logging.warning(
                "state_type=%r 偏离本项目默认 'joint'：state 前 6 维将不是关节角，模型输入分布会错",
                self.cfg.state_type,
            )
        if not self.cfg.absolute_action:
            logging.warning("absolute_action=False：服务端的绝对关节角会被当成增量做 cumsum 累加")
        if self.cfg.enable_gripper_transform:
            logging.warning(
                "enable_gripper_transform=True：夹爪指令 < %.2f 时会被再减 %.2f，本项目默认关闭",
                self.cfg.gripper_transform_threshold,
                self.cfg.gripper_transform_delta,
            )
        if self.cfg.execute_chunk_steps > 7:
            logging.warning(
                "execute_chunk_steps=%d 大于模型 chunk 长度(7)，实际只会执行 7 步",
                self.cfg.execute_chunk_steps,
            )

    # ------------------------------------------------------------------ #
    def get_observation(self) -> t.Dict[str, t.Any]:
        """采集一帧观测：关节角 + 夹爪 + 双相机（保证 HWC3 uint8、224 letterbox）。"""
        raw_obs = self.robot.get_observation()
        state = np.asarray(raw_obs.get("state", np.zeros(7, dtype=np.float32)), dtype=np.float32).reshape(-1)
        if state.shape[0] != 7:
            raise ValueError(f"Piper state 应为 7 维(joint6+gripper)，实际 {state.shape}")

        obs: t.Dict[str, t.Any] = {
            "state": state,
            "image": check_uint8_rgb(adaptive_resize_image(raw_obs.get("cam_head"))),
            "wrist_image": check_uint8_rgb(adaptive_resize_image(raw_obs.get("cam_wrist"))),
        }
        if self.cfg.enable_binary_gripper:
            obs["state"][-1] = 1.0 if obs["state"][-1] > self.cfg.binary_gripper_threshold else 0.0

        # joint_state 只供 delta 分支内部使用，不能混进发往 server 的观测里
        # （server 只读 state/image/wrist_image/cmd，多余键虽被忽略但会白白传输）。
        self.obs = dict(obs)
        self.obs["joint_state"] = state
        return obs

    def get_response(
        self,
        observation: t.Dict[str, t.Any],
        task_instruction: t.Optional[str] = None,
    ) -> t.Dict[str, t.Any]:
        """发观测、取动作块（只读响应里的 `action` 键）。"""
        for key, value in observation.items():
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                logging.debug("observation '%s' shape=%s dtype=%s", key, value.shape, value.dtype)

        observation["cmd"] = task_instruction or self.cfg.task_instruction
        response = self.zmq_client.get_response(obs_dict=observation)
        action = response.get(self.cfg.action_key)
        if action is None:
            raise KeyError(f"服务端响应缺少 '{self.cfg.action_key}' 键，实际键：{sorted(response)}")
        return {"action": action}

    def execute(self, response: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """把动作块下发到机械臂（默认：绝对关节角，直接执行）。"""
        action = np.array(response["action"], dtype=np.float32, copy=True)
        if action.ndim == 1:
            action = action[None, :]
        if action.ndim != 2 or action.shape[1] < 7:
            raise ValueError(f"action 形状应为 (T, >=7)，实际 {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("服务端返回的 action 含 NaN/Inf，拒绝下发（真机前最后一道校验）")

        if self.cfg.enable_binary_gripper:
            action[:, -1] = np.where(
                action[:, -1] > self.cfg.binary_gripper_threshold,
                self.cfg.gripper_open_value,
                self.cfg.gripper_closed_value,
            )
            # 二值夹爪只改第 7 维；前 6 维仍按 absolute_action 解释，
            # 不能因为开了二值夹爪就把绝对关节角当增量 cumsum。
            if self.cfg.absolute_action:
                abs_action = action
            else:
                if self.cfg.state_type == "qpos":
                    abs_action = delta_action_chunk_to_absolute(self.obs.get("joint_state")[:6], action[:, :6])
                else:
                    abs_action = delta_action_chunk_to_absolute(self.obs.get("state")[:6], action[:, :6])
                abs_action = np.concatenate([abs_action, action[:, -1:]], axis=-1)
        elif self.cfg.absolute_action:
            abs_action = action
        elif self.cfg.state_type == "qpos":
            abs_action = delta_action_chunk_to_absolute(self.obs.get("joint_state"), action)
        else:
            abs_action = delta_action_chunk_to_absolute(self.obs.get("state"), action)

        if self.cfg.use_smoothing:
            abs_action = smooth_action_chunk(abs_action, max_angular_acceleration=0.01, max_angular_jerk=0.01)

        if self.cfg.enable_gripper_transform:
            abs_action[:, -1] = abs_action[:, -1] - self.cfg.gripper_transform_delta * (
                action[:, -1] < self.cfg.gripper_transform_threshold
            )

        if self.cfg.enable_action_interpolation:
            if self.cfg.interpolation_target_steps <= 0:
                raise ValueError("enable_action_interpolation=True 时 interpolation_target_steps 必须 > 0")
            abs_action = interpolate_action_chunk(
                abs_action,
                target_steps=self.cfg.interpolation_target_steps,
                method=self.cfg.interpolation_method,
            )

        execute_steps = min(max(1, self.cfg.execute_chunk_steps), abs_action.shape[0])
        for idx in range(execute_steps):
            self.robot.apply_action({"action": abs_action[idx]})
            if self.cfg.control_interval_s > 0:
                time.sleep(self.cfg.control_interval_s)

        return {
            "executed_steps": execute_steps,
            "output_action": abs_action[:execute_steps],
            "action_shape": tuple(abs_action.shape),
        }

    # ------------------------------------------------------------------ #
    def run_once(self) -> t.Dict[str, t.Any]:
        """一轮：观测 → 请求 → 执行。"""
        observation = self.get_observation()
        response = self.get_response(observation)
        execution = self.execute(response)
        return {"action": response, "execution": execution, "observation": observation}

    def run(self, max_steps: t.Optional[int] = None) -> None:
        """连续控制循环；超时按 `stop_on_timeout` 停机。"""
        step_limit = self.cfg.max_steps if max_steps is None else max_steps
        if step_limit <= 0:
            raise ValueError("max_steps 必须 > 0")

        logging.info(
            "VLA-JEPA Piper client loop start | server=%s:%s state_type=%s absolute_action=%s "
            "chunk_steps=%d timeout=%dms instruction=%r",
            self.cfg.server_ip,
            self.cfg.port,
            self.cfg.state_type,
            self.cfg.absolute_action,
            self.cfg.execute_chunk_steps,
            self.cfg.timeout_ms,
            self.cfg.task_instruction,
        )
        for step in range(step_limit):
            try:
                report = self.run_once()
                logging.info(
                    "step=%d executed=%d action_shape=%s action_first=%s",
                    step,
                    report["execution"]["executed_steps"],
                    report["execution"]["action_shape"],
                    np.round(np.asarray(report["execution"]["output_action"])[0], 3).tolist(),
                )
                if self.cfg.show_output_track:
                    self._write_track(step, report)
            except TimeoutError:
                logging.exception("server timeout at step=%s", step)
                if self.cfg.stop_on_timeout:
                    break
            except Exception:
                logging.exception("unexpected error at step=%s", step)
                raise

    def _write_track(self, step: int, report: t.Dict[str, t.Any]) -> None:
        path = Path(self.cfg.track_file)
        if step == 0 and path.exists():
            path.unlink()
        entry = {
            "step": step,
            "output_action": np.asarray(report["execution"]["output_action"]).tolist(),
            "state": np.asarray(report["observation"]["state"]).tolist(),
        }
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"log": []}
        data["log"].append(entry)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def close(self) -> None:
        self.zmq_client.close()


@draccus.wrap()
def main(cfg: VLAJepaInferenceConfig) -> None:
    """入口：起机器人 + ZMQ 客户端，然后进入控制循环。"""
    runtime = VLAJepaPiperClient(cfg=cfg)
    try:
        runtime.run()
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
