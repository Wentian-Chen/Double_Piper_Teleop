"""离线数据集回放：用真实 VLA-JEPA Piper 客户端跑一整条 episode（不接机械臂）。

与 `vlajepa_piper_client.py` 的区别只在 **robot 来源**：
    - `vlajepa_piper_client.py`  : robot = PiperSingleRobot（真机）
    - 本脚本                     : robot = DatasetPiperRobot（LeRobot 数据集回放）

因此走的是完全相同的客户端代码路径（`VLAJepaPiperClient` → `VlaZmqClient` →
`VLAProtocol` → ZMQ），只是把"读机械臂/相机"换成"读数据集帧"、"下发动作"换成
"记录动作"。用于在没有硬件时验证：

    客户端预处理/协议/服务端推理 → 预测动作 chunk → 与数据集真值比对（MSE）→ 轨迹图

运行环境：本仓库控制端 python（需要 pyarrow / av / cv2 / matplotlib / numpy）。
服务端另起进程（VLA-JEPA-Alex 的 .venv）：

    # server（VLA 仓库）
    cd <VLA-JEPA-Alex>
    .venv/bin/python examples/real-robot/piper_zmq_server.py \
        --ckpt_path checkpoints/iclr_adjust_cup/final_model/pytorch_model.pt \
        --host 127.0.0.1 --port 15570 --use-bf16 --no-binarize-gripper \
        --default-instruction "Put the cup the right way up on the table."

    # client（本仓库）
    cd <Double_Piper_Teleop>
    python vla_infer/example/vlajepa/vlajepa_dataset_replay.py \
        --dataset_root <dataset> --episode 0 \
        --output_dir <VLA-JEPA-Alex>/eval_openloop/iclr_adjust_cup_vlajepa_client_replay_ep0 \
        --port 15570 --execute_chunk_steps 7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import typing as t

import numpy as np

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]  # …/Double_Piper_Teleop
sys.path.insert(0, str(REPO_ROOT))

from vla_infer.example.vlajepa.vlajepa_piper_client import (  # noqa: E402
    VLAJepaInferenceConfig,
    VLAJepaPiperClient,
)
from vla_infer.src.zmq.zmq_client import VlaZmqClient  # noqa: E402

DIM_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
EXPECTED_DIM = len(DIM_NAMES)
DEFAULT_INSTRUCTION = "Put the cup the right way up on the table."


# --------------------------------------------------------------------------- #
# 数据集读取（LeRobot v2.1：parquet 存 state/action，mp4 存图像）
# --------------------------------------------------------------------------- #
def read_episode(
    dataset_root: Path,
    episode_index: int,
    chunk_size: int = 100,
) -> t.Dict[str, np.ndarray]:
    """读一条 episode 的 state/action 与两路相机帧（RGB HWC3 uint8）。"""
    import av
    import pyarrow.parquet as pq

    episode_chunk = episode_index // chunk_size
    parquet_path = (
        dataset_root / "data" / f"chunk_{episode_chunk:05d}" / f"episode_{episode_index:06d}.parquet"
    )
    if not parquet_path.is_file():
        raise FileNotFoundError(f"episode parquet 不存在：{parquet_path}")

    table = pq.read_table(parquet_path)
    states = np.stack(table.column("observation.state").to_pylist()).astype(np.float32)
    actions = np.stack(table.column("action").to_pylist()).astype(np.float32)

    images: t.Dict[str, np.ndarray] = {}
    for key in ("observation.images.image", "observation.images.wrist_image"):
        video_path = (
            dataset_root / "videos" / key / f"chunk_{episode_chunk:05d}" / f"episode_{episode_index:06d}.mp4"
        )
        if not video_path.is_file():
            raise FileNotFoundError(f"episode 视频不存在：{video_path}")
        container = av.open(str(video_path))
        try:
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        finally:
            container.close()
        images[key] = np.stack(frames).astype(np.uint8)

    length = min(len(states), len(actions), len(images["observation.images.image"]), len(images["observation.images.wrist_image"]))
    return {
        "state": states[:length],
        "action": actions[:length],
        "cam_head": images["observation.images.image"][:length],
        "cam_wrist": images["observation.images.wrist_image"][:length],
    }


class DatasetPiperRobot:
    """把一条 episode 伪装成 PiperSingleRobot 的观测/执行接口。

    `get_observation()` 返回 `{"state", "cam_head", "cam_wrist"}`，
    `apply_action()` 不驱动硬件，只记录动作并把回放指针前移一帧。
    """

    def __init__(self, episode: t.Dict[str, np.ndarray]) -> None:
        self._state = episode["state"]
        self._cam_head = episode["cam_head"]
        self._cam_wrist = episode["cam_wrist"]
        self.length = len(self._state)
        self.index = 0
        self.applied: t.List[t.Tuple[int, np.ndarray]] = []

    def reset(self) -> None:
        self.index = 0
        self.applied = []

    def get_observation(self) -> t.Dict[str, np.ndarray]:
        i = min(self.index, self.length - 1)
        return {
            "state": self._state[i].copy(),
            "cam_head": self._cam_head[i].copy(),
            "cam_wrist": self._cam_wrist[i].copy(),
        }

    def apply_action(self, action_dict: t.Dict[str, np.ndarray]) -> None:
        action = np.asarray(action_dict["action"], dtype=np.float32).reshape(-1)
        self.applied.append((self.index, action.copy()))
        self.index += 1


class RecordingPiperClient(VLAJepaPiperClient):
    """原样跑客户端循环，额外把每次的完整预测 chunk 与执行结果记下来。"""

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        self.records: t.List[t.Dict[str, t.Any]] = []
        super().__init__(*args, **kwargs)

    def run_once(self) -> t.Dict[str, t.Any]:
        observation = self.get_observation()
        obs_index = int(self.robot.index)
        response = self.get_response(observation)
        chunk = np.asarray(response["action"], dtype=np.float64).copy()
        execution = self.execute(response)
        self.records.append(
            {
                "obs_index": obs_index,
                "chunk": chunk,
                "executed_steps": int(execution["executed_steps"]),
                "executed_action": np.asarray(execution["output_action"], dtype=np.float64).copy(),
            }
        )
        return {"action": response, "execution": execution, "observation": observation}


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def load_action_stats(stats_json: Path) -> t.Tuple[np.ndarray, np.ndarray]:
    """取动作反归一化统计量（框架用 q01/q99，见 base_framework.unnormalize_actions）。"""
    data = json.loads(stats_json.read_text(encoding="utf-8"))
    if "new_embodiment" in data:
        data = data["new_embodiment"]
    stats = data["action"]
    if "q01" in stats and "q99" in stats:
        lo, hi = stats["q01"], stats["q99"]
    elif "min" in stats and "max" in stats:
        lo, hi = stats["min"], stats["max"]
    else:
        raise KeyError(f"action 统计量缺少 q01/q99 或 min/max：{sorted(stats)}")
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if lo.shape != (EXPECTED_DIM,) or hi.shape != (EXPECTED_DIM,):
        raise ValueError(f"统计量维度应为 {EXPECTED_DIM}，实际 {lo.shape}/{hi.shape}")
    return lo, hi


def normalize_minmax(raw: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return 2.0 * (np.asarray(raw, dtype=np.float64) - lo) / np.maximum(hi - lo, 1e-8) - 1.0


def compute_metrics(pred: np.ndarray, gt: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> t.Dict[str, t.Any]:
    error = pred - gt
    norm_error = normalize_minmax(pred, lo, hi) - normalize_minmax(gt, lo, hi)
    return {
        "num_points": int(pred.shape[0]),
        "raw": {
            "mse": float((error**2).mean()),
            "rmse": float(np.sqrt((error**2).mean())),
            "mae": float(np.abs(error).mean()),
            "mse_per_dim": (error**2).mean(axis=0).tolist(),
            "rmse_per_dim": np.sqrt((error**2).mean(axis=0)).tolist(),
            "mae_per_dim": np.abs(error).mean(axis=0).tolist(),
        },
        "normalized_q01q99": {
            "mse": float((norm_error**2).mean()),
            "mae": float(np.abs(norm_error).mean()),
            "mse_per_dim": (norm_error**2).mean(axis=0).tolist(),
            "mae_per_dim": np.abs(norm_error).mean(axis=0).tolist(),
        },
    }


def collect_arrays(
    records: t.List[t.Dict[str, t.Any]],
    actions: np.ndarray,
) -> t.Dict[str, np.ndarray]:
    """两种对齐：

    - ``executed``：客户端真正"下发"的轨迹（chunk-step 模式 = 每 7 步重规划一次）。
    - ``chunk``   ：每个观测点的完整预测 chunk 对齐到未来 7 帧。
    """
    n = len(actions)
    executed_pred = np.full((n, EXPECTED_DIM), np.nan, dtype=np.float64)
    for record in records:
        obs = record["obs_index"]
        for j in range(record["executed_steps"]):
            frame = obs + j
            if frame < n:
                executed_pred[frame] = record["chunk"][j]

    chunk_preds: t.List[np.ndarray] = []
    chunk_gts: t.List[np.ndarray] = []
    chunk_frames: t.List[t.Tuple[int, int]] = []
    for record in records:
        obs = record["obs_index"]
        for j in range(record["chunk"].shape[0]):
            frame = obs + j
            if frame < n:
                chunk_preds.append(record["chunk"][j])
                chunk_gts.append(actions[frame])
                chunk_frames.append((obs, j))

    valid = ~np.isnan(executed_pred).any(axis=1)
    return {
        "executed_pred": executed_pred,
        "executed_gt": actions.astype(np.float64),
        "executed_valid": valid,
        "chunk_pred": np.stack(chunk_preds) if chunk_preds else np.zeros((0, EXPECTED_DIM)),
        "chunk_gt": np.stack(chunk_gts) if chunk_gts else np.zeros((0, EXPECTED_DIM)),
    }


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #
def plot_trajectories(
    output_dir: Path,
    executed_pred: np.ndarray,
    executed_gt: np.ndarray,
    valid: np.ndarray,
    prefix: str,
) -> t.List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = np.arange(len(executed_gt))
    colors = plt.cm.tab10(np.linspace(0, 1, EXPECTED_DIM))
    figures: t.List[str] = []

    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    axes = axes.reshape(-1)
    for dim, name in enumerate(DIM_NAMES):
        ax = axes[dim]
        ax.plot(frames, executed_gt[:, dim], color="black", linewidth=2.0, label="ground truth")
        ax.plot(frames[valid], executed_pred[valid, dim], color=colors[dim], linewidth=1.6, linestyle="--", label="prediction")
        ax.set_title(name)
        ax.set_xlabel("frame")
        ax.set_ylabel("value (rad)")
        ax.grid(alpha=0.3)
        if dim == 0:
            ax.legend(fontsize=8)
    axes[-1].axis("off")
    fig.suptitle(f"VLA-JEPA Piper dataset replay — prediction vs ground truth ({prefix})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = output_dir / f"trajectory_pred_vs_gt_{prefix}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append(path.name)

    error = np.abs(executed_pred - executed_gt)
    fig, ax = plt.subplots(figsize=(12, 5))
    for dim, name in enumerate(DIM_NAMES):
        ax.plot(frames[valid], error[valid, dim], color=colors[dim], linewidth=1.2, label=name)
    ax.set_title(f"per-dim absolute error over time ({prefix})")
    ax.set_xlabel("frame")
    ax.set_ylabel("|error| (rad)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = output_dir / f"error_over_time_{prefix}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append(path.name)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(executed_gt[:, 0], executed_gt[:, 1], color="black", linewidth=2.0, label="ground truth")
    ax.plot(executed_pred[valid, 0], executed_pred[valid, 1], color="tab:red", linewidth=1.5, linestyle="--", label="prediction")
    ax.scatter(executed_gt[0, 0], executed_gt[0, 1], color="green", marker="o", s=60, label="start")
    ax.set_title(f"joint1–joint2 phase plot ({prefix})")
    ax.set_xlabel("joint1")
    ax.set_ylabel("joint2")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = output_dir / f"phase_joint1_joint2_{prefix}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append(path.name)

    return figures


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLA-JEPA Piper dataset replay (offline, no robot)")
    parser.add_argument("--dataset_root", required=True, help="LeRobot v2.1 dataset directory")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--stats_json",
        default=str(REPO_ROOT.parent / "VLA-JEPA-Alex" / "checkpoints" / "iclr_adjust_cup" / "dataset_statistics.json"),
        help="checkpoint dataset_statistics.json（用于 q01/q99 反归一化对比）",
    )
    parser.add_argument("--server_ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout_ms", type=int, default=5000)
    parser.add_argument("--jpeg_quality", type=int, default=80)
    parser.add_argument("--execute_chunk_steps", type=int, default=7, help="7=部署配置(chunk-step)，1=单步重规划")
    parser.add_argument("--task_instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max_frames", type=int, default=0, help="0 = 整条 episode")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--skip_plot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_json = Path(args.stats_json)
    if not stats_json.is_file():
        raise FileNotFoundError(f"统计量文件不存在：{stats_json}")

    episode = read_episode(Path(args.dataset_root), args.episode)
    length = len(episode["action"])
    if args.max_frames > 0:
        length = min(length, args.max_frames)
        episode = {key: value[:length] for key, value in episode.items()}
    print(f"episode {args.episode}: {length} frames, state={episode['state'].shape} action={episode['action'].shape}")

    robot = DatasetPiperRobot(episode)
    cfg = VLAJepaInferenceConfig(
        server_ip=args.server_ip,
        port=args.port,
        timeout_ms=args.timeout_ms,
        jpeg_quality=args.jpeg_quality,
        task_instruction=args.task_instruction,
        max_steps=length,
        execute_chunk_steps=args.execute_chunk_steps,
        log_level=args.log_level,
    )
    zmq_client = VlaZmqClient(
        server_ip=args.server_ip, port=args.port, timeout_ms=args.timeout_ms, jpeg_quality=args.jpeg_quality
    )
    client = RecordingPiperClient(cfg=cfg, robot=robot, client=zmq_client)

    # 客户端 run(max_steps) 数的是"循环次数"，每次循环会执行 execute_chunk_steps 个动作，
    # 因此 chunk-step 模式下观测点数是 ceil(length / steps)，不能直接用 length。
    steps = max(1, int(args.execute_chunk_steps))
    iterations = (length + steps - 1) // steps

    started = time.perf_counter()
    try:
        client.run(max_steps=iterations)
    finally:
        client.close()
    wall_s = time.perf_counter() - started

    records = client.records
    if not records:
        raise RuntimeError("客户端没有产生任何推理记录")
    observed_frames = [r["obs_index"] for r in records]
    if observed_frames != sorted(observed_frames) or len(set(observed_frames)) != len(observed_frames):
        raise RuntimeError(f"观测帧不单调或重复，回放对齐有误：{observed_frames[:8]}...")
    print(f"observations: {len(records)} at frames {observed_frames[:6]}...{observed_frames[-3:]} (steps={steps})")

    arrays = collect_arrays(records, episode["action"])
    lo, hi = load_action_stats(stats_json)
    valid = arrays["executed_valid"]
    executed_metrics = compute_metrics(
        arrays["executed_pred"][valid], arrays["executed_gt"][valid], lo, hi
    )
    chunk_metrics = compute_metrics(arrays["chunk_pred"], arrays["chunk_gt"], lo, hi)

    figures = []
    prefix = f"chunkstep{args.execute_chunk_steps}"
    if not args.skip_plot:
        figures = plot_trajectories(
            output_dir, arrays["executed_pred"], arrays["executed_gt"], valid, prefix
        )

    np.savez_compressed(
        output_dir / f"predictions_{prefix}.npz",
        executed_pred=arrays["executed_pred"],
        executed_gt=arrays["executed_gt"],
        executed_valid=valid,
        chunk_pred=arrays["chunk_pred"],
        chunk_gt=arrays["chunk_gt"],
        action_q01=lo,
        action_q99=hi,
        obs_index=np.asarray([r["obs_index"] for r in records], dtype=np.int64),
    )

    result = {
        "status": "passed",
        "dataset_root": args.dataset_root,
        "episode": args.episode,
        "num_frames": length,
        "num_observations": len(records),
        "execute_chunk_steps": args.execute_chunk_steps,
        "chunk_len": int(records[0]["chunk"].shape[0]),
        "action_dim": EXPECTED_DIM,
        "dimension_names": DIM_NAMES,
        "server": {"host": args.server_ip, "port": args.port},
        "task_instruction": args.task_instruction,
        "stats_json": str(stats_json),
        "normalization": "framework q01/q99 (base_framework.unnormalize_actions)",
        "alignment": {
            "executed": "predicted chunk step j placed at frame obs+j (deployment trajectory)",
            "chunk": "all chunk steps j aligned to frame obs+j",
        },
        "wall_seconds": wall_s,
        "metrics_executed_trajectory": executed_metrics,
        "metrics_chunk": chunk_metrics,
        "figures": figures,
        "records": [
            {
                "obs_index": r["obs_index"],
                "executed_steps": r["executed_steps"],
                "chunk_first": r["chunk"][0].tolist(),
                "chunk_last": r["chunk"][-1].tolist(),
            }
            for r in records
        ],
    }
    result_path = output_dir / f"replay_result_{prefix}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({
        "frames": length,
        "observations": len(records),
        "execute_chunk_steps": args.execute_chunk_steps,
        "executed_raw_mse": executed_metrics["raw"]["mse"],
        "executed_raw_mae": executed_metrics["raw"]["mae"],
        "chunk_raw_mse": chunk_metrics["raw"]["mse"],
        "chunk_raw_mae": chunk_metrics["raw"]["mae"],
        "wall_seconds": round(wall_s, 1),
    }, indent=2))
    print(f"written -> {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
