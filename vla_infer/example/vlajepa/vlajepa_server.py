"""VLA-JEPA server 启动器（在控制端仓库里一键拉起 GPU 机器上的推理服务）。

为什么是"启动器"而不是"服务端实现"：
    VLA-JEPA 的 server 需要 starVLA 及其重型依赖（transformers 4.57 / flash-attn / numpy 1.26），
    而控制端 `vla_infer` 要求 `numpy>=2.0`（见 vla_infer/pyproject.toml:12）——两者**不能装在同一个环境**。
    所以服务端必须跑在 VLA-JEPA 仓库的 VLA 环境里，本脚本只负责用正确的 python、
    在正确的目录、以正确的参数把它拉起来（不做任何模型 import）。

真正的服务端实现：`<VLA-JEPA-Alex>/examples/real-robot/piper_zmq_server.py`
协议与设计依据：`<VLA-JEPA-Alex>/doc/robot/piper_server_plan.md`

用法：
    python vla_infer/example/vlajepa/vlajepa_server.py \
        --vla_repo ~/repo/VLA-JEPA-Alex \
        --ckpt_path ~/repo/VLA-JEPA-Alex/checkpoints/<run_id>/final_model/pytorch_model.pt \
        --port 5555 --use_bf16 --default-instruction "0"

    # 协议自检（不需要权重/GPU）
    python vla_infer/example/vlajepa/vlajepa_server.py --vla_repo ~/repo/VLA-JEPA-Alex --dry_run
"""

from dataclasses import dataclass
import logging
import os
import shlex
import subprocess
import sys
import typing as t
from pathlib import Path

import draccus

SERVER_REL_PATH = "examples/real-robot/piper_zmq_server.py"


@dataclass
class VLAJepaServerLauncherConfig:
    """server 启动参数（会原样透传给 piper_zmq_server.py）。"""

    # VLA-JEPA-Alex 仓库根目录（本机路径；服务端与模型都在这里）
    vla_repo: str = ""
    # VLA 环境的 python（必须已装 torch/starVLA 依赖，见 doc/robot/model.md §2.4）。
    # 留空则自动用 <vla_repo>/.venv/bin/python（本项目 uv 虚拟环境，见 AGENTS.md）。
    vla_python: str = ""

    # ── 透传给 server 的参数 ─────────────────────────────────
    ckpt_path: str = ""
    host: str = "0.0.0.0"
    port: int = 5555
    jpeg_quality: int = 80
    cuda: str = "0"
    use_bf16: bool = True
    num_inference_timesteps: int = 0
    chunk_steps: int = 0
    unnorm_key: str = ""
    # 必须与该 ckpt 训练时的语言条件一致：dataloader 把 task_index 用 meta/tasks.jsonl
    # 解析成字符串（datasets.py:1027），所以要填任务原文。当前 iclr_adjust_cup 为下面这句。
    default_instruction: str = "Put the cup the right way up on the table."
    no_binarize_gripper: bool = False
    dry_run: bool = False
    log_level: str = "INFO"
    # 额外参数（原样追加，例如 "--no-warmup --max-requests 5"）
    extra_args: str = ""

    log_level_launcher: str = "INFO"


def build_command(cfg: VLAJepaServerLauncherConfig) -> t.List[str]:
    repo = Path(os.path.expanduser(cfg.vla_repo)).resolve()
    script = repo / SERVER_REL_PATH
    if not script.is_file():
        raise SystemExit(
            f"找不到服务端脚本：{script}\n"
            f"请用 --vla_repo 指向 VLA-JEPA-Alex 仓库根目录（当前 {cfg.vla_repo!r}）"
        )

    vla_python = cfg.vla_python or str(repo / ".venv" / "bin" / "python")
    cmd: t.List[str] = [os.path.expanduser(vla_python), str(script)]
    cmd += ["--host", cfg.host, "--port", str(cfg.port), "--jpeg-quality", str(cfg.jpeg_quality)]
    cmd += ["--log-level", cfg.log_level]
    if cfg.dry_run:
        cmd += ["--dry-run"]
    else:
        if not cfg.ckpt_path:
            raise SystemExit("真机运行必须给 --ckpt_path（或加 --dry_run 做协议自检）")
        cmd += ["--ckpt_path", os.path.expanduser(cfg.ckpt_path), "--cuda", str(cfg.cuda)]
        if cfg.use_bf16:
            cmd += ["--use-bf16"]
    if cfg.num_inference_timesteps:
        cmd += ["--num-inference-timesteps", str(cfg.num_inference_timesteps)]
    if cfg.chunk_steps:
        cmd += ["--chunk-steps", str(cfg.chunk_steps)]
    if cfg.unnorm_key:
        cmd += ["--unnorm-key", cfg.unnorm_key]
    if cfg.default_instruction:
        cmd += ["--default-instruction", cfg.default_instruction]
    if cfg.no_binarize_gripper:
        cmd += ["--no-binarize-gripper"]
    if cfg.extra_args:
        cmd += shlex.split(cfg.extra_args)
    return cmd


@draccus.wrap()
def main(cfg: VLAJepaServerLauncherConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level_launcher.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    cmd = build_command(cfg)
    repo = Path(os.path.expanduser(cfg.vla_repo)).resolve()

    logging.info("启动 VLA-JEPA server（工作目录=%s）", repo)
    logging.info("命令：%s", " ".join(shlex.quote(part) for part in cmd))
    logging.info(
        "提醒：控制端请用 state_type=joint + absolute_action=True + enable_gripper_transform=False "
        "+ execute_chunk_steps=7（见 vlajepa_piper_client.py 的默认值）"
    )
    try:
        completed = subprocess.run(cmd, cwd=str(repo))
    except FileNotFoundError as exc:
        raise SystemExit(
            f"启动失败：{exc}\n"
            "请确认 --vla_python（留空时默认 <vla_repo>/.venv/bin/python）指向已装 torch/starVLA 的环境"
        ) from exc
    except KeyboardInterrupt:
        logging.info("收到 Ctrl-C，server 已停止")
        return
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    sys.exit(main())
