"""
run_openloop_joint_eval.py

Open-loop evaluation for joint-space actions on Dream-Adapter style episode datasets.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import draccus
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import tqdm
from PIL import Image

# Append VLA-Adapter project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../VLA-Adapter/")))

from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla_action,
)
from experiments.robot.robot_utils import get_model, set_seed_everywhere  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Prevent TensorFlow from reserving GPU memory used by PyTorch
tf.config.set_visible_devices([], "GPU")


@dataclass
class JointEvalConfig:
    # Model parameters
    pretrained_checkpoint: Union[str, Path] = (
        "/home/lxx/repo/VLA-Adapter/outputs/"
        "configs+piper_pick_banana_100_resize_224_converted+b16+lr-0.0002+"
        "lora-r64+dropout-0.0--image_aug--train-0306-02--20000_chkpt"
    )
    base_model_checkpoint: Optional[Union[str, Path]] = None
    model_family: str = "openvla"

    # Dataset parameters (Dream-Adapter converted format)
    dataset_path: Union[str, Path] = (
        "/home/lxx/repo/datasets/dream-adapter/miku112/"
        "piper_pick_banana_100_resize_224_converted"
    )
    max_episodes: Optional[int] = None
    action_key: str = "action"
    proprio_dim: int = 7

    # Runtime/model switches (aligned with existing eval script)
    use_l1_regression: bool = True
    use_minivlm: bool = True
    use_pro_version: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 1
    unnorm_key: str = "piper_pick_banana_100_resize_224_converted"
    save_version: str = "vla-adapter"

    # Output
    plots_dir: Union[str, Path] = "openloop_eval_plots_joint"
    plot_first_n_episodes: int = 5


def initialize_model(cfg: JointEvalConfig):
    """Initialize model and optional heads/projectors."""
    if cfg.base_model_checkpoint and os.path.exists(cfg.base_model_checkpoint):
        logger.info("Detected base model checkpoint: %s", cfg.base_model_checkpoint)
        try:
            from peft import PeftModel
            
            from transformers import (
                AutoConfig,
                AutoImageProcessor,
                AutoModelForVision2Seq,
                AutoProcessor,
            )

            from experiments.robot.openvla_utils import (
                OpenVLAConfig,
                OpenVLAForActionPrediction,
                PrismaticImageProcessor,
                PrismaticProcessor,
            )

            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

            logger.info("Loading base VLA model...")
            model = AutoModelForVision2Seq.from_pretrained(
                cfg.base_model_checkpoint,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            lora_path = os.path.join(cfg.pretrained_checkpoint, "lora_adapter")
            if os.path.exists(lora_path):
                logger.info("Loading LoRA adapter from %s", lora_path)
                model = PeftModel.from_pretrained(model, lora_path)
                model = model.merge_and_unload()
            elif os.path.exists(os.path.join(cfg.pretrained_checkpoint, "adapter_config.json")):
                logger.info("Loading LoRA adapter from %s", cfg.pretrained_checkpoint)
                model = PeftModel.from_pretrained(model, cfg.pretrained_checkpoint)
                model = model.merge_and_unload()
            else:
                logger.warning(
                    "Could not find LoRA adapter files. Proceeding with base model only."
                )
        except Exception as exc:
            logger.error("Failed to load base model + adapter: %s", exc)
            raise
    else:
        logger.info("Loading pretrained checkpoint from %s", cfg.pretrained_checkpoint)
        model = get_model(cfg)

    model.set_version(cfg.save_version)

    action_head = None
    proprio_projector = None

    try:
        llm_dim = model.config.text_config.hidden_size
    except Exception:
        llm_dim = 4096

    if cfg.use_l1_regression:
        action_head = get_action_head(cfg, llm_dim)

    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            llm_dim,
            proprio_dim=cfg.proprio_dim,
        )

    processor = get_processor(cfg) if cfg.model_family == "openvla" else None
    return model, action_head, proprio_projector, processor


def _load_stats(cfg: JointEvalConfig, model) -> None:
    checkpoint_stats = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    dataset_stats = os.path.join(cfg.dataset_path, "data_info", "dataset_statistics.json")

    stats_path = checkpoint_stats if os.path.exists(checkpoint_stats) else dataset_stats
    if not os.path.exists(stats_path):
        logger.warning("No dataset_statistics.json found in checkpoint or dataset.")
        return

    logger.info("Loading dataset statistics from %s", stats_path)
    with open(stats_path, "r", encoding="utf-8") as fp:
        stats = json.load(fp)

    if cfg.unnorm_key not in stats and stats:
        fallback = next(iter(stats.keys()))
        logger.warning("unnorm_key=%s not found, fallback to %s", cfg.unnorm_key, fallback)
        cfg.unnorm_key = fallback

    if not hasattr(model, "norm_stats"):
        model.norm_stats = stats
    else:
        model.norm_stats.update(stats)


def _list_episode_dirs(dataset_path: Path, max_episodes: Optional[int]) -> List[Path]:
    episodes_root = dataset_path / "episodes"
    if not episodes_root.exists():
        raise FileNotFoundError(f"Missing episodes directory: {episodes_root}")

    episode_dirs = [p for p in episodes_root.iterdir() if p.is_dir()]
    episode_dirs.sort(key=lambda p: p.name)

    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]
    return episode_dirs


def _decode_text(value, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _align_action_shape(pred_action: np.ndarray, gt_action: np.ndarray) -> np.ndarray:
    if pred_action.shape == gt_action.shape:
        return pred_action
    if pred_action.size == gt_action.size:
        return pred_action.reshape(gt_action.shape)
    raise ValueError(f"Cannot align prediction shape {pred_action.shape} to {gt_action.shape}")


@draccus.wrap()
def eval_openloop_joint(cfg: JointEvalConfig) -> None:
    set_seed_everywhere(0)

    logger.info("Loading model from %s", cfg.pretrained_checkpoint)
    model, action_head, proprio_projector, processor = initialize_model(cfg)
    model.eval()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if not next(model.parameters()).is_cuda and not (cfg.load_in_8bit or cfg.load_in_4bit):
        model = model.to(device)

    _load_stats(cfg, model)

    dataset_path = Path(cfg.dataset_path).expanduser().resolve()
    logger.info("Loading episodes from %s", dataset_path)
    episode_dirs = _list_episode_dirs(dataset_path, cfg.max_episodes)
    logger.info("Total episodes to evaluate: %d", len(episode_dirs))

    plots_dir = Path(cfg.plots_dir).expanduser().resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    total_mse = 0.0
    total_episodes = 0
    all_episode_mses: List[float] = []

    for ep_idx, ep_dir in enumerate(tqdm.tqdm(episode_dirs, desc="Episodes")):
        steps_root = ep_dir / "steps"
        if not steps_root.exists():
            logger.warning("Skip %s: missing steps/", ep_dir)
            continue

        step_dirs = [p for p in steps_root.iterdir() if p.is_dir()]
        step_dirs.sort(key=lambda p: p.name)

        ep_mse_sum = 0.0
        ep_steps = 0
        gt_actions = []
        pred_actions = []

        for step_dir in step_dirs:
            data_h5 = step_dir / "data.h5"
            image_primary = step_dir / "image_primary.jpg"
            image_wrist = step_dir / "image_wrist.jpg"

            if not data_h5.exists() or not image_primary.exists():
                continue

            with h5py.File(data_h5, "r") as fp:
                if cfg.action_key not in fp:
                    raise KeyError(f"Missing action key '{cfg.action_key}' in {data_h5}")

                gt_action = np.asarray(fp[cfg.action_key][()], dtype=np.float32)
                task_label = _decode_text(fp.get("language_instruction", None), "do something")

                proprio = None
                if cfg.use_proprio and "observation" in fp and "proprio" in fp["observation"]:
                    proprio = np.asarray(fp["observation"]["proprio"][()], dtype=np.float32)

            full_image = np.asarray(Image.open(image_primary).convert("RGB"))
            input_obs = {"full_image": full_image, "state": proprio}
            if image_wrist.exists():
                wrist_img = np.asarray(Image.open(image_wrist).convert("RGB"))
                input_obs["image_wrist"] = wrist_img

            try:
                pred = get_vla_action(
                    cfg,
                    model,
                    processor,
                    input_obs,
                    task_label,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    use_minivlm=cfg.use_minivlm,
                )
                pred_action = np.asarray(pred[0], dtype=np.float32)
                pred_action = _align_action_shape(pred_action, gt_action)
            except Exception as exc:
                logger.error("Prediction failed at %s: %s", step_dir, exc)
                break

            gt_actions.append(gt_action)
            pred_actions.append(pred_action)

            mse = float(np.mean((pred_action - gt_action) ** 2))
            ep_mse_sum += mse
            ep_steps += 1

        if ep_steps == 0:
            continue

        ep_avg_mse = ep_mse_sum / ep_steps
        total_mse += ep_avg_mse
        total_episodes += 1
        all_episode_mses.append(ep_avg_mse)

        if ep_idx % 10 == 0:
            logger.info("Episode %s: joint MSE=%.6f", ep_dir.name, ep_avg_mse)

        if ep_idx < cfg.plot_first_n_episodes:
            gt_arr = np.array(gt_actions)
            pred_arr = np.array(pred_actions)
            dim = gt_arr.shape[1]
            dim_names = [f"Joint {i+1}" for i in range(max(0, dim - 1))] + ["Gripper"]

            fig, axes = plt.subplots(dim, 1, figsize=(10, 2 * dim))
            axes = np.atleast_1d(axes)

            for d in range(dim):
                axes[d].plot(gt_arr[:, d], label="GT", color="blue")
                axes[d].plot(pred_arr[:, d], label="Pred", color="red", linestyle="--")
                axes[d].set_ylabel(dim_names[d] if d < len(dim_names) else f"Dim {d}")
                axes[d].grid(True)
                axes[d].legend()

            plt.xlabel("Step")
            plt.suptitle(f"Episode {ep_dir.name} - Joint Action Comparison (MSE={ep_avg_mse:.4f})")
            plt.tight_layout()
            out_file = plots_dir / f"episode_{ep_dir.name}_joint_comparison.png"
            plt.savefig(out_file)
            plt.close()
            logger.info("Saved %s", out_file)

    if total_episodes == 0:
        logger.info("No valid episodes evaluated.")
        return

    final_avg_mse = total_mse / total_episodes
    logger.info("Final average joint MSE over %d episodes: %.6f", total_episodes, final_avg_mse)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(all_episode_mses)), all_episode_mses, marker="o", linestyle="-", label="Episode MSE")
    plt.xlabel("Episode")
    plt.ylabel("Average Joint MSE")
    model_name = Path(cfg.pretrained_checkpoint).name
    plt.title(f"Joint MSE per Episode ({model_name})")
    plt.grid(True)
    plt.text(
        0.95,
        0.95,
        f"Average over {total_episodes} episodes: {final_avg_mse:.6f}",
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.legend()
    out_curve = plots_dir / "episodes_joint_mse_curve.png"
    plt.savefig(out_curve)
    plt.close()
    logger.info("Saved MSE curve to %s", out_curve)


if __name__ == "__main__":
    eval_openloop_joint()  # type: ignore[call-arg]
