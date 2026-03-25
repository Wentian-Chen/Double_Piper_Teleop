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

# Append Dream-Adapter project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Dream-Adapter/")))

from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla_action,
    get_reconstruct_images,
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
    # Model parameters 需要根据实际情况修改路径和参数
    pretrained_checkpoint: Union[str, Path] = (
        "/home/charles/workspaces/Dream-adapter/outputs/configs+pick_banana_100_newTable_1_offset_state_converted+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--train-1_offset_absolute-0325-02--10000_chkpt"
    )
    base_model_checkpoint: Optional[Union[str, Path]] = None
    model_family: str = "openvla"

    # Dataset parameters (Dream-Adapter converted format) 需要根据实际情况修改路径和参数
    dataset_path: Union[str, Path] = (
        "/home/charles/workspaces/Dream-adapter/datasets/pick_banana_100_newTable_1_offset_state_converted"
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
    use_reconstruct_images: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    unnorm_key: str = "pick_banana_200_newTable_2_offset_state_absolute_converted" # 需要根据实际情况修改路径和参数
    save_version: str = "vla-adapter"

    # Output
    ep_index: int = 15 # 测试的 episode index

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
    reconstruct_images = None

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

    if cfg.use_reconstruct_images:
        reconstruct_images = get_reconstruct_images(cfg, model.llm_dim, image_dim=588, predict_image_frame=1)
    
    processor = get_processor(cfg) if cfg.model_family == "openvla" else None
    return model, action_head, reconstruct_images, proprio_projector, processor

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

def plot_joint_trajectories(gt_states, gt_actions, pred_actions):
    """
    绘制多关节的状态和动作轨迹。
    
    Args:
        gt_states (list of np.ndarray): 每个元素为 shape (7,) 的状态数组，对应每个 step。
        gt_actions (list of np.ndarray): 每个元素为 shape (8, 7) 的动作数组，对应每个 step 的真实动作。
        pred_actions (list of np.ndarray): 每个元素为 shape (8, 7) 的动作数组，对应每个 step 的预测动作。
    """
    # 验证输入长度一致性
    n_steps = len(gt_states)
    if not (len(gt_actions) == n_steps and len(pred_actions) == n_steps):
        raise ValueError("gt_states, gt_actions, pred_actions 的长度必须一致")
    
    # 获取关节数量（假设至少有一个 step 且维度为 7）
    n_joints = gt_states[0].shape[0]
    n_actions = gt_actions[0].shape[0]  # 应为 8
    
    # 创建子图
    fig, axes = plt.subplots(n_joints, 1, figsize=(10, 2 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]
    
    # 为每个关节绘制
    for joint_idx in range(n_joints):
        ax = axes[joint_idx]
        
        # 绘制 gt_states（蓝点）
        step_indices = np.arange(n_steps)
        state_vals = [state[joint_idx] for state in gt_states]
        ax.plot(step_indices, state_vals, 'bo', markersize=4)
        
        # 绘制每个 step 的动作序列
        for step in range(n_steps):
            # 横坐标：均匀分布在 [step, step+1] 区间内
            x_actions = step + np.linspace(0, 1, n_actions, endpoint=True)
            
            # 预测动作（红线）先画，以便后续绿线覆盖
            pred_vals = pred_actions[step][:, joint_idx]
            ax.plot(x_actions, pred_vals, 'r-', linewidth=1.5, alpha=0.7)
            
            # 真实动作（绿线）后画，覆盖红线
            gt_vals = gt_actions[step][:, joint_idx]
            ax.plot(x_actions, gt_vals, 'g-', linewidth=1.5, alpha=0.9)
        
        ax.set_ylabel(f'Joint {joint_idx}')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Step')
    plt.suptitle('Joint State and Action Sequences (Green: gt_action, Red: pred_action)', fontsize=14)
    plt.tight_layout()
    plt.savefig('joint_trajectories.png')  # 保存图像
    plt.show()
    

@draccus.wrap()
def eval_openloop_joint(cfg: JointEvalConfig) -> None:
    set_seed_everywhere(0)

    logger.info("Loading model from %s", cfg.pretrained_checkpoint)
    model, action_head, reconstruct_images, proprio_projector, processor = initialize_model(cfg)
    model.eval()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if not next(model.parameters()).is_cuda and not (cfg.load_in_8bit or cfg.load_in_4bit):
        model = model.to(device)

    _load_stats(cfg, model)

    dataset_path = Path(cfg.dataset_path).expanduser().resolve()
    logger.info("Loading episodes from %s", dataset_path)
    episode_dirs = _list_episode_dirs(dataset_path, cfg.max_episodes)
    ep_dir = episode_dirs[cfg.ep_index] ###############

    steps_root = ep_dir / "steps"

    step_dirs = [p for p in steps_root.iterdir() if p.is_dir()]
    step_dirs.sort(key=lambda p: p.name)

    gt_actions = []
    pred_actions = []
    gt_states = []
    for step_dir in step_dirs[::cfg.num_open_loop_steps]:
        data_h5 = step_dir / "data.h5"
        image_primary = step_dir / "image_primary.jpg"
        image_wrist = step_dir / "image_wrist.jpg"
        
        start_num = int(step_dir.name)
        gt_action = []

        # 根据开环 num_open_loop_steps 获取标签动作数据
        for i in range(cfg.num_open_loop_steps):
            new_data_h5 = step_dir.parent / f"{start_num + i:04d}" / "data.h5"
            print(new_data_h5)
            # 判读文件是否存在,如果不存在用最后一个值填补
            if not new_data_h5.exists():
                gt_action.append(gt_action[-1])
                continue

            with h5py.File(new_data_h5, "r") as fp:
                if cfg.action_key not in fp:
                    raise KeyError(f"Missing action key '{cfg.action_key}' in {data_h5}")
                action = np.asarray(fp[cfg.action_key][()], dtype=np.float32)
                gt_action.append(action)

        with h5py.File(data_h5, "r") as fp:
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
                reconstruct_images=reconstruct_images,
                use_minivlm=cfg.use_minivlm,
            )
            pred_action = np.asarray(pred, dtype=np.float32)
            pred_action = _align_action_shape(pred_action, np.stack(gt_action))
        
        except Exception as exc:
            logger.error("Prediction failed at %s: %s", step_dir, exc)
            break

        gt_actions.append(np.stack(gt_action))
        pred_actions.append(pred_action)
        gt_states.append(proprio)
    
    # 画图
    plot_joint_trajectories(gt_states, gt_actions, pred_actions)


if __name__ == "__main__":
    eval_openloop_joint()  # type: ignore[call-arg]
