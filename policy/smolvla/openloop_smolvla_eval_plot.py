"""
Open-loop evaluation for SmolVLA on LeRobot datasets.

This script keeps the original SmolVLA loading path (LeRobot policy/config/processors)
while adopting the newer episode-based open-loop evaluation style.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import init_logging


logger = logging.getLogger(__name__)


@dataclass
class SmolVLAOpenLoopConfig:
    # Keep original SmolVLA loading style
    pretrained_policy_path: str = (
"/home/lxx/repo/lerobot/outputs/smolvla/miku112/pick_banana_200_newTable_next_state_action_smolvla_0322-2229/checkpoints/050000/pretrained_model"
    )

    # Keep original LeRobot dataset loading style
    dataset_repo_id: str = "miku112/pick_banana_200_newTable_next_state_action"
    dataset_root: str = "/home/lxx/repo/datasets/lerobot/miku112/pick_banana_200_newTable_next_state_action"

    # Open-loop eval settings
    ep_index: int = 0


    max_episodes: Optional[int] = None
    num_open_loop_steps: int = 8
    step_stride: Optional[int] = None  # default: num_open_loop_steps

    # Runtime
    device: Optional[str] = None

    # Output
    save_dir: str = "/home/lxx/repo/Double_Piper_Teleop/policy/smolvla/eval_results"
    save_prefix: str = "smolvla_openloop"
    show_plot: bool = True


def _to_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _to_numpy_action(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def _build_single_batch(sample: dict) -> dict:
    """Convert one dataset sample into a batch dict compatible with preprocessor."""
    batch = {}
    for key, value in sample.items():
        if key == "task":
            batch[key] = [str(value)]
        elif isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0)
        elif isinstance(value, np.ndarray):
            batch[key] = torch.from_numpy(value).unsqueeze(0)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            batch[key] = torch.tensor([value])
        else:
            batch[key] = value
    return batch


def _align_action_shape(pred_action: np.ndarray, gt_action: np.ndarray) -> np.ndarray:
    if pred_action.shape == gt_action.shape:
        return pred_action
    if pred_action.size == gt_action.size:
        return pred_action.reshape(gt_action.shape)
    raise ValueError(f"Cannot align prediction shape {pred_action.shape} to {gt_action.shape}")


def _list_episode_indices(dataset: LeRobotDataset, max_episodes: Optional[int]) -> list[list[int]]:
    episode_column = dataset.hf_dataset["episode_index"]
    episode_to_indices: dict[int, list[int]] = {}

    for abs_idx, ep in enumerate(episode_column):
        ep_id = _to_int(ep)
        episode_to_indices.setdefault(ep_id, []).append(abs_idx)

    episode_ids = sorted(episode_to_indices)
    if max_episodes is not None:
        episode_ids = episode_ids[:max_episodes]

    return [episode_to_indices[ep_id] for ep_id in episode_ids]


def _predict_action_chunk(policy, preprocessor, postprocessor, anchor_sample: dict, horizon: int) -> np.ndarray:
    # Ensure independent open-loop prediction from current anchor frame.
    policy.reset()

    model_batch = _build_single_batch(anchor_sample)
    model_batch = preprocessor(model_batch)

    with torch.no_grad():
        pred = policy.predict_action_chunk(model_batch)
        pred = postprocessor(pred)

    pred_np = pred.detach().cpu().numpy().squeeze(0).astype(np.float32)
    if pred_np.ndim == 1:
        pred_np = pred_np[None, :]

    if pred_np.shape[0] < horizon:
        pad_rows = np.repeat(pred_np[-1:], horizon - pred_np.shape[0], axis=0)
        pred_np = np.concatenate([pred_np, pad_rows], axis=0)

    return pred_np[:horizon]


def plot_joint_trajectories(
    gt_states: list[np.ndarray],
    gt_actions: list[np.ndarray],
    pred_actions: list[np.ndarray],
    save_path: Path,
    show_plot: bool,
) -> None:
    n_steps = len(gt_states)
    if n_steps == 0:
        raise ValueError("No data to plot.")
    if not (len(gt_actions) == n_steps and len(pred_actions) == n_steps):
        raise ValueError("gt_states, gt_actions, pred_actions must have the same length.")

    n_joints = gt_states[0].shape[0]
    n_actions = gt_actions[0].shape[0]

    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 2.2 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]

    for joint_idx in range(n_joints):
        ax = axes[joint_idx]

        step_indices = np.arange(n_steps)
        state_vals = [state[joint_idx] for state in gt_states]
        ax.plot(step_indices, state_vals, "bo", markersize=3, label="State")

        for step in range(n_steps):
            x_actions = step + np.linspace(0.0, 1.0, n_actions, endpoint=True)

            pred_vals = pred_actions[step][:, joint_idx]
            gt_vals = gt_actions[step][:, joint_idx]

            pred_label = "Pred action" if step == 0 else None
            gt_label = "GT action" if step == 0 else None

            ax.plot(x_actions, pred_vals, "r-", linewidth=1.3, alpha=0.75, label=pred_label)
            ax.plot(x_actions, gt_vals, "g-", linewidth=1.5, alpha=0.85, label=gt_label)

        ax.set_ylabel(f"Joint {joint_idx}")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Anchor step")
    plt.suptitle("SmolVLA Open-loop Joint Trajectories (Green: GT, Red: Pred)", fontsize=13)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180)
    logger.info("Saved plot to %s", save_path)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


@draccus.wrap()
def eval_openloop_smolvla(cfg: SmolVLAOpenLoopConfig) -> None:
    init_logging()

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading dataset %s from %s", cfg.dataset_repo_id, cfg.dataset_root)
    dataset = LeRobotDataset(
        repo_id=cfg.dataset_repo_id,
        root=cfg.dataset_root,
        image_transforms=None,
    )

    logger.info("Loading policy from %s", cfg.pretrained_policy_path)
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.pretrained_policy_path)
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    policy = policy.from_pretrained(cfg.pretrained_policy_path)
    policy = policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=cfg.pretrained_policy_path,
    )

    episodes = _list_episode_indices(dataset, cfg.max_episodes)
    if not episodes:
        raise ValueError("No episodes found in dataset.")

    if cfg.ep_index < 0 or cfg.ep_index >= len(episodes):
        raise IndexError(f"ep_index={cfg.ep_index} is out of range [0, {len(episodes) - 1}]")

    episode_indices = episodes[cfg.ep_index]
    if not episode_indices:
        raise ValueError(f"Episode {cfg.ep_index} has no frames.")

    stride = cfg.step_stride or cfg.num_open_loop_steps
    horizon = cfg.num_open_loop_steps

    logger.info(
        "Evaluating episode #%d: total_frames=%d, stride=%d, horizon=%d",
        cfg.ep_index,
        len(episode_indices),
        stride,
        horizon,
    )

    gt_states: list[np.ndarray] = []
    gt_actions: list[np.ndarray] = []
    pred_actions: list[np.ndarray] = []

    anchor_positions = list(range(0, len(episode_indices), stride))

    for pos in tqdm.tqdm(anchor_positions, desc="Open-loop anchors"):
        anchor_abs_idx = episode_indices[pos]
        anchor_sample = dataset[anchor_abs_idx]

        gt_chunk = []
        for offset in range(horizon):
            future_pos = min(pos + offset, len(episode_indices) - 1)
            future_abs_idx = episode_indices[future_pos]
            future_sample = dataset[future_abs_idx]
            gt_chunk.append(_to_numpy_action(future_sample["action"]).reshape(-1))

        gt_chunk_np = np.stack(gt_chunk, axis=0)
        pred_chunk_np = _predict_action_chunk(policy, preprocessor, postprocessor, anchor_sample, horizon)
        pred_chunk_np = _align_action_shape(pred_chunk_np, gt_chunk_np)

        state_np = _to_numpy_action(anchor_sample["observation.state"]).reshape(-1)

        gt_states.append(state_np)
        gt_actions.append(gt_chunk_np)
        pred_actions.append(pred_chunk_np)

    gt_all = np.stack(gt_actions, axis=0)
    pred_all = np.stack(pred_actions, axis=0)
    mse = float(np.mean((gt_all - pred_all) ** 2))

    save_dir = Path(cfg.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{cfg.save_prefix}_ep{cfg.ep_index:03d}.png"
    plot_joint_trajectories(
        gt_states=gt_states,
        gt_actions=gt_actions,
        pred_actions=pred_actions,
        save_path=save_path,
        show_plot=cfg.show_plot,
    )

    logger.info("Open-loop done. anchors=%d, overall MSE=%.6f", len(anchor_positions), mse)


if __name__ == "__main__":
    eval_openloop_smolvla()  # type: ignore[call-arg]
