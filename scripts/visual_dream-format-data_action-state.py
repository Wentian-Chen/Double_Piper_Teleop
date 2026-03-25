import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Iterator, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_state_from_h5(file_path: Path) -> np.ndarray | None:
    """
    从H5文件中读取state（observation/proprio）数据集

    Args:
        file_path: data.h5文件路径

    Returns:
        state数组，shape为(7,)，如果读取失败返回None
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # 优先读取 observation/proprio
            if 'observation/proprio' in f:
                state = f['observation/proprio'][()]
            # 兼容其他可能的键名
            elif 'state' in f:
                state = f['state'][()]
            elif 'proprio' in f:
                state = f['proprio'][()]
            else:
                logger.warning(f"{file_path} 中未找到 state/proprio 数据集")
                return None

            # 验证维度
            if len(state.shape) == 1 and state.shape[0] == 7:
                return state.astype(np.float32)
            elif len(state.shape) == 2 and state.shape[1] == 7:
                # 如果是2D数组，取第一个时间步
                return state[0].astype(np.float32)
            else:
                logger.warning(f"{file_path} state形状异常: {state.shape}")
                return None
    except Exception as e:
        logger.error(f"读取 {file_path} 失败: {e}")
        return None


def read_actions_from_h5(file_path: Path) -> np.ndarray | None:
    """
    从H5文件中读取action数据集

    Args:
        file_path: data.h5文件路径

    Returns:
        action数组，shape为(7,)，如果读取失败返回None
    """
    try:
        with h5py.File(file_path, 'r') as f:
            if 'action' not in f:
                logger.warning(f"{file_path} 中未找到 'action' 数据集")
                return None
            action = f['action'][()]

            # 验证action维度
            if len(action.shape) == 1 and action.shape[0] == 7:
                return action.astype(np.float32)
            elif len(action.shape) == 2 and action.shape[1] == 7:
                # 如果是2D数组，取第一个时间步
                return action[0].astype(np.float32)
            else:
                logger.warning(f"{file_path} action形状异常: {action.shape}")
                return None
    except Exception as e:
        logger.error(f"读取 {file_path} 失败: {e}")
        return None


def read_state_and_action_from_episode(
    root_dir: str | Path,
    episode_id: str
) -> Tuple[np.ndarray | None, np.ndarray | None, List[str]]:
    """
    从指定episode的所有steps中读取state和action数据

    Args:
        root_dir: 数据集根目录
        episode_id: episode ID（如 "000000"）

    Returns:
        (states数组, actions数组, step_ids列表)
        states shape为(num_steps, 7)
        actions shape为(num_steps, 7)
    """
    states = []
    actions = []
    step_ids = []

    steps_dir = Path(root_dir) / "episodes" / episode_id / "steps"

    if not steps_dir.exists():
        logger.error(f"Steps directory not found for episode {episode_id}: {steps_dir}")
        return None, None, []

    step_folders = sorted([f for f in steps_dir.iterdir() if f.is_dir()],
                         key=lambda x: int(x.name))

    for step_folder in step_folders:
        step_id = step_folder.name
        h5_file = step_folder / "data.h5"

        if not h5_file.exists():
            logger.warning(f"data.h5 not found for episode {episode_id}, step {step_id}")
            continue

        state = read_state_from_h5(h5_file)
        action = read_actions_from_h5(h5_file)

        if state is not None and action is not None:
            states.append(state)
            actions.append(action)
            step_ids.append(step_id)
        else:
            logger.warning(f"Skipping step {step_id} due to missing state or action")

    if states and actions:
        return np.array(states), np.array(actions), step_ids
    else:
        logger.warning(f"No valid state/action found for episode {episode_id}")
        return None, None, []


def visualize_episode_state_action(
    root_dir: str = "dream-adapter/miku112/pick_banana_200_newTable_2_offset_state_absolute_converted",
    episode_id: str = "000000",
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    可视化单个episode的state（蓝线）和action（红线）轨迹

    Args:
        root_dir: 数据集根目录
        episode_id: 要可视化的episode ID
        save_path: 保存图片的路径，如果为None则不保存
        show_plot: 是否显示图形窗口
    """
    logger.info(f"Loading episode {episode_id} from {root_dir}...")

    states, actions, step_ids = read_state_and_action_from_episode(root_dir, episode_id)

    if states is None or actions is None:
        logger.error(f"Failed to load data for episode {episode_id}")
        return

    num_steps = len(states)
    logger.info(f"Loaded {num_steps} steps, states shape: {states.shape}, actions shape: {actions.shape}")

    joint_labels = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Gripper']
    num_joints = len(joint_labels)

    fig, axes = plt.subplots(num_joints, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f'Episode {episode_id}: State (blue) vs Action (red)', fontsize=16, fontweight='bold')

    steps = np.arange(num_steps)

    for joint_idx in range(num_joints):
        ax = axes[joint_idx]

        ax.plot(steps, states[:, joint_idx], 'b-', linewidth=1.5, label='State')
        ax.plot(steps, actions[:, joint_idx], 'r-', linewidth=1.5, label='Action')

        ax.set_ylabel(joint_labels[joint_idx], fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)

        if joint_idx == 0:
            ax.legend(loc='upper right', fontsize=10)

        if joint_idx == num_joints - 1:
            ax.set_xlabel('Time Steps', fontsize=11)

    fig.text(
        0.5, 0.98,
        f'Episode {episode_id} | Steps: {num_steps} | Joints: {num_joints} | Blue: State, Red: Action',
        ha='center',
        fontsize=10,
        style='italic'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def quick_validate_actions(
    root_dir: str = "dream-adapter/miku112/pick_banana_200_newTable_2_offset_state_absolute_converted",
    max_episodes: int = 3,
    max_steps_per_episode: int = 5
) -> None:
    """
    快速验证函数：打印每个episode的前几个step的state和action信息
    """
    logger.info("Quick validation of state and action...")

    episodes_dir = Path(root_dir) / "episodes"

    if not episodes_dir.exists():
        logger.error(f"Episodes directory not found: {episodes_dir}")
        return

    episode_folders = sorted([f for f in episodes_dir.iterdir() if f.is_dir()])[:max_episodes]

    for episode_folder in episode_folders:
        episode_id = episode_folder.name
        logger.info(f"\n{'='*50}")
        logger.info(f"Episode: {episode_id}")

        steps_dir = episode_folder / "steps"
        if not steps_dir.exists():
            logger.warning(f"Steps directory not found for episode {episode_id}")
            continue

        step_folders = sorted([f for f in steps_dir.iterdir() if f.is_dir()],
                            key=lambda x: int(x.name))[:max_steps_per_episode]

        for step_folder in step_folders:
            step_id = step_folder.name
            h5_file = step_folder / "data.h5"

            if not h5_file.exists():
                logger.warning(f"  Step {step_id}: data.h5 not found")
                continue

            state = read_state_from_h5(h5_file)
            action = read_actions_from_h5(h5_file)

            if state is not None and action is not None:
                state_joints = state[:6].round(3)
                state_gripper = state[6]
                action_joints = action[:6].round(3)
                action_gripper = action[6]
                logger.info(f"  Step {step_id}: State Joints={state_joints}, Gripper={state_gripper:.2f} | Action Joints={action_joints}, Gripper={action_gripper:.2f}")
            else:
                logger.warning(f"  Step {step_id}: Failed to read state or action")


def main() -> None:
    """
    主函数，演示如何可视化单个episode的state和action
    """
    # 请修改为实际路径和episode ID
    root_dir = "/home/charles/workspaces/Dream-adapter/datasets/pick_banana_100_newTable_1_offset_state_converted"
    episode_id = "000010"

    visualize_episode_state_action(
        root_dir=root_dir,
        episode_id=episode_id,
        save_path=f"episode_{episode_id}_state_action.png",
        show_plot=True
    )

    # 可选：快速验证
    # quick_validate_actions(root_dir, max_episodes=3, max_steps_per_episode=3)


if __name__ == "__main__":
    main()