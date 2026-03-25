import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Iterator, Tuple, Dict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_h5_files(root_dir: str | Path, episode_ids: Optional[List[str]] = None) -> Iterator[Tuple[Path, str, str]]:
    """
    递归获取所有episode文件夹下steps文件夹中的data.h5文件路径
    
    路径结构: root_dir/episodes/{episode_id}/steps/{step_id}/data.h5
    
    Args:
        root_dir: 数据集根目录
        episode_ids: 指定要加载的episode ID列表，如果为None则加载所有episode
        
    Yields:
        (h5_file_path, episode_id, step_id) 元组
    """
    root = Path(root_dir)
    episodes_dir = root / "episodes"
    
    if not episodes_dir.exists():
        logger.error(f"Episodes directory not found: {episodes_dir}")
        return
    
    # 获取所有episode文件夹
    if episode_ids:
        episode_folders = [episodes_dir / episode_id for episode_id in episode_ids if (episodes_dir / episode_id).exists()]
    else:
        episode_folders = sorted([f for f in episodes_dir.iterdir() if f.is_dir()])
    
    for episode_folder in episode_folders:
        episode_id = episode_folder.name
        steps_dir = episode_folder / "steps"
        
        if not steps_dir.exists():
            logger.warning(f"Steps directory not found for episode {episode_id}: {steps_dir}")
            continue
        
        # 获取steps文件夹下的所有子文件夹（按数字排序）
        step_folders = sorted([f for f in steps_dir.iterdir() if f.is_dir()], 
                            key=lambda x: int(x.name))
        
        for step_folder in step_folders:
            step_id = step_folder.name
            h5_file = step_folder / "data.h5"
            
            if h5_file.exists():
                yield h5_file, episode_id, step_id
            else:
                logger.warning(f"data.h5 not found: {h5_file}")


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


def read_all_actions_from_episode(
    root_dir: str | Path, 
    episode_id: str
) -> Tuple[np.ndarray | None, List[str]]:
    """
    从指定episode的所有steps中读取action数据
    
    Args:
        root_dir: 数据集根目录
        episode_id: episode ID（如 "000000"）
        
    Returns:
        (actions数组, step_ids列表)，actions shape为(num_steps, 7)
    """
    actions = []
    step_ids = []
    
    # 构建steps目录路径
    steps_dir = Path(root_dir) / "episodes" / episode_id / "steps"
    
    if not steps_dir.exists():
        logger.error(f"Steps directory not found for episode {episode_id}: {steps_dir}")
        return None, []
    
    # 按step ID数字排序
    step_folders = sorted([f for f in steps_dir.iterdir() if f.is_dir()], 
                         key=lambda x: int(x.name))
    
    for step_folder in step_folders:
        step_id = step_folder.name
        h5_file = step_folder / "data.h5"
        
        if not h5_file.exists():
            logger.warning(f"data.h5 not found for episode {episode_id}, step {step_id}")
            continue
        
        action = read_actions_from_h5(h5_file)
        
        if action is not None:
            actions.append(action)
            step_ids.append(step_id)
    
    if actions:
        return np.array(actions), step_ids
    else:
        logger.warning(f"No valid actions found for episode {episode_id}")
        return None, []


def visualize_actions_from_h5(
    root_dir: str = "dream-adapter/miku112/pick_banana_200_newTable_2_offset_state_absolute_converted",
    episodes: Optional[List[str]] = None,
    save_path: str = "action_comparison_by_episode.png",
    alpha_base: float = 0.3,
    linewidth: float = 0.8
) -> None:
    """
    从H5文件读取action数据并可视化，支持多episode曲线叠加
    
    Args:
        root_dir: 数据集根目录
        episodes: 要加载的episode ID列表（如 ["000000", "000001"]），如果为None则加载所有episode
        save_path: 保存图片的路径
        alpha_base: 曲线透明度基础值，episode数量越多透明度越低
        linewidth: 曲线宽度
    """
    
    logger.info(f"Scanning H5 files in: {root_dir}")
    
    # 获取所有episode ID
    episodes_dir = Path(root_dir) / "episodes"
    
    if not episodes_dir.exists():
        logger.error(f"Episodes directory not found: {episodes_dir}")
        return
    
    # 确定要处理的episode列表
    if episodes is None:
        # 获取所有episode文件夹
        episode_ids = sorted([f.name for f in episodes_dir.iterdir() if f.is_dir()])
        logger.info(f"Found {len(episode_ids)} episodes: {episode_ids[:10]}{'...' if len(episode_ids) > 10 else ''}")
    else:
        episode_ids = episodes
        logger.info(f"Processing specified {len(episode_ids)} episodes: {episode_ids}")
    
    # 定义关节标签
    joint_labels = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Gripper']
    num_joints = len(joint_labels)
    
    # 创建7行1列的子图布局
    fig, axes = plt.subplots(num_joints, 1, figsize=(14, 16))
    fig.suptitle('Action Trajectories Across Different Episodes (H5 Direct Load)', 
                 fontsize=16, fontweight='bold')
    
    # 存储每个episode的步数和actions
    episode_data = {}
    
    # 循环处理每个episode
    for episode_idx, episode_id in enumerate(episode_ids):
        logger.info(f"Processing episode {episode_id} ({episode_idx + 1}/{len(episode_ids)})...")
        
        try:
            # 读取episode的所有actions
            actions, step_ids = read_all_actions_from_episode(root_dir, episode_id)
            
            if actions is None or len(actions) == 0:
                logger.warning(f"Skipping episode {episode_id} due to read error or no data")
                continue
            
            logger.info(f"Episode {episode_id}: Loaded {len(actions)} steps, action shape: {actions.shape}")
            
            # 存储数据
            episode_data[episode_id] = {
                'actions': actions,
                'step_ids': step_ids,
                'num_steps': len(actions)
            }
            
            # 创建时间步轴
            steps = np.arange(len(actions))
            
            # 计算当前episode的透明度
            alpha = max(0.1, alpha_base - (episode_idx * 0.03))
            
            # 为每个关节绘制action曲线
            for joint_idx in range(num_joints):
                ax = axes[joint_idx]
                
                # 绘制action曲线
                ax.plot(
                    steps,
                    actions[:, joint_idx],
                    # label=f'Episode {episode_id}',
                    color='blue',
                    linewidth=linewidth,
                    alpha=alpha
                )
                
                # 设置y轴标签
                ax.set_ylabel(joint_labels[joint_idx], fontsize=10)
                
                # 添加网格线
                ax.grid(True, linestyle='--', alpha=0.6)
                
                # 设置x轴标签（只在最后一个子图显示）
                if joint_idx == num_joints - 1:
                    ax.set_xlabel('Time Steps', fontsize=11)
                else:
                    ax.set_xticklabels([])  # 隐藏非底部子图的x轴标签
                    
        except Exception as e:
            logger.error(f"Failed to process episode {episode_id}: {e}")
            continue
    
    # 统一设置所有子图的x轴范围
    if episode_data:
        max_steps = max([data['num_steps'] for data in episode_data.values()])
        for ax in axes:
            ax.set_xlim(0, max_steps - 1)
    
    # 添加图例
    if episode_data:
        axes[0].legend(
            loc='upper right',
            fontsize=8,
            ncol=2 if len(episode_data) > 4 else 1,
            framealpha=0.9
        )
    
    # 添加总标题说明
    fig.text(
        0.5, 0.98,
        f'Total Episodes: {len(episode_data)} | Action Dimension: {num_joints} | Data Source: H5 Files',
        ha='center',
        fontsize=10,
        style='italic'
    )
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Figure saved to {save_path}")
    
    # 显示统计信息
    # if episode_data:
    #     logger.info(f"Processing completed. Statistics:")
    #     logger.info(f"  - Total episodes processed: {len(episode_data)}")
    #     for episode_id, data in episode_data.items():
    #         logger.info(f"  - Episode {episode_id}: {data['num_steps']} steps")
        
    #     # 输出每个episode的action统计信息
    #     for episode_id, data in episode_data.items():
    #         actions = data['actions']
    #         action_stats = {
    #             'min': actions.min(axis=0),
    #             'max': actions.max(axis=0),
    #             'mean': actions.mean(axis=0),
    #             'std': actions.std(axis=0)
    #         }
    #         logger.info(f"Episode {episode_id} action statistics:")
    #         for joint_idx, joint_label in enumerate(joint_labels):
    #             logger.info(f"  {joint_label}: min={action_stats['min'][joint_idx]:.3f}, "
    #                        f"max={action_stats['max'][joint_idx]:.3f}, "
    #                        f"mean={action_stats['mean'][joint_idx]:.3f}, "
    #                        f"std={action_stats['std'][joint_idx]:.3f}")
    
    plt.close()


def quick_validate_actions(
    root_dir: str = "dream-adapter/miku112/pick_banana_200_newTable_2_offset_state_absolute_converted",
    max_episodes: int = 3,
    max_steps_per_episode: int = 5
) -> None:
    """
    快速验证函数：打印每个episode的前几个step的action信息
    
    Args:
        root_dir: 数据集根目录
        max_episodes: 最多验证的episode数量
        max_steps_per_episode: 每个episode最多验证的step数量
    """
    logger.info("Quick validation of actions...")
    
    episodes_dir = Path(root_dir) / "episodes"
    
    if not episodes_dir.exists():
        logger.error(f"Episodes directory not found: {episodes_dir}")
        return
    
    # 获取前max_episodes个episode
    episode_folders = sorted([f for f in episodes_dir.iterdir() if f.is_dir()])[:max_episodes]
    
    for episode_folder in episode_folders:
        episode_id = episode_folder.name
        logger.info(f"\n{'='*50}")
        logger.info(f"Episode: {episode_id}")
        
        steps_dir = episode_folder / "steps"
        if not steps_dir.exists():
            logger.warning(f"Steps directory not found for episode {episode_id}")
            continue
        
        # 获取前max_steps_per_episode个step
        step_folders = sorted([f for f in steps_dir.iterdir() if f.is_dir()], 
                            key=lambda x: int(x.name))[:max_steps_per_episode]
        
        for step_folder in step_folders:
            step_id = step_folder.name
            h5_file = step_folder / "data.h5"
            
            if not h5_file.exists():
                logger.warning(f"  Step {step_id}: data.h5 not found")
                continue
            
            action = read_actions_from_h5(h5_file)
            
            if action is not None:
                joints = action[:6].round(3)
                gripper = action[6]
                logger.info(f"  Step {step_id}: Joints={joints}, Gripper={gripper:.2f}")
            else:
                logger.warning(f"  Step {step_id}: Failed to read action")


def main() -> None:
    """
    主函数，用于执行可视化
    """
    # 配置参数
    root_dir = "/home/charles/workspaces/Dream-adapter/datasets/pick_banana_100_newTable_1_offset_state_converted"
    
    # 选项1：处理所有找到的episode
    visualize_actions_from_h5(
        root_dir=root_dir,
        episodes=None,  # 处理所有episode
        save_path="action_comparison_all_episodes.png",
        alpha_base=0.3,
        linewidth=0.8
    )
    
    # 选项2：只处理指定的episode（如果需要）
    # specific_episodes = ["000000", "000001", "000002", "000003"]
    # visualize_actions_from_h5(
    #     root_dir=root_dir,
    #     episodes=specific_episodes,
    #     save_path="action_comparison_specific_episodes.png",
    #     alpha_base=0.4,
    #     linewidth=0.8
    # )
    
    # 选项3：快速验证（可选）
    # quick_validate_actions(root_dir, max_episodes=50, max_steps_per_episode=250)


if __name__ == "__main__":
    main()