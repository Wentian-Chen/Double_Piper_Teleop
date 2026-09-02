import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader

# LeRobot 核心组件
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

# 1. 配置参数
# ---------------------------------------------------------
# 你的模型 Checkpoint 路径 (例如: "outputs/train/smolvla_test/checkpoints/last")
PRETRAINED_POLICY_PATH = "/home/lxx/repo/lerobot/outputs/smolvla/miku112/pick_block_0325_1_offset_state_smolvla_0325-2331/checkpoints/050000/pretrained_model" 
# 你的数据集 ID (namespace/repo_name)
DATASET_REPO_ID = "miku112/pick_block_0325_1_offset_state" 
# 你的数据集本地路径
DATASET_ROOT = "/home/lxx/repo/datasets/lerobot/miku112/pick_block_0325_1_offset_state"
# 评估样本数量 (可视化前 N 个样本)
NUM_SAMPLES_TO_PLOT = 100
# 可视化结果保存路径
SAVE_DIR = "/home/lxx/repo/Double_Piper_Teleop/policy/smolvla/eval_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate():
    init_logging()
    
    # 2. 加载数据集 (使用 Test 集进行验证)
    # ---------------------------------------------------------
    print(f"Loading dataset {DATASET_REPO_ID}...")
    dataset = LeRobotDataset(
        repo_id=DATASET_REPO_ID,
        root=DATASET_ROOT,
        # 如果需要特定 episodes，可以在这里传 episodes=[...]
        image_transforms=None # 在这里通常不需要额外的 transform，policy 内部会处理
    )
    
    # 使用 DataLoader 方便批处理（这里 batch_size=1 方便逐帧分析）
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 3. 加载训练好的策略 (Policy)
    # ---------------------------------------------------------
    print(f"Loading policy from {PRETRAINED_POLICY_PATH}...")
    # 构建配置并实例化策略，再加载权重
    cfg = PreTrainedConfig.from_pretrained(PRETRAINED_POLICY_PATH)
    policy = make_policy(cfg=cfg, ds_meta=dataset.meta)
    policy = policy.from_pretrained(PRETRAINED_POLICY_PATH)
    policy.to(DEVICE)
    policy.eval() # 切换到评估模式

    # 获取预处理器和后处理器
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=PRETRAINED_POLICY_PATH,
    )

    # 4. 推理循环
    # ---------------------------------------------------------
    ground_truths = []
    predictions = []

    print("Starting inference...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= NUM_SAMPLES_TO_PLOT:
                break
            
            # 获取任务描述 (从 task_index 映射)
            task_index = batch["task_index"][0].item()
            task_description = dataset.meta.tasks[dataset.meta.tasks['task_index'] == task_index].index[0]
            
            # 准备输入给处理器
            batch["task"] = [task_description]

            # 保存原始真值动作 (未经过 normalization)
            gt_action = batch["action"].clone().cpu().numpy().squeeze()
            
            # 应用预处理器 (包含 Tokenization, Normalization, AddBatchDim 等)
            batch = preprocessor(batch)
            
            # 模型推理
            pred_action = policy.select_action(batch)
            
            # 应用后处理器 (包含 Unnormalization)
            pred_action = postprocessor(pred_action)
            
            # 存储结果 (转回 CPU numpy)
            ground_truths.append(gt_action)
            predictions.append(pred_action.cpu().numpy().squeeze())

    # 5. 可视化对比 (拟合程度)
    # ---------------------------------------------------------
    gt_array = np.array(ground_truths)   # Shape: [N, Action_Dim]
    pred_array = np.array(predictions)   # Shape: [N, Action_Dim]

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "eval_plot.png")
    plot_results(gt_array, pred_array, save_path)

def plot_results(gt, pred, save_path=None):
    """
    绘制动作维度的对比图
    """
    action_dim = gt.shape[1]
    time_steps = np.arange(gt.shape[0])

    # 创建一个多子图的画板
    fig, axes = plt.subplots(action_dim, 1, figsize=(10, 2 * action_dim), sharex=True)
    if action_dim == 1: axes = [axes]

    for dim in range(action_dim):
        ax = axes[dim]
        ax.plot(time_steps, gt[:, dim], label='Ground Truth', color='black', linewidth=1.5, alpha=0.7)
        ax.plot(time_steps, pred[:, dim], label='Prediction', color='red', linestyle='--', linewidth=1.5)
        ax.set_ylabel(f'Dim {dim}')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.xlabel('Time Step')
    plt.suptitle(f'Open-Loop Evaluation: Ground Truth vs Prediction (MSE: {np.mean((gt-pred)**2):.4f})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"图像已保存至: {save_path}")
        
    plt.show()
    print(f"可视化已完成，总体 MSE Loss: {np.mean((gt-pred)**2):.5f}")

if __name__ == "__main__":
    evaluate()