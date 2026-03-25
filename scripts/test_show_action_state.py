import json
import matplotlib.pyplot as plt
import numpy as np

def load_json_data(file_path):
    """
    加载 JSON 文件并解析日志数据。
    
    Args:
        file_path (str): JSON 文件路径。
    
    Returns:
        list: 包含每个 step 的 state 和 output_action 的列表。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['log']

def extract_joint_data(logs):
    """
    从日志中提取每个关节的状态序列和动作序列。
    
    Args:
        logs (list): 日志数据列表，每个元素包含 state 和 output_action。
    
    Returns:
        tuple: (state_series, action_series)
            state_series: 形状 (num_steps, num_joints) 的数组
            action_series: 形状 (num_steps, num_actions, num_joints) 的数组
    """
    num_steps = len(logs)
    num_joints = len(logs[0]['state'])
    num_actions = len(logs[0]['output_action'])
    
    state_series = np.zeros((num_steps, num_joints))
    action_series = np.zeros((num_steps, num_actions, num_joints))
    
    for i, log in enumerate(logs):
        state_series[i] = log['state']
        action_series[i] = log['output_action']
    
    return state_series, action_series

def plot_joint_trajectories(state_series, action_series):
    """
    绘制 7 个子图，每个子图显示一个关节的 state（蓝点）和 action 序列（红线）。
    
    Args:
        state_series (np.ndarray): 形状 (num_steps, num_joints) 的状态数据。
        action_series (np.ndarray): 形状 (num_steps, num_actions, num_joints) 的动作数据。
    """
    num_steps, num_joints = state_series.shape
    num_actions = action_series.shape[1]
    
    fig, axes = plt.subplots(num_joints, 1, figsize=(10, 2 * num_joints), sharex=True)
    if num_joints == 1:
        axes = [axes]
    
    for joint_idx in range(num_joints):
        ax = axes[joint_idx]
        # 绘制 state 点（蓝色圆点）
        step_indices = np.arange(num_steps)
        ax.plot(step_indices, state_series[:, joint_idx], 'bo', markersize=4, label='state')
        
        # 为每个 step 绘制 action 线段（红色）
        for step in range(num_steps):
            # action 横坐标：均匀分布在 [step, step+1] 区间内
            x_actions = step + np.linspace(0, 1, num_actions, endpoint=True)
            y_actions = action_series[step, :, joint_idx]
            ax.plot(x_actions, y_actions, 'r-', linewidth=1.5, alpha=0.7)
        if joint_idx == 6:
            ax.set_ylabel(f'gripper')
        else:
            ax.set_ylabel(f'Joint {joint_idx}')
        ax.grid(True, linestyle='--', alpha=0.6)
        # ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Step')
    plt.suptitle('Joint State and Action Sequences', fontsize=14)
    plt.tight_layout()
    plt.savefig('joint_trajectories.png', dpi=300)
    plt.show()
    

def main():
    # 指定 JSON 文件路径（请根据实际情况修改）
    file_path = '/home/charles/workspaces/Double_Piper_Teleop/tem.json'
    
    try:
        logs = load_json_data(file_path)
        print(f"成功加载 {len(logs)} 个 step 的数据")
        
        state_series, action_series = extract_joint_data(logs)
        print(f"状态数据形状: {state_series.shape}")
        print(f"动作数据形状: {action_series.shape}")
        
        plot_joint_trajectories(state_series, action_series)
    
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到，请检查路径。")
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是有效的 JSON 格式。")
    except KeyError as e:
        print(f"错误：JSON 数据结构缺少必要字段 {e}。")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == '__main__':
    main()