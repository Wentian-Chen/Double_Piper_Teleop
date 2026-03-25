#!/usr/bin/env python3
"""
从H5文件读取action数据并驱动Piper机械臂
目录结构: root/0000/data.h5, root/0001/data.h5, ...
action格式: (7,) float32 - 前6维关节角度，第7维夹爪控制
"""

import sys
sys.path.append("./")

import time
import h5py
import numpy as np
from pathlib import Path
from typing import Iterator
from controller.Piper_controller import PiperController


def get_h5_files(root_dir: str | Path) -> Iterator[Path]:
    """递归获取所有子文件夹中的data.h5文件路径，按文件夹名称排序"""
    root = Path(root_dir)
    files = sorted(root.rglob("data.h5"), key=lambda p: p.parent.name)
    yield from files


def read_action(file_path: Path) -> np.ndarray | None:
    """从H5文件中读取action数据集并验证维度"""
    try:
        with h5py.File(file_path, 'r') as f:
            if 'action' not in f:
                print(f"[警告] {file_path} 中未找到 'action' 数据集")
                return None
            action = f['action'][()]
            if action.shape != (7,):
                print(f"[警告] {file_path} action形状异常: {action.shape}")
            return action.astype(np.float32)
    except Exception as e:
        print(f"[错误] 读取 {file_path} 失败: {e}")
        return None


def execute_action(controller: PiperController, action: np.ndarray, 
                   sleep_time: float = 2.0) -> None:
    """执行单个action：前6维控制关节，第7维控制夹爪"""
    joint_angles = action[:6]
    gripper_value = float(action[6])
    
    controller.set_joint(joint_angles)
    controller.set_gripper(gripper_value)
    time.sleep(sleep_time)


def main():
    """主函数：初始化机械臂并执行所有H5文件中的action"""
    root_dir = "/home/charles/桌面/000000/steps"           # H5文件根目录
    
    # 初始化
    controller = PiperController("piper_arm")
    controller.set_up("can0")
    
    # 初始位置（闭合夹爪）
    controller.set_gripper(0.0)
    init_pose = np.array([0.04867723, -0.02370157, 0.04152138, 
                          0.01727876, 0.33063517, 0.0])
    controller.set_joint(init_pose)
    time.sleep(2)
    
    # 遍历执行所有action
    for h5_file in get_h5_files(root_dir):
        folder_name = h5_file.parent.name
        action = read_action(h5_file)
        
        if action is not None:
            print(f"[{folder_name}] 执行: 关节={action[:6].round(3)}, 夹爪={action[6]:.2f}")
            execute_action(controller, action, sleep_time=1)
    
    # 结束：关闭夹爪
    controller.set_gripper(0.0)


if __name__ == "__main__":
    main()