#!/usr/bin/env python3
"""
循环保存机械臂执行数据到JSON
支持: output_action (8, 7) 和 state (7,) numpy数组
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import gzip


class NumpyEncoder(json.JSONEncoder):
    """JSON编码器：自动转换numpy数组为list"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


class CycleDataLogger:
    """
    循环数据记录器
    自动处理numpy序列化、分卷存储和数据验证
    """
    
    def __init__(self, output_dir: str = "./cycle_data", 
                 max_cycles: int = 1000,
                 compress: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_cycles = max_cycles
        self.compress = compress
        
        self.buffer: List[Dict] = []
        self.file_idx = 0
        self.cycle_idx = 0
    
    def add(self, cycle_report: Dict[str, Any]) -> None:
        """添加单次循环数据"""
        # 提取并验证
        action = np.asarray(cycle_report["execution"]["output_action"])
        state = np.asarray(cycle_report["observation"]["state"])
        
        assert action.shape == (8, 7), f"action形状错误: {action.shape}"
        assert state.shape == (7,), f"state形状错误: {state.shape}"
        
        # 构建记录
        self.buffer.append({
            "cycle_id": self.cycle_idx,
            "timestamp": datetime.now().isoformat(),
            "execution": {"output_action": action, "action_shape": [8, 7]},
            "observation": {"state": state, "state_shape": [7]}
        })
        self.cycle_idx += 1
        
        # 分卷检查
        if len(self.buffer) >= self.max_cycles:
            self._flush()
    
    def _flush(self) -> None:
        """保存当前缓冲区到文件"""
        if not self.buffer:
            return
            
        filename = f"cycles_{self.file_idx:04d}.json"
        if self.compress:
            filename += ".gz"
        
        filepath = self.output_dir / filename
        data = {
            "metadata": {
                "num_cycles": len(self.buffer),
                "created": datetime.now().isoformat()
            },
            "cycles": self.buffer
        }
        
        json_str = json.dumps(data, cls=NumpyEncoder, indent=2)
        
        if self.compress:
            with gzip.open(filepath, 'wt') as f:
                f.write(json_str)
        else:
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        print(f"[保存] {filepath} ({len(self.buffer)} cycles)")
        self.buffer = []
        self.file_idx += 1
    
    def close(self) -> None:
        """保存剩余数据"""
        self._flush()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# ==================== 使用示例 ====================

# 方式1: 批量记录（推荐用于循环）
def example_batch():
    with CycleDataLogger(output_dir="./robot_data") as logger:
        for i in range(100):  # 你的循环
            cycle_report = {
                "execution": {
                    "output_action": np.random.randn(8, 7).astype(np.float32)
                },
                "observation": {
                    "state": np.random.randn(7).astype(np.float32)
                }
            }
            logger.add(cycle_report)  # 自动处理numpy转换


# 方式2: 快速单次保存（简化版）
def quick_save(cycle_report: Dict, filepath: str = "cycle.json"):
    """单次快速保存，无分卷功能"""
    data = {
        "timestamp": datetime.now().isoformat(),
        "execution": {
            "output_action": cycle_report["execution"]["output_action"].tolist()
        },
        "observation": {
            "state": cycle_report["observation"]["state"].tolist()
        }
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    example_batch()