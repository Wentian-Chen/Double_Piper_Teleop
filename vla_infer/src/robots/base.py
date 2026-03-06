'''
Author: boboji11 wendychen112@qq.com
Date: 2026-02-25 19:40:39
LastEditors: boboji11 wendychen112@qq.com
LastEditTime: 2026-02-25 19:41:43
FilePath: \Double_Piper_Teleop\vla_infer\robots\base.py
Description: 

Copyright (c) 2026 by boboji11 , All Rights Reserved. 
'''
from abc import ABC, abstractmethod
import typing
import numpy as np

class BaseCamera(ABC):
    """
    相机硬件抽象基类。
    """
    def __init__(self, camera_id: typing.Union[int, str], camera_name: str = "front_image"):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.connect()

    @abstractmethod
    def connect(self) -> None:
        """初始化相机连接、设置分辨率和帧率等。"""
        pass

    @abstractmethod
    def get_image(self) -> typing.Dict[str, np.ndarray]:
        """
        捕获一帧图像。
        
        :return: 字典格式，键名为 self.camera_name，值为 RGB 格式的 numpy 数组 (uint8)。
                 例如: {"front_image": ndarray(shape=(480, 640, 3))}
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """释放相机资源。"""
        pass


class BaseRobot(ABC):
    """
    机械臂硬件抽象基类。
    """
    @abstractmethod
    def setup(self) -> None:
        """初始化机械臂连接、设置控制模式等。"""
        pass
    @abstractmethod
    def reset(self) -> None:
        """将机械臂重置到初始位置。"""
        pass
    @abstractmethod
    def get_observation(self) -> typing.Dict[str, np.ndarray]:
        """
        获取机械臂当前状态（本体感觉）。
        
        :return: 字典格式。
                 例如: {"state": ndarray(shape=(6,), dtype=float32)}
        """
        pass

    @abstractmethod
    def apply_action(self, action_dict: typing.Dict[str, np.ndarray]) -> None:
        """
        解析动作字典，并下发给底层电机执行。
        如果是 Action Chunk (动作块)，子类可能需要在这里实现插值和高频下发逻辑。
        
        :param action_dict: 包含模型预测动作的字典。
                            例如: {"action": ndarray(shape=(chunk_size, 6))}
        """
        pass
