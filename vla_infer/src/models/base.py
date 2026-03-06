'''
Author: boboji11 wendychen112@qq.com
Date: 2026-02-25 15:57:00
LastEditors: boboji11 wendychen112@qq.com
LastEditTime: 2026-02-25 19:38:05
FilePath: \Double_Piper_Teleop\vla_infer\models\base.py
Description: 

Copyright (c) 2026 by boboji11 , All Rights Reserved. 
'''
from abc import ABC, abstractmethod
import typing
import numpy as np

class BaseVLAModel(ABC):
    """
    VLA 模型统一基类。运行在 Server 端
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.load_model()

    @abstractmethod
    def load_model(self) -> None:
        """
        在此处加载模型权重、Tokenizer 和预处理器。
        强烈建议在此方法末尾进行一次 Dummy Forward (预热)，以避免第一次推理时的延迟尖峰。
        """
        pass

    @abstractmethod
    def predict(self, observation: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        """
        核心推理接口。
        
        :param observation: 包含多模态数据的字典。
                            例如: {"cmd": "pick a banana", "front_image": ndarray, "state": ndarray}
        :return: 包含动作序列的字典。
                 必须包含 "action" 键，例如: {"action": ndarray(shape=(chunk_size, action_dim))}
        """
        pass
