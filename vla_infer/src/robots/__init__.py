"""vla_infer 机器人适配层导出。"""

from .base import BaseCamera, BaseRobot
from .piper_single import PiperSingleRobot

__all__ = [
    "BaseCamera",
    "BaseRobot",
    "PiperSingleRobot",
]
