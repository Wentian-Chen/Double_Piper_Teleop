from __future__ import annotations

import typing as t
import time 
import numpy as np
from .base import BaseRobot


class PiperSingleRobot(BaseRobot):
    """PiperSingle 的 vla_infer 机器人适配层。

    设计原则：
    - 复用 `my_robot.agilex_piper_single_base.PiperSingle` 已有能力；
    - 不重写相机驱动和采图逻辑；
    - 仅做观测/动作格式桥接，满足 vla_infer 的统一协议。

    piper.get() 返回数据结构:
        {
            'arm': {'left_arm': { 
                                    'joint': [0.0, 0.85220935, -0.68542569, 0.0, 0.78588684, -0.05256932], 
                                    'qpos': 0.0,
                                    'gripper': 0.0
                                },
            'image': {
                        'cam_head': {'color': array, 'depth': array}, 
                        'cam_wrist': {'color': array, 'depth': array}}
        }

    观测输出格式::

        {
            "cam_head": np.ndarray(H, W, 3),
            "cam_wrist": np.ndarray(H, W, 3),
            "state": np.ndarray(7,),      # [joint(6), gripper(1)]
            "joint": np.ndarray(6,),
            "qpos": np.ndarray(6,),
            "gripper": np.ndarray(1,)
        }

    动作输入格式::

        {
            "action": np.ndarray(T, D) or np.ndarray(D,)
        }

    其中 D 至少为 7（前 6 维为关节，最后一维为夹爪开合）。
    """

    def __init__(
        self,
        auto_setup: bool = True,
        robot_cls: t.Optional[t.Type[t.Any]] = None,
    ) -> None:
        """初始化 PiperSingle 适配器。

        :param auto_setup: 是否在构造阶段自动执行 :meth:`setup`。
        :param robot_cls: 可注入的机器人实现类，用于测试或自定义替换。
         vla_infer/src/robots/piper_single.py                 为空时默认使用 ``my_robot.agilex_piper_single_base.PiperSingle``。
        """
        if robot_cls is None:
            from my_robot.agilex_piper_single_base import PiperSingle

            robot_cls = PiperSingle

        self._robot = robot_cls()
        self._is_setup = False
        if auto_setup:
            self.setup()
        # initial for warming uo the realsense camera
        self.get_observation()
        time.sleep(4)

    def setup(self) -> None:
        """初始化机器人连接与采集项。"""
        self._robot.set_up()
        self._is_setup = True

    def reset(self) -> None:
        """复位机械臂到默认初始姿态。"""
        self._robot.reset()

    @staticmethod
    def _to_fixed_length_vector(value: t.Any, length: int) -> np.ndarray:
        """将输入统一为固定长度 float32 向量，不足补零、超长截断。"""
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.size >= length:
            return vector[:length]
        padded = np.zeros(length, dtype=np.float32)
        padded[: vector.size] = vector
        return padded

    @classmethod
    def _extract_arm_state(cls, controller_data: t.Dict[str, t.Any]) -> t.Dict[str, np.ndarray]:
        """解析左臂状态，并统一类型为 float32 numpy。
        {'left_arm': { 
                        'joint': [0.0, 0.85220935, -0.68542569, 0.0, 0.78588684, -0.05256932], 
                        'qpos': 0.0,
                                  'cam_head': {'color': array, 'depth': array}, 
            'cam_wrist': {'color': array, 'depth': array}  'gripper': 0.0
                    }
        }

        返回结构：
        {
            "joint": np.ndarray(6,),    # 关节位置
            "qpos": np.ndarray(6,),     # 关节速度
            "gripper": np.ndarray(1,),  # 夹爪开合度
            "state": np.ndarray(7,)     # 关节+夹爪整体状态
        }
        """
        left_arm = controller_data.get("left_arm", {})
        # Keep schema stable even when controller returns scalar or malformed vectors.
        joint = cls._to_fixed_length_vector(left_arm.get("joint", np.zeros(6)), length=6)
        qpos = cls._to_fixed_length_vector(left_arm.get("qpos", np.zeros(6)), length=6)
        gripper_value = np.asarray([left_arm.get("gripper", 0.0)], dtype=np.float32)
        state = np.concatenate([joint, gripper_value], axis=0)
        return {
            "joint": joint,
            "qpos": qpos,
            "gripper": gripper_value,
            "state": state,
        }

    @staticmethod
    def _extract_images(sensor_data: t.Dict[str, t.Any]) -> t.Dict[str, np.ndarray]:
        """解析头部与腕部图像（RGB）。   
        {
            'cam_head': {'color': array, 'depth': array}, 
            'cam_wrist': {'color': array, 'depth': array}
        }

        RGB格式
        """
        head = sensor_data.get("cam_head", {}).get("color")
        wrist = sensor_data.get("cam_wrist", {}).get("color")

        if not isinstance(head, np.ndarray):
            raise ValueError("Invalid cam_head image. expected numpy.ndarray at sensor_data['cam_head']['color']")
        if not isinstance(wrist, np.ndarray):
            raise ValueError("Invalid cam_wrist image. expected numpy.ndarray at sensor_data['cam_wrist']['color']")

        return {
            "cam_head": head,
            "cam_wrist": wrist,
        }

    @staticmethod
    def _split_robot_data(robot_data: t.Any) -> t.Tuple[t.Dict[str, t.Any], t.Dict[str, t.Any]]:
        """兼容解析机器人返回数据结构。

        支持两种输入：
        1) 旧格式: ``[controller_data, sensor_data]``
        2) 新格式: ``{"arm": {...}, "image": {...}}``
        """
        if isinstance(robot_data, (list, tuple)) and len(robot_data) == 2:
            controller_data, sensor_data = robot_data
            return controller_data, sensor_data

        if isinstance(robot_data, dict):
            if "arm" in robot_data and "image" in robot_data:
                return robot_data.get("arm", {}), robot_data.get("image", {})

        raise RuntimeError(
            "Unexpected PiperSingle.get() return format. expected [controller_data, sensor_data] "
            "or {'arm': {...}, 'image': {...}}"
        )

    def get_observation(self) -> t.Dict[str, np.ndarray]:
        """获取当前观测。

        图像色彩空间说明：
        - 返回的 ``cam_head`` 与 ``cam_wrist`` 为 **RGB** 排列（H, W, 3）。
        - 若下游算法需要 OpenCV 默认格式，请自行转换为 BGR。

        :return: 统一观测字典（图像 + 本体状态）。

        {
            "cam_head": np.ndarray(H, W, 3),
            "cam_wrist": np.ndarray(H, W, 3),
            "state": np.ndarray(7,),      # [joint(6), gripper(1)]
            "joint": np.ndarray(6,),
            "qpos": np.ndarray(6,),
            "gripper": np.ndarray(1,)
        }       
        """
        robot_data = self._robot.get()
        controller_data, sensor_data = self._split_robot_data(robot_data)
        obs = {}
        obs.update(self._extract_arm_state(controller_data))
        obs.update(self._extract_images(sensor_data))
        return obs

    @staticmethod
    def _parse_action(action_dict: t.Dict[str, t.Any]) -> t.Tuple[np.ndarray, float]:
        """解析动作字典，提取首帧 joint+gripper。"""
        if "action" not in action_dict:
            raise KeyError("action_dict must contain key 'action'")

        action = np.asarray(action_dict["action"], dtype=np.float32)
        if action.ndim == 2:
            action = action[0]
        if action.ndim != 1:
            raise ValueError(f"action must be 1D or 2D array, got shape={action.shape}")
        if action.shape[0] < 7:
            raise ValueError(f"action dimension must be >= 7, got {action.shape[0]}")

        joint = action[:6]
        gripper = float(action[6])
        return joint, gripper

    def apply_action(self, action_dict: t.Dict[str, np.ndarray]) -> None:
        """将动作字典转换为 PiperSingle 控制指令并执行。"""
        joint, gripper = self._parse_action(action_dict)

        move_data = {
            "arm": {
                "left_arm": {
                    "joint": joint.tolist(),
                    "gripper": gripper,
                }
            }
        }
        self._robot.move(move_data)
