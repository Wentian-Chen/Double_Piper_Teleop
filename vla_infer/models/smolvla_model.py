from __future__ import annotations

import typing as t

import numpy as np

from vla_infer.models.base import BaseVLAModel


class SmolVLAModel(BaseVLAModel):
    """SmolVLA 的 vla_infer Server 端封装实现。

    该类遵循 :class:`BaseVLAModel` 统一接口，负责：
    - 加载 `policy/smolvla/inference_model.py` 中的模型实现；
    - 统一解析 vla_infer 请求字典；
    - 输出标准动作字典 ``{"action": np.ndarray}``。

    请求字典推荐格式::

        {
            "cmd": "pick up the banana",
            "cam_head": np.ndarray(H, W, 3),
            "cam_wrist": np.ndarray(H, W, 3),
            "state": np.ndarray(7,)
        }

    兼容图像键别名：
    - 头相机：``cam_head | image_head | front_image``
    - 腕相机：``cam_wrist | image_wrist | wrist_image``
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dataset_repo_id: t.Optional[str] = None,
        dataset_root: t.Optional[str] = None,
        action_chunk_size: t.Optional[int] = None,
    ) -> None:
        self.dataset_repo_id = dataset_repo_id
        self.dataset_root = dataset_root
        self.action_chunk_size = action_chunk_size
        self._model = None
        super().__init__(model_path=model_path, device=device)

    def load_model(self) -> None:
        """加载 SmolVLA 推理模型。"""
        from policy.smolvla.inference_model import SMOLVLA

        self._model = SMOLVLA(
            model_path=self.model_path,
            dataset_repo_id=self.dataset_repo_id,
            dataset_root=self.dataset_root,
            device=self.device,
            action_chunk_size=self.action_chunk_size,
        )

    @staticmethod
    def _pick_first_image(
        observation: t.Dict[str, t.Any],
        candidate_keys: t.Sequence[str],
        logical_name: str,
    ) -> np.ndarray:
        """从候选键中提取第一张有效图像。

        :param observation: 输入观测字典。
        :param candidate_keys: 候选图像键名列表。
        :param logical_name: 逻辑名称，用于错误提示。
        :return: RGB 图像数组，形状为 ``(H, W, 3)``。
        :raises KeyError: 当未找到对应图像时抛出。
        :raises ValueError: 当图像格式非法时抛出。
        """
        for key in candidate_keys:
            if key in observation:
                image = observation[key]
                if not isinstance(image, np.ndarray):
                    raise ValueError(f"{logical_name} image must be numpy.ndarray, got {type(image)}")
                if image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError(
                        f"{logical_name} image must be HxWx3 RGB array, got shape={image.shape}"
                    )
                return image
        raise KeyError(f"Missing {logical_name} image. tried keys={list(candidate_keys)}")

    @staticmethod
    def _pick_state(observation: t.Dict[str, t.Any]) -> t.Optional[np.ndarray]:
        """从观测中提取本体状态向量。"""
        for key in ("state", "robot_state", "proprio"):
            if key in observation and observation[key] is not None:
                state = np.asarray(observation[key], dtype=np.float32).reshape(-1)
                return state
        return None

    @staticmethod
    def _normalize_action(action: t.Any) -> np.ndarray:
        """规范化模型输出动作为 float32 的二维数组。"""
        action_np = np.asarray(action, dtype=np.float32)
        if action_np.ndim == 1:
            action_np = np.expand_dims(action_np, axis=0)
        return action_np

    def predict(self, observation: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """执行 SmolVLA 推理。

        :param observation: 请求观测字典。推荐包含 ``cmd/cam_head/cam_wrist/state``。
        :return: 动作字典，固定格式 ``{"action": np.ndarray(chunk, dim)}``。
        """
        if self._model is None:
            raise RuntimeError("SmolVLAModel is not loaded. Call load_model first.")

        instruction = observation.get("cmd", "")
        if instruction:
            self._model.random_set_language(instruction)

        cam_head = self._pick_first_image(
            observation,
            candidate_keys=("cam_head", "image_head", "front_image"),
            logical_name="head",
        )
        cam_wrist = self._pick_first_image(
            observation,
            candidate_keys=("cam_wrist", "image_wrist", "wrist_image"),
            logical_name="wrist",
        )
        state = self._pick_state(observation)

        self._model.update_observation_window((cam_head, cam_wrist), state)

        if hasattr(self._model, "get_action_chunk"):
            action = self._model.get_action_chunk()
        else:
            action = self._model.get_action()

        return {"action": self._normalize_action(action)}
