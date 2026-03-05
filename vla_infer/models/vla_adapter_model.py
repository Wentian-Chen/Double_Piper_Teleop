from __future__ import annotations

import importlib.util
import importlib
from pathlib import Path
import typing as t

import numpy as np

from vla_infer.models.base import BaseVLAModel


class VLAAdapterModel(BaseVLAModel):
    """VLA-Adapter 的 vla_infer Server 端封装实现。

    说明：
    - 该实现复用 `policy/vla-adapter/openloop_vlaadapter_eval.py` 的初始化逻辑；
    - 统一对外暴露 :meth:`predict`，输出标准动作字典。

    请求字典推荐格式::

        {
            "cmd": "pick up the banana",
            "cam_head": np.ndarray(H, W, 3),
            "cam_wrist": np.ndarray(H, W, 3),
            "state": np.ndarray(8,)
        }

    注意：`state` 建议为 8 维（Piper 的 7 维可在上游补零到 8 维，或由适配层自动补齐）。
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        base_model_checkpoint: t.Optional[str] = None,
        model_family: str = "openvla",
        use_l1_regression: bool = True,
        use_minivlm: bool = True,
        use_pro_version: bool = True,
        use_proprio: bool = True,
        num_images_in_input: int = 2,
        unnorm_key: str = "pick_banana_50",
        save_version: str = "vla-adapter",
    ) -> None:
        self.base_model_checkpoint = base_model_checkpoint
        self.model_family = model_family
        self.use_l1_regression = use_l1_regression
        self.use_minivlm = use_minivlm
        self.use_pro_version = use_pro_version
        self.use_proprio = use_proprio
        self.num_images_in_input = num_images_in_input
        self.unnorm_key = unnorm_key
        self.save_version = save_version

        self._cfg: t.Any = None
        self._adapter_module: t.Any = None
        self._model: t.Any = None
        self._action_head: t.Any = None
        self._proprio_projector: t.Any = None
        self._noisy_action_projector: t.Any = None
        self._processor: t.Any = None

        super().__init__(model_path=model_path, device=device)

    @staticmethod
    def _load_eval_module(module_path: str):
        """动态加载 `openloop_vlaadapter_eval.py` 模块。"""
        spec = importlib.util.spec_from_file_location("vla_adapter_eval_module", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load vla-adapter module from: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _pick_image(observation: t.Dict[str, t.Any], names: t.Sequence[str], logical_name: str) -> np.ndarray:
        """从观测字典提取图像。"""
        for name in names:
            if name in observation:
                image = observation[name]
                if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[-1] == 3:
                    return image
                raise ValueError(f"{logical_name} image must be HxWx3 numpy.ndarray, got {type(image)}")
        raise KeyError(f"Missing {logical_name} image, candidate keys: {list(names)}")

    @staticmethod
    def _pick_state(observation: t.Dict[str, t.Any]) -> t.Optional[np.ndarray]:
        """提取并标准化本体状态（必要时自动补齐到 8 维）。"""
        for name in ("state", "robot_state", "proprio"):
            if name in observation and observation[name] is not None:
                state = np.asarray(observation[name], dtype=np.float32).reshape(-1)
                if state.shape[-1] == 7:
                    state = np.concatenate([state[:6], np.zeros(1, dtype=np.float32), state[6:]], axis=0)
                return state
        return None

    @staticmethod
    def _format_action(action: t.Any) -> np.ndarray:
        """将模型输出动作规范化为 ``(chunk, dim)`` 的 float32 数组。"""
        action_np = np.asarray(action, dtype=np.float32)
        if action_np.ndim == 1:
            action_np = action_np[None, :]
        return action_np

    def load_model(self) -> None:
        """加载 VLA-Adapter 模型与处理器。"""
        torch_lib = importlib.import_module("torch")

        workspace_root = Path(__file__).resolve().parents[2]
        eval_module_path = workspace_root / "policy" / "vla-adapter" / "openloop_vlaadapter_eval.py"
        if not eval_module_path.exists():
            raise FileNotFoundError(f"vla-adapter eval module not found: {eval_module_path}")

        self._adapter_module = self._load_eval_module(str(eval_module_path))

        cfg_obj = self._adapter_module.EvalConfig(
            pretrained_checkpoint=self.model_path,
            base_model_checkpoint=self.base_model_checkpoint,
            model_family=self.model_family,
            use_l1_regression=self.use_l1_regression,
            use_minivlm=self.use_minivlm,
            use_pro_version=self.use_pro_version,
            use_proprio=self.use_proprio,
            num_images_in_input=self.num_images_in_input,
            unnorm_key=self.unnorm_key,
            save_version=self.save_version,
        )

        self._cfg = cfg_obj
        (
            self._model,
            self._action_head,
            self._proprio_projector,
            self._noisy_action_projector,
            self._processor,
        ) = self._adapter_module.initialize_model(cfg_obj)

        self._model.eval()
        if torch_lib.cuda.is_available() and self.device.startswith("cuda"):
            self._model = self._model.to(torch_lib.device(self.device))

    def predict(self, observation: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """执行 VLA-Adapter 推理。

        :param observation: 请求观测字典。
        :return: 动作字典 ``{"action": np.ndarray(chunk, dim)}``。
        """
        if self._adapter_module is None or self._model is None:
            raise RuntimeError("VLAAdapterModel is not loaded. Call load_model first.")

        task = observation.get("cmd", "do something")
        cam_head = self._pick_image(observation, ("cam_head", "image_head", "front_image", "full_image"), "head")
        cam_wrist = self._pick_image(observation, ("cam_wrist", "image_wrist", "wrist_image"), "wrist")
        state = self._pick_state(observation)

        input_obs = {
            "full_image": cam_head,
            "image_wrist": cam_wrist,
            "state": state,
        }

        pred_actions = self._adapter_module.get_vla_action(
            self._cfg,
            self._model,
            self._processor,
            input_obs,
            task,
            action_head=self._action_head,
            proprio_projector=self._proprio_projector,
            use_minivlm=self._cfg.use_minivlm,
        )

        return {"action": self._format_action(pred_actions)}
