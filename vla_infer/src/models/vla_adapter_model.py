from __future__ import annotations

from dataclasses import dataclass
import typing as t

import numpy as np

from .base import BaseVLAModel


@dataclass
class _AdapterEvalConfig:
    pretrained_checkpoint: str
    base_model_checkpoint: t.Optional[str] = None
    model_family: str = "openvla"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_minivlm: bool = True
    use_pro_version: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 1
    unnorm_key: str = "pick_banana_50"
    save_version: str = "vla-adapter"


class VLAAdapterModel(BaseVLAModel):
    """VLA-Adapter server-side wrapper aligned with official Dream-Adapter inference.

    Expected observation format::

        {
            "cmd": "pick up the banana",
            "image": np.ndarray(H, W, 3),
            "wrist_image": np.ndarray(H, W, 3),
            "state": np.ndarray(7,)
        }

    Notes:
    - This class does not modify ``sys.path``.
    - User should add the VLA-Adapter repo root to ``sys.path`` externally,
      then imports like ``experiments.robot.openvla_utils`` resolve normally.
    - 7-dim state is converted to 8-dim for VLA-Adapter proprio projector.
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
        num_open_loop_steps: int = 1,
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
        self.num_open_loop_steps = num_open_loop_steps
        self.unnorm_key = unnorm_key
        self.save_version = save_version

        self._cfg: t.Optional[_AdapterEvalConfig] = None
        self._model: t.Any = None
        self._action_head: t.Any = None
        self._proprio_projector: t.Any = None
        self._noisy_action_projector: t.Any = None
        self._processor: t.Any = None
        self._get_vla_action: t.Any = None
        self._torch: t.Any = None

        self._default_instruction = ""
        super().__init__(model_path=model_path, device=device)

    @staticmethod
    def _validate_rgb_image(name: str, value: t.Any) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{name} must be numpy.ndarray, got {type(value)}")
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"{name} must be HxWx3 RGB array, got shape={value.shape}")
        if value.strides is not None and any(step < 0 for step in value.strides):
            return value.copy()
        return value

    @staticmethod
    def _validate_state_7d(value: t.Any) -> np.ndarray:
        state = np.asarray(value, dtype=np.float32).reshape(-1)
        if state.shape[0] != 7:
            raise ValueError(f"state must be shape (7,), got {state.shape}")
        return state

    @staticmethod
    def _state_7d_to_8d(state_7d: np.ndarray) -> np.ndarray:
        return np.concatenate([state_7d[:6], np.zeros(1, dtype=np.float32), state_7d[6:]], axis=0)

    @staticmethod
    def _to_action_array(action: t.Any) -> np.ndarray:
        action_np = np.asarray(action, dtype=np.float32)
        if action_np.ndim == 3 and action_np.shape[1] == 1:
            action_np = action_np[:, 0, :]
        if action_np.ndim == 1:
            action_np = action_np[None, :]
        if action_np.ndim != 2:
            raise ValueError(f"Expected action shape (T, D) or (D,), got {action_np.shape}")
        return action_np

    def _ensure_loaded(self) -> None:
        if self._cfg is None or self._model is None or self._get_vla_action is None:
            raise RuntimeError("VLAAdapterModel is not initialized. Call load_model first.")

    def _build_cfg(self) -> _AdapterEvalConfig:
        return _AdapterEvalConfig(
            pretrained_checkpoint=self.model_path,
            base_model_checkpoint=self.base_model_checkpoint,
            model_family=self.model_family,
            use_l1_regression=self.use_l1_regression,
            use_minivlm=self.use_minivlm,
            use_pro_version=self.use_pro_version,
            num_images_in_input=self.num_images_in_input,
            use_proprio=self.use_proprio,
            num_open_loop_steps=max(1, int(self.num_open_loop_steps)),
            unnorm_key=self.unnorm_key,
            save_version=self.save_version,
        )

    @staticmethod
    def _resolve_llm_dim(model: t.Any) -> int:
        llm_dim = getattr(model, "llm_dim", None)
        if isinstance(llm_dim, int):
            return llm_dim

        text_cfg = getattr(getattr(model, "config", None), "text_config", None)
        hidden_size = getattr(text_cfg, "hidden_size", None)
        if isinstance(hidden_size, int):
            return hidden_size
        return 4096

    def load_model(self) -> None:
        """Load VLA-Adapter model and inference components."""
        try:
            import torch
            from experiments.robot.openvla_utils import (
                get_action_head,
                get_processor,
                get_proprio_projector,
                get_vla_action,
            )
            from experiments.robot.robot_utils import get_model
        except Exception as exc:
            raise ImportError(
                "Failed to import VLA-Adapter modules. Please add the VLA-Adapter repo root "
                "to sys.path before creating VLAAdapterModel."
            ) from exc

        self._torch = torch
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"

        cfg = self._build_cfg()
        model = get_model(cfg)
        if hasattr(model, "set_version"):
            model.set_version(cfg.save_version)

        llm_dim = self._resolve_llm_dim(model)

        proprio_projector = None
        if cfg.use_proprio:
            proprio_projector = get_proprio_projector(cfg, llm_dim, proprio_dim=8)

        action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            action_head = get_action_head(cfg, llm_dim)

        processor = get_processor(cfg) if cfg.model_family == "openvla" else None

        if hasattr(model, "to"):
            model = model.to(self.device)
        model.eval()

        self._cfg = cfg
        self._model = model
        self._action_head = action_head
        self._proprio_projector = proprio_projector
        self._noisy_action_projector = None
        self._processor = processor
        self._get_vla_action = get_vla_action

    def _predict_action_chunk_array(self, observation: t.Dict[str, t.Any]) -> np.ndarray:
        self._ensure_loaded()
        cfg = t.cast(_AdapterEvalConfig, self._cfg)

        cmd = str(observation.get("cmd", self._default_instruction) or self._default_instruction)
        image = self._validate_rgb_image("image", observation.get("image"))
        wrist_image = self._validate_rgb_image("wrist_image", observation.get("wrist_image"))
        state_7d = self._validate_state_7d(observation.get("state"))

        policy_obs: t.Dict[str, t.Any] = {
            "full_image": image,
            "image_wrist": wrist_image,
        }
        if cfg.use_proprio:
            policy_obs["state"] = self._state_7d_to_8d(state_7d)

        pred_actions = self._get_vla_action(
            cfg,
            self._model,
            self._processor,
            policy_obs,
            cmd,
            action_head=self._action_head,
            proprio_projector=self._proprio_projector,
            noisy_action_projector=self._noisy_action_projector,
            use_film=cfg.use_film,
            use_minivlm=cfg.use_minivlm,
        )
        return self._to_action_array(pred_actions)

    def predict_action_chunk(self, observation: t.Dict[str, t.Any]) -> np.ndarray:
        """Predict action chunk, returns shape (T, D)."""
        return self._predict_action_chunk_array(observation)

    def predict_action(self, observation: t.Dict[str, t.Any]) -> np.ndarray:
        """Predict single-step action, returns shape (1, D)."""
        action_chunk = self._predict_action_chunk_array(observation)
        return action_chunk[:1]

    def predict(self, observation: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """Run VLA-Adapter inference.

        Behavior:
        - default: return action chunk (`predict_action_chunk`).
        - when `observation.get("return_action_chunk") is False`: return single-step action.
        """
        use_chunk = bool(observation.get("return_action_chunk", True))
        if use_chunk:
            action = self.predict_action_chunk(observation)
        else:
            action = self.predict_action(observation)
        return {"action": action}
