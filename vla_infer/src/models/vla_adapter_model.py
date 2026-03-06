from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import typing as t

import numpy as np

from .base import BaseVLAModel


@dataclass
class _AdapterEvalConfig:
    pretrained_checkpoint: str
    base_model_checkpoint: t.Optional[str] = None
    model_family: str = "openvla"
    use_l1_regression: bool = True
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
    """VLA-Adapter server-side wrapper with in-file loading/inference logic.

    This implementation extracts model initialization and action inference flow
    from `policy/vla-adapter/openloop_vlaadapter_eval.py` and keeps it local to
    vla_infer, so we do not import that policy script directly.

    Request payload preferred format::

        {
            "cmd": "pick up the banana",
            "image": np.ndarray(H, W, 3),
            "wrist_image": np.ndarray(H, W, 3),
            "state": np.ndarray(7,)
        }

    Note:
    - When state is 7-dim ( 6 d + gripper), this wrapper pads to 8-dim
      as required by the VLA-Adapter proprio projector.
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

        self._cfg: t.Optional[_AdapterEvalConfig] = None
        self._model: t.Any = None
        self._action_head: t.Any = None
        self._proprio_projector: t.Any = None
        self._noisy_action_projector: t.Any = None
        self._processor: t.Any = None

        self._torch: t.Any = None
        self._get_vla_action: t.Any = None
        super().__init__(model_path=model_path, device=device)

    @staticmethod
    def _resolve_vla_adapter_root() -> t.Optional[Path]:
        for candidate_root in Path(__file__).resolve().parents:
            candidate = candidate_root / "VLA-Adapter"
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    def _prepare_vla_adapter_import_path(self) -> None:
        vla_adapter_root = self._resolve_vla_adapter_root()
        if vla_adapter_root is None:
            return

        path_str = str(vla_adapter_root)
        if path_str not in sys.path:
            sys.path.append(path_str)

    @staticmethod
    def _pick_image(observation: t.Dict[str, t.Any], names: t.Sequence[str], logical_name: str) -> np.ndarray:
        for name in names:
            if name in observation:
                image = observation[name]
                if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[-1] == 3:
                    return image
                raise ValueError(f"{logical_name} image must be HxWx3 numpy.ndarray, got {type(image)}")
        raise KeyError(f"Missing {logical_name} image, candidate keys: {list(names)}")

    @staticmethod
    def _pick_state(observation: t.Dict[str, t.Any]) -> t.Optional[np.ndarray]:
        for name in ("state", "robot_state", "proprio"):
            if name in observation and observation[name] is not None:
                state = np.asarray(observation[name], dtype=np.float32).reshape(-1)
                if state.shape[-1] == 7:
                    state = np.concatenate([state[:6], np.zeros(1, dtype=np.float32), state[6:]], axis=0)
                return state
        return None

    @staticmethod
    def _format_action(action: t.Any) -> np.ndarray:
        action_np = np.asarray(action, dtype=np.float32)
        if action_np.ndim == 1:
            action_np = action_np[None, :]
        return action_np

    def _build_cfg(self) -> _AdapterEvalConfig:
        return _AdapterEvalConfig(
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

    def _initialize_model_from_cfg(self, cfg: _AdapterEvalConfig) -> t.Tuple[t.Any, t.Any, t.Any, t.Any, t.Any]:
        self._prepare_vla_adapter_import_path()

        openvla_utils = importlib.import_module("experiments.robot.openvla_utils")
        robot_utils = importlib.import_module("experiments.robot.robot_utils")

        get_action_head = getattr(openvla_utils, "get_action_head")
        get_processor = getattr(openvla_utils, "get_processor")
        get_proprio_projector = getattr(openvla_utils, "get_proprio_projector")
        get_model = getattr(robot_utils, "get_model")

        model = get_model(cfg)
        if hasattr(model, "set_version"):
            model.set_version(cfg.save_version)

        proprio_projector = None
        if cfg.use_proprio:
            try:
                llm_dim = model.config.text_config.hidden_size
            except Exception:
                llm_dim = 4096
            proprio_projector = get_proprio_projector(cfg, llm_dim, proprio_dim=8)

        action_head = None
        if cfg.use_l1_regression:
            try:
                llm_dim = model.config.text_config.hidden_size
            except Exception:
                llm_dim = 4096
            action_head = get_action_head(cfg, llm_dim)

        noisy_action_projector = None
        processor = get_processor(cfg) if cfg.model_family == "openvla" else None

        return model, action_head, proprio_projector, noisy_action_projector, processor

    def load_model(self) -> None:
        self._torch = importlib.import_module("torch")
        self._prepare_vla_adapter_import_path()

        openvla_utils = importlib.import_module("experiments.robot.openvla_utils")
        get_vla_action = getattr(openvla_utils, "get_vla_action")

        self._get_vla_action = get_vla_action
        self._cfg = self._build_cfg()

        (
            self._model,
            self._action_head,
            self._proprio_projector,
            self._noisy_action_projector,
            self._processor,
        ) = self._initialize_model_from_cfg(self._cfg)

        self._model.eval()
        if self._torch.cuda.is_available() and str(self.device).startswith("cuda"):
            self._model = self._model.to(self._torch.device(self.device))

    def predict(self, observation: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        if self._cfg is None or self._model is None or self._get_vla_action is None:
            raise RuntimeError("VLAAdapterModel is not loaded. Call load_model first.")

        task = str(observation.get("cmd", "do something"))
        image = self._pick_image(
            observation,
            ("image", "cam_head", "image_head", "front_image", "full_image"),
            "head",
        )
        wrist_image = self._pick_image(
            observation,
            ("wrist_image", "image_wrist", "cam_wrist"),
            "wrist",
        )
        state = self._pick_state(observation)

        if self._cfg.use_proprio and state is None:
            raise ValueError("Missing proprio state while use_proprio=True")

        input_obs = {
            "full_image": image,
            "image_wrist": wrist_image,
            "state": state,
        }

        pred_actions = self._get_vla_action(
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
