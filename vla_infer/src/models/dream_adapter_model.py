import typing as t
import numpy as np
from .base import BaseVLAModel
from dataclasses import dataclass
from pathlib import Path

get_reconstruct_images: t.Optional[t.Callable[..., t.Any]] = None

try:
    from experiments.robot.openvla_utils import (
        get_action_head,
        get_processor,
        get_proprio_projector,
    )
    from experiments.robot import openvla_utils as _openvla_utils

    get_reconstruct_images = t.cast(
        t.Optional[t.Callable[..., t.Any]],
        getattr(_openvla_utils, "get_reconstruct_images", None),
    )
    from experiments.robot.robot_utils import get_model,get_action
except Exception as exc:
    raise ImportError(
        "Failed to import VLA-Adapter modules. Please add the VLA-Adapter repo root "
        "to sys.path before creating VLAAdapterModel."
    ) from exc
    
@dataclass
class DreamAdapterModelConfig:
    pretrained_checkpoint: t.Union[str, Path] = ""
    model_family: str = "openvla"
    use_l1_regression: bool = True
    use_minivlm: bool = True
    use_pro_version: bool = True
    use_proprio: bool = True
    num_images_in_input: int = 2
    num_open_loop_steps: int = 8
    load_in_8bit: bool = False     
    load_in_4bit: bool = False
    task_suite_name: str = ""
    save_version: str = "vla-adapter"
    unnorm_key: str = ""
    use_film: bool = False
    use_reconstruct_images: bool = True
    center_crop: bool = False
    predict_image_frame: int = 1
    proprio_dim: int = 7
class DreamAdapterModel(BaseVLAModel):
    """Dream-Adapter server-side wrapper aligned with official Dream-Adapter inference.

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
        - Proprio state is adapted to the model's expected ``PROPRIO_DIM`` (typically 7 or 8).
    """

    def __init__(
        self,
        pretrained_checkpoint: t.Union[str, Path] = "",
        model_family: str = "openvla",
        use_l1_regression: bool = True,
        use_minivlm: bool = True,   
        use_pro_version: bool = True,
        use_proprio: bool = True,
        num_images_in_input: int = 2,
        num_open_loop_steps: int = 8,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        task_suite_name: str = "",
        save_version: str = "vla-adapter",
        use_film: bool = False,
        proprio_dim : int = 7,
        use_reconstruct_images: bool = True,
        default_instruction: str = "",
        predict_image_frame: int = 1       
    ) -> None:
        self.cfg = DreamAdapterModelConfig(
            pretrained_checkpoint=pretrained_checkpoint,
            model_family=model_family,
            use_l1_regression=use_l1_regression,
            use_minivlm=use_minivlm,    
            use_pro_version=use_pro_version,
            use_proprio=use_proprio,
            num_images_in_input=num_images_in_input,
            num_open_loop_steps=num_open_loop_steps,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            save_version=save_version,
            task_suite_name=task_suite_name,
            proprio_dim=proprio_dim,
            use_reconstruct_images=use_reconstruct_images,
            use_film=use_film,
            predict_image_frame=predict_image_frame
        )
        self._model: t.Any = None
        self._action_head: t.Any = None
        self._proprio_projector: t.Any = None
        self._processor: t.Any = None
        self._get_vla_action: t.Any = None
        self._reconstruct_images: t.Any = None
        self._proprio_dim: int = proprio_dim
        self._default_instruction = default_instruction
        super().__init__()

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
    def _validate_state(state: t.Any, proprio_dim: int) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != proprio_dim:
            raise ValueError(f"state must be shape ({proprio_dim},), got {state.shape}")
        return state

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
        if self.cfg is None or self._model is None or self._get_vla_action is None:
            raise RuntimeError("VLAAdapterModel is not initialized. Call load_model first.")

    @staticmethod
    def _resolve_llm_dim(model: t.Any) -> int:
        llm_dim = getattr(model, "llm_dim", None)
        if isinstance(llm_dim, int):
            return llm_dim
        raise ValueError("Failed to resolve LLM dimension.")
    def check_unnorm_key(self, model) -> None:
        """Check that the model contains the action un-normalization key."""
        # Initialize unnorm_key
        unnorm_key = self.cfg.task_suite_name

        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"

        assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

        # Set the unnorm_key in cfg
        self.cfg.unnorm_key = unnorm_key
    
    def load_model(self) -> None:
        """Load VLA-Adapter model and inference components."""
        self._model = get_model(self.cfg)
        if hasattr(self._model, "set_version"):
            self._model.set_version(self.cfg.save_version)
        llm_dim = self._resolve_llm_dim(self._model)
        if self.cfg.use_proprio:
            self._proprio_projector = get_proprio_projector(self.cfg, llm_dim, proprio_dim=self._proprio_dim)

        if self.cfg.use_l1_regression:
            self._action_head = get_action_head(self.cfg, llm_dim)
            
        if self.cfg.model_family == "openvla":
            self._processor = get_processor(self.cfg)
            self.check_unnorm_key(self._model)

        if self.cfg.use_reconstruct_images:
            if get_reconstruct_images is None:
                raise ImportError(
                    "`get_reconstruct_images` is not available in the current "
                    "VLA-Adapter repo version. Set use_reconstruct_images=False "
                    "or upgrade experiments.robot.openvla_utils."
                )
            self._reconstruct_images = get_reconstruct_images(
                self.cfg,
                self._model.llm_dim,
                image_dim=588,
                predict_image_frame=self.cfg.predict_image_frame,
            )
    
        self._get_vla_action = get_action

    def _predict_action_chunk_array(self, observation: t.Dict[str, t.Any]) -> np.ndarray:
        self._ensure_loaded()

        cmd = str(observation.get("cmd", self._default_instruction) or self._default_instruction)
        image = self._validate_rgb_image("image", observation.get("image"))
        wrist_image = self._validate_rgb_image("wrist_image", observation.get("wrist_image"))
        state_7d = self._validate_state(observation.get("state"), self._proprio_dim)
        # convert obs dict to model input format, run inference, and post-process action output
        policy_obs: t.Dict[str, t.Any] = {
            "full_image": image,
            "image_wrist": wrist_image,
        }
        if self.cfg.use_proprio:
            policy_obs["state"] = state_7d
            
        pred_actions = self._get_vla_action(
            cfg=self.cfg,
            model=self._model,
            processor=self._processor,
            obs=policy_obs,
            task_label=cmd,
            action_head=self._action_head,
            proprio_projector=self._proprio_projector,
            reconstruct_images=self._reconstruct_images,
            use_film=self.cfg.use_film,
            use_minivlm=self.cfg.use_minivlm,
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
