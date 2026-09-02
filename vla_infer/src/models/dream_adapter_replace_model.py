import typing as t
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from .base import BaseVLAModel
from ..process.replace_weights import WeightReplacementConfig, WeightReplacer

# 导入 VLA-Adapter 相关模块
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
    from experiments.robot.robot_utils import get_model, get_action
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

    # 权重替换配置
    weight_replacement: WeightReplacementConfig = field(default_factory=WeightReplacementConfig)


class DreamAdapterReplaceModel(BaseVLAModel):
    """
    Dream-Adapter server-side wrapper with flexible weight replacement.

    Weight replacement can be triggered after model loading by providing
    a JSON configuration file and optionally a base directory for source
    checkpoints. The JSON file can be either:

    1) Simple format: mapping source file names (may include wildcards) to layer lists.
       The module name is inferred from the file name (e.g., "proprio_projector--20000.pt"
       corresponds to module "proprio_projector").
    2) Detailed format: mapping module names to objects with "file" and "layers" keys.

    Example JSON (detailed):
    {
        "proprio_projector": {
            "file": "proprio_projector--20000.pt",
            "layers": ["module.fc1.weight", "module.fc1.bias", "module.fc2.weight"]
        },
        "action_head": {
            "file": "action_head--20000.pt",
            "layers": ["module.model.layer_norm1.weight", "module.model.layer_norm1.bias"]
        }
    }

    After loading the main model, the specified layers are replaced from the
    corresponding source checkpoint.
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
        proprio_dim: int = 7,
        use_reconstruct_images: bool = True,
        default_instruction: str = "",
        predict_image_frame: int = 1,
        # 权重替换参数
        weight_replacement: t.Optional[WeightReplacementConfig] = None,
    ) -> None:
        # 如果没有提供权重替换配置，创建默认的禁用配置
        if weight_replacement is None:
            weight_replacement = WeightReplacementConfig(enabled=False)
        
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
            predict_image_frame=predict_image_frame,
            weight_replacement=weight_replacement,
        )
        self._model: t.Any = None
        self._action_head: t.Any = None
        self._proprio_projector: t.Any = None
        self._processor: t.Any = None
        self._get_vla_action: t.Any = None
        self._reconstruct_images: t.Any = None
        self._proprio_dim: int = proprio_dim
        self._default_instruction = default_instruction
        # 权重替换器
        self._weight_replacer: t.Optional[WeightReplacer] = None
        super().__init__()

    # ------------------------------------------------------------------
    # 静态辅助方法
    # ------------------------------------------------------------------
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
        unnorm_key = self.cfg.task_suite_name
        if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
        self.cfg.unnorm_key = unnorm_key

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
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

        # 应用权重替换（如果启用）
        self._apply_weight_replacements_if_enabled()

    def _apply_weight_replacements_if_enabled(self) -> None:
        """Apply weight replacements if configured."""
        if not self.cfg.weight_replacement.enabled:
            return

        print("\n" + "="*80)
        print("🔄 Starting weight replacement process...")
        print("="*80)

        # 创建权重替换器
        self._weight_replacer = WeightReplacer(self.cfg.weight_replacement)

        # 注册模块
        if self._proprio_projector is not None:
            self._weight_replacer.register_module("proprio_projector", self._proprio_projector)
            print("  ✓ Registered module: proprio_projector")
        if self._action_head is not None:
            self._weight_replacer.register_module("action_head", self._action_head)
            print("  ✓ Registered module: action_head")
        if self._reconstruct_images is not None:
            self._weight_replacer.register_module("reconstruct_images", self._reconstruct_images)
            print("  ✓ Registered module: reconstruct_images")

        print("\n  Starting weight replacement...")
        # 执行替换
        self._weight_replacer.apply_replacements()
        
        print("="*80)
        print("✅ Weight replacement completed successfully!")
        print("="*80 + "\n")

    # ------------------------------------------------------------------
    # 推理方法
    # ------------------------------------------------------------------
    def _predict_action_chunk_array(self, observation: t.Dict[str, t.Any]) -> np.ndarray:
        self._ensure_loaded()

        cmd = str(observation.get("cmd", self._default_instruction) or self._default_instruction)
        image = self._validate_rgb_image("image", observation.get("image"))
        wrist_image = self._validate_rgb_image("wrist_image", observation.get("wrist_image"))
        state_7d = self._validate_state(observation.get("state"), self._proprio_dim)

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
        """Run VLA-Adapter inference."""
        use_chunk = bool(observation.get("return_action_chunk", True))
        if use_chunk:
            action = self.predict_action_chunk(observation)
        else:
            action = self.predict_action(observation)
        return {"action": action}