from __future__ import annotations

from dataclasses import dataclass
import logging
import typing as t

import numpy as np

from .base import BaseVLAModel


@dataclass
class SmolVLAModelConfig:
    """Model loading config aligned with LeRobot record pipeline."""

    model_path: str
    device: str = "cuda"
    dataset_repo_id: t.Optional[str] = None
    dataset_root: t.Optional[str] = None
    action_chunk_size: t.Optional[int] = None
    default_instruction: str = ""


class SmolVLAModel(BaseVLAModel):
    """SmolVLA server-side wrapper using official LeRobot inference pipeline.

    Request payload preferred format::

        {
            "cmd": "pick up the banana",
            "image": np.ndarray(H, W, 3),
            "wrist_image": np.ndarray(H, W, 3),
            "state": np.ndarray(7,)
        }
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dataset_repo_id: t.Optional[str] = None,
        dataset_root: t.Optional[str] = None,
        action_chunk_size: t.Optional[int] = None,
        default_instruction: str = "",
    ) -> None:
        """Initialize SmolVLA wrapper.

        Notes:
        - This class assumes user already added the `lerobot` project root to
          `sys.path`, so standard imports like `import lerobot` work.
        - `dataset_repo_id`/`dataset_root` are kept for compatibility but are
          not required for inference with pretrained processors.
        """
        self.cfg = SmolVLAModelConfig(
            model_path=model_path,
            device=device,
            dataset_repo_id=dataset_repo_id,
            dataset_root=dataset_root,
            action_chunk_size=action_chunk_size,
            default_instruction=default_instruction,
        )

        self.model_path = self.cfg.model_path
        self.device = self.cfg.device
        self.dataset_repo_id = self.cfg.dataset_repo_id
        self.dataset_root = self.cfg.dataset_root
        self.action_chunk_size = self.cfg.action_chunk_size

        self._default_instruction = self.cfg.default_instruction

        self._input_mapping: t.Optional[t.Dict[str, t.Optional[str]]] = None
        self._policy_cfg: t.Any = None
        self._policy: t.Any = None
        self._preprocessor: t.Any = None
        self._postprocessor: t.Any = None

        self._torch: t.Any = None
        super().__init__()

    @staticmethod
    def _ensure_writable_contiguous_array(value: t.Any, dtype: t.Any = None) -> np.ndarray:
        arr = np.asarray(value, dtype=dtype)
        # PyTorch warns on non-writable numpy buffers (common with frombuffer/ZeroMQ payloads).
        if (not arr.flags.writeable) or (not arr.flags.c_contiguous) or any(step < 0 for step in arr.strides):
            arr = np.array(arr, copy=True, order="C")
        return arr

    @staticmethod
    def _validate_rgb_image(name: str, value: t.Any) -> np.ndarray:
        image = SmolVLAModel._ensure_writable_contiguous_array(value)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{name} must be HxWx3 RGB array, got shape={image.shape}")
        return image

    @staticmethod
    def _to_bchw_float_image(name: str, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            image_f32 = image.astype(np.float32) / 255.0
        elif np.issubdtype(image.dtype, np.floating):
            image_f32 = image.astype(np.float32, copy=False)
            min_value = float(np.min(image_f32))
            max_value = float(np.max(image_f32))
            if min_value < -1e-6 or max_value > 1.0 + 1e-6:
                raise ValueError(
                    f"{name} float image must be in [0, 1], got min={min_value:.6f}, max={max_value:.6f}"
                )
        else:
            raise ValueError(f"{name} image dtype must be uint8 or float, got {image.dtype}")

        # SmolVLA policy expects BCHW image tensors in [0,1].
        image_bchw = np.transpose(image_f32, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(image_bchw, dtype=np.float32)

    @staticmethod
    def _validate_state(value: t.Any) -> np.ndarray:
        state = SmolVLAModel._ensure_writable_contiguous_array(value, dtype=np.float32).reshape(-1)
        if state.shape[0] != 7:
            raise ValueError(f"state must be shape (7,), got {state.shape}")
        return state

    @staticmethod
    def _to_action_array(action: t.Any) -> np.ndarray:
        action_np = np.asarray(action, dtype=np.float32)
        if action_np.ndim == 3 and action_np.shape[0] == 1:
            action_np = action_np[0]
        if action_np.ndim == 1:
            action_np = action_np[None, :]
        if action_np.ndim != 2:
            raise ValueError(f"Expected action with shape (T, D) or (D,), got {action_np.shape}")
        return action_np

    @staticmethod
    def _candidate_input_keys(input_features: t.Dict[str, t.Any]) -> t.Tuple[t.List[str], t.List[str]]:
        image_keys = [k for k in input_features.keys() if k.startswith("observation.images.")]
        state_keys = [k for k in input_features.keys() if k.endswith(".state")]
        return image_keys, state_keys

    @staticmethod
    def _resolve_input_mapping(input_features: t.Dict[str, t.Any]) -> t.Dict[str, t.Optional[str]]:
        image_keys, state_keys = SmolVLAModel._candidate_input_keys(input_features)
        if not image_keys:
            raise RuntimeError("SmolVLA config has no image input features under `observation.images.*`")

        head_key = "observation.images.image" if "observation.images.image" in image_keys else image_keys[0]

        wrist_key: t.Optional[str] = None
        for key in image_keys:
            if key == head_key:
                continue
            if "wrist" in key:
                wrist_key = key
                break
        if wrist_key is None:
            for key in image_keys:
                if key != head_key:
                    wrist_key = key
                    break

        state_key: t.Optional[str] = "observation.state" if "observation.state" in state_keys else None
        if state_key is None and state_keys:
            state_key = state_keys[0]

        return {
            "head_image_key": head_key,
            "wrist_image_key": wrist_key,
            "state_key": state_key,
        }

    def _build_policy_input(
        self,
        cmd: str,
        image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
    ) -> t.Dict[str, t.Any]:
        if self._input_mapping is None:
            raise RuntimeError("Input mapping is not initialized. Call load_model first.")
        if self._torch is None:
            raise RuntimeError("Torch is not initialized. Call load_model first.")

        def _to_device_tensor(array: np.ndarray) -> t.Any:
            tensor = self._torch.from_numpy(np.ascontiguousarray(array))
            return tensor.to(self.device)

        payload: t.Dict[str, t.Any] = {
            t.cast(str, self._input_mapping["head_image_key"]): _to_device_tensor(image),
            "task": cmd,
        }

        wrist_image_key = self._input_mapping["wrist_image_key"]
        if wrist_image_key is not None:
            payload[wrist_image_key] = _to_device_tensor(wrist_image)

        state_key = self._input_mapping["state_key"]
        if state_key is not None:
            payload[state_key] = _to_device_tensor(state)

        return payload

    def _ensure_loaded(self) -> None:
        if self._policy is None or self._preprocessor is None or self._postprocessor is None:
            raise RuntimeError("SmolVLAModel is not initialized. Call load_model first.")

    def load_model(self) -> None:
        """Load SmolVLA policy and processors from LeRobot pretrained artifacts."""
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            try:
                # LeRobot <= v0.4.x
                from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            except ImportError:
                # LeRobot >= v0.5.x
                raise ImportError("LeRobotDatasetMetadata not found in lerobot.datasets.lerobot_dataset")
            from lerobot.policies.factory import (
                get_policy_class,
                make_pre_post_processors,
                make_policy,
            )
        except Exception as exc:
            raise ImportError(
                "Failed to import LeRobot SmolVLA modules. "
                "Please add lerobot repo root to sys.path before creating SmolVLAModel."
            ) from exc

        self._torch = torch
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Keep this sequence close to lerobot_record.py for easier cross-debugging.
        policy_cfg = PreTrainedConfig.from_pretrained(self.model_path)
        policy_cfg.pretrained_path = self.model_path
        if getattr(policy_cfg, "device", None) != self.device:
            policy_cfg.device = self.device

        policy = None
        ds_meta = None
        if self.dataset_repo_id is not None:
            try:
                ds_meta = LeRobotDatasetMetadata(repo_id=self.dataset_repo_id, root=self.dataset_root)
            except Exception:
                logging.exception(
                    "Failed to load dataset metadata from repo_id=%s root=%s. "
                    "Falling back to checkpoint-only policy loading.",
                    self.dataset_repo_id,
                    self.dataset_root,
                )
                ds_meta = None

        # Primary path: same factory used by lerobot_record.py when ds_meta is available.
        if ds_meta is not None:
            policy = make_policy(policy_cfg, ds_meta=ds_meta)
        else:
            # Fallback path for deployment when only a checkpoint directory is provided.
            policy_cls = get_policy_class(policy_cfg.type)
            policy = policy_cls.from_pretrained(
                pretrained_name_or_path=self.model_path,
                config=policy_cfg,
            )

        if self.action_chunk_size is not None:
            policy.config.n_action_steps = int(self.action_chunk_size)
            policy.reset()

        policy = policy.to(self.device)
        policy.eval()

        # Keep pre/post processors aligned with pretrained artifacts (same as lerobot_record).
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=self.model_path,
        )

        self._policy_cfg = policy_cfg
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._input_mapping = self._resolve_input_mapping(policy.config.input_features)

    def predict_action_chunk(self, observation: t.Dict[str, t.Any]) -> np.ndarray:
        """Predict an action chunk with shape (T, D)."""
        self._ensure_loaded()

        cmd = str(observation.get("cmd", self._default_instruction) or self._default_instruction)
        image = self._to_bchw_float_image("image", self._validate_rgb_image("image", observation.get("image")))
        wrist_image = self._to_bchw_float_image(
            "wrist_image", self._validate_rgb_image("wrist_image", observation.get("wrist_image"))
        )
        state = self._validate_state(observation.get("state"))

        payload = self._build_policy_input(cmd=cmd, image=image, wrist_image=wrist_image, state=state)
        with self._torch.no_grad():
            model_input = self._preprocessor(payload)
            action_chunk = self._policy.predict_action_chunk(model_input)
            action_chunk = self._postprocessor(action_chunk)

        return self._to_action_array(action_chunk)

    def predict_action(self, observation: t.Dict[str, t.Any]) -> np.ndarray:
        """Predict a single action, returned as shape (1, D)."""
        self._ensure_loaded()

        cmd = str(observation.get("cmd", self._default_instruction) or self._default_instruction)
        image = self._to_bchw_float_image("image", self._validate_rgb_image("image", observation.get("image")))
        wrist_image = self._to_bchw_float_image(
            "wrist_image", self._validate_rgb_image("wrist_image", observation.get("wrist_image"))
        )
        state = self._validate_state(observation.get("state"))

        payload = self._build_policy_input(cmd=cmd, image=image, wrist_image=wrist_image, state=state)
        with self._torch.no_grad():
            model_input = self._preprocessor(payload)
            action = self._policy.select_action(model_input)
            action = self._postprocessor(action)

        return self._to_action_array(action)

    def predict(self, observation: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """Run SmolVLA inference.

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
