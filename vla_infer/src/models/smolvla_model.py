from __future__ import annotations

from dataclasses import dataclass
import typing as t

import numpy as np

from .base import BaseVLAModel


@dataclass
class _SmolVLAInputMapping:
    head_image_key: str
    wrist_image_key: t.Optional[str]
    state_key: t.Optional[str]


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
    ) -> None:
        """Initialize SmolVLA wrapper.

        Notes:
        - This class assumes user already added the `lerobot` project root to
          `sys.path`, so standard imports like `import lerobot` work.
        - `dataset_repo_id`/`dataset_root` are kept for compatibility but are
          not required for inference with pretrained processors.
        """
        self.dataset_repo_id = dataset_repo_id
        self.dataset_root = dataset_root
        self.action_chunk_size = action_chunk_size

        self._default_instruction = ""

        self._input_mapping: t.Optional[_SmolVLAInputMapping] = None
        self._policy: t.Any = None
        self._preprocessor: t.Any = None
        self._postprocessor: t.Any = None

        self._torch: t.Any = None
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
    def _validate_state(value: t.Any) -> np.ndarray:
        state = np.asarray(value, dtype=np.float32).reshape(-1)
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
    def _resolve_input_mapping(input_features: t.Dict[str, t.Any]) -> _SmolVLAInputMapping:
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

        return _SmolVLAInputMapping(
            head_image_key=head_key,
            wrist_image_key=wrist_key,
            state_key=state_key,
        )

    def _build_policy_input(
        self,
        cmd: str,
        image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
    ) -> t.Dict[str, t.Any]:
        if self._input_mapping is None:
            raise RuntimeError("Input mapping is not initialized. Call load_model first.")

        payload: t.Dict[str, t.Any] = {
            self._input_mapping.head_image_key: image,
            "task": cmd,
        }

        if self._input_mapping.wrist_image_key is not None:
            payload[self._input_mapping.wrist_image_key] = wrist_image

        if self._input_mapping.state_key is not None:
            payload[self._input_mapping.state_key] = state

        return payload

    def _ensure_loaded(self) -> None:
        if self._policy is None or self._preprocessor is None or self._postprocessor is None:
            raise RuntimeError("SmolVLAModel is not initialized. Call load_model first.")

    def load_model(self) -> None:
        """Load SmolVLA policy and processors from LeRobot pretrained artifacts."""
        try:
            import torch
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except Exception as exc:
            raise ImportError(
                "Failed to import LeRobot SmolVLA modules. "
                "Please add lerobot repo root to sys.path before creating SmolVLAModel."
            ) from exc

        self._torch = torch
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        policy = SmolVLAPolicy.from_pretrained(self.model_path)
        if self.action_chunk_size is not None:
            policy.config.n_action_steps = int(self.action_chunk_size)
            policy.reset()

        policy = policy.to(self.device)
        policy.eval()

        # Keep pre/post processors aligned with the pretrained model card config.
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=self.model_path,
        )

        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._input_mapping = self._resolve_input_mapping(policy.config.input_features)

    def predict_action_chunk(self, observation: t.Dict[str, t.Any]) -> np.ndarray:
        """Predict an action chunk with shape (T, D)."""
        self._ensure_loaded()

        cmd = str(observation.get("cmd", self._default_instruction) or self._default_instruction)
        image = self._validate_rgb_image("image", observation.get("image"))
        wrist_image = self._validate_rgb_image("wrist_image", observation.get("wrist_image"))
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
        image = self._validate_rgb_image("image", observation.get("image"))
        wrist_image = self._validate_rgb_image("wrist_image", observation.get("wrist_image"))
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
