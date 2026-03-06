from __future__ import annotations

import typing as t
import importlib

import numpy as np

from .base import BaseVLAModel


class SmolVLAModel(BaseVLAModel):
    """SmolVLA server-side wrapper with in-file loading/inference logic.

    This implementation extracts model loading and action inference flow from
    `policy/smolvla/inference_model.py` and keeps it local to vla_infer, so we
    do not import that policy script directly.

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
        self.dataset_repo_id = dataset_repo_id
        self.dataset_root = dataset_root
        self.action_chunk_size = action_chunk_size

        self._instruction = "pick up object"
        self._use_lerobot = False

        self._policy: t.Any = None
        self._preprocessor: t.Any = None
        self._postprocessor: t.Any = None

        self._processor: t.Any = None
        self._vla: t.Any = None

        self._torch: t.Any = None
        super().__init__(model_path=model_path, device=device)

    @staticmethod
    def _pick_first_image(
        observation: t.Dict[str, t.Any],
        candidate_keys: t.Sequence[str],
        logical_name: str,
    ) -> np.ndarray:
        for key in candidate_keys:
            if key in observation:
                image = observation[key]
                if not isinstance(image, np.ndarray):
                    raise ValueError(f"{logical_name} image must be numpy.ndarray, got {type(image)}")
                if image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError(f"{logical_name} image must be HxWx3 RGB array, got shape={image.shape}")
                return image
        raise KeyError(f"Missing {logical_name} image. tried keys={list(candidate_keys)}")

    @staticmethod
    def _pick_state(observation: t.Dict[str, t.Any]) -> t.Optional[np.ndarray]:
        for key in ("state", "robot_state", "proprio"):
            if key in observation and observation[key] is not None:
                return np.asarray(observation[key], dtype=np.float32).reshape(-1)
        return None

    @staticmethod
    def _normalize_action(action: t.Any) -> np.ndarray:
        action_np = np.asarray(action, dtype=np.float32)
        if action_np.ndim == 1:
            action_np = np.expand_dims(action_np, axis=0)
        return action_np

    @staticmethod
    def _ensure_no_negative_strides(array: np.ndarray) -> np.ndarray:
        if array.strides is not None and any(step < 0 for step in array.strides):
            return array.copy()
        return array

    def load_model(self) -> None:
        """Load SmolVLA model.

        Priority:
        - LeRobot pipeline when available and dataset_repo_id is provided.
        - Fallback to raw Transformers pipeline.
        """
        import torch

        self._torch = torch
        runtime_device = self.device if self.device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = runtime_device

        lerobot_available = False
        if self.dataset_repo_id is not None:
            try:
                pretrain_cfg_module = importlib.import_module("lerobot.configs.policies")
                dataset_module = importlib.import_module("lerobot.datasets.lerobot_dataset")
                policy_factory_module = importlib.import_module("lerobot.policies.factory")

                PreTrainedConfig = getattr(pretrain_cfg_module, "PreTrainedConfig")
                LeRobotDataset = getattr(dataset_module, "LeRobotDataset")
                make_policy = getattr(policy_factory_module, "make_policy")
                make_pre_post_processors = getattr(policy_factory_module, "make_pre_post_processors")

                cfg = PreTrainedConfig.from_pretrained(self.model_path)
                if self.action_chunk_size is not None:
                    cfg.n_action_steps = self.action_chunk_size

                dataset = LeRobotDataset(repo_id=self.dataset_repo_id, root=self.dataset_root)
                ds_meta = dataset.meta

                policy = make_policy(cfg=cfg, ds_meta=ds_meta)
                policy = policy.from_pretrained(self.model_path, config=cfg)

                if self.action_chunk_size is not None:
                    policy.config.n_action_steps = self.action_chunk_size
                    if hasattr(policy, "n_action_steps"):
                        policy.n_action_steps = self.action_chunk_size

                policy.to(self.device)
                policy.eval()

                preprocessor, postprocessor = make_pre_post_processors(
                    policy_cfg=cfg,
                    pretrained_path=self.model_path,
                )

                self._policy = policy
                self._preprocessor = preprocessor
                self._postprocessor = postprocessor
                lerobot_available = True
            except Exception:
                lerobot_available = False

        self._use_lerobot = lerobot_available
        if self._use_lerobot:
            return

        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._vla = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)

    def _build_lerobot_batch(
        self,
        image: np.ndarray,
        wrist_image: np.ndarray,
        state: t.Optional[np.ndarray],
        instruction: str,
    ) -> t.Dict[str, t.Any]:
        batch: t.Dict[str, t.Any] = {
            "observation.images.image": image,
            "observation.images.wrist_image": wrist_image,
        }
        if state is not None:
            batch["observation.state"] = state

        prepared: t.Dict[str, t.Any] = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                value = self._ensure_no_negative_strides(value)
                tensor = self._torch.from_numpy(value).to(self.device).float()

                if "image" in key and value.dtype == np.uint8:
                    tensor = tensor / 255.0

                if tensor.ndim > 0:
                    tensor = tensor.unsqueeze(0)

                if "image" in key and tensor.shape[-1] == 3:
                    tensor = tensor.permute(0, 3, 1, 2)

                prepared[key] = tensor
            else:
                prepared[key] = value

        prepared["task"] = [instruction]
        return prepared

    def _predict_lerobot_chunk(
        self,
        image: np.ndarray,
        wrist_image: np.ndarray,
        state: t.Optional[np.ndarray],
        instruction: str,
    ) -> np.ndarray:
        if self._policy is None or self._preprocessor is None or self._postprocessor is None:
            raise RuntimeError("SmolVLA LeRobot components are not initialized")

        batch = self._build_lerobot_batch(image, wrist_image, state, instruction)

        with self._torch.no_grad():
            batch = self._preprocessor(batch)
            if hasattr(self._policy, "predict_action_chunk"):
                action = self._policy.predict_action_chunk(batch)
            else:
                action = self._policy.select_action(batch)
            action = self._postprocessor(action)

        return np.asarray(action.squeeze(0).cpu().numpy(), dtype=np.float32)

    def _predict_transformers_action(
        self,
        image: np.ndarray,
        wrist_image: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        if self._processor is None or self._vla is None:
            raise RuntimeError("SmolVLA transformers components are not initialized")

        from PIL import Image

        image_head = Image.fromarray(image)
        image_wrist = Image.fromarray(wrist_image)
        combined = Image.new("RGB", (image_head.width + image_wrist.width, image_head.height))
        combined.paste(image_head, (0, 0))
        combined.paste(image_wrist, (image_head.width, 0))

        inputs = self._processor(text=instruction, images=combined, return_tensors="pt").to(
            self.device,
            self._torch.bfloat16,
        )
        action = self._vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        return np.asarray(action, dtype=np.float32)

    def predict(self, observation: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """Run SmolVLA inference and return action chunk in standard format."""
        instruction = str(observation.get("cmd", self._instruction) or self._instruction)
        self._instruction = instruction

        image = self._pick_first_image(
            observation,
            candidate_keys=("image", "cam_head", "image_head", "front_image", "full_image"),
            logical_name="head",
        )
        wrist_image = self._pick_first_image(
            observation,
            candidate_keys=("wrist_image", "image_wrist", "cam_wrist"),
            logical_name="wrist",
        )
        state = self._pick_state(observation)

        if self._use_lerobot:
            action = self._predict_lerobot_chunk(image, wrist_image, state, instruction)
        else:
            action = self._predict_transformers_action(image, wrist_image, instruction)

        return {"action": self._normalize_action(action)}
