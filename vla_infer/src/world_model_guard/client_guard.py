"""Torch-only World Model Guard state for the real-robot client.

The runtime is intentionally two phase:

1. ``begin_step`` consumes the fresh visual embedding and evaluates a matured
   prediction.
2. ``commit_executed_action`` registers a new prediction only after the robot
   command completed successfully.

This avoids treating returned, delayed, dropped, or truncated actions as if
they had been executed.
"""

from __future__ import annotations

from collections import deque
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclasses.dataclass(frozen=True)
class ClientWorldModelGuardConfig:
    checkpoint: str
    device: str = "cuda"
    execution_horizon: int = 30
    k_step: int | None = None
    expected_visual_tokens: int = 512

    lvm_mode: str = "official"
    threshold_window_size: int = 15
    threshold_bootstrap_size: int = 15
    threshold_base_noise_floor: float = 0.25
    threshold_ewma_alpha: float = 0.35
    threshold_k_on: float = 3.0
    threshold_k_off: float = 2.0
    threshold_trigger_margin: float = 0.02
    threshold_jump_trigger: float = 0.06
    threshold_trigger_consecutive_steps: int = 5
    threshold_reset_consecutive_steps: int = 5
    cooldown_steps: int | None = None
    threshold_hard_retrigger_margin: float = 0.08
    threshold_hard_retrigger_consecutive_steps: int = 3

    def validate(self) -> None:
        if not self.checkpoint:
            raise ValueError("Client World Model checkpoint is required")
        if self.execution_horizon < 1:
            raise ValueError("execution_horizon must be >= 1")
        if self.k_step is not None and self.k_step < 1:
            raise ValueError("k_step must be >= 1")
        if self.expected_visual_tokens < 1:
            raise ValueError("expected_visual_tokens must be >= 1")
        if self.lvm_mode not in {"official", "paper"}:
            raise ValueError("lvm_mode must be 'official' or 'paper'")
        if self.threshold_window_size < 2 or self.threshold_bootstrap_size < 2:
            raise ValueError("threshold windows must be >= 2")
        if not 0.0 < self.threshold_ewma_alpha <= 1.0:
            raise ValueError("threshold_ewma_alpha must be in (0, 1]")
        if self.cooldown_steps is not None and self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be >= 0")


@dataclasses.dataclass(frozen=True)
class ActionPair:
    action_id: str
    environment_action: np.ndarray
    model_action: np.ndarray


class AlignedActionQueue:
    """Keep environment and normalized model actions paired until execution."""

    def __init__(self) -> None:
        self._items: deque[ActionPair] = deque()
        self._generation = 0

    def clear(self) -> int:
        removed = len(self._items)
        self._items.clear()
        return removed

    def replace(
        self,
        environment_actions: np.ndarray,
        model_actions: np.ndarray,
        *,
        horizon: int,
    ) -> int:
        environment_chunk = _as_action_chunk(environment_actions, "environment_actions")
        model_chunk = _as_action_chunk(model_actions, "model_actions")
        length = min(int(horizon), len(environment_chunk), len(model_chunk))
        if length < 1:
            raise ValueError("Cannot queue an empty action chunk")
        self.clear()
        self._generation += 1
        for index in range(length):
            self._items.append(
                ActionPair(
                    action_id=f"{self._generation}:{index}",
                    environment_action=environment_chunk[index].copy(),
                    model_action=model_chunk[index].copy(),
                )
            )
        return length

    def pop(self) -> ActionPair:
        if not self._items:
            raise IndexError("Action queue is empty")
        return self._items.popleft()

    def __len__(self) -> int:
        return len(self._items)


def _as_action_chunk(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (steps, dim), got {array.shape}")
    return array


class _ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return x + y


class ResidualMLPCorrector(nn.Module):
    """Checkpoint-compatible h1 Corrector used by the server implementation."""

    def __init__(
        self,
        *,
        token_dim: int,
        action_dim: int,
        action_embed_dim: int,
        widths: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("Corrector requires at least two hidden widths")
        self.action_proj = nn.Linear(action_dim, action_embed_dim)
        self.in_proj = nn.Linear(token_dim + action_embed_dim, widths[0])
        self.blocks = nn.ModuleList([_ResidualBlock(width, dropout) for width in widths])
        self.transitions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(widths[index]),
                    nn.Linear(widths[index], widths[index + 1]),
                    nn.SiLU(),
                )
                if widths[index] != widths[index + 1]
                else nn.Identity()
                for index in range(len(widths) - 1)
            ]
        )
        self.out_norm = nn.LayerNorm(widths[-1])
        self.out_proj = nn.Linear(widths[-1], token_dim)
        self.token_dim = token_dim
        self.action_dim = action_dim

    def forward(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        squeeze_batch = z_t.ndim == 2
        if squeeze_batch:
            z_t = z_t.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action_embedding = self.action_proj(action)
        action_embedding = action_embedding[:, None, :].expand(z_t.shape[0], z_t.shape[1], -1)
        x = self.in_proj(torch.cat([z_t, action_embedding], dim=-1))
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index < len(self.transitions):
                x = self.transitions[index](x)
        output = self.out_proj(self.out_norm(x))
        return output[0] if squeeze_batch else output


@dataclasses.dataclass(frozen=True)
class CorrectorMetadata:
    checkpoint_path: str
    token_dim: int
    action_dim: int
    k_step: int
    h_window: int


class FrozenCorrector:
    def __init__(self, checkpoint_path: str, *, device: str) -> None:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "state_dict" not in payload or "model_cfg" not in payload:
            raise ValueError("Corrector checkpoint must contain state_dict and model_cfg")

        model_cfg = dict(payload["model_cfg"])
        train_cfg = dict(payload.get("train_cfg", {}))
        if str(train_cfg.get("model_type", "mlp")).lower() != "mlp":
            raise NotImplementedError("Client Guard supports the deployed h1 MLP Corrector")
        h_window = int(train_cfg.get("h_window", 1))
        if h_window != 1:
            raise NotImplementedError(f"Expected h_window=1, got {h_window}")

        model = ResidualMLPCorrector(
            token_dim=int(model_cfg["token_dim"]),
            action_dim=int(model_cfg["action_dim"]),
            action_embed_dim=int(model_cfg["action_embed_dim"]),
            widths=_widths_from_config(model_cfg),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
        state_dict = _strip_parallel_prefix(payload["state_dict"])
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.model = model.to(self.device)
        self.metadata = CorrectorMetadata(
            checkpoint_path=str(path),
            token_dim=model.token_dim,
            action_dim=model.action_dim,
            k_step=int(train_cfg.get("k_step", 10)),
            h_window=h_window,
        )

    def predict(self, z_t: np.ndarray, model_action: np.ndarray) -> np.ndarray:
        z_array = np.asarray(z_t, dtype=np.float32)
        action_array = np.asarray(model_action, dtype=np.float32)[..., : self.metadata.action_dim]
        if z_array.shape[-1] != self.metadata.token_dim:
            raise ValueError(
                f"Corrector token dim mismatch: {z_array.shape[-1]} != {self.metadata.token_dim}"
            )
        if action_array.shape[-1] != self.metadata.action_dim:
            raise ValueError(
                f"Corrector action dim mismatch: {action_array.shape[-1]} != {self.metadata.action_dim}"
            )
        with torch.inference_mode():
            output = self.model(
                torch.as_tensor(z_array, device=self.device),
                torch.as_tensor(action_array, device=self.device),
            )
        return output.detach().cpu().numpy().astype(np.float32, copy=False)


def _widths_from_config(model_cfg: dict[str, Any]) -> list[int]:
    scale = str(model_cfg.get("scale", "20m")).lower()
    if scale == "4m":
        return [1024, 1024, 1024]
    if scale == "20m":
        return [2048, 2048, 2048, 2048]
    if scale == "100m":
        return [4096, 4096, 4096, 4096, 4096, 4096]
    if scale == "custom":
        return [int(value) for value in model_cfg["custom_widths"]]
    raise ValueError(f"Unknown Corrector scale: {scale!r}")


def _strip_parallel_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


@dataclasses.dataclass(frozen=True)
class MonitorResult:
    raw_score: float
    monitor_score: float
    threshold_on: float
    threshold_off: float
    activation_boundary: float
    bootstrapped: bool
    eligible: bool
    over_activation: bool
    triggered: bool
    trigger_reason: str


@dataclasses.dataclass
class LVMMonitor:
    cfg: ClientWorldModelGuardConfig
    bootstrap_scores: list[float] = dataclasses.field(default_factory=list)
    clean_history: list[float] = dataclasses.field(default_factory=list)
    ewma_previous: float | None = None
    bootstrapped: bool = False
    alert_active: bool = False
    over_count: int = 0
    under_count: int = 0
    hard_over_count: int = 0
    cooldown_remaining: int = 0

    @property
    def _paper_mode(self) -> bool:
        return self.cfg.lvm_mode == "paper"

    @property
    def _trigger_margin(self) -> float:
        return 0.0 if self._paper_mode else self.cfg.threshold_trigger_margin

    @property
    def _cooldown_steps(self) -> int:
        if self.cfg.cooldown_steps is not None:
            return self.cfg.cooldown_steps
        return 10 if self._paper_mode else 0

    def _thresholds(self, current_score: float | None = None) -> tuple[float, float]:
        if self._paper_mode and current_score is not None:
            history_size = max(1, self.cfg.threshold_window_size - 1)
            values = [*self.clean_history[-history_size:], current_score]
        else:
            values = self.clean_history[-self.cfg.threshold_window_size :]
        recent = np.asarray(values, dtype=np.float32)
        median = float(np.median(recent))
        mad = max(float(np.median(np.abs(recent - median))), 1e-6)
        threshold_on = median + self.cfg.threshold_k_on * mad
        threshold_off = median + self.cfg.threshold_k_off * mad
        if not self._paper_mode:
            threshold_on = max(self.cfg.threshold_base_noise_floor, threshold_on)
            threshold_off = max(self.cfg.threshold_base_noise_floor, threshold_off)
        return float(threshold_on), float(min(threshold_off, threshold_on))

    def step(self, raw_score: float) -> MonitorResult:
        if not np.isfinite(raw_score):
            return MonitorResult(
                raw_score=float(raw_score),
                monitor_score=float("nan"),
                threshold_on=float("nan"),
                threshold_off=float("nan"),
                activation_boundary=float("nan"),
                bootstrapped=self.bootstrapped,
                eligible=False,
                over_activation=False,
                triggered=False,
                trigger_reason="",
            )

        previous = self.ewma_previous
        monitor_score = (
            float(raw_score)
            if self._paper_mode or previous is None
            else self.cfg.threshold_ewma_alpha * float(raw_score)
            + (1.0 - self.cfg.threshold_ewma_alpha) * previous
        )
        jump = 0.0 if previous is None else abs(monitor_score - previous)
        self.ewma_previous = monitor_score

        if not self.bootstrapped:
            self.bootstrap_scores.append(monitor_score)
            if len(self.bootstrap_scores) < self.cfg.threshold_bootstrap_size:
                threshold = 10.0
                return MonitorResult(
                    raw_score=float(raw_score),
                    monitor_score=monitor_score,
                    threshold_on=threshold,
                    threshold_off=threshold,
                    activation_boundary=threshold + self._trigger_margin,
                    bootstrapped=False,
                    eligible=False,
                    over_activation=False,
                    triggered=False,
                    trigger_reason="",
                )
            self.clean_history = self.bootstrap_scores[-self.cfg.threshold_bootstrap_size :].copy()
            self.bootstrapped = True
            threshold_on, threshold_off = self._thresholds()
            return MonitorResult(
                raw_score=float(raw_score),
                monitor_score=monitor_score,
                threshold_on=threshold_on,
                threshold_off=threshold_off,
                activation_boundary=threshold_on + self._trigger_margin,
                bootstrapped=True,
                eligible=False,
                over_activation=False,
                triggered=False,
                trigger_reason="",
            )

        threshold_on, threshold_off = self._thresholds(monitor_score)
        activation_boundary = threshold_on + self._trigger_margin
        over_activation = monitor_score > activation_boundary
        jump_condition = (
            not self._paper_mode
            and jump > self.cfg.threshold_jump_trigger
            and monitor_score > threshold_on
        )
        hard_condition = (
            not self._paper_mode
            and monitor_score > threshold_on + self.cfg.threshold_hard_retrigger_margin
        )

        if over_activation:
            self.over_count += 1
        elif not self._paper_mode or monitor_score < threshold_off:
            self.over_count = 0
        self.hard_over_count = self.hard_over_count + 1 if hard_condition else 0
        self.under_count = self.under_count + 1 if monitor_score < threshold_off else 0
        if self.alert_active and self.under_count >= self.cfg.threshold_reset_consecutive_steps:
            self.alert_active = False
            self.over_count = 0
            self.hard_over_count = 0

        triggered = False
        reason = ""
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if self.hard_over_count >= self.cfg.threshold_hard_retrigger_consecutive_steps:
                triggered = True
                reason = "hard_cooldown"
        elif jump_condition:
            triggered = True
            reason = "jump"
        elif self.over_count >= self.cfg.threshold_trigger_consecutive_steps:
            triggered = True
            reason = "persistent"

        if triggered:
            self.alert_active = True
            self.cooldown_remaining = self._cooldown_steps
            self.over_count = 0
            self.hard_over_count = 0
            self.under_count = 0

        history_score = (
            min(monitor_score, threshold_on)
            if self.alert_active and not self._paper_mode
            else monitor_score
        )
        self.clean_history.append(float(history_score))
        return MonitorResult(
            raw_score=float(raw_score),
            monitor_score=monitor_score,
            threshold_on=threshold_on,
            threshold_off=threshold_off,
            activation_boundary=activation_boundary,
            bootstrapped=True,
            eligible=True,
            over_activation=over_activation,
            triggered=triggered,
            trigger_reason=reason,
        )


@dataclasses.dataclass(frozen=True)
class _Prediction:
    source_step: int
    target_step: int
    expected_delta: np.ndarray
    source_model_action: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class GuardDecision:
    step: int
    k_step: int
    triggered: bool
    monitor: MonitorResult | None
    diagnostics: WorldModelDiagnostics | None = None
    expected_terminal_embedding: np.ndarray | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    actual_terminal_embedding: np.ndarray | None = dataclasses.field(
        default=None, repr=False, compare=False
    )


@dataclasses.dataclass(frozen=True)
class WorldModelDiagnostics:
    source_step: int
    target_step: int
    source_model_action: tuple[float, ...]
    cosine_error: float
    expected_delta_norm: float
    actual_delta_norm: float
    delta_error_l2: float
    expected_terminal_norm: float
    actual_terminal_norm: float
    terminal_error_l2: float
    expected_terminal_mean: float
    expected_terminal_std: float


class ClientWorldModelGuard:
    """Episode-local Corrector and LVM state owned by the robot client."""

    def __init__(self, config: ClientWorldModelGuardConfig) -> None:
        config.validate()
        self.config = config
        self.corrector = FrozenCorrector(config.checkpoint, device=config.device)
        self.k_step = self.corrector.metadata.k_step if config.k_step is None else config.k_step
        if self.k_step != self.corrector.metadata.k_step:
            raise ValueError(
                f"Corrector was trained with k_step={self.corrector.metadata.k_step}, "
                f"but runtime k_step={self.k_step}"
            )
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.embeddings: dict[int, np.ndarray] = {}
        self.predictions: dict[int, _Prediction] = {}
        self.monitor = LVMMonitor(self.config)
        self._open_embedding: np.ndarray | None = None

    def begin_step(self, embedding_response: dict[str, Any]) -> GuardDecision:
        if self._open_embedding is not None:
            raise RuntimeError("Previous Guard step has not been committed or cancelled")
        z_t = decode_quantized_embedding(embedding_response)
        expected_shape = (
            self.config.expected_visual_tokens,
            self.corrector.metadata.token_dim,
        )
        if z_t.shape != expected_shape:
            cameras = embedding_response.get("visual_embedding_camera_names")
            raise ValueError(
                f"Visual embedding shape mismatch: {z_t.shape} != {expected_shape}; cameras={cameras}"
            )

        current_step = self.step
        self.embeddings[current_step] = z_t
        self._open_embedding = z_t
        prediction = self.predictions.pop(current_step, None)
        monitor_result = None
        diagnostics = None
        expected_terminal = None
        actual_terminal = None
        if prediction is not None:
            source_embedding = self.embeddings.get(prediction.source_step)
            if source_embedding is None:
                raise RuntimeError(f"Missing source embedding at step {prediction.source_step}")
            actual_delta = z_t - source_embedding
            raw_score = cosine_distance(prediction.expected_delta, actual_delta)
            monitor_result = self.monitor.step(raw_score)
            expected_terminal = source_embedding + prediction.expected_delta
            actual_terminal = z_t
            delta_error = prediction.expected_delta - actual_delta
            terminal_error = expected_terminal - actual_terminal
            diagnostics = WorldModelDiagnostics(
                source_step=prediction.source_step,
                target_step=current_step,
                source_model_action=prediction.source_model_action,
                cosine_error=raw_score,
                expected_delta_norm=float(np.linalg.norm(prediction.expected_delta)),
                actual_delta_norm=float(np.linalg.norm(actual_delta)),
                delta_error_l2=float(np.linalg.norm(delta_error)),
                expected_terminal_norm=float(np.linalg.norm(expected_terminal)),
                actual_terminal_norm=float(np.linalg.norm(actual_terminal)),
                terminal_error_l2=float(np.linalg.norm(terminal_error)),
                expected_terminal_mean=float(np.mean(expected_terminal)),
                expected_terminal_std=float(np.std(expected_terminal)),
            )
        return GuardDecision(
            step=current_step,
            k_step=self.k_step,
            triggered=bool(monitor_result and monitor_result.triggered),
            monitor=monitor_result,
            diagnostics=diagnostics,
            expected_terminal_embedding=expected_terminal,
            actual_terminal_embedding=actual_terminal,
        )

    def commit_executed_action(self, model_action: np.ndarray) -> None:
        if self._open_embedding is None:
            raise RuntimeError("begin_step must be called before committing an action")
        corrector_action = np.asarray(model_action, dtype=np.float32).reshape(-1)[
            : self.corrector.metadata.action_dim
        ]
        expected_delta = self.corrector.predict(self._open_embedding, corrector_action)
        target_step = self.step + self.k_step
        self.predictions[target_step] = _Prediction(
            source_step=self.step,
            target_step=target_step,
            expected_delta=expected_delta,
            source_model_action=tuple(float(value) for value in corrector_action),
        )
        oldest_needed_step = self.step - self.k_step
        for old_step in [key for key in self.embeddings if key < oldest_needed_step]:
            del self.embeddings[old_step]
        self._open_embedding = None
        self.step += 1

    def cancel_open_step(self) -> None:
        if self._open_embedding is not None:
            self.embeddings.pop(self.step, None)
        self._open_embedding = None


def decode_quantized_embedding(response: dict[str, Any]) -> np.ndarray:
    encoding = response.get("visual_embedding_encoding")
    if encoding != "int8_per_token_float16_scale":
        raise ValueError(f"Unsupported visual embedding encoding: {encoding!r}")
    quantized = np.asarray(response["visual_embedding_q"], dtype=np.int8)
    scale = np.asarray(response["visual_embedding_scale"], dtype=np.float16).astype(np.float32)
    if quantized.ndim != 2 or scale.shape != quantized.shape[:-1]:
        raise ValueError(
            f"Invalid quantized embedding shapes: q={quantized.shape}, scale={scale.shape}"
        )
    return quantized.astype(np.float32) * scale[..., None]


def cosine_distance(expected: np.ndarray, actual: np.ndarray) -> float:
    expected_flat = np.asarray(expected, dtype=np.float32).reshape(-1)
    actual_flat = np.asarray(actual, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(expected_flat) * np.linalg.norm(actual_flat))
    if denominator <= 1e-8:
        return 1.0
    cosine = float(np.dot(expected_flat, actual_flat) / denominator)
    return float(1.0 - np.clip(cosine, -1.0, 1.0))
