"""vla_infer 预处理工具函数。

说明：
- 所有函数尽量保持无副作用，输入数组不会在原地被修改。
- 涉及多值返回时统一使用字典，避免使用 tuple。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import typing as t

import draccus
import numpy as np
from PIL import Image


@dataclass
class CropConfig:
    """固定窗口裁剪配置。"""

    top: int
    left: int
    crop_height: int
    crop_width: int


@dataclass
class CenterCropConfig:
    """中心裁剪配置。"""

    crop_height: int
    crop_width: int


@dataclass
class AdaptiveResizeConfig:
    """自适应缩放配置（短边按比例缩放，长边白边补齐到目标分辨率）。"""

    target_height: int
    target_width: int
    pad_value: int = 255
    resample: str = "bilinear"


@dataclass
class ImageRangeCheckConfig:
    """图像范围校验配置。"""

    float_min: float = 0.0
    float_max: float = 1.0
    tolerance: float = 1e-6


@dataclass
class MinMaxNormConfig:
    """动作 Min-Max 归一化配置。"""

    min_values: t.Optional[t.List[float]] = None
    max_values: t.Optional[t.List[float]] = None
    eps: float = 1e-6
    clip_to_unit_range: bool = True


@dataclass
class StandardNormConfig:
    """动作标准化配置。"""

    mean_values: t.Optional[t.List[float]] = None
    std_values: t.Optional[t.List[float]] = None
    eps: float = 1e-6


@dataclass
class ActionSmoothingConfig:
    """动作 chunk 平滑与滤波配置。"""

    moving_average_window: int = 3
    ema_alpha: float = 0.35
    max_angular_acceleration: t.Optional[float] = None
    max_angular_jerk: t.Optional[float] = None


@dataclass
class ProcessUtilitiesConfig:
    """工具层统一配置容器（便于 draccus 管理）。"""

    crop: CropConfig = field(default_factory=lambda: CropConfig(top=0, left=0, crop_height=224, crop_width=224))
    center_crop: CenterCropConfig = field(default_factory=lambda: CenterCropConfig(crop_height=224, crop_width=224))
    adaptive_resize: AdaptiveResizeConfig = field(
        default_factory=lambda: AdaptiveResizeConfig(target_height=224, target_width=224)
    )
    image_range_check: ImageRangeCheckConfig = field(default_factory=ImageRangeCheckConfig)
    minmax_norm: MinMaxNormConfig = field(default_factory=MinMaxNormConfig)
    standard_norm: StandardNormConfig = field(default_factory=StandardNormConfig)
    smoothing: ActionSmoothingConfig = field(default_factory=ActionSmoothingConfig)


def _to_hwc3_uint8(image: np.ndarray) -> t.Dict[str, t.Any]:
    """将输入图像安全转换为 HWC3 的 uint8 图像。

    返回字典结构：
    - image: np.ndarray(H, W, 3), dtype=uint8
    - input_was_float01: bool，输入是否为 float32 且位于 [0,1]
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image)}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must be HxWx3, got shape={image.shape}")

    input_was_float01 = False
    if image.dtype == np.uint8:
        image_u8 = image.copy()
    elif np.issubdtype(image.dtype, np.floating):
        image_min = float(np.min(image))
        image_max = float(np.max(image))
        if image_min >= -1e-6 and image_max <= 1.0 + 1e-6:
            input_was_float01 = True
            image_u8 = np.clip(image, 0.0, 1.0)
            image_u8 = (image_u8 * 255.0).round().astype(np.uint8)
        else:
            raise ValueError(
                "float image must be in [0, 1], "
                f"got min={image_min:.6f}, max={image_max:.6f}"
            )
    else:
        raise TypeError(f"unsupported image dtype: {image.dtype}")

    return {
        "image": image_u8,
        "input_was_float01": input_was_float01,
    }


def detect_color_order(image: np.ndarray) -> t.Dict[str, t.Any]:
    """估计图像通道顺序是否更像 RGB 或 BGR。

    说明：该方法基于通道均值统计进行弱监督估计，适合作为在线告警而非强约束。

    返回字典结构：
    - color_order: str，取值为 "rgb" | "bgr" | "unknown"
    - is_rgb_likely: bool
    - is_bgr_likely: bool
    - channel_means: dict，结构为 {"c0": float, "c1": float, "c2": float}
    """
    checked = _to_hwc3_uint8(image)
    image_u8 = checked["image"]

    channel_means = image_u8.mean(axis=(0, 1))
    c0 = float(channel_means[0])
    c2 = float(channel_means[2])

    # 对自然场景通常红通道均值略高于蓝通道，反之可疑为 BGR。
    margin = 3.0
    if c2 > c0 + margin:
        order = "rgb"
    elif c0 > c2 + margin:
        order = "bgr"
    else:
        order = "unknown"

    return {
        "color_order": order,
        "is_rgb_likely": order == "rgb",
        "is_bgr_likely": order == "bgr",
        "channel_means": {
            "c0": c0,
            "c1": float(channel_means[1]),
            "c2": c2,
        },
    }


def check_image_dtype_and_range(
    image: np.ndarray,
    cfg: ImageRangeCheckConfig,
) -> t.Dict[str, t.Any]:
    """检查图像 dtype 与值域是否合法。

    返回字典结构：
    - ok: bool
    - dtype: str
    - is_uint8: bool
    - is_float32_01: bool
    - min_value: float
    - max_value: float
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image)}")

    min_value = float(np.min(image))
    max_value = float(np.max(image))

    is_uint8 = image.dtype == np.uint8
    is_float32 = image.dtype == np.float32
    is_float32_01 = (
        is_float32
        and min_value >= cfg.float_min - cfg.tolerance
        and max_value <= cfg.float_max + cfg.tolerance
    )

    return {
        "ok": is_uint8 or is_float32_01,
        "dtype": str(image.dtype),
        "is_uint8": is_uint8,
        "is_float32_01": is_float32_01,
        "min_value": min_value,
        "max_value": max_value,
    }


def crop_image(image: np.ndarray, cfg: CropConfig) -> t.Dict[str, t.Any]:
    """执行固定窗口裁剪。

    返回字典结构：
    - image: np.ndarray，裁剪结果
    - meta: dict，结构为 {"top": int, "left": int, "crop_height": int, "crop_width": int}
    """
    checked = _to_hwc3_uint8(image)
    image_u8 = checked["image"]

    h, w = image_u8.shape[:2]
    top = max(0, min(cfg.top, h - 1))
    left = max(0, min(cfg.left, w - 1))
    bottom = min(h, top + max(cfg.crop_height, 1))
    right = min(w, left + max(cfg.crop_width, 1))

    cropped = image_u8[top:bottom, left:right].copy()

    return {
        "image": cropped,
        "meta": {
            "top": int(top),
            "left": int(left),
            "crop_height": int(cropped.shape[0]),
            "crop_width": int(cropped.shape[1]),
        },
    }


def center_crop_image(image: np.ndarray, cfg: CenterCropConfig) -> t.Dict[str, t.Any]:
    """执行中心裁剪。"""
    checked = _to_hwc3_uint8(image)
    image_u8 = checked["image"]

    h, w = image_u8.shape[:2]
    crop_h = min(max(cfg.crop_height, 1), h)
    crop_w = min(max(cfg.crop_width, 1), w)

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    cropped = image_u8[top : top + crop_h, left : left + crop_w].copy()

    return {
        "image": cropped,
        "meta": {
            "top": int(top),
            "left": int(left),
            "crop_height": int(crop_h),
            "crop_width": int(crop_w),
        },
    }


def adaptive_resize_with_padding(image: np.ndarray, cfg: AdaptiveResizeConfig) -> t.Dict[str, t.Any]:
    """根据目标分辨率执行等比例缩放，并在短边方向补齐白边。

    返回字典结构：
    - image: np.ndarray(Ht, Wt, 3)
    - meta: dict，结构为
      {
        "scale": float,
        "resized_height": int,
        "resized_width": int,
        "pad_top": int,
        "pad_bottom": int,
        "pad_left": int,
        "pad_right": int,
        "target_height": int,
        "target_width": int
      }
    """
    checked = _to_hwc3_uint8(image)
    image_u8 = checked["image"]

    h, w = image_u8.shape[:2]
    target_h = max(cfg.target_height, 1)
    target_w = max(cfg.target_width, 1)

    scale = min(target_h / h, target_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resampling = Image.Resampling
    resample_map = {
        "nearest": resampling.NEAREST,
        "bilinear": resampling.BILINEAR,
        "bicubic": resampling.BICUBIC,
        "lanczos": resampling.LANCZOS,
    }
    resample = resample_map.get(cfg.resample.lower(), resampling.BILINEAR)

    resized = np.asarray(Image.fromarray(image_u8).resize((new_w, new_h), resample=resample), dtype=np.uint8)

    canvas = np.full((target_h, target_w, 3), fill_value=int(cfg.pad_value), dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_right = target_w - new_w - pad_left

    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    return {
        "image": canvas,
        "meta": {
            "scale": float(scale),
            "resized_height": int(new_h),
            "resized_width": int(new_w),
            "pad_top": int(pad_top),
            "pad_bottom": int(pad_bottom),
            "pad_left": int(pad_left),
            "pad_right": int(pad_right),
            "target_height": int(target_h),
            "target_width": int(target_w),
        },
    }


def convert_bgr_to_rgb(image: np.ndarray) -> t.Dict[str, t.Any]:
    """将 BGR 图像转换为 RGB 图像。"""
    checked = _to_hwc3_uint8(image)
    image_u8 = checked["image"]
    rgb = image_u8[:, :, ::-1].copy()
    return {"image": rgb}


def ensure_float32_image_01(image: np.ndarray) -> t.Dict[str, t.Any]:
    """将图像规范为 float32 且范围在 [0,1]。"""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image)}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must be HxWx3, got shape={image.shape}")

    if image.dtype == np.uint8:
        image_f32 = image.astype(np.float32) / 255.0
    elif image.dtype == np.float32:
        min_value = float(np.min(image))
        max_value = float(np.max(image))
        if min_value < -1e-6 or max_value > 1.0 + 1e-6:
            raise ValueError(f"float32 image out of [0,1], min={min_value:.6f}, max={max_value:.6f}")
        image_f32 = image.copy()
    else:
        raise TypeError(f"unsupported image dtype: {image.dtype}")

    return {"image": image_f32}


def ensure_action_2d(action: np.ndarray) -> t.Dict[str, t.Any]:
    """将动作规范为二维数组 [T, D]。"""
    action_np = np.asarray(action, dtype=np.float32)
    if action_np.ndim == 1:
        action_np = action_np[None, :]
    if action_np.ndim != 2:
        raise ValueError(f"action must be 1D or 2D, got shape={action_np.shape}")
    return {"action": action_np}


def minmax_normalize_action(action: np.ndarray, cfg: MinMaxNormConfig) -> t.Dict[str, t.Any]:
    """对动作执行 Min-Max 归一化。"""
    action_2d = ensure_action_2d(action)["action"]
    dim = action_2d.shape[1]

    if cfg.min_values is None:
        min_values = np.min(action_2d, axis=0)
    else:
        min_values = np.asarray(cfg.min_values, dtype=np.float32)
    if cfg.max_values is None:
        max_values = np.max(action_2d, axis=0)
    else:
        max_values = np.asarray(cfg.max_values, dtype=np.float32)

    if min_values.shape[0] != dim or max_values.shape[0] != dim:
        raise ValueError("min_values/max_values dimension mismatch with action dimension")

    denom = np.maximum(max_values - min_values, cfg.eps)
    normalized = (action_2d - min_values) / denom
    if cfg.clip_to_unit_range:
        normalized = np.clip(normalized, 0.0, 1.0)

    return {
        "action": normalized.astype(np.float32),
        "meta": {
            "min_values": min_values.astype(np.float32),
            "max_values": max_values.astype(np.float32),
        },
    }


def minmax_denormalize_action(
    normalized_action: np.ndarray,
    min_values: np.ndarray,
    max_values: np.ndarray,
    eps: float = 1e-6,
) -> t.Dict[str, t.Any]:
    """将 Min-Max 归一化动作还原到原始尺度。"""
    action_2d = ensure_action_2d(normalized_action)["action"]
    min_v = np.asarray(min_values, dtype=np.float32)
    max_v = np.asarray(max_values, dtype=np.float32)
    denom = np.maximum(max_v - min_v, eps)
    action = action_2d * denom + min_v
    return {"action": action.astype(np.float32)}


def standard_normalize_action(action: np.ndarray, cfg: StandardNormConfig) -> t.Dict[str, t.Any]:
    """对动作执行标准化（z-score）。"""
    action_2d = ensure_action_2d(action)["action"]
    dim = action_2d.shape[1]

    if cfg.mean_values is None:
        mean_values = np.mean(action_2d, axis=0)
    else:
        mean_values = np.asarray(cfg.mean_values, dtype=np.float32)

    if cfg.std_values is None:
        std_values = np.std(action_2d, axis=0)
    else:
        std_values = np.asarray(cfg.std_values, dtype=np.float32)

    if mean_values.shape[0] != dim or std_values.shape[0] != dim:
        raise ValueError("mean_values/std_values dimension mismatch with action dimension")

    std_safe = np.maximum(std_values, cfg.eps)
    normalized = (action_2d - mean_values) / std_safe

    return {
        "action": normalized.astype(np.float32),
        "meta": {
            "mean_values": mean_values.astype(np.float32),
            "std_values": std_safe.astype(np.float32),
        },
    }


def standard_denormalize_action(
    normalized_action: np.ndarray,
    mean_values: np.ndarray,
    std_values: np.ndarray,
) -> t.Dict[str, t.Any]:
    """将标准化动作还原到原始尺度。"""
    action_2d = ensure_action_2d(normalized_action)["action"]
    mean_v = np.asarray(mean_values, dtype=np.float32)
    std_v = np.asarray(std_values, dtype=np.float32)
    action = action_2d * std_v + mean_v
    return {"action": action.astype(np.float32)}


def _moving_average_filter(action: np.ndarray, window_size: int) -> np.ndarray:
    """对动作序列按时间维执行移动平均滤波。"""
    if window_size <= 1:
        return action.copy()

    padded = np.pad(action, ((window_size - 1, 0), (0, 0)), mode="edge")
    filtered = np.zeros_like(action, dtype=np.float32)
    for t_idx in range(action.shape[0]):
        segment = padded[t_idx : t_idx + window_size]
        filtered[t_idx] = np.mean(segment, axis=0)
    return filtered


def _ema_filter(action: np.ndarray, alpha: float) -> np.ndarray:
    """指数滑动平均滤波。"""
    alpha_safe = min(max(alpha, 0.0), 1.0)
    filtered = action.astype(np.float32).copy()
    for t_idx in range(1, filtered.shape[0]):
        filtered[t_idx] = alpha_safe * filtered[t_idx] + (1.0 - alpha_safe) * filtered[t_idx - 1]
    return filtered


def _limit_angular_acceleration(action: np.ndarray, max_accel: float) -> np.ndarray:
    """限制动作时间序列的角加速度范数。"""
    if action.shape[0] < 3:
        return action

    vel = np.diff(action, axis=0, prepend=action[0:1])
    for t_idx in range(1, vel.shape[0]):
        accel = vel[t_idx] - vel[t_idx - 1]
        accel_norm = float(np.linalg.norm(accel))
        if accel_norm > max_accel:
            scale = max_accel / (accel_norm + 1e-12)
            vel[t_idx] = vel[t_idx - 1] + accel * scale

    limited = np.zeros_like(action, dtype=np.float32)
    limited[0] = action[0]
    for t_idx in range(1, action.shape[0]):
        limited[t_idx] = limited[t_idx - 1] + vel[t_idx]
    return limited


def _limit_angular_jerk(action: np.ndarray, max_jerk: float) -> np.ndarray:
    """限制动作时间序列的角加加速度（jerk）范数。"""
    if action.shape[0] < 4:
        return action

    vel = np.diff(action, axis=0, prepend=action[0:1])
    accel = np.diff(vel, axis=0, prepend=vel[0:1])

    for t_idx in range(1, accel.shape[0]):
        jerk = accel[t_idx] - accel[t_idx - 1]
        jerk_norm = float(np.linalg.norm(jerk))
        if jerk_norm > max_jerk:
            scale = max_jerk / (jerk_norm + 1e-12)
            accel[t_idx] = accel[t_idx - 1] + jerk * scale

    vel_limited = np.zeros_like(vel, dtype=np.float32)
    vel_limited[0] = vel[0]
    for t_idx in range(1, vel.shape[0]):
        vel_limited[t_idx] = vel_limited[t_idx - 1] + accel[t_idx]

    out = np.zeros_like(action, dtype=np.float32)
    out[0] = action[0]
    for t_idx in range(1, action.shape[0]):
        out[t_idx] = out[t_idx - 1] + vel_limited[t_idx]

    return out


def smooth_action_chunk(action_chunk: np.ndarray, cfg: ActionSmoothingConfig) -> t.Dict[str, t.Any]:
    """对动作 chunk 执行平滑和动力学约束滤波。

    返回字典结构：
    - action: np.ndarray(T, D)，平滑后动作
    - meta: dict，结构为 {"window": int, "ema_alpha": float, "accel_limited": bool, "jerk_limited": bool}
    """
    action_2d = ensure_action_2d(action_chunk)["action"]

    smoothed = _moving_average_filter(action_2d, window_size=max(cfg.moving_average_window, 1))
    smoothed = _ema_filter(smoothed, alpha=cfg.ema_alpha)

    accel_limited = False
    jerk_limited = False

    if cfg.max_angular_acceleration is not None and cfg.max_angular_acceleration > 0.0:
        smoothed = _limit_angular_acceleration(smoothed, max_accel=cfg.max_angular_acceleration)
        accel_limited = True

    if cfg.max_angular_jerk is not None and cfg.max_angular_jerk > 0.0:
        smoothed = _limit_angular_jerk(smoothed, max_jerk=cfg.max_angular_jerk)
        jerk_limited = True

    return {
        "action": smoothed.astype(np.float32),
        "meta": {
            "window": int(max(cfg.moving_average_window, 1)),
            "ema_alpha": float(cfg.ema_alpha),
            "accel_limited": accel_limited,
            "jerk_limited": jerk_limited,
        },
    }


__all__ = [
    "draccus",
    "CropConfig",
    "CenterCropConfig",
    "AdaptiveResizeConfig",
    "ImageRangeCheckConfig",
    "MinMaxNormConfig",
    "StandardNormConfig",
    "ActionSmoothingConfig",
    "ProcessUtilitiesConfig",
    "detect_color_order",
    "check_image_dtype_and_range",
    "crop_image",
    "center_crop_image",
    "adaptive_resize_with_padding",
    "convert_bgr_to_rgb",
    "ensure_float32_image_01",
    "ensure_action_2d",
    "minmax_normalize_action",
    "minmax_denormalize_action",
    "standard_normalize_action",
    "standard_denormalize_action",
    "smooth_action_chunk",
]
