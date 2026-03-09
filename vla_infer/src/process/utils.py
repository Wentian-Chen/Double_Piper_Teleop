"""Utility functions for image/action processing in vla_infer.

The API is direct-value in/direct-value out. No dictionary/meta wrappers.
"""

from __future__ import annotations

import typing as t

import numpy as np
from PIL import Image, ImageOps


def _to_hwc3_uint8(image: np.ndarray) -> np.ndarray:
    """Safely convert input image to HWC3 uint8."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image)}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must be HxWx3, got shape={image.shape}")

    if image.dtype == np.uint8:
        return image.copy()
    if np.issubdtype(image.dtype, np.floating):
        image_min = float(np.min(image))
        image_max = float(np.max(image))
        if image_min >= -1e-6 and image_max <= 1.0 + 1e-6:
            image_u8 = np.clip(image, 0.0, 1.0)
            return (image_u8 * 255.0).round().astype(np.uint8)
        raise ValueError(
            "float image must be in [0, 1], "
            f"got min={image_min:.6f}, max={image_max:.6f}"
        )
    raise TypeError(f"unsupported image dtype: {image.dtype}")

def check_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")
    return np.ascontiguousarray(image)
def ensure_hwc3_image(image: np.ndarray) -> np.ndarray:
    """Convert image to HWC3 layout.

    Supports input shapes: HxW, HxWx1, HxWx3, HxWx4, 1xHxW, 3xHxW, 4xHxW.
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    if image.ndim == 2:
        return np.repeat(image[:, :, None], repeats=3, axis=2)

    if image.ndim != 3:
        raise ValueError(f"image must be 2D or 3D, got shape={image.shape}")

    # HWC path
    if image.shape[-1] in (1, 3, 4):
        if image.shape[-1] == 1:
            return np.repeat(image, repeats=3, axis=2)
        if image.shape[-1] == 4:
            return image[:, :, :3]
        return image

    # CHW path
    if image.shape[0] in (1, 3, 4):
        chw_to_hwc = np.transpose(image, (1, 2, 0))
        if chw_to_hwc.shape[-1] == 1:
            return np.repeat(chw_to_hwc, repeats=3, axis=2)
        if chw_to_hwc.shape[-1] == 4:
            return chw_to_hwc[:, :, :3]
        return chw_to_hwc

    raise ValueError(f"unable to convert image to HxWx3, got shape={image.shape}")


def ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    """Convert image to uint8.

    Float input in [0, 1] is scaled to [0, 255]. Other numeric dtypes are
    clipped to [0, 255] then cast to uint8.
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    if image.dtype == np.uint8:
        return image.copy()

    if np.issubdtype(image.dtype, np.floating):
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return np.zeros_like(image, dtype=np.uint8)
        min_value = float(np.min(finite))
        max_value = float(np.max(finite))
        if min_value >= -1e-6 and max_value <= 1.0 + 1e-6:
            return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        return np.clip(image, 0.0, 255.0).round().astype(np.uint8)

    if np.issubdtype(image.dtype, np.integer):
        return np.clip(image, 0, 255).astype(np.uint8)

    raise TypeError(f"unsupported image dtype: {image.dtype}")


def ensure_hwc3_uint8_image(image: np.ndarray) -> np.ndarray:
    """Convert image to HWC3 uint8."""
    return ensure_uint8_image(ensure_hwc3_image(image))


def detect_color_order(image: np.ndarray) -> str:
    """Estimate whether channel ordering is likely RGB or BGR."""
    image_u8 = _to_hwc3_uint8(image)
    channel_means = image_u8.mean(axis=(0, 1))
    c0 = float(channel_means[0])
    c2 = float(channel_means[2])

    margin = 3.0
    if c2 > c0 + margin:
        return "rgb"
    if c0 > c2 + margin:
        return "bgr"
    return "unknown"


def check_image_dtype_and_range(
    image: np.ndarray,
    float_min: float = 0.0,
    float_max: float = 1.0,
    tolerance: float = 1e-6,
) -> bool:
    """Check whether image dtype/range is acceptable."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image)}")

    min_value = float(np.min(image))
    max_value = float(np.max(image))

    is_uint8 = image.dtype == np.uint8
    is_float32_01 = (
        image.dtype == np.float32
        and min_value >= float_min - tolerance
        and max_value <= float_max + tolerance
    )
    return bool(is_uint8 or is_float32_01)


def crop_image(
    image: np.ndarray,
    top: int = 0,
    left: int = 0,
    crop_height: int = 224,
    crop_width: int = 224,
) -> np.ndarray:
    """Crop image with explicit top/left/crop size."""
    image_u8 = _to_hwc3_uint8(image)
    h, w = image_u8.shape[:2]

    top = max(0, min(top, h - 1))
    left = max(0, min(left, w - 1))
    bottom = min(h, top + max(crop_height, 1))
    right = min(w, left + max(crop_width, 1))

    pil_image = Image.fromarray(image_u8)
    return np.asarray(pil_image.crop((left, top, right, bottom)), dtype=np.uint8)


def center_crop_image(
    image: np.ndarray,
    crop_height: int = 224,
    crop_width: int = 224,
) -> np.ndarray:
    """Center crop image."""
    image_u8 = _to_hwc3_uint8(image)
    h, w = image_u8.shape[:2]

    crop_h = min(max(crop_height, 1), h)
    crop_w = min(max(crop_width, 1), w)
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    pil_image = Image.fromarray(image_u8)
    return np.asarray(pil_image.crop((left, top, left + crop_w, top + crop_h)), dtype=np.uint8)


def adaptive_resize_with_padding(
    image: np.ndarray,
    target_height: int = 224,
    target_width: int = 224,
    pad_value: int = 255,
    resample_method: str = "bilinear",
) -> np.ndarray:
    """Resize with aspect-ratio preservation and symmetric padding (PIL)."""
    image_u8 = _to_hwc3_uint8(image)
    target_h = max(int(target_height), 1)
    target_w = max(int(target_width), 1)

    resampling = Image.Resampling
    resample_map = {
        "nearest": resampling.NEAREST,
        "bilinear": resampling.BILINEAR,
        "bicubic": resampling.BICUBIC,
        "lanczos": resampling.LANCZOS,
    }
    pil_resample = resample_map.get(str(resample_method).lower(), resampling.BILINEAR)

    processed = ImageOps.pad(
        Image.fromarray(image_u8),
        size=(target_w, target_h),
        method=pil_resample,
        color=(int(pad_value), int(pad_value), int(pad_value)),
        centering=(0.5, 0.5),
    )
    return np.asarray(processed, dtype=np.uint8)


def adaptive_resize_image(
    image: np.ndarray,
    target_height: int = 224,
    target_width: int = 224,
    pad_value: int = 255,
) -> np.ndarray:
    """Adaptive resize compatible with lerobot_to_vla_libero.py (ImageOps.pad)."""
    return adaptive_resize_with_padding(
        image=image,
        target_height=target_height,
        target_width=target_width,
        pad_value=pad_value,
        resample_method="bilinear",
    )


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB image using PIL channel operations."""
    image_u8 = _to_hwc3_uint8(image)
    pil_image = Image.fromarray(image_u8)
    b, g, r = pil_image.split()
    return np.asarray(Image.merge("RGB", (r, g, b)), dtype=np.uint8)


def ensure_float32_image_01(image: np.ndarray) -> np.ndarray:
    """Normalize image to float32 in [0,1]."""
    image_u8 = _to_hwc3_uint8(image)
    return image_u8.astype(np.float32) / 255.0


def uint8_image_to_float32_01(image: np.ndarray) -> np.ndarray:
    """Convert uint8 HWC3 image to float32 in [0,1]."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image)}")
    if image.dtype != np.uint8:
        raise TypeError(f"image dtype must be uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must be HxWx3, got shape={image.shape}")
    return image.astype(np.float32) / 255.0


def ensure_action_2d(action: np.ndarray) -> np.ndarray:
    """Normalize action to 2D array [T, D]."""
    action_np = np.asarray(action, dtype=np.float32)
    if action_np.ndim == 1:
        action_np = action_np[None, :]
    if action_np.ndim != 2:
        raise ValueError(f"action must be 1D or 2D, got shape={action_np.shape}")
    return action_np


def minmax_normalize_action(
    action: np.ndarray,
    min_values: t.Optional[t.Sequence[float]] = None,
    max_values: t.Optional[t.Sequence[float]] = None,
    eps: float = 1e-6,
    clip_to_unit_range: bool = True,
) -> np.ndarray:
    """Apply min-max normalization on action chunk."""
    action_2d = ensure_action_2d(action)
    dim = action_2d.shape[1]

    min_values_arr = np.min(action_2d, axis=0) if min_values is None else np.asarray(min_values, dtype=np.float32)
    max_values_arr = np.max(action_2d, axis=0) if max_values is None else np.asarray(max_values, dtype=np.float32)

    if min_values_arr.shape[0] != dim or max_values_arr.shape[0] != dim:
        raise ValueError("min_values/max_values dimension mismatch with action dimension")

    denom = np.maximum(max_values_arr - min_values_arr, eps)
    normalized = (action_2d - min_values_arr) / denom
    if clip_to_unit_range:
        normalized = np.clip(normalized, 0.0, 1.0)
    return normalized.astype(np.float32)


def minmax_denormalize_action(
    normalized_action: np.ndarray,
    min_values: np.ndarray,
    max_values: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Invert min-max normalization."""
    action_2d = ensure_action_2d(normalized_action)
    min_v = np.asarray(min_values, dtype=np.float32)
    max_v = np.asarray(max_values, dtype=np.float32)
    denom = np.maximum(max_v - min_v, eps)
    return (action_2d * denom + min_v).astype(np.float32)


def standard_normalize_action(
    action: np.ndarray,
    mean_values: t.Optional[t.Sequence[float]] = None,
    std_values: t.Optional[t.Sequence[float]] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply z-score normalization on action chunk."""
    action_2d = ensure_action_2d(action)
    dim = action_2d.shape[1]

    mean_values_arr = np.mean(action_2d, axis=0) if mean_values is None else np.asarray(mean_values, dtype=np.float32)
    std_values_arr = np.std(action_2d, axis=0) if std_values is None else np.asarray(std_values, dtype=np.float32)

    if mean_values_arr.shape[0] != dim or std_values_arr.shape[0] != dim:
        raise ValueError("mean_values/std_values dimension mismatch with action dimension")

    std_safe = np.maximum(std_values_arr, eps)
    return ((action_2d - mean_values_arr) / std_safe).astype(np.float32)


def standard_denormalize_action(
    normalized_action: np.ndarray,
    mean_values: np.ndarray,
    std_values: np.ndarray,
) -> np.ndarray:
    """Invert z-score normalization."""
    action_2d = ensure_action_2d(normalized_action)
    mean_v = np.asarray(mean_values, dtype=np.float32)
    std_v = np.asarray(std_values, dtype=np.float32)
    return (action_2d * std_v + mean_v).astype(np.float32)


def _moving_average_filter(action: np.ndarray, window_size: int) -> np.ndarray:
    """Apply moving average over time dimension."""
    if window_size <= 1:
        return action.copy()

    padded = np.pad(action, ((window_size - 1, 0), (0, 0)), mode="edge")
    filtered = np.zeros_like(action, dtype=np.float32)
    for t_idx in range(action.shape[0]):
        segment = padded[t_idx : t_idx + window_size]
        filtered[t_idx] = np.mean(segment, axis=0)
    return filtered


def _ema_filter(action: np.ndarray, alpha: float) -> np.ndarray:
    """Apply exponential moving average over time dimension."""
    alpha_safe = min(max(alpha, 0.0), 1.0)
    filtered = action.astype(np.float32).copy()
    for t_idx in range(1, filtered.shape[0]):
        filtered[t_idx] = alpha_safe * filtered[t_idx] + (1.0 - alpha_safe) * filtered[t_idx - 1]
    return filtered


def _limit_angular_acceleration(action: np.ndarray, max_accel: float) -> np.ndarray:
    """Limit acceleration norm of action sequence."""
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
    """Limit jerk norm of action sequence."""
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


def linear_interpolate_action_chunk(action_chunk: np.ndarray, target_steps: int) -> np.ndarray:
    """Resample action chunk to target steps with per-joint linear interpolation."""
    action_2d = ensure_action_2d(action_chunk)
    if target_steps <= 0:
        raise ValueError("target_steps must be > 0")

    src_steps, dim = action_2d.shape
    if src_steps == target_steps:
        return action_2d.copy()
    if src_steps == 1:
        return np.repeat(action_2d, repeats=target_steps, axis=0).astype(np.float32)

    src_x = np.linspace(0.0, 1.0, num=src_steps, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, num=target_steps, dtype=np.float64)
    out = np.empty((target_steps, dim), dtype=np.float32)
    for j in range(dim):
        out[:, j] = np.interp(dst_x, src_x, action_2d[:, j]).astype(np.float32)
    return out


def smooth_interpolate_action_chunk(
    action_chunk: np.ndarray,
    target_steps: int,
    moving_average_window: int = 3,
    ema_alpha: float = 0.35,
) -> np.ndarray:
    """Interpolate then smooth action chunk via moving average + EMA."""
    interpolated = linear_interpolate_action_chunk(action_chunk, target_steps)
    smoothed = _moving_average_filter(interpolated, window_size=max(int(moving_average_window), 1))
    return _ema_filter(smoothed, alpha=float(ema_alpha)).astype(np.float32)


def _limit_joint_acceleration_per_axis(
    action_chunk: np.ndarray,
    max_angular_acceleration: float,
    dt: float,
) -> np.ndarray:
    """Limit per-joint acceleration magnitude with simple forward integration."""
    if action_chunk.shape[0] < 3:
        return action_chunk.copy()
    if max_angular_acceleration <= 0.0:
        return action_chunk.copy()
    if dt <= 0.0:
        raise ValueError("dt must be > 0")

    action = action_chunk.astype(np.float32, copy=True)
    velocity = np.diff(action, axis=0, prepend=action[0:1]) / dt
    max_acc = float(max_angular_acceleration)

    for t_idx in range(1, velocity.shape[0]):
        desired_acc = (velocity[t_idx] - velocity[t_idx - 1]) / dt
        limited_acc = np.clip(desired_acc, -max_acc, max_acc)
        velocity[t_idx] = velocity[t_idx - 1] + limited_acc * dt

    out = np.zeros_like(action, dtype=np.float32)
    out[0] = action[0]
    for t_idx in range(1, action.shape[0]):
        out[t_idx] = out[t_idx - 1] + velocity[t_idx] * dt
    return out


def accel_limited_interpolate_action_chunk(
    action_chunk: np.ndarray,
    target_steps: int,
    max_angular_acceleration: float,
    dt: float = 1.0,
    moving_average_window: int = 3,
    ema_alpha: float = 0.35,
) -> np.ndarray:
    """Smooth interpolation under finite angular acceleration constraints."""
    smoothed = smooth_interpolate_action_chunk(
        action_chunk=action_chunk,
        target_steps=target_steps,
        moving_average_window=moving_average_window,
        ema_alpha=ema_alpha,
    )
    return _limit_joint_acceleration_per_axis(
        smoothed,
        max_angular_acceleration=max_angular_acceleration,
        dt=dt,
    ).astype(np.float32)


def smooth_action_chunk(
    action_chunk: np.ndarray,
    moving_average_window: int = 3,
    ema_alpha: float = 0.35,
    max_angular_acceleration: t.Optional[float] = None,
    max_angular_jerk: t.Optional[float] = None,
) -> np.ndarray:
    """Smooth chunk with optional acceleration/jerk limiting."""
    action_2d = ensure_action_2d(action_chunk)

    smoothed = _moving_average_filter(action_2d, window_size=max(moving_average_window, 1))
    smoothed = _ema_filter(smoothed, alpha=ema_alpha)

    if max_angular_acceleration is not None and max_angular_acceleration > 0.0:
        smoothed = _limit_angular_acceleration(smoothed, max_accel=max_angular_acceleration)

    if max_angular_jerk is not None and max_angular_jerk > 0.0:
        smoothed = _limit_angular_jerk(smoothed, max_jerk=max_angular_jerk)

    return smoothed.astype(np.float32)


def delta_action_chunk_to_absolute(
    current_action: np.ndarray,
    delta_action_chunk: np.ndarray,
) -> np.ndarray:
    """Convert delta action chunk into absolute action chunk.

    The delta chunk is treated as frame-to-frame increments, so the absolute
    trajectory is built by cumulative sum plus current absolute joint state.
    """
    current = np.asarray(current_action, dtype=np.float32).reshape(-1)
    delta = ensure_action_2d(delta_action_chunk)

    if delta.shape[1] != current.shape[0]:
        raise ValueError(
            "delta action dimension mismatch: "
            f"delta dim={delta.shape[1]} vs current dim={current.shape[0]}"
        )

    return (current[None, :] + np.cumsum(delta, axis=0)).astype(np.float32)


__all__ = [
    "ensure_hwc3_image",
    "ensure_uint8_image",
    "ensure_hwc3_uint8_image",
    "detect_color_order",
    "check_image_dtype_and_range",
    "crop_image",
    "center_crop_image",
    "adaptive_resize_with_padding",
    "adaptive_resize_image",
    "convert_bgr_to_rgb",
    "ensure_float32_image_01",
    "uint8_image_to_float32_01",
    "ensure_action_2d",
    "minmax_normalize_action",
    "minmax_denormalize_action",
    "standard_normalize_action",
    "standard_denormalize_action",
    "linear_interpolate_action_chunk",
    "smooth_interpolate_action_chunk",
    "accel_limited_interpolate_action_chunk",
    "smooth_action_chunk",
    "delta_action_chunk_to_absolute",
]
