#!/usr/bin/env python3
"""Visualize and sanity-check one Piper HDF5 episode."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

cv2 = None
h5py = None
plt = None


HEAD_KEY = "cam_head/color"
WRIST_KEY = "cam_wrist/color"
JOINT_KEY = "left_arm/joint"
GRIPPER_KEY = "left_arm/gripper"
QPOS_KEY = "left_arm/qpos"

ALL_KEYS = [HEAD_KEY, WRIST_KEY, JOINT_KEY, GRIPPER_KEY, QPOS_KEY]
REQUIRED_KEYS = [HEAD_KEY, WRIST_KEY, JOINT_KEY, GRIPPER_KEY]


def load_runtime_deps():
    global cv2, h5py, plt
    try:
        import cv2 as cv2_module
        import h5py as h5py_module
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot_module
    except Exception as exc:
        raise RuntimeError(
            "Failed to import required visualization dependencies. "
            "Please install/check numpy, h5py, opencv-python, and matplotlib."
        ) from exc

    cv2 = cv2_module
    h5py = h5py_module
    plt = pyplot_module


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def as_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: as_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [as_jsonable(val) for val in value]
    return value


def read_dataset(root, key, warnings):
    if key not in root:
        level = "WARNING" if key == QPOS_KEY else "MISSING"
        message = f"{level}: HDF5 key not found: {key}"
        print(message)
        warnings.append(message)
        return None
    return root[key][()]


def numeric_min_max(array):
    array = np.asarray(array)
    if not np.issubdtype(array.dtype, np.number):
        return None, None
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None, None
    return float(np.min(finite)), float(np.max(finite))


def print_key_stats(key, array):
    if array is None:
        return
    min_value, max_value = numeric_min_max(array)
    min_text = "nan" if min_value is None else f"{min_value:.6g}"
    max_text = "nan" if max_value is None else f"{max_value:.6g}"
    print(
        f"{key}: shape={array.shape}, dtype={array.dtype}, "
        f"min={min_text}, max={max_text}"
    )


def get_frame_lengths(data, keys=None):
    if keys is None:
        keys = data.keys()
    lengths = {}
    for key in keys:
        value = data.get(key)
        if value is not None and hasattr(value, "shape") and len(value.shape) > 0:
            lengths[key] = int(value.shape[0])
    return lengths


def normalize_gripper(gripper):
    if gripper is None:
        return None
    gripper = np.asarray(gripper, dtype=np.float32)
    if gripper.ndim == 1:
        return gripper[:, None]
    if gripper.ndim == 2 and gripper.shape[1] == 1:
        return gripper
    flattened = gripper.reshape(gripper.shape[0], -1)
    print(
        f"WARNING: gripper shape {gripper.shape} is not (T,) or (T, 1); "
        "using the first flattened column for visualization"
    )
    return flattened[:, :1]


def normalize_image(frame, convert_bgr_to_rgb=False):
    image = np.asarray(frame)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    elif image.ndim == 3 and image.shape[-1] > 3:
        image = image[..., :3]

    if image.dtype != np.uint8:
        image = np.nan_to_num(
            image.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0
        )
        if image.size and np.nanmin(image) >= 0.0 and np.nanmax(image) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    else:
        image = image.copy()

    if convert_bgr_to_rgb and image.ndim == 3 and image.shape[-1] == 3:
        image = image[..., ::-1]
    return np.ascontiguousarray(image)


def rgb_image_for_cv2_write(image):
    if image.ndim == 3 and image.shape[-1] == 3:
        return image[..., ::-1].copy()
    return image


def image_for_video(frame, convert_bgr_to_rgb=False):
    image_rgb = normalize_image(frame, convert_bgr_to_rgb=convert_bgr_to_rgb)
    if image_rgb.ndim == 2:
        return cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
    return rgb_image_for_cv2_write(image_rgb)


def image_black_stats(images):
    if images is None or len(images) == 0:
        return None

    means = []
    stds = []
    for frame in images:
        frame = np.nan_to_num(np.asarray(frame, dtype=np.float32), nan=0.0)
        means.append(float(np.mean(frame)))
        stds.append(float(np.std(frame)))

    means = np.asarray(means, dtype=np.float32)
    stds = np.asarray(stds, dtype=np.float32)
    black_mask = np.logical_or(means < 5.0, stds < 1.0)
    return {
        "mean_min": float(np.min(means)),
        "mean_max": float(np.max(means)),
        "mean_avg": float(np.mean(means)),
        "std_min": float(np.min(stds)),
        "std_max": float(np.max(stds)),
        "std_avg": float(np.mean(stds)),
        "black_frame_ratio": float(np.mean(black_mask)),
    }


def selected_indices(num_frames, max_frames):
    if num_frames <= 0:
        return [], 1
    if max_frames is None or max_frames <= 0:
        stride = 1
    else:
        stride = max(1, int(math.ceil(num_frames / max_frames)))
    return list(range(0, num_frames, stride)), stride


def save_preview_frames(data, output_dir, max_frames, convert_bgr_to_rgb):
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    max_image_frames = max(
        [value.shape[0] for value in [data.get(HEAD_KEY), data.get(WRIST_KEY)] if value is not None],
        default=0,
    )
    indices, stride = selected_indices(max_image_frames, max_frames)
    print(f"Saving image preview frames with stride={stride}")

    for key, prefix in [(HEAD_KEY, "head"), (WRIST_KEY, "wrist")]:
        images = data.get(key)
        if images is None:
            continue
        for index in indices:
            if index >= images.shape[0]:
                continue
            image = normalize_image(
                images[index], convert_bgr_to_rgb=convert_bgr_to_rgb
            )
            out_path = frame_dir / f"{prefix}_{index:06d}.png"
            cv2.imwrite(str(out_path), rgb_image_for_cv2_write(image))


def resize_to_height(image, height):
    if image.shape[0] == height:
        return image
    scale = height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def draw_overlay(canvas, frame_index, joints=None, gripper=None):
    lines = [f"frame: {frame_index}"]
    if joints is not None and frame_index < len(joints):
        joint_values = np.asarray(joints[frame_index]).reshape(-1)[:6]
        first = " ".join(f"{value:.3f}" for value in joint_values[:3])
        second = " ".join(f"{value:.3f}" for value in joint_values[3:6])
        lines.append(f"j1-j3: {first}")
        lines.append(f"j4-j6: {second}")
    if gripper is not None and frame_index < len(gripper):
        lines.append(f"gripper: {float(np.asarray(gripper[frame_index]).reshape(-1)[0]):.3f}")

    y = 24
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24


def make_video_frame(head, wrist, index, joints, gripper, convert_bgr_to_rgb):
    head_frame = None
    wrist_frame = None
    if head is not None and index < head.shape[0]:
        head_frame = image_for_video(head[index], convert_bgr_to_rgb=convert_bgr_to_rgb)
    if wrist is not None and index < wrist.shape[0]:
        wrist_frame = image_for_video(wrist[index], convert_bgr_to_rgb=convert_bgr_to_rgb)

    if head_frame is None and wrist_frame is None:
        return None
    if head_frame is None:
        head_frame = np.zeros_like(wrist_frame)
    if wrist_frame is None:
        wrist_frame = np.zeros_like(head_frame)

    target_height = max(head_frame.shape[0], wrist_frame.shape[0])
    head_frame = resize_to_height(head_frame, target_height)
    wrist_frame = resize_to_height(wrist_frame, target_height)
    canvas = np.concatenate([head_frame, wrist_frame], axis=1)
    draw_overlay(canvas, index, joints=joints, gripper=gripper)
    return canvas


def save_preview_video(data, output_dir, max_frames, fps, convert_bgr_to_rgb):
    head = data.get(HEAD_KEY)
    wrist = data.get(WRIST_KEY)
    if head is None and wrist is None:
        print("WARNING: no head/wrist images available; skipping video")
        return

    num_image_frames = max(
        [value.shape[0] for value in [head, wrist] if value is not None],
        default=0,
    )
    indices, stride = selected_indices(num_image_frames, max_frames)
    out_path = output_dir / "episode_preview.mp4"

    writer = None
    written = 0
    joints = data.get(JOINT_KEY)
    gripper = normalize_gripper(data.get(GRIPPER_KEY))

    try:
        for index in indices:
            frame = make_video_frame(
                head,
                wrist,
                index,
                joints=joints,
                gripper=gripper,
                convert_bgr_to_rgb=convert_bgr_to_rgb,
            )
            if frame is None:
                continue
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    print(f"WARNING: failed to open video writer: {out_path}")
                    return
                print(f"Saving preview video with stride={stride}: {out_path}")
            writer.write(frame)
            written += 1
    finally:
        if writer is not None:
            writer.release()

    print(f"Saved {written} video frames")


def plot_joint_curve(joints, output_path):
    if joints is None:
        print(f"WARNING: {JOINT_KEY} missing; skipping {output_path.name}")
        return
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 2:
        print(f"WARNING: expected {JOINT_KEY} shape (T, 6), got {joints.shape}; skipping {output_path.name}")
        return
    plt.figure(figsize=(12, 7))
    for dim in range(min(6, joints.shape[-1])):
        plt.plot(joints[:, dim], label=f"joint{dim + 1}")
    plt.title("Piper Joint State")
    plt.xlabel("Frame")
    plt.ylabel("Radians")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_gripper_curve(gripper, output_path):
    if gripper is None:
        print(f"WARNING: {GRIPPER_KEY} missing; skipping {output_path.name}")
        return
    gripper = normalize_gripper(gripper)
    plt.figure(figsize=(12, 4))
    plt.plot(gripper[:, 0], label="gripper")
    plt.title("Piper Gripper")
    plt.xlabel("Frame")
    plt.ylabel("Opening")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_joint_delta_curve(joints, output_path):
    if joints is None:
        print(f"WARNING: {JOINT_KEY} missing; skipping {output_path.name}")
        return
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 2:
        print(f"WARNING: expected {JOINT_KEY} shape (T, 6), got {joints.shape}; skipping {output_path.name}")
        return
    plt.figure(figsize=(12, 7))
    if joints.shape[0] > 1:
        deltas = joints[1:, :6] - joints[:-1, :6]
        for dim in range(min(6, deltas.shape[-1])):
            plt.plot(deltas[:, dim], label=f"joint{dim + 1} delta")
    else:
        plt.text(0.5, 0.5, "Need at least 2 frames", ha="center", va="center")
    plt.title("Piper Joint Delta For Smoothness Check Only")
    plt.xlabel("Frame")
    plt.ylabel("Delta radians")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_qpos_curve(qpos, output_path):
    if qpos is None:
        print(f"WARNING: {QPOS_KEY} missing; skipping {output_path.name}")
        return
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2:
        print(f"WARNING: expected {QPOS_KEY} shape (T, 6), got {qpos.shape}; skipping {output_path.name}")
        return
    plt.figure(figsize=(12, 7))
    for dim in range(min(6, qpos.shape[-1])):
        plt.plot(qpos[:, dim], label=f"qpos{dim + 1}")
    plt.title("Piper Qpos / EEF Pose")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def finite_status(name, array):
    if array is None:
        return None
    finite = np.isfinite(np.asarray(array))
    has_bad = not bool(np.all(finite))
    print(f"{name} has NaN/Inf: {has_bad}")
    return has_bad


def build_summary(hdf5_path, data, num_frames, warnings):
    joints = data.get(JOINT_KEY)
    gripper = normalize_gripper(data.get(GRIPPER_KEY))

    summary = {
        "hdf5_path": str(hdf5_path),
        "num_frames": int(num_frames),
        "key_shapes": {
            key: list(value.shape)
            for key, value in data.items()
            if value is not None and hasattr(value, "shape")
        },
        "key_dtypes": {
            key: str(value.dtype)
            for key, value in data.items()
            if value is not None and hasattr(value, "dtype")
        },
        "image_mean_std": {},
        "black_frame_ratio": {},
        "warnings": warnings,
    }

    for key in [HEAD_KEY, WRIST_KEY]:
        stats = image_black_stats(data.get(key))
        if stats is not None:
            summary["image_mean_std"][key] = {
                name: value
                for name, value in stats.items()
                if name != "black_frame_ratio"
            }
            summary["black_frame_ratio"][key] = stats["black_frame_ratio"]

    if joints is not None:
        joints = np.asarray(joints, dtype=np.float32)
        if joints.ndim == 2 and joints.shape[0] > 0:
            summary["joint_min"] = np.min(joints[:, :6], axis=0)
            summary["joint_max"] = np.max(joints[:, :6], axis=0)
            if joints.shape[0] > 1:
                abs_delta = np.abs(joints[1:, :6] - joints[:-1, :6])
                summary["avg_abs_joint_delta"] = float(np.mean(abs_delta))
                summary["avg_abs_joint_delta_per_joint"] = np.mean(abs_delta, axis=0)
                summary["max_abs_joint_delta"] = float(np.max(abs_delta))
                summary["max_abs_joint_delta_per_joint"] = np.max(abs_delta, axis=0)
            else:
                summary["avg_abs_joint_delta"] = 0.0
                summary["avg_abs_joint_delta_per_joint"] = np.zeros((6,), dtype=np.float32)
                summary["max_abs_joint_delta"] = 0.0
                summary["max_abs_joint_delta_per_joint"] = np.zeros((6,), dtype=np.float32)
        else:
            summary["joint_warning"] = f"Expected {JOINT_KEY} shape (T, 6), got {joints.shape}"

    if gripper is not None:
        summary["gripper_min"] = float(np.min(gripper))
        summary["gripper_max"] = float(np.max(gripper))

    return as_jsonable(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and sanity-check a Piper real-robot HDF5 episode."
    )
    parser.add_argument("--hdf5_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--save_video", type=parse_bool, default=True)
    parser.add_argument("--save_frames", type=parse_bool, default=True)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--convert_bgr_to_rgb", type=parse_bool, default=False)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    warnings = []

    if not args.hdf5_path.exists():
        raise FileNotFoundError(args.hdf5_path)

    try:
        load_runtime_deps()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    with h5py.File(args.hdf5_path, "r") as root:
        data = {key: read_dataset(root, key, warnings) for key in ALL_KEYS}

    print("\n=== HDF5 Key Stats ===")
    for key in ALL_KEYS:
        print_key_stats(key, data.get(key))

    required_frame_lengths = get_frame_lengths(data, REQUIRED_KEYS)
    all_frame_lengths = get_frame_lengths(data)
    if required_frame_lengths:
        num_frames = min(required_frame_lengths.values())
        print(f"\nEpisode frames (minimum aligned length): {num_frames}")
        if len(set(all_frame_lengths.values())) != 1:
            message = f"WARNING: frame length mismatch: {all_frame_lengths}"
            print(message)
            warnings.append(message)
    else:
        num_frames = 0
        print("\nWARNING: no frame-like datasets found")

    print("\n=== Image Black-Frame Check ===")
    for key in [HEAD_KEY, WRIST_KEY]:
        stats = image_black_stats(data.get(key))
        if stats is None:
            continue
        print(
            f"{key}: mean_avg={stats['mean_avg']:.3f}, "
            f"std_avg={stats['std_avg']:.3f}, "
            f"black_frame_ratio={stats['black_frame_ratio']:.3f}"
        )

    print("\n=== Robot State Check ===")
    joints = data.get(JOINT_KEY)
    gripper = normalize_gripper(data.get(GRIPPER_KEY))
    finite_status("joint", joints)
    finite_status("gripper", gripper)
    if gripper is not None:
        gripper_min = float(np.min(gripper))
        gripper_max = float(np.max(gripper))
        print(f"gripper range: min={gripper_min:.6g}, max={gripper_max:.6g}")
        if gripper_min < -0.05 or gripper_max > 1.05:
            message = "WARNING: gripper values are outside the expected [0, 1] neighborhood"
            print(message)
            warnings.append(message)

    if args.save_frames:
        save_preview_frames(
            data,
            args.output_dir,
            max_frames=args.max_frames,
            convert_bgr_to_rgb=args.convert_bgr_to_rgb,
        )
    if args.save_video:
        save_preview_video(
            data,
            args.output_dir,
            max_frames=args.max_frames,
            fps=args.fps,
            convert_bgr_to_rgb=args.convert_bgr_to_rgb,
        )

    plot_joint_curve(joints, args.output_dir / "joint_curve.png")
    plot_gripper_curve(gripper, args.output_dir / "gripper_curve.png")
    plot_joint_delta_curve(joints, args.output_dir / "joint_delta_curve.png")
    plot_qpos_curve(data.get(QPOS_KEY), args.output_dir / "qpos_curve.png")

    summary = build_summary(args.hdf5_path, data, num_frames, warnings)
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
