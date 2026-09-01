from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import queue
import threading
import time
import typing as t

import numpy as np


# CODEX MODIFICATION: Shared asynchronous recorder used by both old and new Piper clients.
class PiperMotionDiagnosticsRecorder:
    """Asynchronously persist policy chunks and per-command Piper feedback."""

    _CONTROL_FIELDS = [
        "sample_index",
        "wall_time_ns",
        "monotonic_ns",
        "actual_interval_s",
        "control_step",
        "chunk_id",
        "chunk_action_index",
        "is_chunk_start",
        "rtc_enabled",
        "inference_running",
        "queue_remaining_before",
        "queue_remaining_after",
        "feedback_read_ms",
        "apply_action_ms",
        "feedback_error",
        "command_applied",
        "apply_error",
        *[f"command_joint_{idx + 1}_rad" for idx in range(6)],
        "command_gripper",
        *[f"state_joint_{idx + 1}_rad" for idx in range(6)],
        "state_gripper",
        "ee_x_m",
        "ee_y_m",
        "ee_z_m",
        "ee_rx_rad",
        "ee_ry_rad",
        "ee_rz_rad",
        "ee_x_raw_0p001mm",
        "ee_y_raw_0p001mm",
        "ee_z_raw_0p001mm",
        "ee_rx_raw_0p001deg",
        "ee_ry_raw_0p001deg",
        "ee_rz_raw_0p001deg",
        "joint_feedback_hz",
        "end_pose_feedback_hz",
        "gripper_feedback_hz",
    ]

    def __init__(self, output_dir: Path, metadata: dict[str, t.Any], queue_size: int = 50000) -> None:
        self.output_dir = Path(output_dir) / "motion_diagnostics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[tuple[str, t.Any]] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._closed = False
        self._counter_lock = threading.Lock()
        self._control_count = 0
        self._chunk_count = 0
        self._dropped_count = 0
        self._started_wall_time_ns = time.time_ns()

        metadata_payload = {
            "schema_version": 1,
            "created_wall_time_ns": self._started_wall_time_ns,
            "feedback_phase": "immediately_before_command",
            "files": {
                "chunks": "chunks.jsonl",
                "control_steps": "control_steps.csv",
                "summary": "summary.json",
            },
            **self._json_compatible(metadata),
        }
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as file:
            json.dump(metadata_payload, file, ensure_ascii=False, indent=2)

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="piper-motion-diagnostics-writer",
            daemon=True,
        )
        self._writer_thread.start()

    @staticmethod
    def _json_compatible(value: t.Any) -> t.Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): PiperMotionDiagnosticsRecorder._json_compatible(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [PiperMotionDiagnosticsRecorder._json_compatible(item) for item in value]
        return value

    @staticmethod
    def _vector(value: t.Any, size: int) -> np.ndarray:
        output = np.full(size, np.nan, dtype=np.float64)
        if value is None:
            return output
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        output[: min(size, vector.size)] = vector[:size]
        return output

    def _enqueue(self, kind: str, payload: t.Any) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait((kind, payload))
        except queue.Full:
            with self._counter_lock:
                self._dropped_count += 1

    def record_chunk(self, record: dict[str, t.Any]) -> None:
        payload = {
            "event": "chunk_switch",
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **record,
        }
        self._enqueue("chunk", self._json_compatible(payload))

    def record_control_step(
        self,
        *,
        control_step: int,
        chunk_id: int,
        chunk_action_index: int,
        command_action: np.ndarray,
        feedback: dict[str, t.Any],
        wall_time_ns: int,
        monotonic_ns: int,
        actual_interval_s: float,
        rtc_enabled: bool,
        inference_running: bool,
        queue_remaining_before: int,
        queue_remaining_after: int,
        feedback_read_ms: float,
        apply_action_ms: float,
        feedback_error: str = "",
        command_applied: bool = True,
        apply_error: str = "",
    ) -> None:
        command = self._vector(command_action, 7)
        state = self._vector(feedback.get("state"), 7)
        ee_xyz = self._vector(feedback.get("end_pose_xyz_m"), 3)
        ee_rpy = self._vector(feedback.get("end_pose_rpy_rad"), 3)
        ee_raw = self._vector(feedback.get("end_pose_raw"), 6)
        feedback_hz = self._vector(feedback.get("feedback_hz"), 3)
        row: dict[str, t.Any] = {
            "sample_index": -1,
            "wall_time_ns": int(wall_time_ns),
            "monotonic_ns": int(monotonic_ns),
            "actual_interval_s": float(actual_interval_s),
            "control_step": int(control_step),
            "chunk_id": int(chunk_id),
            "chunk_action_index": int(chunk_action_index),
            "is_chunk_start": int(chunk_action_index == 0),
            "rtc_enabled": int(bool(rtc_enabled)),
            "inference_running": int(bool(inference_running)),
            "queue_remaining_before": int(queue_remaining_before),
            "queue_remaining_after": int(queue_remaining_after),
            "feedback_read_ms": float(feedback_read_ms),
            "apply_action_ms": float(apply_action_ms),
            "feedback_error": str(feedback_error),
            "command_applied": int(bool(command_applied)),
            "apply_error": str(apply_error),
        }
        for idx in range(6):
            row[f"command_joint_{idx + 1}_rad"] = command[idx]
            row[f"state_joint_{idx + 1}_rad"] = state[idx]
        row["command_gripper"] = command[6]
        row["state_gripper"] = state[6]
        for key, value in zip(("ee_x_m", "ee_y_m", "ee_z_m"), ee_xyz, strict=True):
            row[key] = value
        for key, value in zip(("ee_rx_rad", "ee_ry_rad", "ee_rz_rad"), ee_rpy, strict=True):
            row[key] = value
        raw_fields = (
            "ee_x_raw_0p001mm",
            "ee_y_raw_0p001mm",
            "ee_z_raw_0p001mm",
            "ee_rx_raw_0p001deg",
            "ee_ry_raw_0p001deg",
            "ee_rz_raw_0p001deg",
        )
        for key, value in zip(raw_fields, ee_raw, strict=True):
            row[key] = value
        row["joint_feedback_hz"] = feedback_hz[0]
        row["end_pose_feedback_hz"] = feedback_hz[1]
        row["gripper_feedback_hz"] = feedback_hz[2]
        self._enqueue("control", row)

    def _writer_loop(self) -> None:
        chunks_path = self.output_dir / "chunks.jsonl"
        controls_path = self.output_dir / "control_steps.csv"
        try:
            with open(chunks_path, "a", encoding="utf-8") as chunks_file, open(
                controls_path,
                "a",
                newline="",
                encoding="utf-8",
            ) as controls_file:
                controls_writer = csv.DictWriter(controls_file, fieldnames=self._CONTROL_FIELDS)
                if controls_file.tell() == 0:
                    controls_writer.writeheader()
                pending_since_flush = 0
                while True:
                    kind, payload = self._queue.get()
                    if kind == "close":
                        break
                    if kind == "chunk":
                        chunks_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        with self._counter_lock:
                            self._chunk_count += 1
                    elif kind == "control":
                        with self._counter_lock:
                            sample_index = self._control_count
                            self._control_count += 1
                        payload["sample_index"] = sample_index
                        controls_writer.writerow(payload)
                    pending_since_flush += 1
                    if pending_since_flush >= 100:
                        chunks_file.flush()
                        controls_file.flush()
                        pending_since_flush = 0
                chunks_file.flush()
                controls_file.flush()
        except Exception:
            logging.exception("Piper motion diagnostics writer failed for %s", self.output_dir)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(("close", None))
        self._writer_thread.join(timeout=10.0)
        with self._counter_lock:
            summary = {
                "started_wall_time_ns": self._started_wall_time_ns,
                "closed_wall_time_ns": time.time_ns(),
                "control_sample_count": self._control_count,
                "chunk_count": self._chunk_count,
                "dropped_record_count": self._dropped_count,
                "writer_thread_alive": self._writer_thread.is_alive(),
            }
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
