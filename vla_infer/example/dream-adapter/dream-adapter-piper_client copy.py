from dataclasses import dataclass
from datetime import datetime
import csv
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import typing as t
import json
import draccus
import numpy as np
from PIL import Image

from vla_infer.src.inference.client import InferenceClient
from vla_infer.src.robots.piper_single import PiperSingleRobot
from vla_infer.src.zmq.zmq_client import VlaZmqClient
from vla_infer.src.process.utils import (
    adaptive_resize_image,
	ensure_hwc3_uint8_image,
    uint8_image_to_float32_01,
    smooth_action_chunk,
    delta_action_chunk_to_absolute,
	check_uint8_rgb,
	interpolate_action_chunk,
)
import sys
from pathlib import Path
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
sys.path.append(str(REPO_ROOT))
ArrayTransform = t.Callable[[np.ndarray], np.ndarray]


# CODEX MODIFICATION: RTC action queue keeps raw policy chunks for leftover guidance and processed chunks for execution.
class RTCActionQueue:
	"""Thread-safe action queue that keeps raw policy chunks for RTC leftovers."""

	def __init__(self) -> None:
		self._lock = threading.Lock()
		self._original_queue: t.Optional[np.ndarray] = None
		self._processed_queue: t.Optional[np.ndarray] = None
		self._last_processed_index = 0

	@staticmethod
	def _as_2d(actions: np.ndarray) -> np.ndarray:
		actions = np.asarray(actions, dtype=np.float32)
		if actions.ndim == 1:
			return actions[None, :]
		if actions.ndim == 2:
			return actions
		raise ValueError(f"action queue expects 1D or 2D actions, got shape={actions.shape}")

	@staticmethod
	def _processed_to_original_index(processed_index: int, original_len: int, processed_len: int) -> int:
		if original_len <= 0 or processed_len <= 0:
			return 0
		return min(original_len, max(0, int(np.floor(processed_index * original_len / processed_len))))

	@staticmethod
	def _original_to_processed_index(original_index: int, original_len: int, processed_len: int) -> int:
		if original_len <= 0 or processed_len <= 0:
			return 0
		return min(processed_len, max(0, int(np.floor(original_index * processed_len / original_len))))

	def clear(self) -> None:
		with self._lock:
			self._original_queue = None
			self._processed_queue = None
			self._last_processed_index = 0

	def qsize(self) -> int:
		with self._lock:
			if self._processed_queue is None:
				return 0
			return max(0, len(self._processed_queue) - self._last_processed_index)

	def empty(self) -> bool:
		return self.qsize() <= 0

	def get_action_index(self) -> int:
		with self._lock:
			return self._last_processed_index

	def snapshot(self) -> dict[str, int]:
		with self._lock:
			return {
				"original_len": 0 if self._original_queue is None else len(self._original_queue),
				"processed_len": 0 if self._processed_queue is None else len(self._processed_queue),
				"last_processed_index": self._last_processed_index,
			}

	def get_left_over(self) -> t.Optional[np.ndarray]:
		with self._lock:
			if self._original_queue is None or self._processed_queue is None:
				return None
			original_start = self._processed_to_original_index(
				self._last_processed_index,
				len(self._original_queue),
				len(self._processed_queue),
			)
			left_over = self._original_queue[original_start:]
			if left_over.size == 0:
				return None
			return left_over.copy()

	def get(self) -> t.Optional[np.ndarray]:
		with self._lock:
			if self._processed_queue is None or self._last_processed_index >= len(self._processed_queue):
				return None
			action = self._processed_queue[self._last_processed_index].copy()
			self._last_processed_index += 1
			return action

	def merge(self, original_actions: np.ndarray, processed_actions: np.ndarray, real_delay_model_steps: int) -> None:
		original_actions = self._as_2d(original_actions)
		processed_actions = self._as_2d(processed_actions)
		delay_original = min(max(0, int(real_delay_model_steps)), len(original_actions))
		delay_processed = self._original_to_processed_index(
			delay_original,
			len(original_actions),
			len(processed_actions),
		)
		with self._lock:
			self._original_queue = original_actions[delay_original:].copy()
			self._processed_queue = processed_actions[delay_processed:].copy()
			self._last_processed_index = 0


@dataclass
class InferenceConfig:
	"""Single runtime config for Piper inference client.
	"""

	server_ip: str = "127.0.0.1"
	port: int = 5555
	timeout_ms: int = 2000
	jpeg_quality: int = 80

	task_instruction: str = "Pick up the banana and place it in the bowl"
	max_steps: int = 1000
	stop_on_timeout: bool = True

	action_key: str = "action"
	# CODEX MODIFICATION: Request and retain the first N raw policy actions; 0 keeps the server's full chunk.
	action_horizon: int = 0
	execute_chunk_steps: int = 8
	control_interval_s: float = 0.01
	log_level: str = "INFO"
	# CODEX MODIFICATION: Require an Enter press before each robot action when enabled.
	cautious_execute: bool = False

	state_type: str = "qpos"
	action_type: str = "joint"	
	absolute_action: bool = True 

	enable_binary_gripper: bool = False
	binary_gripper_threshold: float = 0.4
	gripper_open_value: float = 0.5
	gripper_closed_value: float = 0.2

	enable_action_interpolation: bool = False
	use_smoothing: bool = False
	interpolation_method: str = "linear"
	interpolation_target_steps: int = 333
	# CODEX MODIFICATION: Generate this many control commands for each retained raw policy action.
	interpolation_steps_per_action: int = 10

	show_output_track: bool = False
	# CODEX MODIFICATION: Save both camera streams observed by the policy client.
	record_cameras: bool = False
	camera_record_dir: str = "/home/charles/workspaces/Double_Piper_Teleop/camera_records"
	camera_record_run_name: str = ""
	camera_record_jpeg_quality: int = 95
	camera_record_video: bool = True
	camera_record_video_fps: float = 0.0
	camera_record_every_n_steps: int = 5
	# CODEX MODIFICATION: Record state/action values and save review plots locally.
	record_state_action: bool = False
	state_action_record_dir: str = "/home/charles/workspaces/Double_Piper_Teleop/state_action_records"
	state_action_record_run_name: str = ""
	state_action_plot: bool = True

	enable_gripper_transform: bool = False
	gripper_transform_threshold: float = 0.55
	gripper_transform_delta: float = 0.3

	# CODEX MODIFICATION: RTC runtime switch and guidance parameters; disabled keeps the original synchronous path.
	enable_rtc: bool = False
	rtc_execution_horizon: int = 10
	rtc_max_guidance_weight: float = 10.0
	rtc_prefix_attention_schedule: str = "LINEAR"
	# CODEX MODIFICATION: A positive fixed threshold overrides ratio and automatic RTC replanning.
	rtc_replan_remaining_steps: int = 0
	# CODEX MODIFICATION: A positive ratio overrides automatic RTC replanning; 0 selects automatic mode.
	rtc_replan_remaining_ratio: float = 0.0
	rtc_empty_queue_sleep_s: float = 0.001

class PiperVLAClient(InferenceClient):
	"""Client runtime that bridges PiperSingleRobot and VLA server.

	Observation contract returned by `get_observation()`:
	{
	  "state": np.ndarray(7,),
	  "image": np.ndarray(H, W, 3),
	  "wrist_image": np.ndarray(H, W, 3),
	  "prev_action": np.ndarray(T, D)   # optional, controlled by config
	}
	"""

	def __init__(
		self,
		cfg: InferenceConfig,
		robot: t.Optional[PiperSingleRobot] = None,
		client: t.Optional[VlaZmqClient] = None,
	) -> None:
		self.cfg = cfg
		if self.cfg.action_horizon < 0:
			raise ValueError("action_horizon must be >= 0")
		if self.cfg.interpolation_steps_per_action <= 0:
			raise ValueError("interpolation_steps_per_action must be > 0")
		if self.cfg.rtc_replan_remaining_steps < 0:
			raise ValueError("rtc_replan_remaining_steps must be >= 0")
		if not 0.0 <= self.cfg.rtc_replan_remaining_ratio <= 1.0:
			raise ValueError("rtc_replan_remaining_ratio must be between 0.0 and 1.0")
		self.show_output_track = cfg.show_output_track
		logging.basicConfig(
			level=getattr(logging, cfg.log_level.upper(), logging.INFO),
			format="%(asctime)s - %(levelname)s - %(message)s",
		)

		# CODEX MODIFICATION: Used by cautious_execute to stop the outer loop safely.
		self._stop_requested = False
		# CODEX MODIFICATION: Camera recording is optional and captures frames for each executed action step.
		self._camera_record_dir: t.Optional[Path] = None
		self._camera_record_log: t.Optional[t.TextIO] = None
		self._camera_record_finalized = False
		self._camera_record_step = 0
		if self.cfg.record_cameras:
			run_name = self.cfg.camera_record_run_name or self._make_camera_record_run_name()
			self._camera_record_dir = Path(self.cfg.camera_record_dir) / run_name
			(self._camera_record_dir / "cam_head").mkdir(parents=True, exist_ok=True)
			(self._camera_record_dir / "cam_wrist").mkdir(parents=True, exist_ok=True)
			metadata = {
				"created_at": datetime.now().isoformat(timespec="seconds"),
				"task_instruction": self.cfg.task_instruction,
				"state_type": self.cfg.state_type,
				"action_type": self.cfg.action_type,
				"absolute_action": self.cfg.absolute_action,
				"camera_record_every_n_steps": self.cfg.camera_record_every_n_steps,
				"image_sources": {
					"cam_head": "observation.image",
					"cam_wrist": "observation.wrist_image",
				},
			}
			with open(self._camera_record_dir / "metadata.json", "w", encoding="utf-8") as f:
				json.dump(metadata, f, indent=2)
			self._camera_record_log = open(self._camera_record_dir / "records.jsonl", "a", encoding="utf-8")
			logging.info("Recording camera frames to %s", self._camera_record_dir)
		# CODEX MODIFICATION: State/action recording is separate from camera recording so either can be enabled alone.
		self._state_action_record_dir: t.Optional[Path] = None
		self._state_action_records: list[dict[str, t.Any]] = []
		self._state_action_finalized = False
		if self.cfg.record_state_action:
			run_name = self.cfg.state_action_record_run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
			self._state_action_record_dir = Path(self.cfg.state_action_record_dir) / run_name
			self._state_action_record_dir.mkdir(parents=True, exist_ok=True)
			metadata = {
				"created_at": datetime.now().isoformat(timespec="seconds"),
				"task_instruction": self.cfg.task_instruction,
				"state_type": self.cfg.state_type,
				"action_type": self.cfg.action_type,
				"absolute_action": self.cfg.absolute_action,
				"execute_chunk_steps": self.cfg.execute_chunk_steps,
				"control_interval_s": self.cfg.control_interval_s,
			}
			with open(self._state_action_record_dir / "metadata.json", "w", encoding="utf-8") as f:
				json.dump(metadata, f, indent=2)
			logging.info("Recording state/action data to %s", self._state_action_record_dir)

		self.robot = robot if robot is not None else PiperSingleRobot()
		time.sleep(2)
		# CODEX MODIFICATION: In cautious mode, reset is also gated because reset moves the real arm.
		if self.cfg.cautious_execute:
			user_input = input(
				"[cautious_execute] Robot is initialized. Press Enter to run reset(), "
				"or type q then Enter to quit before reset: "
			).strip().lower()
			if user_input in {"q", "quit", "stop", "exit"}:
				logging.info("Cautious execution stopped by user before reset().")
				self._stop_requested = True
			else:
				self.robot.reset()
				time.sleep(2)
		else:
			self.robot.reset()
			time.sleep(2)
		self.zmq_client = (
			client
			if client is not None
			else VlaZmqClient(
				server_ip=cfg.server_ip,
				port=cfg.port,
				timeout_ms=cfg.timeout_ms,
			)
		)
		self.obs: t.Dict[str, t.Any] = {}
		# CODEX MODIFICATION: RTC runtime state for action queue and background inference.
		self._rtc_queue = RTCActionQueue()
		self._rtc_inference_thread: t.Optional[threading.Thread] = None
		self._rtc_inference_result: t.Optional[dict[str, t.Any]] = None
		self._rtc_inference_error: t.Optional[BaseException] = None
		self._rtc_inference_lock = threading.Lock()
		self._rtc_last_inference_delay_model_steps = 0
		self._rtc_inference_cycle = 0

	@staticmethod
	def _slugify_for_path(value: str, max_len: int = 64) -> str:
		value = re.sub(r"[^\w]+", "_", value.strip().lower(), flags=re.UNICODE).strip("_")
		value = re.sub(r"_+", "_", value)
		return (value[:max_len].strip("_") or "task")

	def _make_camera_record_run_name(self) -> str:
		# CODEX MODIFICATION: Include the task instruction in camera record folder names.
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		task_slug = self._slugify_for_path(self.cfg.task_instruction)
		return f"{timestamp}_{task_slug}"

	def _apply_action_horizon(self, action: np.ndarray) -> np.ndarray:
		# CODEX MODIFICATION: Trim the raw server chunk before any action post-processing.
		action_2d = np.asarray(action, dtype=np.float32)
		if action_2d.ndim == 1:
			action_2d = action_2d[None, :]
		elif action_2d.ndim != 2:
			raise ValueError(f"server action must be 1D or 2D, got shape={action_2d.shape}")

		action_horizon = int(self.cfg.action_horizon)
		if action_horizon <= 0:
			return action_2d.copy()
		return action_2d[: min(action_horizon, action_2d.shape[0])].copy()
  
	def get_observation(self) -> t.Dict[str, t.Any]:
		"""Abstract step 1: collect and preprocess one observation payload."""
		raw_obs = self.robot.get_observation()
		# set state as qpos
		if self.cfg.state_type == "qpos":
			qpos_value = raw_obs.get("qpos",np.zeros(6,dtype=np.float32))
			gripper_value = np.asarray([raw_obs.get("gripper", 0.0)], dtype=np.float32)
			state = np.concatenate([qpos_value, gripper_value], axis=0)
		elif self.cfg.state_type == "joint":
			state = raw_obs.get("state", np.zeros(7, dtype=np.float32))

		obs = {
			"state": state,
			"image": raw_obs.get("cam_head"),
			"wrist_image": raw_obs.get("cam_wrist"),
		}
		# # adaptive resize image
		obs["image"] = check_uint8_rgb(adaptive_resize_image(obs["image"]))
		obs["wrist_image"] = check_uint8_rgb(adaptive_resize_image(obs["wrist_image"]))
		# Ensure images are HWC3 uint8 before resize to satisfy model input contract.
		# obs["image"] = ensure_hwc3_uint8_image(np.asarray(obs["image"]))
		# obs["wrist_image"] = ensure_hwc3_uint8_image(np.asarray(obs["wrist_image"]))
  
		# binary gripper state (open/close) for discrete action models, determined by thresholding the last element of state vector (gripper position)
		if self.cfg.enable_binary_gripper:
			obs["state"][-1] = 1.0 if obs["state"][-1] > self.cfg.binary_gripper_threshold else 0.0

		# if self.cfg.enable_gripper_transform:	
		# 	# transform gripper state based on threshold to potentially enhance model's ability to learn discrete open/close behavior, by creating a larger gap between open and close states in the input space
		# 	obs["state"][-1] = obs["state"][-1] + self.cfg.gripper_transform_delta if obs["state"][-1] < self.cfg.gripper_transform_threshold else obs["state"][-1]

		self.obs = obs # save for later use in execute
		if self.cfg.action_type == "joint" and self.cfg.state_type == "qpos":
			self.obs["joint_state"] = raw_obs.get("state", np.zeros(7, dtype=np.float32)) # save joint state for later use in absolute conversion

		return obs

	def get_response(
		self,
		observation: t.Dict[str, t.Any],
		task_instruction: t.Optional[str] = None,
		rtc_prev_chunk_left_over: t.Optional[np.ndarray] = None,
		rtc_inference_delay: int = 0,
	) -> t.Any:
		"""send observation an--------------------d get server response."""
		observation = dict(observation)
		# Log observation details for debugging.
		for key, value in observation.items():
			if value is not None and hasattr(value, "shape") and hasattr(value, "dtype"):
				logging.debug(f"Observation '{key}' shape={value.shape} dtype={value.dtype}")

		# set cmd to task_instruction if provided, otherwise use default from config
		observation["cmd"] = task_instruction or self.cfg.task_instruction
		logging.debug(f"Observation 'cmd'='{observation['cmd']}'")
		# CODEX MODIFICATION: Ask compatible servers to return only the requested raw action horizon.
		requested_action_horizon = int(self.cfg.action_horizon)
		if requested_action_horizon > 0:
			observation["requested_action_horizon"] = requested_action_horizon

		# CODEX MODIFICATION: RTC request fields tell the server about leftover actions and inference delay.
		if self.cfg.enable_rtc:
			observation["rtc_enabled"] = True
			observation["rtc_inference_delay"] = int(max(0, rtc_inference_delay))
			observation["rtc_execution_horizon"] = int(max(1, self.cfg.rtc_execution_horizon))
			observation["rtc_max_guidance_weight"] = float(self.cfg.rtc_max_guidance_weight)
			observation["rtc_prefix_attention_schedule"] = self.cfg.rtc_prefix_attention_schedule.upper()
			if rtc_prev_chunk_left_over is not None and np.asarray(rtc_prev_chunk_left_over).size:
				prev_chunk = np.asarray(rtc_prev_chunk_left_over, dtype=np.float32)
				observation["rtc_prev_chunk_left_over"] = prev_chunk
				observation["rtc_prev_chunk_length"] = int(prev_chunk.shape[0])

		# send observation to server and get response
		# response = {"action": np.ndarray(T, D), ...}
		server_response = self.zmq_client.get_response(obs_dict=observation)
		if "error" in server_response:
			raise RuntimeError(f"VLA server error:\n{server_response['error']}")
		action = self._apply_action_horizon(server_response["action"])
		return {"action": action}

	def _postprocess_action_chunk(
		self,
		action: np.ndarray,
		observation_context: t.Optional[t.Dict[str, t.Any]] = None,
	) -> np.ndarray:
		"""Convert raw server actions into robot-executable actions."""
		action = self._apply_action_horizon(action)
		obs_context = observation_context or self.obs

		if self.cfg.enable_binary_gripper:
			# beacause the value of gripper is either 0 or 1, we can reuse the binary threshold to determine open/close 
   			# and set to predefined values for better sim realism (instead of 0/1 which may not be the actual command value for the robot)
			action[:, -1] = np.where(
				action[:, -1] > self.cfg.binary_gripper_threshold,
				self.cfg.gripper_open_value,
				self.cfg.gripper_closed_value,
			)
			# print("Gripper action before smoothing:", action[:, -1])
		# normal absolute + smoothing
		if not self.cfg.enable_binary_gripper:
			if self.cfg.state_type == "qpos":
				abs_action = delta_action_chunk_to_absolute(
					obs_context.get("joint_state", np.zeros(7, dtype=np.float32)),
					action,
				)
			elif self.cfg.state_type == "joint":
				if self.cfg.absolute_action:
					abs_action = action
				else:
					abs_action = delta_action_chunk_to_absolute(
						obs_context.get("state", np.zeros(7, dtype=np.float32)),
						action,
					)
			if self.cfg.use_smoothing:
				smooth_action = smooth_action_chunk(abs_action,max_angular_acceleration=0.01,max_angular_jerk=0.01)
			else:
				smooth_action = abs_action
		# binary gripper
		else:
			# only smooth the first 6 dimensions for the robot joints, keep the gripper command as is to preserve the discrete open/close behavior
			if self.cfg.state_type == "qpos":
				abs_action = delta_action_chunk_to_absolute(
					obs_context.get("joint_state", np.zeros(7, dtype=np.float32))[:6],
					action[:, :6],
				) # only convert the first 6 dimensions for absolute, keep gripper as is
			else:
				abs_action = delta_action_chunk_to_absolute(
					obs_context.get("state", np.zeros(7, dtype=np.float32))[:6],
					action[:, :6],
				)
			abs_action = np.concatenate([abs_action, action[:, -1:]], axis=-1) # concatenate the gripper command back before smoothing, so that the smoothing function can keep it unchanged
			smooth_action = smooth_action_chunk(abs_action,max_angular_acceleration=0.01,max_angular_jerk=0.01)
			print("abs action after smoothing:", smooth_action)
		if self.cfg.enable_gripper_transform:
			delta_gripper= smooth_action[:, -1] - self.cfg.gripper_transform_delta * (action[:, -1] < self.cfg.gripper_transform_threshold) # if the original action command is below the threshold, we assume it's a close command and we further decrease the gripper value in the smoothed action to enhance the close signal; if it's above the threshold, we keep it unchanged to preserve the open signal
			smooth_action[:, -1]= delta_gripper
			
		if self.cfg.enable_action_interpolation:
			# CODEX MODIFICATION: Scale interpolation with the retained horizon (for example, 4 * 10 = 40).
			target_steps = smooth_action.shape[0] * self.cfg.interpolation_steps_per_action
			smooth_action = interpolate_action_chunk(
				smooth_action,
				target_steps=target_steps,
				method=self.cfg.interpolation_method,
			)

  		# ensure action is 2D (T, D)
		if smooth_action.ndim == 1:
			action_2d = smooth_action[None, :]
		elif smooth_action.ndim == 2:
			action_2d = smooth_action
		else:
			raise ValueError(f"action must be 1D or 2D, got shape={smooth_action.shape}")

		return action_2d

	def execute(self, response: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
		"""execute action chunk on robot."""
		action_2d = self._postprocess_action_chunk(response["action"])

		execute_steps = min(max(1, self.cfg.execute_chunk_steps), action_2d.shape[0])

		##################
		# action_2d = action_2d[execute_steps:]
		##################

		for idx in range(execute_steps):
			# CODEX MODIFICATION: Gate each robot action on explicit user confirmation.
			if self.cfg.cautious_execute:
				action_preview = np.round(action_2d[idx], 4).tolist()
				user_input = input(
					f"[cautious_execute] action {idx + 1}/{execute_steps}: {action_preview}\n"
					"Press Enter to execute this action, or type q then Enter to skip the rest: "
				).strip().lower()
				if user_input in {"q", "quit", "stop", "exit"}:
					logging.info("Cautious execution stopped by user before action %s/%s.", idx + 1, execute_steps)
					self._stop_requested = True
					# CODEX MODIFICATION: Report partial execution and let the outer loop stop cleanly.
					return {
						"executed_steps": idx,
						"output_action": action_2d[:idx],
						"action_shape": tuple(action_2d.shape),
						"stopped_by_user": True,
					}

			#  self.robot.get_state()["state"] 
			step_start = time.monotonic()
			self.robot.apply_action({"action":action_2d[idx]})
			self._record_action_camera_frame(action_2d[idx])
			if self.cfg.control_interval_s > 0:
				elapsed = time.monotonic() - step_start
				time.sleep(max(0.0, self.cfg.control_interval_s - elapsed))

		return {
			"executed_steps": execute_steps,
			"output_action": action_2d[:execute_steps],
			"action_shape": tuple(action_2d.shape),
			"stopped_by_user": False,
		}
	
	def run_once(self) -> t.Dict[str, t.Any]:
		"""Run one full observe-send-receive-execute cycle."""
		observation = self.get_observation()
		response = self.get_response(observation)
		execution_result = self.execute(response)

		return {
			"action":  response,
			"execution": execution_result,
			"observation": observation,
		}

	def _process_rtc_response(
		self,
		result: dict[str, t.Any],
		real_delay_model_steps: int,
	) -> t.Dict[str, t.Any]:
		# CODEX MODIFICATION: RTC merges a server chunk into the executable queue after local post-processing.
		raw_action = np.asarray(result["response"]["action"], dtype=np.float32)
		processed_action = self._postprocess_action_chunk(raw_action, result["observation"])
		self._rtc_queue.merge(raw_action, processed_action, real_delay_model_steps)
		self._rtc_last_inference_delay_model_steps = int(max(0, real_delay_model_steps))
		return {
			"action": result["response"],
			"execution": {
				"executed_steps": 0,
				"output_action": processed_action,
				"action_shape": tuple(processed_action.shape),
				"stopped_by_user": False,
			},
			"observation": result["observation"],
		}

	def _rtc_inference_running(self) -> bool:
		thread = self._rtc_inference_thread
		return thread is not None and thread.is_alive()

	def _estimate_model_delay_from_elapsed(self, elapsed_s: float, original_len: int, processed_len: int) -> int:
		# CODEX MODIFICATION: RTC estimates how many model-space steps elapsed during inference.
		if self.cfg.control_interval_s <= 0:
			return 0
		if original_len > 0 and processed_len > 0:
			model_step_s = self.cfg.control_interval_s * processed_len / original_len
			# CODEX MODIFICATION: Round up so a partial raw step cannot replay already-expired interpolated controls.
			return max(0, int(np.ceil(elapsed_s / model_step_s)))
		return max(0, int(np.ceil(elapsed_s / self.cfg.control_interval_s)))

	def _auto_rtc_replan_remaining_steps(self) -> int:
		# CODEX MODIFICATION: Keep enough processed actions for measured policy latency plus a 15% safety margin.
		snapshot = self._rtc_queue.snapshot()
		original_len = snapshot["original_len"]
		processed_len = snapshot["processed_len"]
		if original_len <= 0 or processed_len <= 0:
			return 0

		delay_model_steps = max(1, self._rtc_last_inference_delay_model_steps)
		delay_processed_steps = RTCActionQueue._original_to_processed_index(
			delay_model_steps,
			original_len,
			processed_len,
		)
		safety_steps = max(10, int(round(processed_len * 0.15)))
		return min(processed_len, max(1, delay_processed_steps + safety_steps))

	def _take_rtc_inference_result(self) -> tuple[t.Optional[dict[str, t.Any]], t.Optional[BaseException]]:
		# CODEX MODIFICATION: RTC safely transfers background inference results to the control loop.
		with self._rtc_inference_lock:
			result = self._rtc_inference_result
			error = self._rtc_inference_error
			self._rtc_inference_result = None
			self._rtc_inference_error = None
		return result, error

	def _start_rtc_inference(self) -> bool:
		# CODEX MODIFICATION: RTC starts a background policy request with current leftover actions.
		if self._rtc_inference_running():
			return False
		with self._rtc_inference_lock:
			if self._rtc_inference_result is not None or self._rtc_inference_error is not None:
				return False

		observation = self.get_observation()
		prev_left_over = self._rtc_queue.get_left_over()
		queue_snapshot = self._rtc_queue.snapshot()
		estimated_delay = self._rtc_last_inference_delay_model_steps if prev_left_over is not None else 0
		start_time = time.monotonic()

		def worker() -> None:
			try:
				response = self.get_response(
					observation,
					rtc_prev_chunk_left_over=prev_left_over,
					rtc_inference_delay=estimated_delay,
				)
				elapsed_s = time.monotonic() - start_time
				end_index = self._rtc_queue.get_action_index()
				original_len = queue_snapshot["original_len"]
				processed_len = queue_snapshot["processed_len"]
				if original_len > 0 and processed_len > 0:
					processed_delay = max(0, end_index - queue_snapshot["last_processed_index"])
					# CODEX MODIFICATION: Conservatively map fractional interpolated progress to whole model steps.
					real_delay_model_steps = max(
						0,
						int(np.ceil(processed_delay * original_len / processed_len)),
					)
				elif self.cfg.control_interval_s > 0:
					real_delay_model_steps = self._estimate_model_delay_from_elapsed(elapsed_s, original_len, processed_len)
				else:
					real_delay_model_steps = 0
				with self._rtc_inference_lock:
					self._rtc_inference_result = {
						"observation": observation,
						"response": response,
						"real_delay_model_steps": real_delay_model_steps,
						"elapsed_s": elapsed_s,
						"estimated_delay_model_steps": estimated_delay,
					}
			except BaseException as exc:
				with self._rtc_inference_lock:
					self._rtc_inference_error = exc

		self._rtc_inference_thread = threading.Thread(target=worker, name="piper-vla-rtc-inference", daemon=True)
		self._rtc_inference_thread.start()
		return True

	def _should_start_rtc_inference(self) -> bool:
		# CODEX MODIFICATION: RTC decides when the action queue is low enough to request a new chunk.
		if self._rtc_inference_running():
			return False
		remaining = self._rtc_queue.qsize()
		if remaining <= 0:
			return True
		snapshot = self._rtc_queue.snapshot()
		processed_horizon = snapshot["processed_len"]
		trigger_steps = int(self.cfg.rtc_replan_remaining_steps)
		trigger_mode = "fixed"
		if trigger_steps <= 0:
			ratio = float(self.cfg.rtc_replan_remaining_ratio)
			if ratio > 0.0:
				trigger_mode = "ratio"
				trigger_steps = max(1, int(np.floor(processed_horizon * ratio)))
			else:
				trigger_mode = "auto"
				trigger_steps = self._auto_rtc_replan_remaining_steps()
		should_start = remaining <= trigger_steps
		if should_start:
			logging.info(
				"RTC replan trigger mode=%s remaining_actions=%s trigger_steps=%s processed_horizon=%s remaining_ratio=%.3f",
				trigger_mode,
				remaining,
				trigger_steps,
				snapshot["processed_len"],
				remaining / snapshot["processed_len"] if snapshot["processed_len"] > 0 else 0.0,
			)
		return should_start

	def _execute_rtc_action(self, action: np.ndarray, step: int) -> bool:
		# CODEX MODIFICATION: RTC executes one queued action at the fixed control interval.
		if self.cfg.cautious_execute:
			action_preview = np.round(action, 4).tolist()
			user_input = input(
				f"[cautious_execute][rtc] action step {step}: {action_preview}\n"
				"Press Enter to execute this action, or type q then Enter to stop: "
			).strip().lower()
			if user_input in {"q", "quit", "stop", "exit"}:
				logging.info("RTC cautious execution stopped by user before action step %s.", step)
				self._stop_requested = True
				return False

		self.robot.apply_action({"action": action})
		return True

	def _merge_pending_rtc_result(self) -> bool:
		# CODEX MODIFICATION: RTC swaps in a finished background chunk without blocking execution.
		result, error = self._take_rtc_inference_result()
		if error is not None:
			raise error
		if result is None:
			return False
		cycle_report = self._process_rtc_response(result, result["real_delay_model_steps"])
		self._record_state_action_data(self._rtc_inference_cycle, cycle_report)
		logging.info(
			"RTC merged chunk cycle=%s real_delay_model_steps=%s estimated_delay_model_steps=%s elapsed_ms=%.1f remaining_actions=%s",
			self._rtc_inference_cycle,
			result["real_delay_model_steps"],
			result["estimated_delay_model_steps"],
			result["elapsed_s"] * 1000,
			self._rtc_queue.qsize(),
		)
		self._rtc_inference_cycle += 1
		return True

	def run_rtc(self, max_steps: t.Optional[int] = None) -> None:
		# CODEX MODIFICATION: RTC main loop runs robot actions while policy inference happens in the background.
		"""Run RTC real-time control loop with background policy inference."""
		step_limit = self.cfg.max_steps if max_steps is None else max_steps
		if step_limit <= 0:
			raise ValueError("max_steps must be > 0")
		if self._stop_requested:
			logging.info("Piper RTC loop was not started because user stopped before reset().")
			return

		logging.info(
			"Starting Piper RTC loop. action_step_limit=%s control_interval_s=%s execution_horizon=%s",
			step_limit,
			self.cfg.control_interval_s,
			self.cfg.rtc_execution_horizon,
		)

		initial_observation = self.get_observation()
		initial_start = time.monotonic()
		initial_response = self.get_response(initial_observation, rtc_inference_delay=0)
		initial_elapsed_s = time.monotonic() - initial_start
		initial_report = self._process_rtc_response(
			{"observation": initial_observation, "response": initial_response},
			real_delay_model_steps=0,
		)
		queue_snapshot = self._rtc_queue.snapshot()
		self._rtc_last_inference_delay_model_steps = self._estimate_model_delay_from_elapsed(
			initial_elapsed_s,
			queue_snapshot["original_len"],
			queue_snapshot["processed_len"],
		)
		self._record_state_action_data(self._rtc_inference_cycle, initial_report)
		self._rtc_inference_cycle += 1

		action_step = 0
		while action_step < step_limit and not self._stop_requested:
			self._merge_pending_rtc_result()
			if self._should_start_rtc_inference():
				self._start_rtc_inference()

			action = self._rtc_queue.get()
			if action is None:
				if not self._rtc_inference_running():
					self._start_rtc_inference()
				time.sleep(max(0.0, self.cfg.rtc_empty_queue_sleep_s))
				continue

			step_start = time.monotonic()
			if not self._execute_rtc_action(action, action_step):
				break
			self._record_action_camera_frame(action)
			action_step += 1

			if self.cfg.control_interval_s > 0:
				elapsed = time.monotonic() - step_start
				time.sleep(max(0.0, self.cfg.control_interval_s - elapsed))

		if self._rtc_inference_thread is not None and self._rtc_inference_thread.is_alive():
			logging.info("Waiting for in-flight RTC inference to finish before shutdown.")
			self._rtc_inference_thread.join(timeout=1.0)

	def _record_action_camera_frame(self, action: np.ndarray) -> None:
		# CODEX MODIFICATION: Record camera frames at a lower default rate so logging does not dominate control.
		if not self.cfg.record_cameras:
			return
		action_step = self._camera_record_step
		self._camera_record_step += 1
		record_every = max(1, int(self.cfg.camera_record_every_n_steps))
		if action_step % record_every != 0:
			return

		observation = self.get_observation()
		output_action = np.asarray(action, dtype=np.float32)[None, :]
		cycle_report = {
			"observation": observation,
			"execution": {
				"output_action": output_action,
				"action_shape": tuple(output_action.shape),
				"stopped_by_user": False,
			},
		}
		frame_index = action_step // record_every
		self._record_camera_data(frame_index, cycle_report, action_step)

	@staticmethod
	def _save_rgb_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
		# CODEX MODIFICATION: Persist camera frames without changing the in-memory observation.
		image = check_uint8_rgb(np.asarray(image))
		Image.fromarray(image).save(path, format="JPEG", quality=quality)

	def _record_camera_data(
		self,
		frame_index: int,
		cycle_report: t.Dict[str, t.Any],
		action_step: int,
	) -> None:
		# CODEX MODIFICATION: Store the two camera streams plus a JSONL index for review.
		if not self.cfg.record_cameras or self._camera_record_dir is None or self._camera_record_log is None:
			return

		observation = cycle_report["observation"]
		cam_head_rel = Path("cam_head") / f"{frame_index:06d}.jpg"
		cam_wrist_rel = Path("cam_wrist") / f"{frame_index:06d}.jpg"
		self._save_rgb_jpeg(
			self._camera_record_dir / cam_head_rel,
			observation["image"],
			self.cfg.camera_record_jpeg_quality,
		)
		self._save_rgb_jpeg(
			self._camera_record_dir / cam_wrist_rel,
			observation["wrist_image"],
			self.cfg.camera_record_jpeg_quality,
		)

		record = {
			"step": action_step,
			"frame_index": frame_index,
			"timestamp": datetime.now().isoformat(timespec="milliseconds"),
			"timestamp_ns": time.time_ns(),
			"cam_head": str(cam_head_rel),
			"cam_wrist": str(cam_wrist_rel),
			"state": np.asarray(observation["state"]).tolist(),
			"output_action": np.asarray(cycle_report["execution"]["output_action"]).tolist(),
			"action_shape": list(cycle_report["execution"]["action_shape"]),
			"stopped_by_user": bool(cycle_report["execution"].get("stopped_by_user", False)),
		}
		self._camera_record_log.write(json.dumps(record, ensure_ascii=False) + "\n")
		self._camera_record_log.flush()

	def _camera_video_fps(self) -> float:
		if self.cfg.camera_record_video_fps > 0:
			return self.cfg.camera_record_video_fps
		if self.cfg.control_interval_s > 0:
			record_every = max(1, int(self.cfg.camera_record_every_n_steps))
			return min(120.0, max(1.0, 1.0 / self.cfg.control_interval_s / record_every))
		return 30.0

	def _finalize_camera_recording(self) -> None:
		# CODEX MODIFICATION: Build MP4 videos from recorded JPEG frames after the control loop.
		if self._camera_record_finalized:
			return
		self._camera_record_finalized = True
		if not self.cfg.record_cameras or not self.cfg.camera_record_video or self._camera_record_dir is None:
			return

		for camera_name in ("cam_head", "cam_wrist"):
			frame_dir = self._camera_record_dir / camera_name
			frame_paths = sorted(frame_dir.glob("*.jpg"))
			if not frame_paths:
				logging.warning("No frames found for %s; skipping video export.", camera_name)
				continue
			video_path = self._camera_record_dir / f"{camera_name}.mp4"
			self._write_video_from_jpegs(frame_paths, video_path, fps=self._camera_video_fps())

	@staticmethod
	def _write_video_from_jpegs(frame_paths: list[Path], video_path: Path, fps: float) -> None:
		# CODEX MODIFICATION: Prefer H.264 MP4 for broad player compatibility.
		if PiperVLAClient._write_h264_video_with_ffmpeg(frame_paths, video_path, fps):
			return
		PiperVLAClient._write_video_from_jpegs_with_opencv(frame_paths, video_path, fps)

	@staticmethod
	def _write_h264_video_with_ffmpeg(frame_paths: list[Path], video_path: Path, fps: float) -> bool:
		# CODEX MODIFICATION: Encode H.264 with ffmpeg/libx264; OpenCV mp4v can be hard to play on some systems.
		ffmpeg = shutil.which("ffmpeg")
		if ffmpeg is None:
			logging.warning("ffmpeg was not found; falling back to OpenCV video export.")
			return False
		frame_dir = frame_paths[0].parent
		frame_pattern = str(frame_dir / "*.jpg")
		fps_arg = f"{float(fps):.6f}".rstrip("0").rstrip(".")
		cmd = [
			ffmpeg,
			"-y",
			"-hide_banner",
			"-loglevel",
			"error",
			"-framerate",
			fps_arg,
			"-pattern_type",
			"glob",
			"-i",
			frame_pattern,
			"-c:v",
			"libx264",
			"-pix_fmt",
			"yuv420p",
			"-preset",
			"veryfast",
			"-crf",
			"18",
			"-movflags",
			"+faststart",
			str(video_path),
		]
		try:
			subprocess.run(cmd, check=True, capture_output=True, text=True)
		except (OSError, subprocess.CalledProcessError) as exc:
			if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
				logging.warning("ffmpeg H.264 export failed for %s: %s", video_path, exc.stderr.strip())
			else:
				logging.warning("ffmpeg H.264 export failed for %s: %s", video_path, exc)
			return False
		logging.info("Saved H.264 camera video to %s (%s frames, %.2f fps)", video_path, len(frame_paths), fps)
		return True

	@staticmethod
	def _write_video_from_jpegs_with_opencv(frame_paths: list[Path], video_path: Path, fps: float) -> None:
		# CODEX MODIFICATION: Fallback to OpenCV if ffmpeg is unavailable.
		try:
			import cv2
		except Exception:
			logging.exception("Failed to import cv2; camera frames were saved without MP4 video.")
			return

		first = cv2.imread(str(frame_paths[0]))
		if first is None:
			logging.warning("Failed to read first frame %s; skipping video export.", frame_paths[0])
			return
		height, width = first.shape[:2]
		writer = cv2.VideoWriter(
			str(video_path),
			cv2.VideoWriter_fourcc(*"mp4v"),
			float(fps),
			(width, height),
		)
		if not writer.isOpened():
			logging.warning("Failed to open video writer for %s.", video_path)
			return
		try:
			for frame_path in frame_paths:
				frame = cv2.imread(str(frame_path))
				if frame is None:
					logging.warning("Skipping unreadable frame %s.", frame_path)
					continue
				if frame.shape[:2] != (height, width):
					frame = cv2.resize(frame, (width, height))
				writer.write(frame)
		finally:
			writer.release()
		logging.info("Saved camera video to %s (%s frames, %.2f fps)", video_path, len(frame_paths), fps)

	def _record_state_action_data(self, step: int, cycle_report: t.Dict[str, t.Any]) -> None:
		# CODEX MODIFICATION: Keep the observation state, raw server action, and executed action together.
		if not self.cfg.record_state_action or self._state_action_record_dir is None:
			return

		observation = cycle_report["observation"]
		server_action = np.asarray(cycle_report["action"]["action"], dtype=np.float32)
		executed_action = np.asarray(cycle_report["execution"]["output_action"], dtype=np.float32)
		record = {
			"step": step,
			"timestamp": datetime.now().isoformat(timespec="milliseconds"),
			"timestamp_ns": time.time_ns(),
			"state": np.asarray(observation["state"], dtype=np.float32).tolist(),
			"server_action": server_action.tolist(),
			"executed_action": executed_action.tolist(),
			"server_action_shape": list(server_action.shape),
			"executed_action_shape": list(executed_action.shape),
			"stopped_by_user": bool(cycle_report["execution"].get("stopped_by_user", False)),
		}
		self._state_action_records.append(record)

		with open(self._state_action_record_dir / "records.jsonl", "a", encoding="utf-8") as f:
			f.write(json.dumps(record, ensure_ascii=False) + "\n")

	@staticmethod
	def _value_labels(prefix: str, dim: int) -> list[str]:
		# CODEX MODIFICATION: Name the last 7-DoF Piper dimension explicitly as gripper.
		if dim == 7:
			return [f"{prefix}_joint_{idx}" for idx in range(6)] + [f"{prefix}_gripper"]
		return [f"{prefix}_{idx}" for idx in range(dim)]

	@staticmethod
	def _write_matrix_csv(path: Path, prefix: str, rows: list[dict[str, t.Any]], values_key: str) -> None:
		# CODEX MODIFICATION: Export dense state/action matrices for quick spreadsheet inspection.
		if not rows:
			return
		max_dim = max(len(row[values_key]) for row in rows)
		value_fields = PiperVLAClient._value_labels(prefix, max_dim)
		fieldnames = ["step", "timestamp_ns"] + value_fields
		with open(path, "w", newline="", encoding="utf-8") as f:
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			for row in rows:
				values = list(row[values_key])
				csv_row = {"step": row["step"], "timestamp_ns": row["timestamp_ns"]}
				for idx, field_name in enumerate(value_fields):
					csv_row[field_name] = values[idx] if idx < len(values) else ""
				writer.writerow(csv_row)

	def _finalize_state_action_recording(self) -> None:
		# CODEX MODIFICATION: Save numeric tables and plots after the control loop finishes.
		if self._state_action_finalized:
			return
		self._state_action_finalized = True
		if not self.cfg.record_state_action or self._state_action_record_dir is None or not self._state_action_records:
			return

		state_rows = []
		server_first_rows = []
		executed_rows = []
		for record in self._state_action_records:
			state_rows.append({
				"step": record["step"],
				"timestamp_ns": record["timestamp_ns"],
				"state": record["state"],
			})
			server_action = np.asarray(record["server_action"], dtype=np.float32)
			if server_action.size:
				server_first_rows.append({
					"step": record["step"],
					"timestamp_ns": record["timestamp_ns"],
					"server_action_first": server_action.reshape(-1, server_action.shape[-1])[0].tolist(),
				})
			executed_action = np.asarray(record["executed_action"], dtype=np.float32)
			if executed_action.size:
				executed_action = executed_action.reshape(-1, executed_action.shape[-1])
				for action_idx, action_value in enumerate(executed_action):
					executed_rows.append({
						"step": record["step"],
						"action_index": action_idx,
						"timestamp_ns": record["timestamp_ns"],
						"executed_action": action_value.tolist(),
					})

		self._write_matrix_csv(self._state_action_record_dir / "state.csv", "state", state_rows, "state")
		self._write_matrix_csv(
			self._state_action_record_dir / "server_action_first.csv",
			"action",
			server_first_rows,
			"server_action_first",
		)
		if executed_rows:
			max_dim = max(len(row["executed_action"]) for row in executed_rows)
			action_fields = self._value_labels("action", max_dim)
			fieldnames = ["sample_index", "step", "action_index", "timestamp_ns"] + action_fields
			with open(self._state_action_record_dir / "executed_action.csv", "w", newline="", encoding="utf-8") as f:
				writer = csv.DictWriter(f, fieldnames=fieldnames)
				writer.writeheader()
				for sample_index, row in enumerate(executed_rows):
					values = list(row["executed_action"])
					csv_row = {
						"sample_index": sample_index,
						"step": row["step"],
						"action_index": row["action_index"],
						"timestamp_ns": row["timestamp_ns"],
					}
					for idx, field_name in enumerate(action_fields):
						csv_row[field_name] = values[idx] if idx < len(values) else ""
					writer.writerow(csv_row)

		self._write_gripper_csv(state_rows, server_first_rows, executed_rows)

		self._save_state_action_npz(state_rows, server_first_rows, executed_rows)
		if self.cfg.state_action_plot:
			self._plot_state_action_data(state_rows, server_first_rows, executed_rows)
		logging.info("Saved state/action records to %s", self._state_action_record_dir)

	def _save_state_action_npz(
		self,
		state_rows: list[dict[str, t.Any]],
		server_first_rows: list[dict[str, t.Any]],
		executed_rows: list[dict[str, t.Any]],
	) -> None:
		# CODEX MODIFICATION: Save NumPy arrays for later numerical analysis.
		if self._state_action_record_dir is None:
			return
		states = np.asarray([row["state"] for row in state_rows], dtype=np.float32)
		server_action_first = np.asarray(
			[row["server_action_first"] for row in server_first_rows],
			dtype=np.float32,
		) if server_first_rows else np.empty((0, 0), dtype=np.float32)
		executed_actions = np.asarray(
			[row["executed_action"] for row in executed_rows],
			dtype=np.float32,
		) if executed_rows else np.empty((0, 0), dtype=np.float32)
		np.savez_compressed(
			self._state_action_record_dir / "state_action_arrays.npz",
			states=states,
			server_action_first=server_action_first,
			executed_actions=executed_actions,
			state_steps=np.asarray([row["step"] for row in state_rows], dtype=np.int32),
			server_action_steps=np.asarray([row["step"] for row in server_first_rows], dtype=np.int32),
			executed_action_steps=np.asarray([row["step"] for row in executed_rows], dtype=np.int32),
			executed_action_indices=np.asarray([row["action_index"] for row in executed_rows], dtype=np.int32),
		)

	def _write_gripper_csv(
		self,
		state_rows: list[dict[str, t.Any]],
		server_first_rows: list[dict[str, t.Any]],
		executed_rows: list[dict[str, t.Any]],
	) -> None:
		# CODEX MODIFICATION: Export gripper values separately so they are easy to inspect.
		if self._state_action_record_dir is None:
			return
		server_by_step = {
			row["step"]: row["server_action_first"][-1]
			for row in server_first_rows
			if row["server_action_first"]
		}
		executed_by_step: dict[int, list[float]] = {}
		for row in executed_rows:
			if row["executed_action"]:
				executed_by_step.setdefault(row["step"], []).append(row["executed_action"][-1])

		with open(self._state_action_record_dir / "gripper.csv", "w", newline="", encoding="utf-8") as f:
			fieldnames = [
				"step",
				"timestamp_ns",
				"state_gripper",
				"server_action_first_gripper",
				"executed_action_first_gripper",
				"executed_action_last_gripper",
			]
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			for row in state_rows:
				executed_values = executed_by_step.get(row["step"], [])
				writer.writerow({
					"step": row["step"],
					"timestamp_ns": row["timestamp_ns"],
					"state_gripper": row["state"][-1] if row["state"] else "",
					"server_action_first_gripper": server_by_step.get(row["step"], ""),
					"executed_action_first_gripper": executed_values[0] if executed_values else "",
					"executed_action_last_gripper": executed_values[-1] if executed_values else "",
				})

	def _plot_state_action_data(
		self,
		state_rows: list[dict[str, t.Any]],
		server_first_rows: list[dict[str, t.Any]],
		executed_rows: list[dict[str, t.Any]],
	) -> None:
		# CODEX MODIFICATION: Generate local curve plots for state and action review.
		if self._state_action_record_dir is None:
			return
		try:
			os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
			import matplotlib
			matplotlib.use("Agg")
			import matplotlib.pyplot as plt
		except Exception:
			logging.exception("Failed to import matplotlib; state/action data was saved without plots.")
			return

		fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
		state_steps = np.asarray([row["step"] for row in state_rows], dtype=np.int32)
		states = np.asarray([row["state"] for row in state_rows], dtype=np.float32)
		state_labels = self._value_labels("state", states.shape[1])
		for dim in range(states.shape[1]):
			axes[0].plot(state_steps, states[:, dim], label=state_labels[dim])
		axes[0].set_title("Observation State")
		axes[0].set_xlabel("VLA step")
		axes[0].set_ylabel("value")
		axes[0].grid(True, alpha=0.3)
		axes[0].legend(loc="best", ncol=4, fontsize=8)

		if server_first_rows:
			server_steps = np.asarray([row["step"] for row in server_first_rows], dtype=np.int32)
			server_first = np.asarray([row["server_action_first"] for row in server_first_rows], dtype=np.float32)
			server_labels = self._value_labels("action", server_first.shape[1])
			for dim in range(server_first.shape[1]):
				axes[1].plot(server_steps, server_first[:, dim], label=server_labels[dim])
			axes[1].set_title("Server Action First Step")
		else:
			axes[1].set_title("Server Action First Step (no data)")
		axes[1].set_xlabel("VLA step")
		axes[1].set_ylabel("value")
		axes[1].grid(True, alpha=0.3)
		axes[1].legend(loc="best", ncol=4, fontsize=8)

		if executed_rows:
			executed_actions = np.asarray([row["executed_action"] for row in executed_rows], dtype=np.float32)
			x = np.arange(executed_actions.shape[0])
			executed_labels = self._value_labels("action", executed_actions.shape[1])
			for dim in range(executed_actions.shape[1]):
				axes[2].plot(x, executed_actions[:, dim], label=executed_labels[dim])
			axes[2].set_title("Executed Actions")
		else:
			axes[2].set_title("Executed Actions (no data)")
		axes[2].set_xlabel("executed action sample")
		axes[2].set_ylabel("value")
		axes[2].grid(True, alpha=0.3)
		axes[2].legend(loc="best", ncol=4, fontsize=8)

		fig.savefig(self._state_action_record_dir / "state_action_curves.png", dpi=150)
		plt.close(fig)
		self._plot_gripper_data(state_rows, server_first_rows, executed_rows, plt)

	def _plot_gripper_data(
		self,
		state_rows: list[dict[str, t.Any]],
		server_first_rows: list[dict[str, t.Any]],
		executed_rows: list[dict[str, t.Any]],
		plt: t.Any,
	) -> None:
		# CODEX MODIFICATION: Save a dedicated gripper plot alongside the all-dimension plot.
		if self._state_action_record_dir is None:
			return
		fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
		state_steps = np.asarray([row["step"] for row in state_rows], dtype=np.int32)
		state_gripper = np.asarray([row["state"][-1] for row in state_rows], dtype=np.float32)
		ax.plot(state_steps, state_gripper, marker="o", label="state_gripper")

		if server_first_rows:
			server_steps = np.asarray([row["step"] for row in server_first_rows], dtype=np.int32)
			server_gripper = np.asarray([row["server_action_first"][-1] for row in server_first_rows], dtype=np.float32)
			ax.plot(server_steps, server_gripper, marker="o", label="server_action_first_gripper")

		if executed_rows:
			executed_gripper = np.asarray([row["executed_action"][-1] for row in executed_rows], dtype=np.float32)
			x = np.arange(executed_gripper.shape[0])
			ax.plot(x, executed_gripper, alpha=0.8, label="executed_action_gripper")

		ax.set_title("Gripper Values")
		ax.set_xlabel("step / executed action sample")
		ax.set_ylabel("gripper value")
		ax.grid(True, alpha=0.3)
		ax.legend(loc="best")
		fig.savefig(self._state_action_record_dir / "gripper_curves.png", dpi=150)
		plt.close(fig)

	def run(self, max_steps: t.Optional[int] = None) -> None:
		"""Run the continuous control loop."""
		if self.cfg.enable_rtc:
			self.run_rtc(max_steps=max_steps)
			return

		step_limit = self.cfg.max_steps if max_steps is None else max_steps
		if step_limit <= 0:
			raise ValueError("max_steps must be > 0")
		if self._stop_requested:
			logging.info("Piper VLA client loop was not started because user stopped before reset().")
			return

		logging.info("Starting Piper VLA client loop. step_limit=%s", step_limit)
		for step in range(step_limit):
			try:
				cycle_report = self.run_once()
				self._record_state_action_data(step, cycle_report)
				# CODEX MODIFICATION: Stop requesting new server actions after user aborts cautious execution.
				if self._stop_requested:
					logging.info("Stopping Piper VLA client loop by user request.")
					break
				if self.show_output_track:
					filepath = Path("/home/charles/workspaces/Double_Piper_Teleop/tem.json")
					if filepath.exists() and step == 0: 
						filepath.unlink()

					data_log = {
						"step": step,
						"output_action": np.asarray(cycle_report["execution"]["output_action"]).tolist(),
						"state": np.asarray(cycle_report["observation"]["state"]).tolist()
					}

					if filepath.exists():
						with open(filepath, 'r', encoding='utf-8') as f:
							data = json.load(f)
						data["log"].append(data_log)
					else:
						data = {"log": [data_log]}
				
					# 写回文件（覆盖写入完整更新后的数据）
					with open(filepath, 'w', encoding='utf-8') as f:
						json.dump(data, f, indent=2)

				logging.debug("loop_step=%s report=%s", step, cycle_report)
			except TimeoutError:
				logging.exception("Server timeout at step=%s", step)
				if self.cfg.stop_on_timeout:
					break
			except Exception:
				logging.exception("Unexpected error at step=%s", step)
				raise

	def close(self) -> None:
		"""Close network resources."""
		if self._rtc_inference_thread is not None and self._rtc_inference_thread.is_alive():
			self._rtc_inference_thread.join(timeout=1.0)
		self._finalize_camera_recording()
		if self._camera_record_log is not None:
			self._camera_record_log.close()
			self._camera_record_log = None
		self._finalize_state_action_recording()
		self.zmq_client.close()

@draccus.wrap()
def main(cfg: InferenceConfig) -> None:
	"""Entrypoint for launching the Piper VLA client from CLI."""
	# CODEX MODIFICATION: Removed stray path expression that prevented main() from running.
	runtime = PiperVLAClient(cfg=cfg)
	try:
		runtime.run()
	finally:
		runtime.close()

if __name__ == "__main__":
	main()
