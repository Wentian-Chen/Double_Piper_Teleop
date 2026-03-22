from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import time
import typing as t

import draccus
import numpy as np

from vla_infer.src.inference.client import InferenceClient
from vla_infer.src.robots.piper_single import PiperSingleRobot
from vla_infer.src.zmq.zmq_client import VlaZmqClient
from vla_infer.src.process.utils import (
    adaptive_resize_image,
	ensure_hwc3_uint8_image,
    uint8_image_to_float32_01,
    smooth_action_chunk,
    delta_action_chunk_to_absolute,
	check_uint8_rgb
)
import sys
from pathlib import Path
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
sys.path.append(str(REPO_ROOT))
ArrayTransform = t.Callable[[np.ndarray], np.ndarray]


def _to_jsonable_array(value: t.Any) -> t.Any:
	"""Convert numpy arrays/scalars to JSON-serializable values."""
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	return value


def init_inference_log_file(
	log_dir: str,
	task_instruction: str,
	control_mode: str,
) -> Path:
	"""Create one UTC timestamped log file for the whole run."""
	log_path = Path(log_dir)
	log_path.mkdir(parents=True, exist_ok=True)

	utc_now = datetime.now(timezone.utc)
	filename = utc_now.strftime("%Y%m%dT%H%M%S_%fZ.json")
	file_path = log_path / filename

	payload = {
		"run_start_utc": utc_now.isoformat(),
		"task_instruction": task_instruction,
		"control_mode": control_mode,
		"steps": [],
	}

	with file_path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=False, indent=2)

	return file_path


def append_inference_step_log(
	log_file_path: Path,
	step_index: int,
	log_dir: str,
	state: np.ndarray,
	response_action: np.ndarray,
	abs_action: np.ndarray,
	smooth_action: np.ndarray,
) -> None:
	"""Append one inference step record into the run-level JSON file."""
	_ = log_dir  # keep signature explicit for clarity and future extension
	with log_file_path.open("r", encoding="utf-8") as f:
		payload = json.load(f)

	step_record = {
		"step_index": step_index,
		"utc_timestamp": datetime.now(timezone.utc).isoformat(),
		"state": _to_jsonable_array(np.asarray(state)),
		"response_action": _to_jsonable_array(np.asarray(response_action)),
		"abs_action": _to_jsonable_array(np.asarray(abs_action)),
		"smooth_action": _to_jsonable_array(np.asarray(smooth_action)),
	}
	payload.setdefault("steps", []).append(step_record)

	with log_file_path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=False, indent=2)


@dataclass
class InferenceConfig:
	"""
	Single runtime config for Piper inference client.
	"""

	server_ip: str = "127.0.0.1"
	port: int = 5555
	timeout_ms: int = 2000
	jpeg_quality: int = 80

	
	max_steps: int = 1000
	stop_on_timeout: bool = True
	execute_chunk_steps: int = 8
	control_interval_s: float = 0.04
	log_level: str = "INFO"

	enable_inference_log: bool = False
	log_dir: str = "vla_infer/example/vla-adapter/logs"

	state_type: str = "qpos"
	action_type: str = "joint"
	task_instruction: str = "Pick up the banana and place it in the bowl"

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
		logging.basicConfig(
			level=getattr(logging, cfg.log_level.upper(), logging.INFO),
			format="%(asctime)s - %(levelname)s - %(message)s",
		)

		self.robot = robot if robot is not None else PiperSingleRobot()
		time.sleep(2)
		self.robot.reset()  # Ensure robot is ready before connecting to server
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
		self.current_step_index: int = -1
		self.inference_log_file_path: t.Optional[Path] = None
  

	def get_observation(self) -> t.Dict[str, t.Any]:
		"""Abstract step 1: collect and preprocess one observation payload."""
		raw_obs = self.robot.get_observation()

		if self.cfg.state_type == "qpos":
			qpos_value = raw_obs.get("qpos",np.zeros(6,dtype=np.float32))
			gripper_value = raw_obs.get("gripper",np.zeros(1,dtype=np.float32))
			state = np.concatenate([qpos_value,gripper_value],axis=0)
		elif self.cfg.state_type == "joint":
			# raw_obs["state"] = [joint(6), gripper(1)]
			state = raw_obs.get("state", np.zeros(7, dtype=np.float32))
		else:
			raise ValueError(f"Unsupported state_type: {self.cfg.state_type}")

		obs = {
			"state": state,
			"image": raw_obs.get("cam_head"),
			"wrist_image": raw_obs.get("cam_wrist")
		}

		# Ensure images are HWC3 uint8 before resize to satisfy model input contract.
		obs["image"] = check_uint8_rgb(adaptive_resize_image(obs["image"]))
		obs["wrist_image"] = check_uint8_rgb(adaptive_resize_image(obs["wrist_image"]))
		
		self.obs = obs # save for later use in execute
		if self.cfg.action_type == "joint" and self.cfg.state_type == "qpos":
			self.obs["joint_state"] = state # save joint state for later use in absolute conversion
		
		return obs

	def get_response(
		self,
		observation: t.Dict[str, t.Any],
		task_instruction: t.Optional[str] = None,
	) -> t.Any:
		"""send observation and get server response."""
		# Log observation details for debugging.
		for key, value in observation.items():
			if value is not None and hasattr(value, "shape") and hasattr(value, "dtype"):
				logging.debug(f"Observation '{key}' shape={value.shape} dtype={value.dtype}")
    	# set cmd to task_instruction if provided, otherwise use default from config
		observation["cmd"] = task_instruction or self.cfg.task_instruction
		logging.debug(f"Observation 'cmd'='{observation['cmd']}'")

		# send observation to server and get response
		# response = {"action": np.ndarray(T, D), ...}
		action = self.zmq_client.get_response(obs_dict=observation)["action"]
		
		return {"action": action}

	def execute(self, response: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
		"""execute action chunk on robot."""
		# post-process action if needed (e.g. convert delta to absolute, apply smoothing, etc.)
		action = np.asarray(response["action"], dtype=np.float32)
		if self.cfg.state_type == "qpos":
			abs_action = delta_action_chunk_to_absolute(self.obs.get("joint_state", np.zeros(7, dtype=np.float32)), action)
		else:
			abs_action = delta_action_chunk_to_absolute(self.obs.get("state", np.zeros(7, dtype=np.float32)), action)

		smooth_action = smooth_action_chunk(abs_action,max_angular_acceleration=0.01,max_angular_jerk=0.01)

		if self.cfg.enable_inference_log:
			if self.inference_log_file_path is None:
				raise RuntimeError("inference_log_file_path is not initialized")
			append_inference_step_log(
				log_file_path=self.inference_log_file_path,
				step_index=self.current_step_index,
				log_dir=self.cfg.log_dir,
				state=np.asarray(self.obs["state"], dtype=np.float32),
				response_action=action,
				abs_action=abs_action,
				smooth_action=np.asarray(smooth_action, dtype=np.float32),
			)
			logging.debug("Appended inference step log to: %s", self.inference_log_file_path)

		# ensure action is 2D (T, D)
		if smooth_action.ndim == 1:
			action_2d = smooth_action[None, :]
		elif smooth_action.ndim == 2:
			action_2d = smooth_action
		else:
			raise ValueError(f"action must be 1D or 2D, got shape={smooth_action.shape}")

		execute_steps = min(max(1, self.cfg.execute_chunk_steps), action_2d.shape[0])

		for idx in range(execute_steps):
			#  self.robot.get_state()["state"] 
			self.robot.apply_action({"action":action_2d[idx]})
			if self.cfg.control_interval_s > 0:
				time.sleep(self.cfg.control_interval_s)

		return {
			"executed_steps": execute_steps,
			"action_shape": tuple(action_2d.shape),
		}
	
	def run_once(self) -> t.Dict[str, t.Any]:
		"""Run one full observe-send-receive-execute cycle."""
		observation = self.get_observation()
		response = self.get_response(observation)
		execution_result = self.execute(response)

		return {
			"action":  response,
			"execution": execution_result,
		}

	def run(self, max_steps: t.Optional[int] = None) -> None:
		"""Run the continuous control loop."""
		step_limit = self.cfg.max_steps if max_steps is None else max_steps
		if step_limit <= 0:
			raise ValueError("max_steps must be > 0")

		logging.info("Starting Piper VLA client loop. step_limit=%s", step_limit)
		if self.cfg.enable_inference_log:
			self.inference_log_file_path = init_inference_log_file(
				log_dir=self.cfg.log_dir,
				task_instruction=self.cfg.task_instruction,
				control_mode="",
			)
			logging.info("Inference log file: %s", self.inference_log_file_path)

		for step in range(step_limit):
			try:
				self.current_step_index = step
				cycle_report = self.run_once()
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
		self.zmq_client.close()


@draccus.wrap()
def main(cfg: InferenceConfig) -> None:
	"""Entrypoint for launching the Piper VLA client from CLI."""
	runtime = PiperVLAClient(cfg=cfg)
	try:
		runtime.run()
	finally:
		runtime.close()


if __name__ == "__main__":
	main()