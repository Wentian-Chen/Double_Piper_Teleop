from dataclasses import dataclass
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
	execute_chunk_steps: int = 8
	control_interval_s: float = 0.04
	log_level: str = "INFO"

	state_type: str = "qpos"
	action_type: str = "joint"


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
  

	def get_observation(self) -> t.Dict[str, t.Any]:
		"""Abstract step 1: collect and preprocess one observation payload."""
		raw_obs = self.robot.get_observation()
		# set state as qpos
		if self.cfg.state_type == "qpos":
			qpos_value = raw_obs.get("qpos",np.zeros(6,dtype=np.float32))
			gripper_value = np.asarray([raw_obs.get("gripper", 0.0)], dtype=np.float32)
			state = np.concatenate([qpos_value, gripper_value], axis=0)
		elif self.cfg.state_type == "joint":
			state = raw_obs.get("joint", np.zeros(7, dtype=np.float32))

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
		for step in range(step_limit):
			try:
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