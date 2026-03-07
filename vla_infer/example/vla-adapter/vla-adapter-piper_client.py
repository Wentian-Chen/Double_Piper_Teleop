from __future__ import annotations

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
    uint8_image_to_float32_01,
    smooth_action_chunk,
    delta_action_chunk_to_absolute
)
ArrayTransform = t.Callable[[np.ndarray], np.ndarray]


@dataclass
class InferenceConfig:
	"""Single runtime config for Piper inference client.
	"""

	server_ip: str = "127.0.0.1"
	port: int = 5555
	timeout_ms: int = 2000
	jpeg_quality: int = 80

	auto_setup: bool = True
	task_instruction: str = "Pick up the banana and place it in the container"
	max_steps: int = 1000
	stop_on_timeout: bool = True

	action_key: str = "action"
	execute_chunk_steps: int = 8
	control_interval_s: float = 0.04
	log_level: str = "INFO"


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
		client: t.Optional[Vllog_levelaZmqClient] = None,
	) -> None:
		self.cfg = cfg
		logging.basicConfig(
			level=getattr(logging, cfg.log_level.upper(), logging.INFO),
			format="%(asctime)s - %(levelname)s - %(message)s",
		)

		self.robot = robot if robot is not None else PiperSingleRobot(auto_setup=cfg.auto_setup)
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
		obs = {
			"state": raw_obs.get("state", np.zeros(7, dtype=np.float32)),
			"image": raw_obs.get("cam_head"),
			"wrist_image": raw_obs.get("cam_wrist"),
		}
		# adaptive resize image
		obs["image"] = adaptive_resize_image(np.asarray(obs["image"]))
		obs["wrist_image"] = adaptive_resize_image(np.asarray(obs["wrist_image"]))
		# convert image to float32 [0, 1]
		obs["image"] = uint8_image_to_float32_01(obs["image"])
		obs["wrist_image"] = uint8_image_to_float32_01(obs["wrist_image"])
		self.obs = obs # save for later use in get_response
		return obs

	def get_response(
		self,
		observation: t.Dict[str, t.Any],
		task_instruction: t.Optional[str] = None,
	) -> t.Any:
		"""send observation and get server response."""
		for key, value in observation.items():
			if value is not None and hasattr(value, "shape") and hasattr(value, "dtype"):
				logging.debug(f"Observation '{key}' shape={value.shape} dtype={value.dtype}")
		observation["cmd"] = task_instruction or self.cfg.task_instruction
		logging.debug(f"Observation 'cmd'='{observation['cmd']}'")
		action = self.zmq_client.get_response(obs_dict=observation)["action"]
		abs_action = delta_action_chunk_to_absolute(self.obs["state"],action)
		smooth_action = smooth_action_chunk(abs_action,max_angular_acceleration=0.01,max_angular_jerk=0.01)
		return {"action": smooth_action}


	def execute(self, response: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
		"""execute action chunk on robot."""
		action = np.asarray(response["action"], dtype=np.float32)
		
		if action.ndim == 1:
			action_2d = action[None, :]
		elif action.ndim == 2:
			action_2d = action
		else:
			raise ValueError(f"action must be 1D or 2D, got shape={action.shape}")

		execute_steps = min(max(1, self.cfg.execute_chunk_steps), action_2d.shape[0])

		for idx in range(execute_steps):
			self.robot.apply_action({"action": action_2d[idx]})
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
			"observation": observation,
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


PiperClientConfig = InferenceConfig


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