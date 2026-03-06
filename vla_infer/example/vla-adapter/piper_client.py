from __future__ import annotations

from dataclasses import dataclass
import logging
import time
import typing as t

import draccus
import numpy as np

from vla_infer.src.inference.client import AbstractInferenceClient
from vla_infer.src.process.action_preprocessor import ActionPreprocessor
from vla_infer.src.process.image_preprocessor import ImagePreprocessor
from vla_infer.src.robots.piper_single import PiperSingleRobot
from vla_infer.src.zmq.zmq_client import VLAClient

# Reference
@dataclass(frozen=True)
class ConnectionConfig:
	"""Read-only grouped connection config view."""

	server_ip: str
	port: int
	timeout_ms: int
	jpeg_quality: int


@dataclass(frozen=True)
class RuntimeConfig:
	"""Read-only grouped runtime config view."""

	task_instruction: str
	max_steps: int
	stop_on_timeout: bool


@dataclass(frozen=True)
class ActionConfig:
	"""Read-only grouped action config view."""

	action_key: str
	execute_chunk_steps: int
	control_interval_s: float
	include_prev_action: bool
	prev_action_key: str


@dataclass
class InferenceConfig:
	"""Single runtime config for Piper inference client.

	Use property views (`connection`, `runtime`, `action`) when grouped access is needed.
	"""

	server_ip: str = "127.0.0.1"
	port: int = 5555
	timeout_ms: int = 2000
	jpeg_quality: int = 80

	auto_setup: bool = True
	task_instruction: str = "pick up the banana"
	max_steps: int = 1000
	stop_on_timeout: bool = True

	action_key: str = "action"
	execute_chunk_steps: int = 1
	control_interval_s: float = 0.04
	include_prev_action: bool = True
	prev_action_key: str = "prev_action"

	log_level: str = "INFO"

	@property
	def connection(self) -> ConnectionConfig:
		return ConnectionConfig(
			server_ip=self.server_ip,
			port=self.port,
			timeout_ms=self.timeout_ms,
			jpeg_quality=self.jpeg_quality,
		)

	@property
	def runtime(self) -> RuntimeConfig:
		return RuntimeConfig(
			task_instruction=self.task_instruction,
			max_steps=self.max_steps,
			stop_on_timeout=self.stop_on_timeout,
		)

	@property
	def action(self) -> ActionConfig:
		return ActionConfig(
			action_key=self.action_key,
			execute_chunk_steps=self.execute_chunk_steps,
			control_interval_s=self.control_interval_s,
			include_prev_action=self.include_prev_action,
			prev_action_key=self.prev_action_key,
		)


class PiperVLAClient(AbstractInferenceClient):
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
		client: t.Optional[VLAClient] = None,
		image_preprocessor: t.Optional[ImagePreprocessor] = None,
		observation_action_preprocessor: t.Optional[ActionPreprocessor] = None,
		action_postprocessor: t.Optional[ActionPreprocessor] = None,
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
			else VLAClient(
				server_ip=cfg.server_ip,
				port=cfg.port,
				timeout_ms=cfg.timeout_ms,
			)
		)

		self.image_preprocessor = image_preprocessor
		self.observation_action_preprocessor = observation_action_preprocessor
		self.action_postprocessor = action_postprocessor

		self._last_action_chunk: t.Optional[np.ndarray] = None
		self._last_observation_report: t.Dict[str, t.Any] = {}
		self._last_action_report: t.Dict[str, t.Any] = {}

	def get_observation(self) -> t.Dict[str, t.Any]:
		"""Abstract step 1: collect and preprocess one observation payload."""
		raw_obs = self.robot.get_observation()
		obs = {
			"state": raw_obs.get("state", np.zeros(7, dtype=np.float32)),
			"image": raw_obs.get("cam_head"),
			"wrist_image": raw_obs.get("cam_wrist"),
		}

		if self.image_preprocessor is not None:
			image_result = self.image_preprocessor.preprocess(
				{
					"image": obs["image"],
					"wrist_image": obs["wrist_image"],
				}
			)
			obs["image"] = image_result["image"]
			obs["wrist_image"] = image_result["wrist_image"]

		if self.observation_action_preprocessor is not None:
			action_result = self.observation_action_preprocessor.preprocess(
				{
					"state": obs["state"],
				}
			)
			obs["state"] = action_result["state"]

		return obs

	def get_response(
		self,
		observation: t.Dict[str, t.Any],
		task_instruction: t.Optional[str] = None,
	) -> t.Any:
		"""Abstract step 2: send observation and get server response."""
		for key, value in observation.items():
			if isinstance(value, np.ndarray):
				logging.debug(
					"get_response observation key=%s shape=%s dtype=%s",
					key,
					value.shape,
					value.dtype,
				)
			else:
				logging.debug("get_response observation key=%s type=%s", key, type(value))

		return self.zmq_client.get_action(
			cmd_text=task_instruction or self.cfg.task_instruction,
			obs_dict=observation,
			jpeg_quality=self.cfg.jpeg_quality,
		)

	def unpack_response(self, response: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
		"""Abstract step 3: optional action postprocess + payload normalization."""
		if "action" not in response:
			raise KeyError("Server response missing required key 'action'")

		if self.action_postprocessor is not None:
			processed = self.action_postprocessor.preprocess({"action": response.get("action")})
			response["action"] = processed["action"]

		self._last_action_chunk = np.asarray(response["action"], dtype=np.float32)
		return response

	def execute(self, response: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
		"""Abstract step 4: execute action chunk on robot."""
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
		action = self.unpack_response(response)
		execution_result = self.execute(action)

		return {
			"observation": {
				"payload": observation,
				"report": self._last_observation_report,
			},
			"action": {
				"payload": action,
				"report": self._last_action_report,
			},
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