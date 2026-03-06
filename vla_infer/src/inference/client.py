"""Inference client runtime abstraction."""

from __future__ import annotations

import logging
import typing as t

from .base import BaseInferenceClient


class AbstractInferenceClient(BaseInferenceClient):
	"""Reusable client loop based on five abstract lifecycle methods."""

	def run_once(self) -> t.Dict[str, t.Any]:
		observation = self.get_observation()
		response = self.get_response(observation)
		payload = self.unpack_response(response)
		execution_report = self.execute(payload)
		return {
			"observation": observation,
			"response": response,
			"payload": payload,
			"execution": execution_report,
		}

	def run(self, max_steps: int) -> None:
		if max_steps <= 0:
			raise ValueError("max_steps must be > 0")

		logging.info("Starting inference client loop. step_limit=%s", max_steps)
		for step in range(max_steps):
			try:
				report = self.run_once()
				logging.debug("client_step=%s report=%s", step, report)
			except Exception:
				logging.exception("Inference client loop failed at step=%s", step)
				raise

