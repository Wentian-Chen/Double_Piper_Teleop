"""Inference server runtime abstraction."""

from __future__ import annotations

import logging
import typing as t

from .base import BaseInferenceServer


class AbstractInferenceServer(BaseInferenceServer):
	"""Reusable server loop based on five abstract lifecycle methods."""

	def run_once(self) -> t.Dict[str, t.Any]:
		raw_request = self.get_observation()
		observation = self.unpack_response(raw_request)
		response_payload = self.get_response(observation)
		execute_report = self.execute(response_payload)
		self.pack_and_send(response_payload)
		return {
			"observation": observation,
			"response_payload": response_payload,
			"execution": execute_report,
		}

	def run_forever(self) -> None:
		logging.info("Starting inference server loop")
		while True:
			try:
				report = self.run_once()
				logging.debug("server_report=%s", report)
			except KeyboardInterrupt:
				logging.info("Inference server interrupted by user")
				break
			except Exception:
				logging.exception("Inference server loop failed")
				raise

