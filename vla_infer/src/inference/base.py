"""Inference abstraction interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
import typing as t


class BaseInferenceClient(ABC):
	"""Client-side inference pipeline contract."""

	@abstractmethod
	def get_observation(self) -> t.Dict[str, t.Any]:
		"""Collect one observation payload from robot/sensors."""
		raise NotImplementedError

	@abstractmethod
	def get_response(
		self,
		observation: t.Dict[str, t.Any],
		task_instruction: t.Optional[str] = None,
	) -> t.Any:
		"""Send observation and receive raw response object from inference server."""
		raise NotImplementedError


class BaseInferenceServer(ABC):
	"""Server-side inference pipeline contract."""
	@abstractmethod
	def start(self) -> None:
		"""Start server loop, receive requests, run inference, and send responses."""
		raise NotImplementedError
	@abstractmethod
	def predict(self, request: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
		"""Predict one response payload for a decoded request."""
		raise NotImplementedError

