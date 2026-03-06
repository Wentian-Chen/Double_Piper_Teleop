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

	@abstractmethod
	def unpack_response(self, response: t.Any) -> t.Dict[str, t.Any]:
		"""Decode/normalize raw response into executable action payload."""
		raise NotImplementedError

	@abstractmethod
	def execute(self, payload: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
		"""Execute one action payload and return execution report."""
		raise NotImplementedError


class BaseInferenceServer(ABC):
	"""Server-side inference pipeline contract."""

	@abstractmethod
	def get_observation(self) -> t.Any:
		"""Receive one raw request payload from transport layer."""
		raise NotImplementedError

	@abstractmethod
	def unpack_response(self, response: t.Any) -> t.Dict[str, t.Any]:
		"""Decode raw request bytes to structured observation payload."""
		raise NotImplementedError

	@abstractmethod
	def get_response(self, observation: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
		"""Run model inference and return structured action payload."""
		raise NotImplementedError

	@abstractmethod
	def execute(self, payload: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
		"""Optional side-effects hook before sending response."""
		raise NotImplementedError

	@abstractmethod
	def pack_and_send(self, payload: t.Dict[str, t.Any]) -> None:
		"""Pack and send response payload back to transport layer."""
		raise NotImplementedError

