from __future__ import annotations

from dataclasses import dataclass
import logging
import typing as t

import draccus

from vla_infer.src.models import SmolVLAModel
from vla_infer.src.zmq.zmq_server import VLAServer

draccus_wrap = t.cast(t.Callable[..., t.Callable[..., t.Any]], getattr(draccus, "wrap"))


@dataclass
class SmolVLAServerConfig:
	"""Runtime config for launching SmolVLA inference server."""

	model_path: str = ""
	device: str = "cuda"
	dataset_repo_id: t.Optional[str] = None
	dataset_root: t.Optional[str] = None
	action_chunk_size: t.Optional[int] = None
	port: int = 5555
	log_level: str = "INFO"


def create_server(
	cfg: SmolVLAServerConfig,
	model_cls: t.Type[SmolVLAModel] = SmolVLAModel,
) -> VLAServer:
	"""Construct server instance from config.

	The model class is injectable to make integration tests deterministic.
	"""
	model = model_cls(
		model_path=cfg.model_path,
		device=cfg.device,
		dataset_repo_id=cfg.dataset_repo_id,
		dataset_root=cfg.dataset_root,
		action_chunk_size=cfg.action_chunk_size,
	)
	return VLAServer(model=model, port=cfg.port)


@draccus_wrap()
def main(cfg: t.Optional[SmolVLAServerConfig] = None) -> None:
	if cfg is None:
		cfg = SmolVLAServerConfig()

	if not cfg.model_path:
		raise ValueError("model_path must be provided")

	logging.basicConfig(
		level=getattr(logging, cfg.log_level.upper(), logging.INFO),
		format="%(asctime)s - %(levelname)s - %(message)s",
	)
	logging.info("Starting SmolVLA server on port=%s", cfg.port)

	server = create_server(cfg)
	server.run()


if __name__ == "__main__":
	main()
