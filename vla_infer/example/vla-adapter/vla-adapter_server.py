from __future__ import annotations

from dataclasses import dataclass
import logging
import typing as t

import draccus

from vla_infer.src.models import VLAAdapterModel
from vla_infer.src.zmq.zmq_server import VLAServer

draccus_wrap = t.cast(t.Callable[..., t.Callable[..., t.Any]], getattr(draccus, "wrap"))


@dataclass
class VLAAdapterServerConfig:
	"""Runtime config for launching VLA-Adapter inference server."""

	model_path: str = ""
	device: str = "cuda"

	base_model_checkpoint: t.Optional[str] = None
	model_family: str = "openvla"
	use_l1_regression: bool = True
	use_minivlm: bool = True
	use_pro_version: bool = True
	use_proprio: bool = True
	num_images_in_input: int = 2
	unnorm_key: str = "pick_banana_50"
	save_version: str = "vla-adapter"

	port: int = 5555
	log_level: str = "INFO"


def create_server(
	cfg: VLAAdapterServerConfig,
	model_cls: t.Type[VLAAdapterModel] = VLAAdapterModel,
) -> VLAServer:
	"""Construct server instance from config.

	The model class is injectable to make integration tests deterministic.
	"""
	model = model_cls(
		model_path=cfg.model_path,
		device=cfg.device,
		base_model_checkpoint=cfg.base_model_checkpoint,
		model_family=cfg.model_family,
		use_l1_regression=cfg.use_l1_regression,
		use_minivlm=cfg.use_minivlm,
		use_pro_version=cfg.use_pro_version,
		use_proprio=cfg.use_proprio,
		num_images_in_input=cfg.num_images_in_input,
		unnorm_key=cfg.unnorm_key,
		save_version=cfg.save_version,
	)
	return VLAServer(model=model, port=cfg.port)


@draccus_wrap()
def main(cfg: t.Optional[VLAAdapterServerConfig] = None) -> None:
	if cfg is None:
		cfg = VLAAdapterServerConfig()

	if not cfg.model_path:
		raise ValueError("model_path must be provided")

	logging.basicConfig(
		level=getattr(logging, cfg.log_level.upper(), logging.INFO),
		format="%(asctime)s - %(levelname)s - %(message)s",
	)
	logging.info("Starting VLA-Adapter server on port=%s", cfg.port)

	server = create_server(cfg)
	server.run()


if __name__ == "__main__":
	main()
