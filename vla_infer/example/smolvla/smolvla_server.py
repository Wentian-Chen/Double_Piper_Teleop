from __future__ import annotations

from dataclasses import dataclass
import logging
import typing as t

import draccus

from vla_infer.src.inference.server import ModelZmqInferenceServer
from vla_infer.src.models import SmolVLAModel
from vla_infer.src.zmq.zmq_server import VlaZmqServer

draccus_wrap = t.cast(t.Callable[..., t.Callable[..., t.Any]], getattr(draccus, "wrap"))


@dataclass
class SmolVLAServerConfig:
    """Runtime config for launching SmolVLA inference server."""

    model_path: str = ""
    device: str = "cuda"
    dataset_repo_id: t.Optional[str] = None
    dataset_root: t.Optional[str] = None
    action_chunk_size: t.Optional[int] = None
    ip: str = "127.0.0.1"
    port: int = 5555
    jpeg_quality: int = 80
    log_level: str = "INFO"

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
    logging.info("Starting SmolVLA server on tcp://%s:%s", cfg.ip, cfg.port)

    model = SmolVLAModel(
        model_path=cfg.model_path,
        device=cfg.device,
        dataset_repo_id=cfg.dataset_repo_id,
        dataset_root=cfg.dataset_root,
        action_chunk_size=cfg.action_chunk_size,
    )
    zmq_server = VlaZmqServer(ip=cfg.ip, port=cfg.port, jpeg_quality=cfg.jpeg_quality)
    server = ModelZmqInferenceServer(model=model, zmq_server=zmq_server)
    server.start()


if __name__ == "__main__":
    main()
