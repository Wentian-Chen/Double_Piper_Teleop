from __future__ import annotations

from dataclasses import dataclass
import logging
import typing as t

import draccus

from vla_infer.src.inference.server import ModelZmqInferenceServer
from vla_infer.src.models import VLAAdapterModel
from vla_infer.src.zmq.zmq_server import VlaZmqServer

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
    num_open_loop_steps: int = 1
    unnorm_key: str = "pick_banana_50"
    save_version: str = "vla-adapter"

    ip: str = "127.0.0.1"
    port: int = 5555
    jpeg_quality: int = 80
    log_level: str = "INFO"

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
    logging.info("Starting VLA-Adapter server on tcp://%s:%s", cfg.ip, cfg.port)

    model = VLAAdapterModel(
        model_path=cfg.model_path,
        device=cfg.device,
        base_model_checkpoint=cfg.base_model_checkpoint,
        model_family=cfg.model_family,
        use_l1_regression=cfg.use_l1_regression,
        use_minivlm=cfg.use_minivlm,
        use_pro_version=cfg.use_pro_version,
        use_proprio=cfg.use_proprio,
        num_images_in_input=cfg.num_images_in_input,
        num_open_loop_steps=cfg.num_open_loop_steps,
        unnorm_key=cfg.unnorm_key,
        save_version=cfg.save_version,
    )
    zmq_server = VlaZmqServer(ip=cfg.ip, port=cfg.port, jpeg_quality=cfg.jpeg_quality)
    server = ModelZmqInferenceServer(model=model, zmq_server=zmq_server)
    server.start()


if __name__ == "__main__":
    main()
