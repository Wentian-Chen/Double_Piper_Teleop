"""Launch a ZMQ-based VLA-Adapter inference server."""

from dataclasses import dataclass, field
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
    # VLA-Adapter checkpoint directory.
    model_path: str = ""
    # Model family (must be openvla).
    model_family: str = "openvla"
    # Use L1 action head.
    use_l1_regression: bool = True
    # Use MiniVLM prompt template.
    use_minivlm: bool = True
    # Enable Pro action head variant.
    use_pro_version: bool = True
    # Enable proprio conditioning.
    use_proprio: bool = True
    # Enable FiLM pathway in model forward.
    use_film: bool = False
    # Enable 8-bit quantization loading.
    load_in_8bit: bool = False
    # Enable 4-bit quantization loading.
    load_in_4bit: bool = False
    # Number of camera images consumed by policy.
    num_images_in_input: int = 2
    # Action chunk length returned each call.
    num_open_loop_steps: int = 8
    # Version tag forwarded to model.
    save_version: str = "vla-adapter"
    # Task suite name.
    task_suite_name: str = "piper_pick_banana_100_resize_224_converted"
    ip: str = "127.0.0.1"
    port: int = 5555
    jpeg_quality: int = 80
    log_level: str = "INFO"


@draccus_wrap()
def main(cfg: VLAAdapterServerConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Starting VLA-Adapter server on tcp://%s:%s", cfg.ip, cfg.port)

    zmq_server = VlaZmqServer(ip=cfg.ip, port=cfg.port, jpeg_quality=cfg.jpeg_quality)
    model = VLAAdapterModel(
        pretrained_checkpoint=cfg.model_path,
        model_family=cfg.model_family,
        use_l1_regression=cfg.use_l1_regression,
        use_minivlm=cfg.use_minivlm,
        use_pro_version=cfg.use_pro_version,
        use_proprio=cfg.use_proprio,
        use_film=cfg.use_film,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        num_images_in_input=cfg.num_images_in_input,
        num_open_loop_steps=cfg.num_open_loop_steps,
        save_version=cfg.save_version,
        task_suite_name=cfg.task_suite_name,
    )

    server = ModelZmqInferenceServer(model=model, zmq_server=zmq_server)
    server.start()
if __name__ == "__main__":
    main()
