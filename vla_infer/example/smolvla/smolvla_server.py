from dataclasses import dataclass
import logging
from pprint import pformat
import typing as t

import draccus

from vla_infer.src.inference.server import ModelZmqInferenceServer
from vla_infer.src.models import SmolVLAModel
from vla_infer.src.zmq.zmq_server import VlaZmqServer

draccus_wrap = t.cast(t.Callable[..., t.Callable[..., t.Any]], getattr(draccus, "wrap"))


@dataclass
class SmolVLAServerConfig:
    """Runtime config for launching SmolVLA inference server."""

    # Preferred name, aligned with lerobot-record semantics.
    policy_path: str = ""
    # LeRobot pretrained policy path. Example: HF_USER/FINETUNE_MODEL_NAME
    # Backward-compatible alias of policy_path.
    model_path: str = ""
    # Device used by the policy and pre/post processors.
    device: str = "cuda"
    # Draccus+argparse on Python 3.14 cannot parse Optional[T] reliably here.
    # Use empty-string / negative sentinels and normalize before model init.
    dataset_repo_id: str = ""
    dataset_root: str = ""
    # Override action chunk size (maps to policy.config.n_action_steps).
    # <= 0 means "use model default".
    action_chunk_size: int = -1
    # Fallback instruction when request payload has no cmd.
    default_instruction: str = ""
    ip: str = "0.0.0.0"
    port: int = 5555
    jpeg_quality: int = 80
    log_level: str = "INFO"

@draccus_wrap()
def main(cfg: SmolVLAServerConfig) -> None:

    resolved_model_path = cfg.policy_path or cfg.model_path
    if not resolved_model_path:
        raise ValueError("policy_path must be provided (model_path is accepted as alias)")

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("SmolVLA server config:\n%s", pformat(cfg))
    logging.info("Starting SmolVLA server on tcp://%s:%s", cfg.ip, cfg.port)

    dataset_repo_id = cfg.dataset_repo_id.strip() or None
    dataset_root = cfg.dataset_root.strip() or None
    action_chunk_size = cfg.action_chunk_size if cfg.action_chunk_size > 0 else None
    
    zmq_server = VlaZmqServer(ip=cfg.ip, port=cfg.port, jpeg_quality=cfg.jpeg_quality)

    model = SmolVLAModel(
        model_path=resolved_model_path,
        device=cfg.device,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        action_chunk_size=action_chunk_size,
        default_instruction=cfg.default_instruction,
    )
    server = ModelZmqInferenceServer(model=model, zmq_server=zmq_server)
    server.start()


if __name__ == "__main__":
    main()
