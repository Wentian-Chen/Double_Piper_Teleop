"""Inference server runtime abstraction."""

from __future__ import annotations

import logging
import typing as t

from .base import BaseInferenceServer
from vla_infer.src.models.base import BaseVLAModel
from vla_infer.src.zmq.zmq_server import BaseZmqServer


class ModelZmqInferenceServer(BaseInferenceServer):
    """Inference runtime that composes model + ZMQ transport."""

    def __init__(self, model: BaseVLAModel, zmq_server: BaseZmqServer) -> None:
        self.model = model
        self.zmq_server = zmq_server

    def predict(self, request: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """Run one model forward from decoded request payload."""
        response = self.model.predict(request)
        if not isinstance(response, dict):
            raise TypeError(f"model.predict() must return dict, got {type(response)}")
        return response

    def run_once(self) -> t.Dict[str, t.Any]:
        """Process one REQ/REP cycle and return debug report."""
        request_dict = self.zmq_server.get_request()
        response_dict = self.predict(request_dict)
        self.zmq_server.response(response_dict)
        return {
            "response": response_dict,
        }
    def start(self) -> None:
        """Start continuous inference loop and handle graceful shutdown."""
        logging.info("Starting inference server loop")
        try:
            while True:
                report = self.run_once()
                logging.debug("server_report=%s", report)
        except KeyboardInterrupt:
            logging.info("Inference server interrupted by user")
        except Exception:
            logging.exception("Inference server loop failed")
            raise
        finally:
            self.close()

    def close(self) -> None:
        """Close owned transport resources."""
        self.zmq_server.close()
