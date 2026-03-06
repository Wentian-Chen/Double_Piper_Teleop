import zmq
import logging
import typing as t

from vla_infer.src.inference.server import AbstractInferenceServer
from vla_infer.src.models.base import BaseVLAModel

from .protocol import VLAProtocol

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VLAServer(AbstractInferenceServer):
    def __init__(self, model: BaseVLAModel, port: int = 5555):
        self.model = model
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        
        self.cached_instruction = ""
        logging.info(f"VLA Server ready on port {port}. Model: {model.__class__.__name__}")

    def get_observation(self) -> bytes:
        return self.socket.recv()

    def unpack_response(self, response: t.Any) -> t.Dict[str, t.Any]:
        obs = VLAProtocol.unpack_payload(t.cast(bytes, response))

        if obs.get("use_cached_cmd", False):
            obs["cmd"] = self.cached_instruction
        else:
            self.cached_instruction = obs.get("cmd", "")
            logging.info(f"Server updated instruction cache: '{self.cached_instruction}'")

        return obs

    def get_response(self, observation: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        return self.model.predict(observation)

    def execute(self, payload: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        # Server side default has no side-effect execution hook.
        return {"sent_action_keys": list(payload.keys())}

    def pack_and_send(self, payload: t.Dict[str, t.Any]) -> None:
        reply_bytes = VLAProtocol.pack_payload(payload)
        self.socket.send(reply_bytes)

    def run(self):
        try:
            self.run_forever()
        finally:
            self.socket.close()
            self.context.term()