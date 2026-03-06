import zmq
import logging
import typing as t
from abc import ABC, abstractmethod
from .protocol import VLAProtocol

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaseZmqServer(ABC):

    @abstractmethod
    def get_request(self) -> t.Dict[str, t.Any]:
        pass

    @abstractmethod
    def response(self, obs_dict: t.Dict[str, t.Any]) -> bytes:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class VlaZmqServer(BaseZmqServer):
    def __init__(self, ip: str = "127.0.0.1", port: int = 5555, jpeg_quality: int = 80):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://{ip}:{port}")

        self.jpeg_quality = jpeg_quality
        self.cached_instruction = ""
        self._is_closed = False
        logging.info(f"ZMQ Server ready on tcp://{ip}:{port}")

    def get_request(self) -> t.Dict[str, t.Any]:
        """Receive one request and return decoded payload."""
        request_bytes = self.socket.recv()
        return VLAProtocol.unpack_payload(request_bytes)
    
    def response(self, obs_dict: t.Dict[str, t.Any]) -> bytes:
        """Pack response payload and send it back to client."""
        response_bytes = VLAProtocol.pack_payload(obs_dict, jpeg_quality=self.jpeg_quality)
        self.socket.send(response_bytes)
        return response_bytes

    def close(self) -> None:
        if self._is_closed:
            return

        self._is_closed = True
        self.socket.close()
        self.context.term()
        logging.info("ZMQ server closed.")
