import zmq
import logging
import typing
from .protocol import VLAProtocol

class VLAClient:
    def __init__(self, server_ip: str="127.0.0.1", port: int = 5555, timeout_ms: int = 2000):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        # 强制设置 LINGER 为 0，确保 close() 时立刻销毁 Socket，不等待队列中的残留消息
        self.socket.setsockopt(zmq.LINGER, 0)
        # 强制设置接收超时，防止物理机遇险
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        
        self.socket.connect(f"tcp://{server_ip}:{port}")
        
        self._cached_cmd = None
        logging.info(f"VLA Client connected to tcp://{server_ip}:{port}")

    def get_action(self, cmd_text: str, obs_dict: typing.Dict[str, typing.Any], jpeg_quality: int = 80) -> typing.Dict[str, typing.Any]:
        """
        向 Server 请求动作。
        :param cmd_text: 当前任务指令
        :param obs_dict: 观测字典，例如 {"front_image": img_arr, "piper_state": state_arr}
        :param jpeg_quality: 动态调整图像压缩率
        """
        # 1. 自动维护语言指令的缓存状态
        if cmd_text == self._cached_cmd:
            obs_dict["use_cached_cmd"] = True
            obs_dict["cmd"] = "" 
        else:
            obs_dict["use_cached_cmd"] = False
            obs_dict["cmd"] = cmd_text
            self._cached_cmd = cmd_text
            logging.info(f"[Task Switch] New instruction mapped: '{cmd_text}'")

        # 2. 调用动态协议层打包
        payload_bytes = VLAProtocol.pack_payload(obs_dict, jpeg_quality=jpeg_quality)
        
        # 3. 发送并等待响应
        self.socket.send(payload_bytes)
        try:
            reply_bytes = self.socket.recv()
            return VLAProtocol.unpack_payload(reply_bytes)
        except zmq.error.Again:
            # 触发急停机制
            logging.error("ZMQ Timeout! Emergency stop required.")
            raise TimeoutError("VLA Server timeout.")

    def close(self):
        self.socket.close()
        self.context.term()