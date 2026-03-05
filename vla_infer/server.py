import zmq
import logging
from vla_infer.protocol import VLAProtocol
# 假设 base.py 中有 BaseVLAModel
from vla_infer.models.base import BaseVLAModel 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VLAServer:
    def __init__(self, model: BaseVLAModel, port: int = 5555):
        self.model = model
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        
        self.cached_instruction = ""
        logging.info(f"VLA Server ready on port {port}. Model: {model.__class__.__name__}")

    def run(self):
        try:
            while True:
                # 1. 接收请求字节流
                request_bytes = self.socket.recv()
                
                # 2. 动态解包
                obs = VLAProtocol.unpack_payload(request_bytes)
                
                # 3. 校验并还原语言指令
                if obs.get("use_cached_cmd", False):
                    obs["cmd"] = self.cached_instruction
                else:
                    self.cached_instruction = obs.get("cmd", "")
                    logging.info(f"Server updated instruction cache: '{self.cached_instruction}'")
                
                # 4. 执行模型推理 (传入完整字典)
                action_result = self.model.predict(obs)
                
                # 5. 打包并返回动作字典
                reply_bytes = VLAProtocol.pack_payload(action_result)
                self.socket.send(reply_bytes)
                
        except KeyboardInterrupt:
            logging.info("Shutting down server...")
        finally:
            self.socket.close()
            self.context.term()