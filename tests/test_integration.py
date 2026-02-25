'''
Author: boboji11 wendychen112@qq.com
Date: 2026-02-25 17:27:50
LastEditors: boboji11 wendychen112@qq.com
LastEditTime: 2026-02-25 17:35:53
FilePath: \Double_Piper_Teleop\tests\test_integration.py
Description: 

Copyright (c) 2026 by boboji11 , All Rights Reserved. 
'''
import sys
sys.path.append("..")  # 将上级目录添加到 sys.path，确保能导入 vla_infer 模块
import pytest
import time
import numpy as np
import multiprocessing
import zmq
from vla_infer.client import VLAClient
from vla_infer.server import VLAServer
from vla_infer.models.base import BaseVLAModel

# --- 模拟模型 ---
class MockSmolVLA(BaseVLAModel):
    def load_model(self):
        self.chunk_size = 10
        self.action_dim = 6

    def predict(self, observation: dict) -> dict:
        cmd = observation.get("cmd", "")
        # 我们根据指令返回特定标记，方便测试断言
        if "pick a banana" in cmd:
            action = np.ones((self.chunk_size, self.action_dim), dtype=np.float32)
        else:
            action = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
        return {"action": action}

# --- 后台 Server 运行函数 ---
def run_test_server(port):
    model = MockSmolVLA(model_path="mock")
    server = VLAServer(model=model, port=port)
    server.run()

class TestVLANetworkIntegration:
    
    @pytest.fixture(scope="class")
    def vla_client_server_pair(self):
        """
        类级别的 Fixture：启动 Server 进程，建立 Client，测试结束后自动清理。
        """
        test_port = 5566 # 使用非默认端口避免冲突
        
        # 1. 启动 Server 进程
        server_proc = multiprocessing.Process(target=run_test_server, args=(test_port,))
        server_proc.start()
        
        # 等待端口绑定
        time.sleep(0.5)
        
        # 2. 实例化 Client
        client = VLAClient(server_ip="127.0.0.1", port=test_port, timeout_ms=2000)
        
        # 3. 将 client 交给测试用例使用
        yield client 
        
        # 4. Teardown: 测试结束后的清理工作
        client.close()
        server_proc.terminate()
        server_proc.join()

    def test_end_to_end_inference_and_caching(self, vla_client_server_pair):
        """测试端到端的数据传输与 Client 端指令缓存状态机"""
        client = vla_client_server_pair
        
        dummy_obs = {
            "front_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "state": np.zeros(6, dtype=np.float32)
        }
        
        # Step 1: 首次发送指令
        # 预期：Server 收到完整指令，Mock 模型返回全 1 矩阵
        resp1 = client.get_action("pick a banana", dummy_obs.copy())
        assert np.array_equal(resp1["action"], np.ones((10, 6), dtype=np.float32))
        assert client._cached_cmd == "pick a banana"
        
        # Step 2: 发送相同指令 (触发缓存逻辑)
        # 此时底层协议打包的 cmd 应该是 ""
        resp2 = client.get_action("pick a banana", dummy_obs.copy())
        assert np.array_equal(resp2["action"], np.ones((10, 6), dtype=np.float32))
        
        # Step 3: 切换指令
        # 预期：缓存更新，Mock 模型返回全 0 矩阵
        resp3 = client.get_action("reset position", dummy_obs.copy())
        assert np.array_equal(resp3["action"], np.zeros((10, 6), dtype=np.float32))
        assert client._cached_cmd == "reset position"

    def test_client_timeout_handling(self):
        """
        异常测试：测试 Server 不存在或卡死时，Client 是否能按预期抛出异常。
        这是机器人控制中极其重要的安全机制。
        """
        # 故意连接到一个死端口，设置极短超时 (100ms)
        bad_client = VLAClient(server_ip="127.0.0.1", port=9999, timeout_ms=100)
        dummy_obs = {"state": np.zeros(6, dtype=np.float32)}
        
        # 期望抛出 TimeoutError
        with pytest.raises(TimeoutError, match="VLA Server timeout"):
            bad_client.get_action("test", dummy_obs)
            
        bad_client.close()