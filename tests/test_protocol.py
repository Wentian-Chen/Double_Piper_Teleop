import pytest
import numpy as np
from vla_infer.protocol import VLAProtocol

class TestVLAProtocol:
    
    @pytest.fixture
    def dummy_observation(self):
        """提供基础的测试夹具 (Fixture)"""
        return {
            "front_image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            # 模拟 6 自由度状态，故意使用 float64 以测试降维逻辑
            "state": np.random.randn(6).astype(np.float64), 
            "cmd": "pick a banana",
            "use_cached_cmd": False
        }

    def test_image_compression(self):
        """测试核心逻辑：图像是否被正确压缩和解压，且没有严重的形状丢失"""
        original_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # 编码
        encoded_bytes = VLAProtocol.encode_image(original_img, quality=80)
        assert isinstance(encoded_bytes, bytes)
        assert len(encoded_bytes) > 0
        
        # 解码
        decoded_img = VLAProtocol.decode_image(encoded_bytes)
        assert decoded_img.shape == original_img.shape
        assert decoded_img.dtype == np.uint8

    def test_pack_unpack_payload_fidelity(self, dummy_observation):
        """测试字典打包和解包后的数据保真度及类型转换"""
        payload_bytes = VLAProtocol.pack_payload(dummy_observation)
        unpacked_obs = VLAProtocol.unpack_payload(payload_bytes)
        
        # 1. 验证字符串
        assert unpacked_obs["cmd"] == dummy_observation["cmd"]
        assert unpacked_obs["use_cached_cmd"] == dummy_observation["use_cached_cmd"]
        
        # 2. 验证 float64 是否被自动降维成了 float32
        assert unpacked_obs["state"].dtype == np.float32
        # np.allclose 用于比较浮点数数组是否足够接近
        assert np.allclose(unpacked_obs["state"], dummy_observation["state"], atol=1e-5)
        
        # 3. 验证图像恢复
        assert unpacked_obs["front_image"].shape == dummy_observation["front_image"].shape

    @pytest.mark.parametrize("invalid_img", [
        np.zeros((224, 224), dtype=np.uint8),       # 缺少通道数 (2D)
        np.zeros((224, 224, 3), dtype=np.float32)   # 错误的数据类型
    ])
    def test_pack_payload_ignores_invalid_images(self, invalid_img):
        """边界测试：只有 (H, W, 3) 且为 uint8 的才会被当做图像压缩，否则原样保留"""
        obs = {"weird_tensor": invalid_img}
        payload_bytes = VLAProtocol.pack_payload(obs)
        unpacked = VLAProtocol.unpack_payload(payload_bytes)
        
        # 因为不符合图像特征，协议层不应该将其转为 bytes，而是原样打包为 ndarray
        assert isinstance(unpacked["weird_tensor"], np.ndarray)
        assert unpacked["weird_tensor"].dtype == invalid_img.dtype