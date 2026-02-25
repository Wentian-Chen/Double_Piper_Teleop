'''
Author: boboji11 wendychen112@qq.com
Date: 2026-02-25 15:56:14
LastEditors: boboji11 wendychen112@qq.com
LastEditTime: 2026-02-25 17:31:53
FilePath: \Double_Piper_Teleop\vla_infer\protocol.py
Description: 

Copyright (c) 2026 by boboji11 , All Rights Reserved. 
'''
import io
import msgpack
import msgpack_numpy as m
import numpy as np
from PIL import Image
import logging
import typing

# 启用 msgpack 的 numpy 支持
m.patch()

class VLAProtocol:
    """
    高度可扩展的 VLA 序列化协议层 (支持动态字典输入与内存安全)
    """
    
    @classmethod
    def encode_image(cls, image_array: np.ndarray, quality: int = 80) -> bytes:
        """
        使用 PIL 将 Numpy 图像编码为 JPEG，带有严格的内存管理
        """
        img = Image.fromarray(image_array)
        
        # 使用 with 语句强制管理上下文，确保离开代码块时立刻释放内存
        with io.BytesIO() as buffer:
            img.save(buffer, format="JPEG", quality=quality)
            return buffer.getvalue()

    @classmethod
    def decode_image(cls, image_bytes: bytes) -> np.ndarray:
        """解码 JPEG 字节流"""
        with io.BytesIO(image_bytes) as buffer:
            img = Image.open(buffer)
            # 强制转换为 numpy 数组，并确保加载到内存后关闭图像文件
            img.load() 
            return np.array(img)

    @classmethod
    def pack_payload(cls, payload: typing.Dict[str, typing.Any], jpeg_quality: int = 80) -> bytes:
        """
        通用的动态打包函数 (替代原有的 pack_request / pack_response)
        根据字典 value 的数据类型进行自动优化
        """
        processed_payload = {}
        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                # 启发式判断：3D 且为 uint8 的数组，视为图像进行压缩
                if value.ndim == 3 and value.dtype == np.uint8:
                    processed_payload[key] = cls.encode_image(value, quality=jpeg_quality)
                # 精度优化：将 float64 降级为 float32
                elif value.dtype == np.float64:
                    processed_payload[key] = value.astype(np.float32)
                else:
                    processed_payload[key] = value
            else:
                # 字符串、布尔值（如 use_cached_cmd）等直接保留
                processed_payload[key] = value
                
        return msgpack.packb(processed_payload, use_bin_type=True)

    @classmethod
    def unpack_payload(cls, payload_bytes: bytes) -> typing.Dict[str, typing.Any]:
        """
        通用的动态解包函数
        提取数据，并自动将图像字节流还原为 Numpy 数组
        """
        unpacked = msgpack.unpackb(payload_bytes, raw=False)
        
        for key, value in unpacked.items():
            # 通过 key 命名规范或数据类型来反推图像
            # 企业开发中，通常约定键名包含 'image' 或 'img' 的 bytes 为图像
            if isinstance(value, bytes) and ('img' in key.lower() or 'image' in key.lower()):
                unpacked[key] = cls.decode_image(value)
                
        return unpacked