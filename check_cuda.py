#!/usr/bin/env python3
"""
PyTorch CUDA 测试脚本
测试 GPU 是否可用、CUDA 版本、GPU 数量及详细信息
"""

import torch
import sys

def test_cuda():
    print("=" * 60)
    print("PyTorch CUDA 测试报告")
    print("=" * 60)
    
    # 1. 测试 CUDA 是否可用
    print(f"\n【1】CUDA 是否可用: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA 不可用！请检查：")
        print("   - NVIDIA 驱动是否安装")
        print("   - CUDA Toolkit 是否安装")
        print("   - PyTorch 是否为 CUDA 版本 (不是 CPU 版本)")
        print("\n安装 CUDA 版 PyTorch:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)
    
    # 2. CUDA 版本信息
    print(f"\n【2】CUDA 版本信息:")
    # print(f"   PyTorch 编译时 CUDA 版本: {torch.version.cuda}")
    print(f"   PyTorch 运行时 CUDA 版本: {torch.backends.cudnn.version()}")
    
    # 3. GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"\n【3】检测到 GPU 数量: {gpu_count}")
    
    # 4. 每个 GPU 的详细信息
    print(f"\n【4】GPU 详细信息:")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\n   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   - 总显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"   - 计算能力: {props.major}.{props.minor}")
        print(f"   - 多处理器数量: {props.multi_processor_count}")
    
    # 5. 当前默认 GPU
    current_device = torch.cuda.current_device()
    print(f"\n【5】当前默认 GPU: GPU {current_device} ({torch.cuda.get_device_name(current_device)})")
    
    # 6. 测试实际计算
    print(f"\n【6】执行张量计算测试:")
    try:
        # 创建测试张量
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        
        # 执行矩阵乘法
        z = torch.matmul(x, y)
        
        print(f"   ✓ 成功在 GPU 上创建张量")
        print(f"   ✓ 成功执行矩阵乘法: {x.shape} × {y.shape} = {z.shape}")
        print(f"   ✓ 结果张量位置: {z.device}")
        
        # 显存使用情况
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"\n   显存使用情况:")
        print(f"   - 已分配显存: {allocated:.2f} MB")
        print(f"   - 保留显存: {reserved:.2f} MB")
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！CUDA 工作正常")
        print("=" * 60)
        
    except Exception as e:
        print(f"   ❌ 计算测试失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_cuda()