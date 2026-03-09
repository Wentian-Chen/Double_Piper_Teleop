# vla_infer

基于 ZMQ 的 VLA（Vision-Language-Action）推理部署模块，旨在为机器人提供延迟低、易扩展的远程推理能力。采用 `Client -> Server -> Model` 的解耦架构。

## 1. 核心架构

- **`src/zmq/`**: 底层通信层。基于 ZeroMQ 实现高效的请求-响应模型，内置自动的图像 JPEG 压缩/解压，显著降低网络带宽需求。
- **`src/models/`**: 模型抽象与适配层。
  - `base.py`: 定义 `BaseVLAModel` 接口。
  - `smolvla_model.py`: 封装 [SmolVLA](https://huggingface.co/blog/smolvla)，支持 LeRobot 转换后的模型。
  - `vla_adapter_model.py`: 封装 [VLA-Adapter](https://github.com/vla-adapter/VLA-Adapter)。
- **`src/inference/`**: 服务逻辑封装。
  - `server.py`: `ModelZmqInferenceServer` 串联 ZMQ Server 与模型实例。
  - `client.py`: 提供高层 API，支持动作块（Action ChunkING）平滑处理。
- **`src/robots/`**: 硬件适配层。针对 Piper 机器人及其相机系统的深度集成。
- **`src/process/`**: 图像预处理与动作后处理（如平滑转换）。

## 2. 快速开始

### 2.1 安装

建议在专用虚拟环境中安装：

```bash
cd vla_infer
pip install -e .
```

> **注意**：模型依赖（如 `torch`, `transformers`, `lerobot`）需根据所使用的模型单独配置。

### 2.2 启动推理服务端 (Server)

以 VLA-Adapter 为例，可以使用提供的示例脚本：

```bash
# 需指定模型路径
python example/vla-adapter/vla-adapter_server.py --model_path /path/to/your/checkpoint
```

手动启动逻辑：
```python
from vla_infer.src.models import VLAAdapterModel
from vla_infer.src.zmq.zmq_server import VlaZmqServer
from vla_infer.src.inference.server import ModelZmqInferenceServer

model = VLAAdapterModel(pretrained_checkpoint="/path/to/ckpt")
zmq_server = VlaZmqServer(port=5555)
server = ModelZmqInferenceServer(model=model, zmq_server=zmq_server)
server.start()
```

### 2.3 启动机器人客户端 (Client)

客户端会自动捕获机器人状态与相机图像，发送给服务端并执行返回的动作块：

```bash
python example/vla-adapter/vla-adapter-piper_client.py --server_ip 127.0.0.1 --task_instruction "Pick up the banana"
```

## 3. 协议规范

### 3.1 请求 (Observation)
客户端发送给服务端的字典通常包含：
- `image`: 主视角 RGB 图像 (H, W, 3)
- `wrist_image`: 手腕视角 RGB 图像 (H, W, 3)
- `state`: 机器人当前状态 (7维: 6轴 + 1夹爪)
- `instruction`: 任务指令字符串

### 3.2 响应 (Action)
服务端返回：
- `action`: 动作块 (T, 7)，其中 T 为 chunk size。

## 4. 特色功能

- **图像压缩**: 自动将 `numpy` 图像转换为高质量 JPEG 流传输。
- **动作平滑**: 客户端内置 `smooth_action_chunk` 等工具，减少机器人震动。
- **多模型支持**: 通过统一的 `BaseVLAModel` 快速切换不同的 VLA 模型（如 OpenVLA、SmolVLA）。
- **Piper 深度集成**: 专门针对 Piper 机械臂优化的 `PiperSingleRobot` 类，支持单臂及其双相机同步采集。

## 5. 项目结构详述

```text
vla_infer/
├── src/
│   ├── inference/  # 客户端/服务端逻辑
│   ├── models/     # 模型适配器 (VLA-Adapter, SmolVLA)
│   ├── robots/     # 机器人接口 (Piper)
│   ├── zmq/        # ZMQ 通讯与图像编解码
│   └── process/    # 图像预处理 & 动作后处理
├── example/        # 完整运行示例
└── tests/          # 单元测试与模型验证
```
