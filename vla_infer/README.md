# vla_infer

基于 ZMQ 的 VLA 推理部署模块，采用 `Client -> Server -> Model` 架构。

## 1. 安装

```bash
cd vla_infer
pip install -r requirements.txt
```

> 说明：`smolvla` 与 `vla-adapter` 模型本身依赖 `torch/transformers/lerobot` 等，请按各自模型目录依赖完成安装。

## 2. 架构说明

- `vla_infer/client.py`：发送观测请求，接收动作回复。
- `vla_infer/server.py`：承载模型实例，循环处理请求。
- `vla_infer/protocol.py`：字典序列化协议，自动处理图像编码/解码。
- `vla_infer/models/base.py`：模型统一抽象基类 `BaseVLAModel`。
- `vla_infer/models/smolvla_model.py`：SmolVLA 的标准封装实现。
- `vla_infer/models/vla_adapter_model.py`：VLA-Adapter 的标准封装实现。
- `vla_infer/robots/base.py`：机器人抽象接口。
- `vla_infer/robots/piper_single.py`：PiperSingle 适配层（复用原有相机能力，不重写 camera）。

## 3. 字典协议规范

### 3.1 请求字典（Client -> Server）

推荐格式：

```python
{
	"cmd": "pick up the banana and put it into the bowl",  # str, 任务指令
	"cam_head": np.ndarray((H, W, 3), dtype=np.uint8),      # RGB 头相机
	"cam_wrist": np.ndarray((H, W, 3), dtype=np.uint8),     # RGB 腕相机
	"state": np.ndarray((7,), dtype=np.float32),            # [joint(6), gripper(1)]
}
```

可兼容的图像键别名：

- 头相机：`cam_head | image_head | front_image | full_image`
- 腕相机：`cam_wrist | image_wrist | wrist_image`

可兼容的状态键别名：

- `state | robot_state | proprio`

缓存指令机制字段（可选）：

```python
{
	"use_cached_cmd": bool,
	"cmd": str,
}
```

### 3.2 响应字典（Server -> Client）

统一输出：

```python
{
	"action": np.ndarray((T, D), dtype=np.float32)
}
```

- `T`：动作块长度（chunk size）
- `D`：动作维度（单臂 Piper 至少 7 维，`joint(6)+gripper(1)`）

## 4. 模型封装接口

所有模型遵循：

```python
class BaseVLAModel(ABC):
	def load_model(self) -> None: ...
	def predict(self, observation: dict) -> dict: ...
```

### 4.1 SmolVLA

```python
from vla_infer.models import SmolVLAModel

model = SmolVLAModel(
	model_path="/path/to/smolvla_checkpoint",
	dataset_repo_id="miku112/piper-pick-banana-50",
	dataset_root="/path/to/lerobot_dataset",
	action_chunk_size=50,
)
```

### 4.2 VLA-Adapter

```python
from vla_infer.models import VLAAdapterModel

model = VLAAdapterModel(
	model_path="/path/to/vla_adapter_checkpoint",
	base_model_checkpoint="/path/to/openvla_base",  # 可选
)
```

## 5. PiperSingle 机器人适配

`PiperSingleRobot` 复用 `my_robot/agilex_piper_single_base.py` 中 `PiperSingle` 的控制器与相机，不额外实现 camera。

```python
from vla_infer.robots import PiperSingleRobot

robot = PiperSingleRobot(auto_setup=True)
robot.reset()
obs = robot.get_observation()

# obs 字典示例
# {
#   "cam_head": ...,
#   "cam_wrist": ...,
#   "front_image": ...,
#   "wrist_image": ...,
#   "state": np.ndarray(7,),
#   "joint": np.ndarray(6,),
#   "qpos": np.ndarray(6,),
#   "gripper": np.ndarray(1,)
# }

robot.apply_action({"action": np.zeros((1, 7), dtype=np.float32)})
```

## 6. Server 端最小示例

```python
from vla_infer.server import VLAServer
from vla_infer.models import SmolVLAModel

model = SmolVLAModel(
	model_path="/path/to/smolvla_checkpoint",
	dataset_repo_id="miku112/piper-pick-banana-50",
	dataset_root="/path/to/lerobot_dataset",
)
server = VLAServer(model=model, port=5555)
server.run()
```

## 7. Client 端最小示例

```python
import numpy as np

from vla_infer.client import VLAClient

client = VLAClient(server_ip="127.0.0.1", port=5555, timeout_ms=2000)

obs = {
	"cam_head": np.zeros((480, 640, 3), dtype=np.uint8),
	"cam_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
	"state": np.zeros((7,), dtype=np.float32),
}

result = client.get_action(cmd_text="pick up the banana", obs_dict=obs, jpeg_quality=80)
print(result["action"].shape)
client.close()
```