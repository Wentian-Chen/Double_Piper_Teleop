# VLA-JEPA 真机部署示例（Piper）

本目录是 **VLA-JEPA 真机部署的控制端入口**：服务端（模型推理）跑在 `VLA-JEPA-Alex` 仓库，
客户端（机器人 PC）跑在这里。两侧通过 **ZMQ REQ/REP + msgpack_numpy + JPEG** 通信，
协议契约见 `<VLA-JEPA-Alex>/doc/robot/piper_server_plan.md`。

## 0. 组成与运行位置

| 文件 | 运行位置 | 环境 | 作用 |
|---|---|---|---|
| `<VLA-JEPA-Alex>/examples/real-robot/piper_zmq_server.py` | GPU 机器 | VLA `.venv` | **真正的服务端**：收观测 → 推理 → 返回绝对关节角动作块 |
| `<VLA-JEPA-Alex>/examples/real-robot/selftest_client.py` | GPU 机器 | VLA `.venv` | 协议自检客户端（不需要权重/GPU/机械臂） |
| `vlajepa_server.py` | 机器人 PC | 控制端环境 | server 启动器：用正确的 python/目录/参数拉起上面那个 server |
| `vlajepa_piper_client.py` | 机器人 PC | 控制端环境 | **真机客户端**：采关节角+双相机 → 请求 server → 下发动作块 |
| `vlajepa_dataset_replay.py` | 机器人 PC | 控制端环境 | **离线回放**：把 `PiperSingleRobot` 换成数据集，跑整条 episode 并算 MSE + 轨迹图 |

```
机器人 PC (控制端 env)                          GPU 机器 (VLA .venv)
┌───────────────────────────────┐  ZMQ REQ/REP  ┌──────────────────────────────┐
│ vlajepa_piper_client.py       │ ───────────►  │ piper_zmq_server.py          │
│  PiperSingleRobot → obs       │  msgpack+JPEG │   VLAJepaPiperPolicy         │
│  execute(abs_action) ◄────────│               │   predict_action → 反归一化   │
└───────────────────────────────┘               └──────────────────────────────┘
```

> **为什么 server 不写在本仓库**：VLA-JEPA 需要 starVLA 重型依赖（transformers / flash-attn），
> 且本仓库 `vla_infer` 要求 `numpy>=2.0`，而 VLA 仓库钉 `numpy==1.26.4`——**两个环境不能合并**。
> 所以服务端只在 VLA 仓库实现，本目录只负责“启动它 / 连接它”。

## 1. 环境前提

| 侧 | 需要的环境 | 关键依赖 |
|---|---|---|
| server | `<VLA-JEPA-Alex>/.venv/bin/python` | torch / starVLA / pyzmq / msgpack-numpy / Pillow |
| 启动器 `vlajepa_server.py` | 控制端环境（或任意装了 `draccus` 的 python） | draccus（只做参数解析与 subprocess） |
| 客户端 | 控制端环境 | draccus / numpy / pyzmq / msgpack-numpy / robot 驱动（`my_robot`） |

- `vlajepa_server.py` 的 `--vla_python` 留空时会自动用 `<vla_repo>/.venv/bin/python`，
  一般不需要手填。
- 客户端必须在**能访问机械臂与相机**的控制端环境里运行。

## 2. 启动 server（GPU 机器）

### 2.1 先做协议自检（不需要权重、GPU、机械臂）

```bash
cd <VLA-JEPA-Alex>

# 终端 1：dry-run server（不载入模型，返回“保持位姿”块）
.venv/bin/python examples/real-robot/piper_zmq_server.py \
    --dry-run --host 127.0.0.1 --port 15555 --max-requests 4

# 终端 2：协议自检客户端
.venv/bin/python examples/real-robot/selftest_client.py \
    --host 127.0.0.1 --port 15555 --steps 3 --send-bad-payload
```

期望输出：坏 payload 时 server 回 `{'error': ...}`（不超时），随后 3 次正常请求均返回
`ndarray(7, 7) dtype=float32`，最后打印 `✅ 协议自检通过`。

### 2.2 启动真实权重 server

```bash
cd <VLA-JEPA-Alex>

.venv/bin/python examples/real-robot/piper_zmq_server.py \
    --ckpt_path checkpoints/iclr_adjust_cup/final_model/pytorch_model.pt \
    --host 0.0.0.0 --port 5555 --use-bf16 \
    --default-instruction "Put the cup the right way up on the table."
```

启动日志里应看到（缺任何一条都说明环境/参数有问题）：

- `model loaded in ...s | state_dim=7 chunk_steps=7 ...`
- `warmup done in ...s`
- `ZMQ REP server ready on tcp://0.0.0.0:5555`
- `client must be configured with state_type=joint + absolute_action=True`

> ⚠️ `--default-instruction` 必须填**该 checkpoint 训练时的任务原文**，不是 `task_index` 数字。
> 训练侧 dataloader 会把 parquet 里的 `task_index` 用 `meta/tasks.jsonl` 解析成字符串
> （`starVLA/dataloader/gr00t_lerobot/datasets.py:1027`）。取原文：
>
> ```bash
> python - <<'PY'
> import json
> p = "<dataset>/meta/tasks.jsonl"   # 例如
> # /home/liuxx/repo/datasets/lerobot/miku112/real-robot-v2.1-vlajepa/adjust_cup_0409_1_offset_state_v2_1/meta/tasks.jsonl
> for line in open(p):
>     d = json.loads(line)
>     print(d["task_index"], d["task"])
> PY
> ```
> 当前 `adjust_cup_0409_1_offset_state_v2_1` 的 `task_index=0` 原文是
> `Put the cup the right way up on the table.`

### 2.3 用启动器拉起（可选，在机器人 PC 上一条命令）

```bash
cd <Double_Piper_Teleop>

python vla_infer/example/vlajepa/vlajepa_server.py \
    --vla_repo <VLA-JEPA-Alex> \
    --ckpt_path <VLA-JEPA-Alex>/checkpoints/iclr_adjust_cup/final_model/pytorch_model.pt \
    --port 5555 --use_bf16
```

- 默认 `--vla_python` = `<vla_repo>/.venv/bin/python`；如用别的环境用 `--vla_python` 覆盖。
- `--default-instruction` 默认已填当前任务原文；换 checkpoint 时必须改。
- 透传额外参数用 `--extra_args "--no-binarize-gripper --max-requests 100"`。

## 3. 启动 client（机器人 PC）

**先起 server 并等到 `ZMQ REP server ready`，再起 client。**

```bash
cd <Double_Piper_Teleop>

# 首次上机：小步数 + 有人守急停
python vla_infer/example/vlajepa/vlajepa_piper_client.py \
    --server_ip <GPU_IP> --port 5555 \
    --task_instruction "Put the cup the right way up on the table." \
    --max_steps 20
```

客户端默认值即本项目约定（无需再传）：

| 字段 | 默认 | 为什么 |
|---|---|---|
| `state_type` | `joint` | 发 `joint(6)+gripper(1)`，与训练 state 语义一致 |
| `action_type` | `joint` | 保留 `joint_state`，兼容 delta 分支 |
| `absolute_action` | `True` | 服务端输出绝对关节角，原样下发 |
| `enable_gripper_transform` | `False` | 避免把夹爪指令再下压 0.3 |
| `execute_chunk_steps` | `7` | = 模型 chunk 长度 `future_action_window_size + 1` |
| `timeout_ms` | `5000` | 单次推理 = VLM 前向 + 4 步去噪；超时=急停，故比库默认(2000)宽 |
| `stop_on_timeout` | `True` | 安全默认 |
| `task_instruction` | 当前任务原文 | 与 checkpoint 语言条件一致 |

显式传了非默认值（如 `--state_type qpos`）时，客户端启动会打 warning。

## 4. 联合配置表（server ↔ client，勿随意改）

| 项 | server | client | 说明 |
|---|---|---|---|
| 协议 | ZMQ REP `--port` | ZMQ REQ `--port` | 端口必须一致 |
| 图像 | 接受 HWC3 uint8 224 | `adaptive_resize_image` + `check_uint8_rgb` | letterbox 224，与离线回放一致 |
| state | 要求 7 维有限值 | `state_type=joint` → `joint(6)+gripper(1)` | **不能**用 `qpos`（那是另一字段） |
| 动作 | 输出绝对关节角 `(7,7)` | `absolute_action=True` → 原样下发 | 客户端**不能**按增量 cumsum |
| 夹爪 | 默认二值化 0/1；`--no-binarize-gripper` 走连续 | `enable_binary_gripper=False` + `enable_gripper_transform=False` | 见 §7 未验证项 |
| 指令 | `--default-instruction` | `--task_instruction` | 两者都应等于训练任务原文 |
| chunk | `chunk_steps=7` | `execute_chunk_steps=7` | 执行 7 步/块 |

## 5. 离线验证（可选，不接机器人）

### 5.1 用本仓库客户端跑一整条 episode（推荐，走真实 client 代码路径）

`vlajepa_dataset_replay.py` 把 `PiperSingleRobot` 换成数据集回放，其余代码路径与真机
**完全一致**（`VLAJepaPiperClient` → `VlaZmqClient` → `VLAProtocol` → ZMQ），因此能在没有
硬件时验证"客户端预处理 + 协议 + 服务端推理"，并与数据集真值算 MSE、画轨迹图。

```bash
# server（VLA 仓库）
cd <VLA-JEPA-Alex>
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python examples/real-robot/piper_zmq_server.py \
    --ckpt_path checkpoints/iclr_adjust_cup/final_model/pytorch_model.pt \
    --host 127.0.0.1 --port 15570 --use-bf16 --no-binarize-gripper \
    --default-instruction "Put the cup the right way up on the table."

# client（本仓库，另一个终端）
cd <Double_Piper_Teleop>
python vla_infer/example/vlajepa/vlajepa_dataset_replay.py \
    --dataset_root /home/liuxx/repo/datasets/lerobot/miku112/real-robot-v2.1-vlajepa/adjust_cup_0409_1_offset_state_v2_1 \
    --episode 0 \
    --output_dir <VLA-JEPA-Alex>/eval_openloop/iclr_adjust_cup_vlajepa_client_replay_ep0 \
    --port 15570 --execute_chunk_steps 7     # 7=部署配置；1=单步重规划
```

产物在 `--output_dir`：`replay_result_chunkstep{7,1}.json`（MSE/MAE 逐维）、
`predictions_chunkstep{7,1}.npz`、`trajectory_pred_vs_gt_*.png`、`error_over_time_*.png`、
`phase_joint1_joint2_*.png`。

已实测（checkpoint `iclr_adjust_cup`，episode 0 = 140 帧，任务原文 `Put the cup the right way up on the table.`）：

| 模式 | 观测点数 | 原始 MSE | 原始 MAE | q01/q99 归一化 MAE | 服务端延迟 |
|---|---|---|---|---|---|
| `execute_chunk_steps=7`（部署） | 20 | 0.00416 | 0.0480 | 0.0771 | ~175 ms |
| `execute_chunk_steps=1`（单步） | 140 | 0.00427 | 0.0482 | 0.0789 | ~175 ms |

轨迹图显示 7 个维度（joint1–6 + gripper）的预测曲线均紧贴真值，夹爪开合切换也被正确复现。

### 5.2 用 VLA 仓库自带开环脚本（对照）

```bash
# client（VLA 仓库另一个终端）
cd <VLA-JEPA-Alex>
.venv/bin/python scripts/piper_zmq_openloop_client.py \
    --config_yaml checkpoints/iclr_adjust_cup/config.yaml \
    --output_dir eval_openloop/iclr_adjust_cup_piper_zmq_openloop_ep0 \
    --host 127.0.0.1 --port 15570 \
    --num-episodes 1 --stride 1 --max-steps 0 --server-gripper-mode continuous
```

已实测：P95 延迟约 178 ms，step-0 归一化 MAE 约 0.069（该脚本的图像 resize 与归一化基准
与 §5.1 略有差异，数值不必完全一致）。

## 6. 排错

| 现象 | 可能原因 | 处理 |
|---|---|---|
| client 超时（`TimeoutError`）后急停 | server 未起 / 仍在载模型 / 首帧冷启动 | 先起 server 等到 `ZMQ REP server ready`；必要时调大 `--timeout_ms` |
| server 回 `{'error': ...}` | 请求缺 state/图像，或 handler 异常 | 看 server 日志 traceback；`state_type` 必须 `joint` |
| 启动无 `model loaded` / `warmup done` 日志 | starVLA 的 `overwatch` 曾禁用模块 logger | 已修复（server 会在 load 后重新启用）；确认用的是最新 `examples/real-robot` |
| `ModuleNotFoundError: draccus` | 用了 VLA `.venv` 跑客户端/启动器 | 启动器/客户端在控制端环境跑，server 才用 VLA `.venv` |
| `找不到服务端脚本` | `--vla_repo` 不对 | 指向 `VLA-JEPA-Alex` 仓库根 |
| 动作方向/幅度明显异常 | 任务指令不对 / 客户端被配成 `qpos` / 夹爪被 transform | 核对 §4 联合配置表与客户端启动 warning |

## 7. 安全清单（真机前必读）

1. 机械臂**悬空或垫高**，低速，有人守急停；`--max_steps` 先给小值（如 20）。
2. 先跑 §2.1 协议自检，再跑 §5 离线回放，最后才上真机。
3. 确认 `task_instruction` 与 checkpoint 训练任务原文完全一致。
4. 观察 server 日志里的 `action first/last`（绝对关节角）量级是否合理。
5. 时延必须稳定小于 `timeout_ms`（超时=急停）。
6. 动作语义：全部 Piper 任务权重为 `action_type: absolute`；若换增量权重，server 与 client 必须同时改。

## 8. 已知限制 / 未验证

- **夹爪量纲与方向未确认**：Piper 夹爪指令的合法区间/方向未在 `vla_infer` 暴露标定；
  server 默认输出 0/1，`--no-binarize-gripper` 输出连续物理值（按 `q01/q99`）。真机前必须依据
  Piper 驱动标定二选一（见 `<VLA-JEPA-Alex>/doc/robot/deployment.md` §4.4、plan Q2）。
- **图像预处理与训练不完全一致**：训练侧是直接拉伸到 224（`starVLA/dataloader/gr00t_lerobot/datasets.py`），
  客户端是 letterbox（保持长宽比 + 白边，`src/process/utils.py:214-227`）；同一张图两种预处理像素分布不同，
  可能影响精度（见 plan G6）。需要时先用两种方式各跑一次离线回放对比。
- **关节软限位/最大速度未加**：server 只做形状/有限性校验，未做逐维 clamp。
- **未做真机端到端**：本目录的 client 需在真实 `PiperSingleRobot` + 相机上联调后才能确认闭环安全。
- **无鉴权/客户端租约**：server 默认 `0.0.0.0`，生产环境必须绑可信网卡并加防火墙。
