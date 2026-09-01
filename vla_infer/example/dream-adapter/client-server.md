cd /home/charles/workspaces/openpi-Alex

# 初始化 openpi 虚拟环境
source .venv/bin/activate

# 启动 OpenPI 的 Dream-Adapter ZMQ 兼容 server
python scripts/serve_policy.py \
  --transport DREAM_ZMQ \
  --port 5556 \
  policy:checkpoint \
  --policy.config=pi05_piper \
  --policy.dir=/checkpoint/merge/99999

OPENPI_REPO_ID=miku112/29_task_1_offset_state \
.venv/bin/python scripts/serve_policy.py \
  --transport DREAM_ZMQ \
  --port 5555 \
  policy:checkpoint \
  --policy.config=pi05_piper_recon_vlm \
  --policy.dir=/checkpoint/lxx/first/50000

OPENPI_REPO_ID=miku112/29_task_1_offset_state_converted_1 \
.venv/bin/python scripts/serve_policy.py \
  --transport DREAM_ZMQ \
  --port 5555 \
  policy:checkpoint \
  --policy.config=pi05_piper_recon_vlm \
  --policy.dir=/checkpoint/lxx/third/40000

OPENPI_REPO_ID=miku112/29_task_1_offset_state_converted_1 \
.venv/bin/python scripts/serve_policy.py \
  --transport DREAM_ZMQ \
  --port 5555 \
  policy:checkpoint \
  --policy.config pi05_piper_recon_vlm \
  --policy.dir /checkpoint/lxx/third/48000 \
  --policy.lora-attn-rank 16 \
  --policy.lora-attn-alpha 16.0 \
  --policy.lora-ffn-rank 16 \
  --policy.lora-ffn-alpha 16.0
# can0初始化
cd /home/charles/workspaces/Double_Piper_Teleop

sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

ip -details link show can0

# hdf5 可视化
python scripts/visualize_piper_hdf5.py \
  --hdf5_path /home/charles/workspaces/Double_Piper_Teleop/datasets/pick_eggplant_0415_1/5.hdf5 \
  --output_dir ./tmp/piper_hdf5_vis/pick_eggplant_0415_1 \
  --save_video True \
  --save_frames True \
  --max_frames 300

# 初始化 conda
cd /home/charles/workspaces/Double_Piper_Teleop
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dream-adapter

# 启动真机 client
python vla_infer/example/dream-adapter/dream-adapter-piper_client.py \
  --server_ip 127.0.0.1 \
  --port 5555 \
  --timeout_ms 120000 \
  --task_instruction "open_cabinet_0423" \
  --state_type joint \
  --action_type joint \
  --absolute_action true \
  --execute_chunk_steps 200 \
  --control_interval_s 0.01 \
  --interpolation_target_steps 333 \
  --enable_action_interpolation True \
  --record_cameras false


# 启动真机 client（RTC 模式）
# server 启动方式不变；RTC 由 client 的 --enable_rtc 控制。
python vla_infer/example/dream-adapter/dream-adapter-piper_client_copy_2.py \
  --server_ip 127.0.0.1 \
  --port 5555 \
  --timeout_ms 120000 \
  --task_instruction "pick_eggplant_from_cluttered_0414_1" \
  --state_type joint \
  --action_type joint \
  --absolute_action true \
  --enable_rtc true \
  --rtc_execution_horizon 10 \
  --rtc_max_guidance_weight 10.0 \
  --rtc_prefix_attention_schedule LINEAR \
  --execute_chunk_steps 200 \
  --control_interval_s 0.01 \
  --enable_action_interpolation true \
  --interpolation_target_steps 333 \
  --record_state_action false \
  --record_cameras false

# 0721之后的启动方式：具体修改：每个动作差值10步，action horizon变为可修改。
cd /home/charles/workspaces/Double_Piper_Teleop
conda activate dream-adapter

python vla_infer/example/dream-adapter/dream-adapter-piper_client-0721py \
  --server_ip 127.0.0.1 \
  --port 5555 \
  --timeout_ms 120000 \
  --task_instruction "pick_up_battery_0527" \
  --state_type joint \
  --action_type joint \
  --absolute_action true \
  --enable_rtc true \
  --rtc_execution_horizon 10 \
  --rtc_max_guidance_weight 10.0 \
  --rtc_prefix_attention_schedule LINEAR \
  --action_horizon 30 \
  --control_interval_s 0.01 \
  --initial_replan_count 3 \
  --delay 0 \
  --initial_replan_horizon 20 \
  --gripper_tighten_delta 0.2 \
  --enable_action_interpolation true \
  --interpolation_steps_per_action 10 \
  --record_state_action true \
  --record_cameras false

# mergem model 
python scripts/serve_policy.py \
  --transport DREAM_ZMQ \
  --port 5556 \
  policy:checkpoint \
  --policy.config=pi05_piper \
  --policy.dir=/checkpoint/merge/99999
