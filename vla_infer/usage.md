## vla adapter
```
conda activate dream-adapter
python vla_infer/example/vla-adapter/vla-adapter_server.py \
    --model_path=/home/charles/workspaces/VLA-Adapter/outputs/configs+pick_banana_100_newTable_converted+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--train-vla-0315-01--20000_chkpt \
    --task_suite_name pick_banana_100_newTable_converted

python vla_infer/example/vla-adapter/vla-adapter-piper_client.py \
    --state_type joint \
    --action_type joint \
    --control_interval_s 0.04 \
    --execute_chunk_steps 2
```
## dream adapter
```
conda activate dream-adapter
python vla_infer/example/vla-adapter/vla-adapter_server.py \
    --model_path=/home/charles/workspaces/Dream-adapter/outputs/configs+pick_banana_200_newTable_converted+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--train-0319-01--15000_chkpt \
    --task_suite_name pick_banana_200_newTable_converted
  
python vla_infer/example/dream-adapter/dream-adapter-piper_client.py \
    --state_type joint \
    --action_type joint \
    --control_interval_s 0.04 \
    --execute_chunk_steps 6 \
	--enable_binary_gripper True \
    --binary_gripper_threshold 0.4 \
	--gripper_open_value 0.5 \
	--gripper_closed_value 0.2 \
    --enable_action_interpolation true \
    --interpolation_method linear \
    --interpolation_target_steps 16
```


```
/home/charles/workspaces/VLA-Adapter/outputs/configs+pick_banana_100_newTable_converted+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--train-vla-0315-01--20000_chkpt

```
## 激活机械臂
```
conda activate dream-adapter
bash piper_sdk/can_single_activate.sh 
python piper_sdk/piper_enable_modeJ_after_tech.py

python piper_sdk/piper_ctrl_moveJ_keyboard.py
```
## smolvla
```
source /home/charles/workspaces/lerobot/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
python vla_infer/example/smolvla/smolvla_server.py \
    --policy_path /home/charles/workspaces/lerobot/outputs/smolvla/pick_banana_200_newTable_next_state_action_0322_50k/050000/pretrained_model

/home/charles/workspaces/lerobot/outputs/smolvla/pick_banana_200_newTable_0322_45k/045000/pretrained_model

conda activate dream-adapter
python vla_infer/example/smolvla/smolvla_piper_client.py \
    --execute_chunk_steps 12 \
    --control_interval_s 0.02 \
    --state_type joint \
    --action_type joint \
    --control_type absolute \
    --enable_smooth_action True 
```