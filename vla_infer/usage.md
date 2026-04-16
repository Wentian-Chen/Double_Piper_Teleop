## vla adapter
```
conda activate dream-adapter
python vla_infer/example/vla-adapter/vla-adapter_server.py \
    --model_path=/home/charles/workspaces/VLA-Adapter/outputs/configs+pick_banana_100_newTable_converted+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--train-vla-0315-01--20000_chkpt \
    --task_suite_name pick_banana_100_newTable_converted \
    --num_open_loop_steps 25

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
    --model_path /home/charles/workspaces/Dream-adapter/outputs/configs+adjust_cup_0409_1_offset_state_converted+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--train-adjust_cup_0409_1_offset_state_converted-0414--20000_chkpt \
    --task_suite_name adjust_cup_0409_1_offset_state_converted \
    --num_open_loop_steps 50
  
python vla_infer/example/dream-adapter/dream-adapter-piper_client.py \
    --task_instruction "Right the cup on the table." \
    --state_type joint \
    --action_type joint \
    --control_interval_s 0.1 \
    --execute_chunk_steps 25 \
	--enable_binary_gripper false \
    --binary_gripper_threshold 0.4 \
	--gripper_open_value 0.5 \
	--gripper_closed_value 0.2 \
    --enable_action_interpolation False \
    --interpolation_method linear \
    --interpolation_target_steps 30 \
    --show_output_track false

"Pick up the carrot and put it on the plate."
            "args": [
                "--task_instruction", "Pick up the towel and wipe the table twice, then put the towel on the plate.",
                "--state_type","joint",
                "--action_type","joint",
                "--control_interval_s","0.1",
                "--execute_chunk_steps","25",
                "--enable_binary_gripper","false",
                "--binary_gripper_threshold","0.4",
                "--gripper_open_value","0.5",
                "--gripper_closed_value","0.2",
                "--enable_action_interpolation","False",
                "--interpolation_method","linear",
                "--interpolation_target_steps","30",
                "--show_output_track","true",
                "--enable_gripper_transform","true",
                "--gripper_transform_threshold","0.8",
                "--gripper_transform_delta","0.5"
            ]
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
    --policy_path /home/charles/workspaces/lerobot/outputs/smolvla/pick_block_100_1_offset_state_smolvla_0327-1934/050000/pretrained_model

/home/charles/workspaces/lerobot/outputs/smolvla/pick_banana_200_newTable_0322_45k/045000/pretrained_model

conda activate dream-adapter
python vla_infer/example/smolvla/smolvla_piper_client.py \
    --task_instruction "First grasp the blue block, then grasp the red block." \
    --execute_chunk_steps 25 \
    --control_interval_s 0.2 \
    --state_type joint \
    --action_type joint \
    --control_type absolute \
    --enable_action_interpolation false \
    --interpolation_method linear \
    --interpolation_target_steps 25 \
    --enable_smooth_action false 

# "Pick up the banana and place it in the bowl"
# "First grasp the blue block, then grasp the red block."
```