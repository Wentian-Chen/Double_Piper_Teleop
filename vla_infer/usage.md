## vla adapter
```
conda activate dream-adapter
python vla_infer/example/vla-adapter/vla-adapter_server.py \
    --model_path=/checkpoint/vla-adapter/configs+merged_dataset_1_offset_state_converted+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--train-merged_dataset_1_offset_state_converted-0415--250000_chkpt \
    --task_suite_name merged_dataset_1_offset_state_converted \
    --num_open_loop_steps 50

python vla_infer/example/vla-adapter/vla-adapter-piper_client.py \
    --state_type joint \
    --action_type joint \
    --control_interval_s 0.04 \
    --execute_chunk_steps 2


python vla_infer/example/dream-adapter/dream-adapter-piper_client.py \
  --server_ip 127.0.0.1 \
  --port 5556 \
  --timeout_ms 120000 \
  --task_instruction "move_cup_to_dish_0425" \
  --state_type joint \
  --action_type joint \
  --absolute_action true \
  --enable_rtc true \
  --rtc_execution_horizon 10 \
  --rtc_max_guidance_weight 10.0 \
  --rtc_prefix_attention_schedule LINEAR \
  --server_action_prefix_steps 10 \
  --execute_chunk_steps 200 \
  --control_interval_s 0.01 \
  --enable_action_interpolation true \
  --interpolation_target_steps 100 \
  --record_cameras false
```

## dream adapter
```
conda activate dream-adapter
#  完全体
python vla_infer/example/dream-adapter/dream-adapter_server.py \
    --model_path /home/charles/workspaces/Dream-adapter/outputs/configs+merged_dataset_1_offset_state_converted+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--train-merged_dataset_1_offset_state_converted-0415--240000_chkpt \
    --task_suite_name merged_dataset_1_offset_state_converted \
    --num_open_loop_steps 50
#  抓香蕉
"Pick up the banana and place it into the bowl."
python vla_infer/example/dream-adapter/dream-adapter_server.py \
    --model_path /home/charles/workspaces/Dream-adapter/outputs/configs+pick_banana_200_newTable_1_offset_absolute_converted+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--train-noop_1_offset_absolute-0326--20000_chkpt \
    --task_suite_name pick_banana_200_newTable_1_offset_absolute_converted \
    --num_open_loop_steps 50
#  绿色方块放到黄色
"Pick up the green cube and place it on top of the yellow cube."
python vla_infer/example/dream-adapter/dream-adapter_server.py \
    --model_path /home/charles/workspaces/Dream-adapter/outputs/configs+pick_place_block_all_0408_1_offset_state_converted+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--train-pick_place_block_all_0408_1_offset_state_converted-0414--20000_chkpt \
    --task_suite_name pick_place_block_all_0408_1_offset_state_converted \
    --num_open_loop_steps 50

# 先抓蓝色方块，再抓红色方块
"First grasp the blue block, then grasp the red block."
python vla_infer/example/dream-adapter/dream-adapter_server.py \
    --model_path /home/charles/workspaces/Dream-adapter/outputs/configs+pick_block_100_1_offset_state_converted+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--train-block_100_1_offset_state_absolute-0326--20000_chkpt \
    --task_suite_name pick_block_100_1_offset_state_converted \
    --num_open_loop_steps 50 

# 把胡萝卜放到盘子里
"Pick up the carrot and put it on the plate."
python vla_infer/example/dream-adapter/dream-adapter_server.py \
    --model_path /home/charles/workspaces/Dream-adapter/outputs/configs+carrots_to_plate_0413_1_offset_state_converted+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--train-carrots_to_plate_0413_1_offset_state_converted-0414--20000_chkpt \
    --task_suite_name carrots_to_plate_0413_1_offset_state_converted \
    --num_open_loop_steps 50   

# 抓可乐瓶子或茄子
"Pick up the red bottle and put it into the pot."
"Pick up the eggplant and put it into the pot."
python vla_infer/example/dream-adapter/dream-adapter_server.py \
    --model_path /home/charles/workspaces/Dream-adapter/outputs/configs+pick_red_eggplant_to_pot_0415_1_offset_state_converted+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--train-pick_red_eggplant_to_pot_0415_1_offset_state_converted-0415--20000_chkpt \
    --task_suite_name pick_red_eggplant_to_pot_0415_1_offset_state_converted \
    --num_open_loop_steps 50  

# 抓起海绵擦桌子
"Pick up the sponge and wipe the table completely."
python vla_infer/example/dream-adapter/dream-adapter_server.py \
    --model_path /home/charles/workspaces/Dream-adapter/outputs/configs+sponge_wipe_0423_1_offset_state_converted+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--train-sponge_wipe_0423_1_offset_state_converted-0415--10000_chkpt \
    --task_suite_name sponge_wipe_0423_1_offset_state_converted \
    --num_open_loop_steps 50 

# 打开抽屉
"Open the drawer."
python vla_infer/example/dream-adapter/dream-adapter_server.py \
    --model_path /home/charles/workspaces/Dream-adapter/outputs/configs+open_cabinet_all_0423_1_offset_state_converted+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--train-open_cabinet_all_0423_1_offset_state_converted-0415--10000_chkpt \
    --task_suite_name open_cabinet_all_0423_1_offset_state_converted \
    --num_open_loop_steps 50   

# 把勺子放在盘子旁边
"Put the spoon to the right of the plate."
"Put the spoon to the left of the plate."
python vla_infer/example/dream-adapter/dream-adapter_server.py \
    --model_path /home/charles/workspaces/Dream-adapter/outputs/configs+adjust_fork_0410_1_offset_state_converted+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--train-adjust_fork_0410_1_offset_state_converted-0414--20000_chkpt \
    --task_suite_name adjust_fork_0410_1_offset_state_converted \
    --num_open_loop_steps 50

```


```
"Pick up the green cube and place it on top of the yellow cube."


```
```  
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
