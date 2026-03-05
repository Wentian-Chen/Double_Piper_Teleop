import sys
sys.path.append("./")
import os
import numpy as np
import time
import math
import pdb
from scipy.signal import savgol_filter
from my_robot.agilex_piper_single_base import PiperSingle
from policy.smolvla.inference_model import SMOLVLA
from utils.data_handler import is_enter_pressed

def input_transform(data):
    """
    Process robot data for SmolVLA model
    """
    # 1. State processing (Proprioception)
    # PiperSingle typically has 'left_arm'
    # We construct a 7-DOF state: 6 joints + 1 gripper
    
    left_joint = np.array(data[0]["left_arm"]["joint"]).reshape(-1)
    left_gripper = np.array(data[0]["left_arm"]["gripper"]).reshape(-1)
    
    # Construct state vector (7 dim for single arm)
    # Note: If your model was trained with dual arm state padding, adjust here.
    # Assuming the refined smolvla (finetuned on piper) expects just the relevant arm state 
    # OR standard OpenVLA which primarily uses vision.
    # We pass the state just in case the model wrapper uses it or for logging.
    state = np.concatenate([left_joint, left_gripper])
    
    # 2. Image processing
    # PiperSingle returns: data[1]["cam_head"]["color"], data[1]["cam_wrist"]["color"]
    img_head = data[1]["cam_head"]["color"]
    img_wrist = data[1]["cam_wrist"]["color"]
    
    return (img_head, img_wrist), state

def output_transform(action):
    """
    Process model output action to robot command
    """
    # Action expected: [Joint1, ..., Joint6, Gripper] (7 dims)
    # Or Chunked [T, 7]
    
    # Take first step if chunked
    if len(action.shape) > 1:
        action = action[0]
        
    # Joint Limits (Safety) - Piper specific
    # Radian limits from Piper documentation/code
    joint_limits_rad = [
        (math.radians(-150), math.radians(150)),   # joint1
        (math.radians(0), math.radians(180)),      # joint2
        (math.radians(-170), math.radians(0)),     # joint3
        (math.radians(-100), math.radians(100)),   # joint4
        (math.radians(-70), math.radians(70)),     # joint5
        (math.radians(-120), math.radians(120))    # joint6
    ]

    def clamp(value, min_val, max_val):
        return max(min_val, min(value, max_val))
    
    # Map Action to Joints
    # Assuming model output corresponds to [J1...J6, Gripper]
    target_joints = [
        clamp(action[i], joint_limits_rad[i][0], joint_limits_rad[i][1])
        for i in range(6)
    ]
    target_gripper = action[6]
    
    # Construct Move Command
    move_data = {
        "arm": {
            "left_arm": {
                "joint": target_joints,
                "gripper": target_gripper,
            }
        }
    }
    return move_data

def main():
    # 1. Configuration
    CKPT_PATH = "/home/charles/workspaces/lerobot/weights/train/piper-pick-banana-50/checkpoints/009000/pretrained_model" # TODO: Update this
    DATASET_REPO_ID = "miku112/piper-pick-banana-50" # TODO: Ensure this matches your training dataset for correct normalization
    DATASET_ROOT = "/home/charles/workspaces/lerobot/datasets/miku112/piper-pick-banana-50"
    TASK_INSTRUCTION = "pick up the banana and put it into the bowl" # TODO: Update this
    ACTION_CHUNK_SIZE = 50 # 无效参数 
    EXECUTE_STEPS= 15
    # 2. Initialize Robot
    print("Initializing Robot...")
    robot = PiperSingle()
    robot.set_up()
    robot.reset()
    
    # 3. Initialize Model
    print(f"Loading Model from {CKPT_PATH} with chunk size {ACTION_CHUNK_SIZE}...")
    try:
        model = SMOLVLA(
            model_path=CKPT_PATH, 
            dataset_repo_id=DATASET_REPO_ID, 
            dataset_root=DATASET_ROOT,
            action_chunk_size=ACTION_CHUNK_SIZE
        )
        model.random_set_language(TASK_INSTRUCTION)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Inference Loop
    max_step = 2000
    num_episode = 10
    
    for i in range(num_episode):
        print(f"--- Episode {i+1}/{num_episode} ---")
        
        # Reset
        robot.reset()
        model.reset_obsrvationwindows()
        model.random_set_language(TASK_INSTRUCTION) # Reset instruction just in case
        
        # Wait for start
        print("Press ENTER to start inference...")
        is_start = False
        while not is_start:
            if is_enter_pressed():
                is_start = True
            time.sleep(0.1)
            
        print("Inference Started!")
        
        step = 0
        while step < max_step:
            loop_start = time.time()
            
            # Get Observation
            data = robot.get()
            img_arr, state = input_transform(data)
            
            # Update Model
            model.update_observation_window(img_arr, state)
            
            # Predict
            try:
                # Use get_action_chunk to get the full sequence
                action = model.get_action_chunk()
            except Exception as e:
                print(f"Inference Failed: {e}")
                break
                
            # Execute
            if isinstance(action, np.ndarray) and action.ndim > 1:
                # Smoothing the action chunk
                try:
                    # Use adaptive logic for window length
                    chunk_len = action.shape[0]
                    # Aim for ~10-20% of chunk size or fixed small window
                    window_len = max(5, int(chunk_len * 0.2))
                    if window_len % 2 == 0:
                        window_len += 1
                    
                    # Ensure window_len is not larger than chunk_len
                    if window_len > chunk_len:
                         window_len = chunk_len if chunk_len % 2 == 1 else chunk_len - 1

                    if window_len >= 3:
                        # Smooth each dimension
                        for dim in range(action.shape[1]):
                            action[:, dim] = savgol_filter(action[:, dim], window_length=window_len, polyorder=2)
                except Exception as e:
                    print(f"Smoothing failed: {e}")

                # Execute sequential actions
                print(f"Executing chunk of size {action.shape[0]}")
                for i in range(EXECUTE_STEPS):
                    if step >= max_step:
                        break
                    
                    move_data = output_transform(action[i])
                    robot.move(move_data)
                    step += 1
                    
                    # Control execution frequency (e.g., 50Hz = 0.02s)
                    time.sleep(0.02)
                    
                    # Optional: Check for exit/break condition during chunk execution?
            else:
                move_data = output_transform(action)
                robot.move(move_data)
                step += 1
            
            print(f"Step {step}: action {action} Executed")
            print(f"Step {step}: move_data {move_data} Executed")

        print(f"Episode {i+1} Finished.")
        robot.reset()

if __name__ == "__main__":
    main()
