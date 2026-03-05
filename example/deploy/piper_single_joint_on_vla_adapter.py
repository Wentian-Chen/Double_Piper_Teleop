import sys
import os
import json
import time
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple
import draccus
import numpy as np

# ====== Path Setup ======
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DOUBLE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
VLA_ROOT = os.path.join(REPO_ROOT, "VLA-Adapter")

sys.path.insert(0, DOUBLE_ROOT)
sys.path.insert(0, VLA_ROOT)

from my_robot.agilex_piper_single_base import PiperSingle
from utils.data_handler import is_enter_pressed

from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import (
    get_model,
    get_action,
    set_seed_everywhere,
)


@dataclass
class DeployConfig:
    # Model parameters
    pretrained_checkpoint: str = ''

    model_family: str = "openvla"

    # Input parameters
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    # Match dataset_statistics key in checkpoint (see dataset_statistics.json)
    unnorm_key: str = "pick_banana_50"

    # Action head / adapter settings
    use_l1_regression: bool = True
    use_minivlm: bool = True
    use_pro_version: bool = True
    use_film: bool = False

    # Quantization
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Action chunk execution
    execute_chunk_steps: int = 8
    
    # Joint control parameters
    max_delta_joint: float = 0.5 # radians or unit used in model


# ====== Input / Output Helpers ======

def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")
    return np.ascontiguousarray(image)


def build_proprio(controller_data: Dict[str, Any]) -> np.ndarray:
    # StateEncoding.JOINT: Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)
    # Piper typically has 6 joints
    joints = np.array(controller_data["left_arm"]["joint"], dtype=np.float32).reshape(-1)
    gripper = np.array([controller_data["left_arm"]["gripper"]], dtype=np.float32).reshape(-1)
    
    # Pad to 7 joints if fewer
    if joints.shape[0] < 7:
        pad_len = 7 - joints.shape[0]
        pad = np.zeros((pad_len,), dtype=np.float32)
        proprio = np.concatenate([joints, pad, gripper], axis=0)
    else:
        proprio = np.concatenate([joints[:7], gripper], axis=0)
    
    return proprio


def build_obs(sensor_data: Dict[str, Any], proprio: np.ndarray) -> Dict[str, Any]:
    img_head = to_uint8_rgb(sensor_data["cam_head"]["color"])
    img_wrist = to_uint8_rgb(sensor_data["cam_wrist"]["color"])

    obs = {
        "full_image": img_head,
        "state": proprio,
        "image_wrist": img_wrist,
    }
    return obs


def clamp(value: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(value, lo), hi)


def action_to_move(
    action: np.ndarray,
    current_joints: np.ndarray,
    max_delta_joint: float = 0.5,
) -> Dict[str, Any]:
    # ActionEncoding.JOINT_POS: Joint Delta Position (7) + Gripper Open/Close (1)
    if action.ndim > 1:
        action = action[0]
    if action.shape[-1] < 8:
        raise ValueError(f"Expected action dim=8, got {action.shape[-1]}")

    # The first 7 elements are delta joints (we only need the first 6 for Piper)
    delta_joints = action[:len(current_joints)]
    delta_joints = clamp(delta_joints, -max_delta_joint, max_delta_joint)

    target_joints = current_joints + delta_joints
    
    # The 8th element (index 7) is gripper state
    gripper = float(np.clip(action[7], 0.0, 1.0))
    if gripper > 0.5:
        gripper = 0.7
    else:
        gripper = 0.4
    return {
        "arm": {
            "left_arm": {
                "joint": target_joints.tolist(),
                "gripper": gripper,
            }
        }
    }


def ensure_unnorm_key(model: Any, cfg: DeployConfig) -> None:
    if not hasattr(model, "norm_stats"):
        return
    if cfg.unnorm_key in model.norm_stats:
        return
    keys = list(model.norm_stats.keys())
    if keys:
        print(f"[WARN] unnorm_key {cfg.unnorm_key} not found, use {keys[0]}")
        cfg.unnorm_key = keys[0]

@draccus.wrap()
def main(cfg: DeployConfig):
    set_seed_everywhere(0)

    print("Initializing Robot...")
    robot = PiperSingle()
    robot.set_up()
    robot.reset() # Use joint reset

    # Configure Piper motion mode (Joint position control)
    try:
        piper_ctrl = robot.controllers["arm"]["left_arm"].controller
        # ctrl_mode=0x01 (CAN cmd), move_mode=0x01 (MOVE J), move_spd_rate=100, is_mit_mode=0x00
        piper_ctrl.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        print("Piper MotionCtrl_2 set to CAN mode + MOVE J mode.")
    except Exception as e:
        print(f"[WARN] Failed to set MotionCtrl_2: {e}")

    print(f"Loading VLA-Adapter model from {cfg.pretrained_checkpoint}...")
    model = get_model(cfg)
    processor = get_processor(cfg)

    # Load action head + proprio projector (continuous action & proprio)
    try:
        llm_dim = model.config.text_config.hidden_size
    except Exception:
        llm_dim = 4096

    action_head = get_action_head(cfg, llm_dim)
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, llm_dim, proprio_dim=8)

    ensure_unnorm_key(model, cfg)

    task_instruction = "pick up the banana"
    max_steps = 2000

    print("Press ENTER to start inference...")
    while True:
        if is_enter_pressed():
            break
        time.sleep(0.1)

    print("Inference Started!")
    step = 0
    current_joints: Optional[np.ndarray] = None
    last_feedback_joints: Optional[np.ndarray] = None
    stall_count = 0
    stall_delta_thresh = 1e-4
    stall_err_thresh = 1e-2
    stall_max_steps = 5
    stalled = False
    
    while step < max_steps:
        data = robot.get()
        controller_data, sensor_data = data[0], data[1]

        proprio = build_proprio(controller_data)
        obs = build_obs(sensor_data, proprio)

        try:
            actions = get_action(
                cfg,
                model,
                obs,
                task_instruction,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                use_minivlm=cfg.use_minivlm,
                use_film=cfg.use_film,
            )
        except Exception as e:
            print(f"Inference failed: {e}")
            break

        # Normalize action container to a list for chunk execution
        if isinstance(actions, list):
            action_list = actions
        elif isinstance(actions, np.ndarray) and actions.ndim > 1:
            action_list = [actions[i] for i in range(actions.shape[0])]
        else:
            action_list = [actions]
        
        exec_steps = min(cfg.execute_chunk_steps, len(action_list))
        
        if current_joints is None:
            current_joints = np.array(controller_data["left_arm"]["joint"], dtype=np.float32).reshape(-1)
            
        print(f"action_list sample (0): {action_list[0]}")
        print(f" current_joints: {current_joints}")
        
        for i in range(exec_steps):
            if step >= max_steps:
                break
            action = action_list[i]
            
            # calculate target joints
            move_data = action_to_move(
                action,
                current_joints,
                max_delta_joint=cfg.max_delta_joint,
            )
            print(f"Step {step}: Move Data: {move_data}")

            # Use robot.move as requested
            robot.move(move_data)
            step += 1
            time.sleep(0.02)
            
            # feedback error logging
            feedback = robot.get()[0]["left_arm"]["joint"]
            feedback_joints = np.array(feedback, dtype=np.float32)
            target_joints = np.array(move_data["arm"]["left_arm"]["joint"], dtype=np.float32)
            err = target_joints - feedback_joints
            err_norm = float(np.linalg.norm(err))
            
            print(f"Step {step - 1}: Target joints: {target_joints}")
            print(f"Step {step - 1}: Feedback joints: {feedback_joints}")
            print(f"Step {step - 1}: Joints error norm: {err_norm}")
            
            if last_feedback_joints is not None:
                delta_fb = float(np.linalg.norm(feedback_joints - last_feedback_joints))
                if delta_fb < stall_delta_thresh and err_norm > stall_err_thresh:
                    stall_count += 1
                    print(
                        f"[WARN] Feedback not moving (delta={delta_fb:.6f}) with large error "
                        f"(norm={err_norm:.6f}). stall_count={stall_count}"
                    )
                else:
                    stall_count = 0
            
            last_feedback_joints = feedback_joints
            if stall_count >= stall_max_steps:
                print("[ERROR] Arm appears stalled; stopping to avoid accumulating error.")
                stalled = True
                break
            
            # update current_joints for next step using feedback
            current_joints = feedback_joints
            
        if stalled:
            break

    print("Finished.")
    robot.reset()


if __name__ == "__main__":
    main()
