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
    set_seed_everywhere
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
    # Control behavior
    lock_ry: bool = True
    # rpy delta scale: model outputs in same unit as qpos
    rpy_delta_scale: float = 1.0


# ====== Input / Output Helpers ======

def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")
    return np.ascontiguousarray(image)


def build_proprio(controller_data: Dict[str, Any]) -> np.ndarray:
    # StateEncoding.POS_EULER: XYZ (3) + RPY (3) + PAD (1) + Gripper (1)
    eef = np.array(controller_data["left_arm"]["qpos"], dtype=np.float32).reshape(-1)
    gripper = np.array([controller_data["left_arm"]["gripper"]], dtype=np.float32).reshape(-1)
    if eef.shape[0] != 6:
        raise ValueError(f"Expected EEF state dim=6, got {eef.shape[0]}")
    pad = np.zeros((1,), dtype=np.float32)
    proprio = np.concatenate([eef, pad, gripper], axis=0)
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


def wrap_rpy_error(rpy_err: np.ndarray, full_turn_deg: float = 360.0, unit_scale: float = 1e-6) -> np.ndarray:
    """
    Wrap rpy error to [-180, 180] degrees in the scaled unit used by qpos.
    qpos_rpy unit is (degrees * unit_scale).
    """
    half_turn = (full_turn_deg / 2.0) * unit_scale
    full_turn = full_turn_deg * unit_scale
    return (rpy_err + half_turn) % full_turn - half_turn


def action_to_move(
    action: np.ndarray,
    current_qpos: np.ndarray,
    max_delta_pos: float = 0.05,
    max_delta_rpy: float = 0.2,
    lock_ry: bool = True,
    rpy_delta_scale: float = 1e-6,
) -> Dict[str, Any]:
    # ActionEncoding.EEF_POS: delta XYZ (3) + delta RPY (3) + gripper (1)
    if action.ndim > 1:
        action = action[0]
    if action.shape[-1] < 7:
        raise ValueError(f"Expected action dim=7, got {action.shape[-1]}")

    delta_pos = clamp(action[:3], -max_delta_pos, max_delta_pos)
    delta_rpy = clamp(action[3:6], -max_delta_rpy, max_delta_rpy)

    target_qpos = current_qpos.copy()
    target_qpos[:3] += delta_pos
    if lock_ry:
        # Update only pitch (index 1 in RPY), lock roll and yaw
        target_qpos[3] = current_qpos[3]
        target_qpos[4] += delta_rpy[1] * rpy_delta_scale
        target_qpos[5] = current_qpos[5]
    else:
        target_qpos[3:6] += delta_rpy * rpy_delta_scale

    gripper_raw = float(np.clip(action[6], 0.0, 0.7))
    gripper = gripper_raw
    gripper = 0.7 if gripper_raw > 0.5 else 0.3

    return {
        "arm": {
            "left_arm": {
                "qpos": target_qpos.tolist(),
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
    robot.reset_position()

    # Configure Piper motion mode (EEF position control)
    try:
        piper_ctrl = robot.controllers["arm"]["left_arm"].controller
        # ctrl_mode=0x01 (CAN cmd), move_mode=0x00 (MOVE P), move_spd_rate=100, is_mit_mode=0x00
        piper_ctrl.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        print("Piper MotionCtrl_2 set to CAN mode + MOVE P mode.")
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
    current_qpos: Optional[np.ndarray] = None
    last_feedback_qpos: Optional[np.ndarray] = None
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
        if current_qpos is None:
            current_qpos = np.array(controller_data["left_arm"]["qpos"], dtype=np.float32).reshape(-1)
        print(f"action_list: {action_list}")
        print(f" current_qpos: {current_qpos}")
        for i in range(exec_steps):
            if step >= max_steps:
                break
            action = action_list[i]
            
            # calculate target position
            move_data = action_to_move(
                action,
                current_qpos,
                lock_ry=cfg.lock_ry,
                rpy_delta_scale=cfg.rpy_delta_scale,
            )
            print(f"Step {step}: Move Data: {move_data}")

            robot.move_modeP(move_data["arm"]["left_arm"]["qpos"], move_data["arm"]["left_arm"]["gripper"])
            step += 1
            time.sleep(0.04)
            # feedback error logging
            feedback = robot.get()[0]["left_arm"]["qpos"]
            feedback_qpos = np.array(feedback, dtype=np.float32)
            # target_qpos = np.array(move_data["arm"]["left_arm"]["qpos"], dtype=np.float32)
            # err = target_qpos - feedback_qpos
            # err[3:6] = wrap_rpy_error(err[3:6])
            # err_norm = float(np.linalg.norm(err))
            # print(f"Step {step - 1}: Target qpos: {target_qpos}")
            # print(f"Step {step - 1}: Feedback qpos: {feedback_qpos}")
            # print(f"Step {step - 1}: Qpos error: {err}")
            # if last_feedback_qpos is not None:
            #     delta_fb = float(np.linalg.norm(feedback_qpos - last_feedback_qpos))
            #     if delta_fb < stall_delta_thresh and err_norm > stall_err_thresh:
            #         stall_count += 1
            #         print(
            #             f"[WARN] Feedback not moving (delta={delta_fb:.6f}) with large error "
            #             f"(norm={err_norm:.6f}). stall_count={stall_count}"
            #         )
            #     else:
            #         stall_count = 0
            # last_feedback_qpos = feedback_qpos
            # if stall_count >= stall_max_steps:
            #     print("[ERROR] Arm appears stalled; stopping to avoid accumulating error.")
            #     stalled = True
            #     break
            # update current_qpos for next step using feedback
            current_qpos = feedback_qpos
        # if stalled:
        #     break

    print("Finished.")
    robot.reset()


if __name__ == "__main__":
    main()
