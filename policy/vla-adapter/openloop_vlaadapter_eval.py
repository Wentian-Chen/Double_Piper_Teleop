"""
run_openloop_eval.py

Evaluates a policy in an open-loop manner using a given dataset.
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import draccus
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# Append VLA-Adapter project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../VLA-Adapter/")))

from experiments.robot.openvla_utils import (
    get_processor,
    get_vla_action,
    get_action_head,
    get_proprio_projector,
)
from experiments.robot.robot_utils import (
    get_model,
    set_seed_everywhere,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")

@dataclass
class EvalConfig:
    # Model parameters
    pretrained_checkpoint: Union[str, Path]
    base_model_checkpoint: Optional[Union[str, Path]] = None
    dataset_path: Union[str, Path] = "./data/miku112/pick_banana_50_rlds"
    dataset_split: str = "train"
    model_family: str = "openvla"
    
    # Evaluation parameters
    max_episodes: Optional[int] = None
    
    # Input parameters matching run_libero_eval
    use_l1_regression: bool = True
    use_minivlm: bool = True  
    use_pro_version: bool = True
    
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    num_diffusion_steps: int = 50 
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 1
    unnorm_key: str = "pick_banana_50" # Will be updated from dataset_statistics if possible
    save_version: str = "vla-adapter" 

def initialize_model(cfg: EvalConfig):
    """Initialize model and associated components."""
    # Load model
    # Check if we need to load base model + adapter manually
    if cfg.base_model_checkpoint and os.path.exists(cfg.base_model_checkpoint):
        logger.info(f"Detected base model checkpoint: {cfg.base_model_checkpoint}")
        try:
            # Imports needed for manual loading
            from transformers import AutoModelForVision2Seq, AutoConfig, AutoImageProcessor, AutoProcessor
            from experiments.robot.openvla_utils import OpenVLAConfig, PrismaticImageProcessor, PrismaticProcessor, OpenVLAForActionPrediction
            from peft import PeftModel
            
            # Register classes
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            
            # Load Base Model
            logger.info("Loading base VLA model...")
            model = AutoModelForVision2Seq.from_pretrained(
                cfg.base_model_checkpoint,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # Load LoRA Adapter
            lora_path = os.path.join(cfg.pretrained_checkpoint, "lora_adapter")
            if os.path.exists(lora_path):
                 logger.info(f"Loading LoRA adapter from {lora_path}")
                 model = PeftModel.from_pretrained(model, lora_path)
                 model = model.merge_and_unload()
            elif os.path.exists(os.path.join(cfg.pretrained_checkpoint, "adapter_config.json")):
                 logger.info(f"Loading LoRA adapter from {cfg.pretrained_checkpoint}")
                 model = PeftModel.from_pretrained(model, cfg.pretrained_checkpoint)
                 model = model.merge_and_unload()
            else:
                 logger.warning("Could not find LoRA adapter files. Proceeding with base model only (risky if not intended).")

        except Exception as e:
            logger.error(f"Failed to load base model + adapter: {e}")
            raise e
    else:
        logger.info(f"loading pretrained checkpoint from {cfg.pretrained_checkpoint}")
        model = get_model(cfg)

    model.set_version(cfg.save_version)
    
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        try:
            llm_dim = model.config.text_config.hidden_size
        except:
            llm_dim = 4096
        
        proprio_projector = get_proprio_projector(
            cfg,
            llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for Piper
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression:
        try:
            llm_dim = model.config.text_config.hidden_size
        except:
            llm_dim = 4096
        action_head = get_action_head(cfg, llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)


    return model, action_head, proprio_projector, noisy_action_projector, processor

@draccus.wrap()
def eval_openloop(cfg: EvalConfig) -> None:
    set_seed_everywhere(0)
    
    # 1. Load Model
    logger.info(f"Loading model components from {cfg.pretrained_checkpoint}")
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    model.eval()
    
    # Move to GPU if needed (and not already there)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if not next(model.parameters()).is_cuda and not (cfg.load_in_8bit or cfg.load_in_4bit):
        model = model.to(device)

    # 4. Load Dataset Statistics
    stats_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.exists(stats_path):
        logger.info(f"Loading dataset statistics from {stats_path}")
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        if cfg.unnorm_key not in stats:
             keys = list(stats.keys())
             if len(keys) > 0:
                 logger.warning(f"Key {cfg.unnorm_key} not found in stats. Using {keys[0]}")
                 cfg.unnorm_key = keys[0]
             else:
                 logger.error("Dataset statistics is empty!")
        
        # In openvla/prismatic models, norm_stats is usually an attribute of the model
        if not hasattr(model, 'norm_stats'):
            model.norm_stats = stats
        else:
            model.norm_stats.update(stats)
    else:
        logger.warning("No dataset_statistics.json found! Unnormalization might fail if model does not have it loaded.")

    # 5. Load Dataset
    dataset_path = os.path.abspath(cfg.dataset_path)
    logger.info(f"Loading dataset from {dataset_path}")
    
    builder = None
    try:
        builder = tfds.builder_from_directory(dataset_path)
    except Exception as e:
        logger.info(f"Failed to load builder from {dataset_path}, trying to find inner folder...")
        found = False
        for root, dirs, files in os.walk(dataset_path):
            if "dataset_statistics.json" in files:
                dataset_path = root
                found = True
                break
        if found:
             builder = tfds.builder_from_directory(dataset_path)
        else:
            raise ValueError(f"Could not find dataset_info.json in {cfg.dataset_path}. TFDS Error: {e}")

    logger.info(f"Dataset Info: {builder.info}")
    ds = builder.as_dataset(split=cfg.dataset_split)
    
    # 6. Evaluate
    total_mse = 0.0
    total_samples = 0
    all_episode_mses = []
    proprio_dim = 8 # Default for Piper
    
    logger.info("Starting evaluation...")
    
    if cfg.max_episodes:
        ds = ds.take(cfg.max_episodes)

    # Create plots directory
    plots_dir = os.path.join(os.getcwd(), "openloop_eval_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    iterator = tfds.as_numpy(ds)
        
    for i, episode in enumerate(tqdm.tqdm(iterator, desc="Episodes")):
        episode_mse = 0
        episode_steps = 0
        gt_actions_episode = []
        pred_actions_episode = []
        
        try:
             step_iter = episode['steps']
        except KeyError:
             logger.error("Episode does not contain 'steps'")
             continue

        for step in step_iter:
            obs = step['observation']
            
            # Extract Image
            if 'image' in obs:
                full_image = obs['image']
            elif 'image_primary' in obs:
                full_image = obs['image_primary']
            else:
                img_keys = [k for k in obs.keys() if 'image' in k and 'wrist' not in k]
                if img_keys:
                    full_image = obs[img_keys[0]]
                else:
                    # Maybe it is just keys like '0', '1'? 
                    # Use the first key that looks like an image?
                    raise ValueError(f"No primary image found in observation: {obs.keys()}")
            
            # Extract Proprio
            proprio = None
            if cfg.use_proprio:
                if 'proprio' in obs:
                    proprio = obs['proprio']
                elif 'state' in obs:
                    proprio = obs['state']
                
                if proprio is not None and proprio.shape[-1] == 7 and proprio_dim == 8:
                    # Pad the proprio state: [XYZ (3), RPY (3), PAD (1), Gripper (1)]
                    obs_pose = proprio[:6]
                    obs_gripper = proprio[6:]
                    padding = np.zeros(list(proprio.shape[:-1]) + [1], dtype=proprio.dtype)
                    proprio = np.concatenate([obs_pose, padding, obs_gripper], axis=-1)
            
            # Extract Instruction
            if 'language_instruction' in step:
                task_label = step['language_instruction'].decode('utf-8')
            elif 'language_instruction' in episode:
                task_label = episode['language_instruction'].decode('utf-8')
            else:
                task_label = "do something" 

            # Ground Truth Action
            # The model predicts delta actions [dx, dy, dz, droll, dpitch, dyaw, gripper]
            # We should compare against 'action_eef_delta' if available, otherwise 'action'.
            if 'action_eef_delta' in step:
                 gt_action = step['action_eef_delta']
            else:
                 gt_action = step['action']
            
            # Construct Input
            input_obs = {
                "full_image": full_image,
                "state": proprio
            }
            if "image_wrist" in obs:
                 input_obs["image_wrist"] = obs["image_wrist"]
            elif "wrist_image" in obs:
                 input_obs["image_wrist"] = obs["wrist_image"]
            
            # Predict
            try:
                # Assuming get_vla_action handles everything including moving tensors to device
                pred_actions = get_vla_action(
                    cfg, model, processor, input_obs, task_label,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    use_minivlm=cfg.use_minivlm
                )
                pred_action = pred_actions[0] # Take first step
                
                if pred_action.shape != gt_action.shape:
                   # Try to align shapes if just slightly off (e.g. (7,) vs (1,7))
                   pred_action = pred_action.reshape(gt_action.shape)
                
                gt_actions_episode.append(gt_action)
                pred_actions_episode.append(pred_action)

                mse = np.mean((pred_action - gt_action) ** 2)
                episode_mse += mse
                episode_steps += 1
                
            except Exception as e:
                logger.error(f"Error in prediction: {e}")
                # Don't break the whole loop, just skip step? Or maybe break episode.
                break
        
        if episode_steps > 0:
            avg_ep_mse = episode_mse / episode_steps
            total_mse += avg_ep_mse
            total_samples += 1
            all_episode_mses.append(avg_ep_mse)
            if i % 10 == 0:
                 logger.info(f"Episode {i}: MSE = {avg_ep_mse:.6f}")
            

            # Plot the episode (limit to first 5 episodes to save space)
            if i < 5:
                gt_arr = np.array(gt_actions_episode)
                pred_arr = np.array(pred_actions_episode)
                
                # GT/Pred shape: (T, 7)
                # Dim 0-2: EEF Delta XYZ
                # Dim 3-5: EEF Delta RPY
                # Dim 6: Gripper (Absolute 0/1 usually)
                dim_names = ['dX', 'dY', 'dZ', 'dR', 'dP', 'dY', 'Gripper']
                
                dim = gt_arr.shape[1]
                fig, axes = plt.subplots(dim, 1, figsize=(10, 2 * dim))
                if dim == 1:
                    axes = [axes]
                
                for d in range(dim):
                    axes[d].plot(gt_arr[:, d], label='GT Delta', color='blue')
                    axes[d].plot(pred_arr[:, d], label='Pred Delta', color='red', linestyle='--')
                    axes[d].set_ylabel(dim_names[d] if d < len(dim_names) else f'Dim {d}')
                    axes[d].grid(True)
                    axes[d].legend()
                
                plt.xlabel('Step')
                plt.suptitle(f'Episode {i} - Delta Action Comparison (MSE={avg_ep_mse:.4f})')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"episode_{i}_delta_comparison.png"))
                plt.close()
                logger.info(f"Saved episode_{i}_delta_comparison.png for episode {i} in {plots_dir}")

    if total_samples > 0:
        final_avg_mse = total_mse / total_samples
        logger.info(f"Final Average MSE over {total_samples} episodes: {final_avg_mse:.6f}")

        # Plot Episodes vs MSE curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(all_episode_mses)), all_episode_mses, marker='o', linestyle='-', color='b', label='Episode MSE')
        plt.xlabel('Episode')
        plt.ylabel('Average MSE')
        plt.title(f'MSE per Episode ({cfg.pretrained_checkpoint.split("/")[-1]})')
        plt.grid(True)
        
        # Annotate average MSE in a blank area (top right)
        annotation_text = f"Average MSE over {total_samples} episodes: {final_avg_mse:.6f}"
        plt.text(0.95, 0.95, annotation_text, transform=plt.gca().transAxes, 
                 fontsize=12, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.legend()
        plot_path = os.path.join(plots_dir, "episodes_mse_curve.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved MSE curve to {plot_path}")
    else:
        logger.info("No samples evaluated.")

if __name__ == "__main__":
    eval_openloop()
