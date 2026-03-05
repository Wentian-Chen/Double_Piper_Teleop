'''
pip install tensorflow[and-cuda]>=2.17.0 tensorflow-datasets tensorflow-hub h5py numpy apache-beam mlcroissant

'''
from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import h5py
import os
import json
import random

# Define the base path for datasets
_DATASET_ROOT = '/home/charles/workspaces/Double_Piper_Teleop/datasets'

def _get_configs():
    configs = []
    if os.path.exists(_DATASET_ROOT):
        for child in os.listdir(_DATASET_ROOT):
            child_path = os.path.join(_DATASET_ROOT, child)
            if os.path.isdir(child_path):
                configs.append(
                    tfds.core.BuilderConfig(
                        name=child,
                        version=tfds.core.Version('1.0.0'),
                        description=f'Dataset {child}.'
                    )
                )
    return configs

class PiperDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Piper robot datasets."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    
    # Dynamically generate configs based on folders in datasets directory
    BUILDER_CONFIGS = _get_configs()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if os.environ.get('PIPER_DISABLE_GPU', '0') == '1':
            try:
                tf.config.set_visible_devices([], 'GPU')
            except Exception as e:
                print(f"Warning: failed to disable GPU: {e}")
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Head camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot EEF state, consists of [6x pose (x,y,z, r,p,y), 1x gripper].',
                        ),
                        'joint_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint state, consists of [6x joint angles, 1x gripper].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [6x joint angles, 1x gripper]. Actions are the state of the NEXT step.',
                    ),
                    'action_eef_delta': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot EEF delta action, consists of [3x delta pos, 3x delta rpy, 1x gripper absolute].',
                    ),
                    'joint_delta': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Robot joint delta action, consists of [7x joint delta pos (padded), 1x gripper absolute target].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        # Use the builder config name (e.g. 'pick_banana_50') to find the dataset folder
        
        task_name = self.builder_config.name
        dataset_path = os.path.join(_DATASET_ROOT, task_name)
        
        if not os.path.exists(dataset_path):
             print(f"Warning: Path {dataset_path} not found.")

        return {
            'train': self._generate_examples(path=dataset_path),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        
        if not os.path.exists(path):
            return
            
        # Try load instructions
        task_name = self.builder_config.name
        instruction_path = f'/home/charles/workspaces/Double_Piper_Teleop/task_instructions/{task_name}.json'
        
        possible_instructions = [task_name.replace('_', ' ')] # Default fallback
        
        if os.path.exists(instruction_path):
            try:
                with open(instruction_path, 'r') as f:
                    data = json.load(f)
                    if "instructions" in data and isinstance(data["instructions"], list):
                        possible_instructions = data["instructions"]
            except Exception as e:
                print(f"Error loading instructions from {instruction_path}: {e}")
        
        # Check for MAX_EPISODES env var
        max_episodes = os.environ.get('MAX_EPISODES', '-1')
        try:
            max_episodes = int(max_episodes)
        except ValueError:
            max_episodes = -1

        files = glob.glob(os.path.join(path, '*.hdf5'))
        files.sort()
        
        # Limit files if MAX_EPISODES is set and positive
        if max_episodes > 0:
            print(f"Limiting to {max_episodes} episodes.")
            files = files[:max_episodes]

        for file_path in files:
            try:
                with h5py.File(file_path, 'r') as root:
                    # Verified paths based on user inspection
                    path_head_color = 'cam_head/color'
                    path_wrist_color = 'cam_wrist/color'
                    path_joint = 'left_arm/joint'
                    path_gripper = 'left_arm/gripper'
                    path_qpos = 'left_arm/qpos' # EEF pose

                    # Check existence of minimal required fields
                    if path_head_color not in root or path_joint not in root:
                        print(f"Skipping {file_path}: missing keys")
                        continue

                    # Read data
                    head_imgs = root[path_head_color][()]
                    wrist_imgs = root[path_wrist_color][()]
                    joints = root[path_joint][()]
                    grippers = root[path_gripper][()] 
                    
                    if path_qpos in root:
                        qpos = root[path_qpos][()]
                    else:
                        qpos = np.zeros((joints.shape[0], 6), dtype=np.float32)

                    num_steps = head_imgs.shape[0]

                    # Ensure shapes align
                    if len(grippers.shape) == 1:
                        grippers = grippers[:, np.newaxis]
                    
                    # joint_state = joint + gripper (7 dims)
                    joint_states = np.concatenate([joints, grippers], axis=-1)
                    
                    # state = EEF (6) + gripper (1) (7 dims)
                    eef_states = np.concatenate([qpos, grippers], axis=-1)

                    # Actions: Shifted joint states (predict next joint pos)
                    actions = np.zeros_like(joint_states)
                    actions[:-1] = joint_states[1:]
                    actions[-1] = joint_states[-1]

                    # Actions EEF Delta: Delta Pose + Absolute Gripper Target
                    # qpos is (N, 6) [x,y,z, r,p,y]
                    delta_pose = np.zeros_like(qpos)
                    delta_pose[:-1] = qpos[1:] - qpos[:-1]
                    delta_pose[-1] = 0.0

                    # Gripper target is the gripper state at the next timestep
                    target_gripper = np.zeros_like(grippers)
                    target_gripper[:-1] = grippers[1:]
                    target_gripper[-1] = grippers[-1]

                    action_eef_delta = np.concatenate([delta_pose, target_gripper], axis=-1)

                    # Action Joint Delta: 7x Joint Delta Position (padded) + 1x Gripper Open/Close
                    delta_joints = np.zeros_like(joints)
                    delta_joints[:-1] = joints[1:] - joints[:-1]
                    delta_joints[-1] = 0.0
                    
                    pads = np.zeros((num_steps, 1), dtype=np.float32)
                    joint_delta = np.concatenate([delta_joints, pads, target_gripper], axis=-1)

                    # Pick a random instruction for this episode
                    lang_instruction = random.choice(possible_instructions)
                    
                    # Pre-compute embedding once per episode to speed up
                    lang_embedding = self._embed([lang_instruction])[0].numpy()

                    episode = []
                    for i in range(num_steps):
                        img = head_imgs[i]
                        wrist_img = wrist_imgs[i]
                        
                        episode.append({
                            'observation': {
                                'image': img,
                                'wrist_image': wrist_img,
                                'state': eef_states[i].astype(np.float32),
                                'joint_state': joint_states[i].astype(np.float32),
                            },
                            'action': actions[i].astype(np.float32),
                            'action_eef_delta': action_eef_delta[i].astype(np.float32),
                            'joint_delta': joint_delta[i].astype(np.float32),
                            'discount': 1.0,
                            'reward': float(i == (num_steps - 1)),
                            'is_first': i == 0,
                            'is_last': i == (num_steps - 1),
                            'is_terminal': i == (num_steps - 1),
                            'language_instruction': lang_instruction, # TODO: Read from metadata if available
                            'language_embedding': lang_embedding,
                        })

                    # Create the sample
                    sample = {
                        'steps': episode,
                        'episode_metadata': {
                            'file_path': file_path
                        }
                    }

                    # Use filename as ID
                    file_id = os.path.basename(file_path).replace('.hdf5', '')
                    yield file_id, sample
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
