from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_state_action(save_path="state_action_comparison.png"):
    repo_id = "miku112/pick_banana_200_newTable_next_state_action"
    root = Path("/home/lxx/repo/datasets/lerobot") / repo_id
    
    print(f"Loading dataset from: {root}")
    
    # Initialize the dataset
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=[0]
    )
    
    print(f"Dataset loaded. Total frames: {len(dataset)}")
    
    # Collect all state and action data
    states = []
    actions = []
    
    print("Collecting state and action data...")
    for i in range(len(dataset)):
        item = dataset[i]
        states.append(item['observation.state'].numpy())
        actions.append(item['action'].numpy())
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(dataset)} frames")


    # Convert to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    
    # Create a 7x1 subplot lay)ut
    fig, axes = plt.subplots(7, 1, figsize=(14, 16))
    fig.suptitle('Comparison of State vs Action for Each Joint', fontsize=16)

    joint_labels = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Gripper']
    
    for i in range(7):
        ax = axes[i]
        time_axis = range(len(states))
        
        # Plot state in blue and action in red
        ax.plot(time_axis, states[:, i], label='State', color='blue', linewidth=0.8)
        ax.plot(time_axis, actions[:, i], label='Action', color='red', linewidth=0.8)
        
        ax.set_ylabel(joint_labels[i])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')
        # Add some space between subplots
        if i < 6:
            ax.set_xticklabels([])  # Hide x-axis labels for all but the last subplot
    
    # Set the x label only on the bottom subplot
    axes[-1].set_xlabel('Time Frame')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")


if __name__ == "__main__":
    visualize_state_action("state_action_comparison.png")