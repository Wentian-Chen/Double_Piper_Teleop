import sys
import os

# Add SimPickGen path
sys.path.append("/home/charles/workspaces/SimPickGen/src")
sys.path.append("./")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

try:
    import isaacsim
    print("Isaac Sim module imported.")
except ImportError:
    print("Isaac Sim module not found. Please ensure Isaac Sim is installed and the PYTHONPATH is set correctly.")

import tyro
from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import math
import time
from multiprocessing import Manager, Event

from isaacsim.simulation_app import SimulationApp
from isaacsim.core.utils.types import ArticulationAction

from simpickgen import get_module_logger
from simpickgen.teleop_task import IsaacsimTeleopTask

from utils.data_handler import is_enter_pressed
from utils.worker import Worker
from data.collect_any import CollectAny
from controller.Piper_controller import PiperController

# Logger
logger = get_module_logger(__file__)

@dataclass
class TeleopTaskConfig:
    headless_mode: Optional[str] = None
    """To run headless, use one of [native, websocket]. WebRTC might not work."""
    epoches: int = 100
    """Number of epochs for teleoperation."""
    robot_file_path: str = "config/usd_config/piper/piper.yaml"
    """Robot configuration file to load."""

teleop_args = tyro.cli(TeleopTaskConfig)

# Initialize SimulationApp
simulation_app = SimulationApp(
    {
        "headless": teleop_args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

# Custom Teleop Task
class PiperTeleopTask(IsaacsimTeleopTask):
    def __init__(
            self, 
            teleop_args, 
            simulation_app: SimulationApp, 
            stage_units_in_meters: float = 1.0
        ) -> None:
        super().__init__(teleop_args, simulation_app, stage_units_in_meters)
        self.simulation_step_index = 0

# Configuration for CollectAny
condition = {
    "save_path": "./save/", 
    "task_name": "test", 
    "save_format": "hdf5", 
    "save_freq": 30,
    "collect_type": "teleop",
}

# --- Helper Functions ---
def action_transform(move_data: Dict[str, Any]) -> Dict[str, Any]:
    """ Transform the action from master arm to the slave arm."""
    if "joint" not in move_data:
        return move_data
        
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
    
    joints = np.array(move_data["joint"]).copy()
    
    # Calibration and transform
    joints[1] = joints[1] - math.radians(90)   
    joints[2] = joints[2] + math.radians(175)  
    
    for i in [1, 2, 4]:
        joints[i] = -joints[i]
        
    left_joints = [
        clamp(joints[i], joint_limits_rad[i][0], joint_limits_rad[i][1])
        for i in range(6)
    ]
    
    action = {
        "joint": np.array(left_joints),
        "gripper": move_data["gripper"],
    }
    return action

# --- Workers ---

class MasterWorker(Worker):
    def __init__(self, process_name: str, start_event, end_event):
        super().__init__(process_name, start_event, end_event)
        self.manager = Manager()
        self.data_buffer = self.manager.dict()

    def handler(self):
        # Get data from real robot
        raw_data = self.component.get()
        
        # Transform immediately for simulation usage
        transformed_data = action_transform(raw_data)
        
        # Store transformed data for Simulation
        if transformed_data is not None:
            # We copy key by key to update the manager dict
            for key, value in transformed_data.items():
                self.data_buffer[key] = value

    def component_init(self):
        self.component = PiperController(name="arm")
        self.component.set_up(can="can0") 
        self.component.set_collect_info(["joint","gripper"])

    def finish(self):
        # Safely stop robot
        if hasattr(self.component, 'controller'):
            pass
        return super().finish()

class DataWorker(Worker):
    def __init__(self, process_name: str, start_event, end_event, collect_data_buffer: Manager, episode_id=0, resume=False):
        super().__init__(process_name, start_event, end_event)
        self.collect_data_buffer = collect_data_buffer
        self.episode_id = episode_id
        self.resume = resume
        
    def component_init(self):
        self.collection = CollectAny(condition=condition, start_episode=self.episode_id, move_check=True, resume=self.resume)
    
    def handler(self):
        # Read from shared buffer
        data = dict(self.collect_data_buffer)
        if "controller" in data and "sensor" in data:
            # Reconstruct regular dicts from Manager dicts
            ctrl = dict(data["controller"])
            sensor = dict(data["sensor"])
            if ctrl and sensor:
                self.collection.collect(ctrl, sensor)
    
    def finish(self):
        self.collection.write()

# --- Main Logic ---

if __name__ == "__main__":
    teleop_task = PiperTeleopTask(teleop_args, simulation_app)
    teleop_task.initialize()
    # Ensure recorders are initialized
    teleop_task.initialize_data_recorder()

    # Wait for sim to start
    print("Waiting for Simulation to start...")
    while  teleop_task.simulation_app.is_running() and teleop_task.simulation_step_index < 60: 
        teleop_task.world.step(render=True)
        teleop_task.simulation_step_index += 1
        if teleop_task.simulation_step_index == 10:
             teleop_task.initialize_controller() 
    
    print("Simulation ready. Preparing workers...")

    # Shared resources
    manager = Manager()
    # Buffer for DataWorker: needs "controller" and "sensor" keys
    collect_data_buffer = manager.dict()
    collect_data_buffer["controller"] = manager.dict()
    collect_data_buffer["sensor"] = manager.dict()

    # Events
    start_event = Event()
    end_event = Event()
    
    # Initialize Workers
    master = MasterWorker("master_arm", start_event, end_event)
    data_worker = DataWorker("collect_data", start_event, end_event, collect_data_buffer, episode_id=0, resume=False)
    
    # Start Workers
    master.start()
    data_worker.start()

    print("Workers started. Press ENTER in the terminal to start recording episode.")
    
    # Main Loop
    try:
        is_recording = False
        
        while teleop_task.simulation_app.is_running():
            # 1. Step Physics
            teleop_task.world.step(render=True)
            
            # Handle Start/Stop Recording
            if is_enter_pressed():
                if not is_recording:
                    print(">>> Start Recording Episode...")
                    is_recording = True
                    start_event.set()
                else:
                    print("<<< Stop Recording Episode.")
                    is_recording = False
                    end_event.set()
                    break # Usually stop after one episode or reset logic needed

            # 2. Get Input from Master (Real Robot)
            try:
                master_data = dict(master.data_buffer) # Contains 'joint', 'gripper' (transformed)
            except Exception:
                master_data = {}
            
            if "joint" in master_data:
                # 3. Apply Action to Simulation
                joint_action = master_data["joint"]
                
                # Apply Arm joints to Sim
                if hasattr(teleop_task, 'isaacsim_robot') and teleop_task.isaacsim_robot:
                    # Construct action. 
                    action = ArticulationAction(joint_positions=joint_action)
                    teleop_task.isaacsim_robot.apply_action(action)
                    
            # 4. Get State from Simulation
            sim_data = teleop_task.get_data() # Returns dict with 'joint', 'ee_pos', 'qpos', 'images'...

            # 5. Update Data Buffer for Recording
            if is_recording:
                ctrl_dict = {}
                sensor_dict = {}
                
                # Controller Data (Master Arm)
                if "joint" in master_data:
                    ctrl_dict["slave_left_arm_joint"] = master_data["joint"]
                if "gripper" in master_data:
                    ctrl_dict["slave_left_arm_gripper"] = master_data["gripper"]
                
                # Sensor Data (Sim Arm)
                if "joint" in sim_data:
                    sensor_dict["slave_joint"] = sim_data["joint"]
                if "qpos" in sim_data:
                    sensor_dict["slave_qpos"] = sim_data["qpos"]
                
                # Images
                if "wrist_rgb" in sim_data:
                    sensor_dict["slave_cam_wrist"] = {"color": sim_data["wrist_rgb"]}
                if "table_rgb" in sim_data:
                    sensor_dict["slave_cam_head"] = {"color": sim_data["table_rgb"]}

                collect_data_buffer["controller"] = ctrl_dict
                collect_data_buffer["sensor"] = sensor_dict

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        master.stop()
        data_worker.stop()
        teleop_task.simulation_app.close()