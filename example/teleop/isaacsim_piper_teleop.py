import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.append("./")
try:
    import isaacsim
    print("Isaac Sim module imported.")
    print(f"Module path: {isaacsim.__file__}")
except ImportError:
    print("Isaac Sim module not found. Please ensure Isaac Sim is installed and the PYTHONPATH is set correctly.")

import tyro
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class TeleopTaskConfig:
    headless_mode: Optional[str] = None
    """To run headless, use one of [native, websocket]. WebRTC might not work."""
    epoches: int = 100
    """Number of epochs for teleoperation."""
    robot_file_path: str = "config/usd_config/piper/piper.yaml"
    """Robot configuration file to load."""

teleop_args = tyro.cli(TeleopTaskConfig)
from isaacsim.simulation_app import SimulationApp
from simpickgen import get_module_logger

import numpy as np
logger = get_module_logger(__file__)
simulation_app = SimulationApp(
    {
        "headless": teleop_args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)
from simpickgen.teleop_task import IsaacsimTeleopTask
from isaacsim.core.utils.types import ArticulationAction
from simpickgen.robot import BaseRobot
from simpickgen.gripper import BaseGripper
from simpickgen.camera import BaseCamera
class PiperTeleopTask(IsaacsimTeleopTask):
    def __init__(
            self, 
            teleop_args, 
            simulation_app: SimulationApp, 
            stage_units_in_meters: float = 1.0
        ) -> None:
        super().__init__(teleop_args, simulation_app, stage_units_in_meters)
        self.simulation_step_index = 0
#-------------------------------------------------------
from re import S, T

from h5py._hl.dataset import sel


import time
from multiprocessing import Manager, Event

from utils.data_handler import is_enter_pressed
from utils.time_scheduler import TimeScheduler
from utils.worker import Worker
from data.collect_any import CollectAny
from controller.drAloha_controller import DrAlohaController
from controller.Piper_controller import PiperController
from my_robot.agilex_piper_isaacsim import IsaacsimPiperSingle
from my_robot.agilex_piper_single_base import PiperSingle
import math
from typing import Dict, Any


condition = {
    "save_path": "./save/", 
    "task_name": "test", 
    "save_format": "hdf5", 
    "save_freq": 30,
    "collect_type": "teleop",
}


class MasterWorker(Worker):
    def __init__(self, process_name: str, start_event, end_event):
        super().__init__(process_name, start_event, end_event)
        self.manager = Manager()
        self.data_buffer = self.manager.dict()

    def handler(self):
            
        data = self.component.get()
        for key, value in data.items():
            self.data_buffer[key] = value

    def component_init(self):
        self.component = PiperController(name="arm")
        self.component.set_up(can="can0")
        self.component.set_collect_info(["joint","gripper"])

    def finish(self):
        for i in range(1,7):
            self.component.controller.estop(i)
        return super().finish()
class SlaveWorker(Worker):
    def __init__(self, process_name: str, start_event, end_event, move_data_buffer: Manager,robot: BaseRobot, gripper: BaseGripper, robot_camera: BaseCamera, world_camera: BaseCamera):
        self.isaacsim_robot = robot
        self.isaacsim_gripper = gripper
        self.isaacsim_robot_camera = robot_camera
        self.isaacsim_world_camera = world_camera
        super().__init__(process_name, start_event, end_event)
        self.move_data_buffer = move_data_buffer
        self.manager = Manager()
        self.data_buffer = self.manager.dict()
    
    def handler(self):
        move_data = dict(self.move_data_buffer)
     
        self.component.move({"arm": 
                                {
                                    "left_arm": move_data
                                }
                            })

        data = self.component.get()

        self.data_buffer["controller"] = self.manager.dict()
        self.data_buffer["sensor"] = self.manager.dict()
        # self.data_buffer["controller"]["master_left_arm"] = self.manager.dict()

        for key, value in data[0].items():
            self.data_buffer["controller"]["slave_"+key] = value
        
        for key, value in data[1].items():
            self.data_buffer["sensor"]["slave_"+key] = value

        # for key, value in move_data.items():
        #     self.data_buffer["controller"]["master_left_arm"][key] = value
    
    def component_init(self):
        self.component = IsaacsimPiperSingle()
        self.component.set_up(self.isaacsim_robot, self.isaacsim_gripper, self.isaacsim_robot_camera, self.isaacsim_world_camera)
        self.component.reset()  


class DataWorker(Worker):
    def __init__(self, process_name: str, start_event, end_event, collect_data_buffer: Manager, episode_id=0, resume=False):
        super().__init__(process_name, start_event, end_event)
        self.collect_data_buffer = collect_data_buffer
        self.episode_id = episode_id
        self.resume = resume
    def component_init(self):
        self.collection = CollectAny(condition=condition, start_episode=self.episode_id, move_check=True, resume=self.resume)
    
    def handler(self):
        data = dict(self.collect_data_buffer)
        self.collection.collect(data["controller"], data["sensor"])
    
    def finish(self):
        self.collection.write()

if __name__ == "__main__":
    import os
    teleop_task = PiperTeleopTask(teleop_args, simulation_app)
    teleop_task.initialize()
    os.environ["INFO_LEVEL"] = "INFO"
    num_episode = 10
    avg_collect_time = 0
    while teleop_task.simulation_app.is_running():
        teleop_task.world.step(render=True)
        teleop_task.simulation_step_index = teleop_task.world.current_time_step_index
        if not teleop_task.world.is_playing():
            if i % 50 == 0:
                logger.info("**** Click Play to start simulation *****")
            i += 1
            continue
        if teleop_task.simulation_step_index < 10:
                teleop_task.initialize_controller()
        # skip first 20 steps to allow settling
        if teleop_task.simulation_step_index < 20:
            continue
        for i in range(num_episode):
            is_start = False

            start_event, end_event = Event(), Event()
            
            master = MasterWorker("master_arm", start_event, end_event)
            slave = SlaveWorker(
                "slave_arm", 
                start_event, 
                end_event, 
                master.data_buffer,
                teleop_task.robot,
                teleop_task.gripper,
                teleop_task.robot_camera,
                teleop_task.world_camera
            )
            data = DataWorker("collect_data", start_event, end_event, slave.data_buffer, episode_id=i, resume=True)

            time_scheduler = TimeScheduler(work_events=[master.forward_event], time_freq=30, end_events=[data.next_event])
            
            master.next_to(slave)
            slave.next_to(data)

            master.start()
            slave.start()
            data.start()

            while not is_start:
                time.sleep(0.01)
                if is_enter_pressed():
                    is_start = True
                    start_event.set()
                    
                else:
                    time.sleep(1)

            time_scheduler.start()
            while is_start:
                time.sleep(0.01)
                if is_enter_pressed():
                    end_event.set()  
                    time_scheduler.stop()  
                    is_start = False

            # 给数据写入一定时间缓冲
            time.sleep(1)

            master.stop()
            slave.stop()
            data.stop()
    teleop_task.simulation_app.close()