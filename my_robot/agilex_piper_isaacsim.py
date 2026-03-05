import sys
sys.path.append("./")

import numpy as np

from my_robot.base_robot import Robot

from controller.Piper_controller import PiperController
from sensor.Realsense_sensor import RealsenseSensor

from data.collect_any import CollectAny
from sensor.vision_sensor import VisionSensor
from controller.arm_controller import ArmController
from simpickgen.robot import BaseRobot
from simpickgen.gripper import BaseGripper
from simpickgen.camera import BaseCamera
class IsaacsimPiperController(ArmController):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.controller_type = "user_controller"
        self.controller = None
        self.gripper = None
    def set_up(self,robot:BaseRobot, gripper:BaseGripper):
        piper = robot
        gripper = gripper
        self.controller = piper
        self.gripper = gripper
    def reset(self, start_state):
        try:
            self.set_joint(start_state)
        except :
            print(f"reset error")
        return
    # 返回单位为米
    def get_state(self):
        state = {}
        eef = self.gripper.get_world_pose()
        joint = self.controller.get_joint_position()
        
        state["joint"] = np.array(joint[:-2]) 
        ee_position = eef[0]
        ee_orientation = eef[1]
        state["qpos"] = np.array([ee_position[0], ee_position[1], ee_position[2], ee_orientation[0], ee_orientation[1], ee_orientation[2]])
        state["gripper"] = np.array(joint[-2:])
        return state
    def set_position(self, position):
        pass
    def set_joint(self, joint):
        j1, j2, j3 ,j4, j5, j6 = joint 
        self.controller.set_joint_position(np.array([j1, j2, j3 ,j4, j5, j6]),joint_indices = [0,1,2,3,4,5])
    # The input gripper value is in the range [0, 1], representing the degree of opening.
    def set_gripper(self, gripper):
        self.controller.set_joint_position(gripper,joint_indices = [6,7])


class IsaacsimCameraSensor(VisionSensor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.camera = None
    def set_up(self,camera: BaseCamera):
        self.camera = camera
        
    def get_image(self):
        image = {}
        if "color" in self.collect_info:
            color_frame = self.camera.get_rgb()
            if not color_frame:
                raise RuntimeError("Failed to get color frame.")
            color_image = np.asanyarray(color_frame).copy()
            # BGR -> RGB
            image["color"] = color_image #[:,:,::-1]
        return image


CAMERA_SERIALS = {
    'head': '111',  # Replace with actual serial number
    'wrist': '111',   # Replace with actual serial number
}

# Define start position (in degrees)
START_POSITION_ANGLE_LEFT_ARM = [
    0.0,   # Joint 1
    0.0,    # Joint 2
    0.0,  # Joint 3
    0.,   # Joint 4
    0.0,  # Joint 5
    0.0,    # Joint 6
]

# Define start position (in degrees)
START_POSITION_ANGLE_RIGHT_ARM = [
    0,   # Joint 1
    0,    # Joint 2
    0,  # Joint 3
    0,   # Joint 4
    0,  # Joint 5
    0,    # Joint 6
]

condition = {
    "robot":"piper_single",
    "save_path": "./datasets/", 
    "task_name": "test", 
    "save_format": "hdf5", 
    "save_freq": 10, 
}


class IsaacsimPiperSingle(Robot):
    def __init__(self, condition=condition, move_check=True, start_episode=0):
        super().__init__(condition=condition, move_check=move_check, start_episode=start_episode)

        self.condition = condition
        self.controllers = {
            "arm":{
                "left_arm": IsaacsimPiperController("left_arm"),
            },
        }
        self.sensors = {
            "image":{
                "cam_head": IsaacsimCameraSensor("world_camera"),
                "cam_wrist": IsaacsimCameraSensor("robot_camera"),
            },
        }
    
    # ============== init ==============
    def reset(self):
        self.controllers["arm"]["left_arm"].reset(np.array(START_POSITION_ANGLE_LEFT_ARM))

    def set_up(self,robot:BaseRobot,gripper:BaseGripper, robot_camera:BaseCamera, world_camera:BaseCamera, ):
        super().set_up()

        self.controllers["arm"]["left_arm"].set_up(robot, gripper)
        self.sensors["image"]["cam_head"].set_up(world_camera)
        self.sensors["image"]["cam_wrist"].set_up(robot_camera)

        self.set_collect_type({"arm": ["joint","qpos","gripper"],
                               "image": ["color"]
                               })
        
        print("set up success!")
    def move(self, move_data: Dict[str, Any]) -> None:
        if "arm" in move_data:
            if "left_arm" in move_data["arm"]:
                arm_move_data = move_data["arm"]["left_arm"]
                if "joint" in arm_move_data:
                    joint_positions = arm_move_data["joint"]
                    self.controllers["arm"]["left_arm"].set_joint(joint_positions)
                if "gripper" in arm_move_data:
                    gripper_position = arm_move_data["gripper"]
                    self.controllers["arm"]["left_arm"].set_gripper(gripper_position)
    def get(self) -> Dict[str, Any]:
        data = {}
        # Get arm data
        arm_data = self.controllers["arm"]["left_arm"].get_state()
        data.update(arm_data)

        # Get image data
        image_data = self.sensors["image"]["cam_head"].get_image()
        data.update(image_data)

        wrist_image_data = self.sensors["image"]["cam_wrist"].get_image()
        # Prefix wrist image keys to avoid collision
        wrist_image_data_prefixed = {f"wrist_{key}": value for key, value in wrist_image_data.items()}
        data.update(wrist_image_data_prefixed)

        return data
    
if __name__=="__main__":
    import time
    robot = PiperSingle()
    robot.set_up()
    # collection test
    robot.reset()
    data_list = []
    for i in range(100):
        print(i)
        data = robot.get()
        robot.collect(data)
        time.sleep(0.1)
    robot.finish()
    
    # moving test
    move_data = {
        "arm":{
            "left_arm":{
            "qpos":[0.057, 0.0, 0.216, 0.0, 0.085, 0.0],
            "gripper":0.2,
            },
        },
    }
    robot.move(move_data)
    time.sleep(1)
    move_data = {
        "arm":{
            "left_arm":{
            "joint":[0.00, 0.0, 0.0, 0.0, 0.0, 0.0],
            "gripper":0.2,
            },
        },
    }
    robot.move(move_data)