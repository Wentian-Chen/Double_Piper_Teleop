import sys
sys.path.append("./")
import numpy as np
import time
from controller.Piper_controller import PiperController

if __name__=="__main__":
    controller = PiperController("test_piper")
    controller.set_up("can0")
    controller.set_joint(np.array([0.0, 0.85220935, -0.68542569, 0., 0.78588684, -0.05256932,]))    # 位置A
    controller.set_gripper(1.0)
    time.sleep(3)
    print("pointA gripper:", controller.get_state()['gripper'])

    gripper_value = 0
    controller.set_joint(np.array([0.04867723,-0.02370157,0.04152138, 0.01727876, 0.33063517, 0.])) # 位置B
    controller.set_gripper(gripper_value)
    while True:
        if abs(controller.get_state()['gripper'] - gripper_value) < 0.01:
            break
        time.sleep(0.1)
    print("pointB gripper:", controller.get_state()['gripper'])