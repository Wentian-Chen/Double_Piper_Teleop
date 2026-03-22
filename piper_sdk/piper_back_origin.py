import sys
sys.path.append("./")
import numpy as np
import time
from controller.Piper_controller import PiperController

if __name__=="__main__":
    controller = PiperController("test_piper")
    controller.set_up("can0")
    # print(controller.get_state())
    
    controller.set_gripper(1.0)
    controller.set_joint(np.array([0.04867723,-0.02370157,0.04152138, 0.01727876, 0.33063517, 0.]))
    time.sleep(2)
    controller.set_gripper(0.0)