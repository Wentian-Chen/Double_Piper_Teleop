from piper_sdk import *
import time
# 测试代码
if __name__ == "__main__":
    piper = C_PiperInterface_V2()
    piper.ConnectPort()
    piper.MotionCtrl_1(0x01,0,0)
    print("机械臂已重置，等到复位!!!!")
    time.sleep(2)
    piper.MotionCtrl_1(0x02,0,0)#恢复
    print("机械臂已恢复，等到使能!!!!")
    # 强制待机
    piper.ModeCtrl(ctrl_mode=0x00,move_mode=0x01) # ctrl_mode=0x00: 待机模式  move_mode=0x01:关节控制模式
    time.sleep(0.1)
    print("已切换成待机模式")
    print(piper.GetArmStatus())
    time.sleep(0.1)
    # 使能机械臂
    while( not piper.EnablePiper()):
        time.sleep(0.01)
    piper.ModeCtrl(ctrl_mode=0x01,move_mode=0x01) # ctrl_mode=0x01: CAN模式  move_mode=0x01:关节控制模式
    time.sleep(0.1)
    print(piper.GetArmStatus())
    print("默认切换成CAN模式和MOVE_J----使能成功!!!!")
    print("关节使能状态",piper.GetArmEnableStatus())    