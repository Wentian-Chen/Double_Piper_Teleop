#!/usr/bin/env python3
# -*-coding:utf8-*-
# 键盘控制Piper机械臂末端位置
# 需要安装: pip install piper_sdk pynput

import time
import threading
from pynput import keyboard
from piper_sdk import *

class PiperKeyboardController:
    def __init__(self, can_channel="can0"):
        # 初始化机械臂接口
        self.piper = C_PiperInterface_V2(can_channel)
        self.piper.ConnectPort()
        
        # 等待机械臂使能
        print("正在使能机械臂...")
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        print("机械臂已使能")
        
        # 初始化夹爪
        self.piper.GripperCtrl(0, 1000, 0x01, 0)
        
        # 当前末端位置 (X, Y, Z, RX, RY, RZ, Gripper)
        #  [57.0, 0.0, 215.0, 0, 85.0, 0, 0]
        self.position = [57.0, 0.0, 215.0, 0.0, 85.0, 0.0, 0.0] 
        
        # 步进值 (毫米/度)
        self.step = {
            'position': 5.0,    # XYZ移动步进 5mm
            'rotation': 5.0,    # 旋转步进 5度
            'gripper': 10.0     # 夹爪步进
        }
        
        # 运动限制范围
        self.limits = {
            'X': (0, 400),
            'Y': (-200, 200),
            'Z': (100, 400),
            'RX': (-180, 180),
            'RY': (0, 180),
            'RZ': (-180, 180),
            'Gripper': (0, 70)
        }
        
        # 控制标志
        self.running = True
        self.current_keys = set()
        
        # 因子
        self.factor = 1000 # 1000
        
        print("\n=== 键盘控制说明 ===")
        print("位置控制:")
        print("  W/S - X轴 +/-")
        print("  A/D - Y轴 +/-")
        print("  Q/E - Z轴 上/下")
        print("旋转控制:")
        print("  I/K - RX +/-")
        print("  J/L - RY +/-")
        print("  U/O - RZ +/-")
        print("夹爪控制:")
        print("  空格 - 夹爪开/关切换")
        print("  +/- - 微调夹爪位置")
        print("其他:")
        print("  ↑/↓ - 调整移动速度")
        print("  R - 重置到初始位置")
        print("  ESC - 退出程序")
        print("==================\n")
        
    def clamp(self, value, min_val, max_val):
        """限制数值在范围内"""
        return max(min_val, min(max_val, value))
    
    def update_position(self):
        """根据当前按下的键更新位置"""
        step_pos = self.step['position']
        step_rot = self.step['rotation']
        
        # XYZ平移
        if keyboard.Key.up in self.current_keys or 'w' in self.current_keys:
            self.position[0] += step_pos  # X+
        if keyboard.Key.down in self.current_keys or 's' in self.current_keys:
            self.position[0] -= step_pos  # X-
        if 'a' in self.current_keys:
            self.position[1] += step_pos  # Y+
        if 'd' in self.current_keys:
            self.position[1] -= step_pos  # Y-
        if 'q' in self.current_keys:
            self.position[2] += step_pos  # Z+
        if 'e' in self.current_keys:
            self.position[2] -= step_pos  # Z-
            
        # 旋转
        if 'i' in self.current_keys:
            self.position[3] += step_rot  # RX+
        if 'k' in self.current_keys:
            self.position[3] -= step_rot  # RX-
        if 'j' in self.current_keys:
            self.position[4] += step_rot  # RY+
        if 'l' in self.current_keys:
            self.position[4] -= step_rot  # RY-
        if 'u' in self.current_keys:
            self.position[5] += step_rot  # RZ+
        if 'o' in self.current_keys:
            self.position[5] -= step_rot  # RZ-
            
        # 夹爪微调
        if '+' in self.current_keys or '=' in self.current_keys:
            self.position[6] += self.step['gripper']
        if '-' in self.current_keys:
            self.position[6] -= self.step['gripper']
            
        # 限制范围
        self.position[0] = self.clamp(self.position[0], *self.limits['X'])
        self.position[1] = self.clamp(self.position[1], *self.limits['Y'])
        self.position[2] = self.clamp(self.position[2], *self.limits['Z'])
        self.position[3] = self.clamp(self.position[3], *self.limits['RX'])
        self.position[4] = self.clamp(self.position[4], *self.limits['RY'])
        self.position[5] = self.clamp(self.position[5], *self.limits['RZ'])
        self.position[6] = self.clamp(self.position[6], *self.limits['Gripper'])
    
    def send_command(self):
        """发送控制指令到机械臂"""
        X = round(self.position[0] * self.factor)
        Y = round(self.position[1] * self.factor)
        Z = round(self.position[2] * self.factor)
        RX = round(self.position[3] * self.factor)
        RY = round(self.position[4] * self.factor)
        RZ = round(self.position[5] * self.factor)
        gripper = round(self.position[6] * self.factor)
        
        # 发送运动指令
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        self.piper.GripperCtrl(abs(gripper), 1000, 0x01, 0)
    
    def print_status(self):
        """打印当前状态"""
        print(f"\rPos: X={self.position[0]:6.1f} Y={self.position[1]:6.1f} Z={self.position[2]:6.1f} | "
              f"Rot: RX={self.position[3]:6.1f} RY={self.position[4]:6.1f} RZ={self.position[5]:6.1f} | "
              f"Gripper={self.position[6]:5.1f} | Step={self.step['position']:.1f}mm", end="", flush=True)
    
    def on_press(self, key):
        """键盘按下回调"""
        try:
            # 普通字符键
            if hasattr(key, 'char') and key.char:
                self.current_keys.add(key.char.lower())
                
                # 特殊功能键
                if key.char.lower() == 'r':
                    # 重置位置
                    self.position = [57.0, 0.0, 215.0, 0.0, 85.0, 0.0, 0.0]
                    print("\n[重置到初始位置]")
                    
                elif key.char in ['+', '=']:
                    self.current_keys.add('+')
                elif key.char == '-':
                    self.current_keys.add('-')
                    
            else:
                # 特殊键
                self.current_keys.add(key)
                
                # 调整速度
                if key == keyboard.Key.up:
                    self.step['position'] = min(50.0, self.step['position'] + 1.0)
                    self.step['rotation'] = min(20.0, self.step['rotation'] + 1.0)
                    print(f"\n[速度增加: {self.step['position']:.1f}mm]")
                elif key == keyboard.Key.down:
                    self.step['position'] = max(1.0, self.step['position'] - 1.0)
                    self.step['rotation'] = max(1.0, self.step['rotation'] - 1.0)
                    print(f"\n[速度降低: {self.step['position']:.1f}mm]")
                    
                # 夹爪开关
                elif key == keyboard.Key.space:
                    if self.position[6] < 35:  # 如果夹爪关闭，则打开
                        self.position[6] = 70.0
                        print("\n[夹爪打开]")
                    else:  # 否则关闭
                        self.position[6] = 0.0
                        print("\n[夹爪关闭]")
                        
                # 退出
                elif key == keyboard.Key.esc:
                    print("\n[正在退出...]")
                    self.running = False
                    return False
                    
        except AttributeError:
            pass
    
    def on_release(self, key):
        """键盘释放回调"""
        try:
            if hasattr(key, 'char') and key.char:
                self.current_keys.discard(key.char.lower())
                self.current_keys.discard(key.char)
            else:
                self.current_keys.discard(key)
        except:
            pass
    
    def control_loop(self):
        """主控制循环"""
        while self.running:
            self.update_position()
            self.send_command()
            self.print_status()
            time.sleep(0.02)  # 50Hz控制频率
    
    def run(self):
        """运行控制器"""
        # 启动控制线程
        control_thread = threading.Thread(target=self.control_loop)
        control_thread.daemon = True
        control_thread.start()
        
        # 启动键盘监听
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()
        
        self.running = False
        control_thread.join(timeout=1.0)
        print("\n程序已退出")


if __name__ == "__main__":
    try:
        controller = PiperKeyboardController("can0")
        controller.run()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()