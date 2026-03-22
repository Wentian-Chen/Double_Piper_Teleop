#!/usr/bin/env python3
# -*-coding:utf8-*-
# 键盘控制Piper机械臂关节角度 - 支持长按连续控制
# 需要安装: pip install piper_sdk pynput

import time
import threading
from pynput import keyboard
from piper_sdk import *

class PiperJointKeyboardController:
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
        
        # 当前关节角度 (弧度) [joint0, joint1, joint2, joint3, joint4, joint5, gripper]
        self.joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # 转换因子 (rad to encoder)
        self.factor = 57295.7795  # 1000*180/3.1415926
        
        # 步进值: 1度 = π/180 弧度 ≈ 0.0174533
        self.step_deg = 1.0  # 1度
        self.step_rad = 0.01745329252  # 1度对应的弧度
        
        # 关节限制范围 (度)
        self.limits_deg = [
            (-160, 160),    # joint 0
            (-100, 100),    # joint 1  
            (-140, 140),    # joint 2
            (-100, 100),    # joint 3
            (-120, 120),    # joint 4
            (-100, 100),    # joint 5
            (0, 70)                  # gripper (mm)
        ]
        
        # ===== 长按控制相关参数 =====
        self.LONG_PRESS_DELAY = 0.3  # 长按判定延迟（秒）
        self.CONTROL_FREQ = 15       # 长按连续控制频率（Hz）
        self.control_interval = 1.0 / self.CONTROL_FREQ  # 控制间隔
        
        # 当前按键状态
        self.pressed_keys = set()           # 当前按下的键集合
        self.long_press_active = {}         # 各键的长按状态
        self.key_press_time = {}            # 按键按下时间
        self.active_control = None          # 当前激活的控制线程
        self.control_lock = threading.Lock()
        
        # 控制标志
        self.running = True
        
        # 预设位姿 (度)
        self.poses_deg = {
            'zero': [0, 0, 0, 0, 0, 0, 0],
            'home': [0, 15, -15, 20, -15, 30, 0],
            'grab': [30, 20, -30, 50, -20, 15, 70],
        }
        
        # 按键映射配置 (键 -> [关节索引, 方向])
        self.joint_key_map = {
            'q': (0, +1), 'a': (0, -1),
            'w': (1, +1), 's': (1, -1),
            'e': (2, +1), 'd': (2, -1),
            'r': (3, +1), 'f': (3, -1),
            't': (4, +1), 'g': (4, -1),
            'y': (5, +1), 'h': (5, -1),
            'z': ('gripper', +5), 'x': ('gripper', -5),  # 夹爪用mm单位
        }
        
        self.print_help()
        self.print_status()
        
    def print_help(self):
        """打印控制说明"""
        print("\n" + "="*60)
        print("           机械臂关节键盘控制 (支持长按连续运动)")
        print("="*60)
        print("【关节控制 - 短按±1°精确控制，长按连续运动】")
        print("  Q - 关节0 +1°    A - 关节0 -1°   (底座旋转)")
        print("  W - 关节1 +1°    S - 关节1 -1°   (大臂)")
        print("  E - 关节2 +1°    D - 关节2 -1°   (小臂)")
        print("  R - 关节3 +1°    F - 关节3 -1°   (腕部旋转1)")
        print("  T - 关节4 +1°    G - 关节4 -1°   (腕部旋转2)")
        print("  Y - 关节5 +1°    H - 关节5 -1°   (腕部旋转3)")
        print("\n【夹爪控制】")
        print("  空格 - 夹爪 开/关切换")
        print("  Z - 夹爪 +5mm    X - 夹爪 -5mm  (长按连续)")
        print("\n【预设位姿】")
        print("  1 - 归零位姿    2 - Home位姿    3 - 抓取位姿")
        print("\n【其他功能】")
        print("  0 - 紧急归零    ESC - 退出程序")
        print("-"*60)
        print("提示: 按住 Q/A 等键超过0.3秒进入连续运动模式")
        print("      松开立即停止，短按保持精确±1°控制")
        print("="*60)
        
    def deg_to_rad(self, deg):
        """度转弧度"""
        return deg * 0.01745329252
    
    def rad_to_deg(self, rad):
        """弧度转度"""
        return rad * 57.295779513
    
    def clamp_deg(self, value, min_val, max_val):
        """限制数值在范围内"""
        return max(min_val, min(max_val, value))
    
    def update_joint(self, joint_idx, direction, step_mult=1.0):
        """更新指定关节角度"""
        step = self.step_deg * step_mult
        
        # 当前角度转度
        current_deg = self.rad_to_deg(self.joints[joint_idx])
        # 加减度数
        new_deg = current_deg + (direction * step)
        # 限制范围
        new_deg = self.clamp_deg(new_deg, self.limits_deg[joint_idx][0], self.limits_deg[joint_idx][1])
        # 转回弧度存储
        self.joints[joint_idx] = self.deg_to_rad(new_deg)
        
        return new_deg
    
    def update_gripper(self, delta_mm):
        """更新夹爪位置"""
        current_mm = self.joints[6] * 1000  # 转为mm
        new_mm = current_mm + delta_mm
        new_mm = self.clamp_deg(new_mm, 0, 70)
        self.joints[6] = new_mm / 1000  # 转回m存储
        return new_mm
    
    def set_pose(self, pose_name):
        """设置预设位姿"""
        if pose_name in self.poses_deg:
            pose_deg = self.poses_deg[pose_name]
            self.joints = [self.deg_to_rad(p) for p in pose_deg[:6]]
            self.joints.append(pose_deg[6] / 1000)  # gripper保持m单位
            print(f"\n[切换到位姿: {pose_name}]")
            self.print_status()
            self.send_command()
    
    def send_command(self):
        """发送控制指令到机械臂"""
        try:
            # 转换关节角度为编码器值
            joint_cmds = [round(self.joints[i] * self.factor) for i in range(6)]
            gripper_cmd = round(self.joints[6] * 1000 * 1000)
            
            # 发送运动指令
            self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
            self.piper.JointCtrl(*joint_cmds)
            self.piper.GripperCtrl(abs(gripper_cmd), 1000, 0x01, 0)
        except Exception as e:
            print(f"\n[发送指令错误: {e}]")
    
    def print_status(self):
        """打印当前关节状态"""
        deg_list = [self.rad_to_deg(j) for j in self.joints[:6]]
        gripper_mm = self.joints[6] * 1000
        
        print("\n当前关节角度:")
        print(f"  J0: {deg_list[0]:7.2f}°  J1: {deg_list[1]:7.2f}°  J2: {deg_list[2]:7.2f}°")
        print(f"  J3: {deg_list[3]:7.2f}°  J4: {deg_list[4]:7.2f}°  J5: {deg_list[5]:7.2f}°")
        print(f"  夹爪: {gripper_mm:6.2f}mm")
        print("-" * 40)
    
    def continuous_control_loop(self, key_char):
        """长按连续控制循环（在独立线程中运行）"""
        if key_char not in self.joint_key_map:
            return
        
        config = self.joint_key_map[key_char]
        
        # 标记长按激活
        with self.control_lock:
            self.long_press_active[key_char] = True
        
        print(f"\n[进入连续控制模式: {key_char.upper()}]")
        
        loop_count = 0
        while self.running and key_char in self.pressed_keys:
            if config[0] == 'gripper':
                # 夹爪控制
                new_val = self.update_gripper(config[1])
                if loop_count % 10 == 0:  # 每10次打印一次
                    print(f"\r[夹爪: {'+' if config[1]>0 else '-'}{abs(config[1])}mm → {new_val:.1f}mm]", end='')
            else:
                # 关节控制
                joint_idx, direction = config
                new_deg = self.update_joint(joint_idx, direction)
                if loop_count % 10 == 0:  # 每10次打印一次（约0.66秒）
                    action = "+" if direction > 0 else "-"
                    print(f"\r[关节{joint_idx}: {action}{self.step_deg:.0f}° → {new_deg:.1f}°]", end='')
            
            self.send_command()
            loop_count += 1
            time.sleep(self.control_interval)
        
        # 退出连续控制
        with self.control_lock:
            self.long_press_active[key_char] = False
        
        if loop_count > 0:
            print()  # 换行
            self.print_status()
    
    def start_long_press_timer(self, key_char):
        """启动长按检测定时器"""
        def check_long_press():
            time.sleep(self.LONG_PRESS_DELAY)
            # 检查按键是否仍然按住
            if key_char in self.pressed_keys and self.running:
                # 启动连续控制线程
                thread = threading.Thread(
                    target=self.continuous_control_loop, 
                    args=(key_char,),
                    daemon=True
                )
                thread.start()
                with self.control_lock:
                    self.active_control = thread
        
        timer = threading.Thread(target=check_long_press, daemon=True)
        timer.start()
        return timer
    
    def on_press(self, key):
        """键盘按下回调"""
        try:
            # 处理特殊键
            if key == keyboard.Key.space:
                if 'space' not in self.pressed_keys:
                    self.pressed_keys.add('space')
                    current_mm = self.joints[6] * 1000
                    if current_mm < 35:
                        self.joints[6] = 0.07  # 打开到70mm
                        print("\n[夹爪: 打开 → 70mm]")
                    else:
                        self.joints[6] = 0.0   # 关闭到0mm
                        print("\n[夹爪: 关闭 → 0mm]")
                    self.send_command()
                    self.print_status()
                return True
            
            if key == keyboard.Key.esc:
                print("\n[正在退出...]")
                self.running = False
                return False
            
            # 处理字符键
            if hasattr(key, 'char') and key.char:
                k = key.char.lower()
                
                # 防止重复触发（同一按键）
                if k in self.pressed_keys:
                    return True
                self.pressed_keys.add(k)
                
                # 预设位姿（瞬时触发，不支持长按）
                if key.char == '1':
                    self.set_pose('zero')
                    self.pressed_keys.discard(k)
                    return True
                elif key.char == '2':
                    self.set_pose('home')
                    self.pressed_keys.discard(k)
                    return True
                elif key.char == '3':
                    self.set_pose('grab')
                    self.pressed_keys.discard(k)
                    return True
                elif key.char == '0':
                    self.set_pose('zero')
                    print("[紧急归零]")
                    self.pressed_keys.discard(k)
                    return True
                
                # 关节/夹爪控制键
                if k in self.joint_key_map:
                    # 记录按键时间并启动长按检测
                    self.key_press_time[k] = time.time()
                    self.start_long_press_timer(k)
                    
                    # 立即执行一次短按动作（精确±1°）
                    config = self.joint_key_map[k]
                    if config[0] == 'gripper':
                        new_val = self.update_gripper(config[1])
                        print(f"\n[夹爪: {'+' if config[1]>0 else '-'}{abs(config[1]):.0f}mm → {new_val:.1f}mm]")
                    else:
                        joint_idx, direction = config
                        new_deg = self.update_joint(joint_idx, direction)
                        action = "+" if direction > 0 else "-"
                        print(f"\n[关节{joint_idx}: {action}{self.step_deg:.0f}° → {new_deg:.1f}°]")
                    
                    self.send_command()
                    self.print_status()
                    
        except AttributeError:
            pass
            
        return True
    
    def on_release(self, key):
        """键盘释放回调"""
        try:
            if key == keyboard.Key.space:
                self.pressed_keys.discard('space')
                return
            
            if hasattr(key, 'char') and key.char:
                k = key.char.lower()
                if k in self.pressed_keys:
                    self.pressed_keys.discard(k)
                    
                    # 检查是否是短按（未进入长按模式）
                    with self.control_lock:
                        if not self.long_press_active.get(k, False):
                            # 短按已在on_press中处理，这里只需清理状态
                            pass
                        else:
                            # 长按结束，清理状态
                            self.long_press_active[k] = False
                            
        except AttributeError:
            pass
    
    def run(self):
        """运行控制器"""
        print("控制循环已启动，请按键盘控制机械臂...")
        print(f"长按检测延迟: {self.LONG_PRESS_DELAY*1000:.0f}ms, 连续频率: {self.CONTROL_FREQ}Hz")
        
        # 启动键盘监听
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()
        
        print("程序已安全退出")


if __name__ == "__main__":
    try:
        controller = PiperJointKeyboardController("can0")
        controller.run()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()