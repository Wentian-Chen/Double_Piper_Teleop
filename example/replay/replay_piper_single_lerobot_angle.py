import time
import torch
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from piper_sdk import C_PiperInterface_V2

# Piper配置参数
FACTOR = 57295.7795  # 1000*180/3.1415926
CAN_PORT = "can0"
CONTROL_PERIOD = 0.05  # 控制周期  50ms
SPEED = 50             # 控制速度

def init_piper():
    """初始化并启用Piper机械臂"""
    piper = C_PiperInterface_V2(CAN_PORT)
    piper.ConnectPort()
    
    # 等待使能成功
    while not piper.EnablePiper():
        time.sleep(0.01)
    
    piper.MotionCtrl_2(0x01, 0x01, SPEED, 0x00)
    piper.GripperCtrl(0, 1000, 0x01, 0)
    
    print("Piper机械臂初始化完成")
    
    return piper

def send_action_to_piper(piper, action):
    """
    将LeRobot的action发送到Piper机械臂
    
    action格式: [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, gripper]
    单位: 弧度 (前6个关节), 夹爪行程(0-1或具体数值，需根据数据集确定)
    """
    # 解析动作（假设action是tensor或numpy数组）
    if isinstance(action, torch.Tensor):
        action = action.squeeze().cpu().numpy()
    
    # 前6个是关节角度，最后一个是夹爪
    joint_positions = action[:6]
    gripper_pos = action[6] if len(action) > 6 else 0.0
    
    # 转换关节角度为Piper格式
    joint_0 = round(joint_positions[0] * FACTOR)
    joint_1 = round(joint_positions[1] * FACTOR)
    joint_2 = round(joint_positions[2] * FACTOR)
    joint_3 = round(joint_positions[3] * FACTOR)
    joint_4 = round(joint_positions[4] * FACTOR)
    joint_5 = round(joint_positions[5] * FACTOR)
    
    # 夹爪转换（根据你的原始代码，夹爪乘以1000*1000）
    # 注意：需要根据实际数据集的gripper范围调整
    gripper_value = round(abs(gripper_pos) * 70 * 1000)
    
    # 发送控制指令
    piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
    piper.GripperCtrl(gripper_value, 1000, 0x01, 0)
    
    # 可选：打印状态
    print("夹爪开合",gripper_pos)
    # status = piper.GetArmStatus()
    # print(f"Sent: joints={joint_positions}, gripper={gripper_pos}")

@parser.wrap()
def main(cfg: DatasetConfig):
    """
    从LeRobot数据集读取动作并回放到Piper机械臂
    """
    # 1. 加载数据集
    dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        episodes=cfg.episodes,
        root=cfg.root,
        video_backend=cfg.video_backend,
    )

    print(f"数据集信息:")
    print(f"  - 选定片段: {dataset.episodes}")
    print(f"  - 片段数量: {dataset.num_episodes}")
    print(f"  - 总帧数: {dataset.num_frames}")
    print(f"  - 动作维度: {dataset.features['action']['shape']}")

    # 2. 初始化Piper
    piper = init_piper()
    
    # 等待用户确认开始
    input("机械臂已就绪,按Enter开始回放动作序列...")
    
    # 3. 创建DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,  # 改为0避免多进程问题
        batch_size=1,
        shuffle=False,
    )
    
    # 4. 回放循环
    print("开始回放...")
    try:
        for idx, batch in enumerate(dataloader):
            action = batch['action']  # shape: [1, action_dim]
            
            # 打印信息（每50帧）
            # if idx % 50 == 0:
            #     print(f"Frame {idx}/{dataset.num_frames}: action={action.squeeze().tolist()}")
            
            # 发送到机械臂
            send_action_to_piper(piper, action)
            
            # 保持控制周期
            time.sleep(CONTROL_PERIOD) 
    except KeyboardInterrupt:
        print("\n用户中断回放")
    finally:
        input("回放结束,按Enter机械臂回到安全位置...")
        send_action_to_piper(piper, [0, 0, 0, 0, 0, 0, 0])
        time.sleep(1)

if __name__ == "__main__":
    main()