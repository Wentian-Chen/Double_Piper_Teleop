import os
import json
import glob
import shutil

# 配置路径
base_path = "/home/charles/workspaces/Double_Piper_Teleop/datasets_rlds/piper_dataset/pick_banana_50"
dir_0 = os.path.join(base_path, "1.0.0")
dir_1 = os.path.join(base_path, "1.0.1")

def merge():
    # 1. 加载 JSON 信息
    info_0_path = os.path.join(dir_0, "dataset_info.json")
    info_1_path = os.path.join(dir_1, "dataset_info.json")
    
    with open(info_0_path, 'r') as f:
        info_0 = json.load(f)
    with open(info_1_path, 'r') as f:
        info_1 = json.load(f)

    # 2. 获取现有的 tfrecord 文件列表
    files_0 = sorted(glob.glob(os.path.join(dir_0, "*.tfrecord*")))
    files_1 = sorted(glob.glob(os.path.join(dir_1, "*.tfrecord*")))
    
    # 获取 1.0.1 的物理分片长度（如果 JSON 和物理文件不一致，以物理文件为准）
    # 这里我们假设物理文件数量与 JSON 里的 shardLengths 一致
    
    total_shards = len(files_0) + len(files_1)
    print(f"合并中... 1.0.0 已有 {len(files_0)} 个分片，1.0.1 有 {len(files_1)} 个分片。")
    print(f"总计分片数: {total_shards}")

    # 3. 处理 1.0.0 原有的文件（重命名 -of-00032 为新总数）
    new_files_0 = []
    for i, old_path in enumerate(files_0):
        name = os.path.basename(old_path)
        # 统一格式: piper_dataset-train.tfrecord-000XX-of-000YY
        new_name = name.split('-')[0] + "-train.tfrecord-" + f"{i:05d}-of-{total_shards:05d}"
        new_path = os.path.join(dir_0, new_name)
        if old_path != new_path:
            os.rename(old_path, new_path)
        new_files_0.append(new_path)

    # 4. 移动并重命名 1.0.1 的文件到 1.0.0
    start_idx = len(files_0)
    for i, old_path in enumerate(files_1):
        idx = start_idx + i
        new_name = os.path.basename(old_path).split('-')[0] + "-train.tfrecord-" + f"{idx:05d}-of-{total_shards:05d}"
        new_path = os.path.join(dir_0, new_name)
        shutil.move(old_path, new_path)
        print(f"已移动: {os.path.basename(old_path)} -> {new_name}")

    # 5. 更新 dataset_info.json
    # 合并 shardLengths
    combined_lengths = info_0['splits'][0]['shardLengths'] + info_1['splits'][0]['shardLengths']
    info_0['splits'][0]['shardLengths'] = combined_lengths
    
    # 合并 numBytes
    info_0['splits'][0]['numBytes'] = str(int(info_0['splits'][0]['numBytes']) + int(info_1['splits'][0]['numBytes']))
    
    # 确保版本是 1.0.0
    info_0['version'] = "1.0.0"

    # 保存新的 JSON
    with open(info_0_path, 'w') as f:
        json.dump(info_0, f, indent=2)
    
    print("元数据已更新。")
    print(f"成功合并到: {dir_0}")

if __name__ == "__main__":
    if os.path.exists(dir_0) and os.path.exists(dir_1):
        merge()
    else:
        print("错误: 找不到目录 1.0.0 或 1.0.1")
