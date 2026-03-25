#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HDF5 序列数据筛选工具

根据相邻时间步的 joint 和 gripper 数据的差异（L2范数）进行筛选，
保留差异大于阈值的时间步，并将筛选后的所有文件输出到统一文件夹，
按顺序编号。
"""

import os
import glob
import logging
from typing import List, Optional

import h5py
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def filter_hdf5_sequence(src_path: str, dst_path: str, threshold: float) -> None:
    """
    读取一个 HDF5 文件，根据 joint 和 gripper 的差异阈值筛选时间步，
    并将筛选后的数据写入目标文件。

    参数:
        src_path: 源 HDF5 文件路径
        dst_path: 目标 HDF5 文件路径
        threshold: 差异阈值（L2范数），仅当距离大于该值时保留后一个时间步
    """
    if not os.path.isfile(src_path):
        logger.error(f"源文件不存在: {src_path}")
        return

    # 确保目标目录存在
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    try:
        with h5py.File(src_path, 'r') as src:
            # 读取 joint 和 gripper 数据（假设它们都存在）
            try:
                joint = src['left_arm/joint'][:]          # shape: (N, 6)
                gripper = src['left_arm/gripper'][:]      # shape: (N,)
            except KeyError as e:
                logger.error(f"文件 {src_path} 缺少必要的数据集: {e}")
                return

            N = joint.shape[0]
            if N == 0:
                logger.warning(f"文件 {src_path} 无数据，跳过")
                return

            # 筛选保留的索引
            keep_indices = [0]      # 第一个时间步总是保留
            current_idx = 0
            for i in range(1, N):
                # 构建当前点和候选点的组合向量（6维关节 + 1维夹爪）
                vec_curr = np.concatenate([joint[current_idx], [gripper[current_idx]]])
                vec_i = np.concatenate([joint[i], [gripper[i]]])
                dist = np.linalg.norm(vec_i - vec_curr)
                if dist > threshold:
                    keep_indices.append(i)
                    current_idx = i

            logger.info(f"文件 {src_path} 原始帧数: {N}, 保留帧数: {len(keep_indices)}")

            # 创建新文件并递归复制组/数据集（按索引切片）
            with h5py.File(dst_path, 'w') as dst:
                _copy_group_with_slicing(src, dst, keep_indices, N)

    except Exception as e:
        logger.exception(f"处理文件 {src_path} 时发生错误: {e}")
        raise


def _copy_group_with_slicing(src_group: h5py.Group, dst_group: h5py.Group,
                             indices: List[int], original_len: int) -> None:
    """
    递归复制 HDF5 组及其内容，对第一维长度为 original_len 的数据集应用索引切片。

    参数:
        src_group: 源组对象
        dst_group: 目标组对象
        indices: 要保留的索引列表
        original_len: 原始时间步长度，用于判断数据集是否需要切片
    """
    for name, item in src_group.items():
        if isinstance(item, h5py.Dataset):
            # 读取完整数据
            data = item[()]
            # 如果数据集的第一维长度等于 original_len，则按索引切片
            if data.shape[0] == original_len:
                sliced_data = data[indices]
            else:
                sliced_data = data
            # 创建数据集并写入数据
            ds = dst_group.create_dataset(name, data=sliced_data, dtype=item.dtype)
            # 复制属性
            for key, value in item.attrs.items():
                ds.attrs[key] = value
        elif isinstance(item, h5py.Group):
            sub_group = dst_group.create_group(name)
            # 复制组属性
            for key, value in item.attrs.items():
                sub_group.attrs[key] = value
            # 递归处理子组
            _copy_group_with_slicing(item, sub_group, indices, original_len)
        else:
            # 理论上 h5py 只有 Dataset 和 Group，此处作为安全兜底
            logger.warning(f"未知类型 {type(item)}，跳过: {name}")


def collect_hdf5_files(folders: List[str]) -> List[str]:
    """
    收集多个文件夹中的所有 HDF5 文件（.hdf5 或 .h5）。

    参数:
        folders: 文件夹路径列表

    返回:
        排序后的文件路径列表
    """
    files = []
    for folder in folders:
        if not os.path.isdir(folder):
            logger.warning(f"文件夹不存在，跳过: {folder}")
            continue
        h5_files = glob.glob(os.path.join(folder, "*.hdf5")) + \
                   glob.glob(os.path.join(folder, "*.h5"))
        files.extend(h5_files)
    # 按路径排序，保证处理顺序可预测
    files.sort()
    return files


def main(folders: List[str], threshold: float, output_dir: str) -> None:
    """
    主函数：处理多个文件夹中的 HDF5 文件，输出到统一文件夹，并顺序编号。

    参数:
        folders: 输入文件夹路径列表
        threshold: 差异阈值
        output_dir: 输出文件夹路径（将存放所有处理后的文件）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有 HDF5 文件
    all_files = collect_hdf5_files(folders)
    if not all_files:
        logger.warning("未找到任何 HDF5 文件")
        return

    logger.info(f"共找到 {len(all_files)} 个 HDF5 文件")

    # 逐个处理，按顺序编号
    for idx, src_file in enumerate(all_files):
        # 生成编号：4位数字，不足补零，例如 0000.hdf5
        dst_filename = f"{idx:04d}.hdf5"
        dst_path = os.path.join(output_dir, dst_filename)
        logger.info(f"处理 [{idx+1}/{len(all_files)}]: {src_file} -> {dst_path}")
        filter_hdf5_sequence(src_file, dst_path, threshold)


if __name__ == "__main__":
    # ==================== 配置区 ====================
    # 请根据实际需求修改以下参数
    INPUT_FOLDERS = [
        "/media/lxx/4A21-0000/Data/origin_data/pick_banana/dataset_1",
        "/media/lxx/4A21-0000/Data/origin_data/pick_banana/dataset_2",
        "/media/lxx/4A21-0000/Data/origin_data/pick_banana/dataset_3"
    ]
    THRESHOLD = 0.1          # 差异阈值（根据实际数据调整）
    OUTPUT_DIR = "/media/lxx/4A21-0000/Data/origin_data/pick_banana_new"  # 所有输出文件的存放目录
    # ===============================================

    main(INPUT_FOLDERS, THRESHOLD, OUTPUT_DIR)