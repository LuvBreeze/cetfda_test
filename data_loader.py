# -*- coding: utf-8 -*-
# 数据加载模块
# 功能：加载CWRU轴承数据集，提取特征和标签
# 依赖：numpy, scipy.io, os, feature

import numpy as np
import scipy.io as sio
import os
from feature import extract_features

def load_data(base_path):
    """
    加载凯斯西储大学（CWRU）轴承数据集
    自动递归读取所有子文件夹中的.mat文件，提取DE/FE信号并生成特征样本
    :param base_path: 数据集根目录
    :return: 特征矩阵X，标签数组y
    """
    X = []  # 特征矩阵
    y = []  # 标签数组

    # =========================
    # 内部函数：处理单个.mat文件
    # =========================
    def process_file(filepath, label, signal_type='DE'):
        """
        处理单个mat文件，提取指定类型的信号并切分生成样本
        :param filepath: mat文件路径
        :param label: 样本标签
        :param signal_type: 信号类型（DE:驱动端，FE:风扇端）
        """
        try:
            # 加载mat文件
            data = sio.loadmat(filepath)
        except:
            # 加载失败则跳过
            return

        sig = None
        # 查找包含指定信号类型的字段
        for key in data.keys():
            if signal_type in key:
                sig = data[key].flatten()  # 展平为一维数组
                break

        # 信号长度不足则跳过
        if sig is None or len(sig) < 2048:
            return

        # 滑动窗口切分信号（步长1024，窗口2048）
        for i in range(0, len(sig) - 2048, 1024):
            seg = sig[i:i+2048]  # 截取2048长度的信号段
            feat = extract_features(seg)  # 提取特征
            X.append(feat)  # 添加特征
            y.append(label)  # 添加标签

    # =========================
    # 内部函数：遍历文件夹
    # =========================
    def traverse(folder, label, signal_type='DE'):
        """
        递归遍历文件夹，处理所有.mat文件
        :param folder: 文件夹路径
        :param label: 样本标签
        :param signal_type: 信号类型
        """
        if not os.path.exists(folder):
            return
        # 遍历文件夹中的所有文件
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith('.mat'):
                    # 处理mat文件
                    process_file(os.path.join(root, f), label, signal_type)

    # =========================
    # 数据集路径与标签映射
    # =========================
    datasets = [
        ('Normal Baseline', 0, 'DE'),                      # 正常状态
        ('12k Drive End Bearing Fault Data/Ball', 1, 'DE'),# 12k驱动端球故障
        ('12k Drive End Bearing Fault Data/Inner Race', 2, 'DE'),# 12k驱动端内圈故障
        ('12k Drive End Bearing Fault Data/Outer Race', 3, 'DE'),# 12k驱动端外圈故障
        ('48k Drive End Bearing Fault Data/Ball', 4, 'DE'),# 48k驱动端球故障
        ('48k Drive End Bearing Fault Data/Inner Race', 5, 'DE'),# 48k驱动端内圈故障
        ('48k Drive End Bearing Fault Data/Outer Race', 6, 'DE'),# 48k驱动端外圈故障
        ('12k Fan End Bearing Fault Data/Ball', 7, 'FE'),  # 12k风扇端球故障
        ('12k Fan End Bearing Fault Data/Inner Race', 8, 'FE'),# 12k风扇端内圈故障
        ('12k Fan End Bearing Fault Data/Outer Race', 9, 'FE'),# 12k风扇端外圈故障
    ]

    # =========================
    # 加载所有数据集
    # =========================
    for folder_rel, label, sig_type in datasets:
        folder_path = os.path.join(base_path, folder_rel)
        print('读取:', folder_path)
        traverse(folder_path, label, sig_type)

    # =========================
    # 打印加载信息
    # =========================
    print('\n===== 数据加载完成 =====')
    print('总样本数:', len(y))
    if len(y) > 0:
        print('标签种类:', np.unique(y))
        print('每类数量:', np.bincount(y))
    print('========================\n')

    return np.array(X), np.array(y)