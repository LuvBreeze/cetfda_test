# -*- coding: utf-8 -*-
# 特征提取模块
# 功能：从振动信号中提取时域、频域特征
# 依赖：numpy

import numpy as np

def extract_features(sig):
    """
    从振动信号中提取特征向量
    包含：时域统计特征、偏度、峭度、频域特征
    :param sig: 一维振动信号数组
    :return: 15维特征向量
    """
    feats = []  # 特征列表

    # =========================
    # 时域统计特征
    # =========================
    mean_val = np.mean(sig)  # 均值
    std_val = np.std(sig)    # 标准差
    rms = np.sqrt(np.mean(sig ** 2))  # 均方根

    # 添加基础时域特征
    feats.extend([
        mean_val,                # 均值
        std_val,                 # 标准差
        np.max(sig),             # 最大值
        np.min(sig),             # 最小值
        np.max(sig) - np.min(sig),  # 峰峰值
        rms,                     # 均方根
        np.mean(np.abs(sig)),    # 平均绝对偏差
        np.var(sig),             # 方差
        np.sum(sig ** 2)         # 能量
    ])

    # =========================
    # 形态特征
    # =========================
    # 偏度（分布不对称性）
    skewness = np.mean(sig ** 3) / (std_val ** 3 + 1e-10)  # 加小值避免除零
    # 峭度（分布陡峭程度）
    kurtosis = np.mean(sig ** 4) / (std_val ** 4 + 1e-10)

    feats.append(skewness)
    feats.append(kurtosis)

    # =========================
    # 频域特征（FFT）
    # =========================
    fft_vals = np.abs(np.fft.fft(sig))  # FFT幅度谱
    fft_half = fft_vals[:len(fft_vals)//2]  # 取前半部分（对称）

    # 添加频域特征
    feats.extend([
        np.mean(fft_half),       # 频域均值
        np.std(fft_half),        # 频域标准差
        np.max(fft_half),        # 频域最大值
        np.sum(fft_half)         # 频域能量
    ])

    return np.array(feats)  # 转换为numpy数组返回