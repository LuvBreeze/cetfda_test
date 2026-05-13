# -*- coding: utf-8 -*-
# 边缘端检测模块
# 功能：模型加载、热更新、故障预测、信号生成/加载
# 依赖：os, random, numpy, time, scipy.io, threading, model, feature, trainer

import os
import random

import numpy as np
import time
import scipy.io
import threading

from model import OptimizedELM
from feature import extract_features
from cloud.trainer import train_model

# =========================
# 路径配置（动态计算）
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前文件目录
MODEL_PATH = os.path.join(BASE_DIR, '../models/model.npy')  # 模型文件路径
VERSION_PATH = os.path.join(BASE_DIR, '../models/version.txt')  # 版本文件路径
DATA_PATH = os.path.join(BASE_DIR, '../CRWU')  # CWRU数据集路径

# =========================
# 模型初始化
# =========================
# 创建模型目录（不存在则创建）
models_dir = os.path.dirname(MODEL_PATH)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# 初始化ELM模型
elm = OptimizedELM()

# 加载或训练初始模型
if not os.path.exists(MODEL_PATH):
    print("[警告] 模型不存在，正在训练初始模型...", flush=True)
    try:
        # 检查数据集是否存在
        if os.path.exists(DATA_PATH) and any(os.scandir(DATA_PATH)):
            # 训练初始模型
            elm, scaler = train_model(data_path=DATA_PATH, save_path=MODEL_PATH)
            mean = scaler.mean_  # 标准化均值
            scale = scaler.scale_  # 标准化标准差
            print("[提示] 初始模型训练完成", flush=True)
        else:
            raise ValueError("训练数据为空或路径不存在")
    except Exception as e:
        # 训练失败则使用随机初始化模型
        print(f"[警告] 训练模型失败: {e}", flush=True)
        print("[提示] 使用随机初始模型代替", flush=True)
        feature_len = 15  # 特征维度
        n_classes = 10     # 类别数
        # 随机初始化模型参数
        elm.W = np.random.randn(feature_len, 500)*0.1
        elm.b = np.random.randn(500)*0.1
        elm.beta = np.random.randn(500, n_classes)*0.1
        # 初始化标准化参数
        mean = np.zeros(feature_len)
        scale = np.ones(feature_len)
        # 保存随机模型
        np.save(MODEL_PATH, {'W': elm.W, 'b': elm.b, 'beta': elm.beta, 'mean': mean, 'scale': scale})
else:
    # 加载已存在的模型
    params = np.load(MODEL_PATH, allow_pickle=True).item()
    elm.W = params['W']
    elm.b = params['b']
    elm.beta = params['beta']
    mean = params['mean']
    scale = params['scale']

# =========================
# 模型版本管理与热更新
# =========================
def get_model_version():
    """
    获取当前模型版本号
    :return: 版本号（整数），文件不存在时返回0
    """
    if not os.path.exists(VERSION_PATH):
        return 0
    with open(VERSION_PATH,'r') as f:
        return int(f.read())

# 初始化当前版本号
current_version = get_model_version()

# 版本更新锁（线程安全）
_version_lock = threading.Lock()

def check_model_update():
    """
    检查模型是否有更新，如有则热更新
    :return: True-模型已更新，False-无更新
    """
    global current_version
    latest_version = get_model_version()
    with _version_lock:
        if latest_version != current_version:
            current_version = latest_version
            reload_model()  # 重新加载模型
            return True
    return False

def reload_model():
    """
    重新加载模型参数（热更新）
    线程安全，使用锁保护
    """
    global elm, mean, scale
    with _version_lock:
        # 加载最新模型参数
        params = np.load(MODEL_PATH, allow_pickle=True).item()
        elm.W = params['W']
        elm.b = params['b']
        elm.beta = params['beta']
        mean = params['mean']
        scale = params['scale']
        print("[detector] 模型热更新完成")

# =========================
# 故障预测函数
# =========================
def predict_signal(sig, debug=False, simulation_mode=False):
    """
    对单个振动信号进行故障预测
    :param sig: 一维振动信号数组
    :param debug: 是否输出调试信息
    :param simulation_mode: 是否为模拟模式（随机返回结果）
    :return: (预测结果字符串, 调试信息字符串)
    """
    # 提取特征
    feat = extract_features(sig)
    # 特征缩放处理的索引（数值较大的特征）
    large_idx = [8, 13, 14]

    # =========================
    # 特征预处理
    # =========================
    # 对数缩放（避免数值过大）
    for i in large_idx:
        feat[i] = np.log1p(abs(feat[i])) * np.sign(feat[i])

    # 标准化
    feat_norm = (feat - mean) / scale
    feat_norm = np.clip(feat_norm, -10, 10)  # 限制范围，避免极端值

    # =========================
    # 预测结果生成
    # =========================
    if simulation_mode:
        # 模拟模式：随机返回故障类型
        result = np.random.choice([
            "Normal", "12k Drive Ball", "12k Drive Inner Race", "12k Drive Outer Race",
            "12k Fan Ball", "12k Fan Inner Race", "12k Fan Outer Race",
            "48k Drive Ball", "48k Drive Inner Race", "48k Drive Outer Race"
        ])
    else:
        # 真实预测：使用ELM模型
        # 前向传播计算输出
        out = OptimizedELM._sigmoid(feat_norm.reshape(1, -1) @ elm.W + elm.b) @ elm.beta
        pred_class = np.argmax(out)  # 取最大概率的类别
        # 类别标签映射
        label_map = {
            0: "Normal",
            1: "12k Drive Ball",
            2: "12k Drive Inner Race",
            3: "12k Drive Outer Race",
            4: "12k Fan Ball",
            5: "12k Fan Inner Race",
            6: "12k Fan Outer Race",
            7: "48k Drive Ball",
            8: "48k Drive Inner Race",
            9: "48k Drive Outer Race"
        }
        result = label_map.get(pred_class, f"Unknown({pred_class})")

    # =========================
    # 调试信息生成
    # =========================
    debug_info = None
    if debug:
        if simulation_mode:
            debug_info = f"[调试] 模拟信号类别: {result}"
        else:
            # 详细调试信息
            debug_info = f"""---- 调试信息 ----
时间: {time.strftime('%H:%M:%S')}
信号前10个样本: {sig[:10]}
特征向量: {np.round(feat,3)}
标准化后特征: {np.round(feat_norm,3)}
ELM 输出: {np.round(out,3) if not simulation_mode else '模拟'}
预测类别: {result}
预测置信度: {np.max(out) if not simulation_mode else 1.0:.3f}
-----------------"""

    return result, debug_info

# =========================
# 故障模拟信号生成
# =========================
def simulate_fault_signal(fault_type='Normal', fs=12000, duration=1.0, fr=50):
    """
    生成模拟故障振动信号
    :param fault_type: 故障类型（Normal/Ball/Inner/Outer）
    :param fs: 采样频率（Hz）
    :param duration: 信号时长（秒）
    :param fr: 基础频率（Hz）
    :return: (模拟信号数组, 故障类型字符串)
    """
    # 生成时间轴
    t = np.arange(0, duration, 1/fs)
    # 基础正弦信号
    x = np.sin(2 * np.pi * fr * t)

    # 添加故障脉冲
    if fault_type != 'Normal':
        # 根据故障类型设置脉冲参数
        if fault_type == 'Ball':
            fault_interval, amplitude = 0.05, 1.0
        elif fault_type == 'Inner':
            fault_interval, amplitude = 0.02, 0.8
        elif fault_type == 'Outer':
            fault_interval, amplitude = 0.03, 0.9
        else:
            fault_interval, amplitude = 0.04, 0.7

        # 生成故障脉冲
        pulse = np.zeros_like(t)
        indices = np.arange(0, len(t), int(fault_interval * fs))
        pulse[indices] = amplitude
        x += pulse

    # 添加高斯噪声
    x += np.random.normal(0, 0.05, size=len(t))
    return x, fault_type

# =========================
# Paderborn数据集信号加载
# =========================
def load_paderborn_signal(file_path, target_length=2048):
    """
    加载Paderborn轴承数据集的.mat文件
    :param file_path: mat文件路径
    :param target_length: 目标信号长度（默认2048）
    :return: 固定长度的一维信号数组
    """
    # 加载mat文件
    mat = scipy.io.loadmat(file_path)
    signal_struct = mat['Signal'][0, 0]

    # 提取y值信号（Paderborn数据集格式）
    y_vals = signal_struct['y_values']
    sig = y_vals[0][0][0]

    # 展平为一维数组
    sig = np.array(sig).flatten()

    # 调整信号长度到目标长度
    if len(sig) > target_length:
        sig = sig[:target_length]  # 截断
    else:
        # 补零
        sig = np.pad(sig, (0, target_length - len(sig)), mode='constant')

    return sig

# =========================
# 批量加载Paderborn信号
# =========================
def collect_signals_from_paderborn(
    n_clients=3,
    n_signals=2,
    debug=False,
    dataset_dir=os.path.join(BASE_DIR, '../../数据集/Paderborn_Bearing_Dataset')
):
    """
    从Paderborn数据集批量生成客户端信号
    :param n_clients: 客户端数量
    :param n_signals: 每个客户端的信号数量
    :param debug: 是否输出调试信息
    :param dataset_dir: 数据集目录
    :return: 二维列表 [客户端][信号]
    """
    all_signals = []
    # 检查数据集目录是否存在
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"数据集路径不存在: {dataset_dir}")

    # 获取所有mat文件
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.mat')]
    if not files:
        raise FileNotFoundError("无 .mat 文件")

    # 为每个客户端生成信号
    for client_id in range(n_clients):
        client_sigs = []
        for _ in range(n_signals):
            # 随机选择文件
            f = random.choice(files)
            path = os.path.join(dataset_dir, f)
            # 加载信号
            sig = load_paderborn_signal(path)
            client_sigs.append(sig)
            # 调试信息
            if debug:
                print(f"[Paderborn] 客户端{client_id} 文件: {f}")
        all_signals.append(client_sigs)
    return all_signals

# =========================
# 模拟信号采集
# =========================
def collect_signals(n_clients=3, n_signals=2, debug=False, simulation_mode=False):
    """
    生成模拟振动信号（用于测试）
    :param n_clients: 客户端数量
    :param n_signals: 每个客户端的信号数量
    :param debug: 是否输出调试信息
    :param simulation_mode: 是否生成故障模拟信号
    :return: 二维列表 [客户端][信号]
    """
    all_signals = []
    # 为每个客户端生成信号
    for cid in range(n_clients):
        cs = []
        for _ in range(n_signals):
            if simulation_mode:
                # 故障模拟模式：随机选择故障类型
                selected_label = random.choice([
                    "Normal",
                    "12k Drive Ball",
                    "12k Drive Inner Race",
                    "12k Drive Outer Race",
                    "12k Fan Ball",
                    "12k Fan Inner Race",
                    "12k Fan Outer Race",
                    "48k Drive Ball",
                    "48k Drive Inner Race",
                    "48k Drive Outer Race"
                ])
                # 映射故障类型
                if "Ball" in selected_label:
                    ft = "Ball"
                elif "Inner" in selected_label:
                    ft = "Inner"
                elif "Outer" in selected_label:
                    ft = "Outer"
                else:
                    ft = "Normal"
                # 生成故障模拟信号
                sig, _ = simulate_fault_signal(fault_type=ft)
            else:
                # 普通模拟模式：不同客户端生成不同类型信号
                if cid == 0:
                    sig = np.random.randn(2048)  # 高斯噪声
                elif cid == 1:
                    sig = np.sin(np.linspace(0, 50, 2048))  # 正弦波
                elif cid == 2:
                    sig = np.sign(np.sin(np.linspace(0, 20, 2048)))  # 方波
                else:
                    sig = np.random.randn(2048)  # 默认高斯噪声
            cs.append(sig)
        all_signals.append(cs)
    return all_signals