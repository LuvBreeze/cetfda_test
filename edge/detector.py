#detector.py

import os
import random

import numpy as np
import time
import scipy.io

from model import OptimizedELM
from feature import extract_features
from cloud.trainer import train_model

# -------------------
# 动态计算数据和模型路径
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../models/model.npy')
VERSION_PATH = os.path.join(BASE_DIR, '../models/version.txt')
DATA_PATH = os.path.join(BASE_DIR, '../CWRU')

# -------------------
# 初始化模型
# -------------------
models_dir = os.path.dirname(MODEL_PATH)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

elm = OptimizedELM()

if not os.path.exists(MODEL_PATH):
    print("[警告] 模型不存在，正在训练初始模型...", flush=True)
    try:
        if os.path.exists(DATA_PATH) and any(os.scandir(DATA_PATH)):
            elm, scaler = train_model(data_path=DATA_PATH, save_path=MODEL_PATH)
            mean = scaler.mean_
            scale = scaler.scale_
            print("[提示] 初始模型训练完成", flush=True)
        else:
            raise ValueError("训练数据为空或路径不存在")
    except Exception as e:
        print(f"[警告] 训练模型失败: {e}", flush=True)
        print("[提示] 使用随机初始模型代替", flush=True)
        feature_len = 15
        n_classes = 10
        elm.W = np.random.randn(feature_len, 500)*0.1
        elm.b = np.random.randn(500)*0.1
        elm.beta = np.random.randn(500, n_classes)*0.1
        mean = np.zeros(feature_len)
        scale = np.ones(feature_len)
        np.save(MODEL_PATH, {'W': elm.W, 'b': elm.b, 'beta': elm.beta, 'mean': mean, 'scale': scale})
else:
    params = np.load(MODEL_PATH, allow_pickle=True).item()
    elm.W = params['W']
    elm.b = params['b']
    elm.beta = params['beta']
    mean = params['mean']
    scale = params['scale']

# -------------------
# 模型版本及热更新
# -------------------
def get_model_version():
    if not os.path.exists(VERSION_PATH):
        return 0.0
    with open(VERSION_PATH,'r') as f:
        return float(f.read())

current_version = get_model_version()

def reload_model():
    global elm, mean, scale
    params = np.load(MODEL_PATH, allow_pickle=True).item()
    elm.W = params['W']
    elm.b = params['b']
    elm.beta = params['beta']
    mean = params['mean']
    scale = params['scale']
    print("[detector] 模型热更新完成")

def check_model_update():
    global current_version
    latest_version = get_model_version()
    current_version_rounded = round(current_version, 1)
    latest_version_rounded = round(latest_version, 1)

    if latest_version_rounded != current_version_rounded:
        current_version = latest_version
        reload_model()
        return True
    return False

# -------------------
# 预测
# -------------------
def predict_signal(sig, debug=False, simulation_mode=False):
    debug_info = None
    feat = extract_features(sig)
    large_idx = [8, 13, 14]

    if simulation_mode:
        result = np.random.choice(["Normal",
                    "12k Drive Ball",
                    "12k Drive Inner Race",
                    "12k Drive Outer Race",
                    "12k Fan Ball",
                    "12k Fan Inner Race",
                    "12k Fan Outer Race",
                    "48k Drive Ball",
                    "48k Drive Inner Race",
                    "48k Drive Outer Race"])
        debug_info = f"模拟故障信号类别: {result}" if debug else None
        return result, debug_info

    for i in large_idx:
        feat[i] = np.log1p(abs(feat[i])) * np.sign(feat[i])

    feat_norm = (feat - mean) / scale
    feat_norm = np.clip(feat_norm, -10, 10)

    out = OptimizedELM._sigmoid(feat_norm.reshape(1, -1) @ elm.W + elm.b) @ elm.beta
    pred_class = np.argmax(out)

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

    if debug:
        debug_info = f"""---- 调试信息 ----
时间: {time.strftime('%H:%M:%S')}
信号前10个样本: {sig[:10]}
特征向量: {np.round(feat, 3)}
标准化后特征: {np.round(feat_norm, 3)}
ELM 输出: {np.round(out, 3)}
预测类别: {result}
预测置信度: {np.max(out):.3f}
-----------------"""
    return result, debug_info

# =========================
# 故障模拟信号
# =========================
def simulate_fault_signal(fault_type='Normal', fs=12000, duration=1.0, fr=50):
    t = np.arange(0, duration, 1/fs)
    x = np.sin(2 * np.pi * fr * t)

    if fault_type != 'Normal':
        if fault_type == 'Ball':
            fault_interval, amplitude = 0.05, 1.0
        elif fault_type == 'Inner':
            fault_interval, amplitude = 0.02, 0.8
        elif fault_type == 'Outer':
            fault_interval, amplitude = 0.03, 0.9
        else:
            fault_interval, amplitude = 0.04, 0.7

        pulse = np.zeros_like(t)
        indices = np.arange(0, len(t), int(fault_interval * fs))
        pulse[indices] = amplitude
        x += pulse

    x += np.random.normal(0, 0.05, size=len(t))
    return x, fault_type

# =========================
# 读取 Paderborn 数据集
# =========================
def load_paderborn_signal(file_path, target_length=2048):
    import numpy as np
    import scipy.io

    mat = scipy.io.loadmat(file_path)
    signal_struct = mat['Signal'][0, 0]

    y_vals = signal_struct['y_values']
    sig = y_vals[0][0][0]

    # 展平成一维波形
    sig = np.array(sig).flatten()

    # 固定长度 2048
    if len(sig) > target_length:
        sig = sig[:target_length]
    else:
        sig = np.pad(sig, (0, target_length - len(sig)), mode='constant')

    return sig

# =========================
# 批量读取 Paderborn
# =========================
def collect_signals_from_paderborn(
    n_clients=3,
    n_signals=2,
    debug=False,
    dataset_dir=os.path.join(BASE_DIR, '../../数据集/Paderborn_Bearing_Dataset')
):
    all_signals = []
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"数据集路径不存在: {dataset_dir}")

    files = [f for f in os.listdir(dataset_dir) if f.endswith('.mat')]
    if not files:
        raise FileNotFoundError("无 .mat 文件")

    for client_id in range(n_clients):
        client_sigs = []
        for _ in range(n_signals):
            f = random.choice(files)
            path = os.path.join(dataset_dir, f)
            sig = load_paderborn_signal(path)
            client_sigs.append(sig)
            if debug:
                print(f"[Paderborn] 客户端{client_id} 文件: {f}")
        all_signals.append(client_sigs)
    return all_signals

# -------------------
# 采集信号
# -------------------
def collect_signals(n_clients=3, n_signals=2, debug=False, simulation_mode=False):
    all_signals = []
    for cid in range(n_clients):
        cs = []
        for _ in range(n_signals):
            if simulation_mode:
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
                if "Ball" in selected_label:
                    ft = "Ball"
                elif "Inner" in selected_label:
                    ft = "Inner"
                elif "Outer" in selected_label:
                    ft = "Outer"
                else:
                    ft = "Normal"
                sig, _ = simulate_fault_signal(fault_type=ft)
            else:
                if cid == 0:
                    sig = np.random.randn(2048)
                elif cid == 1:
                    sig = np.sin(np.linspace(0, 50, 2048))
                elif cid == 2:
                    sig = np.sign(np.sin(np.linspace(0, 20, 2048)))
                else:
                    sig = np.random.randn(2048)
            cs.append(sig)
        all_signals.append(cs)
    return all_signals