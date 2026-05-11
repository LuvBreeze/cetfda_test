# detector.py

import os
import numpy as np
import time

from model import OptimizedELM
from feature import extract_features
from cloud.trainer import train_model

# -------------------
# 动态计算数据和模型路径
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # detector.py 所在目录
MODEL_PATH = os.path.join(BASE_DIR, '../models/model.npy')
VERSION_PATH = os.path.join(BASE_DIR, '../models/version.txt')
DATA_PATH = os.path.join(BASE_DIR, '../CRWU')

# -------------------
# 初始化模型
# -------------------
models_dir = os.path.dirname(MODEL_PATH)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

elm = OptimizedELM()

# 尝试训练初始模型，如果数据为空或训练失败，退回随机模型
if not os.path.exists(MODEL_PATH):
    print("[警告] 模型不存在，正在训练初始模型...", flush=True)
    try:
        # 检查数据路径存在且有文件
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
        return 0
    with open(VERSION_PATH,'r') as f:
        return int(f.read())
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
    if latest_version != current_version:
        current_version = latest_version
        reload_model()
        return True
    return False

# -------------------
# Sigmoid
# -------------------
def stable_sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1/(1+np.exp(-x))

# -------------------
# 预测
# -------------------
def predict_signal(sig, debug=False):
    debug_info = None
    feat = extract_features(sig)
    large_idx = [8, 13, 14]
    for i in large_idx:
        feat[i] = np.log1p(abs(feat[i]))*np.sign(feat[i])
    feat_norm = (feat - mean)/scale
    feat_norm = np.clip(feat_norm, -10, 10)
    out = stable_sigmoid(feat_norm.reshape(1,-1) @ elm.W + elm.b) @ elm.beta
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
        debug_info = \
f"""---- 调试信息 ----
时间: {time.strftime('%H:%M:%S')}
信号前10个样本: {sig[:10]}
特征向量: {np.round(feat, 3)}
标准化后特征: {np.round(feat_norm, 3)}
ELM 输出: {np.round(out, 3)}
ELM 输出每列: {out.flatten()}
预测类别: {result}
预测置信度: {np.max(out):.3f}
-----------------"""
    return result, debug_info

# -------------------
# 多客户端多信号采集
# -------------------
def collect_signals(n_clients=3, n_signals=2, debug=False, scada_mode=False):
    all_signals = []
    for client_id in range(n_clients):
        client_signals = []
        for _ in range(n_signals):
            if scada_mode:
                sig = np.random.randn(2048)
            else:
                if client_id==0:
                    sig = np.random.randn(2048)
                elif client_id==1:
                    sig = np.sin(np.linspace(0,50,2048))
                elif client_id==2:
                    sig = np.sign(np.sin(np.linspace(0,20,2048)))
                else:
                    sig = np.random.randn(2048)*0.5 + np.sin(np.linspace(0,5,2048))
            client_signals.append(sig)
            if debug:
                print(f"[调试] 客户端{client_id} 信号均值: {np.mean(sig):.3f}, std: {np.std(sig):.3f}")
        all_signals.append(client_signals)
    return all_signals