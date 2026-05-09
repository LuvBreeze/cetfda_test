# detector.py

import os
import numpy as np
from model import OptimizedELM
from feature import extract_features

MODEL_PATH = r"E:/VScode/Python/毕设/project/models/model.npy"
VERSION_PATH = r"E:/VScode/Python/毕设/project/models/version.txt"

# -------------------
# 初始化模型
# -------------------
models_dir = os.path.dirname(MODEL_PATH)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(MODEL_PATH):
    print("[警告] 模型不存在，正在生成初始模型...")
    feature_len = 15
    n_classes = 4
    elm = OptimizedELM()
    elm.W = np.random.randn(feature_len, 500)*0.1
    elm.b = np.random.randn(500)*0.1
    elm.beta = np.random.randn(500, n_classes)*0.1
    mean = np.zeros(feature_len)
    scale = np.ones(feature_len)
    np.save(MODEL_PATH, {'W': elm.W, 'b': elm.b, 'beta': elm.beta,
                         'mean': mean, 'scale': scale})
else:
    params = np.load(MODEL_PATH, allow_pickle=True).item()
    elm = OptimizedELM()
    elm.W = params['W']
    elm.b = params['b']
    elm.beta = params['beta']
    mean = params['mean']
    scale = params['scale']

# -------------------
# 模型版本
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
    feat = extract_features(sig)
    # 处理极大值特征
    large_idx = [8, 13, 14]
    for i in large_idx:
        feat[i] = np.log1p(abs(feat[i]))*np.sign(feat[i])
    # 标准化
    feat_norm = (feat - mean)/scale
    feat_norm = np.clip(feat_norm, -10, 10)
    # ELM 输出
    out = stable_sigmoid(feat_norm.reshape(1,-1) @ elm.W + elm.b) @ elm.beta
    pred_class = np.argmax(out)
    label_map = {0:"Normal",1:"Ball Fault",2:"Inner Race Fault",3:"Outer Race Fault"}
    result = label_map[pred_class]

    if debug:
        print("---- 调试信息 ----")
        print("信号前10个样本:", sig[:10])
        print("特征向量:", np.round(feat,3))
        print("标准化后特征:", np.round(feat_norm,3))
        print("ELM 输出:", np.round(out,3))
        print("ELM 输出每列:", out.flatten())
        print("预测类别:", result)
        print("-----------------")

    return result

# -------------------
# 多客户端多信号采集
# -------------------
def collect_signals(n_clients=3, n_signals=2, debug=False, scada_mode=False):
    all_signals = []
    for client_id in range(n_clients):
        client_signals = []
        for _ in range(n_signals):
            if scada_mode:
                # 这里需要实现 SCADA 实时采集接口
                sig = np.random.randn(2048)  # 占位
            else:
                # 模拟不同客户端不同模式
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