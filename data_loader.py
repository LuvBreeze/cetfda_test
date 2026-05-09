# data_loader.py(数据加载)
import numpy as np
import scipy.io as sio
import os
from feature import extract_features

def load_data(base_path):
    """
    加载凯斯西储大学CWRU轴承数据集
    自动递归读取所有子文件夹
    提取驱动端振动信号并计算特征
    """
    X = []
    y = []

    def process_file(filepath, label):
        """
        处理单个mat文件
        读取DE信号，切片并提取特征
        """
        try:
            data = sio.loadmat(filepath)
        except:
            return

        # 寻找驱动端DE信号
        sig = None
        for key in data.keys():
            if "DE" in key:
                sig = data[key].flatten()
                break

        # 信号长度不足则跳过
        if sig is None or len(sig) < 2048:
            return

        # 滑动窗口截取信号片段
        for i in range(0, len(sig) - 2048, 1024):
            seg = sig[i:i+2048]
            feat = extract_features(seg)
            X.append(feat)
            y.append(label)

    def traverse(folder, label):
        """
        递归遍历文件夹下所有mat文件
        """
        if not os.path.exists(folder):
            return
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".mat"):
                    process_file(os.path.join(root, f), label)

    # 加载正常数据
    normal_path = os.path.join(base_path, "Normal Baseline")
    print("读取:", normal_path)
    traverse(normal_path, 0)

    # 加载12k驱动端故障数据
    fault_12k = os.path.join(base_path, "12k Drive End Bearing Fault Data")

    # 加载滚珠故障数据
    ball_path = os.path.join(fault_12k, "Ball")
    print("读取:", ball_path)
    traverse(ball_path, 1)

    # 加载内圈故障数据
    inner_path = os.path.join(fault_12k, "Inner Race")
    print("读取:", inner_path)
    traverse(inner_path, 2)

    # 加载外圈故障数据（包含所有子目录）
    outer_path = os.path.join(fault_12k, "Outer Race")
    print("读取:", outer_path)
    traverse(outer_path, 3)

    # 输出数据集信息
    print("\n===== 数据加载完成 =====")
    print("总样本数:", len(y))
    if len(y) > 0:
        print("标签种类:", np.unique(y))
        print("每类数量:", np.bincount(y))
    print("========================\n")

    return np.array(X), np.array(y)