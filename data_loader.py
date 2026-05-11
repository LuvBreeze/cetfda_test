# data_loader.py

import numpy as np
import scipy.io as sio
import os
from feature import extract_features

def load_data(base_path):
    """
    加载凯斯西储大学CWRU轴承数据集
    自动递归读取所有子文件夹，并提取DE和FE信号
    """
    X = []
    y = []

    def process_file(filepath, label, signal_type='DE'):
        try:
            data = sio.loadmat(filepath)
        except:
            return

        sig = None
        for key in data.keys():
            if signal_type in key:
                sig = data[key].flatten()
                break

        if sig is None or len(sig) < 2048:
            return

        for i in range(0, len(sig) - 2048, 1024):
            seg = sig[i:i+2048]
            feat = extract_features(seg)
            X.append(feat)
            y.append(label)

    def traverse(folder, label, signal_type='DE'):
        if not os.path.exists(folder):
            return
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith('.mat'):
                    process_file(os.path.join(root, f), label, signal_type)

    # 数据集路径列表及标签映射
    datasets = [
        ('Normal Baseline', 0, 'DE'),
        ('12k Drive End Bearing Fault Data/Ball', 1, 'DE'),
        ('12k Drive End Bearing Fault Data/Inner Race', 2, 'DE'),
        ('12k Drive End Bearing Fault Data/Outer Race', 3, 'DE'),
        ('48k Drive End Bearing Fault Data/Ball', 4, 'DE'),
        ('48k Drive End Bearing Fault Data/Inner Race', 5, 'DE'),
        ('48k Drive End Bearing Fault Data/Outer Race', 6, 'DE'),
        ('12k Fan End Bearing Fault Data/Ball', 7, 'FE'),
        ('12k Fan End Bearing Fault Data/Inner Race', 8, 'FE'),
        ('12k Fan End Bearing Fault Data/Outer Race', 9, 'FE')
    ]

    # 遍历并加载所有数据
    for folder_rel, label, sig_type in datasets:
        folder_path = os.path.join(base_path, folder_rel)
        print('读取:', folder_path)
        traverse(folder_path, label, sig_type)

    print('\n===== 数据加载完成 =====')
    print('总样本数:', len(y))
    if len(y) > 0:
        print('标签种类:', np.unique(y))
        print('每类数量:', np.bincount(y))
    print('========================\n')

    return np.array(X), np.array(y)