# -*- coding: utf-8 -*-
# 模型训练模块
# 功能：训练优化的ELM模型，支持类别加权、数据平衡、可视化
# 依赖：os, sklearn, matplotlib, seaborn, numpy, data_loader, model

import os
import threading
import time

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

from data_loader import load_data
from model import OptimizedELM

# 路径配置
BASE = os.path.dirname(os.path.abspath(__file__))  # 自动获取当前文件所在目录
NEW_DATA_PATH = os.path.join(BASE, "../uploaded_data/new_data.npy")
MODEL_PATH = os.path.join(BASE, "../models/model.npy")
VERSION_PATH = os.path.join(BASE, "../models/version.txt")
DATA_PATH = os.path.join(BASE, "../CWRU")         # 数据集路径

# 配置matplotlib中文显示
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def train_model(data_path=DATA_PATH,
                extra_data=None,
                save_path=MODEL_PATH,
                n_hidden=1200,
                visualize=True):
    """
    训练优化的ELM模型
    :param data_path: 基础数据集路径
    :param extra_data: 额外数据（增量训练）
    :param save_path: 模型保存路径
    :param n_hidden: ELM隐藏层节点数
    :param visualize: 是否可视化结果
    :return: (训练好的ELM模型, 标准化器)
    """
    # 创建模型保存目录
    models_dir = os.path.dirname(save_path)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # =========================
    # 加载数据
    # =========================
    print("加载训练数据...")
    X, y = load_data(data_path)

    # 处理额外增量数据
    if extra_data is not None:
        X = np.vstack([X, extra_data['X']])  # 合并特征
        y = np.hstack([y, extra_data['y']])  # 合并标签

    # =========================
    # 类别加权
    # =========================
    classes = np.unique(y)  # 获取所有类别
    # 计算类别权重
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weights))
    print("类别权重:", class_weight_dict)

    # =========================
    # 数据标准化
    # =========================
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # =========================
    # 少数类过采样（数据平衡）
    # =========================
    max_count = max(np.bincount(y))  # 最多类别的样本数
    X_aug, y_aug = [], []
    for cls in classes:
        idx = np.where(y == cls)[0]
        X_cls = X[idx]
        y_cls = y[idx]
        # 计算需要重复的次数
        n_repeat = int(np.ceil(max_count / len(idx)))
        # 重复采样并截断到最大数量
        X_aug.append(np.tile(X_cls, (n_repeat, 1))[:max_count])
        y_aug.append(np.tile(y_cls, n_repeat)[:max_count])
    # 合并平衡后的数据
    X_bal = np.vstack(X_aug)
    y_bal = np.hstack(y_aug)

    # =========================
    # 划分训练/测试集
    # =========================
    X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # 类别加权：只根据训练集计算
    # =========================
    classes = np.unique(y_train_raw)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train_raw
    )
    class_weight_dict = dict(zip(classes, class_weights))
    print("类别权重:", class_weight_dict)

    # =========================
    # 数据标准化：只用训练集 fit
    # =========================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # =========================
    # 少数类过采样：只对训练集做
    # =========================
    max_count = max([np.sum(y_train_raw == cls) for cls in classes])

    X_aug, y_aug = [], []
    for cls in classes:
        idx = np.where(y_train_raw == cls)[0]
        X_cls = X_train[idx]
        y_cls = y_train_raw[idx]

        n_repeat = int(np.ceil(max_count / len(idx)))
        X_aug.append(np.tile(X_cls, (n_repeat, 1))[:max_count])
        y_aug.append(np.tile(y_cls, n_repeat)[:max_count])

    X_train_bal = np.vstack(X_aug)
    y_train_bal = np.hstack(y_aug)

    # =========================
    # 训练 ELM 模型
    # =========================
    print(f"训练 ELM 模型，隐藏节点数={n_hidden} ...")
    elm = OptimizedELM(n_hidden=n_hidden)

    sample_weight = np.array([class_weight_dict[label] for label in y_train_bal])
    elm.fit(X_train_bal, y_train_bal, sample_weight=sample_weight)

    # =========================
    # 模型评估：测试集不参与过采样
    # =========================
    pred = elm.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f'训练完成，测试集 Accuracy: {acc:.4f}')

    # =========================
    # 结果可视化
    # =========================
    if visualize:
        # 绘制混淆矩阵
        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("预测标签")
        plt.ylabel("真实标签")
        plt.title("ELM 混淆矩阵")
        plt.show()

        # 绘制每类F1分数
        f1_scores = f1_score(y_test, pred, average=None)
        plt.figure(figsize=(10,6))
        sns.barplot(x=classes, y=f1_scores)
        plt.xlabel("类别")
        plt.ylabel("F1 分数")
        plt.title("每类 F1 分数")
        plt.show()

    # =========================
    # 保存模型
    # =========================
    params = {'W': elm.W, 'b': elm.b, 'beta': elm.beta, 'mean': scaler.mean_, 'scale': scaler.scale_}
    np.save(save_path, params)
    print(f'模型保存完成: {save_path[save_path.find("../"):]}')
    return elm, scaler