# trainer.py

import os
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

MODEL_PATH = r"../models/model.npy"
DATA_PATH = r"../CRWU"

# 中文显示
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def train_model(data_path=DATA_PATH,
                extra_data=None,
                save_path=MODEL_PATH,
                n_hidden=1200,
                visualize=True):

    models_dir = os.path.dirname(save_path)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    print("加载训练数据...")
    X, y = load_data(data_path)

    # 处理额外数据
    if extra_data is not None:
        X = np.vstack([X, extra_data['X']])
        y = np.hstack([y, extra_data['y']])

    # 类别加权
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weights))
    print("类别权重:", class_weight_dict)

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 少数类过采样
    max_count = max(np.bincount(y))
    X_aug, y_aug = [], []
    for cls in classes:
        idx = np.where(y == cls)[0]
        X_cls = X[idx]
        y_cls = y[idx]
        n_repeat = int(np.ceil(max_count / len(idx)))
        X_aug.append(np.tile(X_cls, (n_repeat, 1))[:max_count])
        y_aug.append(np.tile(y_cls, n_repeat)[:max_count])
    X_bal = np.vstack(X_aug)
    y_bal = np.hstack(y_aug)

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )

    # 训练 ELM
    print(f"训练 ELM 模型，隐藏节点数={n_hidden} ...")
    elm = OptimizedELM(n_hidden=n_hidden)

    # 使用类别权重训练
    sample_weight = np.array([class_weight_dict[label] for label in y_train])
    elm.fit(X_train, y_train, sample_weight=sample_weight)

    # 测试
    pred = elm.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f'训练完成，测试集 Accuracy: {acc:.4f}')

    if visualize:
        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("预测标签")
        plt.ylabel("真实标签")
        plt.title("ELM 混淆矩阵")
        plt.show()

        f1_scores = f1_score(y_test, pred, average=None)
        plt.figure(figsize=(10,6))
        sns.barplot(x=classes, y=f1_scores)
        plt.xlabel("类别")
        plt.ylabel("F1 分数")
        plt.title("每类 F1 分数")
        plt.show()

    # 保存模型
    params = {'W': elm.W, 'b': elm.b, 'beta': elm.beta, 'mean': scaler.mean_, 'scale': scaler.scale_}
    np.save(save_path, params)
    print(f'模型保存完成: {save_path}')

    return elm, scaler


