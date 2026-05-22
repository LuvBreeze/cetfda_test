# -*- coding: utf-8 -*-
"""
模型对比测试
功能：
1. 对比 ELM、SVM、RandomForest 在 CWRU 数据集上的分类性能；
2. 实验流程与论文第 5.1 节保持一致：
   - 先将原始样本按照 8:2 划分为训练集和测试集；
   - 标准化参数仅由训练集统计；
   - 重复过采样仅作用于训练集；
   - 测试集保持原始类别分布不变；
3. 输出 Accuracy、F1-score、训练与测试总耗时；
4. 额外输出 ELM 单样本平均推理时间和模型文件大小，可用于论文补充说明。

依赖：
time, os, numpy, sklearn, data_loader, model
"""

import os
import time
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from data_loader import load_data
from model import OptimizedELM


# =========================
# 基本参数设置
# =========================
DATA_PATH = "CWRU"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_HIDDEN = 1200

TEMP_MODEL_PATH = os.path.join("models", "temp_elm_compare.npy")


def print_class_distribution(name, y):
    """打印类别分布，便于检查训练集、测试集和过采样结果。"""
    classes, counts = np.unique(y, return_counts=True)
    print(f"\n{name}类别分布：")
    for cls, count in zip(classes, counts):
        print(f"类别 {cls}: {count}")
    print(f"合计: {len(y)}")


def oversample_training_set(X_train, y_train):
    """
    只对训练集进行重复过采样，使各类别样本数与训练集中最大类别样本数一致。
    注意：测试集不参与过采样，避免数据泄漏。
    """
    classes, counts = np.unique(y_train, return_counts=True)
    max_count = counts.max()

    X_aug_list = []
    y_aug_list = []

    for cls in classes:
        idx = np.where(y_train == cls)[0]
        X_cls = X_train[idx]
        y_cls = y_train[idx]

        n_repeat = int(np.ceil(max_count / len(idx)))

        X_aug = np.tile(X_cls, (n_repeat, 1))[:max_count]
        y_aug = np.tile(y_cls, n_repeat)[:max_count]

        X_aug_list.append(X_aug)
        y_aug_list.append(y_aug)

    X_train_balanced = np.vstack(X_aug_list)
    y_train_balanced = np.hstack(y_aug_list)

    # 打乱过采样后的训练集，避免类别顺序影响模型训练
    rng = np.random.default_rng(RANDOM_STATE)
    perm = rng.permutation(len(y_train_balanced))

    return X_train_balanced[perm], y_train_balanced[perm]


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, fit_kwargs=None):
    """
    训练并评估模型，返回 Accuracy、F1-score 和训练测试总耗时。
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    start = time.perf_counter()

    model.fit(X_train, y_train, **fit_kwargs)
    pred = model.predict(X_test)

    end = time.perf_counter()

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    total_time = end - start

    print(
        f"{model_name}: "
        f"Acc={acc:.4f}, "
        f"F1={f1:.4f}, "
        f"Time={total_time:.4f}s"
    )

    return {
        "model": model_name,
        "accuracy": acc,
        "f1_score": f1,
        "time": total_time,
        "pred": pred
    }


def test_elm_inference_efficiency(elm, scaler, X_test):
    """
    测试 ELM 单样本平均推理时间，并保存临时模型文件以统计模型大小。
    该结果可用于论文中补充说明 ELM 的边缘端部署潜力。
    """
    repeat = min(1000, len(X_test))

    # 预热，避免第一次调用影响计时
    for i in range(min(50, repeat)):
        elm.predict(X_test[i:i + 1])

    start = time.perf_counter()
    for i in range(repeat):
        elm.predict(X_test[i:i + 1])
    end = time.perf_counter()

    avg_infer_time_ms = (end - start) / repeat * 1000

    os.makedirs(os.path.dirname(TEMP_MODEL_PATH), exist_ok=True)

    params = {
        "W": elm.W,
        "b": elm.b,
        "beta": elm.beta,
        "mean": scaler.mean_,
        "scale": scaler.scale_
    }
    np.save(TEMP_MODEL_PATH, params)

    model_size_kb = os.path.getsize(TEMP_MODEL_PATH) / 1024

    print("\nELM 边缘端推理效率测试：")
    print(f"输入特征维度: {X_test.shape[1]}")
    print(f"隐含层节点数: {N_HIDDEN}")
    print(f"单样本平均推理时间: {avg_infer_time_ms:.6f} ms")
    print(f"模型文件大小: {model_size_kb:.2f} KB")

    return avg_infer_time_ms, model_size_kb


def main():
    # =========================
    # 1. 加载原始数据
    # =========================
    print("加载 CWRU 数据集...")
    X, y = load_data(DATA_PATH)

    print_class_distribution("原始样本", y)

    # =========================
    # 2. 先划分训练集和测试集
    # =========================
    X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print_class_distribution("原始训练集", y_train_raw)
    print_class_distribution("原始测试集", y_test)

    # =========================
    # 3. 标准化：只用训练集 fit
    # =========================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # =========================
    # 4. 类别权重：只根据训练集计算
    # =========================
    classes = np.unique(y_train_raw)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_raw
    )
    class_weight_dict = dict(zip(classes, class_weights))

    print("\n类别权重：")
    for cls, weight in class_weight_dict.items():
        print(f"类别 {cls}: {weight:.6f}")

    # =========================
    # 5. 只对训练集进行重复过采样
    # =========================
    X_train_balanced, y_train_balanced = oversample_training_set(X_train, y_train_raw)

    print_class_distribution("过采样后的训练集", y_train_balanced)

    # =========================
    # 6. ELM 模型测试
    # =========================
    print("\n开始模型对比实验：")

    elm = OptimizedELM(n_hidden=N_HIDDEN)

    # ELM 使用类别权重
    elm_sample_weight = np.array([
        class_weight_dict[label] for label in y_train_balanced
    ])

    elm_result = evaluate_model(
        model=elm,
        X_train=X_train_balanced,
        y_train=y_train_balanced,
        X_test=X_test,
        y_test=y_test,
        model_name="ELM",
        fit_kwargs={"sample_weight": elm_sample_weight}
    )

    # =========================
    # 7. SVM 模型测试
    # =========================
    svm = SVC()

    svm_result = evaluate_model(
        model=svm,
        X_train=X_train_balanced,
        y_train=y_train_balanced,
        X_test=X_test,
        y_test=y_test,
        model_name="SVM"
    )

    # =========================
    # 8. RandomForest 模型测试
    # =========================
    rf = RandomForestClassifier(random_state=RANDOM_STATE)

    rf_result = evaluate_model(
        model=rf,
        X_train=X_train_balanced,
        y_train=y_train_balanced,
        X_test=X_test,
        y_test=y_test,
        model_name="RF"
    )

    # =========================
    # 9. ELM 边缘端推理效率测试
    # =========================
    avg_infer_time_ms, model_size_kb = test_elm_inference_efficiency(
        elm=elm,
        scaler=scaler,
        X_test=X_test
    )

    # =========================
    # 10. 汇总结果
    # =========================
    print("\n模型对比结果汇总：")
    print("模型\tAccuracy\tF1-score\t训练与测试总耗时/s")
    for result in [elm_result, svm_result, rf_result]:
        print(
            f"{result['model']}\t"
            f"{result['accuracy']:.4f}\t\t"
            f"{result['f1_score']:.4f}\t\t"
            f"{result['time']:.4f}"
        )

    print("\nELM推理效率结果：")
    print("测试指标\t\t测试结果")
    print(f"输入特征维度\t\t{X_test.shape[1]}")
    print(f"隐含层节点数\t\t{N_HIDDEN}")
    print(f"模型文件大小/KB\t\t{model_size_kb:.2f}")
    print(f"单样本平均推理时间/ms\t{avg_infer_time_ms:.6f}")


if __name__ == "__main__":
    main()