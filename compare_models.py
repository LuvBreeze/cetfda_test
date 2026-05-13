# -*- coding: utf-8 -*-
# 模型对比测试
# 功能：对比ELM、SVM、RandomForest在CWRU数据集上的性能
# 依赖：time, sklearn, data_loader, model

import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from data_loader import load_data
from model import OptimizedELM

# =========================
# 加载并预处理数据
# =========================
# 加载CWRU数据集
X, y = load_data('./CRWU')

# 划分训练集/测试集（8:2）
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 分层抽样，保持类别比例
)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# ELM模型测试
# =========================
start = time.time()  # 计时开始
# 初始化并训练ELM
elm = OptimizedELM(n_hidden=1000)
elm.fit(X_train, y_train)
# 预测
pred = elm.predict(X_test)
end = time.time()  # 计时结束
# 输出结果
print(
    f'ELM: Acc={accuracy_score(y_test,pred):.4f}, '
    f'F1={f1_score(y_test,pred,average="weighted"):.4f}, '
    f'Time={end-start:.4f}s'
)

# =========================
# SVM模型测试
# =========================
start = time.time()
# 初始化并训练SVM
svm = SVC()
svm.fit(X_train, y_train)
# 预测
pred = svm.predict(X_test)
end = time.time()
# 输出结果
print(
    f'SVM: Acc={accuracy_score(y_test,pred):.4f}, '
    f'F1={f1_score(y_test,pred,average="weighted"):.4f}, '
    f'Time={end-start:.4f}s'
)

# =========================
# 随机森林模型测试
# =========================
start = time.time()
# 初始化并训练RandomForest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
# 预测
pred = rf.predict(X_test)
end = time.time()
# 输出结果
print(
    f'RF: Acc={accuracy_score(y_test,pred):.4f}, '
    f'F1={f1_score(y_test,pred,average="weighted"):.4f}, '
    f'Time={end-start:.4f}s'
)