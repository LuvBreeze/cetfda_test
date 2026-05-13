# -*- coding: utf-8 -*-
# 优化的ELM模型实现
# 功能：实现带加权最小二乘的极限学习机（ELM）
# 依赖：numpy

import numpy as np

class OptimizedELM:
    """
    优化的极限学习机（ELM）类
    支持加权最小二乘训练，解决类别不平衡问题
    """
    def __init__(self, n_hidden=800):
        """
        初始化ELM模型
        :param n_hidden: 隐藏层节点数
        """
        self.n_hidden = n_hidden  # 隐藏层节点数
        self.W = None             # 输入层到隐藏层权重
        self.b = None             # 隐藏层偏置
        self.beta = None          # 隐藏层到输出层权重

    @staticmethod
    def _sigmoid(x):
        """
        Sigmoid激活函数
        :param x: 输入数组
        :return: 激活后数组
        """
        x = np.clip(x, -500, 500)  # 限制范围避免溢出
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, sample_weight=None):
        """
        训练ELM模型
        :param X: 特征矩阵 (n_samples, n_features)
        :param y: 标签数组 (n_samples,)
        :param sample_weight: 样本权重 (n_samples,) 可选
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))  # 类别数

        # 随机初始化输入层到隐藏层的权重和偏置
        self.W = np.random.randn(n_features, self.n_hidden)
        self.b = np.random.randn(self.n_hidden)

        # 计算隐藏层输出矩阵H
        H = self._sigmoid(X @ self.W + self.b)
        # 标签独热编码
        Y = np.eye(n_classes)[y.astype(int)]

        # =========================
        # 加权最小二乘求解beta
        # =========================
        if sample_weight is not None:
            # 矢量化加权计算（避免创建巨型对角矩阵）
            sw = sample_weight[:, np.newaxis]  # 扩展维度 (n_samples, 1)
            H_weighted = H * np.sqrt(sw)       # 每行乘以权重的平方根
            Y_weighted = Y * np.sqrt(sw)       # 标签矩阵同样加权
            # 伪逆求解
            self.beta = np.linalg.pinv(H_weighted) @ Y_weighted
        else:
            # 普通最小二乘求解
            self.beta = np.linalg.pinv(H) @ Y

    def predict(self, X):
        """
        模型预测
        :param X: 特征矩阵 (n_samples, n_features)
        :return: 预测标签数组 (n_samples,)
        """
        # 计算隐藏层输出
        H = self._sigmoid(X @ self.W + self.b)
        # 计算输出层结果
        out = H @ self.beta
        # 返回最大概率的类别
        return np.argmax(out, axis=1)