# model.py

import numpy as np

class OptimizedELM:
    def __init__(self, n_hidden=800):
        self.n_hidden = n_hidden
        self.W = None
        self.b = None
        self.beta = None

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        self.W = np.random.randn(n_features, self.n_hidden)
        self.b = np.random.randn(self.n_hidden)
        H = self._sigmoid(X @ self.W + self.b)
        Y = np.eye(n_classes)[y.astype(int)]

        if sample_weight is not None:
            # 使用加权最小二乘的矢量化实现，避免创建巨型对角矩阵
            sw = sample_weight[:, np.newaxis]  # shape (n_samples, 1)
            H_weighted = H * np.sqrt(sw)  # 每行乘 sqrt(weight)
            Y_weighted = Y * np.sqrt(sw)  # 同样处理标签矩阵
            self.beta = np.linalg.pinv(H_weighted) @ Y_weighted
        else:
            self.beta = np.linalg.pinv(H) @ Y

    def predict(self, X):
        H = self._sigmoid(X @ self.W + self.b)
        out = H @ self.beta
        return np.argmax(out, axis=1)