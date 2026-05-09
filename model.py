# models.py
import numpy as np


class OptimizedELM:

    def __init__(self, n_hidden=500):

        self.n_hidden = n_hidden

        self.W = None
        self.b = None
        self.beta = None

    def _sigmoid(self, x):

        x = np.clip(x, -500, 500)  # 防止 exp 溢出
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):

        n_samples, n_features = X.shape

        n_classes = len(np.unique(y))

        self.W = np.random.randn(n_features, self.n_hidden)

        self.b = np.random.randn(self.n_hidden)

        H = self._sigmoid(X @ self.W + self.b)

        Y = np.eye(n_classes)[y.astype(int)]

        self.beta = np.linalg.pinv(H) @ Y

    def predict(self, X):

        H = self._sigmoid(X @ self.W + self.b)

        out = H @ self.beta

        return np.argmax(out, axis=1)