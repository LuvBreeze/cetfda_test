#tsne_visualization.py（t-SNE可视化）

from data_loader import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np


X, y = load_data('../数据集/CRWU')

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 抽样
n_samples = 2000

idx = np.random.choice(len(X), n_samples, replace=False)

X = X[idx]
y = y[idx]

print('正在进行t-SNE降维...')


tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30
)

X_tsne = tsne.fit_transform(X)


plt.figure(figsize=(8,6))

colors = ['blue', 'red', 'green', 'purple']
labels = ['Normal', 'Ball', 'Inner', 'Outer']

for i in range(4):

    plt.scatter(
        X_tsne[y == i, 0],
        X_tsne[y == i, 1],
        c=colors[i],
        label=labels[i],
        s=10
    )

plt.legend()
plt.title('t-SNE Visualization')

plt.savefig('../figures/tsne.png', dpi=300)

plt.show()