# -*- coding: utf-8 -*-
# t-SNE特征可视化
# 功能：对CWRU数据集的特征进行t-SNE降维并可视化
# 依赖：data_loader, sklearn, matplotlib, numpy

from data_loader import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np

# =========================
# 加载并预处理数据
# =========================
# 加载CWRU数据集特征和标签
X, y = load_data('../数据集/CRWU')

# 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 抽样（避免计算量过大）
n_samples = 2000
idx = np.random.choice(len(X), n_samples, replace=False)
X = X[idx]
y = y[idx]

# =========================
# t-SNE降维
# =========================
print('正在进行t-SNE降维...')
# 初始化t-SNE模型
tsne = TSNE(
    n_components=2,  # 降维到2维
    random_state=42, # 随机种子
    perplexity=30    # 困惑度（影响聚类效果）
)

# 执行降维
X_tsne = tsne.fit_transform(X)

# =========================
# 可视化绘制
# =========================
plt.figure(figsize=(8,6))

# 颜色和标签配置
colors = ['blue', 'red', 'green', 'purple']
labels = ['Normal', 'Ball', 'Inner', 'Outer']

# 绘制散点图
for i in range(4):
    plt.scatter(
        X_tsne[y == i, 0],  # x轴坐标
        X_tsne[y == i, 1],  # y轴坐标
        c=colors[i],        # 颜色
        label=labels[i],    # 标签
        s=10                # 点大小
    )

# 添加图例和标题
plt.legend()
plt.title('t-SNE Visualization')

# 保存图片
plt.savefig('./tsne.png', dpi=300)

# 显示图片
plt.show()