# trainer.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from data_loader import load_data
from model import OptimizedELM

MODEL_PATH = r"../models/model.npy"
DATA_PATH = r"../CRWU"

def train_model(data_path=DATA_PATH,
                extra_data=None,
                save_path=MODEL_PATH,
                n_hidden=500):

    models_dir = os.path.dirname(save_path)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    print("加载训练数据...")
    X, y = load_data(data_path)

    # -------------------
    # 计算类别权重
    # -------------------
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    print("类别权重:", dict(zip(classes, class_weights)))

    # -------------------
    # 增量数据处理
    # -------------------
    if extra_data is not None:
        if extra_data.ndim == 1:
            print("[提示] 附加数据为一维信号，跳过增量训练")
        elif extra_data.ndim == 2 and extra_data.shape[1] > 1:
            print(f"附加增量数据: {len(extra_data)} 条")
            X_extra = extra_data[:, :-1]
            y_extra = extra_data[:, -1].astype(int)
            X = np.vstack([X, X_extra])
            y = np.hstack([y, y_extra])
        else:
            print("[警告] 附加数据格式不正确，跳过增量训练")

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 训练 ELM
    print("训练 ELM 模型...")
    elm = OptimizedELM(n_hidden=n_hidden)
    elm.fit(X_train, y_train)  # 可扩展 sample_weight 支持

    # 测试
    pred = elm.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f'训练完成，测试集 Accuracy: {acc:.4f}')

    # 保存模型
    params = {'W': elm.W, 'b': elm.b, 'beta': elm.beta, 'mean': scaler.mean_, 'scale': scaler.scale_}
    np.save(save_path, params)
    print(f'模型保存完成: {save_path}')

    return elm, scaler

if __name__ == "__main__":
    train_model()