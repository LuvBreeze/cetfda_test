# -*- coding: utf-8 -*-
# 云端模型更新器
# 功能：监听新数据上传，自动重新训练模型并更新版本号
# 依赖：os, time, numpy, trainer, model

import os
import time
import numpy as np
from trainer import train_model
from model import OptimizedELM

# 路径配置
UPLOAD_PATH = '../uploaded_data/new_data.npy'  # 新数据上传路径
VERSION_PATH = '../models/version.txt'  # 模型版本文件路径
MODEL_PATH = '../models/model.npy'  # 模型保存路径

print('云端监听启动...')

# =========================
# 版本管理函数
# =========================
def get_version():
    """
    获取当前模型版本号
    :return: 版本号（整数），文件不存在时返回0
    """
    if not os.path.exists(VERSION_PATH):
        return 0
    with open(VERSION_PATH, 'r') as f:
        return int(f.read())

def set_version(v):
    """
    设置模型版本号
    :param v: 新的版本号（整数）
    """
    with open(VERSION_PATH, 'w') as f:
        f.write(str(v))

# 初始化当前版本号
current_version = get_version()

# =========================
# 主监听循环
# =========================
while True:
    # 检查是否有新数据上传
    if os.path.exists(UPLOAD_PATH):
        print('发现新数据！')
        # 加载新数据
        new_data = np.load(UPLOAD_PATH)
        print('新数据长度:', len(new_data))

        # 调用训练函数重新训练模型
        train_model(extra_data=new_data, save_path=MODEL_PATH)
        print('模型训练完成！')

        # 更新版本号
        current_version += 1
        set_version(current_version)
        print(f'模型版本已更新为 v{current_version}')

        # 删除已处理的上传文件
        os.remove(UPLOAD_PATH)

    # 每秒检查一次
    time.sleep(1)