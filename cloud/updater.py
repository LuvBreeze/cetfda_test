# updater.py

import os
import time
import numpy as np
from trainer import train_model
from model import OptimizedELM

UPLOAD_PATH = '../uploaded_data/new_data.npy'
VERSION_PATH = '../models/version.txt'
MODEL_PATH = '../models/model.npy'

print('云端监听启动...')

def get_version():
    if not os.path.exists(VERSION_PATH):
        return 0
    with open(VERSION_PATH, 'r') as f:
        return int(f.read())

def set_version(v):
    with open(VERSION_PATH, 'w') as f:
        f.write(str(v))

current_version = get_version()

while True:
    if os.path.exists(UPLOAD_PATH):
        print('发现新数据！')
        new_data = np.load(UPLOAD_PATH)
        print('新数据长度:', len(new_data))

        # 调用训练函数重新训练模型
        train_model(extra_data=new_data, save_path=MODEL_PATH)
        print('模型训练完成！')

        # 更新版本号
        current_version += 1
        set_version(current_version)
        print(f'模型版本已更新为 v{current_version}')

        os.remove(UPLOAD_PATH)

    time.sleep(1)