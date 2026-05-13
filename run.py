# -*- coding: utf-8 -*-
# 系统启动入口
# 功能：启动云端训练线程和GUI界面，管理数据上传队列
# 依赖：threading, time, os, shutil, queue, numpy

import threading
import time
import os
import shutil
from queue import Queue
import numpy as np

from cloud.trainer import train_model
from edge.detector import elm, check_model_update, reload_model, collect_signals, predict_signal, VERSION_PATH
from gui import root, start_detection, stop_detection, ui_destroyed

# =========================
# 数据队列配置
# =========================
data_queue = Queue()  # 数据上传队列
HISTORY_DIR = './uploaded_data/history'  # 历史数据目录
os.makedirs(HISTORY_DIR, exist_ok=True)  # 创建目录（不存在则创建）

# =========================
# 云端训练线程
# =========================
def cloud_training_worker():
    """
    云端训练工作线程
    从队列中获取新数据，训练模型并更新版本号
    """
    while not ui_destroyed:
        # 阻塞等待队列数据
        new_data_dict = data_queue.get()
        print("[云端] 发现新数据，开始训练...")
        # 训练模型
        train_model(extra_data=new_data_dict)

        # 更新版本号
        if not os.path.exists(VERSION_PATH):
            version = 1
        else:
            with open(VERSION_PATH, 'r') as f:
                version = int(f.read()) + 1
        with open(VERSION_PATH, 'w') as f:
            f.write(str(version))
        print(f"[云端] 模型训练完成，版本更新为 v{version}")

        # 移动处理过的数据到历史文件夹
        if 'file_path' in new_data_dict and os.path.exists(new_data_dict['file_path']):
            shutil.move(new_data_dict['file_path'], HISTORY_DIR)

        # 标记任务完成
        data_queue.task_done()

# 启动云端训练线程（守护线程）
threading.Thread(target=cloud_training_worker, daemon=True).start()

# =========================
# 数据上传回调函数
# =========================
def on_new_data_uploaded(file_path):
    """
    新数据上传回调函数
    将数据加入训练队列
    :param file_path: 上传的npy文件路径
    """
    new_data = np.load(file_path)
    # 包装数据并加入队列
    data_queue.put({'X': new_data, 'y': np.zeros(len(new_data)), 'file_path': file_path})

# =========================
# 启动GUI
# =========================
start_detection()  # 启动检测
root.mainloop()    # 启动主窗口