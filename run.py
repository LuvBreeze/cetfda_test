# run.py

import threading
import time
import os
import shutil
from queue import Queue
import numpy as np

from cloud.trainer import train_model
from edge.detector import elm, check_model_update, reload_model, collect_signals, predict_signal, VERSION_PATH
from gui import root, start_detection, stop_detection

# =========================
# 队列管理增量数据
# =========================
data_queue = Queue()
HISTORY_DIR = './uploaded_data/history'
os.makedirs(HISTORY_DIR, exist_ok=True)

# =========================
# 云端训练线程
# =========================
def cloud_training_worker():
    while True:
        new_data_dict = data_queue.get()  # 阻塞直到有数据
        print("[云端] 发现新数据，开始训练...")
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

        data_queue.task_done()

# 启动云端训练线程
threading.Thread(target=cloud_training_worker, daemon=True).start()

# =========================
# 数据上传回调
# =========================
def on_new_data_uploaded(file_path):
    new_data = np.load(file_path)
    data_queue.put({'X': new_data, 'y': np.zeros(len(new_data)), 'file_path': file_path})  # 使用 dict 包装

# =========================
# 启动 GUI
# =========================
start_detection()
root.mainloop()