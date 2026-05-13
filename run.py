# -*- coding: utf-8 -*-
import threading
import time
import os
import numpy as np
from cloud.trainer import train_model
from edge.detector import VERSION_PATH, MODEL_PATH
from gui import run_gui

# 路径配置
BASE = os.path.dirname(os.path.abspath(__file__))
NEW_DATA_PATH = os.path.join(BASE, "uploaded_data/new_data.npy")
os.makedirs("uploaded_data", exist_ok=True)



def cloud_listener_worker():
    while True:
        if os.path.exists(NEW_DATA_PATH):
            print(f"监听到文件：{NEW_DATA_PATH}")
            new_data = np.load(NEW_DATA_PATH, allow_pickle=True)
            print("文件加载成功！")
            # 版本号更新代码
            ver = 0.0
            if os.path.exists(VERSION_PATH):
                with open(VERSION_PATH, 'r') as f:
                    ver = float(f.read())
            print(f"原版本号：{ver}")
            ver += 0.1
            with open(VERSION_PATH, 'w') as f:
                f.write(f"{ver:.1f}")
            print(f"新版本号：{ver}，已写入文件！")
            os.remove(NEW_DATA_PATH)
            print("文件已删除！")
        time.sleep(0.5)


# 启动线程
threading.Thread(target=cloud_listener_worker, daemon=True).start()

# 启动GUI
run_gui()