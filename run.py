import threading
import time
import os
import numpy as np
from cloud.trainer import train_model
from edge.detector import elm, check_model_update, reload_model, collect_signals, predict_signal, VERSION_PATH
from gui import root, start_detection, stop_detection  # 直接使用 gui.py 的窗口和按钮

# =========================
# 云端训练线程
# =========================
def cloud_training_loop(data_path='E:/VScode/Python/毕设/数据集/CRWU', model_path='E:/VScode/Python/毕设/project/models/model.npy'):
    while True:
        # 检查是否有新数据
        upload_file = 'E:/VScode/Python/毕设/project/uploaded_data/new_data.npy'
        if os.path.exists(upload_file):
            print("[云端] 发现新数据，开始训练...")
            new_data = np.load(upload_file)
            train_model(data_path=data_path, extra_data=new_data, save_path=model_path)
            # 更新版本号
            if not os.path.exists(VERSION_PATH):
                version = 1
            else:
                with open(VERSION_PATH, 'r') as f:
                    version = int(f.read()) + 1
            with open(VERSION_PATH, 'w') as f:
                f.write(str(version))
            print(f"[云端] 模型训练完成，版本更新为 v{version}")
            os.remove(upload_file)
        time.sleep(1)

# =========================
# 启动云端训练线程
# =========================
cloud_thread = threading.Thread(target=cloud_training_loop, daemon=True)
cloud_thread.start()

# =========================
# 启动 GUI
# =========================
# GUI 内部会调用 detector.py 的 check_model_update、predict_signal
start_detection()
root.mainloop()