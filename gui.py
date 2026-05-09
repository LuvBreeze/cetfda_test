# gui.py
import customtkinter as ctk
import numpy as np
import threading
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from edge.detector import predict_signal, collect_signals, check_model_update, current_version

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("轴承故障诊断系统")
root.geometry("1200x700")

# =========================
# 左侧控制区
# =========================
left_frame = ctk.CTkFrame(root, width=320)
left_frame.pack(side="left", fill="y", padx=10, pady=10)

title_label = ctk.CTkLabel(left_frame, text="轴承故障诊断系统", font=("微软雅黑", 28, "bold"))
title_label.pack(pady=20)

sub_label = ctk.CTkLabel(left_frame, text="基于ELM的云边协同诊断", font=("微软雅黑", 14))
sub_label.pack(pady=5)

status_label = ctk.CTkLabel(left_frame, text="等待检测", font=("微软雅黑", 24, "bold"), text_color="cyan")
status_label.pack(pady=30)

log_box = ctk.CTkTextbox(left_frame, width=320, height=300, font=("Consolas", 12))
log_box.pack(pady=20)

version_label = ctk.CTkLabel(left_frame, text=f"模型版本: {current_version}", font=("微软雅黑", 12))
version_label.pack(pady=5)

# =========================
# 右侧图像区
# =========================
right_frame = ctk.CTkFrame(root)
right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

fig = Figure(figsize=(8,5), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill="both", expand=True)

# =========================
# 波形图更新
# =========================
def update_wave(sig):
    ax.clear()
    ax.plot(sig[:500])
    ax.set_title("实时振动信号")
    ax.set_xlabel("样本")
    ax.set_ylabel("振幅")
    canvas.draw()

# =========================
# 检测循环
# =========================
running = False

def detect_loop():
    global running
    while running:
        # 热更新模型
        updated = check_model_update()
        if updated:
            version_label.configure(text=f"模型版本: {current_version}")
            log_box.insert("end", "[系统] 检测到新模型，已热更新\n")
            log_box.see("end")

        # 多客户端、多信号采集
        signals = collect_signals(n_clients=3, n_signals=2)
        status_text = []

        # 遍历每个客户端的信号
        for client_id, client_signals in enumerate(signals):
            for sig_idx, sig in enumerate(client_signals):
                result = predict_signal(sig)
                status_text.append(result)
                current_time = time.strftime("%H:%M:%S")
                log_box.insert("end", f"[{current_time}] 客户端{client_id} 信号{sig_idx}: {result}\n")
                log_box.see("end")

        # 状态栏显示摘要：总信号数 + 故障数量
        total_signals = len(status_text)
        fault_count = sum(1 for r in status_text if "Fault" in r)
        summary_text = f"总信号: {total_signals} | 故障数: {fault_count}"
        color = "red" if fault_count > 0 else "green"
        status_label.configure(text=summary_text, text_color=color)

        # 显示波形图（第一条信号）
        update_wave(signals[0][0])

        time.sleep(1)

# =========================
# 按钮功能
# =========================
def start_detection():
    global running
    if not running:
        running = True
        thread = threading.Thread(target=detect_loop)
        thread.daemon = True
        thread.start()

def stop_detection():
    global running
    running = False

def upload_data():
    sig = np.random.randn(2048)
    save_path = "../uploaded_data/new_data.npy"
    np.save(save_path, sig)
    current_time = time.strftime("%H:%M:%S")
    log_box.insert("end", f"[{current_time}] 新数据已上传至云端\n")
    log_box.see("end")

start_btn = ctk.CTkButton(left_frame, text="开始检测", font=("微软雅黑",16), command=start_detection)
start_btn.pack(pady=8)
stop_btn = ctk.CTkButton(left_frame, text="停止检测", font=("微软雅黑",16), fg_color="darkred", command=stop_detection)
stop_btn.pack(pady=8)
upload_btn = ctk.CTkButton(left_frame, text="上传新数据", font=("微软雅黑",16), fg_color="darkorange", command=upload_data)
upload_btn.pack(pady=8)

root.mainloop()