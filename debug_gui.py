import threading
import time
import numpy as np
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import rcParams
from edge.detector import collect_signals, predict_signal, check_model_update, current_version

# ---------------------
# matplotlib 中文支持
# ---------------------
rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
rcParams['axes.unicode_minus'] = False    # 负号正常显示

# ---------------------
# GUI 设置
# ---------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("轴承故障诊断系统")
root.geometry("1200x700")

# 左侧控制区
left_frame = ctk.CTkFrame(root, width=300)
left_frame.pack(side="left", fill="y", padx=10, pady=10)

title_label = ctk.CTkLabel(left_frame, text="轴承故障诊断系统", font=("微软雅黑", 28, "bold"))
title_label.pack(pady=20)
sub_label = ctk.CTkLabel(left_frame, text="基于ELM的云边协同诊断", font=("微软雅黑", 14))
sub_label.pack(pady=5)

status_label = ctk.CTkLabel(left_frame, text="等待检测", font=("微软雅黑", 16, "bold"), text_color="cyan")
status_label.pack(pady=20)

log_box = ctk.CTkTextbox(left_frame, width=250, height=300, font=("Consolas", 12))
log_box.pack(pady=20)

version_label = ctk.CTkLabel(left_frame, text=f"模型版本: {current_version}", font=("微软雅黑", 12))
version_label.pack(pady=5)

# 右侧波形区
right_frame = ctk.CTkFrame(root)
right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

fig = Figure(figsize=(8,5), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill="both", expand=True)

# ---------------------
# 波形更新函数
# ---------------------
def update_wave(signals):
    ax.clear()
    for sig in signals:
        ax.plot(sig[:500], alpha=0.7)
    ax.set_title("实时振动信号")
    ax.set_xlabel("样本")
    ax.set_ylabel("振幅")
    canvas.draw()

# ---------------------
# 检测循环
# ---------------------
running = False

def detect_loop(debug=False):
    global running
    while running:
        # 检查模型更新
        updated = check_model_update()
        if updated:
            version_label.configure(text=f"模型版本: {current_version}")
            log_box.insert("end", "[系统] 检测到新模型，已热更新\n")
            log_box.see("end")

        # 多客户端、多条信号
        signals = collect_signals(n_clients=3, n_signals=2, debug=debug)
        all_results = []

        for client_id, client_signals in enumerate(signals):
            for sig_idx, sig in enumerate(client_signals):
                result = predict_signal(sig, debug=debug)
                current_time = time.strftime("%H:%M:%S")
                log_box.insert("end", f"[{current_time}] 客户端{client_id} 信号{sig_idx}: {result}\n")
                log_box.see("end")
                all_results.append(result)

        # 统计每类预测数量并换行显示
        total_signals = len(all_results)
        unique, counts = np.unique(all_results, return_counts=True)
        summary = dict(zip(unique, counts))
        fault_count = sum([v for k,v in summary.items() if "Fault" in k])

        lines = [f"总信号: {total_signals} | 故障数: {fault_count}"]
        for k,v in summary.items():
            lines.append(f"{k}: {v}")
        color = "red" if fault_count>0 else "green"
        status_label.configure(text="\n".join(lines), text_color=color)

        # 波形图显示所有信号叠加
        flattened_signals = [sig for client_signals in signals for sig in client_signals]
        update_wave(flattened_signals)

        time.sleep(1)

# ---------------------
# 按钮功能
# ---------------------
def start_detection():
    global running
    if not running:
        running = True
        thread = threading.Thread(target=detect_loop, kwargs={"debug": True})
        thread.daemon = True
        thread.start()

def stop_detection():
    global running
    running = False

start_btn = ctk.CTkButton(left_frame, text="开始检测", font=("微软雅黑", 16), command=start_detection)
start_btn.pack(pady=8)
stop_btn = ctk.CTkButton(left_frame, text="停止检测", font=("微软雅黑", 16), fg_color="darkred", command=stop_detection)
stop_btn.pack(pady=8)

root.mainloop()