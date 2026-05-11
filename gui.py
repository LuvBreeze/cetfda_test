# gui.py (优化后的完整版本)

import customtkinter as ctk
import numpy as np
import threading
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox
import traceback

from edge.detector import predict_signal, collect_signals, check_model_update, current_version

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("轴承故障诊断系统")
root.geometry("1440x840")

# =========================
# 全局常量
# =========================
fault_labels = [
    "12k Drive Ball", "12k Drive Inner Race", "12k Drive Outer Race",
    "12k Fan Ball", "12k Fan Inner Race", "12k Fan Outer Race",
    "48k Drive Ball", "48k Drive Inner Race", "48k Drive Outer Race"
]

# =========================
# 左侧控制区
# =========================
left_frame = ctk.CTkFrame(root)
left_frame.pack(side="left", fill="y", padx=10, pady=10)

# 标题
title_label = ctk.CTkLabel(left_frame, text="轴承故障诊断系统", font=("微软雅黑", 28, "bold"))
title_label.pack(pady=10)
sub_label = ctk.CTkLabel(left_frame, text="基于ELM的云边协同诊断", font=("微软雅黑", 14))
sub_label.pack(pady=5)

# 状态栏
status_label = ctk.CTkLabel(left_frame, text="等待检测", font=("微软雅黑", 24, "bold"), text_color="cyan")
status_label.pack(pady=10)

# 调试信息复选框
debug_var = ctk.BooleanVar(value=False)
debug_check = ctk.CTkCheckBox(left_frame, text="开启调试信息", variable=debug_var)
debug_check.pack(pady=5)

# 显示当前调试状态
debug_status_label = ctk.CTkLabel(left_frame, text="调试模式: 关闭", font=("微软雅黑",12))
debug_status_label.pack(pady=2)

def update_debug_status():
    debug_status_label.configure(text=f"调试模式: {'开启' if debug_var.get() else '关闭'}")
    root.after(500, update_debug_status)
update_debug_status()

# 日志滚动框
log_frame = ctk.CTkScrollableFrame(left_frame, width=300, height=300)
log_frame.pack(fill="both", expand=True, pady=10)
log_box = ctk.CTkTextbox(log_frame, font=("Consolas", 12),height=300)
log_box.pack(fill="both", expand=True, padx=2, pady=2)

# 模型版本显示
version_label = ctk.CTkLabel(left_frame, text=f"模型版本: {current_version}", font=("微软雅黑", 12))
version_label.pack(pady=5)

# 按钮区
button_frame = ctk.CTkFrame(left_frame)
button_frame.pack(pady=10, fill="x")

# =========================
# 右侧波形图区
# =========================
right_frame = ctk.CTkFrame(root)
right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
fig = Figure(figsize=(8,5), dpi=150)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)

# =========================
# 日志安全更新
# =========================
def safe_insert_log(msg):
    def insert():
        log_box.insert("end", msg)
        log_box.see("end")
        lines = log_box.get("1.0", "end-1c").split('\n')
        if len(lines) > 500:
            log_box.delete("1.0", f"{len(lines)-500}.0")
    root.after(0, insert)

# =========================
# 清空日志
# =========================
def clear_log():
    log_box.delete("1.0", "end")

# =========================
# 导出日志
# =========================
def export_log():
    file_path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text Files','*.txt')])
    if file_path:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(log_box.get('1.0', 'end-1c'))
        messagebox.showinfo("导出日志", f"日志已导出到 {file_path}")

# =========================
# 波形图更新
# =========================
def update_wave(sig):
    ax.clear()
    ax.plot(sig[:500])
    ax.set_title("实时振动信号")
    ax.set_xlabel("样本")
    ax.set_ylabel("振幅")
    fig.tight_layout()
    canvas.draw()

# =========================
# 检测循环
# =========================
running = False

def detect_loop_after():
    if not running:
        return
    try:
        updated = check_model_update()
        if updated:
            version_label.configure(text=f"模型版本: {current_version}")
            safe_insert_log("[系统] 检测到新模型，已热更新\n")

        signals = collect_signals(n_clients=3, n_signals=2)
        status_text = []

        for client_id, client_signals in enumerate(signals):
            for sig_idx, sig in enumerate(client_signals):
                result, debug_info = predict_signal(sig, debug=debug_var.get())
                status_text.append(result)
                current_time = time.strftime("%H:%M:%S")
                safe_insert_log(f"[{current_time}] 客户端{client_id} 信号{sig_idx}: {result}\n")
                if debug_info:
                    safe_insert_log(f"[客户端{client_id} 信号{sig_idx}] {debug_info}\n")
                    safe_insert_log(debug_info)

        # 故障统计
        fault_count = sum(1 for r in status_text if r in fault_labels)
        summary_text = f"总信号: {len(status_text)} | 故障数: {fault_count}"
        color = "red" if fault_count>0 else "green"
        status_label.configure(text=summary_text, text_color=color)

        # 显示第一条信号波形
        update_wave(signals[0][0])

    except Exception as e:
        safe_insert_log(f"[Error] {e}\n{traceback.format_exc()}")

    root.after(1000, detect_loop_after)

# =========================
# 按钮功能
# =========================
start_btn = ctk.CTkButton(button_frame, text="开始检测", font=("微软雅黑",16), command=lambda: start_detection())
start_btn.pack(side="top", fill="x", pady=5)
stop_btn = ctk.CTkButton(button_frame, text="停止检测", font=("微软雅黑",16), fg_color="darkred", command=lambda: stop_detection())
stop_btn.pack(side="top", fill="x", pady=5)
upload_btn = ctk.CTkButton(button_frame, text="上传新数据", font=("微软雅黑",16), fg_color="darkorange", command=lambda: upload_data())
upload_btn.pack(side="top", fill="x", pady=5)
clear_log_btn = ctk.CTkButton(button_frame, text="清空日志", font=("微软雅黑",16), command=clear_log)
clear_log_btn.pack(side="top", fill="x", pady=5)
export_log_btn = ctk.CTkButton(button_frame, text="导出日志", font=("微软雅黑",16), command=export_log)
export_log_btn.pack(side="top", fill="x", pady=5)

# =========================
# 启动/停止函数
# =========================
def start_detection():
    global running
    if not running:
        running = True
        detect_loop_after()

def stop_detection():
    global running
    running = False

def upload_data():
    file_path = filedialog.askopenfilename(filetypes=[("Numpy Files", "*.npy")])
    if not file_path:
        return
    sig = np.load(file_path)
    save_path = "../uploaded_data/new_data.npy"
    np.save(save_path, sig)
    current_time = time.strftime("%H:%M:%S")
    safe_insert_log(f"[{current_time}] 新数据已上传至云端\n")

root.mainloop()
