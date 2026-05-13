# gui.py

import customtkinter as ctk
import numpy as np
import threading
import time

from fontTools.cffLib import width
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox
import traceback

from win32comext.ifilter import ifilter

from edge.detector import predict_signal, collect_signals, check_model_update, current_version, collect_signals_from_paderborn

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("轴承故障诊断系统")
root.geometry("1600x900")

# =========================
# 全局常量
# =========================
fault_labels = [
    "Normal",
    "12k Drive Ball",
    "12k Drive Inner Race",
    "12k Drive Outer Race",
    "12k Fan Ball",
    "12k Fan Inner Race",
    "12k Fan Outer Race",
    "48k Drive Ball",
    "48k Drive Inner Race",
    "48k Drive Outer Race"
]
fault_labels_zh = [
    "正常",
    "12k 驱动端 球故障",
    "12k 驱动端 内圈故障",
    "12k 驱动端 外圈故障",
    "12k 风扇端 球故障",
    "12k 风扇端 内圈故障",
    "12k 风扇端 外圈故障",
    "48k 驱动端 球故障",
    "48k 驱动端 内圈故障",
    "48k 驱动端 外圈故障"
]
client_options = ["客户端0", "客户端1", "客户端2"]
signal_options = ["信号0", "信号1"]

# =========================
# 左侧控制区
# =========================
left_frame = ctk.CTkFrame(root, width=400)
left_frame.pack(side="left", fill="y", padx=10, pady=10)

title_label = ctk.CTkLabel(left_frame, text="轴承故障诊断系统", font=("微软雅黑", 28, "bold"))
title_label.pack(pady=10)
sub_label = ctk.CTkLabel(left_frame, text="基于ELM的云边协同诊断", font=("微软雅黑", 14))
sub_label.pack(pady=5)

status_label = ctk.CTkLabel(left_frame, text="等待检测", font=("微软雅黑", 24, "bold"), text_color="cyan")
status_label.pack(pady=10)



# 调试信息
debug_var = ctk.BooleanVar(value=False)
debug_check = ctk.CTkCheckBox(left_frame, text="开启调试信息", variable=debug_var)
debug_check.pack(pady=5)
debug_status_label = ctk.CTkLabel(left_frame, text="调试模式: 关闭", font=("微软雅黑",12))
debug_status_label.pack(pady=2)

def update_debug_status():
    status_label.configure(text=f"检测状态: {'运行中' if running else '等待检测'}")
    debug_status_label.configure(text=f"调试模式: {'开启' if debug_var.get() else '关闭'}")
    root.after(500, update_debug_status)

simulation_mode_var = ctk.StringVar(value='Random')
simulation_mode_combobox = ctk.CTkComboBox(
    left_frame,
    values=['Random/正弦/方波模拟', 'Ball/Inner/Outer 故障模拟', 'Paderborn 信号模拟'],
    variable=simulation_mode_var
)
simulation_mode_combobox.pack(pady=5, fill='x')

# 客户端/信号选择 ComboBox
client_var = ctk.StringVar(value=client_options[0])
client_combobox = ctk.CTkComboBox(left_frame, values=client_options, variable=client_var)
client_combobox.pack(pady=5, fill='x')

signal_var = ctk.StringVar(value=signal_options[0])
signal_combobox = ctk.CTkComboBox(left_frame, values=signal_options, variable=signal_var)
signal_combobox.pack(pady=5, fill='x')

# 日志框
log_box = ctk.CTkTextbox(left_frame, font=("Consolas", 12),width=300,height=300)
log_box.pack(fill="both", expand=True, padx=2, pady=2)

version_label = ctk.CTkLabel(left_frame, text=f"模型版本: {current_version}", font=("微软雅黑", 12))
version_label.pack(pady=5)

# 按钮
button_frame = ctk.CTkFrame(left_frame)
button_frame.pack(pady=10, fill="x")

# 日志功能
def safe_insert_log(msg):
    def insert():
        log_box.insert("end", msg)
        log_box.see("end")
        lines = log_box.get("1.0", "end-1c").split('\n')
        if len(lines) > 1000:
            log_box.delete("1.0", f"{len(lines)-1000}.0")
    root.after(0, insert)

def clear_log():
    log_box.delete("1.0", "end")

def export_log():
    file_path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text Files','*.txt')])
    if file_path:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(log_box.get('1.0', 'end-1c'))
        messagebox.showinfo("导出日志", f"日志已导出到 {file_path}")

# =========================
# 右侧波形和统计图
# =========================
right_frame = ctk.CTkFrame(root)
right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
fig = Figure(figsize=(10,6), dpi=150)
ax = fig.add_subplot(211)
stat_ax = fig.add_subplot(212)
canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)

# 波形更新
running = False

def update_wave(sig):
    ax.clear()
    ax.plot(sig[:500])
    ax.set_title("实时振动信号")
    ax.set_xlabel("样本")
    ax.set_ylabel("振幅")
    canvas.draw()


def update_stat(status_list):
    stat_ax.clear()
    counts = [status_list.count(label) for label in fault_labels]
    stat_ax.bar(range(len(fault_labels_zh)), counts, color='orange')
    stat_ax.set_xticks(range(len(fault_labels_zh)))
    stat_ax.set_xticklabels(fault_labels_zh, rotation=45, ha='right', fontproperties='Microsoft YaHei')
    stat_ax.set_ylabel("数量")
    stat_ax.set_title("故障统计")
    fig.tight_layout()
    canvas.draw()

# =========================
# 检测循环
# =========================
n_signals = 2
signal_options = [f"信号{i}" for i in range(n_signals)]
signal_var = ctk.StringVar(value=signal_options[0])
signal_combobox.configure(values=signal_options, variable=signal_var)
def detect_loop_after():
    global running
    if not running:
        status_label.configure(text="检测状态: 等待检测")
        return
    status_label.configure(text="检测状态: 运行中")
    try:
        # 热更新模型
        updated = check_model_update()
        if updated:
            version_label.configure(text=f"模型版本: {current_version}")
            safe_insert_log("[系统] 检测到新模型，已热更新\n")

        # 根据模拟模式选择信号生成方式
        mode = simulation_mode_var.get()
        if "Random" in mode:
            signals = collect_signals(n_clients=3, n_signals=n_signals, debug=debug_var.get(), simulation_mode=False)
            simulation_flag = False
        elif "Ball" in mode:
            signals = collect_signals(n_clients=3, n_signals=n_signals, debug=debug_var.get(), simulation_mode=True)
            simulation_flag = True
        elif "Paderborn" in mode:
            signals = collect_signals_from_paderborn(n_clients=3, n_signals=n_signals, debug=debug_var.get())
            simulation_flag = False
        else:
            safe_insert_log(f"[Error] 未知模拟模式: {mode}\n")
            signals = collect_signals(n_clients=3, n_signals=n_signals, debug=debug_var.get(), simulation_mode=False)
            simulation_flag = False

        # 确保 signals 有效
        if not signals or len(signals) == 0:
            safe_insert_log("[Error] 信号列表为空，使用随机模式生成代替。\n")
            signals = collect_signals(n_clients=3, n_signals=n_signals, debug=debug_var.get(), simulation_mode=False)

        status_text = []

        # 遍历每个客户端和信号
        for client_id, client_signals in enumerate(signals):
            for sig_idx, sig in enumerate(client_signals):
                result, debug_info = predict_signal(sig, debug=debug_var.get(), simulation_mode=simulation_flag)
                status_text.append(result)
                current_time = time.strftime("%H:%M:%S")
                # 安全获取信号名称，防止越界
                sig_name = signal_options[sig_idx] if sig_idx < len(signal_options) else f"信号{sig_idx}"
                if debug_info:
                    safe_insert_log(f"[{current_time}] {client_options[client_id]} {sig_name}: {debug_info}\n")
                else:
                    safe_insert_log(f"[{current_time}] {client_options[client_id]} {sig_name}: {result}\n")

        # 更新右侧波形显示选中客户端和信号
        current_client_idx = client_options.index(client_var.get())
        current_signal_idx = signal_options.index(signal_var.get()) if signal_var.get() in signal_options else 0

        # 安全检查索引是否有效
        if 0 <= current_client_idx < len(signals) and 0 <= current_signal_idx < len(signals[current_client_idx]):
            selected_signal = signals[current_client_idx][current_signal_idx]
            # 打印调试信息，确认信号是否正常
            print(
                f"[调试] 客户端{current_client_idx} 信号{current_signal_idx} 均值: {np.mean(selected_signal):.3f}, 标准差: {np.std(selected_signal):.3f}")
            update_wave(selected_signal)
        else:
            print(
                f"[错误] 索引越界！客户端索引{current_client_idx}, 信号索引{current_signal_idx}, 实际信号数{len(signals)}")
            # 画一条水平线，避免报错
            update_wave(np.zeros(500))
        update_stat(status_text)
    except Exception as e:
        safe_insert_log(f"[Error] 获取信号失败: {e}\n")

    root.after(1000, detect_loop_after)

# 启动/停止

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

# 按钮绑定
start_btn = ctk.CTkButton(button_frame, text="开始检测", font=("微软雅黑",16), command=start_detection)
start_btn.pack(side="top", fill="x", pady=5)
stop_btn = ctk.CTkButton(button_frame, text="停止检测", font=("微软雅黑",16), fg_color="darkred", command=stop_detection)
stop_btn.pack(side="top", fill="x", pady=5)
upload_btn = ctk.CTkButton(button_frame, text="上传新数据", font=("微软雅黑",16), fg_color="darkorange", command=upload_data)
upload_btn.pack(side="top", fill="x", pady=5)
clear_log_btn = ctk.CTkButton(button_frame, text="清空日志", font=("微软雅黑",16), command=clear_log)
clear_log_btn.pack(side="top", fill="x", pady=5)
export_log_btn = ctk.CTkButton(button_frame, text="导出日志", font=("微软雅黑",16), command=export_log)
export_log_btn.pack(side="top", fill="x", pady=5)

root.mainloop()
