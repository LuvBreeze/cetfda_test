# -*- coding: utf-8 -*-
# 轴承故障诊断系统GUI界面
# 功能：提供可视化界面，实现信号采集、故障诊断、结果展示、日志管理等功能
# 依赖：customtkinter, numpy, matplotlib, win32comext, edge.detector等

import customtkinter as ctk
import numpy as np
import threading
import time
import os

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox
import traceback

# 导入边缘端检测模块
from edge.detector import predict_signal, collect_signals, check_model_update, current_version, \
    collect_signals_from_paderborn, get_model_version

# 配置matplotlib中文显示
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文显示字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

UPLOAD_DIR = os.path.join(PROJECT_ROOT, "./project/uploaded_data")
UPLOAD_PATH = os.path.join(UPLOAD_DIR, "new_data.npy")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 设置CustomTkinter外观
ctk.set_appearance_mode("dark")  # 深色模式
ctk.set_default_color_theme("blue")  # 蓝色主题

# =========================
# 全局常量定义
# =========================
# 故障标签（英文）
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

# 故障标签（中文）
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

# 客户端选项
client_options = ["客户端0", "客户端1", "客户端2"]
# 信号选项（初始值）
signal_options = ["信号0", "信号1"]

# 全局变量
root = None
running = False
ui_destroyed = False

# =========================
# 启动图形化
# =========================
def run_gui():
    global root, running, ui_destroyed
    n_signals = 2
    signal_options = [f"信号{i}" for i in range(n_signals)]
    # 主窗口初始化
    root = ctk.CTk()
    root.title("轴承故障诊断系统")
    root.geometry("1600x900")

    # =========================
    # 左侧控制区布局
    # =========================
    # 左侧主框架
    left_frame = ctk.CTkFrame(root, width=400)
    left_frame.pack(side="left", fill="y", padx=10, pady=10)

    # 标题标签
    title_label = ctk.CTkLabel(left_frame, text="轴承故障诊断系统", font=("微软雅黑", 28, "bold"))
    title_label.pack(pady=10)

    # 副标题标签
    sub_label = ctk.CTkLabel(left_frame, text="基于ELM的云边协同诊断", font=("微软雅黑", 14))
    sub_label.pack(pady=5)

    # 检测状态标签
    status_label = ctk.CTkLabel(left_frame, text="等待检测", font=("微软雅黑", 24, "bold"), text_color="cyan")
    status_label.pack(pady=10)

    # 调试模式复选框
    debug_var = ctk.BooleanVar(value=False)
    debug_check = ctk.CTkCheckBox(left_frame, text="开启调试信息", variable=debug_var)
    debug_check.pack(pady=5)

    # 调试状态显示标签
    debug_status_label = ctk.CTkLabel(left_frame, text="调试模式: 关闭", font=("微软雅黑", 12))
    debug_status_label.pack(pady=2)

    # 界面是否已销毁标记
    ui_destroyed = False

    def on_closing():
        """窗口关闭时的安全处理函数"""
        global running, ui_destroyed
        running = False
        ui_destroyed = True
        root.destroy()

    # 绑定窗口关闭事件
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # 更新调试状态和检测状态
    def update_debug_status():
        if ui_destroyed:
            return
        status_label.configure(text=f"检测状态: {'运行中' if running else '等待检测'}")
        debug_status_label.configure(text=f"调试模式: {'开启' if debug_var.get() else '关闭'}")
        version_label.configure(text=f"模型版本: v{get_model_version():.1f}")
        root.after(500, update_debug_status)

    # 模拟模式选择下拉框
    simulation_mode_var = ctk.StringVar(value='Random')
    simulation_mode_combobox = ctk.CTkComboBox(
        left_frame,
        values=['Random/正弦/方波模拟', 'Ball/Inner/Outer 故障模拟', 'Paderborn 信号模拟'],
        variable=simulation_mode_var
    )
    simulation_mode_combobox.pack(pady=5, fill='x')

    # 客户端选择下拉框
    client_var = ctk.StringVar(value=client_options[0])
    client_combobox = ctk.CTkComboBox(left_frame, values=client_options, variable=client_var)
    client_combobox.pack(pady=5, fill='x')

    # 信号选择下拉框
    signal_var = ctk.StringVar(value=signal_options[0])
    signal_combobox = ctk.CTkComboBox(left_frame, values=signal_options, variable=signal_var)
    signal_combobox.pack(pady=5, fill='x')

    # 日志显示文本框
    log_box = ctk.CTkTextbox(left_frame, font=("Consolas", 12), width=300, height=300)
    log_box.pack(fill="both", expand=True, padx=2, pady=2)

    # 模型版本显示标签
    version_label = ctk.CTkLabel(left_frame, text=f"模型版本: v{get_model_version():.1f}", font=("微软雅黑", 12))
    version_label.pack(pady=5)

    # 按钮框架
    button_frame = ctk.CTkFrame(left_frame)
    button_frame.pack(pady=10, fill="x")

    # =========================
    # 日志管理功能
    # =========================
    def safe_insert_log(msg):
        def insert():
            if ui_destroyed:
                return
            log_box.insert("end", msg)
            log_box.see("end")
            lines = log_box.get("1.0", "end-1c").split('\n')
            if len(lines) > 1000:
                log_box.delete("1.0", f"{len(lines) - 1000}.0")
        root.after(0, insert)

    def clear_log():
        if ui_destroyed:
            return
        log_box.delete("1.0", "end")

    def export_log():
        file_path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text Files', '*.txt')])
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(log_box.get('1.0', 'end-1c'))
            messagebox.showinfo("导出日志", f"日志已导出到 {file_path}")

    # =========================
    # 右侧波形和统计图布局
    # =========================
    right_frame = ctk.CTkFrame(root)
    right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    fig = Figure(figsize=(10, 6), dpi=150)
    ax = fig.add_subplot(211)
    stat_ax = fig.add_subplot(212)
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # =========================
    # 波形和统计更新函数
    # =========================
    def update_wave(sig):
        if ui_destroyed:
            return
        ax.clear()
        ax.plot(sig[:500])
        ax.set_title("实时振动信号")
        ax.set_xlabel("样本")
        ax.set_ylabel("振幅")
        canvas.draw()

    def update_stat(status_list):
        if ui_destroyed:
            return
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
    # 信号数量配置
    # =========================
    signal_var = ctk.StringVar(value=signal_options[0])
    signal_combobox.configure(values=signal_options, variable=signal_var)

    # =========================
    # 检测循环函数
    # =========================
    def detect_loop_after():
        global running
        if ui_destroyed or not running:
            return

        try:
            status_label.configure(text="检测状态: 运行中")
        except:
            return

        try:
            updated = check_model_update()
            if updated:
                version_label.configure(text=f"模型版本: v{current_version:.1f}")
                safe_insert_log("[系统] 检测到新模型，已热更新\n")

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

            if not signals or len(signals) == 0:
                safe_insert_log("[Error] 信号列表为空，使用随机模式生成代替。\n")
                signals = collect_signals(n_clients=3, n_signals=n_signals, debug=debug_var.get(), simulation_mode=False)

            status_text = []
            for client_id, client_signals in enumerate(signals):
                for sig_idx, sig in enumerate(client_signals):
                    result, debug_info = predict_signal(sig, debug=debug_var.get(), simulation_mode=simulation_flag)
                    status_text.append(result)
                    current_time = time.strftime("%H:%M:%S")
                    sig_name = signal_options[sig_idx] if sig_idx < len(signal_options) else f"信号{sig_idx}"
                    if debug_info:
                        safe_insert_log(f"[{current_time}] {client_options[client_id]} {sig_name}: {debug_info}\n")
                    else:
                        safe_insert_log(f"[{current_time}] {client_options[client_id]} {sig_name}: {result}\n")

            current_client_idx = client_options.index(client_var.get())
            current_signal_idx = signal_options.index(signal_var.get()) if signal_var.get() in signal_options else 0

            if 0 <= current_client_idx < len(signals) and 0 <= current_signal_idx < len(signals[current_client_idx]):
                selected_signal = signals[current_client_idx][current_signal_idx]
                update_wave(selected_signal)
            else:
                update_wave(np.zeros(500))

            update_stat(status_text)

        except Exception as e:
            safe_insert_log(f"[Error] 获取信号失败: {e}\n")

        root.after(1000, detect_loop_after)

    # =========================
    # 控制按钮功能函数
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

        try:
            sig = np.load(file_path)
            save_path = UPLOAD_PATH
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, sig)

            current_time = time.strftime("%H:%M:%S")
            safe_insert_log(f"[{current_time}] 新数据已上传至云端，正在训练新模型...\n")
            check_model_update()

        except Exception as e:
            current_time = time.strftime("%H:%M:%S")
            safe_insert_log(f"[{current_time}] 上传失败：{str(e)}\n")

    # =========================
    # 按钮绑定与布局
    # =========================
    start_btn = ctk.CTkButton(button_frame, text="开始检测", font=("微软雅黑", 16), command=start_detection)
    start_btn.pack(side="top", fill="x", pady=5)

    stop_btn = ctk.CTkButton(button_frame, text="停止检测", font=("微软雅黑", 16), fg_color="darkred", command=stop_detection)
    stop_btn.pack(side="top", fill="x", pady=5)

    upload_btn = ctk.CTkButton(button_frame, text="上传新数据", font=("微软雅黑", 16), fg_color="darkorange", command=upload_data)
    upload_btn.pack(side="top", fill="x", pady=5)

    clear_log_btn = ctk.CTkButton(button_frame, text="清空日志", font=("微软雅黑", 16), command=clear_log)
    clear_log_btn.pack(side="top", fill="x", pady=5)

    export_log_btn = ctk.CTkButton(button_frame, text="导出日志", font=("微软雅黑", 16), command=export_log)
    export_log_btn.pack(side="top", fill="x", pady=5)

    # 启动
    update_debug_status()
    root.mainloop()