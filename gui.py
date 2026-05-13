# -*- coding: utf-8 -*-
# 轴承故障诊断系统GUI界面
# 功能：提供可视化界面，实现信号采集、故障诊断、结果展示、日志管理等功能
# 依赖：customtkinter, numpy, matplotlib, win32comext, edge.detector等

import customtkinter as ctk
import numpy as np
import threading
import time

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox
import traceback

# 导入边缘端检测模块
from edge.detector import predict_signal, collect_signals, check_model_update, current_version, \
    collect_signals_from_paderborn

# 配置matplotlib中文显示
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文显示字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置CustomTkinter外观
ctk.set_appearance_mode("dark")  # 深色模式
ctk.set_default_color_theme("blue")  # 蓝色主题

# 主窗口初始化
root = ctk.CTk()
root.title("轴承故障诊断系统")
root.geometry("1600x900")

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
    """
    定时更新调试模式状态和检测状态显示
    每500ms刷新一次
    """
    if ui_destroyed:
        return
    status_label.configure(text=f"检测状态: {'运行中' if running else '等待检测'}")
    debug_status_label.configure(text=f"调试模式: {'开启' if debug_var.get() else '关闭'}")
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
version_label = ctk.CTkLabel(left_frame, text=f"模型版本: {current_version}", font=("微软雅黑", 12))
version_label.pack(pady=5)

# 按钮框架
button_frame = ctk.CTkFrame(left_frame)
button_frame.pack(pady=10, fill="x")


# =========================
# 日志管理功能
# =========================
def safe_insert_log(msg):
    """
    线程安全的日志插入函数
    避免UI线程阻塞，限制日志最大行数为1000行
    :param msg: 要插入的日志信息字符串
    """

    def insert():
        if ui_destroyed:
            return
        log_box.insert("end", msg)  # 插入日志到末尾
        log_box.see("end")  # 滚动到最新日志
        # 限制日志行数，超过1000行时删除最早的行
        lines = log_box.get("1.0", "end-1c").split('\n')
        if len(lines) > 1000:
            log_box.delete("1.0", f"{len(lines) - 1000}.0")

    # 使用after方法确保在UI线程执行
    root.after(0, insert)


def clear_log():
    """清空日志文本框"""
    if ui_destroyed:
        return
    log_box.delete("1.0", "end")


def export_log():
    """导出日志到本地文本文件"""
    # 弹出保存文件对话框
    file_path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text Files', '*.txt')])
    if file_path:
        # 写入日志内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(log_box.get('1.0', 'end-1c'))
        # 提示导出成功
        messagebox.showinfo("导出日志", f"日志已导出到 {file_path}")


# =========================
# 右侧波形和统计图布局
# =========================
# 右侧主框架
right_frame = ctk.CTkFrame(root)
right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# 创建matplotlib图表
fig = Figure(figsize=(10, 6), dpi=150)
ax = fig.add_subplot(211)  # 上半部分：实时振动信号波形
stat_ax = fig.add_subplot(212)  # 下半部分：故障统计柱状图
# 将图表嵌入tkinter窗口
canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)

# =========================
# 波形和统计更新函数
# =========================
# 检测运行状态标记
running = False


def update_wave(sig):
    """
    更新实时振动信号波形图
    :param sig: 振动信号数组
    """
    if ui_destroyed:
        return
    ax.clear()  # 清空原有绘图
    ax.plot(sig[:500])  # 绘制前500个样本点
    ax.set_title("实时振动信号")
    ax.set_xlabel("样本")
    ax.set_ylabel("振幅")
    canvas.draw()  # 刷新画布


def update_stat(status_list):
    """
    更新故障统计柱状图
    :param status_list: 故障诊断结果列表
    """
    if ui_destroyed:
        return
    stat_ax.clear()  # 清空原有绘图
    # 统计各类故障数量
    counts = [status_list.count(label) for label in fault_labels]
    # 绘制柱状图
    stat_ax.bar(range(len(fault_labels_zh)), counts, color='orange')
    stat_ax.set_xticks(range(len(fault_labels_zh)))
    stat_ax.set_xticklabels(fault_labels_zh, rotation=45, ha='right', fontproperties='Microsoft YaHei')
    stat_ax.set_ylabel("数量")
    stat_ax.set_title("故障统计")
    fig.tight_layout()  # 自动调整布局
    canvas.draw()  # 刷新画布


# =========================
# 信号数量配置
# =========================
n_signals = 2  # 每个客户端的信号数量
signal_options = [f"信号{i}" for i in range(n_signals)]  # 更新信号选项列表
signal_var = ctk.StringVar(value=signal_options[0])
signal_combobox.configure(values=signal_options, variable=signal_var)  # 更新下拉框选项


# =========================
# 检测循环函数
# =========================
def detect_loop_after():
    """
    检测循环主函数（使用after实现非阻塞循环）
    1. 检查模型更新
    2. 根据模拟模式采集信号
    3. 故障诊断预测
    4. 更新波形和统计图表
    5. 记录日志
    """
    global running
    if ui_destroyed or not running:
        return

    try:
        status_label.configure(text="检测状态: 运行中")
    except:
        return

    try:
        # 检查并热更新模型
        updated = check_model_update()
        if updated:
            version_label.configure(text=f"模型版本: {current_version}")
            safe_insert_log("[系统] 检测到新模型，已热更新\n")

        # 根据模拟模式选择信号生成方式
        mode = simulation_mode_var.get()
        if "Random" in mode:
            # 随机/正弦/方波模拟信号
            signals = collect_signals(n_clients=3, n_signals=n_signals, debug=debug_var.get(), simulation_mode=False)
            simulation_flag = False
        elif "Ball" in mode:
            # 故障类型模拟信号（Ball/Inner/Outer）
            signals = collect_signals(n_clients=3, n_signals=n_signals, debug=debug_var.get(), simulation_mode=True)
            simulation_flag = True
        elif "Paderborn" in mode:
            # Paderborn数据集信号模拟
            signals = collect_signals_from_paderborn(n_clients=3, n_signals=n_signals, debug=debug_var.get())
            simulation_flag = False
        else:
            # 未知模式，默认使用随机模拟
            safe_insert_log(f"[Error] 未知模拟模式: {mode}\n")
            signals = collect_signals(n_clients=3, n_signals=n_signals, debug=debug_var.get(), simulation_mode=False)
            simulation_flag = False

        # 信号有效性检查，为空则生成随机信号
        if not signals or len(signals) == 0:
            safe_insert_log("[Error] 信号列表为空，使用随机模式生成代替。\n")
            signals = collect_signals(n_clients=3, n_signals=n_signals, debug=debug_var.get(), simulation_mode=False)

        # 存储诊断结果
        status_text = []

        # 遍历每个客户端和信号进行诊断
        for client_id, client_signals in enumerate(signals):
            for sig_idx, sig in enumerate(client_signals):
                # 故障预测
                result, debug_info = predict_signal(sig, debug=debug_var.get(), simulation_mode=simulation_flag)
                status_text.append(result)
                current_time = time.strftime("%H:%M:%S")
                # 安全获取信号名称（防止索引越界）
                sig_name = signal_options[sig_idx] if sig_idx < len(signal_options) else f"信号{sig_idx}"
                # 记录调试/普通日志
                if debug_info:
                    safe_insert_log(f"[{current_time}] {client_options[client_id]} {sig_name}: {debug_info}\n")
                else:
                    safe_insert_log(f"[{current_time}] {client_options[client_id]} {sig_name}: {result}\n")

        # 获取当前选中的客户端和信号索引
        current_client_idx = client_options.index(client_var.get())
        current_signal_idx = signal_options.index(signal_var.get()) if signal_var.get() in signal_options else 0

        # 安全检查索引有效性，防止越界
        if 0 <= current_client_idx < len(signals) and 0 <= current_signal_idx < len(signals[current_client_idx]):
            selected_signal = signals[current_client_idx][current_signal_idx]
            # 调试信息：打印信号统计特征
            print(
                f"[信息] 客户端{current_client_idx} 信号{current_signal_idx} 均值: {np.mean(selected_signal):.3f}, 标准差: {np.std(selected_signal):.3f}")
            update_wave(selected_signal)  # 更新波形图
        else:
            # 索引越界时打印错误信息，并绘制空波形
            print(
                f"[错误] 索引越界！客户端索引{current_client_idx}, 信号索引{current_signal_idx}, 实际信号数{len(signals)}")
            update_wave(np.zeros(500))

        # 更新故障统计图表
        update_stat(status_text)

    except Exception as e:
        # 异常捕获并记录日志
        safe_insert_log(f"[Error] 获取信号失败: {e}\n")

    # 1秒后再次执行循环
    root.after(1000, detect_loop_after)


# =========================
# 控制按钮功能函数
# =========================
def start_detection():
    """启动故障检测循环"""
    global running
    if not running:
        running = True
        detect_loop_after()


def stop_detection():
    """停止故障检测循环"""
    global running
    running = False


def upload_data():
    """上传新的振动数据到云端（npy格式）"""
    # 弹出文件选择对话框
    file_path = filedialog.askopenfilename(filetypes=[("Numpy Files", "*.npy")])
    if not file_path:
        return
    # 加载npy数据
    sig = np.load(file_path)
    # 保存到云端上传目录
    save_path = "../uploaded_data/new_data.npy"
    np.save(save_path, sig)
    # 记录日志
    current_time = time.strftime("%H:%M:%S")
    safe_insert_log(f"[{current_time}] 新数据已上传至云端\n")


# =========================
# 按钮绑定与布局
# =========================
# 开始检测按钮
start_btn = ctk.CTkButton(button_frame, text="开始检测", font=("微软雅黑", 16), command=start_detection)
start_btn.pack(side="top", fill="x", pady=5)

# 停止检测按钮
stop_btn = ctk.CTkButton(button_frame, text="停止检测", font=("微软雅黑", 16), fg_color="darkred",
                         command=stop_detection)
stop_btn.pack(side="top", fill="x", pady=5)

# 上传数据按钮
upload_btn = ctk.CTkButton(button_frame, text="上传新数据", font=("微软雅黑", 16), fg_color="darkorange",
                           command=upload_data)
upload_btn.pack(side="top", fill="x", pady=5)

# 清空日志按钮
clear_log_btn = ctk.CTkButton(button_frame, text="清空日志", font=("微软雅黑", 16), command=clear_log)
clear_log_btn.pack(side="top", fill="x", pady=5)

# 导出日志按钮
export_log_btn = ctk.CTkButton(button_frame, text="导出日志", font=("微软雅黑", 16), command=export_log)
export_log_btn.pack(side="top", fill="x", pady=5)

# 启动状态更新循环
update_debug_status()

# 启动主窗口消息循环
root.mainloop()