# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import librosa # 虽然在此特定脚本中不直接用于声音特征提取，但保留以防未来扩展或理解
import numpy as np
import pandas as pd
import os
import re
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg') # 使用 Agg 后端，避免在无图形界面的服务器上或线程中出错
# import matplotlib.pyplot as plt # 此脚本中不直接绘图

# ===============================================
# 核心处理函数
# ===============================================

def extract_eeg_features_timeseries_gui(eeg_segment_data_full, eeg_sampling_rate_hz, # <-- **已修改参数名**
                                    smoothing_sigma=2.0,
                                    window_ms=11.0, step_ms=11.0, status_callback=None):
    """
    从单个 EEG 分段数据中提取滑动窗口平均特征和对应的时间戳。
    Args:
        eeg_segment_data_full (np.array): N_points x 34 (时间, 标记, 32 EEG)
        eeg_sampling_rate_hz (float): EEG 数据的实际采样率。 <--- **参数名已修改**
        smoothing_sigma (float or None): 高斯平滑 sigma。
        window_ms (float): 滑动窗口大小 (毫秒)。
        step_ms (float): 滑动窗口步长 (毫秒)。
        status_callback (callable): 更新状态的回调。
    Returns:
        tuple: (eeg_features_ts, eeg_times_ts_relative) 或 (None, None)
               eeg_features_ts: (N_eeg_frames, 32)
               eeg_times_ts_relative: (N_eeg_frames,) 相对分段开始的时间戳 (秒)。
    """
    def _log(msg):
        if status_callback: status_callback(msg)
        # else: print(msg) # 用于独立测试

    if eeg_segment_data_full is None or eeg_segment_data_full.ndim != 2 or eeg_segment_data_full.shape[1] != 34:
        _log("错误：EEG 分段数据格式无效，预期形状 (N, 34)。")
        return None, None

    original_eeg_times_sec = eeg_segment_data_full[:, 0]
    eeg_data_raw = eeg_segment_data_full[:, 2:] # EEG 数据 (列 2 到 33)

    if original_eeg_times_sec.shape[0] < 2: # 至少需要两个时间点来估算或使用采样率
        _log("警告：EEG 数据点太少 (<2)，无法提取特征。")
        return None, None
    if eeg_sampling_rate_hz <= 0: # <-- **使用修改后的参数名**
        _log(f"错误：无效的采样率 ({eeg_sampling_rate_hz} Hz)。")
        return None,None

    data_to_process = eeg_data_raw.astype(np.float64)
    if smoothing_sigma and smoothing_sigma > 0:
        # _log(f"  对 EEG 应用高斯平滑 (sigma={smoothing_sigma})...")
        data_to_process = np.apply_along_axis(
            gaussian_filter1d, axis=0, arr=data_to_process,
            sigma=smoothing_sigma, mode='reflect'
        )

    # <-- **使用修改后的参数名 eeg_sampling_rate_hz** -->
    window_points = int(round(window_ms / 1000.0 * eeg_sampling_rate_hz))
    step_points = int(round(step_ms / 1000.0 * eeg_sampling_rate_hz))

    if window_points <= 0: window_points = 1 # 确保窗口至少为1
    if step_points <= 0: step_points = 1   # 确保步长至少为1

    # _log(f"  EEG 特征提取: 采样率={eeg_sampling_rate_hz:.1f} Hz, 窗口={window_points}点 ({window_ms}ms), 步长={step_points}点 ({step_ms}ms)")


    if window_points > data_to_process.shape[0]:
        _log(f"警告：窗口点数 ({window_points}) 大于数据长度 ({data_to_process.shape[0]})。无法提取特征。")
        return None, None

    num_frames = (data_to_process.shape[0] - window_points) // step_points + 1
    if num_frames <= 0:
        _log("警告：计算出的 EEG 特征帧数为 0 或负数。")
        return None, None

    num_eeg_channels = data_to_process.shape[1]
    eeg_features_ts = np.zeros((num_frames, num_eeg_channels))
    eeg_times_ts_absolute = np.zeros(num_frames) # 绝对时间戳

    for i in range(num_frames):
        start = i * step_points
        end = start + window_points
        # 由于 num_frames 的计算方式，end 不应超出 data_to_process.shape[0]
        window_data = data_to_process[start:end, :]
        eeg_features_ts[i, :] = np.mean(window_data, axis=0)
        # 时间戳对应窗口中心
        center_point_index = start + window_points // 2
        if center_point_index < len(original_eeg_times_sec):
            eeg_times_ts_absolute[i] = original_eeg_times_sec[center_point_index]
        else: # Fallback, 理论上不应发生
             eeg_times_ts_absolute[i] = original_eeg_times_sec[-1]


    # 转换为相对于分段开始的时间
    if eeg_times_ts_absolute.size > 0 :
         # 确保时间从0开始，且相对于分段内第一个特征帧的中心时间
         eeg_times_ts_relative = eeg_times_ts_absolute - eeg_times_ts_absolute[0]
    else:
         eeg_times_ts_relative = eeg_times_ts_absolute # 如果为空则保持

    # _log(f"  提取的 EEG 特征形状: {eeg_features_ts.shape}")
    return eeg_features_ts, eeg_times_ts_relative


def align_features_gui(eeg_features_ts, eeg_times_ts_relative,
                   sound_features_ts, sound_times_ts,
                   target_duration_sec=4.0, target_hop_sec=0.011, status_callback=None):
    """
    将 EEG 和声音特征时间序列对齐到共同的时间轴和长度。
    """
    def _log(msg):
        if status_callback: status_callback(msg)
        # else: print(msg)

    if eeg_features_ts is None or sound_features_ts is None:
        _log("错误：EEG 或声音特征为空，无法对齐。")
        return None, None
    if eeg_times_ts_relative.size == 0 or sound_times_ts.size == 0:
        _log("错误：EEG 或声音时间戳为空，无法对齐。")
        return None, None

    # 1. 创建目标时间轴 (从 0 到 target_duration)
    num_target_frames = int(round(target_duration_sec / target_hop_sec))
    if num_target_frames <=0:
        _log(f"错误：计算出的目标帧数 ({num_target_frames}) 无效 (时长={target_duration_sec}, 步长={target_hop_sec})。")
        return None, None
    target_times = np.linspace(0, (num_target_frames - 1) * target_hop_sec, num_target_frames)
    # _log(f"  目标对齐: {num_target_frames} 帧, 步长 {target_hop_sec*1000:.1f}ms, 总时长 {target_times[-1]:.3f}s")


    aligned_eeg = np.zeros((num_target_frames, eeg_features_ts.shape[1]))
    aligned_sound = np.zeros((num_target_frames, sound_features_ts.shape[1]))

    try:
        # 2. 插值 EEG 特征
        # 确保原始时间戳是单调递增的且没有重复（插值函数要求）
        unique_eeg_indices = np.sort(np.unique(eeg_times_ts_relative, return_index=True)[1])
        eeg_times_unique = eeg_times_ts_relative[unique_eeg_indices]
        eeg_features_unique = eeg_features_ts[unique_eeg_indices, :]

        if len(eeg_times_unique) < 2:
            # _log("警告：EEG 特征时间点不足 (<2) 以进行插值。")
            if len(eeg_times_unique) == 1: # 如果只有一个点，则用该值填充所有目标帧
                 for i in range(eeg_features_ts.shape[1]):
                     aligned_eeg[:, i] = eeg_features_unique[0, i]
                 # _log("  EEG 特征仅1点，用该值填充。")
            else: return None, None # 否则失败
        else: # 正常插值
            for i in range(eeg_features_ts.shape[1]):
                interp_func_eeg = interp1d(eeg_times_unique, eeg_features_unique[:, i],
                                           kind='linear', bounds_error=False,
                                           fill_value=(eeg_features_unique[0,i], eeg_features_unique[-1,i]))
                aligned_eeg[:, i] = interp_func_eeg(target_times)

        # 3. 插值声音特征
        unique_sound_indices = np.sort(np.unique(sound_times_ts, return_index=True)[1])
        sound_times_unique = sound_times_ts[unique_sound_indices]
        sound_features_unique = sound_features_ts[unique_sound_indices, :]

        if len(sound_times_unique) < 2:
            # _log("警告：声音特征时间点不足 (<2) 以进行插值。")
            if len(sound_times_unique) == 1:
                 for i in range(sound_features_ts.shape[1]):
                     aligned_sound[:, i] = sound_features_unique[0, i]
                 # _log("  声音特征仅1点，用该值填充。")
            else: return None, None
        else:
            for i in range(sound_features_ts.shape[1]):
                interp_func_sound = interp1d(sound_times_unique, sound_features_unique[:, i],
                                             kind='linear', bounds_error=False,
                                             fill_value=(sound_features_unique[0,i], sound_features_unique[-1,i]))
                aligned_sound[:, i] = interp_func_sound(target_times)

        return aligned_eeg, aligned_sound

    except Exception as e:
        _log(f"特征对齐时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ===============================================
# GUI 主类 (修改后支持批处理)
# ===============================================
class Task3BatchAlignApp:
    def __init__(self, master):
        self.master = master
        master.title("任务3 (批量) - EEG与声音特征对齐工具 v2")
        master.geometry("750x600")

        # --- GUI 变量 ---
        self.eeg_npy_dir_path = tk.StringVar()
        self.artifact_report_path = tk.StringVar()
        self.sound_features_dir_path = tk.StringVar()
        self.output_dir_path = tk.StringVar()

        self.eeg_sampling_rate_var = tk.DoubleVar(value=1001.0)
        self.eeg_smooth_sigma_var = tk.DoubleVar(value=2.0)
        self.eeg_window_ms_var = tk.DoubleVar(value=11.0)
        self.eeg_step_ms_var = tk.DoubleVar(value=11.0)

        self.align_duration_sec_var = tk.DoubleVar(value=4.0)
        self.align_hop_sec_var = tk.DoubleVar(value=0.011)

        # --- 创建控件 ---
        frame = ttk.Frame(master, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)

        row_idx = 0
        # EEG .npy 文件目录选择
        ttk.Label(frame, text="EEG 分段 .npy 文件夹:").grid(row=row_idx, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frame, textvariable=self.eeg_npy_dir_path, width=60).grid(row=row_idx, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(frame, text="浏览...", command=self.browse_eeg_npy_dir).grid(row=row_idx, column=3, padx=5, pady=5)
        frame.columnconfigure(1, weight=1)
        row_idx += 1

        # 伪迹报告 CSV 文件选择
        ttk.Label(frame, text="伪迹报告 CSV 文件:").grid(row=row_idx, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frame, textvariable=self.artifact_report_path, width=60).grid(row=row_idx, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(frame, text="浏览...", command=self.browse_artifact_report).grid(row=row_idx, column=3, padx=5, pady=5)
        row_idx += 1

        # 声音特征文件夹选择
        ttk.Label(frame, text="包含声音特征的文件夹:").grid(row=row_idx, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frame, textvariable=self.sound_features_dir_path, width=60).grid(row=row_idx, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(frame, text="浏览...", command=self.browse_sound_features_dir).grid(row=row_idx, column=3, padx=5, pady=5)
        row_idx += 1

        # 输出文件夹选择
        ttk.Label(frame, text="保存对齐特征的输出文件夹:").grid(row=row_idx, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frame, textvariable=self.output_dir_path, width=60).grid(row=row_idx, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(frame, text="浏览...", command=self.browse_output_dir).grid(row=row_idx, column=3, padx=5, pady=5)
        row_idx += 1

        # EEG 处理参数框架
        eeg_params_frame = ttk.LabelFrame(frame, text="EEG 特征提取参数", padding="10")
        eeg_params_frame.grid(row=row_idx, column=0, columnspan=4, padx=5, pady=10, sticky="ew")
        row_idx += 1

        epf_row = 0; epf_col = 0
        ttk.Label(eeg_params_frame, text="EEG 采样率 (Hz):").grid(row=epf_row, column=epf_col, padx=5, pady=3, sticky="w"); epf_col+=1
        ttk.Entry(eeg_params_frame, textvariable=self.eeg_sampling_rate_var, width=10).grid(row=epf_row, column=epf_col, padx=5, pady=3, sticky="w"); epf_col+=1

        ttk.Label(eeg_params_frame, text="平滑 Sigma (点数):").grid(row=epf_row, column=epf_col, padx=5, pady=3, sticky="w"); epf_col+=1
        ttk.Entry(eeg_params_frame, textvariable=self.eeg_smooth_sigma_var, width=10).grid(row=epf_row, column=epf_col, padx=5, pady=3, sticky="w")
        epf_col=0; epf_row+=1

        ttk.Label(eeg_params_frame, text="窗口大小 (ms):").grid(row=epf_row, column=epf_col, padx=5, pady=3, sticky="w"); epf_col+=1
        ttk.Entry(eeg_params_frame, textvariable=self.eeg_window_ms_var, width=10).grid(row=epf_row, column=epf_col, padx=5, pady=3, sticky="w"); epf_col+=1

        ttk.Label(eeg_params_frame, text="步长 (ms):").grid(row=epf_row, column=epf_col, padx=5, pady=3, sticky="w"); epf_col+=1
        ttk.Entry(eeg_params_frame, textvariable=self.eeg_step_ms_var, width=10).grid(row=epf_row, column=epf_col, padx=5, pady=3, sticky="w")


        # 对齐参数框架
        align_params_frame = ttk.LabelFrame(frame, text="特征对齐参数", padding="10")
        align_params_frame.grid(row=row_idx, column=0, columnspan=4, padx=5, pady=10, sticky="ew")
        row_idx += 1

        apf_row=0; apf_col=0
        ttk.Label(align_params_frame, text="目标分析时长 (秒):").grid(row=apf_row, column=apf_col, padx=5, pady=3, sticky="w"); apf_col+=1
        ttk.Entry(align_params_frame, textvariable=self.align_duration_sec_var, width=10).grid(row=apf_row, column=apf_col, padx=5, pady=3, sticky="w"); apf_col+=1

        ttk.Label(align_params_frame, text="目标时间步长 (秒):").grid(row=apf_row, column=apf_col, padx=5, pady=3, sticky="w"); apf_col+=1
        ttk.Entry(align_params_frame, textvariable=self.align_hop_sec_var, width=10).grid(row=apf_row, column=apf_col, padx=5, pady=3, sticky="w")


        # 启动按钮
        self.start_button = ttk.Button(frame, text="开始批量处理", command=self.start_batch_processing_thread, style='Accent.TButton')
        self.start_button.grid(row=row_idx, column=1, columnspan=2, padx=5, pady=15)
        row_idx += 1

        # 进度条、状态标签、日志区域
        self.progress_bar = ttk.Progressbar(frame, orient="horizontal", length=600, mode="determinate")
        self.progress_bar.grid(row=row_idx, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        row_idx += 1
        self.status_label = ttk.Label(frame, text="准备就绪", anchor="w", justify="left")
        self.status_label.grid(row=row_idx, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        row_idx += 1
        self.log_text = scrolledtext.ScrolledText(frame, height=10, width=80, state='disabled', wrap=tk.WORD)
        self.log_text.grid(row=row_idx, column=0, columnspan=4, padx=5, pady=5, sticky="ewns")
        frame.rowconfigure(row_idx, weight=1)

        # 应用主题
        try:
            from ttkthemes import ThemedStyle
            style = ThemedStyle(master); style.set_theme("arc")
            style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'), padding=6)
        except ImportError:
             s = ttk.Style(); s.configure('TButton', font=('Segoe UI',10),padding=5); s.configure('Accent.TButton', font=('Segoe UI',10,'bold'),padding=6)

    # --- 文件浏览方法 ---
    def browse_eeg_npy_dir(self):
        dirname = filedialog.askdirectory(title="选择 EEG .npy 文件所在的文件夹")
        if dirname: self.eeg_npy_dir_path.set(dirname)

    def browse_artifact_report(self):
        filename = filedialog.askopenfilename(title="选择伪迹报告 CSV 文件", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filename: self.artifact_report_path.set(filename)

    def browse_sound_features_dir(self):
        dirname = filedialog.askdirectory(title="选择包含声音特征的文件夹")
        if dirname: self.sound_features_dir_path.set(dirname)

    def browse_output_dir(self):
        dirname = filedialog.askdirectory(title="选择保存对齐特征的输出文件夹")
        if dirname: self.output_dir_path.set(dirname)

    # --- 日志和进度更新方法 ---
    def log_message(self, text):
        try:
            if self.master.winfo_exists(): self.log_text.config(state='normal'); self.log_text.insert(tk.END, text + '\n'); self.log_text.see(tk.END); self.log_text.config(state='disabled')
        except Exception: print(f"Log Error: {text}")
    def update_progress(self, value):
        try:
             if self.master.winfo_exists(): self.master.after(0, lambda: self.progress_bar.config(value=min(max(0, int(value)), 100)))
        except Exception: pass
    def update_status(self, text):
        try:
             if self.master.winfo_exists(): self.master.after(0, lambda: self.status_label.config(text=str(text))); self.log_message(str(text))
        except Exception: pass
    def processing_finished(self, success_count, total_processed, failed_files):
        try:
             if self.master.winfo_exists():
                 self.master.after(0, lambda: self.start_button.config(state="normal"))
                 if success_count == total_processed and total_processed > 0:
                     message = f"批量处理完成！\n成功处理并保存: {success_count} 个文件。"
                     self.master.after(0, lambda: messagebox.showinfo("完成", message))
                 elif success_count > 0 :
                     message = f"批量处理部分完成。\n成功: {success_count}, 失败/跳过: {len(failed_files)} (共 {total_processed} 个计划文件)。"
                     if failed_files:
                         message += "\n失败文件列表 (最多显示5个):\n" + "\n".join(failed_files[:5])
                         if len(failed_files) > 5: message += "\n..."
                     self.master.after(0, lambda: messagebox.showwarning("部分完成", message))
                 else:
                     message = "批量处理失败，没有文件成功处理。"
                     if failed_files: message += f"\n失败/跳过: {len(failed_files)}。\n查看日志获取详情。"
                     self.master.after(0, lambda: messagebox.showerror("错误", message))
                 self.update_status(f"最终状态: {message.splitlines()[0]}")
        except Exception: pass

    # --- 启动批处理线程 ---
    def start_batch_processing_thread(self):
        eeg_dir = self.eeg_npy_dir_path.get()
        report_path = self.artifact_report_path.get()
        sound_feat_dir = self.sound_features_dir_path.get()
        output_dir = self.output_dir_path.get()

        if not os.path.isdir(eeg_dir): messagebox.showerror("错误", "请选择有效的 EEG .npy 文件夹"); return
        if not os.path.isfile(report_path): messagebox.showerror("错误", "请选择有效的伪迹报告 CSV 文件"); return
        if not os.path.isdir(sound_feat_dir): messagebox.showerror("错误", "请选择有效的声音特征文件夹"); return
        if not output_dir: messagebox.showerror("错误", "请选择有效的输出文件夹"); return
        if not os.path.isdir(output_dir):
            try: os.makedirs(output_dir, exist_ok=True); self.log_message(f"输出目录不存在，已创建: {output_dir}")
            except Exception as e: messagebox.showerror("错误", f"无法创建输出目录: {e}"); return

        try:
            params = {
                'eeg_sampling_rate_hz': self.eeg_sampling_rate_var.get(),
                'eeg_smoothing_sigma': self.eeg_smooth_sigma_var.get() if self.eeg_smooth_sigma_var.get() > 0 else None,
                'eeg_window_ms': self.eeg_window_ms_var.get(),
                'eeg_step_ms': self.eeg_step_ms_var.get(),
                'align_duration_sec': self.align_duration_sec_var.get(),
                'align_hop_sec': self.align_hop_sec_var.get()
            }
            if any(p <= 0 for k, p in params.items() if k not in ['eeg_smoothing_sigma'] and isinstance(p, (int,float))):
                raise ValueError("数值参数（采样率，窗口，步长，时长）必须为正。")
            if params['eeg_smoothing_sigma'] is not None and params['eeg_smoothing_sigma'] <= 0 :
                 params['eeg_smoothing_sigma'] = None # 如果sigma设为0或负数，则禁用平滑
        except (tk.TclError, ValueError) as e: messagebox.showerror("错误", f"无效的参数输入: {e}"); return

        self.start_button.config(state="disabled")
        self.update_progress(0)
        self.log_text.config(state='normal'); self.log_text.delete('1.0', tk.END); self.log_text.config(state='disabled')
        self.log_message("开始批量处理干净分段...")

        self.processing_thread = threading.Thread(
            target=self._run_batch_alignment_processing,
            args=(eeg_dir, report_path, sound_feat_dir, output_dir, params),
            daemon=True
        )
        self.processing_thread.start()

    def _run_batch_alignment_processing(self, eeg_dir, report_path, sound_feat_dir, output_dir, params):
        """在后台线程中执行批量对齐处理"""
        processed_count = 0
        success_count = 0
        failed_files_list = []

        try:
            # 1. 读取伪迹报告并筛选干净文件
            self.master.after(0, lambda: self.update_status("读取伪迹报告..."))
            try:
                artifact_df = pd.read_csv(report_path)
                if 'Is_Artifact' not in artifact_df.columns or 'Epoch_Filename' not in artifact_df.columns:
                    raise ValueError("伪迹报告中缺少 'Is_Artifact' 或 'Epoch_Filename' 列。")
                clean_eeg_filenames = artifact_df[artifact_df['Is_Artifact'] == False]['Epoch_Filename'].tolist()
                if not clean_eeg_filenames:
                    raise ValueError("未从报告中找到标记为 '干净' (Is_Artifact=False) 的文件。")
                self.master.after(0, lambda: self.log_message(f"找到 {len(clean_eeg_filenames)} 个干净的 EEG 分段进行处理。"))
            except Exception as e_report:
                self.master.after(0, lambda: self.log_message(f"错误：无法读取或解析伪迹报告: {e_report}"))
                self.processing_finished(0,0,[])
                return

            total_to_process = len(clean_eeg_filenames)
            if total_to_process == 0: self.processing_finished(0,0,[]); return

            # 2. 循环处理每个干净的文件
            for idx, eeg_filename in enumerate(clean_eeg_filenames):
                self.master.after(0, lambda i=idx, t=total_to_process, f=eeg_filename: self.update_status(f"处理 ({i+1}/{t}): {f}"))
                self.master.after(0, lambda p=int(100*(idx)/total_to_process) : self.update_progress(p) )
                processed_count += 1
                current_file_success = False
                try:
                    eeg_npy_path = os.path.join(eeg_dir, eeg_filename)
                    if not os.path.exists(eeg_npy_path):
                         self.master.after(0, lambda f=eeg_filename: self.log_message(f"  跳过：EEG 文件 {f} 未找到。"))
                         failed_files_list.append(eeg_filename + " (EEG NPY missing)")
                         continue

                    eeg_segment_data_full = np.load(eeg_npy_path)
                    # 使用参数字典解包传递参数
                    eeg_features_ts, eeg_times_ts_relative = extract_eeg_features_timeseries_gui(
                        eeg_segment_data_full,
                        eeg_sampling_rate_hz=params['eeg_sampling_rate_hz'], # <-- 确保这里的参数名正确
                        smoothing_sigma=params['eeg_smoothing_sigma'],
                        window_ms=params['eeg_window_ms'],
                        step_ms=params['eeg_step_ms'],
                        status_callback=lambda m, fn=eeg_filename: self.master.after(0, lambda: self.log_message(f"  EEG ({fn}): {m}"))
                    )
                    if eeg_features_ts is None:
                        failed_files_list.append(eeg_filename + " (EEG 特征提取失败)")
                        continue

                    # 从 EEG 文件名解析原始声音文件名
                    # 假设文件名格式: SUBJID_SEQNUM_WAVBASE.npy (SEQNUM 是1-3位数字)
                    match = re.match(r"^(.*?)_(\d{1,3})_(.*)\.npy$", eeg_filename)
                    if not match:
                        self.master.after(0, lambda f=eeg_filename: self.log_message(f"  跳过：无法从 {f} 解析声音文件名。"))
                        failed_files_list.append(eeg_filename + " (文件名解析错误)")
                        continue
                    original_wav_base = match.group(3)
                    subj_id = match.group(1)
                    stim_seq = match.group(2) # 字符串形式的序号

                    # 加载对应的声音特征
                    sound_feat_path = os.path.join(sound_feat_dir, f"{original_wav_base}_features.npy")
                    sound_times_path = os.path.join(sound_feat_dir, f"{original_wav_base}_feature_times.npy")

                    if not (os.path.exists(sound_feat_path) and os.path.exists(sound_times_path)):
                        self.master.after(0, lambda f=original_wav_base: self.log_message(f"  跳过 {eeg_filename}：未找到声音特征文件 for {f}。"))
                        failed_files_list.append(eeg_filename + " (声音特征文件缺失)")
                        continue
                    sound_features_ts = np.load(sound_feat_path)
                    sound_times_ts = np.load(sound_times_path)

                    # 特征对齐
                    aligned_eeg, aligned_sound = align_features_gui(
                        eeg_features_ts, eeg_times_ts_relative,
                        sound_features_ts, sound_times_ts,
                        target_duration_sec=params['align_duration_sec'],
                        target_hop_sec=params['align_hop_sec'],
                        status_callback=lambda m, fn=eeg_filename: self.master.after(0, lambda: self.log_message(f"  对齐 ({fn}): {m}"))
                    )

                    if aligned_eeg is not None and aligned_sound is not None:
                        output_eeg_aligned_filename = f"{subj_id}_{stim_seq}_{original_wav_base}_EEG_aligned.npy"
                        output_sound_aligned_filename = f"{subj_id}_{stim_seq}_{original_wav_base}_Sound_aligned.npy"
                        np.save(os.path.join(output_dir, output_eeg_aligned_filename), aligned_eeg.astype(np.float32))
                        np.save(os.path.join(output_dir, output_sound_aligned_filename), aligned_sound.astype(np.float32))
                        self.master.after(0, lambda f1=output_eeg_aligned_filename, f2=output_sound_aligned_filename: self.log_message(f"  已保存: {f1}, {f2}"))
                        success_count += 1
                        current_file_success = True
                    else:
                        failed_files_list.append(eeg_filename + " (特征对齐失败)")

                except Exception as file_e:
                    self.master.after(0, lambda f=eeg_filename, e=file_e: self.log_message(f"  处理文件 {f} 时出错: {e}"))
                    if not current_file_success:
                        failed_files_list.append(eeg_filename + f" (处理错误: {type(file_e).__name__})")

            self.master.after(0, lambda: self.update_progress(100))
            self.processing_finished(success_count, total_to_process, failed_files_list)

        except Exception as e_batch:
            self.master.after(0, lambda m=f"批量处理线程主错误: {e_batch}": self.log_message(m))
            import traceback
            self.master.after(0, lambda tb=traceback.format_exc(): self.log_message(tb))
            self.processing_finished(success_count, processed_count, failed_files_list)


# ===============================================
# 应用程序入口点
# ===============================================
if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ttkthemes import ThemedTk
        root = ThemedTk(theme="arc")
    except ImportError:
        print("ttkthemes 未安装，使用默认 ttk 风格。")
        s = ttk.Style(); s.theme_use('clam')
        s.configure('TButton', padding=6); s.configure('TEntry', padding=5)
        s.configure('TLabel', padding=2); s.configure('TCheckbutton', padding=3)
        s.configure('TLabelframe.Label', font=('Segoe UI', 9, 'bold'))
        s.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'), padding=6)

    app = Task3BatchAlignApp(root) # 运行新的 GUI 类
    root.mainloop()