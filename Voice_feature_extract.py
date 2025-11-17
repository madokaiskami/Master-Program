# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import librosa
import numpy as np
import pandas as pd
import os
import re
import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt # 此脚本中不直接绘图

# ===============================================
# 声音特征提取函数 (修正版)
# ===============================================
def extract_detailed_sound_features_librosa(
    wav_path,
    frame_length_ms=25.0,
    hop_length_ms=10.0,
    n_mfcc=13,
    fmin_hz_param=75.0,
    fmax_hz_param=600.0,
    status_callback=None
    ):
    """
    从 WAV 文件中提取一组详细的动态声音特征时间序列。
    包含参数调整和错误处理。
    """
    def _log(message):
        if status_callback: status_callback(message)
        # else: print(message) # 用于独立测试

    try:
        y, sr = librosa.load(wav_path, sr=None)
        nyquist = sr / 2.0
        duration = librosa.get_duration(y=y, sr=sr)
        _log(f"Info ({os.path.basename(wav_path)}): 加载音频 sr={sr} Hz, Nyquist={nyquist:.1f} Hz, 时长={duration:.2f}s")

        if y.size == 0:
            _log(f"警告：音频文件 '{os.path.basename(wav_path)}' 为空或无法加载。")
            return None, None, None

        n_fft = int(sr * frame_length_ms / 1000.0)
        hop_length = int(sr * hop_length_ms / 1000.0)

        if n_fft <= 0 or hop_length <= 0:
            _log(f"错误 ({os.path.basename(wav_path)}): 计算得到的 n_fft ({n_fft}) 或 hop_length ({hop_length}) 无效。")
            return None, None, None
        if n_fft > y.size: # FFT窗口不能大于信号长度
            _log(f"警告 ({os.path.basename(wav_path)}): n_fft ({n_fft}) 大于信号长度 ({y.size})。将使用信号长度作为 n_fft。")
            n_fft = y.size


        all_features_list = []
        feature_names_list = []
        num_target_frames = 0
        target_times_sec = np.array([]) # 初始化为空数组

        # A. MFCCs + Deltas
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            if mfcc.ndim != 2 or mfcc.shape[1] == 0: raise ValueError("MFCC 结果为空或形状不正确")
            num_target_frames = mfcc.shape[1]
            target_times_sec = librosa.frames_to_time(np.arange(num_target_frames), sr=sr, hop_length=hop_length, n_fft=n_fft) # 定义 target_times_sec
            delta_mfcc = librosa.feature.delta(mfcc, order=1)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            all_features_list.extend([mfcc, delta_mfcc, delta2_mfcc])
            feature_names_list.extend([f"mfcc_{i+1}" for i in range(n_mfcc)])
            feature_names_list.extend([f"delta_mfcc_{i+1}" for i in range(n_mfcc)])
            feature_names_list.extend([f"delta2_mfcc_{i+1}" for i in range(n_mfcc)])
        except Exception as e_mfcc:
            _log(f"错误 ({os.path.basename(wav_path)}) 提取 MFCC 时: {e_mfcc}"); return None,None,None

        # 内部辅助函数，用于对齐特征帧数
        def _align_feature(feat_array, target_frames, feature_name_debug, expected_dims=1):
            if feat_array is None or feat_array.size == 0:
                _log(f"警告 ({os.path.basename(wav_path)}): 特征 '{feature_name_debug}' 为空，将填充 NaN。")
                return np.full((expected_dims, target_frames), np.nan)

            if feat_array.ndim == 1: feat_array = feat_array.reshape(1, -1)
            current_dims, current_frames = feat_array.shape
            if current_dims == 0 : # 另一种空特征的情况
                 _log(f"警告 ({os.path.basename(wav_path)}): 特征 '{feature_name_debug}' 维度为0，将填充 NaN。")
                 return np.full((expected_dims, target_frames), np.nan)


            if current_frames == target_frames: return feat_array

            aligned_feat = np.full((current_dims, target_frames), np.nan)
            copy_frames = min(current_frames, target_frames)
            aligned_feat[:, :copy_frames] = feat_array[:, :copy_frames]

            if current_frames < target_frames and current_frames > 0 :
                for r_idx in range(current_dims):
                    aligned_feat[r_idx, current_frames:] = feat_array[r_idx, -1]
            return aligned_feat

        # B. RMS Energy
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
        all_features_list.append(_align_feature(rms, num_target_frames, "RMS"))
        feature_names_list.append("rms_energy")

        # C. Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
        all_features_list.append(_align_feature(zcr, num_target_frames, "ZCR"))
        feature_names_list.append("zero_crossing_rate")

        # D. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        all_features_list.append(_align_feature(spectral_centroid, num_target_frames, "Spectral Centroid"))
        feature_names_list.append("spectral_centroid")

        # E. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        all_features_list.append(_align_feature(spectral_bandwidth, num_target_frames, "Spectral Bandwidth"))
        feature_names_list.append("spectral_bandwidth")

        # F. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)
        all_features_list.append(_align_feature(spectral_rolloff, num_target_frames, "Spectral Rolloff"))
        feature_names_list.append("spectral_rolloff_85")

        # G. Spectral Flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
        all_features_list.append(_align_feature(spectral_flatness, num_target_frames, "Spectral Flatness"))
        feature_names_list.append("spectral_flatness")

        # H. Spectral Contrast
        n_bands_for_contrast = 6
        fmin_contrast = 200.0
        # Librosa 内部计算最高中心频率大约是 fmin * 2^(n_bands - 0.5)
        # 确保奈奎斯特频率大于这个范围的某个点
        if nyquist < (fmin_contrast * (2**(n_bands_for_contrast - 1.5))): # 检查比最高中心频率稍低的频率
            # 根据采样率动态调整 n_bands
            if sr >= 16000: n_bands_for_contrast = 5 # 或 6，取决于具体 fmin
            elif sr >= 8000: n_bands_for_contrast = 4
            else: n_bands_for_contrast = 3
            _log(f"Info ({os.path.basename(wav_path)}): For spectral_contrast, n_bands 调整为 {n_bands_for_contrast} (sr={sr} Hz)")
        num_expected_contrast_features = n_bands_for_contrast + 1

        try:
            spectral_contrast_feat = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft,
                                                                hop_length=hop_length,
                                                                n_bands=n_bands_for_contrast,
                                                                fmin=fmin_contrast)
            sc_aligned = _align_feature(spectral_contrast_feat, num_target_frames, "Spectral Contrast", expected_dims=num_expected_contrast_features)
        except Exception as e_sc:
             _log(f"错误 ({os.path.basename(wav_path)}) 提取 Spectral Contrast 时: {e_sc}")
             sc_aligned = np.full((num_expected_contrast_features, num_target_frames), np.nan)

        all_features_list.append(sc_aligned)
        feature_names_list.extend([f"spectral_contrast_band_{i+1}" for i in range(sc_aligned.shape[0])]) # 使用实际返回的维度

        # I. F0 (pYIN) and Voiced Flag
        effective_fmax_hz = min(fmax_hz_param, nyquist - 50.0) # 从奈奎斯特减去 buffer
        if effective_fmax_hz <= fmin_hz_param: effective_fmax_hz = fmin_hz_param + 100

        # 确保 frame_length (n_fft) 对于 pyin 来说足够大
        # pyin 要求 frame_length > sr / fmin
        min_frame_len_for_pyin = int(sr / fmin_hz_param) + 1 if fmin_hz_param > 0 else n_fft
        pyin_n_fft_param = max(n_fft, min_frame_len_for_pyin)

        _log(f"Info ({os.path.basename(wav_path)}): For pyin, fmin={fmin_hz_param:.1f}, fmax={effective_fmax_hz:.1f}, frame_length={pyin_n_fft_param}")

        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=fmin_hz_param, fmax=effective_fmax_hz,
                                              frame_length=pyin_n_fft_param, # 使用调整后的帧长
                                              hop_length=hop_length, # hop_length 与其他特征一致
                                              fill_na=np.nan)
            f0_aligned = _align_feature(f0.reshape(1,-1) if f0.ndim==1 else f0, num_target_frames, "F0")
            voiced_flag_aligned = _align_feature(voiced_flag.astype(float).reshape(1,-1) if voiced_flag.ndim==1 else voiced_flag.astype(float),
                                                 num_target_frames, "Voiced Flag")
        except Exception as e_pyin:
            _log(f"错误 ({os.path.basename(wav_path)}) 提取 F0 (pyin) 时: {e_pyin}")
            f0_aligned = np.full((1, num_target_frames), np.nan)
            voiced_flag_aligned = np.full((1, num_target_frames), np.nan)

        all_features_list.append(f0_aligned)
        feature_names_list.append("f0_pyin")
        all_features_list.append(voiced_flag_aligned)
        feature_names_list.append("voiced_flag_pyin")

        # --- 合并 ---
        if not all_features_list:
            _log(f"错误 ({os.path.basename(wav_path)}): 未能处理任何特征。")
            return None, None, None

        features_timeseries = np.vstack(all_features_list).T

        # 最终检查特征名数量
        if features_timeseries.shape[1] != len(feature_names_list):
             _log(f"严重警告 ({os.path.basename(wav_path)}): 最终特征数量 ({features_timeseries.shape[1]}) 与名称列表长度 ({len(feature_names_list)}) 不匹配！特征名可能不准确。")
             if features_timeseries.shape[1] > len(feature_names_list):
                 feature_names_list.extend([f"unknown_feat_{k}" for k in range(len(feature_names_list), features_timeseries.shape[1])])
             else:
                 feature_names_list = feature_names_list[:features_timeseries.shape[1]]

        return features_timeseries, feature_names_list, target_times_sec

    except Exception as e:
        _log(f"提取详细声音特征时发生严重错误 '{os.path.basename(wav_path)}': {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ===============================================
# 批处理函数 (与之前版本类似)
# ===============================================
def batch_process_sound_features(input_dir, output_dir, params, progress_callback, status_callback):
    """
    批量处理输入目录中的所有 WAV 文件，提取特征并保存为 .npy 文件。
    """
    try:
        wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]
        total_files = len(wav_files)
        status_callback(f"找到 {total_files} 个 .wav 文件在目录 '{input_dir}'")
        if total_files == 0:
            status_callback("错误：未找到 .wav 文件。")
            progress_callback(100)
            return False

        os.makedirs(output_dir, exist_ok=True)
        saved_count = 0
        failed_count = 0

        for i, filename in enumerate(wav_files):
            current_progress = int(100 * (i + 1) / total_files)
            progress_callback(current_progress)
            status_callback(f"处理中 ({i+1}/{total_files}): {filename}")

            wav_path = os.path.join(input_dir, filename)
            # 从 params 字典中获取参数
            features_ts, feature_names, times_ts = extract_detailed_sound_features_librosa(
                wav_path,
                frame_length_ms=params['frame_length_ms'],
                hop_length_ms=params['hop_length_ms'],
                n_mfcc=params['n_mfcc'],
                fmin_hz_param=params.get('fmin_hz', 75.0), # 使用 get 获取可选参数
                fmax_hz_param=params.get('fmax_hz', 600.0),
                status_callback=status_callback # 传递回调
            )

            if features_ts is not None and feature_names is not None and times_ts is not None:
                base, _ = os.path.splitext(filename)
                output_npy_path = os.path.join(output_dir, f"{base}_features.npy")
                output_names_path = os.path.join(output_dir, f"{base}_feature_names.txt")
                output_times_path = os.path.join(output_dir, f"{base}_feature_times.npy")

                try:
                    np.save(output_npy_path, features_ts.astype(np.float32)) # 保存为 float32 节省空间
                    np.save(output_times_path, times_ts.astype(np.float32))
                    with open(output_names_path, 'w', encoding='utf-8') as f_names:
                        for name in feature_names:
                            f_names.write(name + '\n')
                    saved_count += 1
                except Exception as e_save:
                    status_callback(f"错误：无法保存文件 '{output_npy_path}': {e_save}")
                    failed_count += 1
            else:
                status_callback(f"跳过文件 '{filename}'，无法提取特征。")
                failed_count += 1

        status_callback(f"\n批处理完成。成功保存: {saved_count} 个文件, 失败/跳过: {failed_count} 个文件。")
        progress_callback(100)
        return True

    except FileNotFoundError:
        status_callback(f"错误：输入目录 '{input_dir}' 未找到。")
        progress_callback(0)
        return False
    except Exception as e:
        status_callback(f"批处理过程中发生未知错误: {e}")
        import traceback
        traceback.print_exc()
        progress_callback(0)
        return False

# ===============================================
# GUI 主类 (与之前版本基本相同)
# ===============================================
class SoundFeatureExtractorApp:
    def __init__(self, master):
        self.master = master
        master.title("声音特征提取工具 (任务 3 前期) v3")
        master.geometry("700x550")

        # --- GUI 变量 ---
        self.input_dir_path = tk.StringVar()
        self.output_dir_path = tk.StringVar()
        self.n_mfcc_var = tk.IntVar(value=13)
        self.frame_length_var = tk.DoubleVar(value=25.0)
        self.hop_length_var = tk.DoubleVar(value=10.0)
        self.fmin_var = tk.DoubleVar(value=75.0) # F0 最小频率
        self.fmax_var = tk.DoubleVar(value=600.0) # F0 最大频率

        # --- 创建控件 ---
        frame = ttk.Frame(master, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)

        row_idx = 0
        # 输入目录
        ttk.Label(frame, text="包含 .wav 文件的输入文件夹:").grid(row=row_idx, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frame, textvariable=self.input_dir_path, width=60).grid(row=row_idx, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(frame, text="浏览...", command=self.browse_input_dir).grid(row=row_idx, column=3, padx=5, pady=5)
        frame.columnconfigure(1, weight=1)
        row_idx += 1

        # 输出目录
        ttk.Label(frame, text="保存 .npy 特征的输出文件夹:").grid(row=row_idx, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frame, textvariable=self.output_dir_path, width=60).grid(row=row_idx, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(frame, text="浏览...", command=self.browse_output_dir).grid(row=row_idx, column=3, padx=5, pady=5)
        row_idx += 1

        # 参数设置框架
        params_frame = ttk.LabelFrame(frame, text="特征提取参数", padding="10")
        params_frame.grid(row=row_idx, column=0, columnspan=4, padx=5, pady=10, sticky="ew")
        row_idx += 1

        pf_row = 0; pf_col = 0
        ttk.Label(params_frame, text="MFCC 数量 (n_mfcc):").grid(row=pf_row, column=pf_col, padx=5, pady=3, sticky="w"); pf_col+=1
        ttk.Entry(params_frame, textvariable=self.n_mfcc_var, width=8).grid(row=pf_row, column=pf_col, padx=5, pady=3, sticky="w"); pf_col+=1

        ttk.Label(params_frame, text="帧长 (ms, frame_length):").grid(row=pf_row, column=pf_col, padx=5, pady=3, sticky="w"); pf_col+=1
        ttk.Entry(params_frame, textvariable=self.frame_length_var, width=8).grid(row=pf_row, column=pf_col, padx=5, pady=3, sticky="w");
        pf_col=0; pf_row+=1 # 换行

        ttk.Label(params_frame, text="帧移 (ms, hop_length):").grid(row=pf_row, column=pf_col, padx=5, pady=3, sticky="w"); pf_col+=1
        ttk.Entry(params_frame, textvariable=self.hop_length_var, width=8).grid(row=pf_row, column=pf_col, padx=5, pady=3, sticky="w"); pf_col+=1

        ttk.Label(params_frame, text="F0 最小频率 (Hz):").grid(row=pf_row, column=pf_col, padx=5, pady=3, sticky="w"); pf_col+=1
        ttk.Entry(params_frame, textvariable=self.fmin_var, width=8).grid(row=pf_row, column=pf_col, padx=5, pady=3, sticky="w");
        pf_col=0; pf_row+=1 # 换行

        ttk.Label(params_frame, text="F0 最大频率 (Hz):").grid(row=pf_row, column=pf_col, padx=5, pady=3, sticky="w"); pf_col+=1
        ttk.Entry(params_frame, textvariable=self.fmax_var, width=8).grid(row=pf_row, column=pf_col, padx=5, pady=3, sticky="w")


        # 启动按钮
        self.start_button = ttk.Button(frame, text="开始提取特征", command=self.start_batch_thread, style='Accent.TButton')
        self.start_button.grid(row=row_idx, column=1, columnspan=2, padx=5, pady=15)
        row_idx += 1

        # 进度条和状态标签
        self.progress_bar = ttk.Progressbar(frame, orient="horizontal", length=600, mode="determinate")
        self.progress_bar.grid(row=row_idx, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        row_idx += 1
        self.status_label = ttk.Label(frame, text="准备就绪", anchor="w", justify="left")
        self.status_label.grid(row=row_idx, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        row_idx += 1

        # 日志/消息区域
        self.log_text = scrolledtext.ScrolledText(frame, height=10, width=80, state='disabled', wrap=tk.WORD)
        self.log_text.grid(row=row_idx, column=0, columnspan=4, padx=5, pady=5, sticky="ewns")
        frame.rowconfigure(row_idx, weight=1) # 使日志区域可伸缩

        # 应用主题 (如果可用)
        try:
            from ttkthemes import ThemedStyle
            style = ThemedStyle(master)
            style.set_theme("arc") # 或者 "adapta", "plastik", "ubuntu", etc.
            # 自定义 Accent button 风格 (可选)
            style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'), padding=6)
        except ImportError:
             # print("ttkthemes 未安装，将使用默认 ttk 风格。")
             # 为默认 ttk button 添加一些样式
             s = ttk.Style()
             s.configure('TButton', font=('Segoe UI', 10), padding=5)
             s.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'), padding=6)


    # --- 文件/目录浏览方法 ---
    def browse_input_dir(self):
        dirname = filedialog.askdirectory(title="选择包含 .wav 文件的输入文件夹")
        if dirname: self.input_dir_path.set(dirname)

    def browse_output_dir(self):
        dirname = filedialog.askdirectory(title="选择保存 .npy 特征的输出文件夹")
        if dirname: self.output_dir_path.set(dirname)

    # --- 日志和进度更新方法 (与之前版本相同) ---
    def log_message(self, text):
        try:
            if self.master.winfo_exists():
                self.log_text.config(state='normal')
                self.log_text.insert(tk.END, text + '\n')
                self.log_text.see(tk.END)
                self.log_text.config(state='disabled')
        except Exception: print(f"Log Error (GUI likely closed): {text}")

    def update_progress(self, value):
        try:
             if self.master.winfo_exists(): self.master.after(0, lambda: self.progress_bar.config(value=min(max(0, int(value)), 100)))
        except Exception: pass

    def update_status(self, text):
        try:
             if self.master.winfo_exists():
                 self.master.after(0, lambda: self.status_label.config(text=str(text)))
                 self.log_message(str(text))
        except Exception: pass

    def processing_finished(self, success):
        try:
             if self.master.winfo_exists():
                 self.master.after(0, lambda: self.start_button.config(state="normal"))
                 if success: self.master.after(0, lambda: messagebox.showinfo("完成", "特征提取成功完成！"))
                 else: self.master.after(0, lambda: messagebox.showerror("错误", "特征提取过程中发生错误。"))
        except Exception: pass

    # --- 启动处理线程 ---
    def start_batch_thread(self):
        input_dir = self.input_dir_path.get()
        output_dir = self.output_dir_path.get()

        if not os.path.isdir(input_dir): messagebox.showerror("错误", "请选择有效的输入文件夹"); return
        if not output_dir: messagebox.showerror("错误", "请选择有效的输出文件夹"); return
        if not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                self.log_message(f"输出目录不存在，已创建: {output_dir}")
            except Exception as e:
                messagebox.showerror("错误", f"无法创建输出目录: {e}"); return

        try:
            params = {
                'n_mfcc': self.n_mfcc_var.get(),
                'frame_length_ms': self.frame_length_var.get(),
                'hop_length_ms': self.hop_length_var.get(),
                'fmin_hz': self.fmin_var.get(),
                'fmax_hz': self.fmax_var.get()
            }
            if params['n_mfcc'] <=0 or params['frame_length_ms'] <=0 or params['hop_length_ms'] <=0:
                raise ValueError("参数 (MFCC数, 帧长, 帧移) 必须为正数")
            if params['fmin_hz'] < 0 or params['fmax_hz'] <= 0 or params['fmin_hz'] >= params['fmax_hz']:
                 raise ValueError("F0 频率范围设置无效")
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("错误", f"无效的参数输入: {e}"); return

        self.start_button.config(state="disabled")
        self.update_progress(0)
        self.log_text.config(state='normal'); self.log_text.delete('1.0', tk.END); self.log_text.config(state='disabled')
        self.log_message(f"开始批处理声音特征提取 (MFCCs={params['n_mfcc']}, Frame={params['frame_length_ms']}ms, Hop={params['hop_length_ms']}ms)...")

        self.processing_thread = threading.Thread(
            target=self._run_batch_processing,
            args=(input_dir, output_dir, params),
            daemon=True
        )
        self.processing_thread.start()

    def _run_batch_processing(self, input_dir, output_dir, params):
        """在后台线程中执行批处理"""
        success = False
        try:
            success = batch_process_sound_features(
                input_dir, output_dir, params,
                self.update_progress, self.update_status
            )
        except Exception as e:
            self.master.after(0, lambda m=f"批处理线程出错: {e}": self.log_message(m))
            import traceback
            traceback.print_exc() # 打印完整错误堆栈到控制台
            success = False
        finally:
            self.processing_finished(success)

# ===============================================
# 应用程序入口点
# ===============================================
if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ttkthemes import ThemedTk
        root = ThemedTk(theme="arc") # 或者 "plastik", "clearlooks", "radiance" 等
    except ImportError:
        print("ttkthemes 未安装，将使用默认 ttk 风格。可以尝试 `pip install ttkthemes`")
        s = ttk.Style()
        try:
            s.theme_use('clam') # 'clam', 'alt', 'default', 'classic'
        except tk.TclError:
            print("Clam 主题不可用，使用默认主题。")
        # 为默认 ttk 添加一些样式
        s.configure('TButton', padding=6)
        s.configure('TEntry', padding=5)
        s.configure('TLabel', padding=2)
        s.configure('TCheckbutton', padding=3)
        s.configure('TLabelframe.Label', font=('Segoe UI', 9, 'bold'))
        s.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'), padding=6)


    app = SoundFeatureExtractorApp(root)
    root.mainloop()