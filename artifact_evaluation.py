# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
# import struct # struct 在这个任务中不再需要，因为我们读 .npy
import numpy as np
import pandas as pd
import os
import re
import matplotlib
matplotlib.use('Agg') # 使用 Agg 后端，避免在无图形界面的服务器上或线程中出错
import matplotlib.pyplot as plt

# ===============================================
# 任务 2: 伪迹评估函数
# ===============================================

def calculate_artifact_metrics_all(epoch_data_with_meta):
    """
    计算单个分段的多种伪迹指标。
    假设输入 N x 34, 列 0=时间, 列 1=标记, 列 2-33=EEG。
    指标只基于 EEG 通道 (列 2 及之后) 计算。

    Args:
        epoch_data_with_meta (np.array): 分段数据 (N_points, 34), float32 或 float64.

    Returns:
        dict: 包含所有指标的字典，或 None 如果出错。
               Keys: 'std_mean', 'rms_mean', 'max_abs_amp',
                     'ratio_max_meanabs', 'ratio_max_std', 'ratio_maxdev_mad'
    """
    # 检查输入的有效性
    if epoch_data_with_meta is None or epoch_data_with_meta.ndim != 2 or epoch_data_with_meta.shape[1] < 3:
        # print("警告：calculate_artifact_metrics_all 输入数据无效。") # 可选
        return None
    eeg_data = epoch_data_with_meta[:, 2:] # **关键：只选择 EEG 列**
    if eeg_data.size == 0:
        # print("警告：未能提取有效的 EEG 数据列。") # 可选
        return None

    # 初始化指标字典，确保所有键都存在
    metrics = {
        'std_mean': np.nan,
        'rms_mean': np.nan,
        'max_abs_amp': np.nan,
        'ratio_max_meanabs': np.nan, # 方法 A: Max / Mean Abs
        'ratio_max_std': np.nan,     # 方法 B: Max / Mean Std
        'ratio_maxdev_mad': np.nan   # 方法 C: Max Deviation from Median / Mean MAD
    }

    try:
        # 使用 float64 进行计算以保证精度
        eeg_data_float = eeg_data.astype(np.float64)

        # --- 基础指标 ---
        channel_stds = np.std(eeg_data_float, axis=0)
        mean_std = np.mean(channel_stds)
        if np.isfinite(mean_std): metrics['std_mean'] = float(mean_std)

        channel_rms = np.sqrt(np.mean(eeg_data_float**2, axis=0))
        mean_rms = np.mean(channel_rms)
        if np.isfinite(mean_rms): metrics['rms_mean'] = float(mean_rms)

        max_abs_amp = np.max(np.abs(eeg_data_float))
        if np.isfinite(max_abs_amp): metrics['max_abs_amp'] = float(max_abs_amp)

        # --- 比率指标计算 ---

        # 方法 A: Ratio Max / Mean Abs
        channel_mean_abs_amp = np.mean(np.abs(eeg_data_float), axis=0)
        mean_abs_amp_overall = np.mean(channel_mean_abs_amp)
        if pd.notna(metrics['max_abs_amp']) and pd.notna(mean_abs_amp_overall) and mean_abs_amp_overall > 1e-9:
            ratio_a = metrics['max_abs_amp'] / mean_abs_amp_overall
            metrics['ratio_max_meanabs'] = float(ratio_a) if np.isfinite(ratio_a) else 999999.9 # 用大数代替 Inf
        elif pd.notna(metrics['max_abs_amp']) and metrics['max_abs_amp'] > 1e-9:
            metrics['ratio_max_meanabs'] = 999999.9 # 用大数代替 Inf

        # 方法 B: Ratio Max / Mean Std
        if pd.notna(metrics['max_abs_amp']) and pd.notna(metrics['std_mean']) and metrics['std_mean'] > 1e-9:
            ratio_b = metrics['max_abs_amp'] / metrics['std_mean']
            metrics['ratio_max_std'] = float(ratio_b) if np.isfinite(ratio_b) else 999999.9
        elif pd.notna(metrics['max_abs_amp']) and metrics['max_abs_amp'] > 1e-9:
             metrics['ratio_max_std'] = 999999.9

        # 方法 C: Ratio Max Deviation from Median / Mean MAD (主要方法)
        try:
            channel_medians = np.median(eeg_data_float, axis=0)
            channel_mad = np.median(np.abs(eeg_data_float - channel_medians), axis=0)
            mean_mad = np.mean(channel_mad)
            overall_median = np.median(eeg_data_float)
            max_abs_deviation = np.max(np.abs(eeg_data_float - overall_median))

            if pd.notna(max_abs_deviation) and pd.notna(mean_mad) and mean_mad > 1e-9:
                estimated_sd_from_mad = mean_mad * 1.4826
                if estimated_sd_from_mad > 1e-9:
                    ratio_c = max_abs_deviation / estimated_sd_from_mad
                    metrics['ratio_maxdev_mad'] = float(ratio_c) if np.isfinite(ratio_c) else 999999.9
                else:
                     metrics['ratio_maxdev_mad'] = 999999.9 if max_abs_deviation > 1e-9 else 0.0
            elif pd.notna(max_abs_deviation) and max_abs_deviation > 1e-9:
                 metrics['ratio_maxdev_mad'] = 999999.9
        except Exception as mad_e:
             print(f"计算 MAD 指标时出错: {mad_e}")
             # metrics['ratio_maxdev_mad'] 保持为 NaN

        return metrics
    except Exception as e:
        print(f"计算所有 EEG 指标时出错: {e}")
        return metrics # 返回包含 NaN 的字典

def extract_info_from_filename(filename):
    """
    从标准化的分段文件名中提取信息。
    文件名格式: 被试ID_刺激序号(1-3位)_声音文件名.npy
    """
    match = re.match(r"^(.*?)_(\d{1,3})_(.*)\.npy$", filename)
    if match:
        subject_id = match.group(1)
        try: sequence_number = int(match.group(2))
        except ValueError: sequence_number = -1
        wav_filename_base = match.group(3)
        return subject_id, sequence_number, wav_filename_base
    else:
        # print(f"警告：文件名 '{filename}' 不匹配预期格式。尝试基本分割...") # 减少日志噪音
        base = filename.replace('.npy', '')
        parts = base.split('_')
        if len(parts) >= 3:
            subject_id = parts[0]
            try: sequence_number = int(parts[1])
            except ValueError: sequence_number = -1
            wav_filename_base = "_".join(parts[2:])
            return subject_id, sequence_number, wav_filename_base
        else:
             print(f"警告：无法从文件名 '{filename}' 解析信息。")
             return "Unknown", -1, base

# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
# import struct # 不再需要
import numpy as np
import pandas as pd
import os
import re
import matplotlib
matplotlib.use('Agg') # 使用 Agg 后端，避免在无图形界面的服务器上或线程中出错
import matplotlib.pyplot as plt
from scipy import stats # 用于计算 Z-score
from scipy.ndimage import gaussian_filter1d # 用于高斯平滑

# ===============================================
# 任务 2: 伪迹评估函数 (含平滑选项)
# ===============================================

def calculate_artifact_metrics_all(epoch_data_with_meta, smoothing_sigma=None):
    """
    计算单个分段的多种伪迹指标，在计算前可选地对 EEG 数据进行高斯平滑。
    假设输入 N x 34, 列 0=时间, 列 1=标记, 列 2-33=EEG。

    Args:
        epoch_data_with_meta (np.array): 分段数据 (N_points, 34).
        smoothing_sigma (float, optional): 高斯平滑的标准差（点数）。
                                           如果为 None, 0 或负数，则不平滑。

    Returns:
        dict: 包含所有指标的字典，或 None 如果出错。
               Keys: 'std_mean', 'rms_mean', 'max_abs_amp',
                     'ratio_max_meanabs', 'ratio_max_std', 'ratio_maxdev_mad'
    """
    # 检查输入的有效性
    if epoch_data_with_meta is None or epoch_data_with_meta.ndim != 2 or epoch_data_with_meta.shape[1] < 3:
        # print("警告：calculate_artifact_metrics_all 输入数据无效。") # 可选
        return None
    eeg_data_raw = epoch_data_with_meta[:, 2:] # 原始 EEG 数据 (从第2列开始)
    if eeg_data_raw.size == 0:
        # print("警告：未能提取有效的 EEG 数据列。") # 可选
        return None

    # 初始化指标字典，确保所有键都存在，值为 NaN
    metrics = {
        'std_mean': np.nan, 'rms_mean': np.nan, 'max_abs_amp': np.nan,
        'ratio_max_meanabs': np.nan, 'ratio_max_std': np.nan, 'ratio_maxdev_mad': np.nan
    }

    try:
        # 使用 float64 进行计算以保证精度
        eeg_data_float = eeg_data_raw.astype(np.float64)

        # --- 可选的高斯平滑处理 ---
        if smoothing_sigma and smoothing_sigma > 0:
            # print(f"   平滑处理 (sigma={smoothing_sigma})...") # Debugging
            data_to_calculate = np.apply_along_axis(
                gaussian_filter1d, axis=0, arr=eeg_data_float,
                sigma=smoothing_sigma, mode='reflect'
            )
        else:
            # print("   跳过平滑处理。") # Debugging
            data_to_calculate = eeg_data_float
        # --- 平滑处理结束 ---


        # --- 基础指标 (基于 data_to_calculate) ---
        channel_stds = np.std(data_to_calculate, axis=0)
        mean_std = np.mean(channel_stds)
        if np.isfinite(mean_std): metrics['std_mean'] = float(mean_std)

        channel_rms = np.sqrt(np.mean(data_to_calculate**2, axis=0))
        mean_rms = np.mean(channel_rms)
        if np.isfinite(mean_rms): metrics['rms_mean'] = float(mean_rms)

        # 最大绝对幅值 (在平滑/原始数据上计算)
        max_abs_amp_calc = np.max(np.abs(data_to_calculate))
        if np.isfinite(max_abs_amp_calc): metrics['max_abs_amp'] = float(max_abs_amp_calc)

        # --- 比率指标计算 (基于 data_to_calculate) ---
        # 方法 A: Ratio Max / Mean Abs
        channel_mean_abs_amp = np.mean(np.abs(data_to_calculate), axis=0)
        mean_abs_amp_overall = np.mean(channel_mean_abs_amp)
        current_max_abs = metrics['max_abs_amp'] # 使用已计算的值
        if pd.notna(current_max_abs) and pd.notna(mean_abs_amp_overall) and mean_abs_amp_overall > 1e-9:
            ratio_a = current_max_abs / mean_abs_amp_overall
            metrics['ratio_max_meanabs'] = float(ratio_a) if np.isfinite(ratio_a) else 999999.9 # 用大数代替 Inf
        elif pd.notna(current_max_abs) and current_max_abs > 1e-9: # 如果均值为0但最大值不为0
            metrics['ratio_max_meanabs'] = 999999.9

        # 方法 B: Ratio Max / Mean Std
        current_mean_std = metrics['std_mean']
        if pd.notna(current_max_abs) and pd.notna(current_mean_std) and current_mean_std > 1e-9:
            ratio_b = current_max_abs / current_mean_std
            metrics['ratio_max_std'] = float(ratio_b) if np.isfinite(ratio_b) else 999999.9
        elif pd.notna(current_max_abs) and current_max_abs > 1e-9: # 如果std为0但最大值不为0
             metrics['ratio_max_std'] = 999999.9

        # 方法 C: Ratio Max Deviation from Median / Mean MAD
        try:
            channel_medians = np.median(data_to_calculate, axis=0)
            # 计算 MAD: median(|x - median(x)|)
            channel_mad = np.median(np.abs(data_to_calculate - channel_medians), axis=0)
            mean_mad = np.mean(channel_mad)
            # 计算最大绝对偏差（相对于整体中位数）
            overall_median = np.median(data_to_calculate)
            max_abs_deviation = np.max(np.abs(data_to_calculate - overall_median))

            if pd.notna(max_abs_deviation) and pd.notna(mean_mad) and mean_mad > 1e-9:
                estimated_sd_from_mad = mean_mad * 1.4826 # 转换为近似标准差
                if estimated_sd_from_mad > 1e-9:
                    ratio_c = max_abs_deviation / estimated_sd_from_mad
                    metrics['ratio_maxdev_mad'] = float(ratio_c) if np.isfinite(ratio_c) else 999999.9
                else: # 如果 MAD 估计的标准差接近0
                     metrics['ratio_maxdev_mad'] = 999999.9 if max_abs_deviation > 1e-9 else 0.0 # 如果偏差不为0，比率极大
            elif pd.notna(max_abs_deviation) and max_abs_deviation > 1e-9: # 如果偏差不为0，但 MAD 接近0
                 metrics['ratio_maxdev_mad'] = 999999.9
        except Exception as mad_e:
             print(f"计算 MAD 指标时出错: {mad_e}")
             # metrics['ratio_maxdev_mad'] 保持为 NaN

        return metrics
    except Exception as e:
        print(f"计算所有 EEG 指标时出错: {e}")
        return metrics # 返回包含 NaN 的字典

def extract_info_from_filename(filename):
    """
    从标准化的分段文件名中提取信息。
    文件名格式: 被试ID_刺激序号(1-3位)_声音文件名.npy
    """
    # 尝试匹配主要格式，允许序号为 1-3 位数字
    match = re.match(r"^(.*?)_(\d{1,3})_(.*)\.npy$", filename)
    if match:
        subject_id = match.group(1)
        try:
            sequence_number = int(match.group(2))
        except ValueError:
            sequence_number = -1 # 如果序号不是数字
        wav_filename_base = match.group(3)
        return subject_id, sequence_number, wav_filename_base
    else:
        # print(f"警告：文件名 '{filename}' 不匹配预期格式 'ID_XXX_WAVBASE.npy'。尝试基本分割...")
        base = filename.replace('.npy', '')
        parts = base.split('_')
        if len(parts) >= 3:
            subject_id = parts[0]
            try:
                sequence_number = int(parts[1])
            except ValueError:
                 sequence_number = -1
                 # print(f"警告：无法从 '{filename}' 的第二部分 ('{parts[1]}') 解析序号。")
            wav_filename_base = "_".join(parts[2:]) # 文件名本身可能包含下划线
            return subject_id, sequence_number, wav_filename_base
        else: # 无法解析
             print(f"警告：无法从文件名 '{filename}' 中按 '_' 分割解析信息。")
             return "Unknown", -1, base

# ===============================================
# 报告生成核心函数 (带平滑和综合评分)
# ===============================================
def generate_artifact_report_composite_with_smoothing(
    epoch_dir, output_csv_path,
    weights, # 权重字典
    composite_score_threshold, # 综合评分阈值
    smoothing_sigma=None, # 新增：平滑参数
    visualize_dir=None,
    progress_callback=None, status_callback=None):
    """
    计算多种指标（可选平滑），生成综合评分，并基于综合评分进行报告和可视化。
    假设输入的 .npy 文件包含 N x 34 列 (时间, 标记, 32 EEG)。
    """
    def _update_status(msg):
        if status_callback: status_callback(msg)
        else: print(msg)
    def _update_progress(val):
        if progress_callback: progress_callback(val)

    all_metrics_data = [] # 用于存储最终的行字典
    _update_progress(0)
    try:
        npy_files = [f for f in os.listdir(epoch_dir) if f.lower().endswith(".npy")]
        total_files = len(npy_files)
        _update_status(f"找到 {total_files} 个 .npy 文件在 '{epoch_dir}'")
        if total_files == 0:
            _update_status("错误：目录中未找到 .npy 文件。")
            return False

        _update_status("第一步：计算所有文件的基础指标...")
        # --- 步骤 1: 计算所有文件的基础指标 ---
        for i, filename in enumerate(npy_files):
            if (i + 1) % 50 == 0 or i == total_files - 1: # 减少状态更新频率
                _update_status(f"  计算指标: {i+1}/{total_files}")
            file_path = os.path.join(epoch_dir, filename)
            subject_id, sequence_number, wav_base = extract_info_from_filename(filename)

            metrics = None
            epoch_data_full = None
            try:
                epoch_data_full = np.load(file_path) # 加载 N x 34 数组
                if epoch_data_full.ndim == 2 and epoch_data_full.shape[0] > 0 and epoch_data_full.shape[1] >= 3:
                    # 调用计算指标的函数, 传递平滑参数
                    metrics = calculate_artifact_metrics_all(epoch_data_full, smoothing_sigma=smoothing_sigma)
                else:
                     _update_status(f"警告：文件 '{filename}' 数据形状无效 ({epoch_data_full.shape})。")
            except Exception as e:
                _update_status(f"错误：加载或处理文件 '{filename}' 时出错: {e}")
                metrics = None

            # 构建包含所有目标列的字典
            row_data = {
                'Epoch_Filename': filename,
                'Subject_ID': subject_id,
                'Sequence_Number': sequence_number if sequence_number != -1 else np.nan,
                'WAV_Filename_Base': wav_base,
                # 直接从 metrics 获取值，如果 metrics 为 None 或键不存在，则为 NaN
                'Std_Mean': metrics.get('std_mean', np.nan) if metrics else np.nan,
                'RMS_Mean': metrics.get('rms_mean', np.nan) if metrics else np.nan,
                'Max_Abs_Amplitude': metrics.get('max_abs_amp', np.nan) if metrics else np.nan,
                'Ratio_MaxMeanAbs': metrics.get('ratio_max_meanabs', np.nan) if metrics else np.nan,
                'Ratio_MaxStd': metrics.get('ratio_max_std', np.nan) if metrics else np.nan,
                'Ratio_MaxDevMeanMAD': metrics.get('ratio_maxdev_mad', np.nan) if metrics else np.nan
            }
            all_metrics_data.append(row_data) # 添加行数据到列表

            _update_progress(int(50 * (i + 1) / total_files)) # 假设指标计算占 50%

        if not all_metrics_data:
            _update_status("错误：未能处理任何文件或计算指标。"); return False

        # --- 步骤 2: 指标标准化 (Z-score) ---
        _update_status("\n第二步：标准化指标...")
        metrics_df = pd.DataFrame(all_metrics_data)
        metric_columns = ['Std_Mean', 'RMS_Mean', 'Max_Abs_Amplitude',
                          'Ratio_MaxMeanAbs', 'Ratio_MaxStd', 'Ratio_MaxDevMeanMAD']

        # 计算 Z-scores 并添加到 DataFrame
        for col in metric_columns:
             zscore_col = f"{col}_zscore"
             metrics_df[zscore_col] = np.nan # 初始化 Z-score 列
             valid_data = metrics_df[col].dropna() # 选择非 NaN 值进行计算
             if len(valid_data) > 1: # 至少需要两个点来计算标准差
                 z_scores = stats.zscore(valid_data, nan_policy='omit')
                 metrics_df.loc[valid_data.index, zscore_col] = z_scores # 使用原始索引放回
             elif len(valid_data) == 1: # 如果只有一个有效点
                  metrics_df.loc[valid_data.index, zscore_col] = 0.0 # Z-score 为 0

        _update_progress(60)

        # --- 步骤 3: 计算综合评分 ---
        _update_status("第三步：计算综合伪迹评分...")
        metrics_df['Composite_Score'] = 0.0
        used_weights_info = []
        # 权重字典的键应与 metrics 字典的键 (小写) 一致
        weight_keys_map = {
            'std_mean': 'Std_Mean', 'rms_mean': 'RMS_Mean', 'max_abs_amp': 'Max_Abs_Amplitude',
            'ratio_max_meanabs': 'Ratio_MaxMeanAbs', 'ratio_max_std': 'Ratio_MaxStd',
            'ratio_maxdev_mad': 'Ratio_MaxDevMeanMAD'
        }
        for weight_key, df_col_base in weight_keys_map.items():
             weight = weights.get(weight_key, 0) # 使用小写键获取权重
             if weight != 0:
                 zscore_col = f"{df_col_base}_zscore" # 对应的 zscore 列名 (大写开头)
                 if zscore_col in metrics_df.columns:
                      # 用 0 替换 NaN Z-score，然后乘以权重累加
                      metrics_df['Composite_Score'] += metrics_df[zscore_col].fillna(0) * weight
                      used_weights_info.append(f"{weight_key}(w={weight})")
                 else:
                      _update_status(f"警告：未找到 Z-score 列 {zscore_col}。")

        _update_status(f"综合评分基于: {', '.join(used_weights_info)}")
        _update_progress(70)

        # --- 步骤 4: 标记伪迹并准备可视化 ---
        _update_status("第四步：标记伪迹并准备可视化...")
        is_threshold_valid = isinstance(composite_score_threshold, (int, float))
        if is_threshold_valid:
            metrics_df['Is_Artifact'] = metrics_df['Composite_Score'] > composite_score_threshold
        else:
            metrics_df['Is_Artifact'] = False # 如果阈值无效，则不标记
            _update_status("警告：未提供有效的综合评分阈值，不标记伪迹。")

        visualize_enabled = False
        if is_threshold_valid and visualize_dir: # 只有阈值有效且目录存在才启用
             try:
                 os.makedirs(visualize_dir, exist_ok=True)
                 visualize_enabled = True
                 _update_status(f"伪迹图表将保存到: '{visualize_dir}' (基于综合评分 > {composite_score_threshold:.2f})")
             except Exception as e:
                  _update_status(f"警告：无法创建图表目录 ({e}). 可视化已禁用。")
                  visualize_enabled = False
        elif is_threshold_valid:
             _update_status("警告：提供了综合评分阈值，但未提供目录。可视化已禁用。")

        # --- 步骤 5: 可选可视化 + 保存报告 ---
        num_artifacts_found = 0
        for index, report_row in metrics_df.iterrows():
            current_progress = 70 + int(30 * (index + 1) / total_files)
            _update_progress(current_progress)

            if visualize_enabled and report_row['Is_Artifact']:
                num_artifacts_found += 1
                filename = report_row['Epoch_Filename']
                score = report_row['Composite_Score']
                plot_filename = os.path.join(visualize_dir, filename.replace('.npy', '.png'))
                # _update_status(f"  检测到伪迹 ({filename}, Score={score:.2f})，生成图表...") # 减少日志
                try:
                    epoch_data_full = np.load(os.path.join(epoch_dir, filename)) # 重新加载以绘图
                    if epoch_data_full.ndim == 2 and epoch_data_full.shape[0] > 0 and epoch_data_full.shape[1] >= 3:
                        # --- 绘图逻辑 ---
                        plt.figure(figsize=(12, 6))
                        eeg_data_only = epoch_data_full[:, 2:]
                        num_channels_to_plot = min(8, eeg_data_only.shape[1])
                        time_axis = epoch_data_full[:, 0] if epoch_data_full.shape[0] > 0 else np.array([])
                        time_label = "Time (s, from col 0)"
                        if time_axis.size != eeg_data_only.shape[0]: # Fallback 时间轴
                            sampling_rate_est = eeg_data_only.shape[0]/4.0 if eeg_data_only.shape[0]>1 else 1.0
                            time_axis = np.arange(eeg_data_only.shape[0]) / sampling_rate_est if sampling_rate_est > 0 else np.arange(eeg_data_only.shape[0])
                            time_label = f"Time (s, assuming {sampling_rate_est:.1f} Hz)"

                        valid_eeg = eeg_data_only[:, :num_channels_to_plot][np.isfinite(eeg_data_only[:, :num_channels_to_plot])]
                        offset_scale = np.std(valid_eeg) * 2 if valid_eeg.size > 0 and np.std(valid_eeg) > 1e-9 else 1.0

                        for ch_idx in range(num_channels_to_plot):
                            actual_col_idx = ch_idx + 2
                            plt.plot(time_axis, epoch_data_full[:, actual_col_idx] + ch_idx * offset_scale, linewidth=0.8)

                        plt.title(f"{filename}\nComposite Score = {score:.2f} (> {composite_score_threshold:.2f})")
                        plt.xlabel(time_label)
                        plt.ylabel("Amplitude (EEG channels offset)")
                        plt.yticks([])
                        plt.grid(True, axis='x', linestyle=':')
                        plt.tight_layout()
                        plt.savefig(plot_filename)
                        plt.close()
                        # --- 绘图结束 ---
                    else: _update_status(f"警告：无法为 {filename} 生成图表，数据无效。")
                except Exception as plot_e: _update_status(f"  错误：无法为 {filename} 创建图表: {plot_e}")

        _update_status(f"检测到 {num_artifacts_found} 个伪迹分段（基于综合评分）。")

        # --- 保存 CSV ---
        _update_status("\n正在保存最终 CSV 报告...")
        # 定义最终列顺序
        final_columns = [
            'Epoch_Filename', 'Subject_ID', 'Sequence_Number', 'WAV_Filename_Base',
            'Std_Mean', 'RMS_Mean', 'Max_Abs_Amplitude', # 原始指标
            'Ratio_MaxMeanAbs', 'Ratio_MaxStd', 'Ratio_MaxDevMeanMAD', # 比率指标
            'Composite_Score', # 综合评分
            'Is_Artifact'      # 是否被标记为伪迹
            # 可以取消注释以包含 Z-score 列
            # 'Std_Mean_zscore', 'RMS_Mean_zscore', 'Max_Abs_Amplitude_zscore',
            # 'Ratio_MaxMeanAbs_zscore', 'Ratio_MaxStd_zscore', 'Ratio_MaxDevMeanMAD_zscore'
        ]
        # 使用 reindex 获取需要的列并按顺序排列，不存在的列会填充 NaN
        report_df_to_save = metrics_df.reindex(columns=final_columns)
        # 确保序号列是整数类型 (可空)
        report_df_to_save['Sequence_Number'] = pd.to_numeric(report_df_to_save['Sequence_Number'], errors='coerce').astype('Int64')
        report_df_to_save = report_df_to_save.sort_values(by=['Subject_ID', 'Sequence_Number'])
        try:
            report_df_to_save.to_csv(output_csv_path, index=False, float_format='%.4f', encoding='utf-8-sig')
            _update_status(f"最终报告已保存到: {output_csv_path}")
            return True
        except Exception as e:
            _update_status(f"错误：无法保存最终 CSV 报告: {e}")
            return False

    except FileNotFoundError:
        _update_status(f"错误：找不到目录 '{epoch_dir}'")
        _update_progress(0)
        return False
    except Exception as e:
        _update_status(f"处理过程中发生未知错误: {e}")
        import traceback
        traceback.print_exc()
        _update_progress(0)
        return False

# ===============================================
# GUI 主类 (Tkinter) - 支持综合评分和新指标
# ===============================================
class CompositeArtifactReporterApp:
    def __init__(self, master):
        self.master = master
        master.title("伪迹报告生成工具 v3 (综合评分)")
        master.geometry("700x650") # 增加高度

        # --- GUI 变量 ---
        self.epoch_dir_path = tk.StringVar()
        self.report_csv_path = tk.StringVar()
        self.visualize_var = tk.BooleanVar(value=False)
        self.visualize_dir_path = tk.StringVar()
        # 平滑选项
        self.smooth_var = tk.BooleanVar(value=True)
        self.sigma_var = tk.DoubleVar(value=2.0)
        # 权重变量 (键使用小写，与 metrics 字典匹配)
        self.weight_std_mean = tk.DoubleVar(value=1.0)
        self.weight_rms_mean = tk.DoubleVar(value=0.5)
        self.weight_max_abs = tk.DoubleVar(value=1.0)
        self.weight_ratio_maxmeanabs = tk.DoubleVar(value=0.5)
        self.weight_ratio_maxstd = tk.DoubleVar(value=0.5)
        self.weight_ratio_mad = tk.DoubleVar(value=1.5)
        # 综合评分阈值
        self.composite_threshold_var = tk.DoubleVar(value=3.0)

        # --- 创建控件 ---
        frame = ttk.Frame(master, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)

        row_idx = 0
        # 输入/输出路径选择
        ttk.Label(frame, text="包含 .npy 分段的文件夹:").grid(row=row_idx, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frame, textvariable=self.epoch_dir_path, width=60).grid(row=row_idx, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(frame, text="浏览...", command=self.browse_epoch_dir).grid(row=row_idx, column=3, padx=5, pady=5)
        frame.columnconfigure(1, weight=1)
        row_idx += 1

        ttk.Label(frame, text="保存 CSV 报告路径:").grid(row=row_idx, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frame, textvariable=self.report_csv_path, width=60).grid(row=row_idx, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(frame, text="选择/输入...", command=self.browse_report_csv).grid(row=row_idx, column=3, padx=5, pady=5)
        row_idx += 1

        # --- 平滑选项 ---
        smooth_frame = ttk.LabelFrame(frame, text="预处理", padding="5")
        smooth_frame.grid(row=row_idx, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        row_idx += 1
        sf_row = 0
        self.smooth_check = ttk.Checkbutton(smooth_frame, text="启用高斯平滑", variable=self.smooth_var, command=self.toggle_sigma_entry)
        self.smooth_check.grid(row=sf_row, column=0, padx=5, pady=2, sticky="w")
        ttk.Label(smooth_frame, text="Sigma (点数):").grid(row=sf_row, column=1, padx=(10,0), pady=2, sticky="w")
        self.sigma_entry = ttk.Entry(smooth_frame, textvariable=self.sigma_var, width=6, state='normal' if self.smooth_var.get() else 'disabled')
        self.sigma_entry.grid(row=sf_row, column=2, padx=5, pady=2, sticky="w")

        # --- 权重设置框架 ---
        weights_frame = ttk.LabelFrame(frame, text="指标权重 (用于综合评分)", padding="5")
        weights_frame.grid(row=row_idx, column=0, columnspan=4, padx=5, pady=10, sticky="ew")
        row_idx += 1

        wf_col = 0
        ttk.Label(weights_frame, text="Avg Std:").grid(row=0, column=wf_col, padx=(5,0), pady=2, sticky='w'); wf_col+=1
        ttk.Entry(weights_frame, textvariable=self.weight_std_mean, width=5).grid(row=0, column=wf_col, padx=(0,10), pady=2); wf_col+=1
        ttk.Label(weights_frame, text="Avg RMS:").grid(row=0, column=wf_col, padx=(5,0), pady=2, sticky='w'); wf_col+=1
        ttk.Entry(weights_frame, textvariable=self.weight_rms_mean, width=5).grid(row=0, column=wf_col, padx=(0,10), pady=2); wf_col+=1
        ttk.Label(weights_frame, text="Max Abs:").grid(row=0, column=wf_col, padx=(5,0), pady=2, sticky='w'); wf_col+=1
        ttk.Entry(weights_frame, textvariable=self.weight_max_abs, width=5).grid(row=0, column=wf_col, padx=(0,10), pady=2); wf_col+=1
        wf_col = 0 # New row
        ttk.Label(weights_frame, text="Max/MeanAbs:").grid(row=1, column=wf_col, padx=(5,0), pady=2, sticky='w'); wf_col+=1
        ttk.Entry(weights_frame, textvariable=self.weight_ratio_maxmeanabs, width=5).grid(row=1, column=wf_col, padx=(0,10), pady=2); wf_col+=1
        ttk.Label(weights_frame, text="Max/Std:").grid(row=1, column=wf_col, padx=(5,0), pady=2, sticky='w'); wf_col+=1
        ttk.Entry(weights_frame, textvariable=self.weight_ratio_maxstd, width=5).grid(row=1, column=wf_col, padx=(0,10), pady=2); wf_col+=1
        ttk.Label(weights_frame, text="MaxDev/MAD:").grid(row=1, column=wf_col, padx=(5,0), pady=2, sticky='w'); wf_col+=1
        ttk.Entry(weights_frame, textvariable=self.weight_ratio_mad, width=5).grid(row=1, column=wf_col, padx=(0,10), pady=2); wf_col+=1

        # 可视化选项
        ttk.Checkbutton(frame, text="为超阈值分段生成图表 (基于综合评分)", variable=self.visualize_var, command=self.toggle_visualization_options).grid(row=row_idx, column=0, columnspan=4, padx=5, pady=5, sticky="w")
        row_idx += 1

        # 可视化设置框架
        self.vis_frame = ttk.LabelFrame(frame, text="可视化选项", padding="5")
        self.vis_frame.grid(row=row_idx, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        row_idx += 1

        vis_row = 0
        ttk.Label(self.vis_frame, text="保存图表的文件夹:").grid(row=vis_row, column=0, padx=5, pady=2, sticky="w")
        self.vis_dir_entry = ttk.Entry(self.vis_frame, textvariable=self.visualize_dir_path, width=58, state='disabled')
        self.vis_dir_entry.grid(row=vis_row, column=1, columnspan=2, padx=5, pady=2, sticky=(tk.W, tk.E))
        self.vis_dir_button = ttk.Button(self.vis_frame, text="浏览...", command=self.browse_visualize_dir, state='disabled')
        self.vis_dir_button.grid(row=vis_row, column=3, padx=5, pady=2)
        self.vis_frame.columnconfigure(1, weight=1)
        vis_row += 1

        thresh_frame = ttk.Frame(self.vis_frame)
        thresh_frame.grid(row=vis_row, column=0, columnspan=4, pady=3, sticky='w')
        ttk.Label(thresh_frame, text="触发阈值: Composite Score >").pack(side=tk.LEFT, padx=5)
        self.vis_composite_threshold_entry = ttk.Entry(thresh_frame, textvariable=self.composite_threshold_var, width=8, state='disabled')
        self.vis_composite_threshold_entry.pack(side=tk.LEFT, padx=5)
        vis_row += 1

        # 启动按钮
        self.start_button = ttk.Button(frame, text="生成报告", command=self.start_report_generation_thread, style='Accent.TButton')
        self.start_button.grid(row=row_idx, column=1, columnspan=2, padx=5, pady=15)
        try:
            from ttkthemes import ThemedStyle
            style = ThemedStyle(master)
            style.set_theme("arc")
            style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'), padding=6)
        except ImportError:
             # print("ttkthemes 未安装，使用默认 ttk 风格。")
             s = ttk.Style()
             s.configure('TButton', font=('Segoe UI', 10), padding=5)
             s.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'), padding=6)
        row_idx += 1

        # 进度条和状态标签
        self.progress_bar = ttk.Progressbar(frame, orient="horizontal", length=600, mode="determinate")
        self.progress_bar.grid(row=row_idx, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        row_idx += 1
        self.status_label = ttk.Label(frame, text="准备就绪", anchor="w", justify="left")
        self.status_label.grid(row=row_idx, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        row_idx += 1

        # 日志/消息区域
        self.log_text = scrolledtext.ScrolledText(frame, height=8, width=80, state='disabled', wrap=tk.WORD)
        self.log_text.grid(row=row_idx, column=0, columnspan=4, padx=5, pady=5, sticky="ewns")
        frame.rowconfigure(row_idx, weight=1)
        row_idx += 1

        # 初始化控件状态
        self.toggle_sigma_entry()
        self.toggle_visualization_options()

    # --- 文件/目录浏览方法 ---
    def browse_epoch_dir(self):
        dirname = filedialog.askdirectory(title="选择包含 .npy 分段的文件夹")
        if dirname: self.epoch_dir_path.set(dirname)

    def browse_report_csv(self):
        filename = filedialog.asksaveasfilename(title="选择或输入 CSV 报告文件名", defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filename: self.report_csv_path.set(filename)

    def browse_visualize_dir(self):
        dirname = filedialog.askdirectory(title="选择保存图表的文件夹")
        if dirname: self.visualize_dir_path.set(dirname)

    # --- 切换控件状态 ---
    def toggle_sigma_entry(self):
         """根据复选框状态启用/禁用 Sigma 输入框"""
         new_state = 'normal' if self.smooth_var.get() else 'disabled'
         try: self.sigma_entry.config(state=new_state)
         except tk.TclError: pass

    def toggle_visualization_options(self):
        """切换可视化相关控件的可用性"""
        new_state = 'normal' if self.visualize_var.get() else 'disabled'
        try:
            for widget in self.vis_frame.winfo_children():
                if isinstance(widget, ttk.Frame): # 阈值框架
                    for sub_widget in widget.winfo_children():
                        try: sub_widget.config(state=new_state)
                        except tk.TclError: pass
                else: # 目录选择
                    try: widget.config(state=new_state)
                    except tk.TclError: pass
        except tk.TclError: pass

    # --- 日志和进度更新方法 ---
    def log_message(self, text):
        try:
            if self.master.winfo_exists():
                self.log_text.config(state='normal')
                self.log_text.insert(tk.END, text + '\n')
                self.log_text.see(tk.END)
                self.log_text.config(state='disabled')
        except Exception: print(f"Log Error: {text}")

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
                 if success: self.master.after(0, lambda: messagebox.showinfo("完成", "报告生成成功！"))
                 else: self.master.after(0, lambda: messagebox.showerror("错误", "报告生成过程中发生错误。"))
        except Exception: pass

    # --- 启动报告生成线程 ---
    def start_report_generation_thread(self):
        epoch_dir = self.epoch_dir_path.get()
        report_path = self.report_csv_path.get()
        do_visualize = self.visualize_var.get()
        vis_dir = self.visualize_dir_path.get()

        if not os.path.isdir(epoch_dir): messagebox.showerror("错误", "请选择有效的分段文件夹"); return
        if not report_path: messagebox.showerror("错误", "请指定 CSV 报告输出路径"); return

        # 获取权重
        try:
            weights = { # 使用小写键，与 calculate_artifact_metrics_all 返回的键匹配
                'std_mean': self.weight_std_mean.get(),
                'rms_mean': self.weight_rms_mean.get(),
                'max_abs_amp': self.weight_max_abs.get(),
                'ratio_max_meanabs': self.weight_ratio_maxmeanabs.get(),
                'ratio_max_std': self.weight_ratio_maxstd.get(),
                'ratio_maxdev_mad': self.weight_ratio_mad.get()
            }
            if any(w < 0 for w in weights.values()): raise ValueError("权重不能为负数")
        except (tk.TclError, ValueError) as e: messagebox.showerror("错误", f"无效权重: {e}"); return

        # 获取平滑设置
        sigma_value = None
        if self.smooth_var.get():
            try:
                sigma_value = self.sigma_var.get()
                if sigma_value <= 0: raise ValueError("Sigma 必须为正数")
            except (tk.TclError, ValueError) as e: messagebox.showerror("错误", f"无效 Sigma: {e}"); return

        # 获取可视化阈值（仅综合评分）
        vis_comp_threshold = None
        if do_visualize:
            if not vis_dir or not os.path.isdir(vis_dir):
                 default_vis_dir = os.path.join(epoch_dir, "artifact_plots_composite")
                 try:
                     os.makedirs(default_vis_dir, exist_ok=True)
                     self.visualize_dir_path.set(default_vis_dir)
                     vis_dir = default_vis_dir
                     self.log_message(f"图表目录无效，使用默认：{default_vis_dir}")
                 except Exception as e: messagebox.showerror("错误", f"无法创建图表目录：{e}"); return
            try:
                vis_comp_threshold = self.composite_threshold_var.get() # 获取综合评分阈值
            except (tk.TclError, ValueError) as e: messagebox.showerror("错误", f"无效综合评分阈值: {e}"); return

        self.start_button.config(state="disabled")
        self.update_progress(0)
        self.log_text.config(state='normal'); self.log_text.delete('1.0', tk.END); self.log_text.config(state='disabled')
        self.log_message(f"开始生成报告 (平滑: {'启用, Sigma='+str(sigma_value) if sigma_value else '禁用'})...")

        self.processing_thread = threading.Thread(
            target=self._run_report_generation,
            # 传递所有需要的参数
            args=(epoch_dir, report_path, weights, vis_comp_threshold, sigma_value, vis_dir),
            daemon=True
        )
        self.processing_thread.start()

    def _run_report_generation(self, epoch_dir, report_path, weights, vis_comp_threshold, sigma_value, vis_dir):
        """在后台线程中执行报告生成"""
        success = False
        try:
            # 调用包含所有功能的核心函数
            success = generate_artifact_report_composite_with_smoothing(
                epoch_dir, report_path,
                weights=weights,
                composite_score_threshold=vis_comp_threshold,
                smoothing_sigma=sigma_value, # 传递平滑参数
                visualize_dir=vis_dir if vis_comp_threshold is not None else None,
                progress_callback=self.update_progress,
                status_callback=self.update_status
            )
        except Exception as e:
            self.master.after(0, lambda m=f"报告生成线程出错: {e}": self.log_message(m))
            import traceback
            traceback.print_exc()
            success = False
        finally:
            self.processing_finished(success)


# ===============================================
# 应用程序入口点
# ===============================================
if __name__ == "__main__":
    root = tk.Tk()
    try: # 尝试应用主题
        from ttkthemes import ThemedTk
        root = ThemedTk(theme="arc") # 使用 'arc' 主题
    except ImportError:
        print("ttkthemes 未安装，使用默认 ttk 风格。")
        s = ttk.Style()
        s.theme_use('clam') # 或者 'alt', 'default', 'classic'
        # 为默认 ttk 添加一些样式
        s.configure('TButton', padding=6)
        s.configure('TEntry', padding=5)
        s.configure('TLabel', padding=2)
        s.configure('TCheckbutton', padding=3)
        s.configure('TLabelframe.Label', font=('Segoe UI', 9, 'bold'))
        s.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'), padding=6)

    app = CompositeArtifactReporterApp(root) # 运行新的 GUI 类
    root.mainloop()