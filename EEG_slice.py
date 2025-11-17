# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import struct
import numpy as np
import pandas as pd
import os
import re # 用于解析时间输入

# ===============================================
# 数据处理函数
# ===============================================

def read_stimulus_map_interactive(file_path):
    """
    读取 Описание.xlsx 的 'Лист3' 工作表。
    返回 {刺激序号: {'wav_filename': str}}
    """
    stimulus_map = {}
    status_updates = []
    sheet_name_to_read = "Лист3"
    try:
        # header=None, skiprows=1: 明确跳过标题行
        df_stim = pd.read_excel(file_path, sheet_name=sheet_name_to_read, header=None, skiprows=1)
        status_updates.append(f"尝试从 Excel '{os.path.basename(file_path)}' Sheet '{sheet_name_to_read}' 读取 (跳过第1行)...")
    except Exception as e_excel:
        status_updates.append(f"Excel 读取失败 ({e_excel}), 尝试 CSV...")
        try:
            df_stim = pd.read_csv(file_path, header=None, skiprows=1) # CSV 也跳过第一行
            status_updates.append(f"尝试从 CSV '{os.path.basename(file_path)}' 读取 (跳过第1行)...")
        except Exception as e_csv:
            status_updates.append(f"错误：无法读取刺激文件。Excel: {e_excel}, CSV: {e_csv}")
            return None, status_updates

    expected_col_idx_seq = 0 # 'Номер по порядку предъявления'
    expected_col_idx_wav = 3 # 'Звуковой файл'

    if df_stim.shape[1] <= max(expected_col_idx_seq, expected_col_idx_wav):
         status_updates.append(f"错误：'{sheet_name_to_read}' 列数不足。")
         return None, status_updates

    read_count = 0
    skipped_rows = [] # 记录因为序号无效而跳过的行
    for index, row in df_stim.iterrows():
        actual_file_row_num = index + 2 # 文件中的实际行号
        try:
            seq_num_val = row.iloc[expected_col_idx_seq]
            if pd.isna(seq_num_val):
                skipped_rows.append(actual_file_row_num)
                continue
            sequence_number = int(seq_num_val)

            wav_filename = "UnknownWav"
            if expected_col_idx_wav < len(row) and not pd.isna(row.iloc[expected_col_idx_wav]):
                wav_filename = str(row.iloc[expected_col_idx_wav])

            stimulus_map[sequence_number] = {'wav_filename': wav_filename}
            read_count += 1
        except (IndexError, ValueError, TypeError) as e:
             skipped_rows.append(actual_file_row_num)
             status_updates.append(f"警告：处理文件行 {actual_file_row_num} 时出错: {e}. 跳过。")
             continue

    if skipped_rows:
         status_updates.append(f"加载刺激映射时跳过了以下行（无效序号或错误）：{skipped_rows}")
    status_updates.append(f"成功从 '{sheet_name_to_read}' 加载 {read_count} 条刺激信息。")
    return stimulus_map, status_updates


def load_cbyt_data_interactive(cbyt_path):
    """加载 .CBYT 文件"""
    # ... (代码与上个回答相同) ...
    status_updates = []
    try:
        status_updates.append(f"正在加载 .CBYT 文件: {os.path.basename(cbyt_path)}...")
        bytes_per_row = 268
        cbyt_dtype = np.dtype([('time_sec', 'd'), ('original_marker', 'i'), ('data', 'd', (32,))])
        file_size = os.path.getsize(cbyt_path)
        if file_size % bytes_per_row != 0:
             status_updates.append(f"警告：文件大小 ({file_size}) 不能被行大小 ({bytes_per_row}) 整除。")
        original_data_struct = np.fromfile(cbyt_path, dtype=cbyt_dtype)
        n_times = len(original_data_struct)
        if n_times == 0:
            status_updates.append("错误：未能从 .CBYT 文件读取任何数据")
            return None, None, status_updates
        original_times_sec = original_data_struct['time_sec']
        original_eeg_data = original_data_struct['data']
        status_updates.append(f"成功加载 {n_times} 个时间点。")
        return original_times_sec, original_eeg_data, status_updates
    except FileNotFoundError:
        status_updates.append(f"错误: 输入文件未找到 {cbyt_path}")
        return None, None, status_updates
    except Exception as e:
        status_updates.append(f"加载 .CBYT 文件时发生错误: {e}")
        return None, None, status_updates


def read_marker_times_file(file_path):
    """
    读取包含 'time of beginning' 和 'time of the end' 的文件。
    返回一个字典 {stimulus_number: {'start': float or None, 'end': float or None}}
    假设第一行是标题，刺激序号从 1 开始对应文件的第 2 行。
    """
    marker_times = {}
    status_updates = []
    try:
        # 尝试读取 Excel，跳过标题行
        df_times = pd.read_excel(file_path, header=0)
        status_updates.append(f"尝试从 Excel '{os.path.basename(file_path)}' 读取标记时间 (带标题)...")
    except Exception as e_excel:
        status_updates.append(f"Excel 读取失败 ({e_excel}), 尝试 CSV...")
        try:
            # 尝试 CSV，跳过标题行
            df_times = pd.read_csv(file_path, header=0)
            status_updates.append(f"尝试从 CSV '{os.path.basename(file_path)}' 读取标记时间 (带标题)...")
        except Exception as e_csv:
            status_updates.append(f"错误：无法读取标记时间文件。Excel: {e_excel}, CSV: {e_csv}")
            return None, status_updates

    # 检查必需的列名 (不区分大小写，去除首尾空格)
    expected_cols = ['time of beginning', 'time of the end']
    actual_cols = [str(col).strip().lower() for col in df_times.columns]

    if expected_cols[0] not in actual_cols or expected_cols[1] not in actual_cols:
        status_updates.append(f"错误：标记时间文件缺少必需的列标题: '{expected_cols[0]}' 和/或 '{expected_cols[1]}'")
        status_updates.append(f"   找到的列标题: {df_times.columns.tolist()}")
        return None, status_updates

    # 获取列的实际索引（因为用户可能大小写或空格不同）
    col_start_idx = actual_cols.index(expected_cols[0])
    col_end_idx = actual_cols.index(expected_cols[1])

    read_count = 0
    # index 对应 DataFrame 行号 (0 开始)，刺激序号是 index + 1
    for index, row in df_times.iterrows():
        stimulus_number = index + 1
        start_time = None
        end_time = None

        try:
            # 尝试读取开始时间
            start_val = row.iloc[col_start_idx]
            if pd.notna(start_val):
                try:
                    start_time = float(start_val)
                except (ValueError, TypeError):
                    status_updates.append(f"警告：标记时间文件，刺激 {stimulus_number}，开始时间 '{start_val}' 无效。忽略。")

            # 尝试读取结束时间
            end_val = row.iloc[col_end_idx]
            if pd.notna(end_val):
                try:
                    end_time = float(end_val)
                except (ValueError, TypeError):
                    status_updates.append(f"警告：标记时间文件，刺激 {stimulus_number}，结束时间 '{end_val}' 无效。忽略。")

            # 只有当至少有一个有效时间时才添加到字典
            if start_time is not None or end_time is not None:
                marker_times[stimulus_number] = {'start': start_time, 'end': end_time}
                read_count += 1
            # else: # 如果两个时间都无效或为空，则不添加，稍后会被跳过
            #    status_updates.append(f"提示：刺激 {stimulus_number} 在标记时间文件中没有有效的开始或结束时间。")

        except IndexError:
            status_updates.append(f"警告：处理标记时间文件行 {index+2} 时发生索引错误。跳过。")
            continue
        except Exception as e:
            status_updates.append(f"警告：处理标记时间文件行 {index+2} 时发生未知错误: {e}。跳过。")
            continue

    status_updates.append(f"成功从标记时间文件加载 {read_count} 个刺激的时间信息。")
    return marker_times, status_updates

def process_and_save_all_epochs(
    original_times_sec, original_eeg_data,
    stim_map, marker_times_dict, # 使用新的标记时间字典
    output_dir, epoch_duration_sec, cbyt_filename_base,
    progress_callback, status_callback):
    """
    根据读取的标记时间文件处理所有刺激并保存分段。
    """
    try:
        status_callback("开始分段处理...")
        progress_callback(0)

        n_times = len(original_times_sec)
        if n_times == 0: raise ValueError("EEG 数据未加载。")

        # 估算采样率
        sampling_rate = 0
        epoch_points = int(epoch_duration_sec * 1000) # Default
        if n_times > 1:
             mean_diff_sec = np.mean(np.diff(original_times_sec))
             if mean_diff_sec > 1e-9:
                 sampling_rate = 1.0 / mean_diff_sec
                 epoch_points = int(epoch_duration_sec * sampling_rate)
        status_callback(f"每个分段目标点数: {epoch_points} (基于采样率 ~{sampling_rate:.1f} Hz)")

        num_processed = 0
        num_skipped = 0
        num_saved = 0
        total_stimuli = max(stim_map.keys()) if stim_map else 0 # 找到最大的刺激序号

        if total_stimuli == 0:
            status_callback("错误：无法确定刺激总数。")
            return False

        for stim_num in range(1, total_stimuli + 1):
            status_callback(f"处理刺激 {stim_num}/{total_stimuli}...")
            current_progress = int(90 * stim_num / total_stimuli) # 进度条 (0-90%)
            progress_callback(current_progress)

            stim_info = stim_map.get(stim_num)
            marker_time_info = marker_times_dict.get(stim_num)

            if not stim_info:
                status_callback(f"警告：刺激 {stim_num} 在 Описание 文件中未找到。跳过。")
                num_skipped += 1
                continue

            if not marker_time_info:
                status_callback(f"提示：刺激 {stim_num} 在标记时间文件中无有效时间。跳过。")
                num_skipped += 1
                continue

            start_time = marker_time_info.get('start')
            end_time = marker_time_info.get('end')
            anchor_time = None
            anchor_type = None

            # 决定使用哪个时间点
            if start_time is not None:
                anchor_time = start_time
                anchor_type = "Start"
            elif end_time is not None:
                anchor_time = end_time
                anchor_type = "End"
            else:
                # 这个理论上不应该发生，因为我们在 read_marker_times_file 中已经过滤掉了
                status_callback(f"提示：刺激 {stim_num} 无有效的开始或结束时间。跳过。")
                num_skipped += 1
                continue

            # --- 提取逻辑 ---
            nearest_idx = np.argmin(np.abs(original_times_sec - anchor_time))
            actual_marked_time = original_times_sec[nearest_idx]
            time_diff = abs(actual_marked_time - anchor_time)
            sub_status = ""
            if time_diff > 0.1:
                sub_status += f"时间点警告 (差 {time_diff:.3f}s). "

            start_idx = -1
            end_idx = -1
            if anchor_type == "Start":
                start_idx = nearest_idx
                end_idx = start_idx + epoch_points
            elif anchor_type == "End":
                end_idx = nearest_idx + 1 # 结束索引是开区间，所以用下一个点
                start_idx = end_idx - epoch_points
                if start_idx < 0: start_idx = 0 # 防止负索引

            # 检查边界并提取
            if start_idx < 0 or start_idx >= n_times:
                status_callback(f"错误：刺激 {stim_num} 计算出的起始索引({start_idx})无效。跳过。")
                num_skipped += 1
                continue
            if end_idx > n_times:
                sub_status += f"超出末尾，截断. "
                end_idx = n_times
            if start_idx >= end_idx:
                 status_callback(f"错误：刺激 {stim_num} 无效的分段区间 [{start_idx}, {end_idx})。跳过。")
                 num_skipped += 1
                 continue

            epoch_original_times_sec_float64 = original_times_sec[start_idx:end_idx]
            epoch_eeg_data_float64 = original_eeg_data[start_idx:end_idx, :]
            actual_points_extracted = epoch_eeg_data_float64.shape[0]

            if actual_points_extracted == 0:
                status_callback(f"错误：刺激 {stim_num} 未能提取任何数据点。跳过。")
                num_skipped += 1
                continue

            # --- 创建标记列和相对时间 ---
            marker_col = np.full((actual_points_extracted, 1), -1.0, dtype=np.float32)
            if actual_points_extracted > 0: marker_col[0, 0] = float(stim_num)
            if actual_points_extracted > 0:
                 t0 = epoch_original_times_sec_float64[0]
                 epoch_relative_times_sec_float64 = epoch_original_times_sec_float64 - t0
            else: epoch_relative_times_sec_float64 = np.array([], dtype=np.float64)

            epoch_relative_times_sec_float32 = epoch_relative_times_sec_float64.astype(np.float32)
            epoch_eeg_data_float32 = epoch_eeg_data_float64.astype(np.float32)

            data_to_save = np.hstack((
                epoch_relative_times_sec_float32.reshape(-1, 1),
                marker_col,
                epoch_eeg_data_float32
            ))

            # --- 保存文件 ---
            wav_filename_raw = stim_info.get('wav_filename', f"Stim{stim_num}")
            wav_filename_base = os.path.basename(wav_filename_raw).replace('.wav', '')
            wav_filename_safe = re.sub(r'[\\/*?:"<>|]', '_', wav_filename_base)
            epoch_filename = f"{cbyt_filename_base}_{stim_num:03d}_{wav_filename_safe}.npy"
            output_epoch_path = os.path.join(output_dir, epoch_filename)

            try:
                np.save(output_epoch_path, data_to_save)
                num_saved += 1
            except Exception as e:
                status_callback(f"错误：无法保存刺激 {stim_num} 的分段 {epoch_filename}: {e}")
                num_skipped += 1 # 保存失败也算跳过

            num_processed += 1

        progress_callback(100)
        status_callback(f"处理完成。总刺激数: {total_stimuli}, 已处理: {num_processed}, 成功保存: {num_saved}, 跳过: {num_skipped}。")
        return True

    except Exception as e:
        status_callback(f"批处理分段时发生错误: {e}")
        progress_callback(0)
        return False

# ===============================================
# GUI 主类 (Tkinter) - 修改后支持批处理
# ===============================================
class BatchBytProcessorApp:
    def __init__(self, master):
        self.master = master
        master.title("CBYT 批处理分段工具 (基于标记时间文件)")
        master.geometry("700x500") # 调整大小

        # --- 数据存储 ---
        self.cbyt_file_path = tk.StringVar()
        self.xlsx_opisan_path = tk.StringVar() # Описание.xlsx
        self.xlsx_markers_path = tk.StringVar() # 新的标记时间文件
        self.output_dir_path = tk.StringVar()
        self.stimulus_map = None
        self.marker_times_dict = None
        self.original_times_sec = None
        self.original_eeg_data = None
        self.cbyt_filename_base = ""

        # --- GUI 变量 ---
        self.epoch_duration_var = tk.DoubleVar(value=4.0)

        # --- 创建控件 ---
        row_idx = 0
        # 文件/目录选择
        tk.Label(master, text="原始 .CBYT 文件:").grid(row=row_idx, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(master, textvariable=self.cbyt_file_path, width=60).grid(row=row_idx, column=1, padx=5, pady=5)
        tk.Button(master, text="浏览...", command=self.browse_cbyt).grid(row=row_idx, column=2, padx=5, pady=5)
        row_idx += 1

        tk.Label(master, text="Описание.xlsx/csv:").grid(row=row_idx, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(master, textvariable=self.xlsx_opisan_path, width=60).grid(row=row_idx, column=1, padx=5, pady=5)
        tk.Button(master, text="浏览...", command=self.browse_opisan_xlsx).grid(row=row_idx, column=2, padx=5, pady=5)
        row_idx += 1

        # --- 新增：选择标记时间文件 ---
        tk.Label(master, text="标记时间文件 (Excel/CSV):").grid(row=row_idx, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(master, textvariable=self.xlsx_markers_path, width=60).grid(row=row_idx, column=1, padx=5, pady=5)
        tk.Button(master, text="浏览...", command=self.browse_markers_xlsx).grid(row=row_idx, column=2, padx=5, pady=5)
        row_idx += 1
        # -----------------------------

        tk.Label(master, text="保存分段的文件夹:").grid(row=row_idx, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(master, textvariable=self.output_dir_path, width=60).grid(row=row_idx, column=1, padx=5, pady=5)
        tk.Button(master, text="浏览...", command=self.browse_output_dir).grid(row=row_idx, column=2, padx=5, pady=5)
        row_idx += 1

        tk.Label(master, text="分段时长 (秒):").grid(row=row_idx, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(master, textvariable=self.epoch_duration_var, width=10).grid(row=row_idx, column=1, padx=5, pady=5, sticky="w")
        row_idx += 1

        # 启动按钮
        self.start_button = tk.Button(master, text="开始批处理", command=self.start_batch_processing_thread, width=15, height=2)
        self.start_button.grid(row=row_idx, column=1, padx=5, pady=15)
        row_idx += 1

        # 进度条和状态标签
        self.progress_bar = ttk.Progressbar(master, orient="horizontal", length=600, mode="determinate")
        self.progress_bar.grid(row=row_idx, column=0, columnspan=3, padx=5, pady=5)
        row_idx += 1
        self.status_label = tk.Label(master, text="准备就绪", anchor="w", justify="left")
        self.status_label.grid(row=row_idx, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        row_idx += 1

        # 日志/消息区域
        self.log_text = scrolledtext.ScrolledText(master, height=8, width=80, state='disabled')
        self.log_text.grid(row=row_idx, column=0, columnspan=3, padx=5, pady=5)
        row_idx += 1


    # --- 文件/目录浏览方法 ---
    def browse_cbyt(self):
        filename = filedialog.askopenfilename(title="选择 .CBYT 文件", filetypes=[("CBYT files", "*.cbyt *.CBYT"), ("All files", "*.*")])
        if filename: self.cbyt_file_path.set(filename)

    def browse_opisan_xlsx(self): # 修改函数名以区分
        filename = filedialog.askopenfilename(title="选择 Описание 文件", filetypes=[("Excel/CSV", "*.xlsx *.csv"), ("All files", "*.*")])
        if filename: self.xlsx_opisan_path.set(filename)

    def browse_markers_xlsx(self): # 新增：浏览标记时间文件
        filename = filedialog.askopenfilename(title="选择标记时间文件", filetypes=[("Excel/CSV", "*.xlsx *.csv"), ("All files", "*.*")])
        if filename: self.xlsx_markers_path.set(filename)

    def browse_output_dir(self):
        dirname = filedialog.askdirectory(title="选择保存分段的文件夹")
        if dirname: self.output_dir_path.set(dirname)

    # --- 日志和进度更新方法 (与之前类似) ---
    def log_message(self, text):
        try:
            self.log_text.config(state='normal')
            self.log_text.insert(tk.END, text + '\n')
            self.log_text.see(tk.END)
            self.log_text.config(state='disabled')
        except tk.TclError: pass # Ignore if GUI closed

    def update_progress(self, value):
        try: self.master.after(0, lambda: self.progress_bar.config(value=value))
        except tk.TclError: pass

    def update_status(self, text):
        try:
            self.master.after(0, lambda: self.status_label.config(text=text))
            self.log_message(text) # 同时记录到日志
        except tk.TclError: pass

    def processing_finished(self, success):
        try:
            self.master.after(0, lambda: self.start_button.config(state="normal"))
            if success: self.master.after(0, lambda: messagebox.showinfo("完成", "批处理成功完成！"))
            else: self.master.after(0, lambda: messagebox.showerror("错误", "批处理过程中发生错误，请查看日志。"))
        except tk.TclError: pass

    # --- 启动批处理线程 ---
    def start_batch_processing_thread(self):
        # 1. 获取并验证输入
        cbyt_path = self.cbyt_file_path.get()
        opisan_path = self.xlsx_opisan_path.get()
        markers_path = self.xlsx_markers_path.get() # 获取新文件路径
        output_dir = self.output_dir_path.get()

        if not os.path.isfile(cbyt_path): messagebox.showerror("错误", ".CBYT 文件未找到"); return
        if not os.path.isfile(opisan_path): messagebox.showerror("错误", "Описание 文件未找到"); return
        if not os.path.isfile(markers_path): messagebox.showerror("错误", "标记时间文件未找到"); return # 验证新文件
        if not os.path.isdir(output_dir): messagebox.showerror("错误", "输出文件夹不存在"); return

        try:
            epoch_duration = self.epoch_duration_var.get()
            if epoch_duration <= 0: raise ValueError
        except: messagebox.showerror("错误", "无效的分段时长"); return

        # 2. 清空日志，禁用按钮，启动线程
        self.start_button.config(state="disabled")
        self.update_progress(0)
        self.log_text.config(state='normal'); self.log_text.delete('1.0', tk.END); self.log_text.config(state='disabled')
        self.log_message("开始批处理...")

        self.processing_thread = threading.Thread(
            target=self._run_batch_processing,
            args=(cbyt_path, opisan_path, markers_path, output_dir, epoch_duration),
            daemon=True
        )
        self.processing_thread.start()

    def _run_batch_processing(self, cbyt_path, opisan_path, markers_path, output_dir, epoch_duration):
        """在后台线程中执行加载和处理"""
        success = False
        try:
            # 1. 加载 Описание 文件
            stim_map, stim_status = read_stimulus_map_interactive(opisan_path)
            for msg in stim_status: self.master.after(0, lambda m=msg: self.log_message(m))
            if stim_map is None: raise ValueError("无法加载刺激映射")

            # 2. 加载标记时间文件
            marker_times, marker_status = read_marker_times_file(markers_path)
            for msg in marker_status: self.master.after(0, lambda m=msg: self.log_message(m))
            if marker_times is None: raise ValueError("无法加载标记时间")

            # 3. 加载 CBYT 数据
            times, data, cbyt_status = load_cbyt_data_interactive(cbyt_path)
            for msg in cbyt_status: self.master.after(0, lambda m=msg: self.log_message(m))
            if times is None or data is None: raise ValueError("无法加载 CBYT 数据")

            cbyt_base = os.path.splitext(os.path.basename(cbyt_path))[0]

            # 4. 执行批处理分段
            success = process_and_save_all_epochs(
                times, data, stim_map, marker_times,
                output_dir, epoch_duration, cbyt_base,
                self.update_progress, self.update_status
            )
        except Exception as e:
            # 捕获加载或处理过程中的任何异常
            self.master.after(0, lambda m=f"批处理线程出错: {e}": self.log_message(m))
            success = False
        finally:
            # 确保完成后调用 finished 回调
            self.processing_finished(success)


# ===============================================
# 应用程序入口点
# ===============================================
if __name__ == "__main__":
    root = tk.Tk()
    app = BatchBytProcessorApp(root)
    root.mainloop()