# EEG Audio Benchmark 项目说明

本仓库提供了一条从原始 EEG/音频数据到可直接用于基准回归实验的完整流水线，并配套了一个可配置的实验框架 `eeg_audio_benchmark`。本说明文档旨在帮助你理解目前的代码架构、主要模块之间的关系以及常见的使用方式。

## 架构概览

```
Master-Program/
├── src/eeg_audio_benchmark/        # 核心 Python 包
│   ├── preprocessing/              # EEG/音频预处理流水线
│   ├── representations.py          # EEG/音频特征注册器
│   ├── datasets.py                 # 对齐后的数据加载与组装
│   ├── models/                     # 可插拔的回归模型
│   ├── evaluation.py               # 统一的评估指标
│   ├── experiment.py               # 配置驱动的实验运行器
│   └── bin/                        # 命令行入口（预处理）
├── configs/                        # 示例 YAML 配置
├── data/                           # 放置原始及中间数据的目录
└── notebooks/                      # 探索性分析或示例 notebook
```

核心思想如下：

1. **预处理流水线**（`src/eeg_audio_benchmark/preprocessing/`）将四个历史 Tk GUI 脚本拆解为纯函数：
   - `eeg_epochs.py`：从 CBYT 文件与刺激注释切片出固定长度的 EEG epoch。
   - `artifacts.py`：计算多种伪迹指标，输出包含 `Is_Artifact` 的 CSV 报告。
   - `audio_features.py`：批量抽取 Librosa 音频特征并保存矩阵/时间戳。
   - `alignment.py`：对“干净”epoch 与音频特征做滑窗、插值并输出对齐后的 `.npy`。
2. **实验框架**（`datasets.py`, `representations.py`, `models/`, `evaluation.py`, `experiment.py`）负责将对齐好的 EEG/音频矩阵装配成 lagged design matrix，执行划分、建模与评估。
3. **配置系统**（`config.py` + `preprocessing/utils.py`）提供加载 YAML 与 dataclass 解析的统一入口，使 CLI 与 Python API 都能通过同一份配置运行。

## 预处理流水线

四个步骤彼此独立，可以单独运行，也可以通过总入口串联执行。所有函数都接受对应的 dataclass 配置，字段含义与历史 GUI 选项一致。

### 1. EEG 切片 (`slice_eeg_to_epochs`)
- 输入：CBYT 原始 EEG、刺激说明 Excel/CSV、可选 marker 表。
- 输出：`<cbyt>_<seq>_<wav>.npy`，包含时间、标记、32 个通道。
- 关键配置：`epoch_duration`、`anchor`（start/end/auto）、`resample_hz`、`smoothing_sigma`。
- CLI：
  ```bash
  python -m eeg_audio_benchmark.bin.run_eeg_epochs --config configs/eeg_epochs.yaml
  ```

### 2. 伪迹评分 (`compute_artifact_report`)
- 输入：步骤 1 生成的 epoch 目录。
- 输出：含各类指标、Z 分数、`Composite_Score`、`Is_Artifact` 的 CSV，可选伪迹可视化。
- 关键配置：`metric_weights`、`composite_threshold`、`smoothing_sigma`、`artifact_plots_dir`。
- CLI：
  ```bash
  python -m eeg_audio_benchmark.bin.run_artifacts --config configs/artifacts.yaml
  ```

### 3. 音频特征 (`extract_audio_features`)
- 输入：WAV 音频目录。
- 输出：每个音频的 `*_features.npy`、`*_feature_times.npy`、`*_feature_names.txt`。
- 关键配置：`frame_length`、`hop_length`、`n_mfcc`、`pyin_fmin/fmax`、统一采样率。
- CLI：
  ```bash
  python -m eeg_audio_benchmark.bin.run_audio_features --config configs/audio_features.yaml
  ```

### 4. EEG/音频对齐 (`align_eeg_audio_pairs`)
- 输入：epoch 目录、伪迹 CSV、音频特征目录。
- 输出：`*_EEG_aligned.npy` 与 `*_Sound_aligned.npy`，采样在统一的目标网格上。
- 关键配置：`eeg_sampling_rate`、`eeg_window_ms`、`eeg_step_ms`、`target_duration`、`target_hop`、`missing_policy`。
- CLI：
  ```bash
  python -m eeg_audio_benchmark.bin.run_alignment --config configs/alignment.yaml
  ```

### 串联运行

若想一次性跑完整条流水线，可编写如下顶层配置并调用 `run_preprocessing`：

```yaml
# configs/pipeline.yaml
log_level: INFO

eeg_epochs:
  enabled: true
  output_dir: data/epochs
  cbyt_dir: data/raw/cbyt
  stimulus_table: data/meta/stimuli.xlsx
  epoch_duration: 4.0
  anchor: auto

artifacts:
  enabled: true
  epoch_dir: data/epochs
  output_csv: data/epochs/artifact_report.csv

audio_features:
  enabled: true
  wav_dir: data/audio
  output_dir: data/audio_features

alignment:
  enabled: true
  eeg_epoch_dir: data/epochs
  artifact_report: data/epochs/artifact_report.csv
  audio_feature_dir: data/audio_features
  output_dir: data/aligned
```

运行命令：

```bash
python -m eeg_audio_benchmark.bin.run_preprocessing --config configs/pipeline.yaml
```

## 与实验框架衔接

完成对齐后即可在 `ExperimentRunner` 中引用 `.npy`：

1. 在 `configs/` 下编写实验 YAML，包含 `dataset`、`model`、`splits` 与 `output_dir`。`dataset` 区域需要填写 EEG/音频文件路径、任务类型（`encoding`/`decoding`）、representation 名称（参见 `representations.py`）与滞后列表。
2. 通过 Python API 运行：
   ```python
   from eeg_audio_benchmark.experiment import run_from_config
   run_from_config("configs/experiment.yaml")
   ```
3. 亦可在 notebook 中直接实例化 `ExperimentRunner`，检查 `runner.dataset.X`/`Y` 或 `runner.run()` 的指标（`r2`、`pearson`）。

## 开发与扩展建议

- **新增预处理步骤**：在 `preprocessing/` 下新增模块，并在 `bin/` 中创建 CLI，复用 `preprocessing.utils.parse_config` 以保持配置体验一致。
- **自定义特征/模型**：
  - 在 `representations.py` 注册新的 EEG/音频表示函数。
  - 在 `models/` 中实现 `build_model` 可识别的新模型名称及其参数。
- **配置复用**：`config.load_config` 支持任意 YAML，因此可以在单个文件中集中放多个步骤的配置，通过 `enabled` 字段选择执行哪些步骤。
- **日志与调试**：每个步骤的 dataclass 均包含 `log_level` 字段，可设置为 `DEBUG` 以输出更多处理细节。

如需进一步了解特定函数的使用方式，可直接阅读 `src/eeg_audio_benchmark/preprocessing/` 中的源码——所有核心逻辑均为纯函数，便于脚本或测试调用。
