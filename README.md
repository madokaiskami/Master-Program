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

1. **预处理流水线**（`src/eeg_audio_benchmark/preprocessing/`）围绕 HuggingFace 数据集 `Aurelianous/eeg-audio` 展开，使用 HF manifest、原始 EEG `.npz`、事件 CSV 与 WAV 为唯一输入：
   - `eeg_epochs.py`：基于事件表切片 EEG，输出 epoch `.npy` 与 epoch manifest。
   - `artifacts.py`：计算多种伪迹指标，输出包含 `Is_Artifact` 的 CSV 报告。
   - `audio_features.py`：批量抽取 Librosa 音频特征并保存矩阵/时间戳，同时生成特征 manifest。
   - `alignment.py`：对“干净”epoch 与音频特征做滑窗、插值并输出对齐后的 `.npy` 与对齐 manifest。
2. **实验框架**（`datasets.py`, `representations.py`, `models/`, `evaluation.py`, `experiment.py`）负责将对齐好的 EEG/音频矩阵装配成 lagged design matrix，执行划分、建模与评估。
3. **配置系统**（`config.py` + `preprocessing/utils.py`）提供加载 YAML 与 dataclass 解析的统一入口，使 CLI 与 Python API 都能通过同一份配置运行。

## 预处理流水线

四个步骤彼此独立，可以单独运行，也可以通过总入口串联执行。所有函数都接受对应的 dataclass 配置，字段围绕 HF manifest 设计。

### 1. EEG 切片 (`slice_eeg_to_epochs`)
- 输入：`manifest_raw_runs.csv` + 对应的 `raw/eeg/*.npz` 和 `raw/events/*.csv`。
- 输出：`{subject}_{run}_evt-<idx>_<stim>.npy`（时间、事件索引、通道）和 `derivatives/epoch_manifest.csv`。
- 关键配置：`epoch_duration_sec`、`anchor`（onset/center）、`resample_hz`、`smoothing_sigma`。

### 2. 伪迹评分 (`compute_artifact_report`)
- 输入：epoch manifest + epoch 目录。
- 输出：含各类指标、Z 分数、`Composite_Score`、`Is_Artifact` 的 CSV，可选伪迹可视化。

### 3. 音频特征 (`extract_audio_features`)
- 输入：epoch manifest 中引用的 WAV（`raw/audio/stimuli`）。
- 输出：每个音频的 `*_features.npy`、`*_feature_times.npy`、`*_feature_names.txt`，并记录特征 manifest。

### 4. EEG/音频对齐 (`align_eeg_audio_pairs`)
- 输入：epoch manifest、伪迹 CSV、音频特征目录。
- 输出：`*_EEG_aligned.npy` 与 `*_Sound_aligned.npy`，采样在统一的目标网格上，并生成 `manifest_epochs.csv`。

### 串联运行

推荐使用 HuggingFace 数据缓存 + 预处理一条龙命令：

```bash
python -m eeg_audio_benchmark.bin.sync_hf_data --preproc-config configs/preproc_on_hf.yaml
```

`configs/preproc_on_hf.yaml` 展示了以 HF 目录为核心的配置格式，可直接在其中调整切片长度、特征参数或输出目录。

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
