# 项目技术路线重建

## 1. 项目与任务概述
- 本项目构建了一个 **EEG-音频对齐与建模基线**，目标是研究连续自然音频与脑电（EEG）之间的映射关系，覆盖编码（音频→EEG）与解码（EEG→音频特征）两类回归任务。
- 数据输入为对齐后的 EEG 序列与对应的音频特征序列（时间帧对齐）；输出为回归模型对目标模态的预测（例如编码任务中预测 EEG，多通道；解码任务中预测音频特征或包络）。
- 预处理围绕 HuggingFace 数据集 `Aurelianous/eeg-audio` 及其派生文件展开，生成对齐片段并支持 ROI/偏移搜索的 TRF 基线实验。

## 2. 整体技术路线 / 系统架构
1) **数据预处理与对齐**：`preprocessing/` 模块实现 EEG 切段、伪迹打分、音频特征提取与 EEG-音频时间对齐，输出 `manifest_epochs.csv` 及对齐的 `.npy` 对。
2) **特征表示与设计矩阵构建**：`representations.py` 定义 EEG（时间域、bandpower）与音频（时间、包络）特征；`datasets.py` 根据任务类型构造滞后特征矩阵，并可按被试标准化。
3) **模型与训练**：
   - 通用回归基线通过 `models/`（如 Ridge/Lasso/ElasticNet/RandomForest）实现，`experiment.py` 驱动交叉验证评估。
   - TRF 基线位于 `trf/`：声学特征生成、滞后矩阵、ROI 选择、偏移搜索及 Ridge 回归（`TRFEncoder`）。
4) **评估与分析**：`evaluation.py` 计算 $R^2$ 与 Pearson 相关；`trf/eval.py` 额外提供 null baseline、分段相关等统计；结果保存为 JSON/CSV，`analysis/` 进一步绘图。

## 3. 数据与预处理
- **EEG 切段与伪迹评分**：依据 README 中的流程，`eeg_epochs.py` 产生 epoch 清单，`artifacts.py` 生成包含 `Is_Artifact` 的质控表（用于筛除伪迹）。
- **音频特征**：`preprocessing/audio_features.py`（由 `audio_features.yaml` 配置）使用 Librosa 以 22.05 kHz 采样率、46 ms 帧长/11 ms 步长提取 MFCC/音高等特征并保存时间戳。
- **对齐**（`preprocessing/alignment.py`）：
  - 读取干净 epoch 清单与音频特征，按滑动窗口（默认 250 ms 窗、50 ms 步）对 EEG 求均值，生成对应时间戳，再与音频特征时间轴插值到统一目标网格（默认 4 s、11 ms hop）。
  - 输出对齐的 EEG/音频 `.npy` 及 `manifest_epochs.csv`，包含 subject/run/event/stim/audio 文件路径等元数据，供后续加载。
- **加载与滞后展开**（`datasets.py`）：
  - 使用指定表示函数生成 EEG/音频特征；根据任务（encoding/decoding）决定自变量与因变量，并通过 `_lag_matrix` 添加用户指定的时间滞后栈。
  - 可选 `per_subject_scaling` 进行被试级 z-score。

## 4. 模型结构与关键模块
- **通用回归基线（Sklearn 模型）**
  - 输入：形状 $(N, F)$ 的设计矩阵；输出：目标序列预测 $(N, D)$。
  - 通过 `models/sklearn_models.py` 构造 Ridge/Lasso/ElasticNet/RandomForest 回归器；`BaseRegressor` 规范 `fit/predict` 接口。
- **TRF 编码器（`trf/models.py`）**
  - 输入：滞后展开的声学特征 $X \in \mathbb{R}^{T \times L}$；输出：预测 EEG $\hat{Y} \in \mathbb{R}^{T \times C}$。
  - 核心为 Ridge 回归，支持 `ridge_alpha_grid` + `GridSearchCV` 超参搜索；滞后窗口由 `n_pre`/`n_post` 控制。
- **声学与 EEG 特征构造（`trf/features.py` & `representations.py`）**
  - 音频：mel 能量、多带或宽带包络、能量特征，支持因果平滑与慢包络；可附加有声性掩码。EEG：按被试/通道或分段 z-score，并可因果滑动高通滤波。
  - `build_lagged_features` 生成包含 $[-n_{pre}, n_{post}]$ 滞后的设计矩阵；可拼接有声性掩码作为额外列。
- **ROI 选择与偏移搜索（`trf/roi.py`, `trf/offset.py`）**
  - ROI：基于 EEG 通道与宽带包络的最大相关（跨滞后）选择 top-k 通道。
  - 偏移：对不同时间偏移的声学特征计算 ROI 相关，选取最佳全局偏移用于 TRF 训练。

## 5. 损失函数与训练目标
- Sklearn 回归器内部采用最小化均方误差（如 Ridge/Lasso/ElasticNet）；随机森林最小化袋外误差（隐含 MSE）。
- TRF 使用 Ridge 回归；可通过 `ridge_alpha_grid`+交叉验证选择正则系数，评价与调参目标均为负均方误差（`neg_mean_squared_error`）。
- 评价指标：$R^2$（解释度）和 Pearson 相关系数；TRF 额外记录 null baseline 相关/ $R^2$ 以检验显著性。

## 6. 实验设置与评估指标
- **数据划分**：`splits.py` 支持基于元数据的 GroupKFold（默认 5 折）或 LOSO；TRF 评估按 segment 组构建折并保持被试分层。
- **指标**：
  - 通用基线：在 `experiment.py` 中对每折计算 $R^2$ 与 Pearson 并写入 `results.json`。
  - TRF：记录 `mean_r2`、`median_pred_r`、对应 null 版本，以及所用 ROI 通道、最优偏移和折数。
- **配置示例**：`configs/example_ridge.yaml`（解码任务，滞后 [0,1,2]）、`configs/trf_envelope*.yaml`（宽带/多带特征、偏移候选 0–121 ms、top-3 ROI、5 折）。

## 7. 工程实现细节
- 通过 dataclass 配置+路径自动解析（`_apply_data_root`）简化实验复现；结果自动保存到 `output_dir`。
- 支持按被试标准化、随机种子控制（TRF 评估使用 `np.random.default_rng`）。
- 日志使用 Python `logging` 统一控制；对齐与 TRF 模块在缺失文件或异常情况下提供警告。

## 8. 摘要版技术路线
项目采用“HF 数据预处理 → EEG/音频特征提取与时间对齐 → 滞后展开设计矩阵 → Ridge 等回归器（含 TRF 专用声学特征、ROI/偏移搜索） → 使用 $R^2$ 与 Pearson 相关的交叉验证评估”的流水线，为连续音频与脑电映射提供可复现的基线框架。
