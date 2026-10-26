# 模型训练代码阅读指南

## 先明确：现在训练什么

当前是 **PANNs Cnn14 预训练深度特征 + 12 维时频特征 + 逻辑回归**。不是重新设计网络；不是对 Cnn14 做端到端微调。

| 部分 | 是否更新参数 | 当前作用 |
| --- | --- | --- |
| PANNs Cnn14 | 否，eval/inference_mode | 从窗口波形提取 2048 维深度嵌入 |
| 时域/频域统计 | 无可训练参数 | 提取 12 维幅值、冲击性及频谱信息 |
| StandardScaler | 是，只在 train 上拟合 | 估计每维均值、尺度 |
| LogisticRegression | 是，只在 train 上拟合 | 学习窗口是否含目标的分类权重与偏置 |
| 报警阈值 | 在 val 上选择 | 按预测时间段与人工真值的匹配效果选择 |
| test | 不参与拟合/选阈值 | 给出独立评估 |

默认 `feature_set=fusion` 有 2060 维；`panns` 仅用深度特征；`acoustic` 仅用传统统计，是对照路线，不是深度学习训练。

## 建议阅读顺序

本次新增的手动冒烟入口为`scripts/smoke_train.py`，运行说明见`docs/SMOKE_TEST.md`。先读`preflight`了解只读预检，再读`run_smoke`跟踪`extract_manifest → train_classifier(smoke_test=True) → LinearClassifier.load → BoltDetector.detect`。冒烟模式使用固定阈值0.5、不评估准确率；常规训练的独立验证要求保持启用。

所有路径相对于 `D:\project_sound`。先读项目主链路，再读 vendor 的底层网络。

| 顺序 | 文件 / 主要函数 | 你要理解的内容 |
| --- | --- | --- |
| 1 | `configs/default.json`、`src/bolt_audio/config.py: Settings` | 采样率、窗口、步长、特征类型、设备与事件合并设置 |
| 2 | `src/bolt_audio/dataset.py: load_manifest` | 标注数据结构、区间范围、reviewed/split、组和相同录音泄漏检查 |
| 3 | `src/bolt_audio/annotations.py: read_label_file, read_audition_labels, import_labels` | 自动识别 Audition CSV / Audacity 文本，将开始+持续时间转成训练区间，保留描述等元信息 |
| 4 | `src/bolt_audio/audio.py: read_audio, window_starts` | 质量检查、转单声道、32 kHz 重采样、滑窗与尾部覆盖 |
| 5 | `src/bolt_audio/features.py: FeatureExtractor, acoustic_features` | 深度嵌入、12 维统计、特征定义和官方权重 SHA-256 |
| 6 | `src/bolt_audio/dataset.py: window_labels, extract_manifest` | 时间标签怎样变成 1/0/-1；NPZ 缓存的 X/y/时间/录音组/划分如何对应 |
| 7 | `src/bolt_audio/classifier.py: train_classifier` | **核心训练函数**：拟合 scaler、均衡录音贡献、拟合 LR、验证集选阈值、导出模型 |
| 8 | `src/bolt_audio/classifier.py: merge_windows, temporal_metrics`、`src/bolt_audio/evaluation.py: event_metrics` | 窗口分数怎样成为时间段；一对一 IoU 匹配、事件漏检误报、录到时长下的每小时误报 |
| 9 | `src/bolt_audio/classifier.py: LinearClassifier` | 从 JSON 恢复数值参数，做标准化与分类，避免 pickle |
| 10 | `src/bolt_audio/detector.py: BoltDetector` | 把单条 WAV 的特征、分类、合并接起来；形成结果与详细日志 |
| 11 | `src/bolt_audio/cli.py: main` | 对照命令了解 extract/train/infer 的调用顺序 |

测试入口 `tests/test_pipeline.py` 的 `test_acoustic_training_detection_and_api` 用临时合成数据演示一条完整训练→推理链路，不依赖大权重；其余测试重点覆盖数据划分、标签及错误时间段。

## train_classifier 内部重点看哪些语句

1. `np.load(..., allow_pickle=False)`：读取窗口特征、标签、录音/分组/split、真实区间等元信息。
2. `train = (splits == "train") & (y >= 0)`：只拿 train 且已确定的窗口拟合；-1 模糊窗口排除。
3. `StandardScaler().fit(X[train])`：只从训练集估计标准化参数。
4. `weights = ...`：让每条录音贡献相近，避免窗口多的录音支配训练；`class_weight="balanced"` 同时处理正负类别不平衡。
5. `model.fit(...)`：真正更新分类参数的地方。
6. `candidates`、`for threshold in candidates`：候选阈值来自 val 得分；先比 val 事件 F1，再用录音 F1、精确率处理平局。
7. `reports`：用选好的同一阈值报告 train/val/test，不能拿测试指标反复选阈值。
8. `artifact`、`write_json`：保存均值、尺度、分类权重、偏置、阈值、特征定义、训练来源与评估结果。

`artifacts/bolt_v1/classifier.json` 不是 PANNs 神经网络权重；它是训练出的分类器和配置。`checkpoints/Cnn14_mAP=0.431.pth` 才是冻结的官方神经网络权重。推理需要二者，特征定义必须匹配。

## 想读网络实现时

- `src/bolt_audio/vendor/panns/models.py: Cnn14.forward`：波形→STFT→log-mel→卷积→时频池化→2048 维 embedding。
- `src/bolt_audio/vendor/panns/inference.py: _Inference._forward, AudioTagging.inference`：官方权重加载、eval、inference_mode、取嵌入。**这些函数没有 optimizer、loss.backward 或更新权重。**
- 同文件中的 `Cnn14_DecisionLevelMax` / `SoundEventDetection` 是通用 AudioSet 诊断模型，当前不用于训练你的螺栓分类器，也没有自动变成螺栓定位网络。

## 修改时的直接对应

| 需求 | 主要修改处 | 之后做什么 |
| --- | --- | --- |
| 补标签、修正起止、补困难负例 | manifest / Audition 或 Audacity 标签 | 重新 extract、train，再评估 |
| 改 2 秒窗口、0.5 秒步长 | configs/default.json | 全部重新 extract、train；推理配置同步 |
| 改正负窗口判定 | dataset.window_labels | 重新 extract、train；说明边界规则 |
| 改特征或加统计量 | features.py | 更新 feature_spec，重新提取与训练 |
| 改 LR 参数、录音权重 | classifier.train_classifier | 用原缓存重训，只用 val 作选择 |
| 改合并规则、最少正窗口 | config + classifier.merge_windows | 重选阈值、重评估；推理以训练产物为准 |
| 改时间匹配严格程度 | train --event-iou | 重训/选阈值，保留设置以便对比 |
| 将来微调 Cnn14 | 需要新增基于已有架构的监督训练循环 | 现在尚未实现，不能只改 learning_rate 就视为已微调 |

先通过真实现场数据评估这条基线。若窗口语义分类能检出、但定位误差仍明显，应考虑以时间标注监督的事件检测微调；若目标被背景完全掩盖，单靠更复杂模型无法保证恢复信息。
