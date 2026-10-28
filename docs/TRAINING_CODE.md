# 模型训练代码阅读指南

## 当前训练什么

PANNs Cnn14保持冻结，提供2048维深度特征；叠加12维时频统计共2060维。训练集拟合StandardScaler与LogisticRegression；验证集按事件F1选择阈值；测试集独立评估。当前没有端到端微调Cnn14。

## 阅读顺序

| 顺序 | 文件与函数 | 内容 |
|---|---|---|
| 1 | `scripts/full_test.py: main, run_full` | 默认第二版818条，预检、提取、训练评估、重载三步 |
| 2 | `scripts/smoke_train.py: preflight` | 只读检查、来源隔离、标签、窗口数量 |
| 3 | `src/bolt_audio/config.py: Settings` | 训练配置与路径 |
| 4 | `src/bolt_audio/annotations.py: import_labels` | Audition/Audacity标注导入 |
| 5 | `src/bolt_audio/dataset.py: load_manifest` | 标注范围、来源组、重复录音和集合检查 |
| 6 | `src/bolt_audio/audio.py` | 声道、重采样、质量、滑窗 |
| 7 | `src/bolt_audio/features.py: FeatureExtractor` | 深度及12维时频特征，与推理定义保持一致 |
| 8 | `src/bolt_audio/dataset.py: window_labels, extract_manifest` | 标注转窗口标签，生成特征缓存 |
| 9 | `src/bolt_audio/classifier.py: train_classifier` | 核心：拟合标准化和LR、验证选阈值、计算三个集合指标 |
| 10 | `src/bolt_audio/evaluation.py: event_metrics` | 一对一时间IoU匹配、误报漏检与边界误差 |
| 11 | `src/bolt_audio/classifier.py: LinearClassifier` | 训练产物的数值重载检查 |
| 12 | `src/bolt_audio/model_export.py: export_model` | 导出参数与匹配权重，剔除训练路径和记录 |

全量入口保留原始来源隔离和已有划分，不重新随机分组；只有训练集增强。第一版360条可用`--manifest`切回，见`FULL_TEST.md`。

## 推理已独立

训练项目的`infer`、`serve`、`detector.py`和`api.py`已移出。部署代码从`D:\project_sound_inference\src\bolt_inference\cli.py`开始读，之后是`config.py → model.py → features.py → detector.py → api.py`。

训练评估仍计算窗口分数和时间段，这是指标计算需要的部分；全量入口不再从WAV重复执行部署推理。训练后用`export-model`导出模型目录，独立项目通过`--model`加载。导出本身不拟合任何参数，见`INFERENCE_SPLIT.md`。

## 修改位置

| 需求 | 修改处 |
|---|---|
| 增强、来源划分 | `D:\data_process` |
| 标注区间 | JSONL清单及Audition CSV |
| 特征或窗口 | 训练配置与features；同步推理特征并重训 |
| 分类器与阈值选择 | `classifier.py` |
| 时间段评估 | `evaluation.py` |
| 导出格式 | `model_export.py`，与推理模型格式保持兼容 |
| HTTP、部署、单WAV结果 | 独立推理项目 |
