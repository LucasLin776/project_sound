# 风机螺栓掉落：模型训练项目

本项目负责标注导入、数据检查、特征提取、模型训练、验证测试和模型包导出。数据增强在`D:\data_process`执行；独立WAV推理和HTTP服务位于`D:\project_sound_inference`。

```text
D:/data_process              数据准备、来源划分、数据增强
          ↓ 已标注WAV与train/val/test清单
D:/project_sound             特征提取、训练与评估、导出模型包
          ↓ classifier.json + Cnn14权重 + bundle.json
D:/project_sound_inference   指定 --model，执行WAV/HTTP推理
```

## 1. 当前数据与训练

默认第二版：第一批15条加第二批19条独立故障来源，第二批原编号19按要求排除。已有数据、划分和训练结果保留。

| 集合 | 正例 | 反例 | 合计 |
|---|---:|---:|---:|
| 训练集 | 400 | 400 | 800 |
| 验证集 | 7 | 2 | 9 |
| 测试集 | 7 | 2 | 9 |

数据在`data/raw/bolt_dataset_v2/`，总清单为`data/manifests/bolt_dataset_v2_all.jsonl`。每组含两批来源，只有训练集增强。

```powershell
conda activate wind-bolt-audio
cd D:\project_sound
python scripts/full_test.py --check-only
```

由你开始训练：

```powershell
python scripts/full_test.py
```

当前全量入口执行三步：提取全部特征；训练集拟合并在验证集选阈值、评估三组数据；保存和重载模型检查。训练仍为冻结PANNs Cnn14的2048维特征加12维时频统计，拟合标准化与逻辑回归权重。

**训练结束不再自动逐条运行部署推理或启动HTTP服务。** 评估直接使用提取的特征计算窗口、整段和事件指标；输出模型、指标及总报告到新的`artifacts/full_runs/<运行编号>/`。之前已生成的预测和日志保留。

切换数据用`--manifest`，只对本次命令生效：

```powershell
python scripts/full_test.py --manifest data/manifests/bolt_aug_v3_split_all.jsonl --check-only
python scripts/full_test.py --manifest data/manifests/bolt_dataset_v2_all.jsonl --check-only
```

去掉`--check-only`才会训练。不指定清单时默认第二版。详见[训练与数据切换](docs/FULL_TEST.md)。

## 2. 导出模型，交给独立推理项目

无需重新训练，已有产物可以直接导出。将`<训练运行编号>`换为实际目录：

```powershell
python -m bolt_audio export-model --classifier "artifacts/full_runs/<训练运行编号>/model/classifier.json" --output-dir D:\models\bolt_v2
```

导出目录包含`classifier.json`、匹配的`Cnn14_mAP=0.431.pth`及校验索引`bundle.json`。原始分类器中的训练录音路径、报告和训练记录不进入模型包。输出目录需是新的；不覆盖已有模型。可用`--checkpoint`指定其他位置的匹配权重，代码检查SHA-256。

在推理项目中：

```powershell
cd D:\project_sound_inference
python scripts/check_model.py --model D:\models\bolt_v2
python scripts/infer.py --model D:\models\bolt_v2 --audio D:\incoming\test.wav --output results/test.json
python scripts/serve.py --model D:\models\bolt_v2 --host 127.0.0.1 --port 8000
```

推理项目也支持`--model <原始classifier.json> --checkpoint <匹配权重>`。完整模型包只需一个`--model`路径。部署机器无需训练代码或数据增强项目。当前本地演示模型是第一版360条训练产物的导出，第二版训练仍由你启动。

## 3. 标注与分步训练

继续支持Adobe Audition范围标记CSV和Audacity TXT。多个事件在同一WAV的标注文件中保留多个范围；每段目标标为`bolt_drop`，正常录音显式确认。详见[数据与标注](docs/DATASET.md)。

```powershell
python -m bolt_audio init-manifest --audio-dir data/raw --output data/manifests/train.jsonl
python -m bolt_audio check-data --manifest data/manifests/train.jsonl
python -m bolt_audio extract --manifest data/manifests/train.jsonl --output data/features/train.npz
python -m bolt_audio train --features data/features/train.npz --output-dir artifacts/bolt_v2
python -m bolt_audio export-model --classifier artifacts/bolt_v2/classifier.json --output-dir D:\models\bolt_v2
```

全量第二版已标注，不需要重新执行`init-manifest`。测试集不参与拟合或阈值选择。事件匹配默认IoU≥0.3。诊断工具仍用于质量和特征查看，`diagnose`不再加载故障分类器。

## 4. 代码和迁移说明

| 代码 | 职责 |
|---|---|
| `src/bolt_audio/dataset.py` | 清单检查、时间标签、特征缓存 |
| `src/bolt_audio/features.py` | 训练时的PANNs与时频特征 |
| `src/bolt_audio/classifier.py` | 标准化、分类器拟合、阈值选择、重载检查 |
| `src/bolt_audio/evaluation.py` | 时间区间匹配、指标计算 |
| `src/bolt_audio/model_export.py` | 无训练的模型包导出 |
| `scripts/full_test.py` | 全量训练与评估入口 |

部署用的`api.py`、`detector.py`、HTTP启动与检查工具从本项目移出；`infer`和`serve`命令改由独立项目提供。训练配置移除`classifier_path`和上传限制，API依赖移至推理项目。完整迁移记录见[拆分说明](docs/INFERENCE_SPLIT.md)，代码阅读见[训练代码指南](docs/TRAINING_CODE.md)。

环境继续使用Conda，见[环境说明](docs/CONDA.md)。安装开发依赖用`python -m pip install -e ".[dev]"`。第三方模型源码与许可保留，见`third_party/NOTICE.md`。原始音频、已有模型和Git未提交修改均保留，本次不启动训练、不提交或推送Git。
