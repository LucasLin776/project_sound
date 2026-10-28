# 第二版数据的全量训练与版本切换

## 1. 当前默认数据

增强项目`D:\data_process\output\bolt_dataset_v2`完整复制到`D:\project_sound\data\raw\bolt_dataset_v2`。音频、Audition CSV、波形图、时频图和生成记录均保留。中央训练清单中的音频、标签和图片路径指向项目内部副本；生成记录中的历史源路径用于审计，不用于训练读文件。

| 集合 | 总数 | 正例 | 反例 | 第一批/第二批独立故障来源 | 中央清单 |
|---|---:|---:|---:|---:|---|
| 训练集 | 800 | 400 | 400 | 9 / 11 | `data/manifests/bolt_dataset_v2_train.jsonl` |
| 验证集 | 9 | 7 | 2 | 3 / 4 | `data/manifests/bolt_dataset_v2_val.jsonl` |
| 测试集 | 9 | 7 | 2 | 3 / 4 | `data/manifests/bolt_dataset_v2_test.jsonl` |
| 合计 | 818 | 414 | 404 | 15 / 19 | `data/manifests/bolt_dataset_v2_all.jsonl` |

第二批最初20条中的原编号19按用户要求排除。先按原始录音来源划分，再仅对训练集增强；导入时不重新划分、不重新随机打乱、不改变事件标注。训练集正反例1:1，验证与测试各保留7条真实故障和2条正常/干扰录音。

专门合成的双故障样本有40条，索引见`data/raw/bolt_dataset_v2/multi_event_examples.json`。计入原始多事件录音后，全数据共70条包含多个区间：训练62条、验证4条、测试4条。总标注事件551个，分别为510、20、21个。模型保持单文件多区间输出。

文件复制校验和预检记录见`data/manifests/bolt_dataset_v2_import.json`。原始来源、来源组、来源哈希、重复音频和时间标注检查保持启用。

## 2. 手动启动第二版

PowerShell：

```powershell
conda activate wind-bolt-audio
cd D:\project_sound
python scripts/full_test.py --check-only
```

预检应显示818条、训练800条、验证9条、测试9条，预计46626个窗口，每个窗口2060维。预检只读配置、元数据、音频头及哈希；不初始化模型、不提取特征、不拟合参数。数据切换准备阶段只执行这一步。

由你开始训练：

```powershell
python scripts/full_test.py
```

Miniconda Prompt切换目录用`cd /d D:\project_sound`。

## 3. 如何切换数据版本

`--manifest`指定本次命令使用的完整数据清单。不传时，`scripts/full_test.py`默认选择第二版；该参数不修改下次运行的默认值。清单中的`split`字段已经决定train/val/test成员。

```powershell
# 第一版：360条
python scripts/full_test.py --manifest data/manifests/bolt_aug_v3_split_all.jsonl --check-only

# 第二版：818条
python scripts/full_test.py --manifest data/manifests/bolt_dataset_v2_all.jsonl --check-only
```

以上均只预检。确认清单和数量后，由你去掉`--check-only`启动对应版本：

```powershell
# 训练第一版
python scripts/full_test.py --manifest data/manifests/bolt_aug_v3_split_all.jsonl

# 训练第二版
python scripts/full_test.py --manifest data/manifests/bolt_dataset_v2_all.jsonl
```

无需移动数据、重新标注或修改`configs/default.json`。单独的`_train.jsonl`、`_val.jsonl`、`_test.jsonl`方便单组查看；全量入口使用`_all.jsonl`。

切换数据不会自动切换现有推理模型。每次训练在新目录保存自己的特征和模型；导出模型包后，在独立推理项目用`--model`指定路径。本次导入不启动训练、不更新现有模型。

## 4. 当前训练与评估流程

1. 对所选版本的全部WAV提取冻结PANNs Cnn14的2048维深度特征及12维时频统计，窗口2秒、步长0.5秒。
2. 只用训练集拟合标准化及逻辑回归，用验证集选阈值，再基于缓存特征评估三个集合的窗口级、录音级及事件级指标。
3. 保存并重载分类器，检查全部缓存窗口的分数数值。

训练不再执行部署用的逐条WAV推理。单文件检测、日志和HTTP服务移到`D:\project_sound_inference`。本次拆分不启动训练。详细边界见`INFERENCE_SPLIT.md`。

## 5. 输出与模型交付

每次训练创建独立目录：

```text
D:/project_sound/artifacts/full_runs/<本次编号>/
  plan.json                  数据清单、环境与窗口数
  features.npz               本次特征、标注及划分
  model/classifier.json      分类器、标准化、特征配置与阈值
  model/metrics.json         三个集合的指标及事件匹配详情
  full_report.json           总报告，含source_manifest
```

成功时显示`全量流程测试完成`，总报告`status=passed`表示流程执行成功。训练不再生成新的`predictions/`、`logs/`和`detection_summary.csv`；既有结果保持原样。依赖库提示如存在，保存在`training_warnings.json`。失败时保留错误报告与`RUNNING.json`。

把`<本次编号>`替换为实际目录，导出新模型包：

```powershell
python -m bolt_audio export-model --classifier "artifacts/full_runs/<本次编号>/model/classifier.json" --output-dir D:\models\bolt_v2
```

导出不重新训练。在独立推理项目执行：

```powershell
cd D:\project_sound_inference
python scripts/check_model.py --model D:\models\bolt_v2
python scripts/infer.py --model D:\models\bolt_v2 --audio D:\incoming\test.wav --output results/test.json
```

模型包可放任意目录；`--model`指定路径，推理机器无需训练仓库。第一版已有训练模型可以直接导出，第二版训练仍由你启动。

## 6. 历史数据

第一版360条仍在`data/raw/bolt_aug_v3_split/`，其清单和已有训练结果完整保留。`scripts/smoke_train.py`仍使用历史6条数据进行小规模冒烟。

更早的82条相关来源批次仅供历史流程复现，需显式指定：

```powershell
python scripts/full_test.py --manifest data/manifests/bolt_aug_v2_eq_all.jsonl --allow-related-splits
```

报告会标记为`pipeline_test`。第一版和第二版均保持来源隔离，不需要此选项。`scripts/split_dataset.py`服务于旧82条批次，不用于重新随机划分第二版增强样本。
