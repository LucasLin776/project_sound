# 当前360条数据的全量训练、验证和测试

## 1. 当前划分

从`D:\data_process\output\bolt_aug_v3_split`完整复制到`D:\project_sound\data\raw\bolt_aug_v3_split`，包括音频、Audition CSV、波形图、时频图和增强记录。训练清单指向项目内部相对路径，不依赖增强项目的文件位置。

| 集合 | 总数 | 正例 | 反例 | 清单 |
|---|---:|---:|---:|---|
| 训练集 | 350 | 180 | 170 | `data/manifests/bolt_aug_v3_split_train.jsonl` |
| 验证集 | 5 | 3 | 2 | `data/manifests/bolt_aug_v3_split_val.jsonl` |
| 测试集 | 5 | 3 | 2 | `data/manifests/bolt_aug_v3_split_test.jsonl` |
| 合计 | 360 | 186 | 174 | `data/manifests/bolt_aug_v3_split_all.jsonl` |

15次独立螺栓掉落录音按9/3/3分到三个集合；4条正常背景和4条干扰反例也按原始来源隔离。先划分来源，之后仅训练集做混合、随机均衡等增强。验证和测试只做格式统一。固定种子20261027，训练清单已经打乱，导入不重新划分。

训练集含18条双故障正例，使用两个不同训练来源的掉落片段，每条有两个独立时间区间。示例：`data/raw/bolt_aug_v3_split/audio/325.wav`，区间为8.1306875～14.7926875秒及17.8630625～25.3190625秒，对应`labels/325.csv`。全部示例索引见该批次的`multi_event_examples.json`。

导入记录和校验结果在`data/manifests/bolt_aug_v3_split_import.json`。默认训练入口检查原始来源、来源组、来源哈希不跨集合，另外检查重复录音内容和标注范围。

## 2. 由你手动启动

PowerShell：

```powershell
conda activate wind-bolt-audio
cd D:\project_sound
python scripts/full_test.py
```

Miniconda Prompt切换目录用`cd /d D:\project_sound`。

只预检、不启动训练：

```powershell
python scripts/full_test.py --check-only
```

预检核对三组正反例、来源隔离、数据路径、标注区间、环境依赖及权重文件。当前配置预计20520个窗口，每个窗口2060维。预检不初始化模型、不提取特征、不拟合参数。

## 3. 一次完整执行包含什么

1. 全部360条录音提取特征：冻结PANNs Cnn14的2048维深度特征 + 12维时频统计；2秒窗口、0.5秒步长。
2. 仅用训练集的确定窗口拟合标准化和逻辑回归；在验证集选阈值，输出训练、验证和测试三个集合的窗口级、录音级、事件级指标。
3. 重新加载模型，核对全部窗口的预测数值。
4. 再通过现有WAV接口逐条检测全部360条音频，每条保存是否掉落、时间段、分数、质量和耗时，并生成详细日志。

终端进度和摘要使用中文，文件名、命令参数和机器可读JSON键保持原格式。训练产生的依赖库提示保存在`training_warnings.json`，终端给出中文提示。

沿用现有PANNs特征 + 分类器路线：本次拟合标准化参数与分类权重，Cnn14保持冻结。后半程会逐条重新提取特征进行实际检测，因此整个命令的耗时包括特征提取、分类器训练和360条音频检测。

## 4. 输出与完成标志

每次运行创建新目录：

```text
D:/project_sound/artifacts/full_runs/<本次编号>/
  plan.json                  数据、环境与窗口数
  features.npz               全部360条的特征与划分信息
  model/classifier.json      模型及选定阈值
  model/metrics.json         三个集合的完整指标
  predictions/01.json...     全部360条的逐条检测结果
  logs/01.json...            全部360条的详细日志
  detection_summary.csv      中文列名的逐条结果汇总
  full_report.json           总报告
```

成功时终端显示`全量流程测试完成`，`full_report.json`中`status=passed`。它表示完整执行成功，预测对错不作为程序完成条件。录音级摘要包含准确率、精确率、召回率、F1，以及事件级F1。CSV可用Excel打开。

失败时总报告记录错误、完整调用栈和已完成步骤；`RUNNING.json`保留。修复后重新执行会生成新目录。

## 5. 历史批次

旧82条批次和已有训练结果保留，默认入口已切换到新360条批次。`scripts/split_dataset.py`操作旧批次，不要用于重新随机划分新增强样本。

如需复现旧批次的相关来源流程测试，可由你手动运行：

```powershell
python scripts/full_test.py --manifest data/manifests/bolt_aug_v2_eq_all.jsonl --allow-related-splits
```

该显式选项仍检查重复录音和标注，报告标记为`pipeline_test`；新批次默认使用严格来源隔离。
