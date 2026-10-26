# 当前82条数据的全量训练、验证和测试

## 1. 当前划分

全部数据来自`D:\project_sound\data\raw\bolt_aug_v2_eq`，按录音划分，音频和标签文件不改名、不搬动。固定随机种子20261026，约70%/15%/15%。

| 集合 | 总数 | 正例 | 反例 | 清单 |
|---|---:|---:|---:|---|
| 训练集 | 58 | 24 | 34 | `data/manifests/bolt_aug_v2_eq_train.jsonl` |
| 验证集 | 12 | 5 | 7 | `data/manifests/bolt_aug_v2_eq_val.jsonl` |
| 测试集 | 12 | 5 | 7 | `data/manifests/bolt_aug_v2_eq_test.jsonl` |
| 合计 | 82 | 34 | 48 | `data/manifests/bolt_aug_v2_eq_all.jsonl` |

每条录音只属于一个集合；32组混合正例与配对背景反例保持成对。标签起止、原始`group_id`和来源哈希保留。划分明细、编号列表和种子保存在`data/manifests/bolt_aug_v2_eq_split.json`。

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

预检核对三组正反例、数据路径、标注区间、环境依赖及权重文件，预计4674个窗口、每个窗口2060维。预检不初始化模型、不提取特征、不拟合参数。

## 3. 一次完整执行包含什么

1. 全部82条录音提取特征：冻结PANNs Cnn14的2048维深度特征 + 12维时频统计；2秒窗口、0.5秒步长。
2. 仅用训练集的确定窗口拟合标准化和逻辑回归；在验证集选阈值，输出训练、验证和测试三个集合的窗口级、录音级、事件级指标。
3. 重新加载模型，核对全部窗口的预测数值。
4. 再通过现有WAV接口逐条检测全部82条音频，每条保存是否掉落、时间段、分数、质量和耗时，并生成详细日志。

终端进度和摘要使用中文，文件名、命令参数和机器可读JSON键保持原格式。训练产生的依赖库提示保存在`training_warnings.json`，终端给出中文提示。

这是按当前数据进行的流程测试，预测对错不作为流程完成条件。模型仍沿用现有PANNs特征 + 分类器路线。

## 4. 输出与完成标志

每次运行创建新目录：

```text
D:/project_sound/artifacts/full_runs/<本次编号>/
  plan.json                  数据、环境与窗口数
  features.npz               全部82条的特征与划分信息
  model/classifier.json      模型及选定阈值
  model/metrics.json         三个集合的完整指标
  predictions/01.json...     全部82条的逐条检测结果
  logs/01.json...            全部82条的详细日志
  detection_summary.csv      中文列名的逐条结果汇总
  full_report.json           总报告
```

成功时终端显示`全量流程测试完成`，`full_report.json`中`status=passed`。录音级摘要包含准确率、精确率、召回率、F1，以及事件级F1。CSV可用Excel打开。

失败时总报告记录错误、完整调用栈和已完成步骤；`RUNNING.json`保留。修复后重新执行会生成新目录。

## 5. 重新划分

```powershell
python scripts/split_dataset.py
```

它从项目内保留的完整原始清单读取82条，再生成四份中央清单和划分说明；相同种子得到相同划分，不启动训练。改变种子可使用`--seed 123`。

本次全量入口在读取和训练时明确允许原始来源组跨集合；每条录音、文件路径及相同音频内容的基本一致性检查仍启用。JSON中记录本次用途为`pipeline_test`，不把来源组改写成新的独立采集组。
