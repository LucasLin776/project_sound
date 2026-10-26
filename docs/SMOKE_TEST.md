# 增强数据与手动训练冒烟测试

## 1. 已准备的数据

- 来源：`D:\data_process\output\bolt_aug_v2_eq`。
- 项目内副本：`D:\project_sound\data\raw\bolt_aug_v2_eq`；原数据保留。
- 82条WAV：34条有目标、48条无目标；每条30秒、32 kHz、单声道PCM16。
- 音频、34份正例Audition标签、82张波形图、82张时频图及增强日志一并复制。
- 当前全量划分清单：`data/manifests/bolt_aug_v2_eq_all.jsonl`，使用新入口`python scripts/full_test.py`执行全量流程，见`docs/FULL_TEST.md`。
- 冒烟清单：`data/manifests/bolt_aug_v2_eq_smoke.jsonl`，6条录音、3正3负。
- 导入记录：`data/manifests/bolt_aug_v2_eq_import.json`，含文件校验、来源和所选录音。

中央清单中的音频、标签和图片路径均已调整为指向项目内副本。复制的生成日志保留生成时的历史路径，仅供来源审计，训练读取中央清单，不依赖历史路径。

6条包括螺栓录音EQ版本、两种背景下的混合正例、胶粒干扰负例、正常录音EQ版本和配对背景负例。沿用真实的时间区间、`group_id`、`split=train`；未伪造独立验证集。

## 2. 你来启动

在PowerShell中：

```powershell
conda activate wind-bolt-audio
cd D:\project_sound
python scripts/smoke_train.py
```

若使用Miniconda Prompt，切换目录用`cd /d D:\project_sound`。直接使用现有环境，不需要安装新增依赖或重新下载权重。

只做只读预检：

```powershell
python scripts/smoke_train.py --check-only
```

该选项检查标注、音频路径、依赖版本和权重文件是否存在，计算预计窗口数；不实例化特征模型、不提取特征、不拟合分类器、不建立运行输出目录。预检通过不等于训练已跑通。

## 3. 实际运行的四步

1. 使用当前默认`fusion`设置：冻结PANNs Cnn14的2048维嵌入 + 12维时频统计。6条完整录音切成342个窗口，窗口长2秒、步长0.5秒；不自动退回纯统计特征。
2. 在确定的训练窗口上拟合StandardScaler和LogisticRegression。边界不确定窗口仍排除，阈值固定0.5，不要求独立验证集。
3. 保存并重新加载JSON分类器，检查特征维数、数值有效性及分数范围。
4. 调用现有`BoltDetector.detect`对一条正例和一条负例执行完整WAV推理，保存事件区间、分数、质量和耗时，以及详细日志。

训练的仍是原有分类器，Cnn14权重不更新。本次只要求四步执行完成、输出结构有效，**预测正确与否不作为通过条件**。

## 4. 成功时看到什么

终端最后显示：

```text
冒烟测试通过
测试报告：D:\project_sound\artifacts\smoke_runs\<本次编号>\smoke_report.json
```

每次创建独立目录，不覆盖上次结果：

```text
artifacts/smoke_runs/<本次编号>/
  plan.json                  本次预检信息
  features.npz               训练特征缓存
  model/classifier.json      本次训练出的分类器
  model/metrics.json         标明未评估准确率，记录窗口数和维数
  predictions/<编号>.json    两条WAV的检测结果
  logs/<编号>.json           推理详细日志
  smoke_report.json          status=passed表示四步成功
```

如中途失败，`smoke_report.json`记录`status=failed`、已经完成的步骤和错误；`RUNNING.json`保留，表示本次流程未成功结束。修复后重跑会建立新目录。

## 5. 与后续完整训练的区别

本次只做程序流程检查，不计算验证准确率，也不承诺检测效果。所有增强版本仍只有一个真实螺栓来源事件。

冒烟产物标记`smoke_test_only=true`，仅冒烟脚本显式允许加载。常规CLI、服务和`train_classifier`的正式路径继续使用独立验证流程；补齐验证数据后使用项目原有`extract`、`train`命令。

仅准备代码、导入数据和执行`--check-only`不会生成模型。由你运行不带`--check-only`的命令后才会开始特征提取和训练。
