# 数据与快速标注指南

## 一、标什么

目标只有一种：`bolt_drop`，代表确认由螺栓掉落产生的撞击、滚动及后续碰撞声。标签是时间区间 `[start_s, end_s)`，时间相对于原 WAV 开头，单位为秒。

- 从最早可确认的目标声音开始，到同一掉落过程的目标声音结束为止。紧接着的滚动和反复碰撞可归为一个事件，不要把每个峰值算作一次掉落。
- 另一次独立掉落分别标注。一次试验中存在较长静默时，应按预先约定的一致规则处理，并在 notes 说明；不能通过声音波形确定真实次数时，不宣称推断出螺栓个数。
- 起止只围绕目标声音，不包含整条 30 秒背景；没有目标的前后片段会用于训练负例。
- 听音与波形、频谱一起确认。风噪或其他金属撞击不是目标；仅有大峰值不构成正例。困难负例可以写在 notes。
- 完整听完且确认没有目标的文件才标正常。如果有听不清/无法确定的片段，暂保留 reviewed=false；当前导入器不支持忽略区间，不要当成正常训练。
- 只有整条核对完所有事件后，才能设为 reviewed=true。静止参考录音还没有精确人工标注，程序不会替你猜起止。

## 二、Adobe Audition 2023 标注（当前使用）

### 一个 WAV 中的多个时间段

1. 在波形编辑器打开原始完整 WAV，听完整条；保留原音频和原始时间位置。
2. 拖动选中第一段掉落、滚动和后续碰撞声，按 `M` 建立范围标记，将名称改为 `bolt_drop`。
3. 选中第二段独立事件，再按 `M`，名称同样填 `bolt_drop`。后面的事件依此操作。
4. 在“窗口 → 标记”面板检查每条标记的开始、结束、持续时间。每个区间的持续时间必须大于 0；你的示例中 Type=Cue、Duration=0:04.764 已能表达一个范围。
5. 在标记列表选中本文件的全部目标标记，使用“导出所选标记到 CSV”。一条 WAV 对应一份 CSV，其中每个事件占一行。建议成对命名 `trial_001.wav`、`trial_001.csv`。

### 项目支持的 Audition 文件

根据你提供的 `标记.csv` 实际格式适配：UTF-8（支持 BOM）、Tab 分列，包含 Name / Start / Duration / Time Format / Type / Description。也兼容逗号分隔、带引号的 CSV。

下面第一行时间来自你的示例；第二行仅演示同一 WAV 中的另一个事件：

```text
Name	Start	Duration	Time Format	Type	Description
bolt_drop	0:12.808	0:04.764	decimal	Cue	第一次掉落
bolt_drop	0:23.100	0:00.900	decimal	Cue	第二次掉落
```

转换为训练区间：

```json
[{"start_s":12.808,"end_s":17.572},{"start_s":23.1,"end_s":24.0}]
```

结束时间由**开始时间 + 持续时间**计算，Duration 不是结束时间。`Time Format` 选择 `decimal`，支持纯秒数、`分:秒.小数` 或 `时:分:秒.小数`；暂不解析采样点和视频帧时间码。

你的导出文件按内容自动识别，保持 `.csv` 即可。程序会保留 Type、Description、原始时间字符串和来源路径到 manifest 的 annotation 字段，随训练来源元信息保存。

导入器拒绝零持续时间的点标记、非 bolt_drop 名称、重叠或超出音频的区间。Audition 的毫秒舍入可能导致文件末尾区间最多超出真实时长 1 毫秒，程序在此范围内贴齐真实结尾，并记录 `end_clamped_to_audio`；其他越界报错。无法确认的区间暂不设为 reviewed。

## 三、Audacity 标注（同样支持）

[官方下载 Windows 版](https://www.audacityteam.org/download/windows/)。适合现在这种 30 秒录音，标注时可以同时看波形和频谱。

1. 打开原 WAV，听完整条；不要裁剪、移动音频或改变速度后再保存时间标签。
2. 拖动选中目标的**一段范围**，使用添加标签（常用快捷键 `Ctrl+B`），标签文字统一填 `bolt_drop`。
3. 同一文件内所有目标事件都要标；调整选区边界后复听。不要只做起点标记。
4. 从标签编辑器的 Export，或 File → Export Other → Export Labels 导出**标准格式**、UTF-8 标签文本。版本菜单可能略有不同。
5. 保存为 `data/labels/trial_001.txt`。一条标签三列，用真实 Tab 分隔，例如：

```text
12.300000	16.800000	bolt_drop
23.100000	24.000000	bolt_drop
```

以上是格式示例，**不是参考录音的真实标签**。

官方说明：[创建区间标签](https://manual.audacityteam.org/man/creating_and_selecting_labels.html)、[导入导出与文本格式](https://manual.audacityteam.org/man/importing_and_exporting_labels.html)。只导出时间区间标签，不用带频率范围的扩展格式；脚本会拒绝未知类别、点标签、重叠或超出文件时长的区间。

## 四、导入本项目

在项目根目录、激活 Conda 后先执行一次清单生成；已有清单不要重新覆盖：

```bat
python -m bolt_audio init-manifest --audio-dir data/raw --output data/manifests/train.jsonl
python -m bolt_audio check-data --manifest data/manifests/train.jsonl
```

在清单中找到相应 WAV 的 `recording_id`，例如 `recording_00001`；将 Audition CSV 放入 `data/labels/`，导入并指定真实试验组、数据划分：

```bat
python -m bolt_audio import-labels --manifest data/manifests/train.jsonl --recording-id recording_00001 --labels data/labels/trial_001.csv --group-id turbine01_session01 --split train
```

同一 CSV 内所有时间段一次导入。格式默认自动识别，也可加 `--label-format audition`；Audacity 的 `.txt` 沿用同一命令。`examples/audition_markers.csv` 保存了你提供的格式示例，可用于对照。

该命令表示你已核对**整个文件**和所有目标区间，程序将设置 reviewed=true。确认没有任何目标的另一条录音：

```bat
python -m bolt_audio import-labels --manifest data/manifests/train.jsonl --recording-id recording_00002 --normal --group-id turbine01_session01 --split train
```

空标签文件不会自动变成正常样本，必须用 `--normal`。已核对的标签默认禁止覆盖；要有意修正时加 `--replace`。每次修改前在清单旁保存 `.jsonl.bak`，保留上一步版本。标签文件务必对应正确的 WAV，程序无法通过两个不同文件的名字保证标注内容一致。

## 五、最终清单与划分

JSONL 每行对应一个原始完整文件。示例：

```json
{"recording_id":"recording_00001","audio":"../raw/trial_001.wav","group_id":"turbine01_session01","split":"train","reviewed":true,"events":[{"start_s":12.3,"end_s":16.8}],"notes":"同一掉落后的滚动和碰撞"}
```

`audio` 相对清单目录解析。`events=[]` 表示整条已核对的正常文件；`split` 可为 train/val/test，未完成时保留 unassigned。

**先按独立试验/采集批次划分，再切窗口。** 同次试验的重复录音、连续录音切片以及其增强版本必须共用 group_id，放在同一 split。要测试新风机泛化时，应按风机整体留出。默认不会随机切散窗口，避免泄漏。

训练、验证都要有正负窗口；验证还必须同时有完整正常录音与事件录音。建议测试集也具备两类。数据应覆盖正常运行背景、不同位置和螺栓试验、人员操作/启停/其他撞击等困难负例。不要让正例全来自静止、负例全来自运行，模型可能只学到工况差异。静止声与背景混音只作为补充；同源前景或背景的混音不要跨 split。

标注完成后按 README 执行 `check-data → extract → train → infer`。

## 六、标注如何影响训练与时间输出

- 默认 2 秒窗口、0.5 秒步长，覆盖文件尾部；短文件右补零。输入统一 32 kHz，不做增益归一化。
- 窗口与事件有正时长重叠时标 1；未重叠但接近边界 0.2 秒内标 -1，排除监督拟合；其余已核对窗口标 0。任意重叠规则是初始基线，对很短或很弱事件仍需迭代。
- -1 窗口仍保留用于整条文件和事件评估，因为实际推理也会看到这些区域。
- 分类器逐窗打分，相邻/重叠正窗口合并成区间。2 秒上下文会扩大预测边界，0.5 秒步长不代表边界有 0.5 秒准确度；缩短窗口需要重新提取特征、训练并验证短事件识别能力。
- 阈值按验证集时间区间匹配 F1 选择，整条录音 F1 和精确率依次用于平局。测试集只作独立评估。
- `metrics.json` 记录窗口、录音及事件指标。事件采用一对一时间 IoU 匹配（默认 0.3），一个预测不会匹配多个真实事件；错时间的报警计误报并留下漏检。IoU 对短事件较严格，应结合预测明细与起止误差一起看。
- 同时记录每小时误报，以**实际录到的时长**作分母，不包含未录音的 90 秒。正常录音上的每小时误报和全部录音上的未匹配预测分别统计。

已有参考录音约 30.003 秒，44.1 kHz、单声道，约 1.589% 样本达到量程边界。保留原文件，后续采集应检查设备增益和削波；该比例不等同于已识别的故障强度。

## 七、训练后用 Audition 修正模型建议

```bat
cd D:\project_sound_inference
python scripts/infer.py --model D:\models\bolt_v2 --audio D:\project_sound\data\raw\trial_001.wav --output results/trial_001.json --labels-output D:\project_sound\data\labels\trial_001.proposal.csv
```

`.csv` 自动导出为与你的示例相同的 Audition 表头、Tab 分隔、UTF-8 BOM 格式；`.txt` 则导出为 Audacity 格式。可用 `--label-format` 明确指定。将 CSV 通过 Audition 的标记导入功能载入对应 WAV，人工核对全部区间后再导入项目；模型建议本身不自动设为人工真值。

代码已验证导出→导入的数值往返；你的 Audition 2023 中实际载入建议 CSV 尚需首次操作确认。
