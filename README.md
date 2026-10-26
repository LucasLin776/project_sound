# Acoustic_Deep_Learning — 风机螺栓掉落声音检测

基础版本，目前主要用于检测轮毂中螺栓掉落。

输入一条 WAV，输出是否掉落、发生区间，以及检测分数、录音质量和处理耗时。不依赖转速、风速、功率或运行状态。只对录到的音频负责，不覆盖未录音的 90 秒。

**已接入增强训练数据，尚未启动训练，没有训练好的真实螺栓分类器。** `diagnose` 可以立即运行；`infer` 必须加载训练产物，缺少模型时明确报错，HTTP 接口返回 503。

## 1. 环境与立即运行

在 Miniconda Prompt 中：

```bat
conda activate wind-bolt-audio
cd /d D:\project_sound
python -m bolt_audio diagnose --audio data/reference/bolt_drop.wav --output results/reference/diagnostics.json
```

在 PowerShell 中切换目录使用 `cd D:\project_sound`。现有环境位置、重建方法见 [Conda 说明](docs/CONDA.md)。

需要波形、STFT 时频图辅助查看时：

```bat
python scripts/diagnose_audio.py --audio data/reference/bolt_drop.wav --mode both --output-dir results/reference
```

通用 AudioSet 标签和 SED 都只用于诊断，不能代替螺栓模型。

### 已导入增强数据：手动运行冒烟测试

随机均衡批次从`D:\data_process\output\bolt_aug_v2_eq`复制到`data/raw/bolt_aug_v2_eq/`，共82条：34条正例、48条负例；包含音频、时间标注、两类图片和来源日志。全量清单为`data/manifests/bolt_aug_v2_eq_train.jsonl`，小规模冒烟清单为`data/manifests/bolt_aug_v2_eq_smoke.jsonl`（3条正例、3条负例）。所有数据仍为train。

激活Conda环境并进入`D:\project_sound`后，由你手动运行：

```bat
python scripts/smoke_train.py
```

它使用默认的冻结PANNs Cnn14 + 时频统计特征，执行特征提取、分类器训练、JSON模型重载及两条WAV推理。固定阈值0.5，不选择验证阈值，不以预测对错判定成功。终端显示`SMOKE TEST PASSED`、输出的`smoke_report.json`中`status=passed`即表示流程完成。输出在`artifacts/smoke_runs/<本次编号>/`。

只检查数据、依赖和权重文件是否存在，不提取特征、不加载模型、不训练：

```bat
python scripts/smoke_train.py --check-only
```

详细说明见[冒烟测试指南](docs/SMOKE_TEST.md)。此次接入仅做准备和只读预检，未启动特征提取、训练或真实录音推理。常规`train`仍需要独立验证集；本次冒烟模式使用同一训练组检查程序流程，不生成准确率结论。

## 2. 标注、训练、检测

将原始录音放入 `data/raw/`，建立待标注清单：

```bat
python -m bolt_audio init-manifest --audio-dir data/raw --output data/manifests/train.jsonl
```

**支持你正在使用的 Adobe Audition 2023**：每段目标声音建立一个名为 `bolt_drop` 的范围标记，导出同一 WAV 的全部标记到一份 CSV，再用 `import-labels` 自动识别导入；也支持 Audacity 文本标签。正常录音用 `--normal` 明确确认。多区间操作和命令见 [数据与标注指南](docs/DATASET.md)。

标注并按独立试验分配 train/val/test 后：

```bat
python -m bolt_audio check-data --manifest data/manifests/train.jsonl
python -m bolt_audio extract --manifest data/manifests/train.jsonl --output data/features/train.npz
python -m bolt_audio train --features data/features/train.npz --output-dir artifacts/bolt_v1
python -m bolt_audio infer --audio data/reference/bolt_drop.wav --classifier artifacts/bolt_v1/classifier.json --output results/detection.json --log-output logs/latest_detection.json
```

训练生成 `classifier.json` 和 `metrics.json`。验证集选阈值，测试集不参与选择。指标同时包含窗口、整条录音和时间区间匹配；时间匹配默认要求 IoU≥0.3，可用 `train --event-iou` 调整。该阈值是当前原型的评估规则，需要在现场数据上确认。

### 检测结果

以下数值仅为格式示例：

```json
{
  "bolt_detected": true,
  "events": [{"start_s": 12.0, "end_s": 17.0, "score": 0.91}],
  "score": 0.91,
  "quality": {
    "duration_seconds": 30.0,
    "sample_rate": 44100,
    "channels": 1,
    "peak_amplitude": 0.95,
    "rms_dbfs": -25.0,
    "full_scale_sample_fraction": 0.0,
    "clipping_warning": false
  },
  "processing_seconds": 8.2
}
```

- 无掉落时 `bolt_detected=false`、`events=[]`，分数、质量与耗时仍返回。
- 时间单位为秒，相对于文件开头；允许多个不相连事件。默认 2 秒窗口、0.5 秒步长，区间是正窗口覆盖，**不承诺精确撞击边界**。
- `score` 是最大窗口分类得分；事件 `score` 是该事件内最大得分。分数未经概率校准，不能解释为真实发生概率。若以后设置至少多个正窗口才报警，最大得分超过阈值也可能不形成事件。
- `processing_seconds` 包括解码、重采样、特征和分类，不包括首次模型加载、网络上传和日志写盘。常驻服务只在启动时加载模型。
- 逐窗分数、阈值、通用标签、完整质量信息、模型文件哈希、配置和警告写入 `logs/inference/`；`--log-output` 可指定路径。CLI 标准输出只打印检测结果，日志位置打印到标准错误。

训练后可将预测导出为 Audition CSV，人工修正，辅助下一轮标注：

```bat
python -m bolt_audio infer --audio data/raw/trial_001.wav --classifier artifacts/bolt_v1/classifier.json --output results/trial_001.json --labels-output data/labels/trial_001.proposal.csv
```

`.csv` 自动使用 Audition 格式，`.txt` 使用 Audacity 格式。预测不会自动变成真值标签。验证/测试集应独立人工标注，避免照着模型输出验收模型。

## 3. 服务器

复制 `configs/default.json` 为 `configs/server.json`，将 `classifier_path` 改成 `artifacts/bolt_v1/classifier.json`：

```bat
python -m bolt_audio --config configs/server.json serve --host 127.0.0.1 --port 8000
curl.exe -F "file=@D:/project_sound/data/reference/bolt_drop.wav" http://127.0.0.1:8000/detect
```

`POST /detect` 与 CLI 返回相同字段。`X-Request-ID` 响应头对应 `results/api/<id>.json` 和 `logs/api/<id>.json`。`GET /health` 检查服务与目标分类器状态；`/docs` 可手动上传。默认限制 20 MB、60 秒、单进程一次推理，繁忙返回 503 并提示重试。可用环境变量 `BOLT_API_TOKEN` 开启 Bearer token；TLS、设备协议、数据长期保存及正式告警渠道需按现场系统接入。

## 4. 代码与目录

```text
src/bolt_audio/        标注导入、特征、训练、检测、API；vendor/panns 保留现有模型
configs/              统一参数
data/raw/             原始录音
data/labels/          Audition CSV / Audacity 标签文本
data/manifests/       每条录音的标注与数据划分
data/features/        提取的训练特征
checkpoints/          官方预训练权重
artifacts/            训练出的目标分类器与指标
results/              检测结果、诊断图
logs/                 详细检测日志
scripts/              Conda 安装、诊断、服务与验证工具
tests/                数据隔离、标注、训练推理及时间匹配测试
docs/                 标注、环境、清理及训练代码阅读指南
third_party/          来源与校验记录；许可证随 vendor 分发
```

**读训练代码请从 [训练代码阅读指南](docs/TRAINING_CODE.md) 开始。** 当前路线是冻结的 PANNs Cnn14 深度特征 + 时频统计 + 逻辑回归；训练更新标准化和分类器参数，**不更新 Cnn14 的神经网络权重**。这适合作为数据量有限、CPU 可运行的首版基线；真实背景遮蔽与跨风机泛化必须用独立现场数据验证，后续再决定是否微调网络。

清理清单和备份位置见 [清理说明](docs/CLEANUP.md)。源码来源见 [第三方说明](third_party/NOTICE.md)。本次未执行 Git 暂存、提交或上传，已有 `.git`、`.vscode`、原始录音和权重保留。

本地验证命令：

```bat
python -m pytest -q
python -m ruff check src tests scripts
python -m pip check
python scripts/check_server.py
```

默认 HTTP 检查验证缺少分类器时明确返回 503；训练后使用 `python scripts/check_server.py --config configs/server.json` 验证两次完整上传与结果/日志对应。合成数据测试只证明代码流程，不代表真实识别性能。
