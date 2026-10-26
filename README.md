# Acoustic_Deep_Learning — 风机螺栓掉落声音检测

基础版本，目前主要用于检测轮毂中螺栓掉落。

统一的本地开发项目，路径为 **`D:\project_sound`**。整合 PANNs 主仓库和配套推理工具，具备数据检查、时频特征提取、目标分类器训练、录音检测、诊断图和 HTTP 推理接口。

**目前只有一条参考音频，尚未训练真实螺栓分类器。默认运行输出 `bolt_detected: null`，不能把通用 AudioSet 标签直接作为螺栓报警。** 项目提供的是可运行、可继续训练的基础架构。

## 立即运行

在已初始化 Conda 的 PowerShell 中执行（若用 Miniconda Prompt，切换目录的命令写为 `cd /d D:\project_sound`）：

```powershell
cd D:\project_sound
conda activate wind-bolt-audio
python -m bolt_audio infer --audio data/reference/bolt_drop.wav --output results/reference/inference.json
```

生成 `results/reference/inference.json`，包含录音质量、通用标签、特征维数和处理耗时。默认融合特征为 **2048 维 PANNs 嵌入 + 12 维时频统计特征**。

波形、STFT 时频图和通用声音事件诊断：

```powershell
python scripts/diagnose_audio.py --audio data/reference/bolt_drop.wav --mode both --output-dir results/reference
```

两种诊断模型各约 327 MB，已复制并校验；普通推理不会自动联网。项目环境由 Conda 管理，名称为 `wind-bolt-audio`。

## 项目结构

```text
src/bolt_audio/
  audio.py                   音频读取、重采样、削波检查、窗口覆盖
  features.py                PANNs 与时域/频域特征
  dataset.py                 标注读取、泄漏检查、特征缓存
  classifier.py              分类器训练、验证集选阈值、评估、事件合并
  detector.py                统一检测接口，可常驻复用模型
  api.py                     FastAPI 上传与检测接口
  cli.py                     统一命令入口
  diagnostics.py             波形/STFT/通用事件得分图
  vendor/panns/              训练和推理共用的一份上游模型实现
src/panns_inference/         兼容原有常用导入方式
research/audioset_tagging_cnn/ 原始训练、数据打包、评估、微调研究工具
configs/default.json        参数配置，无风速/转速/功率依赖
data/raw/                   待标注原始录音
data/reference/             用户参考录音副本
data/manifests/             JSONL 标注表，可纳入版本管理
data/features/              特征缓存
checkpoints/                官方预训练权重
artifacts/                  训练后的目标分类器
results/                    推理结果和诊断图
scripts/                    环境安装、权重下载、启动脚本
tests/                      数据隔离、训练推理及 API 检查
third_party/                上游版本、许可证
docs/                       架构、数据规范、整合变更和本机验证结果
```

## 训练自己的分类器

### 1. 准备并标注录音

将真实正常运行录音、掉落试验和干扰录音放入 `data/raw/`，生成待标注清单：

```powershell
python -m bolt_audio init-manifest --audio-dir data/raw --output data/manifests/train.jsonl
```

逐条修改清单：填写真实掉落/滚动/碰撞区间、独立试验 `group_id`、`train/val/test` 划分，完整核对后设置 `reviewed: true`。**正常录音使用空 `events`，必须确认整条没有目标事件。** JSONL 每行一条录音，音频相对路径从清单所在目录计算。

参考清单 `data/manifests/reference.jsonl` 保持未标注状态，程序不会猜测准确事件时间。具体规范见 [数据规范](docs/DATASET.md)。

### 2. 检查、提取、训练

```powershell
python -m bolt_audio check-data --manifest data/manifests/train.jsonl
python -m bolt_audio extract --manifest data/manifests/train.jsonl --output data/features/train.npz
python -m bolt_audio train --features data/features/train.npz --output-dir artifacts/bolt_v1
```

分类器使用已冻结的 PANNs 深度特征，配合标准化和逻辑回归。少量真实数据时，可先训练、检查误报，再考虑微调深度网络。`--feature-set panns` 使用纯 PANNs 特征，`--feature-set acoustic` 运行传统特征对照；默认 `fusion` 融合二者，训练和推理必须一致。

程序强制检查同一录音、同一试验组和相同文件内容不能跨数据集。先划分独立试验，再生成窗口。只有与标注事件重叠的窗口为正例，邻近但不重叠的模糊窗口默认排除。阈值以验证集的**整条录音 F1**选择，测试集不参与阈值选择。输出 `classifier.json` 与 `metrics.json`；前者包含数值参数、特征定义、权重哈希、阈值及训练来源，无 pickle 加载。

验证集必须同时有事件录音和完整正常录音。建议独立测试集也覆盖两类、不同风机和背景。上述检查无法自动识别“同一次试验切成不同文件”这种关系，必须正确填写 `group_id`。

### 3. 检测新录音

```powershell
python -m bolt_audio infer --audio data/reference/bolt_drop.wav --classifier artifacts/bolt_v1/classifier.json --output results/bolt_v1.json
```

有目标分类器时返回 `bolt_detected`、窗口得分和合并后的候选事件区间；区间代表窗口覆盖范围，不能当作精确碰撞起止。默认 2 秒窗口、0.5 秒步长，必须按真实试验验证是否合适。

## 服务器接口

本地启动：

```powershell
python -m bolt_audio serve --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/docs` 上传文件测试。接口包括 `GET /health` 和 `POST /detect`，文件字段名为 `file`。

```powershell
curl.exe -F "file=@D:/project_sound/data/reference/bolt_drop.wav" http://127.0.0.1:8000/detect
```

训练后复制 `configs/default.json` 为 `configs/server.json`，将 `classifier_path` 改为 `artifacts/bolt_v1/classifier.json`，再执行：

```powershell
python -m bolt_audio --config configs/server.json serve --host 127.0.0.1 --port 8000
```

模型在启动时加载一次，推理请求逐个处理；繁忙时返回 503，由上传端重试。默认最多 20 MB、60 秒录音，每次结果写到 `results/api/`。接口临时音频处理后清理，不作长期原始数据归档。

可设置环境变量 `BOLT_API_TOKEN` 启用 Bearer token。远程访问需要按服务器环境配置监听地址、TLS/反向代理、上传体积限制及重试。当前提供的是同步 HTTP 原型；设备数据上传协议、正式消息队列、告警短信/平台和长期存储尚未接入。检测 JSON 可供后续告警系统消费。

## 环境和工具

当前使用 Conda 的 `wind-bolt-audio` 环境，Python 3.12、CPU PyTorch 2.6。`environment.yml` 定义解释器与 pip 安装项；Conda 管理环境，PyTorch 等 Python 包安装在该环境内部。

环境路径、IDE 解释器选择和无需激活的运行方法见 [Conda 说明](docs/CONDA.md)。

从项目根目录重建：

```powershell
conda env create -f environment.yml
conda activate wind-bolt-audio
python scripts/download_checkpoints.py
python -m pip check
```

Windows 也可执行 `.\scripts\setup.ps1` 自动创建/复用该 Conda 环境、按已验证的 Windows 锁文件安装，并验证权重。脚本支持 `-EnvName` 自定义环境名；不会向 base 环境安装项目依赖。若 PowerShell 提示 conda activate 尚未初始化，可直接使用 Miniconda Prompt，或使用 `conda run --no-capture-output -n wind-bolt-audio python ...`。

环境包含 PyTorch、librosa、soundfile、torchlibrosa、scikit-learn、SciPy、NumPy、matplotlib、pandas、h5py、FastAPI、Uvicorn、pytest、httpx 和 Ruff。版本实录见 `requirements.lock.windows.txt`，不包含旧项目 editable 路径。

CPU Linux 服务器可参考 `sh scripts/setup.sh`。`environment.yml` 固定 CPU 版 PyTorch；GPU 服务器应新建 Python 3.12 Conda 环境，安装与驱动匹配的 CUDA 版 PyTorch，再安装 `pip install -e '.[api]'` 并配置 `device: cuda`。Windows 环境锁不用于 Linux。

```powershell
python -m pip check
python -m pytest -q
python -m ruff check src tests scripts
```

CPU 容器模板见 `Dockerfile` 和 [架构说明](docs/ARCHITECTURE.md)，模板未在本机构建验证。

本机验证明细见 [LOCAL_CHECK.md](docs/LOCAL_CHECK.md)。需要再次检查真实 HTTP 上传时可运行 `python scripts/check_server.py`（先激活上述 Conda 环境）；脚本会临时启动服务、连续上传两次参考录音并停止服务。

## 当前判断的边界

- 仅对收到的录音作判断，无法覆盖未录音的 90 秒；未使用任何运行工况数据。
- 已有的静止参考录音存在明显削波，只能用来跑通流程；不能代表不同尺寸、落点、安装距离和运行背景下的所有掉落。
- 时域与频域融合、深度预训练特征都有验证价值，但不能保证分离被背景掩盖的信息。需要真实运行背景、独立掉落试验和易混淆干扰声验证。
- 当前训练路线是深度特征加轻量分类器；没有声源分离、自监督现场适配或端到端目标网络微调。上游微调模板保留用于后续研究。
- 通用 SED 仅作诊断，10 ms 插值输出间隔不等于 10 ms 定位精度。训练后仍需独立现场验证；`deployment_validated` 保持 false。

## 来源与本地状态

上游地址、固定提交和许可见 [第三方说明](third_party/NOTICE.md)。本次移除了两仓库重复模型实现和嵌套 Git 管理，整合成一个包；详情见 [整合记录](docs/INTEGRATION.md)。原 C 盘工作区保持原样，D 盘未复制 `.git` 元数据，也未执行 Git 初始化、暂存、提交或上传。
