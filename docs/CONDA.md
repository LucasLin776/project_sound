# Conda 环境

本机已建环境：

```text
Miniconda: C:\Users\tlzn\miniconda3
环境名:    wind-bolt-audio
解释器:    C:\Users\tlzn\miniconda3\envs\wind-bolt-audio\python.exe
项目:      D:\project_sound
```

在 Miniconda Prompt 中：

```bat
conda activate wind-bolt-audio
cd /d D:\project_sound
python -m bolt_audio --help
```

PowerShell 切换目录使用 `cd D:\project_sound`。IDE 选择上面的解释器。

从项目根目录重建：

```bat
conda env create -f environment.yml
conda activate wind-bolt-audio
python scripts/download_checkpoints.py
python -m pip check
```

Windows 可运行 `scripts/setup.ps1` 创建/复用环境，并使用 Windows 锁文件安装；Linux CPU 环境参考 `scripts/setup.sh`。`environment.yml` 固定 Python 3.12 和 CPU PyTorch，Python 包由该 Conda 环境里的 pip 安装。不要把 Windows 锁文件用于 Linux。GPU 服务器应另建环境，安装匹配驱动的 CUDA 版 PyTorch，再安装本项目。

项目安装依赖统一以 `pyproject.toml` 为准。代码通过 editable 安装即时生效；依赖定义改变后可在项目根目录运行：

```bat
python -m pip install -e ".[dev]"
```

旧 `.venv` 的删除被自动审批策略拒绝，暂时保留，现有运行和安装脚本只使用 Conda。此次已从项目依赖和 Windows 锁文件中移除研究工具专用的 h5py/pandas 等条目；没有从你现有的 Conda 环境强制卸载这些包。

推理项目使用独立包`bolt_inference`；可先复用本环境运行其`scripts/*.py`，或在推理目录用`environment.yml`创建`wind-bolt-inference`环境。拆分不自动安装或卸载现有环境的包。
