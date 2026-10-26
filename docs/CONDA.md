# Conda 环境

本项目按用户习惯改为 Conda 管理，默认环境名 **wind-bolt-audio**。

## 本机位置

```text
Conda:  C:\Users\tlzn\miniconda3
环境:   C:\Users\tlzn\miniconda3\envs\wind-bolt-audio
解释器: C:\Users\tlzn\miniconda3\envs\wind-bolt-audio\python.exe
项目:   D:\project_sound
```

环境沿用本机 Conda 的默认存放目录，源码、模型权重、数据与结果仍位于 D 盘项目中。

迁移后安装脚本和启动脚本不再使用 `.venv`。旧 `.venv` 的清理命令被系统策略拦截，目录暂时保留，不影响 Conda 运行。

## 日常运行

在已初始化 Conda 的 PowerShell 中（若用 Miniconda Prompt，切换目录写为 `cd /d D:\project_sound`）：

```powershell
conda activate wind-bolt-audio
cd D:\project_sound
python -m bolt_audio infer --audio data/reference/bolt_drop.wav --output results/reference/inference.json
```

无需激活也可以：

```powershell
cd D:\project_sound
conda run --no-capture-output -n wind-bolt-audio python -m bolt_audio --help
```

IDE 选择上面列出的 Conda 环境解释器。不要把项目依赖安装到 base。

## 重建与更新

在尚无该环境的电脑上，进入项目根目录：

```powershell
conda env create -f environment.yml
conda activate wind-bolt-audio
python scripts/download_checkpoints.py
python -m pip check
```

`environment.yml` 用 Conda 创建 Python 3.12 和 pip，再在这个环境内部安装 CPU 版 PyTorch 和本地项目的 api/dev 依赖。使用 pip 安装的是 Conda 环境内的包；项目代码修改通过 editable 安装即时生效。

Windows 可执行 `scripts/setup.ps1` 创建或复用环境，安装 requirements.lock.windows.txt 中的已验证版本，重新安装 editable 项目并核验权重。可传入 `-EnvName` 改名，默认 wind-bolt-audio；脚本拒绝使用 base。环境已存在时不会删除重建，但要求 Python 3.12。

在已经激活的环境内更新本地项目：

```powershell
python -m pip install -e ".[api,dev]"
python -m pip check
```

YAML 明确固定 CPU 版 PyTorch。迁移至 GPU 服务器时，应单独建立 Conda 环境并按服务器驱动安装合适的 CUDA 版 PyTorch，之后安装本项目；不要直接用 CPU YAML 覆盖 GPU 环境。
