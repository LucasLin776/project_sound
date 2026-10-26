# D 盘整合项目本机验证

## Conda 迁移验证（当前环境）

- 环境名 wind-bolt-audio，位置 C:\Users\tlzn\miniconda3\envs\wind-bolt-audio。
- Python 3.12.15（conda-forge），PyTorch 2.6.0+cpu；项目通过 editable 安装至该环境。
- environment.yml 创建成功；Conda 版 scripts/setup.ps1 的复用、Python 版本检查、Windows 锁依赖安装、权重校验和 pip check 均通过。
- 在 Conda 环境中重新运行 pytest：6 passed，原有 Starlette 弃用警告仍为 1 条；Ruff 检查通过。
- 用改后的 scripts/run_reference.ps1 完整处理参考录音，融合特征仍为 (58, 2060)，未训练标志与 null 检测结果保持正确。
- 新环境未向原有 base 或 aiot 安装项目依赖；当前启动方式见 README 与 CONDA.md。
- 递归删除先前 D 盘 .venv 的命令被系统策略拒绝，未提供更具体原因；旧目录保留，当前脚本不再使用它。

## 初次部署的环境与来源（历史记录）

- 目录：`D:\project_sound`。
- 初次部署使用独立 `.venv`，Python 3.12.14、PyTorch 2.6.0+cpu、NumPy 1.26.4、scikit-learn 1.6.1。之后按用户习惯迁移至 Conda；当前操作以 README 和 CONDA.md 为准。
- FastAPI 0.162.0、Uvicorn 0.54.0；完整已装版本见根目录 requirements.lock.windows.txt。
- `pip check`：No broken requirements found。
- 两个官方权重大小及 MD5 校验通过；两个权重及参考录音与原 C 盘副本 SHA-256 一致，详见 integration_sources.json。
- 未复制 .git；未执行 Git 初始化、暂存、提交、推送或音频远程上传。

## 代码与训练流程

- `pytest -q`：6 passed。测试包括尾部覆盖、短文件、局部事件标注、试验组/内容泄漏、未核对录音阻断、模糊窗口保留，以及合成音频上的特征缓存→训练→数值 JSON 加载→检测→API 流程。
- 合成音频仅用于验证软件链路，不能证明真实螺栓识别性能。临时合成分类器没有放进正式 artifacts。
- 当前 Starlette 对测试客户端的 httpx 路径发出 1 条弃用警告，未导致测试失败；生产 HTTP 检测也已单独通过。
- Ruff：All checks passed。范围为 src/tests/scripts；上游 vendor 模型和保留的诊断脚本不做风格改写。
- 原主仓库 main.py、finetune_template.py 的 --help 可运行，统一模型导入桥接正常；未执行原始 AudioSet 训练或深度微调。

## 用户参考音频

新 CLI 输出 `results/reference/inference.json`：

- 原始 44.1 kHz、单声道 PCM16、约 30.002721 秒。
- 融合特征形状 `(58, 2060)`。
- 未加载螺栓分类器，`bolt_detector_trained=false`、`bolt_detected=null`。
- 原始量程边界样本比例约 1.588896%，削波提示正常。

诊断入口输出：

- PANNs 嵌入 `(58, 2048)`；通用标签 `(58, 527)`。
- 通用 SED `(3001, 527)`。
- 所有数值数组有限；诊断图已目视检查，波形、STFT 与得分图时间轴对齐。
- 输出文件：`results/reference/bolt_drop_result.json`、`bolt_drop_features.npz`、`bolt_drop_diagnostics.png`。

## 实际 HTTP 服务

`scripts/check_server.py` 启动临时 Uvicorn 服务，在本机回环地址进行 GET /health 及两次完整参考音频 multipart 上传，随后停止自己创建的服务进程。

- 两次 POST /detect 返回 200、特征 `(58, 2060)`、`bolt_detected=null`。
- 请求 ID 各不相同，加载耗时字段相同，模型在启动时只加载一次。
- 两次请求内部处理时间约 8.64 秒、7.75 秒；不包含网络上传，且仅代表本机本次运行，不能视为服务器生产延迟承诺。
- 检查实录位于 results/server_check/check.json、server.log，服务结果各保存到 results/api。
- 本机存在代理环境变量，检查脚本已明确设置 trust_env=False，使回环请求直接访问本地服务。

## 尚未验证或实现

真实掉落/正常数据训练及独立现场效果、GPU、Linux 安装、Docker 构建、设备上传协议、正式告警渠道、持久消息队列、长期数据归档。当前没有常驻服务进程；需要时按 README 启动。
