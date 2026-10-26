# 本次项目简化记录

## 实际处理方式

自动审批策略拒绝了递归删除命令，未给出更具体原因。为完成目录简化，采用**移出项目、保留可恢复备份**的方式，未绕过限制永久删除。

备份位置：`D:\project_sound_cleanup_backup\20261026_training`。

## 已移出项目的内容

| 原位置 | 原因 |
| --- | --- |
| `research/` | 原 AudioSet 数据下载/打包、训练、评估和不完整微调模板，不是本项目的螺栓训练入口 |
| `docs/legacy/` | 旧上游 README、过期部署说明和历史清理记录 |
| `src/panns_inference/` | 旧包名兼容入口；诊断脚本已直接导入统一 vendor 模块 |
| `third_party/licenses/` | 重复许可证副本；随包分发的两份 MIT 许可证仍保留在 vendor/panns |
| `docs/ARCHITECTURE.md`、`INTEGRATION.md`、`LOCAL_CHECK.md` | 过期/重复描述；当前用法统一到 README、DATASET、CONDA、TRAINING_CODE |
| `Dockerfile`、`.dockerignore` | 当前未使用、未验证的容器模板，环境按 Conda 管理 |
| `requirements.txt` | 与 pyproject.toml 重复的依赖入口；保留 Conda YAML、pyproject 和 Windows 锁文件 |

上述共 39 个文件，包含旧目录内的 Python 缓存；逐文件清单、原大小与备份根目录记录在 `cleanup_inventory.json`。

## 已在原文件中精简

- `vendor/panns/models.py` 只保留 Cnn14、诊断用 Cnn14_DecisionLevelMax、ConvBlock、初始化及必要导入，约 3300 行缩减为 260 行；其余 CNN/ResNet/MobileNet/Wavegram 等未使用结构已去掉。
- `vendor/panns/pytorch_utils.py` 只保留两个模型使用的三个张量函数，缩减为 52 行。
- 保留部分的类/函数体没有改写。裁剪前两个文件另存到备份内 `vendor_before_pruning/`。
- 项目依赖不再声明仅研究代码使用的 h5py、pandas；Windows 锁文件也移除这两项及 pandas 的 pytz/tzdata 项。未强制卸载现有环境中的包。
- 原 `docs/integration_sources.json` 移到 `third_party/sources.json`，保留权重和音频哈希来源记录。

## 保留内容

已有 `.git`、`.vscode`、Conda 环境、原始 WAV、标注清单、两个官方权重、现有结果和必要 MIT 许可证均保留。本次未暂存、提交或上传 Git。

旧 `.venv` 曾被同一策略阻止删除，继续保留；现在的安装、训练和启动只使用 Conda。没有修改原 C 盘项目。

## 本次功能变更

- 检测返回是否掉落、时间段、检测分数、录音质量与处理耗时；详细逐窗结果与模型信息写入独立日志。
- 无目标分类器时停止检测并明确报错；另用 diagnose 跑通预训练特征诊断。
- 增加 Audacity 区间标签导入、显式正常样本确认、覆盖保护与清单备份；支持将训练模型的预测导出为待人工修正的标签。
- 增加一对一时间区间匹配评估，选阈值时使用验证集事件 F1；保留窗口/录音指标、边界误差与按录到时长统计的误报指标。

## 验证记录（2026-10-26）

- Conda 环境 `wind-bolt-audio`：pytest 8 passed；有一条测试客户端依赖的弃用提示，不影响本次测试。Ruff、pip check 通过。
- 合成数据验证标注→特征→训练→JSON 模型→正负检测→API，以及时间报错/重复报警的指标；不代表真实螺栓准确率。
- 两个官方权重均以 strict=True 加载，完整处理参考录音：58 个窗口、2048 维嵌入，通用 SED 为 (3001, 527)；融合诊断为 (58, 2060)。当前输出在 `results/reference_current/`。
- 真实本地 HTTP 连续两次上传验证缺少目标分类器时返回 503，生成不同请求 ID 和对应错误日志，服务检查后已停止。实录在 `results/server_check/check.json`。
- 对比备份的 AST，保留的上游函数/类内容一致；原始参考 WAV 和两个权重 SHA-256 与原来源记录一致。
- 尚无真实螺栓模型；实际运行工况下的识别/时间定位效果还需标注数据训练与独立测试。
