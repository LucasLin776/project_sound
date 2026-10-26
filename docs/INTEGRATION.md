# 本次整合变更

## 目的地和保留内容

新项目根目录：`D:\project_sound`。原 C 盘工作区未删除或修改。

已复制两个官方模型权重及参考音频。初次部署新建了 Python 3.12 虚拟环境，随后按用户要求改为 Conda 的 wind-bolt-audio 环境，安装与启动脚本已同步更新，详见 CONDA.md。`data/reference/bolt_drop.wav` 保持原始字节内容；两个 checkpoint 保持原始官方内容。

## 两个仓库如何合并

1. 主仓库完整模型文件和模型工具移至 `src/bolt_audio/vendor/panns/`，模型工具导入改成相对导入。
2. 推理工具的接口设计、标签读取与主仓库模型整合，推理使用同一份模型源码。去掉重复 Cnn14/SED 模型定义和重复工具，不再独立安装 panns-inference 包。
3. 保留常用 `from panns_inference import AudioTagging, SoundEventDetection, labels` 导入兼容。该兼容层不承诺覆盖上游包全部内部模块路径。
4. 原主仓库研究训练、评估、数据工具放入 `research/audioset_tagging_cnn/`，其中模型文件改成统一模型导入桥接。其原始训练帮助入口也纳入检查。
5. 两个 MIT 许可证、固定上游提交、历史 README 和之前的删除清单放在 third_party 与 docs/legacy 中保留溯源。

## 本次没有带入或替换的内容

- 两个上游 `.git` 和原项目 `.git` 均未复制，新项目不含嵌套仓库。
- 原 `.venv`、Python 缓存、egg-info、旧运行结果、tmp、旧环境锁未复制。
- 推理仓库独立的 setup.py、重复模型/工具实现、旧 example 入口不再单独放进项目；由 pyproject、统一 vendor 和 CLI 替代。
- 原主仓库的古老依赖清单改为提示安装统一项目；研究模型文件改为导入桥接。
- 之前已清理的演示音频、论文图表、示例 shell、绘图统计脚本及 AudioSet 黑名单未恢复。旧的 24 项详细删除记录见 docs/legacy/CLEANUP.md。

这些操作是对新目录的复制与重组，没有删除原 C 盘两个仓库里的新文件，也没有操作用户其他目录。

## 新增工具

- 独立 Python 包、统一 CLI、JSON 配置。
- WAV 清单生成、人工标注格式与数据检查、分组/录音/内容哈希隔离检查。
- 统一音频质量检查、PANNs/传统时频/融合特征提取、NPZ 缓存。
- 冻结深度特征加轻量分类器训练、验证选阈值、独立测试评估、JSON 模型产物。
- 未训练时返回 null 的检测接口、粗粒度事件合并、常驻 HTTP 服务和结果 JSON。
- PowerShell/Linux 安装脚本、启动脚本、依赖版本锁、CPU 容器模板、pytest/Ruff 检查。

没有执行 Git 初始化、git add、git commit 或 git push。后续需要管理版本时，可以从本目录建立一个单独项目；环境、权重、录音、特征和结果已有忽略规则。
