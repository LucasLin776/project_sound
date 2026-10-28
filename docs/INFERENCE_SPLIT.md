# 数据处理、训练、推理拆分说明

## 三个项目的边界

| 项目 | 内容 | 交付物 |
|---|---|---|
| `D:\data_process` | 原始标注、按来源划分、训练数据增强 | WAV、标签、图片、JSONL |
| `D:\project_sound` | 数据导入检查、特征、训练、评估、模型导出 | 模型包 |
| `D:\project_sound_inference` | 指定模型的WAV检测、日志、HTTP | 检测JSON |

推理项目使用自己的`bolt_inference`包，不导入`bolt_audio`或`data_process`，不需要训练仓库在同一台机器上。不读取训练清单、不包含训练、拟合或阈值选择函数。特征提取源代码保持与训练一致，模型特征格式版本为1；后续改动特征定义需同步两边并做兼容性验证。

## 从训练项目移出的文件

- `src/bolt_audio/api.py` → 推理项目`src/bolt_inference/api.py`。
- `src/bolt_audio/detector.py` → 推理项目`src/bolt_inference/detector.py`。
- `scripts/start_server.ps1`、`scripts/check_server.py` → 推理项目的启动和检查工具。
- 命令行`infer`、`serve`以及预测标记导出入口移到推理项目。
- HTTP令牌的`.env.example`示例移到推理项目。

原始文件在本次工作区的`tmp/project_sound_inference_split/before_install/`保留备份。训练项目新增`model_export.py`和`export-model`命令。PANNs源码、许可证、训练用音频处理和数值分类器重载仍是训练评估需要的部分。

## 行为变化

`scripts/full_test.py`和`scripts/smoke_train.py`执行特征、训练及模型重载检查；全量训练仍输出三个集合的指标。训练不再创建新的`predictions/`、`logs/`或`detection_summary.csv`；独立推理项目负责这些部署输出。已有结果不修改。

训练项目不再提供HTTP服务，`configs/default.json`移除`classifier_path`及上传大小配置；推理时从指定模型恢复预处理参数和阈值，运行资源参数由CLI指定。独立项目提供`infer.py`、`serve.py`、`check_model.py`源码入口，复用现有Conda环境即可运行。

## 模型交付约定

推荐导出完整目录：`classifier.json`、`Cnn14_mAP=0.431.pth`、`bundle.json`。纯时频模型不需要`.pth`。JSON包含数值权重、标准化、特征配置、阈值及事件合并设置；bundle文件记录格式和哈希。训练数据路径及评估详情不进入部署模型。

推理项目支持旧的原始`classifier.json`，搭配`--checkpoint`指定匹配的Cnn14。模型包可放在任意磁盘路径；更换`--model`即可切换模型，HTTP服务换模型后需重启。官方Cnn14权重也属于运行依赖，单独复制分类器JSON不足以运行fusion/panns模型。

## 仓库范围

独立项目内的`models/`、音频、权重、日志、结果和特征缓存由`.gitignore`排除。可以由你把源码作为新仓库管理。迁移不初始化Git、不设置remote、不提交或推送，也不修改现有数据与训练产物。
