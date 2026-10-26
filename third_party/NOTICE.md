# 第三方源码与许可证

本项目在本地整合以下两个 MIT 许可项目，保留各自许可证及版本来源：

| 来源 | 固定版本 | 本项目用途 |
| --- | --- | --- |
| https://github.com/qiuqiangkong/audioset_tagging_cnn | d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4 | 全部 PANNs 模型架构、原始研究训练工具 |
| https://github.com/qiuqiangkong/panns_inference | f673f604ec6f4805a61c5b3be087e24776ec5fda | 推理接口设计、AudioSet 标签表 |

`src/bolt_audio/vendor/panns/models.py` 和 `pytorch_utils.py` 来自主仓库；模型工具导入改为包内相对导入。`config.py`、标签 CSV 来自配套推理工具。

`inference.py` 基于配套工具接口改写，加载主仓库的同一套模型，实现统一设备检查、本地权重加载与 `torch.inference_mode()`。上游 SED 的 nearest 插值与主仓库行为一致，其他插值方式不开放。两种官方权重均以 strict=True 加载。

`src/panns_inference/__init__.py` 保留常用导入兼容入口，不再安装独立的上游推理包。训练和推理共用 vendor 源码，避免两套模型定义分叉。

`research/audioset_tagging_cnn` 保留原始训练、数据打包、评估和微调模板；其中 `pytorch/models.py` 改为统一模型的导入桥接。上游 README 保留历史说明，当前安装与运行方式以根目录 README 为准。上游作者代码的 MIT 许可见 `licenses/`。

官方预训练权重来源：https://zenodo.org/records/3987831 。源码许可不自动代表录音或权重的全部使用条件；原始来源保持可追溯。
