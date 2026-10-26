# 第三方源码与许可证

本项目在本地整合以下两个 MIT 许可项目，保留各自许可证及版本来源：

| 来源 | 固定版本 | 本项目用途 |
| --- | --- | --- |
| https://github.com/qiuqiangkong/audioset_tagging_cnn | d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4 | PANNs Cnn14、诊断用 Cnn14_DecisionLevelMax 及必要张量函数 |
| https://github.com/qiuqiangkong/panns_inference | f673f604ec6f4805a61c5b3be087e24776ec5fda | 推理接口设计、AudioSet 标签表 |

`src/bolt_audio/vendor/panns/models.py` 和 `pytorch_utils.py` 来自主仓库；模型工具导入改为包内相对导入。`config.py`、标签 CSV 来自配套推理工具。

`inference.py` 基于配套工具接口改写，加载主仓库的同一套模型，实现统一设备检查、本地权重加载与 `torch.inference_mode()`。上游 SED 的 nearest 插值与主仓库行为一致，其他插值方式不开放。两种官方权重均以 strict=True 加载。

项目只保留当前使用的两种模型架构、ConvBlock、初始化函数及 do_mixup/interpolate/pad_framewise_output；这些保留部分的函数和类内容没有改动。原始 AudioSet 研究工具、其余架构和旧包名兼容入口已移出项目，详见 docs/CLEANUP.md。

两个上游 MIT 许可文本保留在 `src/bolt_audio/vendor/panns/LICENSE.audioset_tagging_cnn.txt` 和 `LICENSE.panns_inference.txt`，随安装包分发。重复的许可证副本已移出项目；未移除许可义务。权重及参考音频的校验实录见 `sources.json`。

官方预训练权重来源：https://zenodo.org/records/3987831 。源码许可不自动代表录音或权重的全部使用条件；原始来源保持可追溯。
