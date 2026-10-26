# 清理和适配记录

## 上游来源

- [audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)，提交：d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4。
- [panns_inference](https://github.com/qiuqiangkong/panns_inference)，提交：f673f604ec6f4805a61c5b3be087e24776ec5fda。

## 已删除文件

从刚下载的上游副本删除了 24 个文件，共 2502192 字节。用户原有录音及工作区顶层 Git 元数据未删除。

| 文件 | 原始大小（字节） |
|---|---:|
| third_party/audioset_tagging_cnn/resources/classwise_iteration_map.pdf | 94675 |
| third_party/audioset_tagging_cnn/resources/complexity_mAP.pdf | 16081 |
| third_party/audioset_tagging_cnn/resources/long_fig.pdf | 156683 |
| third_party/audioset_tagging_cnn/resources/mAP_table.png | 113434 |
| third_party/audioset_tagging_cnn/resources/R9_ZSCveAHg_7s.mp3 | 42561 |
| third_party/audioset_tagging_cnn/resources/R9_ZSCveAHg_7s.wav | 448078 |
| third_party/audioset_tagging_cnn/resources/sed_R9_ZSCveAHg_7s.png | 220721 |
| third_party/audioset_tagging_cnn/resources/six_figures.pdf | 141749 |
| third_party/audioset_tagging_cnn/resources/six_figures.png | 508742 |
| third_party/audioset_tagging_cnn/scripts/0_inference.sh | 1036 |
| third_party/audioset_tagging_cnn/scripts/1_download_dataset.sh | 2357 |
| third_party/audioset_tagging_cnn/scripts/2_pack_waveforms_to_hdf5s.sh | 1412 |
| third_party/audioset_tagging_cnn/scripts/3_create_training_indexes.sh | 1315 |
| third_party/audioset_tagging_cnn/scripts/4_train.sh | 692 |
| third_party/audioset_tagging_cnn/utils/plot_for_paper.py | 26649 |
| third_party/audioset_tagging_cnn/utils/plot_statistics.py | 102697 |
| third_party/audioset_tagging_cnn/utils/crash.py | 378 |
| third_party/audioset_tagging_cnn/utils/create_black_list.py | 1950 |
| third_party/audioset_tagging_cnn/metadata/black_list/groundtruth_weak_label_evaluation_set.csv | 75925 |
| third_party/audioset_tagging_cnn/metadata/black_list/groundtruth_weak_label_testing_set.csv | 34805 |
| third_party/panns_inference/resources/R9_ZSCveAHg_7s.wav | 448078 |
| third_party/panns_inference/resources/sed_result.png | 29975 |
| third_party/panns_inference/resources/_old_sed_result.png | 29764 |
| third_party/panns_inference/example.py | 2435 |

删除原因：演示音频与配图不属于风机数据；论文绘图/统计脚本服务于论文展示；shell 示例脚本面向 Linux/AudioSet 预训练；DCASE 黑名单仅供特定基准去重；crash.py 是交互式调试钩子。原推理 example.py 由项目入口 scripts/infer_audio.py 替代。

## 保留内容

模型实现、训练入口、微调模板、数据读取/打包/索引、损失和评估代码、527 类标签、上游 README、MIT 许可证以及两个仓库的本地 Git 历史均保留。多种模型实现仍可用于后续消融比较。

## 本地适配

- 推理工具随包携带标签 CSV，移除导入时联网下载标签的副作用。
- 显式检查本地权重文件，缺失时提示下载命令，移除 wget/os.system 自动下载。
- 权重统一存放于项目 checkpoints 目录，可用 checkpoint_path 或 PANNS_CHECKPOINT_DIR 覆盖。
- 推理工具使用 weights_only=True 加载官方权重。
- 主仓库标签路径改为相对源码定位，支持从项目根目录调用。
- np.bool 改为 np.bool_，兼容 NumPy 1.26。
- 使用项目根目录兼容依赖，取代上游 2019 年依赖锁定。
- 新增可复现安装入口、官方权重校验下载器和音频诊断入口。

逐文件原始 SHA-256、大小和上游提交号见 upstream_sources_and_cleanup.json。普通源码删除均可根据上游提交恢复。没有提交或推送 Git。
