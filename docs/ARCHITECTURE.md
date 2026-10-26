# 架构与运行方式

```text
设备录音/上传
    ↓
解码 + 质量检查 + 转单声道 + 32kHz 重采样（不做增益归一化）
    ↓
2s 窗口 / 0.5s 步长 / 覆盖尾部 / 短音频右补零
    ├─ PANNs Cnn14 → 2048维深度嵌入
    └─ 时频统计 → RMS、峰值、峰值因子、峭度、过零率、谱质心、
                 谱带宽、4段频带能量比例、正谱通量
    ↓
训练集 StandardScaler + LogisticRegression
    ↓
验证集选阈值 → 相邻正窗口合并 → JSON 检测结果
```

不依赖风速、转速、功率或设备运行状态字段。特征提取不进行降噪或声源分离，原始幅值信息保留；不同设备增益的影响应通过多样化数据验证。多通道输入目前取均值，若设备后续使用空间阵列或通道之间反相，应改为单独通道策略并重新训练。

## 配置与模型一致性

默认配置在 `configs/default.json`。相对 checkpoint/classifier 路径基于配置中 project_dir，project_dir 相对配置文件目录解析。CLI 的音频、清单和输出路径相对当前工作目录。

特征缓存和分类器记录 window/hop、特征类型、采样率、通道策略、幅值策略及官方权重 SHA-256。检测加载分类器时会逐项对比，避免无声使用不匹配特征。分类器的阈值、事件合并间隔和最少正窗口数以训练产物为准。

PANNs 模型保持上游结构；`torch.load(weights_only=True)` 加载本地官方权重。推理包装器和原始研究脚本共用 `vendor/panns/models.py`。`research/` 内仍以 AudioSet 组织的训练命令不能直接替代项目目标分类器训练流程。

## 服务

API 使用 FastAPI lifespan 在启动时加载一个模型，检测调用在工作线程中执行，锁限制单个进程一次推理。繁忙返回 503，避免并发推理耗尽内存；不提供持久化队列或设备侧重试。默认单 Uvicorn worker，增加 worker 会复制模型内存。

接口支持 multipart 文件上传，读取解码前检查上传字节数，解码前通过文件头限制时长、通道和采样率。FastAPI 在进入端点前解析 multipart，因此暴露到远程网络时也应在反向代理设置上传体积限制。可选 Bearer token 通过环境变量注入。

每次检测记录 JSON 到 results/api，临时音频随后清理。没有向第三方发送录音、结果、报警或远程上传操作。现场告警、持久化原始录音、数据库、设备身份和上传协议需要后续接入。

实现参考：[FastAPI 文件上传](https://fastapi.tiangolo.com/tutorial/request-files/)、[FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)、[scikit-learn 1.6 逻辑回归](https://scikit-learn.org/1.6/modules/linear_model.html#logistic-regression)。

## CPU 容器模板

本机未构建验证 Docker。Linux 主机安装 Docker 后，可从项目根目录运行：

```bash
docker build -t wind-bolt-audio .
docker run --rm -p 8000:8000 \
  -v "$PWD/checkpoints:/app/checkpoints:ro" \
  -v "$PWD/artifacts:/app/artifacts:ro" \
  -v "$PWD/results:/app/results" \
  wind-bolt-audio
```

默认没有螺栓分类器，输出 null。加载自己的配置时需挂载配置文件，并将容器命令覆盖为 `python -m bolt_audio --config configs/server.json serve --host 0.0.0.0 --port 8000`。容器不自动下载权重。
