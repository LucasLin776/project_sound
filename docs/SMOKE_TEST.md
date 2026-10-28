# 手动训练冒烟测试

历史6条训练录音位于`data/manifests/bolt_aug_v2_eq_smoke.jsonl`，3正3反，仅用来检查训练链路。当前全量入口默认使用第二版818条数据，见`FULL_TEST.md`。

```powershell
conda activate wind-bolt-audio
cd D:\project_sound
python scripts/smoke_train.py --check-only
```

只预检，不初始化模型或训练。由你开始冒烟训练：

```powershell
python scripts/smoke_train.py
```

依次检查特征提取、拟合、保存并重载分类器。使用固定阈值0.5；不计算泛化性能，不使用验证或测试集。模型标记`smoke_test_only`，独立推理项目和模型导出均拒绝使用它。

输出到新建`artifacts/smoke_runs/<编号>/`，包含plan、features、model和smoke_report。拆分后不再执行正反例WAV部署检测，也不生成推理日志。推理接口的检查位于独立项目`D:\project_sound_inference`。
