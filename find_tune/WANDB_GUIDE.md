# WandB 集成使用指南

## 📊 WandB 监控功能

训练脚本已集成 WandB（Weights & Biases），可以实时监控以下指标：

### 训练指标
- **train/loss**: 训练损失
- **train/learning_rate**: 学习率变化
- **train/epoch**: 当前训练轮数

### 评估指标
- **eval/loss**: 验证集损失
- **eval/wer**: 词错误率（Word Error Rate）

### 最终指标
- **final/eval_loss**: 最终验证损失
- **final/eval_wer**: 最终词错误率

## 🚀 快速开始

### 1. 安装 WandB

```bash
pip install wandb
```

### 2. 登录 WandB

首次使用需要登录：

```bash
wandb login
```

这会打开浏览器，复制你的 API key 并粘贴到终端。

或者直接使用 API key：

```bash
wandb login YOUR_API_KEY
```

### 3. 开始训练

正常运行训练脚本即可：

```bash
python find_tune/train_shanghai.py
```

训练开始后，终端会显示 WandB 运行链接，点击即可查看实时训练进度。

## 📈 查看训练结果

### 在线查看

训练开始后，访问显示的链接：
```
https://wandb.ai/YOUR_USERNAME/whisper-shanghai-finetuning/runs/RUN_ID
```

### 主要图表

WandB 会自动生成以下图表：

1. **Loss 曲线**
   - 训练 loss vs 验证 loss
   - 帮助识别过拟合

2. **WER 曲线**
   - 词错误率随训练步数的变化
   - 越低越好

3. **学习率曲线**
   - 学习率调度可视化

4. **系统监控**
   - GPU 使用率
   - 内存使用
   - CPU 使用率

## ⚙️ 自定义配置

### 修改项目名称

编辑 `train_shanghai.py` 第 95 行：

```python
wandb_project = "your-project-name"
```

### 修改运行名称

编辑 `train_shanghai.py` 第 96 行：

```python
wandb_run_name = "your-run-name"
```

### 添加标签

在 `wandb.init()` 中添加：

```python
wandb.init(
    project=wandb_project,
    name=wandb_run_name,
    tags=["shanghai-dialect", "whisper-small", "v1"],
    config={...}
)
```

### 记录额外指标

在 `WandBCallback` 类中添加：

```python
def on_log(self, args, state, control, logs=None, **kwargs):
    if logs is not None and WANDB_AVAILABLE and wandb.run is not None:
        # 添加你的自定义指标
        if "your_metric" in logs:
            wandb.log({"custom/your_metric": logs["your_metric"]}, step=state.global_step)
```

## 🔧 高级功能

### 1. 对比多次运行

在 WandB 界面中可以：
- 选择多个运行进行对比
- 查看不同超参数的影响
- 生成对比报告

### 2. 超参数搜索

使用 WandB Sweeps 进行自动超参数调优：

```python
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'eval/wer', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'values': [1e-5, 5e-6, 1e-6]},
        'per_device_train_batch_size': {'values': [4, 8, 16]},
    }
}

sweep_id = wandb.sweep(sweep_config, project="whisper-shanghai-finetuning")
wandb.agent(sweep_id, function=train)
```

### 3. 保存模型到 WandB

训练脚本已自动保存配置文件到 WandB。如需保存完整模型：

```python
# 在训练结束后添加
wandb.save(os.path.join(output_dir, "*.bin"))
```

### 4. 离线模式

如果网络不稳定，可以使用离线模式：

```bash
export WANDB_MODE=offline
python find_tune/train_shanghai.py
```

训练完成后同步：

```bash
wandb sync wandb/offline-run-*
```

## 📊 关键指标解读

### Train Loss
- **正常范围**: 开始时较高（2-4），逐渐下降到 0.5-1.5
- **异常情况**: 
  - 不下降：学习率可能太小
  - 震荡剧烈：学习率可能太大
  - 突然上升：可能遇到坏数据

### Eval Loss
- **正常范围**: 略高于 train loss
- **异常情况**:
  - 远高于 train loss：过拟合
  - 不下降：模型容量不足或数据质量问题

### WER (Word Error Rate)
- **优秀**: < 10%
- **良好**: 10-20%
- **可接受**: 20-30%
- **需改进**: > 30%

## 🎯 最佳实践

1. **定期检查**: 每隔几个 epoch 查看一次 WandB
2. **保存快照**: 在 WandB 中标记重要的运行
3. **添加注释**: 在关键点添加注释说明
4. **团队协作**: 邀请团队成员查看训练进度
5. **设置告警**: 配置 WandB 告警，在指标异常时通知

## 🔗 相关资源

- [WandB 官方文档](https://docs.wandb.ai/)
- [WandB Transformers 集成](https://docs.wandb.ai/guides/integrations/huggingface)
- [WandB 最佳实践](https://wandb.ai/site/articles/best-practices-for-ml-experiment-tracking)

## ❓ 常见问题

### Q: 如何禁用 WandB？

A: 卸载 wandb 包即可：
```bash
pip uninstall wandb
```

或设置环境变量：
```bash
export WANDB_DISABLED=true
```

### Q: WandB 会影响训练速度吗？

A: 影响很小（< 1%），因为日志记录是异步的。

### Q: 如何删除旧的运行记录？

A: 在 WandB 网页界面中选择运行，点击删除按钮。

### Q: 可以在多个 GPU 上使用 WandB 吗？

A: 可以，WandB 会自动处理分布式训练的日志聚合。
