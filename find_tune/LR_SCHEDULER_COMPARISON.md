# 学习率调度器对比调研

## 概述

学习率调度器（Learning Rate Scheduler）是深度学习训练中的关键组件，它控制学习率在训练过程中的变化方式。合适的学习率调度策略可以显著提升模型收敛速度和最终性能。

本文档对比了常见的学习率调度器，并重点介绍了**带平顶的 Cosine 调度器**（Cosine with Plateau），这是一种结合了 warmup、平顶和 cosine 衰减的调度策略。

---

## 调度器类型对比

### 1. Linear（线性衰减）

```
学习率
  ↑
  │    /\
  │   /  \
  │  /    \
  │ /      \
  │/        \
  └──────────→ 训练步数
    warmup  decay
```

**特点**：
- Warmup 阶段线性上升到最大值
- 然后线性下降到 0
- **最大值只是一个"尖峰"，停留时间为 0**

**适用场景**：
- 传统的全参数微调
- 训练数据量较大时

**缺点**：
- 学习率在最大值停留时间太短
- 可能导致模型在高学习率阶段学习不充分

---

### 2. Cosine（余弦衰减）

```
学习率
  ↑
  │    ___
  │   /   \
  │  /     \
  │ /       \_
  │/          \_
  └──────────────→ 训练步数
    warmup  cosine decay
```

**特点**：
- Warmup 阶段线性上升
- 按余弦曲线平滑下降
- 在最大值和最小值附近变化较慢

**适用场景**：
- LoRA 微调（推荐）
- 需要平滑收敛的场景

**优点**：
- 比线性衰减更平滑
- 在最大值附近下降较慢

**缺点**：
- 仍然没有真正的"平顶"阶段

---

### 3. Cosine with Plateau（带平顶的余弦衰减）⭐ 推荐

```
学习率
  ↑
  │    ________
  │   /        \
  │  /          \
  │ /            \_
  │/               \_
  └──────────────────→ 训练步数
    warmup  plateau  cosine decay
```

**特点**：
- **Warmup 阶段**：线性从 0 上升到 max_lr
- **Plateau 阶段**：保持在 max_lr（平顶）
- **Cosine 衰减阶段**：从 max_lr 余弦衰减到 min_lr

**参数说明**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `warmup_steps` | warmup 步数 | 500 |
| `plateau_ratio` | 平顶阶段占（总步数-warmup）的比例 | 0.1 (10%) |
| `min_lr_ratio` | 最小学习率与最大学习率的比例 | 0.0 |

**适用场景**：
- 需要模型在高学习率阶段充分学习
- 数据量较小，需要更多高学习率训练
- ASR 微调任务

**优点**：
- 在最大学习率停留更长时间，学习更充分
- 结合了 warmup 的稳定性和 cosine 的平滑衰减
- 可调节平顶时长

---

### 4. Constant with Warmup（常数 + Warmup）

```
学习率
  ↑
  │    _______________
  │   /
  │  /
  │ /
  │/
  └──────────────────→ 训练步数
    warmup   constant
```

**特点**：
- Warmup 后保持恒定学习率
- 不衰减

**适用场景**：
- 短期训练
- 需要持续高学习率

**缺点**：
- 训练后期可能震荡
- 不利于精细收敛

---

### 5. Cosine with Restarts（余弦重启）

```
学习率
  ↑
  │  /\    /\    /\
  │ /  \  /  \  /  \
  │/    \/    \/    \
  └──────────────────→ 训练步数
```

**特点**：
- 周期性地重新回到最大学习率
- 每个周期使用余弦衰减

**适用场景**：
- 需要跳出局部最优
- 长期训练

**缺点**：
- 需要调节周期参数
- 可能导致训练不稳定

---

### 6. Polynomial（多项式衰减）

```
学习率
  ↑
  │    _
  │   / \
  │  /   \__
  │ /       \___
  │/            \___
  └──────────────────→ 训练步数
```

**特点**：
- 按多项式函数衰减
- 可调节衰减速度（power 参数）

**适用场景**：
- 需要自定义衰减曲线
- NLP 预训练

---

## 对比总结

| 调度器 | 平顶时间 | 衰减平滑度 | 参数复杂度 | 推荐场景 |
|--------|----------|------------|------------|----------|
| Linear | ❌ 无 | ⭐⭐ | ⭐ | 全参微调 |
| Cosine | ⭐ 短 | ⭐⭐⭐⭐ | ⭐ | LoRA 微调 |
| **Cosine with Plateau** | ⭐⭐⭐⭐ 可调 | ⭐⭐⭐⭐ | ⭐⭐ | **ASR 微调（推荐）** |
| Constant with Warmup | ⭐⭐⭐⭐⭐ 全程 | ❌ | ⭐ | 短期训练 |
| Cosine with Restarts | ⭐⭐ 周期性 | ⭐⭐⭐ | ⭐⭐⭐ | 长期训练 |
| Polynomial | ❌ 无 | ⭐⭐⭐ | ⭐⭐ | 预训练 |

---

## 使用方法

### 在 train_shanghai_wandb.py 中使用

```bash
# 使用带平顶的 Cosine 调度器（默认）
python find_tune/train_shanghai_wandb.py \
    --lr_scheduler_type cosine_with_plateau \
    --plateau_ratio 0.1 \
    --min_lr_ratio 0.0 \
    --warmup_steps 500

# 使用标准 Cosine 调度器
python find_tune/train_shanghai_wandb.py \
    --lr_scheduler_type cosine

# 使用 Linear 调度器
python find_tune/train_shanghai_wandb.py \
    --lr_scheduler_type linear
```

### 参数调优建议

| 场景 | plateau_ratio | min_lr_ratio | warmup_steps |
|------|---------------|--------------|--------------|
| 数据量小（<1000 样本） | 0.2 | 0.01 | 200 |
| 数据量中等（1000-5000） | 0.1 | 0.0 | 500 |
| 数据量大（>5000） | 0.05 | 0.0 | 1000 |
| LoRA 微调 | 0.15 | 0.01 | 300 |

---

## 实现原理

带平顶的 Cosine 调度器的学习率计算公式：

```python
def lr_lambda(current_step):
    # 阶段 1: Warmup（线性上升）
    if current_step < num_warmup_steps:
        return current_step / num_warmup_steps
    
    # 阶段 2: Plateau（保持最大值）
    step_after_warmup = current_step - num_warmup_steps
    if step_after_warmup < plateau_steps:
        return 1.0
    
    # 阶段 3: Cosine 衰减
    step_in_cosine = step_after_warmup - plateau_steps
    progress = step_in_cosine / cosine_steps
    cosine_decay = 0.5 * (1.0 + cos(π * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
```

---

## 可视化对比

假设总训练步数为 10000，warmup_steps=500：

| 步数 | Linear | Cosine | Cosine with Plateau (10%) |
|------|--------|--------|---------------------------|
| 0 | 0.0 | 0.0 | 0.0 |
| 250 | 0.5 | 0.5 | 0.5 |
| 500 | 1.0 | 1.0 | 1.0 |
| 1000 | 0.89 | 0.97 | **1.0** (平顶) |
| 1450 | 0.84 | 0.93 | **1.0** (平顶) |
| 2000 | 0.79 | 0.85 | 0.97 |
| 5000 | 0.47 | 0.50 | 0.50 |
| 8000 | 0.16 | 0.15 | 0.15 |
| 10000 | 0.0 | 0.0 | 0.0 |

可以看到，**Cosine with Plateau 在步数 500-1450 之间保持在最大学习率**，比其他调度器多了约 950 步的高学习率训练时间。

---

---

## 前沿优化器：Muon 和 Schedule-Free

除了传统的学习率调度器，近年来出现了一些革命性的优化方法，它们从根本上改变了学习率调度的范式。

### 7. Muon 优化器 ⭐ 前沿

**论文**：[Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) (2024)

**核心思想**：
Muon (MomentUm Orthogonalized by Newton-Schulz) 通过**正交化**更新矩阵来优化神经网络的隐藏层参数。

```
更新流程：
1. 计算 SGD-Momentum 的更新 G
2. 使用 Newton-Schulz 迭代将 G 正交化为 Ortho(G)
3. 应用正交化后的更新
```

**为什么正交化有效**：
- SGD-momentum 和 Adam 产生的更新矩阵通常条件数很高（接近低秩）
- 所有神经元的更新被少数几个方向主导
- 正交化有效地增加了"稀有方向"的权重

**实验结果**：
| 任务 | 提升 |
|------|------|
| CIFAR-10 94% 准确率 | 3.3 → 2.6 A100-秒 |
| NanoGPT 3.28 val loss | **1.35x 加速** |
| 1.5B 参数 LLM | 13.3h → 10h (8xH100) |

**使用方式**：
```python
# pip install muon-optimizer
from muon import Muon

# Muon 用于隐藏层，AdamW 用于 embedding 和输出层
optimizer = Muon(
    model.hidden_layers.parameters(),
    lr=0.02,
    momentum=0.95,
)
# 输入/输出层仍需使用 AdamW
adamw = torch.optim.AdamW(
    [model.embed.parameters(), model.head.parameters()],
    lr=1e-4,
)
```

**注意事项**：
- 仅适用于 2D 参数（隐藏层）
- Embedding 和输出层仍需使用 AdamW
- FLOP 开销 < 1%（通过 Newton-Schulz 迭代实现）
- 目前主要验证于预训练，微调效果待验证

---

### 8. Schedule-Free 优化器 ⭐ 前沿

**论文**：[The Road Less Scheduled](https://arxiv.org/abs/2405.15682) (Meta, 2024)

**核心思想**：
**完全消除学习率调度的需求**！通过插值和平均的组合替代传统的动量机制。

```
Schedule-Free 更新公式：
y_t = (1-β)z_t + β x_t        # 插值
z_{t+1} = z_t - γ∇f(y_t)      # 梯度步
x_{t+1} = (1-1/(t+1))x_t + (1/(t+1))z_{t+1}  # 平均
```

**三个序列**：
- `z`：主迭代序列
- `y`：梯度计算位置
- `x`：评估/测试时使用的序列

**优势**：
- ✅ **无需指定训练步数**：不需要提前知道训练多久
- ✅ **无需学习率调度**：自动适应训练进度
- ✅ **性能匹配或超越** cosine decay 等精心调优的调度器
- ✅ **内存开销与基础优化器相同**

**实验结果**：
- 🏆 **MLCommons 2024 AlgoPerf 算法效率挑战赛冠军**
- 在多种任务上匹配或超越 cosine schedule

**使用方式**：
```python
# pip install schedulefree
import schedulefree

optimizer = schedulefree.AdamWScheduleFree(
    model.parameters(),
    lr=1e-3,
    warmup_steps=500,
    betas=(0.9, 0.99),
)

# 训练时
model.train()
optimizer.train()  # 重要！

# 评估时
model.eval()
optimizer.eval()  # 重要！切换到 x 序列
```

**注意事项**：
- 必须在 `model.train()/eval()` 时同步调用 `optimizer.train()/eval()`
- 保存 checkpoint 时需要先调用 `optimizer.eval()`
- 如果使用 BatchNorm，需要额外处理
- β 参数比传统动量更敏感，长训练可能需要 0.95-0.98
- 仍然推荐使用 warmup

---

## 优化器/调度器选择指南

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| **快速实验** | Schedule-Free AdamW | 无需调参调度器 |
| **LLM 预训练** | Muon + AdamW | 1.35x 加速 |
| **ASR 微调** | Cosine with Plateau | 平顶阶段学习更充分 |
| **LoRA 微调** | Cosine | 平滑收敛 |
| **全参微调** | Linear 或 Cosine | 经典方案 |
| **不确定训练时长** | Schedule-Free | 无需指定步数 |

---

## 参考资料

1. [Cosine Annealing (SGDR)](https://arxiv.org/abs/1608.03983) - Loshchilov & Hutter, 2016
2. [Warmup for Training Neural Networks](https://arxiv.org/abs/1706.02677) - Goyal et al., 2017
3. [Hugging Face Transformers Schedulers](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
4. [Muon: An optimizer for hidden layers](https://kellerjordan.github.io/posts/muon/) - Keller Jordan et al., 2024
5. [The Road Less Scheduled](https://arxiv.org/abs/2405.15682) - Defazio et al., Meta 2024
6. [Schedule-Free GitHub](https://github.com/facebookresearch/schedule_free)
7. [Muon GitHub](https://github.com/KellerJordan/Muon)
