# Day 5-6: MNIST 手写数字分类项目 | Day 5-6: MNIST Handwritten Digit Classification Project

## 项目概述 | Project Overview

Week 1 的综合项目，整合 Day 1-4 所有知识，完成一个从数据到训练到推理的完整深度学习流程。

Comprehensive project for Week 1, integrating all knowledge from Day 1-4 to complete a full deep learning pipeline from data to training to inference.

---

## 学习目标 | Learning Objectives

- 掌握 torchvision 数据集的加载和预处理 | Master loading and preprocessing torchvision datasets
- 搭建 MLP 和 CNN 两种图像分类模型 | Build both MLP and CNN image classification models
- 完整运行训练 + 评估循环，记录 loss 和 accuracy | Run complete training + evaluation loop, track loss and accuracy
- 保存和加载模型权重（state_dict） | Save and load model weights (state_dict)
- 对单张图像做推理，分析错误样本 | Run inference on single images, analyze misclassified samples

---

## 文件说明 | File Descriptions

| 文件 File | 内容 Content |
|-----------|-------------|
| `01_load_and_explore.py` | 下载 MNIST，探索数据集结构，可视化分布，创建 DataLoader Download MNIST, explore dataset structure, visualize distribution, create DataLoader |
| `02_build_model.py` | 搭建 MLP 和 CNN 两种网络，逐层分析形状变化 Build MLP and CNN networks, analyze shape changes layer by layer |
| `03_train_and_evaluate.py` | 完整训练循环，每 epoch 评估，逐类别准确率分析 Complete training loop, per-epoch evaluation, per-class accuracy analysis |
| `04_save_and_load.py` | state_dict 保存/加载，训练检查点（checkpoint）机制 state_dict save/load, training checkpoint mechanism |
| `05_inference.py` | 单张图像推理，批量错误分析，高置信度错误定位 Single image inference, batch error analysis, high-confidence error localization |

---

## 技术栈 | Tech Stack

```
PyTorch (torch)
torchvision (datasets, transforms)
torch.utils.data (DataLoader, Dataset)
torch.nn (Conv2d, Linear, BatchNorm2d, MaxPool2d, Dropout)
torch.optim (Adam)
torch.nn.functional (softmax)
```

---

## 核心概念 | Core Concepts

### 数据流水线 | Data Pipeline

```
原始 PIL Image (28×28, 像素值 0-255)
         ↓ ToTensor()
(1, 28, 28) 张量，值域 [0.0, 1.0]
         ↓ Normalize(mean=0.1307, std=0.3081)
(1, 28, 28) 张量，值域约 [-0.4, 2.8]
         ↓ DataLoader(batch_size=64, shuffle=True)
(64, 1, 28, 28) 批量张量
```

### CNN 形状变化 | CNN Shape Transformations

```
输入 Input:    (batch, 1, 28, 28)
Conv1 + Pool:  (batch, 32, 14, 14)   # 通道 1→32，宽高 28→14
Conv2 + Pool:  (batch, 64,  7,  7)   # 通道 32→64，宽高 14→7
Flatten:       (batch, 3136)          # 64×7×7 = 3136
FC1 + ReLU:   (batch, 128)
FC2 (输出):   (batch, 10)            # 10 类 logit
```

### 训练循环（五步法）| Training Loop (5 Steps)

```python
for images, labels in train_loader:
    optimizer.zero_grad()       # 1. 清零梯度
    logits = model(images)      # 2. 前向传播
    loss = criterion(logits, labels)  # 3. 计算损失
    loss.backward()             # 4. 反向传播
    optimizer.step()            # 5. 更新参数
```

### 训练 vs 推理模式 | Train vs Inference Mode

| 场景 Scenario | 模式 Mode | 梯度 Gradient |
|---|---|---|
| 训练 Training | `model.train()` | `loss.backward()` |
| 验证/测试 Validation/Test | `model.eval()` | `with torch.no_grad():` |

---

## 模型对比 | Model Comparison

| 指标 Metric | MLP | CNN |
|---|---|---|
| 参数量 Parameters | ~235,000 | ~422,000 |
| 空间感知 Spatial Awareness | 否（展平丢失）No (flatten loses it) | 是（卷积保留）Yes (convolution preserves it) |
| MNIST 预期准确率 Expected Accuracy | ~97% | ~99% |
| 适用数据 Suitable Data | 表格/向量 tabular/vector | 图像/空间 image/spatial |

---

## 常见错误 | Common Mistakes

**1. 忘记 unsqueeze(0)**
单张推理：`image.unsqueeze(0)` 把 `(1,28,28)` → `(1,1,28,28)`，加上批次维度。

**2. 推理忘记 eval() + no_grad()**
不调用 `model.eval()` 会导致 Dropout 在推理时随机丢弃，结果不稳定。
不加 `torch.no_grad()` 会构建计算图，浪费内存。

**3. CrossEntropyLoss 输出层加了 Softmax**
`CrossEntropyLoss` 内部已含 softmax，重复添加会导致梯度消失，训练不收敛。

**4. 加载 state_dict 前忘记创建模型实例**
必须先 `model = MNISTCnn()` 创建空模型，再 `model.load_state_dict(state)`。

---

## 运行顺序 | Execution Order

```bash
# 建议按顺序运行 | Run in order
python 01_load_and_explore.py   # 先下载数据
python 02_build_model.py        # 验证模型结构
python 03_train_and_evaluate.py # 完整训练（约 5-15 分钟 CPU）
python 04_save_and_load.py      # 保存/加载（依赖 03 生成的模型）
python 05_inference.py          # 推理分析（依赖 04 生成的权重文件）
```

---

## 检查点 | Checkpoints

- [ ] 运行 01，确认 MNIST 下载成功，DataLoader 输出 `(64, 1, 28, 28)` | Run 01, confirm MNIST downloaded, DataLoader outputs `(64, 1, 28, 28)`
- [ ] 运行 02，确认 CNN 最终输出形状为 `(4, 10)` | Run 02, confirm CNN final output shape is `(4, 10)`
- [ ] 运行 03，5 epochs 后测试准确率 > 98% | Run 03, test accuracy > 98% after 5 epochs
- [ ] 运行 04，加载后准确率与保存前完全一致 | Run 04, accuracy after loading matches before saving
- [ ] 运行 05，能正确识别单张图像并分析错误 | Run 05, correctly identifies single images and analyzes errors

---

## 与前几天的联系 | Connections to Previous Days

| Day | 概念 Concept | 在本项目中的体现 Application in This Project |
|-----|-------------|---------------------------------------------|
| Day 1 | 张量操作 Tensor ops | 图像张量 `(batch, 1, 28, 28)` 的操作 |
| Day 2 | 自动梯度 Autograd | `loss.backward()` 自动计算所有参数梯度 |
| Day 3 | nn.Module | `MNISTCnn` 继承 `nn.Module`，实现 `__init__` 和 `forward` |
| Day 4 | DataLoader + 训练循环 | 完整的 epoch + batch 双层循环 |

---

*Day 5-6 MNIST 项目 — Week 1 综合实践*
*Day 5-6 MNIST Project — Week 1 Comprehensive Practice*
