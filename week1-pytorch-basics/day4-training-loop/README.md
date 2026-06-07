# Day 4: 训练循环与 DataLoader

## 学习目标

- 理解完整训练循环的每一步（前向 → 损失 → 反向 → 更新）
- 掌握 `torch.utils.data.Dataset` 和 `DataLoader`，实现批量数据加载
- 学会拆分训练集与验证集，在每个 epoch 结束后评估模型表现
- 理解 `model.train()` 和 `model.eval()` 的实际作用场景

---

## 第一部分：完整训练循环（从 Day 3 的四步到工程写法）

### 1.1 Day 3 的简化版 vs Day 4 的完整版

Day 3 学的四步循环，每次只处理「全部数据一次」（batch = 全集）：

```
optimizer.zero_grad()
y_pred = model(x)
loss = criterion(y_pred, y)
loss.backward()
optimizer.step()
```

Day 4 引入两个重要概念：

| 概念 | 是什么 | 为什么需要它 |
|------|--------|------------|
| **epoch** | 完整遍历一次全部数据 | 数据需要被看多次才能充分训练 |
| **batch（批次）** | 每次只取一小批数据来更新 | 全量数据太大，内存装不下；小批次还有正则化效果 |

### 1.2 完整训练循环结构

```
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:       ← 按批次迭代
        optimizer.zero_grad()                  ← 1. 清零梯度
        y_pred = model(batch_x)               ← 2. 前向传播
        loss = criterion(y_pred, batch_y)     ← 3. 计算损失
        loss.backward()                        ← 4. 反向传播
        optimizer.step()                       ← 5. 更新参数
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### 1.3 为什么 zero_grad() 必须在每个 batch 之前？

PyTorch 默认**累积**梯度，不清零的话本次梯度会叠加到上次，训练发散。

```
                  第1批           第2批
梯度（清零前）:  [0.5, -0.3]  → [0.5+0.8, -0.3+0.2] = [1.3, -0.1]  ← 错误！
梯度（正确）:    [0.5, -0.3]  → [0.8, 0.2]                          ← 正确
```

---

## 第二部分：DataLoader — 自动批量加载数据

### 2.1 手动方式 vs DataLoader

手动切分数据：

```python
for i in range(0, len(X), batch_size):
    batch_x = X[i:i+batch_size]
    batch_y = y[i:i+batch_size]
    ...
```

用 DataLoader（推荐）：

```python
dataset = TensorDataset(X, y)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_x, batch_y in loader:
    ...
```

### 2.2 DataLoader 的关键参数

| 参数 | 含义 | 常用值 |
|------|------|--------|
| `batch_size` | 每批多少个样本 | 32 / 64 / 128 |
| `shuffle` | 每个 epoch 是否打乱顺序 | 训练集 True，验证集 False |
| `drop_last` | 最后一批不足 batch_size 时是否丢弃 | False（默认） |

### 2.3 Dataset 的两种写法

**方式一：TensorDataset（数据已在内存，最简单）**

```python
from torch.utils.data import TensorDataset, DataLoader

X = torch.randn(100, 4)
y = torch.randint(0, 3, (100,))

dataset = TensorDataset(X, y)
loader  = DataLoader(dataset, batch_size=16, shuffle=True)
```

**方式二：自定义 Dataset（数据来自文件、数据库等）**

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):               # DataLoader 需要知道总样本数
        return len(self.X)

    def __getitem__(self, idx):      # DataLoader 按 idx 取一条样本
        return self.X[idx], self.y[idx]
```

---

## 第三部分：训练集 / 验证集拆分

### 3.1 为什么需要验证集？

只看训练损失会造成「过拟合」幻觉：模型死记训练数据，在新数据上表现差。

```
训练集（80%）  →  训练，更新参数
验证集（20%）  →  每个 epoch 后评估，不更新参数
```

### 3.2 评估时要切换模式

```python
model.eval()
with torch.no_grad():          # 不需要梯度，省内存、加速
    val_pred = model(val_x)
    val_loss = criterion(val_pred, val_y)
model.train()                  # 评估完切回训练模式
```

`torch.no_grad()` 告诉 PyTorch 不必记录计算图，推理时必用。

---

## 练习文件

### 01_complete_training_loop.py

标准完整训练循环：
- 手动构造小数据集（100 个线性回归样本）
- 实现带 epoch 和 batch 的双层循环
- 每 10 个 epoch 打印一次 loss

**运行：**
```bash
cd day4-training-loop
python 01_complete_training_loop.py
```

### 02_dataloader.py

Dataset 与 DataLoader 的两种写法：
- `TensorDataset` 快速封装
- 自定义 `Dataset` 类（实现 `__len__` 和 `__getitem__`）
- 演示 shuffle 对 batch 顺序的影响

**运行：**
```bash
python 02_dataloader.py
```

### 03_train_eval_split.py

完整训练 + 验证流程：
- 80/20 拆分训练集和验证集
- 每个 epoch 计算训练损失和验证损失
- `model.train()` / `model.eval()` + `torch.no_grad()` 的正确用法

**运行：**
```bash
python 03_train_eval_split.py
```

---

## 第四部分：Day 4 完整流程图

```
         原始数据 (X, y)
               │
       ┌───────┴────────┐
       │  80% 训练集     │  20% 验证集
       │  TensorDataset  │  TensorDataset
       │  DataLoader     │  DataLoader
       └───────┬────────┘
               │
       for epoch in range(N):
               │
         ┌─────┴────────────────────────────────────┐
         │  for batch_x, batch_y in train_loader:   │
         │     optimizer.zero_grad()                │  ← 清零梯度
         │     y_pred = model(batch_x)              │  ← 前向传播
         │     loss = criterion(y_pred, batch_y)    │  ← 计算损失
         │     loss.backward()                      │  ← 反向传播
         │     optimizer.step()                     │  ← 更新参数
         └─────┬────────────────────────────────────┘
               │
         model.eval()
         with torch.no_grad():
             val_loss = criterion(model(val_x), val_y)
         model.train()
               │
         print(f"Epoch {e}: train={train_loss:.4f}, val={val_loss:.4f}")
```

---

## 常见错误

### 1. 验证时忘记 model.eval() 和 torch.no_grad()

```python
# 错误：Dropout 仍在随机丢弃，且浪费内存记录梯度
val_loss = criterion(model(val_x), val_y)

# 正确
model.eval()
with torch.no_grad():
    val_loss = criterion(model(val_x), val_y)
model.train()
```

### 2. 验证集 DataLoader 开启了 shuffle

```python
# 错误：验证集打乱没意义，且影响可重现性
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# 正确
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### 3. 在 epoch 循环里只打印最后一个 batch 的 loss

```python
# 错误：只是最后一个 batch 的 loss，不代表整个 epoch
for batch_x, batch_y in loader:
    ...
print(f"Loss: {loss.item()}")

# 正确：累积后求平均
total_loss = 0.0
for batch_x, batch_y in loader:
    ...
    total_loss += loss.item()
avg_loss = total_loss / len(loader)
```

---

## 检查点

完成今天的学习后，你应该能够：

- [ ] 解释 epoch 和 batch 的区别
- [ ] 说出 `optimizer.zero_grad()` 必须在每个 batch 前调用的原因
- [ ] 用 `TensorDataset` + `DataLoader` 实现批量加载
- [ ] 实现自定义 `Dataset`（`__len__` 和 `__getitem__`）
- [ ] 正确使用 `model.eval()` + `torch.no_grad()` 进行验证
- [ ] 拆分训练集和验证集，并每个 epoch 打印双损失

---

**上一步：** Day 3 - 神经网络基础（nn.Module）
**下一步：** Day 5 - MNIST 手写数字分类（第一个真实数据集）

---

# Day 4: Training Loop & DataLoader (English)

## Learning Objectives

- Understand every step of the complete training loop (forward → loss → backward → update)
- Master `torch.utils.data.Dataset` and `DataLoader` for batch data loading
- Learn to split training and validation sets and evaluate model performance after each epoch
- Understand the practical use cases of `model.train()` and `model.eval()`

---

## Part 1: The Complete Training Loop (From Day 3's Four Steps to Production Code)

### 1.1 Day 3 Simplified vs. Day 4 Complete

Day 3's four-step loop processes all data at once (batch = full dataset):

```
optimizer.zero_grad()
y_pred = model(x)
loss = criterion(y_pred, y)
loss.backward()
optimizer.step()
```

Day 4 introduces two key concepts:

| Concept | What it is | Why we need it |
|---------|-----------|----------------|
| **epoch** | One complete pass through all data | Data must be seen multiple times for sufficient training |
| **batch** | Only a small subset of data per update | Full data may not fit in memory; mini-batches also have a regularization effect |

### 1.2 Complete Training Loop Structure

```
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:       ← iterate by batch
        optimizer.zero_grad()                  ← 1. zero gradients
        y_pred = model(batch_x)               ← 2. forward pass
        loss = criterion(y_pred, batch_y)     ← 3. compute loss
        loss.backward()                        ← 4. backpropagation
        optimizer.step()                       ← 5. update parameters
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### 1.3 Why Must zero_grad() Come Before Each Batch?

PyTorch **accumulates** gradients by default. Without clearing, gradients from the current batch stack onto the previous ones, causing unstable training.

```
                  Batch 1         Batch 2
Grad (no clear): [0.5, -0.3] → [0.5+0.8, -0.3+0.2] = [1.3, -0.1]  ← Wrong!
Grad (correct):  [0.5, -0.3] → [0.8, 0.2]                          ← Correct
```

---

## Part 2: DataLoader — Automatic Batch Loading

### 2.1 Manual Slicing vs. DataLoader

Manual slicing:

```python
for i in range(0, len(X), batch_size):
    batch_x = X[i:i+batch_size]
    batch_y = y[i:i+batch_size]
    ...
```

With DataLoader (recommended):

```python
dataset = TensorDataset(X, y)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_x, batch_y in loader:
    ...
```

### 2.2 Key DataLoader Parameters

| Parameter | Meaning | Common values |
|-----------|---------|---------------|
| `batch_size` | Samples per batch | 32 / 64 / 128 |
| `shuffle` | Shuffle order each epoch | True for training, False for validation |
| `drop_last` | Drop the last incomplete batch | False (default) |

### 2.3 Two Ways to Write a Dataset

**Option 1: TensorDataset (data already in memory, simplest)**

```python
from torch.utils.data import TensorDataset, DataLoader

X = torch.randn(100, 4)
y = torch.randint(0, 3, (100,))

dataset = TensorDataset(X, y)
loader  = DataLoader(dataset, batch_size=16, shuffle=True)
```

**Option 2: Custom Dataset (data from files, databases, etc.)**

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):               # DataLoader needs to know total samples
        return len(self.X)

    def __getitem__(self, idx):      # DataLoader fetches one sample by idx
        return self.X[idx], self.y[idx]
```

---

## Part 3: Train / Validation Split

### 3.1 Why Do We Need a Validation Set?

Monitoring only training loss can create an illusion of good performance (overfitting): the model memorizes training data and performs poorly on new data.

```
Training set (80%)   →  train, update parameters
Validation set (20%) →  evaluate after each epoch, no parameter updates
```

### 3.2 Switch Mode During Evaluation

```python
model.eval()
with torch.no_grad():          # no need for gradients, saves memory and speeds up
    val_pred = model(val_x)
    val_loss = criterion(val_pred, val_y)
model.train()                  # switch back to training mode after evaluation
```

`torch.no_grad()` tells PyTorch not to record the computation graph — always use it during inference.

---

## Exercises

### 01_complete_training_loop.py

Standard complete training loop:
- Manually construct a small dataset (100 linear regression samples)
- Implement a two-level loop with epochs and batches
- Print loss every 10 epochs

**Run:**
```bash
cd day4-training-loop
python 01_complete_training_loop.py
```

### 02_dataloader.py

Two approaches to Dataset and DataLoader:
- `TensorDataset` for quick wrapping
- Custom `Dataset` class (implement `__len__` and `__getitem__`)
- Demonstrate how shuffle affects batch order

**Run:**
```bash
python 02_dataloader.py
```

### 03_train_eval_split.py

Complete training + validation workflow:
- 80/20 split into training and validation sets
- Compute both training loss and validation loss each epoch
- Correct usage of `model.train()` / `model.eval()` + `torch.no_grad()`

**Run:**
```bash
python 03_train_eval_split.py
```

---

## Common Mistakes

### 1. Forgetting model.eval() and torch.no_grad() During Validation

```python
# Wrong: Dropout is still randomly dropping, and memory is wasted recording gradients
val_loss = criterion(model(val_x), val_y)

# Correct
model.eval()
with torch.no_grad():
    val_loss = criterion(model(val_x), val_y)
model.train()
```

### 2. Enabling shuffle for the Validation DataLoader

```python
# Wrong: shuffling the validation set is meaningless and hurts reproducibility
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# Correct
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### 3. Only Printing the Last Batch Loss per Epoch

```python
# Wrong: this is only the last batch's loss, not the full epoch
for batch_x, batch_y in loader:
    ...
print(f"Loss: {loss.item()}")

# Correct: accumulate and average
total_loss = 0.0
for batch_x, batch_y in loader:
    ...
    total_loss += loss.item()
avg_loss = total_loss / len(loader)
```

---

## Checklist

After today's learning, you should be able to:

- [ ] Explain the difference between epoch and batch
- [ ] State why `optimizer.zero_grad()` must be called before each batch
- [ ] Use `TensorDataset` + `DataLoader` for batch loading
- [ ] Implement a custom `Dataset` (`__len__` and `__getitem__`)
- [ ] Correctly use `model.eval()` + `torch.no_grad()` for validation
- [ ] Split training and validation sets and print dual losses each epoch

---

**Previous:** Day 3 - Neural Network Basics (nn.Module)
**Next:** Day 5 - MNIST Handwritten Digit Classification (First Real Dataset)
