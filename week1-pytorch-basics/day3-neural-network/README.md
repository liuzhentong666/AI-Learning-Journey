# Day 3: 神经网络基础

## 学习目标

- 理解神经网络是什么、它如何工作
- 掌握 PyTorch 的 `nn.Module` 类，学会定义自己的模型
- 熟悉常用层：`Linear`（全连接）、`Conv2d`（卷积）、`ReLU`（激活函数）
- 理解损失函数和优化器的作用
- 能把「模型 + 损失 + 优化器」组合起来，完成一次简单的训练

---

## 第一部分：神经网络到底是什么？（从零开始）

### 1.1 用生活例子理解

想象你在教一个小朋友识别猫和狗：

1. 你给他看很多猫的照片，告诉他「这是猫」
2. 再给他看很多狗的照片，告诉他「这是狗」
3. 看多了之后，小朋友自己总结规律：猫耳朵更尖、狗鼻子更长……
4. 下次看到新照片，他就能猜这是猫还是狗

**神经网络做的就是这个「看例子 → 总结规律 → 预测新数据」的事情**，只不过它用的是数学计算，而不是人脑。

### 1.2 神经网络的基本结构

一个最简单的神经网络长这样：

```
输入层          隐藏层           输出层
  x1  ──→   [神经元]  ──→   [神经元]  ──→   y（预测结果）
  x2  ──→   [神经元]  ──→   [神经元]  ──→
  x3  ──→   [神经元]  ──→   [神经元]  ──→
```

各部分含义：

| 名称 | 是什么 | 举例 |
|------|--------|------|
| **输入层** | 原始数据 | 一张图片的像素值、一个学生的身高体重 |
| **隐藏层** | 中间加工层，提取特征 | 识别「边缘」「纹理」等 |
| **输出层** | 最终预测结果 | 「是猫的概率 90%」「房价 50 万」 |
| **神经元** | 做一次加权求和 + 激活 | 数学公式：`y = f(w1*x1 + w2*x2 + b)` |
| **权重 w** | 每个输入的重要程度 | 模型要学习的参数 |
| **偏置 b** | 整体偏移量 | 模型要学习的参数 |

**Day 1** 你学了张量（数据容器），**Day 2** 你学了自动微分（算梯度）。  
**Day 3** 就是把数据和梯度结合起来，搭出一个能「学习」的模型。

### 1.3 前向传播：数据怎么流过网络

「前向传播」= 数据从输入层一层层算到输出层，得到预测值。

以单个神经元为例：

```
输入: x = [1.0, 2.0, 3.0]
权重: w = [0.5, 0.3, 0.2]
偏置: b = 0.1

第1步：加权求和  z = w1*x1 + w2*x2 + w3*x3 + b
              z = 0.5*1 + 0.3*2 + 0.2*3 + 0.1 = 1.4

第2步：激活函数  a = ReLU(z) = max(0, 1.4) = 1.4

输出: a = 1.4
```

多层网络就是重复这个过程：上一层的输出，变成下一层的输入。

### 1.4 为什么需要激活函数？（ReLU）

如果每层只做「加权求和」，没有激活函数，那不管叠多少层，等价于一个线性函数，学不了复杂规律。

**激活函数**给网络加入「非线性」，让它能拟合曲线、识别图像等复杂任务。

最常用的激活函数是 **ReLU**：

```
ReLU(x) = max(0, x)

x = -2  →  ReLU = 0
x =  0  →  ReLU = 0
x =  3  →  ReLU = 3
```

直观理解：负数归零，正数保留。简单、高效，是深度学习默认选择。

### 1.5 反向传播与训练（和 Day 2 的连接）

训练分四步，循环进行：

```
┌─────────────────────────────────────────────────────┐
│  1. 前向传播：输入数据 → 模型 → 得到预测值 y_pred      │
│  2. 算损失：比较 y_pred 和真实值 y_true，得到 loss     │
│  3. 反向传播：loss.backward()，计算每个参数的梯度      │
│  4. 更新参数：optimizer.step()，让 loss 变小          │
└─────────────────────────────────────────────────────┘
         ↑                                              │
         └──────────── 重复很多轮（epoch）───────────────┘
```

- **损失函数（Loss）**：衡量「预测有多差」，越小越好
- **优化器（Optimizer）**：根据梯度，自动更新权重和偏置

Day 2 你手动写过 `w -= learning_rate * w.grad`，Day 3 用 PyTorch 内置的优化器代替手写。

---

## 第二部分：PyTorch 中的神经网络（技术栈概览）

本日代码使用的技术栈：

| 技术 | 包/模块 | 作用 |
|------|---------|------|
| PyTorch 核心 | `torch` | 张量运算 |
| 神经网络模块 | `torch.nn` | 层、损失函数、模型基类 |
| 函数式 API | `torch.nn.functional` | 激活函数等无参数操作 |
| 优化器 | `torch.optim` | SGD、Adam 等优化算法 |

### 2.1 nn.Module：所有模型的「父类」

在 PyTorch 里，**每个神经网络都继承 `nn.Module`**。你需要实现两个方法：

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()          # 必须调用，初始化父类
        self.layer1 = nn.Linear(3, 4)  # 在 __init__ 里定义层

    def forward(self, x):
        return self.layer1(x)       # 在 forward 里写数据怎么流动
```

- `__init__`：定义有哪些层（只执行一次）
- `forward`：定义数据怎么流过这些层（每次预测都执行）

### 2.2 常用层速查

#### nn.Linear（全连接层）

把每个输入和每个输出都连起来，适合表格数据、向量输入。

```python
layer = nn.Linear(in_features=784, out_features=128)
# 输入 784 个数 → 输出 128 个数
# 内部参数：权重 W(128×784) + 偏置 b(128)
```

#### nn.Conv2d（二维卷积层）

专门处理图像，通过「滑动小窗口」提取局部特征（边缘、纹理等）。

```python
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
# 输入：3 通道彩色图 (R/G/B)
# 输出：16 个特征图
# kernel_size=3：用 3×3 的小窗口扫描
```

#### nn.ReLU（激活函数）

```python
relu = nn.ReLU()   # 作为层使用
# 或
F.relu(x)          # 作为函数使用（torch.nn.functional）
```

### 2.3 损失函数

| 损失函数 | 适用场景 | PyTorch 写法 |
|----------|----------|--------------|
| MSE（均方误差） | 回归（预测连续数值，如房价） | `nn.MSELoss()` |
| CrossEntropy（交叉熵） | 多分类（预测类别，如猫/狗/鸟） | `nn.CrossEntropyLoss()` |

### 2.4 优化器

| 优化器 | 特点 | PyTorch 写法 |
|--------|------|--------------|
| SGD | 最基础，Day 2 手写过 | `optim.SGD(params, lr=0.01)` |
| Adam | 自适应学习率，常用默认选择 | `optim.Adam(params, lr=0.001)` |

---

## 练习文件

### 01_nn_module_basics.py

学习 `nn.Module` 的基本用法：
- 为什么要继承 `nn.Module`
- `__init__` 和 `forward` 的分工
- 查看模型结构和参数
- `train()` 和 `eval()` 模式的区别

**运行：**
```bash
cd day3-neural-network
python 01_nn_module_basics.py
```

### 02_common_layers.py

学习常用神经网络层：
- `nn.Linear` 全连接层
- `nn.Conv2d` 卷积层（图像）
- `nn.ReLU` 等激活函数
- 用 `nn.Sequential` 快速堆叠层
- 搭建简单的 MLP 和 CNN

**运行：**
```bash
python 02_common_layers.py
```

### 03_loss_and_optimizer.py

学习损失函数与优化器，并完成一次完整训练：
- `nn.MSELoss` 回归任务
- `nn.CrossEntropyLoss` 分类任务
- `optim.SGD` 和 `optim.Adam`
- 标准训练四步循环

**运行：**
```bash
python 03_loss_and_optimizer.py
```

---

## 第三部分：完整训练流程（Day 3 最重要的一张图）

```
  数据 x, 标签 y
       │
       ▼
  ┌─────────┐
  │  model  │  ← nn.Module，你定义的神经网络
  │ forward │
  └────┬────┘
       │ y_pred（预测值）
       ▼
  ┌─────────┐
  │  loss   │  ← nn.MSELoss / nn.CrossEntropyLoss
  │  fn     │
  └────┬────┘
       │ loss（标量，越小越好）
       ▼
  loss.backward()     ← 自动微分（Day 2），计算梯度
       │
       ▼
  optimizer.step()    ← 更新 model 的所有参数
  optimizer.zero_grad() ← 清零梯度，准备下一轮
```

标准训练代码模板（Day 4 会详细展开）：

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = MyModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
  optimizer.zero_grad()       # 1. 清零梯度
  y_pred = model(x)           # 2. 前向传播
  loss = criterion(y_pred, y) # 3. 计算损失
  loss.backward()             # 4. 反向传播
  optimizer.step()            # 5. 更新参数
```

---

## 练习任务

1. **基础练习**
   - 定义一个 3 层 MLP：`784 → 128 → 64 → 10`
   - 打印模型结构和参数总数

2. **实战练习**
   - 用 `nn.Sequential` 搭建一个简单 CNN
   - 输入形状 `(1, 1, 28, 28)`（1 张 28×28 灰度图）
   - 输出 10 个类别的 logits

3. **挑战练习**
   - 生成假数据，用 MSE + Adam 训练线性回归（对比 Day 2 手写梯度下降）
   - 观察 loss 是否下降

**答案参考：** 见 `exercises/day3_exercise_answers.py`（后续补充）

---

## 常见错误

### 1. 忘记调用 super().__init__()

```python
# ❌ 错误
class MyModel(nn.Module):
    def __init__(self):
        self.layer = nn.Linear(10, 5)  # 报错！

# ✅ 正确
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 5)
```

### 2. CrossEntropyLoss 输入格式不对

```python
# ❌ 错误：CrossEntropyLoss 不需要手动做 softmax
loss = criterion(F.softmax(output), target)

# ✅ 正确：直接传入原始 logits，CrossEntropyLoss 内部会做 softmax
loss = criterion(output, target)
```

### 3. 忘记 optimizer.zero_grad()

```python
# ❌ 错误：梯度会累积，训练不稳定
loss.backward()
optimizer.step()

# ✅ 正确
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 学习资源

- [PyTorch nn.Module 文档](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch 神经网络教程](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
- [常用层文档](https://pytorch.org/docs/stable/nn.html)

---

## 检查点

完成今天的学习后，你应该能够：

- [ ] 用自己的话解释：输入层、隐藏层、输出层、权重、偏置
- [ ] 解释为什么需要激活函数（ReLU）
- [ ] 继承 `nn.Module` 并定义 `__init__` 和 `forward`
- [ ] 使用 `nn.Linear`、`nn.Conv2d`、`nn.ReLU`
- [ ] 选择合适的损失函数（MSE / CrossEntropy）
- [ ] 使用 `optim.SGD` 或 `optim.Adam` 更新参数
- [ ] 写出标准训练四步循环

---

**上一步：** Day 2 - 自动微分（autograd）  
**下一步：** Day 4 - 训练循环与 DataLoader
