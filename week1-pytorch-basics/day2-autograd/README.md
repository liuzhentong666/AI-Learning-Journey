# Day 2: 自动微分与梯度计算

## 学习目标
- 理解自动微分(autograd)机制
- 掌握梯度的计算和使用
- 理解计算图的构建和释放
- 实现基本的梯度下降算法

## 练习文件

### 01_basic_autograd.py
自动微分基础：
- requires_grad参数
- backward()方法
- 梯度累积与清零
- 阻止梯度追踪
- 简单函数的梯度计算

**运行：**
```bash
python 01_basic_autograd.py
```

### 02_computational_graph.py
计算图与梯度流：
- 计算图的构建
- 链式法则
- 分支计算图
- 高阶导数
- 神经网络中的梯度流

**运行：**
```bash
python 02_computational_graph.py
```

### 03_gradient_descent.py
梯度下降实战：
- 函数优化
- 线性回归
- 多元线性回归
- 动量梯度下降
- 小批量梯度下降

**运行：**
```bash
python 03_gradient_descent.py
```

## 核心概念

### 1. 什么是自动微分？

自动微分(Automatic Differentiation)是一种计算导数的技术。PyTorch使用**反向模式自动微分**（也叫反向传播）。

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2  # y = 4

y.backward()  # 计算 dy/dx
print(x.grad)  # 输出: tensor([4.]) 因为 dy/dx = 2x = 4
```

### 2. 为什么需要自动微分？

深度学习的核心是**梯度下降**：
- 需要计算损失函数对每个参数的梯度
- 手动计算梯度容易出错，且效率低
- 自动微分让我们专注于设计网络结构

### 3. 计算图

PyTorch在前向传播时构建**动态计算图**：

```
输入 x ---[操作1]---> 中间值 a ---[操作2]---> 输出 y
                                                  |
                            反向传播 <-------------
                           (计算梯度)
```

### 4. 关键API

```python
# 创建需要梯度的张量
x = torch.tensor([1.0], requires_grad=True)

# 计算梯度
y = f(x)
y.backward()

# 访问梯度
print(x.grad)

# 清零梯度
x.grad.zero_()

# 阻止梯度追踪
with torch.no_grad():
    y = f(x)  # y不会追踪梯度

# 或使用detach
y = f(x).detach()
```

### 5. 梯度下降

```python
# 标准梯度下降
for epoch in range(epochs):
    # 前向传播
    y_pred = model(x)
    loss = loss_fn(y_pred, y_true)
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    with torch.no_grad():
        param -= learning_rate * param.grad
    
    # 清零梯度（重要！）
    param.grad.zero_()
```

## 常见概念

### 叶子节点 (Leaf Node)
- 用户创建的张量（不是运算结果）
- 梯度只保存在叶子节点
- `x.is_leaf` 检查是否是叶子

### 梯度累积
- 默认情况下，梯度会累积
- 每次backward前需要手动清零：`x.grad.zero_()`

### 计算图释放
- backward()后计算图默认释放
- 节省内存
- 如需多次backward，使用`retain_graph=True`

### 向量函数的梯度
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# 需要传入gradient参数
y.backward(torch.ones_like(x))
```

## 数学背景

### 链式法则
```
如果 z = f(g(x))
则 dz/dx = df/dg * dg/dx
```

### 示例
```python
x = 2
u = 2*x + 1  # u = 5
z = u**2     # z = 25

# dz/dx = dz/du * du/dx
#       = 2u * 2
#       = 2*5*2 = 20
```

## 实战技巧

### 1. 调试梯度
```python
# 检查梯度是否为None
if x.grad is None:
    print("梯度未计算")

# 检查梯度值
print(f"梯度: {x.grad}")

# 检查梯度是否包含NaN
if torch.isnan(x.grad).any():
    print("梯度包含NaN!")
```

### 2. 梯度裁剪
```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
```

### 3. 冻结参数
```python
# 不需要梯度的参数
for param in model.parameters():
    param.requires_grad = False
```

## 练习任务

1. **基础练习**
   - 计算 f(x) = sin(x^2) 在 x=π/4 处的梯度
   - 验证链式法则：手动计算并与autograd对比

2. **实战练习**
   - 实现完整的线性回归（生成数据、训练、评估）
   - 尝试不同的学习率，观察收敛速度
   - 实现动量SGD

3. **挑战练习**
   - 实现二次函数拟合：y = ax^2 + bx + c
   - 可视化损失下降曲线
   - 对比不同优化算法（SGD、动量SGD、Adam）

**答案参考：** 见 `exercises/day2_answers.py`

## 学习资源

- [PyTorch Autograd文档](https://pytorch.org/docs/stable/autograd.html)
- [Autograd教程](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [自动微分详解](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

## 常见错误

### 1. 忘记清零梯度
```python
# ❌ 错误
loss.backward()
optimizer.step()  # 梯度会累积！

# ✅ 正确
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 2. 对非标量调用backward
```python
# ❌ 错误
y = x ** 2  # shape: (3,)
y.backward()  # 报错！

# ✅ 正确
y.backward(torch.ones_like(y))
```

### 3. 在no_grad下创建需要梯度的张量
```python
# ❌ 错误
with torch.no_grad():
    x = torch.tensor([1.0], requires_grad=True)
    # requires_grad会被忽略

# ✅ 正确
x = torch.tensor([1.0], requires_grad=True)
```

## 检查点

完成今天的学习后，你应该能够：
- [ ] 理解什么是自动微分
- [ ] 使用backward()计算梯度
- [ ] 实现基本的梯度下降
- [ ] 理解计算图的概念
- [ ] 知道何时清零梯度
- [ ] 能调试梯度相关问题

---

**下一步：** Day 3 - 神经网络基础（nn.Module）

---

# Day 2: Automatic Differentiation and Gradient Computation

## Learning Objectives
- Understand the automatic differentiation (autograd) mechanism
- Master gradient computation and usage
- Understand computational graph construction and release
- Implement basic gradient descent algorithms

## Exercise Files

### 01_basic_autograd.py
Autograd fundamentals:
- requires_grad parameter
- backward() method
- Gradient accumulation and zeroing
- Preventing gradient tracking
- Gradient computation for simple functions

**Run:**
```bash
python 01_basic_autograd.py
```

### 02_computational_graph.py
Computational graphs and gradient flow:
- Construction of computational graphs
- Chain rule
- Branching computational graphs
- Higher-order derivatives
- Gradient flow in neural networks

**Run:**
```bash
python 02_computational_graph.py
```

### 03_gradient_descent.py
Gradient descent in practice:
- Function optimization
- Linear regression
- Multiple linear regression
- Momentum gradient descent
- Mini-batch gradient descent

**Run:**
```bash
python 03_gradient_descent.py
```

## Core Concepts

### 1. What is Automatic Differentiation?

Automatic Differentiation is a technique for computing derivatives. PyTorch uses **reverse-mode automatic differentiation** (also called backpropagation).

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2  # y = 4

y.backward()  # Compute dy/dx
print(x.grad)  # Output: tensor([4.]) because dy/dx = 2x = 4
```

### 2. Why Do We Need Automatic Differentiation?

The core of deep learning is **gradient descent**:
- We need to compute the gradient of the loss function with respect to each parameter
- Manual gradient computation is error-prone and inefficient
- Automatic differentiation lets us focus on designing network architectures

### 3. Computational Graph

PyTorch builds a **dynamic computational graph** during forward propagation:

```
Input x ---[Op 1]---> Intermediate a ---[Op 2]---> Output y
                                                       |
                              Backpropagation <---------
                            (gradient computation)
```

### 4. Key API

```python
# Create a tensor that requires gradients
x = torch.tensor([1.0], requires_grad=True)

# Compute gradient
y = f(x)
y.backward()

# Access gradient
print(x.grad)

# Zero the gradient
x.grad.zero_()

# Prevent gradient tracking
with torch.no_grad():
    y = f(x)  # y will not track gradients

# Or use detach
y = f(x).detach()
```

### 5. Gradient Descent

```python
# Standard gradient descent
for epoch in range(epochs):
    # Forward pass
    y_pred = model(x)
    loss = loss_fn(y_pred, y_true)

    # Backward pass
    loss.backward()

    # Update parameters
    with torch.no_grad():
        param -= learning_rate * param.grad

    # Zero gradients (important!)
    param.grad.zero_()
```

## Common Concepts

### Leaf Node
- User-created tensors (not results of operations)
- Gradients are only stored on leaf nodes
- `x.is_leaf` checks if a tensor is a leaf

### Gradient Accumulation
- By default, gradients accumulate
- Must manually zero before each backward: `x.grad.zero_()`

### Computational Graph Release
- The computational graph is released by default after backward()
- This saves memory
- Use `retain_graph=True` if multiple backward passes are needed

### Gradients of Vector Functions
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# Must pass the gradient argument
y.backward(torch.ones_like(x))
```

## Mathematical Background

### Chain Rule
```
If z = f(g(x))
Then dz/dx = df/dg * dg/dx
```

### Example
```python
x = 2
u = 2*x + 1  # u = 5
z = u**2     # z = 25

# dz/dx = dz/du * du/dx
#       = 2u * 2
#       = 2*5*2 = 20
```

## Practical Tips

### 1. Debugging Gradients
```python
# Check if gradient is None
if x.grad is None:
    print("Gradient not computed")

# Check gradient values
print(f"Gradient: {x.grad}")

# Check if gradient contains NaN
if torch.isnan(x.grad).any():
    print("Gradient contains NaN!")
```

### 2. Gradient Clipping
```python
# Prevent gradient explosion
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
```

### 3. Freezing Parameters
```python
# Parameters that don't need gradients
for param in model.parameters():
    param.requires_grad = False
```

## Practice Tasks

1. **Basic Exercises**
   - Compute the gradient of f(x) = sin(x^2) at x = π/4
   - Verify the chain rule: manually compute and compare with autograd

2. **Practical Exercises**
   - Implement complete linear regression (generate data, train, evaluate)
   - Try different learning rates and observe convergence speed
   - Implement momentum SGD

3. **Challenge Exercises**
   - Implement quadratic function fitting: y = ax^2 + bx + c
   - Visualize the loss descent curve
   - Compare different optimization algorithms (SGD, momentum SGD, Adam)

**Reference answers:** See `exercises/day2_answers.py`

## Learning Resources

- [PyTorch Autograd Documentation](https://pytorch.org/docs/stable/autograd.html)
- [Autograd Tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [Automatic Differentiation in Detail](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

## Common Errors

### 1. Forgetting to Zero Gradients
```python
# ❌ Wrong
loss.backward()
optimizer.step()  # Gradients will accumulate!

# ✅ Correct
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 2. Calling backward() on Non-Scalars
```python
# ❌ Wrong
y = x ** 2  # shape: (3,)
y.backward()  # Error!

# ✅ Correct
y.backward(torch.ones_like(y))
```

### 3. Creating Requires-Grad Tensors Under no_grad
```python
# ❌ Wrong
with torch.no_grad():
    x = torch.tensor([1.0], requires_grad=True)
    # requires_grad will be ignored

# ✅ Correct
x = torch.tensor([1.0], requires_grad=True)
```

## Checklist

After completing today's study, you should be able to:
- [ ] Understand what automatic differentiation is
- [ ] Use backward() to compute gradients
- [ ] Implement basic gradient descent
- [ ] Understand the concept of computational graphs
- [ ] Know when to zero gradients
- [ ] Debug gradient-related issues

**Next Step:** Day 3 - Neural Network Basics (nn.Module)
