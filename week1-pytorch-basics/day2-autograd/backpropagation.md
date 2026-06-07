# Day 2: 自动微分与梯度计算

## 学习目标
- 理解自动微分(autograd)机制
- 掌握梯度的计算和使用
- 理解计算图的构建和释放
- 实现基本的梯度下降算法

## 为什么需要反向传播？

### 1. 机器学习的核心问题：优化
在机器学习中，我们的根本目标是通过不断调整模型参数来最小化损失函数。反向传播提供了一种高效的方法来计算损失函数相对于每个参数的梯度。

### 2. 传统梯度计算的局限性
- 手动计算梯度：
  - 对于复杂的神经网络，手动求导几乎是不可能的
  - 容易出错，尤其是在多层网络中
  - 计算效率极低

### 3. 反向传播的优势
- **自动化梯度计算**：自动微分技术可以精确地计算每个参数的梯度
- **计算效率高**：通过链式法则高效地传播梯度信息
- **可扩展性强**：适用于各种复杂的神经网络架构
- **减少人工干预**：开发者可以专注于模型设计，而非繁琐的梯度计算

### 4. 反向传播的工作原理
```
输入层 → 中间层 → 输出层 
   ↑           ↑
梯度回传   损失计算
```

- 前向传播：计算预测值
- 反向传播：根据损失函数计算梯度
- 参数更新：使用梯度下降法调整权重

### 5. 实际示例：线性回归
```python
# 前向传播
y_pred = weights * x + bias

# 计算损失
loss = (y_pred - y_true)²

# 反向传播
loss.backward()  # 自动计算梯度

# 更新参数
weights -= learning_rate * weights.grad
bias -= learning_rate * bias.grad
```

### 6. 反向传播的数学本质：链式法则
对于复合函数 z = f(g(x))，其导数计算遵循链式法则：
dz/dx = df/dg * dg/dx

## 实践意义
反向传播使得深度学习成为可能：
- 训练复杂的神经网络
- 自动学习特征表示
- 实现从简单线性模型到深度神经网络的平滑过渡

## 注意事项
- 梯度可能出现vanishing或exploding问题
- 学习率的选择至关重要
- 不同的优化算法会影响收敛速度

## 练习建议
1. 尝试手动推导简单函数的梯度
2. 观察不同学习率对模型收敛的影响
3. 使用PyTorch实现简单的神经网络并观察反向传播

## 推荐资源
- 深度学习教材：《深度学习》(Goodfellow)
- 反向传播详细解析论文：Understanding Backpropagation Algorithm

---

# Day 2: Autograd and Gradient Computation (English)

## Learning Objectives
- Understand the autograd mechanism
- Master gradient computation and usage
- Understand computational graph construction and release
- Implement basic gradient descent algorithms

## Why Do We Need Backpropagation?

### 1. The Core Problem of Machine Learning: Optimization
In machine learning, our fundamental goal is to minimize the loss function by continuously adjusting model parameters. Backpropagation provides an efficient method to compute the gradient of the loss function with respect to each parameter.

### 2. Limitations of Traditional Gradient Computation
- Manual gradient calculation:
  - For complex neural networks, manual differentiation is nearly impossible
  - Error-prone, especially in multi-layer networks
  - Extremely inefficient computationally

### 3. Advantages of Backpropagation
- **Automated gradient computation**: Autograd technology precisely calculates gradients for each parameter
- **High computational efficiency**: Efficiently propagates gradient information via the chain rule
- **Strong scalability**: Applicable to various complex neural network architectures
- **Reduced manual intervention**: Developers can focus on model design rather than tedious gradient calculations

### 4. How Backpropagation Works
```
Input Layer → Hidden Layer → Output Layer
     ↑              ↑
Gradient Return  Loss Calculation
```

- Forward propagation: Compute predicted values
- Backward propagation: Compute gradients based on the loss function
- Parameter update: Adjust weights using gradient descent

### 5. Practical Example: Linear Regression
```python
# Forward propagation
y_pred = weights * x + bias

# Compute loss
loss = (y_pred - y_true)²

# Backward propagation
loss.backward()  # Automatically compute gradients

# Update parameters
weights -= learning_rate * weights.grad
bias -= learning_rate * bias.grad
```

### 6. The Mathematical Essence of Backpropagation: The Chain Rule
For a composite function z = f(g(x)), its derivative follows the chain rule:
dz/dx = df/dg * dg/dx

## Practical Significance
Backpropagation makes deep learning possible:
- Train complex neural networks
- Automatically learn feature representations
- Achieve smooth transitions from simple linear models to deep neural networks

## Important Notes
- Gradients may suffer from vanishing or exploding problems
- The choice of learning rate is crucial
- Different optimization algorithms affect convergence speed

## Practice Suggestions
1. Try to manually derive gradients of simple functions
2. Observe the effect of different learning rates on model convergence
3. Use PyTorch to implement a simple neural network and observe backpropagation

## Recommended Resources
- Deep Learning textbook: "Deep Learning" (Goodfellow)
- Backpropagation detailed analysis paper: Understanding Backpropagation Algorithm
