"""
Day 2: 自动微分基础 - autograd 入门 | Day 2: Automatic Differentiation Basics - Introduction to autograd
学习目标：理解 PyTorch 的自动微分机制，掌握 requires_grad、backward、grad 三件套 | Learning Goals: Understand PyTorch's autograd mechanism; master requires_grad, backward, and grad

技术栈： | Tech stack:
- PyTorch (torch)
- torch.Tensor（requires_grad 属性）| torch.Tensor (requires_grad attribute)
- torch.autograd（backward, grad）| torch.autograd (backward, grad)
"""

import torch

print("=" * 55)
print("PyTorch 自动微分基础 | PyTorch Autograd Basics")
print("=" * 55)


# =============================================================================
# 1. 创建需要梯度的张量 | 1. Create tensors requiring gradients
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# autograd 的核心是「追踪计算，自动求导」。先看怎么声明一个「需要追踪」的张量。 | autograd's core is "track computation, auto-differentiate." First, see how to declare a "trackable" tensor.
#
# 技术栈：torch.Tensor 的 requires_grad 属性 | Tech stack: torch.Tensor's requires_grad attribute

print("\n1. 创建需要梯度的张量 | 1. Create tensors requiring gradients")
print("-" * 40)

x = torch.tensor([2.0], requires_grad=True)
print(f"x = {x}")
print(f"x.requires_grad = {x.requires_grad}")  # True


# =============================================================================
# 2. 简单函数的自动求导 | 2. Simple function automatic differentiation
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 从最简单的 y = x^2 开始，验证 autograd 计算的梯度与手算结果一致。 | Start with the simplest y = x^2, verify autograd's gradient matches manual calculation.
# 手算：dy/dx = 2x，当 x=2 时，梯度 = 4。 | Manual: dy/dx = 2x, at x=2, gradient = 4.
#
# 技术栈：torch.Tensor 的 backward() 和 grad | Tech stack: torch.Tensor's backward() and grad

print("\n2. 简单函数 y = x^2 的梯度 | 2. Gradient of y = x^2")
print("-" * 40)

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2                               # y = x^2 = 4.0

# 为什么 y.requires_grad 也是 True？ | Why is y.requires_grad also True?
# 有 requires_grad=True 的张量参与运算，结果自动继承。 | If any requires_grad=True tensor participates in the computation, the result inherits it.
print(f"y = x^2 = {y}")                  # y = tensor([4.])
print(f"y.requires_grad = {y.requires_grad}")  # True

y.backward()                             # 计算 dy/dx | Compute dy/dx
print(f"dy/dx = {x.grad.item():.4f} (手算: 2x = 4) | (manual: 2x = 4)")
# dy/dx = 4.0000 (手算: 2x = 4)


# =============================================================================
# 3. 复杂函数：多项式求导 | 3. Complex function: polynomial differentiation
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 多项式是多个基本运算的组合。验证 autograd 能自动处理链式法则。 | A polynomial combines multiple basic operations. Verify autograd handles the chain rule automatically.
# z = 3x^2 + 2x + 1，dz/dx = 6x + 2，x=2 时 = 14。 | z = 3x^2 + 2x + 1, dz/dx = 6x + 2, at x=2 = 14.
#
# 技术栈：torch.Tensor 的 backward()，多步计算图 | Tech stack: backward(), multi-step computation graph

print("\n3. 复杂函数 z = 3x^2 + 2x + 1 | 3. Complex function z = 3x^2 + 2x + 1")
print("-" * 40)

x = torch.tensor([2.0], requires_grad=True)

z = 3 * x**2 + 2 * x + 1
print(f"x = {x.item():.4f}")
print(f"z = 3*x^2 + 2*x + 1 = {z.item():.4f}")   # z = 17.0

z.backward()
print(f"dz/dx = {x.grad.item():.4f} (手算: 6x+2 = 14) | (manual: 6x+2 = 14)")
# dz/dx = 14.0000 (手算: 6x+2 = 14)


# =============================================================================
# 4. 多变量函数：偏导数 | 4. Multivariable function: partial derivatives
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 真实模型有多个参数。同时声明多个 requires_grad=True 的张量，一次 backward 算出所有偏导数。 | Real models have multiple parameters. Declare multiple requires_grad=True tensors; one backward call computes all partial derivatives.
# f = x^2 + y^2，∂f/∂x = 2x，∂f/∂y = 2y。 | f = x^2 + y^2, ∂f/∂x = 2x, ∂f/∂y = 2y.
#
# 技术栈：torch.Tensor 的 backward()，多变量梯度 | Tech stack: backward(), multi-variable gradients

print("\n4. 多变量函数 f(x,y) = x^2 + y^2 | 4. Multivariable function f(x,y) = x^2 + y^2")
print("-" * 40)

x = torch.tensor([3.0], requires_grad=True)
y = torch.tensor([4.0], requires_grad=True)
f = x**2 + y**2

print(f"x = {x.item():.4f}, y = {y.item():.4f}")
print(f"f = x^2 + y^2 = {f.item():.4f}")  # f = 25.0

f.backward()
print(f"∂f/∂x = {x.grad.item():.4f} (手算: 2x = 6) | (manual: 2x = 6)")
print(f"∂f/∂y = {y.grad.item():.4f} (手算: 2y = 8) | (manual: 2y = 8)")
# ∂f/∂x = 6.0000 (手算: 2x = 6)
# ∂f/∂y = 8.0000 (手算: 2y = 8)


# =============================================================================
# 5. 向量函数：非标量输出的 backward | 5. Vector function: backward with non-scalar output
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 当输出是向量时，backward() 需要一个「起始梯度」来展开计算。 | When the output is a vector, backward() needs an "initial gradient" to unfold the computation.
# 原理：向量不能直接标量求导，需要给定每个分量的权重（通常用全1）。 | Principle: vectors can't be differentiated directly; each component needs a weight (usually all-ones).
# 这里 y = 2x，dy/dx 每个位置都是 2。 | Here y = 2x, dy/dx at each position is 2.
#
# 技术栈：torch.Tensor 的 backward(gradient=...) | Tech stack: backward(gradient=...)

print("\n5. 向量函数 y = 2x | 5. Vector function y = 2x")
print("-" * 40)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)  # shape: (3,)
y = x * 2                                                # y = [2.0, 4.0, 6.0]

print(f"x = {x}")              # tensor([1., 2., 3.])
print(f"y = 2x = {y}")          # tensor([2., 4., 6.])
print(f"y 的形状: {y.shape} | y shape: {y.shape}")  # torch.Size([3])

# 为什么传入 torch.ones_like(x)？ | Why pass torch.ones_like(x)?
# 等价于先对 y 各分量求和（得到标量），再求导。 | Equivalent to summing all y components first (getting a scalar), then differentiating.
y.backward(torch.ones_like(x))
print(f"dy/dx = {x.grad} (手算: 每个位置都是 2) | (manual: 2 at every position)")
# dy/dx = tensor([2., 2., 2.])


# =============================================================================
# 6. 梯度累积与清零 | 6. Gradient accumulation and zeroing
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# PyTorch 默认累积梯度——每次 backward() 把新梯度加到 .grad 上。 | PyTorch accumulates gradients by default — each backward() adds new gradients to .grad.
# 如果不清理，后续 backward 的结果会叠加上一次的值。这是 Day 4 训练循环的关键知识点。 | Without clearing, subsequent backward() results stack onto previous ones. This is crucial for Day 4 training loops.
#
# 技术栈：torch.Tensor 的 backward() 和 .grad.zero_() | Tech stack: backward() and .grad.zero_()

print("\n6. 梯度累积与清零 | 6. Gradient accumulation and zeroing")
print("-" * 40)

# --- 6a. 演示梯度累积 | 6a. Demonstrate gradient accumulation ---
x = torch.tensor([2.0], requires_grad=True)

y1 = x ** 2            # y1 = 4, dy1/dx = 2x = 4
y1.backward()
print(f"第一次 backward 后: x.grad = {x.grad.item():.4f} (4) | After 1st backward: {x.grad.item():.4f}")

y2 = x ** 3            # y2 = 8, dy2/dx = 3x^2 = 12
y2.backward()          # 梯度不会自动清零！| Gradients are NOT auto-cleared!
print(f"第二次 backward 后: x.grad = {x.grad.item():.4f} (4+12=16) | After 2nd backward: {x.grad.item():.4f} (accumulated!)")

# --- 6b. 正确做法：每次 backward 前清零 | 6b. Correct approach: zero before each backward ---
x = torch.tensor([2.0], requires_grad=True)

y1 = x ** 2
y1.backward()
print(f"\ny1=x^2 的梯度: x.grad = {x.grad.item():.4f} | grad of y1=x^2: {x.grad.item():.4f}")

# 为什么要 zero_()？ | Why zero_()?
# 清零后再做第二次 backward，grad 只包含本次结果。 | Zero first, then second backward — grad only contains this round's result.
x.grad.zero_()
print(f"zero_() 后: x.grad = {x.grad.item():.4f} | After zero_(): {x.grad.item():.4f}")

y2 = x ** 3
y2.backward()
print(f"y2=x^3 的梯度: x.grad = {x.grad.item():.4f} (12) | grad of y2=x^3: {x.grad.item():.4f}")
# y2=x^3 的梯度: x.grad = 12.0000


# =============================================================================
# 7. 阻止梯度追踪 | 7. Preventing gradient tracking
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 推理/评估时不需要梯度，构建计算图浪费内存和时间。两种方式关闭追踪。 | Inference/evaluation doesn't need gradients; building a computation graph wastes memory and time. Two ways to disable tracking.
# torch.no_grad()：上下文管理器，临时关闭。 | torch.no_grad(): context manager, temporary disable.
# .detach()：创建脱离计算图的副本。 | .detach(): create a copy detached from the computation graph.
#
# 技术栈：torch.no_grad(), torch.Tensor.detach() | Tech stack: torch.no_grad(), torch.Tensor.detach()

print("\n7. 阻止梯度追踪 | 7. Preventing gradient tracking")
print("-" * 40)

x = torch.tensor([2.0], requires_grad=True)

# 方法 1：torch.no_grad() 上下文管理器 | Method 1: torch.no_grad() context manager
# 为什么要在这段代码中使用 no_grad？ | Why use no_grad here?
# 在推理阶段，不需要反向传播，使用 no_grad 可以大幅节省内存。 | During inference, no backprop needed — no_grad saves substantial memory.
with torch.no_grad():
    y = x ** 2
    print(f"torch.no_grad() 内: y.requires_grad = {y.requires_grad} | Inside no_grad(): {y.requires_grad}")
    # torch.no_grad() 内: y.requires_grad = False

# 方法 2：.detach() 创建脱离副本 | Method 2: .detach() creates detached copy
y = x ** 2                   # y.requires_grad = True
y_detached = y.detach()       # 新的张量，requires_grad = False
print(f"y.requires_grad = {y.requires_grad}")                        # True
print(f"y_detached.requires_grad = {y_detached.requires_grad}")      # False


# =============================================================================
# 8. 实战：线性回归的梯度计算 | 8. Hands-on: gradient computation for linear regression
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 把前面学的内容串起来：定义参数、前向传播、计算损失、反向传播、查看梯度。 | Tie everything together: define parameters, forward pass, compute loss, backprop, inspect gradients.
# 这是 Day 3 使用 nn.Module 之前的手写版本，帮助理解优化器内部工作原理。 | This is the manual version before using nn.Module in Day 3 — helps understand what optimizers do internally.
# 模型：y_pred = w*x + b，损失：L = (y_pred - y_true)^2 | Model: y_pred = w*x + b, Loss: L = (y_pred - y_true)^2
#
# 技术栈：requires_grad, backward(), .grad | Tech stack: requires_grad, backward(), .grad

print("\n8. 实战：线性回归的梯度 | 8. Hands-on: gradients of linear regression")
print("-" * 40)

# 定义参数（需要梯度）| Define parameters (need gradients)
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)

# 训练数据（固定值，不需要梯度）| Training data (fixed values, no gradients needed)
x_data  = torch.tensor([1.0])
y_true  = torch.tensor([2.0])

# 前向传播 | Forward pass
y_pred = w * x_data + b               # y_pred = 0.5 * 1 + 0.1 = 0.6
loss   = (y_pred - y_true) ** 2       # loss = (0.6 - 2.0)^2 = 1.96

print(f"w = {w.item():.4f}, b = {b.item():.4f}")
print(f"y_pred = {y_pred.item():.4f}, y_true = {y_true.item():.4f}")
print(f"loss = (y_pred - y_true)^2 = {loss.item():.4f}")
# loss = 1.9600

# 反向传播 | Backpropagation
loss.backward()

# 为什么查看 gradient？ | Why inspect gradients?
# 梯度告诉我们「往哪个方向调整参数可以减小 loss」。 | Gradients tell us "which direction to adjust parameters to reduce loss."
# ∂L/∂w = 2*(y_pred - y_true) * x = 2*(0.6-2)*1 = -2.8
# ∂L/∂b = 2*(y_pred - y_true)     = 2*(0.6-2)   = -2.8
print(f"∂L/∂w = {w.grad.item():.4f} (手算: -2.8) | (manual: -2.8)")
print(f"∂L/∂b = {b.grad.item():.4f} (手算: -2.8) | (manual: -2.8)")
# ∂L/∂w = -2.8000 (手算: -2.8)
# ∂L/∂b = -2.8000 (手算: -2.8)

# 梯度下降更新演示 | Gradient descent update demo
# 参数 -= 学习率 * 梯度 → w 变大, b 变大 → y_pred 更接近 y_true
learning_rate = 0.1
with torch.no_grad():
    w -= learning_rate * w.grad  # w = 0.5 - 0.1*(-2.8) = 0.78
    b -= learning_rate * b.grad  # b = 0.1 - 0.1*(-2.8) = 0.38

print(f"\n更新后参数: w = {w.item():.4f}, b = {b.item():.4f} | Updated: w = {w.item():.4f}, b = {b.item():.4f}")
print(f"新预测值: y_pred = {w.item()*x_data.item() + b.item():.4f} | New y_pred = {w.item()*x_data.item() + b.item():.4f}")
# 更新后参数: w = 0.7800, b = 0.3800
# 新预测值: y_pred = 1.1600 (更接近 2.0 了！| closer to 2.0!)


print("\n" + "=" * 55)
print("autograd 基础学习完成！ | autograd basics learning complete!")
print("=" * 55)
print("\n核心要点： | Key takeaways:")
print("1. requires_grad=True 开启梯度追踪 | 1. requires_grad=True enables gradient tracking")
print("2. .backward() 自动计算梯度，结果存入 .grad | 2. .backward() auto-computes gradients, stored in .grad")
print("3. 梯度默认累积，每次 backward 前须 .grad.zero_() | 3. Gradients accumulate by default; must .grad.zero_() before each backward")
print("4. torch.no_grad() 和 .detach() 阻止梯度追踪 | 4. torch.no_grad() and .detach() prevent gradient tracking")
print("5. 向量输出需要传入 gradient 参数（通常用全1）| 5. Vector output needs gradient argument (usually all-ones)")
print("6. 线性回归梯度 = 链式法则自动计算，无需手算偏导 | 6. Linear regression gradients = chain rule auto-computed, no manual partial derivatives needed")
