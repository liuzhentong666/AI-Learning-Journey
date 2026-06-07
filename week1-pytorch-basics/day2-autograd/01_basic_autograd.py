"""
Day 2: 自动微分基础 - autograd入门 | Day 2: Automatic Differentiation Basics - Introduction to autograd
学习目标：理解PyTorch的自动微分机制 | Learning objective: Understand PyTorch's automatic differentiation mechanism
"""

import torch

print("=" * 50)
print("PyTorch自动微分(Autograd)基础 | PyTorch Automatic Differentiation (Autograd) Basics")
print("=" * 50)

# 1. 创建需要梯度的张量 | 1. Create tensors requiring gradients
print("\n1. 创建需要梯度的张量 | 1. Create tensors requiring gradients")
x = torch.tensor([2.0], requires_grad=True)
print(f"x = {x}")
print(f"requires_grad = {x.requires_grad}")

# 2. 简单的函数计算 | 2. Simple function computation
print("\n2. 计算函数 y = x^2 | 2. Compute function y = x^2")
y = x ** 2
print(f"y = x^2 = {y}")
print(f"y.requires_grad = {y.requires_grad}")  # 自动继承 | Automatically inherited

# 3. 反向传播计算梯度 | 3. Backpropagation to compute gradients
print("\n3. 反向传播 | 3. Backpropagation")
y.backward()  # 计算 dy/dx | Compute dy/dx
print(f"dy/dx = {x.grad}")  # 理论值：2x = 2*2 = 4 | Theoretical value: 2x = 2*2 = 4

# 4. 更复杂的函数 | 4. More complex function
print("\n4. 复杂函数：z = 3x^2 + 2x + 1 | 4. Complex function: z = 3x^2 + 2x + 1")
x = torch.tensor([2.0], requires_grad=True)
z = 3 * x**2 + 2 * x + 1
print(f"x = {x.item()}")
print(f"z = {z.item()}")

z.backward()
print(f"dz/dx = {x.grad}")  # 理论值：6x + 2 = 6*2 + 2 = 14 | Theoretical value: 6x + 2 = 6*2 + 2 = 14

# 5. 多变量函数 | 5. Multivariable function
print("\n5. 多变量函数：f(x,y) = x^2 + y^2 | 5. Multivariable function: f(x,y) = x^2 + y^2")
x = torch.tensor([3.0], requires_grad=True)
y = torch.tensor([4.0], requires_grad=True)
f = x**2 + y**2

print(f"x = {x.item()}, y = {y.item()}")
print(f"f = {f.item()}")

f.backward()
print(f"∂f/∂x = {x.grad} (理论值: 2x = 6) | (theoretical: 2x = 6)")
print(f"∂f/∂y = {y.grad} (理论值: 2y = 8) | (theoretical: 2y = 8)")

# 6. 向量函数 | 6. Vector function
print("\n6. 向量函数 | 6. Vector Function")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(x.shape)
y = x * 2  # y = 2x
print(f"x = {x}")
print(f"y = 2x = {y}")

# 向量函数需要指定gradient参数 | Vector function requires specified gradient parameter
y.backward(torch.ones_like(x))  # 相当于对每个元素求和后求导 | Equivalent to summing all elements then differentiating
print(f"dy/dx = {x.grad}")  # 每个位置都是2 | Each position is 2

# 7. 梯度累积 | 7. Gradient accumulation
print("\n7. 梯度累积 | 7. Gradient Accumulation")
x = torch.tensor([2.0], requires_grad=True)

# 第一次计算 | First computation
y1 = x ** 2
y1.backward()
print(f"第一次backward后: x.grad = {x.grad} | After first backward: x.grad = {x.grad}")

# 第二次计算（梯度会累积！）| Second computation (gradients will accumulate!)
y2 = x ** 3
y2.backward()
print(f"第二次backward后: x.grad = {x.grad} (梯度累积了！) | After second backward: x.grad = {x.grad} (gradients accumulated!)")

# 8. 梯度清零 | 8. Zeroing gradients
print("\n8. 梯度清零 | 8. Zeroing Gradients")
x = torch.tensor([2.0], requires_grad=True)

y1 = x ** 2
y1.backward()
print(f"第一次backward: x.grad = {x.grad} | First backward: x.grad = {x.grad}")

# 清零梯度 | Zero gradients
x.grad.zero_()
print(f"清零后: x.grad = {x.grad} | After zeroing: x.grad = {x.grad}")

y2 = x ** 3
y2.backward()
print(f"第二次backward: x.grad = {x.grad} | Second backward: x.grad = {x.grad}")

# 9. 阻止梯度追踪 | 9. Preventing gradient tracking
print("\n9. 阻止梯度追踪 | 9. Preventing Gradient Tracking")
x = torch.tensor([2.0], requires_grad=True)

# 方法1: torch.no_grad() | Method 1: torch.no_grad()
with torch.no_grad():
    y = x ** 2
    print(f"torch.no_grad()下: y.requires_grad = {y.requires_grad} | Under torch.no_grad(): y.requires_grad = {y.requires_grad}")

# 方法2: .detach() | Method 2: .detach()
y = x ** 2
y_detached = y.detach()
print(f"detach()后: y_detached.requires_grad = {y_detached.requires_grad} | After detach(): y_detached.requires_grad = {y_detached.requires_grad}")

# 10. 实际应用：线性回归的梯度 | 10. Practical application: Linear regression gradients
print("\n10. 实际应用：线性回归 | 10. Practical Application: Linear Regression")
# y = wx + b，损失函数 L = (y_pred - y_true)^2 | y = wx + b, loss function L = (y_pred - y_true)^2

w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)

# 训练数据 | Training data
x_data = torch.tensor([1.0])
y_true = torch.tensor([2.0])

# 前向传播 | Forward propagation
y_pred = w * x_data + b
loss = (y_pred - y_true) ** 2

print(f"w = {w.item():.4f}, b = {b.item():.4f}")
print(f"y_pred = {y_pred.item():.4f}, y_true = {y_true.item():.4f}")
print(f"loss = {loss.item():.4f}")

# 反向传播 | Backpropagation
loss.backward()
print(f"∂L/∂w = {w.grad.item():.4f}")
print(f"∂L/∂b = {b.grad.item():.4f}")

print("\n" + "=" * 50)
print("自动微分基础学习完成！ | Automatic differentiation basics learning complete!")
print("=" * 50)
print("\n核心要点： | Key points:")
print("1. requires_grad=True 开启梯度追踪 | 1. requires_grad=True enables gradient tracking")
print("2. .backward() 计算梯度 | 2. .backward() computes gradients")
print("3. .grad 访问梯度 | 3. .grad accesses gradients")
print("4. .grad.zero_() 清零梯度 | 4. .grad.zero_() zeroes gradients")
print("5. torch.no_grad() 或 .detach() 阻止梯度追踪 | 5. torch.no_grad() or .detach() prevents gradient tracking")
