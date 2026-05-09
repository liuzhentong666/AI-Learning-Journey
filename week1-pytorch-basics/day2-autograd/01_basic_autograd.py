"""
Day 2: 自动微分基础 - autograd入门
学习目标：理解PyTorch的自动微分机制
"""

import torch

print("=" * 50)
print("PyTorch自动微分(Autograd)基础")
print("=" * 50)

# 1. 创建需要梯度的张量
print("\n1. 创建需要梯度的张量")
x = torch.tensor([2.0], requires_grad=True)
print(f"x = {x}")
print(f"requires_grad = {x.requires_grad}")

# 2. 简单的函数计算
print("\n2. 计算函数 y = x^2")
y = x ** 2
print(f"y = x^2 = {y}")
print(f"y.requires_grad = {y.requires_grad}")  # 自动继承

# 3. 反向传播计算梯度
print("\n3. 反向传播")
y.backward()  # 计算 dy/dx
print(f"dy/dx = {x.grad}")  # 理论值：2x = 2*2 = 4

# 4. 更复杂的函数
print("\n4. 复杂函数：z = 3x^2 + 2x + 1")
x = torch.tensor([2.0], requires_grad=True)
z = 3 * x**2 + 2 * x + 1
print(f"x = {x.item()}")
print(f"z = {z.item()}")

z.backward()
print(f"dz/dx = {x.grad}")  # 理论值：6x + 2 = 6*2 + 2 = 14

# 5. 多变量函数
print("\n5. 多变量函数：f(x,y) = x^2 + y^2")
x = torch.tensor([3.0], requires_grad=True)
y = torch.tensor([4.0], requires_grad=True)
f = x**2 + y**2

print(f"x = {x.item()}, y = {y.item()}")
print(f"f = {f.item()}")

f.backward()
print(f"∂f/∂x = {x.grad} (理论值: 2x = 6)")
print(f"∂f/∂y = {y.grad} (理论值: 2y = 8)")

# 6. 向量函数
print("\n6. 向量函数")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2  # y = 2x
print(f"x = {x}")
print(f"y = 2x = {y}")

# 向量函数需要指定gradient参数
y.backward(torch.ones_like(x))  # 相当于对每个元素求和后求导
print(f"dy/dx = {x.grad}")  # 每个位置都是2

# 7. 梯度累积
print("\n7. 梯度累积")
x = torch.tensor([2.0], requires_grad=True)

# 第一次计算
y1 = x ** 2
y1.backward()
print(f"第一次backward后: x.grad = {x.grad}")

# 第二次计算（梯度会累积！）
y2 = x ** 3
y2.backward()
print(f"第二次backward后: x.grad = {x.grad} (梯度累积了！)")

# 8. 梯度清零
print("\n8. 梯度清零")
x = torch.tensor([2.0], requires_grad=True)

y1 = x ** 2
y1.backward()
print(f"第一次backward: x.grad = {x.grad}")

# 清零梯度
x.grad.zero_()
print(f"清零后: x.grad = {x.grad}")

y2 = x ** 3
y2.backward()
print(f"第二次backward: x.grad = {x.grad}")

# 9. 阻止梯度追踪
print("\n9. 阻止梯度追踪")
x = torch.tensor([2.0], requires_grad=True)

# 方法1: torch.no_grad()
with torch.no_grad():
    y = x ** 2
    print(f"torch.no_grad()下: y.requires_grad = {y.requires_grad}")

# 方法2: .detach()
y = x ** 2
y_detached = y.detach()
print(f"detach()后: y_detached.requires_grad = {y_detached.requires_grad}")

# 10. 实际应用：线性回归的梯度
print("\n10. 实际应用：线性回归")
# y = wx + b，损失函数 L = (y_pred - y_true)^2

w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)

# 训练数据
x_data = torch.tensor([1.0])
y_true = torch.tensor([2.0])

# 前向传播
y_pred = w * x_data + b
loss = (y_pred - y_true) ** 2

print(f"w = {w.item():.4f}, b = {b.item():.4f}")
print(f"y_pred = {y_pred.item():.4f}, y_true = {y_true.item():.4f}")
print(f"loss = {loss.item():.4f}")

# 反向传播
loss.backward()
print(f"∂L/∂w = {w.grad.item():.4f}")
print(f"∂L/∂b = {b.grad.item():.4f}")

print("\n" + "=" * 50)
print("自动微分基础学习完成！")
print("=" * 50)
print("\n核心要点：")
print("1. requires_grad=True 开启梯度追踪")
print("2. .backward() 计算梯度")
print("3. .grad 访问梯度")
print("4. .grad.zero_() 清零梯度")
print("5. torch.no_grad() 或 .detach() 阻止梯度追踪")
