"""
Day 2: 计算图与梯度流
学习目标：理解计算图的构建和梯度传播
"""

import torch
import torch.nn.functional as F

print("=" * 50)
print("计算图与梯度流")
print("=" * 50)

# 1. 简单计算图
print("\n1. 简单计算图")
print("计算: z = (x + y) * (x - y)")
print("     = x^2 - y^2")

x = torch.tensor([3.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)

# 中间节点
a = x + y  # a = 5
b = x - y  # b = 1
z = a * b  # z = 5

print(f"\nx = {x.item()}, y = {y.item()}")
print(f"a = x + y = {a.item()}")
print(f"b = x - y = {b.item()}")
print(f"z = a * b = {z.item()}")

z.backward()
print(f"\n∂z/∂x = {x.grad.item():.4f}")  # 理论: 2x = 6
print(f"∂z/∂y = {y.grad.item():.4f}")    # 理论: -2y = -4

# 2. 链式法则示例
print("\n2. 链式法则：z = f(g(x))")
print("f(u) = u^2, g(x) = 2x + 1")
print("z = (2x + 1)^2")

x = torch.tensor([1.0], requires_grad=True)
u = 2 * x + 1  # g(x) = 2x + 1 = 3
z = u ** 2      # f(u) = u^2 = 9

print(f"\nx = {x.item()}")
print(f"u = 2x + 1 = {u.item()}")
print(f"z = u^2 = {z.item()}")

z.backward()
# dz/dx = dz/du * du/dx = 2u * 2 = 2*3*2 = 12
print(f"dz/dx = {x.grad.item():.4f} (理论: 12)")

# 3. 分支计算图
print("\n3. 分支计算图")
print("a = x^2")
print("b = x^3")
print("c = a + b")

x = torch.tensor([2.0], requires_grad=True)
a = x ** 2  # 4
b = x ** 3  # 8
c = a + b   # 12

print(f"\nx = {x.item()}")
print(f"a = x^2 = {a.item()}")
print(f"b = x^3 = {b.item()}")
print(f"c = a + b = {c.item()}")

c.backward()
# dc/dx = da/dx + db/dx = 2x + 3x^2 = 4 + 12 = 16
print(f"dc/dx = {x.grad.item():.4f} (理论: 16)")

# 4. 多次使用同一变量
print("\n4. 多次使用同一变量")
print("y = x * x * x (即 x^3)")

x = torch.tensor([2.0], requires_grad=True)
y = x * x * x

print(f"\nx = {x.item()}")
print(f"y = x^3 = {y.item()}")

y.backward()
print(f"dy/dx = {x.grad.item():.4f} (理论: 3x^2 = 12)")

# 5. 非标量输出的backward
print("\n5. 非标量输出的backward")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
print(f"\nx = {x}")
print(f"y = x^2 = {y}")

# 需要传入gradient参数（通常是全1向量）
y.backward(torch.ones_like(x))
print(f"dy/dx = {x.grad} (理论: 2x = [2, 4, 6])")

# 6. 高阶导数
print("\n6. 高阶导数")
print("计算 f(x) = x^3 的二阶导数")

x = torch.tensor([2.0], requires_grad=True)
y = x ** 3

# 一阶导数
grad_1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"\nx = {x.item()}")
print(f"f(x) = x^3 = {y.item()}")
print(f"f'(x) = {grad_1.item():.4f} (理论: 3x^2 = 12)")

# 二阶导数
grad_2 = torch.autograd.grad(grad_1, x)[0]
print(f"f''(x) = {grad_2.item():.4f} (理论: 6x = 12)")

# 7. 梯度检查点（内存优化）
print("\n7. 计算图的释放")
x = torch.tensor([1.0], requires_grad=True)

# 默认情况下，计算图在backward后会被释放
y = x ** 2
y.backward()
print(f"第一次backward成功")

try:
    y.backward()  # 会报错，因为计算图已释放
    print("第二次backward成功")
except RuntimeError as e:
    print(f"第二次backward失败: 计算图已释放")

# 如果需要多次backward，设置retain_graph=True
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)
print(f"\n使用retain_graph=True:")
print(f"第一次backward: x.grad = {x.grad}")

x.grad.zero_()
y.backward(retain_graph=True)
print(f"第二次backward: x.grad = {x.grad}")

# 8. 实战：神经网络中的梯度流
print("\n8. 实战：简单神经网络的梯度")
print("网络: y = W2 * ReLU(W1 * x + b1) + b2")

# 参数
W1 = torch.tensor([[0.5]], requires_grad=True)
b1 = torch.tensor([0.1], requires_grad=True)
W2 = torch.tensor([[0.8]], requires_grad=True)
b2 = torch.tensor([0.2], requires_grad=True)

# 输入
x = torch.tensor([[1.0]])
y_true = torch.tensor([[2.0]])

# 前向传播
h = torch.matmul(x, W1.T) + b1  # 隐藏层
h = F.relu(h)                    # 激活函数
y_pred = torch.matmul(h, W2.T) + b2  # 输出层
loss = (y_pred - y_true) ** 2    # 损失

print(f"\n前向传播:")
print(f"隐藏层输出: {h.item():.4f}")
print(f"预测值: {y_pred.item():.4f}")
print(f"损失: {loss.item():.4f}")

# 反向传播
loss.backward()
print(f"\n梯度:")
print(f"∂L/∂W1 = {W1.grad.item():.4f}")
print(f"∂L/∂b1 = {b1.grad.item():.4f}")
print(f"∂L/∂W2 = {W2.grad.item():.4f}")
print(f"∂L/∂b2 = {b2.grad.item():.4f}")

print("\n" + "=" * 50)
print("计算图学习完成！")
print("=" * 50)
print("\n核心要点：")
print("1. 计算图记录操作历史")
print("2. backward()从输出到输入传播梯度")
print("3. 链式法则自动应用")
print("4. 计算图在backward后默认释放")
print("5. 梯度在叶子节点累积")
