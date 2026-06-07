"""
Day 2: 计算图与梯度流 | Day 2: Computational Graph and Gradient Flow
学习目标：理解计算图的构建和梯度传播 | Learning objective: Understand computational graph construction and gradient propagation
"""

import torch
import torch.nn.functional as F

print("=" * 50)
print("计算图与梯度流 | Computational Graph and Gradient Flow")
print("=" * 50)

# 1. 简单计算图 | 1. Simple computational graph
print("\n1. 简单计算图 | 1. Simple Computational Graph")
print("计算: z = (x + y) * (x - y) | Compute: z = (x + y) * (x - y)")
print("     = x^2 - y^2")

x = torch.tensor([3.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)

# 中间节点 | Intermediate nodes
a = x + y  # a = 5
b = x - y  # b = 1
z = a * b  # z = 5

print(f"\nx = {x.item()}, y = {y.item()}")
print(f"a = x + y = {a.item()}")
print(f"b = x - y = {b.item()}")
print(f"z = a * b = {z.item()}")

z.backward()
print(f"\n∂z/∂x = {x.grad.item():.4f}")  # 理论: 2x = 6 | Theory: 2x = 6
print(f"∂z/∂y = {y.grad.item():.4f}")    # 理论: -2y = -4 | Theory: -2y = -4

# 2. 链式法则示例 | 2. Chain rule example
print("\n2. 链式法则：z = f(g(x)) | 2. Chain Rule: z = f(g(x))")
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
print(f"dz/dx = {x.grad.item():.4f} (理论: 12) | (theory: 12)")

# 3. 分支计算图 | 3. Branching computational graph
print("\n3. 分支计算图 | 3. Branching Computational Graph")
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
print(f"dc/dx = {x.grad.item():.4f} (理论: 16) | (theory: 16)")

# 4. 多次使用同一变量 | 4. Using the same variable multiple times
print("\n4. 多次使用同一变量 | 4. Using the Same Variable Multiple Times")
print("y = x * x * x (即 x^3) | y = x * x * x (i.e. x^3)")

x = torch.tensor([2.0], requires_grad=True)
y = x * x * x

print(f"\nx = {x.item()}")
print(f"y = x^3 = {y.item()}")

y.backward()
print(f"dy/dx = {x.grad.item():.4f} (理论: 3x^2 = 12) | (theory: 3x^2 = 12)")

# 5. 非标量输出的backward | 5. Backward with non-scalar output
print("\n5. 非标量输出的backward | 5. Backward with Non-scalar Output")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
print(f"\nx = {x}")
print(f"y = x^2 = {y}")

# 需要传入gradient参数（通常是全1向量）| Need to pass gradient parameter (usually all-ones vector)
y.backward(torch.ones_like(x))
print(f"dy/dx = {x.grad} (理论: 2x = [2, 4, 6]) | (theory: 2x = [2, 4, 6])")

# 6. 高阶导数 | 6. Higher-order derivatives
print("\n6. 高阶导数 | 6. Higher-order Derivatives")
print("计算 f(x) = x^3 的二阶导数 | Compute second derivative of f(x) = x^3")

x = torch.tensor([2.0], requires_grad=True)
y = x ** 3

# 一阶导数 | First derivative
grad_1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"\nx = {x.item()}")
print(f"f(x) = x^3 = {y.item()}")
print(f"f'(x) = {grad_1.item():.4f} (理论: 3x^2 = 12) | (theory: 3x^2 = 12)")

# 二阶导数 | Second derivative
grad_2 = torch.autograd.grad(grad_1, x)[0]
print(f"f''(x) = {grad_2.item():.4f} (理论: 6x = 12) | (theory: 6x = 12)")

# 7. 梯度检查点（内存优化）| 7. Computational graph release (memory optimization)
print("\n7. 计算图的释放 | 7. Computational Graph Release")
x = torch.tensor([1.0], requires_grad=True)

# 默认情况下，计算图在backward后会被释放 | By default, the computational graph is released after backward
y = x ** 2
y.backward()
print(f"第一次backward成功 | First backward succeeded")

try:
    y.backward()  # 会报错，因为计算图已释放 | Will error because computational graph has been released
    print("第二次backward成功 | Second backward succeeded")
except RuntimeError as e:
    print(f"第二次backward失败: 计算图已释放 | Second backward failed: computational graph released")

# 如果需要多次backward，设置retain_graph=True | If multiple backward passes needed, set retain_graph=True
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)
print(f"\n使用retain_graph=True: | Using retain_graph=True:")
print(f"第一次backward: x.grad = {x.grad} | First backward: x.grad = {x.grad}")

x.grad.zero_()
y.backward(retain_graph=True)
print(f"第二次backward: x.grad = {x.grad} | Second backward: x.grad = {x.grad}")

# 8. 实战：神经网络中的梯度流 | 8. Practical: Gradient flow in neural networks
print("\n8. 实战：简单神经网络的梯度 | 8. Practical: Gradients in a Simple Neural Network")
print("网络: y = W2 * ReLU(W1 * x + b1) + b2 | Network: y = W2 * ReLU(W1 * x + b1) + b2")

# 参数 | Parameters
W1 = torch.tensor([[0.5]], requires_grad=True)
b1 = torch.tensor([0.1], requires_grad=True)
W2 = torch.tensor([[0.8]], requires_grad=True)
b2 = torch.tensor([0.2], requires_grad=True)

# 输入 | Input
x = torch.tensor([[1.0]])
y_true = torch.tensor([[2.0]])

# 前向传播 | Forward propagation
h = torch.matmul(x, W1.T) + b1  # 隐藏层 | Hidden layer
h = F.relu(h)                    # 激活函数 | Activation function
y_pred = torch.matmul(h, W2.T) + b2  # 输出层 | Output layer
loss = (y_pred - y_true) ** 2    # 损失 | Loss

print(f"\n前向传播: | Forward propagation:")
print(f"隐藏层输出: {h.item():.4f} | Hidden layer output: {h.item():.4f}")
print(f"预测值: {y_pred.item():.4f} | Prediction: {y_pred.item():.4f}")
print(f"损失: {loss.item():.4f} | Loss: {loss.item():.4f}")

# 反向传播 | Backpropagation
loss.backward()
print(f"\n梯度: | Gradients:")
print(f"∂L/∂W1 = {W1.grad.item():.4f}")
print(f"∂L/∂b1 = {b1.grad.item():.4f}")
print(f"∂L/∂W2 = {W2.grad.item():.4f}")
print(f"∂L/∂b2 = {b2.grad.item():.4f}")

print("\n" + "=" * 50)
print("计算图学习完成！ | Computational graph learning complete!")
print("=" * 50)
print("\n核心要点： | Key Points:")
print("1. 计算图记录操作历史 | 1. Computational graph records operation history")
print("2. backward()从输出到输入传播梯度 | 2. backward() propagates gradients from output to input")
print("3. 链式法则自动应用 | 3. Chain rule is automatically applied")
print("4. 计算图在backward后默认释放 | 4. Computational graph is released by default after backward")
print("5. 梯度在叶子节点累积 | 5. Gradients accumulate at leaf nodes")
