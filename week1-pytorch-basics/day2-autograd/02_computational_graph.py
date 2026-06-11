"""
Day 2: 计算图与梯度流 | Day 2: Computational Graph and Gradient Flow
学习目标：理解 autograd 内部如何构建计算图，掌握链式法则、分支图的梯度传播 | Learning Goals: Understand how autograd builds computation graphs internally; master chain rule and branching graph gradient propagation

技术栈： | Tech stack:
- PyTorch (torch)
- torch.autograd.grad（高阶导数）| torch.autograd.grad (higher-order derivatives)
- torch.nn.functional（ReLU）| torch.nn.functional (ReLU)
"""

import torch
import torch.nn.functional as F

print("=" * 55)
print("计算图与梯度流 | Computational Graph and Gradient Flow")
print("=" * 55)


# =============================================================================
# 1. 简单计算图：z = (x + y) * (x - y) | 1. Simple computational graph: z = (x + y) * (x - y)
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 计算图是 autograd 的核心数据结构——每个运算是一个节点，边代表数据流。 | The computation graph is autograd's core data structure — each operation is a node, edges represent data flow.
# 从可视化拆解开始：z = x^2 - y^2，手算 ∂z/∂x = 2x, ∂z/∂y = -2y。 | Start by visualizing the breakdown: z = x^2 - y^2, manual: ∂z/∂x = 2x, ∂z/∂y = -2y.
#
# 技术栈：torch.Tensor 的 backward()，中间节点 | Tech stack: backward(), intermediate nodes

print("\n1. 简单计算图：z = (x+y)(x-y) = x^2-y^2 | 1. Simple graph: z = (x+y)(x-y) = x^2-y^2")
print("-" * 40)

x = torch.tensor([3.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)

# 中间节点（autograd 自动记录这些操作）| Intermediate nodes (autograd records these automatically)
a = x + y  # a = 3 + 2 = 5
b = x - y  # b = 3 - 2 = 1
z = a * b  # z = 5 * 1 = 5

print(f"x = {x.item():.4f}, y = {y.item():.4f}")
print(f"a = x + y = {a.item():.4f}")          # a = 5.0000
print(f"b = x - y = {b.item():.4f}")          # b = 1.0000
print(f"z = a * b = {z.item():.4f}")          # z = 5.0000

z.backward()
print(f"\n∂z/∂x = {x.grad.item():.4f} (手算: 2x = 6)   | (manual: 2x = 6)")
print(f"∂z/∂y = {y.grad.item():.4f} (手算: -2y = -4) | (manual: -2y = -4)")
# ∂z/∂x = 6.0000 (手算: 2x = 6)
# ∂z/∂y = -4.0000 (手算: -2y = -4)

# 为什么 autograd 不需要我们手算 2x 和 -2y？ | Why doesn't autograd need us to compute 2x and -2y manually?
# 链式法则：∂z/∂x = ∂z/∂a * ∂a/∂x + ∂z/∂b * ∂b/∂x = b*1 + a*1 = 1 + 5 = 6 ✓
# Chain rule: ∂z/∂x = ∂z/∂a * ∂a/∂x + ∂z/∂b * ∂b/∂x = b*1 + a*1 = 1 + 5 = 6 ✓


# =============================================================================
# 2. 链式法则：z = f(g(x)) | 2. Chain rule: z = f(g(x))
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 链式法则是一切自动求导的数学基础。用嵌套函数 z = (2x+1)^2 验证。 | The chain rule is the mathematical foundation of all automatic differentiation. Verify with nested function z = (2x+1)^2.
# 手算：dz/dx = dz/du * du/dx = 2u * 2 = 4u = 4*(2x+1) = 8x+4 = 12 (x=1) | Manual: dz/dx = dz/du * du/dx = 2u * 2 = 4u = 4*(2x+1) = 8x+4 = 12 (x=1)
#
# 技术栈：torch.Tensor 的 backward()，链式法则 | Tech stack: backward(), chain rule

print("\n2. 链式法则：z = f(g(x)) = (2x+1)^2 | 2. Chain rule: z = f(g(x)) = (2x+1)^2")
print("-" * 40)

x = torch.tensor([1.0], requires_grad=True)

u = 2 * x + 1       # g(x) = 2x + 1 = 3
z = u ** 2          # f(u) = u^2 = 9

print(f"x = {x.item():.4f}")
print(f"u = 2x + 1 = {u.item():.4f}")      # u = 3.0000
print(f"z = u^2 = {z.item():.4f}")         # z = 9.0000

z.backward()
# dz/dx = dz/du * du/dx = 2u * 2 = 2*3*2 = 12
print(f"dz/dx = {x.grad.item():.4f} (手算: 2u * 2 = 2*3*2 = 12) | (manual: 12)")
# dz/dx = 12.0000


# =============================================================================
# 3. 分支计算图：一个变量分叉后汇合 | 3. Branching graph: variable forks and merges
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 真实网络中一个变量可能流经多个路径（如残差连接）。 | In real networks, one variable may flow through multiple paths (e.g., residual connections).
# c = a + b，a = x^2，b = x^3。dc/dx = da/dx + db/dx = 2x + 3x^2 = 4+12 = 16。 | c = a + b, a = x^2, b = x^3. dc/dx = da/dx + db/dx = 2x + 3x^2 = 4+12 = 16.
#
# 技术栈：torch.Tensor 的 backward()，分支梯度求和 | Tech stack: backward(), branch gradient summation

print("\n3. 分支计算图：c = x^2 + x^3 | 3. Branching graph: c = x^2 + x^3")
print("-" * 40)

x = torch.tensor([2.0], requires_grad=True)

a = x ** 2          # a = 4
b = x ** 3          # b = 8
c = a + b           # c = 12

print(f"x = {x.item():.4f}")
print(f"a = x^2 = {a.item():.4f}")          # a = 4.0000
print(f"b = x^3 = {b.item():.4f}")          # b = 8.0000
print(f"c = a + b = {c.item():.4f}")        # c = 12.0000

c.backward()
# dc/dx = da/dx + db/dx = 2x + 3x^2 = 4 + 12 = 16
print(f"dc/dx = {x.grad.item():.4f} (手算: 2x + 3x^2 = 4+12 = 16) | (manual: 16)")
# dc/dx = 16.0000


# =============================================================================
# 4. 同一变量多次使用（雅可比累加）| 4. Same variable used multiple times (Jacobian accumulation)
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# y = x * x * x 看似三个不同节点，实际是同一变量被「取出」三次。autograd 自动对每次使用的梯度累加。 | y = x * x * x looks like three different nodes, but it's the same variable "read" three times. Autograd accumulates gradients from each use.
# 结果等同于 y = x^3 的导数：dy/dx = 3x^2 = 12。 | Result equals derivative of y = x^3: dy/dx = 3x^2 = 12.
#
# 技术栈：torch.Tensor 的 backward()，雅可比累加 | Tech stack: backward(), Jacobian accumulation

print("\n4. 同一变量多次使用：y = x * x * x | 4. Same variable used multiple times: y = x * x * x")
print("-" * 40)

x = torch.tensor([2.0], requires_grad=True)

y = x * x * x       # y = 2 * 2 * 2 = 8

print(f"x = {x.item():.4f}")
print(f"y = x * x * x = {y.item():.4f}")    # y = 8.0000

y.backward()
print(f"dy/dx = {x.grad.item():.4f} (手算: 3x^2 = 12) | (manual: 3x^2 = 12)")
# dy/dx = 12.0000


# =============================================================================
# 5. 非标量输出的 backward（Jacobian 向量积）| 5. Non-scalar backward (Jacobian-vector product)
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 当输出是向量时，PyTorch 使用 Jacobian-vector product 而非显式计算整个雅可比矩阵（省内存）。 | When output is a vector, PyTorch uses Jacobian-vector product instead of explicitly building the full Jacobian matrix (saves memory).
# y = [x1^2, x2^2, x3^2]，∂y/∂x 是 3x3 对角矩阵，全1乘得 [2x1, 2x2, 2x3]。 | y = [x1^2, x2^2, x3^2], ∂y/∂x is a 3x3 diagonal matrix, all-ones gives [2x1, 2x2, 2x3].
#
# 技术栈：torch.Tensor 的 backward(gradient=...)，Jacobian-vector product | Tech stack: backward(gradient=...), Jacobian-vector product

print("\n5. 非标量输出的 backward | 5. Backward with non-scalar output")
print("-" * 40)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2          # y = [1, 4, 9]，每个分量独立于 x 的对应元素

print(f"x = {x}")                    # tensor([1., 2., 3.])
print(f"y = x^2 = {y}")             # tensor([1., 4., 9.])

# 全1 向量的含义：对每个输出分量赋予权重 1（等效于先求和再求导）
# All-ones meaning: weight each output component equally (equivalent to sum-then-differentiate)
y.backward(torch.ones_like(x))
print(f"dy/dx = {x.grad} (手算: 2x = [2, 4, 6]) | (manual: 2x = [2, 4, 6])")
# dy/dx = tensor([2., 4., 6.])


# =============================================================================
# 6. 高阶导数 | 6. Higher-order derivatives
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 部分优化方法（如 Newton 法）需要二阶导数。用 torch.autograd.grad + create_graph 实现。 | Some optimization methods (e.g., Newton's) need second-order derivatives. Use torch.autograd.grad + create_graph.
# f(x) = x^3，f'(x) = 3x^2 = 12，f''(x) = 6x = 12（x=2 时）。 | f(x) = x^3, f'(x) = 3x^2 = 12, f''(x) = 6x = 12 (at x=2).
#
# 技术栈：torch.autograd.grad(create_graph=True) | Tech stack: torch.autograd.grad(create_graph=True)

print("\n6. 高阶导数：f(x) = x^3 的一阶和二阶 | 6. Higher-order derivatives: 1st & 2nd of f(x) = x^3")
print("-" * 40)

x = torch.tensor([2.0], requires_grad=True)
y = x ** 3           # y = 8

# 一阶导数（create_graph=True 保留计算图用于二阶）| First derivative (create_graph=True to keep graph for 2nd order)
grad_1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"f(x) = x^3 = {y.item():.4f}")                           # f(x) = 8.0000
print(f"f'(x) = {grad_1.item():.4f} (手算: 3x^2 = 12)   | (manual: 3x^2 = 12)")

# 二阶导数 | Second derivative
grad_2 = torch.autograd.grad(grad_1, x)[0]
print(f"f''(x) = {grad_2.item():.4f} (手算: 6x = 12)    | (manual: 6x = 12)")
# f''(x) = 12.0000 (手算: 6x = 12)


# =============================================================================
# 7. 计算图的释放与 retain_graph | 7. Graph release and retain_graph
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# backward() 后计算图默认被销毁，释放内存。但有时需要多次 backward（如 GAN 训练），用 retain_graph=True 保留。 | After backward(), the graph is destroyed by default to free memory. But sometimes multiple backward() calls are needed (e.g., GAN training); use retain_graph=True.
#
# 技术栈：torch.Tensor 的 backward(retain_graph=True) | Tech stack: backward(retain_graph=True)

print("\n7. 计算图的释放与保留 | 7. Graph release and retention")
print("-" * 40)

# 7a. 默认行为：backward 后图被销毁 | 7a. Default: graph destroyed after backward
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
y.backward()
print("第一次 backward 成功 | 1st backward succeeded")          # OK

try:
    y.backward()  # 计算图已释放，无法再次 backward | Graph already released, can't backward again
    print("第二次 backward 成功 | 2nd backward succeeded")
except RuntimeError as e:
    print(f"第二次 backward 失败：计算图已释放 | 2nd backward failed: graph released")

# 7b. retain_graph=True：保留计算图 | 7b. retain_graph=True: keep graph alive
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)
print(f"\n使用 retain_graph=True: | Using retain_graph=True:")
print(f"第一次 backward: x.grad = {x.grad.item():.4f} | 1st backward: {x.grad.item():.4f}")

x.grad.zero_()  # 记得清零！| Don't forget to zero!
y.backward(retain_graph=True)
print(f"第二次 backward: x.grad = {x.grad.item():.4f} | 2nd backward: {x.grad.item():.4f}")
# 第二次 backward: x.grad = 2.0000

# 注意：长时间 retain 可能 OOM！实际训练很少需要 | Caution: long retain may cause OOM! Rarely needed in real training


# =============================================================================
# 8. 实战：小型神经网络的梯度流 | 8. Hands-on: gradient flow in a mini neural network
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 把所有概念串起来：参数声明 → 前向构建计算图 → 损失反向 → 查看每层梯度。 | Tie everything together: declare parameters → forward builds graph → loss backprop → inspect per-layer gradients.
# 这是 Day 3 nn.Module 的手写版，理解内部原理后使用高级 API 更有底气。 | This is the manual version of Day 3's nn.Module — understanding internals gives confidence when using higher-level APIs.
# 网络结构: W1[0.5], b1[0.1] → ReLU → W2[0.8], b2[0.2] → 预测 | Network: W1[0.5], b1[0.1] → ReLU → W2[0.8], b2[0.2] → predict
#
# 技术栈：torch.matmul, F.relu, backward() | Tech stack: torch.matmul, F.relu, backward()

print("\n8. 实战：神经网络的梯度流 | 8. Hands-on: gradient flow in a neural network")
print("-" * 40)

# 定义参数（需要梯度）| Define parameters (require gradients)
W1 = torch.tensor([[0.5]], requires_grad=True)
b1 = torch.tensor([0.1],  requires_grad=True)
W2 = torch.tensor([[0.8]], requires_grad=True)
b2 = torch.tensor([0.2],  requires_grad=True)

# 输入和标签（固定值，不需要梯度）| Input and label (fixed values, no gradients needed)
x      = torch.tensor([[1.0]])
y_true = torch.tensor([[2.0]])

# 前向传播 —— 构建计算图 | Forward pass — builds computation graph
h      = torch.matmul(x, W1.T) + b1    # 线性层 1 | Linear layer 1: out = x*W1^T + b1
h      = F.relu(h)                      # ReLU 激活 | ReLU activation
y_pred = torch.matmul(h, W2.T) + b2    # 输出层 | Output layer: out = h*W2^T + b2
loss   = (y_pred - y_true) ** 2        # MSE 损失 | MSE loss

print("前向传播结果：| Forward pass results:")
print(f"  W1={W1.item():.4f}, b1={b1.item():.4f}")
print(f"  隐藏层输出: h = relu(1*0.5+0.1) = {h.item():.4f} | Hidden: {h.item():.4f}")
print(f"  预测值: y_pred = {y_pred.item():.4f} | Prediction: {y_pred.item():.4f}")
print(f"  损失: loss = {loss.item():.4f} | Loss: {loss.item():.4f}")
# 隐藏层输出: h = 0.6000
# 预测值: y_pred = 0.6800
# 损失: loss = 1.7424

# 反向传播 —— 沿计算图传播梯度 | Backprop — propagate gradients along the computation graph
loss.backward()

print(f"\n每层梯度：| Per-layer gradients:")
print(f"  ∂L/∂W1 = {W1.grad.item():.4f}")
print(f"  ∂L/∂b1 = {b1.grad.item():.4f}")
print(f"  ∂L/∂W2 = {W2.grad.item():.4f}")
print(f"  ∂L/∂b2 = {b2.grad.item():.4f}")
# 所有梯度都是 autograd 自动计算的，无需手推链式法则


print("\n" + "=" * 55)
print("计算图学习完成！ | Computational graph learning complete!")
print("=" * 55)
print("\n核心要点： | Key takeaways:")
print("1. 计算图 = 操作节点 + 张量边，记录所有运算历史 | 1. Computation graph = operation nodes + tensor edges, recording all computation history")
print("2. backward() 沿计算图从输出到输入自动传播梯度（链式法则）| 2. backward() auto-propagates gradients along graph from output to input (chain rule)")
print("3. 分支图中同一变量的梯度自动求和（雅可比累加）| 3. In branching graphs, gradients of same variable auto-sum (Jacobian accumulation)")
print("4. 计算图默认在 backward 后销毁；retain_graph=True 可保留 | 4. Graph destroyed by default after backward; retain_graph=True keeps it alive")
print("5. torch.autograd.grad(create_graph=True) 保留图用于高阶导数 | 5. torch.autograd.grad(create_graph=True) keeps graph for higher-order derivatives")
print("6. 真实网络中，loss.backward() 自动算出所有参数的梯度，无需干预 | 6. In real networks, loss.backward() auto-computes all parameter gradients — no manual intervention needed")
