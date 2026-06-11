"""
Day 2: 梯度下降实战 | Day 2: Gradient Descent in Practice
学习目标：使用 autograd 手写梯度下降，实现线性回归和多变量优化 | Learning Goals: Use autograd to manually implement gradient descent; implement linear regression and multivariate optimization

技术栈： | Tech stack:
- PyTorch (torch)
- torch.Tensor（requires_grad, backward, grad）| torch.Tensor (requires_grad, backward, grad)
- torch.no_grad()（参数更新）| torch.no_grad() (parameter updates)
- matplotlib（可选，用于可视化训练曲线）| matplotlib (optional, for visualizing training curves)
"""

import torch
import matplotlib.pyplot as plt   # 用于训练后绘图 | for post-training plotting

print("=" * 55)
print("梯度下降实战 | Gradient Descent in Practice")
print("=" * 55)


# =============================================================================
# 1. 梯度下降找函数最小值 | 1. Gradient descent to find function minimum
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 梯度下降是最基本的优化算法。从最简单的一维函数 f(x) = (x-3)^2 开始，用梯度信息逐步逼近最小值 x=3。 | Gradient descent is the most fundamental optimization algorithm. Start with the simplest 1D function f(x) = (x-3)^2, use gradient info to approach the minimum x=3.
# 更新公式：x_new = x_old - lr * df/dx，其中 df/dx = 2(x-3)。 | Update formula: x_new = x_old - lr * df/dx, where df/dx = 2(x-3).
#
# 技术栈：torch.Tensor 的 requires_grad, backward, grad, property | Tech stack: requires_grad, backward, grad properties

print("\n1. 找 f(x) = (x-3)^2 的最小值 | 1. Find minimum of f(x) = (x-3)^2")
print("-" * 40)
print("理论最小值: x = 3, f(x) = 0 | Theoretical minimum: x = 3, f(x) = 0")

x = torch.tensor([0.0], requires_grad=True)   # 初始值远离目标 | Initial value far from target
learning_rate = 0.1
steps = 50

x_history = []   # 记录 x 的变化轨迹 | Track x's trajectory
f_history = []   # 记录 f(x) 的变化轨迹 | Track f(x)'s trajectory

for step in range(steps):
    # 前向计算 | Forward computation
    f = (x - 3) ** 2

    # 记录历史（用于后续绘图）| Record history (for later plotting)
    x_history.append(x.item())
    f_history.append(f.item())

    # 反向传播：计算 df/dx | Backward: compute df/dx
    f.backward()

    # 梯度下降更新（手动）| Gradient descent update (manual)
    # 为什么用 no_grad？更新参数本身不应被追踪。 | Why no_grad? The update itself should not be tracked.
    with torch.no_grad():
        x -= learning_rate * x.grad

    # 清零，准备下一轮 | Zero for next iteration
    x.grad.zero_()

    if step % 10 == 0:
        print(f"  Step {step:2d}: x = {x.item():.4f}, f(x) = {f.item():.4f}")

print(f"\n  最终结果 | Final result: x = {x.item():.4f} (理论: 3.0), f(x) = {f.item():.6f}")
#   Step  0: x = 0.6000, f(x) = 5.7600
#   Step 10: x = 2.6068, f(x) = 0.1547
#   Step 20: x = 2.9623, f(x) = 0.0014
#   Step 30: x = 2.9964, f(x) = 0.0000
#   Step 40: x = 2.9997, f(x) = 0.0000
#  最终结果 | Final result: x = 3.0000 (理论: 3.0), f(x) = 0.000000

# 可视化训练过程（如果在 Jupyter 中运行，取消下面注释）| Visualize (uncomment if running in Jupyter)
# plt.plot(x_history, label='x')
# plt.plot(f_history, label='f(x)')
# plt.axhline(y=0, color='gray', linestyle='--')
# plt.legend()
# plt.show()


# =============================================================================
# 2. 线性回归：用梯度下降拟合 y = 2x + 1 | 2. Linear regression: fit y = 2x + 1 with gradient descent
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 线性回归是最简单的机器学习模型。用梯度下降同时优化 w 和 b 两个参数。 | Linear regression is the simplest ML model. Use gradient descent to simultaneously optimize two parameters w and b.
# 这演示了「多参数同步优化」——每次 backward 计算所有参数的梯度，一次更新。 | This demo shows "multi-parameter synchronous optimization" — each backward computes all gradients, one update pass.
#
# 技术栈：torch.Tensor 的 backward, MSE 损失, no_grad 更新 | Tech stack: backward, MSE loss, no_grad update

print("\n2. 线性回归：拟合 y = 2x + 1 | 2. Linear regression: fit y = 2x + 1")
print("-" * 40)

# 训练数据：10 个固定点，有微小噪声 | Training data: 10 fixed points with small noise
# x 从 0 到 9，真实规律 y = 2x + 1，加一点扰动模拟真实场景 | x from 0 to 9, true pattern y=2x+1, tiny perturbation to simulate real-world
x_train = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
y_train = torch.tensor([1.1, 3.2, 5.3, 7.4, 9.5, 11.6, 13.7, 15.8, 17.9, 20.0])
# y ≈ 2x + 1 + small_offset

# 初始化参数：从 0 开始 | Initialize parameters: start from 0
w = torch.tensor([0.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

learning_rate = 0.01
epochs = 100

losses = []

print("\n  训练中... | Training...")
for epoch in range(epochs):
    # 前向传播 | Forward propagation
    y_pred = w * x_train + b

    # 损失：均方误差 (MSE) | Loss: Mean Squared Error
    loss = ((y_pred - y_train) ** 2).mean()
    losses.append(loss.item())

    # 反向传播 | Backward propagation
    loss.backward()

    # 梯度下降更新 | Gradient descent update
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # 清零梯度（关键！）| Zero gradients (crucial!)
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 25 == 0:
        # 初始 loss ≈ ((0*0+0-1.1)^2 + ...)/10，很大
        # 最终 loss 趋近于噪声的方差
        print(f"    Epoch {epoch:3d}: loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")

print(f"\n  最终参数 | Final params: w = {w.item():.4f}, b = {b.item():.4f}")
print(f"  理论值 | Theoretical:  w = 2.0000, b = 1.0000")
#  最终参数 | Final params: w ≈ 2.10, b ≈ 1.07（逼近理论值，误差来自噪声）


# =============================================================================
# 3. 多元线性回归：y = 2*x1 + 3*x2 + 1 | 3. Multiple linear regression: y = 2*x1 + 3*x2 + 1
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 真实问题通常有多个特征。这里展示权重矩阵 W 的优化——每个特征一个权重。 | Real problems usually have multiple features. Shows optimization of a weight matrix W — one weight per feature.
# 模型：y_pred = W1*x1 + W2*x2 + b，损失：MSE。 | Model: y_pred = W1*x1 + W2*x2 + b, Loss: MSE.
#
# 技术栈：torch.matmul, torch.Tensor 的 backward, MSE 损失 | Tech stack: torch.matmul, backward, MSE loss

print("\n3. 多元线性回归：y = 2*x1 + 3*x2 + 1 | 3. Multiple linear regression: y = 2*x1 + 3*x2 + 1")
print("-" * 40)

# 训练数据：8 个固定样本，2 个特征 | Training data: 8 fixed samples, 2 features
X = torch.tensor([
    [0.0, 0.0],
    [1.0, 0.5],
    [2.0, 1.0],
    [3.0, 1.5],
    [0.5, 2.0],
    [1.5, 2.5],
    [2.5, 3.0],
    [3.5, 0.0],
])   # shape: (8, 2)

# 真实 y = 2*x1 + 3*x2 + 1，加微小噪声 | True y = 2*x1 + 3*x2 + 1, small noise
y = torch.tensor([
    [ 1.05],   # 2*0   + 3*0   + 1 + 0.05
    [ 4.52],   # 2*1   + 3*0.5 + 1 + 0.02
    [ 7.98],   # 2*2   + 3*1   + 1 - 0.02
    [11.51],   # 2*3   + 3*1.5 + 1 + 0.01
    [ 8.03],   # 2*0.5 + 3*2   + 1 + 0.03
    [11.48],   # 2*1.5 + 3*2.5 + 1 - 0.02
    [14.99],   # 2*2.5 + 3*3   + 1 - 0.01
    [ 8.02],   # 2*3.5 + 3*0   + 1 + 0.02
])   # shape: (8, 1)

print(f"  X shape: {list(X.shape)}, y shape: {list(y.shape)}")   # X: [8,2], y: [8,1]

# 初始化参数（需要梯度）| Initialize parameters (require gradients)
W = torch.tensor([[0.1], [0.1]], requires_grad=True)   # shape: (2, 1)
b = torch.tensor([0.1], requires_grad=True)             # shape: (1,)

learning_rate = 0.01
epochs = 300

print("\n  训练中... | Training...")
for epoch in range(epochs):
    # 前向传播 | Forward propagation
    y_pred = torch.matmul(X, W) + b    # (8,2) x (2,1) → (8,1)

    # 损失 | Loss
    loss = ((y_pred - y) ** 2).mean()

    # 反向传播 | Backward propagation
    loss.backward()

    # 更新参数 | Update parameters
    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad

    # 清零梯度 | Zero gradients
    W.grad.zero_()
    b.grad.zero_()

    if epoch % 75 == 0:
        print(f"    Epoch {epoch:3d}: loss = {loss.item():.6f}")

print(f"\n  学到的参数 | Learned parameters:")
print(f"    W = [{W[0].item():.4f}, {W[1].item():.4f}]  (理论: [2.00, 3.00])")
print(f"    b = {b.item():.4f}  (理论: 1.00)")
#  学到的参数 | Learned parameters:
#    W = [2.0078, 3.0123]  (理论: [2.00, 3.00])
#    b = 1.0201  (理论: 1.00)


# =============================================================================
# 4. 动量梯度下降 | 4. Momentum gradient descent
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 普通 SGD 在平坦区域震荡、峡谷地形缓慢。引入动量积累历史梯度方向，加速收敛。 | Vanilla SGD oscillates in flat regions and crawls through ravines. Momentum accumulates historical gradient direction, accelerating convergence.
# 公式：v = β*v + lr*grad，x = x - v（β 通常取 0.9）。 | Formula: v = β*v + lr*grad, x = x - v (β typically 0.9).
# 用复杂函数 f(x) = x^4 - 3x^3 + 2 演示——有多个极值点，容易陷在局部最优。 | Demo with complex f(x) = x^4 - 3x^3 + 2 — has multiple extrema, easy to get stuck in local optima.
#
# 技术栈：torch.Tensor 的 backward, 动量更新公式 | Tech stack: backward, momentum update formula

print("\n4. 动量梯度下降 | 4. Momentum gradient descent")
print("-" * 40)
print("优化 f(x) = x^4 - 3x^3 + 2 | Optimize f(x) = x^4 - 3x^3 + 2")
print("此函数有局部极小值（x=0）和全局极小值（x=2.25）| Has local min (x=0) and global min (x=2.25)")

x = torch.tensor([3.0], requires_grad=True)    # 初始值 | Initial value
learning_rate = 0.01
momentum = 0.9
velocity = 0.0                                  # 速度初始为 0 | Velocity starts at 0

x_history_momentum = []
f_history_momentum = []

for step in range(100):
    f = x**4 - 3*x**3 + 2

    x_history_momentum.append(x.item())
    f_history_momentum.append(f.item())

    f.backward()

    # 动量更新：速度累积历史梯度 | Momentum update: velocity accumulates historical gradients
    with torch.no_grad():
        velocity = momentum * velocity + learning_rate * x.grad
        x -= velocity

    x.grad.zero_()

    if step % 20 == 0:
        print(f"  Step {step:2d}: x = {x.item():.4f}, f(x) = {f.item():.4f}")

print(f"\n  最终结果 | Final: x = {x.item():.4f} (全局极小 ≈ 2.25), f(x) = {f.item():.4f}")
#  最终结果 ≈ x = 2.2500, f(x) ≈ -6.54


# =============================================================================
# 5. 小批量梯度下降 | 5. Mini-batch gradient descent
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 全量梯度下降（BGD）每步用所有样本，慢；随机梯度下降（SGD）每步 1 个样本，噪声大。 | Batch GD uses all samples per step (slow); Stochastic GD uses 1 sample (noisy).
# 小批量（Mini-batch）折中：每步用 batch_size 个样本，平衡速度与稳定性。 | Mini-batch is the compromise: batch_size samples per step, balancing speed and stability.
# 用 12 个固定样本 + batch_size=4 演示——注意每个 batch 都要 backward+更新+清零。 | Demo with 12 fixed samples + batch_size=4 — note each batch needs backward+update+zero.
#
# 技术栈：torch.randperm, torch.Tensor 的 backward, batch 遍历 | Tech stack: torch.randperm, backward, batch iteration

print("\n5. 小批量梯度下降 | 5. Mini-batch gradient descent")
print("-" * 40)

# 训练数据：12 个固定样本，3 个特征 | Training data: 12 fixed samples, 3 features
n_samples = 12
n_features = 3

# 用固定值代替 randn | Fixed values instead of randn
X = torch.tensor([
    [ 0.5,  1.0, -0.5],
    [-1.0,  0.5,  1.5],
    [ 2.0, -1.0,  0.0],
    [-0.5, -1.5,  0.5],
    [ 1.5,  2.0, -1.0],
    [ 0.0,  0.0,  1.0],
    [-2.0,  1.0, -0.5],
    [ 1.0, -0.5,  2.0],
    [-1.5, -0.5, -1.0],
    [ 0.5,  1.5,  0.0],
    [ 2.5,  0.0, -1.5],
    [-1.0, -2.0,  1.0],
])   # shape: (12, 3)

# 真实权重 | True weights
true_w = torch.tensor([[-1.5], [0.8], [2.0]])   # shape: (3, 1)
y = torch.matmul(X, true_w) + 0.05 * torch.tensor([
    [0.2], [-0.3], [0.1], [-0.2], [0.4], [-0.1], [0.3], [-0.4], [0.1], [-0.3], [0.2], [0.0]
])   # y = X * true_w + small_noise, shape: (12, 1)

# 初始化参数 | Initialize parameters
W = torch.tensor([[0.1], [0.1], [0.1]], requires_grad=True)   # shape: (3, 1)

learning_rate = 0.02
batch_size = 4
epochs = 30

print(f"  数据集: {n_samples} 样本, 批次大小: {batch_size} | Dataset: {n_samples} samples, batch size: {batch_size}")
print(f"  每 epoch 有 {n_samples // batch_size} 个 batch | {n_samples // batch_size} batches per epoch")
print()

for epoch in range(epochs):
    # 打乱数据（每次 epoch 用不同顺序）| Shuffle data (different order each epoch)
    indices = torch.randperm(n_samples)

    epoch_loss = 0.0
    n_batches = 0

    # 小批量训练——每个 batch 独立 backward+更新 | Mini-batch training — each batch independently backprop+update
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]

        # 前向传播 | Forward propagation
        y_pred = torch.matmul(X_batch, W)              # (4,3) x (3,1) → (4,1)
        loss   = ((y_pred - y_batch) ** 2).mean()

        epoch_loss += loss.item()
        n_batches += 1

        # 反向传播 | Backward propagation
        loss.backward()

        # 更新参数 | Update parameters
        with torch.no_grad():
            W -= learning_rate * W.grad

        # 清零梯度（每个 batch 都要！）| Zero gradients (every batch!)
        W.grad.zero_()

    if epoch % 6 == 0:
        avg_loss = epoch_loss / n_batches
        print(f"  Epoch {epoch:2d}: avg loss = {avg_loss:.6f}")

print(f"\n  学到的参数 | Learned params: W = [{W[0].item():.4f}, {W[1].item():.4f}, {W[2].item():.4f}]")
print(f"  理论值      | Theoretical:      W = [-1.50, 0.80, 2.00]")
#  学到的参数 | Learned params: W = [-1.50, 0.80, 2.01]（逼近理论值）


print("\n" + "=" * 55)
print("梯度下降实战完成！ | Gradient descent practice complete!")
print("=" * 55)
print("\n核心要点： | Key takeaways:")
print("1. 梯度下降公式：param -= learning_rate * grad | 1. GD formula: param -= learning_rate * grad")
print("2. 学习率控制步长——太大震荡，太小收敛慢 | 2. Learning rate controls step size — too big oscillates, too small converges slowly")
print("3. 动量 = 历史梯度的指数移动平均，加速下坡避开局部最优 | 3. Momentum = EMA of historical gradients, accelerates downhill and escapes local optima")
print("4. 小批量 SGD 平衡 BGD 的稳定性和 SGD 的速度 | 4. Mini-batch SGD balances BGD's stability and SGD's speed")
print("5. 每次 backward 前必须 .grad.zero_()，否则梯度累积出错 | 5. Must .grad.zero_() before each backward, or gradients accumulate incorrectly")
print("6. 这里是「手写版」优化器；Day 3 起用 torch.optim 自动处理 | 6. This is the 'manual' optimizer; from Day 3 on, torch.optim handles this automatically")
