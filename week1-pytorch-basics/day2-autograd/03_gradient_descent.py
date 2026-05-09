"""
Day 2: 梯度下降实战
学习目标：使用autograd实现梯度下降优化
"""

import torch
import matplotlib.pyplot as plt

print("=" * 50)
print("梯度下降实战")
print("=" * 50)

# 1. 简单的梯度下降：找函数最小值
print("\n1. 找 f(x) = (x-3)^2 的最小值")
print("理论最小值: x = 3")

x = torch.tensor([0.0], requires_grad=True)  # 初始值
learning_rate = 0.1
steps = 50

x_history = []
f_history = []

for step in range(steps):
    # 前向计算
    f = (x - 3) ** 2
    
    # 记录历史
    x_history.append(x.item())
    f_history.append(f.item())
    
    # 反向传播
    f.backward()
    
    # 梯度下降更新（手动）
    with torch.no_grad():
        x -= learning_rate * x.grad
    
    # 清零梯度
    x.grad.zero_()
    
    if step % 10 == 0:
        print(f"Step {step}: x = {x.item():.4f}, f(x) = {f.item():.4f}")

print(f"\n最终结果: x = {x.item():.4f} (理论值: 3.0)")

# 2. 线性回归：拟合直线
print("\n2. 线性回归：y = 2x + 1 + noise")

# 生成训练数据
torch.manual_seed(42)
x_train = torch.linspace(0, 10, 50)
y_train = 2 * x_train + 1 + torch.randn(50) * 0.5

# 初始化参数
w = torch.tensor([0.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

learning_rate = 0.01
epochs = 100

losses = []

print("\n训练线性回归模型...")
for epoch in range(epochs):
    # 前向传播
    y_pred = w * x_train + b
    
    # 计算损失（MSE）
    loss = ((y_pred - y_train) ** 2).mean()
    losses.append(loss.item())
    
    # 反向传播
    loss.backward()
    
    # 梯度下降（手动更新）
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")

print(f"\n最终参数: w = {w.item():.4f}, b = {b.item():.4f}")
print(f"理论值: w = 2.0, b = 1.0")

# 3. 多元线性回归
print("\n3. 多元线性回归：y = w1*x1 + w2*x2 + b")

# 生成数据：y = 2*x1 + 3*x2 + 1
torch.manual_seed(42)
n_samples = 100
X = torch.randn(n_samples, 2)  # 100个样本，2个特征
y_true_params = torch.tensor([[2.0], [3.0]])  # 真实权重
b_true = 1.0

y = torch.matmul(X, y_true_params) + b_true + torch.randn(n_samples, 1) * 0.1

# 初始化参数
W = torch.randn(2, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

learning_rate = 0.01
epochs = 200

print("\n训练多元线性回归...")
for epoch in range(epochs):
    # 前向传播
    y_pred = torch.matmul(X, W) + b
    
    # 损失
    loss = ((y_pred - y) ** 2).mean()
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad
    
    # 清零梯度
    W.grad.zero_()
    b.grad.zero_()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")

print(f"\n学到的参数:")
print(f"W = {W.squeeze().detach().numpy()} (理论: [2.0, 3.0])")
print(f"b = {b.item():.4f} (理论: 1.0)")

# 4. 带动量的梯度下降
print("\n4. 动量梯度下降")
print("优化 f(x) = x^4 - 3x^3 + 2")

x = torch.tensor([3.0], requires_grad=True)
learning_rate = 0.01
momentum = 0.9
velocity = 0.0

x_history_momentum = []
f_history_momentum = []

for step in range(100):
    f = x**4 - 3*x**3 + 2
    
    x_history_momentum.append(x.item())
    f_history_momentum.append(f.item())
    
    f.backward()
    
    with torch.no_grad():
        # 动量更新
        velocity = momentum * velocity + learning_rate * x.grad
        x -= velocity
    
    x.grad.zero_()
    
    if step % 20 == 0:
        print(f"Step {step}: x = {x.item():.4f}, f(x) = {f.item():.4f}")

# 5. 小批量梯度下降
print("\n5. 小批量梯度下降")

# 生成大数据集
torch.manual_seed(42)
n_samples = 1000
X = torch.randn(n_samples, 5)
true_w = torch.randn(5, 1)
y = torch.matmul(X, true_w) + torch.randn(n_samples, 1) * 0.1

# 参数
W = torch.randn(5, 1, requires_grad=True)

learning_rate = 0.01
batch_size = 32
epochs = 50

print(f"\n数据集大小: {n_samples}, 批次大小: {batch_size}")

for epoch in range(epochs):
    # 打乱数据
    indices = torch.randperm(n_samples)
    
    epoch_loss = 0
    n_batches = 0
    
    # 小批量训练
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        # 前向传播
        y_pred = torch.matmul(X_batch, W)
        loss = ((y_pred - y_batch) ** 2).mean()
        
        epoch_loss += loss.item()
        n_batches += 1
        
        # 反向传播
        loss.backward()
        
        # 更新
        with torch.no_grad():
            W -= learning_rate * W.grad
        
        W.grad.zero_()
    
    if epoch % 10 == 0:
        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch}: avg loss = {avg_loss:.6f}")

print("\n" + "=" * 50)
print("梯度下降实战完成！")
print("=" * 50)
print("\n核心要点：")
print("1. 梯度下降：参数 -= 学习率 * 梯度")
print("2. 学习率控制更新步长")
print("3. 动量加速收敛")
print("4. 小批量梯度下降平衡速度和稳定性")
print("5. 记得每次backward前清零梯度")
