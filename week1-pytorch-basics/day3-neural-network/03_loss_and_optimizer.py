"""
Day 3: 神经网络基础 - 损失函数与优化器
学习目标：理解损失函数和优化器，完成一次完整的模型训练

技术栈：
- PyTorch (torch)
- torch.nn（MSELoss, CrossEntropyLoss, Module）
- torch.optim（SGD, Adam）
- torch.nn.functional（可选）
"""

import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 50)
print("损失函数与优化器")
print("=" * 50)


# =============================================================================
# 1. 损失函数：衡量预测有多差
# =============================================================================
# 为什么要写这段代码？
# 训练需要数值化的「错误程度」；损失越小表示预测越接近真实值。
#
# 技术栈：torch.nn.MSELoss, torch.nn.CrossEntropyLoss

print("\n1. 损失函数基础")
print("-" * 40)

# --- MSELoss：回归（预测连续数值）---
# 为什么要用 MSE？
# 回归任务关心预测值与真实值的平方差，对大误差惩罚更重。
mse_loss = nn.MSELoss()

y_pred = torch.tensor([2.5, 3.0, 4.5])
y_true = torch.tensor([2.0, 3.5, 4.0])

loss_mse = mse_loss(y_pred, y_true)
print(f"MSE 损失: {loss_mse.item():.4f}")
print("MSE = mean((y_pred - y_true)^2)，用于回归")

# --- CrossEntropyLoss：多分类 ---
# 为什么要用 CrossEntropy？
# 多分类需要衡量预测类别分布与真实类别的差距；内部含 softmax + 负对数似然。
ce_loss = nn.CrossEntropyLoss()

logits = torch.tensor([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 0.3],
])
labels = torch.tensor([0, 1])

loss_ce = ce_loss(logits, labels)
print(f"\nCrossEntropy 损失: {loss_ce.item():.4f}")
print("CrossEntropy 用于多分类；输入 logits（未 softmax），标签为类别整数")


# =============================================================================
# 2. 优化器：根据梯度更新参数
# =============================================================================
# 为什么要写这段代码？
# Day 2 手写 w -= lr * w.grad；优化器封装这一步，并支持 Adam 等高级算法。
#
# 技术栈：torch.optim.SGD, torch.optim.Adam

print("\n2. 优化器基础")
print("-" * 40)

model = nn.Linear(3, 1)

# 为什么要传 model.parameters()？
# 优化器需要知道更新哪些张量；parameters() 返回所有 requires_grad=True 的参数。
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)

print("SGD：最基础，Day 2 手写梯度下降的封装")
print("Adam：自适应学习率，多数场景默认首选")
print(f"当前模型参数量: {sum(p.numel() for p in model.parameters())}")


# =============================================================================
# 3. 标准训练四步循环
# =============================================================================
# 为什么要写这段代码？
# 这是 PyTorch 训练的核心模板，Day 4 及 MNIST 项目都会重复使用。
#
# 技术栈：torch, torch.nn, torch.optim, autograd（backward）

print("\n3. 标准训练四步循环")
print("-" * 40)

print("""
for epoch in range(num_epochs):
    optimizer.zero_grad()   # 步骤1：清零梯度（防止累积）
    y_pred = model(x)       # 步骤2：前向传播
    loss = criterion(y_pred, y_true)  # 步骤3：计算损失
    loss.backward()         # 步骤4：反向传播（autograd）
    optimizer.step()        # 步骤5：更新参数
""")


# =============================================================================
# 4. 实战：线性回归（MSE + SGD）
# =============================================================================
# 为什么要写这段代码？
# 与 Day 2 梯度下降对比：同样任务，用 nn + MSELoss + SGD 更简洁。
#
# 技术栈：torch.nn.Linear, MSELoss, optim.SGD, autograd

print("\n4. 实战：线性回归（MSE + SGD）")
print("-" * 40)

torch.manual_seed(42)

x_train = torch.linspace(0, 10, 50).unsqueeze(1)
y_train = 2 * x_train + 1 + torch.randn(50, 1) * 0.5

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 100
print("训练线性回归...")
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        w = model.weight.item()
        b = model.bias.item()
        print(f"  Epoch {epoch:3d}: loss = {loss.item():.4f}, w = {w:.4f}, b = {b:.4f}")

print(f"\n最终: w = {model.weight.item():.4f}, b = {model.bias.item():.4f}")
print("理论值: w ~ 2.0, b ~ 1.0")


# =============================================================================
# 5. 实战：多分类（CrossEntropy + Adam）
# =============================================================================
# 为什么要写这段代码？
# 分类是深度学习最常见任务；演示 logits 输出 + CrossEntropy + Adam 的组合。
#
# 技术栈：torch.nn.Sequential, Linear, ReLU, CrossEntropyLoss, optim.Adam

print("\n5. 实战：多分类（CrossEntropy + Adam）")
print("-" * 40)

torch.manual_seed(42)

n_samples = 200
n_features = 10
n_classes = 3

X = torch.randn(n_samples, n_features)
true_W = torch.randn(n_features, n_classes)
logits_true = X @ true_W
y = logits_true.argmax(dim=1)

classifier = nn.Sequential(
    nn.Linear(n_features, 32),
    nn.ReLU(),
    nn.Linear(32, n_classes),
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.01)

epochs = 50
print("训练分类器...")
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = classifier(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            acc = (preds == y).float().mean().item()
        print(f"  Epoch {epoch:3d}: loss = {loss.item():.4f}, acc = {acc:.2%}")

with torch.no_grad():
    final_preds = classifier(X).argmax(dim=1)
    final_acc = (final_preds == y).float().mean().item()
print(f"\n最终准确率: {final_acc:.2%}")


# =============================================================================
# 6. SGD vs Adam 对比
# =============================================================================
# 为什么要写这段代码？
# 同一任务换优化器，观察收敛差异，便于日后调参。
#
# 技术栈：optim.SGD, optim.Adam, MSELoss

print("\n6. SGD vs Adam 对比")
print("-" * 40)

torch.manual_seed(42)
x_data = torch.randn(100, 5)
y_data = torch.randn(100, 1)

results = {}
for opt_name, opt_class, lr in [("SGD", optim.SGD, 0.01), ("Adam", optim.Adam, 0.01)]:
    model = nn.Linear(5, 1)
    criterion = nn.MSELoss()
    optimizer = opt_class(model.parameters(), lr=lr)

    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(x_data), y_data)
        loss.backward()
        optimizer.step()

    results[opt_name] = loss.item()

print(f"SGD  最终 loss: {results['SGD']:.4f}")
print(f"Adam 最终 loss: {results['Adam']:.4f}")
print("Adam 通常收敛更快；SGD + 调 lr 有时泛化更好")


# =============================================================================
# 7. 完整训练模板（回归）
# =============================================================================
# 为什么要写这段代码？
# 汇总 Day 3 所有组件，形成可复用的训练模板。
#
# 技术栈：torch.nn.Module, MSELoss, optim.Adam, autograd

print("\n7. 完整训练模板")
print("-" * 40)


class RegressionModel(nn.Module):
    """小型回归网络。技术栈：nn.Module, Linear, ReLU"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_regression(model, x, y, epochs=100, lr=0.01):
    """
    可复用的回归训练函数。

    技术栈：MSELoss, Adam, autograd（zero_grad, backward, step）
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0:
            print(f"  Epoch {epoch:3d}: loss = {loss.item():.6f}")

    return loss.item()


torch.manual_seed(42)
x = torch.linspace(-3, 3, 100).unsqueeze(1)
y = x ** 2 + torch.randn(100, 1) * 0.3

model = RegressionModel()
print("拟合 y ~ x^2（非线性回归）:")
final_loss = train_regression(model, x, y, epochs=100, lr=0.05)
print(f"最终 loss: {final_loss:.6f}")


# =============================================================================
# 8. 练习：观察 loss 是否下降
# =============================================================================
# 为什么要写这段代码？
# loss 下降是训练有效的基本信号；养成每个 epoch 或定期打印 loss 的习惯。
#
# 技术栈：nn.Linear, MSELoss, optim.SGD

print("\n8. 练习：监控 loss 变化")
print("-" * 40)

torch.manual_seed(0)
x = torch.randn(50, 1)
y = 3 * x + 2 + torch.randn(50, 1) * 0.1

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

losses = []
for epoch in range(30):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print(f"初始 loss: {losses[0]:.4f}")
print(f"最终 loss: {losses[-1]:.4f}")
print(f"loss 下降: {losses[0] - losses[-1]:.4f}")
print("loss 持续下降 -> 训练正常；不变或上升 -> 检查 lr、数据或模型")


print("\n" + "=" * 50)
print("损失函数与优化器学习完成！")
print("=" * 50)
print("\n核心要点：")
print("1. MSELoss：回归；CrossEntropyLoss：多分类")
print("2. 优化器通过 model.parameters() 更新所有参数")
print("3. 训练循环：zero_grad -> forward -> loss -> backward -> step")
print("4. Adam 常用默认；SGD 需更多调 lr")
print("5. 始终监控 loss 是否下降")
