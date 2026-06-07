"""
Day 3: 神经网络基础 - 损失函数与优化器 | Day 3: Neural Network Basics - Loss Functions and Optimizers
学习目标：理解损失函数和优化器，完成一次完整的模型训练 | Learning objectives: Understand loss functions and optimizers, complete a full model training

技术栈： | Tech stack:
- PyTorch (torch)
- torch.nn（MSELoss, CrossEntropyLoss, Module） | torch.nn (MSELoss, CrossEntropyLoss, Module)
- torch.optim（SGD, Adam） | torch.optim (SGD, Adam)
- torch.nn.functional（可选） | torch.nn.functional (optional)
"""

import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 50)
print("损失函数与优化器 | Loss Functions and Optimizers")
print("=" * 50)


# =============================================================================
# 1. 损失函数：衡量预测有多差 | 1. Loss Functions: Measuring how bad the prediction is
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 训练需要数值化的「错误程度」；损失越小表示预测越接近真实值。 | Training needs a numerical 'error magnitude'; smaller loss means predictions are closer to true values.
#
# 技术栈：torch.nn.MSELoss, torch.nn.CrossEntropyLoss | Tech stack: torch.nn.MSELoss, torch.nn.CrossEntropyLoss

print("\n1. 损失函数基础 | 1. Loss Function Basics")
print("-" * 40)

# --- MSELoss：回归（预测连续数值）--- | --- MSELoss: Regression (predicting continuous values) ---
# 为什么要用 MSE？ | Why use MSE?
# 回归任务关心预测值与真实值的平方差，对大误差惩罚更重。 | Regression tasks care about the squared difference between predictions and true values, penalizing large errors more heavily.
mse_loss = nn.MSELoss()

y_pred = torch.tensor([2.5, 3.0, 4.5])
y_true = torch.tensor([2.0, 3.5, 4.0])

loss_mse = mse_loss(y_pred, y_true)
print(f"MSE 损失: {loss_mse.item():.4f} | MSE Loss: {loss_mse.item():.4f}")
print("MSE = mean((y_pred - y_true)^2)，用于回归 | MSE = mean((y_pred - y_true)^2), used for regression")

# --- CrossEntropyLoss：多分类 --- | --- CrossEntropyLoss: Multi-class classification ---
# 为什么要用 CrossEntropy？ | Why use CrossEntropy?
# 多分类需要衡量预测类别分布与真实类别的差距；内部含 softmax + 负对数似然。 | Multi-class classification needs to measure the gap between predicted class distribution and true classes; internally contains softmax + negative log-likelihood.
ce_loss = nn.CrossEntropyLoss()

logits = torch.tensor([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 0.3],
])
labels = torch.tensor([0, 1])

loss_ce = ce_loss(logits, labels)
print(f"\nCrossEntropy 损失: {loss_ce.item():.4f} | CrossEntropy Loss: {loss_ce.item():.4f}")
print("CrossEntropy 用于多分类；输入 logits（未 softmax），标签为类别整数 | CrossEntropy used for multi-class classification; input logits (pre-softmax), labels as class integers")


# =============================================================================
# 2. 优化器：根据梯度更新参数 | 2. Optimizers: Updating parameters based on gradients
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# Day 2 手写 w -= lr * w.grad；优化器封装这一步，并支持 Adam 等高级算法。 | Day 2 handwritten w -= lr * w.grad; optimizers encapsulate this step and support advanced algorithms like Adam.
#
# 技术栈：torch.optim.SGD, torch.optim.Adam | Tech stack: torch.optim.SGD, torch.optim.Adam

print("\n2. 优化器基础 | 2. Optimizer Basics")
print("-" * 40)

model = nn.Linear(3, 1)

# 为什么要传 model.parameters()？ | Why pass model.parameters()?
# 优化器需要知道更新哪些张量；parameters() 返回所有 requires_grad=True 的参数。 | The optimizer needs to know which tensors to update; parameters() returns all tensors with requires_grad=True.
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)

print("SGD：最基础，Day 2 手写梯度下降的封装 | SGD: Most basic, encapsulation of Day 2's handwritten gradient descent")
print("Adam：自适应学习率，多数场景默认首选 | Adam: Adaptive learning rate, default choice for most scenarios")
print(f"当前模型参数量: {sum(p.numel() for p in model.parameters())} | Current model parameter count: {sum(p.numel() for p in model.parameters())}")


# =============================================================================
# 3. 标准训练四步循环 | 3. Standard Training Loop (Four Steps)
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 这是 PyTorch 训练的核心模板，Day 4 及 MNIST 项目都会重复使用。 | This is the core template for PyTorch training, reused in Day 4 and the MNIST project.
#
# 技术栈：torch, torch.nn, torch.optim, autograd（backward） | Tech stack: torch, torch.nn, torch.optim, autograd (backward)

print("\n3. 标准训练四步循环 | 3. Standard Training Loop (Four Steps)")
print("-" * 40)

print("""
for epoch in range(num_epochs):
    optimizer.zero_grad()   # 步骤1：清零梯度（防止累积） | Step 1: Zero gradients (prevent accumulation)
    y_pred = model(x)       # 步骤2：前向传播 | Step 2: Forward pass
    loss = criterion(y_pred, y_true)  # 步骤3：计算损失 | Step 3: Compute loss
    loss.backward()         # 步骤4：反向传播（autograd） | Step 4: Backward pass (autograd)
    optimizer.step()        # 步骤5：更新参数 | Step 5: Update parameters
""")


# =============================================================================
# 4. 实战：线性回归（MSE + SGD） | 4. Practice: Linear Regression (MSE + SGD)
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 与 Day 2 梯度下降对比：同样任务，用 nn + MSELoss + SGD 更简洁。 | Compare with Day 2 gradient descent: same task, cleaner with nn + MSELoss + SGD.
#
# 技术栈：torch.nn.Linear, MSELoss, optim.SGD, autograd | Tech stack: torch.nn.Linear, MSELoss, optim.SGD, autograd

print("\n4. 实战：线性回归（MSE + SGD） | 4. Practice: Linear Regression (MSE + SGD)")
print("-" * 40)

torch.manual_seed(42)

x_train = torch.linspace(0, 10, 50).unsqueeze(1)
y_train = 2 * x_train + 1 + torch.randn(50, 1) * 0.5

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 100
print("训练线性回归... | Training linear regression...")
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

print(f"\n最终: w = {model.weight.item():.4f}, b = {model.bias.item():.4f} | Final: w = {model.weight.item():.4f}, b = {model.bias.item():.4f}")
print("理论值: w ~ 2.0, b ~ 1.0 | Theoretical values: w ~ 2.0, b ~ 1.0")


# =============================================================================
# 5. 实战：多分类（CrossEntropy + Adam） | 5. Practice: Multi-class Classification (CrossEntropy + Adam)
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 分类是深度学习最常见任务；演示 logits 输出 + CrossEntropy + Adam 的组合。 | Classification is the most common deep learning task; demonstrates the logits output + CrossEntropy + Adam combination.
#
# 技术栈：torch.nn.Sequential, Linear, ReLU, CrossEntropyLoss, optim.Adam | Tech stack: torch.nn.Sequential, Linear, ReLU, CrossEntropyLoss, optim.Adam

print("\n5. 实战：多分类（CrossEntropy + Adam） | 5. Practice: Multi-class Classification (CrossEntropy + Adam)")
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
print("训练分类器... | Training classifier...")
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
print(f"\n最终准确率: {final_acc:.2%} | Final accuracy: {final_acc:.2%}")


# =============================================================================
# 6. SGD vs Adam 对比 | 6. SGD vs Adam Comparison
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 同一任务换优化器，观察收敛差异，便于日后调参。 | Swap optimizers on the same task, observe convergence differences for easier future tuning.
#
# 技术栈：optim.SGD, optim.Adam, MSELoss | Tech stack: optim.SGD, optim.Adam, MSELoss

print("\n6. SGD vs Adam 对比 | 6. SGD vs Adam Comparison")
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

print(f"SGD  最终 loss: {results['SGD']:.4f} | SGD  final loss: {results['SGD']:.4f}")
print(f"Adam 最终 loss: {results['Adam']:.4f} | Adam final loss: {results['Adam']:.4f}")
print("Adam 通常收敛更快；SGD + 调 lr 有时泛化更好 | Adam usually converges faster; SGD + tuned lr sometimes generalizes better")


# =============================================================================
# 7. 完整训练模板（回归） | 7. Complete Training Template (Regression)
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 汇总 Day 3 所有组件，形成可复用的训练模板。 | Combine all Day 3 components into a reusable training template.
#
# 技术栈：torch.nn.Module, MSELoss, optim.Adam, autograd | Tech stack: torch.nn.Module, MSELoss, optim.Adam, autograd

print("\n7. 完整训练模板 | 7. Complete Training Template")
print("-" * 40)


class RegressionModel(nn.Module):
    """小型回归网络。技术栈：nn.Module, Linear, ReLU | Small regression network. Tech stack: nn.Module, Linear, ReLU"""

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
    可复用的回归训练函数。 | Reusable regression training function.

    技术栈：MSELoss, Adam, autograd（zero_grad, backward, step） | Tech stack: MSELoss, Adam, autograd (zero_grad, backward, step)
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
print("拟合 y ~ x^2（非线性回归）: | Fitting y ~ x^2 (nonlinear regression):")
final_loss = train_regression(model, x, y, epochs=100, lr=0.05)
print(f"最终 loss: {final_loss:.6f} | Final loss: {final_loss:.6f}")


# =============================================================================
# 8. 练习：观察 loss 是否下降 | 8. Exercise: Observe whether loss decreases
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# loss 下降是训练有效的基本信号；养成每个 epoch 或定期打印 loss 的习惯。 | Loss decreasing is the basic signal of effective training; develop the habit of printing loss every epoch or periodically.
#
# 技术栈：nn.Linear, MSELoss, optim.SGD | Tech stack: nn.Linear, MSELoss, optim.SGD

print("\n8. 练习：监控 loss 变化 | 8. Exercise: Monitor loss changes")
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

print(f"初始 loss: {losses[0]:.4f} | Initial loss: {losses[0]:.4f}")
print(f"最终 loss: {losses[-1]:.4f} | Final loss: {losses[-1]:.4f}")
print(f"loss 下降: {losses[0] - losses[-1]:.4f} | Loss decrease: {losses[0] - losses[-1]:.4f}")
print("loss 持续下降 -> 训练正常；不变或上升 -> 检查 lr、数据或模型 | Loss consistently decreasing -> training normal; flat or increasing -> check lr, data, or model")


print("\n" + "=" * 50)
print("损失函数与优化器学习完成！ | Loss Functions and Optimizers learning complete!")
print("=" * 50)
print("\n核心要点： | Key takeaways:")
print("1. MSELoss：回归；CrossEntropyLoss：多分类 | 1. MSELoss: regression; CrossEntropyLoss: multi-class classification")
print("2. 优化器通过 model.parameters() 更新所有参数 | 2. Optimizers update all parameters via model.parameters()")
print("3. 训练循环：zero_grad -> forward -> loss -> backward -> step | 3. Training loop: zero_grad -> forward -> loss -> backward -> step")
print("4. Adam 常用默认；SGD 需更多调 lr | 4. Adam is the common default; SGD needs more lr tuning")
print("5. 始终监控 loss 是否下降 | 5. Always monitor whether loss is decreasing")
