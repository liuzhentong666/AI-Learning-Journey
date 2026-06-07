"""
Day 4: 训练循环与 DataLoader - 完整训练循环 | Day 4: Training Loop & DataLoader - Complete Training Loop
学习目标：理解 epoch 和 batch 的概念，掌握完整训练循环的双层结构 | Learning Objectives: Understand epoch and batch, master the two-level training loop structure

技术栈： | Tech stack:
- PyTorch (torch)
- torch.nn（Module, Linear, ReLU, MSELoss） | torch.nn (Module, Linear, ReLU, MSELoss)
- torch.optim（Adam） | torch.optim (Adam)
- torch.utils.data（TensorDataset, DataLoader） | torch.utils.data (TensorDataset, DataLoader)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

print("=" * 55)
print("Day 4: 完整训练循环 | Day 4: Complete Training Loop")
print("=" * 55)


# =============================================================================
# 1. 手动构造线性回归数据 | 1. Manually construct linear regression data
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 先用已知规律的假数据验证训练循环是否正确。 | Use fake data with known pattern to verify the training loop is correct.
# 真实规律：y = 2*x1 + 3*x2 - x3 + 1 + 噪声 | True rule: y = 2*x1 + 3*x2 - x3 + 1 + noise
#
# 技术栈：torch.tensor（固定值构造） | Tech stack: torch.tensor (fixed-value construction)

print("\n1. 构造数据集 | 1. Construct Dataset")
print("-" * 40)

# 为什么用固定值而非 randn？ | Why use fixed values instead of randn?
# 固定值让输出可预期，便于验证程序逻辑。 | Fixed values make output predictable, easy to verify program logic.
# 20 个样本，每个 3 个特征 | 20 samples, each with 3 features
X = torch.tensor([
    [1.0, 2.0, 0.5],   # 样本 0 | sample 0
    [0.5, 1.0, 1.5],   # 样本 1 | sample 1
    [2.0, 0.5, 1.0],   # 样本 2 | sample 2
    [1.5, 3.0, 0.5],   # 样本 3 | sample 3
    [0.5, 0.5, 2.0],   # 样本 4 | sample 4
    [3.0, 1.0, 0.5],   # 样本 5 | sample 5
    [1.0, 1.5, 1.5],   # 样本 6 | sample 6
    [2.5, 2.0, 1.0],   # 样本 7 | sample 7
    [0.5, 2.5, 0.5],   # 样本 8 | sample 8
    [1.5, 1.0, 2.5],   # 样本 9 | sample 9
    [2.0, 2.5, 1.5],   # 样本 10 | sample 10
    [0.5, 3.0, 1.0],   # 样本 11 | sample 11
    [1.0, 0.5, 0.5],   # 样本 12 | sample 12
    [3.0, 2.0, 2.0],   # 样本 13 | sample 13
    [1.5, 1.5, 1.0],   # 样本 14 | sample 14
    [2.5, 0.5, 1.5],   # 样本 15 | sample 15
    [0.5, 1.5, 2.5],   # 样本 16 | sample 16
    [2.0, 3.0, 0.5],   # 样本 17 | sample 17
    [1.0, 2.0, 1.5],   # 样本 18 | sample 18
    [3.0, 0.5, 2.0],   # 样本 19 | sample 19
])  # 形状 (20, 3) | shape (20, 3)

# 按真实规律生成标签：y = 2*x1 + 3*x2 - x3 + 1 | Generate labels by true rule: y = 2*x1 + 3*x2 - x3 + 1
# 为什么这样做？ | Why do this?
# 模型训练后，权重应该接近 [2, 3, -1]，偏置接近 1。 | After training, weights should be close to [2, 3, -1], bias close to 1.
y = 2.0 * X[:, 0] + 3.0 * X[:, 1] - 1.0 * X[:, 2] + 1.0
y = y.unsqueeze(1)  # (20,) → (20, 1)，与模型输出形状对齐 | align with model output shape

print(f"X 形状: {X.shape} | X shape: {X.shape}")    # torch.Size([20, 3])
print(f"y 形状: {y.shape} | y shape: {y.shape}")    # torch.Size([20, 1])
print(f"y 前 5 个值: {y[:5].squeeze().tolist()} | First 5 y values: {y[:5].squeeze().tolist()}")
# y 前5: [8.5, 3.5, 4.5, 11.5, 0.0]


# =============================================================================
# 2. 封装为 DataLoader | 2. Wrap into DataLoader
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# DataLoader 自动把数据切成 batch，省去手动切片的工作。 | DataLoader automatically splits data into batches, eliminating manual slicing.
#
# 技术栈：torch.utils.data.TensorDataset, DataLoader | Tech stack: torch.utils.data.TensorDataset, DataLoader

print("\n2. 封装 DataLoader | 2. Wrap DataLoader")
print("-" * 40)

# 为什么 batch_size=5？ | Why batch_size=5?
# 20 个样本 / 5 = 4 个 batch，方便演示循环。 | 20 samples / 5 = 4 batches, convenient for demonstration.
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=5, shuffle=True)
# shuffle=True：每个 epoch 打乱顺序，避免模型记住顺序 | shuffle=True: shuffle each epoch to prevent memorizing order

print(f"数据集总样本数: {len(dataset)} | Total samples: {len(dataset)}")   # 20
print(f"batch_size: 5，共 {len(loader)} 个 batch | batch_size: 5, total {len(loader)} batches")  # 4
print(f"每个 batch 形状: X=(5,3), y=(5,1) | Each batch shape: X=(5,3), y=(5,1)")


# =============================================================================
# 3. 定义模型 | 3. Define Model
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 用一个简单的线性模型（无激活函数）拟合线性数据。 | Use a simple linear model (no activation) to fit linear data.
# 因为真实规律本身是线性的，加 ReLU 反而会限制表达。 | The true rule is linear, so adding ReLU would limit expressiveness.
#
# 技术栈：torch.nn.Module, torch.nn.Linear | Tech stack: torch.nn.Module, torch.nn.Linear

print("\n3. 定义模型 | 3. Define Model")
print("-" * 40)


class LinearRegression(nn.Module):
    """
    线性回归模型：y = Wx + b | Linear Regression: y = Wx + b

    技术栈：torch.nn.Module, torch.nn.Linear | Tech stack: torch.nn.Module, torch.nn.Linear
    """

    def __init__(self):
        super().__init__()
        # 为什么 Linear(3, 1)？ | Why Linear(3, 1)?
        # 3 个输入特征（x1, x2, x3），输出 1 个预测值 y_pred。 | 3 input features (x1, x2, x3), output 1 prediction y_pred.
        self.linear = nn.Linear(in_features=3, out_features=1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegression()
criterion = nn.MSELoss()                          # 回归任务用 MSE | MSE for regression
optimizer = optim.Adam(model.parameters(), lr=0.05)  # Adam 优化器，学习率 0.05 | Adam optimizer, lr=0.05

print(f"模型结构: | Model structure:")
print(model)
print(f"参数量: {sum(p.numel() for p in model.parameters())} | Parameter count: {sum(p.numel() for p in model.parameters())}")
# Linear(3,1) 有 3 个权重 + 1 个偏置 = 4 个参数 | Linear(3,1) has 3 weights + 1 bias = 4 parameters


# =============================================================================
# 4. 完整训练循环 | 4. Complete Training Loop
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 这是 Day 4 的核心。演示「外层 epoch 循环 + 内层 batch 循环」的双层结构。 | This is the core of Day 4. Demonstrates the two-level structure of outer epoch loop + inner batch loop.
#
# 技术栈：DataLoader 迭代, optimizer.zero_grad, loss.backward, optimizer.step
# Tech stack: DataLoader iteration, optimizer.zero_grad, loss.backward, optimizer.step

print("\n4. 完整训练循环 | 4. Complete Training Loop")
print("-" * 40)

num_epochs = 100  # 完整遍历数据集 100 次 | Traverse the full dataset 100 times

for epoch in range(num_epochs):
    # 外层循环：一个 epoch = 完整遍历一次全部数据 | Outer loop: one epoch = one complete pass over all data
    epoch_loss = 0.0  # 累积本 epoch 的总 loss | Accumulate total loss for this epoch

    for batch_x, batch_y in loader:
        # 内层循环：每次取一个 batch（5 个样本）| Inner loop: take one batch (5 samples) each time

        # 步骤 1：清零梯度 | Step 1: Zero gradients
        # 为什么必须清零？ | Why must we zero?
        # PyTorch 默认累积梯度，不清零会叠加上一个 batch 的梯度，导致错误更新。 | PyTorch accumulates gradients by default; not zeroing adds last batch's gradients, causing wrong updates.
        optimizer.zero_grad()

        # 步骤 2：前向传播 | Step 2: Forward pass
        y_pred = model(batch_x)  # 形状 (5, 1) | shape (5, 1)

        # 步骤 3：计算损失 | Step 3: Compute loss
        loss = criterion(y_pred, batch_y)

        # 步骤 4：反向传播 | Step 4: Backpropagation
        # 从 loss 开始，自动计算所有参数的梯度 | Starting from loss, automatically compute gradients for all parameters
        loss.backward()

        # 步骤 5：更新参数 | Step 5: Update parameters
        # 用 optimizer 根据梯度更新 model 的权重和偏置 | Use optimizer to update model weights and biases based on gradients
        optimizer.step()

        epoch_loss += loss.item()  # .item() 把标量张量转为 Python float | .item() converts scalar tensor to Python float

    # 每 10 个 epoch 打印平均 loss | Print average loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(loader)  # 总 loss / batch 数量 = 平均 loss | total loss / num batches = avg loss
        print(f"Epoch [{epoch+1:3d}/{num_epochs}]  avg_loss = {avg_loss:.6f} | Epoch [{epoch+1:3d}/{num_epochs}]  avg_loss = {avg_loss:.6f}")


# =============================================================================
# 5. 验证训练结果 | 5. Verify Training Results
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 查看训练后的权重是否接近真实值 [2, 3, -1]，偏置是否接近 1。 | Check if trained weights are close to true values [2, 3, -1] and bias close to 1.
#
# 技术栈：model.linear.weight, model.linear.bias | Tech stack: model.linear.weight, model.linear.bias

print("\n5. 验证训练结果 | 5. Verify Training Results")
print("-" * 40)

w = model.linear.weight.data   # 形状 (1, 3) | shape (1, 3)
b = model.linear.bias.data     # 形状 (1,)  | shape (1,)

print(f"学到的权重: {w.squeeze().tolist()} | Learned weights: {w.squeeze().tolist()}")
print(f"期望权重:   [2.0, 3.0, -1.0]      | Expected weights: [2.0, 3.0, -1.0]")
print(f"学到的偏置: {b.item():.4f}         | Learned bias:    {b.item():.4f}")
print(f"期望偏置:   1.0                    | Expected bias:   1.0")

# 用第一个样本验证预测 | Verify prediction with first sample
# X[0] = [1.0, 2.0, 0.5]，真实 y = 2*1 + 3*2 - 1*0.5 + 1 = 8.5
with torch.no_grad():
    test_x = torch.tensor([[1.0, 2.0, 0.5]])
    pred = model(test_x)
print(f"\n测试样本 [1.0, 2.0, 0.5]         | Test sample [1.0, 2.0, 0.5]")
print(f"  真实值: 8.5                       | True value: 8.5")
print(f"  预测值: {pred.item():.4f}         | Predicted:  {pred.item():.4f}")


print("\n" + "=" * 55)
print("完整训练循环学习完成！ | Complete training loop learning done!")
print("=" * 55)
print("\n核心要点： | Key takeaways:")
print("1. 外层 epoch 循环：完整遍历数据次数 | 1. Outer epoch loop: number of full data passes")
print("2. 内层 batch 循环：DataLoader 自动切批 | 2. Inner batch loop: DataLoader auto-splits batches")
print("3. 每个 batch 必须 zero_grad() → forward → loss → backward → step | 3. Each batch: zero_grad() → forward → loss → backward → step")
print("4. epoch_loss 累积后除以 len(loader) 得平均 loss | 4. epoch_loss divided by len(loader) gives average loss")
