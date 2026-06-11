"""
Day 5-6: MNIST 项目 - 训练与评估 | Day 5-6: MNIST Project - Training and Evaluation
学习目标：完整运行 MNIST 训练循环，记录 loss 和 accuracy，用测试集做最终评估 | Learning Objectives: Run full MNIST training loop, track loss and accuracy, evaluate on test set

技术栈： | Tech stack:
- PyTorch (torch)
- torch.nn（CrossEntropyLoss） | torch.nn (CrossEntropyLoss)
- torch.optim（Adam） | torch.optim (Adam)
- torchvision（datasets, transforms） | torchvision (datasets, transforms)
- torch.utils.data（DataLoader） | torch.utils.data (DataLoader)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

print("=" * 60)
print("Day 5-6: MNIST 训练与评估 | Day 5-6: MNIST Training and Evaluation")
print("=" * 60)


# =============================================================================
# 1. 数据准备（复用 01 的结论）| 1. Data Preparation (reuse findings from 01)
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 把 01 里探索好的 transform 和 DataLoader 直接用于训练，不重复探索。 | Apply the transform and DataLoader from 01 directly to training without re-exploring.
#
# 技术栈：torchvision.transforms, torchvision.datasets.MNIST, DataLoader

print("\n1. 数据准备 | 1. Data Preparation")
print("-" * 40)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

print(f"  训练集: {len(train_dataset)} 样本, {len(train_loader)} 个 batch | Train: {len(train_dataset)} samples, {len(train_loader)} batches")
print(f"  测试集: {len(test_dataset)} 样本, {len(test_loader)} 个 batch | Test:  {len(test_dataset)} samples, {len(test_loader)} batches")


# =============================================================================
# 2. 定义 CNN 模型（复用 02 的结论）| 2. Define CNN Model (reuse findings from 02)
# =============================================================================
# 为什么选 CNN 不选 MLP？ | Why CNN instead of MLP?
# CNN 保留空间结构，在图像任务上通常更准，参数量也更合理。 | CNN preserves spatial structure, usually more accurate on image tasks with reasonable parameter count.

print("\n2. 定义模型 | 2. Define Model")
print("-" * 40)


class MNISTCnn(nn.Module):
    """
    CNN 分类模型：(1,28,28) → Conv→Pool → Conv→Pool → FC → 10
    CNN Classifier: (1,28,28) → Conv→Pool → Conv→Pool → FC → 10

    技术栈：nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.Dropout, nn.Linear
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(0.25)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))   # (batch, 32, 14, 14)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))   # (batch, 64, 7, 7)
        x = self.drop(x.view(x.size(0), -1))                # (batch, 3136)
        x = self.relu(self.fc1(x))                          # (batch, 128)
        return self.fc2(x)                                   # (batch, 10)


# 自动选择设备（有 GPU 用 GPU，没有用 CPU）| Auto-select device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  使用设备: {device} | Device: {device}")

model     = MNISTCnn().to(device)
# 为什么用 CrossEntropyLoss？ | Why CrossEntropyLoss?
# 多分类标准损失：内含 softmax + 负对数似然，数学上对应最大化正确类的概率。 | Standard multi-class loss: contains softmax + NLL, mathematically corresponds to maximizing correct class probability.
criterion = nn.CrossEntropyLoss()
# 为什么 lr=0.001？ | Why lr=0.001?
# Adam 的经典默认值，对大多数任务开箱即用，MNIST 上收敛快且稳定。 | Adam's classic default, works out-of-the-box for most tasks, fast and stable convergence on MNIST.
optimizer = optim.Adam(model.parameters(), lr=0.001)

total_params = sum(p.numel() for p in model.parameters())
print(f"  参数量: {total_params:,} | Parameters: {total_params:,}")


# =============================================================================
# 3. 定义评估函数 | 3. Define Evaluation Function
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 训练和测试都需要计算准确率，抽成函数避免重复代码。 | Both training and testing need accuracy; extract to a function to avoid code duplication.
#
# 技术栈：model.eval(), torch.no_grad(), argmax | Tech stack: model.eval(), torch.no_grad(), argmax

def evaluate(model, loader, criterion, device):
    """
    在给定 DataLoader 上计算平均 loss 和准确率
    Compute average loss and accuracy on a given DataLoader

    返回 (avg_loss, accuracy) | Returns (avg_loss, accuracy)
    """
    # 为什么 model.eval()？ | Why model.eval()?
    # 关闭 Dropout（推理时不随机丢弃），BatchNorm 使用全局统计量。 | Disables Dropout (no random dropping during inference), BatchNorm uses global statistics.
    model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0

    # 为什么 torch.no_grad()？ | Why torch.no_grad()?
    # 推理不需要反向传播，不构建计算图，节省内存和时间。 | Inference doesn't need backprop; not building the computation graph saves memory and time.
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)   # 移到 GPU/CPU | move to GPU/CPU
            labels = labels.to(device)

            logits    = model(images)                          # (batch, 10)
            loss      = criterion(logits, labels)
            total_loss += loss.item()

            # 为什么 argmax(dim=1)？ | Why argmax(dim=1)?
            # 10 个 logit 中取最大值的下标 = 模型认为最可能的类别。 | Index of max among 10 logits = class the model deems most likely.
            predicted = logits.argmax(dim=1)                   # (batch,)
            correct  += (predicted == labels).sum().item()
            total    += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


# =============================================================================
# 4. 训练循环 | 4. Training Loop
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 把 Day 4 学到的完整训练循环应用到真实数据集 MNIST 上。 | Apply the complete training loop learned in Day 4 to the real MNIST dataset.
# 每个 epoch 结束后做测试集评估，观察 loss 和 accuracy 的变化趋势。 | Evaluate on test set after each epoch to observe trends in loss and accuracy.
#
# 技术栈：model.train(), DataLoader, optimizer.zero_grad, loss.backward, optimizer.step

print("\n3. 开始训练 | 3. Start Training")
print("-" * 40)

num_epochs = 5   # 5 个 epoch，CPU 约需 5-15 分钟 | 5 epochs, ~5-15 minutes on CPU

# 记录每个 epoch 的指标，用于最后汇总 | Record metrics per epoch for final summary
history = {
    "train_loss": [],
    "test_loss":  [],
    "test_acc":   []
}

print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Test Loss':>9}  {'Test Acc':>8}  {'说明 | Note'}")
print("-" * 65)

for epoch in range(num_epochs):

    # ── 训练阶段 | Training phase ──────────────────────────────────────
    # 为什么每个 epoch 开始都要 model.train()？ | Why model.train() at the start of each epoch?
    # evaluate() 函数调用了 model.eval()，训练前必须切回训练模式。 | evaluate() calls model.eval(); must switch back to train mode before training.
    model.train()
    epoch_train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 标准五步训练 | Standard five-step training
        optimizer.zero_grad()                    # 1. 清零梯度 | zero gradients
        logits = model(images)                   # 2. 前向传播 | forward pass
        loss   = criterion(logits, labels)       # 3. 计算损失 | compute loss
        loss.backward()                          # 4. 反向传播 | backpropagation
        optimizer.step()                         # 5. 更新参数 | update parameters

        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)

    # ── 测试阶段 | Test phase ──────────────────────────────────────────
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # 记录历史 | Record history
    history["train_loss"].append(avg_train_loss)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)

    # 打印当前 epoch 结果 | Print current epoch results
    note = "↓ 收敛中 | converging" if epoch > 0 and test_acc > history["test_acc"][-2] else ""
    if epoch == 0:
        note = "← 基线 | baseline"
    print(f"{epoch+1:>6}  {avg_train_loss:>10.4f}  {test_loss:>9.4f}  {test_acc:>7.2%}  {note}")

    # 训练完一个 epoch 后要切回 train 模式（虽然下一轮循环头部也会切） | Switch back to train mode (loop head also does this, but be explicit)
    model.train()


# =============================================================================
# 5. 最终评估：逐类别准确率 | 5. Final Evaluation: Per-Class Accuracy
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 总体准确率掩盖了「某个数字识别特别差」的问题。 | Overall accuracy can hide "one digit is recognized very poorly".
# 逐类别分析帮助发现模型弱点。 | Per-class analysis reveals model weaknesses.
#
# 技术栈：model.eval(), torch.no_grad(), 字典统计 | Tech stack: model.eval(), torch.no_grad(), dict stats

print("\n4. 逐类别准确率 | 4. Per-Class Accuracy")
print("-" * 40)

# 每个数字类别的正确数和总数 | Correct count and total count per digit class
class_correct = [0] * 10
class_total   = [0] * 10

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits    = model(images)
        predicted = logits.argmax(dim=1)

        for i in range(labels.size(0)):
            lbl = labels[i].item()
            class_correct[lbl] += (predicted[i] == labels[i]).item()
            class_total[lbl]   += 1

print("  数字  正确数  总数   准确率 | Digit  Correct  Total  Accuracy")
print("  " + "-" * 40)
for cls in range(10):
    acc = class_correct[cls] / class_total[cls]
    bar = "█" * int(acc * 20)   # 20 格代表 100% | 20 blocks = 100%
    print(f"    {cls}:  {class_correct[cls]:4d} / {class_total[cls]:4d}  ({acc:.1%})  {bar}")


# =============================================================================
# 6. 训练总结 | 6. Training Summary
# =============================================================================

print("\n5. 训练总结 | 5. Training Summary")
print("-" * 40)

best_epoch = history["test_acc"].index(max(history["test_acc"])) + 1
best_acc   = max(history["test_acc"])

print(f"  训练 epoch 数: {num_epochs} | Trained epochs: {num_epochs}")
print(f"  最佳测试准确率: {best_acc:.2%}（Epoch {best_epoch}）| Best test accuracy: {best_acc:.2%} (Epoch {best_epoch})")
print(f"  最终测试 loss:  {history['test_loss'][-1]:.4f} | Final test loss: {history['test_loss'][-1]:.4f}")
print()
print("  各 epoch 汇总: | Epoch summary:")
print(f"  {'Epoch':>6}  {'Train Loss':>10}  {'Test Loss':>9}  {'Test Acc':>8}")
print("  " + "-" * 40)
for i in range(num_epochs):
    marker = " ← 最佳 | best" if (i + 1) == best_epoch else ""
    print(f"  {i+1:>6}  {history['train_loss'][i]:>10.4f}  {history['test_loss'][i]:>9.4f}  {history['test_acc'][i]:>7.2%}{marker}")


print("\n" + "=" * 60)
print("MNIST 训练与评估完成！ | MNIST training and evaluation done!")
print("=" * 60)
print("\n核心要点： | Key takeaways:")
print("1. 设备选择：device = 'cuda' if available else 'cpu'，模型和数据都要 .to(device) | 1. Device: 'cuda' if available else 'cpu'; both model and data need .to(device)")
print("2. evaluate() 函数：model.eval() + torch.no_grad() 是固定搭配 | 2. evaluate(): model.eval() + torch.no_grad() always go together")
print("3. 每个 epoch 训练后立即测试，观察 loss 是否同步下降 | 3. Test after each epoch, watch if loss decreases in sync")
print("4. 逐类别准确率帮助发现模型弱点 | 4. Per-class accuracy reveals model weaknesses")
print("5. train_loss 应低于 test_loss；若 train 很低而 test 很高，说明过拟合 | 5. train_loss should be below test_loss; if train very low but test high, it's overfitting")
