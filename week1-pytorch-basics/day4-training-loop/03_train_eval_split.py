"""
Day 4: 训练循环与 DataLoader - 训练集/验证集拆分 | Day 4: Training Loop & DataLoader - Train/Validation Split
学习目标：掌握 80/20 数据拆分，正确使用 model.train()/model.eval() 和 torch.no_grad() | Learning Objectives: Master 80/20 data split, correctly use model.train()/model.eval() and torch.no_grad()

技术栈： | Tech stack:
- PyTorch (torch)
- torch.nn（Module, Linear, ReLU, Sequential, CrossEntropyLoss） | torch.nn (Module, Linear, ReLU, Sequential, CrossEntropyLoss)
- torch.optim（Adam） | torch.optim (Adam)
- torch.utils.data（TensorDataset, DataLoader, random_split） | torch.utils.data (TensorDataset, DataLoader, random_split)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

print("=" * 55)
print("Day 4: 训练集/验证集拆分 | Day 4: Train/Validation Split")
print("=" * 55)


# =============================================================================
# 1. 构造分类数据集 | 1. Construct Classification Dataset
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 用二分类数据演示完整的「训练+验证」流程，比回归更贴近真实场景。 | Use binary classification to demonstrate the full train+val flow, closer to real-world use.
# 规律：x1 + x2 > 5 → 类别 1，否则 → 类别 0 | Rule: x1 + x2 > 5 → class 1, else → class 0
#
# 技术栈：torch.tensor（固定值构造） | Tech stack: torch.tensor (fixed-value construction)

print("\n1. 构造数据集 | 1. Construct Dataset")
print("-" * 40)

# 固定 20 个样本，每个 2 个特征 | Fixed 20 samples, each with 2 features
X = torch.tensor([
    [1.0, 2.0],   # 和=3.0 → 类别 0 | sum=3.0 → class 0
    [2.0, 1.0],   # 和=3.0 → 类别 0 | sum=3.0 → class 0
    [1.5, 1.5],   # 和=3.0 → 类别 0 | sum=3.0 → class 0
    [0.5, 3.0],   # 和=3.5 → 类别 0 | sum=3.5 → class 0
    [2.5, 2.0],   # 和=4.5 → 类别 0 | sum=4.5 → class 0
    [3.0, 1.5],   # 和=4.5 → 类别 0 | sum=4.5 → class 0
    [2.0, 3.0],   # 和=5.0 → 类别 0 | sum=5.0 → class 0
    [1.0, 4.0],   # 和=5.0 → 类别 0 | sum=5.0 → class 0
    [4.0, 1.5],   # 和=5.5 → 类别 1 | sum=5.5 → class 1
    [3.0, 3.0],   # 和=6.0 → 类别 1 | sum=6.0 → class 1
    [5.0, 1.5],   # 和=6.5 → 类别 1 | sum=6.5 → class 1
    [2.0, 5.0],   # 和=7.0 → 类别 1 | sum=7.0 → class 1
    [4.0, 3.0],   # 和=7.0 → 类别 1 | sum=7.0 → class 1
    [3.5, 4.0],   # 和=7.5 → 类别 1 | sum=7.5 → class 1
    [6.0, 2.0],   # 和=8.0 → 类别 1 | sum=8.0 → class 1
    [5.0, 3.5],   # 和=8.5 → 类别 1 | sum=8.5 → class 1
    [4.5, 4.5],   # 和=9.0 → 类别 1 | sum=9.0 → class 1
    [7.0, 2.5],   # 和=9.5 → 类别 1 | sum=9.5 → class 1
    [6.0, 4.0],   # 和=10.0 → 类别 1 | sum=10.0 → class 1
    [5.5, 5.0],   # 和=10.5 → 类别 1 | sum=10.5 → class 1
])  # 形状 (20, 2) | shape (20, 2)

# 为什么用 Long 类型？ | Why Long type?
# CrossEntropyLoss 要求标签是 torch.long（整数），不能是 float。 | CrossEntropyLoss requires labels to be torch.long (integer), not float.
y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 dtype=torch.long)  # 形状 (20,) | shape (20,)

print(f"X 形状: {X.shape} | X shape: {X.shape}")    # torch.Size([20, 2])
print(f"y 形状: {y.shape} | y shape: {y.shape}")    # torch.Size([20,])
print(f"类别 0 样本数: {(y == 0).sum().item()} | Class 0 count: {(y == 0).sum().item()}")  # 8
print(f"类别 1 样本数: {(y == 1).sum().item()} | Class 1 count: {(y == 1).sum().item()}")  # 12


# =============================================================================
# 2. 拆分训练集和验证集 | 2. Split Train and Validation Sets
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 验证集用于检测模型是否「过拟合」，即在未见过的数据上的表现。 | Validation set detects if model is overfitting — how it performs on unseen data.
#
# 技术栈：torch.utils.data.random_split | Tech stack: torch.utils.data.random_split

print("\n2. 拆分数据集 | 2. Split Dataset")
print("-" * 40)

dataset = TensorDataset(X, y)

# 为什么 80/20 拆分？ | Why 80/20 split?
# 80% 用于训练，20% 用于验证，是最常见的比例。 | 80% for training, 20% for validation — the most common ratio.
train_size = 16   # 20 * 0.8 = 16
val_size   = 4    # 20 * 0.2 = 4

# 为什么要固定随机种子？ | Why fix random seed?
# 保证每次运行的拆分结果相同，便于复现实验。 | Ensures the same split every run, enabling reproducible experiments.
torch.manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"总样本数: {len(dataset)} | Total samples: {len(dataset)}")       # 20
print(f"训练集: {len(train_dataset)} 个样本 | Train: {len(train_dataset)} samples")  # 16
print(f"验证集: {len(val_dataset)} 个样本 | Val:   {len(val_dataset)} samples")      # 4

# 为什么训练集 shuffle=True，验证集 shuffle=False？ | Why train shuffle=True, val shuffle=False?
# 训练：打乱顺序防止顺序偏差；验证：固定顺序方便对比不同 epoch 的结果。 | Train: shuffle prevents order bias; Val: fixed order allows comparing across epochs.
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False)

print(f"训练 DataLoader: {len(train_loader)} 个 batch | Train DataLoader: {len(train_loader)} batches")  # 4
print(f"验证 DataLoader: {len(val_loader)} 个 batch | Val DataLoader:   {len(val_loader)} batches")      # 1


# =============================================================================
# 3. 定义分类模型 | 3. Define Classification Model
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 用两层 MLP（多层感知机）做二分类，比 Day 3 更完整：有隐藏层 + 激活函数。 | Use a two-layer MLP for binary classification, more complete than Day 3: has hidden layer + activation.
#
# 技术栈：torch.nn.Sequential, Linear, ReLU | Tech stack: torch.nn.Sequential, Linear, ReLU

print("\n3. 定义模型 | 3. Define Model")
print("-" * 40)


class SimpleClassifier(nn.Module):
    """
    两层 MLP 二分类模型 | Two-layer MLP binary classifier

    结构：Linear(2→8) → ReLU → Linear(8→2)
    输出：2 个 logit（对应类别 0 和类别 1）| Output: 2 logits (for class 0 and class 1)

    技术栈：nn.Sequential, nn.Linear, nn.ReLU | Tech stack: nn.Sequential, nn.Linear, nn.ReLU
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 为什么第一层 Linear(2, 8)？ | Why first layer Linear(2, 8)?
            # 输入 2 个特征，隐藏层 8 个神经元，增加模型表达能力。 | 2 input features, 8 hidden neurons to increase model capacity.
            nn.Linear(2, 8),
            nn.ReLU(),
            # 为什么输出层 Linear(8, 2)？ | Why output layer Linear(8, 2)?
            # 二分类输出 2 个 logit，CrossEntropyLoss 内部做 softmax。 | Binary classification outputs 2 logits; CrossEntropyLoss applies softmax internally.
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)


model     = SimpleClassifier()
# 为什么用 CrossEntropyLoss 不用 MSELoss？ | Why CrossEntropyLoss instead of MSELoss?
# 分类任务用交叉熵：内含 softmax + 负对数似然，数学上与分类概率匹配。 | Classification uses cross-entropy: contains softmax + NLL, mathematically matches class probabilities.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

print(f"模型结构: | Model structure:")
print(model)
print(f"参数量: {sum(p.numel() for p in model.parameters())} | Parameter count: {sum(p.numel() for p in model.parameters())}")
# Linear(2,8): 16+8=24, Linear(8,2): 16+2=18 → 总计 42 个参数 | Total 42 parameters


# =============================================================================
# 4. 训练 + 验证循环 | 4. Training + Validation Loop
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 这是 Day 4 的核心：在每个 epoch 结束后做验证，对比训练 loss 和验证 loss。 | Core of Day 4: validate after each epoch, compare train loss vs. val loss.
#
# 技术栈：model.train(), model.eval(), torch.no_grad() | Tech stack: model.train(), model.eval(), torch.no_grad()

print("\n4. 训练 + 验证循环 | 4. Training + Validation Loop")
print("-" * 40)
print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val Acc':>8}")
print("-" * 40)

num_epochs = 30

for epoch in range(num_epochs):

    # ── 训练阶段 | Training phase ──────────────────────────────────────────
    # 为什么要调用 model.train()？ | Why call model.train()?
    # 告诉模型进入训练模式：启用 Dropout 随机丢弃、BatchNorm 用当前批统计量。 | Tells model to enter training mode: enables Dropout, BatchNorm uses current batch statistics.
    # 本模型没有 Dropout/BN，但养成习惯在真实模型中很重要。 | This model has no Dropout/BN, but the habit is important for real models.
    model.train()

    train_loss_total = 0.0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()              # 1. 清零梯度 | zero gradients
        logits = model(batch_x)            # 2. 前向传播，输出 (batch, 2) | forward pass, output (batch, 2)
        loss = criterion(logits, batch_y)  # 3. 交叉熵损失 | cross-entropy loss
        loss.backward()                    # 4. 反向传播 | backpropagation
        optimizer.step()                   # 5. 更新参数 | update parameters
        train_loss_total += loss.item()

    avg_train_loss = train_loss_total / len(train_loader)

    # ── 验证阶段 | Validation phase ──────────────────────────────────────
    # 为什么要调用 model.eval()？ | Why call model.eval()?
    # 告诉模型进入推理模式：关闭 Dropout、BatchNorm 用全局统计量。 | Tells model to enter inference mode: disables Dropout, BatchNorm uses global statistics.
    model.eval()

    val_loss_total  = 0.0
    correct         = 0
    total           = 0

    # 为什么用 torch.no_grad()？ | Why use torch.no_grad()?
    # 验证不需要反向传播，不记录计算图可以节省内存、加速运算。 | Validation doesn't need backprop; not recording the computation graph saves memory and speeds up.
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            logits = model(batch_x)                        # 输出 (batch, 2) | output (batch, 2)
            loss   = criterion(logits, batch_y)
            val_loss_total += loss.item()

            # 计算准确率 | Compute accuracy
            # argmax(dim=1)：取每个样本 logit 最大的那个类别索引 | argmax(dim=1): take class index with highest logit per sample
            predicted = logits.argmax(dim=1)               # 形状 (batch,) | shape (batch,)
            correct  += (predicted == batch_y).sum().item()
            total    += batch_y.size(0)

    avg_val_loss = val_loss_total / len(val_loader)
    val_acc      = correct / total  # 准确率 = 正确数 / 总数 | accuracy = correct / total

    # 每个 epoch 都打印，方便观察收敛过程 | Print every epoch to observe convergence
    print(f"{epoch+1:>6}  {avg_train_loss:>10.4f}  {avg_val_loss:>8.4f}  {val_acc:>7.1%}")

    # 验证结束后切回训练模式，为下一个 epoch 做准备 | Switch back to train mode after validation
    model.train()


# =============================================================================
# 5. 观察训练结果 | 5. Observe Training Results
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 训练结束后用完整验证集打一次最终指标，确认模型确实学到了规律。 | After training, run a final evaluation on the full val set to confirm the model learned the pattern.

print("\n5. 最终评估 | 5. Final Evaluation")
print("-" * 40)

model.eval()
with torch.no_grad():
    # 用整个验证集（不走 DataLoader，直接取全部数据）| Use full val set (bypass DataLoader, get all data directly)
    val_X_list, val_y_list = [], []
    for bx, by in val_loader:
        val_X_list.append(bx)
        val_y_list.append(by)
    val_X_all = torch.cat(val_X_list)  # (4, 2) | (4, 2)
    val_y_all = torch.cat(val_y_list)  # (4,)   | (4,)

    logits_all   = model(val_X_all)          # (4, 2)
    pred_all     = logits_all.argmax(dim=1)  # (4,)
    final_acc    = (pred_all == val_y_all).float().mean().item()

print(f"验证集最终准确率: {final_acc:.1%} | Final val accuracy: {final_acc:.1%}")
print()
print("逐样本预测结果: | Per-sample predictions:")
for i in range(len(val_X_all)):
    x_vals  = val_X_all[i].tolist()
    true_y  = val_y_all[i].item()
    pred_y  = pred_all[i].item()
    mark    = "✓" if pred_y == true_y else "✗"
    print(f"  x={[f'{v:.1f}' for v in x_vals]}  真实={true_y}  预测={pred_y}  {mark} | true={true_y}  pred={pred_y}  {mark}")

model.train()


print("\n" + "=" * 55)
print("训练集/验证集拆分学习完成！ | Train/Val split learning done!")
print("=" * 55)
print("\n核心要点： | Key takeaways:")
print("1. random_split 按比例拆分，固定 seed 保证可复现 | 1. random_split splits by ratio; fixed seed ensures reproducibility")
print("2. 训练前 model.train()，验证前 model.eval() | 2. model.train() before training, model.eval() before validation")
print("3. 验证时必须 with torch.no_grad() | 3. Always use with torch.no_grad() during validation")
print("4. 用 argmax(dim=1) 把 logit 转成类别预测 | 4. Use argmax(dim=1) to convert logits to class predictions")
print("5. 每 epoch 同时打印 train_loss 和 val_loss，观察是否过拟合 | 5. Print both train_loss and val_loss each epoch to watch for overfitting")
