"""
Day 5-6: MNIST 项目 - 模型保存与加载 | Day 5-6: MNIST Project - Model Saving and Loading
学习目标：掌握 state_dict 保存/加载，理解两种保存方式的区别，验证加载后模型正确工作 | Learning Objectives: Master state_dict save/load, understand two saving methods, verify loaded model works correctly

技术栈： | Tech stack:
- PyTorch (torch)
- torch.save / torch.load
- model.state_dict() / model.load_state_dict()
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

print("=" * 60)
print("Day 5-6: 模型保存与加载 | Day 5-6: Model Saving and Loading")
print("=" * 60)


# =============================================================================
# 1. 为什么需要保存模型？ | 1. Why Save Models?
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 训练耗时几分钟到几天，不保存就意味着每次使用都要重新训练。 | Training takes minutes to days; without saving you'd retrain every time you want to use the model.
# 保存后可以：部署到生产环境、继续微调、分享给他人。 | After saving: deploy to production, continue fine-tuning, share with others.

print("\n1. 保存模型的两种方式 | 1. Two Ways to Save Models")
print("-" * 40)
print("  方式 A：保存 state_dict（推荐）| Method A: Save state_dict (recommended)")
print("    torch.save(model.state_dict(), 'model.pth')")
print("    优点：只保存参数，文件小；加载时与代码解耦 | Pros: only saves parameters, small file; decoupled from code on load")
print()
print("  方式 B：保存整个模型（不推荐）| Method B: Save entire model (not recommended)")
print("    torch.save(model, 'model.pth')")
print("    缺点：依赖类定义文件路径，换环境容易报错 | Cons: depends on class definition path, breaks easily in different environments")


# =============================================================================
# 2. 定义模型（与 03 保持一致）| 2. Define Model (same as 03)
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 加载 state_dict 时必须先创建同结构的模型实例，才能把参数填进去。 | Loading state_dict requires a model instance of the same architecture first, to fill parameters into.

print("\n2. 定义模型结构 | 2. Define Model Architecture")
print("-" * 40)


class MNISTCnn(nn.Module):
    """
    与 03 完全相同的 CNN 结构 | Identical CNN architecture as in 03
    加载 state_dict 要求结构完全匹配 | Loading state_dict requires exact architecture match
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
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.drop(x.view(x.size(0), -1))
        x = self.relu(self.fc1(x))
        return self.fc2(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  使用设备: {device} | Device: {device}")


# =============================================================================
# 3. 快速训练一个小模型（3 个 epoch）| 3. Quickly Train a Small Model (3 epochs)
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 演示保存/加载需要一个已训练好的模型，用 3 个 epoch 快速得到一个。 | Demonstrating save/load requires a trained model; 3 epochs gives us one quickly.
# 实际使用时，这里换成 03 里保存好的完整训练结果。 | In real use, replace this with the fully trained result from 03.

print("\n3. 快速训练模型（3 epochs）| 3. Quick Training (3 epochs)")
print("-" * 40)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data   = torchvision.datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data    = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

model     = MNISTCnn().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/3  avg_loss = {total_loss/len(train_loader):.4f}")

# 训练后在测试集上评估一次，记录基准准确率 | Evaluate on test set after training, record baseline accuracy
model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
baseline_acc = correct / len(test_data)
print(f"  训练后测试准确率（保存前）: {baseline_acc:.2%} | Test accuracy before saving: {baseline_acc:.2%}")


# =============================================================================
# 4. 保存 state_dict | 4. Save state_dict
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 演示标准的 state_dict 保存方式，并解释 state_dict 里都有什么。 | Demonstrate standard state_dict saving and explain what's inside state_dict.
#
# 技术栈：torch.save, model.state_dict() | Tech stack: torch.save, model.state_dict()

print("\n4. 保存模型 | 4. Save Model")
print("-" * 40)

# 保存路径 | Save path
save_dir  = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)   # 目录不存在则创建 | Create directory if it doesn't exist

save_path = os.path.join(save_dir, "mnist_cnn.pth")

# 为什么只保存 state_dict 而不是整个模型？ | Why save only state_dict, not the whole model?
# state_dict 是一个 Python 字典：{层名: 参数张量}。 | state_dict is a Python dict: {layer name: parameter tensor}.
# 它不依赖 Python 路径，换环境也能加载，只要类定义相同。 | It doesn't depend on Python paths; loads in any environment as long as class definition matches.
torch.save(model.state_dict(), save_path)

# 计算文件大小 | Check file size
file_size_kb = os.path.getsize(save_path) / 1024
print(f"  已保存到: {save_path} | Saved to: {save_path}")
print(f"  文件大小: {file_size_kb:.1f} KB | File size: {file_size_kb:.1f} KB")

# 查看 state_dict 的键（层名）| Inspect state_dict keys (layer names)
state = torch.load(save_path, map_location="cpu", weights_only=True)
print(f"\n  state_dict 包含 {len(state)} 个张量: | state_dict contains {len(state)} tensors:")
for key, val in state.items():
    print(f"    {key:<30} shape: {list(val.shape)}")
# conv1.weight, conv1.bias, bn1.weight, bn1.bias, bn1.running_mean, ... fc2.weight, fc2.bias


# =============================================================================
# 5. 加载 state_dict 并验证 | 5. Load state_dict and Verify
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 模拟「换了一个进程/机器」的场景：从零创建模型，加载参数，确认准确率不变。 | Simulate "new process/machine": create model from scratch, load parameters, confirm accuracy unchanged.
#
# 技术栈：torch.load, model.load_state_dict() | Tech stack: torch.load, model.load_state_dict()

print("\n5. 加载模型并验证 | 5. Load Model and Verify")
print("-" * 40)

# 步骤 1：创建同结构的空模型 | Step 1: Create empty model with same architecture
new_model = MNISTCnn().to(device)
print("  步骤 1: 创建新模型实例（参数随机初始化）| Step 1: New model instance (randomly initialized)")

# 步骤 2：加载 state_dict | Step 2: Load state_dict
# weights_only=True：只加载张量，不执行任意代码，更安全 | weights_only=True: only load tensors, no arbitrary code execution, safer
loaded_state = torch.load(save_path, map_location=device, weights_only=True)
new_model.load_state_dict(loaded_state)
print("  步骤 2: 加载 state_dict 完成 | Step 2: state_dict loaded")

# 步骤 3：切换到 eval 模式并评估 | Step 3: Switch to eval mode and evaluate
new_model.eval()
correct_new = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        correct_new += (new_model(images).argmax(dim=1) == labels).sum().item()
loaded_acc = correct_new / len(test_data)
print(f"  步骤 3: 加载后测试准确率: {loaded_acc:.2%} | Accuracy after loading: {loaded_acc:.2%}")

# 验证准确率是否与保存前一致 | Verify accuracy matches pre-save baseline
match = abs(loaded_acc - baseline_acc) < 1e-6
print(f"\n  保存前准确率: {baseline_acc:.2%} | Before save: {baseline_acc:.2%}")
print(f"  加载后准确率: {loaded_acc:.2%} | After load:  {loaded_acc:.2%}")
print(f"  结果一致: {'✓ 是 | Yes' if match else '✗ 否（有问题！）| No (something is wrong!)'}")


# =============================================================================
# 6. 保存训练检查点（含 epoch 和 optimizer）| 6. Save Training Checkpoint (with epoch and optimizer)
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 长时间训练中途中断（断电、超时），需要从上次保存的位置继续，而不是从头再来。 | Long training interrupted (power cut, timeout) needs to resume from last save, not restart from scratch.
# 检查点保存：模型参数 + 优化器状态 + 当前 epoch。 | Checkpoint saves: model params + optimizer state + current epoch.
#
# 技术栈：torch.save（字典形式）| Tech stack: torch.save (dict form)

print("\n6. 保存/加载训练检查点 | 6. Save/Load Training Checkpoint")
print("-" * 40)

checkpoint_path = os.path.join(save_dir, "mnist_checkpoint.pth")

# 为什么要保存 optimizer.state_dict()？ | Why save optimizer.state_dict()?
# Adam 等优化器有动量（momentum）等内部状态，不保存会导致继续训练时优化器「重置」，收敛变慢。
# Optimizers like Adam have internal states (momentum, etc.); without saving, resuming training resets the optimizer and slows convergence.
checkpoint = {
    "epoch":      3,
    "model_state":     model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "test_acc":   baseline_acc,
}
torch.save(checkpoint, checkpoint_path)

ckpt_size_kb = os.path.getsize(checkpoint_path) / 1024
print(f"  检查点保存到: {checkpoint_path} | Checkpoint saved to: {checkpoint_path}")
print(f"  检查点大小: {ckpt_size_kb:.1f} KB（包含优化器状态，比 state_dict 大）| Size: {ckpt_size_kb:.1f} KB (larger than state_dict, includes optimizer state)")

# 从检查点恢复 | Resume from checkpoint
ckpt       = torch.load(checkpoint_path, map_location=device, weights_only=True)
resume_model = MNISTCnn().to(device)
resume_optim = torch.optim.Adam(resume_model.parameters(), lr=0.001)

resume_model.load_state_dict(ckpt["model_state"])
resume_optim.load_state_dict(ckpt["optimizer_state"])
start_epoch = ckpt["epoch"]

print(f"\n  从检查点恢复: | Resume from checkpoint:")
print(f"    已训练 epoch 数: {start_epoch} | Trained epochs: {start_epoch}")
print(f"    检查点测试准确率: {ckpt['test_acc']:.2%} | Checkpoint test accuracy: {ckpt['test_acc']:.2%}")
print(f"  可继续从 epoch {start_epoch + 1} 开始训练 | Can continue training from epoch {start_epoch + 1}")


print("\n" + "=" * 60)
print("模型保存与加载完成！ | Model saving and loading done!")
print("=" * 60)
print("\n核心要点： | Key takeaways:")
print("1. 推荐 state_dict 保存：torch.save(model.state_dict(), path) | 1. Recommended: torch.save(model.state_dict(), path)")
print("2. 加载顺序：创建同结构模型 → load_state_dict() → eval() | 2. Load order: create same-arch model → load_state_dict() → eval()")
print("3. torch.load 用 weights_only=True 更安全 | 3. torch.load with weights_only=True is safer")
print("4. 检查点（checkpoint）同时保存模型 + 优化器 + epoch，支持断点续训 | 4. Checkpoint saves model + optimizer + epoch, supports resume training")
print("5. 加载后验证准确率与保存前一致，确认保存/加载无误 | 5. Verify accuracy matches before/after to confirm save/load correctness")
