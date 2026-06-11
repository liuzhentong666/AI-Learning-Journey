"""
Day 5-6: MNIST 项目 - 推理与错误分析 | Day 5-6: MNIST Project - Inference and Error Analysis
学习目标：加载已训练模型，对单张图像做推理，分析预测错误的样本 | Learning Objectives: Load trained model, run inference on single images, analyze misclassified samples

技术栈： | Tech stack:
- PyTorch (torch)
- torch.nn.functional（softmax） | torch.nn.functional (softmax)
- torchvision（datasets, transforms） | torchvision (datasets, transforms)
- torch.utils.data（DataLoader） | torch.utils.data (DataLoader)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

print("=" * 60)
print("Day 5-6: MNIST 推理与错误分析 | Day 5-6: MNIST Inference and Error Analysis")
print("=" * 60)


# =============================================================================
# 1. 定义模型结构并加载权重 | 1. Define Model and Load Weights
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 推理阶段模拟真实部署：加载已训练好的模型，不再训练，只做预测。 | Inference stage simulates real deployment: load trained model, no more training, only prediction.

print("\n1. 加载已训练模型 | 1. Load Trained Model")
print("-" * 40)


class MNISTCnn(nn.Module):
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


device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path  = "./checkpoints/mnist_cnn.pth"

# 检查模型文件是否存在 | Check if model file exists
if os.path.exists(save_path):
    model = MNISTCnn().to(device)
    state = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    # 推理前必须 eval()，关闭 Dropout 和 BatchNorm 的训练行为 | Must call eval() before inference to disable Dropout and BatchNorm training behavior
    model.eval()
    print(f"  ✓ 已加载模型: {save_path} | Model loaded: {save_path}")
else:
    # 如果还没运行 04，先快速训练一个 | If 04 hasn't been run, train quickly
    print(f"  未找到 {save_path}，先快速训练... | {save_path} not found, quick training...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader     = DataLoader(train_data, batch_size=64, shuffle=True)
    model      = MNISTCnn().to(device)
    criterion  = nn.CrossEntropyLoss()
    optim      = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(3):
        model.train()
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optim.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optim.step()
        print(f"  Epoch {epoch+1}/3 done")
    model.eval()
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"  ✓ 训练完成并保存 | Training done and saved")


# =============================================================================
# 2. 对单张图像做推理 | 2. Run Inference on a Single Image
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 演示「真实场景」：只有一张图像，不走 DataLoader，直接预处理后送入模型。 | Demonstrates "real scenario": single image, no DataLoader, directly preprocess and feed to model.
#
# 技术栈：unsqueeze(0), torch.no_grad(), F.softmax | Tech stack: unsqueeze(0), torch.no_grad(), F.softmax

print("\n2. 单张图像推理 | 2. Single Image Inference")
print("-" * 40)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

# 取 5 个不同数字的样本做演示 | Take 5 samples of different digits for demo
demo_indices = [0, 1, 2, 3, 4]   # 这些样本在测试集中对应不同数字 | These map to different digits in the test set

print("  样本索引  真实标签  预测标签  置信度   是否正确 | Idx  True  Pred  Conf    Correct")
print("  " + "-" * 55)

for idx in demo_indices:
    image, true_label = test_dataset[idx]   # image: (1, 28, 28), true_label: int

    # 为什么要 unsqueeze(0)？ | Why unsqueeze(0)?
    # 模型期望输入为 (batch, C, H, W)，单张图像是 (C, H, W)，加一维变成 (1, C, H, W)。
    # Model expects (batch, C, H, W); single image is (C, H, W); add dim to get (1, C, H, W).
    input_tensor = image.unsqueeze(0).to(device)   # (1, 1, 28, 28)

    with torch.no_grad():
        logits      = model(input_tensor)                 # (1, 10)
        # 为什么用 softmax 把 logit 转成概率？ | Why softmax to convert logits to probabilities?
        # logit 是任意实数，softmax 把它们归一化为和为 1 的概率分布，方便解读「置信度」。
        # Logits are arbitrary real numbers; softmax normalizes them to a probability distribution summing to 1, easier to interpret as "confidence".
        probs       = F.softmax(logits, dim=1)            # (1, 10)
        pred_label  = probs.argmax(dim=1).item()          # 预测类别 | predicted class
        confidence  = probs[0, pred_label].item()         # 预测类别的概率 | probability of predicted class

    correct = "✓" if pred_label == true_label else "✗"
    print(f"  [{idx:5d}]   {true_label:4d}     {pred_label:4d}   {confidence:.1%}   {correct}")
    # 格式：样本索引 真实标签 预测标签 置信度 是否正确


# =============================================================================
# 3. 批量推理并收集错误样本 | 3. Batch Inference and Collect Errors
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 分析模型哪些情况下预测错误，是改进模型的第一步。 | Analyzing when the model makes errors is the first step to improving it.
# 收集所有错误样本：真实标签、预测标签、置信度。 | Collect all errors: true label, predicted label, confidence.
#
# 技术栈：DataLoader 批量推理，列表收集错误 | Tech stack: DataLoader batch inference, list to collect errors

print("\n3. 错误样本分析 | 3. Error Analysis")
print("-" * 40)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 存储所有错误 | Store all errors
errors = []   # 每个元素：(样本索引, 真实标签, 预测标签, 置信度) | each: (index, true_label, pred_label, confidence)
sample_offset = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits     = model(images)
        probs      = F.softmax(logits, dim=1)
        preds      = probs.argmax(dim=1)
        confs      = probs.max(dim=1).values   # 每个样本的最高概率 | highest probability per sample

        # 找出这个 batch 中预测错误的样本 | Find misclassified samples in this batch
        wrong_mask = (preds != labels)
        wrong_idxs = wrong_mask.nonzero(as_tuple=True)[0]

        for i in wrong_idxs:
            errors.append({
                "sample_idx":  sample_offset + i.item(),
                "true_label":  labels[i].item(),
                "pred_label":  preds[i].item(),
                "confidence":  confs[i].item(),
            })

        sample_offset += labels.size(0)

total_samples = len(test_dataset)
error_count   = len(errors)
accuracy      = (total_samples - error_count) / total_samples

print(f"  测试集总样本: {total_samples} | Total samples: {total_samples}")
print(f"  预测错误数:   {error_count}  | Errors:        {error_count}")
print(f"  整体准确率:   {accuracy:.2%} | Accuracy:      {accuracy:.2%}")


# =============================================================================
# 4. 分析最容易混淆的类别对 | 4. Analyze Most Confused Class Pairs
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 仅知道错误数量不够，还要知道「把数字 X 认成 Y」最常发生，才能针对性改进。 | Just knowing error count isn't enough; knowing "digit X most often confused with Y" enables targeted improvements.

print("\n4. 最常见的混淆类别对 | 4. Most Confused Class Pairs")
print("-" * 40)

# 统计 (true, pred) 的次数 | Count occurrences of (true, pred)
confusion_counts = {}
for err in errors:
    key = (err["true_label"], err["pred_label"])
    confusion_counts[key] = confusion_counts.get(key, 0) + 1

# 按次数降序排列 | Sort by count descending
sorted_pairs = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)

print("  真实 → 预测   次数   | True → Pred  Count")
print("  " + "-" * 30)
for (true_lbl, pred_lbl), count in sorted_pairs[:10]:   # 只显示前 10 个 | show top 10
    bar = "█" * count
    print(f"  {true_lbl} → {pred_lbl}        {count:3d}   {bar}")


# =============================================================================
# 5. 高置信度错误分析 | 5. High-Confidence Error Analysis
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 「高置信度地预测错误」比「低置信度地预测错误」更危险，模型不知道自己错了。 | "Wrong with high confidence" is more dangerous than "wrong with low confidence" — model doesn't know it's wrong.
# 找出这些样本帮助发现模型的盲点。 | Finding these samples reveals model blindspots.

print("\n5. 高置信度错误（置信度 > 90%）| 5. High-Confidence Errors (confidence > 90%)")
print("-" * 40)

# 置信度 > 90% 但预测错误的样本 | Samples with > 90% confidence but wrong prediction
high_conf_errors = [e for e in errors if e["confidence"] > 0.9]
high_conf_errors.sort(key=lambda x: x["confidence"], reverse=True)   # 按置信度降序 | sort by confidence descending

print(f"  高置信度错误数: {len(high_conf_errors)} / {error_count} 个错误 | High-confidence errors: {len(high_conf_errors)} / {error_count}")
print()
print(f"  {'样本索引 | Idx':>12}  {'真实 | True':>10}  {'预测 | Pred':>10}  {'置信度 | Conf':>12}")
print("  " + "-" * 50)
for e in high_conf_errors[:10]:   # 只展示前 10 个 | show top 10
    print(f"  {e['sample_idx']:>12}  {e['true_label']:>10}  {e['pred_label']:>10}  {e['confidence']:>11.1%}")
    # 例如：真实是 5，预测是 3，置信度 95%。这种错误值得重点关注。


print("\n" + "=" * 60)
print("推理与错误分析完成！ | Inference and error analysis done!")
print("=" * 60)
print("\n核心要点： | Key takeaways:")
print("1. 单张推理：unsqueeze(0) 加批次维度 → no_grad() → softmax → argmax | 1. Single inference: unsqueeze(0) add batch dim → no_grad() → softmax → argmax")
print("2. softmax 把 logit 转为概率（0~1），argmax 取最大概率的类别 | 2. softmax converts logits to probabilities (0~1), argmax picks highest-prob class")
print("3. 错误分析：(真实, 预测) 对的频次图帮助发现混淆的类别 | 3. Error analysis: (true, pred) pair frequencies reveal confused classes")
print("4. 高置信度错误最危险，模型「自信地错」 | 4. High-confidence errors are most dangerous — model is 'confidently wrong'")
print("5. eval() + no_grad() 是推理阶段的固定搭配 | 5. eval() + no_grad() are always paired in inference")
