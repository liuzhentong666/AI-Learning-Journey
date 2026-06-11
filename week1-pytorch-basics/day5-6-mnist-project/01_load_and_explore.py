"""
Day 5-6: MNIST 项目 - 数据加载与探索 | Day 5-6: MNIST Project - Data Loading and Exploration
学习目标：用 torchvision 下载 MNIST，理解数据集结构，可视化样本 | Learning Objectives: Download MNIST with torchvision, understand dataset structure, visualize samples

技术栈： | Tech stack:
- PyTorch (torch)
- torchvision（datasets, transforms） | torchvision (datasets, transforms)
- torch.utils.data（DataLoader） | torch.utils.data (DataLoader)
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

print("=" * 60)
print("Day 5-6: MNIST 数据加载与探索 | Day 5-6: MNIST Data Loading and Exploration")
print("=" * 60)


# =============================================================================
# 1. 为什么选 MNIST？ | 1. Why MNIST?
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# MNIST 是深度学习的 "Hello World"：28×28 灰度图，10 类（数字 0-9）。 | MNIST is the "Hello World" of deep learning: 28×28 grayscale images, 10 classes (digits 0-9).
# 数据量小（60000 训练 + 10000 测试），CPU 可在几分钟内训练完。 | Small dataset (60000 train + 10000 test), CPU can train in a few minutes.
# Day 3 已经手写了 MNISTClassifier 结构，Day 5-6 就是把它跑起来。 | Day 3 already hand-coded MNISTClassifier structure; Day 5-6 runs it for real.

print("\n1. MNIST 数据集介绍 | 1. MNIST Dataset Introduction")
print("-" * 40)
print("  图像尺寸: 28 × 28 像素，灰度（单通道） | Image size: 28×28 pixels, grayscale (1 channel)")
print("  类别数:   10（数字 0-9）               | Classes: 10 (digits 0-9)")
print("  训练集:   60000 张                      | Train set: 60000 images")
print("  测试集:   10000 张                      | Test set:  10000 images")


# =============================================================================
# 2. 定义数据变换（Transform） | 2. Define Data Transforms
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 原始 MNIST 图像是 PIL Image，像素值 [0, 255]。 | Raw MNIST images are PIL Images, pixel values [0, 255].
# 需要两步变换：ToTensor（转张量且归一化到 [0,1]）+ Normalize（标准化到均值0方差1）。 | Two transforms needed: ToTensor (convert to tensor, normalize to [0,1]) + Normalize (standardize to mean=0, std=1).
#
# 技术栈：torchvision.transforms.Compose, ToTensor, Normalize | Tech stack: torchvision.transforms.Compose, ToTensor, Normalize

print("\n2. 定义数据变换 | 2. Define Data Transforms")
print("-" * 40)

# 为什么 Normalize(mean=0.1307, std=0.3081)？ | Why Normalize(mean=0.1307, std=0.3081)?
# 这是 MNIST 训练集所有像素的均值和标准差，提前计算好的固定值。 | These are the pre-computed mean and std of all MNIST training pixels — fixed values.
# 标准化后模型收敛更快，梯度更稳定。 | Normalization speeds up convergence and stabilizes gradients.
transform = transforms.Compose([
    transforms.ToTensor(),                        # PIL Image → (1, 28, 28) 张量，值域 [0, 1] | PIL Image → (1,28,28) tensor, range [0,1]
    transforms.Normalize((0.1307,), (0.3081,))   # (x - 0.1307) / 0.3081，值域约 [-0.4, 2.8] | standardize, range approx [-0.4, 2.8]
])

print("  变换流水线: | Transform pipeline:")
print("    步骤 1: ToTensor()          → PIL Image 转 (1,28,28) 张量，/ 255 归一化 | PIL Image → (1,28,28) tensor, /255 normalize")
print("    步骤 2: Normalize(0.1307, 0.3081) → (x - mean) / std 标准化 | (x - mean) / std standardize")


# =============================================================================
# 3. 下载并加载 MNIST 数据集 | 3. Download and Load MNIST Dataset
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# torchvision.datasets.MNIST 封装了下载+加载，一行代码搞定。 | torchvision.datasets.MNIST wraps download+load into one line.
# root='./data'：数据存储路径；train=True/False 区分训练集和测试集。 | root='./data': storage path; train=True/False distinguishes train/test.
#
# 技术栈：torchvision.datasets.MNIST | Tech stack: torchvision.datasets.MNIST

print("\n3. 下载并加载数据集 | 3. Download and Load Dataset")
print("-" * 40)

# download=True：如果本地没有就从网络下载 | download=True: download from network if not locally available
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print(f"  训练集样本数: {len(train_dataset)} | Train samples: {len(train_dataset)}")   # 60000
print(f"  测试集样本数: {len(test_dataset)} | Test samples:  {len(test_dataset)}")    # 10000
print(f"  类别列表: {train_dataset.classes} | Classes: {train_dataset.classes}")
# ['0 - zero', '1 - one', ..., '9 - nine']


# =============================================================================
# 4. 探索单个样本 | 4. Explore a Single Sample
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 在送入模型之前，先直观理解数据长什么样：形状、数值范围、标签含义。 | Before feeding to the model, intuitively understand what the data looks like: shape, value range, label meaning.
#
# 技术栈：Dataset 索引访问 dataset[i] | Tech stack: Dataset index access dataset[i]

print("\n4. 探索单个样本 | 4. Explore a Single Sample")
print("-" * 40)

# 取第 0 个样本 | Get 0th sample
image_0, label_0 = train_dataset[0]

print(f"  image 形状: {image_0.shape}       | image shape: {image_0.shape}")   # torch.Size([1, 28, 28])
print(f"  image 类型: {image_0.dtype}        | image dtype: {image_0.dtype}")  # torch.float32
print(f"  像素最小值: {image_0.min():.4f}    | pixel min:   {image_0.min():.4f}")
print(f"  像素最大值: {image_0.max():.4f}    | pixel max:   {image_0.max():.4f}")
print(f"  label: {label_0}（数字 '{label_0}'）| label: {label_0} (digit '{label_0}')")

# 为什么形状是 (1, 28, 28) 不是 (28, 28)？ | Why shape (1, 28, 28) not (28, 28)?
# 第 1 维是通道数（Channel）。灰度图 = 1 个通道，彩色 RGB = 3 个通道。 | Dim 1 is channel count. Grayscale = 1 channel, color RGB = 3 channels.
# 卷积网络统一要求输入为 (batch, channel, H, W) 格式。 | CNNs require input in (batch, channel, H, W) format.
print()
print("  说明：(1, 28, 28) = (C, H, W) | Note: (1, 28, 28) = (C, H, W)")
print("       C=1（灰度通道）, H=28（高）, W=28（宽）| C=1 (grayscale), H=28 (height), W=28 (width)")


# =============================================================================
# 5. 查看每类样本数量分布 | 5. Check Class Distribution
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 检查数据是否均衡：如果某类样本极少，模型会偏向多数类。 | Check if data is balanced: too few samples of one class causes model bias.
#
# 技术栈：train_dataset.targets（标签张量） | Tech stack: train_dataset.targets (label tensor)

print("\n5. 类别样本数量分布 | 5. Class Distribution")
print("-" * 40)

# train_dataset.targets 是所有标签的张量，形状 (60000,) | train_dataset.targets is the full label tensor, shape (60000,)
targets = train_dataset.targets   # torch.Size([60000])

print("  各类别样本数: | Samples per class:")
for cls in range(10):
    count = (targets == cls).sum().item()
    bar = "█" * (count // 600)   # 每 600 个样本画一格 | one block per 600 samples
    print(f"    数字 {cls}: {count:5d}  {bar} | Digit {cls}: {count:5d}")
# 每类约 6000 个样本，分布均衡 | ~6000 per class, balanced distribution


# =============================================================================
# 6. 创建 DataLoader | 6. Create DataLoader
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# DataLoader 负责自动切批、打乱顺序、并行加载。 | DataLoader handles auto-batching, shuffling, and parallel loading.
# batch_size=64 是 MNIST 常用值：内存友好，梯度估计稳定。 | batch_size=64 is common for MNIST: memory-friendly, stable gradient estimate.
#
# 技术栈：torch.utils.data.DataLoader | Tech stack: torch.utils.data.DataLoader

print("\n6. 创建 DataLoader | 6. Create DataLoader")
print("-" * 40)

# 为什么训练集 shuffle=True，测试集 shuffle=False？ | Why train shuffle=True, test shuffle=False?
# 训练：每 epoch 打乱，避免模型记住顺序。 | Train: shuffle each epoch to prevent memorizing order.
# 测试：不需要打乱，按固定顺序评估，结果可复现。 | Test: no shuffling needed, fixed order for reproducible evaluation.
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

print(f"  train_loader: {len(train_loader)} 个 batch（60000 / 64 = 937 整批 + 1 尾批）| {len(train_loader)} batches")  # 938
print(f"  test_loader:  {len(test_loader)} 个 batch（10000 / 64）| {len(test_loader)} batches")   # 157


# =============================================================================
# 7. 验证 DataLoader 输出形状 | 7. Verify DataLoader Output Shape
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 拿第一个 batch 出来，确认形状符合预期，避免在模型里才发现 shape 错误。 | Take first batch, confirm shape matches expectations before shape errors hit the model.
#
# 技术栈：DataLoader 迭代，iter() + next() | Tech stack: DataLoader iteration, iter() + next()

print("\n7. 验证 DataLoader 输出形状 | 7. Verify DataLoader Output Shape")
print("-" * 40)

# iter() + next() 取第一个 batch，不遍历全部 | iter() + next() fetches first batch without iterating all
images, labels = next(iter(train_loader))

print(f"  images 形状: {images.shape}  | images shape: {images.shape}")  # torch.Size([64, 1, 28, 28])
print(f"  labels 形状: {labels.shape}  | labels shape: {labels.shape}")  # torch.Size([64])
print(f"  labels 前 10 个: {labels[:10].tolist()} | First 10 labels: {labels[:10].tolist()}")
# 随机打乱后的前 10 个标签，例如 [5, 0, 4, 1, 9, 2, 1, 3, 1, 4]

print()
print("  形状解读: images = (batch, C, H, W) = (64, 1, 28, 28) | Shape: images = (batch, C, H, W) = (64, 1, 28, 28)")
print("           labels = (batch,)          = (64,)           | labels = (batch,) = (64,)")


print("\n" + "=" * 60)
print("数据加载与探索完成！ | Data loading and exploration done!")
print("=" * 60)
print("\n核心要点： | Key takeaways:")
print("1. transforms.Compose 把多个变换串联：ToTensor + Normalize | 1. transforms.Compose chains transforms: ToTensor + Normalize")
print("2. MNIST 图像形状为 (1, 28, 28)，C=1 灰度通道 | 2. MNIST image shape (1, 28, 28), C=1 grayscale channel")
print("3. train_dataset.targets 可直接查看所有标签 | 3. train_dataset.targets gives all labels directly")
print("4. DataLoader batch_size=64，训练集 shuffle=True | 4. DataLoader batch_size=64, train shuffle=True")
print("5. iter() + next() 取单个 batch 验证形状 | 5. iter() + next() fetches one batch to verify shape")
