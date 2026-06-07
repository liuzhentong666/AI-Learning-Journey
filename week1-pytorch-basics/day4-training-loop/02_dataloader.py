"""
Day 4: 训练循环与 DataLoader - Dataset 与 DataLoader | Day 4: Training Loop & DataLoader - Dataset and DataLoader
学习目标：掌握 TensorDataset 和自定义 Dataset，理解 DataLoader 参数 | Learning Objectives: Master TensorDataset and custom Dataset, understand DataLoader parameters

技术栈： | Tech stack:
- PyTorch (torch)
- torch.utils.data（Dataset, TensorDataset, DataLoader） | torch.utils.data (Dataset, TensorDataset, DataLoader)
"""

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

print("=" * 55)
print("Day 4: Dataset 与 DataLoader | Day 4: Dataset and DataLoader")
print("=" * 55)


# =============================================================================
# 1. 为什么需要 DataLoader？ | 1. Why Do We Need DataLoader?
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 演示手动切批的繁琐，对比 DataLoader 的简洁。 | Demonstrate the tedium of manual batching vs. the simplicity of DataLoader.
#
# 技术栈：Python list 手动切片 | Tech stack: Python list manual slicing

print("\n1. 手动切批 vs DataLoader | 1. Manual batching vs DataLoader")
print("-" * 40)

# 手动方式（不推荐） | Manual way (not recommended)
X_raw = torch.tensor([
    [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],
    [9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0],
])  # 8 个样本，每个 2 维 | 8 samples, each 2-dimensional
y_raw = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])  # 二分类标签 | binary classification labels

batch_size = 3
print("手动切批（不推荐）: | Manual batching (not recommended):")
for i in range(0, len(X_raw), batch_size):
    bx = X_raw[i:i + batch_size]
    by = y_raw[i:i + batch_size]
    print(f"  batch {i // batch_size}: X 形状={bx.shape}, y 形状={by.shape} | X shape={bx.shape}, y shape={by.shape}")
# batch 0: X(3,2), batch 1: X(3,2), batch 2: X(2,2)  ← 最后一批不足 batch_size | last batch is short

print("\n用 DataLoader（推荐，下面详细演示）: | With DataLoader (recommended, detailed demo below):")
print("  只需写: for batch_x, batch_y in loader: | Just write: for batch_x, batch_y in loader:")
print("  DataLoader 自动切批、打乱、并行加载 | DataLoader auto-batches, shuffles, and loads in parallel")


# =============================================================================
# 2. TensorDataset：最简单的封装方式 | 2. TensorDataset: Simplest Wrapping Method
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 数据已在内存（张量形式），TensorDataset 是最快速的封装。 | Data is already in memory (tensor form); TensorDataset is the fastest wrapping method.
#
# 技术栈：torch.utils.data.TensorDataset | Tech stack: torch.utils.data.TensorDataset

print("\n2. TensorDataset 封装 | 2. TensorDataset Wrapping")
print("-" * 40)

# 构造分类数据：学生的两门考试成绩 → 是否通过（0/1） | Build classification data: two exam scores → pass/fail (0/1)
# 固定 12 个样本 | Fixed 12 samples
scores = torch.tensor([
    [85.0, 72.0],   # 高分，通过 | high scores, pass
    [90.0, 88.0],   # 高分，通过 | high scores, pass
    [78.0, 65.0],   # 中等，通过 | medium, pass
    [92.0, 95.0],   # 高分，通过 | high scores, pass
    [88.0, 80.0],   # 高分，通过 | high scores, pass
    [75.0, 70.0],   # 中等，通过 | medium, pass
    [45.0, 38.0],   # 低分，不通过 | low scores, fail
    [52.0, 40.0],   # 低分，不通过 | low scores, fail
    [35.0, 55.0],   # 低分，不通过 | low scores, fail
    [60.0, 45.0],   # 低分，不通过 | low scores, fail
    [42.0, 60.0],   # 低分，不通过 | low scores, fail
    [50.0, 35.0],   # 低分，不通过 | low scores, fail
])  # 形状 (12, 2) | shape (12, 2)

labels = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])  # 形状 (12,) | shape (12,)

# 为什么用 TensorDataset？ | Why use TensorDataset?
# 自动把多个张量对齐：第 i 个样本 = (scores[i], labels[i])。 | Automatically aligns multiple tensors: i-th sample = (scores[i], labels[i]).
dataset = TensorDataset(scores, labels)

print(f"数据集大小: {len(dataset)} 个样本 | Dataset size: {len(dataset)} samples")   # 12

# 测试索引访问 | Test index access
sample_x, sample_y = dataset[0]  # 取第 0 个样本 | get 0th sample
print(f"dataset[0]: X={sample_x.tolist()}, y={sample_y.item()} | dataset[0]: X={sample_x.tolist()}, y={sample_y.item()}")
# dataset[0]: X=[85.0, 72.0], y=1

sample_x, sample_y = dataset[6]  # 取第 6 个样本 | get 6th sample
print(f"dataset[6]: X={sample_x.tolist()}, y={sample_y.item()} | dataset[6]: X={sample_x.tolist()}, y={sample_y.item()}")
# dataset[6]: X=[45.0, 38.0], y=0


# =============================================================================
# 3. DataLoader 参数详解 | 3. DataLoader Parameter Deep Dive
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# batch_size 和 shuffle 是最常用的两个参数，需要直观演示其效果。 | batch_size and shuffle are the two most-used parameters; need to demonstrate their effects concretely.
#
# 技术栈：torch.utils.data.DataLoader | Tech stack: torch.utils.data.DataLoader

print("\n3. DataLoader 参数演示 | 3. DataLoader Parameter Demo")
print("-" * 40)

# --- 3.1 batch_size 的作用 | 3.1 Effect of batch_size ---
print("3.1 batch_size 对比 | 3.1 batch_size comparison:")

loader_bs4 = DataLoader(dataset, batch_size=4, shuffle=False)
loader_bs6 = DataLoader(dataset, batch_size=6, shuffle=False)

print(f"  batch_size=4: {len(loader_bs4)} 个 batch | {len(loader_bs4)} batches")   # 3
print(f"  batch_size=6: {len(loader_bs6)} 个 batch | {len(loader_bs6)} batches")   # 2

# 查看第一个 batch 的形状 | Check first batch shape
for bx, by in loader_bs4:
    print(f"  batch_size=4 的第一个 batch: X={bx.shape}, y={by.shape} | First batch with batch_size=4: X={bx.shape}, y={by.shape}")
    # X=torch.Size([4, 2]), y=torch.Size([4])
    break

# --- 3.2 shuffle 的作用 | 3.2 Effect of shuffle ---
# 为什么训练集要 shuffle？ | Why shuffle the training set?
# 打乱顺序可以：1) 避免模型记住样本顺序；2) 每个 batch 的标签分布更均匀。 | Shuffling: 1) prevents model from memorizing sample order; 2) more balanced label distribution per batch.
print("\n3.2 shuffle 对比（看第一个 batch 的标签）| 3.2 shuffle comparison (look at first batch labels):")

# shuffle=False：标签按顺序 0,0,0,0... 然后 1,1,1,1 | shuffle=False: labels in order 0,0,0,0... then 1,1,1,1
loader_no_shuffle = DataLoader(dataset, batch_size=4, shuffle=False)
for bx, by in loader_no_shuffle:
    print(f"  shuffle=False 第一批标签: {by.tolist()} | First batch labels: {by.tolist()}")
    # [1, 1, 1, 1] — 全是 pass，不均衡 | all pass, unbalanced
    break

# shuffle=True：标签混合 | shuffle=True: mixed labels
# 为了结果可重现，固定随机种子 | Fix random seed for reproducible results
torch.manual_seed(42)
loader_with_shuffle = DataLoader(dataset, batch_size=4, shuffle=True)
for bx, by in loader_with_shuffle:
    print(f"  shuffle=True  第一批标签: {by.tolist()} | First batch labels: {by.tolist()}")
    # 标签混合，更均衡 | mixed labels, more balanced
    break

# --- 3.3 drop_last 的作用 | 3.3 Effect of drop_last ---
print("\n3.3 drop_last 对比（12 个样本，batch_size=5）| 3.3 drop_last comparison (12 samples, batch_size=5):")
# 12 / 5 = 2 个完整 batch + 剩余 2 个 | 12/5 = 2 full batches + 2 remaining

loader_keep = DataLoader(dataset, batch_size=5, shuffle=False, drop_last=False)
loader_drop = DataLoader(dataset, batch_size=5, shuffle=False, drop_last=True)

print(f"  drop_last=False: {len(loader_keep)} 个 batch（最后一批有 {12 - 5*2} 个样本）| {len(loader_keep)} batches (last batch has {12-5*2} samples)")  # 3
print(f"  drop_last=True:  {len(loader_drop)} 个 batch（最后不足的丢弃）| {len(loader_drop)} batches (incomplete last batch dropped)")   # 2

for bx, by in loader_keep:
    pass  # 遍历完，最后一个 batch | iterate through, last batch
print(f"  drop_last=False 最后一批 X 形状: {bx.shape} | Last batch X shape: {bx.shape}")  # torch.Size([2, 2])


# =============================================================================
# 4. 自定义 Dataset 类 | 4. Custom Dataset Class
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 真实项目中数据往往来自文件/数据库，需要自定义加载逻辑。 | In real projects, data often comes from files/databases, requiring custom loading logic.
# 自定义 Dataset 必须实现 __len__ 和 __getitem__ 两个方法。 | Custom Dataset must implement __len__ and __getitem__.
#
# 技术栈：torch.utils.data.Dataset | Tech stack: torch.utils.data.Dataset

print("\n4. 自定义 Dataset 类 | 4. Custom Dataset Class")
print("-" * 40)


class ExamDataset(Dataset):
    """
    自定义考试数据集 | Custom exam dataset

    数据：学生两门考试成绩 → 是否通过 | Data: two exam scores per student → pass/fail
    技术栈：继承 torch.utils.data.Dataset，实现 __len__ 和 __getitem__ | Inherit torch.utils.data.Dataset, implement __len__ and __getitem__
    """

    def __init__(self, scores_data, labels_data):
        # 为什么在 __init__ 里保存数据？ | Why save data in __init__?
        # __init__ 只执行一次，数据保存在 self 上供后续 __getitem__ 使用。 | __init__ runs once; data saved on self for subsequent __getitem__ calls.
        self.scores = scores_data    # 形状 (N, 2) | shape (N, 2)
        self.labels = labels_data    # 形状 (N,) | shape (N,)

    def __len__(self):
        # 为什么需要 __len__？ | Why is __len__ needed?
        # DataLoader 需要知道总样本数才能计算 batch 数量。 | DataLoader needs to know total samples to calculate number of batches.
        return len(self.scores)      # 返回样本总数 | return total sample count

    def __getitem__(self, idx):
        # 为什么需要 __getitem__？ | Why is __getitem__ needed?
        # DataLoader 内部按 idx 取单个样本，再组合成 batch。 | DataLoader internally fetches individual samples by idx, then combines into a batch.
        # idx: 整数，从 0 到 len(self)-1 | idx: integer from 0 to len(self)-1
        return self.scores[idx], self.labels[idx]


# 实例化自定义 Dataset | Instantiate custom Dataset
custom_dataset = ExamDataset(scores, labels)

print(f"自定义 Dataset 大小: {len(custom_dataset)} | Custom Dataset size: {len(custom_dataset)}")   # 12

# 测试 __getitem__ | Test __getitem__
x0, y0 = custom_dataset[0]
print(f"custom_dataset[0]: 成绩={x0.tolist()}, 标签={y0.item()} | scores={x0.tolist()}, label={y0.item()}")
# 成绩=[85.0, 72.0], 标签=1

x6, y6 = custom_dataset[6]
print(f"custom_dataset[6]: 成绩={x6.tolist()}, 标签={y6.item()} | scores={x6.tolist()}, label={y6.item()}")
# 成绩=[45.0, 38.0], 标签=0

# 用自定义 Dataset 创建 DataLoader | Create DataLoader from custom Dataset
torch.manual_seed(0)
custom_loader = DataLoader(custom_dataset, batch_size=4, shuffle=True)
print(f"\n自定义 DataLoader: {len(custom_loader)} 个 batch | Custom DataLoader: {len(custom_loader)} batches")  # 3

print("遍历所有 batch: | Iterate all batches:")
for i, (bx, by) in enumerate(custom_loader):
    print(f"  batch {i}: X 形状={bx.shape}, y={by.tolist()} | X shape={bx.shape}, y={by.tolist()}")
# batch 0: X(4,2), batch 1: X(4,2), batch 2: X(4,2)


# =============================================================================
# 5. TensorDataset vs 自定义 Dataset 对比 | 5. TensorDataset vs Custom Dataset Comparison
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 帮助选择合适的方式。 | Help choose the right approach.

print("\n5. 两种方式对比 | 5. Comparison of Two Approaches")
print("-" * 40)
print("TensorDataset（适合）:  | TensorDataset (suitable when):")
print("  - 数据已全部在内存中 | Data is all in memory")
print("  - 不需要特殊预处理 | No special preprocessing needed")
print("  - 快速原型验证 | Quick prototyping")
print()
print("自定义 Dataset（适合）: | Custom Dataset (suitable when):")
print("  - 数据来自文件/数据库，按需加载 | Data from files/databases, load on demand")
print("  - 需要数据增强或变换 | Need data augmentation or transforms")
print("  - 数据量太大无法全部放入内存 | Data too large to fit in memory")


print("\n" + "=" * 55)
print("Dataset 与 DataLoader 学习完成！ | Dataset and DataLoader learning done!")
print("=" * 55)
print("\n核心要点： | Key takeaways:")
print("1. TensorDataset: 直接封装内存中的张量 | 1. TensorDataset: directly wraps in-memory tensors")
print("2. 自定义 Dataset: 实现 __len__ 和 __getitem__ | 2. Custom Dataset: implement __len__ and __getitem__")
print("3. DataLoader 关键参数: batch_size, shuffle, drop_last | 3. DataLoader key params: batch_size, shuffle, drop_last")
print("4. 训练集 shuffle=True，验证/测试集 shuffle=False | 4. Training set shuffle=True, val/test shuffle=False")
