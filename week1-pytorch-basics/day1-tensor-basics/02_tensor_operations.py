"""
Day 1: PyTorch张量基础 - 张量操作
学习目标：掌握张量的索引、切片、变形等操作
"""

import torch

print("=" * 50)
print("张量操作练习")
print("=" * 50)

# 1. 张量的索引和切片
print("\n1. 索引和切片")
tensor = torch.arange(12).reshape(3, 4)
print(f"原张量:\n{tensor}")
print(f"第一行: {tensor[0]}")
print(f"第一列: {tensor[:, 0]}")
print(f"最后一行: {tensor[-1]}")
print(f"前两行前两列:\n{tensor[:2, :2]}")

# 2. 张量的形状操作
print("\n2. 形状操作")
x = torch.arange(12)
print(f"原始张量: {x}")
print(f"形状: {x.shape}")

# reshape - 改变形状
reshaped = x.reshape(3, 4)
print(f"reshape(3,4):\n{reshaped}")

# view - 视图（共享内存）
viewed = x.view(2, 6)
print(f"view(2,6):\n{viewed}")

# unsqueeze - 增加维度
unsqueezed = x.unsqueeze(0)  # 在第0维增加
print(f"unsqueeze(0)形状: {unsqueezed.shape}")

# squeeze - 删除大小为1的维度
squeezed = unsqueezed.squeeze()
print(f"squeeze后形状: {squeezed.shape}")

# 3. 张量的拼接
print("\n3. 张量拼接")
a = torch.ones(2, 3)
b = torch.zeros(2, 3)

# cat - 沿指定维度拼接
cat_dim0 = torch.cat([a, b], dim=0)  # 垂直拼接
cat_dim1 = torch.cat([a, b], dim=1)  # 水平拼接
print(f"垂直拼接(dim=0):\n{cat_dim0}")
print(f"水平拼接(dim=1):\n{cat_dim1}")

# stack - 创建新维度拼接
stacked = torch.stack([a, b], dim=0)
print(f"stack后形状: {stacked.shape}")

# 4. 张量的数学运算
print("\n4. 数学运算")
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# 逐元素运算
print(f"加法: {x + y}")
print(f"减法: {x - y}")
print(f"乘法: {x * y}")
print(f"除法: {x / y}")
print(f"幂运算: {x ** 2}")

# 矩阵运算
A = torch.randn(2, 3)
B = torch.randn(3, 2)
C = torch.matmul(A, B)  # 或 A @ B
print(f"矩阵乘法结果形状: {C.shape}")

# 5. 广播机制
print("\n5. 广播机制")
x = torch.ones(3, 4)
y = torch.ones(4)
result = x + y  # y会被广播成(3, 4)
print(f"广播加法结果形状: {result.shape}")

# 6. 就地操作（in-place）
print("\n6. 就地操作")
x = torch.tensor([1.0, 2.0, 3.0])
print(f"原始x: {x}")
x.add_(1)  # 就地加1（带下划线的操作都是就地操作）
print(f"就地加1后: {x}")

# 7. 张量与NumPy互转
print("\n7. 张量与NumPy互转")
import numpy as np

# Tensor -> NumPy
tensor = torch.ones(2, 3)
numpy_array = tensor.numpy()
print(f"张量转NumPy: {type(numpy_array)}")

# NumPy -> Tensor
np_arr = np.ones((2, 3))
tensor_from_np = torch.from_numpy(np_arr)
print(f"NumPy转张量: {type(tensor_from_np)}")

# 8. 练习：实现张量的标准化
print("\n8. 练习：标准化张量")
data = torch.randn(100)  # 生成100个随机数
mean = data.mean()
std = data.std()
normalized = (data - mean) / std
print(f"原始数据均值: {data.mean():.4f}, 标准差: {data.std():.4f}")
print(f"标准化后均值: {normalized.mean():.4f}, 标准差: {normalized.std():.4f}")

print("\n" + "=" * 50)
print("张量操作练习完成！")
print("=" * 50)
