"""
Day 1: PyTorch张量基础 - 张量创建
学习目标：掌握创建张量的各种方法
"""

import torch
import numpy as np

print("=" * 50)
print("PyTorch版本:", torch.__version__)
print("=" * 50)

# 1. 从Python列表创建张量
print("\n1. 从列表创建张量")
data = [[1, 2], [3, 4]]
tensor_from_list = torch.tensor(data)
print(f"从列表创建: {tensor_from_list}")
print(f"形状: {tensor_from_list.shape}")
print(f"数据类型: {tensor_from_list.dtype}")

# 2. 从NumPy数组创建
print("\n2. 从NumPy数组创建")
np_array = np.array([[1, 2], [3, 4]])
tensor_from_numpy = torch.from_numpy(np_array)
print(f"从NumPy创建: {tensor_from_numpy}")

# 3. 创建特殊张量
print("\n3. 创建特殊张量")

# 全零张量
zeros = torch.zeros(2, 3)
print(f"全零张量:\n{zeros}")

# 全一张量
ones = torch.ones(2, 3)
print(f"全一张量:\n{ones}")

# 单位矩阵
eye = torch.eye(3)
print(f"单位矩阵:\n{eye}")

# 随机张量
random_tensor = torch.rand(2, 3)  # [0, 1)均匀分布
print(f"随机张量:\n{random_tensor}")

# 标准正态分布
randn_tensor = torch.randn(2, 3)  # 标准正态分布
print(f"正态分布张量:\n{randn_tensor}")

# 4. 指定数据类型
print("\n4. 指定数据类型")
float_tensor = torch.ones(2, 3, dtype=torch.float32)
int_tensor = torch.ones(2, 3, dtype=torch.int64)
print(f"Float32张量: {float_tensor.dtype}")
print(f"Int64张量: {int_tensor.dtype}")

# 5. 创建与已有张量相同形状的张量
print("\n5. 创建相同形状的张量")
x = torch.rand(2, 3)
zeros_like = torch.zeros_like(x)
ones_like = torch.ones_like(x)
rand_like = torch.rand_like(x)
print(f"原张量形状: {x.shape}")
print(f"zeros_like形状: {zeros_like.shape}")

# 6. 序列张量
print("\n6. 序列张量")
arange_tensor = torch.arange(0, 10, 2)  # [0, 10), 步长2
print(f"arange: {arange_tensor}")

linspace_tensor = torch.linspace(0, 1, 5)  # [0, 1], 5个点
print(f"linspace: {linspace_tensor}")

# 7. 练习：创建一个3x3的随机矩阵，数值在[10, 20)之间
print("\n7. 练习：创建[10, 20)区间的随机矩阵")
random_range = torch.rand(3, 3) * 10 + 10  # rand生成[0,1)，变换到[10,20)
print(f"随机矩阵[10, 20):\n{random_range}")

print("\n" + "=" * 50)
print("Day 1 练习完成！")
print("=" * 50)
