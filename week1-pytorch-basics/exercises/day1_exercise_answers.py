"""
Day 1: PyTorch张量练习答案 | Day 1: PyTorch Tensor Exercise Answers
包含第一天练习的详细解答和示例 | Contains detailed solutions and examples for Day 1 exercises
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def exercise1_tensor_creation():
    """
    练习1：创建特殊张量 | Exercise 1: Create special tensors
    1. 创建5x5的单位矩阵 | 1. Create a 5x5 identity matrix
    2. 创建3x4的随机矩阵，值在[-1, 1]之间 | 2. Create a 3x4 random matrix with values in [-1, 1]
    3. 创建从0到100的序列，步长为5 | 3. Create a sequence from 0 to 100 with step 5
    """
    print("=" * 50)
    print("练习1：张量创建 | Exercise 1: Tensor Creation")
    print("=" * 50)
    
    # 1. 5x5单位矩阵 | 1. 5x5 identity matrix
    identity_matrix = torch.eye(5)
    print("5x5单位矩阵 | 5x5 identity matrix:\n", identity_matrix)
    
    # 2. 3x4的[-1, 1]随机矩阵 | 2. 3x4 random matrix in [-1, 1]
    random_matrix = torch.rand(3, 4) * 2 - 1
    print("\n3x4的随机矩阵([-1, 1]) | 3x4 random matrix ([-1, 1]):\n", random_matrix)
    
    # 3. 从0到100的序列，步长为5 | 3. Sequence from 0 to 100, step 5
    sequence = torch.arange(0, 101, 5)
    print("\n0到100的序列(步长5) | Sequence from 0 to 100 (step 5):\n", sequence)
    
    return identity_matrix, random_matrix, sequence

def exercise2_tensor_operations():
    """
    练习2：张量操作 | Exercise 2: Tensor operations
    给定张量 x = torch.arange(24).reshape(2, 3, 4) | Given tensor x = torch.arange(24).reshape(2, 3, 4)
    1. 提取第一个"页"的所有数据 | 1. Extract all data from the first "page"
    2. 将其reshape成(6, 4) | 2. Reshape it to (6, 4)
    3. 计算每列的平均值 | 3. Compute the mean of each column
    """
    print("\n" + "=" * 50)
    print("练习2：张量操作 | Exercise 2: Tensor Operations")
    print("=" * 50)
    
    x = torch.arange(24).reshape(2, 3, 4)
    
    # 1. 提取第一个"页"的所有数据 | 1. Extract all data from the first "page"
    first_page = x[0]
    print("第一个页的数据 | First page data:\n", first_page)
    
    # 2. reshape成(6, 4) | 2. Reshape to (6, 4)
    reshaped_x = x.reshape(6, 4)
    print("\nreshape后的结果 | Reshaped result:\n", reshaped_x)
    
    # 3. 计算每列的平均值 | 3. Compute column means
    col_means = reshaped_x.float().mean(dim=0)
    print("\n每列平均值 | Column means:\n", col_means)
    
    return first_page, reshaped_x, col_means

def exercise3_tensor_standardization():
    """
    练习3：实现张量的标准化 | Exercise 3: Implement tensor standardization
    创建一个张量，实现标准化（减去均值，除以标准差） | Create a tensor and implement standardization (subtract mean, divide by std)
    验证标准化后的张量均值为0，标准差为1 | Verify that the standardized tensor has mean 0 and std 1
    """
    print("\n" + "=" * 50)
    print("练习3：张量标准化 | Exercise 3: Tensor Standardization")
    print("=" * 50)
    
    # 生成随机张量 | Generate random tensor
    data = torch.randn(100)
    
    # 标准化函数 | Standardization function
    def standardize(tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / std
    
    # 标准化 | Standardize
    normalized = standardize(data)
    
    print("原始数据 | Original data:")
    print(f"均值 | Mean: {data.mean():.4f}")
    print(f"标准差 | Std: {data.std():.4f}")
    
    print("\n标准化后 | After standardization:")
    print(f"均值 | Mean: {normalized.mean():.4f}")
    print(f"标准差 | Std: {normalized.std():.4f}")
    
    return normalized

def bonus_exercise():
    """
    附加练习：张量可视化与高级操作 | Bonus exercise: Tensor visualization and advanced operations
    1. 创建正态分布随机张量 | 1. Create normally distributed random tensor
    2. 绘制直方图 | 2. Plot histogram
    3. 计算累积分布 | 3. Compute cumulative distribution
    """
    print("\n" + "=" * 50)
    print("附加练习：张量可视化 | Bonus Exercise: Tensor Visualization")
    print("=" * 50)
    
    # 创建两个不同参数的正态分布 | Create two normal distributions with different parameters
    x1 = torch.randn(1000)  # 标准正态 | Standard normal
    x2 = torch.randn(1000) * 2 + 1  # 方差为2，均值为1 | Variance=2, mean=1
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(x1.numpy(), bins=30, alpha=0.7, color='blue', density=True)
    plt.title('标准正态分布')
    plt.xlabel('值')
    plt.ylabel('频率')
    
    plt.subplot(1, 2, 2)
    plt.hist(x2.numpy(), bins=30, alpha=0.7, color='red', density=True)
    plt.title('方差=2, 均值=1的正态分布')
    plt.xlabel('值')
    plt.ylabel('频率')
    
    plt.tight_layout()
    plt.show()

def main():
    # 运行所有练习 | Run all exercises
    print("\n" + "=" * 50)
    print("Day 1 PyTorch张量练习 - 答案 | Day 1 PyTorch Tensor Exercises - Answers")
    print("=" * 50)

    print("\n【练习1】张量创建 | [Exercise 1] Tensor Creation")
    exercise1_tensor_creation()

    print("\n【练习2】张量操作 | [Exercise 2] Tensor Operations")
    exercise2_tensor_operations()

    print("\n【练习3】张量标准化 | [Exercise 3] Tensor Standardization")
    exercise3_tensor_standardization()

    print("\n【附加练习】张量可视化 | [Bonus Exercise] Tensor Visualization")
    bonus_exercise()

if __name__ == "__main__":
    main()

"""
练习解析 | Exercise Analysis:

练习1：张量创建 | Exercise 1: Tensor Creation
- torch.eye(): 创建单位矩阵 | Create identity matrix
- torch.rand() * 2 - 1: 生成[-1, 1]的随机数 | Generate random numbers in [-1, 1]
- torch.arange(): 生成等差数列 | Generate arithmetic sequence

练习2：张量操作 | Exercise 2: Tensor Operations
- x[0]: 提取多维张量的第一页 | Extract the first page of a multi-dimensional tensor
- reshape(): 改变张量形状 | Change tensor shape
- mean(dim=0): 计算每列平均值 | Compute column-wise mean

练习3：张量标准化 | Exercise 3: Tensor Standardization
- 标准化公式: (x - mean) / std | Standardization formula: (x - mean) / std
- 目标是使数据变为零均值、单位方差 | Goal: transform data to zero mean, unit variance
"""
