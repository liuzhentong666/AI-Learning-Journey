"""
Day 1: PyTorch张量练习答案
包含第一天练习的详细解答和示例
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def exercise1_tensor_creation():
    """
    练习1：创建特殊张量
    1. 创建5x5的单位矩阵
    2. 创建3x4的随机矩阵，值在[-1, 1]之间
    3. 创建从0到100的序列，步长为5
    """
    print("=" * 50)
    print("练习1：张量创建")
    print("=" * 50)
    
    # 1. 5x5单位矩阵
    identity_matrix = torch.eye(5)
    print("5x5单位矩阵:\n", identity_matrix)
    
    # 2. 3x4的[-1, 1]随机矩阵
    random_matrix = torch.rand(3, 4) * 2 - 1
    print("\n3x4的随机矩阵([-1, 1]):\n", random_matrix)
    
    # 3. 从0到100的序列，步长为5
    sequence = torch.arange(0, 101, 5)
    print("\n0到100的序列(步长5):\n", sequence)
    
    return identity_matrix, random_matrix, sequence

def exercise2_tensor_operations():
    """
    练习2：张量操作
    给定张量 x = torch.arange(24).reshape(2, 3, 4)
    1. 提取第一个"页"的所有数据
    2. 将其reshape成(6, 4)
    3. 计算每列的平均值
    """
    print("\n" + "=" * 50)
    print("练习2：张量操作")
    print("=" * 50)
    
    x = torch.arange(24).reshape(2, 3, 4)
    
    # 1. 提取第一个"页"的所有数据
    first_page = x[0]
    print("第一个页的数据:\n", first_page)
    
    # 2. reshape成(6, 4)
    reshaped_x = x.reshape(6, 4)
    print("\nreshape后的结果:\n", reshaped_x)
    
    # 3. 计算每列的平均值
    col_means = reshaped_x.float().mean(dim=0)
    print("\n每列平均值:\n", col_means)
    
    return first_page, reshaped_x, col_means

def exercise3_tensor_standardization():
    """
    练习3：实现张量的标准化
    创建一个张量，实现标准化（减去均值，除以标准差）
    验证标准化后的张量均值为0，标准差为1
    """
    print("\n" + "=" * 50)
    print("练习3：张量标准化")
    print("=" * 50)
    
    # 生成随机张量
    data = torch.randn(100)
    
    # 标准化函数
    def standardize(tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / std
    
    # 标准化
    normalized = standardize(data)
    
    print("原始数据:")
    print(f"均值: {data.mean():.4f}")
    print(f"标准差: {data.std():.4f}")
    
    print("\n标准化后:")
    print(f"均值: {normalized.mean():.4f}")
    print(f"标准差: {normalized.std():.4f}")
    
    return normalized

def bonus_exercise():
    """
    附加练习：张量可视化与高级操作
    1. 创建正态分布随机张量
    2. 绘制直方图
    3. 计算累积分布
    """
    print("\n" + "=" * 50)
    print("附加练习：张量可视化")
    print("=" * 50)
    
    # 创建两个不同参数的正态分布
    x1 = torch.randn(1000)  # 标准正态
    x2 = torch.randn(1000) * 2 + 1  # 方差为2，均值为1
    
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
    # 运行所有练习
    print("\n" + "=" * 50)
    print("Day 1 PyTorch张量练习 - 答案")
    print("=" * 50)

    print("\n【练习1】张量创建")
    exercise1_tensor_creation()

    print("\n【练习2】张量操作")
    exercise2_tensor_operations()

    print("\n【练习3】张量标准化")
    exercise3_tensor_standardization()

    print("\n【附加练习】张量可视化")
    bonus_exercise()

if __name__ == "__main__":
    main()

"""
练习解析：

练习1：张量创建
- torch.eye(): 创建单位矩阵
- torch.rand() * 2 - 1: 生成[-1, 1]的随机数
- torch.arange(): 生成等差数列

练习2：张量操作
- x[0]: 提取多维张量的第一页
- reshape(): 改变张量形状
- mean(dim=0): 计算每列平均值

练习3：张量标准化
- 标准化公式: (x - mean) / std
- 目标是使数据变为零均值、单位方差
"""