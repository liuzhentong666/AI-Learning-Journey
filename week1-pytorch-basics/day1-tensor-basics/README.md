# Day 1: PyTorch张量基础

## 学习目标
- 理解张量(Tensor)的概念
- 掌握张量的创建、操作和变形
- 了解PyTorch与NumPy的关系
- 学习GPU加速基础

## 练习文件

### 01_tensor_creation.py
学习各种创建张量的方法：
- 从列表/NumPy创建
- 创建特殊张量（全0、全1、随机等）
- 指定数据类型
- 创建序列张量

**运行：**
```bash
python 01_tensor_creation.py
```

### 02_tensor_operations.py
学习张量的各种操作：
- 索引和切片
- 形状变换（reshape、view、squeeze、unsqueeze）
- 张量拼接（cat、stack）
- 数学运算
- 广播机制
- 与NumPy互转

**运行：**
```bash
python 02_tensor_operations.py
```

### 03_gpu_basics.py
了解GPU加速基础：
- CUDA可用性检查
- 设备选择
- 张量在设备间移动
- CPU vs GPU性能对比

**运行：**
```bash
python 03_gpu_basics.py
```

## 核心概念

### 1. 什么是张量？
张量是PyTorch中的基本数据结构，类似于NumPy的ndarray，但可以在GPU上运行。

```python
# 0维张量（标量）
scalar = torch.tensor(3.14)

# 1维张量（向量）
vector = torch.tensor([1, 2, 3])

# 2维张量（矩阵）
matrix = torch.tensor([[1, 2], [3, 4]])

# 3维张量（立方体）
cube = torch.randn(2, 3, 4)
```

### 2. 张量属性
- `shape` / `size()`: 形状
- `dtype`: 数据类型
- `device`: 所在设备（CPU/GPU）
- `requires_grad`: 是否需要梯度（下节课讲）

### 3. PyTorch vs NumPy

| 特性 | PyTorch | NumPy |
|------|---------|-------|
| 基本单位 | Tensor | ndarray |
| GPU支持 | ✅ | ❌ |
| 自动微分 | ✅ | ❌ |
| 深度学习 | ✅ | ❌ |
| 科学计算 | ✅ | ✅ |

### 4. 常用操作速查

```python
# 创建
torch.tensor(data)           # 从数据创建
torch.zeros(shape)           # 全0
torch.ones(shape)            # 全1
torch.rand(shape)            # [0,1)随机
torch.randn(shape)           # 标准正态分布

# 形状
x.shape / x.size()           # 查看形状
x.reshape(shape)             # 改变形状
x.view(shape)                # 视图（共享内存）
x.unsqueeze(dim)             # 增加维度
x.squeeze(dim)               # 删除维度

# 拼接
torch.cat([a, b], dim)       # 拼接
torch.stack([a, b], dim)     # 堆叠

# 运算
x + y, x - y, x * y, x / y   # 逐元素
torch.matmul(a, b) / a @ b   # 矩阵乘法
x.sum(), x.mean(), x.max()   # 聚合运算

# 设备
x.to(device)                 # 移动设备
x.cuda()                     # 移到GPU
x.cpu()                      # 移到CPU
```

## 练习任务

完成以下练习以巩固知识：

1. **张量创建练习**
   - 创建一个5x5的单位矩阵
   - 创建一个3x4的随机矩阵，值在[-1, 1]之间
   - 创建一个序列张量，从0到100，步长为5

2. **张量操作练习**
   - 给定张量 `x = torch.arange(24).reshape(2, 3, 4)`
   - 提取第一个"页"的所有数据
   - 将其reshape成(6, 4)
   - 计算每列的平均值

3. **实战练习**
   - 实现一个函数，输入任意形状的张量，返回标准化后的张量
   - 标准化公式：`(x - mean) / std`

**答案参考：** 见 `exercises/day1_answers.py`

## 学习资源

- [PyTorch张量文档](https://pytorch.org/docs/stable/tensors.html)
- [PyTorch教程 - Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [从NumPy到PyTorch](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)

## 检查点

完成今天的学习后，你应该能够：
- [ ] 创建各种类型的张量
- [ ] 对张量进行索引、切片、变形
- [ ] 理解广播机制
- [ ] 在CPU和GPU间移动张量
- [ ] 知道何时使用PyTorch而非NumPy

---

**下一步：** Day 2 - 自动微分机制（autograd）
