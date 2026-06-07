# Week 1: PyTorch基础与深度学习入门

## 🎯 本周目标

掌握PyTorch基础操作，理解深度学习训练流程，完成MNIST手写数字识别项目。

## 📚 学习内容

### Day 1: 环境搭建 + PyTorch基础
- PyTorch安装（CPU/GPU版本）
- Tensor张量基础操作
- 与NumPy的转换

### Day 2: 张量操作与自动微分
- 张量的创建、索引、切片
- 广播机制
- 自动微分（autograd）
- 梯度计算

### Day 3: 神经网络基础
- nn.Module类
- 常用层：Linear、Conv2d、ReLU
- 损失函数
- 优化器

### Day 4: 训练循环
- 前向传播
- 损失计算
- 反向传播
- 参数更新
- 数据加载器（DataLoader）

### Day 5-6: MNIST项目实战
- 数据集加载
- 模型设计（CNN）
- 训练与验证
- 模型保存与加载
- 可视化结果

### Day 7: 周总结与复习
- 整理学习笔记
- 代码重构
- 准备下周内容

## 📁 项目结构

```
week1-pytorch-basics/
├── day1-tensor-basics/        # 张量基础
├── day2-autograd/             # 自动微分
├── day3-neural-network/       # 神经网络
├── day4-training-loop/        # 训练循环
├── day5-6-mnist-project/      # MNIST项目
│   ├── data/                  # 数据集
│   ├── models/                # 模型定义
│   ├── train.py               # 训练脚本
│   ├── test.py                # 测试脚本
│   └── visualize.py           # 可视化
├── exercises/                 # 练习题
└── README.md
```

## 🔧 环境要求

```bash
# Python 3.8+
python --version

# 安装PyTorch (CPU版本)
pip install torch torchvision torchaudio

# 安装其他依赖
pip install numpy matplotlib jupyter
```

## 📊 学习检查点

- [ ] 能创建和操作张量
- [ ] 理解自动微分机制
- [ ] 能用nn.Module构建神经网络
- [ ] 能编写完整的训练循环
- [ ] 完成MNIST项目（准确率>95%）

## 📚 学习资源

1. **官方教程**
   - [PyTorch 60分钟入门](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
   - [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)

2. **视频课程**
   - [动手学深度学习 - PyTorch版](https://d2l.ai/)
   - [Stanford CS231n](http://cs231n.stanford.edu/)

3. **推荐阅读**
   - 《深度学习入门：基于Python的理论与实现》

## 💡 学习建议

1. **边学边练**：每学一个概念立即写代码验证
2. **对比学习**：与NumPy对比理解PyTorch
3. **可视化**：用matplotlib绘制训练曲线
4. **记录笔记**：整理重点知识和常见错误
5. **提问思考**：为什么需要自动微分？为什么用SGD？

## 🎓 本周项目：MNIST手写数字识别

**项目目标：**
- 实现CNN模型识别手写数字
- 训练准确率达到95%以上
- 能可视化训练过程和预测结果
- 理解完整的深度学习流程

**技术要点：**
- 卷积神经网络
- 数据增强
- 模型保存与加载
- 训练曲线可视化

---

**开始日期:** 2026-05-08  
**预计完成:** 2026-05-14

---

# Week 1: PyTorch Basics & Deep Learning Introduction

## 🎯 Weekly Goals

Master PyTorch basic operations, understand the deep learning training process, and complete an MNIST handwritten digit recognition project.

## 📚 Learning Content

### Day 1: Environment Setup + PyTorch Basics
- PyTorch installation (CPU/GPU version)
- Tensor basic operations
- Conversion with NumPy

### Day 2: Tensor Operations and Automatic Differentiation
- Tensor creation, indexing, slicing
- Broadcasting mechanism
- Automatic differentiation (autograd)
- Gradient computation

### Day 3: Neural Network Basics
- nn.Module class
- Common layers: Linear, Conv2d, ReLU
- Loss functions
- Optimizers

### Day 4: Training Loop
- Forward propagation
- Loss calculation
- Backward propagation
- Parameter update
- DataLoader

### Day 5-6: MNIST Project Practice
- Dataset loading
- Model design (CNN)
- Training and validation
- Model saving and loading
- Result visualization

### Day 7: Weekly Summary and Review
- Organize learning notes
- Code refactoring
- Prepare for next week's content

## 📁 Project Structure

```
week1-pytorch-basics/
├── day1-tensor-basics/        # Tensor basics
├── day2-autograd/             # Automatic differentiation
├── day3-neural-network/       # Neural networks
├── day4-training-loop/        # Training loop
├── day5-6-mnist-project/      # MNIST project
│   ├── data/                  # Dataset
│   ├── models/                # Model definitions
│   ├── train.py               # Training script
│   ├── test.py                # Testing script
│   └── visualize.py           # Visualization
├── exercises/                 # Exercises
└── README.md
```

## 🔧 Environment Requirements

```bash
# Python 3.8+
python --version

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install other dependencies
pip install numpy matplotlib jupyter
```

## 📊 Learning Checkpoints

- [ ] Can create and manipulate tensors
- [ ] Understand automatic differentiation mechanism
- [ ] Can build neural networks with nn.Module
- [ ] Can write a complete training loop
- [ ] Complete MNIST project (accuracy > 95%)

## 📚 Learning Resources

1. **Official Tutorials**
   - [PyTorch 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
   - [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)

2. **Video Courses**
   - [Dive into Deep Learning - PyTorch Edition](https://d2l.ai/)
   - [Stanford CS231n](http://cs231n.stanford.edu/)

3. **Recommended Reading**
   - "Deep Learning from Scratch: Theory and Implementation with Python"

## 💡 Study Tips

1. **Learn by doing**: Write code to verify every concept as soon as you learn it
2. **Comparative learning**: Compare with NumPy to understand PyTorch
3. **Visualization**: Use matplotlib to plot training curves
4. **Take notes**: Organize key knowledge and common errors
5. **Ask questions**: Why do we need automatic differentiation? Why use SGD?

## 🎓 Weekly Project: MNIST Handwritten Digit Recognition

**Project Goals:**
- Implement a CNN model to recognize handwritten digits
- Achieve training accuracy above 95%
- Visualize the training process and prediction results
- Understand the complete deep learning workflow

**Technical Points:**
- Convolutional neural networks
- Data augmentation
- Model saving and loading
- Training curve visualization

---

**Start Date:** 2026-05-08  
**Expected Completion:** 2026-05-14
