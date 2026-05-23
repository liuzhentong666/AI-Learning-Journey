"""
Day 3: 神经网络基础 - nn.Module 入门
学习目标：理解 nn.Module 是什么，以及如何定义自己的神经网络

技术栈：
- PyTorch (torch)
- torch.nn（神经网络层与模型基类 nn.Module）
"""

import torch
import torch.nn as nn

print("=" * 50)
print("PyTorch 神经网络基础 - nn.Module")
print("=" * 50)


# =============================================================================
# 1. 为什么需要 nn.Module？
# =============================================================================
# Day 2 我们手动创建 requires_grad=True 的张量，手写梯度下降。
# 但真实神经网络有成百上千个参数，手写不现实。
#
# nn.Module 是 PyTorch 提供的「模型容器」：
# - 自动管理所有层的参数（权重、偏置）
# - 提供 .parameters() 给优化器
# - 支持 .train() / .eval() 模式切换
# - 支持 GPU 迁移 .to(device)
#
# 技术栈：torch.nn.Module

print("\n1. 为什么需要 nn.Module？")
print("-" * 40)
print("Day 2：手动管理参数和梯度")
print("Day 3：用 nn.Module 自动管理，专注设计网络结构")


# =============================================================================
# 2. 最简单的 nn.Module 示例
# =============================================================================
# 为什么要写这段代码？
# 先从一个「只有一层 Linear」的最小模型开始，
# 理解 nn.Module 的两个核心方法：__init__ 和 forward。
#
# 技术栈：torch.nn.Module, torch.nn.Linear

print("\n2. 最简单的 nn.Module 示例")
print("-" * 40)


class SimpleLinearModel(nn.Module):
    """
    最简单的线性模型：y = Wx + b

    技术栈：继承 torch.nn.Module，使用 torch.nn.Linear
    """

    def __init__(self):
        # 为什么要写 super().__init__()？
        # nn.Module 内部需要注册子模块和参数，不调用会导致模型无法正常工作。
        super().__init__()

        # 为什么要在这里定义层？
        # __init__ 只在创建模型时执行一次，层（及其参数）在这里被创建并注册。
        # Linear(3, 1)：3 个输入特征 → 1 个输出
        self.linear = nn.Linear(in_features=3, out_features=1)

    def forward(self, x):
        # 为什么要写 forward？
        # 每次调用 model(x) 时，PyTorch 会自动执行 forward，定义数据如何流过各层。
        return self.linear(x)


# 为什么要实例化并传入数据？
# 验证模型能接收张量并输出预测值，这是所有神经网络的基本工作方式。
model = SimpleLinearModel()
x = torch.tensor([[1.0, 2.0, 3.0]])
y = model(x)

print(f"输入 x: {x}")
print(f"输出 y: {y}")
print(f"输出形状: {y.shape}")  # (1, 1)：1 个样本，1 个预测值


# =============================================================================
# 3. 多层网络：在 __init__ 中堆叠层
# =============================================================================
# 为什么要写这段代码？
# 真实网络有多层。在 __init__ 里定义好层，在 forward 里按顺序调用。
#
# 技术栈：torch.nn.Module, torch.nn.Linear, torch.nn.ReLU

print("\n3. 多层网络示例")
print("-" * 40)


class SmallMLP(nn.Module):
    """
    小型多层感知机（MLP）：3 → 8 → 4 → 1

    技术栈：torch.nn.Linear（全连接层）, torch.nn.ReLU（激活函数）
    """

    def __init__(self):
        super().__init__()

        # 为什么要分三层？
        # 每层提取不同抽象级别的特征：原始输入 → 中间表示 → 最终预测
        self.layer1 = nn.Linear(3, 8)   # 输入层 → 隐藏层1
        self.layer2 = nn.Linear(8, 4)   # 隐藏层1 → 隐藏层2
        self.layer3 = nn.Linear(4, 1)   # 隐藏层2 → 输出层
        self.relu = nn.ReLU()           # 激活函数，引入非线性

    def forward(self, x):
        # 为什么要交替使用 Linear 和 ReLU？
        # Linear 做加权求和，ReLU 引入非线性，两者配合才能学习复杂规律。
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)  # 输出层通常不加 ReLU（回归任务）
        return x


mlp = SmallMLP()
x = torch.randn(2, 3)  # 2 个样本，每个 3 个特征
output = mlp(x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")  # (2, 1)：2 个样本各 1 个预测


# =============================================================================
# 4. 查看模型结构和参数
# =============================================================================
# 为什么要写这段代码？
# 调试和优化时需要知道模型有哪些层、参数有多少，避免「黑盒」。
#
# 技术栈：nn.Module 的 .parameters(), 内置 print(model)

print("\n4. 查看模型结构")
print("-" * 40)

print("模型结构:")
print(mlp)

# 为什么要遍历 parameters()？
# 优化器需要所有可训练参数；numel() 统计参数总数，反映模型复杂度。
total_params = sum(p.numel() for p in mlp.parameters())
trainable_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)

print(f"\n总参数量: {total_params}")
print(f"可训练参数量: {trainable_params}")

print("\n各层参数详情:")
for name, param in mlp.named_parameters():
    print(f"  {name}: 形状 {param.shape}")


# =============================================================================
# 5. train() 和 eval() 模式
# =============================================================================
# 为什么要写这段代码？
# 某些层（如 Dropout、BatchNorm）在训练和推理时行为不同。
# .train() 训练模式，.eval() 推理模式。Day 4+ 会详细用到。
#
# 技术栈：nn.Module 的 .train() 和 .eval()

print("\n5. train() 和 eval() 模式")
print("-" * 40)

print(f"默认 training 属性: {mlp.training}")

mlp.eval()
print(f"调用 eval() 后: {mlp.training}")

mlp.train()
print(f"调用 train() 后: {mlp.training}")

print("\n说明：")
print("- model.train()：训练时使用（Dropout 随机丢弃神经元等）")
print("- model.eval()：推理/验证时使用（关闭 Dropout 等随机行为）")


# =============================================================================
# 6. 用 nn.Sequential 简化写法
# =============================================================================
# 为什么要写这段代码？
# 当层只是简单串联、无分支时，Sequential 比手写 forward 更简洁。
#
# 技术栈：torch.nn.Sequential

print("\n6. 使用 nn.Sequential 简化模型")
print("-" * 40)

# 为什么要用 Sequential？
# 层按顺序执行，等价于 SmallMLP，代码更短。
sequential_model = nn.Sequential(
    nn.Linear(3, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)

x = torch.randn(2, 3)
output = sequential_model(x)
print(f"Sequential 模型输出形状: {output.shape}")
print("\nSequential 模型结构:")
print(sequential_model)


# =============================================================================
# 7. 练习：定义一个 784 → 128 → 10 的分类网络
# =============================================================================
# 为什么要写这段代码？
# 784 = 28×28 像素展平，10 = 10 类（如 MNIST 数字 0-9），为 Day 5-6 做准备。
#
# 技术栈：torch.nn.Module, torch.nn.Linear, torch.nn.ReLU

print("\n7. 练习：MNIST 风格的分类网络")
print("-" * 40)


class MNISTClassifier(nn.Module):
    """
    784 维输入 → 128 维隐藏 → 10 类输出

    技术栈：torch.nn.Module, Linear, ReLU
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 为什么要 view(-1, 784)？
        # 图像可能是 (batch, 1, 28, 28)，展平成 (batch, 784) 才能进 Linear。
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # 输出 10 个 logits，不加 softmax（CrossEntropyLoss 内部处理）
        return x


classifier = MNISTClassifier()
fake_image = torch.randn(4, 1, 28, 28)  # 4 张 28×28 灰度图
logits = classifier(fake_image)

print(f"输入形状: {fake_image.shape}")
print(f"输出 logits 形状: {logits.shape}")  # (4, 10)
print(f"参数量: {sum(p.numel() for p in classifier.parameters())}")


print("\n" + "=" * 50)
print("nn.Module 基础学习完成！")
print("=" * 50)
print("\n核心要点：")
print("1. 所有模型继承 nn.Module，实现 __init__ 和 forward")
print("2. __init__ 定义层，forward 定义数据流向")
print("3. model.parameters() 供优化器使用")
print("4. model.train() / model.eval() 切换训练与推理模式")
print("5. nn.Sequential 适合简单串联结构")
