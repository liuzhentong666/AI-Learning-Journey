"""
Day 3: 神经网络基础 - nn.Module 入门 | Day 3: Neural Network Basics - Introduction to nn.Module
学习目标：理解 nn.Module 是什么，以及如何定义自己的神经网络 | Learning objectives: Understand what nn.Module is and how to define your own neural network

技术栈： | Tech stack:
- PyTorch (torch)
- torch.nn（神经网络层与模型基类 nn.Module） | torch.nn (neural network layers and the model base class nn.Module)
"""

import torch
import torch.nn as nn

print("=" * 50)
print("PyTorch 神经网络基础 - nn.Module | PyTorch Neural Network Basics - nn.Module")
print("=" * 50)


# =============================================================================
# 1. 为什么需要 nn.Module？ | 1. Why do we need nn.Module?
# =============================================================================
# Day 2 我们手动创建 requires_grad=True 的张量，手写梯度下降。 | Day 2 we manually created requires_grad=True tensors and hand-coded gradient descent.
# 但真实神经网络有成百上千个参数，手写不现实。 | But real neural networks have hundreds or thousands of parameters — hand-coding is impractical.
#
# nn.Module 是 PyTorch 提供的「模型容器」： | nn.Module is the "model container" provided by PyTorch:
# - 自动管理所有层的参数（权重、偏置） | - Automatically manages all layer parameters (weights, biases)
# - 提供 .parameters() 给优化器 | - Provides .parameters() to optimizers
# - 支持 .train() / .eval() 模式切换 | - Supports .train() / .eval() mode switching
# - 支持 GPU 迁移 .to(device) | - Supports GPU transfer via .to(device)
#
# 技术栈：torch.nn.Module | Tech stack: torch.nn.Module

print("\n1. 为什么需要 nn.Module？ | 1. Why do we need nn.Module?")
print("-" * 40)
print("Day 2：手动管理参数和梯度 | Day 2: Manually manage parameters and gradients")
print("Day 3：用 nn.Module 自动管理，专注设计网络结构 | Day 3: Use nn.Module for automatic management, focus on designing network architecture")


# =============================================================================
# 2. 最简单的 nn.Module 示例 | 2. Simplest nn.Module Example
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 先从一个「只有一层 Linear」的最小模型开始， | Start with a minimal model having just one Linear layer,
# 理解 nn.Module 的两个核心方法：__init__ 和 forward。 | to understand nn.Module's two core methods: __init__ and forward.
#
# 技术栈：torch.nn.Module, torch.nn.Linear | Tech stack: torch.nn.Module, torch.nn.Linear

print("\n2. 最简单的 nn.Module 示例 | 2. Simplest nn.Module Example")
print("-" * 40)


class SimpleLinearModel(nn.Module):
    """
    最简单的线性模型：y = Wx + b | Simplest linear model: y = Wx + b

    技术栈：继承 torch.nn.Module，使用 torch.nn.Linear | Tech stack: Inherit torch.nn.Module, use torch.nn.Linear
    """

    def __init__(self):
        # 为什么要写 super().__init__()？ | Why write super().__init__()?
        # nn.Module 内部需要注册子模块和参数，不调用会导致模型无法正常工作。 | nn.Module internally needs to register submodules and parameters; not calling it will cause the model to malfunction.
        super().__init__()

        # 为什么要在这里定义层？ | Why define layers here?
        # __init__ 只在创建模型时执行一次，层（及其参数）在这里被创建并注册。 | __init__ executes only once when the model is created; layers (and their parameters) are created and registered here.
        # Linear(3, 1)：3 个输入特征 → 1 个输出 | Linear(3, 1): 3 input features → 1 output
        self.linear = nn.Linear(in_features=3, out_features=1)

    def forward(self, x):
        # 为什么要写 forward？ | Why write forward?
        # 每次调用 model(x) 时，PyTorch 会自动执行 forward，定义数据如何流过各层。 | Every time model(x) is called, PyTorch automatically runs forward, defining how data flows through each layer.
        return self.linear(x)


# 为什么要实例化并传入数据？ | Why instantiate and pass in data?
# 验证模型能接收张量并输出预测值，这是所有神经网络的基本工作方式。 | Verify that the model can accept tensors and output predictions — the fundamental way all neural networks work.
model = SimpleLinearModel()
x = torch.tensor([[1.0, 2.0, 3.0]])
print(x.shape)
y = model(x)

print(f"输入 x: {x} | Input x: {x}")
print(f"输出 y: {y} | Output y: {y}")
print(f"输出形状: {y.shape} | Output shape: {y.shape}")  # (1, 1)：1 个样本，1 个预测值 | (1, 1): 1 sample, 1 prediction


# =============================================================================
# 3. 多层网络：在 __init__ 中堆叠层 | 3. Multi-layer Network: Stacking Layers in __init__
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 真实网络有多层。在 __init__ 里定义好层，在 forward 里按顺序调用。 | Real networks have multiple layers. Define layers in __init__, call them in sequence in forward.
#
# 技术栈：torch.nn.Module, torch.nn.Linear, torch.nn.ReLU | Tech stack: torch.nn.Module, torch.nn.Linear, torch.nn.ReLU

print("\n3. 多层网络示例 | 3. Multi-layer Network Example")
print("-" * 40)


class SmallMLP(nn.Module):
    """
    小型多层感知机（MLP）：3 → 8 → 4 → 1 | Small Multi-Layer Perceptron (MLP): 3 → 8 → 4 → 1

    技术栈：torch.nn.Linear（全连接层）, torch.nn.ReLU（激活函数） | Tech stack: torch.nn.Linear (fully connected layer), torch.nn.ReLU (activation function)
    """

    def __init__(self):
        super().__init__()

        # 为什么要分三层？ | Why three layers?
        # 每层提取不同抽象级别的特征：原始输入 → 中间表示 → 最终预测 | Each layer extracts features at different abstraction levels: raw input → intermediate representation → final prediction
        self.layer1 = nn.Linear(3, 8)   # 输入层 → 隐藏层1 | Input layer → Hidden layer 1
        self.layer2 = nn.Linear(8, 4)   # 隐藏层1 → 隐藏层2 | Hidden layer 1 → Hidden layer 2
        self.layer3 = nn.Linear(4, 1)   # 隐藏层2 → 输出层 | Hidden layer 2 → Output layer
        self.relu = nn.ReLU()           # 激活函数，引入非线性 | Activation function, introduces non-linearity

    def forward(self, x):
        # 为什么要交替使用 Linear 和 ReLU？ | Why alternate between Linear and ReLU?
        # Linear 做加权求和，ReLU 引入非线性，两者配合才能学习复杂规律。 | Linear does weighted summation, ReLU introduces non-linearity — both together enable learning complex patterns.
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)  # 输出层通常不加 ReLU（回归任务） | Output layer usually doesn't use ReLU (regression task)
        return x


mlp = SmallMLP()
x = torch.randn(2, 3)  # 2 个样本，每个 3 个特征 | 2 samples, each with 3 features
output = mlp(x)

print(f"输入形状: {x.shape} | Input shape: {x.shape}")
print(f"输出形状: {output.shape} | Output shape: {output.shape}")  # (2, 1)：2 个样本各 1 个预测 | (2, 1): 2 samples each with 1 prediction


# =============================================================================
# 4. 查看模型结构和参数 | 4. Inspecting Model Structure and Parameters
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 调试和优化时需要知道模型有哪些层、参数有多少，避免「黑盒」。 | When debugging and optimizing, you need to know what layers the model has and how many parameters — avoid the "black box".
#
# 技术栈：nn.Module 的 .parameters(), 内置 print(model) | Tech stack: nn.Module's .parameters(), built-in print(model)

print("\n4. 查看模型结构 | 4. Inspecting Model Structure")
print("-" * 40)

print("模型结构: | Model structure:")
print(mlp)

# 为什么要遍历 parameters()？ | Why iterate over parameters()?
# 优化器需要所有可训练参数；numel() 统计参数总数，反映模型复杂度。 | Optimizers need all trainable parameters; numel() counts total parameters, reflecting model complexity.
total_params = sum(p.numel() for p in mlp.parameters())
trainable_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)

print(f"\n总参数量: {total_params} | Total parameters: {total_params}")
print(f"可训练参数量: {trainable_params} | Trainable parameters: {trainable_params}")

print("\n各层参数详情: | Layer parameter details:")
for name, param in mlp.named_parameters():
    print(f"  {name}: 形状 {param.shape} | shape {param.shape}")


# =============================================================================
# 5. train() 和 eval() 模式 | 5. train() and eval() Modes
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 某些层（如 Dropout、BatchNorm）在训练和推理时行为不同。 | Certain layers (like Dropout, BatchNorm) behave differently during training and inference.
# .train() 训练模式，.eval() 推理模式。Day 4+ 会详细用到。 | .train() for training mode, .eval() for inference mode. Will be used in detail from Day 4 onward.
#
# 技术栈：nn.Module 的 .train() 和 .eval() | Tech stack: nn.Module's .train() and .eval()

print("\n5. train() 和 eval() 模式 | 5. train() and eval() Modes")
print("-" * 40)

print(f"默认 training 属性: {mlp.training} | Default training attribute: {mlp.training}")

mlp.eval()
print(f"调用 eval() 后: {mlp.training} | After calling eval(): {mlp.training}")

mlp.train()
print(f"调用 train() 后: {mlp.training} | After calling train(): {mlp.training}")

print("\n说明： | Notes:")
print("- model.train()：训练时使用（Dropout 随机丢弃神经元等） | - model.train(): Use during training (Dropout randomly drops neurons, etc.)")
print("- model.eval()：推理/验证时使用（关闭 Dropout 等随机行为） | - model.eval(): Use during inference/validation (disables Dropout and other random behaviors)")


# =============================================================================
# 6. 用 nn.Sequential 简化写法 | 6. Simplifying with nn.Sequential
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 当层只是简单串联、无分支时，Sequential 比手写 forward 更简洁。 | When layers are simply sequential with no branches, Sequential is cleaner than writing forward manually.
#
# 技术栈：torch.nn.Sequential | Tech stack: torch.nn.Sequential

print("\n6. 使用 nn.Sequential 简化模型 | 6. Simplifying Models with nn.Sequential")
print("-" * 40)

# 为什么要用 Sequential？ | Why use Sequential?
# 层按顺序执行，等价于 SmallMLP，代码更短。 | Layers execute in order, equivalent to SmallMLP, with shorter code.
sequential_model = nn.Sequential(
    nn.Linear(3, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)

x = torch.randn(2, 3)
output = sequential_model(x)
print(f"Sequential 模型输出形状: {output.shape} | Sequential model output shape: {output.shape}")
print("\nSequential 模型结构: | Sequential model structure:")
print(sequential_model)


# =============================================================================
# 7. 练习：定义一个 784 → 128 → 10 的分类网络 | 7. Exercise: Define a 784 → 128 → 10 Classification Network
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 784 = 28×28 像素展平，10 = 10 类（如 MNIST 数字 0-9），为 Day 5-6 做准备。 | 784 = 28×28 pixels flattened, 10 = 10 classes (e.g. MNIST digits 0-9), preparing for Day 5-6.
#
# 技术栈：torch.nn.Module, torch.nn.Linear, torch.nn.ReLU | Tech stack: torch.nn.Module, torch.nn.Linear, torch.nn.ReLU

print("\n7. 练习：MNIST 风格的分类网络 | 7. Exercise: MNIST-style Classification Network")
print("-" * 40)


class MNISTClassifier(nn.Module):
    """
    784 维输入 → 128 维隐藏 → 10 类输出 | 784-dim input → 128-dim hidden → 10-class output

    技术栈：torch.nn.Module, Linear, ReLU | Tech stack: torch.nn.Module, Linear, ReLU
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 为什么要 view(-1, 784)？ | Why view(-1, 784)?
        # 图像可能是 (batch, 1, 28, 28)，展平成 (batch, 784) 才能进 Linear。 | Images may be (batch, 1, 28, 28); flatten to (batch, 784) to feed into Linear.
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # 输出 10 个 logits，不加 softmax（CrossEntropyLoss 内部处理） | Output 10 logits, no softmax (CrossEntropyLoss handles it internally)
        return x


classifier = MNISTClassifier()
fake_image = torch.randn(4, 1, 28, 28)  # 4 张 28×28 灰度图 | 4 grayscale images of 28×28
logits = classifier(fake_image)

print(f"输入形状: {fake_image.shape} | Input shape: {fake_image.shape}")
print(f"输出 logits 形状: {logits.shape} | Output logits shape: {logits.shape}")  # (4, 10)
print(f"参数量: {sum(p.numel() for p in classifier.parameters())} | Parameter count: {sum(p.numel() for p in classifier.parameters())}")


print("\n" + "=" * 50)
print("nn.Module 基础学习完成！ | nn.Module basics learning complete!")
print("=" * 50)
print("\n核心要点： | Key takeaways:")
print("1. 所有模型继承 nn.Module，实现 __init__ 和 forward | 1. All models inherit nn.Module, implement __init__ and forward")
print("2. __init__ 定义层，forward 定义数据流向 | 2. __init__ defines layers, forward defines data flow")
print("3. model.parameters() 供优化器使用 | 3. model.parameters() for use by optimizers")
print("4. model.train() / model.eval() 切换训练与推理模式 | 4. model.train() / model.eval() switch between training and inference modes")
print("5. nn.Sequential 适合简单串联结构 | 5. nn.Sequential suits simple sequential architectures")
