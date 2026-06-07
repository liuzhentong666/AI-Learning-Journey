"""
Day 3: 神经网络基础 - 常用层详解 | Day 3: Neural Network Basics - Common Layers Explained
学习目标：掌握 Linear、Conv2d、ReLU 等常用层，并搭建 MLP 和 CNN | Learning Goals: Master Linear, Conv2d, ReLU and other common layers, and build MLP and CNN

技术栈： | Tech Stack:
- PyTorch (torch)
- torch.nn（Linear, Conv2d, ReLU, MaxPool2d, Flatten, Sequential） | torch.nn (Linear, Conv2d, ReLU, MaxPool2d, Flatten, Sequential)
- torch.nn.functional（函数式激活等） | torch.nn.functional (functional activations, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 50)
print("PyTorch 常用神经网络层 | PyTorch Common Neural Network Layers")
print("=" * 50)


# =============================================================================
# 1. nn.Linear 全连接层 | 1. nn.Linear Fully Connected Layer
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# Linear 是最基础的层，处理向量/表格数据。先理解其输入输出形状和计算方式。 | Linear is the most basic layer, handling vector/tabular data. First understand its input/output shapes and computation.
#
# 技术栈：torch.nn.Linear | Tech stack: torch.nn.Linear

print("\n1. nn.Linear 全连接层 | 1. nn.Linear Fully Connected Layer")
print("-" * 40)

# 为什么要创建 Linear(4, 3)？ | Why create Linear(4, 3)?
# 4 个输入特征 → 3 个输出，对应 3 个神经元，每个对 4 个输入做加权求和。 | 4 input features → 3 outputs, corresponding to 3 neurons, each performing weighted sum on 4 inputs.
linear = nn.Linear(in_features=4, out_features=3)

x = torch.randn(2, 4)  # 2 个样本，每个 4 维 | 2 samples, each 4-dimensional
output = linear(x)

print(f"输入形状: {x.shape} | Input shape: {x.shape}")
print(f"输出形状: {output.shape} | Output shape: {output.shape}")  # (2, 3)
print(f"权重 W 形状: {linear.weight.shape} | Weight W shape: {linear.weight.shape}")  # (3, 4)
print(f"偏置 b 形状: {linear.bias.shape} | Bias b shape: {linear.bias.shape}")    # (3,)

print("\nLinear 计算: output = x @ W.T + b | Linear computation: output = x @ W.T + b")


# =============================================================================
# 2. nn.ReLU 激活函数 | 2. nn.ReLU Activation Function
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 没有激活函数，多层 Linear 等价于单层，无法学非线性。ReLU 是最常用选择。 | Without activation functions, multiple Linear layers are equivalent to a single layer and cannot learn nonlinearity. ReLU is the most common choice.
#
# 技术栈：torch.nn.ReLU, torch.nn.functional.relu | Tech stack: torch.nn.ReLU, torch.nn.functional.relu

print("\n2. nn.ReLU 激活函数 | 2. nn.ReLU Activation Function")
print("-" * 40)

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# 为什么要两种方式演示？ | Why demonstrate both approaches?
# nn.ReLU() 作为层；F.relu() 作为函数，在 forward 里常用 F.relu。 | nn.ReLU() as a layer; F.relu() as a function, commonly used as F.relu in forward.
relu_layer = nn.ReLU()
output_layer = relu_layer(x)
output_func = F.relu(x)

print(f"输入: {x} | Input: {x}")
print(f"nn.ReLU() 输出: {output_layer} | nn.ReLU() output: {output_layer}")
print(f"F.relu() 输出:  {output_func} | F.relu() output:  {output_func}")
print("ReLU: 负数→0，正数不变 | ReLU: negative → 0, positive unchanged")


# =============================================================================
# 3. 其他常用激活函数 | 3. Other Common Activation Functions
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 不同任务可能用不同激活；了解 Sigmoid、Tanh 便于阅读他人代码。 | Different tasks may use different activations; understanding Sigmoid and Tanh helps when reading others' code.
#
# 技术栈：torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.functional | Tech stack: torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.functional

print("\n3. 其他激活函数（了解即可） | 3. Other Activation Functions (for reference)")
print("-" * 40)

x = torch.linspace(-2, 2, 5)

sigmoid = torch.sigmoid(x)   # (0, 1)，二分类输出层历史用法 | (0, 1), historically used for binary classification output layer
tanh = torch.tanh(x)         # (-1, 1)

print(f"输入:     {x} | Input:     {x}")
print(f"Sigmoid:  {sigmoid}")
print(f"Tanh:     {tanh}")
print("现代网络隐藏层首选 ReLU；多分类输出常用 raw logits + CrossEntropyLoss | Modern networks prefer ReLU for hidden layers; multi-class output commonly uses raw logits + CrossEntropyLoss")


# =============================================================================
# 4. nn.Conv2d 二维卷积层 | 4. nn.Conv2d 2D Convolution Layer
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 图像有空间结构，Conv2d 用局部窗口提取边缘、纹理，是 CNN 核心。 | Images have spatial structure; Conv2d uses local windows to extract edges and textures — the core of CNN.
#
# 技术栈：torch.nn.Conv2d | Tech stack: torch.nn.Conv2d

print("\n4. nn.Conv2d 二维卷积层 | 4. nn.Conv2d 2D Convolution Layer")
print("-" * 40)

# 为什么要 Conv2d(3, 16, kernel_size=3)？ | Why Conv2d(3, 16, kernel_size=3)?
# 3 通道 RGB → 16 个特征图；3×3 卷积核扫描图像。 | 3-channel RGB → 16 feature maps; 3×3 kernel scanning the image.
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

# 为什么要 padding=1？ | Why padding=1?
# 3×3 卷积不加 padding 会缩小尺寸；padding=1 保持 H、W 不变（stride=1 时）。 | 3×3 convolution without padding shrinks dimensions; padding=1 preserves H and W (when stride=1).
image = torch.randn(1, 3, 32, 32)  # 1 张 32×32 RGB | 1 image of 32×32 RGB
feature_maps = conv(image)

print(f"输入图像形状: {image.shape} | Input image shape: {image.shape}")       # (1, 3, 32, 32)
print(f"卷积输出形状: {feature_maps.shape} | Conv output shape: {feature_maps.shape}")  # (1, 16, 32, 32)
print(f"卷积核权重形状: {conv.weight.shape} | Conv kernel weight shape: {conv.weight.shape}")  # (16, 3, 3, 3)

print("\nConv2d 维度: (batch, channels, height, width) | Conv2d dimensions: (batch, channels, height, width)")


# =============================================================================
# 5. nn.MaxPool2d 池化层 | 5. nn.MaxPool2d Pooling Layer
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 池化降采样、扩大感受野、减少计算，并带来一定平移不变性。 | Pooling downsamples, expands receptive field, reduces computation, and provides some translation invariance.
#
# 技术栈：torch.nn.MaxPool2d | Tech stack: torch.nn.MaxPool2d

print("\n5. nn.MaxPool2d 池化层 | 5. nn.MaxPool2d Pooling Layer")
print("-" * 40)

pool = nn.MaxPool2d(kernel_size=2, stride=2)
pooled = pool(feature_maps)

print(f"池化前: {feature_maps.shape} | Before pooling: {feature_maps.shape}")  # (1, 16, 32, 32)
print(f"池化后: {pooled.shape} | After pooling: {pooled.shape}")          # (1, 16, 16, 16)
print("2×2 池化 + stride=2：高宽各减半 | 2×2 pooling + stride=2: height and width each halved")


# =============================================================================
# 6. nn.Flatten 展平层 | 6. nn.Flatten Flatten Layer
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 卷积输出是 4D，全连接需要 2D (batch, features)，Flatten 负责转换。 | Conv output is 4D; fully connected layers need 2D (batch, features); Flatten handles the conversion.
#
# 技术栈：torch.nn.Flatten | Tech stack: torch.nn.Flatten

print("\n6. nn.Flatten 展平层 | 6. nn.Flatten Flatten Layer")
print("-" * 40)

flatten = nn.Flatten()
flat = flatten(pooled)

print(f"展平前: {pooled.shape} | Before flatten: {pooled.shape}")   # (1, 16, 16, 16)
print(f"展平后: {flat.shape} | After flatten: {flat.shape}")     # (1, 4096)


# =============================================================================
# 7. 实战：搭建简单 MLP | 7. Hands-on: Build a Simple MLP
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 串联 Linear + ReLU，完成表格/向量数据的分类网络。 | Chain Linear + ReLU to build a classification network for tabular/vector data.
#
# 技术栈：torch.nn.Sequential, Linear, ReLU | Tech stack: torch.nn.Sequential, Linear, ReLU

print("\n7. 实战：简单 MLP（多层感知机） | 7. Hands-on: Simple MLP (Multi-Layer Perceptron)")
print("-" * 40)


class SimpleMLP(nn.Module):
    """
    输入 20 维 → 64 → 32 → 5 类 | Input 20-dim → 64 → 32 → 5 classes

    技术栈：torch.nn.Module, Sequential, Linear, ReLU | Tech stack: torch.nn.Module, Sequential, Linear, ReLU
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

    def forward(self, x):
        return self.network(x)


mlp = SimpleMLP()
x = torch.randn(8, 20)
output = mlp(x)

print(f"MLP 输入: {x.shape} | MLP input: {x.shape}")
print(f"MLP 输出: {output.shape} | MLP output: {output.shape}")  # (8, 5)
print(f"MLP 参数量: {sum(p.numel() for p in mlp.parameters())} | MLP parameter count: {sum(p.numel() for p in mlp.parameters())}")


# =============================================================================
# 8. 实战：搭建简单 CNN | 8. Hands-on: Build a Simple CNN
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 图像任务标准范式：Conv → ReLU → Pool → Flatten → Linear。 | Standard image task paradigm: Conv → ReLU → Pool → Flatten → Linear.
#
# 技术栈：Conv2d, ReLU, MaxPool2d, Flatten, Linear, Sequential | Tech stack: Conv2d, ReLU, MaxPool2d, Flatten, Linear, Sequential

print("\n8. 实战：简单 CNN（卷积神经网络） | 8. Hands-on: Simple CNN (Convolutional Neural Network)")
print("-" * 40)


class SimpleCNN(nn.Module):
    """
    28×28 灰度图 → 10 类（如 MNIST） | 28×28 grayscale → 10 classes (e.g., MNIST)

    技术栈：Conv2d, ReLU, MaxPool2d, Flatten, Linear | Tech stack: Conv2d, ReLU, MaxPool2d, Flatten, Linear
    """

    def __init__(self):
        super().__init__()

        # 为什么要分 features 和 classifier？ | Why separate features and classifier?
        # 特征提取（卷积+池化）与分类（全连接）职责分离，结构更清晰。 | Feature extraction (conv+pool) and classification (fully connected) are separated by responsibility for clearer structure.
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 1→16 通道，28×28 | 1→16 channels, 28×28
            nn.ReLU(),
            nn.MaxPool2d(2),   # 28→14
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 14×14
            nn.ReLU(),
            nn.MaxPool2d(2),   # 14→7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


cnn = SimpleCNN()
image = torch.randn(4, 1, 28, 28)
logits = cnn(image)

print(f"CNN 输入: {image.shape} | CNN input: {image.shape}")
print(f"CNN 输出: {logits.shape} | CNN output: {logits.shape}")  # (4, 10)
print(f"CNN 参数量: {sum(p.numel() for p in cnn.parameters())} | CNN parameter count: {sum(p.numel() for p in cnn.parameters())}")

print("\nCNN 结构: | CNN structure:")
print(cnn)


# =============================================================================
# 9. 练习：追踪中间层输出形状 | 9. Exercise: Trace Intermediate Layer Output Shapes
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 调试 CNN 时经常遇到 shape 错误；手动追踪每层输出是必备技能。 | Shape errors are common when debugging CNNs; manually tracing each layer's output is an essential skill.
#
# 技术栈：torch.nn 各层 forward | Tech stack: torch.nn layer forward passes

print("\n9. 练习：追踪 CNN 各层输出形状 | 9. Exercise: Trace CNN Layer Output Shapes")
print("-" * 40)

x = torch.randn(1, 1, 28, 28)
print(f"输入:           {x.shape} | Input:           {x.shape}")

x = cnn.features[0](x)  # Conv2d
print(f"Conv2d 后:      {x.shape} | After Conv2d:      {x.shape}")
x = cnn.features[1](x)  # ReLU
print(f"ReLU 后:        {x.shape} | After ReLU:        {x.shape}")
x = cnn.features[2](x)  # MaxPool2d
print(f"MaxPool2d 后:   {x.shape} | After MaxPool2d:   {x.shape}")
x = cnn.features[3](x)  # Conv2d
print(f"Conv2d 后:      {x.shape} | After Conv2d:      {x.shape}")
x = cnn.features[4](x)  # ReLU
print(f"ReLU 后:        {x.shape} | After ReLU:        {x.shape}")
x = cnn.features[5](x)  # MaxPool2d
print(f"MaxPool2d 后:   {x.shape} | After MaxPool2d:   {x.shape}")

x = cnn.classifier(x)
print(f"Classifier 后:  {x.shape} | After Classifier:  {x.shape}")


print("\n" + "=" * 50)
print("常用层学习完成！ | Common layers learning complete!")
print("=" * 50)
print("\n核心要点： | Key Takeaways:")
print("1. Linear：向量/表格数据，全连接 | 1. Linear: vector/tabular data, fully connected")
print("2. Conv2d：图像数据，局部特征 | 2. Conv2d: image data, local features")
print("3. ReLU：隐藏层默认激活 | 3. ReLU: default hidden layer activation")
print("4. MaxPool2d：降采样 | 4. MaxPool2d: downsampling")
print("5. Flatten：卷积输出 → 全连接输入 | 5. Flatten: conv output → fully connected input")
print("6. MLP = Linear + ReLU；CNN = Conv + ReLU + Pool + Linear | 6. MLP = Linear + ReLU; CNN = Conv + ReLU + Pool + Linear")
