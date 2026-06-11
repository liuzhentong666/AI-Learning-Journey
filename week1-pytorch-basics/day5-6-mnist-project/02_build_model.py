"""
Day 5-6: MNIST 项目 - 搭建分类模型 | Day 5-6: MNIST Project - Build Classification Model
学习目标：搭建两种网络（MLP 和 CNN），理解 Flatten、Conv2d、MaxPool2d 的作用 | Learning Objectives: Build two networks (MLP and CNN), understand Flatten, Conv2d, MaxPool2d

技术栈： | Tech stack:
- PyTorch (torch)
- torch.nn（Module, Linear, ReLU, Conv2d, MaxPool2d, Flatten, Dropout, BatchNorm2d） | torch.nn (Module, Linear, ReLU, Conv2d, MaxPool2d, Flatten, Dropout, BatchNorm2d)
"""

import torch
import torch.nn as nn

print("=" * 60)
print("Day 5-6: 搭建 MNIST 分类模型 | Day 5-6: Build MNIST Classification Model")
print("=" * 60)


# =============================================================================
# 1. 模型一：全连接 MLP（多层感知机） | 1. Model 1: Fully Connected MLP
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# Day 3 已经搭过 784→128→10 的雏形，这里补全：加 Dropout 防过拟合。 | Day 3 sketched 784→128→10; here we complete it by adding Dropout to prevent overfitting.
# MLP 把图像展平成一维向量，丢失了像素间的空间关系，但结构简单易理解。 | MLP flattens the image into a 1D vector, losing spatial relationships — simple and easy to understand.
#
# 技术栈：nn.Flatten, nn.Linear, nn.ReLU, nn.Dropout | Tech stack: nn.Flatten, nn.Linear, nn.ReLU, nn.Dropout

print("\n1. 全连接 MLP 模型 | 1. Fully Connected MLP Model")
print("-" * 40)


class MNISTMlp(nn.Module):
    """
    全连接多层感知机：(1,28,28) → 展平 784 → 256 → 128 → 10
    Fully Connected MLP: (1,28,28) → flatten 784 → 256 → 128 → 10

    技术栈：nn.Flatten, nn.Linear, nn.ReLU, nn.Dropout | Tech stack: nn.Flatten, nn.Linear, nn.ReLU, nn.Dropout
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 为什么先 Flatten？ | Why Flatten first?
            # Linear 层要求输入是二维 (batch, features)，Flatten 把 (batch,1,28,28) → (batch, 784)。
            # Linear requires 2D input (batch, features); Flatten converts (batch,1,28,28) → (batch,784).
            nn.Flatten(),                   # (batch, 1, 28, 28) → (batch, 784)

            nn.Linear(784, 256),            # 784 输入特征 → 256 个隐藏神经元 | 784 input → 256 hidden
            nn.ReLU(),
            # 为什么加 Dropout(0.2)？ | Why add Dropout(0.2)?
            # 训练时随机关闭 20% 的神经元，防止模型过度依赖某些特征（过拟合）。 | During training, randomly turn off 20% of neurons to prevent over-reliance on specific features (overfitting).
            # eval() 模式下 Dropout 自动关闭，不影响推理。 | In eval() mode, Dropout is automatically disabled, no effect on inference.
            nn.Dropout(0.2),

            nn.Linear(256, 128),            # 256 → 128 | 256 → 128
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 10),             # 128 → 10 类的 logit | 128 → 10-class logits
            # 为什么输出层不加 ReLU 或 Softmax？ | Why no ReLU or Softmax on output?
            # CrossEntropyLoss 内部已经做了 softmax + log + NLL，不要重复。 | CrossEntropyLoss already applies softmax + log + NLL internally — don't add twice.
        )

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        return self.net(x)  # 输出 (batch, 10) logit | output (batch, 10) logits


mlp = MNISTMlp()
print("MLP 模型结构: | MLP model structure:")
print(mlp)
mlp_params = sum(p.numel() for p in mlp.parameters())
print(f"\n参数总量: {mlp_params:,} | Total parameters: {mlp_params:,}")
# Linear(784,256)=200960, Linear(256,128)=32896, Linear(128,10)=1290 → 约 235,146

# 验证输出形状 | Verify output shape
dummy = torch.zeros(4, 1, 28, 28)   # 4 张假图 | 4 dummy images
out   = mlp(dummy)
print(f"输入形状: {dummy.shape} → 输出形状: {out.shape} | Input: {dummy.shape} → Output: {out.shape}")
# torch.Size([4, 1, 28, 28]) → torch.Size([4, 10])


# =============================================================================
# 2. 模型二：卷积神经网络 CNN | 2. Model 2: Convolutional Neural Network CNN
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# CNN 通过卷积核扫描图像，保留空间结构（相邻像素的关系），通常比 MLP 在图像上更准。 | CNN scans images with kernels, preserving spatial structure (neighbor pixel relationships), usually more accurate on images than MLP.
# 这是图像识别的主流架构，理解它是进阶的关键。 | This is the mainstream architecture for image recognition — understanding it is key to advancing.
#
# 技术栈：nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d, nn.Flatten, nn.Linear | Tech stack: nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d, nn.Flatten, nn.Linear

print("\n2. 卷积神经网络 CNN 模型 | 2. CNN Model")
print("-" * 40)


class MNISTCnn(nn.Module):
    """
    小型 CNN：(1,28,28) → Conv→Pool → Conv→Pool → Flatten → FC → 10
    Small CNN: (1,28,28) → Conv→Pool → Conv→Pool → Flatten → FC → 10

    技术栈：nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d, nn.ReLU, nn.Flatten, nn.Linear
    """

    def __init__(self):
        super().__init__()

        # ── 卷积块 1 | Convolutional Block 1 ─────────────────────────────
        # 为什么 Conv2d(1, 32, kernel_size=3, padding=1)？ | Why Conv2d(1, 32, kernel_size=3, padding=1)?
        # in_channels=1（灰度），out_channels=32（32 个不同的特征探测器），kernel=3×3，padding=1 保持宽高不变。
        # in_channels=1 (grayscale), out_channels=32 (32 different feature detectors), kernel=3×3, padding=1 keeps H,W unchanged.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # 为什么加 BatchNorm2d？ | Why add BatchNorm2d?
        # 对每个 channel 的特征值做标准化，让训练更稳定，收敛更快。 | Normalizes feature values per channel, making training more stable and converging faster.
        self.bn1   = nn.BatchNorm2d(32)

        # ── 卷积块 2 | Convolutional Block 2 ─────────────────────────────
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # ── 最大池化 | Max Pooling ────────────────────────────────────────
        # 为什么用 MaxPool2d(2)？ | Why MaxPool2d(2)?
        # 2×2 最大池化把宽高各减半：28→14→7。保留最强的特征，减少计算量。 | 2×2 max pooling halves H and W: 28→14→7. Keeps strongest features, reduces computation.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

        # ── 全连接分类头 | Fully Connected Classification Head ─────────────
        # 为什么 Linear(64 * 7 * 7, 128)？ | Why Linear(64 * 7 * 7, 128)?
        # 两次 MaxPool 后宽高从 28→14→7，通道数 64，展平后 = 64×7×7 = 3136。 | After two MaxPool: 28→14→7, channels=64, flattened = 64×7×7 = 3136.
        self.fc1 = nn.Linear(64 * 7 * 7, 128)   # 3136 → 128 | 3136 → 128
        self.fc2 = nn.Linear(128, 10)             # 128 → 10 类 logit | 128 → 10-class logits

    def forward(self, x):
        # x: (batch, 1, 28, 28)

        # 卷积块 1 | Conv block 1
        x = self.relu(self.bn1(self.conv1(x)))   # (batch, 32, 28, 28)
        x = self.pool(x)                          # (batch, 32, 14, 14)

        # 卷积块 2 | Conv block 2
        x = self.relu(self.bn2(self.conv2(x)))   # (batch, 64, 14, 14)
        x = self.pool(x)                          # (batch, 64, 7, 7)

        # 展平 + 全连接 | Flatten + FC
        x = x.view(x.size(0), -1)                # (batch, 64*7*7) = (batch, 3136)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))               # (batch, 128)
        x = self.fc2(x)                          # (batch, 10)

        return x


cnn = MNISTCnn()
print("CNN 模型结构: | CNN model structure:")
print(cnn)
cnn_params = sum(p.numel() for p in cnn.parameters())
print(f"\n参数总量: {cnn_params:,} | Total parameters: {cnn_params:,}")
# Conv(1,32,3): 32*1*3*3+32=320, BN(32): 64, Conv(32,64,3): 64*32*3*3+64=18496, ...

# 验证 CNN 输出形状及中间形状 | Verify CNN output shape and intermediate shapes
dummy = torch.zeros(4, 1, 28, 28)

# 手动走一遍前向，打印每层形状 | Manually step through forward, print each shape
x = dummy
x = cnn.relu(cnn.bn1(cnn.conv1(x)));  print(f"  conv1 + bn1 + relu 后: {x.shape} | After conv1+bn1+relu: {x.shape}")  # (4,32,28,28)
x = cnn.pool(x);                       print(f"  pool  后:              {x.shape} | After pool: {x.shape}")             # (4,32,14,14)
x = cnn.relu(cnn.bn2(cnn.conv2(x)));  print(f"  conv2 + bn2 + relu 后: {x.shape} | After conv2+bn2+relu: {x.shape}")  # (4,64,14,14)
x = cnn.pool(x);                       print(f"  pool  后:              {x.shape} | After pool: {x.shape}")             # (4,64,7,7)
x = x.view(x.size(0), -1);            print(f"  flatten 后:            {x.shape} | After flatten: {x.shape}")          # (4,3136)
x = cnn.relu(cnn.fc1(x));             print(f"  fc1 + relu 后:         {x.shape} | After fc1+relu: {x.shape}")         # (4,128)
x = cnn.fc2(x);                       print(f"  fc2（输出）后:          {x.shape} | After fc2 (output): {x.shape}")     # (4,10)


# =============================================================================
# 3. MLP vs CNN 对比 | 3. MLP vs CNN Comparison
# =============================================================================
# 为什么要写这段代码？ | Why write this code?
# 帮助理解两种架构的优缺点，为选择合适的模型建立直觉。 | Builds intuition for choosing the right architecture.

print("\n3. MLP vs CNN 对比 | 3. MLP vs CNN Comparison")
print("-" * 40)
print(f"  {'模型 | Model':<12} {'参数量 | Params':>16} {'空间感知 | Spatial Awareness':>22} {'适用场景 | Use Case'}")
print(f"  {'-'*12} {'-'*16} {'-'*22} {'-'*30}")
print(f"  {'MLP':<12} {mlp_params:>16,} {'否（展平丢失）| No':>22} {'表格/向量数据 | tabular/vector data'}")
print(f"  {'CNN':<12} {cnn_params:>16,} {'是（卷积保留）| Yes':>22} {'图像/空间数据 | image/spatial data'}")

print("\n  核心差异: | Core difference:")
print("  - MLP: 784 个像素各自独立，不知道「哪些像素相邻」 | MLP: 784 pixels treated independently, no notion of 'which pixels are neighbors'")
print("  - CNN: 3×3 卷积核在图像上滑动，天然感知局部空间结构 | CNN: 3×3 kernel slides over image, naturally captures local spatial structure")


print("\n" + "=" * 60)
print("模型搭建完成！ | Model building done!")
print("=" * 60)
print("\n核心要点： | Key takeaways:")
print("1. MLP 先 Flatten 把图像展平，再接 Linear | 1. MLP first Flattens image, then Linear layers")
print("2. CNN 用 Conv2d 保留空间结构，MaxPool2d 下采样 | 2. CNN uses Conv2d to preserve spatial structure, MaxPool2d to downsample")
print("3. 两次 MaxPool(2) 后宽高 28→14→7 | 3. Two MaxPool(2) reduce H,W: 28→14→7")
print("4. Dropout 防过拟合，BatchNorm 加速收敛 | 4. Dropout prevents overfitting, BatchNorm speeds convergence")
print("5. 输出层不加激活函数，让 CrossEntropyLoss 内部处理 | 5. No activation on output layer; CrossEntropyLoss handles it internally")
