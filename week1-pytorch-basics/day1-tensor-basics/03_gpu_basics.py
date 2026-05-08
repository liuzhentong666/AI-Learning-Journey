"""
Day 1: PyTorch张量基础 - GPU加速基础
学习目标：了解如何在GPU上运行PyTorch
"""

import torch

print("=" * 50)
print("GPU加速基础")
print("=" * 50)

# 1. 检查CUDA是否可用
print("\n1. CUDA可用性检查")
cuda_available = torch.cuda.is_available()
print(f"CUDA是否可用: {cuda_available}")

if cuda_available:
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU名称: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA不可用，将使用CPU进行计算")

# 2. 设备选择
print("\n2. 设备选择")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 3. 在指定设备上创建张量
print("\n3. 在不同设备上创建张量")

# CPU张量
cpu_tensor = torch.ones(3, 3)
print(f"CPU张量设备: {cpu_tensor.device}")

# GPU张量（如果可用）
if cuda_available:
    gpu_tensor = torch.ones(3, 3, device='cuda')
    print(f"GPU张量设备: {gpu_tensor.device}")
    
    # 将CPU张量移到GPU
    cpu_to_gpu = cpu_tensor.to('cuda')
    print(f"移动到GPU后: {cpu_to_gpu.device}")
    
    # 将GPU张量移回CPU
    gpu_to_cpu = gpu_tensor.to('cpu')
    print(f"移回CPU后: {gpu_to_cpu.device}")
else:
    print("GPU不可用，跳过GPU操作")

# 4. 设备间计算
print("\n4. 设备间计算")
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)

# 矩阵乘法
import time
start = time.time()
c = torch.matmul(a, b)
end = time.time()
print(f"矩阵乘法在{device}上用时: {(end-start)*1000:.2f}ms")

# 5. 实用的设备管理模式
print("\n5. 推荐的设备管理方式")
print("""
# 在代码开头设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建张量时指定设备
tensor = torch.randn(3, 3, device=device)

# 或移动已有张量
tensor = torch.randn(3, 3).to(device)

# 模型也需要移到相同设备
model = MyModel().to(device)
""")

# 6. 练习：对比CPU和GPU性能（如果有GPU）
print("\n6. 性能对比练习")
size = 5000

# CPU计算
cpu_a = torch.randn(size, size)
cpu_b = torch.randn(size, size)
start = time.time()
cpu_result = torch.matmul(cpu_a, cpu_b)
cpu_time = time.time() - start
print(f"CPU矩阵乘法({size}x{size})用时: {cpu_time*1000:.2f}ms")

if cuda_available:
    # GPU计算
    gpu_a = torch.randn(size, size, device='cuda')
    gpu_b = torch.randn(size, size, device='cuda')
    
    # 预热GPU
    _ = torch.matmul(gpu_a, gpu_b)
    
    start = time.time()
    gpu_result = torch.matmul(gpu_a, gpu_b)
    torch.cuda.synchronize()  # 等待GPU计算完成
    gpu_time = time.time() - start
    
    print(f"GPU矩阵乘法({size}x{size})用时: {gpu_time*1000:.2f}ms")
    print(f"加速比: {cpu_time/gpu_time:.2f}x")
else:
    print("无GPU，跳过GPU性能测试")

print("\n" + "=" * 50)
print("GPU基础学习完成！")
print("提示：如果没有GPU，不影响学习，PyTorch会自动在CPU上运行")
print("=" * 50)
