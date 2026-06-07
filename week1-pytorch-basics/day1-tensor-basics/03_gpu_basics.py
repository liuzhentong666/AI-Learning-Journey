"""
Day 1: PyTorch张量基础 - GPU加速基础 | Day 1: PyTorch Tensor Basics - GPU Acceleration Basics
学习目标：了解如何在GPU上运行PyTorch | Learning objective: Learn how to run PyTorch on GPU
"""

import torch

print("=" * 50)
print("GPU加速基础 | GPU Acceleration Basics")
print("=" * 50)

# 1. 检查CUDA是否可用 | 1. Check CUDA availability
print("\n1. CUDA可用性检查 | 1. CUDA Availability Check")
cuda_available = torch.cuda.is_available()
print(f"CUDA是否可用: {cuda_available} | CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA版本: {torch.version.cuda} | CUDA version: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()} | GPU count: {torch.cuda.device_count()}")
    print(f"当前GPU名称: {torch.cuda.get_device_name(0)} | Current GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA不可用，将使用CPU进行计算 | CUDA not available, will use CPU for computation")

# 2. 设备选择 | 2. Device selection
print("\n2. 设备选择 | 2. Device Selection")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device} | Using device: {device}")

# 3. 在指定设备上创建张量 | 3. Create tensors on specified devices
print("\n3. 在不同设备上创建张量 | 3. Create tensors on different devices")

# CPU张量 | CPU tensor
cpu_tensor = torch.ones(3, 3)
print(f"CPU张量设备: {cpu_tensor.device} | CPU tensor device: {cpu_tensor.device}")

# GPU张量（如果可用）| GPU tensor (if available)
if cuda_available:
    gpu_tensor = torch.ones(3, 3, device='cuda')
    print(f"GPU张量设备: {gpu_tensor.device} | GPU tensor device: {gpu_tensor.device}")
    
    # 将CPU张量移到GPU | Move CPU tensor to GPU
    cpu_to_gpu = cpu_tensor.to('cuda')
    print(f"移动到GPU后: {cpu_to_gpu.device} | After moving to GPU: {cpu_to_gpu.device}")
    
    # 将GPU张量移回CPU | Move GPU tensor back to CPU
    gpu_to_cpu = gpu_tensor.to('cpu')
    print(f"移回CPU后: {gpu_to_cpu.device} | After moving back to CPU: {gpu_to_cpu.device}")
else:
    print("GPU不可用，跳过GPU操作 | GPU not available, skipping GPU operations")

# 4. 设备间计算 | 4. Cross-device computation
print("\n4. 设备间计算 | 4. Cross-device Computation")
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)

# 矩阵乘法 | Matrix multiplication
import time
start = time.time()
c = torch.matmul(a, b)
end = time.time()
print(f"矩阵乘法在{device}上用时: {(end-start)*1000:.2f}ms | Matrix multiplication on {device} took: {(end-start)*1000:.2f}ms")

# 5. 实用的设备管理模式 | 5. Practical device management pattern
print("\n5. 推荐的设备管理方式 | 5. Recommended device management approach")
print("""
# 在代码开头设置 | # Set at the beginning of code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建张量时指定设备 | # Specify device when creating tensors
tensor = torch.randn(3, 3, device=device)

# 或移动已有张量 | # Or move existing tensors
tensor = torch.randn(3, 3).to(device)

# 模型也需要移到相同设备 | # Model also needs to be moved to the same device
model = MyModel().to(device)
""")

# 6. 练习：对比CPU和GPU性能（如果有GPU）| 6. Exercise: Compare CPU and GPU performance (if GPU available)
print("\n6. 性能对比练习 | 6. Performance Comparison Exercise")
size = 5000

# CPU计算 | CPU computation
cpu_a = torch.randn(size, size)
cpu_b = torch.randn(size, size)
start = time.time()
cpu_result = torch.matmul(cpu_a, cpu_b)
cpu_time = time.time() - start
print(f"CPU矩阵乘法({size}x{size})用时: {cpu_time*1000:.2f}ms | CPU matrix multiplication ({size}x{size}) time: {cpu_time*1000:.2f}ms")

if cuda_available:
    # GPU计算 | GPU computation
    gpu_a = torch.randn(size, size, device='cuda')
    gpu_b = torch.randn(size, size, device='cuda')
    
    # 预热GPU | Warm up GPU
    _ = torch.matmul(gpu_a, gpu_b)
    
    start = time.time()
    gpu_result = torch.matmul(gpu_a, gpu_b)
    torch.cuda.synchronize()  # 等待GPU计算完成 | Wait for GPU computation to complete
    gpu_time = time.time() - start
    
    print(f"GPU矩阵乘法({size}x{size})用时: {gpu_time*1000:.2f}ms | GPU matrix multiplication ({size}x{size}) time: {gpu_time*1000:.2f}ms")
    print(f"加速比: {cpu_time/gpu_time:.2f}x | Speedup ratio: {cpu_time/gpu_time:.2f}x")
else:
    print("无GPU，跳过GPU性能测试 | No GPU, skipping GPU performance test")

print("\n" + "=" * 50)
print("GPU基础学习完成！ | GPU basics learning complete!")
print("提示：如果没有GPU，不影响学习，PyTorch会自动在CPU上运行 | Tip: Without GPU, learning is unaffected; PyTorch will automatically run on CPU")
print("=" * 50)
