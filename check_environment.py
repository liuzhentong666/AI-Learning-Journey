#!/usr/bin/env python3
"""
环境检查脚本 - 验证所需的包是否已安装
"""

import sys

def check_python_version():
    """检查Python版本"""
    print("=" * 50)
    print("检查Python环境")
    print("=" * 50)
    
    version = sys.version_info
    print(f"\nPython版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python版本符合要求 (>=3.8)")
        return True
    else:
        print("❌ Python版本过低，需要3.8或更高版本")
        return False

def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', '未知版本')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: 未安装")
        return False

def check_cuda():
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用")
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA不可用（将使用CPU，不影响学习）")
    except ImportError:
        print("❌ PyTorch未安装，无法检查CUDA")

def main():
    """主函数"""
    # 检查Python版本
    python_ok = check_python_version()
    
    if not python_ok:
        print("\n请升级Python版本后再试")
        return False
    
    # 检查必需的包
    print("\n" + "=" * 50)
    print("检查Python包")
    print("=" * 50 + "\n")
    
    packages = [
        ('numpy', 'numpy'),
        ('PyTorch', 'torch'),
        ('torchvision', 'torchvision'),
        ('matplotlib', 'matplotlib'),
    ]
    
    results = []
    for package_name, import_name in packages:
        results.append(check_package(package_name, import_name))
    
    # 检查CUDA
    print("\n" + "=" * 50)
    print("检查CUDA支持")
    print("=" * 50 + "\n")
    check_cuda()
    
    # 总结
    print("\n" + "=" * 50)
    print("检查结果")
    print("=" * 50)
    
    if all(results):
        print("\n✅ 所有必需包已安装，可以开始学习！")
        print("\n运行以下命令开始Day 1练习:")
        print("  ./run_day1.sh")
        print("或")
        print("  cd week1-pytorch-basics/day1-tensor-basics")
        print("  python 01_tensor_creation.py")
        return True
    else:
        print("\n❌ 有些包未安装，请运行以下命令安装:")
        print("  pip install torch torchvision numpy matplotlib")
        print("或")
        print("  pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
