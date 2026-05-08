#!/bin/bash
# Day 1 学习脚本 - 运行所有练习

echo "=================================="
echo "Day 1: PyTorch张量基础练习"
echo "=================================="

cd /mnt/f/AI-Learning-Journey/week1-pytorch-basics/day1-tensor-basics

echo -e "\n【练习1】张量创建"
echo "=================================="
python 01_tensor_creation.py

echo -e "\n\n【练习2】张量操作"
echo "=================================="
python 02_tensor_operations.py

echo -e "\n\n【练习3】GPU基础"
echo "=================================="
python 03_gpu_basics.py

echo -e "\n\n=================================="
echo "Day 1 所有练习完成！"
echo "=================================="
echo ""
echo "下一步："
echo "1. 复习今天的代码，理解每个操作"
echo "2. 尝试修改参数，观察结果变化"
echo "3. 完成 exercises/ 中的练习题"
echo "4. 更新学习日志 daily-logs/2026-05-08.md"
echo "5. 提交到GitHub: git add . && git commit -m 'Day 1完成' && git push"
