#!/bin/bash
# Day 2 学习脚本 - 运行所有练习

echo "=================================="
echo "Day 2: 自动微分与梯度计算"
echo "=================================="

cd /mnt/f/AI-Learning-Journey/week1-pytorch-basics/day2-autograd

echo -e "\n【练习1】自动微分基础"
echo "=================================="
python 01_basic_autograd.py

echo -e "\n\n【练习2】计算图与梯度流"
echo "=================================="
python 02_computational_graph.py

echo -e "\n\n【练习3】梯度下降实战"
echo "=================================="
python 03_gradient_descent.py

echo -e "\n\n=================================="
echo "Day 2 所有练习完成！"
echo "=================================="
echo ""
echo "下一步："
echo "1. 复习自动微分的原理"
echo "2. 理解计算图和链式法则"
echo "3. 尝试实现不同的优化算法"
echo "4. 更新学习日志 daily-logs/2026-05-09.md"
echo "5. 提交到GitHub: git add . && git commit -m 'Day 2完成' && git push"
