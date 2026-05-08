# 快速开始指南

欢迎来到AI大模型开发学习之旅！

## 第一步：检查Python环境

```bash
# 检查Python版本（需要3.8+）
python --version

# 或
python3 --version
```

## 第二步：安装依赖

```bash
cd /mnt/f/AI-Learning-Journey

# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 安装PyTorch和其他依赖
pip install torch torchvision torchaudio
pip install numpy matplotlib jupyter

# 或一次性安装所有
pip install -r requirements.txt
```

## 第三步：验证安装

```bash
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
```

## 第四步：运行Day 1练习

```bash
# 方法1: 运行脚本
./run_day1.sh

# 方法2: 手动运行每个文件
cd week1-pytorch-basics/day1-tensor-basics
python 01_tensor_creation.py
python 02_tensor_operations.py
python 03_gpu_basics.py
```

## 第五步：设置GitHub（见 setup_github.md）

1. 在GitHub网页创建仓库 `AI-Learning-Journey`
2. 连接本地仓库：
```bash
cd /mnt/f/AI-Learning-Journey
git remote add origin https://github.com/YOUR_USERNAME/AI-Learning-Journey.git
git branch -M main
git push -u origin main
```

## 项目结构

```
AI-Learning-Journey/
├── README.md                    # 项目总览
├── QUICKSTART.md               # 本文件
├── setup_github.md             # GitHub设置指南
├── requirements.txt            # Python依赖
├── daily-logs/                 # 每日学习日志
│   └── 2026-05-08.md
├── week1-pytorch-basics/       # 第1周学习内容
│   ├── README.md
│   └── day1-tensor-basics/     # Day 1练习
│       ├── 01_tensor_creation.py
│       ├── 02_tensor_operations.py
│       ├── 03_gpu_basics.py
│       └── README.md
└── ... (更多周的内容)
```

## 学习流程

### 每日流程
1. 阅读当天的README.md，了解学习目标
2. 运行示例代码，理解概念
3. 完成练习题
4. 更新学习日志
5. 提交到GitHub

### 每周流程
1. 周一到周五：理论学习 + 代码练习
2. 周末：项目实战
3. 周日晚：复习总结，准备下周

## 常见问题

**Q: 我没有GPU怎么办？**  
A: 没关系！所有练习都可以在CPU上运行，只是训练速度会慢一些。

**Q: 安装PyTorch失败？**  
A: 访问 https://pytorch.org/get-started/locally/ 选择适合你系统的安装命令。

**Q: 代码运行报错？**  
A: 
1. 检查Python版本是否>=3.8
2. 确认PyTorch已正确安装
3. 查看错误信息，通常会提示缺少的包
4. 可以在daily-logs中记录问题和解决方法

**Q: 学习进度跟不上？**  
A: 没关系，按自己的节奏来。重要的是理解概念，而不是赶进度。

**Q: 想深入学习某个主题？**  
A: 在resources/目录下可以添加额外的学习资料和笔记。

## 学习建议

1. **每天坚持** - 每天1-2小时比周末突击6小时效果更好
2. **动手实践** - 不要只看代码，一定要自己敲一遍
3. **记录笔记** - 在daily-logs中记录学习心得和遇到的问题
4. **修改尝试** - 修改代码参数，观察结果变化，加深理解
5. **及时复习** - 每周末复习本周内容，巩固记忆

## 寻求帮助

- PyTorch官方论坛: https://discuss.pytorch.org/
- Stack Overflow: 搜索错误信息
- GitHub Issues: 在相关项目提问
- 学习笔记: 记录在daily-logs中，方便回顾

## 下一步

阅读 `week1-pytorch-basics/README.md` 开始Week 1的学习！

---

**祝学习顺利！记住：从零到一最难，坚持就是胜利！** 🚀
