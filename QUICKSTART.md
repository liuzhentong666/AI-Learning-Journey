# AI 大模型学习之路 - 快速开始

## 第一步：环境准备

### 1. 安装Python（如未安装）
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip

# macOS
brew install python3

# Windows
# 从 https://python.org 下载安装包
```

### 2. 克隆项目
```bash
git clone https://github.com/YOUR_USERNAME/AI-Learning-Journey.git
cd AI-Learning-Journey
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

## 第二步：验证环境

```bash
cd AI-Learning-Journey
python check_environment.py
```

确认输出显示：
- Python >= 3.8
- PyTorch 已安装
- 环境正常

## 第三步：开始学习

```bash
# Day 1：张量基础
cd week1-pytorch-basics/day1-tensor-basics
python 01_tensor_creation.py

# Day 2：自动微分
cd ../day2-autograd
python 01_basic_autograd.py

# Day 3：神经网络基础
cd ../day3-neural-network
python 01_nn_module_basics.py
```

## 第四步：记录学习

在 `daily-logs/` 目录下创建学习日志：

```bash
# 创建当天的日志（以日期命名）
touch daily-logs/YYYY-MM-DD.md
```

## 第五步：设置GitHub（见 setup_github.md）

1. 在GitHub网页创建仓库 `AI-Learning-Journey`
2. 连接本地仓库：
```bash
cd AI-Learning-Journey
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

---

# AI LLM Learning Journey - Quick Start (English)

## Step 1: Environment Setup

### 1. Install Python (if not installed)
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip

# macOS
brew install python3

# Windows
# Download from https://python.org
```

### 2. Clone the Project
```bash
git clone https://github.com/YOUR_USERNAME/AI-Learning-Journey.git
cd AI-Learning-Journey
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 2: Verify Environment

```bash
cd AI-Learning-Journey
python check_environment.py
```

Confirm output shows:
- Python >= 3.8
- PyTorch installed
- Environment OK

## Step 3: Start Learning

```bash
# Day 1: Tensor Basics
cd week1-pytorch-basics/day1-tensor-basics
python 01_tensor_creation.py

# Day 2: Autograd
cd ../day2-autograd
python 01_basic_autograd.py

# Day 3: Neural Network Basics
cd ../day3-neural-network
python 01_nn_module_basics.py
```

## Step 4: Log Your Learning

Create daily logs in the `daily-logs/` directory:

```bash
# Create today's log (named by date)
touch daily-logs/YYYY-MM-DD.md
```

## Step 5: Set Up GitHub (see setup_github.md)

1. Create the `AI-Learning-Journey` repository on GitHub
2. Connect your local repository:
```bash
cd AI-Learning-Journey
git remote add origin https://github.com/YOUR_USERNAME/AI-Learning-Journey.git
git branch -M main
git push -u origin main
```

## Project Structure

```
AI-Learning-Journey/
├── README.md                    # Project overview
├── QUICKSTART.md               # This file
├── setup_github.md             # GitHub setup guide
├── requirements.txt            # Python dependencies
├── daily-logs/                 # Daily learning logs
│   └── 2026-05-08.md
├── week1-pytorch-basics/       # Week 1 learning content
│   ├── README.md
│   └── day1-tensor-basics/     # Day 1 exercises
│       ├── 01_tensor_creation.py
│       ├── 02_tensor_operations.py
│       ├── 03_gpu_basics.py
│       └── README.md
└── ... (more weeks)
```

## Learning Workflow

### Daily Workflow
1. Read the day's README.md to understand learning objectives
2. Run example code to grasp concepts
3. Complete practice exercises
4. Update your learning log
5. Commit to GitHub

### Weekly Workflow
1. Monday-Friday: Theory study + coding practice
2. Weekend: Project practice
3. Sunday evening: Review and prepare for next week

## FAQ

**Q: I don't have a GPU, what should I do?**  
A: No problem! All exercises can run on CPU, though training will be slightly slower.

**Q: PyTorch installation failed?**  
A: Visit https://pytorch.org/get-started/locally/ and select the appropriate install command for your system.

**Q: Code throws errors when running?**  
A: 
1. Check that Python version >= 3.8
2. Verify PyTorch is correctly installed
3. Read the error message — it usually hints at missing packages
4. Record issues and solutions in your daily-logs

**Q: Falling behind on the learning schedule?**  
A: No worries, go at your own pace. Understanding concepts matters more than keeping up with a timeline.

**Q: Want to dive deeper into a topic?**  
A: You can add extra learning materials and notes in the resources/ directory.

## Study Tips

1. **Stay Consistent** — 1–2 hours daily works better than 6-hour weekend marathons
2. **Hands-On Practice** — Don't just read code; type it out yourself
3. **Take Notes** — Record insights and challenges in daily-logs
4. **Experiment** — Tweak code parameters, observe results, deepen understanding
5. **Review Regularly** — Revisit each week's content on weekends to reinforce memory

## Getting Help

- PyTorch Official Forum: https://discuss.pytorch.org/
- Stack Overflow: Search error messages
- GitHub Issues: Ask questions on relevant projects
- Study Notes: Record in daily-logs for easy review

## Next Steps

Read `week1-pytorch-basics/README.md` to start your Week 1 learning!

---

**Happy learning! Remember: The hardest step is the first one — persistence leads to success!** 🚀
