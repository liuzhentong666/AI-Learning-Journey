# GitHub仓库设置指南

由于系统中未安装GitHub CLI (gh)，请按以下步骤手动创建GitHub仓库：

## 方法1：通过GitHub网页创建（推荐）

1. **登录GitHub**
   访问 https://github.com 并登录

2. **创建新仓库**
   - 点击右上角 "+" → "New repository"
   - Repository name: `AI-Learning-Journey`
   - Description: `3个月AI大模型开发学习计划 - Python AI开发 + Go后端集成`
   - 选择 Public（公开）
   - 不要勾选 "Initialize this repository with..."（因为本地已初始化）
   - 点击 "Create repository"

3. **连接本地仓库**
   复制GitHub显示的命令，或使用以下命令（替换YOUR_USERNAME）：
   
   ```bash
   cd AI-Learning-Journey
   git remote add origin https://github.com/YOUR_USERNAME/AI-Learning-Journey.git
   git branch -M main
   git push -u origin main
   ```

## 方法2：使用GitHub CLI（需要先安装）

如果想使用命令行，可以安装GitHub CLI：

```bash
# 下载并安装 gh (GitHub CLI)
# 方法1: 使用snap (推荐)
sudo snap install gh

# 方法2: 使用apt
sudo apt update
sudo apt install gh

# 认证
gh auth login

# 创建仓库
cd AI-Learning-Journey
gh repo create AI-Learning-Journey --public --source=. --remote=origin --push
```

## 推送代码

创建远程仓库后，推送代码：

```bash
cd AI-Learning-Journey
git branch -M main  # 重命名分支为main
git push -u origin main
```

## 后续提交流程

每天学习结束后，提交更新：

```bash
cd AI-Learning-Journey

# 查看修改
git status

# 添加所有修改
git add .

# 提交（写清楚今天做了什么）
git commit -m "Day X: 完成XXX学习和练习"

# 推送到GitHub
git push
```

## 建议的提交信息格式

```
Day 1: 项目初始化，完成PyTorch张量基础
Day 2: 完成自动微分学习，理解梯度计算
Day 3: 神经网络基础，实现简单前馈网络
...
Week 1 完成: MNIST项目达到95%准确率
```

## 查看GitHub仓库

推送后，访问：
https://github.com/YOUR_USERNAME/AI-Learning-Journey

---

**下一步：** 设置好GitHub仓库后，运行第一天的练习代码！

---

# GitHub Repository Setup Guide

Since GitHub CLI (gh) is not installed on the system, follow these steps to manually create a GitHub repository:

## Method 1: Create via GitHub Website (Recommended)

1. **Log into GitHub**
   Visit https://github.com and log in

2. **Create a New Repository**
   - Click "+" in the top right → "New repository"
   - Repository name: `AI-Learning-Journey`
   - Description: `3-month AI Large Model Development Learning Plan - Python AI Development + Go Backend Integration`
   - Select Public
   - Do NOT check "Initialize this repository with..." (since local repo is already initialized)
   - Click "Create repository"

3. **Connect Local Repository**
   Copy the commands shown on GitHub, or use the following (replace YOUR_USERNAME):

   ```bash
   cd AI-Learning-Journey
   git remote add origin https://github.com/YOUR_USERNAME/AI-Learning-Journey.git
   git branch -M main
   git push -u origin main
   ```

## Method 2: Using GitHub CLI (requires installation)

If you prefer command line, you can install GitHub CLI:

```bash
# Download and install gh (GitHub CLI)
# Method 1: Using snap (recommended)
sudo snap install gh

# Method 2: Using apt
sudo apt update
sudo apt install gh

# Authenticate
gh auth login

# Create repository
cd AI-Learning-Journey
gh repo create AI-Learning-Journey --public --source=. --remote=origin --push
```

## Push Your Code

After creating the remote repository, push your code:

```bash
cd AI-Learning-Journey
git branch -M main  # Rename branch to main
git push -u origin main
```

## Subsequent Commit Workflow

At the end of each day of study, commit your updates:

```bash
cd AI-Learning-Journey

# View changes
git status

# Add all changes
git add .

# Commit (clearly describe what you did today)
git commit -m "Day X: Completed XXX learning and exercises"

# Push to GitHub
git push
```

## Suggested Commit Message Format

```
Day 1: Project initialization, completed PyTorch tensor basics
Day 2: Completed autograd study, understood gradient computation
Day 3: Neural network basics, implemented simple feedforward network
...
Week 1 Complete: MNIST project achieved 95% accuracy
```

## View GitHub Repository

After pushing, visit:
https://github.com/YOUR_USERNAME/AI-Learning-Journey

**Next Step:** After setting up the GitHub repository, run the first day's practice code!
