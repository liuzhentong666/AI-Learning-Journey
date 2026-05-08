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
   cd /mnt/f/AI-Learning-Journey
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
cd /mnt/f/AI-Learning-Journey
gh repo create AI-Learning-Journey --public --source=. --remote=origin --push
```

## 推送代码

创建远程仓库后，推送代码：

```bash
cd /mnt/f/AI-Learning-Journey
git branch -M main  # 重命名分支为main
git push -u origin main
```

## 后续提交流程

每天学习结束后，提交更新：

```bash
cd /mnt/f/AI-Learning-Journey

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
