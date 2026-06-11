# Day 1 复盘总结

---

## 一句话总结

> Day 1 学的是 PyTorch 的基本数据单位——张量。学会了 6 种创建方式、4 类操作、3 条广播规则、1 个设备管理模板。

---

## 一、张量创建（6 种核心方法）

| 方法 | 用途 | 示例 |
|------|------|------|
| `torch.tensor(data)` | 从列表/数组创建 | `torch.tensor([1, 2, 3])` |
| `torch.zeros(shape)` | 全 0 张量 | `torch.zeros(2, 3)` |
| `torch.ones(shape)` | 全 1 张量 | `torch.ones(2, 3)` |
| `torch.rand(shape)` | [0,1) 均匀随机 | `torch.rand(2, 3)` |
| `torch.arange(start, end, step)` | 等差数列 | `torch.arange(0, 10, 2)` |
| `torch.eye(n)` | 单位矩阵 | `torch.eye(3)` |

**其他补充：** `torch.linspace()` 等间距生成，`torch.randn()` 标准正态随机，`torch.from_numpy()` 从 NumPy 转换。

---

## 二、张量操作（4 大类）

### 2.1 索引与切片
和 NumPy 一样：`x[0]`、`x[:, 1:3]`、`x[0, 2]`

### 2.2 形状变换

| 方法 | 内存 | 何时用 |
|------|------|--------|
| `reshape()` | 连续则共享，不连续则复制 | **优先用，更安全** |
| `view()` | 必须共享（要求内存连续） | 确定连续时用，更省内存 |
| `squeeze(dim)` | 共享 | 去掉大小为 1 的维度 |
| `unsqueeze(dim)` | 共享 | 在指定位置插入大小为 1 的维度 |

**dim 参数含义：**
```
正数（0, 1, 2...）：从前往后数维度位置
负数（-1, -2...）：从后往前数维度位置
```

**squeeze(dim)：** dim 告诉我要挤掉 shape 中的第几个数字（前提是这个数字必须是 1）
**unsqueeze(dim)：** dim 告诉我在 shape 的第几个位置塞一个 1

### 2.3 数学运算
加减乘除、矩阵乘法 `@` 或 `torch.matmul()`、求和 `sum()`、均值 `mean()`

### 2.4 张量拼接
- `torch.cat([a, b], dim=0)`：沿 dim 拼接（dim=0 垂直，dim=1 水平）
- `torch.stack([a, b], dim=0)`：沿新维度堆叠

---

## 三、广播机制（3 条规则）

**从右往左**逐个维度比较：

1. **维度相等** → 直接匹配
2. **一方为 1** → 扩展为对方大小
3. **一方不存在** → 补 1，然后按规则 2 扩展

**例子：**
```
x: (3, 4)     y: (4,)      → y 补为 (1, 4) → 扩展为 (3, 4) ✓
x: (4, 3)     y: (4, 1)    → 从右往左：3≠1 且都不是1 → 失败 ✗
```

---

## 四、设备管理（1 个模板）

```python
# 自动选择可用设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 张量移到设备
x = torch.randn(3, 4).to(device)

# 模型移到设备
model = MyModel().to(device)
```

**核心思想：** 代码里不写死 `'cuda'` 也不写死 `'cpu'`，用 `device` 变量统一管理。有 GPU 自动用 GPU，没有则降级用 CPU。

---

## 五、与 NumPy 的关系

```python
# NumPy → PyTorch
t = torch.from_numpy(np_array)

# PyTorch → NumPy
arr = t.numpy()            # 要求 t 在 CPU 上
arr = t.cpu().numpy()      # 安全写法，先移到 CPU
```

**注意：** `t.numpy()` 和 `torch.from_numpy()` 共享内存，改一个会影响另一个。

---

## 六、Day 1 知识结构图

```
张量基础
├── 创建：tensor, zeros, ones, rand, arange, eye
├── 操作
│   ├── 索引切片
│   ├── 形状变换：reshape, view, squeeze, unsqueeze
│   ├── 数学运算
│   └── 拼接：cat, stack
├── 广播：3 条规则（从右往左）
├── 设备：device 变量 + .to()
└── NumPy 互转：from_numpy / .numpy()
```

---

## 七、自检清单

闭卷回答以下问题，全部通过才算 Day 1 到位：

- [ ] `view()` 和 `reshape()` 的区别是什么？
- [ ] `squeeze(dim)` 和 `unsqueeze(dim)` 的 dim 参数含义？正负数怎么理解？
- [ ] 广播三条规则是什么？`(3, 4) + (4,)` 为什么能广播？
- [ ] 兼容 CPU/GPU 的 device 变量怎么写？
- [ ] `torch.arange()` 和 `torch.rand()` 的区别是什么？
- [ ] `torch.cat()` 和 `torch.stack()` 的区别是什么？

---

*复盘时间：2026-06-11*

---

# Day 1 Recap Summary (English)

## One-liner

> Day 1 covered PyTorch's fundamental data unit — the Tensor. Learned 6 creation methods, 4 operation categories, 3 broadcasting rules, 1 device management template.

## I. Tensor Creation (6 Core Methods)

| Method | Purpose | Example |
|--------|---------|---------|
| `torch.tensor(data)` | From list/array | `torch.tensor([1, 2, 3])` |
| `torch.zeros(shape)` | All zeros | `torch.zeros(2, 3)` |
| `torch.ones(shape)` | All ones | `torch.ones(2, 3)` |
| `torch.rand(shape)` | Uniform random [0,1) | `torch.rand(2, 3)` |
| `torch.arange(start, end, step)` | Arithmetic sequence | `torch.arange(0, 10, 2)` |
| `torch.eye(n)` | Identity matrix | `torch.eye(3)` |

**Also available:** `torch.linspace()` (evenly spaced), `torch.randn()` (standard normal), `torch.from_numpy()` (NumPy conversion).

## II. Tensor Operations (4 Categories)

### 2.1 Indexing & Slicing
Same as NumPy: `x[0]`, `x[:, 1:3]`, `x[0, 2]`

### 2.2 Shape Manipulation

| Method | Memory | When to Use |
|--------|--------|-------------|
| `reshape()` | Shares if contiguous, copies otherwise | **Preferred — safer** |
| `view()` | Must share (requires contiguous memory) | Use when sure memory is contiguous — saves memory |
| `squeeze(dim)` | Shares | Remove dimension of size 1 |
| `unsqueeze(dim)` | Shares | Insert dimension of size 1 at position |

**dim parameter meaning:**
```
Positive (0, 1, 2...): count dimension from the left
Negative (-1, -2...): count dimension from the right
```

**squeeze(dim):** dim tells which position in the shape to squeeze (must be size 1)
**unsqueeze(dim):** dim tells where in the shape to insert a 1

### 2.3 Math Operations
Arithmetic, matrix multiplication `@` / `torch.matmul()`, `sum()`, `mean()`

### 2.4 Tensor Concatenation
- `torch.cat([a, b], dim=0)`: concatenate along dim (dim=0 vertical, dim=1 horizontal)
- `torch.stack([a, b], dim=0)`: stack along a new dimension

## III. Broadcasting (3 Rules)

Compare dimensions **from right to left**:

1. **Equal dimensions** → match directly
2. **One side is 1** → expand to match the other
3. **One side missing** → pad with 1, then apply rule 2

**Examples:**
```
x: (3, 4)     y: (4,)      → y padded to (1, 4) → expanded to (3, 4) ✓
x: (4, 3)     y: (4, 1)    → right to left: 3≠1 and neither is 1 → fail ✗
```

## IV. Device Management (1 Template)

```python
# Auto-select available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move tensor to device
x = torch.randn(3, 4).to(device)

# Move model to device
model = MyModel().to(device)
```

**Core idea:** Never hardcode `'cuda'` or `'cpu'` in code. Use a `device` variable to manage it uniformly. Auto-uses GPU if available, falls back to CPU if not.

## V. NumPy Interop

```python
# NumPy → PyTorch
t = torch.from_numpy(np_array)

# PyTorch → NumPy
arr = t.numpy()            # requires t on CPU
arr = t.cpu().numpy()      # safe: always works
```

**Note:** `t.numpy()` and `torch.from_numpy()` share memory — modifying one affects the other.

## VI. Day 1 Knowledge Structure

```
Tensor Basics
├── Creation: tensor, zeros, ones, rand, arange, eye
├── Operations
│   ├── Indexing & slicing
│   ├── Shape manipulation: reshape, view, squeeze, unsqueeze
│   ├── Math operations
│   └── Concatenation: cat, stack
├── Broadcasting: 3 rules (right to left)
├── Device: device variable + .to()
└── NumPy interop: from_numpy / .numpy()
```

## VII. Self-Check Checklist

Answer these without looking at code — all passed means Day 1 is solid:

- [ ] What's the difference between `view()` and `reshape()`?
- [ ] What does the `dim` parameter mean in `squeeze(dim)` and `unsqueeze(dim)`? How do positive vs. negative numbers work?
- [ ] What are the 3 broadcasting rules? Why does `(3, 4) + (4,)` work?
- [ ] How do you write the CPU/GPU-compatible `device` variable?
- [ ] What's the difference between `torch.arange()` and `torch.rand()`?
- [ ] What's the difference between `torch.cat()` and `torch.stack()`?

---

*Recap date: 2026-06-11*
