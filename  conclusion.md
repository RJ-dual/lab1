# Lab1 考点整理

> 来源：`pytorch_toturial(2).pdf`、`pytorch_preliminary.ipynb`、`linear_regression.ipynb`

---

## 一、PyTorch 基础概念

### 1.1 什么是 PyTorch
- 开源机器学习库
- 核心功能：定义神经网络、自动计算梯度（Autograd）、数据集管理、优化器、GPU 加速
- 与 NumPy 的区别：PyTorch 支持 GPU 运算和自动微分

### 1.2 Tensor（张量）
- 定义：多维数组，类似 NumPy 的 ndarray，但具备 GPU 和自动微分能力
- 导入方式：`import torch`

---

## 二、张量创建与基本属性

### 2.1 创建张量的方式
| 方法 | 说明 | 示例 |
|------|------|------|
| `torch.arange()` | 创建等差一维张量 | `torch.arange(12)` → tensor([0,1,...,11]) |
| `torch.zeros()` | 全零张量 | `torch.zeros((3, 4))` |
| `torch.ones()` | 全一张量 | `torch.ones((3, 4))` |
| `torch.randn()` | 标准正态分布随机张量 | `torch.randn((3, 4))` |
| `torch.tensor()` | 从 Python 列表创建 | `torch.tensor([[2,1],[3,4]])` |
| `torch.from_numpy()` | 从 NumPy 数组转换 | `torch.from_numpy(np_array)` |

### 2.2 查看张量属性
- `x.shape` — 查看形状
- `x.numel()` — 查看元素总数

---

## 三、数据操作（Data Manipulation）

### 3.1 形状变换（Reshaping）
- `torch.reshape(x, (3, 4))` — 改变张量形状
- `x.view(shape)` — 类似 reshape，但要求内存连续
- 自动维度计算：使用 `-1` 让 PyTorch 自动推断维度大小

### 3.2 索引与切片（Indexing & Slicing）
- 基本索引：`X[0]`（第一行）、`X[-1]`（最后一行）
- 切片：`X[1:3]`（第1到第2行）
- 多维索引：`X[1, 2]`（第1行第2列）
- 赋值操作：`X[1, 2] = 9`、`X[0:2, :] = 12`（批量赋值）

### 3.3 拼接与拆分
- `torch.cat((X, Y), dim=0)` — 沿行拼接
- `torch.cat((X, Y), dim=1)` — 沿列拼接
- 张量拆分（Splitting）

### 3.4 数学运算
- 逐元素运算：加 `+`、减 `-`、乘 `*`、除 `/`、幂 `**`
- 指数函数：`torch.exp(x)`
- 点积：`torch.dot(A, B)`
- 余弦相似度：`torch.nn.functional.cosine_similarity(A, B, dim=0)`
  - 注意：需要浮点类型输入，使用 `.to(dtype=torch.float32)` 转换

### 3.5 广播机制（Broadcasting）
- 当两个张量形状不同时，自动扩展较小张量以匹配较大张量的维度
- 优点：避免不必要的内存分配、简化数学运算
- 示例：`a = tensor([[1],[2],[3]])` + `b = tensor([10,20,30])` → 3×3 矩阵

### 3.6 聚合运算
- `torch.mean(x, dim=0)` — 按列求均值
- 注意：`torch.mean()` 要求浮点类型，需先 `.to(dtype=torch.float32)`

### 3.7 节省内存
- 原地操作可避免分配新内存（如 `X[:] = X + Y`）
- 了解运算是否会创建新的内存空间

---

## 四、数据类型转换

### 4.1 Python 列表 ↔ Tensor
```python
A = [[2,1,4,3],[1,2,3,4]]
A = torch.tensor(A)        # list → tensor
```

### 4.2 NumPy ↔ Tensor
```python
A = torch.tensor(np_array)       # numpy → tensor
A = torch.from_numpy(np_array)   # numpy → tensor（共享内存）
A = tensor.numpy()               # tensor → numpy
```

### 4.3 Tensor → Python 标量
```python
a = torch.tensor([3.5])
a.item()  # → 3.5
```

---

## 五、数据预处理（Data Preprocessing）

### 5.1 为什么需要数据预处理
- 真实数据通常是杂乱的，需要清洗
- 深度学习模型需要结构化的数值数据

### 5.2 关键步骤
1. **读取数据**：使用 Pandas 读取 CSV 文件
2. **处理缺失值**：
   - 删除缺失行（Drop）
   - 用启发式方法填充（如均值填充）
3. **转换分类数据**：将类别型数据转为数值型
4. **转换为 PyTorch Tensor**：Pandas DataFrame → Tensor

---

## 六、线性回归（Linear Regression）

### 6.1 完整工作流程
```
数据准备 → 定义模型 → 选择损失函数 → 选择优化器 → 训练循环 → 可视化
```

### 6.2 数据准备
```python
from sklearn import datasets
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=30, random_state=4)
X = torch.from_numpy(X_numpy).float()   # numpy → tensor，转为float
y = torch.from_numpy(y_numpy).float()
y = y.view(y.shape[0], 1)               # 调整形状为列向量
```
- 考点：`torch.from_numpy()` 转换 + `.float()` 类型转换
- 考点：`view()` 与 `reshape()` 的区别（view 要求内存连续）

### 6.3 定义模型
```python
model = nn.Linear(input_size, output_size)  # nn.Linear(1, 1)
```
- `nn.Linear(in_features, out_features)` 实现 `f = wx + b`

### 6.4 损失函数
```python
criterion = nn.MSELoss()  # 均方误差损失
```

### 6.5 优化器
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降
```
- `model.parameters()` 获取模型可训练参数
- `lr` 为学习率

### 6.6 训练循环（核心考点）
```python
for epoch in range(num_epochs):
    # 1. 前向传播
    y_predicted = model(X)
    # 2. 计算损失
    loss = criterion(y_predicted, y)
    # 3. 反向传播
    loss.backward()
    # 4. 更新参数
    optimizer.step()
    # 5. 梯度清零
    optimizer.zero_grad()
```
- 五步顺序：前向传播 → 计算损失 → 反向传播 → 更新参数 → 梯度清零
- `loss.backward()` 计算梯度
- `optimizer.step()` 根据梯度更新权重
- `optimizer.zero_grad()` 清除累积梯度（PyTorch 默认累积梯度）

### 6.7 可视化结果
```python
predicted = model(X).detach().numpy()  # 从计算图分离，再转numpy
plt.plot(X_numpy, y_numpy, 'ro')       # 原始数据（红点）
plt.plot(X_numpy, predicted, 'b')      # 预测结果（蓝线）
plt.show()
```
- 考点：`.detach()` 将张量从计算图中分离（不再追踪梯度）
- 考点：必须先 `detach()` 再 `.numpy()`，否则报错

---

## 七、练习题汇总

### pytorch_preliminary.ipynb（Exercise 1-4）
| 练习 | 内容 |
|------|------|
| Exercise 1 | 张量创建与形状操作：`arange`、`reshape`、`zeros`、`ones`、`randn` |
| Exercise 2 | 索引与切片：基本索引、多维索引、布尔索引、花式索引 |
| Exercise 3 | 数学运算：逐元素运算、点积、余弦相似度 |
| Exercise 4 | 广播机制：不同形状张量相加、`torch.mean()` 按列求均值 |

### linear_regression.ipynb（Exercise 1-6）
| 练习 | 内容 |
|------|------|
| Exercise 1 | `torch.from_numpy()` 将 numpy 数组转为 tensor |
| Exercise 2 | 使用 `nn.Linear(1,1)` 创建线性模型 |
| Exercise 3 | 定义损失函数 `nn.MSELoss()` |
| Exercise 4 | 定义优化器 `torch.optim.SGD(model.parameters(), lr=0.01)` |
| Exercise 5 | 实现训练循环：前向传播 + 计算损失 |
| Exercise 6 | `.detach().numpy()` 转换后用 Matplotlib 画图 |
