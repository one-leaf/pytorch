# 03/ - PyTorch 自动微分机制

这个目录包含 PyTorch autograd 机制的深入教程，从手动反向传播到自动微分，涵盖自定义 Function、nn.Sequential 和 optim 优化器。

## 文件清单

### 核心模块

- **01.py**: NumPy 手动反向传播
  - 手动计算前向传播
  - 手动计算梯度（链式法则）
  - 权重更新：`w -= learning_rate * grad`
  - 两层网络：ReLU 激活

- **02.py**: PyTorch autograd 基础
  - `torch.tensor(requires_grad=True)` 自动追踪梯度
  - `.backward()` 自动计算梯度
  - `.grad` 访问梯度
  - `.detach()` 从计算图分离

- **03.py**: 自定义 autograd Function
  - `torch.autograd.Function` 子类
  - `forward()`: 保存中间结果
  - `backward()`: 手动定义梯度
  - ReLU 的自定义实现

- **04.py**: nn.Sequential 构建网络
  - `torch.nn.Sequential` 顺序模型
  - `torch.nn.Linear`, `torch.nn.ReLU`
  - 损失函数：`torch.nn.MSELoss()`

- **05.py**: optim 优化器
  - `torch.optim.SGD`, `torch.optim.Adam`
  - `optimizer.zero_grad()`, `optimizer.step()`
  - 学习率调度器

- **06.py**: 自定义优化器
  - 手动实现 SGD 更新
  - 学习率衰减策略

- **07.py**: 动态图特性
  - PyTorch 动态计算图
  - 控制流（if/while）在计算图中

- **08.py**: 神经网络训练循环
  - 完整训练流程
  - 验证集评估
  - 早停机制

- **09.py**: 神经正切核（NTK）
  - NTK 理论简介
  - 无限宽神经网络的训练动态
  - 核方法对比
