# 00/ - PyTorch 基础操作

这个目录包含 PyTorch 张量操作和梯度计算的基础教程，涵盖了张量创建、算术运算、重塑、索引、view 操作以及 autograd 自动微分机制。

## 文件清单

### 核心模块

- **01.py**: PyTorch 张量基础操作
  - 张量创建：`torch.tensor()`, `torch.rand()`, `torch.zeros()`, `torch.ones()`
  - 算术运算：加法、乘法、矩阵乘法（`@`）
  - 重塑操作：`view()`, `reshape()`
  - 索引操作：基本索引、布尔索引
  - 标量提取：`.item()`
  - 设备操作：`.to(device)`, `.cuda()`, `.cpu()`

- **02.py**: PyTorch 梯度计算（autograd）
  - `requires_grad=True` 启用梯度追踪
  - `backward()` 计算梯度
  - `.grad` 访问梯度值
  - `torch.no_grad()` 上下文禁用梯度计算
  - 自定义梯度函数
  - 计算图概念
