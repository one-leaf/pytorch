# -*- coding: utf-8 -*-
import torch


batch_size, D_x, D_h, D_y = 64, 1000, 100, 10

#创建随机输入和输出数据
x = torch.randn(batch_size, D_x)
y = torch.randn(batch_size, D_y)

# 为权重创建随机Tensors。
# 设置requires_grad = True表示我们想要计算梯度
# 在向后传递期间更新这些张量。
w1 = torch.randn(D_x, D_h, requires_grad=True)
w2 = torch.randn(D_h, D_y, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 前向传播：使用tensors上的操作计算预测值y; 
      # 由于w1和w2有requires_grad=True，涉及这些张量的操作将让PyTorch构建计算图，
    # 从而允许自动计算梯度。由于我们不再手工实现反向传播，所以不需要保留中间值的引用。
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # 使用Tensors上的操作计算和打印损失。
    # loss是一个形状为()的张量
    # loss.item() 得到这个张量对应的python数值
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # 使用autograd计算反向传播。这个调用将计算loss对所有requires_grad=True的tensor的梯度。
    # 这次调用后，w1.grad和w2.grad将分别是loss对w1和w2的梯度张量。
    loss.backward()

    # 使用梯度下降更新权重。对于这一步，只想对w1和w2的值进行改变；不想为更新阶段构建计算图，
    # 所以使用torch.no_grad()上下文管理器防止PyTorch为更新构建计算图
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 反向传播后手动将梯度设置为零
        w1.grad.zero_()
        w2.grad.zero_()