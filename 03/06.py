import torch


batch_size, D_x, D_h, D_y = 64, 1000, 100, 10

x = torch.randn(D_h, D_x)
y = torch.randn(D_h, D_y)

# 使用nn包定义模型和损失函数
model = torch.nn.Sequential(
          torch.nn.Linear(D_x, D_h),
          torch.nn.ReLU(),
          torch.nn.Linear(D_h, D_y),
        )
loss_fn = torch.nn.MSELoss(reduction='sum')

# 使用optim包定义优化器（Optimizer）。Optimizer将会为我们更新模型的权重。
# 这里我们使用SGD优化方法；optim包还包含了许多别的优化算法。
# SGD构造函数的第一个参数告诉优化器应该更新哪些张量。
learning_rate = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(500):

    # 前向传播：通过像模型输入x计算预测的y
    y_pred = model(x)

    # 计算并打印loss
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # 在反向传播之前，使用optimizer将它要更新的所有张量的梯度清零(这些张量是模型可学习的权重)
    optimizer.zero_grad()

    # 反向传播：根据模型的参数计算loss的梯度
    loss.backward()

    # 调用Optimizer的step函数使它所有参数更新
    optimizer.step()