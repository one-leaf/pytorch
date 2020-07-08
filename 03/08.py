import random
import torch

class DynamicNet(torch.nn.Module):
    def __init__(self, D_input, D_hidden, D_output):
        """
        在构造函数中，我们构造了三个nn.Linear实例，它们将在前向传播时被使用。
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_input, D_hidden)
        self.middle_linear = torch.nn.Linear(D_hidden, D_hidden)
        self.output_linear = torch.nn.Linear(D_hidden, D_output)

    def forward(self, x):
        """
        对于模型的前向传播，我们随机选择0、1、2、3，
        并重用了多次计算隐藏层的middle_linear模块。
        由于每个前向传播构建一个动态计算图，
        我们可以在定义模型的前向传播时使用常规Python控制流运算符，如循环或条件语句。
        在这里，我们还看到，在定义计算图形时多次重用同一个模块是完全安全的。
        这是Lua Torch的一大改进，因为Lua Torch中每个模块只能使用一次。
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


batch_size, D_x, D_h, D_y = 64, 1000, 100, 10

x = torch.randn(D_h, D_x)
y = torch.randn(D_h, D_y)

# 实例化上面定义的类来构造我们的模型
model = DynamicNet(D_x, D_h, D_y)

# 构造我们的损失函数（loss function）和优化器（Optimizer）。
# 用随机随机梯度下降训练这个模型会非常困难的，所以我们使用了Adam方法。
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for t in range(500):

    # 前向传播：通过向模型传入x计算预测的y。
    y_pred = model(x)

    # 计算并打印损失
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # 清零梯度，反向传播，更新权重 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()