# from https://zhuanlan.zhihu.com/p/682231092
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import numpy as np

# hyper-params
in_dim = 1
# hidden_dim = 400
hidden_dim = 256
# n_layer = 4
n_layer = 3
out_dim = 1
n_samples = 200
eps = 1e-3

x_train = 3.0 * torch.randn(n_samples, 1)
# y_train = x_train.pow(2) # use ntk approximate y=x^2
y_train = np.sin(x_train)

x_fun = torch.linspace(-3, 3, 100)
# y_fun = x_fun.pow(2)
y_fun = np.sin(x_fun)
model = []
model.extend([nn.Linear(in_features=in_dim, out_features=hidden_dim), nn.ReLU()])
for i in range(n_layer-1):
    model.extend([nn.Linear(in_features=hidden_dim, out_features=hidden_dim), nn.ReLU()])

model.extend([nn.Linear(in_features=hidden_dim, out_features=out_dim)])
model = nn.Sequential(*model)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

fake_loss = model(x_train).sum()
fake_loss.backward()

# 参数的长度132353，每一层都需要计算梯度，所以需要除以sqrt(hidden_dim)，确保梯度大小一致
param_dim = torch.cat(
        [w.grad.detach().flatten() / np.sqrt(hidden_dim) for w in model.parameters() if w.requires_grad]
    ).size()[0]

# 计算梯度
def compute_grad(model, optimizer, x):
    assert x.shape[0] == 1
    y = model(x)
    optimizer.zero_grad()
    y.backward()
    return torch.cat(
        [w.grad.detach().flatten() / np.sqrt(hidden_dim) for w in model.parameters() if w.requires_grad]
    )

# 计算雅克比矩阵，根据样本计算梯度/偏导数
def compute_jac(model, optimizer, xs):
    jacobian = torch.zeros(xs.shape[0], param_dim, device=xs.device)
    for i, x in enumerate(xs):
        x = x.view(1, -1)
        jacobian[i] = compute_grad(model, optimizer, x)
    return jacobian

# 求雅可比矩阵的逆
def compute_kernel_inv(jac, eps=1e-5):
    return torch.linalg.inv(torch.matmul(jac,jac.T)+eps*torch.diag(torch.ones(n_samples)))

def predict(xs_test, x, y, jac, G_inv, model, optimizer):
    ys_test = torch.zeros(xs_test.shape[0], device=xs_test.device)
    for i, x_test in enumerate(xs_test):
        x_test = x_test.view(1,-1)
        # 计算单个验证数据的梯度
        g = compute_grad(model, optimizer, x_test).view(1,-1) # (1, param-dim,)
        # 与雅可比矩阵的转置相乘得到协方差矩阵
        K = torch.matmul(g, jac.T) # (1, n_samples)
        # y_test = 模型训练集x_test的结果 + x_test和训练集y的积 - x_test和训练集x的积
        ys_test[i] = model(x_test).detach() + torch.matmul(torch.matmul(K, G_inv), y) - torch.matmul(torch.matmul(K, G_inv), model(x_train).detach())
    return ys_test

# exec
jac = compute_jac(model, optimizer, xs=x_train)
G_inv = compute_kernel_inv(jac, eps=eps)

y_test = predict(x_fun, x_train, y_train, jac, G_inv, model, optimizer)
plt.plot(x_fun, y_test.numpy())
plt.plot(x_fun, y_fun)
plt.show()
print(G_inv)
print('='*20)
print(jac)