import numpy as np

# batch_size 是批量大小; D_x是输入维度;
# 49/5000 D_h是隐藏的维度; D_y是输出维度。
batch_size, D_x, D_h, D_y = 64, 1000, 100, 10

# 创建随机输入和输出数据
x = np.random.randn(batch_size, D_x)
y = np.random.randn(batch_size, D_y)

# 随机初始化权重
w1 = np.random.randn(D_x, D_h)
w2 = np.random.randn(D_h, D_y)

learning_rate = 1e-6
for t in range(500):
    # 前向传递：计算预测值y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # 计算和打印损失loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # 反向传播，计算w1和w2对loss的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # 更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2