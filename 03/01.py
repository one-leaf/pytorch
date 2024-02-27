import numpy as np

# batch_size 是批量大小; D_x是输入维度;
# D_h是隐藏的维度; D_y是输出维度。
batch_size, D_x, D_h, D_y = 64, 1000, 200, 10

# 随机初始化权重
w1 = np.random.randn(D_x, D_h)
w2 = np.random.randn(D_h, D_y)
b1 = np.zeros((1, D_h))
b2 = np.zeros((1, D_y))

# 需要拟合的函数
# 计算每一批的方差和均值，重新产生一个输出维度为D_y的高斯分布
def dst_fun(x):
    _batch_size = x.shape[0] 
    sigma = np.std(x, axis=1)
    mu = np.average(x, axis=1)
    y = np.zeros((_batch_size, D_y))
    for i in range(_batch_size):
        y[i]=np.random.normal(mu[i], sigma[i], size=D_y)
    return y

# 学习率
learning_rate = 1e-8
# 训练次数
epochs = 50000

for epoch in range(epochs):
    mu = np.random.uniform(-1,1)
    sigma = np.random.uniform(0.1,2)
    x = np.random.normal(mu,sigma,size=(batch_size, D_x))
        
    y = dst_fun(x)

    # 前向传递：计算预测值y
    h = np.dot(x, w1)+b1
    h_relu = np.maximum(h, 0)
    y_pred = np.dot(h_relu, w2)+b2

    # 计算和打印损失loss
    loss = np.mean(np.square(y_pred - y))
    if epoch%100==0:
        print(epoch, loss)

    # 反向传播，计算w1和w2对loss的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = np.dot(h_relu.T, grad_y_pred)
    grad_h_relu = np.dot(grad_y_pred, w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = np.dot(x.T,grad_h)

    # 更新权重
    w1 -= learning_rate * grad_w1
    b1 -= learning_rate * np.sum(grad_h, axis=0, keepdims=True)
    w2 -= learning_rate * grad_w2
    b2 -= learning_rate * np.sum(grad_y_pred, axis=0, keepdims=True)

# 验证模型，产生一个正态分布    
x = np.random.randn(D_x)
x = x.reshape(1, -1)
# 应该输出的结果
y = dst_fun(x)
print("sigma:", np.std(y,axis=1), "mu:", np.average(y,axis=1))
# 预测的结果
y_pred = np.maximum(x.dot(w1)+b1,0).dot(w2)+b2
print("pred:", y_pred)
print("sigma:", np.std(y_pred,axis=1), "mu:", np.average(y_pred,axis=1))