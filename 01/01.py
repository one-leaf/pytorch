import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchvision import datasets, transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os

# 定义模型
class Net(nn.Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        self.fc1 = nn.Linear(4*4*20, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 训练
def train(train_loader, net, optimizer, ceriation, use_cuda, epoch):
    # 启用 BN 和 Dropout
    net.train()
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        # 梯度先清零
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        out = net(x)
        loss = ceriation(out, target)
        ave_loss = ave_loss * 0.9 + loss.item() * 0.1
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
            print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                epoch, batch_idx+1, ave_loss))

def test(test_loader, net, ceriation, use_cuda, epoch):
    # 固定住 BN 和 Dropout
    net.eval()

    # 创建混淆矩阵
    confusion = torch.zeros(10, 10)

    correct_cnt, ave_loss = 0, 0
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, targe = x.cuda(), target.cuda()
        out = net(x)
        loss = ceriation(out, target)
        pred = out.argmax(dim=1, keepdim=True)

        # 计算正确率
        correct_cnt += pred.eq(target.view_as(pred)).sum().item()
        # smooth average
        ave_loss = ave_loss * 0.9 + loss.item() * 0.1
        
        if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_loader):
            print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                epoch, batch_idx+1, ave_loss, correct_cnt))

        # 显示混淆矩阵，但只显示错误相关性
        for idx in range(len(target)):
            if target[idx]!=pred[idx]:
                confusion[target[idx]][pred[idx]] += 1

    # 混淆矩阵数据归一化
    for i in range(10):
        confusion[i] = confusion[i] / confusion[i].sum()

    # 设置绘图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # 设置轴
    ax.set_xticklabels([''] + list(range(10)), rotation=90)
    ax.set_yticklabels([''] + list(range(10)))

    # 每个刻度线强制标签
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

def show(train_loader):
    images, label = next(iter(train_loader))
    images_example = torchvision.utils.make_grid(images)
    images_example = images_example.numpy().transpose(1,2,0) # 将图像的通道值置换到最后的维度，符合图像的格式
    mean = [0.1307,]
    std =  [0.3081,]
    images_example = images_example * std + mean
    plt.imshow(images_example, cmap="gray")
    plt.show()

    images_example = images[0]#把一个批数的训练数据的第一个取出
    print(images_example.shape)
    images_example = images_example.reshape(28,28) #转换成28*28的矩阵
    plt.imshow(images_example, cmap="gray")
    plt.show()

def main():
    net = Net()
    print(net)
    print(summary(net,(1,28,28)))

    # 是否采用GPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    # 下载MNIST数据并进行归一化处理
    root = "./data"

    # transforms.ToTensor() 将 Image [0,255] 转为 FloatTensor [0.0, 1.0] 注意数据轴会发生变化 [H W C] ==> [C H W]
    # transforms.Normalize((0.5,), (1.0,)) 将 [0.0, 1.0] 转为 [-0.5, 0.5] 区间
    # Normalize公式： output = (input - mean) / std 即： (0-0.5)/1.0 = -0.5  1-0.5/1.0 = 0.5 
    # mean 和 std 就是数据的均值和方差，需要统计求出来。
    # 针对MNIST数据集黑白图像，则采用 (0.1307,), (0.3081,) 这样可以数据更好的分布  
    # 如果是imagenet数据集（RGB），则 transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)))
    transform = transforms.Normalize((0.1307,), (0.3081,))
    # transforms.Compose() 将多个处理方法集成一起
    trans = transforms.Compose([transforms.ToTensor(), transform])
    train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = datasets.MNIST(root=root, train=False, transform=trans)
    batch_size = 128
    
    # 加载训练集和测试集
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,  batch_size=batch_size, shuffle=False)

    show(train_loader)

    # 小批量梯度下降方法
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    # 损失函数定义 交叉熵损失
    ceriation = nn.CrossEntropyLoss()

    savefile="mnist_cnn.pt"
    if os.path.exists(savefile):
        net.load_state_dict(torch.load(savefile))  #读取网络参数

    # 训练
    # 动态调整学习率
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    for epoch in range(1):
        train(train_loader, net, optimizer, ceriation, use_cuda, epoch)
        test(test_loader, net, ceriation, use_cuda, epoch)
        scheduler.step()

    torch.save(net.state_dict(), savefile)

if __name__ == "__main__":
    main()