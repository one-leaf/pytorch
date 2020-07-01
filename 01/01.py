import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary

# 定义模型
class Net(nn.Module):
    # 定义层
    def __init__(self):
        super(Net, self).__init__()
        # Conv2d (输入 维度，输出维度， 窗口尺寸)
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        # BatchNorm
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        # Dropout (比例)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # Linear (输入维度， 输出维度)
        self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    # 前向网络
    def forward(self, x):
        # CNN 先 relu 然后再 池化 2*2
        x=F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x,2)
        x=F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x,2)
        x = self.dropout1(x)
        # 扁平化
        x = torch.flatten(x, 1)
        # 全连接
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        # 最后一层不做激活函数，避免最终输出不完整
        x = self.fc3(x)
        # 将结果转为概率
        output = F.log_softmax(x, dim=1)
        return output

def train(train_loader, net, optimizer, ceriation, use_cuda, epoch):
    # trainning
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = net(x)
        loss = ceriation(out, target)
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
            print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                epoch, batch_idx+1, ave_loss))

def main():
    net = Net()
    print(net)
    print(summary(net,(1,32,32)))

    # 是否采用GPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    root = "./data"
    download=True
    # 下载的数据进行归一化处理
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = datasets.MNIST(root=root, train=True, transform=trans, download=download)
    test_set = datasets.MNIST(root=root, train=False, transform=trans)
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

    # 梯度下降方法 随机梯度下降
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # 损失函数定义 交叉熵损失
    ceriation = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(10):
        train(train_loader, net, optimizer, ceriation, use_cuda, epoch)

    # input = torch.randn(1, 1, 32, 32)
    # out = net(input)
    # print(out)



if __name__ == "__main__":
    main()