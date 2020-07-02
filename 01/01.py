import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchvision import datasets, transforms
from torchsummary import summary
import os

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
        self.fc1 = nn.Linear(400, 128)
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

    # transforms.ToTensor() 将Image [0,255] 转为 FloatTensor [0.0, 1.0]
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