# MLP-Mixer
# 参考 https://arxiv.org/pdf/2105.01601.pdf

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

import torch.nn as nn

# 多层感知机，加了dropout
# 输入 x: (n_samples, n_channels, n_patches) 或 (n_samples, n_patches, n_channels)
# 输出：  和输入 x 的张量保存一致
# 构造函数 mlp_dim = 等于 x 的最后一个维度
class MlpBlock(nn.Module):
    def __init__(self, mlp_dim:int, hidden_dim:int, dropout = 0.):
        super(MlpBlock, self).__init__()
        self.Linear1 = nn.Linear(mlp_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.Linear2 = nn.Linear(hidden_dim, mlp_dim)
    def forward(self,x):
        y = self.Linear1(x)
        y = self.gelu(y)
        y = self.dropout(y)
        y = self.Linear2(y)
        y = self.dropout(y)
        return y

# 混合感知机块
# 输入 x： (n_samples, n_patches, hidden_dim)
# 输出 ： 和输入 x 的张量保存一致
class MixerBlock(nn.Module):
    def __init__(self, n_patches: int , hidden_dim: int, token_dim: int, channel_dim: int, dropout = 0.):
        super(MixerBlock, self).__init__()
        self.MLP_block_token = MlpBlock(n_patches, token_dim, dropout)
        self.MLP_block_chan = MlpBlock(hidden_dim, channel_dim, dropout)
        self.LayerNorm_token = nn.LayerNorm(hidden_dim)
        self.LayerNorm_chan = nn.LayerNorm(hidden_dim)

    def forward(self,x):
        # 针对 n_patches 做全连接(token)
        y = self.LayerNorm_token(x)           # (n_samples, n_patches, hidden_dim)
        y = y.permute(0, 2, 1)          # (n_samples, hidden_dim, n_patches)
        y = self.MLP_block_token(y)     # (n_samples, hidden_dim, n_patches)
        y = y.permute(0, 2, 1)          # (n_samples, n_patches, hidden_dim)
        x = x + y   # (n_samples, n_patches, hidden_dim)
        # 针对 hidden_dim 做全连接(channel)
        y = self.LayerNorm_chan(x)  # (n_samples, n_patches, hidden_dim)
        y = self.MLP_block_chan(y) # (n_samples, n_patches, hidden_dim)
        return x + y

# 混合多层感知机网络
# 输入 x (n_samples, n_channels, image_size, image_size)
# 输出 逻辑分类张量 (n_samples, n_classes)
# 构造函数：
# image_size  : 输入图片的边长
# n_channels  : 输入图片的层数
# patch_size  : 图片分割边长，是 image_size 的约数， n_patches 为分割的块数 为（图片边长/分割边长）的平方
# hidden_dim  : 每个图片块的最后维度
# token_dim   : token 混合的维度
# channel_dim : channel 混合的维度
# n_classes   : 输出类别个数
# n_blocks    : 多少个模型块相当于残差的层数
class MLP_Mixer(nn.Module):
    def __init__(self, image_size, n_channels, patch_size, hidden_dim, token_dim, channel_dim, n_classes, n_blocks, dropout = 0.):
        super(MLP_Mixer, self).__init__()
        n_patches =(image_size//patch_size) ** 2 # image_size 可以整除 patch_size
        self.patch_size_embbeder = nn.Conv2d(kernel_size=patch_size, stride=patch_size, in_channels=n_channels, out_channels= hidden_dim)
        self.blocks = nn.ModuleList([
            MixerBlock(n_patches=n_patches, hidden_dim=hidden_dim, token_dim=token_dim, channel_dim=channel_dim, dropout=dropout) for i in range(n_blocks)
        ])

        self.flatten = nn.Flatten(start_dim=2)
        self.Layernorm1 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.classifier.weight.data.fill_(0)
        self.classifier.bias.data.fill_(0)

    def forward(self,x):
        x = self.patch_size_embbeder(x) # (n_samples, hidden_dim, image_size/patch_size, image_size/patch_size)
        x = self.flatten(x)         # (n_samples, hidden_dim, n_patches)
        x = x.permute(0, 2, 1)      # (n_samples, n_patches, hidden_dim)
        for block in self.blocks:
            x = block(x)            # (n_samples, n_patches, hidden_dim)
        x = self.Layernorm1(x)      # (n_samples, n_patches, hidden_dim)
        x = x.mean(dim = 1)         # (n_sample, hidden_dim)
        return self.classifier(x)   # (n_samples, n_classes)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.resnet = torchvision.models.resnet18(num_classes=10)
 
    def forward(self, x):
        x= self.conv(x)
        x= self.resnet(x)
        return x

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
            x, target = x.cuda(), target.cuda()
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

    # # 混淆矩阵数据归一化
    # for i in range(10):
    #     confusion[i] = confusion[i] / confusion[i].sum()

    # # 设置绘图
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(confusion.numpy())
    # fig.colorbar(cax)

    # # 设置轴
    # ax.set_xticklabels([''] + list(range(10)), rotation=90)
    # ax.set_yticklabels([''] + list(range(10)))

    # # 每个刻度线强制标签
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # # sphinx_gallery_thumbnail_number = 2
    # plt.show()

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

# resnet 18                    {test loss: 0.013704, acc: 9905.000}     params: 11,181,648
# vs
# patch_size=14,  hidden_dim=64,  token_dim=32, channel_dim=128 
# n_blocks = 8, dropout = 0   {test loss: 0.055842, acc: 9820.000}     params: 150,378
# vs
# patch_size=7,  hidden_dim=64,  token_dim=32, channel_dim=128 
# n_blocks = 3,  dropout = 0   {test loss: 0.061474, acc: 9789.000}     params: 57,690
# n_blocks = 8,  dropout = 0   {test loss: 0.040890, acc: 9834.000}     params: 147,210
# n_blocks = 8,  dropout = 0.1 {test loss: 0.033129, acc: 9822.000}
# n_blocks = 8,  dropout = 0.5 {test loss: 0.078637, acc: 9743.000}
# n_blocks = 18, dropout = 0   {test loss: 0.060316, acc: 9831.000}     params: 326,250
# n_blocks = 18, dropout = 0.1 {test loss: 0.036490, acc: 9817.000}
# n_blocks = 18, dropout = 0.5 {test loss: 0.047998, acc: 9762.000}
# n_blocks = 34, dropout = 0   {test loss: 0.050164, acc: 9815.000}     params: 612,714
# n_blocks = 50, dropout = 0   {test loss: 0.060606, acc: 9827.000}     params: 899,178
# vs
# patch_size=4,  hidden_dim=64,  token_dim=32, channel_dim=128      
# n_blocks = 8,  dropout = 0   {test loss: 0.042324, acc: 9824.000}     params: 162,258
# vs
# patch_size=2,  hidden_dim=64,  token_dim=32, channel_dim=128 
# n_blocks = 8 , dropout = 0   {test loss: 0.066434, acc: 9793.000}     params: 237,930
# vs 
# patch_size=7,  hidden_dim=64,  token_dim=64, channel_dim=128
# n_blocks = 8,  dropout = 0   {test loss: 0.058035, acc: 9785.000}     params: 155,658
# patch_size=7,  hidden_dim=64,  token_dim=64, channel_dim=256 
# n_blocks = 18, dropout = 0   {test loss: 0.054196, acc: 9833.000}     params: 287,754
# patch_size=7,  hidden_dim=64,  token_dim=128, channel_dim=256 
# n_blocks = 18, dropout = 0   {test loss: 0.032596, acc: 9845.000}     params: 1,242,026
# patch_size=7,  hidden_dim=64,  token_dim=128, channel_dim=512 
# n_blocks = 18, dropout = 0   {}     params: 2,426,282


def main():
    net = MLP_Mixer(
        image_size=28, 
        n_channels=1, 
        patch_size=7, 
        hidden_dim=64,
        token_dim=64, 
        channel_dim=256, 
        n_classes=10, 
        n_blocks=8,
        dropout=0    
        )
   
    # 是否采用GPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()

    # net = ResNet()
    # print(net)
    print("########### print net end ##############")
    print(summary(net,(1,28,28)))
    print("########### print summary end ##############")  

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

    # show(train_loader)

    # 小批量梯度下降方法
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # 损失函数定义 交叉熵损失
    ceriation = nn.CrossEntropyLoss()

    savefile="mnist_mlp_mixer.pt"
    # if os.path.exists(savefile):
        # net.load_state_dict(torch.load(savefile))  #读取网络参数

    # 训练
    # 动态调整学习率
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    for epoch in range(10):
        train(train_loader, net, optimizer, ceriation, use_cuda, epoch)
        scheduler.step()
        torch.save(net.state_dict(), savefile)

    test(test_loader, net, ceriation, use_cuda, epoch)


if __name__ == "__main__":
    main()