# MLP-Mixer
# 参考 https://github.com/jankrepl/mildlyoverfitted/blob/master/github_adventures/mixer/ours.py

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
import einops

# 多层感知机，加了dropout
# 输入 x: (n_samples, n_channels, n_patches) 或 (n_samples, n_patches, n_channels)
# 输出：  和输入 x 的张量保存一致
# 构造函数 mlp_dim = 等于 x 的最后一个维度
class MLPBlock(nn.Module):
    def __init__(self, mlp_dim:int, hidden_dim:int, dropout = 0.):
        super(MLPBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.Linear1 = nn.Linear(mlp_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.Linear2 = nn.Linear(hidden_dim, mlp_dim)
    def forward(self,x):
        x = self.Linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        x = self.dropout(x)
        return x

# 混合感知机块
# 输入 x： (n_samples, n_patches, hidden_dim)
# 输出 ： 和输入 x 的张量保存一致
class MixerBlock(nn.Module):
    def __init__(self, n_patches: int , hidden_dim: int, token_dim: int, channel_dim: int, dropout = 0.):
        super(MixerBlock, self).__init__()
        self.n_patches = n_patches
        self.channel_dim = channel_dim
        self.token_dim = token_dim
        self.dropout = dropout

        self.MLP_block_token = MLPBlock(n_patches, token_dim, self.dropout)
        self.MLP_block_chan = MLPBlock(hidden_dim, channel_dim, self.dropout)
        self.LayerNorm = nn.LayerNorm(hidden_dim)

    def forward(self,x):
        # 针对 n_patches 做全连接(token)
        out = self.LayerNorm(x)             # (n_samples, n_patches, hidden_dim)
        out = out.permute(0, 2, 1)          # (n_samples, hidden_dim, n_patches)
        out = self.MLP_block_token(out)     # (n_samples, hidden_dim, n_patches)
        out = out.permute(0, 2, 1)          # (n_samples, n_patches, hidden_dim)
        out = x + out   # (n_samples, n_patches, hidden_dim)
        # 针对 hidden_dim 做全连接(channel)
        out2 = self.LayerNorm(out)  # (n_samples, n_patches, hidden_dim)
        out2 = self.MLP_block_chan(out2) # (n_samples, n_patches, hidden_dim)
        res = out + out2
        return res

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
    def __init__(self, image_size, n_channels, patch_size, hidden_dim, token_dim, channel_dim, n_classes, n_blocks):
        super(MLP_Mixer, self).__init__()
        n_patches =(image_size//patch_size) **2 # image_size 可以整除 patch_size
        self.patch_size_embbeder = nn.Conv2d(kernel_size=patch_size, stride=patch_size, in_channels=n_channels, out_channels= hidden_dim)
        self.blocks = nn.ModuleList([
            MixerBlock(n_patches=n_patches, hidden_dim=hidden_dim, token_dim=token_dim, channel_dim=channel_dim) for i in range(n_blocks)
        ])

        self.Layernorm1 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self,x):
        out = self.patch_size_embbeder(x) # (n_samples, hidden_dim, image_size//patch_size, image_size//patch_size)
        out = einops.rearrange(out,"n c h w -> n (h w) c")  # (n_samples, n_patches, hidden_dim)
        for block in self.blocks:
            out = block(out)            # (n_samples, n_patches, hidden_dim)
        out = self.Layernorm1(out)      # (n_samples, n_patches, hidden_dim)
        out = out.mean(dim = 1)         # (n_sample, hidden_dim)
        result = self.classifier(out)   # (n_samples, n_classes)
        return result

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
    net = MLP_Mixer(
        image_size=28, 
        n_channels=1, 
        patch_size=7, 
        hidden_dim=64,
        token_dim=32, 
        channel_dim=128, 
        n_classes=10, 
        n_blocks=19
        )
    print(net)
    print("########### print net end ##############")
    print(summary(net,(1,28,28)))
    print("########### print summary end ##############")

    # 是否采用GPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = net.cuda()

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

    savefile="mnist_mlp_mixer.pt"
    if os.path.exists(savefile):
        net.load_state_dict(torch.load(savefile))  #读取网络参数

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