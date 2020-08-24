# 生成对抗网络
from __future__ import print_function
#%matplotlib inline
import argparse
import os
from os import curdir
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# 为再现性设置随机seem
# manualSeed = 999
manualSeed = random.randint(1, 10000) # 如果你想要新的结果就是要这段代码
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

curr_dir = os.path.dirname(__file__)
dataroot = os.path.join(curr_dir, "../data/celeba")
workers = 2
batch_size = 128
image_size = 64
nc = 3              # 图像颜色通道 
nz = 100            # 随机参数的向量长度
ngf = 64            # 生成器的特征深度
ndf = 64            # 判别器的特征深度
num_epochs = 10000
lr =  0.0002*0.5     # 学习率为 0.0002 * beta1
beta1 = 0.5         # 应该为 0.5 * GPU个数
ngpu = 1            # GPU 个数

# 我们可以按照设置的方式使用图像文件夹数据集。
# 创建数据集
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.RandomHorizontalFlip(p=0.5),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# 创建加载器
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# 选择我们运行在上面的设备
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

print("runing on", device)

# 绘制部分我们的输入图像
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

# netG 和 netD 所有模型权重应从正态分布中随机初始化，mean = 0，stdev = 0.02。
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# 生成器
# 生成器用于将潜在空间矢量映射到数据空间。输入和输出是一致的
# 跨步的二维卷积转置层
# 生成器的输出通过tanh函数输入，使其返回到[-1,1]范围的输入数据。
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入是 (b, nz, 1, 1)，进入反向卷积
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (b, (ngf*8), 4, 4)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (b, (ngf*4), 8, 8)
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (b, (ngf*2), 16, 16)
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (b, ngf, 32, 32)
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            # (b, nc, 64, 64)
            nn.Tanh()
            # 输出是图片，数据压缩到 [0，1]区间 (b, nc, 64, 64)
        )

    def forward(self, input):
        return self.main(input)

netG = Generator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1): netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)

print(netG)

# 判别器
# 判别器是二进制分类网络，它将图像作为输入并输出输入图像是真实的标量概率（与假的相反）。
# 通过Sigmoid激活函数输出 最终概率
# DCGAN论文提到使用跨步卷积而不是池化到降低采样是一种很好的做法，因为它可以让网络学习自己的池化功能。
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入是图片 (b, nc, 64, 64)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (b, ndf, 32, 32)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (b, (ndf*2), 16, 16)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (b, (ndf*4), 8, 8)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (b, (ndf*8), 4, 4)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # (b, 1, 1, 1)
            nn.Sigmoid()
            # 输出是正常还是虚假图片的概率，数据范围【0~1】 (b, 1) 
        )

    def forward(self, input):
        return self.main(input)

# 创建判别器
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# 应用weights_init函数随机初始化所有权重，mean= 0，stdev = 0.2
netD.apply(weights_init)

# 打印模型
print(netD)

# 加载模型
modle_file = "data/save/13_checkpoint.tar"
if os.path.exists(modle_file):
    checkpoint = torch.load(modle_file, map_location=device)
    netG_sd = checkpoint["netG"] 
    netD_sd = checkpoint["netD"] 
    netG.load_state_dict(netG_sd)
    netD.load_state_dict(netD_sd)

# 初始化BCELoss函数 二进制交叉熵损失
criterion = nn.BCELoss()

# 创建一批潜在的向量，我们将用它来可视化生成器的进程
# 混合噪声，按高斯分布采样 (64, 100, 1, 1)
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# with torch.no_grad():
#     fake = netG(fixed_noise).detach().cpu()
#     img = vutils.make_grid(fake, padding=2, normalize=True)
#     plt.imshow(np.transpose(img,(1,2,0)))
#     plt.show()
#     raise "only test"

# 在训练期间建立真假标签的惯例
real_label = 1
fake_label = 0

# 为 G 和 D 设置 Adam 优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# 训练

# 训练中间状态
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# 每一轮训练
for epoch in range(num_epochs):
    # 对于数据加载器中的每个batch
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## 用真实样本进行训练
        netD.zero_grad()
        # [128, 3, 64, 64]
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # 输出真实图片的概率 D [128]
        output = netD(real_cpu).view(-1)
        # 计算真实图片和真实标签的的损失
        errD_real = criterion(output, label)
        # 计算真实样本下D的梯度
        errD_real.backward()
        # 真样本的概率 1 --> 0.5
        D_x = output.mean().item()

        ## 用假样本进行训练
        # 输入一组随机高斯分布噪声 [128, 100, 1, 1] 产生一个假图片
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # 根据随机噪声通过 G 产生假的图片
        fake = netG(noise)
        label.fill_(fake_label)

        # 对假图片进行判别
        output = netD(fake.detach()).view(-1)
        # 计算所有假图片和假标签的损失
        errD_fake = criterion(output, label)
        # 计算假样本下D的梯度
        errD_fake.backward()
        # 假样本的概率 0 --> 0.5
        D_G_z1 = output.mean().item()

        # 将真的和假的损失梯度混合再一起
        # errD_real 最初这应该从接近1开始，随着G提升然后理论上收敛到0.5。
        # errD_fake 最初这应该从接近0开始，随着G提升然后理论上收敛到0.5。
        errD = errD_real + errD_fake
        # 判别器损失计算为所有实际批次和所有假批次的损失总和，同时禁止向下传播假图片G的梯度，只更新 D 的参数
        # 也就是只会计算 真实图片的 D 的梯度 和 假图片的 D 的梯度，让D的判别真图能力增强，不更新 G
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # 多训练几次G
        for _ in range(5):
            netG.zero_grad()
            label.fill_(real_label)  # 假图片却采用真的标签
            # 所有假图片重新计算概率，但允许更新G的梯度
            output = netD(fake).view(-1)
            # 计算假图片和真样本之间的损失
            errG = criterion(output, label)
            # 计算 G 的梯度
            errG.backward()
            # 输出假图片到真标签的距离 0 --> 0.5
            D_G_z2 = output.mean().item()
            # 用假数据却赋予正确标签，同时计算 D 和 G，通过D推动G的学习，但只更新 G 的参数
            optimizerG.step()
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # 根据随机噪声通过 G 产生假的图片
            fake = netG(noise)

        # 输出训练状态
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # 保存损失后续绘制
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # 用固定的噪声产生同样的图片输出，看G的训练过程
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

    torch.save({    'netG': netG.state_dict(),
                    'netD': netD.state_dict(),
                }, modle_file)


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

# 从数据加载器中获取一批真实图像
real_batch = next(iter(dataloader))

# 绘制真实图像
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# 在最后一个epoch中绘制伪图像
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()














