import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import os


def main():

    batch_size = 4

    # 加载 CIFAR10 图像数据
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # show images
    imshow(torchvision.utils.make_grid(images))


def imshow(img):
    print(img.size())
    img = img.permute(1,2,0)
    img = img * torch.Tensor([0.229, 0.224, 0.225]) + torch.Tensor([0.485, 0.456, 0.406])    # unnormalize
    npimg = img.numpy()
    plt.imshow(npimg)
    plt.show()


if __name__ == "__main__":
    main()