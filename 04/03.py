# 定义模型
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型
model = TheModelClass()

# 初始化优化器
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 打印模型的状态字典
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# 打印优化器的状态字典
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])


# 保存模型
def save_model(PATH):
    torch.save(model.state_dict(),PATH)

# 加载模型
def load_model(PATH):
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    # 会设置dropout和batch normalization
    model.eval()

# 保存 checkpoint
def save_checkpoint(PATH):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                ...
                }, PATH)

# 加载 checkpoint
def load_checkpoint(PATH):
    model = TheModelClass(*args, **kwargs)
    optimizer = TheOptimizerClass(*args, **kwargs)

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()
    # - or -
    model.train()

# 保存多个模型
def save_models(PATH):
    torch.save({
            'modelA_state_dict': modelA.state_dict(),
            'modelB_state_dict': modelB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            ...
            }, PATH)

# 加载多个模型
def load_models(PATH):
    modelA = TheModelAClass(*args, **kwargs)
    modelB = TheModelBClass(*args, **kwargs)
    optimizerA = TheOptimizerAClass(*args, **kwargs)
    optimizerB = TheOptimizerBClass(*args, **kwargs)

    checkpoint = torch.load(PATH)
    modelA.load_state_dict(checkpoint['modelA_state_dict'])
    modelB.load_state_dict(checkpoint['modelB_state_dict'])
    optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
    optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

    modelA.eval()
    modelB.eval()
    # - or -
    modelA.train()
    modelB.train()

# 将模型A给模型B
def modelA_to_modelB(PATH):
    torch.save(modelA.state_dict(), PATH)
    modelB = TheModelBClass(*args, **kwargs)
    # strict=False 忽略不匹配项
    modelB.load_state_dict(torch.load(PATH), strict=False)

# 将模型加载到cpu
def load_model_cpu(PATH):
    device = torch.device('cpu')
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH, map_location=device))

# 将模型加载到gpu
def load_model_gpu(PATH):
    device = torch.device("cuda")
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.to(device)

# 将模型加载到第一个gpu
def load_model_gpu0(PATH):
    device = torch.device("cuda")
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  
    # 将参数转为cude张量
    model.to(device)

# 保存 多GPU torch.nn.DataParallel 模型
def save_model_gpus(PATH):
    torch.save(model.module.state_dict(), PATH)


