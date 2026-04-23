# 01/ - PyTorch 图像分类网络

这个目录包含使用 PyTorch 进行图像分类的教程，涵盖 CNN、MLP-Mixer 等架构，以及 MNIST 和 CIFAR-10 数据集的训练和评估。

## 文件清单

### 核心模块

- **01.py**: MNIST 手写数字分类 CNN
  - 网络结构：2 个卷积层 + 2 个全连接层
  - 训练循环：交叉熵损失、Adam 优化器
  - 测试循环：准确率计算
  - 混淆矩阵可视化（matplotlib）
  - 数据归一化：均值 0.1307，标准差 0.3081

- **02.py**: CIFAR-10 图像分类 CNN
  - 网络结构：3 个卷积层 + 3 个全连接层
  - 模型保存/加载：`torch.save()`, `model.load_state_dict()`
  - 每类准确率评估（10 个类别分别统计）
  - ImageNet 归一化：mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  - 数据增强：RandomCrop, RandomHorizontalFlip

- **03.py**: MLP-Mixer 架构
  - `MlpBlock`: 全连接层块（GELU 激活）
  - `MixerBlock`: Token 混合 + Channel 混合
  - `MLP_Mixer`: 完整 MLP-Mixer 模型
  - 与 ResNet 比较
  - Token 混合：跨空间位置的信息交换
  - Channel 混合：跨通道的信息交换
