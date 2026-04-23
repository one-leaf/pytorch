# 02/ - PyTorch 自定义数据集

这个目录包含 PyTorch 自定义数据集的创建和使用教程，涵盖 CSV 数据加载、自定义变换和数据可视化。

## 文件清单

### 核心模块

- **01.py**: 面部 landmarks CSV 数据加载
  - pandas `iloc` 读取 CSV 文件
  - matplotlib scatter 可视化面部关键点
  - 数据解析：图像路径、坐标点
  - 坐标转换：一维数组 → 二维坐标点

- **02.py**: 自定义 FaceLandmarksDataset 类
  - `FaceLandmarksDataset` 类继承 `torch.utils.data.Dataset`
  - `__len__`: 返回数据集大小
  - `__getitem__`: 获取单个样本（图像 + landmarks）
  - `Rescale` 变换：缩放到指定尺寸
  - `RandomCrop` 变换：随机裁剪
  - `Compose` 变换：组合多个变换
  - `DataLoader` 使用：batch_size, shuffle, num_workers
