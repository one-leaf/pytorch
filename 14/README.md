# 14/ - 石头剪刀布 AI（MediaPipe 手势识别）

这个目录包含石头剪刀布游戏的 AI 实现，使用 MediaPipe 进行手部关键点检测，RNN 进行手势预测。

## 文件清单

### 核心模块

- **getData.py**: 数据采集
  - MediaPipe Hands 模型
  - 手部关键点提取
  - 数据录制和保存
  - 类别标签分配
  - 实时摄像头捕捉

- **hand.py**: 手势处理
  - 手部关键点预处理
  - 特征工程
  - 数据增强
  - 序列构建

- **model.py**: RNN 预测模型
  - 循环神经网络
  - LSTM/GRU 层
  - 全连接分类层
  - 手势类别输出（石头/剪刀/布）
  - 序列预测

- **run.py**: 主运行程序
  - 模型加载
  - 实时预测
  - 游戏逻辑
  - 结果显示

- **train.py**: 训练循环
  - 加载采集数据
  - 模型训练
  - 交叉熵损失
  - 准确率评估
  - 模型保存

- **test.py**: 模型测试
  - 测试集评估
  - 混淆矩阵
  - 准确率/召回率/F1
  - 错误分析

### 其他文件

- **README.md**: 项目说明
  - 使用方法
  - 数据采集流程
  - 模型训练步骤

- **data/**: 采集的手势数据
- **model/**: 训练好的模型

## 安装

1. 安装 miniconda
   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/

2. 安装 opencv

## 参考

- MediaPipe Hands
- RNN/LSTM 序列预测
- 石头剪刀布游戏逻辑

 pip install opencv-python opencv-contrib-python -i https://pypi.tuna.tsinghua.edu.cn/simple

3 安装 mediapipe

 pip install mediapipe -i https://pypi.tuna.tsinghua.edu.cn/simple

4 安装 pytorch

 conda install pytorch torchvision torchaudio cpuonly -c pytorch-lts
