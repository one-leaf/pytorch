# 04/ - PyTorch 高级应用

这个目录包含 PyTorch 在迁移学习、目标检测、生成模型、强化学习等领域的高级应用教程。

## 文件清单

### 核心模块

- **01.py**: 迁移学习（微调）
  - ResNet 预训练模型
  - 替换最后全连接层
  - 冻结/解冻层训练
  - 蚂蚁/蜜蜂二分类

- **02.py**: 迁移学习（特征提取）
  - 冻结所有层
  - 只训练新添加的分类器
  - 对比微调方法

- **03.py**: 空间变换器网络（STN）
  - `SpatialTransformerNetwork` 类
  - 定位网络：回归变换参数
  - 网格采样：应用变换
  - 增强模型对几何变换的不变性

- **04.py**: 序列到序列翻译（seq2seq）
  - Encoder-Decoder 架构
  - GRU 循环神经网络
  - 注意力机制
  - 英语→法语翻译

- **05.py**: DCGAN（深度卷积生成对抗网络）
  - Generator：转置卷积上采样
  - Discriminator：卷积下采样
  - 对抗训练：minimax 博弈
  - 生成手写数字图像

- **06.py**: 自定义数据集加载
  - 图像文件夹读取
  - 数据增强管道
  - DataLoader 配置

- **07.py**: 目标检测基础
  - 边界框回归
  - 非极大值抑制（NMS）
  - IoU 计算

- **08.py**: 神经风格迁移
  - VGG 特征提取
  - 内容损失 + 风格损失
  - Gram 矩阵计算
  - 图像优化（非模型训练）

- **09.py**: FGSM 对抗攻击
  - Fast Gradient Sign Method
  - 生成对抗样本
  - 模型鲁棒性测试

- **10.py**: 保存和加载模型
  - `torch.save()`, `torch.load()`
  - 检查点恢复
  - 跨设备加载

- **11.py**: 多 GPU 训练
  - `DataParallel` 使用
  - 分布式训练基础

- **12.py**: 模型可视化
  - TensorBoard 集成
  - 特征图可视化
  - 梯度可视化

- **13.py**: 自定义损失函数
  - Focal Loss
  - Dice Loss
  - 类别不平衡处理

- **14.py**: 模型部署
  - ONNX 导出
  - TorchScript 序列化

- **15.py**: 性能优化
  - 混合精度训练（AMP）
  - 梯度累积
  - 数据加载优化

### 工具模块

- **coco_eval.py**: COCO 评估器
  - 边界框评估
  - 分割掩码评估
  - mAP 计算

- **coco_utils.py**: COCO 工具函数
  - 数据集转换
  - 数据加载器创建

- **engine.py**: 训练/评估引擎
  - `train_one_epoch()`: 单轮训练
  - `evaluate()`: 模型评估
  - 损失监控、学习率调度
  - 多 GPU 分布式损失聚合

- **transforms.py**: 数据变换
  - 图像增强
  - 目标检测专用变换

- **utils.py**: 工具函数
  - `MetricLogger`: 指标记录
  - `SmoothedValue`: 平滑值
  - `warmup_lr_scheduler`: 学习率预热
  - `reduce_dict`: 多 GPU 损失聚合
