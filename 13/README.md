# 13/ - 人脸检测与识别

这个目录包含人脸检测和识别的完整实现，使用 OpenCV DNN 模块进行人脸检测，LBPH 算法进行人脸识别。

## 文件清单

### 核心模块

- **face_detect.py**: 人脸检测
  - OpenCV DNN 模块
  - Caffe SSD 模型
  - 部署 prototxt 加载
  - 人脸边界框预测
  - 置信度阈值过滤
  - 图像预处理（缩放、归一化）

- **face_recognizer.py**: 人脸识别
  - LBPH（局部二值模式直方图）算法
  - 人脸特征提取
  - 人脸匹配和识别
  - 模型训练和保存
  - 预测置信度评估

### 模型文件

- **deploy.prototxt.txt**: Caffe 部署配置文件
  - SSD 网络结构定义
  - 层参数配置
  - 输入输出规范

- **res10_300x300_ssd_iter_140000_fp16.caffemodel**: 预训练人脸检测模型
  - 300×300 输入尺寸
  - SSD 架构
  - 140000 次迭代训练
  - FP16 半精度权重

### 其他文件

- **README.md**: 项目说明
  - 使用方法
  - 依赖说明
  - 示例代码

- **images/**: 示例图像
  - 测试图片
  - 结果展示

## 依赖

```bash
pip install numpy
pip install opencv-python
pip install opencv-contrib-python
```

## 参考

- OpenCV DNN 模块
- Caffe SSD 人脸检测模型
- LBPH 人脸识别算法