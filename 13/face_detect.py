# 人脸检测

import cv2
import numpy as np
import os

# 当前路径
curr_dir = os.path.dirname(os.path.abspath(__file__))

# 下载链接：https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
prototxt_path = os.path.join(curr_dir, "deploy.prototxt.txt")
# 下载链接：https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel 
model_path = os.path.join(curr_dir, "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# 加载 Caffe model
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# 读取所需图像
image = cv2.imread(os.path.join(curr_dir, "test.jpeg"))

# 获取图像的宽度和高度
h, w = image.shape[:2]

print("图片高:", h, "图片宽:", w)

# 预处理图像：调整大小并执行平均减法。 104.0, 177.0, 123.0 表示b通道的值-104，g-177,r-123  
# 在深度学习中通过减去数人脸据集的图像均值而不是当前图像均值来对图像进行归一化，因此这里写死了
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))

# 将图像输入神经网络
model.setInput(blob)
# 得到结果
output = np.squeeze(model.forward())

font_scale = 1.0

print("output shape:", output.shape)

face_list = []
for i in range(0, output.shape[0]):
    # 置信度 
    confidence = output[i, 2]
    # 如果置信度高于50%，则绘制周围的方框
    if confidence > 0.5:
        # 之前将图片变成300*300，接下来提取检测到的对象的模型的置信度后，我们得到周围的框 output[i, 3:7]，然后将其width与height原始图像的和相乘，以获得正确的框坐标
        box = output[i, 3:7] * np.array([w, h, w, h])
        # 转换为整数
        start_x, start_y, end_x, end_y = box.astype(np.int32)

        face = cv2.resize(image[start_y:end_y, start_x:end_x],(200,200))

        # 绘制矩形
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)
        # 添加文本
        cv2.putText(image, f"{confidence*100:.2f}%", (start_x, start_y-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), 2)
        
        face_list.append(face)


print("检测到人脸数:", len(face_list))

cv2.imshow("image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite(os.path.join(curr_dir,"beauty_detected.jpg"), image)

