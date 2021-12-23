import cv2
import numpy as np
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

prototxt_path = os.path.join(curr_dir, "deploy.prototxt.txt")
model_path = os.path.join(curr_dir, "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# 加载 Caffe model
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def train_face_recognizer(recognizer):
    image_path = os.path.join(curr_dir, "images")
    labels = []
    images = []
    for label in os.listdir(image_path):
        if os.path.isfile(os.path.join(image_path, label)): continue
        for image_name in os.listdir(os.path.join(image_path, label )):
            if not image_name.endswith(".jpeg"): continue
            print("reading image:", os.path.join(image_path, label, image_name))
            image = cv2.imread(os.path.join(image_path, label, image_name))
            _, faces = detect_face(image, model)

            for face in faces:
                labels.append(int(label))
                images.append(face)

    recognizer.train(np.array(images), np.array(labels))

def detect_face(image, model, confidence_threshold=0.5, recognizer=None):
    # 获取图像的宽度和高度
    h, w = image.shape[:2]
  
    # 预处理图像：调整大小并执行平均减法。 104.0, 177.0, 123.0 表示b通道的值-104，g-177,r-123  
    # 在深度学习中通过减去数人脸据集的图像均值而不是当前图像均值来对图像进行归一化，因此这里写死了
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))

    # 将图像输入神经网络
    model.setInput(blob)
    # 得到结果
    output = np.squeeze(model.forward())

    font_scale = 1.0

    face_list = []
    for i in range(0, output.shape[0]):
        # 置信度 
        confidence = output[i, 2]
        # 如果置信度高于50%，则绘制周围的方框
        if confidence > confidence_threshold:
            # 之前将图片变成300*300，接下来提取检测到的对象的模型的置信度后，我们得到周围的框 output[i, 3:7]，然后将其width与height原始图像的和相乘，以获得正确的框坐标
            box = output[i, 3:7] * np.array([w, h, w, h])
            # 转换为整数
            start_x, start_y, end_x, end_y = box.astype(np.int32)

            face = cv2.resize(image[start_y:end_y, start_x:end_x], (300, 300))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            if recognizer!=None:
                
                label, confidence = recognizer.predict(face)  
                print(label, confidence)             

                # 绘制矩形
                cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)
                # 添加文本
                if confidence <50:
                    if label == 0:
                        label_str = "zhangxueyou"
                    elif label == 1:
                        label_str = "liudehua"
                else:
                    label_str = "unknown"
                cv2.putText(image, label_str, (start_x, start_y-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), 2)
            
            face_list.append(face)

    return image, face_list


if __name__ == "__main__":

    # 创建一个简单的人脸识别模型
    # 局部二值，置信度 0 完全匹配，50以下可以接受，80以上不行
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    train_face_recognizer(recognizer)
    image, _ = detect_face(cv2.imread(os.path.join(curr_dir, "test.jpeg")), model, recognizer=recognizer)

    cv2.imshow("image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

