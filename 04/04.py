import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# https://github.com/pytorch/vision/blob/master/references/detection/transforms.py
import transforms as T

# https://github.com/pytorch/vision/blob/master/references/detection/engine.py
from engine import train_one_epoch, evaluate

# https://github.com/pytorch/vision/blob/master/references/detection/utils.py
import utils

class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # 下载所有图像文件，为其排序
        # 确保它们对齐
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # 请注意我们还没有将mask转换为RGB,
        # 因为每种颜色对应一个不同的实例
        # 0是背景
        mask = Image.open(mask_path)
        # 将PIL图像转换为numpy数组
        # mask [h, w]
        # 注意这个图片的高和宽不是固定的
        mask = np.array(mask)

        # 实例被编码为不同的颜色
        # 去重得到序列，并按从小到大排列
        obj_ids = np.unique(mask)
        # 第一个id是背景，所以删除它
        # obj_ids [obj_ids_len]
        obj_ids = obj_ids[1:]

        # 将颜色编码的mask分成一组
        # 二进制格式
        # masks [obj_ids_len, h, w]
        masks = mask == obj_ids[:, None, None]
        
        # 获取每个mask的边界框坐标
        # num_objs 目标的个数
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            # pos 输出符合条件的所有坐标
            # ([1,2,3],[4,5,6])
            pos = np.where(masks[i])        
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        # 将所有转换为torch.Tensor
        # boxes [obj_ids_len, 4]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # 这里仅有一个类
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        
        # 计算所有box面积 area [obj_ids_len]
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 假设所有实例都不是人群
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # 加载在COCO上预训练的预训练的实例分割模型
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # 获取分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 用新的头部替换预先训练好的头部
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 现在获取掩膜分类器的输入特征数
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # 并用新的掩膜预测器替换掩膜预测器
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

# 定义图片预处理
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # 在GPU上训练，若无GPU，可选择在CPU上训练
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 我们的数据集只有两个类 - 背景和人
    num_classes = 2
    # 使用我们的数据集和定义的转换
    dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('data/PennFudanPed', get_transform(train=False))

    # 在训练和测试集中拆分数据集
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # 定义训练和验证数据加载器
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # 使用我们的辅助函数获取模型
    model = get_model_instance_segmentation(num_classes)

    # 将我们的模型迁移到合适的设备
    model.to(device)

    # 构造一个优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # 和学习率调度程序
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # 训练10个epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # 训练一个epoch，每10次迭代打印一次
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # 更新学习速率
        lr_scheduler.step()
        # 在测试集上评价
        evaluate(model, data_loader_test, device=device)

    print("That's it!")

if __name__ == "__main__":
    main()
