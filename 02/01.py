import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import warnings
warnings.filterwarnings("ignore")

plt.ion()

landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')
print(landmarks_frame)
n = 65
# iloc 选取特定的列
img_name = landmarks_frame.iloc[n, 0]
print('Image name: {}'.format(img_name))

landmarks = landmarks_frame.iloc[n, 1:].to_numpy()
print(landmarks.shape, landmarks)

landmarks = landmarks.astype('float').reshape(-1, 2)
print(landmarks.shape, landmarks)
print('First 4 Landmarks: {}'.format(landmarks[:4]))

def show_lanmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:,0],landmarks[:,1],s=10,marker='.',c='r')
    plt.pause(10)

plt.figure()
show_lanmarks(io.imread(os.path.join('./data/faces/',img_name)),landmarks)
plt.show()
