import numpy as np
import torch
import cv2
import os
import argparse
import time
import torch.nn.functional as F
# 指定使用CPU还是GPU
parse = argparse.ArgumentParser(description='Centroid algorithm parameter')
parse.add_argument('--device', default='cuda', type=str)
args = parse.parse_args()
if args.device == 'cuda':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = torch.device('cpu')
# 读取文件数据
# time_start = time.time()
folder = '.\data'
names = os.listdir(folder)
data = []
for i in range(5):
    for name in names[0:1000]:  # 在此处更改测试图像的张数
        path = os.path.join(folder, name)
        img = cv2.imread(path, 0)
        # img = np.expand_dims(img, 0)
        data.append(img)
data = np.stack(([i for i in data]), axis=0)
data_tensor = torch.from_numpy(data)
Number, Height, Width = data_tensor.shape
data_tensor = data_tensor.to(DEVICE)
time_start = time.time()
print(data_tensor.device)

sumImg = torch.sum(data_tensor.reshape(Number, -1), dim=1)
sum_x = torch.sum(data_tensor, dim=1)
sum_x = torch.transpose(sum_x, 0, 1).float()
gridX = torch.linspace(1, Width, Width).unsqueeze(0).to(DEVICE)
x = torch.mm(gridX, sum_x) / sumImg

sum_y = torch.sum(data_tensor, dim=2)
sum_y = torch.transpose(sum_y, 0, 1).float()
gridY = torch.linspace(1, Height, Height).unsqueeze(0).to(DEVICE)
y = torch.mm(gridY, sum_y) / sumImg
time_end = time.time()
print('The x dimension is {}, the y dimension is {}\n' .format(x.shape, y.shape))
print('Using {}, Running time is {:.3f}s'.format(DEVICE, (time_end - time_start)))