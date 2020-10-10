from torch.nn import functional as F
import math
import torch
import scipy.io as scio
import matplotlib.pyplot as plt
A_path='./dataset/trainB/SVCTdattest1_1.mat'
A_path1='./dataset/trainB/SVCTdattest1_30.mat'
data=torch.from_numpy(scio.loadmat(A_path)['data']).unsqueeze(0).float()
data1=torch.from_numpy(scio.loadmat(A_path1)['data']).unsqueeze(0).float()
angle = 30*math.pi/180
theta = torch.tensor([
    [math.cos(angle),math.sin(-angle),0],
    [math.sin(angle),math.cos(angle) ,0]
], dtype=torch.float)
theta=theta.unsqueeze(0).expand(1,-1,-1)
data=data.expand(1,-1,-1,-1)
grid=F.affine_grid(theta,data.size(),align_corners=False)
output = F.grid_sample(data, grid,mode='nearest',align_corners=False)
new_img_torch = output[0,32]
old_img_torch = data1[0,32]

plt.imshow((new_img_torch-old_img_torch).numpy())
plt.show()
plt.imshow(old_img_torch.numpy())
plt.show()
plt.pause(5)