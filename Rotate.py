import numpy as np
import cv2
import torch
import torch.nn.functional as F
import math

def project3D(ct):
    dr = torch.sum(ct, dim=1, keepdim=True)
    return dr

def single_rotate_2d(input_fmap, theta, out_dims=None, **kwargs):
    matrix = np.zeros((1, 6))
    matrix[0,] = np.array([math.cos(theta), - math.sin(theta), 0, math.sin(theta), math.cos(theta), 0])
    #matrix[0,] = np.array([0,math.cos(theta), - math.sin(theta),0, math.sin(theta), math.cos(theta)])
    matrix = torch.FloatTensor(matrix).cuda()

    B = input_fmap.size()[0]
    H = input_fmap.size()[1]
    W = input_fmap.size()[2]

    matrix = torch.reshape(matrix, (B, 2, 3))

    batch_grids = affine_grid_generator_2d(H, W, matrix)

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    out = bilinear_sampler_2d_jchen(input_fmap, x_s, y_s)

    return out

def bilinear_sampler_2d_jchen(img, x, y):
    H = img.size()[1]
    W = img.size()[2]

    max_y = W - 1
    max_x = H - 1

    zero = 0

    x = x.float()
    y = y.float()
    x = 0.5 * ((x + 1.0) * float(max_x))
    y = 0.5 * ((y + 1.0) * float(max_y))

    # grab 4 nearest corner points for each (x_i, y_i) and 2 nearest z-plane, make of 8 nearest corner points
    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = torch.clamp(x0, zero, max_x)
    x1 = torch.clamp(x1, zero, max_x)
    y0 = torch.clamp(y0, zero, max_y)
    y1 = torch.clamp(y1, zero, max_y)

    Ia = get_pixel_value_2d(img, x0, y0)
    Ib = get_pixel_value_2d(img, x0, y1)
    Ic = get_pixel_value_2d(img, x1, y0)
    Id = get_pixel_value_2d(img, x1, y1)

    dx = x - x0.float()
    dy = y - y0.float()

    wa = torch.unsqueeze((1. - dx) * (1. - dy), dim=3)
    wb = torch.unsqueeze((1. - dx) * dy, dim=3)
    wc = torch.unsqueeze(dx * (1. - dy), dim=3)
    wd = torch.unsqueeze(dx * dy, dim=3)

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return out

def affine_grid_generator_2d(height, width, theta):
    num_batch = theta.size()[0]

    y = torch.linspace(-1.0, 1.0, width)
    x = torch.linspace(-1.0, 1.0, height)

    x_t, y_t = torch.meshgrid(x, y)

    x_t_flat = torch.reshape(x_t, (-1,))
    y_t_flat = torch.reshape(y_t, (-1,))
    ones = torch.ones_like(x_t_flat)

    sampling_grid = torch.stack((x_t_flat, y_t_flat, ones))

    sampling_grid = torch.unsqueeze(sampling_grid, 0)  # (1, 3, h*w)

    sampling_grid = sampling_grid.repeat((num_batch, 1, 1))  # (1, 3, h*w), 在这里复制1次

    theta = theta.float()

    sampling_grid = sampling_grid.float().cuda()

    batch_grids = theta.matmul(sampling_grid)

    batch_grids = batch_grids.view(num_batch, 2, height, width)

    return batch_grids

def get_pixel_value_2d(img, x, y):
    shape = x.size()
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    width_tensor = torch.FloatTensor([(width - 1) * 0.5]).cuda()
    height_tensor = torch.FloatTensor([(height - 1) * 0.5]).cuda()
    one = torch.FloatTensor([1.0]).cuda()

    x = x.float().cuda() / (height_tensor) - one

    y = y.float().cuda() / (width_tensor) - one
    indices_test = torch.stack((x, y), 3)
    indices_test = indices_test.float()  # grid_sample 的输入必须是 float
    img = img.permute(0, 3, 1, 2)

    gather_test = F.grid_sample(img, indices_test,align_corners=True)

    gather_test = gather_test.permute(0, 2, 3, 1)

    gather = gather_test.float()

    return gather