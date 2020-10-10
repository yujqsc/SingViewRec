import torch
from utils.ctprojection import ctprojection
class projection3d2d(torch.nn.Module):
    def __init__(self):
        super(projection3d2d,self).__init__()
    def forward(self,input,args):
        return ctprojection.apply(input,args)

