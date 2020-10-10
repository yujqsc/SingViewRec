import  torch
from  utils.DataStruct import DataStruct
import numpy as np

class InitParam_CT:
    def __init__(self):
        return
    def parse(self):
        # 设置参数
        args=DataStruct()
        args.nx=65                        #断层分辨率
        args.ny=args.nx
        args.nz=65                       #断层层数
        args.dx=0.25                  #体素尺寸
        args.dy=args.dx
        args.dz=0.25
        args.nv=1                     #视角数量
        args.SO=110.0                          #射线到转轴
        args.OD=60.0                        # 转轴到探测器
        args.na=257#探测器分辨率height
        args.nb=257#探测器分辨率width
        args.dy_det=0.1 #探测器像元尺寸
        args.dz_det=args.dy_det#探测器像元尺寸
        args.sd_phi=torch.tensor((2*np.pi)/args.nv*torch.arange(0,args.nv))
        args.cos_phi=torch.tensor(np.cos(args.sd_phi))
        args.sin_phi=torch.tensor(np.sin(args.sd_phi))
        args.sd_z=torch.zeros([1,args.nv])
        args.y_os=0*args.dy_det#焦点水平偏移
        args.y_det=(torch.arange(-args.na/2,args.na/2)+0.5)*args.dy_det+args.y_os
        args.z_det=(torch.arange(-args.nb/2,args.nb/2)+0.5)*args.dz_det
        # 尺度缩放  normalize so that dx=dy=1 %
        args.scale=args.dx
        args.sd_z=args.sd_z/args.scale
        args.sd_z=args.sd_z/args.scale;args.SO=args.SO/args.scale;args.OD=args.OD/args.scale;
        args.y_det=args.y_det/args.scale;args.z_det=args.z_det/args.scale;
        args.dy_det=args.dy_det/args.scale;args.dz_det=args.dz_det/args.scale;
        args.y_os=args.y_os/args.scale;args.dz=args.dz/args.scale;
        Idv=torch.arange(0,args.nv)
        args.id_X=(Idv*0).int()
        args.id_Y=[]
        args.id_Y=Idv.view(-1,1).int()
        args.Nv=torch.tensor(args.nv).int()
        args.tmp_size=args.nv
        args.nv_block=4
        args.nt = 1
        self.args=args
        return self.args

class RegisterCT:
    def __init__(self):
        return
    def param(self):
        self.args=InitParam_CT().parse()
        return self.args
