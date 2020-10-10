from torch.autograd import Function,Variable
import torch
import CTProjection
from utils.DataStruct import DataStruct

class ctprojection(Function):
    @staticmethod
    def forward(ctx,input:torch.Tensor,args:DataStruct):

        #assert input.is_contiguous()
        ctx.args=args
        ctx.In_shape=input.size(0)*input.size(1)*input.size(2)*input.size(3)*input.size(4)
        out=torch.cuda.FloatTensor(args.na*args.nb*args.nv).fill_(0)
        CTProjection.forward(input.reshape(-1,1), out.reshape(-1,1), args.sd_phi, args.sd_z, args.y_det, args.z_det, args.id_Y,
                             args.Nv, args.SO, args.OD, args.scale, args.dz, args.nx,
                             args.ny, args.nz, args.nt,
                             args.na, args.nb, args.nv,
                             args.tmp_size,
                             args.nv_block)
        ctx.out=out
        return out.reshape(args.nv,args.nb,args.na).permute(0,1,2).unsqueeze(0)

    def backward(ctx, grad_outputs):
        #grad_out=grad_outputs.data.contiguous()
        args=ctx.args
        grad_3d=Variable(torch.cuda.FloatTensor(ctx.In_shape,1).fill_(0))
        CTProjection.backward(grad_3d,grad_outputs.data.contiguous().reshape(-1,1),args.cos_phi,args.sin_phi,args.sd_z,args.y_det,
                              args.z_det,args.id_Y,args.Nv,args.SO,args.OD,args.scale,args.dy_det,
                              args.y_os, args.dz_det, args.dz, args.nx,args.ny,args.nz, args.nt,
                              args.na, args.nb, args.nv, args.tmp_size, args.nv_block)

        return grad_3d.reshape(args.nz,args.nx,args.ny).permute(0,1,2).unsqueeze(0).unsqueeze(0),None


