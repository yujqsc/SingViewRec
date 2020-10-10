import torch
import scipy.io as scio
import torch.nn as nn
import torch.functional as Function
#x = torch.tensor([[[1,2],[3,4]]]).float()
data = scio.loadmat('C:/MyPJ/ProjFourier/dataset/trainA/' +'0010.mat')
x =torch.from_numpy(data['data'])
x_fft=torch.rfft(x,2)
x_ifft=torch.irfft(x_fft, 2,signal_sizes=x.shape)
out= torch.max(torch.abs(x_ifft-x))
a=1


# class FFT(nn.Module):
#     def __init__(self,input_features,output_features,bias=True):
#         super(FFT,self).__init__()
#         self.input_features=input_features
#         self.output_features=output_features
#
#         self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(output_features))
#         else:
#             # You should always register all possible parameters, but the
#             # optional ones can be None if you want.
#             self.register_parameter('bias', None)
#
#     def forward(self,input):
#         return
#     def extra_repr(self):
#         return 'in_features={},out_features={},bias={}'.format(self.input_features,self.output_features,self.bias is not None)
# class fftFunction(nn.Module):

