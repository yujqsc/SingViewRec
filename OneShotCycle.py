# coding=gbk
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.autograd as autograd
import functools
from options.train_options import TrainOptions
from data import create_dataset
from torch.nn import init
import os
from PIL import Image
import os.path
import scipy.io as scio
from models import networks
from util import util
from PIL import Image
from torchsummary import summary
SAMPLE_GAP = 0.2
SAMPLE_NUM = 50
N_GNET = 50
BATCH_SIZE = 65
USE_CUDA = True
MAX_EPOCH = 1000000

LAMBDA=10
CRITIC_ITERS = 5
POINT = np.linspace(0, SAMPLE_GAP * SAMPLE_NUM, SAMPLE_NUM)
load_net=False
epoch_start=1
def calc_gradient_penaltyold(netD, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(0) if torch.cuda.is_available() else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if torch.cuda.is_available():
        interpolates = interpolates.cuda(0)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(0) if torch.cuda.is_available() else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
def calc_gradient_penalty(net_D, real_data, fake_data):
    # print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(0) if torch.cuda.is_available() else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if torch.cuda.is_available():
        interpolates = interpolates.cuda(0)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = net_D(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(
                                  0) if torch.cuda.is_available() else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty
class FFT(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
       return torch.rfft(x,2,onesided=False).squeeze(0).permute(3,0,1,2)

class IFFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.dim()==5:
            x=x.squeeze(0)
        return torch.irfft(x.permute(1,2,3,0), 2,onesided=False).unsqueeze(0)
# 判别器
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 3
        padw = 1
        #sequence = [IFFT()]
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    norm_layer(ndf),
                    #nn.ReLU(True)
                    nn.LeakyReLU(0.2, True)
                    ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
                #nn.ReLU(True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
            #nn.ReLU(True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input.squeeze(0))

DIM=128
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(input_nc,int(DIM/4), 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(int(DIM/4), int(DIM / 2), 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(int(DIM/2), DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            # nn.Conv2d(4 * DIM, 8 * DIM, 3, 2, padding=1),
            # nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4, 1)

    def forward(self, input):
        output = self.main(input.squeeze(0))
        output = output.view(-1, 4*4)
        output = self.linear(output)
        return output

# 生成器


def init_weights(net, init_type='xavier', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
           (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        __patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
def load_network(net,load_path):
    # if isinstance(net, torch.nn.DataParallel):
    #     net = net.module
    print('loading the model from %s' % load_path)
    # if you are using PyTorch newer than 0.4 (e.g., built from
    # GitHub source), you can remove str() on self.device
    state_dict = torch.load(load_path, map_location=str('cuda:0'))
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        __patch_instance_norm_state_dict(state_dict, net, key.split('.'))
    net.load_state_dict(state_dict)

def ceshi(net):
    pathA=os.path.join('C:/MyPJ/ProjFourier128gANCycle/Fourier128Wgan/dataset/testA/SVCTdattest32_34.mat')
    #pathB = os.path.join('C:/MyPJ/ProjFourier128gANCycle/Fourier128Wgan/dataset/testB/SVCTdattest32_34.mat')
    A= torch.from_numpy(scio.loadmat(pathA)['data']).unsqueeze(0).float()
    A = torch.rfft(A, 2, onesided=False).permute(3, 0, 1, 2).squeeze(0)
    A = torch.roll(torch.roll(A, 128, 2), 128, 3)
    #B = torch.from_numpy(scio.loadmat(pathB)['data']).float()
    fake_date=net(A)
    fk_im = toimage(torch.irfft(torch.roll(torch.roll(fake_date, -32, 2), -32, 3).permute(1, 2, 3, 0), 2, onesided=False)[32, :,:].unsqueeze(0))
    # img_test.append(fk_im)
    save_filenamet = 'fake%s.bmp' % 34
    img_path = os.path.join('./check/test_run/', save_filenamet)
    save_image(fk_im, img_path)


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_params(model):
    for param in model.parameters():
        param.requires_grad = True

def main():
    plt.ion()  # 开启interactive mode，便于连续plot
    opt = TrainOptions().parse()
    # 用于计算的设备 CPU or GPU
    device = torch.device("cuda" if USE_CUDA else "cpu")

    # 定义判别器与生成器的网络
    #net_d = NLayerDiscriminator(opt.output_nc, opt.ndf, n_layers=3)#batchnorm
    #net_d = Discriminator(opt.output_nc)
    net_d_ct =networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)
    net_d_dr =networks.define_D(opt.input_nc, opt.ndf, 'ProjNet',
                                             opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)
    net_g_dr=networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)

    #net_g = CTGenerator(opt.input_nc, opt.output_nc, opt.ngf, n_blocks=6)
    net_g_ct = networks.define_G(1, 65, opt.ngf, 'CTnet', opt.norm,
                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
    # init_weights(net_d_dr)
    # init_weights(net_d_ct)
    # init_weights(net_g_dr)
    # init_weights(net_g_ct)
    net_d_ct.to(device)
    net_d_dr.to(device)
    net_g_dr.to(device)
    net_g_ct.to(device)

    one = torch.FloatTensor([1])
    mone = one * -1
    one = one.to(device)
    mone= mone.to(device)
    #summary(net_g_dr, (2,65, 65,65))
    if load_net:
        # save_filename = 'net_d%s.pth' % epoch_start
        # save_path = os.path.join('./check/', save_filename)
        # load_network(net_d, save_path)
        save_filename = 'net_g%s.pth' % epoch_start
        save_path = os.path.join('./check/', save_filename)
        load_network(net_g_ct, save_path)
    # 损失函数
    #criterion = nn.BCELoss().to(device)
    criterion = nn.MSELoss().to(device)
    criterion1 = nn.L1Loss().to(device)

    # 优化器
    optimizer_d = torch.optim.Adam(itertools.chain(net_d_ct.parameters(),net_d_dr.parameters()), lr=0.0001,betas=[0.5,0.9])
    optimizer_g = torch.optim.Adam(itertools.chain(net_g_ct.parameters(),net_g_dr.parameters()), lr=0.0001,betas=[0.5,0.9])

    #optimizer_d = torch.optim.AdamW(net_d.parameters(), lr=0.0001)
    #optimizer_g = torch.optim.AdamW(net_g.parameters(), lr=0.0001)
    #one = torch.FloatTensor([1]).cuda()
    #mone = one * -1
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    def gensample():
        for image in enumerate(dataset):
            yield image
    gen=gensample()
    ii=0

    for epoch in range(MAX_EPOCH):
        # 为真实数据加上噪声
        for it,data in enumerate(dataset):


            #载入数据
            dr_real = autograd.Variable(data['A'].cuda())
            dr_real=dr_real.squeeze(0)
            ct_real = autograd.Variable(data['B'].cuda())
            ct_real=ct_real.squeeze(0)
            # 训练
            #内循环
            freeze_params(net_g_ct)
            freeze_params(net_g_dr)
            unfreeze_params(net_d_ct)
            unfreeze_params(net_d_dr)

            ct_fake = autograd.Variable(net_g_ct(dr_real).data)
            dr_fake = autograd.Variable(net_g_dr(ct_real).data)

            optimizer_d.zero_grad()
            loss_dsc_realct = net_d_ct(ct_real).mean()
            #loss_dsc_realct.backward()
            loss_dsc_fakect = net_d_ct(ct_fake.detach()).mean()
            #loss_dsc_fakect.backward()
            gradient_penalty_ct = calc_gradient_penalty(net_d_ct, ct_real, ct_fake)
            #gradient_penalty_ct.backward()
            loss_d_ct=loss_dsc_fakect - loss_dsc_realct+gradient_penalty_ct
            loss_d_ct.backward()
            Wd_ct=loss_dsc_realct-loss_dsc_fakect

            loss_dsc_realdr = net_d_dr(dr_real).mean()
            #loss_dsc_realdr.backward()
            loss_dsc_fakedr = net_d_dr(dr_fake.detach()).mean()
            #loss_dsc_fakedr.backward()
            gradient_penalty_dr = calc_gradient_penalty(net_d_dr, dr_real, dr_fake)
            #gradient_penalty_dr.backward()
            loss_d_dr = loss_dsc_fakedr - loss_dsc_realdr + gradient_penalty_dr
            loss_d_dr.backward()
            Wd_dr = loss_dsc_realdr - loss_dsc_fakedr
            optimizer_d.step()
            if it%CRITIC_ITERS==0:
            #if True:
                unfreeze_params(net_g_ct)
                freeze_params(net_d_ct)
                unfreeze_params(net_g_dr)
                freeze_params(net_d_dr)


                ct_fake_g=net_g_ct(dr_real)
                dr_fake_g=net_g_dr(ct_real)

                #外循环ct_dr
                # optimizer_g.zero_grad()
                loss_out_dr=criterion1(net_g_ct(dr_fake_g),ct_real)
                # loss_out_dr.backward()
                # optimizer_g.step()
                #net_g_ct.load_state_dict(dict_g_ct)

                #外循环dr_ct
                # optimizer_g.zero_grad()
                loss_out_ct=criterion1(net_g_dr(ct_fake_g),dr_real)
                # loss_out_ct.backward()
                # optimizer_g.step()

                #内循环gan
                loss_g_ct = - net_d_ct(ct_fake_g).mean()
                #loss_g_ct.backward()
                loss_g_dr = - net_d_dr(dr_fake_g).mean()
                #loss_g_dr.backward()
                loss_gan = loss_out_dr + loss_out_ct
                #loss_gan=loss_out_dr+loss_out_ct+loss_g_ct+loss_g_dr
                #loss_gan = criterion(net_g_ct(dr_real), ct_real) + criterion(net_g_dr(ct_real), dr_real)
                optimizer_g.zero_grad()
                loss_gan.backward()
                optimizer_g.step()

            if it%1==0:
                fk_im=toimage(torch.irfft(torch.roll(torch.roll(ct_fake,-32,2),-32,3).permute(1, 2, 3, 0), 2, onesided=False)[32, :, :].unsqueeze(0))

                #fk_im=toimage(ct_fake[0,32,:,:].unsqueeze(0))
                 #img_test.append(fk_im)
                save_filenamet = 'fakect%s.bmp' % int(epoch/dataset_size)
                img_path = os.path.join('./check/img/', save_filenamet)
                save_image(fk_im, img_path)

                rel_im=toimage(torch.irfft(torch.roll(torch.roll(ct_real,-32,2),-32,3).permute(1, 2, 3, 0), 2, onesided=False)[32, :, :].unsqueeze(0))
                #rel_im = toimage(ct_real[0,32, :, :].unsqueeze(0))
                # img_test.append(rel_im)
                save_image(rel_im, os.path.join('./check/img/', 'Realct%s.bmp' % int(epoch)))

                fake_im = toimage(torch.irfft(torch.roll(torch.roll(dr_fake, -128, 2), -128, 3).permute(1, 2, 3, 0), 2, onesided=False))
                #fake_im =toimage(dr_fake.squeeze(0))
                save_image(fake_im, os.path.join('./check/img/', 'fakedr%s.bmp' % int(epoch)))
                ceshi(net_g_ct)
                message = '(epoch: %d, iters: %d, D_ct: %.3f;[real:%.3f;fake:%.3f], G_ct: %.3f, D_dr: %.3f, G_dr: %.3f) ' % (int(epoch), ii,loss_d_ct,loss_dsc_realct,loss_dsc_fakect,loss_g_ct,loss_d_dr,loss_g_dr)
                print(message)

        save_filename = 'net_g%s.pth' % epoch
        save_path = os.path.join('./check/', save_filename)
        torch.save(net_g_ct.cpu().state_dict(), save_path)
        net_g_ct.cuda(0)


def toimage(image_tensor):
    image_numpy = image_tensor[0].data.cpu().float().numpy()  # convert it into a numpy array
    image_numpy = (image_numpy - image_numpy.min()) / (
                image_numpy.max() - image_numpy.min()) * 255.0
    return image_numpy.astype(np.uint8)
def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

if __name__ == '__main__':
    main()