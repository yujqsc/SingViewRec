# coding=gbk
import torch
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
SAMPLE_GAP = 0.2
SAMPLE_NUM = 50
N_GNET = 50
BATCH_SIZE = 64
USE_CUDA = True
MAX_EPOCH = 1000
LAMBDA=10
CRITIC_ITERS = 5
POINT = np.linspace(0, SAMPLE_GAP * SAMPLE_NUM, SAMPLE_NUM)
load_net=False
epoch_start=1
def calc_gradient_penalty(netD, real_data, fake_data):
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

def test(net):
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
def main():
    plt.ion()  # 开启interactive mode，便于连续plot
    opt = TrainOptions().parse()
    # 用于计算的设备 CPU or GPU
    device = torch.device("cuda" if USE_CUDA else "cpu")
    # 定义判别器与生成器的网络
    #net_d = NLayerDiscriminator(opt.output_nc, opt.ndf, n_layers=3)#batchnorm
    net_d = Discriminator(opt.output_nc)
    init_weights(net_d)
    #net_g = CTGenerator(opt.input_nc, opt.output_nc, opt.ngf, n_blocks=6)
    net_g = networks.define_G(1, 65, opt.ngf, 'CTnet', opt.norm,
                      False, opt.init_type, opt.init_gain, opt.gpu_ids)
    init_weights(net_g)
    net_d.to(device)
    net_g.to(device)
    if load_net:
        # save_filename = 'net_d%s.pth' % epoch_start
        # save_path = os.path.join('./check/', save_filename)
        # load_network(net_d, save_path)
        save_filename = 'net_g%s.pth' % epoch_start
        save_path = os.path.join('./check/', save_filename)
        load_network(net_g, save_path)
    # 损失函数
    #criterion = nn.BCELoss().to(device)
    criterion = nn.MSELoss().to(device)
    criterion1 = nn.L1Loss().to(device)
    # 真假数据的标签
    true_lable = Variable(torch.ones(BATCH_SIZE)).to(device)
    fake_lable = Variable(torch.zeros(BATCH_SIZE)).to(device)
    # 优化器
    #optimizer_d = torch.optim.Adam(net_d.parameters(), lr=0.0008,betas=[0.3,0.9])
    optimizer_g = torch.optim.Adam(net_g.parameters(), lr=0.001,betas=[0.9,0.9])

    #optimizer_d = torch.optim.AdamW(net_d.parameters(), lr=0.0001)
    #optimizer_g = torch.optim.AdamW(net_g.parameters(), lr=0.0001)
    #one = torch.FloatTensor([1]).cuda()
    #mone = one * -1
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    for epoch in range(MAX_EPOCH):
        if epoch==900:
            at=1
        # 为真实数据加上噪声
        for ii, data in enumerate(dataset):

            g_noises = data['A'].cuda()
            g_noises=g_noises.squeeze(0)
            real_data = data['B'].cuda()
            real_data=real_data.squeeze(0)
            optimizer_g.zero_grad()
            fake_date = net_g(g_noises)
            # aa=torch.roll(torch.roll(real_data,32,2),32,3)
            # a1=aa[0,32,:,:].data.cpu().numpy()
            # bb=torch.roll(torch.roll(aa.flip(2).flip(3),1,2),1,3)
            # b1=bb[0,32,:,:].data.cpu().numpy()
            # c1=b1-a1
            #loss=criterion(torch.roll(torch.roll(fake_date,32,2),32,3),torch.roll(torch.roll(real_data,32,2),32,3))\
                # +criterion(torch.roll(torch.roll(torch.roll(torch.roll(fake_date,32,2),32,3).flip(2).flip(3),1,2),1,3),torch.roll(torch.roll(real_data,32,2),32,3))
            # if ii%2==0:
            #     #loss =criterion(torch.roll(torch.roll(torch.roll(torch.roll(fake_date,32,2),32,3).flip(2).flip(3),1,2),1,3),torch.roll(torch.roll(real_data,32,2),32,3))
            #     loss = criterion(torch.roll(torch.roll(fake_date,32,2),32,3).flip(2).flip(3),torch.roll(torch.roll(real_data,32,2),32,3))
            #
            # else:
            #     loss=criterion(torch.roll(torch.roll(fake_date,32,2),32,3),torch.roll(torch.roll(real_data,32,2),32,3))
            # loss1 = criterion(torch.roll(torch.roll(fake_date,32,2),32,3),torch.roll(torch.roll(real_data,32,2),32,3))
            # p1=fake_date[:,:,0:32,0:65]
            # p2=fake_date[:,:,32,0:32].unsqueeze(2)
            # p2=torch.cat((p2,fake_date[:,:,32,32].unsqueeze(2).unsqueeze(3)),3)
            # p2= torch.cat((p2, fake_date[:, :, 32, 0:32].unsqueeze(2).flip(3)), 3)
            # P=torch.cat((p1,p2),2)
            # P=torch.cat((P,p1.flip(2).flip(3)),2)
            # loss2=criterion(P,real_data)
            loss1 = criterion(fake_date, real_data)
            #loss2 = criterion(torch.abs(torch.roll(torch.roll(fake_date,32,2),32,3)-torch.roll(torch.roll(torch.roll(torch.roll(fake_date,32,2),32,3).flip(2).flip(3),1,2),1,3)),torch.zeros(fake_date.size()).cuda())
            loss2 =criterion1(fake_date,fake_date.flip(2).flip(3))#暂时没有预想的效果
            loss=loss1
            #loss = loss1
            #loss1.backward(retain_graph=True)
            loss.backward()
            optimizer_g.step()


            # #real_data = np.vstack([POINT*POINT + np.random.normal(0, 0.01, SAMPLE_NUM) for _ in range(BATCH_SIZE)])
            # #real_data = np.vstack([np.sin(POINT) + np.random.normal(0, 0.01, SAMPLE_NUM) for _ in range(BATCH_SIZE)])
            # #real_data = Variable(torch.Tensor(real_data)).to(device)
            # # 用随机噪声作为生成器的输入
            # #g_noises = np.random.randn(BATCH_SIZE, N_GNET)
            # #g_noises = Variable(torch.Tensor(g_noises)).to(device)
            #
            #
            # # 训练辨别器
            # # for p in net_d.parameters():  # reset requires_grad
            # #     p.requires_grad = True  # they are set to False below in netG update
            #
            # optimizer_d.zero_grad()
            # # 辨别器辨别真图的loss
            # d_real = net_d(real_data)
            # #loss_d_real = criterion(d_real, true_lable)
            # loss_d_real = -d_real.mean()
            # #loss_d_real.backward()
            # # 辨别器辨别假图的loss
            # fake_date = net_g(g_noises)
            # d_fake = net_d(fake_date.detach())
            # #loss_d_fake = criterion(d_fake, fake_lable)
            # loss_d_fake =d_fake.mean()
            # #loss_d_fake.backward()
            #
            # # train with gradient penalty
            # gradient_penalty = calc_gradient_penalty(net_d, real_data, fake_date)
            # #gradient_penalty.backward()
            #
            # D_cost = loss_d_fake + loss_d_real + gradient_penalty
            # D_cost.backward()
            # Wasserstein_D = loss_d_real - loss_d_fake
            # optimizer_d.step()
            # if ii%CRITIC_ITERS==0:
            #     # 训练生成器
            #     # for p in net_d.parameters():
            #     #     p.requires_grad = False  # to avoid computation
            #     optimizer_g.zero_grad()
            #     fake_date = net_g(g_noises)
            #     d_fake = net_d(fake_date)
            #     # 生成器生成假图的loss
            #     #loss_g = criterion(d_fake, true_lable)
            #     loss_g =-d_fake.mean()
            #     loss_g.backward()
            #     optimizer_g.step()
            #     G_cost = -loss_g
            #     for name, parms in net_g.named_parameters():
            #         if name=='model.2.weight':
            #             print('层:',name,parms.size(),'-->name:', name, '-->grad_requirs:', parms.requires_grad, \
            #               ' -->grad_value:', parms.grad[0])
            #     a = 1
            #     # for name, parms in self.netG_A.named_parameters():

            # 每200步画出生成的数字图片和相关的数据
            if ii % 10 == 0:
                #print(fake_date[0]) plt.ion()
    #             plt.ion()
    #             plt.cla()
    #             # plt.plot(POINT, fake_date[0].to('cpu').detach().numpy(), c='#4AD631', lw=2,
    #             #          label="generated line")  # 生成网络生成的数据
    #             # plt.plot(POINT, real_data[0].to('cpu').detach().numpy(), c='#74BCFF', lw=3, label="real sin")  # 真实数据
    #             #prob = (loss_d_real.mean() + 1 - loss_d_fake.mean()) / 2.
    #
    #             img_test=[]
                 fk_im=toimage(torch.irfft(torch.roll(torch.roll(fake_date,-32,2),-32,3).permute(1, 2, 3, 0), 2, onesided=False)[32, :, :].unsqueeze(0))
                 #fk_im = toimage(torch.irfft(fake_date.permute(1, 2, 3, 0), 2, onesided=False)[32, :, :].unsqueeze(0))
                 #fk_im=toimage(fake_date[0,32,:,:].unsqueeze(0).unsqueeze(0))
                 #img_test.append(fk_im)
                 save_filenamet = 'fake%s.bmp' % epoch
                 img_path = os.path.join('./check/img/', save_filenamet)
                 save_image(fk_im, img_path)

                 rel_im=toimage(torch.irfft(torch.roll(torch.roll(real_data,-32,2),-32,3).permute(1, 2, 3, 0), 2, onesided=False)[32, :, :].unsqueeze(0))
                # img_test.append(rel_im)
                 save_image(rel_im, os.path.join('./check/img/', 'Real%s.bmp' % epoch))
                 test(net_g)
                 message = '(epoch: %d, iters: %d, loss1: %.3f, loss2: %.3f) ' % (epoch, ii,loss1,loss2)
                 print(message)

    #             rel_im=toimage(torch.irfft(real_data.squeeze(0).permute(1, 2, 3, 0), 2, onesided=False)[32, :, :].unsqueeze(0))
    #             img_test.append(rel_im)
    #
    #
    #             for it in range(1, 3):
    #                 plt.subplot(1, 2, it)
    #                 plt.imshow(img_test[it - 1])
    #             plt.text(-1, 81, 'D accuracy=%.2f ' % (D_cost.mean()),
    #                      fontdict={'size': 15})
    #             plt.text(-1, 85, 'G accuracy=%.2f ' % (G_cost),
    #                      fontdict={'size': 15})
    #             plt.text(-1, 89, 'W accuracy=%.2f ' % (Wasserstein_D),
    #                      fontdict={'size': 15})
    #             plt.text(-1, 95,  'epoch=%.2f ' % (epoch),
    #                      fontdict={'size': 15})
    #             plt.show()
    #            # plt.ylim(-2, 2)
    #             plt.draw(), plt.pause(0.1),plt.clf()
    #     save_filename = 'net_d%s.pth' % epoch
    #     save_path = os.path.join('./check/', save_filename)
    #     torch.save(net_d.cpu().state_dict(), save_path)
    #     net_d.cuda(0)
        save_filename = 'net_g%s.pth' % epoch
        save_path = os.path.join('./check/', save_filename)
        torch.save(net_g.cpu().state_dict(), save_path)
        net_g.cuda(0)
    #
    # plt.ioff()
    # plt.show()

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