import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import math
from Modules.Projection3D2D import projection3d2d


###############################################################################
# Helper Functions
###############################################################################

class Transform(nn.Module):
    def __init__(self):
        super(Transform, self).__init__()

    def transform(self, input, rotang):
        angle = rotang * math.pi / 180
        theta = torch.tensor([
            [math.cos(angle), math.sin(-angle), 0],
            [math.sin(angle), math.cos(angle), 0]
        ], dtype=torch.float)
        theta = theta.unsqueeze(0).expand(input.size(0), -1, -1).cuda()
        data = input.expand(input.size(0), -1, -1, -1)
        grid = F.affine_grid(theta, data.size(), align_corners=False)
        return F.grid_sample(input, grid, mode='nearest', align_corners=False)

    def forward(self, input,ang):
        return self.transform(input.squeeze(0),ang).unsqueeze(0)

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
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


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        #net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=1)
        #net = DRUnetGenerator(input_nc, output_nc, 3, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        #net = DRGen(input_nc, output_nc, norm_layer=norm_layer)
        net=projection3d2d()
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG =='CTnet':
        #net=CTGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
        net = CTUnetGenerator(input_nc, output_nc, 3, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_G_A(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        #net = NLayerDiscriminator(input_nc, ndf, n_layers=5, norm_layer=norm_layer)
        net = Dsc3D()
        #net = NLayerDiscriminatorG(input_nc, ndf, n_layers=5, norm_layer=norm_layer)
        #net = Discriminator(input_nc)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'ProjNet':
        net = NLayerDiscriminatorPj(input_nc, ndf, n_layers=6, norm_layer=norm_layer)
        #net =PixelDiscriminatorPj(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.BCELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model=[IFFT()]
        model += [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(2):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        out=int(ngf * mult / 2)
        in_num=int(ngf * mult / 2)
        for i in range(2):
            model += [nn.ConvTranspose2d(in_num, out,
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=use_bias),
                  norm_layer(int(out)),
                  nn.ReLU(True)]
            in_num=out
            out=int(out / 2)
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(out*2, output_nc, kernel_size=7, padding=0)]
        #model += [nn.Tanh()]
        #model+=[nn.ReLU(True)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input.squeeze(0))
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

class Filter(nn.Module):
    def __init__(self):
        super(Filter,self).__init__()
        kernel=[[1,1,1],[1,-8,1],[1,1,1]]
        kernel=torch.FloatTensor(kernel).cuda()
        kernel=kernel.expand(64,64,-1,-1)
        self.weight=nn.Parameter(data=kernel,requires_grad=False)
    def forward(self,x):
        return F.conv2d(x,weight=self.weight,padding=1)


class CTGenerator0(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(CTGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model=[FFT()]
        model+=[nn.ReflectionPad2d(3),
                  nn.Conv2d(1,ngf, kernel_size=3, padding=0, bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]

        model += [
                  nn.Conv2d(ngf, 2*ngf, kernel_size=3, stride=2,padding=0, bias=use_bias),
                  norm_layer(2*ngf),
                  nn.ReLU(True)]
        model += [
            nn.Conv2d(2*ngf, 2 * ngf, kernel_size=3, stride=2, padding=0, bias=use_bias),
            norm_layer(2*ngf),
            nn.ReLU(True)]
        model += [
            nn.Conv2d(2*ngf, 2*ngf, kernel_size=3, stride=2, padding=0, bias=use_bias),
            norm_layer(2*ngf),
            nn.ReLU(True)]
        model += [
            nn.Conv2d(2 * ngf, 2 * ngf, kernel_size=3, stride=2, padding=0, bias=use_bias),
            norm_layer(2*ngf),
            nn.ReLU(True)]
        model += [
            nn.Conv2d(2 * ngf, 2 * ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(2*ngf),
            nn.ReLU(True)]
        model += [
            nn.ConvTranspose2d(2 * ngf,  ngf, kernel_size=3, stride=2, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)]
        model += [
            nn.ConvTranspose2d(ngf, int(ngf/2), kernel_size=3, stride=2, padding=0, bias=use_bias),
            norm_layer(ngf/2),
            nn.ReLU(True)]
        model += [
            nn.ConvTranspose2d(int(ngf/2), output_nc, kernel_size=3, stride=2, padding=4, output_padding=1,bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(True)]
        model += [IFFT()]

        # model = [nn.ReflectionPad2d(3),
        #          nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
        #          norm_layer(ngf),
        #          nn.ReLU(True)]
        #
        # n_downsampling = 2
        # for i in range(n_downsampling):  # add downsampling layers
        #     mult = 2 ** i
        #     model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
        #               norm_layer(ngf * mult * 2),
        #               nn.ReLU(True)]
        #
        # mult = 2 ** n_downsampling
        # for i in range(n_blocks):       # add ResNet blocks
        #
        #     model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        #
        # for i in range(n_downsampling):  # add upsampling layers
        #     mult = 2 ** (n_downsampling - i)
        #     model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
        #                                  kernel_size=3, stride=2,
        #                                  padding=1, output_padding=1,
        #                                  bias=use_bias),
        #               norm_layer(int(ngf * mult / 2)),
        #               nn.ReLU(True)]
        # model += [nn.ReflectionPad2d(3)]
        # model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # #model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class CTGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(CTGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # norm_layer=Normalize
        model = [FFT()]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(1, ngf, kernel_size=3, padding=0, bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]

        model += [
            nn.Conv2d(ngf, 2 * ngf, kernel_size=3, stride=2, padding=0, bias=use_bias),
            norm_layer(2 * ngf),
            nn.ReLU(True)]
        model += [
            nn.Conv2d(2 * ngf, 2 * ngf, kernel_size=3, stride=2, padding=0, bias=use_bias),
            norm_layer(2 * ngf),
            nn.ReLU(True)]
        model += [
            nn.Conv2d(2 * ngf, 2 * ngf, kernel_size=3, stride=2, padding=0, bias=use_bias),
            norm_layer(2 * ngf),
            nn.ReLU(True)]
        model += [
            nn.Conv2d(2 * ngf, 2 * ngf, kernel_size=3, stride=2, padding=0, bias=use_bias),
            norm_layer(2 * ngf),
            nn.ReLU(True)]
        model += [
            nn.Conv2d(2 * ngf, 2 * ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(2 * ngf),
            nn.ReLU(True)]
        model += [
            nn.ConvTranspose2d(2 * ngf, 2 *ngf, kernel_size=7, stride=2, padding=0, bias=use_bias),
            norm_layer(2 *ngf),
            nn.ReLU(True)]
        model += [
            nn.ConvTranspose2d(2 * ngf,  ngf, kernel_size=7, stride=1, padding=0, bias=use_bias),
            norm_layer(2 * ngf),
            nn.ReLU(True)]
        model += [
            nn.ConvTranspose2d(ngf, ngf, kernel_size=5, stride=2, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)]
        model += [
            nn.ConvTranspose2d(ngf, int(ngf / 2), kernel_size=5, stride=1, padding=0, bias=use_bias),
            norm_layer(int(ngf / 2)),
            nn.ReLU(True)]
        model += [
            nn.ConvTranspose2d(int(ngf / 2), int(ngf / 2), kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(int(ngf / 2)),
            nn.ReLU(True)]
        model += [
            nn.ConvTranspose2d(int(ngf / 2), output_nc, kernel_size=4,  padding=1,
                               bias=use_bias),
        ]
        # model += [IFFT()]

        # model = [nn.ReflectionPad2d(3),
        #          nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
        #          norm_layer(ngf),
        #          nn.ReLU(True)]
        #
        # n_downsampling = 2
        # for i in range(n_downsampling):  # add downsampling layers
        #     mult = 2 ** i
        #     model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
        #               norm_layer(ngf * mult * 2),
        #               nn.ReLU(True)]
        #
        # mult = 2 ** n_downsampling
        # for i in range(n_blocks):       # add ResNet blocks
        #
        #     model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        #
        # for i in range(n_downsampling):  # add upsampling layers
        #     mult = 2 ** (n_downsampling - i)
        #     model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
        #                                  kernel_size=3, stride=2,
        #                                  padding=1, output_padding=1,
        #                                  bias=use_bias),
        #               norm_layer(int(ngf * mult / 2)),
        #               nn.ReLU(True)]
        # model += [nn.ReflectionPad2d(3)]
        # model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # #model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 1
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        # conv_block+=[norm_layer(dim),nn.ReLU(True),nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),norm_layer(dim),nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 1
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        # conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]

        p = 1
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        # conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
class ResnetBlock2Conv(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock2Conv, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        # conv_block+=[norm_layer(dim),nn.ReLU(True),nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),norm_layer(dim),nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        # conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
class DRGen(nn.Module):
    def __init__(self,input_nc,output_nc,norm_layer=nn.InstanceNorm2d):
        super(DRGen, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.model = []
        channel_in = input_nc
        channel_out =input_nc

        for n in range(1, 3):
            channel_in =channel_out
            channel_out = 2 * channel_in - 1
            resnet = [ResnetBlock2Conv(channel_in, padding_type='zero', norm_layer=norm_layer, use_dropout=False,
                                       use_bias=use_bias)]
            up = [nn.ConvTranspose2d(channel_in, channel_out, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(channel_out), nn.ReLU(True)]
            self.model += resnet + up
        channel_in = channel_out
        channel_out = int((channel_in-1)/2)
        self.model += [ResnetBlock2Conv(channel_in, padding_type='zero', norm_layer=norm_layer, use_dropout=False,
                                       use_bias=use_bias)] + [nn.Conv2d(channel_in,channel_out,kernel_size=3, padding=1, bias=use_bias)]
        for n in range(1, 8):
            channel_in = channel_out
            channel_out = int(channel_in/2)
            resnet = [ResnetBlock2Conv(channel_in, padding_type='zero', norm_layer=norm_layer, use_dropout=False,
                                       use_bias=use_bias)]
            down=[nn.Conv2d(channel_in,channel_out,kernel_size=3, padding=1, bias=use_bias)]
            self.model += resnet + down
        #self.model += [FFT()]
        self.model = nn.Sequential(*self.model)


    def forward(self, input):
        """Standard forward"""
        return self.model(input.squeeze(1))
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c, bn_momentum):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                      bias=False),
                            nn.BatchNorm2d(c * (2 ** j), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused
class HRNet(nn.Module):
    def __init__(self, c=48, nof_joints=17, bn_momentum=0.1):
        super(HRNet, self).__init__()

        # Input (stem net)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.InstanceNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False)
        self.bn2 = nn.InstanceNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1 (layer1)      - First group of bottleneck (resnet) modules
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(256, c * (2 ** 1), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 1), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 2 (stage2)      - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 2 (transition2)      - Creation of the third branch (1/4 resolution)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(c * (2 ** 1), c * (2 ** 2), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 2), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 3 (stage3)      - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 3 (transition3)      - Creation of the fourth branch (1/8 resolution)
        self.transition3 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(c * (2 ** 2), c * (2 ** 3), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 3), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 4 (stage4)      - Fourth module with 3 groups of bottleneck (resnet) modules. This has 4 branches
        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=1, c=c, bn_momentum=bn_momentum),
        )

        # Final layer (final_layer)
        self.final_layer = nn.Conv2d(c, nof_joints, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)

        x = self.stage2(x)
        # x = [trans(x[-1]) for trans in self.transition2]    # New branch derives from the "upper" branch only
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        # x = [trans(x) for trans in self.transition3]    # New branch derives from the "upper" branch only
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)

        x = self.final_layer(x[0])

        return x


class CTUnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(CTUnetGenerator, self).__init__()
        #self. model = [FFT()]
        # ---->in 256 channel: 1-> 32 out 256
        self.model = [
                  nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d),
                  norm_layer(32),
                  nn.ReLU(True)]
        # ---->in 256 channel 32->64 out 256
        self.model += [
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d),
            norm_layer(64),
            nn.ReLU(True)]
        # ---->in 256 channel 64->64 out 128
        self.model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=nn.InstanceNorm2d),
            norm_layer(128),
            nn.ReLU(True)]
        # ---->in 128  out 128 channel 128->256
        self.model += [
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d),
            norm_layer(256),
            nn.ReLU(True)]
        # ---->in 128  out 64 channel 256->256
        self.model += [
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=nn.InstanceNorm2d),
            norm_layer(256),
            nn.ReLU(True)]
        # construct unet structure
        unet_block = UnetSkipConnectionBlockX(256, 256, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        unet_block = UnetSkipConnectionBlockX(256, 256, input_nc=None, submodule=unet_block,
                                          norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlockX(256, 256, input_nc=None, submodule=unet_block,
                                              norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block= UnetSkipConnectionBlockX(256, 256, input_nc=256, submodule=unet_block, outermost=True,
                                               norm_layer=norm_layer)  # add the outermost layer
        self.model +=[unet_block]
        # ---->in 128  out 128 channel 128->256
        self.model += [
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d),
            norm_layer(128),
            nn.ReLU(True)]
        # ---->in 128  out 128 channel 128->256
        self.model += [
            nn.Conv2d(128, 65, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d),
           ]
        # for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
        #     unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # # gradually reduce the number of filters from ngf * 8 to ngf
        # unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # self.model += UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

        self.model = nn.Sequential(*self.model)
    def forward(self, input):
        """Standard forward"""
        return self.model(input).unsqueeze(1)

class CTUnetGenerator512(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(CTUnetGenerator512, self).__init__()
        self. model = [FFT()]
        # ---->in 256 channel: 1-> 32 out 256
        self.model += [
                  nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d),
                  norm_layer(32),
                  nn.ReLU(True)]
        # ---->in 256 channel 32->64 out 256
        self.model += [
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d),
            norm_layer(64),
            nn.ReLU(True)]
        # ---->in 256 channel 64->64 out 128
        self.model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=nn.InstanceNorm2d),
            norm_layer(128),
            nn.ReLU(True)]
        # ---->in 128  out 128 channel 128->256
        self.model += [
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d),
            norm_layer(256),
            nn.ReLU(True)]
        # ---->in 128  out 128 channel 256->512
        self.model += [
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d),
            norm_layer(512),
            nn.ReLU(True)]
        # ---->in 128  out 64 channel 256->256
        self.model += [
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=nn.InstanceNorm2d),
            norm_layer(512),
            nn.ReLU(True)]
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(512, 512, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        unet_block = UnetSkipConnectionBlock(512, 512, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block= UnetSkipConnectionBlock(512, 512, input_nc=512, submodule=unet_block, outermost=True,
                                              norm_layer=norm_layer)  # add the outermost layer
        self.model +=[unet_block]
        # ---->in 128  out 128 channel 512->256
        self.model += [
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d),
            norm_layer(256),
            nn.ReLU(True)]
        # ---->in 128  out 128 channel 256->128
        self.model += [
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d),
            norm_layer(128),
            nn.ReLU(True)
           ]
        # ---->in 128  out 128 channel 128->64
        self.model += [
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d),
        ]
        # for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
        #     unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # # gradually reduce the number of filters from ngf * 8 to ngf
        # unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # self.model += UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

        self.model = nn.Sequential(*self.model)
    def forward(self, input):
        """Standard forward"""
        return self.model(input)
class DRUnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(DRUnetGenerator, self).__init__()

        # ---->in 64 channel: 1-> 64 out 128
        self.model = [
                  nn.Conv2d(65, 128, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d),
                  norm_layer(128),
                  nn.ReLU(True)]
        # ---->in 64 channel 32->128 out 256
        self.model += [
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d),
            norm_layer(256),
            nn.ReLU(True)]

        # construct unet structure
        unet_block = UnetSkipConnectionBlockX(256, 256, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        unet_block = UnetSkipConnectionBlockX(256, 256, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block= UnetSkipConnectionBlockX(256, 256, input_nc=256, submodule=unet_block, outermost=True,
                                              norm_layer=norm_layer)  # add the outermost layer
        self.model +=[unet_block]

        # ---->in 64 channel 256->256 out 128
        self.model += [
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, bias=nn.InstanceNorm2d),
            norm_layer(256),
            nn.ReLU(True)]
        # ---->in 128  out 128 channel 256->128
        self.model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d),
            norm_layer(128),
            nn.ReLU(True)]
        # ---->in 128  out 256 channel 128->64
        self.model += [
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=nn.InstanceNorm2d),
            norm_layer(64),
            nn.ReLU(True)]
        # ---->in 256  out 256 channel 64->32
        self.model += [
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d),
            norm_layer(32),
            nn.ReLU(True)]
        # ---->in 128  out 128 channel 128->256
        self.model += [
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d),
           ]
        # for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
        #     unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # # gradually reduce the number of filters from ngf * 8 to ngf
        # unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # self.model += UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

        self.model = nn.Sequential(*self.model)
    def forward(self, input):
        """Standard forward"""
        return self.model(input)

# class ParaGenerator(nn.Module):
#     """Create a Unet-based generator"""
#
#     def __init__(self):
#
#
#     def forward(self, input):
#         """Standard forward"""
#         return torch.sum(input, dim=1, keepdim=True)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)


        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
class UnetSkipConnectionBlockX(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlockX, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=1, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        pool=nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        # pool =nn.Conv2d(input_nc, inner_nc, kernel_size=3,
        #                      stride=2, padding=1, bias=use_bias)
        resnet=[ResnetBlock(inner_nc, padding_type='zero', norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        if outermost:
            upconv = [nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1)]
            # down = [downconv]
            # up = [uprelu, upconv, nn.Tanh()]
            down = resnet + [pool]
            up = upconv
            model = down + [submodule] + up
        elif innermost:
            upconv =[nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1, bias=use_bias)]

            down = resnet+[pool]
            up = upconv+resnet
            model = down + up
        else:
            upconv = [nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1, bias=use_bias)]
            down =  resnet + [pool]
            up = upconv + resnet

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
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
class Dsc3D(torch.nn.Module):
    def __init__(self):
        super(Dsc3D, self).__init__()
        self.leak_value = 0.2
        self.bias = False
        self.cube_len =64

        padd = (0,0,0)
        if self.cube_len == 32:
            padd = (1,1,1)

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, self.cube_len, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.InstanceNorm3d(self.cube_len),
            torch.nn.LeakyReLU(self.leak_value),
            #nn.MaxPool3d(kernel_size=4, stride=2,padding=1)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len, self.cube_len*2, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.InstanceNorm3d(self.cube_len*2),
            torch.nn.LeakyReLU(self.leak_value),
            #nn.MaxPool3d(kernel_size=4, stride=2, padding=1)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*2, self.cube_len*4, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.InstanceNorm3d(self.cube_len*4),
            torch.nn.LeakyReLU(self.leak_value),
            #nn.MaxPool3d(kernel_size=4, stride=2, padding=1)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*4, self.cube_len*8, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.InstanceNorm3d(self.cube_len*8),
            torch.nn.LeakyReLU(self.leak_value),
            #nn.MaxPool3d(kernel_size=4, stride=2, padding=1)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*8, self.cube_len*16, kernel_size=4, stride=1, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.InstanceNorm3d(self.cube_len*16),
            torch.nn.LeakyReLU(self.leak_value),
            #nn.MaxPool3d(kernel_size=4, stride=2, padding=1)
        )
        # self.layer5 = torch.nn.Sequential(
        #     torch.nn.Conv3d(self.cube_len*8, 1, kernel_size=4, stride=2, bias=self.bias, padding=padd),
        #
        #     #torch.nn.Sigmoid()
        # )
        self.layer6 = torch.nn.Sequential(nn.Linear(64, 1))
        self.layer7 = torch.nn.Sequential( nn.Linear(512, 1))

    def forward(self, x):
        out = x
        #print(out.size()) # torch.Size([100, 1, 64, 64, 64])
        out = self.layer1(out)
        #print(out.size())  # torch.Size([100, 64, 32, 32, 32])
        out = self.layer2(out)
        #print(out.size())  # torch.Size([100, 128, 16, 16, 16])
        out = self.layer3(out)
        #print(out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.layer4(out)
        #out = self.layer5(out)
        out=out.view(-1,64)
        out=self.layer6(out)
        out=out.view(x.size(0),-1)
        out = self.layer7(out)
        #print(out.size())  # torch.Size([100, 512, 4, 4, 4])
        #out = self.layer5(out)
        #print(out.size())  # torch.Size([100, 200, 1, 1, 1])

        return out.view(x.size(0),-1)
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
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
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input.squeeze(0))
class NLayerDiscriminatorG(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminatorG, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 3
        padw = 1
        #sequence = [IFFT()]
        # in 1 out 256 shape in 64 out 64
        sequence = [nn.Conv2d(input_nc, 256, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True),norm_layer(256),nn.LeakyReLU(0.2, True)]
        # in 256 out 256 shape in 64 out 32
        sequence += [nn.Conv2d(256, 256, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True),norm_layer(256),nn.LeakyReLU(0.2, True)]
        # in 256 out 512 shape in 32 out 32
        sequence += [nn.Conv2d(256, 512, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True),norm_layer(512),nn.LeakyReLU(0.2, True)]
        # in 512 out 512 shape in 32 out 16
        sequence += [nn.Conv2d(512, 512, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True),norm_layer(512),nn.LeakyReLU(0.2, True)]
        # in 512 out 1024 shape in 32 out 16
        sequence += [nn.Conv2d(512, 1024, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True),norm_layer(1024),nn.LeakyReLU(0.2, True)]
        # in 512 out 1024 shape in 16 out 8
        sequence += [nn.Conv2d(1024, 1024, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True),norm_layer(1024),nn.LeakyReLU(0.2, True)]
        # in 512 out 1024 shape in 16 out 8
        sequence += [nn.Conv2d(1024, 1, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True)
                    ]
        # nf_mult = 1
        # nf_mult_prev = 1
        # for n in range(1, n_layers):  # gradually increase the number of filters
        #     nf_mult_prev = nf_mult
        #     nf_mult = min(2 ** n, 8)
        #     sequence += [
        #         nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
        #         norm_layer(ndf * nf_mult),
        #         nn.LeakyReLU(0.2, True)
        #     ]
        #
        # nf_mult_prev = nf_mult
        # nf_mult = min(2 ** n_layers, 8)
        # sequence += [
        #     nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
        #     norm_layer(ndf * nf_mult),
        #     nn.LeakyReLU(0.2, True)
        # ]
        #
        # sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input.squeeze(0))
class NLayerDiscriminatorPj(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminatorPj, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf* 2, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input.squeeze(0))

class PixelDiscriminatorPj(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminatorPj, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)