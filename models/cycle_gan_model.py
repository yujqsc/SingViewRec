import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import torch.nn as nn
from torchsummary import summary
import torch.autograd as autograd
from PIL import Image
import os.path
import scipy.io as scio
from utils.RegisterCT import RegisterCT
from torch.autograd import Variable
def project3D(ct):
    dr = torch.sum(ct, dim=1, keepdim=True)
    return dr
class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=30, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=30, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt=opt
        self.No=0
        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1
        self.one = self.one.cuda()
        self.mone =self. mone.cuda()
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        #self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B','D_real','D_fake']
        self.loss_names = [ 'D_A','G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B','gradient_penalty','D_real','D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        #visual_names_A = ['real_A', 'fake_B', 'rec_A','f_b_filter','r_b_filter','realB_filter']
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        # if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        #     visual_names_A.append('idt_B')
        #     visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['D_A', 'D_B','G_A', 'G_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
        self.ctargs = RegisterCT().param()
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'CTnet', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.Filter=networks.Filter()
        self.Trans = networks.Transform()
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            #self.netD_A = networks.Discriminator(opt.output_nc)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, 'ProjNet',
                                             opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            #self.netD_B = networks.Discriminator(opt.input_nc)

        if self.isTrain:
            # if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
            #     assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(1)  # create image buffer to store previously generated images
            # define loss functions
            #self.criterionGANL1 = torch.nn.L1Loss()
            self.criterionGAN = networks.GANLoss('lsgan').to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionCondition=torch.nn.L1Loss()
            #self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=0.0001,betas=[0.5,0.9])
            #self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=0.0001,betas=[0.5,0.9])
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    # def netG_B(self,ct):
    #     dr = torch.sum(ct, dim=2, keepdim=True)
    #     dr =dr.permute(0,2,1,3)
    #     return dr/128
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        #self.real_A =self.real_A.squeeze(0)
        self.real_B =self.real_B.unsqueeze(1)
        #self.real_A = self.real_A.view(-1,1,257,257)
        self.real_A.requires_grad=True
        #self.real_B = self.real_B.view(-1,65,65,65)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.ang = input['B_rotate_ang']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        #self.modelsize(self.netG_A,self.real_A)
        self.fake_B = self.netG_A(self.real_A)
        #self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.fake_A  =self.real_A
        #self.fake_A =self.real_A
        self.rec_B =self.real_B
        # self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        #self.fake_A =self.real_A
        self.rec_A= self.netG_B(self.Trans(self.fake_B, self.ang),self.ctargs)  # G_B(B)
        # self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        #self.f_b_filter = self.Filter(torch.irfft(self.fake_B.permute(1,2,3,0), 2,onesided=False).unsqueeze(0))
        #self.r_b_filter= self.Filter(torch.irfft(self.rec_B.permute(1,2,3,0), 2,onesided=False).unsqueeze(0))
        #self.realB_filter= self.Filter(torch.irfft(self.real_B.squeeze(0).permute(1, 2, 3, 0), 2, onesided=False).unsqueeze(0))
    def modelsize(model1, input1, type_size=4):
        model=model1.netG_A
        input=model1.real_A
        # para = sum([np.prod(list(p.size())) for p in model.parameters()])
        # print('Model  : params: {:4f}M'.format(para * type_size / 1000 / 1000))

        input_ = input.clone()
        input_.requires_grad_(requires_grad=False)

        mods = list(model.modules())
        out_sizes = []

        for i in range(1, len(mods)):
            m = mods[i]
            if isinstance(m, nn.ReLU):
                if m.inplace:
                    continue
            out = m(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out

        total_nums = 0
        for i in range(len(out_sizes)):
            s = out_sizes[i]
            nums = np.prod(np.array(s))
            total_nums += nums

        # print('Model {} : intermedite variables: {:3f} M (without backward)'
        #       .format(model._get_name(), total_nums * type_size / 1000 / 1000))
        # print('Model {} : intermedite variables: {:3f} M (with backward)'
        #       .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        if self.opt.gan_mode != 'wggp':
            # Real
            pred_real = netD(real)
            self.loss_D_real = self.criterionGAN(pred_real, True)
            # Fake
            pred_fake = netD(fake.detach())
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            # Combined loss and calculate gradients
            loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
            loss_D.backward()
            return loss_D
        else:
            # Real
            self.loss_D_real = netD(real)
            self.loss_D_real = self.loss_D_real.mean().unsqueeze(0)
            #self.loss_D_real.backward(self.mone,retain_graph=True)
            # Fake
            self.loss_D_fake = netD(fake.detach())
            self.loss_D_fake = self.loss_D_fake.mean().unsqueeze(0)
            #self.loss_D_fake.backward(self.one,retain_graph=True)
            # gp weight
            #gradient_penalty,gradient= networks.cal_gradient_penalty(netD, real, fake.detach(), torch.device('cuda'))


            #self.loss_D_real.backward()
            #self.loss_D_fake = pred_fake.mean()
            #self.loss_D_fake.backward()
           # self.gradient_penalty = self.calc_gradient_penaltynew(netD, real.data, fake.data)
            self.loss_gradient_penalty =self._gradient_penalty(netD,real,fake,10)
           # self.loss_gradient_penalty =self.calculate_gradient_penalty(netD,real,fake)
            #self.loss_gradient_penalty.backward()
            loss_D= self.loss_D_fake-self.loss_D_real+self.loss_gradient_penalty
            loss_D.backward()
            #loss_D = self.loss_D_fake + self.loss_D_real
            #loss_D.backward(retain_graph=True)
            Wasserstein_D = self.loss_D_real - self.loss_D_fake
            return loss_D



    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        #fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B =self.fake_B
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        #fake_A = self.fake_A_pool.query(self.fake_A)
        fake_A =self.fake_A
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        #self.loss_D_B =0
    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        #lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        if self.opt.gan_mode != 'wggp':
            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
            #self.loss_G_A = self.criterionGAN(self.fake_B,self.real_B.squeeze(0))
            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
            #self.loss_G_B = self.criterionGAN(self.fake_A,self.real_A)
            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B.squeeze(0)) * lambda_B
            # combined loss and calculate gradients
            self.loss_condition1= self.criterionCondition(self.r_b_filter,self.Filter(torch.irfft(self.real_B.squeeze(0).permute(1,2,3,0), 2,onesided=False).unsqueeze(0)))*20
            self.loss_condition2 = self.criterionCondition(self.f_b_filter, self.Filter(torch.irfft(self.real_B.squeeze(0).permute(1, 2, 3, 0), 2, onesided=False).unsqueeze(0)))*0.1
            self.loss_G = self.loss_G_A
            self.loss_G.backward()
            #print(self.loss_condition1)
            # for name, parms in self.netG_A.named_parameters():
            #     if self.opt.current_iter%10==0:
            #         if name=='module.model.2.weight':
            #             print('层:',name,parms.size(),'-->name:', name, '-->grad_requirs:', parms.requires_grad, \
            #               ' -->grad_value:', parms.grad)
            # a = 1
        else:
            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.netD_A(self.fake_B)
            self.loss_G_A =self.loss_G_A.mean().unsqueeze(0)
            self.loss_G_A.backward(self.mone,retain_graph=True)
            # GAN loss D_B(G_B(B))
            #self.loss_G_B = self.netD_B(self.fake_A)
            #self.loss_G_B = self.loss_G_B .mean().unsqueeze(0)
            #self.loss_G_B.backward(self.mone)
            self.loss_G_B =0
            #self.loss_G_B =0;
            #self.loss_G_B.backward(retain_graph=True)
            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
            #self.loss_cycle_A =0;
            self.loss_cycle_A.backward()
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B.squeeze(0)) * lambda_B
            #self.loss_cycle_B =0
            #self.loss_cycle_B.backward()
            # combined loss and calculate gradients
            self.loss_G_A=-self.loss_G_A
            #self.loss_G_B = -self.loss_G_B
            #self.loss_G = self.loss_G_A+self.loss_G_B+self.loss_cycle_A +self.loss_cycle_B
            #self.loss_G =-self.loss_G_A
           #self.loss_G.backward()

    def calc_gradient_penaltynew(self,net_D, real_data, fake_data):
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
    def _gradient_penalty(self,net_D, data, generated_data, gamma=10):
        batch_size = data.size(0)
        if data.dim()==5:
            epsilon = torch.rand(batch_size, 1, 1,1, 1)
        else:
            epsilon = torch.rand(batch_size, 1, 1, 1)

        epsilon = epsilon.expand_as(data)


        #if self.use_cuda:
        epsilon = epsilon.cuda()

        interpolation = epsilon * data.data + (1 - epsilon) * generated_data.data
        interpolation = Variable(interpolation, requires_grad=True)

        #if self.use_cuda:
        interpolation = interpolation.cuda()

        interpolation_logits = net_D(interpolation)
        grad_outputs = torch.ones(interpolation_logits.size())

        #if self.use_cuda:
        grad_outputs = grad_outputs.cuda()

        gradients = autograd.grad(outputs=interpolation_logits,
                                  inputs=interpolation,
                                  grad_outputs=grad_outputs,
                                  create_graph=True,
                                  retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return gamma* ((gradients_norm - 1) ** 2).mean()

    def calculate_gradient_penalty(self,net_D,real_data, fake_data, k=2, p=6):
        #real_grad_outputs = torch.full((real_data.size(0), 1), 1, requires_grad=True)
        #fake_grad_outputs = torch.full((fake_data.size(0), 1), 1, requires_grad=True)
        fake_outputs = net_D(fake_data)
        real_outputs = net_D(real_data)
        real_grad_outputs = torch.ones(fake_outputs.size(), requires_grad=True).cuda()
        fake_grad_outputs = torch.ones(real_outputs.size(), requires_grad=True).cuda()
        real_data.requires_grad = True
        #real_grad_outputs = real_grad_outputs.cuda(non_blocking=True)
        #fake_grad_outputs = fake_grad_outputs.cuda(non_blocking=True)

        real_gradient = autograd.grad(
            outputs=real_outputs,
            inputs=real_data,
            grad_outputs=real_grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        fake_gradient = autograd.grad(
            outputs=fake_outputs,
            inputs=fake_data,
            grad_outputs=fake_grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        real_gradient_norm = real_gradient.view(real_gradient.size(0), -1).pow(2).sum(1) ** (p / 2)
        fake_gradient_norm = fake_gradient.view(fake_gradient.size(0), -1).pow(2).sum(1) ** (p / 2)

        gradient_penalty = torch.mean(real_gradient_norm + fake_gradient_norm) * k / 2
        return gradient_penalty
    def divpenalty(self,real_imgs,real_validity,fake_imgs,fake_validity):
        k=2
        p=6
        real_imgs.requires_grad=True
        Tensor = torch.cuda.FloatTensor
        #real_validity=net_D(real_imgs)
        # Compute W-div gradient penalty
        #real_grad_out = Variable(Tensor(real_imgs.size(0), 1).fill_(1.0), requires_grad=False)
        real_grad_out =torch.ones(real_validity.size(), requires_grad=False)
        real_grad_out=real_grad_out.cuda()
        real_grad = autograd.grad(
            real_validity, real_imgs, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

        fake_grad_out = Variable(Tensor(fake_imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake_grad = autograd.grad(
            fake_validity, fake_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

        return torch.mean(real_grad_norm + fake_grad_norm) * k / 2


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # D_A and D_B
        #self.set_requires_grad([self.netD_A, self.netD_B], True)
        if (self.opt.current_iter) % (self.opt.critic_iter*self.opt.batch_size) != 0:
            for p in self.netD_A.parameters():  # reset requires_grad
                p.requires_grad = True
            self.netD_A.zero_grad()
            #self.optimizer_G.zero_grad()
            #self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
            self.backward_D_A()      # calculate gradients for D_A
            #self.loss_D_B=0
            #self.backward_D_B()      # calculate graidents for D_B
            self.loss_D_B = 0
            self.optimizer_D.step()  # update D_A and D_B's weights
        else:
            # G_A and G_B
            for p in self.netD_A.parameters():
                p.requires_grad = False  # to avoid computation
           # self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
            #self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            self.netG_A.zero_grad()
            self.backward_G()             # calculate gradients for G_A and G_B
            self.optimizer_G.step()       # update G_A and G_B's weights
            if(self.opt.current_iter%100==0):
                self.test(self.netG_A,self.opt.epochnow)
    def print_networkStructure(self,opt):
        #summary(self.netD_A,(2,64,64,64))
        summary(self.netD_B, (1,256, 256))



    def toimage(self,image_tensor):
        image_numpy = image_tensor[0].data.cpu().float().numpy()  # convert it into a numpy array
        image_numpy = (image_numpy - image_numpy.min()) / (
                image_numpy.max() - image_numpy.min()) * 255.0
        return image_numpy.astype(np.uint8)

    def save_image(self,image_numpy, image_path, aspect_ratio=1.0):
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

    def test(self,net,count):
        pathA = os.path.join('C:/MyPJ/ProjFourier128gANCycle/Fourier128Wgan/dataset/testA/SVCTdattest12_34.mat')
        # pathB = os.path.join('C:/MyPJ/ProjFourier128gANCycle/Fourier128Wgan/dataset/testB/SVCTdattest32_34.mat')
        A = torch.from_numpy(scio.loadmat(pathA)['data']).unsqueeze(0).float()
        # A = torch.rfft(A, 2, onesided=False).permute(3, 0, 1, 2).squeeze(0)
        # A = torch.roll(torch.roll(A, 128, 2), 128, 3)
        # B = torch.from_numpy(scio.loadmat(pathB)['data']).float()
        A=A.view(-1,1,257,257)
        fake_date = net(A)
        #fk_im = self.toimage(
            #torch.irfft(torch.roll(torch.roll(fake_date, -32, 2), -32, 3).permute(1, 2, 3, 0), 2, onesided=False)[32, :,:].unsqueeze(0))
        fk_im=self.toimage(fake_date[0,0,32, :, :].unsqueeze(0))
        # img_test.append(fk_im)
        if self.opt.current_iter%1000==0:
            self.No=self.No+1
        save_filenamet = 'fake%s_%s.bmp' % (count,self.No)
        img_path = os.path.join('C:/MyPJ/ProjFourier128gANCycle/Fourier128Wgan/checkpoints/experiment_name/web/', save_filenamet)
        self.save_image(fk_im, img_path)
        del fake_date
        del fk_im
        del A