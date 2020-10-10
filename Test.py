import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
from models import networks
from PIL import Image
import numpy as np

save_dir='C:\MyPJ\ProjFourier128gANCycle\Fourier128Wgan\check'
load_net=True
epoch_start=4
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
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model=networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'CTnet',
                      opt.norm, False, opt.init_type, opt.init_gain, opt.gpu_ids)

    if load_net:
        # save_filename = 'net_d%s.pth' % epoch_start
        # save_path = os.path.join('./check/', save_filename)
        # load_network(net_d, save_path)
        save_filename = 'net_g%s.pth' % epoch_start
        save_path = os.path.join('./check/', save_filename)
        load_network(model, save_path)

    # model = create_model(opt)      # create a model given opt.model and other options
    # model.setup(opt)               # regular setup: load and print networks; create schedulers
    #
    # load_filename = 'netd_%d.pth' % (epoch)
    # load_path = os.path.join(save_dir, load_filename)
    # print('loading the model from %s' % load_path)
    # # if you are using PyTorch newer than 0.4 (e.g., built from
    # # GitHub source), you can remove str() on self.device
    # state_dict = torch.load(load_path, map_location=str(opt.gpu_ids[0]))
    # if hasattr(state_dict, '_metadata'):
    #     del state_dict._metadata
    # net = getattr(model, 'net')
    # # patch InstanceNorm checkpoints prior to 0.4
    # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
    #     __patch_instance_norm_state_dict(state_dict, net, key.split('.'))
    # net.load_state_dict(state_dict)

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        g_noises = data['A'].cuda()
        g_noises = g_noises.squeeze(0)
        real_data = data['B'].cuda()
        real_data = real_data.squeeze(0)
        fake_date = model(g_noises)
        fake_im = toimage(
            torch.irfft(torch.roll(torch.roll(fake_date, -32, 2), -32, 3).permute(1, 2, 3, 0), 2, onesided=False)[32, :,
            :].unsqueeze(0))
        # img_test.append(rel_im)
        save_image(fake_im, os.path.join('./check/test/', 'fake%s.bmp' % i))
        rel_im = toimage(
            torch.irfft(torch.roll(torch.roll(real_data, -32, 2), -32, 3).permute(1, 2, 3, 0), 2, onesided=False)[32, :,
            :].unsqueeze(0))
        # img_test.append(rel_im)
        save_image(rel_im, os.path.join('./check/test/', 'real%s.bmp' % i))



