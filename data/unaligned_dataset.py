import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
from torchvision.transforms import ToTensor
import torch
import scipy.io as scio

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=False)
        self.B=torch.Tensor(torch.randn(output_nc,output_nc,output_nc))
        i=0

        # for filenname in os.listdir(r"./dataset/trainB/"):
        #     i = i + 1
        #     #image = Image.open("./dataset/trainB/" + filenname).convert('RGB')
        #     image = scio.loadmat("./dataset/trainB/" + filenname)
        #     B_temp = self.transform_B(image)
        #     if i == 1:
        #         self.B = B_temp
        #     else:
        #         self.B = torch.cat((self.B, B_temp), 0)


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path=self.B_paths[index%self.B_size]
        B_rotate_ang = int(os.path.split(B_path)[1].split('_')[1].split('.')[0])
        B_path = os.path.split(B_path)[0] + '\\' + os.path.split(B_path)[1].split('_')[0] + '_1.mat'
        #B_path = self.B_paths[0]
        self.A =torch.from_numpy(scio.loadmat(A_path)['data']).unsqueeze(0).float()
        self.B = torch.from_numpy(scio.loadmat(B_path)['data']).float()
        # # if self.opt.serial_batches:   # make sure index is within then range
        # #     index_B = index % self.B_size
        # # else:   # randomize the index for domain B to avoid fixed pairs.
        # #     index_B = random.randint(0, self.B_size - 1)
        # # B_path = self.B_paths[index_B]
        #
        # A_img = Image.open(A_path).convert('RGB')
        # # B_img = Image.open(B_path).convert('L')
        # # apply image transformation
        # A = self.transform_A(A_img)
        # # B = self.transform_B(B_img)
        #A=(self.A/self.A.max()-0.5)/0.5
        #B=(self.B/self.B.max()-0.5)/0.5
        #A=self.A
        #A=(self.A-self.A.min())/(self.A.max()-self.A.min())
        #self.B[-1,:,:]=(self.B[-1,:,:]-self.B[-1,:,:].min())/(self.B[-1,:,:].max()-self.B[-1,:,:].min())

        # A = torch.rfft(self.A,2,onesided=False).permute(3,0,1,2).squeeze(0)
        # A = torch.roll(torch.roll(A, 128, 2), 128, 3)
        # B = torch.rfft(self.B,2,onesided=False).permute(3,0,1,2)
        # B = torch.roll(torch.roll(B, 32, 2), 32, 3)

        A = self.A
        B = self.B

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_rotate_ang': B_rotate_ang}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
