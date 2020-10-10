from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import os.path
import scipy.io as scio
class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.
    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        # input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=False)
        self.B = torch.Tensor(torch.randn(output_nc, output_nc, output_nc))

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        # A_path = self.A_paths[index]
        # A_img = Image.open(A_path).convert('RGB')
        # A = self.transform(A_img)
        # return {'A': A, 'A_paths': A_path}
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path=self.B_paths[index%self.B_size]
        #B_path = self.B_paths[0]
        self.A = torch.from_numpy(scio.loadmat(A_path)['data']).unsqueeze(0).float()
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
        # A=(self.A/self.A.max()-0.5)/0.5
        # B=(self.B/self.B.max()-0.5)/0.5
        A = torch.rfft(self.A, 2, onesided=False).permute(3, 0, 1, 2).squeeze(0)
        A = torch.roll(torch.roll(A, 128, 2), 128, 3)
        B = torch.rfft(self.B, 2, onesided=False).permute(3, 0, 1, 2)
        B = torch.roll(torch.roll(B, 32, 2), 32, 3)

        # A = self.A
        # B = self.B
        # A=(self.A-self.A.min())/(self.A.max()-self.A.min())
        # self.B[-1,:,:]=(self.B[-1,:,:]-self.B[-1,:,:].min())/(self.B[-1,:,:].max()-self.B[-1,:,:].min())
        # A = torch.rfft(self.A,2,onesided=False).permute(3,0,1,2).squeeze(0)
        # B = torch.rfft(self.B, 2, onesided=False).permute(3, 0, 1, 2)
        # B_rotate_ang = index * 1
        return {'A': A, 'B': B,'A_paths': A_path,'B_paths':B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)