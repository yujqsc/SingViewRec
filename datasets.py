from torch.utils.data import Dataset
import  os
import cv2
import torchvision.transforms as transforms
tran = transforms.ToTensor()


class DataImport(Dataset):
    def __init__(self,img_size,train=True):
        self.train=train
        self.imgs=[]
        self.img_size=img_size
        if train:
            dr_path='./dataset/train/'
        else:
            dr_path='./dataset/test/'

        for img_num,im in enumerate(os.listdir(dr_path)):
            str=im.split('.png')
            index = int(str[0])
            im = os.path.join(dr_path, im)
            self.imgs.append((im, index))

    def __getitem__(self, index):
        (img_path, no) = self.imgs[index]
        data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #data = cv2.resize(data, (self.img_size, self.img_size))
        data = tran(data)

        return data, no

    def __len__(self):
        return len(self.imgs)