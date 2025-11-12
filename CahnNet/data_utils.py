import json
import os
import random
import glob
import torch.utils.data as data
import torchvision.transforms as tfs
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as FF

from option import opt


class real_Dataset(data.Dataset):
    def __init__(self, path, train=True, origin=False, size='no', format='.png'):

        super(real_Dataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.origin = origin

        dir = 'dataset'
        if origin:
            dir = 'origin'
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy', dir))
        self.haze_imgs = [os.path.join(path, 'hazy', dir, img) for img in self.haze_imgs_dir]

        self.clear_dir = os.path.join(path, 'clear', dir)

    def __getitem__(self, index):

        haze = Image.open(self.haze_imgs[index])

        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 20000)
                haze = Image.open(self.haze_imgs[index])

        img = self.haze_imgs[index]
        haze_name = os.path.split(img)[1]

        if self.origin:
            id = haze_name.split('.')[0].split('_')[0]
            clear_name = id + '_GT' + self.format
        else:
            id = haze_name.split('.')[0]
            clear_name = id + self.format


        clear = Image.open(os.path.join(self.clear_dir, clear_name))

        clear = tfs.CenterCrop(haze.size[::-1])(clear)

        if not isinstance(self.size, str):

            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))

            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)

        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))

        return haze, clear, clear_name, haze_name

    def augData(self, data, target):

        if self.train:

            rand_hor = random.randint(0, 1)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)

            rand_rot = random.randint(0, 3)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)

        data = tfs.ToTensor()(data)

        if opt.norm:
            data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)

        return data, target

    def __len__(self):

        return len(self.haze_imgs)


class RESIDE_Dataset(data.Dataset):
    def __init__(self, path, train, size='no', format='.png'):
        super(RESIDE_Dataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        if train:
            self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
            self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
            self.clear_dir = os.path.join(path, 'clear')
        else:
            self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
            self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
            self.clear_dir = os.path.join(path, 'gt')

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 20000)
                haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        haze_name = os.path.split(img)[1]
        id = haze_name.split('.')[0].split('_')[0]
        clear_name = id + self.format
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        clear = tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear, clear_name, haze_name

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        data = tfs.ToTensor()(data)
        if opt.norm:
            data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        return data, target

    def __len__(self):
        return len(self.haze_imgs)


class outRESIDE_Dataset(data.Dataset):
    def __init__(self, path, train, size='no', format='.png'):
        super(outRESIDE_Dataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        if train:
            self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy/hazy_images'))
            self.haze_imgs = [os.path.join(path, 'hazy/hazy_images', img) for img in self.haze_imgs_dir]
            self.clear_dir = os.path.join(path, 'clear/clear_images')
        else:
            self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
            self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
            self.clear_dir = os.path.join(path, 'gt2')

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 20000)
                haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        haze_name = os.path.split(img)[1]
        id = haze_name.split('.')[0].split('_')[0]
        clear_name = id + '.png'
        if self.train:
            clear_name = id + self.format
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        clear = tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear, clear_name

    def augData(self, data, target):
        if self.train:
            if random.random() < 0.5:
                data = tfs.RandomHorizontalFlip(p=1.0)(data)
                target = tfs.RandomHorizontalFlip(p=1.0)(target)

        data = tfs.ToTensor()(data)
        if opt.norm:
            data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        return data, target

    def __len__(self):
        return len(self.haze_imgs)


"""
O_Haze .jpg
D_Haze .png
I_Haze .jpg
NH_Haze .png
"""

if __name__ == '__main__':
    train_loader = DataLoader(
        dataset=real_Dataset(os.path.join('/media/StudentGroup/LZ/Dataset/', 'Dense_Haze/test'), train=True,
                             origin=False,
                             size='no',
                             format='.png'),
        batch_size=8, shuffle=True, num_workers=24)
    in_train_lodar = DataLoader(
        dataset=RESIDE_Dataset(os.path.join('/media/StudentGroup/LZ/Dataset/', 'ITS'), train=True,
                               size=128,
                               format='.png'),
        batch_size=8, shuffle=True, num_workers=24)
    in_test_lodar = DataLoader(
        dataset=RESIDE_Dataset(os.path.join('/media/StudentGroup/LZ/Dataset/', 'ITS/test/indoor'), train=False,
                               size=128,
                               format='.png'),
        batch_size=8, shuffle=True, num_workers=24)

    print("测试训练数据集...")
    for i, (haze, clear, clear_name) in enumerate(train_loader):
        print(f"批次 {i + 1}:")
        print(f"清晰图像名称: {clear_name}")
        print(f"雾图形状: {haze.shape}")
        print(f"清晰图像形状: {clear.shape}")
        if i == 2:
            break
