import os
from torch.utils.data import Dataset, DataLoader
import csv
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable



class EmojiDataset(Dataset):
    '''
    Dataset of 1 million bitmoji images.
    start_idx - image number dataset should start at
    end_idx - data number where dataset ends
    '''
    def __init__(self, data_dir, start_idx=0, end_idx=1000000, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_len = end_idx - start_idx
    
    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        """
        img_name = os.path.join(self.data_dir, 'emoji_{}.png'.format(idx))
        img = Image.open(img_name)
        img = img.convert('RGB') # b/c it's a png

        if self.transform is not None:
            img = self.transform(img)
                                   
        return img

    def __len__(self):
        return self.data_len    

class CelebADataset(Dataset):
    '''
    CelebA face image dataset. This is the aligned and cropped version. 
    data_dir - directory of image data
    ann_dir - directory of annotation data
    split - either 'train', 'eval', or 'test'
    '''
    def __init__(self, data_dir, ann_dir, split, transform=None):
                
        data_splits = ['train', 'eval', 'test']
        self.data_dir = data_dir
        self.transform = transform
        
        split = data_splits.index(split)
        split_data = []
        with open(os.path.join(ann_dir, 'list_eval_partition.txt')) as split_file:
            reader = csv.reader(split_file, delimiter=' ')
            for row in reader:
                split_data.append(row)
        bbox_data = []
        with open(os.path.join(ann_dir, 'list_bbox_celeba.txt')) as bbox_file:
            reader = csv.reader(bbox_file, delimiter=' ', skipinitialspace=True)
            test_row = next(reader) # header row
            test_row = next(reader) # header row
            for row in reader:
                bbox_data.append(row)
                
        split_data = np.array(split_data)
        bbox_data = np.array(bbox_data)
        split_inds = np.where(split_data[:,1] == str(split))[0]
        
        self.split_info = split_data[split_inds, :]
        self.bbox_info = bbox_data[split_inds, :]
        self.data_len = self.split_info.shape[0]

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        """
        img_name = os.path.join(self.data_dir, self.split_info[idx, 0])
        img = Image.open(img_name)
        
        if self.transform is not None:
            img = self.transform(img)
                           
        return img

    def __len__(self):
        return self.data_len
    
class MSCeleb1MDataset(Dataset):
    '''
    MS-Celeb-1M face image dataset. This is the aligned and cropped version. 
    data_dir - directory of data. This directory should contain annotation files and a subdirectory for image data.
    split - either 'train' or 'test'
    '''
    def __init__(self, data_dir, split, transform=None):
        data_splits = ['train', 'test']
        self.transform = transform
        
        split = data_splits.index(split)
        if split == 0:
            info_path = 'train_data_info.txt'
            self.data_path = os.path.join(data_dir, 'images_train/')
        elif split == 1:
            info_path = 'test_data_info.txt'
            self.data_path = os.path.join(data_dir, 'images_test/')
        
        info_data = []
        with open(os.path.join(data_dir, info_path)) as info_file:
            reader = csv.reader(info_file, delimiter=' ')
            for row in reader:
                info_data.append(row)
                
        self.info = np.array(info_data)
        self.data_len = self.info.shape[0]

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        """
        img_name = os.path.join(self.data_path, self.info[idx, 0])
        img = Image.open(img_name)
        
        if self.transform is not None:
            img = self.transform(img)
                       
        return img

    def __len__(self):
        return self.data_len
    
class ResizeTransform(object):
    ''' Resizes a PIL image to (size, size) to feed into OpenFace net and returns a torch tensor.'''
    def __init__(self, size):
        self.size = size
        
    def __call__(self, sample):
        img = sample.resize((self.size, self.size), Image.BILINEAR)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img)
    
class ZeroPadBottom(object):
    ''' Zero pads batch of image tensor Variables on bottom to given size. Input (B, C, H, W) - padded on H axis. '''
    def __init__(self, size, use_gpu=True):
        self.size = size
        self.use_gpu = use_gpu
        
    def __call__(self, sample):
        B, C, H, W = sample.size()
        diff = self.size - H
        padding = Variable(torch.zeros(B, C, diff, W), requires_grad=False)
        if self.use_gpu:
            padding = padding.cuda()
        zero_padded = torch.cat((sample, padding), dim=2)
        return zero_padded
    
class NormalizeRangeTanh(object):
    ''' Normalizes a tensor with values from [0, 1] to [-1, 1]. '''
    def __init__(self):
        pass
    
    def __call__(self, sample):
        sample = sample * 2.0 - 1.0
        return sample
    
class UnNormalizeRangeTanh(object):
    ''' Unnormalizes a tensor with values from [-1, 1] to [0, 1]. '''
    def __init__(self):
        pass
    
    def __call__(self, sample):
        sample = (sample + 1.0) * 0.5
        return sample
        
    
class UnNormalize(object):
    ''' from https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3'''
    def __init__(self, mean, std):
        mean_arr = []
        for dim in range(len(mean)):
            mean_arr.append(dim)
        std_arr = []
        for dim in range(len(std)):
            std_arr.append(dim)
        self.mean = torch.Tensor(mean_arr).view(1, len(mean), 1, 1)
        self.std = torch.Tensor(std_arr).view(1, len(std), 1, 1)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor *= self.std
        tensor += self.mean
        return tensor
