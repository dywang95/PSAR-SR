import os
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class RandomRotate(object):
    def __call__(self, sample):
        # k1 = np.random.randint(0, 4)
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.rot90(sample['LR'], k=1).copy()
            sample['HR'] = np.rot90(sample['HR'], k=1).copy()
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.rot90(sample['LR'], k=2).copy()
            sample['HR'] = np.rot90(sample['HR'], k=2).copy()
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.rot90(sample['LR'], k=3).copy()
            sample['HR'] = np.rot90(sample['HR'], k=3).copy()
        return sample

class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.fliplr(sample['LR']).copy()
            sample['HR'] = np.fliplr(sample['HR']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.flipud(sample['LR']).copy()
            sample['HR'] = np.flipud(sample['HR']).copy()
        return sample


class ToTensor(object):
    def __call__(self, sample):
        LR, HR = sample['LR'], sample['HR']
        LR = LR.transpose((2, 0, 1))
        HR = HR.transpose((2, 0, 1))
        return {'LR': torch.from_numpy(LR).float(),
                'HR': torch.from_numpy(HR).float()}


class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()])):
        self.input_list = sorted([os.path.join(args.dataset_dir, 'LR/X4', name) for name in
                                  os.listdir(os.path.join(args.dataset_dir, 'LR/X4'))])
        self.lab_list = sorted([os.path.join(args.dataset_dir, 'HR', name) for name in
                                os.listdir(os.path.join(args.dataset_dir, 'HR'))])
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        LR = cv2.imread(self.input_list[idx], cv2.IMREAD_UNCHANGED)
        HR = cv2.imread(self.lab_list[idx], cv2.IMREAD_UNCHANGED)

        ### LR and LR_sr
        LR = np.array(Image.fromarray(LR).resize((84, 84), Image.BICUBIC))
        HR = np.array(Image.fromarray(HR).resize((84*4, 84*4), Image.BICUBIC))

        ### change type
        LR = LR.astype(np.float32)
        HR = HR.astype(np.float32)

        LR = LR[:, :, [2, 1, 0]]
        HR = HR[:, :, [2, 1, 0]]

        sample = {'LR': LR,
                  'HR': HR}

        sample = self.transform(sample)
        return sample


class TestSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([ToTensor()])):
        self.input_list = sorted([os.path.join(args.dataset_dir, 'LR1', name) for name in
                                  os.listdir(os.path.join(args.dataset_dir, 'LR1'))])
        self.lab_list = sorted([os.path.join(args.dataset_dir, 'HR1', name) for name in
                                os.listdir(os.path.join(args.dataset_dir, 'HR1'))])
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        LR = cv2.imread(self.input_list[idx], cv2.IMREAD_UNCHANGED)
        HR = cv2.imread(self.lab_list[idx], cv2.IMREAD_UNCHANGED)

        ### change type
        LR = LR.astype(np.float32)
        HR = HR.astype(np.float32)

        LR = LR[:, :, [2, 1, 0]]
        HR = HR[:, :, [2, 1, 0]]

        sample = {'LR': LR,
                  'HR': HR}

        sample = self.transform(sample)
        return sample