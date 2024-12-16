from torch.utils.data import DataLoader
from dataset.cufed import TrainSet
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from torch.utils.data import Dataset
import numpy as np

def get_dataloader(args):
    data_train = TrainSet(args)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)
    dataloader = {'train': dataloader_train}

    return dataloader, len(data_train)

class MyTestDataSet(Dataset):

    def __init__(self, image_path, label_path):
        """可以在初始化函数当中对数据进行一些操作，比如读取、归一化等"""
        self.img_dir = sorted(os.listdir(image_path))
        num = len(self.img_dir)
        x_img = []
        y_lab = []
        for i in range(num):
            read_img_path = image_path + self.img_dir[i]
            img = (cv2.imread(read_img_path, cv2.IMREAD_UNCHANGED)).astype(np.float32)
            x_img.append(img[:, :, [2, 1, 0]])
            read_lab_path = label_path + self.img_dir[i]
            lab = (cv2.imread(read_lab_path, cv2.IMREAD_UNCHANGED)).astype(np.float32)
            y_lab.append(lab[:, :, [2, 1, 0]])
        # x_img = np.array(x_img)
        x_img = np.array(x_img)
        y_lab = np.array(y_lab)
        self.x_inputs = x_img
        self.y_labels = y_lab

    def __len__(self):
        """返回数据集当中的样本个数"""
        return len(self.img_dir)

    def __getitem__(self, index):
        """返回样本集中的第 index 个样本；输入变量在前，输出变量在后"""
        return self.x_inputs[index, ], self.y_labels[index, ]

