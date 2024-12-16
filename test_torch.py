import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import numpy as np
from thop import profile
import os
from dataset.dataloader import MyTestDataSet
from common import calc_psnr
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_file', type=str, default='')
    parser.add_argument('--save_file', type=str, default='')
    parser.add_argument('--test_file', type=str, default='')
    parser.add_argument('--testlabel_file', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--factor', type=int, default=4)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torch.load(args.weights_file)
    model.eval()

    test_dataset = MyTestDataSet(args.test_file, args.testlabel_file)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)
    num = 0
    pn = 0.0
    fl = 0.0
    pa = 0.0
    for data in test_dataloader:
        inputs, labels = data

        inputs = inputs.permute(0, 3, 1, 2)
        labels = labels.permute(0, 3, 1, 2)

        inputs = inputs.float()
        labels = labels.float()

        inputs = inputs.to(device)
        labels = labels.to(device)

        flops, params = profile(model, inputs=(inputs, args.mode))
        fl = flops / 1000000 + fl
        pa = params / 1000 + pa

        with torch.no_grad():
            preds = model(inputs, args.mode)[0].to(device)
            preds = preds * 255.
        for i in range(args.batch_size):
            psnr = calc_psnr(labels[i, :, :, :], preds[i, :, :, :])
            pn += psnr
            output = preds[i, :, :, :].permute(1, 2, 0)
            output = np.array(output.data.cpu().numpy())
            cv2.imwrite(args.save_file + str(num) + ".png", output)
            num += 1
    print('avg PSNR: {:.4f}'.format(pn/num),'avg params: {:.4f}'.format(pa/num),'avg flops: {:.4f}'.format(fl/num))