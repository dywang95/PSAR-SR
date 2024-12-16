import torch
from torch import nn
from common import Conv_act, calc_asm


class IICM(nn.Module):
    def __init__(self, Channel, cfc):
        super(IICM, self).__init__()

        self.conv1 = Conv_act(in_channel=Channel, out_channel=cfc, kernel_size=3)
        self.conv2 = Conv_act(in_channel=cfc, out_channel=cfc, kernel_size=3)
        self.conv31 = Conv_act(in_channel=cfc, out_channel=1, kernel_size=1)
        self.conv32 = Conv_act(in_channel=cfc, out_channel=1, kernel_size=1)
        self.pool1 = nn.AdaptiveMaxPool2d(16)
        self.pool2 = nn.AdaptiveMaxPool2d(4)
        self.relu = nn.ReLU()
        self.sig = torch.nn.Sigmoid()

    def forward(self, x, patch_size):
        ssza = torch.Tensor(patch_size, 1).to(x.device)

        xo = self.conv1(x)
        x1 = self.pool1(xo)
        x2 = self.conv2(x1)
        x21 = self.conv31(x1)
        x22 = self.sig(x21)
        x3 = self.conv32(x22 * x2)
        x3 = self.pool2(x3)
        for i in range(patch_size):
            sst = x3[i, :, :, :].view(4, 4)
            ssza[i] = calc_asm(self.relu(sst))
        ssza = ssza.to(x.device)
        return xo, ssza