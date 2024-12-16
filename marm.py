import torch
from torch import nn
from common import Conv_act


class MARM(nn.Module):
    def __init__(self, Channel, cac):
        super(MARM, self).__init__()

        self.Channel = Channel
        self.convg1 = nn.Conv2d(cac, 1, kernel_size=1, padding=1 // 2)
        self.convb = nn.Conv2d(Channel, cac, kernel_size=1, padding=1 // 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(cac * 2, 8, 1, bias=False)
        self.fc2 = nn.Conv2d(8, cac * 2, 1, bias=False)
        self.convt = Conv_act(in_channel=cac * 2, out_channel=cac, kernel_size=3, groups=1)
        self.convf1 = Conv_act(in_channel=cac, out_channel=cac, kernel_size=3, groups=cac)
        self.convf2 = Conv_act(in_channel=cac, out_channel=cac, kernel_size=3, groups=cac)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x, xoutput):
        batch_size = x.shape[0]
        img_high = x.shape[2]
        img_width = x.shape[3]
        xcropiaf = torch.Tensor(batch_size, self.Channel, img_high * 1, img_width * 1).to(x.device)
        xcropiaff = torch.Tensor(batch_size, self.Channel, img_high * 1, img_width * 1).to(x.device)

        x = self.convb(x)
        xo = x
        xoutputtc = xoutput - xo
        xoutputt = xcropiaf[:, 0, :, :].unsqueeze(1) * xoutputtc
        xoutput_ah = xoutputtc * xcropiaff[:, 0, :, :].unsqueeze(1)
        xoutputg = self.sig(self.convg1(xoutput_ah))
        outcat = torch.cat([xo, xoutput_ah * xoutputg + xoutputt], dim=1)
        outcat_ca = self.sig(self.fc2(self.fc1(self.avg_pool(outcat))) + self.fc2(self.fc1(self.max_pool(outcat))))
        xoutput1 = self.convt(outcat * outcat_ca)
        xoutput1 = self.convf1(xoutput1)
        xoutput1 = self.convf2(xoutput1)
        return xoutput1