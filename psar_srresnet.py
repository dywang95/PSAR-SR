import functools
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from common import Conv_act, combination_img, decompsition_img, make_layer, ResidualBlock_noBN
from iicm import IICM
from marm import MARM


class PSAR(nn.Module):
    def __init__(self, ps=0.0001, pm=0.0002, pl=0.0003, Channel=3, factor=4, crop_high=32, crop_width=32, cfc=12, csc=64, cmc=64, chc=64, cac=64):
        super(PSAR, self).__init__()

        self.factor = factor
        self.crop_high = crop_high
        self.crop_width = crop_width
        self.Channel = Channel
        self.cfc = cfc
        self.csc = csc
        self.cmc = cmc
        self.chc = chc
        self.cac = cac
        self.ps = np.array(ps)
        self.pm = np.array(pm)
        self.pl = np.array(pl)
        self.tl = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.tr = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.iicm = IICM(Channel=self.Channel, cfc=self.cfc)
        self.marm = MARM(Channel=self.Channel, cac=self.cac)

        ## PCSM
        # s
        self.conv_first1 = Conv_act(in_channel=self.cfc, out_channel=self.csc, kernel_size=3, act='lelu')
        self.conv_first2 = Conv_act(in_channel=self.csc, out_channel=self.csc, kernel_size=3, act='lelu')
        basic_block = functools.partial(ResidualBlock_noBN, nf=self.csc)
        self.recon_trunks = make_layer(basic_block, 5)
        # m
        self.recon_trunkm = make_layer(basic_block, 5)
        # l
        self.recon_trunkl = make_layer(basic_block, 6)
        # tail
        self.conv_mid = nn.Conv2d(self.cac, self.cac, 3, 1, 1, bias=True)
        # up
        modules_up = [
            nn.Conv2d(self.csc, self.csc * 4, 3, 1, 1, bias=True),
            nn.Sequential(nn.PixelShuffle(2), nn.LeakyReLU(negative_slope=0.1, inplace=True)),
            nn.Conv2d(self.csc, self.csc * 4, 3, 1, 1, bias=True),
            nn.Sequential(nn.PixelShuffle(2), nn.LeakyReLU(negative_slope=0.1, inplace=True)),
            Conv_act(in_channel=self.cac, out_channel=self.cac, kernel_size=3, act='lelu'),
            nn.Conv2d(self.cac, self.Channel, 3, 1, 1, bias=True)]
        self.up = nn.Sequential(*modules_up)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.constant_(self.tl.data, 0.1)
        nn.init.constant_(self.tr.data, 0.1)

    def forward(self, x, mode):
        base = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        batch_size = x.shape[0]
        img_high = x.shape[2]
        img_width = x.shape[3]
        xcropia = torch.Tensor(batch_size, self.Channel, img_high * self.factor, img_width * self.factor).to(x.device)
        xcropiaf = torch.Tensor(batch_size, self.Channel, img_high * 1, img_width * 1).to(x.device)
        xcropiaff = torch.Tensor(batch_size, self.Channel, img_high * 1, img_width * 1).to(x.device)
        xoutput = torch.Tensor(batch_size, self.cac, img_high * 1, img_width * 1).to(x.device)
        timep = torch.Tensor(batch_size, 1).to(x.device)

        for batchsizei in range(batch_size):
            ## decompsition
            xcrop, xcropiaff[batchsizei, :, :, :], xcropiaf[batchsizei, :, :, :], xcropia[batchsizei, :, :, :] = decompsition_img(
                x[batchsizei, :, :, :], self.crop_high, self.crop_width, self.Channel, 1)
            patch_size = xcrop.shape[0]

            list = np.arange(patch_size)
            ilist = []
            mlist = []
            h1list = []
            h2list = []
            xhhlist = []
            time = torch.Tensor(patch_size, 1).to(x.device)
            xout = torch.Tensor(patch_size, self.csc, self.crop_high, self.crop_width).to(x.device)
            xcrop = torch.from_numpy(xcrop).type(torch.FloatTensor).to(x.device)

            # IICM
            xcrop, ssza = self.iicm(xcrop, patch_size)

            ## PCSM
            ## S
            xcrop = self.conv_first2(self.conv_first1(xcrop))
            xs = self.recon_trunks(xcrop)
            if mode == 'train2' or mode == 'test':
                for i1 in range(patch_size):
                    if ssza[i1] > torch.add(self.tl, self.tr):
                        # print('s:', i1, ssza[i1])
                        xout[i1, :, :, :] = xs[i1, :, :, :].unsqueeze(0)
                        time[i1, 0] = torch.from_numpy(self.ps) * (
                                ssza[i1] - torch.add(self.tl, self.tr))
                        ilist.append(i1)

            for i in list:
                if i not in ilist:
                    mlist.append(i)
            xs1 = xs[mlist, :, :, :]

            ## M
            if len(mlist) != 0:
                xm = self.recon_trunkm(xs1)
                if mode == 'train2' or mode == 'test':
                    for i2 in range(len(mlist)):
                        if ssza[mlist[i2]] >= self.tl and ssza[mlist[i2]] <= torch.add(self.tl, self.tr):
                            # print('m:', mlist[i2], ssza[mlist[i2]])
                            xout[mlist[i2], :, :, :] = xm[i2, :, :, :].unsqueeze(0)
                            time[mlist[i2], 0] = torch.from_numpy(self.pm) * (
                                    ssza[mlist[i2]] - self.tl + torch.add(self.tl, self.tr) - ssza[mlist[i2]])
                            h1list.append(mlist[i2])
                xhh = 0
                for j in mlist:
                    if j not in h1list:
                        xhhlist.append(xhh)
                        h2list.append(j)
                    xhh += 1
                xm1 = xm[xhhlist, :, :, :]

            ## L
            if len(xhhlist) != 0:
                xh = self.recon_trunkl(xm1)
                if mode == 'train1':
                    xout = xh
                if mode == 'train2' or mode == 'test':
                    for i3 in range(len(h2list)):
                        if ssza[h2list[i3]] < self.tl:
                            # print('h:', h2list[i3], ssza[h2list[i3]])
                            xout[h2list[i3], :, :, :] = xh[i3, :, :, :].unsqueeze(0)
                            time[h2list[i3], 0] = torch.from_numpy(self.pl) * (self.tl - ssza[h2list[i3]])
            xout = self.conv_mid(xout) + xcrop

            ## combination
            label = (combination_img(xout, img_high, img_width, self.cac, 1, self.crop_high, self.crop_width)).to(x.device)
            xoutput[batchsizei, :, :, :] = label
            timep[batchsizei, 0] = torch.mean(time[:, 0])

        ## MARM
        xoutput1 = self.marm(x, xoutput)

        ## UP
        xoutput1 = self.up(xoutput1) + base
        xoutputa = xoutput1 * xcropia[:, 0, :, :].reshape(xoutput1.shape[0], 1, xoutput1.shape[2], xoutput1.shape[3])
        return xoutput1, timep, xcropia[:, 0, :, :].unsqueeze(1), xoutputa


if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = int(256//upscale)
    width = int(256//upscale)
    model = PSAR().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    x = torch.randn((1, 3, height, width)).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    from thop import profile
    flops, params = profile(model, inputs=(x, 'test', ))
    print("params:", params/1000000, "flops:", flops/1000000000)