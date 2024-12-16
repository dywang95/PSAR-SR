import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn import preprocessing
from skimage.feature import graycomatrix, graycoprops
import torch.nn.init as init
import cv2


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class MeanShift1(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift1, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
                # m.append(nn.ReLU())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
            # m.append(nn.ReLU())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

def init_weights(modules):
    pass

class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out

class EResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out

class _UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale,
                 group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out

class UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale, multi_scale,
                 group=1):
        super(UpsampleBlock, self).__init__()

        self.scale = scale
        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x):
        if self.multi_scale:
            if self.scale == 2:
                return self.up2(x)
            elif self.scale == 3:
                return self.up3(x)
            elif self.scale == 4:
                return self.up4(x)
        else:
            return self.up(x)

class Block(nn.Module):
    def __init__(self, nf,
                 group=1):
        super(Block, self).__init__()

        self.b1 = EResidualBlock(nf, nf, group=group)
        self.c1 = BasicBlock(nf * 2, nf, 1, 1, 0)
        self.c2 = BasicBlock(nf * 3, nf, 1, 1, 0)
        self.c3 = BasicBlock(nf * 4, nf, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            # nn.ReLU(inplace=True),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            # if bn: modules_body.append(act)
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class Conv_act(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, act='relu', groups=1):
        super(Conv_act, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups)
        if act == 'relu':
            self.act = nn.ReLU()
        if act == 'prelu':
            self.act = nn.PReLU()
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.act(self.conv(x))
        return out

def calc_asm(img):
    img = np.array(img.data.cpu().numpy())
    img = preprocessing.normalize(img, norm='l2') * 99
    img = img.astype(np.int32)
    glcm = graycomatrix(img, distances=[1], angles=[0, np.pi / 2], levels=100,
                        symmetric=True, normed=True)
    asm = graycoprops(glcm, 'ASM')
    asmm = np.mean(asm)
    return asmm

def decompsition_img(image, crop_high, crop_width, Channel, factor1):
    m = image.shape[1]
    n = image.shape[2]
    oasize = 1
    osize = 2 * oasize
    h1 = math.ceil(m/crop_high)
    z1 = math.ceil(n/crop_high)
    crop_img = np.zeros([h1*z1, Channel, crop_high, crop_width])
    cropif_img = torch.from_numpy(np.zeros([Channel, m * factor1, n * factor1]))
    num = 0
    for i in range(h1):
        for j in range(z1):
            if i == h1 - 1 and j == z1 - 1:
                imgc = image[:, m - crop_high:m, n - crop_high:n]
                cropif_img[:, (m - crop_high) * factor1 - osize:(m - crop_high) * factor1 + osize, :] = torch.tensor(1)
                cropif_img[:, :, (n - crop_high) * factor1 - osize:(n - crop_high) * factor1 + osize] = torch.tensor(1)
            elif i == 0 and j != z1-1:
                imgc = image[:, (i * crop_high):(i * crop_high + crop_high), (j * crop_high):(j * crop_high + crop_high)]
            elif j == 0 and i != h1-1:
                imgc = image[:, (i * crop_high):(i * crop_high + crop_high), (j * crop_high):(j * crop_high + crop_high)]
            elif i == 0 and j == z1-1:
                imgc = image[:, (i * crop_high):(i * crop_high + crop_high), n-crop_high:n]
            elif j == 0 and i == h1-1:
                imgc = image[:, m - crop_high:m, (j * crop_high):(j * crop_high + crop_high)]
            elif j == z1-1:
                imgc = image[:, (i*crop_high):(i*crop_high + crop_high), n-crop_high:n]
                cropif_img[:, :, (n - crop_high) * factor1 - osize:(n - crop_high) * factor1 + osize] = torch.tensor(1)
            elif i == h1-1:
                imgc = image[:, m - crop_high:m, (j * crop_high):(j * crop_high + crop_high)]
                cropif_img[:, (m - crop_high) * factor1 - osize:(m - crop_high) * factor1 + osize, :] = torch.tensor(1)
            else:
                imgc = image[:, (i*crop_high):(i*crop_high + crop_high), (j*crop_high):(j*crop_high + crop_high)]
                cropif_img[:, (i * crop_high) * factor1 - osize:(i * crop_high) * factor1 + osize, :] = torch.tensor(1)
                cropif_img[:, :, (j * crop_high) * factor1 - osize:(j * crop_high) * factor1 + osize] = torch.tensor(1)
            crop_img[num, :, :, :] = imgc.cpu().detach()
            num = num + 1
    cropiff_img = torch.from_numpy(np.ones([Channel, m * factor1, n * factor1])) - cropif_img
    cropifu_img = F.interpolate(cropif_img.unsqueeze(0), scale_factor=4, mode='bilinear')
    return crop_img, cropif_img, cropiff_img, cropifu_img.squeeze(0)

def combination_img(lab_image,img_high,img_width,Channel,factor,crop_high, crop_width):
    m1 = img_high * factor
    n1 = img_width * factor
    lab_h = torch.zeros((Channel, m1, n1))
    h1 = math.ceil(img_high/crop_high)
    z1 = math.ceil(img_width/crop_width)
    num = 0
    for i in range(h1):
        for j in range(z1):
            labh1 = lab_image[num, :, :, :]
            if i == h1-1 and j == z1-1:
                lab_h[:,m1-crop_high*factor:m1,  n1-crop_high*factor:n1] = labh1
            elif j == z1-1:
                lab_h[:,i*crop_high*factor:(crop_high *factor + i*crop_high * factor), n1-crop_high*factor:n1] = labh1
            elif i == h1-1:
                lab_h[:,m1-crop_high*factor:m1, j*crop_high*factor:(crop_high *factor + j*crop_high * factor)] = labh1
            else:
                lab_h[:,i*crop_high*factor:(crop_high *factor + i*crop_high * factor), j*crop_high*factor:(crop_high *factor + j*crop_high * factor)] = labh1
            num = num+1
    return lab_h


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_psnr(img1, img2):
    return 10. * torch.log10(255 ** 2. / torch.mean((img1 - img2) ** 2))

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()