# -*- coding: utf-8 -*-
# @Author  : LG
import torch.nn as nn
import torch.nn.functional as F
import os
import wget
import torch
from torch.nn import init
from Utils import GetFileMd5

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


def add_vgg(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, size=300):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return layers


vgg_base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras_base = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}


class VGG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        vgg_config = vgg_base[str(self.cfg.MODEL.INPUT.IMAGE_SIZE)]
        extras_config = extras_base[str(self.cfg.MODEL.INPUT.IMAGE_SIZE)]

        self.vgg = nn.ModuleList(add_vgg(vgg_config))
        self.extras = nn.ModuleList(add_extras(extras_config, i=1024, size=self.cfg.MODEL.INPUT.IMAGE_SIZE))
        self.l2_norm = L2Norm(512, scale=20)

    def forward(self, x):
        features = []
        for i in range(23):
            x = self.vgg[i](x)
        s = self.l2_norm(x)  # Conv4_3 L2 normalization
        features.append(s)

        # apply vgg up to fc7
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        features.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                features.append(x)

        return tuple(features)

    def reset_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def load_weights(self):
        url = 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth'
        weight_name = url.split('/')[-1]
        weight_path = self.cfg.FILE.PRETRAIN_WEIGHT_ROOT
        weight_file = os.path.join(weight_path, weight_name)

        if not os.path.exists(weight_file):
            if not os.path.exists(weight_path):
                os.makedirs(weight_path)

            print('{} no exist ,downloading .....'.format(weight_name))
            wget.download(url=url,out=weight_file)

            print('donwload finish')
        md5 = GetFileMd5(weight_file)
        if md5 =='9fb5a8dfd5f42dc1090365b6179aa659':
            self.load_state_dict(torch.load(weight_file), strict=False)
            print(' --- load weight finish ---')
        else:
            print('VGG weights file check failed ')

def vgg(cfg, pretrained=True):
    print(' --- base_model = vgg16_ssd{} --- '.format(cfg.MODEL.INPUT.IMAGE_SIZE))
    model = VGG(cfg)
    if pretrained:
        model.load_weights()
    else:
        model.reset_parameters()
    return model