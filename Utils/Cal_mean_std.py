# -*- coding: utf-8 -*-
# @Author  : LG
# 计算数据集均值方差
import numpy as np
import os
from PIL import Image

def get_mean_std(img_root):

    means = 0
    stds = 0
    img_list = os.listdir(img_root)
    num = len(img_list)
    for i, img in enumerate(img_list):
        i +=1
        img = os.path.join(img_root,img)
        img = np.array(Image.open(img))
        mean = np.mean(img, axis=(0,1))
        std = np.std(img, axis=(0,1))

        means += mean
        stds += std
        print(' {}/{} , mean : [{:.2f}, {:.2f}, {:.2f}], std : [{:.2f}, {:.2f}, {:.2f}]'.format(i, num, means[0]/i, means[1]/i, means[2]/i, stds[0]/i, stds[1]/i, stds[2]/i))
    mean = means / i
    std = stds / i
    return mean, std