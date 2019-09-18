# -*- coding: utf-8 -*-
# @Author  : LG
import numpy as np
import cv2
from tqdm import tqdm
import os

def cal_mean_std(images_dir):
    """
    给定数据图片根目录,计算图片整体均值与方差
    :param images_dir:
    :return:
    """
    img_filenames = os.listdir(images_dir)
    m_list, s_list = [], []
    for img_filename in tqdm(img_filenames):
        img = cv2.imread(images_dir + '/' + img_filename)
        img = img / 255.0
        m, s = cv2.meanStdDev(img)

        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
        print(m_list)
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print('mean: ',m[0][::-1])
    print('std:  ',s[0][::-1])
    return m
