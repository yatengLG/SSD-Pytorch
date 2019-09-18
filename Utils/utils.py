# -*- coding: utf-8 -*-
# @Author  : LG
import numpy as np
import cv2
from tqdm import tqdm
import json
import os
import shutil

def json_to_txt(json_file, root):
    """
    解析json文件
            文件名是图片名,
            数据以   label1, x1, y1, w1, h1
                    label2, x2, y2, w2, h2
            格式存放
    eg:
        json_to_txt('/home/super/guangdong1_round1_train1_20190809/Annotations/gt_result.json',
                    '/home/super/guangdong1_round1_train1_20190809/Our')

    :param json_file:
    :param root:
    :return:
    """
    if os.path.exists(root):
        shutil.rmtree(root)
    os.mkdir(root)

    with open(json_file, 'r')as f:
        json_dict_list = json.load(f)
        for json_dict in json_dict_list:
            name = json_dict['name']
            defect_name = json_dict['defect_name']
            bbox = json_dict['bbox']
            content = [defect_name]
            for xywh in bbox:
                content.append(str(xywh))
            content = ','.join(content)

            with open(os.path.join(root, name.split('.')[0] + '.txt'), 'a')as f:
                f.write(content + '\n')
    return True


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
