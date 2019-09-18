# -*- coding: utf-8 -*-
# @Author  : LG

"""
没写完
"""
import hashlib
import os
from Utils import GetFileMd5
from Configs import _C as cfg
import wget

url = 'XXXXXX/vgg_ssd300_voc0712.pkl'
weight_name = url.split('/')[-1]
weight_path = cfg.FILE.PRETRAIN_WEIGHT_ROOT
weight_file = os.path.join(weight_path, weight_name)

if not os.path.exists(weight_file):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    print('{} no exist ,downloading .....'.format(weight_name))
    wget.download(url=url,out=weight_file)

    print('donwload finish')
md5 = GetFileMd5(weight_file)
if md5 =='2acbd3bcd23ec7378a2ee466697bcc03':
    print('模型权重文件已就绪')
else:
    print('模型权重文件校验失败')