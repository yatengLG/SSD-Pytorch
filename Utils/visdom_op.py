# -*- coding: utf-8 -*-
# @Author  : LG
import visdom
import torch
import numpy as np

"""
visdom使用，
首先先安装visdom pip install visdom
启动 visdom服务器，python -m visdom.server
默认为 http://localhost:8097/
"""

"""
"""

def setup_visdom(**kwargs):

    """
    eg :
        vis_eval = setup_visdom(env='SSD_eval')

    :param kwargs:
    :return:
    """
    vis = visdom.Visdom(**kwargs)
    return vis


def visdom_line(vis, y, x, win_name, update='append'):

    """
    eg :
        visdom_line(vis_train, y=[loss], x=iteration, win_name='loss')

    :param vis:         由　setup_visdom　函数创建
    :param y:           Ｙ轴数据，为一系列数据，可同时传入多种数据。　ｅｇ　：　[loss1, loss2]
    :param x:           Ｘ轴，格式同Ｙ
    :param win_name:    绘图窗口名称，必须指定，不然会一直创建窗口
    :param update:      绘图方式。　这里默认append连续绘图,用于记录损失变化曲线
    :return:
    """
    if not isinstance(y,torch.Tensor):
        y=torch.Tensor(y)
    y = y.unsqueeze(0)
    x = torch.Tensor(y.size()).fill_(x)
    vis.line(Y=y, X=x, win=win_name, update=update, opts={'title':win_name})
    return True


def visdom_images(vis, images,win_name,num_show=None,nrow=None):
    """
    eg:
        visdom_images(vis_train, images, num_show=3, nrow=3, win_name='Image')

    visdom　展示图片，默认只展示６张，每行３张。

    :param vis:         由　setup_visdom　函数创建
    :param images:      多幅图片张量，shape:[B,N,W,H]
    :param win_name:    绘图窗口名称，必须指定，不然会一直创建窗口
    :param num_show:    展示的图片数量，默认六张
    :param nrow:        每行展示的图片数量，默认三张
    :return:
    """
    if not num_show:
        num_show = 6
    if not nrow:
        nrow = 3
    num = images.size(0)
    if num > num_show:
        images = images [:num_show]
    vis.images(tensor=images,nrow=nrow,win=win_name)
    return True


def visdom_image(vis, image,win_name):
    """
    eg :
        visdom_image(vis=vis, image=drawn_image, win_name='image')

    :param vis:         由　setup_visdom　函数创建
    :param image:       单幅图片张量，ｓｈａｐｅ:[n,w,h]
    :param win_name:    绘图窗口名称，必须指定，不然会一直创建窗口
    :return:
    """
    vis.image(img=image, win=win_name)
    return True

def visdom_bar(vis, X, Y, win_name):
    """
    绘制柱形图
    eg:
        visdom_bar(vis_train, X=cfg.DATASETS.CLASS_NAME, Y=ap, win_name='ap', title='ap')

    :param vis:
    :param X:           类别
    :param Y:           数值
    :param win_name:    绘图窗口名称，必须指定，不然会一直创建窗口
    :return:
    """
    dic = dict(zip(X,Y))
    del_list = []
    for val in dic:
        if np.isnan(dic[val]):
            del_list.append(val)

    for val in del_list:
        del dic[val]

    vis.bar(X=list(dic.values()),Y=list(dic.keys()),win=win_name, opts={'title':win_name})
    return True