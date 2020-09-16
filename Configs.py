# -*- coding: utf-8 -*-
# @Author  : LG
from yacs.config import CfgNode as CN
import os

### 参数请结合自身项目设定，才能跑出较好的效果。

project_root = os.getcwd()

_C = CN()


_C.FILE = CN()

_C.FILE.PRETRAIN_WEIGHT_ROOT = project_root+'/Weights/pretrained'   # 会使用到的预训练模型
_C.FILE.MODEL_SAVE_ROOT = project_root+'/Weights/trained'           # 训练模型的保存
_C.FILE.VGG16_WEIGHT = 'vgg16_reducedfc.pth'                        # vgg预训练模型

_C.DEVICE = CN()

_C.DEVICE.MAINDEVICE = 'cuda:0' # 主gpu
_C.DEVICE.TRAIN_DEVICES = [0,1] # 训练gpu
_C.DEVICE.TEST_DEVICES = [0,1]  # 检测gpu

_C.MODEL = CN()

_C.MODEL.INPUT = CN()
_C.MODEL.INPUT.IMAGE_SIZE = 300         # 模型输入尺寸
_C.MODEL.INPUT.PIXEL_MEAN = [0, 0, 0]   # 数据集均值
_C.MODEL.INPUT.PIXEL_STD = [1, 1, 1]    # 数据集方差

_C.MODEL.ANCHORS = CN()
_C.MODEL.ANCHORS.FEATURE_MAPS = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]  # 特征图大小
_C.MODEL.ANCHORS.MIN_SIZES = [30, 60, 111, 162, 213, 264]   # 检测框大小
_C.MODEL.ANCHORS.MAX_SIZES = [60, 111, 162, 213, 264, 315]  # 检测框大小
_C.MODEL.ANCHORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]    # 不同特征图上检测框绘制比例
_C.MODEL.ANCHORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 4, 4]  # 不同特征图上特征点的检测框数量
_C.MODEL.ANCHORS.OUT_CHANNELS = [512, 1024, 512, 256, 256, 256] # 特征图数量
_C.MODEL.ANCHORS.CLIP = True            # 越界检测框截断,0~1
_C.MODEL.ANCHORS.THRESHOLD = 0.5        # 交并比阈值
_C.MODEL.ANCHORS.CENTER_VARIANCE = 0.1  # 解码
_C.MODEL.ANCHORS.SIZE_VARIANCE = 0.2    # 解码

_C.TRAIN = CN()

_C.TRAIN.NEG_POS_RATIO = 3      # 负正例比例
_C.TRAIN.MAX_ITER = 120000      # 训练轮数
_C.TRAIN.BATCH_SIZE = 10        # 训练批次
_C.TRAIN.NUM_WORKERS = 4        # 数据数据所使用的线程数
_C.OPTIM = CN()

_C.OPTIM.LR = 1e-3              # 初始学习率.默认优化器为SGD
_C.OPTIM.MOMENTUM = 0.9         # 优化器动量.默认优化器为SGD
_C.OPTIM.WEIGHT_DECAY = 5e-4    # 权重衰减,L2正则化.默认优化器为SGD

_C.OPTIM.SCHEDULER = CN()       # 默认使用MultiStepLR
_C.OPTIM.SCHEDULER.GAMMA = 0.1  # 学习率衰减率
_C.OPTIM.SCHEDULER.LR_STEPS = [80000, 100000]


_C.MODEL.TEST = CN()

_C.MODEL.TEST.NMS_THRESHOLD = 0.45              # 非极大抑制阈值
_C.MODEL.TEST.CONFIDENCE_THRESHOLD = 0.01       # 分数阈值,
_C.MODEL.TEST.MAX_PER_IMAGE = 100               # 预测结果最大数量
_C.MODEL.TEST.MAX_PER_CLASS = -1                # 测试时,top-N


_C.DATA = CN()

# 由于在使用时,是自己的数据集.所以这里,并没有写0712合并的数据集格式,这里以VOC2007为例
_C.DATA.DATASET = CN()
_C.DATA.DATASET.NUM_CLASSES =21
_C.DATA.DATASET.CLASS_NAME = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

_C.DATA.DATASET.DATA_DIR = '/home/XXX/VOCdevkit/VOC2007'   # 数据集voc格式,根目录
_C.DATA.DATASET.TRAIN_SPLIT = 'train'       # 训练集,对应于 /VOCdevkit/VOC2007/ImageSets/Main/train.txt'
_C.DATA.DATASET.TEST_SPLIT = 'val'          # 测试集,对应于 /VOCdevkit/VOC2007/ImageSets/Main/val.txt'

_C.DATA.DATALOADER = CN()


_C.STEP = CN()
_C.STEP.VIS_STEP = 10           # visdom可视化训练过程,打印步长
_C.STEP.MODEL_SAVE_STEP = 1000  # 训练过程中,模型保存步长
_C.STEP.EVAL_STEP = 1000        # 在训练过程中,并没有进行检测流程,建议保存模型后另外检测

