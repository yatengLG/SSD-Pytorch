# -*- coding: utf-8 -*-
# @Author  : LG
from Utils.Boxs_op import center_form_to_corner_form, assign_priors,\
    corner_form_to_center_form, convert_boxes_to_locations
from Data.Transfroms_utils import *

__all__ = ['SSDTramsfrom', 'SSDTargetTransform']

class SSDTramsfrom:
    """
    targets_transfroms
    eg:
        transform = SSDTramsfrom(cfg,is_train=True)
    """
    def __init__(self,cfg, is_train):
        if is_train:
            self.transforms = [
                ConvertFromInts(),  # 图像数据转float32
                PhotometricDistort(),   # 光度畸变,对比度,亮度,光噪声,色调,饱和等(详情看函数,有详细备注.)
                # SubtractMeans(cfg.MODEL.INPUT.PIXEL_MEAN),  # 减均值
                # DivideStds(cfg.MODEL.INPUT.PIXEL_STD),    # 除方差
                Expand(), # 随机扩充
                RandomSampleCrop(), # 随机交兵比裁剪
                RandomMirror(),     # 随机镜像
                ToPercentCoords(),  # boxes 坐标转百分比制
                Resize(cfg.MODEL.INPUT.IMAGE_SIZE),

                ToTensor(),
            ]
        else:
            self.transforms = [
                Resize(cfg.MODEL.INPUT.IMAGE_SIZE),
                SubtractMeans(cfg.MODEL.INPUT.PIXEL_MEAN),
                ToTensor()
            ]

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class SSDTargetTransform:
    """
    targets_transfroms
    eg:
        transform = SSDTargetTransform(cfg)
    """

    def __init__(self, cfg):
        from Model.structs import priorbox  # 避免循环导入.(模型中detect方法会使用transfrom,而targettransfrom会使用到priorbox, 这样写可以避免循环导入)

        self.center_form_priors = priorbox(cfg)()
        self.corner_form_priors = center_form_to_corner_form(self.center_form_priors)
        self.center_variance = cfg.MODEL.ANCHORS.CENTER_VARIANCE
        self.size_variance = cfg.MODEL.ANCHORS.SIZE_VARIANCE
        self.iou_threshold = cfg.MODEL.ANCHORS.THRESHOLD

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = corner_form_to_center_form(boxes)
        locations = convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance,
                                                         self.size_variance)
        return locations, labels
