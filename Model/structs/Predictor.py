# -*- coding: utf-8 -*-
# @Author  : LG
import torch
from torch import nn

__all__ = ['predictor']

class predictor(nn.Module):
    """
    分类(cls)及回归(reg)网络
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cls_headers = nn.ModuleList()
        self.reg_headers = nn.ModuleList()
        for boxes_per_location, out_channels in zip(cfg.MODEL.ANCHORS.BOXES_PER_LOCATION, cfg.MODEL.ANCHORS.OUT_CHANNELS ):
            self.cls_headers.append(self.cls_block(out_channels, boxes_per_location))
            self.reg_headers.append(self.reg_block(out_channels, boxes_per_location))
        self.reset_parameters()

    def cls_block(self, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * self.cfg.DATA.DATASET.NUM_CLASSES, kernel_size=3, stride=1, padding=1)

    def reg_block(self, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        """
        对输入的特征图中每个特征点进行分类及回归(不同特征图特征点对应的输出数是不一样的,以检测框数量为准)
        :param features:    # base_model 输出的特征图,这里SSD_VGG_300 为六层特征图
        :return:            # 每个特征点的类别预测与回归预测(输出数量以各自特征点上检测框数量为准)
        """
        cls_logits = []
        bbox_pred = []
        for feature, cls_header, reg_header in zip(features, self.cls_headers, self.reg_headers):
            cls_logits.append(cls_header(feature).permute(0, 2, 3, 1).contiguous())
            bbox_pred.append(reg_header(feature).permute(0, 2, 3, 1).contiguous())

        batch_size = features[0].shape[0]
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits], dim=1).view(batch_size, -1, self.cfg.DATA.DATASET.NUM_CLASSES)
        bbox_pred = torch.cat([l.view(l.shape[0], -1) for l in bbox_pred], dim=1).view(batch_size, -1, 4)

        return cls_logits, bbox_pred

