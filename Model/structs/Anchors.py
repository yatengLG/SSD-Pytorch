# -*- coding: utf-8 -*-
# @Author  : LG
import torch
from math import sqrt

__all__ = ['priorbox']

class priorbox:
    def __init__(self, cfg):
        """
        SSD默认检测框生成器
        :param cfg:
        """
        self.image_size = cfg.MODEL.INPUT.IMAGE_SIZE        # 模型输入图片大小
        anchor_config = cfg.MODEL.ANCHORS
        self.feature_maps = anchor_config.FEATURE_MAPS      # 特征图大小 [38,19,10,5,3,1]
        self.min_sizes = anchor_config.MIN_SIZES            # 检测框大框 [60, 111, 162, 213, 264, 315]
        self.max_sizes = anchor_config.MAX_SIZES            # [30, 60, 111, 162, 213, 264]
        self.aspect_ratios = anchor_config.ASPECT_RATIOS    # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.clip = anchor_config.CLIP                      # True ,检测框越界截断.  0<检测框尺寸<300

    def __call__(self):
        """SSD默认检测框生成
            :return
                Tensor(num_priors,boxes)
                其中boxes(x, y, w, h)
                检测框为比例存储,0~1
        """
        priors = []
        for k, (feature_map_w, feature_map_h) in enumerate(self.feature_maps):
            for i in range(feature_map_w):
                for j in range(feature_map_h):

                    cx = (j + 0.5) / feature_map_w
                    cy = (i + 0.5) / feature_map_h

                    size = self.min_sizes[k]
                    h = w = size / self.image_size
                    priors.append([cx, cy, w, h])

                    size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                    h = w = size / self.image_size
                    priors.append([cx, cy, w, h])

                    size = self.min_sizes[k]
                    h = w = size / self.image_size
                    for ratio in self.aspect_ratios[k]:
                        ratio = sqrt(ratio)
                        priors.append([cx, cy, w * ratio, h / ratio])
                        priors.append([cx, cy, w / ratio, h * ratio])

        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors


if __name__ == '__main__':
    # 运行 查看生成的 检测框
    from Configs import _C as cfg
    boxes = priorbox(cfg = cfg)()
    for box in boxes:
        print(box)
    print(len(boxes))
