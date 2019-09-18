# -*- coding: utf-8 -*-
# @Author  : LG
import torch
from torch import nn
from .base_models import vgg
from .structs import predictor, multiboxloss, postprocessor, priorbox
from vizer.draw import draw_boxes
from PIL import Image
from Data.Transfroms import SSDTramsfrom
import numpy as np
import time

__all__ = ['SSD']

class SSD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 基础网络、额外层 通过cfg进行配置
        self.backbone = vgg(cfg, pretrained=True)    # VGG,分ssd300,ssd512两种, 使用ssd512需自行配置Config文件
        # conf、loc网络
        self.predictor = predictor(cfg)
        self.postprocessor = postprocessor(cfg)
        self.priors = priorbox(self.cfg)()

    def forward(self, images):
        # 获取特征图  6层特征图组成的元组
        features = self.backbone(images)
        # 将特征图输入 conf以及loc网络 获取类别评分以及回归评分
        cls_logits, bbox_pred = self.predictor(features)
        return cls_logits, bbox_pred

    def load_pretrained_weight(self, weight_pkl):
        self.load_state_dict(torch.load(weight_pkl))

    def forward_with_postprocess(self, images):
        """
        前向传播并后处理
        :param images:
        :return:
        """
        cls_logits, bbox_pred = self.forward(images)
        detections = self.postprocessor(cls_logits, bbox_pred)
        return detections

    @torch.no_grad()
    def Detect_single_img(self, image, score_threshold=0.7, device='cuda'):
        """
        检测单张照片
        eg:
            image, boxes, labels, scores= net.Detect_single_img(img)
            plt.imshow(image)
            plt.show()

        :param image:           图片,PIL.Image.Image
        :param score_threshold: 阈值
        :param device:          检测时所用设备,默认'cuda'
        :return:                添加回归框的图片(np.array),回归框,标签,分数
        """
        self.eval()
        assert isinstance(image,Image.Image)
        w, h = image.width, image.height
        images_tensor = SSDTramsfrom(self.cfg, is_train=False)(np.array(image))[0].unsqueeze(0)

        self.to(device)
        images_tensor = images_tensor.to(device)
        time1 = time.time()
        detections = self.forward_with_postprocess(images_tensor)[0]
        boxes, labels, scores = detections
        boxes, labels, scores = boxes.to('cpu').numpy(), labels.to('cpu').numpy(), scores.to('cpu').numpy()
        boxes[:, 0::2] *= (w / self.cfg.MODEL.INPUT.IMAGE_SIZE)
        boxes[:, 1::2] *= (h / self.cfg.MODEL.INPUT.IMAGE_SIZE)

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        print("Detect {} object, inference cost {:.2f} ms".format(len(scores),(time.time()-time1)*1000))
        # 图像数据加框
        drawn_image = draw_boxes(image=image, boxes=boxes, labels=labels,
                                 scores=scores, class_name_map=self.cfg.DATA.DATASET.CLASS_NAME).astype(np.uint8)
        return drawn_image, boxes, labels, scores

    @torch.no_grad()
    def Detect_video(self, video_path, score_threshold=0.5, save_video_path=None, show=True):
        """
        检测视频
        :param video_path:      视频路径  eg: /XXX/aaa.mp4
        :param score_threshold:
        :param save_video_path: 保存路径,不指定则不保存
        :param show:            在检测过程中实时显示,(会存在卡顿现象,受检测效率影响)
        :return:
        """
        import cv2
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        weight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if save_video_path:
            out = cv2.VideoWriter(save_video_path, fourcc, cap.get(5), (weight, height))
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                drawn_image, boxes, labels, scores =self.Detect_single_img(image=image,
                                                                           device='cuda:0',
                                                                           score_threshold=score_threshold)
                frame = cv2.cvtColor(np.asarray(drawn_image), cv2.COLOR_RGB2BGR)
                if show:
                    cv2.imshow('frame', frame)
                if save_video_path:
                    out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        if save_video_path:
            out.release()
        cv2.destroyAllWindows()
        return True
