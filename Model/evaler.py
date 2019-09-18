# -*- coding: utf-8 -*-
# @Author  : LG
import torch
from Configs import _C as cfg
from Utils import eval_detection_voc
from tqdm import tqdm
from torch.nn import DataParallel
from torch import nn
from Data import Our_Dataloader_test
from .structs import postprocessor


class Evaler(object):
    """
    模型测试器,不指定参数时,均默认使用Configs中配置的参数
    *** 推荐使用Configs文件管理参数, 不推荐在函数中进行参数指定, 只是为了扩展  ***

    模型在测试时,会使用DataParallel进行包装,以便于在多GPU上进行测试
    本测试器只支持GPU训练,单机单卡与单机单卡均可,但不支持cpu,不支持多机多卡(别问为啥不支持多机多卡.穷!!!)

    eg:
        evaler = Evaler(cfg,eval_devices=[0,1])         # 使用俩块GPU进行测试,使用时请指定需使用的gpu编号,终端运行nvidia-smi进行查看
        ap, map = evaler(net,test_dataset=test_dataset)
    """
    def __init__(self, cfg, eval_devices=None):
        self.cfg = cfg
        self.postprocessor = postprocessor(cfg)

        self.eval_devices = self.cfg.DEVICE.TEST_DEVICES
        if eval_devices:
            self.eval_devices = eval_devices

    def __call__(self, model, test_dataset):
        model.eval()
        if not isinstance(model, nn.DataParallel):
            model = DataParallel(model, device_ids=self.eval_devices)
        else:
            model = DataParallel(model.module, device_ids=self.eval_devices)
        test_loader = Our_Dataloader_test(dataset=test_dataset, batch_size=20)
        results_dict = self.eval_model_inference(model, data_loader=test_loader)
        result = cal_ap_map(results_dict, test_dataset=test_loader.dataset)
        ap, map = result['ap'], result['map']
        return ap, map

    def eval_model_inference(self, model, data_loader):
        with torch.no_grad():
            results_dict = {}
            print(' Evaluating...... use GPU : {}'.format(self.eval_devices))
            for images, boxes, labels, image_names in tqdm(data_loader):
                cls_logits, bbox_pred = model(images)
                results = self.postprocessor(cls_logits, bbox_pred)
                for image_name, result in zip(image_names, results):
                    pred_boxes, pred_labels, pred_scores = result
                    pred_boxes, pred_labels, pred_scores = pred_boxes.to('cpu').numpy(), \
                                                           pred_labels.to('cpu').numpy(), \
                                                           pred_scores.to('cpu').numpy()
                    results_dict.update({image_name: {'pred_boxes': pred_boxes,
                                                      'pred_labels': pred_labels,
                                                      'pred_scores': pred_scores}})
        return results_dict


def cal_ap_map(results_dict,test_dataset):
    pred_boxes_list = []
    pred_labels_list = []
    pred_scores_list = []
    gt_boxs_list = []
    gt_labels_list = []
    gt_difficult_list = []
    for img_name in results_dict:
        gt_boxs, gt_labels, gt_difficult = test_dataset._get_annotation(img_name)
        size = test_dataset.get_img_size(img_name)
        w, h = size['width'],size['height']
        pred_boxes, pred_labels, pred_scores= results_dict[img_name]['pred_boxes'],results_dict[img_name]['pred_labels'],results_dict[img_name]['pred_scores']
        pred_boxes[:, 0::2] *= (w / cfg.MODEL.INPUT.IMAGE_SIZE)
        pred_boxes[:, 1::2] *= (h / cfg.MODEL.INPUT.IMAGE_SIZE)
        pred_boxes_list.append(pred_boxes)
        pred_labels_list.append(pred_labels)
        pred_scores_list.append(pred_scores)
        gt_boxs_list.append(gt_boxs)
        gt_labels_list.append(gt_labels)
        gt_difficult_list.append(gt_difficult)
    result = eval_detection_voc(pred_bboxes=pred_boxes_list,
                                pred_labels=pred_labels_list,
                                pred_scores=pred_scores_list,
                                gt_bboxes=gt_boxs_list,
                                gt_labels=gt_labels_list,
                                gt_difficults=gt_difficult_list)
    return result
