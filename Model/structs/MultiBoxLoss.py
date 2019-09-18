# -*- coding: utf-8 -*-
# @Author  : LG
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

__all__ = ['multiboxloss']

class multiboxloss(nn.Module):
    def __init__(self, neg_pos_ratio):
        """
        SSD损失函数,分为类别损失(使用cross_entropy)
        框体回归损失(使用smooth_l1_loss)
        这里并没有在返回时,采用分别返回的方式返回.便于训练过程中分析处理
        :param neg_pos_ratio:
        """
        super(multiboxloss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """
            计算类别损失和框体回归损失
        Args:
            confidence (batch_size, num_priors, num_classes): 预测类别
            predicted_locations (batch_size, num_priors, 4): 预测位置
            labels (batch_size, num_priors): 所有框的真实类别
            gt_locations (batch_size, num_priors, 4): 所有框真实的位置
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.view(-1, num_classes), labels[mask], reduction='sum')

        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos


def hard_negative_mining(loss, labels, neg_pos_ratio=3):
    """
    用于训练过程中正负例比例的限制.默认在训练时,负例数量是正例数量的三倍
    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  正负例比例: 负例数量/正例数量
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf  # 无穷
    # 两次sort 找出元素在排序中的位置
    _, indexes = loss.sort(dim=1, descending=True)  # descending 降序 ,返回 value,index
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg  # loss 降序排, 背景为-无穷, 选择排前num_neg的 负无穷,也就是 背景
    return pos_mask | neg_mask  # 目标 或 背景