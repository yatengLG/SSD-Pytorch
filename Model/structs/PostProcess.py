# -*- coding: utf-8 -*-
# @Author  : LG
import torch
import torchvision
from Utils.Boxs_op import center_form_to_corner_form, convert_locations_to_boxes
from .Anchors import priorbox
import torch.nn.functional as F

__all__ = ['postprocessor']

class postprocessor:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.width = cfg.MODEL.INPUT.IMAGE_SIZE
        self.height = cfg.MODEL.INPUT.IMAGE_SIZE

    def __call__(self, cls_logits, bbox_pred):
        priors = priorbox(self.cfg)().to(self.cfg.DEVICE.MAINDEVICE)
        batches_scores = F.softmax(cls_logits, dim=2)
        boxes = convert_locations_to_boxes(
            bbox_pred, priors, self.cfg.MODEL.ANCHORS.CENTER_VARIANCE, self.cfg.MODEL.ANCHORS.CENTER_VARIANCE
        )
        batches_boxes = center_form_to_corner_form(boxes)

        device = batches_scores.device
        batch_size = batches_scores.size(0)
        results = []
        for batch_id in range(batch_size):
            processed_boxes = []
            processed_scores = []
            processed_labels = []

            per_img_scores, per_img_boxes = batches_scores[batch_id], batches_boxes[batch_id]  # (N, #CLS) (N, 4)
            for class_id in range(1, per_img_scores.size(1)):  # skip background
                scores = per_img_scores[:, class_id]
                mask = scores > self.cfg.MODEL.TEST.CONFIDENCE_THRESHOLD
                scores = scores[mask]
                if scores.size(0) == 0:
                    continue
                boxes = per_img_boxes[mask, :]
                boxes[:, 0::2] *= self.width
                boxes[:, 1::2] *= self.height

                keep = boxes_nms(boxes, scores, self.cfg.MODEL.TEST.NMS_THRESHOLD, self.cfg.MODEL.TEST.MAX_PER_CLASS)

                nmsed_boxes = boxes[keep, :]
                nmsed_labels = torch.tensor([class_id] * keep.size(0), device=device)
                nmsed_scores = scores[keep]

                processed_boxes.append(nmsed_boxes)
                processed_scores.append(nmsed_scores)
                processed_labels.append(nmsed_labels)

            if len(processed_boxes) == 0:
                processed_boxes = torch.empty(0, 4)
                processed_labels = torch.empty(0)
                processed_scores = torch.empty(0)
            else:
                processed_boxes = torch.cat(processed_boxes, 0)
                processed_labels = torch.cat(processed_labels, 0)
                processed_scores = torch.cat(processed_scores, 0)

            if processed_boxes.size(0) > self.cfg.MODEL.TEST.MAX_PER_IMAGE > 0:
                processed_scores, keep = torch.topk(processed_scores, k=self.cfg.MODEL.TEST.MAX_PER_IMAGE)
                processed_boxes = processed_boxes[keep, :]
                processed_labels = processed_labels[keep]
            results.append([processed_boxes, processed_labels, processed_scores])
        return results

def boxes_nms(boxes, scores, nms_thresh, max_count=-1):
    """ Performs non-maximum suppression, run on GPU or CPU according to
    boxes's device.
    Args:
        boxes(Tensor): `xyxy` mode boxes, use absolute coordinates(or relative coordinates), shape is (n, 4)
        scores(Tensor): scores, shape is (n, )
        nms_thresh(float): thresh
        max_count (int): if > 0, then only the top max_proposals are kept  after non-maximum suppression
    Returns:
        indices kept.
    """
    keep = torchvision.ops.nms(boxes, scores, nms_thresh)
    if max_count > 0:
        keep = keep[:max_count]
    return keep

