# -*- coding: utf-8 -*-
# @Author  : LG
import os
import torch.utils.data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

__all__ = ['VOCDataset']

class VOCDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, is_train=True, data_dir=None, transform=None, target_transform=None, keep_difficult=False):
        """VOC格式数据集
        Args:
            data_dir: VOC格式数据集根目录，该目录下包含：
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
            split： train、test 或者 eval， 对应于 ImageSets/Main/train.txt,eval.txt
        """
        # 类别
        self.class_names = cfg.DATA.DATASET.CLASS_NAME
        self.data_dir = cfg.DATA.DATASET.DATA_DIR
        self.is_train = is_train
        if data_dir:
            self.data_dir = data_dir
        self.split = cfg.DATA.DATASET.TRAIN_SPLIT       # train     对应于ImageSets/Main/train.txt
        if not self.is_train:
            self.split = cfg.DATA.DATASET.TEST_SPLIT    # test      对应于ImageSets/Main/test.txt
        self.transform = transform
        self.target_transform = target_transform
        image_sets_file = os.path.join(self.data_dir, "ImageSets", "Main", "{}.txt".format(self.split))
        # 从train.txt 文件中读取图片 id 返回ids列表
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_name = self.ids[index]
        # 解析Annotations/id.xml 读取id图片对应的 boxes, labels, is_difficult 均为列表
        boxes, labels, is_difficult = self._get_annotation(image_name)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        # 读取 JPEGImages/id.jpg 返回Image.Image
        image = self._read_image(image_name)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        return image, boxes, labels, image_name

    # 返回 id, boxes， labels， is_difficult
    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    # 解析xml，返回 boxes， labels， is_difficult   numpy.array格式
    def _get_annotation(self, image_name):
        annotation_file = os.path.join(self.data_dir, "Annotations", "{}.xml".format(image_name))
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects: # .encode('utf-8').decode('UTF-8-sig') 解决Windows下中文编码问题
            class_name = obj.find('name').text.encode('utf-8').decode('UTF-8-sig').lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text.encode('utf-8').decode('UTF-8-sig')) - 1
            y1 = float(bbox.find('ymin').text.encode('utf-8').decode('UTF-8-sig')) - 1
            x2 = float(bbox.find('xmax').text.encode('utf-8').decode('UTF-8-sig')) - 1
            y2 = float(bbox.find('ymax').text.encode('utf-8').decode('UTF-8-sig')) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    # 获取图片尺寸信息，返回字典 {'height': , 'width': }
    def get_img_size(self, img_name):
        annotation_file = os.path.join(self.data_dir, "Annotations", "{}.xml".format(img_name))
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    # 读取图片数据，返回Image.Image
    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "JPEGImages", "{}.jpg".format(image_id))
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image

    def get_one_image(self,image_name = None):
        import random

        if not image_name:
            image_name = random.choice(self.ids)
        # 解析Annotations/id.xml 读取id图片对应的 boxes, labels, is_difficult 均为列表
        boxes, labels, is_difficult = self._get_annotation(image_name)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        # 读取 JPEGImages/id.jpg 返回Image.Image
        image = self._read_image(image_name)
        image_after_transfrom = None
        if self.transform:
            image_after_transfrom, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        return image, image_after_transfrom, boxes, labels, image_name