# -*- coding: utf-8 -*-
# @Author  : LG

from Model import SSD, Evaler
from Data import VOCDataset
from Configs import _C as cfg
from Data import SSDTramsfrom,SSDTargetTransform


# 训练数据集,VOC格式数据集, 训练数据取自 ImageSets/Main/train.txt'
train_dataset=VOCDataset(cfg, is_train=True, transform=SSDTramsfrom(cfg,is_train=True),
                         target_transform=SSDTargetTransform(cfg))

# 测试数据集,VOC格式数据集, 测试数据取自 ImageSets/Main/eval.txt'
test_dataset = VOCDataset(cfg=cfg, is_train=False,
                          transform=SSDTramsfrom(cfg=cfg, is_train=False),
                          target_transform=SSDTargetTransform(cfg))

if __name__ == '__main__':
    # 模型测试只支持GPU单卡或多卡,不支持cpu
    net = SSD(cfg)
    # 将模型移动到gpu上,cfg.DEVICE.MAINDEVICE定义了模型所使用的主GPU
    net.to(cfg.DEVICE.MAINDEVICE)
    # 模型从权重文件中加载权重
    net.load_pretrained_weight('Weights/pretrained/vgg_ssd300_voc0712.pkl')
    # 初始化验证器,验证器参数通过cfg进行配置;也可传入参数进行配置,但不建议
    evaler = Evaler(cfg, eval_devices=None)
    # 验证器开始在数据集上验证模型
    ap, map = evaler(model=net,
                     test_dataset=test_dataset)
    print(ap)
    print(map)