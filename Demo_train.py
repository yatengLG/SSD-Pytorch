# -*- coding: utf-8 -*-
# @Author  : LG

from Model import SSD, Trainer
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
    """
    使用时,请先打开visdom
    
    命令行 输入  pip install visdom          进行安装 
    输入        python -m visdom.server'    启动
    """

    # 实例化模型. 模型的具体各种参数在Config文件中进行配置
    net = SSD(cfg)
    # 将模型移动到gpu上,cfg.DEVICE.MAINDEVICE定义了模型所使用的主GPU
    net.to(cfg.DEVICE.MAINDEVICE)

    # 初始化训练器,训练器参数通过cfg进行配置;也可传入参数进行配置,但不建议
    trainer = Trainer(cfg)

    print(trainer.optimizer)
    # 训练器开始在 数据集上训练模型
    trainer(net, train_dataset)