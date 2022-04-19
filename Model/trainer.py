# -*- coding: utf-8 -*-
# @Author  : LG
import torch
from torch.optim.lr_scheduler import MultiStepLR
from Data import Our_Dataloader
from .structs import multiboxloss
from Utils.visdom_op import visdom_line, setup_visdom, visdom_bar
from torch import nn
from torch.nn import DataParallel
import os

__all__ = ['Trainer']

class Trainer(object):
    """
    模型训练器,不指定参数时,均默认使用Configs中配置的参数
    *** 推荐使用Configs文件管理参数, 不推荐在函数中进行参数指定, 只是为了扩展  ***
    *** 默认使用 SGD 优化器, 如需使用其他优化器, 继承该类,对build_optimizer方法进行重写即可***

    模型在训练时,会使用DataParallel进行包装,以便于在多GPU上进行训练
    本训练器只支持GPU训练,单机单卡与单机单卡均可,但不支持cpu,不支持多机多卡(别问为啥不支持多机多卡.穷!!!)

    eg:
        trainer = Trainer(cfg)          # 实例化训练器
        trainer(net,train_dataset)      # 在train_dataset数据集上训练模型net
    """
    def __init__(self, cfg, max_iter=None, batch_size=None, num_workers = None, train_devices=None,
                 model_save_step=None, model_save_root=None, vis = None, vis_step=None):
        """
        训练器初始化
        值为None的参数项不指定时为默认,已在配置文件中设置.
        如需更改参数建议在Configs配置文件中进行更改
        不建议直接指定参数,只留做扩展用.

        :param cfg:             配置
        :param max_iter:        最大训练轮数
        :param batch_size:      批次数,
        :param train_devices:   训练设备,列表,eg:[0,1],使用0,1俩个GPU,这里0,1为gpu编号,可用nvidia-smi查看.,不指定时为默认,已在配置文件中设置
        :param vis:             visdom.Visdom(),用于训练过程可视化.绘制损失曲线已经学习率
        :param model_save_step: 模型保存步长
        :param vis_step:        visdom可视化步长
        """
        self.cfg = cfg

        self.iterations = self.cfg.TRAIN.MAX_ITER
        if max_iter:
            self.iterations = max_iter

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        if batch_size:
            self.batch_size = batch_size

        self.num_workers = cfg.TRAIN.NUM_WORKERS
        if num_workers:
            self.num_workers = num_workers

        self.train_devices = cfg.DEVICE.TRAIN_DEVICES
        if train_devices:
            self.train_devices = train_devices

        self.model_save_root = cfg.FILE.MODEL_SAVE_ROOT
        if model_save_root:
            self.model_save_root = model_save_root

        if not os.path.exists(self.model_save_root):
            os.mkdir(self.model_save_root)
        self.model_save_step = self.cfg.STEP.MODEL_SAVE_STEP
        if model_save_step:
            self.model_save_step = model_save_step

        self.vis = setup_visdom()
        if vis:
            self.vis = vis
        self.vis_step = self.cfg.STEP.VIS_STEP
        if vis_step:
            self.vis_step = vis_step

        self.model = None
        self.loss_func = None
        self.optimizer = None
        self.scheduler = None

    def __call__(self, model, dataset):
        """
        训练器使用, 传入 模型 与数据集.
        :param model:
        :param dataset:
        :return:
        """
        if not isinstance(model, nn.DataParallel):
            # raise TypeError('请用 DataParallel 包装模型. eg: model = DataParallel(model, device_ids=[0,1,2]),使用device_ids指定需要使用的gpu')
            model = DataParallel(model, device_ids=self.train_devices)
        self.model = model
        data_loader = Our_Dataloader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        print(' Max_iter = {}, Batch_size = {}'.format(self.iterations, self.batch_size))
        print(' Model will train on cuda:{}'.format(self.train_devices))

        num_gpu_use = len(self.train_devices)
        if (self.batch_size % num_gpu_use) != 0:
            raise ValueError(
                'You use {} gpu to train , but set batch_size={}'.format(num_gpu_use, data_loader.batch_size))

        self.set_lossfunc()
        self.set_optimizer()
        self.set_scheduler()

        print("Set optimizer : {}".format(self.optimizer))
        print("Set scheduler : {}".format(self.scheduler))
        print("Set lossfunc : {}".format(self.loss_func))


        print(' Start Train......')
        print(' -------' * 20)

        for iteration, (images, boxes, labels, image_names) in enumerate(data_loader):
            iteration+=1
            boxes, labels = boxes.to('cuda'), labels.to('cuda')
            cls_logits, bbox_preds = self.model(images)
            reg_loss, cls_loss = self.loss_func(cls_logits, bbox_preds, labels, boxes)

            reg_loss = reg_loss.mean()
            cls_loss = cls_loss.mean()
            loss = reg_loss + cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']

            if iteration % 10 == 0:
                print('Iter : {}/{} | Lr : {} | Loss : {:.4f} | cls_loss : {:.4f} | reg_loss : {:.4f}'.format(iteration, self.iterations, lr, loss.item(), cls_loss.item(), reg_loss.item()))

            if self.vis and iteration % self.vis_step == 0:
                visdom_line(self.vis, y=[loss], x=iteration, win_name='loss')
                visdom_line(self.vis, y=[reg_loss], x=iteration, win_name='reg_loss')
                visdom_line(self.vis, y=[cls_loss], x=iteration, win_name='cls_loss')
                visdom_line(self.vis, y=[lr], x=iteration, win_name='lr')

            if iteration % self.model_save_step == 0:
                torch.save(model.module.state_dict(), '{}/model_{}.pkl'.format(self.model_save_root, iteration))

            if iteration == self.iterations:
                return True
        return True

    def set_optimizer(self, lr=None, momentum=None, weight_decay=None):
        """
        配置优化器
        :param lr:              初始学习率,  默认0.001
        :param momentum:        动量, 默认 0.9
        :param weight_decay:    权重衰减,L2, 默认 5e-4
        :return:
        """
        if not lr:
            lr= self.cfg.OPTIM.LR
        if not momentum:
            momentum = self.cfg.OPTIM.MOMENTUM
        if not weight_decay:
            weight_decay = self.cfg.OPTIM.WEIGHT_DECAY

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=lr,
                                         momentum=momentum,
                                         weight_decay=weight_decay)

    def set_lossfunc(self, neg_pos_ratio=None):
        """
        配置损失函数
        :param neg_pos_ratio:   负正例 比例,默认3, 负例数量是正例的三倍
        :return:
        """
        if not neg_pos_ratio:
            neg_pos_ratio = self.cfg.TRAIN.NEG_POS_RATIO
        self.loss_func = multiboxloss(neg_pos_ratio=neg_pos_ratio)
        # print(' Trainer set loss_func : {}, neg_pos_ratio = {}'.format('multiboxloss', neg_pos_ratio))

    def set_scheduler(self, lr_steps=None, gamma=None):
        """
        配置学习率衰减策略
        :param lr_steps:    默认 [80000, 10000],当训练到这些轮次时,学习率*gamma
        :param gamma:       默认 0.1,学习率下降10倍
        :return:
        """
        if not lr_steps:
            lr_steps = self.cfg.OPTIM.SCHEDULER.LR_STEPS
        if not gamma:
            gamma = self.cfg.OPTIM.SCHEDULER.GAMMA
        self.scheduler = MultiStepLR(optimizer=self.optimizer,
                                     milestones=lr_steps,
                                     gamma=gamma)
        # print(' Trainer set scheduler : {}, lr_steps={}, gamma={}'.format('MultiStepLR', lr_steps, gamma))
