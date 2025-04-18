import logging
import os
from functools import partial
import time

import cv2
import torch
import torch.optim as optim
import tqdm
import yaml
from torch.utils.data import DataLoader

from adversarial_trainer import GANFactory
from dataset import PairedDataset
from metric_counter import MetricCounter
from models.losses import get_loss
from models.models import get_model
from models.networks import get_nets
from schedulers import LinearDecay, WarmRestart
from fire import Fire
import torch.nn as nn
import numpy as np
import random

cv2.setNumThreads(0)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, config, train: DataLoader, val: DataLoader):
        self.config = config
        self.start_epoch = 0
        self.train_dataset = train
        self.val_dataset = val
        self.adv_lambda = config['model']['adv_lambda']
        self.metric_counter = MetricCounter(config['experiment_desc'])
        self.warmup_epochs = config['warmup_num']
        self.alpha = config['alpha']

    def train(self):
        self._init_params()
        if self.config['RESUME'] != '':
            print('########## resume training ##########')
            self.load_checkpoint(self.config['RESUME'])
            print('Start at %d epoch' % self.start_epoch)

        for epoch in range(self.start_epoch, self.config['num_epochs']):
            epoch_start_time = time.time()

            if (epoch == self.warmup_epochs) and not (self.warmup_epochs == 0):
                self.netG.module.unfreeze()
                self.optimizer_G = self._get_optim(self.netG.parameters())
                self.scheduler_G = self._get_scheduler(self.optimizer_G)

            self._run_epoch(epoch)
            self._validate(epoch)
            self.scheduler_G.step()
            self.scheduler_D.step()

            if self.metric_counter.update_best_model():
                torch.save({
                    'model': self.netG.state_dict(),
                    'model_D_patch': self.netD['patch'].state_dict(),
                    'model_D_full': self.netD['full'].state_dict(),
                    'epoch': epoch,
                    'optimizer_G': self.optimizer_G.state_dict(),
                    'scheduler_G': self.scheduler_G.state_dict(),
                    'optimizer_D': self.optimizer_D.state_dict(),
                    'scheduler_D': self.scheduler_D.state_dict(),
                    'optimizer_cla': self.cla_optimizer.state_dict(),
                    'scheduler_cla': self.cla_scheduler.state_dict()
                }, 'best_{}.h5'.format(self.config['experiment_desc']))
            if (epoch+1) % 100 == 0:
                torch.save({
                    'model': self.netG.state_dict(),
                    'model_D_patch': self.netD['patch'].state_dict(),
                    'model_D_full': self.netD['full'].state_dict(),
                    'epoch': epoch,
                    'optimizer_G': self.optimizer_G.state_dict(),
                    'scheduler_G': self.scheduler_G.state_dict(),
                    'optimizer_D': self.optimizer_D.state_dict(),
                    'scheduler_D': self.scheduler_D.state_dict(),
                    'optimizer_cla': self.cla_optimizer.state_dict(),
                    'scheduler_cla': self.cla_scheduler.state_dict()
                }, 'epoch{}_{}.h5'.format(epoch, self.config['experiment_desc']))
            torch.save({
                'model': self.netG.state_dict(),
                'model_D_patch': self.netD['patch'].state_dict(),
                'model_D_full': self.netD['full'].state_dict(),
                'epoch': epoch,
                'optimizer_G': self.optimizer_G.state_dict(),
                'scheduler_G': self.scheduler_G.state_dict(),
                'optimizer_D': self.optimizer_D.state_dict(),
                'scheduler_D': self.scheduler_D.state_dict(),
                'optimizer_cla': self.cla_optimizer.state_dict(),
                'scheduler_cla': self.cla_scheduler.state_dict()
            }, 'last_{}.h5'.format(self.config['experiment_desc']))
            print(self.metric_counter.loss_message())
            logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (
                self.config['experiment_desc'], epoch, self.metric_counter.loss_message()))

    def _run_epoch(self, epoch):
        self.metric_counter.clear()
        for param_group in self.optimizer_G.param_groups:
            lr = param_group['lr']

        epoch_size = self.config.get('train_batches_per_epoch') or len(self.train_dataset)
        tq = tqdm.tqdm(self.train_dataset, total=epoch_size)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        i = 0
        for data in tq:
            inputs, targets, target_class = self.model.get_input(data)
            outputs, logits, *_ = self.netG(inputs)

            # Classification
            cla_loss = self.cla_loss(logits, target_class)
            self.cla_optimizer.zero_grad()

            loss_D = self._update_d(outputs, targets)
            self.optimizer_G.zero_grad()
            loss_content = self.criterionG(outputs, targets)
            loss_adv = self.adv_trainer.loss_g(outputs, targets)
            loss_G = loss_content + self.adv_lambda * loss_adv + self.alpha * cla_loss

            loss_G.backward()
            self.cla_optimizer.step()
            self.optimizer_G.step()
            self.metric_counter.add_losses(loss_G.item(), loss_content.item(), loss_adv.item(), cla_loss.item(), loss_D)
            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)

            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            tq.set_postfix(loss=self.metric_counter.loss_message())
            if not i:
                self.metric_counter.add_image(img_for_vis, tag='train')
            i += 1
            if i > epoch_size:
                break

        tq.close()
        self.metric_counter.write_to_tensorboard(epoch)

    def _validate(self, epoch):
        self.metric_counter.clear()
        epoch_size = self.config.get('val_batches_per_epoch') or len(self.val_dataset)
        tq = tqdm.tqdm(self.val_dataset, total=epoch_size)
        tq.set_description('Validation')
        i = 0
        for data in tq:
            inputs, targets, target_class = self.model.get_input(data)
            with torch.no_grad():
                outputs, logits, *_ = self.netG(inputs)

                cla_loss = self.cla_loss(logits, target_class)

                loss_content = self.criterionG(outputs, targets)
                loss_adv = self.adv_trainer.loss_g(outputs, targets)
            loss_G = loss_content + self.adv_lambda * loss_adv + self.alpha * cla_loss

            self.metric_counter.add_losses(loss_G.item(), loss_content.item(), loss_adv, cla_loss.item())
            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            if not i:
                self.metric_counter.add_image(img_for_vis, tag='val')
            i += 1
            if i > epoch_size:
                break

        tq.close()
        self.metric_counter.write_to_tensorboard(epoch, validation=True)

    def load_checkpoint(self, weight_path):
        checkpoint = torch.load(weight_path)
        self.start_epoch = checkpoint['epoch'] + 1

        if self.start_epoch >= self.warmup_epochs:
            self.netG.module.unfreeze()
            self.optimizer_G = self._get_optim(self.netG.parameters())
            self.scheduler_G = self._get_scheduler(self.optimizer_G)

        self.netG.load_state_dict(checkpoint['model'])
        self.netD['patch'].load_state_dict(checkpoint['model_D_patch'])
        self.netD['full'].load_state_dict(checkpoint['model_D_full'])

        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D'])
        self.cla_optimizer.load_state_dict(checkpoint['optimizer_cla'])
        self.cla_scheduler.load_state_dict(checkpoint['scheduler_cla'])

    def _update_d(self, outputs, targets):
        if self.config['model']['d_name'] == 'no_gan':
            return 0
        self.optimizer_D.zero_grad()
        loss_D = self.adv_lambda * self.adv_trainer.loss_d(outputs, targets)
        loss_D.backward(retain_graph=True)
        self.optimizer_D.step()
        return loss_D.item()

    def _get_optim(self, params):
        if self.config['optimizer']['name'] == 'adam':
            optimizer = optim.Adam(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'sgd':
            optimizer = optim.SGD(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'adadelta':
            optimizer = optim.Adadelta(params, lr=self.config['optimizer']['lr'])
        else:
            raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
        return optimizer

    def _get_optim_d(self, params):
        if self.config['optimizer']['name'] == 'adam':
            optimizer = optim.Adam(params, lr=self.config['optimizer']['lr_D'])
        elif self.config['optimizer']['name'] == 'sgd':
            optimizer = optim.SGD(params, lr=self.config['optimizer']['lr_D'])
        elif self.config['optimizer']['name'] == 'adadelta':
            optimizer = optim.Adadelta(params, lr=self.config['optimizer']['lr_D'])
        else:
            raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
        return optimizer

    def _get_scheduler(self, optimizer):
        if self.config['scheduler']['name'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='min',
                                                             patience=self.config['scheduler']['patience'],
                                                             factor=self.config['scheduler']['factor'],
                                                             min_lr=self.config['scheduler']['min_lr'])
        elif self.config['optimizer']['name'] == 'sgdr':
            scheduler = WarmRestart(optimizer)
        elif self.config['scheduler']['name'] == 'linear':
            scheduler = LinearDecay(optimizer,
                                    min_lr=self.config['scheduler']['min_lr'],
                                    num_epochs=self.config['num_epochs'],
                                    start_epoch=self.config['scheduler']['start_epoch'])
        else:
            raise ValueError("Scheduler [%s] not recognized." % self.config['scheduler']['name'])
        return scheduler

    @staticmethod
    def _get_adversarial_trainer(d_name, net_d, criterion_d):
        if d_name == 'no_gan':
            return GANFactory.create_model('NoGAN')
        elif d_name == 'patch_gan' or d_name == 'multi_scale':
            return GANFactory.create_model('SingleGAN', net_d, criterion_d)
        elif d_name == 'double_gan':
            return GANFactory.create_model('DoubleGAN', net_d, criterion_d)
        else:
            raise ValueError("Discriminator Network [%s] not recognized." % d_name)

    def _init_params(self):
        self.criterionG, criterionD = get_loss(self.config['model'])
        self.netG, self.netD = get_nets(self.config['model'])
        self.netG.cuda()
        self.adv_trainer = self._get_adversarial_trainer(self.config['model']['d_name'], self.netD, criterionD)
        self.model = get_model(self.config['model'])
        self.optimizer_G = self._get_optim(filter(lambda p: p.requires_grad, self.netG.parameters()))
        self.optimizer_D = self._get_optim_d(self.adv_trainer.get_params())
        self.scheduler_G = self._get_scheduler(self.optimizer_G)
        self.scheduler_D = self._get_scheduler(self.optimizer_D)
        self.cla_loss = nn.CrossEntropyLoss()
        self.cla_optimizer = optim.Adam(self.netG.module.fpn.inception.parameters(), lr=self.config['optimizer']['lr'])
        self.cla_scheduler = self._get_scheduler(self.cla_optimizer)


def main(config_path='config/config.yaml'):
    with open(config_path, 'r',encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    batch_size = config.pop('batch_size')
    get_dataloader = partial(DataLoader,
                             batch_size=batch_size,
                             shuffle=True, drop_last=True, num_workers=2)

    datasets = map(config.pop, ('train', 'val'))
    datasets = map(PairedDataset.from_config, datasets)
    train, val = map(get_dataloader, datasets)
    trainer = Trainer(config, train=train, val=val)
    trainer.train()


if __name__ == '__main__':
    Fire(main)
