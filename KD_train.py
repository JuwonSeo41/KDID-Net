import logging
import os
from functools import partial
import time
import kd_losses

import cv2
import pandas as pd
import torch
import torch.optim as optim
import tqdm
import yaml
from joblib import cpu_count
from torch.utils.data import DataLoader

from adversarial_trainer import GANFactory
from dataset import PairedDataset
from KD_metric_counter import MetricCounter
from models.losses import get_loss
from models.models import get_model
from models.networks import get_nets
from models.networks_small import get_nets as get_nets_small
from schedulers import LinearDecay, WarmRestart
from fire import Fire
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import random
import warnings


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
        # self.alpha = config['alpha']

        self.start_time = time.time()
        self.epoch_times = []

    def train(self):
        self._init_params()

        # MLP
        # self.MLP1 = kd_losses.MLP(32, 32).to('cuda')
        # self.MLP2 = kd_losses.MLP(192, 192).to('cuda')
        # self.MLP3 = kd_losses.MLP(64, 64).to('cuda')
        # self.optimizer_G_small.add_param_group({'params': [*self.MLP1.parameters(), *self.MLP2.parameters(),
        #                                                    *self.MLP3.parameters()]})

        # AttnFD
        # self.Attn1 = kd_losses.AttnFD(32).to('cuda')
        # self.Attn2 = kd_losses.AttnFD(192).to('cuda')
        # self.Attn3 = kd_losses.AttnFD(64).to('cuda')
        # self.optimizer_G_small.add_param_group({'params': [*self.Attn1.parameters(), *self.Attn2.parameters(),
        #                                                    *self.Attn3.parameters()]})

        # Teacher weight(pretrained)
        weight_path = self.config['pre-weight']
        pretrained = torch.load(weight_path)
        self.netG.module.unfreeze()
        self.netG.load_state_dict(pretrained['model'])

        if self.config['RESUME'] != '':
            print('########## resume training ##########')
            self.load_checkpoint(self.config['RESUME'])
            print('Start at %d epoch' % self.start_epoch)

        for epoch in range(self.start_epoch, self.config['num_epochs']):
            epoch_start_time = time.time()

            self._run_epoch(epoch)
            self._validate(epoch)
            # self.scheduler_G.step()
            # self.scheduler_D.step()
            self.scheduler_G_small.step()
            self.scheduler_D_small.step()
            # self.cla_scheduler.step()

            if self.metric_counter.update_best_model():
                torch.save({
                    'model': self.netG_small.state_dict(),
                    'model_D_patch': self.netD_small['patch'].state_dict(),
                    'model_D_full': self.netD_small['full'].state_dict(),
                    'epoch': epoch,
                    'optimizer_G': self.optimizer_G_small.state_dict(),
                    'scheduler_G': self.scheduler_G_small.state_dict(),
                    'optimizer_D': self.optimizer_D_small.state_dict(),
                    'scheduler_D': self.scheduler_D_small.state_dict()
                }, 'best_{}_small.h5'.format(self.config['experiment_desc']))
            if (epoch+1) % 100 == 0:
                torch.save({
                    'model': self.netG_small.state_dict(),
                    'model_D_patch': self.netD_small['patch'].state_dict(),
                    'model_D_full': self.netD_small['full'].state_dict(),
                    'epoch': epoch,
                    'optimizer_G': self.optimizer_G_small.state_dict(),
                    'scheduler_G': self.scheduler_G_small.state_dict(),
                    'optimizer_D': self.optimizer_D_small.state_dict(),
                    'scheduler_D': self.scheduler_D_small.state_dict()
                }, 'epoch_{}_small.h5'.format(epoch, self.config['experiment_desc']))
            torch.save({
                'model': self.netG_small.state_dict(),
                'model_D_patch': self.netD_small['patch'].state_dict(),
                'model_D_full': self.netD_small['full'].state_dict(),
                'epoch': epoch,
                'optimizer_G': self.optimizer_G_small.state_dict(),
                'scheduler_G': self.scheduler_G_small.state_dict(),
                'optimizer_D': self.optimizer_D_small.state_dict(),
                'scheduler_D': self.scheduler_D_small.state_dict()
            }, 'last_{}_small.h5'.format(self.config['experiment_desc']))
            print(self.metric_counter.loss_message())
            logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (
                self.config['experiment_desc'], epoch, self.metric_counter.loss_message()))

            epoch_duration = time.time() - epoch_start_time
            self.epoch_times.append(epoch_duration)
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = self.config['num_epochs'] - (epoch + 1)
            estimated_total_time = avg_epoch_time * self.config['num_epochs']
            estimated_remaining_time = avg_epoch_time * remaining_epochs

            print(f"Estimated Total Time: {estimated_total_time:.2f} seconds")
            print(f"Estimated Time Remaining: {estimated_remaining_time:.2f} seconds")

    def _run_epoch(self, epoch):
        self.metric_counter.clear()
        for param_group in self.optimizer_G_small.param_groups:
            lr = param_group['lr']

        epoch_size = self.config.get('train_batches_per_epoch') or len(self.train_dataset)
        tq = tqdm.tqdm(self.train_dataset, total=epoch_size)
        # tq = tqdm.tqdm(self.train_dataset)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        i = 0

        # MLP
        # self.MLP1.train()
        # self.MLP2.train()
        # self.MLP3.train()

        # AttnKD
        # self.Attn1.train()
        # self.Attn2.train()
        # self.Attn3.train()

        for data in tq:
            inputs, targets, target_class = self.model.get_input(data)
            with torch.no_grad():
                outputs, logits, feature_first, feature_middle, feature_last = self.netG(inputs)
            outputs_small, feature_small_first, feature_small_middle, feature_small_last = self.netG_small(inputs)

            loss_D_small = self._update_d(outputs_small, targets)
            self.optimizer_G_small.zero_grad()

            loss_content_small = self.criterionG(outputs_small, targets)
            loss_adv_small = self.adv_trainer_small.loss_g(outputs_small, targets)


            # FTKD
            feature_last = torch.fft.rfft2(feature_last)
            feature_small_last = torch.fft.rfft2(feature_small_last)
            outputs = torch.fft.rfft2(outputs)
            outputs_small_fft = torch.fft.rfft2(outputs_small)

            KD_loss = torch.nn.functional.l1_loss(outputs_small_fft, outputs)

            KD_feature_loss_first = torch.nn.functional.l1_loss(feature_first, feature_small_first)
            KD_feature_loss_middle = torch.nn.functional.l1_loss(feature_middle, feature_small_middle)
            KD_feature_loss_last = torch.nn.functional.l1_loss(feature_last, feature_small_last)

            KD_feature_loss = 0.1 * KD_feature_loss_first + 0.05 * KD_feature_loss_middle + KD_feature_loss_last          # case1 0.1 0.05
            # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last  # case2 no weight


            # FTKD_1
            # _, _, H, W = outputs_small.shape
            #
            # def FFT(x):
            #     x = torch.fft.fft2(x)
            #     x = torch.fft.fftshift(x)
            #     magnitude = torch.abs(x).squeeze()
            #     return magnitude
            #
            # def cal_distance(x):
            #     _, _, H, W = x.shape
            #     y, x = torch.meshgrid(torch.arange(H), torch.arange(W))  # 좌표 행렬 생성
            #     center_y, center_x = H // 2, W // 2
            #     distance = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            #     return distance  # 각 좌표의 distance 행렬
            #
            # def extract_high_freq(x, threshold_ratio=0.1):
            #     distance = cal_distance(x)
            #     max_distance = distance.max()
            #     threshold = max_distance * threshold_ratio
            #     high_freq_mask = distance > threshold  # threshold 보다 크면 True, 아니면 False
            #     x_high = x * high_freq_mask.cuda()  # magnitude * high_freq_mask
            #     return x_high
            #
            # feature_last = FFT(feature_last)
            # feature_small_last = FFT(feature_small_last)
            # outputs = FFT(outputs)
            # outputs_small_FFT = FFT(outputs_small)
            #
            # feature_last = extract_high_freq(feature_last)  # extract high freq
            # feature_small_last = extract_high_freq(feature_small_last)
            # outputs = extract_high_freq(outputs)
            # outputs_small_high = extract_high_freq(outputs_small_FFT)
            #
            # KD_loss = torch.nn.functional.l1_loss(outputs_small_high, outputs)
            #
            # KD_feature_loss_first = torch.nn.functional.l1_loss(feature_first, feature_small_first)
            # KD_feature_loss_middle = torch.nn.functional.l1_loss(feature_middle, feature_small_middle)
            # KD_feature_loss_last = torch.nn.functional.l1_loss(feature_last, feature_small_last)
            # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + 0.2 * KD_feature_loss_last
            #
            # KD_total_loss = 0.5 * KD_loss + KD_feature_loss
            # KD_total_loss = 0.1 * KD_total_loss


            # FTKD_2
            # def FFT(x):
            #     x = torch.fft.fft2(x)
            #     x = torch.fft.fftshift(x)
            #     magnitude = torch.abs(x)
            #     return magnitude
            #
            # feature_last = FFT(feature_last)
            # feature_small_last = FFT(feature_small_last)
            # outputs = FFT(outputs)
            # outputs_small_mag = FFT(outputs_small)
            #
            # KD_loss = torch.nn.functional.l1_loss(outputs_small_mag, outputs)
            #
            # KD_feature_loss_first = torch.nn.functional.l1_loss(feature_first, feature_small_first)
            # KD_feature_loss_middle = torch.nn.functional.l1_loss(feature_middle, feature_small_middle)
            # KD_feature_loss_last = torch.nn.functional.l1_loss(feature_last, feature_small_last)
            # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last


            # FTKD_3   phase, mag 분리해서
            # def FFT_mag(x):
            #     x = torch.fft.fft2(x)
            #     x = torch.fft.fftshift(x)
            #     magnitude = torch.abs(x)
            #     return magnitude
            #
            # def FFT_phase(x):
            #     x = torch.fft.fft2(x)
            #     phase = torch.angle(x)
            #     zero_mag = torch.ones_like(x)
            #     phase_img = zero_mag * torch.exp(1j * phase)
            #     phase_img = torch.fft.ifft2(phase_img).real
            #     return phase_img
            #
            # feature_last_mag = FFT_mag(feature_last)        # mag
            # feature_small_last_mag = FFT_mag(feature_small_last)
            # outputs_mag = FFT_mag(outputs)
            # outputs_small_mag = FFT_mag(outputs_small)
            #
            # feature_last_phase = FFT_phase(feature_last)        # phase
            # feature_small_last_phase = FFT_phase(feature_small_last)
            # outputs_phase = FFT_phase(outputs)
            # outputs_small_phase = FFT_phase(outputs_small)
            #
            # KD_loss_mag = torch.nn.functional.l1_loss(outputs_small_mag, outputs_mag)       # output 비교
            # KD_loss_phase = torch.nn.functional.l1_loss(outputs_small_phase, outputs_phase)
            # KD_loss = KD_loss_mag + KD_loss_phase
            #
            # KD_feature_loss_first = torch.nn.functional.l1_loss(feature_first, feature_small_first)
            # KD_feature_loss_middle = torch.nn.functional.l1_loss(feature_middle, feature_small_middle)
            #
            # KD_feature_loss_last_mag = torch.nn.functional.l1_loss(feature_last_mag, feature_small_last_mag)        # feature last 비교
            # KD_feature_loss_last_phase = torch.nn.functional.l1_loss(feature_last_phase, feature_small_last_phase)
            # KD_feature_loss_last = KD_feature_loss_last_mag + KD_feature_loss_last_phase
            #
            # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last

            # FFT 모두 적용
            # feature_first = torch.fft.rfft2(feature_first)
            # feature_small_first = torch.fft.rfft2(feature_small_first)
            # feature_middle = torch.fft.rfft2(feature_middle)
            # feature_small_middle = torch.fft.rfft2(feature_small_middle)
            # feature_last = torch.fft.rfft2(feature_last)
            # feature_small_last = torch.fft.rfft2(feature_small_last)
            # outputs = torch.fft.rfft2(outputs)
            # outputs_small_FFT = torch.fft.rfft2(outputs_small)
            #
            # KD_loss = torch.nn.functional.l1_loss(outputs_small_FFT, outputs)
            #
            # KD_feature_loss_first = torch.nn.functional.l1_loss(feature_first, feature_small_first)
            # KD_feature_loss_middle = torch.nn.functional.l1_loss(feature_middle, feature_small_middle)
            # KD_feature_loss_last = torch.nn.functional.l1_loss(feature_last, feature_small_last)
            #
            # KD_feature_loss = 0.1 * KD_feature_loss_first + 0.05 * KD_feature_loss_middle + KD_feature_loss_last

            # FFT 미적용
            # KD_loss = torch.nn.functional.l1_loss(outputs_small, outputs)
            #
            # KD_feature_loss_first = torch.nn.functional.l1_loss(feature_first, feature_small_first)
            # KD_feature_loss_middle = torch.nn.functional.l1_loss(feature_middle, feature_small_middle)
            # KD_feature_loss_last = torch.nn.functional.l1_loss(feature_last, feature_small_last)
            #
            # KD_feature_loss = 0.1 * KD_feature_loss_first + 0.05 * KD_feature_loss_middle + KD_feature_loss_last


            # Logits
            # Logits = kd_losses.logits.Logits()
            # KD_loss = Logits(outputs_small, outputs)
            #
            # # Total loss
            # loss_G_small = loss_content_small + self.adv_lambda * loss_adv_small + KD_loss
            #
            # self.metric_counter.add_losses(loss_G_small.item(), loss_content_small.item(), loss_adv_small.item(),
            #                                KD_loss.item(), 0, 0, 0, loss_D_small)


            # FitNet
            # KD_loss = F.mse_loss(outputs_small, outputs)
            #
            # Hint = kd_losses.fitnet.Hint()
            # KD_feature_loss_first = Hint(feature_small_first, feature_first)
            # KD_feature_loss_middle = Hint(feature_small_middle, feature_middle)
            # KD_feature_loss_last = Hint(feature_small_last, feature_last)
            #
            # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last


            # Attention Transfer
            # KD_loss = F.mse_loss(outputs_small, outputs)
            #
            # AT = kd_losses.at.AT(2)
            # KD_feature_loss_first = AT(feature_small_first, feature_first)
            # KD_feature_loss_middle = AT(feature_small_middle, feature_middle)
            # KD_feature_loss_last = AT(feature_small_last, feature_last)
            #
            # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last


            # Similarity Preserving
            # KD_loss = F.mse_loss(outputs_small, outputs)
            #
            # SP = kd_losses.sp.SP()
            # KD_feature_loss_first = SP(feature_small_first, feature_first)
            # KD_feature_loss_middle = SP(feature_small_middle, feature_middle)
            # KD_feature_loss_last = SP(feature_small_last, feature_last)
            #
            # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last


            # Simple Recipe Distillation
            # KD_loss = F.mse_loss(outputs_small, outputs) * 10
            #
            # SRD1 = kd_losses.SRD.DistillationLoss(32, 32)
            # SRD2 = kd_losses.SRD.DistillationLoss(192, 192)
            # SRD3 = kd_losses.SRD.DistillationLoss(64, 64)
            # KD_feature_loss_first = SRD1(feature_small_first, feature_first)
            # KD_feature_loss_middle = SRD2(feature_small_middle, feature_middle)
            # KD_feature_loss_last = SRD3(feature_small_last, feature_last)
            #
            # KD_feature_loss = 0.1 * (KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last)

            # CWD
            # KD_loss = F.mse_loss(outputs_small, outputs)

            # cwd = kd_losses.CWD.CriterionCWD()
            # KD_feature_loss_first = cwd(feature_small_first, feature_first)
            # KD_feature_loss_middle = cwd(feature_small_middle, feature_middle)
            # KD_feature_loss_last = cwd(feature_small_last, feature_last)

            # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last


            # MLP
            # KD_loss = F.mse_loss(outputs_small, outputs)
            #
            # KD_feature_loss_first = self.MLP1(feature_small_first, feature_first)
            # KD_feature_loss_middle = self.MLP2(feature_small_middle, feature_middle)
            # KD_feature_loss_last = self.MLP3(feature_small_last, feature_last)
            #
            # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last

            # AttnFD
            # KD_loss = F.mse_loss(outputs_small, outputs)
            #
            # KD_feature_loss_first = self.Attn1(feature_small_first, feature_first)
            # KD_feature_loss_middle = self.Attn2(feature_small_middle, feature_middle)
            # KD_feature_loss_last = self.Attn3(feature_small_last, feature_last)
            #
            # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last

            # WKD
            # wkd = kd_losses.WKD()
            # KD_loss = wkd(outputs_small, outputs)
            # # Total loss
            # loss_G_small = loss_content_small + self.adv_lambda * loss_adv_small + KD_loss
            #
            # self.metric_counter.add_losses(loss_G_small.item(), loss_content_small.item(), loss_adv_small.item(),
            #                                KD_loss.item(), 0, 0, 0, loss_D_small)



            # Total Loss
            loss_G_small = loss_content_small + self.adv_lambda * loss_adv_small + KD_loss + KD_feature_loss

            # FTKD_1의 경우 Total Loss
            # loss_G_small = loss_content_small + self.adv_lambda * loss_adv_small + 0.1 * KD_total_loss

            self.metric_counter.add_losses(loss_G_small.item(), loss_content_small.item(), loss_adv_small.item(),
                                           KD_loss.item(), KD_feature_loss_first.item(),
                                           KD_feature_loss_middle.item(), KD_feature_loss_last.item(), loss_D_small)

            loss_G_small.backward()
            self.optimizer_G_small.step()

            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs_small, targets)

            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            tq.set_postfix(loss='PSNR={:.4f}; SSIM={:.4f}'.format(curr_psnr, curr_ssim))
            if not i:
                self.metric_counter.add_image(img_for_vis, tag='train')
            i += 1
            if i > epoch_size:
                break

        tq.close()
        print("\nloss_content :", loss_content_small)
        # print("self.adv_lambda * loss_adv :", self.adv_lambda * loss_adv_small)
        print("KD feature 1", KD_feature_loss_first)
        print("KD feature 2", KD_feature_loss_middle)
        print("KD feature 3", KD_feature_loss_last)
        print("KD_loss :", KD_loss)
        self.metric_counter.write_to_tensorboard(epoch)

    def _validate(self, epoch):
        self.metric_counter.clear()
        epoch_size = self.config.get('val_batches_per_epoch') or len(self.val_dataset)
        tq = tqdm.tqdm(self.val_dataset, total=epoch_size)
        # tq = tqdm.tqdm(self.val_dataset)
        tq.set_description('Validation')
        i = 0

        # MLP
        # self.MLP1.eval()
        # self.MLP2.eval()
        # self.MLP3.eval()

        # AttnFD
        # self.Attn1.eval()
        # self.Attn2.eval()
        # self.Attn3.eval()

        for data in tq:
            inputs, targets, target_class = self.model.get_input(data)
            with torch.no_grad():
                outputs, logits, feature_first, feature_middle, feature_last = self.netG(inputs)
                outputs_small, feature_small_first, feature_small_middle, feature_small_last = self.netG_small(inputs)

                # cla_loss = self.cla_loss(logits, target_class)

                loss_content_small = self.criterionG(outputs_small, targets)
                loss_adv_small = self.adv_trainer_small.loss_g(outputs_small, targets)


                # FTKD
                feature_last = torch.fft.rfft2(feature_last)
                feature_small_last = torch.fft.rfft2(feature_small_last)
                outputs = torch.fft.rfft2(outputs)
                outputs_small_fft = torch.fft.rfft2(outputs_small)

                KD_loss = torch.nn.functional.l1_loss(outputs_small_fft, outputs)

                KD_feature_loss_first = torch.nn.functional.l1_loss(feature_first, feature_small_first)
                KD_feature_loss_middle = torch.nn.functional.l1_loss(feature_middle, feature_small_middle)
                KD_feature_loss_last = torch.nn.functional.l1_loss(feature_last, feature_small_last)

                KD_feature_loss = 0.1 * KD_feature_loss_first + 0.05 * KD_feature_loss_middle + KD_feature_loss_last          # case1 0.1 0.05
                # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last  # case2 no weight


                # FTKD_1 임계값 0.1
                # _, _, H, W = outputs_small.shape
                #
                # def FFT(x):
                #     x = torch.fft.fft2(x)
                #     x = torch.fft.fftshift(x)
                #     magnitude = torch.abs(x).squeeze()
                #     return magnitude
                #
                # def cal_distance(x):
                #     _, _, H, W = x.shape
                #     y, x = torch.meshgrid(torch.arange(H), torch.arange(W))  # 좌표 행렬 생성
                #     center_y, center_x = H // 2, W // 2
                #     distance = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                #     return distance  # 각 좌표의 distance 행렬
                #
                # def extract_high_freq(x, threshold_ratio=0.1):
                #     distance = cal_distance(x)
                #     max_distance = distance.max()
                #     threshold = max_distance * threshold_ratio
                #     high_freq_mask = distance > threshold  # threshold 보다 크면 True, 아니면 False
                #     x_high = x * high_freq_mask.cuda()  # magnitude * high_freq_mask
                #     return x_high
                #
                # feature_last = FFT(feature_last)
                # feature_small_last = FFT(feature_small_last)
                # outputs = FFT(outputs)
                # outputs_small_FFT = FFT(outputs_small)
                #
                # feature_last = extract_high_freq(feature_last)  # extract high freq
                # feature_small_last = extract_high_freq(feature_small_last)
                # outputs = extract_high_freq(outputs)
                # outputs_small_high = extract_high_freq(outputs_small_FFT)
                #
                # KD_loss = torch.nn.functional.l1_loss(outputs_small_high, outputs)
                #
                # KD_feature_loss_first = torch.nn.functional.l1_loss(feature_first, feature_small_first)
                # KD_feature_loss_middle = torch.nn.functional.l1_loss(feature_middle, feature_small_middle)
                # KD_feature_loss_last = torch.nn.functional.l1_loss(feature_last, feature_small_last)
                # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + 0.2 * KD_feature_loss_last
                #
                # KD_total_loss = 0.5 * KD_loss + KD_feature_loss
                # KD_total_loss = 0.1 * KD_total_loss

                # FTKD_2
                # def FFT(x):
                #     x = torch.fft.fft2(x)
                #     x = torch.fft.fftshift(x)
                #     magnitude = torch.abs(x)
                #     return magnitude
                #
                # feature_last = FFT(feature_last)
                # feature_small_last = FFT(feature_small_last)
                # outputs = FFT(outputs)
                # outputs_small_mag = FFT(outputs_small)
                #
                # KD_loss = torch.nn.functional.l1_loss(outputs_small_mag, outputs)
                #
                # KD_feature_loss_first = torch.nn.functional.l1_loss(feature_first, feature_small_first)
                # KD_feature_loss_middle = torch.nn.functional.l1_loss(feature_middle, feature_small_middle)
                # KD_feature_loss_last = torch.nn.functional.l1_loss(feature_last, feature_small_last)
                # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last


                # FTKD_3   phase, mag 분리해서
                # def FFT_mag(x):
                #     x = torch.fft.fft2(x)
                #     x = torch.fft.fftshift(x)
                #     magnitude = torch.abs(x)
                #     return magnitude
                #
                # def FFT_phase(x):
                #     x = torch.fft.fft2(x)
                #     phase = torch.angle(x)
                #     zero_mag = torch.ones_like(x)
                #     phase_img = zero_mag * torch.exp(1j * phase)
                #     phase_img = torch.fft.ifft2(phase_img).real
                #     return phase_img
                #
                # feature_last_mag = FFT_mag(feature_last)        # mag
                # feature_small_last_mag = FFT_mag(feature_small_last)
                # outputs_mag = FFT_mag(outputs)
                # outputs_small_mag = FFT_mag(outputs_small)
                #
                # feature_last_phase = FFT_phase(feature_last)        # phase
                # feature_small_last_phase = FFT_phase(feature_small_last)
                # outputs_phase = FFT_phase(outputs)
                # outputs_small_phase = FFT_phase(outputs_small)
                #
                # KD_loss_mag = torch.nn.functional.l1_loss(outputs_small_mag, outputs_mag)       # output 비교
                # KD_loss_phase = torch.nn.functional.l1_loss(outputs_small_phase, outputs_phase)
                # KD_loss = KD_loss_mag + KD_loss_phase
                #
                # KD_feature_loss_first = torch.nn.functional.l1_loss(feature_first, feature_small_first)
                # KD_feature_loss_middle = torch.nn.functional.l1_loss(feature_middle, feature_small_middle)
                #
                # KD_feature_loss_last_mag = torch.nn.functional.l1_loss(feature_last_mag, feature_small_last_mag)        # feature last 비교
                # KD_feature_loss_last_phase = torch.nn.functional.l1_loss(feature_last_phase, feature_small_last_phase)
                # KD_feature_loss_last = KD_feature_loss_last_mag + KD_feature_loss_last_phase
                #
                # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last

                # FFT 모두 적용
                # feature_first = torch.fft.rfft2(feature_first)
                # feature_small_first = torch.fft.rfft2(feature_small_first)
                # feature_middle = torch.fft.rfft2(feature_middle)
                # feature_small_middle = torch.fft.rfft2(feature_small_middle)
                # feature_last = torch.fft.rfft2(feature_last)
                # feature_small_last = torch.fft.rfft2(feature_small_last)
                # outputs = torch.fft.rfft2(outputs)
                # outputs_small_FFT = torch.fft.rfft2(outputs_small)
                #
                # KD_loss = torch.nn.functional.l1_loss(outputs_small_FFT, outputs)
                #
                # KD_feature_loss_first = torch.nn.functional.l1_loss(feature_first, feature_small_first)
                # KD_feature_loss_middle = torch.nn.functional.l1_loss(feature_middle, feature_small_middle)
                # KD_feature_loss_last = torch.nn.functional.l1_loss(feature_last, feature_small_last)
                #
                # KD_feature_loss = 0.1 * KD_feature_loss_first + 0.05 * KD_feature_loss_middle + KD_feature_loss_last

                # FFT 미적용
                # KD_loss = torch.nn.functional.l1_loss(outputs_small, outputs)
                #
                # KD_feature_loss_first = torch.nn.functional.l1_loss(feature_first, feature_small_first)
                # KD_feature_loss_middle = torch.nn.functional.l1_loss(feature_middle, feature_small_middle)
                # KD_feature_loss_last = torch.nn.functional.l1_loss(feature_last, feature_small_last)
                #
                # KD_feature_loss = 0.1 * KD_feature_loss_first + 0.05 * KD_feature_loss_middle + KD_feature_loss_last


                # Logits
                # Logits = kd_losses.logits.Logits()
                # KD_loss = Logits(outputs_small, outputs)
                #
                # # Total loss
                # loss_G_small = loss_content_small + self.adv_lambda * loss_adv_small + KD_loss
                #
                # self.metric_counter.add_losses(loss_G_small.item(), loss_content_small.item(), loss_adv_small.item(),
                #                                KD_loss.item(), 0, 0, 0)


                # FitNet
                # KD_loss = F.mse_loss(outputs_small, outputs)
                #
                # Hint = kd_losses.fitnet.Hint()
                # KD_feature_loss_first = Hint(feature_small_first, feature_first)
                # KD_feature_loss_middle = Hint(feature_small_middle, feature_middle)
                # KD_feature_loss_last = Hint(feature_small_last, feature_last)
                #
                # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last


                # Attention Transfer
                # KD_loss = F.mse_loss(outputs_small, outputs)
                #
                # AT = kd_losses.at.AT(2)
                # KD_feature_loss_first = AT(feature_small_first, feature_first)
                # KD_feature_loss_middle = AT(feature_small_middle, feature_middle)
                # KD_feature_loss_last = AT(feature_small_last, feature_last)
                #
                # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last


                # Similarity Preserving
                # KD_loss = F.mse_loss(outputs_small, outputs)
                #
                # SP = kd_losses.sp.SP()
                # KD_feature_loss_first = SP(feature_small_first, feature_first)
                # KD_feature_loss_middle = SP(feature_small_middle, feature_middle)
                # KD_feature_loss_last = SP(feature_small_last, feature_last)
                #
                # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last


                # Simple Recipe Distillation
                # KD_loss = F.mse_loss(outputs_small, outputs) * 10
                #
                # SRD1 = kd_losses.SRD.DistillationLoss(32, 32)
                # SRD2 = kd_losses.SRD.DistillationLoss(192, 192)
                # SRD3 = kd_losses.SRD.DistillationLoss(64, 64)
                # KD_feature_loss_first = SRD1(feature_small_first, feature_first)
                # KD_feature_loss_middle = SRD2(feature_small_middle, feature_middle)
                # KD_feature_loss_last = SRD3(feature_small_last, feature_last)
                #
                # KD_feature_loss = 0.1 * (KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last)


                # CWD
                # KD_loss = F.mse_loss(outputs_small, outputs)

                # cwd = kd_losses.CWD.CriterionCWD()
                # KD_feature_loss_first = cwd(feature_small_first, feature_first)
                # KD_feature_loss_middle = cwd(feature_small_middle, feature_middle)
                # KD_feature_loss_last = cwd(feature_small_last, feature_last)

                # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last


                # MLP
                # KD_loss = F.mse_loss(outputs_small, outputs)
                #
                # KD_feature_loss_first = self.MLP1(feature_small_first, feature_first)
                # KD_feature_loss_middle = self.MLP2(feature_small_middle, feature_middle)
                # KD_feature_loss_last = self.MLP3(feature_small_last, feature_last)
                #
                # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last


                # AttnFD
                # KD_loss = F.mse_loss(outputs_small, outputs)
                #
                # KD_feature_loss_first = self.Attn1(feature_small_first, feature_first)
                # KD_feature_loss_middle = self.Attn2(feature_small_middle, feature_middle)
                # KD_feature_loss_last = self.Attn3(feature_small_last, feature_last)
                #
                # KD_feature_loss = KD_feature_loss_first + KD_feature_loss_middle + KD_feature_loss_last


                # WKD
                # wkd = kd_losses.WKD()
                # KD_loss = wkd(outputs_small, outputs)
                # # Total loss
                # loss_G_small = loss_content_small + self.adv_lambda * loss_adv_small + KD_loss
                #
                # self.metric_counter.add_losses(loss_G_small.item(), loss_content_small.item(), loss_adv_small.item(),
                #                                KD_loss.item(), 0, 0, 0)


            # Total Loss
            loss_G_small = loss_content_small + self.adv_lambda * loss_adv_small + KD_loss + KD_feature_loss

            # FTKD_1, 2의 경우 Total Loss
            # loss_G_small = loss_content_small + self.adv_lambda * loss_adv_small + 0.1 * KD_total_loss

            self.metric_counter.add_losses(loss_G_small.item(), loss_content_small.item(), loss_adv_small.item(),
                                           KD_loss.item(), KD_feature_loss_first.item(),
                                           KD_feature_loss_middle.item(), KD_feature_loss_last.item())

            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs_small, targets)
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

        self.netG_small.load_state_dict(checkpoint['model'])
        self.netD_small['patch'].load_state_dict(checkpoint['model_D_patch'])
        self.netD_small['full'].load_state_dict(checkpoint['model_D_full'])

        self.optimizer_G_small.load_state_dict(checkpoint['optimizer_G'])
        self.scheduler_G_small.load_state_dict(checkpoint['scheduler_G'])
        self.optimizer_D_small.load_state_dict(checkpoint['optimizer_D'])
        self.scheduler_D_small.load_state_dict(checkpoint['scheduler_D'])
        # self.cla_optimizer.load_state_dict(checkpoint['optimizer_cla'])
        # self.cla_scheduler.load_state_dict(checkpoint['scheduler_cla'])

    def _update_d(self, outputs, targets):
        if self.config['model']['d_name'] == 'no_gan':
            return 0
        self.optimizer_D_small.zero_grad()
        loss_D = self.adv_lambda * self.adv_trainer_small.loss_d(outputs, targets)
        loss_D.backward(retain_graph=True)
        self.optimizer_D_small.step()
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
        self.netG, _ = get_nets(self.config['model'])    # fpn_inception, double_gan
        self.netG_small, self.netD_small = get_nets_small((self.config['model']))
        self.netG.cuda()
        self.netG_small.cuda()

        self.adv_trainer_small = self._get_adversarial_trainer(self.config['model']['d_name'], self.netD_small, criterionD)

        self.model = get_model(self.config['model'])

        self.optimizer_G_small = self._get_optim(filter(lambda p: p.requires_grad, self.netG_small.parameters()))
        self.optimizer_D_small = self._get_optim_d(self.adv_trainer_small.get_params())
        self.scheduler_G_small = self._get_scheduler(self.optimizer_G_small)
        self.scheduler_D_small = self._get_scheduler(self.optimizer_D_small)


def main(config_path='config/KD_config.yaml'):
    with open(config_path, 'r',encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    batch_size = config.pop('batch_size')
    get_dataloader = partial(DataLoader,
                             batch_size=batch_size,
                             shuffle=True, drop_last=True)

    datasets = map(config.pop, ('train', 'val'))
    datasets = map(PairedDataset.from_config, datasets)
    train, val = map(get_dataloader, datasets)
    trainer = Trainer(config, train=train, val=val)
    trainer.train()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    Fire(main)
