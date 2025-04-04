import logging
from collections import defaultdict

import numpy as np
import os
from tensorboardX import SummaryWriter

WINDOW_SIZE = 100


class MetricCounter:
    def __init__(self, exp_name, save_dir):
        self.writer = SummaryWriter(exp_name)
        logging.basicConfig(filename=os.path.join(save_dir, '{}.log'.format(exp_name)), level=logging.DEBUG)
        self.metrics = defaultdict(list)
        self.images = defaultdict(list)
        self.best_metric = 0

    def add_image(self, x: np.ndarray, tag: str):
        self.images[tag].append(x)

    def clear(self):
        self.metrics = defaultdict(list)
        self.images = defaultdict(list)

    def add_losses(self, l_G, l_content, l_adv, l_KD, l_KD_1, l_KD_2, l_KD_3, l_D=0):
        for name, value in zip(('G_loss', 'G_loss_content', 'G_loss_adv', 'KD_loss', 'KD_feature_first', 'KD_feature_middle', 'KD_feature_last', 'D_loss'),
                               (l_G, l_content, l_adv, l_KD, l_KD_1, l_KD_2, l_KD_3, l_D)):
            self.metrics[name].append(value)

    # def add_metrics(self, psnr, ssim):
    def add_metrics(self, psnr):
        for name, value in zip(('PSNR',), (psnr,)):
            self.metrics[name].append(value)

    def loss_message(self):
        metrics = ((k, np.mean(self.metrics[k][-WINDOW_SIZE:])) for k in ('G_loss', 'PSNR'))
        return '; '.join(map(lambda x: f'{x[0]}={x[1]:.4f}', metrics))

    def write_to_tensorboard(self, epoch_num, validation=False):
        scalar_prefix = 'Validation' if validation else 'Train'
        for tag in ('G_loss', 'D_loss', 'G_loss_adv', 'G_loss_content', 'KD_loss', 'KD_feature_first', 'KD_feature_middle', 'KD_feature_last', 'PSNR'):
            self.writer.add_scalar(f'{scalar_prefix}_{tag}', np.mean(self.metrics[tag]), global_step=epoch_num)
        for tag in self.images:
            imgs = self.images[tag]
            if imgs:
                imgs = np.array(imgs)
                self.writer.add_images(tag, imgs[:, :, :, ::-1].astype('float32') / 255, dataformats='NHWC',
                                       global_step=epoch_num)
                self.images[tag] = []

    def update_best_model(self):
        cur_metric = np.mean(self.metrics['PSNR'])
        if self.best_metric < cur_metric:
            self.best_metric = cur_metric
            return True
        return False
