import numpy as np
import logging
import os
import torch
import shutil
from scipy.misc import imsave, imresize
from torch.utils.tensorboard import SummaryWriter


class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    @property
    def metrics(self):
        return [name for name in self.meters]

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


class Saver:
    def __init__(self, save_dir, image_size):
        self.save_dir = save_dir
        self.image_size = image_size
        self.writer = SummaryWriter(self.save_dir['logs'])

    def save_checkpoint(self, state, epoch, is_best):
        filename = 'checkpoint.{}.ckpt'.format(epoch)
        checkpoint_path = os.path.join(self.save_dir['model'], filename)
        torch.save(state, checkpoint_path)
        if is_best:
            # filename = 'checkpoint.{}.ckpt'.format(epoch)
            # checkpoint_path = os.path.join(self.save_dir['model'], filename)
            best_path = os.path.join(self.save_dir['model'], 'best.ckpt')
            torch.save(state, best_path)
            # shutil.copyfile(checkpoint_path, best_path)
            # log.debug("--- checkpoint copied to %s ---" % best_path)


    def save_images(self, imgs, iter_time, prefix=''):
        # check (N, C, W, H)
        if imgs.shape[-1]>3:
            # reshape image from vector to (N, H, W, C)
            imgs = np.transpose(imgs,(0, 2, 3, 1))
        if imgs.shape[3]>1:
            imgs_fake = np.reshape(imgs, (-1, self.image_size[0], self.image_size[1], 3))
            channel = 3
        else:
            imgs_fake = np.reshape(imgs, (-1, self.image_size[0], self.image_size[1], 1))
            channel = 1

        h_imgs, w_imgs = int(np.sqrt(imgs_fake.shape[0])), int(np.sqrt(imgs_fake.shape[0]))
        imsave(os.path.join(self.save_dir['output'], prefix+'_{}.png'.format(str(iter_time))),
               self._merge(imgs_fake, size=[h_imgs, w_imgs, channel], resize_ratio=1.))

    def _merge(self, images, size, resize_ratio=1.):
        h, w = images.shape[1], images.shape[2]
        h_ = int(h * resize_ratio)
        w_ = int(w * resize_ratio)
        if size[2] > 1:
            img_canvas = np.zeros((h_ * size[0], w_ * size[1], size[2]))
            im_size = (h_, w_, size[2])
        else:
            img_canvas = np.zeros((h_ * size[0], w_ * size[1]))
            im_size = (h_, w_)
            images = np.squeeze(images)
            if len(images.shape)<3:
                images = np.expand_dims(images, axis=0)
        n = 0
        for j in range(size[0]):
            for i in range(size[1]):
                image_resize = imresize(images[n,...], size=im_size, interp='bicubic')
                img_canvas[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_resize
                n=n+1

        return img_canvas

    def save_log(self, loss, epoch):
        self.writer.add_scalar('Loss/L2', loss, epoch)
        return

def create_log(path):
    LOG = logging.getLogger()
    LOG.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

    file_handler = logging.FileHandler(os.path.join(path['logs'],'log.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    LOG.addHandler(file_handler)
    LOG.addHandler(stream_handler)
    return LOG


def create_save_path(root, type):
    if not os.path.exists(root):
        os.mkdir(root)
    folder_name = ['model', 'logs', 'output']
    paths = {}
    for i in range(len(folder_name)):
        folder_path = os.path.join(root, folder_name[i])
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        type_path = os.path.join(folder_path, type)
        if not os.path.exists(type_path):
            os.mkdir(type_path)
        paths[folder_name[i]] = type_path

    return paths


def parameters_string(module):
    lines = [
        "",
        "List of model parameters:",
        "=========================",
    ]

    row_format = "{name:<40} {shape:>20} ={total_size:>12,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(
            name=name,
            shape=" * ".join(str(p) for p in param.size()),
            total_size=param.numel()
        ))
    lines.append("=" * 75)
    lines.append(row_format.format(
        name="all parameters",
        shape="sum of above",
        total_size=sum(int(param.numel()) for name, param in params)
    ))
    lines.append("")
    return "\n".join(lines)

def log_data(epoch, meters, log, is_train):
    if is_train:
        log.debug(
            'Epoch: [{0}]\t'
            'Train time {meters[train_time]:.3f}\t'
            'losses {meters[losses]:.4f}\t'.format(
                epoch, meters=meters))
    else:
        log.debug(
            'Evaluate:\t'
            'Loss {meters[class_loss]:.4f}\t'.format(meters=meters))
    return

def get_mIOU(c):
    return c[1, 1]/(c[1, 1]+c[0, 1]+c[1, 0])