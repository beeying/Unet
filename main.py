# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import re
import argparse
import os
import shutil
import time
import math
import logging
import h5py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from PIL import Image
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter
import imageio

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
from scipy.misc import imsave, imresize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0

args = cli.parse_commandline_args()

experiment_path = '/home/beeying/Desktop/NUS/experiment/mean_teacher'
if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)

model_path = os.path.join(experiment_path, 'model')
if not os.path.exists(model_path):
    os.mkdir(model_path)
checkpoint_path = os.path.join(model_path, args.type)
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

log_path = os.path.join(experiment_path, 'logs')
if not os.path.exists(log_path):
    os.mkdir(log_path)
log_dir = os.path.join(log_path, args.type)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

sample_path = os.path.join(experiment_path, 'sample')
if not os.path.exists(sample_path):
    os.mkdir(sample_path)
sample_dir = os.path.join(experiment_path,sample_path,args.type)
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

def main():
    global global_step
    global best_prec1

    dataset_config = datasets.__dict__[args.dataset](args.inference, args.is_nontexture)

    def create_model(ema=False):
        LOG.info("=> creating {pretrained}{ema}model '{arch}'".format(
            pretrained='pre-trained ' if args.pretrained else '',
            ema='EMA ' if ema else '',
            arch=args.arch))

        model_factory = architectures.__dict__[args.arch]
        model_params = dict(pretrained=args.pretrained)

        model = model_factory(**model_params)
        model = nn.DataParallel(model).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)
    if args.inference:
        eval_loader = create_eval_loaders(**dataset_config, args=args)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(os.path.join(model_path, args.type, 'checkpoint.250.ckpt'))
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        validate(eval_loader, ema_model)
    else:
        unlabeled_loader, labeled_loader = create_data_loaders(**dataset_config, args=args)
        LOG.info(parameters_string(model))
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

        cudnn.benchmark = True
        writer = SummaryWriter(log_dir)

        for epoch in range(args.start_epoch, args.epochs):
            start_time = time.time()
            # train for one epoch
            train(unlabeled_loader, labeled_loader, model, ema_model, optimizer, epoch, writer)
            LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

            is_best = False

            if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0 and epoch > 200:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_path, epoch + 1)


def read_h5(path, field, is_uint8=True):
    data = h5py.File(path, 'r')
    data_dict = {}
    for i in field:
        read_data = data[i]
        if is_uint8:
            read_data = np.asanyarray(read_data).astype(np.uint8)

        data_dict.update({i : read_data})
    return data_dict


def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)


def create_data_loaders(train_transformation,
                        datadir,
                        args):
    if args.is_nontexture:
        path = [os.path.join(datadir, 'unlabeled_non_texture'), os.path.join(datadir, 'labeled_non_texture')]
    else:
        # path = [os.path.join(datadir, 'unlabeled_texture'), os.path.join(datadir, 'labeled_texture')]
        path = [os.path.join(datadir, 'unlabeled'), os.path.join(datadir, 'labeled', '3_samples')]

    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])
    unlabeled_minibatch_size = args.batch_size - args.labeled_batch_size

    unlabeled = data.getH5Data(path, 'unlabeled',args.type, transform=train_transformation)
    labeled = data.getH5Data(path, 'labeled',args.type, transform=train_transformation)

    unlabeled_loader = torch.utils.data.DataLoader(unlabeled, batch_size=unlabeled_minibatch_size, shuffle=True, drop_last=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    labeled_loader = torch.utils.data.DataLoader(labeled, batch_size=args.labeled_batch_size, shuffle=True, drop_last=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    return unlabeled_loader, labeled_loader

def create_eval_loaders(datadir, args):
    if args.is_nontexture:
        path = os.path.join(datadir, 'test_non_texture')
    else:
        # path = os.path.join(datadir, 'test_texture')
        path = os.path.join(datadir, 'test')

    dataset = data.getH5Data(path, 'test',args.type, transform=None)
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                               num_workers=args.workers,
                                               pin_memory=True)
    return eval_loader


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(unlabeled_loader, labeled_loader, model, ema_model, optimizer, epoch, writer):
    global global_step

    class_criterion = nn.BCEWithLogitsLoss(size_average=False).cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()

    end = time.time()
    for i, ((unlabeled_image, unlabeled_mask), (labeled_image, labeled_mask)) in enumerate(zip(unlabeled_loader, labeled_loader)):
        # measure data loading time
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(labeled_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])

        images = torch.cat((unlabeled_image, labeled_image), 0)
        masks = torch.cat((unlabeled_mask, labeled_mask), 0)

        # ims = images.numpy().reshape((-1,256,256))*255
        # m = masks.numpy().reshape((-1,256,256))*255

        # for n in range(ims.shape[0]):
        #     imageio.imsave('{}.png'.format(n), ims[n,...].astype(np.uint8))
        #     imageio.imsave('{}_mask.png'.format(n), m[n,...].astype(np.uint8))

        input_var = torch.autograd.Variable(images)
        input_var = input_var.permute(0, 3, 1, 2)
        mask_train = torch.autograd.Variable(masks.cuda().round())
        mask_train = mask_train.permute(0, 3, 1, 2)
        model_out = model(input_var)
        output = model_out.permute(0, 2, 3, 1)
        predict_mask = torch.sigmoid(output)


        im = images.numpy()
        m = masks.numpy()
        p = predict_mask.cpu().data.numpy()

        ema_images = np.zeros((0, 256, 256, 1))
        ema_masks = np.zeros((0, 256, 256, 1))
        ema_predict = np.zeros((0, 256, 256, 1))

        for n in range(16):
            ime, me, pe = Rotate_Translate(im[n,...], m[n,...], p[n,...])
            ema_images = np.concatenate((ema_images, ime), 0)
            ema_masks = np.concatenate((ema_masks, me), 0)
            ema_predict = np.concatenate((ema_predict, pe), 0)

        ema_images = torch.Tensor(ema_images)
        ema_masks = torch.Tensor(ema_masks)
        ema_predict = torch.Tensor(ema_predict)

        with torch.no_grad():
            ema_input_var = torch.autograd.Variable(ema_images)
            ema_input_var = ema_input_var.permute(0, 3, 1, 2)
            mask_ema = torch.autograd.Variable(ema_masks.cuda().round())
            mask_ema = mask_ema.permute(0, 3, 1, 2)

        minibatch_size = args.batch_size
        unlabeled_minibatch_size = args.batch_size - args.labeled_batch_size
        assert args.labeled_batch_size > 0
        meters.update('labeled_minibatch_size', args.labeled_batch_size)

        ema_model_out = ema_model(ema_input_var)
        plots(images.data.numpy(), [256, 256, 1], i, sample_dir, prefix='input')
        plots(ema_images.data.numpy(), [256, 256, 1], i, sample_dir, prefix='input_ema')
        plots(ema_masks.data.numpy(), [256, 256, 1], i, sample_dir, prefix='mask')
        output1 = model_out.permute(0, 2, 3, 1)
        output1 = torch.sigmoid(output1).round()
        output1 = output1.cpu()
        plots(output1.data.numpy(), [256, 256, 1], i, sample_dir, prefix='predict')


        if isinstance(model_out, Variable):
            assert args.logit_distance_cost < 0
            logit1 = model_out
            ema_logit = ema_model_out
        else:
            assert len(model_out) == 2
            assert len(ema_model_out) == 2
            logit1, logit2 = model_out
            ema_logit, _ = ema_model_out

        ema_logit = Variable(ema_logit.detach().data, requires_grad=False)

        if args.logit_distance_cost >= 0:
            class_logit, cons_logit = logit1, logit2
            res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
            meters.update('res_loss', res_loss.data[0])
        else:
            class_logit, cons_logit = logit1, logit1
            res_loss = 0
        class_criterion.reduction = 'none'
        class_loss = class_criterion(class_logit[unlabeled_minibatch_size:,:,:,:], mask_train[unlabeled_minibatch_size:,:,:,:])
        class_loss = torch.sum(class_loss, dim=(1,2,3))
        class_loss = torch.mean(class_loss)
        meters.update('class_loss', class_loss.item())

        ema_class_loss = class_criterion(ema_logit[unlabeled_minibatch_size:,:,:,:], mask_ema[unlabeled_minibatch_size:,:,:,:])
        ema_class_loss = torch.sum(ema_class_loss, dim=(1, 2, 3))
        ema_class_loss = torch.mean(ema_class_loss)
        meters.update('ema_class_loss', ema_class_loss.item())

        if args.consistency:
            consistency_weight = get_current_consistency_weight(epoch)
            meters.update('cons_weight', consistency_weight)
            ema_logit = torch.sigmoid(ema_logit)
            consistency_loss = consistency_weight * consistency_criterion(ema_predict.cuda().permute(0, 3, 1, 2), ema_logit) / minibatch_size
            meters.update('cons_loss', consistency_loss.item())
        else:
            consistency_loss = 0
            meters.update('cons_loss', 0)

        loss = class_loss + consistency_loss + res_loss
        assert not (np.isnan(loss.item())), 'Loss explosion: {}'.format(loss.item())
        meters.update('loss', loss.item())

        # miou = meanIOU(class_logit, mask_train)
        # meters.update('miou', miou, unlabeled_minibatch_size)


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'EMA Class {meters[ema_class_loss]:.4f}\t'
                'Cons {meters[cons_loss]:.4f}\t'.format(
                    epoch, i, len(labeled_loader), meters=meters))

        writer.add_scalar('Loss/student', class_loss.item(), epoch)
        writer.add_scalar('Loss/teacher', ema_class_loss.item(), epoch)


def validate(eval_loader, model):
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    hw =256
    y_true = np.zeros((0, hw, hw))
    y_score = np.zeros((0, hw, hw))
    for i, (train_input) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        input_var = torch.autograd.Variable(train_input[0])
        input_var = input_var.permute(0, 3, 1, 2)
        target_var = torch.autograd.Variable(train_input[1].cuda().round())
        target_var = target_var.permute(0, 3, 1, 2)

        output1 = model(input_var)
        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        out = torch.sigmoid(output1).round()
        out = out.permute(0, 2, 3, 1)
        out = out.cpu()
        plots(train_input[0].data.numpy(), [256,256,1], i, sample_dir, prefix='input')
        plots(train_input[1].data.numpy(), [256, 256, 1], i, sample_dir, prefix='mask')
        imgs_mask =np.concatenate((train_input[0].data.numpy()+(out.data.numpy()*0.5), train_input[0].data.numpy(), train_input[0].data.numpy()), axis=3)
        plots(imgs_mask, [256, 256, 1], i, sample_dir, prefix='predict_mask', is_color=True)

        y_true = np.concatenate((y_true, train_input[1].data.round().numpy().reshape(1, hw, hw)), axis=0)
        y_score = np.concatenate((y_score, out.data.numpy().reshape(1, hw, hw)), axis=0)

    c = confusion_matrix(y_true.flatten(), y_score.flatten())
    t = precision_recall_fscore_support(y_true.flatten(), y_score.flatten())
    miou = meanIOU(c)
    print('miou', miou)
    meters.update('mean IOU', miou.item())
    print(t[1])
        # if i % args.print_freq == 0:
        #     LOG.info(
        #         'Test: [{0}/{1}]\t'
        #         'Time {meters[batch_time]:.3f}\t'
        #         'Data {meters[data_time]:.3f}\t'
        #         'Class {meters[class_loss]:.4f}\t'
        #         'Prec@1 {meters[top1]:.3f}\t'
        #         'Prec@5 {meters[top5]:.3f}'.format(
        #             i, len(eval_loader), meters=meters))


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        LOG.info("--- checkpoint copied to %s ---" % best_path)


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def meanIOU(c):
    """Computes the precision@k for the specified values of k"""
    iou_fg = c[0,0]/(c[0,0]+c[0,1]+c[1,0])
    iou_bg = c[1,1]/(c[1,1]+c[0,1]+c[1,0])
    # miou = (iou_fg+iou_bg)/2
    return iou_bg

def plots(imgs, image_size, iter_time, save_file, prefix=None, is_color=False):
# reshape image from vector to (N, H, W, C)
    if is_color:
        imgs_fake = np.reshape(imgs, (-1, image_size[0], image_size[1], 3))
        channel = 3
    else:
        imgs_fake = np.reshape(imgs, (-1, image_size[0], image_size[1], image_size[2]))
        channel = image_size[2]

    h_imgs, w_imgs = int(np.sqrt(imgs_fake.shape[0])), int(np.sqrt(imgs_fake.shape[0]))
    imsave(os.path.join(save_file, prefix+'_{}.png'.format(str(iter_time))),
           merge(imgs_fake, size=[h_imgs, w_imgs, channel], resize_ratio=1.))

def merge(images, size, resize_ratio=1.):
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
    n=0
    for j in range(size[0]):
        for i in range(size[1]):
            image_resize = imresize(images[n,...], size=im_size, interp='bicubic')
            img_canvas[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_resize
            n=n+1

    return img_canvas

def Rotate_Translate(old, old_mask, old_predict, max_translation=10, max_rotation=90, flip=0.3):
        xtranslation = np.random.randint(-max_translation,
                                                       max_translation,
                                                       size=1)
        ytranslation = np.random.randint(-max_translation,
                                                       max_translation,
                                                       size=1)
        degree = np.random.randint(-max_rotation,
                                    max_rotation,
                                    size=1)


        old_image = Image.fromarray(255*old.reshape((256,256)))
        old_mask = Image.fromarray(255*old_mask.reshape((256, 256)))
        old_predict = Image.fromarray(255*old_predict.reshape((256, 256)))
        xsize, ysize = old_image.size

        new_image = Image.new("L", (xsize, ysize))
        new_mask = Image.new("L", (xsize, ysize))
        new_predict = Image.new("L", (xsize, ysize))

        new_image.paste(old_image, box=None)
        new_mask.paste(old_mask, box=None)
        new_predict.paste(old_predict, box=None)

        new_image = new_image.rotate(degree, translate=(xtranslation, ytranslation))
        new_mask = new_mask.rotate(degree, translate=(xtranslation, ytranslation))
        new_predict = new_predict.rotate(degree, translate=(xtranslation, ytranslation))
        chance = np.random.randint(0, 100, size=1)
        if chance < flip:
            new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)
            new_mask = new_mask.transpose(Image.FLIP_LEFT_RIGHT)
            new_predict = new_predict.transpose(Image.FLIP_LEFT_RIGHT)

        new_image = np.array(new_image).astype(np.float32)
        new_mask = np.array(new_mask).astype(np.float32)
        new_predict = np.array(new_predict).astype(np.float32)

        pixel_coor = np.where(new_image == 0)
        new_image[pixel_coor] = 255*old[10, 10]


        return new_image.reshape((1,256,256,1))/255, new_mask.reshape((1,256,256,1))/255,  new_predict.reshape((1,256,256,1))/255

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    main()
