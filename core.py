import numpy as np
import os
import time
import torch
import losses.loss as loss
from torch.nn import functional as F
import pandas as pd

from config import cfg
from pytorch_lightning.metrics.functional.classification import iou, auroc
from sklearn.metrics import roc_curve

class Train:
    def __init__(self, cfg, data_loader, model, optimizer, saver, loss_fn, meters, log):
        self.cfg = cfg
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        self.saver = saver
        self.meters = meters
        self.loss_fn = loss.__dict__[loss_fn]
        self.log = log

    def step(self, epoch):
        raise NotImplementedError

    def _compute_loss(self, output, epoch):
        loss = self.loss_fn(**output)
        self.saver.save_log(loss, epoch)
        self.meters.update('losses', loss.item())
        return loss

    def save_image(self, epoch, output):
        self.saver.save_images(output['input'], epoch, prefix='in')
        self.saver.save_images(output['output'], epoch, prefix='out')

    def save_model(self, epoch, is_best):
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, epoch + 1, is_best=is_best)

    def log_data(self, epoch):
        self.log.debug(
            '******Epoch: [{0}]\t'
            'Train time {meters[train_time]:.3f}\t'
            'losses {meters[losses]:.4f}\t'.format(
                epoch, meters=self.meters))


class TrainRecon(Train):
    def __init__(self, cfg, data_loader, model, optimizer, saver, loss_fn, meters, log):
        super().__init__(cfg, data_loader, model, optimizer, saver, loss_fn, meters, log)


    def step(self, epoch):
        self.meters.reset()
        self.model.train()

        end = time.time()
        for i, (image, mask) in enumerate(self.data_loader):
            input_var = torch.autograd.Variable(image).cuda()
            output = self.model(input_var)
            output['target'] = input_var
            loss = self._compute_loss(output, epoch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.saver.save_log(loss, epoch)


        out_images = output['pred']
        self.meters.update('train_time', time.time() - end)
        return {'input': image.cpu(), 'output': out_images.cpu().data.numpy()}


class TrainSeg(Train):
    def __init__(self, cfg, data_loader, model, optimizer, saver, loss_fn, meters, log):
        super().__init__(cfg, data_loader, model, optimizer, saver, loss_fn, meters, log)


    def step(self, epoch):
        self.meters.reset()
        self.model.train()
        end = time.time()
        y_pred = torch.Tensor([]).cuda()
        y_true = torch.Tensor([]).cuda()
        for i, (image, mask) in enumerate(self.data_loader):
            input_var = torch.autograd.Variable(image).cuda()
            mask = torch.autograd.Variable(mask).cuda()
            output = self.model(input_var)
            output['target'] = mask
            loss = self._compute_loss(output, epoch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.saver.save_log(loss, epoch)

            out_images = torch.sigmoid(output['pred'])
            out_images = out_images.round()
            out_images = out_images.view(cfg.train.batch_size, -1)
            mask = mask.view(cfg.train.batch_size, -1)
            y_pred = torch.cat((y_pred, out_images), 0)
            y_true = torch.cat((y_true, mask), 0)

        iou_score = iou(y_pred, y_true, reduction='none')
        self.meters.update('iou_train', iou_score.cpu().numpy())
        self.meters.update('train_time', time.time() - end)
        self.log_data(epoch)
        return {'input': image.cpu(), 'output': output['pred'].cpu().data.numpy()}

    def log_data(self, epoch):
        self.log.info(
            '******Epoch: [{0}]\t'
            'Train time {meters[train_time]:.3f}\t'
            'losses {meters[losses]:.4f}\t'
            'iou_train {meters[iou_train]}\t'.format(
                epoch, meters=self.meters))

class TrainGAN(Train):
    def __init__(self, cfg, data_loader, model, optimizer, saver, loss_fn, meters, log):
        super().__init__(cfg, data_loader, model, optimizer, saver, loss_fn, meters, log)

    def step(self, epoch):
        self.meters.reset()
        self.model['G'].train()
        self.model['D'].train()
        real_label = 1
        fake_label = 0

        end = time.time()
        for i, (image, mask) in enumerate(self.data_loader):
            input_var = torch.autograd.Variable(image).cuda()
            # update D
            b_size = input_var.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=torch.device("cuda:0"))
            output = self.model['D'](input_var)
            output['target'] = label
            errD_real = self._compute_loss(output, epoch)
            self.optimizer['D'].zero_grad()
            errD_real.backward()

            noise = torch.randn(b_size, 100, 1, 1, device=torch.device("cuda:0"))
            fake = self.model['G'](noise)
            label.fill_(fake_label)
            output = self.model['D'](fake.detach())
            output['target'] = label
            errD_fake = self._compute_loss(output, epoch)
            errD_fake.backward()
            lossD = errD_real + errD_fake
            self.optimizer['D'].step()
            self.meters.update('lossD', lossD.item())
            self.saver.save_log(lossD, epoch)

            # update G
            for k in range(5):

                label.fill_(real_label)  # fake labels are real for generator cost
                noise = torch.randn(b_size, 100, 1, 1)
                fake = self.model['G'](noise)
                output = self.model['D'](fake)
                output['target'] = label
                errG = self._compute_loss(output, epoch)
                self.optimizer['G'].zero_grad()
                errG.backward()
                self.optimizer['G'].step()
                self.meters.update('lossG', errG.item())


        out_images = fake
        self.meters.update('train_time', time.time() - end)
        return {'input': image.cpu(), 'output': out_images.cpu().data.numpy()}

    def _compute_loss(self, output, epoch):
        loss = self.loss_fn(**output)
        self.saver.save_log(loss, epoch)
        return loss

    def log_data(self, epoch):
        self.log.debug(
            '******Epoch: [{0}]\t'
            'Train time {meters[train_time]:.3f}\t'
            'loss D {meters[lossD]:.4f}\t'
            'loss G {meters[lossG]:.4f}\t'.format(
                epoch, meters=self.meters))


class Evaluate:
    def __init__(self, data_loader, model, meters, loss_fn, log, measure='iou'):
        self.data_loader = data_loader
        self.model = model
        self.loss_fn = loss.__dict__[loss_fn]
        self.meters = meters
        self.log = log
        if measure=='iou':
            self.measure = iou
        else:
            self.measure = auroc

    def step(self, print_metric=False):
        raise NotImplementedError

    def log_data(self):
        self.log.info(
            '=================Evaluate================\t')
        metrics = self.meters.metrics
        for i in metrics:
            self.log.info('{0}: {1}'.format(i, self.meters[i]))

    def _compute_loss(self, output):
        loss = self.loss_fn(**output)
        self.meters.update('losses', loss.item())
        return loss


class EvalRecon(Evaluate):
    def __init__(self, data_loader, model, meters, loss_fn, log, measure='iou'):
        super().__init__(data_loader, model, meters, loss_fn, log, measure)
    def step(self, print_metric=False):
        self.meters.reset()
        # switch to evaluate mode
        self.model.eval()
        y_pred = torch.Tensor([]).cuda()
        y_true = torch.Tensor([]).cuda()
        for i, (image, mask) in enumerate(self.data_loader):
            with torch.no_grad():
                mask = torch.autograd.Variable(mask.cuda(async=True))
                input_var = torch.autograd.Variable(image.cuda(async=True))
                output = self.model(input_var)
                output_images = torch.abs(output['pred']-input_var[:,0,:,:].reshape((-1,1,256,256)))
            output['target'] = input_var
            loss = self._compute_loss(output)
            output_images = output_images.view(-1)
            mask = mask.view(-1)
            y_pred = torch.cat((y_pred, output_images), 0)
            y_true = torch.cat((y_true, mask), 0)
            self.meters.update('losses', loss.item())
        if print_metric:
            auc = self.measure(y_pred, y_true)
            self.meters.update('auc', auc)
            threshold = self.find_optimal_cutoff(y_true.cpu().numpy().astype(np.int32), y_pred.cpu())
            y_pred_round = y_pred > threshold[0]
            IOU = iou(y_pred_round, y_true, reduction='none')
            self.meters.update('iou', IOU)
        self.log_data()
        return {'input': image, 'output': output_images}

    def find_optimal_cutoff(self, target, predicted):
        """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        target : Matrix with dependent or target data, where rows are observations

        predicted : Matrix with predicted data, where rows are observations

        Returns
        -------
        list type, with optimal cutoff value

        """
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

        return list(roc_t['threshold'])


class EvalSeg(Evaluate):
    def __init__(self, data_loader, model, meters, loss_fn, log, measure='iou'):
        super().__init__(data_loader, model, meters, loss_fn, log, measure)
    def step(self, print_metric=False):
        self.meters.reset()
        # switch to evaluate mode
        self.model.eval()
        y_pred = torch.Tensor([]).cuda()
        y_true = torch.Tensor([]).cuda()
        for i, (image, mask) in enumerate(self.data_loader):
            with torch.no_grad():
                mask = torch.autograd.Variable(mask.cuda(async=True))
                input_var = torch.autograd.Variable(image.cuda(async=True))
                output = self.model(input_var)
                output_images = torch.sigmoid(output['pred'])
                output_images = output_images.round()
            output['target'] = mask
            loss = self._compute_loss(output)
            output_images = output_images.view(cfg.train.batch_size, -1)
            mask = mask.view(cfg.train.batch_size, -1)
            y_pred = torch.cat((y_pred, output_images), 0)
            y_true = torch.cat((y_true, mask), 0)
            self.meters.update('losses', loss.item())
        if print_metric:
            iou = self.measure(y_pred, y_true, reduction='none')
            self.meters.update('iou', iou.cpu().numpy())
        self.log_data()
        return {'input': image, 'output': output_images}


# def train_pi(data_loader, model, optimizer, epoch, saver, meters):
#     # switch to train mode
#     meters.reset()
#     model.train()
#     end = time.time()
#     loader = zip(data_loader[0], data_loader[1])
#     for i, ((im_unlabeled, _), (im_labeled, mask)) in enumerate(loader):
#         meters.update('data_time', time.time() - end)
#
#         output_labeled = model(im_labeled)
#         output_unlabeled = model(im_unlabeled)
#
#         #pi model perturbation
#
#         im_unlabeled_np = im_unlabeled.cpu().data.numpy()
#         output_unlabeled_np = output_unlabeled.cpu().data.numpy()
#         im_unlabeled_trans = np.zeros_like(im_unlabeled_np)
#         output_unlabeled_trans = np.zeros_like(im_unlabeled_np)
#         for i in range(im_unlabeled_np.shape[0]):
#             im, out = \
#                 transformation(np.squeeze(im_unlabeled_np[i,...]), np.squeeze(output_unlabeled_np[i,...]))
#             im_unlabeled_trans[i, ...] = np.squeeze(im)
#             output_unlabeled_trans[i,...] = np.squeeze(out)
#
#         im_unlabeled_trans = torch.tensor(im_unlabeled_trans)
#         output_unlabeled_trans = torch.tensor(output_unlabeled_trans).cuda()
#         out_pi = model(im_unlabeled_trans)
#
#         loss_labeled, out_images = get_loss(output_labeled, mask.cuda(), cfg.model, batch_size=cfg.train.batch_size_labeled)
#         loss_unlabeled, out_images = get_loss(output_unlabeled_trans, {'Autoencoder': out_pi.cuda()}, 'Autoencoder', batch_size=cfg.train.batch_size_unlabeled)
#
#         alpha = 0.1
#
#         loss = loss_labeled+(alpha*loss_unlabeled)
#         print('label losses ', loss_labeled)
#         print('unlabel losses ', loss_unlabeled)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         meters.update('losses', loss.item())
#         saver.save_log(loss, epoch)
#
#     return {'input': im_labeled, 'output': out_images.cpu().data.numpy()}, meters







