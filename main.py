import numpy as np
import torch
import torch.nn as nn
import argparse
import models
import utils
from core import *

root = '/home/beeying/Desktop/NUS'


parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=0,
                    help='gpu index if you have multiple gpus')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size')
parser.add_argument('--batch_size_labeled', type=int,
                    default=0, help='batch_size')
parser.add_argument('--dataset', default='all_mask', help='dataset')
parser.add_argument('--is_train', action='store_false', help='train or test')
parser.add_argument('--learning_rate', type=float,
                    default=0.0001, help='initial learning rate for Adam')
parser.add_argument('--is_decay', action='store_false',
                    help='learning rate decay')
parser.add_argument('--num_epoch', type=int, default=200,
                    help='number of trining epoch')
parser.add_argument('--load_model', default=type,
                    help='folder of saved model that you wish to continue training')
parser.add_argument('--print_freq', type=int, default=10,
                    help='print frequency for loss, default: 100')
parser.add_argument('--save_freq', type=int, default=10,
                    help='save frequency for model, default: 5000')
parser.add_argument('--data_dir', default=root +
                    '/data/pytorch_semi/unlabeled_non_texture', help='files directory')
parser.add_argument('--out_dir', default=root +
                    '/experiment/AE_pytorch/', help='output directory')
parser.add_argument('--z_chanel', type=int, default=500,
                    help='output directory')
parser.add_argument('--type', default=type, help='dataset category')
parser.add_argument('--w', type=int, default=w, help='SSIM window size')
parser.add_argument('--is_continue', default=False, help='continue to train')
parser.add_argument('--is_AE_pretrained', default=True,
                    help='continue to train')
parser.add_argument('--is_nontexture', default=is_nontexture,
                    help='continue to train')

args = parser.parse_args()
LOG = logging.getLogger('main')


def main(args):
    iter_time = 0
    meters = utils.AverageMeterSet()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    train_loader = create_data_loaders(args)
    encoder = models.encoder()
    decoder = models.decoder()
    while iter_time < args.num_epoch:
        end = time.time()
        loss = train(train_loader, args, encoder, decoder)
        meters.update('data_time', time.time() - end)
        meters.update('loss', loss)
        if iter_time % args.save_freq == 0:
            LOG.info(
                'Epoch: [{0}]\t'
                'Data {meters[data_time]:.3f}\t'
                'loss {meters[loss]:.4f}\t'.format(
                    iter_time, meters=meters))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(args)
