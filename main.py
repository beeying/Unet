import numpy as np
import torch
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=gpu,
                    help='gpu index if you have multiple gpus')
parser.add_argument('--batch_size_labeled', type=int,
                    default=batch_size_labeled, help='batch_size')
parser.add_argument('--dataset', default=dataset, help='dataset')
parser.add_argument('--is_train', action=is_train, help='train or test')
parser.add_argument('--learning_rate', type=float,
                    default=learning_rate, help='initial learning rate for Adam')
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
parser.add_argument('--data_dir', default=path, help='files directory')
parser.add_argument('--out_dir', default=out_path, help='output directory')
parser.add_argument('--z_chanel', type=int, default=500,
                    help='output directory')
parser.add_argument('--type', default=type, help='dataset category')
parser.add_argument('--w', type=int, default=w, help='SSIM window size')
parser.add_argument('--is_continue', default=False, help='continue to train')
parser.add_argument('--is_AE_pretrained', default=True,
                    help='continue to train')
parser.add_argument('--is_nontexture', default=is_nontexture,
                    help='continue to train')

FLAGS = parser.parse_args()


def main(args):
    iter_time = 0
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    while iter_time < args.num_epoch


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(FLAGS)
