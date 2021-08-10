import argparse

import os

from .yacs import CfgNode as CN

parser = argparse.ArgumentParser()

parser.add_argument('--is_train', action='store_false', help='train or test')
parser.add_argument('--type', default='bottle', help='train or test')
parser.add_argument('--model', default='Unet', help='model')

args = parser.parse_args()

root = '.'

cfg = CN()
# gpus
cfg.gpus = [0]

# if load the pretrained network
cfg.resume = False
cfg.model = args.model
cfg.type = args.type

texture_cats = ['grid', 'wood', 'tile', 'carpet', 'leather']

if cfg.type not in texture_cats:
    cfg.texture = False
else:
    cfg.texture = True

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.start_epoch = 0
cfg.train.num_epoch = 200
cfg.train.num_workers = 8

# use adam as default
cfg.train.lr = 1e-3

cfg.train.batch_size = 16
cfg.train.batch_size_labeled = 16
cfg.train.batch_size_unlabeled = cfg.train.batch_size - cfg.train.batch_size_labeled

cfg.train.save_freq = 10

# -----------------------------------------------------------------------------
# path
# -----------------------------------------------------------------------------
cfg.path = CN()
cfg.path.out_dir = root + '/experiment/pytorch' + cfg.model
cfg.path.data_dir = root + '/dataset/MVTecDATA/'
if cfg.texture:
    cfg.path.labeled = os.path.join(cfg.path.data_dir, 'labeled_texture', cfg.type)
    cfg.path.unlabeled = os.path.join(cfg.path.data_dir, 'unlabeled_texture', cfg.type)
    cfg.path.test = os.path.join(cfg.path.data_dir, 'test_texture', cfg.type)
else:
    cfg.path.labeled = os.path.join(cfg.path.data_dir, 'labeled_non_texture', cfg.type)
    cfg.path.unlabeled = os.path.join(cfg.path.data_dir, 'unlabeled_non_texture', cfg.type)
    cfg.path.test = os.path.join(cfg.path.data_dir, 'test_non_texture', cfg.type)
