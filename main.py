import models
import utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from config import args
from dataset import create_data_loaders
from core import *
import core


out_paths = utils.create_save_path(cfg.path.out_dir, cfg.type)
saver = utils.Saver(out_paths, image_size=(256, 256))
LOG = utils.create_log(out_paths)

best_score = 10e10


def main(args, cfg):
    print('Model: ', cfg.model)
    print('Data: ', cfg.type)
    Model = models.__dict__[cfg.model]()
    loss_fn = Model.loss_fn
    Train = core.__dict__[Model.train_obj]
    Eval = core.__dict__[Model.eval_obj]
    measure = Model.measure

    model = nn.DataParallel(Model)
    # model=Model.cuda()
    LOG.debug(utils.parameters_string(model))
    cudnn.benchmark = True

    if args.is_train:
        data_loader = create_data_loaders(cfg)
        optimizer = torch.optim.Adam(model.parameters(), cfg.train.lr)
        train_meter = utils.AverageMeterSet()
        train = Train(cfg, data_loader, model, optimizer, saver, loss_fn, train_meter, LOG)

        eval_loader = create_data_loaders(cfg, is_transform=False)
        eval_meter = utils.AverageMeterSet()
        eval = Eval(eval_loader, model, eval_meter, loss_fn, LOG, measure=measure)
        for epoch in range(cfg.train.start_epoch, cfg.train.num_epoch):
            output = train.step(epoch)
            if epoch % cfg.train.save_freq == 0:
                eval.step(print_metric=True)
                train.save_image(epoch, output)
                train.save_model(epoch, is_best=False)
                if eval_meter['losses'].avg > best_score:
                    train.best_score = max(eval_meter['losses'].avg, best_score)
                    train.save_model(epoch, is_best=False)
        eval.step(print_metric=True)

    else:
        eval_loader = create_data_loaders(cfg, is_transform=False)
        eval_meter = utils.AverageMeterSet()
        eval = Evaluate(eval_loader, model, eval_meter, LOG)
        _, score = eval.step()


def load_model(path, model, log):
    assert os.path.isfile(path), "=> no checkpoint found at '{}'".format(path)
    log.info("=> loading checkpoint '{}'".format(path))
    checkpoint = torch.load(path)
    cfg.train.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    log.info("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))


if __name__ == '__main__':
    main(args, cfg)

