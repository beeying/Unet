import numpy as np
import torch
import torch.nn as nn
import transformation as trans
from torch.utils.data import DataLoader


class getH5Data(Dataset):
    def __init__(self, h5path, transform=None):
        self.name = name
        self.transform = transform

        if transform == None:
            data = self.read_h5(os.path.join(
                h5path, 'data.hy'), ['Image', 'Mask'])
            self.image = data['Image'].astype(np.float32) / 255
            self.mask = data['Mask'].astype(np.float32) / 255

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, index):
        if self.transform:
            img, mask = self.transform(self.image, self.mask)
            return img, mask
        else:
            return self.image, self.mask

    def read_h5(self, path, field, is_uint8=True):
        data = h5py.File(path, 'r')
        data_dict = {}
        for i in field:
            read_data = data[i]
            if is_uint8:
                read_data = np.asanyarray(read_data).astype(np.uint8)
            data_dict.update({i: read_data})
        return data_dict


def create_data_loaders(args):
    if args.is_train:
        if args.is_non_texture:
            path = [os.path.join(args.datadir, 'unlabeled_non_texture', args.type),
                    os.path.join(args.datadir, 'unlabeled_non_texture', args.type)]
        else:
            path = [os.path.join(args.datadir, 'unlabeled_texture', args.type),
                    os.path.join(args.datadir, 'labeled_texture', args.type)]
            # path = [os.path.join(datadir, 'unlabeled'), os.path.join(datadir, 'labeled', '3_samples')]

        unlabeled_minibatch_size = args.batch_size - args.labeled_batch_size
        train_transformation = trans.Translate_and_Rotate(
            max_xtranslation=10, max_ytranslation=10, max_rotation=90, flip=0.3)

        unlabeled = getH5Data(path, transform=train_transformation)
        labeled = getH5Data(path, transform=train_transformation)

        unlabeled_loader = torch.utils.data.DataLoader(unlabeled, batch_size=unlabeled_minibatch_size, shuffle=True, drop_last=True,
                                                       pin_memory=True)
        labeled_loader = torch.utils.data.DataLoader(labeled, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                                     pin_memory=True)
        return unlabeled_loader, labeled_loader
    else:
        if is_non_texture:
            path = os.path.join(datadir, 'test_non_texture')
        else:
            path = os.path.join(datadir, 'test_texture')

        loader = getH5Data(path, transform=None)
        data_loader = torch.utils.data.DataLoader(loader, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                                  pin_memory=True)

        return loader
