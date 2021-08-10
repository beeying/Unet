import numpy as np
import os
import h5py
from dataset import transformation as trans
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class getH5Data(Dataset):
    def __init__(self, h5path, transform=None):
        self.transform = transform

        data = self.read_h5(os.path.join(
            h5path, 'data.hy'), ['Image', 'Mask'])
        self.image = data['Image'].astype(np.float32) / 255
        self.mask = data['Mask'].astype(np.float32) / 255

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, index):
        if self.transform:
            img, mask = self.transform(self.image[index], self.mask[index])
            img = img.permute(2, 0, 1)
            mask = mask.permute(2, 0, 1)
            return img, mask
        else:
            # image = np.concatenate((self.image[index], self.image[index], self.image[index]), axis=2)
            image = self.image[index].reshape(256, 256, 1)
            return image.transpose(2, 0, 1), self.mask[index].transpose(2, 0, 1)

    def read_h5(self, path, field, is_uint8=True):
        data = h5py.File(path, 'r')
        data_dict = {}
        for i in field:
            read_data = data[i]
            if is_uint8:
                read_data = np.asanyarray(read_data).astype(np.uint8)
            data_dict.update({i: read_data})
        return data_dict


def create_data_loaders(cfg, is_transform=True):
    if is_transform:
        if cfg.texture:
            train_transformation = trans.RandomRotateb4Crop(max_rotation=90)

        else:
            train_transformation = trans.Translate_and_Rotate(
                max_xtranslation=10, max_ytranslation=10, max_rotation=90, flip=0.3)

        data = getH5Data(cfg.path.labeled, transform=train_transformation)
        data_loader = DataLoader(data, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True,
                                                       pin_memory=True)
        return data_loader
    else:
        loader = getH5Data(cfg.path.test)
        data_loader = DataLoader(loader, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True,
                                                  pin_memory=True)
        return data_loader
