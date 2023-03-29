import os

import numpy as np
import torch
from PIL import Image


# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class CUB200Dataset(torch.utils.data.Dataset):
    # cutouts - if dataloader is to load cutouts or images
    # train - loads train or test set
    # subset - limits classes to only the selected ones, None or False for all classes
    def __init__(self, path, transform=None, cutouts=True, train=True,
                 subset=[3, 5, 9, 15, 17, 18, 20, 21, 27, 29, 36, 44, 45, 46, 47, 51, 64, 72, 82, 84, 87, 90, 91, 92,
                         93, 98, 99, 100, 104, 106, 107, 108, 110, 111, 134, 139, 141, 149, 173, 187, 199, 200]
                 ):
        self.train = train
        self.cutouts = cutouts
        self.path = path
        self.transform = transform
        self.subset = subset
        if subset:
            self.mapping_cl_to_idx = {cl:i for i,cl in enumerate(subset)}
            self.mapping_idx_to_cl = {i:cl for i,cl in enumerate(subset)}
        self.imgs_class = {}  # id -> class
        with open(os.path.join(path, 'image_class_labels.txt')) as file:
            for row in file:
                row = row.split(' ')
                id, cls = int(row[0]), int(row[1])
                self.imgs_class[id] = cls
        self.imgs_ids = {}  # name -> id
        self.imgs_names = {}  # id -> name
        self.imgs_path = {}
        with open(os.path.join(path, 'images.txt')) as file:
            for row in file:
                row = row.split(' ')
                # the '-1' removes trailing \n
                id, name = int(row[0]), row[1][row[1].find('/') + 1:]
                if subset and self.imgs_class[id] in subset:
                    self.imgs_path[id] = row[1][:-1]
                    self.imgs_ids[name] = id
                    self.imgs_names[id] = name
        self.train_ids = {}  # idx -> id
        self.test_ids = {}  # idx -> id
        with open(os.path.join(path, 'train_test_split.txt')) as file:
            for row in file:
                row = row.split(' ')
                id, is_train = int(row[0]), (int(row[1]) == 1)
                if subset and self.imgs_class[id] in subset:
                    if is_train:
                        self.train_ids[len(self.train_ids)] = id
                    else:
                        self.test_ids[len(self.test_ids)] = id

    def __len__(self):
        if self.train:
            return len(self.train_ids)
        else:
            return len(self.test_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        id = (self.train_ids[idx] if self.train else self.test_ids[idx])
        if self.cutouts:
            img = Image.open(os.path.join(self.path, 'cutouts', self.imgs_path[id]))
        else:
            img = Image.open(os.path.join(self.path, 'images', self.imgs_path[id]))
        if self.transform:
            img = self.transform(img)
        label = self.imgs_class[id] if self.subset is None else self.mapping_cl_to_idx[self.imgs_class[id]]
        return img, label
    def get_filename(self,idx):
        id = (self.train_ids[idx] if self.train else self.test_ids[idx])
        return self.imgs_path[id]
    def get_id(self,idx):
        id = (self.train_ids[idx] if self.train else self.test_ids[idx])
        return id

