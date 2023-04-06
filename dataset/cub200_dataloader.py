import os

import numpy as np
import torch
from PIL import Image


# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class CUB200Dataset(torch.utils.data.Dataset):
    # cutouts - if dataloader is to load cutouts or images
    # train - loads train or test set
    # subset - limits classes to only the selected ones, None or False for all classes
    def __init__(self, path, transform=None, cutouts=True, data_set='ALL',
                 subset=[3, 5, 9, 15, 17, 18, 20, 21, 27, 29, 36, 44, 45, 46, 47, 51, 64, 72, 82, 84, 87, 90, 91, 92,
                         93, 98, 99, 100, 104, 106, 107, 108, 110, 111, 134, 139, 141, 149, 173, 187, 199, 200]
                 ):
        self.data_set = data_set.upper() if data_set else 'ALL'
        assert self.data_set in ['ALL', 'TRAIN', 'TEST']
        self.cutouts = cutouts
        self.path = path
        self.transform = transform
        self.subset = subset
        if subset:
            self.mapping_cl_to_idx = {cl: i for i, cl in enumerate(subset)}
            self.mapping_idx_to_cl = {i: cl for i, cl in enumerate(subset)}
        self.imgs_class = {}  # id -> class
        self.classes = set()
        try:
            self.logits = torch.load(os.path.join(path,'trained_logits.pt'))
        except:
            self.logits = None
        with open(os.path.join(path, 'image_class_labels.txt')) as file:
            for row in file:
                row = row.split(' ')
                id, cls = int(row[0]), int(row[1])-1 # classes are numbered from 1, I need them to start at 0
                self.imgs_class[id] = cls
                if self.subset is None or cls+1 in subset:
                    self.classes.add(cls)
        self.is_train = {}
        with open(os.path.join(path, 'train_test_split.txt')) as file:
            for row in file:
                row = row.split(' ')
                id, is_train = int(row[0]), (int(row[1]) == 1)
                self.is_train[id] = is_train

        def allow(id):
            if self.data_set == 'ALL':
                return True
            elif self.data_set == 'TRAIN':
                return self.is_train[id]
            else:
                return not self.is_train[id]

        self.imgs_ids = {}  # name -> id
        self.imgs_names = {}  # id -> name
        self.imgs_path = {}
        self.ids = {}  # idx -> id
        with open(os.path.join(path, 'images.txt')) as file:
            for row in file:
                row = row.split(' ')
                # the '-1' removes trailing \n
                id, name = int(row[0]), row[1][row[1].find('/') + 1:]
                if (self.subset is None or self.imgs_class[id] in subset) and allow(id):
                    self.imgs_path[id] = row[1][:-1]
                    self.imgs_ids[name] = id
                    self.imgs_names[id] = name
                    self.ids[len(self.ids)] = id
        print('Found {} classes'.format(self.num_classes()) )
        self.bounding_boxes  = {}
        with open(os.path.join(path, 'bounding_boxes.txt')) as file:
            for row in file:
                row = row.split(' ')
                # the '-1' removes trailing \n
                id, tmp_bbox = int(row[0]), row[1:]
                bbox = []
                for coord in tmp_bbox:
                    bbox.append(int(coord))
                self.bounding_boxes[id] = bbox

    def __len__(self):
        return len(self.ids)
    def num_classes(self):
        return len(self.classes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        id = self.ids[idx]
        if self.cutouts:
            img = Image.open(os.path.join(self.path, 'cutouts', self.imgs_path[id]))
        else:
            img = Image.open(os.path.join(self.path, 'images', self.imgs_path[id]))
        img = img.crop(tuple(self.bounding_boxes[id]))
        if self.transform:
            img = self.transform(img)
        label = self.imgs_class[id] if self.subset is None else self.mapping_cl_to_idx[self.imgs_class[id]]
        item = {'img':img, 'label':label, 'id':id}
        if self.logits and id in self.logits.keys():
            item['logit'],item['cls'] = self.logits[id]
        return item

    def get_filename(self, idx):
        return self.imgs_path[self.get_id(idx)]

    def get_id(self, idx):
        return self.ids[idx]
