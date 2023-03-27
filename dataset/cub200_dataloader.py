import os
import torch
from PIL import Image
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class CUB200DataLoader(torch.utils.data.Dataset):
    # cutouts - if dataloader is to load cutouts or images
    # train - loads train or test set
    def __init__(self,path,transform=None,cutouts=True,train=True):
        self.train = train
        self.cutouts = cutouts
        self.path = path
        self.transform = transform
        self.imgs_ids = {}  # name -> id
        self.imgs_names = {}  # id -> name
        self.imgs_path = {}
        with open(os.path.join(path,'images.txt')) as file:
            for row in file:
                row = row.split(' ')
                # the '-1' removes trailing \n
                id, name = int(row[0]), row[1][row[1].find('/') + 1:]
                self.imgs_path[id] = row[1][:-1]
                self.imgs_ids[name] = id
                self.imgs_names[id] = name
        self.train_ids = {} # idx -> id
        self.test_ids = {} # idx -> id
        with open(os.path.join(path,'train_test_split.txt')) as file:
            for row in file:
                row = row.split(' ')
                id, is_train = int(row[0]), (int(row[1]) == 1)
                if is_train:
                    self.train_ids[len(self.train_ids)] = id
                else:
                    self.test_ids[len(self.train_ids)] = id

        self.imgs_class = {}  # name -> class
        with open(os.path.join(path,'image_class_labels.txt')) as file:
            for row in file:
                row = row.split(' ')
                id, cls = int(row[0]), int(row[1])
                self.imgs_class[id] = cls
    def __len__(self):
        if self.train:
            return len(self.train_ids)
        else:
            return len(self.test_ids)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        id = ( self.train_ids[idx] if self.train else self.test_ids[idx])
        if self.cutouts:
            img = Image.open(os.path.join(self.path,'cutouts',self.imgs_path[id]))
        else:
            img = Image.open(os.path.join(self.path,'images',self.imgs_path[id]))
        if self.transform:
            img = self.transform(img)
        return (img,self.imgs_class[id])
