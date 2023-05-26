import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from dataset.CUB200Dataset import CUB200Dataset


class CUB200DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./dataset/caltech_birds2011/CUB_200_2011', subset=None, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        if subset == 'small':
            self.subset = [3, 5, 9, 15, 17, 18, 20, 21, 27, 29, 36, 44, 45, 46, 47, 51, 64, 72, 82, 84, 87, 90, 91, 92,
                           93, 98, 99, 100, 104, 106, 107, 108, 110, 111, 134, 139, 141, 149, 173, 187, 199, 200]
        else:
            self.subset = subset
        self.batch_size = batch_size

        self.ds_train = None
        self.ds_test = None
        self.ds_all = None

    def prepare_data(self):
        # This method is used for any data download or preparation that only needs to be done once
        pass

    def setup(self, stage=None):
        # This method is used for data loading and splitting

        # https://github.com/M-Nauta/ProtoTree/blob/main/util/data.py
        train_transforms = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.RandomOrder([
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ColorJitter((0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, shear=(-2, 2), translate=[0.05, 0.05]),
            ]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        # ViT
        test_transforms = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.ds_train = CUB200Dataset(self.data_dir, data_set='TRAIN', transform=train_transforms, subset=self.subset)
        self.ds_test = CUB200Dataset(self.data_dir, data_set='TEST', transform=test_transforms, subset=self.subset)
        self.ds_all = CUB200Dataset(self.data_dir, data_set='ALL', transform=test_transforms, subset=self.subset)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.ds_all, batch_size=self.batch_size, num_workers=0)
