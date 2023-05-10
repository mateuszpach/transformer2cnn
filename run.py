import pytorch_lightning as pl
import pytorch_lightning.loggers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from transformers import AdamW

from dataset.cub200_dataloader import CUB200Dataset

bs = 64
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

test_transforms = transforms.Compose([  # as in vit extractor
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
model = models.resnet18()
# Train on all
subset = None
# Train on selected
# subset = [3, 5]
# subset = [3, 5, 9, 15, 17, 18, 20, 21, 27, 29, 36, 44, 45, 46, 47, 51, 64, 72, 82, 84, 87, 90, 91, 92,93, 98, 99, 100, 104, 106, 107, 108, 110, 111, 134, 139, 141, 149, 173, 187, 199, 200]

ds_train = CUB200Dataset('./dataset/caltech_birds2011/CUB_200_2011',
                         data_set='TRAIN',
                         transform=train_transforms,
                         subset=subset)
ds_test = CUB200Dataset('./dataset/caltech_birds2011/CUB_200_2011',
                        data_set='TEST',
                        transform=test_transforms,
                        subset=subset)
ds_all = CUB200Dataset('./dataset/caltech_birds2011/CUB_200_2011',
                       data_set='ALL',
                       transform=test_transforms,
                       subset=subset)
dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True,num_workers=0)
dl_test = DataLoader(ds_test, batch_size=bs, shuffle=False,num_workers=0)
dl_all = DataLoader(ds_all, batch_size=bs, shuffle=False,num_workers=0)


class ResNet18_distillation(pl.LightningModule):
    def __init__(self, model, teacher_temp, student_temp, num_labels=ds_train.num_classes(),cls_size=384,loss_ratio=0.5,replace_fc=True):
        super(ResNet18_distillation, self).__init__()
        self.resnet = model.cuda()
        print(resnet)
        # replace resnet finisher with identity
        num_features = 2048
        if replace_fc:
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Identity()

        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.loss_ratio = loss_ratio
        self.cls_size = cls_size

        self.final = nn.Sequential(
            nn.Linear(num_features,(num_features+num_labels)//2),
            nn.Linear((num_features+num_labels)//2, num_labels)).cuda()
        self.final_cls = nn.Sequential(
            nn.Linear(num_features,(num_features+cls_size)//2),
            nn.Linear((num_features+cls_size)//2, cls_size)).cuda()
        self.unfreeze()

    def forward(self, pixel_values):
        outputs = self.resnet(pixel_values)
        logits = self.final(outputs)
        cls = self.final_cls(outputs)
        return {'logits':logits,'cls':cls}

    def freeze(self) -> None:
        for name, layer in self.resnet.named_modules():
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.final.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for name, layer in self.resnet.named_modules():
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.final.parameters():
            param.requires_grad = True

    def common_step(self, batch, batch_idx):
        imgs, labels, teacher_logits, teacher_cls = batch['img'], batch['label'], batch['logits'], batch['cls']
        labels = labels.cuda()
        imgs = imgs.cuda()
        outs =  self(imgs)

        student_logits = outs['logits']
        teacher_log_probs = F.log_softmax(teacher_logits / self.teacher_temp, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.student_temp, dim=1)

        # loss_dist = nn.CrossEntropyLoss()(student_logits, labels)
        # loss_cls = 0
        criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
        loss_dist = criterion(student_log_probs, teacher_log_probs)

        student_cls = outs['cls']
        criterion_cls = nn.MSELoss()
        loss_cls = criterion_cls(student_cls,teacher_cls) / self.cls_size

        predictions = student_log_probs.argmax(-1)
        accuracy = torch.where(predictions == labels, 1.0, 0.0)
        accuracy = torch.mean(accuracy)

        base_predictions = teacher_log_probs.argmax(-1)
        base_accuracy = torch.where(base_predictions == labels, 1.0, 0.0)
        base_accuracy = torch.mean(base_accuracy)

        return loss_dist, loss_cls, accuracy, base_accuracy

    def training_step(self, batch, batch_idx):
        loss_dist,loss_cls, accuracy, base_accuracy = self.common_step(batch, batch_idx)

        loss_dist = self.loss_ratio * loss_dist
        loss_cls = (1-self.loss_ratio)*loss_cls
        loss = loss_dist + loss_cls

        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_distillation_loss", loss_dist,on_epoch=True,batch_size=bs)
        self.log("training_cls_loss", loss_cls,on_epoch=True,batch_size=bs)
        self.log("training_loss", loss,prog_bar=True,on_epoch=True,batch_size=bs)
        self.log("training_accuracy", accuracy,prog_bar=True,on_epoch=True,batch_size=bs)
        self.log("training_base_accuracy", base_accuracy, on_epoch=True,batch_size=bs)

        return loss

    def validation_step(self, batch, batch_idx):
        loss_dist, loss_cls, accuracy, base_accuracy = self.common_step(batch, batch_idx)

        loss_dist = self.loss_ratio * loss_dist
        loss_cls = (1-self.loss_ratio)*loss_cls
        loss = loss_dist + loss_cls

        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("val_distillation_loss", loss_dist, on_epoch=True,batch_size=bs)
        self.log("val_cls_loss", loss_cls, on_epoch=True,batch_size=bs)
        self.log("val_loss", loss, on_epoch=True,batch_size=bs)
        self.log("val_accuracy", accuracy, on_epoch=True,batch_size=bs)
        self.log("val_base_accuracy", base_accuracy, on_epoch=True,batch_size=bs)

        return loss

    def test_step(self, batch, batch_idx):
        loss_dist, loss_cls, accuracy, base_accuracy = self.common_step(batch, batch_idx)

        loss_dist = self.loss_ratio * loss_dist
        loss_cls = (1-self.loss_ratio)*loss_cls
        loss = loss_dist + loss_cls

        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("test_distillation_loss", loss_dist, on_epoch=True,batch_size=bs)
        self.log("test_cls_loss", loss_cls, on_epoch=True,batch_size=bs)
        self.log("test_loss", loss, on_epoch=True,batch_size=bs)
        self.log("test_accuracy", accuracy, on_epoch=True,batch_size=bs)
        self.log("test_base_accuracy", base_accuracy, on_epoch=True,batch_size=bs)

        return loss

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return AdamW(self.parameters(), lr=5e-5)

    def train_dataloader(self):
        return dl_train

    def test_dataloader(self):
        return dl_test

    def val_dataloader(self):
        return dl_test


# for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
early_stop_callback = EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    strict=False,
    verbose=False,
    mode='max'
)
# resnet = models.resnet18()
resnet = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50') # dino has last layer removed. set replace_fc to False

# loss = loss_ratio * loss_dist + (1-loss_ratio)*loss_cls
model = ResNet18_distillation(resnet, teacher_temp= 0.06, student_temp= 0.1,loss_ratio=0.1,replace_fc=False)
trainer = Trainer(accelerator='gpu', callbacks=[early_stop_callback], log_every_n_steps=5, max_epochs=300,
                  logger=pytorch_lightning.loggers.TensorBoardLogger('logs',name='Transformer2cnn'))
trainer.fit(model)
trainer.test()
trainer.save_checkpoint('final.ckpt')
