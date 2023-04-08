import pytorch_lightning as pl
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
crop_size = 160

train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(crop_size),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([  # as in vit extractor
    transforms.Resize(224),
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
dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True)
dl_test = DataLoader(ds_test, batch_size=bs, shuffle=False)
dl_all = DataLoader(ds_all, batch_size=bs, shuffle=False)


class ResNet18_distillation(pl.LightningModule):
    def __init__(self, model, teacher_temp, student_temp, num_labels=ds_train.num_classes()):
        super(ResNet18_distillation, self).__init__()
        self.resnet = model.cuda()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.final = nn.Linear(1000, num_labels).cuda()
        self.unfreeze()

    def forward(self, pixel_values):
        outputs = self.resnet(pixel_values)
        logits = self.final(outputs)
        return logits

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
        imgs, labels, teacher_logits = batch['img'], batch['label'], batch['logits']
        labels = labels.cuda()
        imgs = imgs.cuda()
        student_logits = self(imgs)
        teacher_log_probs = F.log_softmax(teacher_logits / self.teacher_temp, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.student_temp, dim=1)

        criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
        loss = criterion(student_log_probs, teacher_log_probs)

        predictions = student_log_probs.argmax(-1)
        accuracy = torch.where(predictions == labels, 1.0, 0.0)
        accuracy = torch.mean(accuracy)

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_accuracy", accuracy, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return AdamW(self.parameters(), lr=5e-5)

    def train_dataloader(self):
        return dl_train

    def test_dataloader(self):
        return dl_test


# for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
early_stop_callback = EarlyStopping(
    monitor='training_accuracy',
    patience=5,
    strict=False,
    verbose=False,
    mode='max'
)

model = ResNet18_distillation(models.resnet18(), 0.06, 0.1)
trainer = Trainer(accelerator='gpu', callbacks=[early_stop_callback], log_every_n_steps=5, max_epochs=250)
trainer.fit(model)
trainer.save_checkpoint('final.ckpt')
