from typing import Dict, Any

import torch
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
import numpy as np
from tqdm import tqdm
# from cub200_dataloader import CUB200Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import pytorch_lightning as pl
from transformers import AdamW
import torch.nn as nn
from pytorch_lightning import Trainer
import os
from pytorch_lightning.callbacks import EarlyStopping

bs = 32
feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits8')
# resize to 224 normalize with
# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.Lambda(lambda x: feature_extractor(images=x, return_tensors="pt"))
])
# Train on all
subset = None
# Train on selected
# subset = [3, 5, 9, 15, 17, 18, 20, 21, 27, 29, 36, 44, 45, 46, 47, 51, 64, 72, 82, 84, 87, 90, 91, 92,93, 98, 99, 100, 104, 106, 107, 108, 110, 111, 134, 139, 141, 149, 173, 187, 199, 200]

# ds_train = CUB200Dataset('./caltech_birds2011/CUB_200_2011', data_set='TRAIN', transform=trans, subset=subset)
# ds_test = CUB200Dataset('./caltech_birds2011/CUB_200_2011', data_set='TEST', transform=trans, subset=subset)
# ds_all = CUB200Dataset('./caltech_birds2011/CUB_200_2011', data_set='ALL', transform=trans, subset=subset)
# dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True)
# dl_test = DataLoader(ds_test, batch_size=bs, shuffle=False)
# dl_all = DataLoader(ds_all, batch_size=bs, shuffle=False)


class ViTLightningModule(pl.LightningModule):
    def __init__(self, num_labels=200):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTModel.from_pretrained('facebook/dino-vits8')
        self.final = nn.Linear(384, num_labels)
        self.unfreeze()
        self.save_logits_frequency = 30
        self.till_logits_save = self.save_logits_frequency
        self.save_logits()

    def forward(self, pixel_values, return_cls=False):
        outputs = self.vit(pixel_values=pixel_values)
        last_hidden_states = outputs.last_hidden_state
        cls = last_hidden_states[:, 0, :]
        outputs = self.final(cls)
        return {'outputs': outputs, 'cls': cls}

    def freeze(self) -> None:
        for name, layer in self.vit.named_modules():
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.final.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for name, layer in self.vit.named_modules():
            for param in layer.parameters():
                param.requires_grad = True if name in ['pooler.dense', 'pooler', 'pooler.activation'] else False
        for param in self.final.parameters():
            param.requires_grad = True

    def common_step(self, batch, batch_idx):
        imgs, labels = batch['img'], batch['label']
        labels = labels
        imgs = imgs['pixel_values'].squeeze()
        logits = self(imgs)['outputs']

        criterion = nn.CrossEntropyLoss()
        predictions = logits.argmax(-1)
        loss = criterion(logits, labels)
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

    def save_logits(self, force=False) -> None:
        if self.till_logits_save != 0 and not force:
            self.till_logits_save -= 1
            return
        self.till_logits_save = self.save_logits_frequency
        self.freeze()
        self.eval()
        with torch.no_grad():
            path = os.path.join('./caltech_birds2011/CUB_200_2011')
            all = 0
            correct = 0
            logits = {}
            # for batch in tqdm(dl_all, leave=False, desc='Saving logits'):
            #     imgs, labels, ids = batch['img'], batch['label'], batch['id']
            #     imgs = imgs['pixel_values'].cuda().squeeze()
            #     labels = labels.cuda()
            #     self.cuda()
            #     output = self(imgs, return_cls=True)
            #     outs, clses = output['outputs'], output['cls']
            #     prediction = outs.argmax(-1)
            #     correct += torch.where(prediction == labels, 1.0, 0.0).sum()
            #     all += len(labels)
            #     for id, out, cls in zip(ids, outs, clses):
            #         logits[id] = (out.cpu(), cls.cpu())
            torch.save(logits, os.path.join(path, 'logits.pt'))
            print("Acc on {} = {:.5}".format('ALL', correct / all))
        self.train()
        self.unfreeze()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.save_logits()
        for name, params in self.vit.named_parameters():
            if name in ['pooler.dense.bias', 'pooler.dense.weight']:
                checkpoint['vit_head_' + name] = params

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for name, params in self.vit.named_parameters():
            if name in ['pooler.dense.bias', 'pooler.dense.weight']:
                params = checkpoint['vit_head_' + name]


# for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
# early_stop_callback = EarlyStopping(
#     monitor='training_accuracy',
#     patience=5,
#     strict=False,
#     verbose=False,
#     mode='max'
# )

model = ViTLightningModule()
# trainer = Trainer(accelerator='gpu', callbacks=[early_stop_callback], log_every_n_steps=5, max_epochs=250)
# trainer.fit(model)
# trainer.save_checkpoint('final.ckpt')  # also saves logits
# model.save_logits(True)
