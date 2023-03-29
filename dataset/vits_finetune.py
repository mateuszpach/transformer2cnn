from typing import Dict, Any

import torch
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
import numpy as np
from cub200_dataloader import CUB200Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import pytorch_lightning as pl
from transformers import AdamW
import torch.nn as nn
from pytorch_lightning import Trainer
import os
from pytorch_lightning.callbacks import EarlyStopping

bs = 64
feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits8')
trans = transforms.Compose([
    transforms.Lambda(lambda x: feature_extractor(images=x, return_tensors="pt"))
])
ds_train = CUB200Dataset('./caltech_birds2011/CUB_200_2011', train=True, transform=trans)
ds_test = CUB200Dataset('./caltech_birds2011/CUB_200_2011', train=False, transform=trans)
dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True)
dl_test = DataLoader(ds_test, batch_size=bs, shuffle=False)


class ViTLightningModule(pl.LightningModule):
    def __init__(self, num_labels=42):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTModel.from_pretrained('facebook/dino-vits8').cuda()
        self.final = nn.Linear(384, num_labels).cuda()
        self.unfreeze()
        self.save_logits()

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        last_hidden_states = outputs.last_hidden_state
        cls = last_hidden_states[:, 0, :]
        outputs = self.final(cls)
        return outputs

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
        imgs, labels = batch
        labels = labels.cuda()
        imgs = imgs['pixel_values'].cuda().squeeze()
        logits = self(imgs)

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

    def save_logits(self) -> None:
        self.freeze()
        self.eval()
        with torch.no_grad():
            path = os.path.join('./caltech_birds2011/CUB_200_2011/logits')
            for ds, ds_name in zip([ds_train, ds_test], ['train', 'test']):
                all = 0
                correct = 0
                for idx in range(len(ds)):
                    # ds has __len__, but not __next__ or __iter__ so iterating over it is not allowed
                    imgs, label = ds[idx]
                    imgs = imgs['pixel_values'].cuda()
                    outs = self(imgs)
                    name = str(ds.get_id(idx)) + '.pt'
                    torch.save(outs, os.path.join(path, name))
                    prediction = outs.argmax(-1).cpu().item()
                    correct += 1 if prediction == label else 0
                    all += 1
                print("Acc on {} = {:.5}".format(ds_name, correct / all))
        self.train()
        self.unfreeze()

    def on_train_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.save_logits()
        for name, params in self.vit.named_parameters():
            if name in ['pooler.dense.bias', 'pooler.dense.weight']:
                checkpoint['vit_head_' + name] = params

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for name, params in self.vit.named_parameters():
            if name in ['pooler.dense.bias', 'pooler.dense.weight']:
                params = checkpoint['vit_head_' + name]


# for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
early_stop_callback = EarlyStopping(
    monitor='training_accuracy',
    patience=4,
    strict=False,
    verbose=False,
    mode='max'
)

model = ViTLightningModule()
trainer = Trainer(devices=1, callbacks=[early_stop_callback], log_every_n_steps=5, enable_checkpointing=False)
trainer.fit(model)
trainer.save_checkpoint('final.ckpt')  # also saves logits
