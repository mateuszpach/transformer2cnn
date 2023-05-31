import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW
from rkd import RkdDistance, RkdAngle


class DistilledResNetModel(pl.LightningModule):
    def __init__(self,
                 resnet_model,
                 teacher_temp=1,
                 student_temp=1,
                 num_labels=200,
                 cls_size=384,
                 cls_weight=1,
                 logits_weight=1,
                 ce_weight=1,
                 ikd_weight=1,
                 rkd_d_weight=1,
                 rkd_a_weight=1,
                 backbone_lr=5e-7,
                 head_lr=5e-5,
                 replace_fc=True):
        super(DistilledResNetModel, self).__init__()
        self.save_hyperparameters()
        self.resnet = resnet_model
        self.head_lr = head_lr
        self.backbone_lr = backbone_lr
        print(self.resnet)

        # replace resnet finisher with identity
        num_features = 2048
        if replace_fc:
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Identity()

        self.cls_projection = nn.Linear(num_features, cls_size)
        self.final_projection = nn.Linear(cls_size, num_labels)

        self.automatic_optimization = False

    def forward(self, pixel_values):
        embedding = self.resnet(pixel_values)
        cls = self.cls_projection(embedding)
        logits = self.final_projection(cls)
        return {'logits': logits, 'cls': cls}

    def freeze(self) -> None:
        for name, layer in self.resnet.named_modules():
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze(self) -> None:
        for name, layer in self.resnet.named_modules():
            for param in layer.parameters():
                param.requires_grad = False

    def common_step(self, batch, batch_idx):
        imgs, labels, teacher_logits, teacher_cls = batch['img'], batch['label'], batch['logits'], batch['cls']
        outs = self(imgs)
        student_logits = outs['logits']
        student_cls = outs['cls']

        # Cross-entropy
        loss_logits_ce = nn.CrossEntropyLoss()(student_logits, labels)

        # Individual knowledge distillation
        teacher_log_probs = F.log_softmax(teacher_logits / self.hparams.teacher_temp, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.hparams.student_temp, dim=1)
        loss_logits_ikd = nn.KLDivLoss(reduction='batchmean', log_target=True)(student_log_probs, teacher_log_probs)

        # Relational knowledge distillation (distance-wise)
        loss_logits_rkd_d = RkdDistance()(student_logits, teacher_logits)

        # Relational knowledge distillation (angle-wise)
        loss_logits_rkd_a = RkdAngle()(student_logits, teacher_logits)

        # MSE loss between cls token and resnet embedding
        loss_cls = nn.MSELoss()(student_cls, teacher_cls) / self.hparams.cls_size

        loss_logits_ce = self.hparams.ce_weight * loss_logits_ce
        loss_logits_ikd = self.hparams.ikd_weight * loss_logits_ikd
        loss_logits_rkd_d = self.hparams.rkd_d_weight * loss_logits_rkd_d
        loss_logits_rkd_a = self.hparams.rkd_a_weight * loss_logits_rkd_a

        loss_logits = loss_logits_ce + loss_logits_ikd + loss_logits_rkd_d + loss_logits_rkd_a

        loss_logits = self.hparams.logits_weight * loss_logits
        loss_cls = self.hparams.cls_weight * loss_cls

        loss = loss_logits + loss_cls

        predictions = student_log_probs.argmax(-1)
        accuracy = torch.where(predictions == labels, 1.0, 0.0)
        accuracy = torch.mean(accuracy)

        base_predictions = teacher_log_probs.argmax(-1)
        base_accuracy = torch.where(base_predictions == labels, 1.0, 0.0)
        base_accuracy = torch.mean(base_accuracy)

        return {
            'loss': loss,
            'loss_logits': loss_logits,
            'loss_logits_ce': loss_logits_ce,
            'loss_logits_rkd_d': loss_logits_rkd_d,
            'loss_logits_rkd_a': loss_logits_rkd_a,
            'loss_logits_ikd': loss_logits_ikd,
            'loss_cls': loss_cls,
            'accuracy': accuracy,
            'base_accuracy': base_accuracy
        }

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()

        metrics = self.common_step(batch, batch_idx)

        self.manual_backward(metrics['loss'])
        optimizer.step()
        optimizer.zero_grad()

        for name, value in metrics.items():
            self.log("training/" + name, value, prog_bar=False, on_epoch=True, batch_size=len(batch))

        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        metrics = self.common_step(batch, batch_idx)

        for name, value in metrics.items():
            self.log("validation/" + name, value, prog_bar=False, on_epoch=True, batch_size=len(batch))

        return metrics['loss']

    def test_step(self, batch, batch_idx):
        metrics = self.common_step(batch, batch_idx)

        for name, value in metrics.items():
            self.log("test/" + name, value, prog_bar=False, on_epoch=True, batch_size=len(batch))

        return metrics['loss']

        return loss

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        params = []
        params.append({'params':self.resnet.parameters(),'lr':self.hparams.backbone_lr})
        params.append({'params':list(self.cls_projection.parameters()) + list(self.final_projection.parameters()),'lr':self.hparams.head_lr})
        optimizer = AdamW(params)
        return optimizer