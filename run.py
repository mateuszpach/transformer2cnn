from itertools import product

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from CUB200DataModule import CUB200DataModule
from DistilledResnetModel import DistilledResNetModel

configs = {
    "teacher_temp": [0.06],
    "student_temp": [0.1],
    "num_labels": [200],
    "cls_size": [384],
    "weights": [
        # w rkd
        # {"cls_weight": 3, "logits_weight": 1, "ce_weight": 1/50, "ikd_weight": 1/30, "rkd_d_weight": 3, "rkd_a_weight": 4.5}
        # w/o rkd
        {"cls_weight": 3, "logits_weight": 1, "ce_weight": 1/25, "ikd_weight": 1/12, "rkd_d_weight": 0., "rkd_a_weight": 0.}
    ],
    "backbone_lr": [5e-6],
    "head_lr": [5e-4]
}

configs = product(
    *[zip([name] * len(values), values) for name, values in configs.items()]
)

for hyperparams in configs:
    hyperparams = dict(hyperparams)

    datamodule = CUB200DataModule()

    resnet = torch.hub.load('facebookresearch/dino:main',
                            'dino_resnet50')  # dino has last layer removed, set replace_fc to False

    model = DistilledResNetModel(resnet,
                                 teacher_temp=hyperparams["teacher_temp"],
                                 student_temp=hyperparams["student_temp"],
                                 num_labels=hyperparams["num_labels"],
                                 cls_size=hyperparams["cls_size"],
                                 cls_weight=hyperparams["weights"]["cls_weight"],
                                 logits_weight=hyperparams["weights"]["logits_weight"],
                                 ce_weight=hyperparams["weights"]["ce_weight"],
                                 ikd_weight=hyperparams["weights"]["ikd_weight"],
                                 rkd_d_weight=hyperparams["weights"]["rkd_d_weight"],
                                 rkd_a_weight=hyperparams["weights"]["rkd_a_weight"],
                                 backbone_lr=hyperparams["backbone_lr"],
                                 head_lr=hyperparams["head_lr"],
                                 replace_fc=False)

    wandb_logger = WandbLogger(
        project="transformer2cnn",
        entity="drug_repositioning",
        save_dir="logs",
        tags=[],
        reinit=True,
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        strict=False,
        verbose=False,
        mode='max'
    )
    trainer = Trainer(accelerator='auto',
                      callbacks=[early_stop_callback],
                      log_every_n_steps=5,
                      max_epochs=500,
                      logger=wandb_logger,
                      # fast_dev_run=1
                      )
    trainer.fit(model=model, datamodule=datamodule)

    # trainer.test()
    trainer.save_checkpoint('final.ckpt')

    wandb_logger.experiment.finish()
    wandb_logger.finalize("success")
