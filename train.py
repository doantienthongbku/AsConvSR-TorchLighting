import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, Callback

from model.lit_model import LitAsConvSR
from model.lit_dataset import SRDataModule


def main():
    model = LitAsConvSR(learning_rate=1e-3, scale_factor=2)
    dm = SRDataModule(train_dir='/home/taft/SISR/DIV2K_raw/DIV2K_train_HR',
                      val_dir='/home/taft/SISR/DIV2K_raw/DIV2K_valid_HR',
                      batch_size=2,
                      num_workers=4,
                      crop_size=112,
                      upscale_factor=2,
                      image_format='png',
                      preupsample=False)
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        min_epochs=1,
        max_epochs=10,
        precision=16,
        callbacks=[EarlyStopping(monitor="val_loss")],
    )

    trainer.fit(model, dm)
    trainer.validate(model, dm)

if __name__ == "__main__":
    main()