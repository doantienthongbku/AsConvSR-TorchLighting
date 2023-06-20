import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DeepSpeedStrategy

from model.lit_model import LitAsConvSR
from model.lit_dataset import SRDataModule

torch.set_float32_matmul_precision("medium")    # https://pytorch.org/docs/stable/notes/cuda.html#torch-backends

def main():
    # strategy = DeepSpeedStrategy()
    logger = TensorBoardLogger("tb_logs", name="AsConvSR_model_v0")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = LitAsConvSR(learning_rate=5e-4, scale_factor=2, device=device)
    dm = SRDataModule(train_dir='/home/taft/SISR/DIV2K_small/train_HR',
                      val_dir='/home/taft/SISR/DIV2K_small/valid_HR',
                      batch_size=2,
                      num_workers=4,
                      crop_size=256,
                      upscale_factor=2,
                      image_format='png',
                      preupsample=False)
    trainer = Trainer(
        # strategy=strategy,
        logger=logger,
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