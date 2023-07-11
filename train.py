import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
# from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.strategies import DeepSpeedStrategy

from model import LitAsConvSR
from model import SRDataModule

torch.set_float32_matmul_precision("medium")    # https://pytorch.org/docs/stable/notes/cuda.html#torch-backends

def main():
    # strategy = DeepSpeedStrategy()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LitAsConvSR(learning_rate=5e-4, scale_factor=2, device=device)
    dm = SRDataModule(
        train_dir='../DIV2K_raw/DIV2K_train_HR',
        val_dir='../DIV2K_raw/DIV2K_valid_HR',
        batch_size=2,
        num_workers=4,
        crop_size=256,
        upscale_factor=2,
        image_format='png',
        preupsample=False,
        prefetch_factor=16
    )
    dm.setup(stage="fit")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        verbose=True,
        monitor="val_loss",
        mode="min",
        dirpath="checkpoints/",
        filename="AsConvSR-{epoch:02d}-{val_loss:.2f}-{psnr:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # wandb_logger = WandbLogger(project="AsConvSR", log_model=True)
    tb_logger = TensorBoardLogger("logs/", name="AsConvSR")
    
    trainer = Trainer(
        # strategy=strategy,
        accelerator="auto",
        devices="auto",
        min_epochs=1,
        max_epochs=20,
        precision="16-mixed",
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor, EarlyStopping(monitor="val/loss", patience=10)],
    )

    trainer.fit(model, dm)
    trainer.validate(model, dm)

if __name__ == "__main__":
    main()