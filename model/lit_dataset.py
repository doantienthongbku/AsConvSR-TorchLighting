from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import torch

from .dataset import SRDataset


class SRDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_dir="data/train",
        val_dir="data/val",
        batch_size=32,
        num_workers=4,
        crop_size=256, 
        upscale_factor=2,
        image_format="png",
        preupsample=False,
        prefetch_factor=16,
    ) -> None:
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.image_format = image_format
        self.preupsample = preupsample
        self.prefetch_factor = prefetch_factor
        
        self.dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "pin_memory": True,
        }
        
    def setup(self, stage):
        self.train_ds = SRDataset(
            images_dir=self.train_dir,
            crop_size=self.crop_size,
            upscale_factor=self.upscale_factor,
            mode="train",
            image_format=self.image_format,
            preupsample=self.preupsample
        )
        self.valid_ds = SRDataset(
            images_dir=self.val_dir,
            crop_size=self.crop_size,
            upscale_factor=self.upscale_factor,
            mode="valid",
            image_format=self.image_format,
            preupsample=self.preupsample
        )
        
        # print information of dataset
        print("============================================================")
        print(f"Train dataset: {len(self.train_ds)} images")
        print(f"Valid dataset: {len(self.valid_ds)} images")
        print("============================================================")
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True,**self.dataloader_kwargs)
    
    def val_dataloader(self):
        return DataLoader(self.valid_ds, shuffle=False, **self.dataloader_kwargs)