from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import torch

from .dataset import SRDataset


class SRDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, batch_size, num_workers, crop_size, 
                 upscale_factor, image_format, preupsample):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.image_format = image_format
        self.preupsample = preupsample
        
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
    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    