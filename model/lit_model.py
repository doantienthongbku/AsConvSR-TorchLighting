import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .AsConvSR import AsConvSR
from .losses import ContentLoss


class LitAsConvSR(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, scale_factor=2):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = AsConvSR(scale_factor=scale_factor)
        
        self.l1_loss_fn = nn.L1Loss()
        self.content_loss_fn = ContentLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        image_lr, image_hr = batch['lr'], batch['hr']
        image_sr = self.forward(image_lr)
        l1_loss = self.l1_loss_fn(image_sr, image_hr)
        content_loss = self.content_loss_fn(image_sr, image_hr)
        loss = l1_loss + 0.001 * content_loss
        
        return {'loss': loss, 'l1_loss': l1_loss, 'content_loss': content_loss}
    
    def validation_step(self, batch, batch_idx):
        image_lr, image_hr = batch['lr'], batch['hr']
        image_sr = self.forward(image_lr)
        l1_loss = self.l1_loss_fn(image_sr, image_hr)
        content_loss = self.content_loss_fn(image_sr, image_hr)
        loss = l1_loss + 0.001 * content_loss
        
        return {'val_loss': loss, 'val_l1_loss': l1_loss, 'val_content_loss': content_loss}
    
    def test_step(self, batch, batch_idx):
        image_lr, image_hr = batch['lr'], batch['hr']
        image_sr = self.forward(image_lr)
        l1_loss = self.l1_loss_fn(image_sr, image_hr)
        content_loss = self.content_loss_fn(image_sr, image_hr)
        loss = l1_loss + 0.001 * content_loss
        
        return {'test_loss': loss, 'test_l1_loss': l1_loss, 'test_content_loss': content_loss}
    
    def predict_step(self, batch, batch_idx):
        image_lr, image_hr = batch['lr'], batch['hr']
        image_sr = self.forward(image_lr)
        return {'image_lr': image_lr, 'image_sr': image_sr, 'image_hr': image_hr}
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
