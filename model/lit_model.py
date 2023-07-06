import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

from .AsConvSR import AsConvSR
from .losses import ContentLoss, PSNR


class LitAsConvSR(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, scale_factor=2, device=torch.device('cpu')):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = AsConvSR(scale_factor=scale_factor, device=device)
        
        self.l1_loss_fn = nn.L1Loss()
        self.psnr_fn = PSNR()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        image_lr, image_hr = batch['lr'], batch['hr']
        image_lr = image_lr.to(self.device)
        image_sr = self.forward(image_lr)
        loss = self.l1_loss_fn(image_sr, image_hr)
        psnr = self.psnr_fn(image_sr, image_hr)
        
        self.log_dict({'train_loss': loss, 'train_psnr': psnr},
                      on_step=False, on_epoch=True, prog_bar=True)
        
        if batch_idx % 1000 == 0:
            # show single images lr, sr, hr in tensorboard
            grid = torchvision.utils.make_grid(torch.cat((image_sr[:1], image_hr[:1]), dim=0))
            self.logger.experiment.add_image('train_images', grid, self.global_step)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        image_lr, image_hr = batch['lr'], batch['hr']
        image_sr = self.forward(image_lr)
        loss = self.l1_loss_fn(image_sr, image_hr)
        psnr = self.psnr_fn(image_sr, image_hr)
        
        self.log_dict({'val_loss': loss, 'val_psnr': psnr},
                      on_step=False, on_epoch=True, prog_bar=True)
        
        if batch_idx % 1000 == 0:
            # show single images lr, sr, hr in tensorboard
            grid = torchvision.utils.make_grid(torch.cat((image_sr[:1], image_hr[:1]), dim=0))
            self.logger.experiment.add_image('val_images', grid, self.global_step)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        image_lr, image_hr = batch['lr'], batch['hr']
        image_sr = self.forward(image_lr)
        loss = self.l1_loss_fn(image_sr, image_hr)
        psnr = self.psnr_fn(image_sr, image_hr)
        
        self.log_dict({'test_loss': loss, 'test_psnr': psnr},
                      on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        image_lr, image_hr = batch['lr'], batch['hr']
        image_sr = self.forward(image_lr)
        return {'image_lr': image_lr, 'image_sr': image_sr, 'image_hr': image_hr}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.9999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]
    
