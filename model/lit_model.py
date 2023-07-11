import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

# import PSNR and SSIM metrics from torchmetrics
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from .AsConvSR import AsConvSR


class LitAsConvSR(pl.LightningModule):
    def __init__(
        self, 
        learning_rate=1e-3, 
        scale_factor=2, 
        device=torch.device('cpu')
    ):
        super().__init__()
        self.lr = learning_rate
        self.model = AsConvSR(scale_factor=scale_factor, device=device)

        self.l1_loss_fn = nn.L1Loss()
        
        # add metrics to monitor during training
        self.train_psnr = PeakSignalNoiseRatio()
        self.val_psnr = PeakSignalNoiseRatio()
        self.train_ssim = StructuralSimilarityIndexMeasure()
        self.val_ssim = StructuralSimilarityIndexMeasure()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_psnr_best = MaxMetric()
        
    def forward(self, x):
        return self.model(x)
    
    def on_train_start(self):
        self.val_psnr_best.reset()
    
    def training_step(self, batch, batch_idx):
        image_lr, image_hr = batch['lr'], batch['hr']
        image_lr = image_lr.to(self.device)
        image_sr = self.forward(image_lr)
        loss = self.l1_loss_fn(image_sr, image_hr)
        psnr = self.train_psnr(image_sr, image_hr)
        ssim = self.train_ssim(image_sr, image_hr)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/ssim", ssim, on_step=False, on_epoch=True, prog_bar=True)
        
        # log images to wandb log 4 times per epoch
        if batch_idx % 500 == 0:
            grid = torchvision.utils.make_grid(torch.cat((image_sr[:1], image_hr[:1]), dim=0))
            self.logger.experiment.add_image('train_images', grid, self.global_step)
        
        return {"loss": loss, "psnr": psnr, "ssim": ssim}
    
    def validation_step(self, batch, batch_idx):
        image_lr, image_hr = batch['lr'], batch['hr']
        image_sr = self.forward(image_lr)
        loss = self.l1_loss_fn(image_sr, image_hr)
        psnr = self.val_psnr(image_sr, image_hr)
        ssim = self.val_ssim(image_sr, image_hr)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ssim", ssim, on_step=False, on_epoch=True, prog_bar=True)
        
        if batch_idx % 500 == 0:
            # show random single images lr, sr, hr in tensorboard
            index = torch.randint(0, image_sr.shape[0], (1,))
            grid = torchvision.utils.make_grid(torch.cat((image_sr[index:index+1], image_hr[index:index+1]), dim=0))
            self.logger.experiment.add_image('val_images', grid, self.global_step)
        
        return {"loss": loss, "psnr": psnr, "ssim": ssim}
    
    def on_validation_epoch_end(self) -> None:
        self.val_psnr_best(self.val_psnr.compute())
        self.log("val/psnr_best", self.val_psnr_best, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        image_lr, image_hr = batch['lr'], batch['hr']
        image_sr = self.forward(image_lr)
        loss = self.l1_loss_fn(image_sr, image_hr)
        psnr = self.val_psnr(image_sr, image_hr)
        ssim = self.val_ssim(image_sr, image_hr)
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        image_lr, image_hr = batch['lr'], batch['hr']
        image_sr = self.forward(image_lr)
        return {'image_lr': image_lr, 'image_sr': image_sr, 'image_hr': image_hr}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5,
                                      amsgrad=False)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200*20)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "step",
                "frequency": 1,
            },
        }
    
