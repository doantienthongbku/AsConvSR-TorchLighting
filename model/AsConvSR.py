import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dynamic_conv import DynamicConv

# Implementation of AsConvSR 
class AsConvSR(nn.Module):
    def __init__(self, scale_factor: int=2):
        super(AsConvSR, self).__init__()
        self.scale_factor = scale_factor
        
        self.pixelUnShuffle = nn.PixelUnshuffle(2)
        self.conv1 = nn.Conv2d(12, 48, kernel_size=3, stride=1, padding=1)
        self.dymamic1 = DynamicConv(48, 48, kernel_size=3, stride=1, padding=1, K=4, temprature=30, ratio=4)
        self.dymamic2 = DynamicConv(48, 48, kernel_size=3, stride=1, padding=1, K=4, temprature=30, ratio=4)
        self.dymamic3 = DynamicConv(48, 48, kernel_size=3, stride=1, padding=1, K=4, temprature=30, ratio=4)
        self.conv2 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.pixelShuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.pixelUnShuffle(x)
        out2 = self.conv1(out1)
        out3 = self.dymamic1(out2)
        out4 = self.dymamic2(out3)
        out5 = self.dymamic3(out4)
        out6 = self.conv2(out5)
        out7 = self.pixelShuffle(out6)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        out8 = self.pixelUnShuffle(x)
        out9 = torch.add(out7, out8)
        out10 = self.pixelShuffle(out9)
        
        return out10
        
if __name__ == '__main__':
    from torchsummary import summary
    model = AsConvSR()
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print(out.shape)    # torch.Size([1, 3, 256, 256])
    
    # summary(model, (3, 1080, 1920), device='cpu')