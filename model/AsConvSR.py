import torch
import torch.nn as nn
import torch.nn.functional as F

from Assembled_conv import AssembledBlock

# Implementation of AsConvSR 
class AsConvSR(nn.Module):
    def __init__(self, scale_factor: int=2):
        super(AsConvSR, self).__init__()
        self.scale_factor = scale_factor
        
        self.pixelUnShuffle = nn.PixelUnshuffle(2)
        self.conv1 = nn.Conv2d(12, 48, kernel_size=3, stride=1, padding=1)
        self.assemble = AssembledBlock(48, 48, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.pixelShuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.pixelUnShuffle(x)
        out2 = self.conv1(out1)
        out3 = self.assemble(out2)
        out4 = self.conv2(out3)
        out5 = self.pixelShuffle(out4)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        out6 = self.pixelUnShuffle(x)
        out7 = torch.add(out5, out6)
        out8 = self.pixelShuffle(out7)
        
        return out8
        
if __name__ == '__main__':
    from torchsummary import summary
    model = AsConvSR()
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print(out.shape)    # torch.Size([1, 3, 256, 256])
    
    summary(model, (3, 1080, 1920), device='cpu')