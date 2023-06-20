import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlModule(nn.Module):
    def __init__(self, in_channels, out_channels, temperature=30, ratio=4, E=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.E = E
        self.temperature = temperature
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        assert in_channels > ratio
        hidden_channels = in_channels // ratio
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, E * out_channels, kernel_size=1, bias=False)
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        coeff = self.avgpool(x) # bs, channels, 1, 1
        coeff = self.net(coeff).view(x.shape[0], -1) # bs, E * out_channels
        coeff = coeff.view(coeff.shape[0], self.out_channels, self.E)
        coeff = F.softmax(coeff / self.temperature, dim=2)
        return coeff
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    
class AssembledBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 dilation=1, groups=1, bias=True, E=4, temperature=30, ratio=4, device=torch.device('cpu')):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride=stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.E = E
        self.temperature = temperature
        self.ratio = ratio
        self.device = device
        
        self.control_module = ControlModule(in_channels, out_channels, temperature, ratio, E)
        self.weight1 = nn.Parameter(torch.randn(E, out_channels, in_channels // groups, kernel_size, kernel_size), requires_grad=True)
        self.weight2 = nn.Parameter(torch.randn(E, out_channels, out_channels // groups, kernel_size, kernel_size), requires_grad=True)
        self.weight3 = nn.Parameter(torch.randn(E, out_channels, out_channels // groups, kernel_size, kernel_size), requires_grad=True)
        
        if bias:
            self.bias1 = nn.Parameter(torch.randn(E, out_channels), requires_grad=True)
            self.bias2 = nn.Parameter(torch.randn(E, out_channels), requires_grad=True)
            self.bias3 = nn.Parameter(torch.randn(E, out_channels), requires_grad=True)
    
    def forward(self, x):
        bs, in_channels, h, w = x.shape
        coeff = self.control_module(x) # bs, out_channels, E
        weight1 = self.weight1.view(self.E, self.out_channels, -1) # E, out_channels, in_channels // groups * k * k
        weight2 = self.weight2.view(self.E, self.out_channels, -1) # E, out_channels, in_channels // groups * k * k
        weight3 = self.weight3.view(self.E, self.out_channels, -1) # E, out_channels, in_channels // groups * k * k
        x = x.view(1, bs * in_channels, h, w) # 1, bs * in_channels, h, w
        
        aggregate_weight1 = torch.zeros(bs, self.out_channels, self.in_channels // self.groups, self.kernel_size, 
                                        self.kernel_size).to(self.device) # bs, out_channels, in_channels // groups, k, k
        aggregate_weight2 = torch.zeros(bs, self.out_channels, self.out_channels // self.groups, self.kernel_size,
                                        self.kernel_size).to(self.device) # bs, out_channels, in_channels // groups, k, k
        aggregate_weight3 = torch.zeros(bs, self.out_channels, self.out_channels // self.groups, self.kernel_size,
                                        self.kernel_size).to(self.device) # bs, out_channels, in_channels // groups, k, k
        
        for i in range(self.out_channels):
            sub_coeff = coeff[:, i, :] # bs, E
            sub_weight1 = weight1[:, i, :] # E, in_channels // groups * k * k
            sub_weight2 = weight2[:, i, :] # E, in_channels // groups * k * k
            sub_weight3 = weight3[:, i, :] # E, in_channels // groups * k * k
            
            sub_aggregate_weight1 = torch.mm(sub_coeff, sub_weight1) # bs, in_channels // groups * k * k
            aggregate_weight1[:, i, :, :, :] = sub_aggregate_weight1.view(bs, self.in_channels // self.groups, 
                                                                          self.kernel_size, self.kernel_size)
            
            sub_aggregate_weight2 = torch.mm(sub_coeff, sub_weight2) # bs, in_channels // groups * k * k
            aggregate_weight2[:, i, :, :, :] = sub_aggregate_weight2.view(bs, self.out_channels // self.groups,
                                                                          self.kernel_size, self.kernel_size)
            
            sub_aggregate_weight3 = torch.mm(sub_coeff, sub_weight3) # bs, in_channels // groups * k * k
            aggregate_weight3[:, i, :, :, :] = sub_aggregate_weight3.view(bs, self.out_channels // self.groups,
                                                                          self.kernel_size, self.kernel_size)
            
        aggregate_weight1 = aggregate_weight1.view(bs * self.out_channels, self.in_channels // self.groups, 
                                                   self.kernel_size, self.kernel_size)
        aggregate_weight2 = aggregate_weight2.view(bs * self.out_channels, self.out_channels // self.groups,
                                                   self.kernel_size, self.kernel_size)
        aggregate_weight3 = aggregate_weight3.view(bs * self.out_channels, self.out_channels // self.groups,
                                                   self.kernel_size, self.kernel_size)
        
        aggregate_weight1 = aggregate_weight1.to(self.device)
        aggregate_weight2 = aggregate_weight2.to(self.device)
        aggregate_weight3 = aggregate_weight3.to(self.device)
        
        out = F.conv2d(x, weight=aggregate_weight1, bias=None, stride=self.stride, padding=self.padding, 
                       dilation=self.dilation, groups=self.groups * bs)   # bs * out_channels, in_channels // groups, h, w
        out = out.view(1, self.out_channels * bs, out.shape[2], out.shape[3])
        out = F.conv2d(out, weight=aggregate_weight2, bias=None, stride=self.stride, padding=self.padding,
                       dilation=self.dilation, groups=self.groups * bs)  # bs * out_channels, in_channels // groups, h, w
        out = out.view(1, self.out_channels * bs, out.shape[2], out.shape[3])
        out = F.conv2d(out, weight=aggregate_weight3, bias=None, stride=self.stride, padding=self.padding,
                       dilation=self.dilation, groups=self.groups * bs)  # bs * out_channels, in_channels // groups, h, w
        out = out.view(bs, self.out_channels, out.shape[2], out.shape[3])
        
        return out
    
    
def test_control_module():
    x = torch.randn(1, 32, 64, 64)
    net = ControlModule(32, 64)
    print(net(x).shape)


def test_assembled_block():
    x = torch.randn(2, 32, 64, 64)
    net = AssembledBlock(32, 64, 3, 1, 1, groups=1)
    print(net(x).shape)
    
if __name__ == '__main__':
    test_assembled_block()
