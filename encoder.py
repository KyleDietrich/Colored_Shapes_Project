import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

if torch.accelerator.is_available():
    our_device = torch.accelerator.current_accelerator().type 
else:
    our_device = "cpu"
our_device="cpu"

print(f"Using {our_device} device")

class Encoder(nn.Module):
    def __init__(self):
        super().__init__() #call super class
        self.flatten = nn.Flatten()
        self.encoding_stack = nn.Sequential(
            nn.Conv2d( in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device=our_device, dtype=None), # 32, 32
            nn.ReLU(),

            nn.Conv2d( in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device=our_device, dtype=None),# 16, 16 
            nn.ReLU(), 

            nn.Conv2d( in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device=our_device, dtype=None),# 8, 8 
            nn.ReLU(),

            nn.Conv2d( in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device=our_device, dtype=None),# 4, 4 
            nn.ReLU(), 

            nn.Conv2d( in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device=our_device, dtype=None),# 2, 2
            nn.ReLU(),

            nn.Conv2d( in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device=our_device, dtype=None),# 2, 2
            nn.ReLU(),

            nn.Flatten(0,2),

            nn.LazyLinear(128)
        )
    def forward(self, x):
        representation = self.encoding_stack(x)
        return representation 
