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

class Decoder(nn.Module):
    def __init__(self):
        super().__init__() #call super class
        self.flatten = nn.Flatten()
        self.decoding_stack = nn.Sequential(
                #nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
                nn.ConvTranspose2d(in_channels=4,out_channels= 3, kernel_size=3, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None),
                nn.ReLU(True),
        )
    def forward(self, x):
        x = x.view(4, 4, 4, 2)
        representation = self.decoding_stack(x)
        print(representation.shape)
        return representation 
