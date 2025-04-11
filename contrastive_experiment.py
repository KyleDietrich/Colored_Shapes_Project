# contrastive_experiment.py

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import ColoredShapes32

# Definitions for transformations 
Tc = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

Ts = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
    transforms.ColorJitter(
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.5
    ),
    transforms.ToTensor()
])

"""
A small CNN that outputs 2D embeddings from 32x32 inputs.
We apply L2 normalization at the end so the embedding lie on the unit circle.
Total "6 layers" 3 downsampling, 2 refinement, final flatten+fc.
"""
class SmallCNN(nn.Module):
    def __init__(self, out_dim=2):
        super().__init__()

        # Convolutional feature extractor
        self.conv_net = nn.Sequential(
            # Layer 1: input (3,32,32) -> (32,16,16)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True))

            # Layer 2: (32,16,16) -> (64,8,8)

            # Layer 3: (64,8,8) -> (128,4,4)

            # Layer 4: refine (128,4,4)

            # Layer 5: refine (128,4,4)

            # Then we have feature map -> 128*4*4 = 2048
            # Map out with FC layer
    
    """
    Foward pass:
        x: (batch_size, 3, 32, 32)
    returns:
        A 2D embedding for each sample, normalized to lie on unit circle
    """
    def forward(self, x):
        # Extract features
        x = self.conv_net(x) 

        # Flatten
        x = x.view(x.size(0), -1)

        # Map to 2D
        x = self.fc(x) # (batch_size, 2)

        # Normalize
        x = F.normalize(x, dim=1)

        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data set with no transforms
    dataset = ColoredShapes32(length=1000, transform=None)

    # Simple loader to test later on
    basic_tranform = transforms.ToTensor()

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # INstantiate SmallCNN
    model = SmallCNN(out_dim=2).to(device) 
    model.eval() # forward test, no training

    # test forward pass
    # IDK WHAT IM DOING I HOPE THIS WORKS


if __name__ == "__main__":
    main()
