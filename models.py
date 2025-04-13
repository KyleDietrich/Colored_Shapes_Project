# models.py

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    """
    Small Convolutional Network that outputs 2D embeddings.
    
    The architecture:
      - 5 convolutional layers:
          3 strided convs to downsample from 32x32 -> 4x4
          2 conv layers at 4x4 (no downsampling)
      - Flatten + fully connected to produce out_dim 
      - L2-normalize the output so points lie on the unit circle.
    """
    def __init__(self, out_dim=2):
        super().__init__()
        # Convolutional feature extractor
        self.conv_net = nn.Sequential(
            # Layer 1: input (3,32,32) -> (32,16,16)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Layer 2: (32,16,16) -> (64,8,8)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Layer 3: (64,8,8) -> (128,4,4)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Layer 4: refine (128,4,4)
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # Layer 5: refine (128,4,4)
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        # Then we have feature map -> 128*4*4 = 2048
        # Map out with FC layer
        self.fc = nn.Linear(128*4*4, out_dim)

    def forward(self, x):
        """
        Foward pass:
            x: (batch_size, 3, 32, 32)
        returns:
            A 2D embedding for each sample, normalized to lie on unit circle
        """
        # Extract features
        x = self.conv_net(x) 

        # Flatten
        x = x.view(x.size(0), -1)

        # Map to 2D
        x = self.fc(x) # (batch_size, 2)

        # Normalize
        x = F.normalize(x, dim=1)
        return x

class ConvEncoder(nn.Module):
    """
    Convolutional Encoder for 32x32 RGB images, used in the ConvAutoencoder.
    - 6 total conv layers: 4 downsampling (stride=2) + 2 "refinement" layers.
    - Outputs a latent vector of size `latent_dim`.
    """

    # TODO: model for encoder here

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch_size, 3, 32, 32)
        Returns:
            Tensor: shape (batch_size, latent_dim) (e.g. 128)
        """
        # TODO

        return x

class ConvDecoder(nn.Module):
    """
    Convolutional Decoder for 32x32 RGB images, used in the ConvAutoencoder.
    Mirrors the encoder structure in reverse:
    - Start from latent_dim -> expand to (256,2,2) with a fully connected layer
    - 6 total layers of ConvTranspose2d to upsample back to (3,32,32).
    """

    # TODO: model here for decoder

    def forward(self, z):
        """
        Args:
            z (Tensor): shape (batch_size, latent_dim) e.g. 128
        Returns:
            Tensor: shape (batch_size, 3, 32, 32)
        """
        # TODO 

        return z

class ConvAutoencoder(nn.Module):
    """
    Full Convolutional Autoencoder that combines ConvEncoder + ConvDecoder.
    - Input: 32x32 RGB
    - Latent: 128 dim (configurable)
    - Output: 32x32 RGB
    """

    # TODO: Combine encoder and decoder

    def forward(self, x):
        """
        Autoencoder forward pass:
          1) Encode x -> z
          2) Decode z -> reconstruction
        """

        return reconstruction
