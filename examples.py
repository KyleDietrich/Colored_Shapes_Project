# examples.py

import torch
from torchvision.utils import make_grid
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import ColoredShapes32

def main():
    ds = ColoredShapes32(length=12, transform=transforms.ToTensor())

    imgs = torch.stack([ds[i][0] for i in range(12)])

    grid = make_grid(imgs, nrow=4, padding=2)

    plt.figure(figsize=(6, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.tight_layout()

    out_path = "Example_Shapes"
    plt.savefig(out_path, dpi=300)
    print("saved")

if __name__ == "__main__":
    main()