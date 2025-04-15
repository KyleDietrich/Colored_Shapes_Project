# contrastive_experiment.py

import torch
from torchvision import transforms
from dataset import ColoredShapes32
from models import SmallCNN
from training import train_contrastive

def main():
    """
    Main script to run the contrastive experiment
    
    We'll train a 2D embedding CNN with two different augmentation strategies:
      1) Tc (random crop only) -> encourages color-sensitive embeddings.
      2) Ts (random crop + color jitter) -> encourages shape-sensitive embeddings.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Data set with no transforms
    dataset = ColoredShapes32(length=64000, transform=None)
    print("Dataset size:", len(dataset))

    # Instantiate SmallCNN
    model = SmallCNN(out_dim=2).to(device) 

    # Definitions for transformations 
    #   "Tc" -> random crop only
    #   "Ts -> random crop + color jitter
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

    # Color run, for color sensitivity
    print("=== Training: Color-Sensitive Embedding (Tc) ===")
    model_color = SmallCNN(out_dim=2)
    model_color = train_contrastive(
        model=model_color,
        dataset=dataset,
        transform_fn=Tc,
        num_steps=20000,
        batch_size=128,
        lr=1e-3,
        t=2.0,
        print_every=500
    )
    
    torch.save(model_color.state_dict(), "contrastive_model_color.pth")
    print("Model save successfully as contrastive_model_color.pth")

    # Shape run, for shape-sensitivity
    print("=== Training: Shape-Sensitive Embedding (Ts) ===")
    model_shape = SmallCNN(out_dim=2)
    model_shape = train_contrastive(
        model=model_shape,
        dataset=dataset,
        transform_fn=Ts,
        num_steps=20000,
        batch_size=128,
        lr=1e-3,
        t=2.0,
        print_every=500
    )

    torch.save(model_shape.state_dict(), "contrastive_model_shape.pth")
    print("Model save successfully as contrastive_model_shape.pth")

if __name__ == "__main__":
    main()
