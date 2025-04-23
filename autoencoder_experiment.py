# autoecoder_experiment.py

import torch
from dataset import ColoredShapes32
from training import train_autoencoder
from models import ConvAutoencoder

def main():
    data = ColoredShapes32(length=64000)
    model = ConvAutoencoder(latent_dim=128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = model.to(device)

    trained_model = train_autoencoder(model, data, num_epochs=40, batch_size=128, lr=1e-3, num_workers=4)

    torch.save(trained_model.state_dict(), "Autoencoder_model.pth")
    print("Autoencoder_model saved succesfully")


if __name__ == "__main__":
    main()