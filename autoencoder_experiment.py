# autoecoder_experiment.py

from dataset import ColoredShapes32
import dataset
from training import train_autoencoder
from models import ConvAutoencoder

def main():
    data = ColoredShapes32(length=64000)
    model = ConvAutoencoder(latent_dim=128)
    train_autoencoder(model, dataset, num_epochs=40)


if __name__ == "__main__":
    main()