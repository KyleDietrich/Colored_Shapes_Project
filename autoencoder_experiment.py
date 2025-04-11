# autoecoder_experiment.py

from dataset import ColoredShapes32
import dataset
from training import train_autoencoder

def main():
    data = ColoredShapes32(length=64000)

    # model go here

    # train autoencoder
    trained_model = train_autoencoder(
        model=model,
        dataset=dataset,
        num_epochs=40,
        batch_size=128,
        lr=1e-3,
        print_every=1
    )

if __name__ == "__main__":
    main()