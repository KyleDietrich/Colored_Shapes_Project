# visualize_embeddings.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import ColoredShapes32
from models import SmallCNN

def get_embeddings(model, dataset, n_points=2000, device=None):
    """
    Computes embeddings for a random subset of the dataset.
    
    Args:
        model: Trained model that outputs 2D embeddings.
        dataset: The dataset from which to sample images.
        n_points: Number of samples to visualize. 
        device: Device on which computations are performed.
    
    Returns:
        embeddings: Array of shape (n_points, 2) with the embeddings.
        colors: Array of color labels corresponding to each embedding.
        shapes: Array of shape labels corresponding to each embedding.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval() # evaluation mode
    embeddings = []
    color_labels = []
    shape_labels = []

    # Randomly choose indices from the dataset
    indices = np.random.choice(len(dataset), size=n_points, replace=False)

    with torch.no_grad():
        for idx in indices:
            img, (c_label, s_label) = dataset[idx]

            # Batch dimension: (3,32,32) -> (1,3,32,32)
            img = img.unsqueeze(0).to(device)

            # Forward pass through the model
            emb = model(img) # we want (1,2)
            embeddings.append(emb.cpu().numpy()[0])

            color_labels.append(c_label)
            shape_labels.append(s_label)

    embeddings = np.array(embeddings)
    color_labels = np.array(color_labels)
    shape_labels = np.array(shape_labels)
    return embeddings, color_labels, shape_labels


def plot_embeddings(embeddings, labels, title="2D Embeddings", label_type="color"):
    """
    Plots 2D embeddings using matplotlib.
    
    Args:
        embeddings (np.ndarray): Array of shape with the embeddings.
        labels (np.ndarray): Array of labels for each embedding.
        title (str): Title of the plot.
        label_type (str): Which label type is being used 
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                          c=labels, cmap="viridis", alpha=0.7)
    plt.title(title)
    plt.xlabel("Dimesion 1")
    plt.ylabel("Dimension 2")
    plt.axis("equal")
    plt.colorbar(scatter, label=label_type)
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Visualizing embeddings on device:", device)

    # Create dataset
    dataset = ColoredShapes32(length=64000, transform=transforms.ToTensor())

    model = SmallCNN(out_dim=2).to(device)

    # Instantiate trained model or a checkpoint
    saved_model = "contrastive_model.pth"
    model.load_state_dict(torch.load(saved_model, map_location=device))
    model.eval()
    print("Loaded saved model:", saved_model)

    embeddings, color_labels, shape_labels = get_embeddings(model, dataset, n_points=2000, device=device)

    plot_embeddings(embeddings, color_labels, title="Embeddings by Color Label", label_type="color")

    plot_embeddings(embeddings, shape_labels, title="Embeddings by Shape Label", Label_type="shape")


if __name__ == "__main__":
    main()

