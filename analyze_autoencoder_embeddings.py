# analyze_autoencoder_embeddings

from matplotlib.cm import get_cmap
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.modules import batchnorm
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from dataset import ColoredShapes32
from models import ConvAutoencoder

def extract_embeddings(model, dataset, device, n_points=None):
    """
    Extract latent representations (embeddings) using the encoder part 
    of the autoencoder.
    """
    model.eval()
    encoder = model.encoder

    embeddings = []
    labels = [] # each label is a tuple (color_label, shape_label)

    indices = np.arange(len(dataset))
    if n_points is not None and n_points < len(dataset):
        indices = np.random.choice(len(dataset), size=n_points, replace=False)

    with torch.no_grad():
        for idx in indices:
            x, label = dataset[idx] # x = tensor (3, 32, 32)
            x = x.unsqueeze(0).to(device) # add batch dimension (1, 3, 32, 32)
            z = encoder(x) # get latent vector: (1, 128)
            embeddings.append(z.cpu().numpy()[0])
            labels.append(label)

    embeddings = np.array(embeddings)
    return embeddings, labels



def plot_embeddings_pca(embeddings, labels, label_idx=0, title="Embeddings (PCA)"):
    """
    Use PCA to reduce the dimension of embeddings to 2D, and plot them.
    """
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    label_values = np.array([lb[label_idx] for lb in labels])
    num_classes = int(label_values.max() + 1)

    cmap = plt.colormaps.get_cmap("tab10")
    colors = [cmap(i) for i in range(num_classes)]
    markers = ["o", "^", "s", "P", "X", "D", "v", "*"]
    
    plt.figure(figsize=(8,6))
    for cls in range(num_classes):
        mask = label_values == cls
        plt.scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1],
            color=colors[cls], marker=markers[cls % len(markers)], 
            label=f"class {cls}", s=35, alpha=0.8, 
            edgecolors="k", linewidths=0.2
        )

    plt.title(title)
    plt.legend(loc="best", fontsize=8, markerscale=0.8)
    plt.tight_layout(); 
    plt.show()


def nearest_neighbors_visualization(model, dataset, device, n_neighbors=5):
    """
    For a randomly selected query image, find its nearest neighbors in the latent space 
    and visualize them.
    """
    model.eval()
    encoder = model.encoder

    # Randomly select query index
    query_idx = np.random.choice(len(dataset))
    query_img, query_label = dataset[query_idx]

    # randomly select query image
    query_tensor = query_img.unsqueeze(0).to(device)
    with torch.no_grad():
        query_embedding = encoder(query_tensor).cpu().numpy()[0]

    # Collect embeddings
    embeddings, _ = extract_embeddings(model, dataset, device, n_points=1000)

    # Compute Euclidean distance from query_embeddings to all other embeddings
    dists = np.linalg.norm(embeddings - query_embedding, axis=1)
    # get indices of nearest neighbors
    nn_indices = np.argsort(dists)[1:n_neighbors+1]

    # visualize query and nearest neighbors
    plt.figure(figsize=(15, 3))
    # plot query image first
    ax = plt.subplot(1, n_neighbors+1, 1)
    plt.imshow(query_img.permute(1,2,0).numpy())
    plt.title("Query")
    plt.axis("off")

    for i, idx in enumerate(nn_indices):
        neighbor_img, neighbor_label = dataset[idx]
        ax = plt.subplot(1, n_neighbors+1, i+2)
        plt.imshow(neighbor_img.permute(1,2,0).numpy())
        plt.title(f"NN {i+1}")
        plt.axis("off")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Analyzing on device:", device)

    transform = transforms.ToTensor()
    dataset = ColoredShapes32(length=64000, transform=transform)
    print("Dataset size:", len(dataset))

    model = ConvAutoencoder(latent_dim=128).to(device)

    autoencoder_model = "Autoencoder_model.pth"
    model.load_state_dict(torch.load(autoencoder_model, map_location=device))
    model.eval()
    print("Loaded model from", autoencoder_model)

    # Embeddings and PCA visualization
    n_points = 1000
    embeddings, labels = extract_embeddings(model, dataset, device, n_points=n_points)
    labels = np.array(labels)
    print(np.bincount(labels[:,1].astype(int)))

    plot_embeddings_pca(embeddings, labels, label_idx=0, title="Embeddings by Color Label")
    plot_embeddings_pca(embeddings, labels, label_idx=1, title="Embeddings by Shape Label")

    # Nearest Neighbors Visualization
    nearest_neighbors_visualization(model, dataset, device, n_neighbors=5)

    # 1-NN Classification on embeddings
    all_embeddings, all_labels = extract_embeddings(model, dataset, device, n_points=3000)
    all_labels = np.array(all_labels)

    # shuffle 
    perm = np.random.permutation(len(all_embeddings))
    all_embeddings = all_embeddings[perm]
    all_labels = all_labels[perm]

    # train and test split, 80% train, 20% test
    split_idx = int(0.8 * len(all_embeddings))
    X_train, X_test = all_embeddings[:split_idx], all_embeddings[split_idx:]
    y_train_color, y_test_color = all_labels[:split_idx, 0], all_labels[split_idx:, 0]
    y_train_shape, y_test_shape = all_labels[:split_idx, 1], all_labels[split_idx:, 1]

    # train simple 1-NN classifier from scikit-learn
    from sklearn.neighbors import KNeighborsClassifier
    knn_color = KNeighborsClassifier(n_neighbors=1)
    knn_color.fit(X_train, y_train_color)
    color_preds = knn_color.predict(X_test)
    from sklearn.metrics import accuracy_score
    color_acc = accuracy_score(y_test_color, color_preds)

    knn_shape = KNeighborsClassifier(n_neighbors=1)
    knn_shape.fit(X_train, y_train_shape)
    shape_preds = knn_shape.predict(X_test)
    shape_acc = accuracy_score(y_test_shape, shape_preds)

    print("1-NN Color Classification Accuracy: {:.2f}%".format(color_acc * 100))
    print("1-NN Shape Classification Accuracy: {:.2f}%".format(shape_acc * 100))

if __name__ == "__main__":
    main()