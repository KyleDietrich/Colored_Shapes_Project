# training.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from loss_functions import alignment_loss, uniformity_loss
from torchvision import transforms

"""
    train_autoencoder:
    Trains a convolutional autoencoder to reconstruct images from the dataset.
    We use MSELoss between the original images and the reconstructed outputs.

    We do 40 epochs on a dataset of 64,000 images with batch_size=128,
    which is ~20k training steps total.

    Args:
        model: The autoencoder (encoder+decoder) to train.
        dataset: A PyTorch Dataset of (image, label) pairs.
        num_epochs: Number of epochs to iterate over the dataset. 
        batch_size: Mini-batch size.
        lr: Learning rate for Adam optimizer.
        print_every: Print the loss every N epochs. 
    
    Returns:
        model: The trained autoencoder, updated in-place.
"""
def train_autoencoder(
    model,
    dataset,
    num_epochs=40,
    batch_size=128,
    lr=1e-3,
    print_every=1
):
    # training loop go here

    return model

"""
    contrastive_train:
    This function trains a small CNN that outputs 2D embeddings
    using a contrastive learning approach with two main losses:
      1) alignment_loss to pull embeddings of the same sample's augmented views together
      2) uniformity_loss to push all embeddings apart, avoiding collapse.

    We repeatedly sample mini-batches from the given dataset, transform each
    image twice (forming pairs), compute embeddings, and minimize the sum
    of alignment and uniformity losses.

    Args:
        model: PyTorch model that outputs embeddings.
        dataset: PyTorch Dataset (e.g., ColoredShapes32).
        transform_fn: transform to apply (e.g. random crop, color jitter)
        num_steps: total number of training steps.
        batch_size: how many samples per batch.
        lr: learning rate for the Adam optimizer.
        t: temperature in uniformity_loss (controls how strongly embeddings are pushed apart).
        print_every: frequency (in steps) to print the loss.

    Returns:
        model: The trained model
"""
def contrastive_train(
    model, 
    dataset, 
    transfom_fn,
    num_steps=20000, 
    batch_size=128, 
    lr=1e-3, 
    t=2.0, 
    print_every=500
):
    # training loop go here

    return model