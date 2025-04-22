# training.py

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from loss_functions import alignment_loss, uniformity_loss

# ============================================================================
# Autoencoder Training Function
# ============================================================================

def train_autoencoder(
    model,
    dataset,
    num_epochs=40,
    batch_size=128,
    lr=1e-3,
    print_every=1
):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Autoencoder training using device:", device)
    model = model.to(device)
    model.train()

    # Loss function for image reconstruction
    criterion = nn.MSELoss()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # DataLoader for dataset
    dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=4,
            pin_memory=True
    )

    # Training loop over epochs
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        steps = 0

        # Loop over batches
        for imgs, _ in dataloader:
            imgs = imgs.to(device)

            # Forward pass through autoencoder
            recon = model(imgs)

            # Reconstruction loss
            loss = criterion(recon, imgs)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / steps

        if epoch % print_every == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Reconstruction Loss: {avg_loss:.4f}")


    return model

# ============================================================================
# Contrastive Training Function
# ============================================================================

def train_contrastive(
    model, 
    dataset, 
    transform_fn,
    num_steps=20000, 
    batch_size=128, 
    lr=1e-3, 
    t=2.0, 
    print_every=500
):
    """
    train_contrastive:
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Contrastive training using device:", device)
    model = model.to(device)
    model.train()

    # Define Adam optimizer for model parameters
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Datalaoder for dataset
    dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=4,
            pin_memory=True
    )
    data_iter = iter(dataloader)

    for step in range(1, num_steps + 1):
        try:
            imgs, _ = next(data_iter)
        except StopIteration:
            # Recreate dataloader if exhuasted
            data_iter = iter(dataloader)
            imgs, _ = next(data_iter)

        imgs = imgs.to(device)
        # imgs: shape (batch_size, 3 , 32, 32) in [0,1] from dataset
        # Create 2 augmented views of each image (x1, x2)

        pil_imgs = []
        for i in range(imgs.shape[0]):
            # Covert the tensor to PIL image
            np_img = (imgs[i].permute(1,2,0).cpu().numpy() * 255).astype('uint8')
            pil_imgs.append(transforms.ToPILImage()(np_img))

        # applying transfom_fn to get x1, x2
        x1_list = [transform_fn(img) for img in pil_imgs]
        x2_list = [transform_fn(img) for img in pil_imgs]

        x1 = torch.stack(x1_list, dim=0).to(device)
        x2 = torch.stack(x2_list, dim=0).to(device)

        # Forward pass
        z1 = model(x1)
        z2 = model(x2)

        L_align = alignment_loss(z1, z2)

        z_cat = torch.cat([z1, z2], dim=0)
        L_unif = uniformity_loss(z_cat, t=t)

        # Total loss
        loss = L_align + L_unif

        # Backprop and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % print_every == 0:
            print(f"Step [{step}/{num_steps}], "
                  f"Alignment Loss: {L_align.item():.4f}, "
                  f"Uniformity Loss: {L_unif.item():.4f}, "
                  f"Total loss: {loss.item():.4f}")


    return model