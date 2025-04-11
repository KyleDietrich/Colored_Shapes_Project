# loss_functions.py

import torch
import torch.nn.functional as F

"""
    Alignment loss:
    This measures how close each pair of embeddings (z1_i, z2_i) are,
    typically used to pull embeddings of two augmented views of the
    same sample closer together. We compute the mean squared distance
    across all pairs in the batch.
    
    Args:
        z1 (Tensor): shape (batch_size, embedding_dim)
        z2 (Tensor): shape (batch_size, embedding_dim)
    Returns:
        Tensor: a single scalar (mean squared distance)
"""
def alignment_loss(z1, z2):
    return (z1 - z2).pow(2).sum(dim=1).mean()

"""
    Uniformity loss:
    This allows the embeddings to spread out in space rather than
    collapsing to the same point. We calculate pairwise distances between
    all embeddings and push them apart via an exponential term.
    
    Args:
        z (Tensor): shape (batch_size, embedding_dim), embeddings 
        t (float): temperature parameter controlling the strength of repulsion
    Returns:
        Tensor: a single scalar representing how uniformly spread out the embeddings are
"""
def uniformity_loss(z, t=2.0):
    batch_size = z.size(0)
    dist_matrix = (z.unsqueeze(1) - z.unsqueeze(0)).pow(2).sum(dim=2)
    off_diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=z.device)
    dist_off_diag = dist_matrix[off_diag_mask]
    unif = torch.log(torch.mean(torch.exp(-t * dist_off_diag)))
    return unif