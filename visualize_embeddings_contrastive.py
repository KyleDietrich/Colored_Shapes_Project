# visualize_embeddings_contrastive.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import ColoredShapes32
from models import SmallCNN

# ---------- CONFIG ----------
MODEL = "contrastive_model_shape.pth" # Change to shape model for second plot
N_POINTS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------ Color & Markers -----
color_palette = [
    "#E24A33", "#348ABD", "#988ED5", "#777777",
    "#FBC15E", "#8EBA42", "#FFB5B8", "#56B4E9"
]

shape_markers = {
    0: "o",
    1: "^",
    2: "s"
}

def get_embeddings(model, dataset, n_pts, device):
    model.eval()
    idx = np.random.choice(len(dataset), n_pts, replace=False)
    zs, colors, shapes = [], [], []

    with torch.no_grad():
        for i in idx:
            x, lab = dataset[i]
            colors.append(int(lab[0]))
            shapes.append(int(lab[1]))
            z = model(x.unsqueeze(0).to(device))
            zs.append(z.cpu().numpy()[0])

    zs = np.array(zs)
    return zs, np.array(colors), np.array(shapes)

def plot_ring(zs, colors, shapes, title):
    theta = np.linspace(0, 2*np.pi, 400)
    plt.figure(figsize=(6,6))
    plt.plot(np.cos(theta), np.sin(theta), color="lightgray", lw=1)

    for c_id in range(8):
        for s_id in range(3):
            mask = (colors == c_id) & (shapes == s_id)
            if mask.sum() == 0: continue
            plt.scatter(
                zs[mask,0], zs[mask,1],
                c=[color_palette[c_id]],
                marker=shape_markers[s_id],
                edgecolors="k", linewidths=0.4, s=70, alpha=0.9
            )
    plt.title(title, fontsize=14)
    plt.axis("equal"); plt.axis("off")
    plt.tight_layout(); plt.show()

def main():
    ds = ColoredShapes32(length=64000, transform=transforms.ToTensor())

    model = SmallCNN(out_dim=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL, map_location=DEVICE))
    model = model.to(DEVICE)

    zs, colors, shapes = get_embeddings(model, ds, N_POINTS, DEVICE)

    if "color" in MODEL:
        plot_ring(zs, colors, shapes, "Color-Sensitive Contrastive Embedding")
    else:
        plot_ring(zs, colors, shapes, "Shape-Sensitive Contrastive Embedding")

if __name__ == "__main__":
    main()