# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import random

# List of "base" colors in RGB [0..255]
COLOR_CLASSES = [
    (255, 0, 0),      # red
    (0, 255, 0),      # green
    (0, 0, 255),      # blue
    (255, 165, 0),    # orange
    (255, 255, 0),    # yellow
    (128, 0, 128),    # purple
    (0, 255, 255),    # cyan
    (255, 192, 203),  # pink
]

SHAPE_CLASSES = ["circle", "triangle", "square"]
IMAGE_SIZE = 32
if torch.accelerator.is_available():
    our_device = torch.accelerator.current_accelerator().type 
else:
    our_device = "cpu"

our_device="cpu"
torch.set_default_device(our_device)

print(f"Using {our_device} device")


def random_color_perturbation(base_color, perturb_range = 20):
    """
    Given an (R, G, B) tuple, add a random offset to each channel in 
    the range. Then clip to [0, 255] and return new (R, G, B)
    """
    offset = np.random.randint(-perturb_range, perturb_range+1, size = 3)
    color = np.array(base_color) + offset
    color = np.clip(color, 0, 255)
    return tuple(color.astype(np.uint8))


class ColoredShapes32(Dataset):
    """
    A pytorch Dataset that generates 32x32 images, each with one shape 
    (circle, triangle, square), random size, position, rotation, and a 
    randomly perturbed color.
    Args:
        length: Number of images to generate
        transform: Optional transform to apply to each image
    """
    def __init__(self, length = 64000, transform=None):
        super().__init__()
        self.length = length
        self.transform = transform

        # Pre generate which color and shape class each sample will have
        self.color_indices = np.random.randint(0, len(COLOR_CLASSES), size=self.length)
        self.shape_indices = np.random.randint(0, len(SHAPE_CLASSES), size=self.length)


    def __len__(self):
        """
        Return how many samples in the dataset
        """
        return self.length

    
    def _draw_shape_on_image(self, shape_type, color):
        """
        Create a 32x32 image w/ white background
        draw a single shape random postion/size/rotation
        """
        # draw on 32x32 RGBA image first 
        # then combine with white background
        overlay = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Pick a random top left corner to start
        x0 = random.randint(2, IMAGE_SIZE - 10)
        y0 = random.randint(2, IMAGE_SIZE - 10)

        shape_size = random.randint(8, 16)

        # Draw shape onto overlay
        if shape_type == "circle":
            draw.ellipse((x0, y0, x0 + shape_size, y0 + shape_size), fill=color + (255,))
        elif shape_type == "square":
            draw.rectangle((x0, y0, x0 + shape_size, y0 + shape_size), fill=color + (255,))
        elif shape_type == "triangle":
            half = shape_size / 2
            points = [
                (x0 + half, y0),
                (x0, y0 + shape_size),
                (x0 + shape_size, y0 + shape_size)
            ]
            draw.polygon(points, fill=color + (255,))

        # Random rotation
        angle = random.uniform(0, 360)
        overlay = overlay.rotate(angle, expand=False, resample=Image.BICUBIC)

        # White background
        background = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255, 255))
        combined = Image.alpha_composite(background, overlay).convert("RGB")
        
        return combined


    def __getitem__(self, idx):
        """
        Retrieve the idx-th sample from the dataset:
        1) Determine color + shape
        2) Draw shape as 32x32 RBG image
        3) Apply transform (none for this)
        4) Covert to Pytorch tensor since transform is none
        5) Return (image, (color_label, shape_label))

        color_label is an int [0, 7]
        shape_label is an int [0, 2]
        """
        c_idx = self.color_indices[idx]
        s_idx = self.shape_indices[idx]

        # Choose color and perturb
        base_color = COLOR_CLASSES[c_idx]
        color = random_color_perturbation(base_color, perturb_range=20)

        shape_type = SHAPE_CLASSES[s_idx]

        # Draw shape
        img = self._draw_shape_on_image(shape_type, color)

        # default to ransforms.ToTensor()
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img) # Tensor in [0, 1], shape (3, 32, 32)

        return img, (c_idx, s_idx)

# Small test to enusre it works
if __name__ == "__main__":
    encoder = Encoder()
    decoder = Decoder()

    ds = ColoredShapes32(length=10)
    print("Dataset length:", len(ds))
    dataloader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, generator= torch.Generator(device=our_device))
    for image, label in dataloader:
        img = image[0]
        lab = label[0]
        img.to(our_device)
        lab.to(our_device)
        output = encoder(img)
        decoder(output)




    # one sample
    sample_img, (color_label, shape_label) = ds[0]
    #print("Sample image shape:", sample_img.shape)
    #print("Color label:", color_label, "Shape label:", shape_label)

                            
