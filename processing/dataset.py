"""Custom geospatial dataset class."""

import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors


class GeospatialDataset(Dataset):
    """Custom dataset class to load the geospatial dataset."""

    def __init__(self, img_dir, img_mask, transform=None):
        self.img_dir = img_dir
        self.img_mask = img_mask
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(img_mask))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.img_mask, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        image, mask = tv_tensors.Image(image), tv_tensors.Mask(mask)

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        mask = mask.float()
        return image, mask
