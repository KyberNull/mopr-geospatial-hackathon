import albumentations as A
import cv2
import numpy as np
import torch
from torchvision import datasets

SIZE = (256, 256)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


VOCEvalTransforms = A.Compose([
    A.SmallestMaxSize(max_size=SIZE[0], interpolation=cv2.INTER_LINEAR),
    # Takes a center slice of the image and important stuff is mostly near the center anyways
    A.CenterCrop(height=SIZE[0], width=SIZE[1]),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    A.ToTensorV2(),
])

VOCTrainTransforms = A.Compose([
    # Takes an image and scales the shorter side to 256 * 2 and scales the longer side with the same ratio (keeping the aspact ratio)
    A.SmallestMaxSize(max_size=SIZE[0] * 2, interpolation=cv2.INTER_LINEAR),
    A.RandomCrop(height=SIZE[0], width=SIZE[1]),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=255, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    # A.GaussNoise(std_range=(0.1, 0.2), p=0.2),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    A.ToTensorV2(),
])

class AlbumentationsVOC(datasets.VOCSegmentation):
    def __init__(self, root, year='2012', image_set='train', transforms=None):
        super().__init__(root, year=year, image_set=image_set, transform=None)
        self.Atransforms = transforms


    def __len__(self):
        return super().__len__()    


    def __getitem__(self, index):
        image, mask = super().__getitem__(index)
        image = np.array(image.convert('RGB'))
        mask = np.array(mask)

        if self.Atransforms is not None:
            augmented = self.Atransforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Convert numpy arrays to torch tensors (C, H, W) and proper dtypes
            image = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
            mask = torch.from_numpy(mask).long()

        return image, mask.long()
