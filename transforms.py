"""Data Augmentation and preprocessing transforms for Segmentation."""

import cv2
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2, InterpolationMode
from torchvision.transforms.v2 import functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def apply_clahe(image):
    # image: torch tensor [C, H, W] in float32 [0,1]
    img = image.permute(1, 2, 0).cpu().numpy()  # HWC

    img = (img * 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)

class EvalTransforms:
    '''Transforms for evaluation, including resizing and type conversions. 
        Does only resize and type conversions for consistent evaluation.
    '''
    def __init__(self, size=(512, 512)):
        self.size = size

    def __call__(self, image, mask=None):
        if mask is None:
            if isinstance(image, dict):
                sample = image
                image, mask = sample['image'], sample['mask']
            else:
                raise TypeError("Invalid arguments")
        image, mask = tv_tensors.Image(image), tv_tensors.Mask(mask)
        image = F.to_dtype(image, torch.float32, scale=True)

        image = torch.nn.functional.interpolate(image.unsqueeze(0), self.size, mode="area", antialias=False).squeeze(0)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        image = F.to_image(image)
        image = apply_clahe(image)
        image = F.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        mask = mask.to(torch.int64)

        return image, mask

class TrainTransforms:
    '''Data augmentation transforms for training,
    including random resized cropPIng, horizontal flipping, and rotation.
    '''
    def __init__(self, size=(512, 512), rotation_degrees=10):
        self.size = size
        self.rotation_degrees = rotation_degrees
        self.flips = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
        ])
        self.rotate90 = v2.RandomChoice([
            v2.RandomRotation((0, 0)),
            v2.RandomRotation((90, 90)),
            v2.RandomRotation((180, 180)),
            v2.RandomRotation((270, 270)),
        ])
      
    def __call__(self, image, mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            if isinstance(image, dict):
                sample = image
                image, mask = sample['image'], sample['mask']
            else:
                raise TypeError("Invalid arguments")
        image, mask = tv_tensors.Image(image), tv_tensors.Mask(mask)
        image = F.to_dtype(image, torch.float32, scale=True)

        image = torch.nn.functional.interpolate(image.unsqueeze(0), self.size, mode="area", antialias=False).squeeze(0)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        image, mask = self.flips(image, mask)
        image, mask = self.rotate90(image, mask)
        
        #Converting the image to float32 and mask to int64 as only one channel in mask
        image = F.to_image(image)
        image = apply_clahe(image)
        image = F.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        mask = mask.to(torch.int64)

        return image, mask

class PostProcessing:
    def __init__(self, num_classes, min_area=50, threshold=0.5):
        self.num_classes = num_classes
        self.min_area = min_area
        self.threshold = threshold
        self.kernel = np.ones((3, 3), np.uint8)

    def __call__(self, logits):
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)

        if probs.ndim == 4:
            return torch.stack([self._process_single(p) for p in probs])

        return self._process_single(probs)

    def _process_single(self, probs):
        probs = probs.cpu().numpy()  # (C, H, W)

        final_mask = np.zeros(probs.shape[1:], dtype=np.int32)

        for cls in range(self.num_classes):
            cls_prob = probs[cls]

            # Threshold FIRST
            binary = (cls_prob > self.threshold).astype(np.uint8)

            # Morphological cleanup
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)

            # Connected components
            num_labels, labels = cv2.connectedComponents(binary)

            for label in range(1, num_labels):
                component = (labels == label).astype(np.uint8)

                area = np.sum(component)

                if area < self.min_area:
                    continue

                final_mask[component == 1] = cls

        return torch.from_numpy(final_mask).long()