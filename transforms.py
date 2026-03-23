"""Data Augmentation and preprocessing transforms for Segmentation."""

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2, InterpolationMode
from torchvision.transforms.v2 import functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        image = F.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        mask = mask.to(torch.int64)

        return image, mask