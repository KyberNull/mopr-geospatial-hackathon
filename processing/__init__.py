"""Processing package: dataset loading, augmentations, and postprocessing."""

from .dataset import GeospatialDataset
from .transforms import EvalTransforms, TrainTransforms
from .preprocessing import apply_clahe, shadow_correction
from .postprocessing import PostProcessing, PostProcessingBuildings, PostProcessingRoads, PostProcessingWater

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

__all__ = [
    "GeospatialDataset",
    "EvalTransforms",
    "TrainTransforms",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "apply_clahe",
    "shadow_correction",
    "PostProcessing",
    "PostProcessingBuildings",
    "PostProcessingRoads",
    "PostProcessingWater",
]