"""SegFormer model definition used for semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# ===========================
# SegFormer Model
# ===========================

class SegFormer(nn.Module):
    """
    SegFormer architecture using a hierarchical Transformer encoder (MiT) 
     and a lightweight MLP decoder.
    """
    def __init__(self, num_classes: int, encoder_name: str = "mit_b2"):
        super().__init__()
        
        # We use smp.Segformer which implements the MiT backbone and MLP decoder.
        # mit_b2 is the recommended 'Quality' choice for RTX 50-series.
        self.model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            classes=num_classes,
        )

    def forward(self, x):
        """
        Forward pass. 
        Note: SegFormer output is typically 1/4 of input resolution.
        SMP handles the upsampling to the original input size internally.
        """
        input_size = x.shape[2:]
        
        # Get logits from SegFormer
        logits = self.model(x)
        
        # Ensure output matches input size exactly (handles odd input dimensions)
        if logits.shape[2:] != input_size:
            logits = F.interpolate(
                logits, size=input_size, mode="bilinear", align_corners=False
            )
            
        return logits
    
class TinyRefiner(nn.Module):
    """A small convolutional network to refine SegFormer outputs. Right now it's only targeting edges"""
    def __init__(self, in_channels=7, out_channels=4, hidden=32):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv_out = nn.Conv2d(hidden, out_channels, 1)

        self.act = nn.ReLU(inplace=True)

    def forward(self, rgb, logits):
        """
        rgb: (B, 3, H, W)
        logits: (B, C, H, W)
        """

        x = torch.cat([rgb, logits], dim=1) # concatenate along channel dimension

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))  # small context boost
        x = self.act(self.conv3(x))

        delta = self.conv_out(x)

        # residual refinement
        refined_logits = logits + delta

        return refined_logits