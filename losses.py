"""Loss and metric helpers for segmentation training and evaluation."""

import torch
import torch.nn.functional as F

def dice_loss(pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth=1e-8):
    pred = torch.softmax(pred, dim=1)

    # VOC uses label 255 as ignore; exclude those pixels from Dice computation.
    valid_mask = (target != 255)
    target = target.clone()
    target[~valid_mask] = 0  # temporary safe value

    # one_hot returns [N, H, W, C], so permute to [N, C, H, W] for channel-wise math.
    target_onehot = torch.nn.functional.one_hot(target, num_classes)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()

    valid_mask = valid_mask.unsqueeze(1)

    pred = pred * valid_mask
    target_onehot = target_onehot * valid_mask

    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

    dice = (2 * intersection + smooth) / (union + smooth)
    # Average only over classes present in the target to avoid skew from absent classes.
    class_present = target_onehot.sum(dim=(2, 3)) > 0
    dice = (dice * class_present).sum(dim=1) / class_present.sum(dim=1).clamp_min(1)
    return 1 - dice.mean()

def compute_means(pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth = 1e-8):
    target = target.long()
    pred_labels = torch.argmax(pred, dim=1)
    # Ignore VOC's 255 label and compute class stats from a compact confusion matrix.
    valid_mask = (target != 255)
    if not valid_mask.any():
        zero = pred.new_tensor(0.0)
        return zero, zero

    target_valid = target[valid_mask]
    pred_valid = pred_labels[valid_mask]

    indices = target_valid * num_classes + pred_valid
    confusion = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion = confusion.reshape(num_classes, num_classes).float()

    true_positive = confusion.diag()
    pred_sum = confusion.sum(dim=0)
    target_sum = confusion.sum(dim=1)

    dice_per_class = (2 * true_positive + smooth) / (pred_sum + target_sum + smooth)
    union = pred_sum + target_sum - true_positive
    iou_per_class = (true_positive + smooth) / (union + smooth)

    class_present = (pred_sum + target_sum) > 0
    present_count = class_present.sum().clamp_min(1)

    dice = (dice_per_class * class_present).sum() / present_count
    iou = (iou_per_class * class_present).sum() / present_count

    return dice.mean(), iou.mean()

def CBCE(pred_logits: torch.Tensor, target: torch.Tensor, num_classes: int, smooth = 1e-8):

    class_pixels = target.sum(dim=(0, 2, 3))
    background_pixels = (1 - target).sum(dim=(0, 2, 3))

    pos_weight=((background_pixels + smooth) / (class_pixels + smooth))
    pos_weight = pos_weight.view(-1, 1, 1).to(pred_logits.device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    return criterion(pred_logits, target)


def focal_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth=1e-8,
    ignore_index: int = 255,
):
    """
    Supports either class-index targets [N, H, W] or one-hot targets [N, C, H, W].
    """
    del smooth  # kept for call-site compatibility
    gamma = 2.0
    alpha = 0.75

    if target.ndim == 3:
        # Class-index mask path: convert valid labels to one-hot and mask ignore pixels.
        valid_mask = (target != ignore_index) & (target >= 0) & (target < num_classes)
        safe_target = target.clone()
        safe_target[~valid_mask] = 0
        target = F.one_hot(safe_target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1).expand(-1, num_classes, -1, -1)
    elif target.ndim == 4:
        # One-hot/multi-channel path.
        target = target.float()
        valid_mask = torch.ones_like(target, dtype=torch.bool)
    else:
        raise ValueError(f"Unsupported target shape {tuple(target.shape)}. Expected [N,H,W] or [N,C,H,W].")

    target = target.to(pred_logits.device)
    valid_mask = valid_mask.to(pred_logits.device)

    if not valid_mask.any():
        return pred_logits.sum() * 0.0

    bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    p = torch.sigmoid(pred_logits)
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)

    focal_term = torch.pow(1 - p_t, gamma)
    loss = alpha_t * focal_term * bce_loss

    # Average over valid elements only so ignored pixels do not affect loss scale.
    loss = loss[valid_mask]
    return loss.mean()
