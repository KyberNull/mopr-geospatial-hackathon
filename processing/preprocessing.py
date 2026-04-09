
import cv2
import numpy as np
import torch

def apply_clahe(image):
    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def shadow_correction(image):
    img = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l = np.asarray(lab[:, :, 0], dtype=np.float32)
    a = np.asarray(lab[:, :, 1], dtype=np.uint8)
    b = np.asarray(lab[:, :, 2], dtype=np.uint8)
    thresh = np.percentile(l, 35)
    shadow_mask = l < thresh
    mean_shadow = np.mean(l[shadow_mask]) if np.any(shadow_mask) else 1.0
    mean_light = np.mean(l[~shadow_mask]) if np.any(~shadow_mask) else 1.0
    scale = mean_light / (mean_shadow + 1e-6)
    l_corrected = l.copy()
    l_corrected[shadow_mask] *= scale
    l_corrected = np.clip(l_corrected, 0, 255)
    shadow_mask_blur = cv2.GaussianBlur(np.asarray(shadow_mask, dtype=np.float32), (21, 21), 0)
    l_final = shadow_mask_blur * l_corrected + (1 - shadow_mask_blur) * l
    l_final = np.uint8(l_final)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    l_final = clahe.apply(l_final)  # type: ignore[arg-type]
    lab_final = cv2.merge((l_final, a, b))
    result = cv2.cvtColor(lab_final, cv2.COLOR_LAB2RGB)
    result = result.astype(np.float32) / 255.0
    return torch.from_numpy(result).permute(2, 0, 1)


def apply_preprocessing(image, mode="original"):
    if mode == "original":
        return image
    if mode == "clahe":
        return apply_clahe(image)
    if mode == "shadow":
        return shadow_correction(image)
    return image