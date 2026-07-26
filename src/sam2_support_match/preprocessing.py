"""
Preprocessing operations for sam2 image's input. 
It includes, for each slice, normalization uint8[0, 255], resizing to 512
"""

import numpy as np
import torch 
from scipy.ndimage import binary_fill_holes
from skimage.measure import label as cc_label
from PIL import Image

IMG_SIZE = 512

def mask_to_box(mask: np.ndarray, margin: int = 8) -> tuple: 
    """
    From mask return bbox coordinates with margin
    """
    ys, xs = np.nonzero(mask)
    H, W = mask.shape
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(W - 1, int(xs.max()) + margin)
    y1 = min(H - 1, int(ys.max()) + margin)
    return (float(x0), float(y0), float(x1), float(y1))

def volume_to_uint8(vol: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray: 
    """
    Volume clipping and normalization as uint8
    """
    lo, hi = np.percentile(vol, [p_low, p_high])
    v = np.clip(vol, lo, hi)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8) * 255.0
    return v.astype(np.uint8)

def resize_grayscale_to_rgb_and_resize(array: np.ndarray, image_size: int) -> np.ndarray: 
    """
    array with dimension [Z, H, W]
    """
    d = array.shape[0]
    out = np.zeros((d, 3, image_size, image_size), dtype=np.float32)
    for i in range(d): 
        img = Image.fromarray(array[i].astype(np.uint8)).convert("RGB")
        img = img.resize((image_size, image_size))
        out[i] = np.array(img).transpose(2, 0, 1)
    return out

def body_mask_2d(frame: np.ndarray, body_thresh: float = 10.0, body_min_prox: int = 3) -> np.ndarray: 
    m = frame > body_thresh
    if not m.any(): 
        return m
    labeled = cc_label(m, connectivity=2) # return labeled connected regions
    sizes = np.bincount(labeled.flat) # count number of cc as label
    keep = np.zeros_like(m)
    for comp_id, size in enumerate(sizes):
        if comp_id == 0: 
            continue
        if size >= body_min_prox: # if cc ≥ body_min_prox add label to keep
            keep |= (labeled == comp_id)
    return binary_fill_holes(keep) # remove holes from final mask