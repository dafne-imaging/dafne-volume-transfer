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

def volume_to_uint8(vol: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray: 
    """Any-dtype volume -> uint8 [0,255], clipped to the [p_low, p_high] percentiles
    first so a few extreme voxels cannot flatten the rest of the range."""
    lo, hi = np.percentile(vol, [p_low, p_high])
    v = np.clip(vol, lo, hi)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8) * 255.0
    return v.astype(np.uint8)

def resize_grayscale_to_rgb_and_resize(array: np.ndarray, image_size: int) -> np.ndarray: 
    """[Z,H,W] uint8 grayscale -> [Z,3,image_size,image_size] float32, each slice
    replicated to RGB and resized: SAM2's encoder takes 3-channel square input."""
    d = array.shape[0]
    out = np.zeros((d, 3, image_size, image_size), dtype=np.float32)
    for i in range(d): 
        img = Image.fromarray(array[i].astype(np.uint8)).convert("RGB")
        img = img.resize((image_size, image_size))
        out[i] = np.array(img).transpose(2, 0, 1)
    return out

def body_mask_2d(frame: np.ndarray, body_thresh: float = 10.0, body_min_prox: int = 3) -> np.ndarray:
    """[H,W] frame -> [H,W] bool body mask: threshold, keep connected components of at
    least body_min_prox px, fill holes."""
    m = frame > body_thresh
    if not m.any():
        return m
    labeled = cc_label(m, connectivity=2)
    sizes = np.bincount(labeled.flat)
    keep = np.zeros_like(m)
    for comp_id, size in enumerate(sizes):
        if comp_id == 0:
            continue
        if size >= body_min_prox:
            keep |= (labeled == comp_id)
    return binary_fill_holes(keep)


def volume_to_slices(vol: np.ndarray) -> dict:
    """[Z,H,W] array -> dict[z -> frame [H,W]], one entry per slice index."""
    return {z: vol[z] for z in range(vol.shape[0])}


def labels_to_masks(lbl: np.ndarray, roi_names: list) -> dict:
    """[Z,H,W] int label volume -> dict[roi_name -> dict[z -> bool mask]],
    label i+1 in roi_names maps to roi_names[i]; slices with no mask pixels
    are omitted."""
    out = {}
    for i, name in enumerate(roi_names, start=1):
        m = lbl == i
        out[name] = {z: m[z] for z in range(m.shape[0]) if m[z].any()}
    return out


def masks_to_labels(masks: dict, shape: tuple, roi_names: list) -> np.ndarray:
    """Inverse of labels_to_masks: dict[roi_name -> dict[z -> bool mask]] -> [Z,H,W]
    int label volume of the given shape."""
    lbl = np.zeros(shape, dtype=np.int32)
    for i, name in enumerate(roi_names, start=1):
        for z, m in masks.get(name, {}).items():
            lbl[z][m] = i
    return lbl


def body_z_extent(slices: dict, body_thresh: float = 10.0, body_min_px: int = 50) -> tuple:
    """
    Input: slices dict[slice_idx -> frame [H,W] uint8]
    Return: (z0, z1), first/last slice index (by key, not position) with a non-empty
            body mask. Intrinsic to this volume, no GT: it lets a roi's z-position be
            normalized by body extent rather than raw slice count
            (matching.estimate_z_window). Raises if no slice has a body.
    """
    present = [idx for idx, frame in slices.items()
               if body_mask_2d(frame, body_thresh, body_min_px).any()]
    if not present:
        raise ValueError("no slice has a non-empty body mask")
    return min(present), max(present)