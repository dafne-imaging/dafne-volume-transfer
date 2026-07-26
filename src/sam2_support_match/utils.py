from skimage.measure import label as cc_label
import numpy as np
import torch
import torch.nn.functional as F

from sam2_support_match.preprocessing import body_mask_2d

def _largest_cc(mask_2d: np.ndarray) -> np.ndarray: 
    """
    Input: mask2d [H,W] bool
    Return: mask2d [H,W] bool, only largest connected component kept
    Drops stray noise blobs (e.g. coil/marker) that would pollute the region gate.
    """
    labeled = cc_label(mask_2d)
    if labeled.max() == 0:
        return mask_2d
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == sizes.argmax()


def _to_grid(mask2d: np.ndarray, h: int, w: int, device=None):
    """
    Input: mask2d [H,W], target grid size (h,w)
    Return: mask resized to [h,w] (torch tensor)
    Resample full-res mask to feature-grid resolution via bilinear interp.
    """
    t = torch.from_numpy(mask2d.astype(np.float32))[None, None]
    g = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)[0, 0]
    return g.to(device) if device is not None else g


def _two_legs_cc(body2d: np.ndarray, min_leg_ratio: float = 0.2):
    """
    Input: body2d [H,W] bool, min_leg_ratio (min size of 2nd component vs 1st)
    Return: (left_mask, right_mask) [H,W] bool each, ordered by x, or None
    """
    lab = cc_label(body2d)
    sizes = np.bincount(lab.flat)
    sizes[0] = 0
    comps = [c for c in np.argsort(sizes)[::-1][:2] if sizes[c] > 0]
    if len(comps) < 2 or sizes[comps[1]] < min_leg_ratio * sizes[comps[0]]:
        return None

    a, b = sorted(comps, key=lambda c: np.where(lab == c)[1].mean())
    return lab == a, lab == b

def _split_at_midline(body2d: np.ndarray) -> tuple:
    """
    Input: body2d [H,W] bool
    Return: (left_mask, right_mask) [H,W] bool each
    Fallback when legs touch (_two_legs_cc fails): cuts body2d at the column with
    fewest body pixels near the centroid, the gap between the two legs.
    """
    cols = body2d.sum(axis=0).astype(np.float64)
    w = body2d.shape[1]
    cx = int(np.average(np.arange(w), weights=cols)) if cols.sum() else w // 2

    lo, hi = max(1, cx - w // 6), min(w - 1, cx + w // 6 + 1)
    cut = lo + int(np.argmin(cols[lo:hi])) if hi > lo else cx

    left, right = body2d.copy(), body2d.copy()
    left[:, cut:] = False
    right[:, :cut] = False
    return left, right

def side_masks(body2d: np.ndarray, left_is_low_x: bool,
               min_leg_ratio: float = 0.2) -> dict:
    """
    Input: body2d [H,W] bool, left_is_low_x (which x-side is anatomical left),
           min_leg_ratio (passed to _two_legs_cc)
    Return: {'L': mask [H,W] bool, 'R': mask [H,W] bool}
    Splits body into the two legs: CC split when reliable, midline cut fallback.
    """
    legs = _two_legs_cc(body2d, min_leg_ratio)
    lo_m, hi_m = legs if legs is not None else _split_at_midline(body2d)
    l, r = (lo_m, hi_m) if left_is_low_x else (hi_m, lo_m)
    return {'L': l, 'R': r}


def leg_crop_boxes(vol_u8: np.ndarray, body_thresh: float = 10.0, body_min_px: int = 50,
                   margin_frac: float = 0.15, min_leg_ratio: float = 0.2) -> dict:
    """
    Input: vol_u8 [Z,H,W] uint8
    Return: {'L': (y0,y1,x0,x1), 'R': (y0,y1,x0,x1)} pixel windows, one per side,
    valid across the whole volume (union of each slice's side bbox + margin).
    Side is a pure geometric split (no anatomical L/R meaning needed): a class is
    """
    H, W = vol_u8.shape[1], vol_u8.shape[2]
    acc = {'L': None, 'R': None}
    for z in range(vol_u8.shape[0]):
        body = body_mask_2d(vol_u8[z], body_thresh, body_min_px)
        if not body.any():
            continue
        for side, smask in side_masks(body, left_is_low_x=True, min_leg_ratio=min_leg_ratio).items():
            if not smask.any():
                continue
            ys, xs = np.where(smask)
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
            if acc[side] is None:
                acc[side] = [y0, y1, x0, x1]
            else:
                acc[side][0] = min(acc[side][0], y0)
                acc[side][1] = max(acc[side][1], y1)
                acc[side][2] = min(acc[side][2], x0)
                acc[side][3] = max(acc[side][3], x1)

    out = {}
    for side, b in acc.items():
        if b is None:
            continue
        y0, y1, x0, x1 = b
        mh = int(round((y1 - y0) * margin_frac))
        mw = int(round((x1 - x0) * margin_frac))
        out[side] = (max(0, y0 - mh), min(H, y1 + mh + 1),
                     max(0, x0 - mw), min(W, x1 + mw + 1))
    return out