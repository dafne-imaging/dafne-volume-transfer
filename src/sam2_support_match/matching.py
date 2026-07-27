import numpy as np
import torch
from collections import defaultdict
import torch.nn.functional as F
from skimage.measure import label as cc_label
from scipy.ndimage import binary_dilation

from sam2_support_match.preprocessing import body_mask_2d
from sam2_support_match.utils import _largest_cc, _to_grid, _two_legs_cc

BG_KEY = "__background__"

def build_multiclass_bags(seg,
                          supp_slices: dict,
                          supp_mask_slices: dict,
                          thr_hi: float = 0.7,
                          thr_lo: float = 0.3,
                          body_thresh: float = 10.0,
                          body_min_px: int = 50) -> dict:
    """
    Input: seg (encoder wrapper), supp_slices dict[slice_idx, image],
           supp_mask_slices dict[roi_name, dict[slice_idx, mask]]
    Return: dict[roi_name -> [C, N] bag, BG_KEY -> [C, N] bag], L2-normalized
    Builds one bag of foreground feature-vectors per ROI plus one shared background bag.
    """
    bags = defaultdict(list)

    # group masks by slice so feature/body are computed once per slice,
    # and BG can be defined against ALL rois on that slice, not just one
    per_slice = defaultdict(dict)
    for roi_name, slice_masks in supp_mask_slices.items():
        for slice_idx, mask in slice_masks.items():
            per_slice[slice_idx][roi_name] = mask

    for slice_idx, roi_masks in per_slice.items():
        frame_u8 = supp_slices[slice_idx]
        feat = seg.encoder_f_extractor(frame_u8)  # extract features from uint8 frame, once per slice
        C, h, w = feat.shape  # feature matrix dimensions
        flat = feat.reshape(C, h * w)
        body = _to_grid(body_mask_2d(frame_u8, body_thresh, body_min_px), h, w, feat.device)
        is_body = body.reshape(-1) > 0.5

        max_mask = None
        for roi_name, mask in roi_masks.items():
            m = _to_grid(mask, h, w, feat.device).reshape(-1)  # reshape corrispondent mask
            max_mask = m if max_mask is None else torch.maximum(max_mask, m)

            pos_idx = (m > thr_hi).nonzero(as_tuple=True)[0]
            if pos_idx.numel():
                bags[roi_name].append(F.normalize(flat[:, pos_idx], dim=0))

        # background: low for EVERY roi on this slice, and inside the body
        neg_idx = ((max_mask < thr_lo) & is_body).nonzero(as_tuple=True)[0]
        if neg_idx.numel():
            bags[BG_KEY].append(F.normalize(flat[:, neg_idx], dim=0))

    return {cls: torch.cat(vecs, dim=1) for cls, vecs in bags.items() if vecs}


def multiclass_score_maps(feat_query: torch.Tensor, bags: dict) -> dict:
    """
    Input: feat_query [C,h,w] query feature grid, bags dict[roi_name/BG_KEY -> [C,N]]
    Return: dict[roi_name -> score_map [h,w]] (BG_KEY excluded from output)
    Per-class similarity minus best rival class (margin), not raw cosine similarity.
    """
    C, h, w = feat_query.shape
    Qn = F.normalize(feat_query.reshape(C, h * w), dim=0)

    def _max_sim(B):
        """Best-matching bag vector per query cell, for one class's bag B."""
        sim = B.t() @ Qn  # [N_bag, h*w] similarity of every bag vector to every query cell
        return sim.max(dim=0).values.reshape(h, w)  # best-matching bag vector per cell

    pos = {c: _max_sim(B) for c, B in bags.items() if B.shape[1] > 0}

    out = {}
    for c in pos:
        if c == BG_KEY:
            continue
        rivals = torch.stack([p for k, p in pos.items() if k != c])  # every other class + BG
        out[c] = (pos[c] - rivals.max(dim=0).values).cpu().numpy()
    return out


def _mask_from_blob(blob: np.ndarray, score: np.ndarray, cell_px: float,
                    dilate_iters: int = 1, min_frac: float = 0.3,
                    min_abs_cells: float = 2.0) -> np.ndarray:
    """
    Input: blob [H,W] bool (raw winning region, full-res), score [H,W], cell_px (grid
           cell size in pixels)
    Return: mask [H,W] bool, cleaned
    Dilate+largest-CC to merge a region's own nearby pieces (drop distant noise);
    fall back to blob's filled bbox if that leaves too little of the original blob.
    """
    dilate_px = max(1, round(dilate_iters * cell_px))

    grown = binary_dilation(blob, iterations=dilate_px)
    lab = cc_label(grown)
    if lab.max() == 0:
        sel = blob
    else:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        sel = (lab == sizes.argmax()) & blob
        if not sel.any():
            sel = blob

    min_abs_px = min_abs_cells * cell_px * cell_px
    min_area_px = max(min_frac * blob.sum(), min_abs_px)
    if sel.sum() < min_area_px:
        ys, xs = np.where(blob)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sel = np.zeros_like(blob)
        sel[y0:y1, x0:x1] = True  # filled bbox fallback, whole blob is trusted as one piece

    return sel


def multiclass_masks(score_maps: dict, query_body2d: np.ndarray, img_hw: tuple,
                     score_thresh: float = 0.0, cc_mode: str = 'dilate_largest') -> dict:
    """
    Input: score_maps dict[roi_name -> [h,w]], query_body2d [H,W] bool, img_hw (H,W)
    Return: dict[roi_name -> (score: float, mask: [H,W] bool)]
    Winner-take-all per pixel, upsampled to full-res. Gated per body component
    (both legs if bilateral, else whole body) so a class keeps every blob it
    wins, not just the largest -- merged back under one roi_name.
    """
    if cc_mode != 'dilate_largest':
        raise ValueError(f"unsupported cc_mode: {cc_mode!r}")

    names = sorted(score_maps)
    h, w = score_maps[names[0]].shape
    H, W = img_hw
    cell_px = ((H / h) + (W / w)) / 2.0  # avg coarse-grid cell size in full-res pixels

    stack = np.stack([score_maps[c] for c in names])  # [K,h,w]
    t = torch.from_numpy(stack.astype(np.float32))[None]
    ups = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)[0].numpy()  # [K,H,W]
    win, best = ups.argmax(0), ups.max(0)  # per-pixel winning class index / its score

    legs = _two_legs_cc(query_body2d)
    regions = legs if legs is not None else (_largest_cc(query_body2d) > 0.5,)

    out = {}
    for ci, c in enumerate(names):
        for leg in regions:
            blob = (win == ci) & (best > score_thresh) & leg
            if not blob.any():
                continue
            mask = _mask_from_blob(blob, best, cell_px)
            score = float(best[blob].max())
            if c not in out:
                out[c] = [score, mask]
            else:
                out[c][0] = max(out[c][0], score)
                out[c][1] = out[c][1] | mask
    return {c: (s, m) for c, (s, m) in out.items()}


def pick_anchors(candidates: list, n_anchors: int = 3, min_gap: int = 3) -> list:
    """
    Input: candidates list[(slice_idx, score, mask)], n_anchors, min_gap
    Return: list[(slice_idx, mask)], length <= n_anchors
    Greedy: take highest-score candidate, drop every remaining candidate within
    min_gap slices of an already-picked one, repeat until n_anchors picked or
    candidates exhausted.
    """
    remaining = sorted(candidates, key=lambda c: c[1], reverse=True)
    picked = []
    while remaining and len(picked) < n_anchors:
        idx, _score, mask = remaining.pop(0)
        picked.append((idx, mask))
        remaining = [c for c in remaining if abs(c[0] - idx) >= min_gap]
    return picked


def multiclass_masks_for_frame(seg, bags: dict, frame_u8: np.ndarray,
                               body_thresh: float = 10.0, body_min_px: int = 50,
                               score_thresh: float = 0.0, cc_mode: str = 'dilate_largest') -> tuple:
    """
    Input: seg (encoder wrapper), bags dict[roi_name/BG_KEY -> [C,N]], frame_u8 [H,W] uint8
    Return: (dict[roi_name -> (score, mask)], score_maps dict[roi_name -> [h,w]])
    Single entry point: one query frame -> mask prompt for every class in bags.
    """
    feat = seg.encoder_f_extractor(frame_u8)
    score_maps = multiclass_score_maps(feat, bags)
    body = body_mask_2d(frame_u8, body_thresh, body_min_px)
    masks = multiclass_masks(score_maps, body, frame_u8.shape, score_thresh, cc_mode=cc_mode)
    return masks, score_maps
