import numpy as np
import torch
from collections import defaultdict
import torch.nn.functional as F
from skimage.measure import label as cc_label
from scipy.ndimage import binary_dilation

from sam2_support_match.preprocessing import body_mask_2d
from sam2_support_match.utils import _largest_cc, _to_grid, _two_legs_cc

BG_KEY = "__background__"


def _positional_channels(body_grid: torch.Tensor) -> torch.Tensor:
    """[h,w] body mask -> [3,h,w] (r, cos theta, sin theta), centered on this slice's own
    body centroid (grid center if empty). Position still separates rois where appearance
    degrades (muscle's tapering end); concatenated to the appearance feature on both
    sides (build_multiclass_bags, api.find_prompts)."""
    h, w = body_grid.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=body_grid.device, dtype=torch.float32),
                            torch.arange(w, device=body_grid.device, dtype=torch.float32),
                            indexing='ij')
    mask = body_grid > 0.5
    if mask.any():
        cy, cx = ys[mask].mean(), xs[mask].mean()
    else:
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dy, dx = ys - cy, xs - cx
    r = torch.sqrt(dy * dy + dx * dx)
    r = r / (r.max() + 1e-8)
    theta = torch.atan2(dy, dx)
    return torch.stack([r, torch.cos(theta), torch.sin(theta)], dim=0)  # [3,h,w]


def build_multiclass_bags(seg,
                          supp_slices: dict,
                          supp_mask_slices: dict,
                          thr_hi: float = 0.7,
                          thr_lo: float = 0.3,
                          body_thresh: float = 10.0,
                          body_min_px: int = 50) -> dict:
    """
    Return: dict[roi_name -> [C,N] bag, BG_KEY -> [C,N] bag], L2-normalized. One
    foreground bag per roi, one shared background bag (cells below thr_lo for every roi
    on that slice, inside the body). Vector = fused-FPN feature + 3 positional channels
    (_positional_channels); api.find_prompts appends the same 3 query-side.
    """
    bags = defaultdict(list)

    # grouped by slice so feature/body compute once per slice, and BG is defined
    # against every roi on that slice, not just one
    per_slice = defaultdict(dict)
    for roi_name, slice_masks in supp_mask_slices.items():
        for slice_idx, mask in slice_masks.items():
            per_slice[slice_idx][roi_name] = mask

    for slice_idx, roi_masks in per_slice.items():
        frame_u8 = supp_slices[slice_idx]
        feat = seg.encoder_f_extractor(frame_u8)
        C, h, w = feat.shape
        body_grid = _to_grid(body_mask_2d(frame_u8, body_thresh, body_min_px), h, w, feat.device)
        pos = _positional_channels(body_grid).reshape(3, h * w)
        flat = torch.cat([feat.reshape(C, h * w), pos.to(feat.dtype)], dim=0)
        is_body = body_grid.reshape(-1) > 0.5

        max_mask = None
        for roi_name, mask in roi_masks.items():
            m = _to_grid(mask, h, w, feat.device).reshape(-1)
            max_mask = m if max_mask is None else torch.maximum(max_mask, m)

            pos_idx = (m > thr_hi).nonzero(as_tuple=True)[0]
            if pos_idx.numel():
                bags[roi_name].append(F.normalize(flat[:, pos_idx], dim=0))

        # background: low for EVERY roi on this slice, and inside the body
        neg_idx = ((max_mask < thr_lo) & is_body).nonzero(as_tuple=True)[0]
        if neg_idx.numel():
            bags[BG_KEY].append(F.normalize(flat[:, neg_idx], dim=0))

    return {cls: torch.cat(vecs, dim=1) for cls, vecs in bags.items() if vecs}


def multiclass_score_maps(feat_query: torch.Tensor, bags: dict,
                          bag_chunk: int = 4096) -> dict:
    """Return: dict[roi_name -> score_map [h,w]] (BG_KEY excluded). Per-class similarity
    minus best rival class (margin), not raw cosine similarity."""
    C, h, w = feat_query.shape
    Qn = F.normalize(feat_query.reshape(C, h * w), dim=0)

    def _max_sim(B):
        """Best-matching bag vector per query cell. Chunked over the bag with a running
        max to avoid materializing the full [N_bag, h*w] similarity matrix (background
        bag alone can reach several GB); max is associative, same result either way."""
        best = None
        for start in range(0, B.shape[1], bag_chunk):
            sim = B[:, start:start + bag_chunk].t() @ Qn
            block = sim.max(dim=0).values
            best = block if best is None else torch.maximum(best, block)
            del sim
        return best.reshape(h, w)

    pos = {c: _max_sim(B) for c, B in bags.items() if B.shape[1] > 0}

    out = {}
    for c in pos:
        if c == BG_KEY:
            continue
        rivals = torch.stack([p for k, p in pos.items() if k != c])
        out[c] = (pos[c] - rivals.max(dim=0).values).cpu().numpy()
    return out


def _mask_from_blob(blob: np.ndarray, score: np.ndarray, cell_px: float,
                    dilate_iters: int = 1, min_frac: float = 0.3,
                    min_abs_cells: float = 2.0, keep_frac: float = 0.2) -> np.ndarray:
    """
    Dilate to merge nearby pieces, keep every component big enough to be real, fall back
    to the whole blob if too little survives. keep_frac is RELATIVE to the largest
    component, not largest-only: a muscle split into two bellies along z (hamstrings
    distally) is two comparable components, and argmax would silently delete one.
    """
    dilate_px = max(1, round(dilate_iters * cell_px))
    min_abs_px = min_abs_cells * cell_px * cell_px

    grown = binary_dilation(blob, iterations=dilate_px)
    lab = cc_label(grown)
    if lab.max() == 0:
        sel = blob
    else:
        sizes = np.bincount(lab[blob].ravel(), minlength=lab.max() + 1)
        sizes[0] = 0
        keep = np.flatnonzero(sizes >= max(keep_frac * sizes.max(), min_abs_px))
        if keep.size == 0:
            # whole blob below the absolute floor (roi at its tapering end): fall
            # back to the largest component
            keep = np.array([sizes.argmax()])
        sel = np.isin(lab, keep) & blob
        if not sel.any():
            sel = blob

    if sel.sum() < max(min_frac * blob.sum(), min_abs_px):
        sel = blob  # kept pieces hold too little, keep everything

    return sel


def _blob_score(blob: np.ndarray, best: np.ndarray, mode: str) -> float:
    """Rank a slice for one class. 'peak' = best single pixel (one lucky pixel outranks a
    slice fully covering the organ, says little about presence). 'area'/'sum_margin'
    aggregate over the region -- what pick_anchors needs."""
    if mode == 'peak':
        return float(best[blob].max())
    if mode == 'area':
        return float(blob.sum())
    if mode == 'sum_margin':
        return float(best[blob].clip(min=0).sum())
    raise ValueError(f"unsupported score_mode: {mode!r}")


def multiclass_masks(score_maps: dict, query_body2d: np.ndarray, img_hw: tuple,
                     score_thresh: float = 0.0, cc_mode: str = 'dilate_largest',
                     score_mode: str = 'peak') -> dict:
    """
    Return: dict[roi_name -> (score, mask)]. Winner-take-all per pixel, upsampled to
    full-res. Gated per body component (both legs if bilateral, else whole body) so a
    class keeps every blob it wins, not just the largest -- merged back under one roi_name.
    """
    if cc_mode != 'dilate_largest':
        raise ValueError(f"unsupported cc_mode: {cc_mode!r}")

    names = sorted(score_maps)
    h, w = score_maps[names[0]].shape
    H, W = img_hw
    cell_px = ((H / h) + (W / w)) / 2.0

    stack = np.stack([score_maps[c] for c in names])
    t = torch.from_numpy(stack.astype(np.float32))[None]
    ups = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)[0].numpy()
    win, best = ups.argmax(0), ups.max(0)

    legs = _two_legs_cc(query_body2d)
    regions = legs if legs is not None else (_largest_cc(query_body2d) > 0.5,)

    out = {}
    for ci, c in enumerate(names):
        blob_all, mask_all = None, None
        for leg in regions:
            blob = (win == ci) & (best > score_thresh) & leg
            if not blob.any():
                continue
            mask = _mask_from_blob(blob, best, cell_px)
            blob_all = blob if blob_all is None else (blob_all | blob)
            mask_all = mask if mask_all is None else (mask_all | mask)
        if blob_all is None:
            continue
        # scored on the union of legs, so an aggregate mode sums a bilateral class
        # instead of keeping only the stronger side ('peak' is unaffected)
        out[c] = (_blob_score(blob_all, best, score_mode), mask_all)
    return out


def estimate_z_window(mask_slices: dict, supp_body_z: tuple, query_body_z: tuple) -> tuple:
    """
    Return: (lo, hi) suggested query slice range (float, unclamped). Roi's support slice
    range as a fraction of the support's body extent, applied to the query's -- lines up
    even when the two stacks cover different z-slabs, unlike a raw slice-count ratio. No
    query GT used. A SUGGESTION: tracks large central organs well (liver, IoU ~0.76),
    unreliable on small/paired ones (kidneys, spleen, calf muscles: 0.15-0.65).
    """
    idxs = sorted(mask_slices)
    z0_s, z1_s = supp_body_z
    span_s = max(1, z1_s - z0_s)
    rel_lo = (idxs[0] - z0_s) / span_s
    rel_hi = (idxs[-1] - z0_s) / span_s

    z0_q, z1_q = query_body_z
    span_q = z1_q - z0_q
    lo = z0_q + rel_lo * span_q
    hi = z0_q + rel_hi * span_q
    return (lo, hi) if lo <= hi else (hi, lo)


def _edge_candidates(candidates: list, edge_frac: float, edge_score_frac: float) -> list:
    """
    Return: 0-2 candidates, best-scoring of the first/last edge_frac band -- best inside
    the band, not the band's outermost slice, since the window's first/last slice may sit
    past the roi's real extent. edge_score_frac drops a band pick scoring too low relative
    to the window's best (kept near zero: a roi tapering towards its real end still scores
    low while genuinely being the roi).
    """
    if not candidates:
        return []

    idxs = [c[0] for c in candidates]
    span = max(idxs) - min(idxs)
    if span <= 0:
        return []

    width = edge_frac * span
    lo_band = [c for c in candidates if c[0] <= min(idxs) + width]
    hi_band = [c for c in candidates if c[0] >= max(idxs) - width]

    best_overall = max(c[1] for c in candidates)
    floor = edge_score_frac * best_overall if best_overall > 0 else float('-inf')

    out = []
    for band in (lo_band, hi_band):
        if not band:
            continue
        best = max(band, key=lambda c: c[1])
        if best[1] >= floor:
            out.append(best)
    return out


def _overlap_frac(mask: np.ndarray, ref: np.ndarray) -> float:
    """Fraction of mask's OWN area inside ref. Asymmetric on purpose: "does this blob sit
    where the roi was", without punishing the roi for being larger here than on ref."""
    area = int(mask.sum())
    if area == 0:
        return 0.0
    return float(np.count_nonzero(mask & ref)) / area


def pick_anchors(candidates: list, n_anchors: int = 3, min_gap: int = 3,
                 must_include: int | None = None, edge_frac: float = 0.2,
                 edge_score_frac: float = 0.02,
                 min_overlap_frac: float = 0.2) -> list:
    """
    Return: list[(slice_idx, mask)], length <= n_anchors. Pick order: centre
    (must_include), one anchor near each window end (score peaks mid-belly, so greedy
    alone never reaches the ends), then greedy by score. min_gap drops nearby remaining
    candidates per pick, so a narrow window yields fewer than n_anchors rather than
    near-duplicates. min_overlap_frac gates candidates against the centre anchor first,
    so a window overrunning the roi's real extent doesn't seed an end anchor on the
    neighbouring structure.
    """
    centre = None
    if must_include is not None:
        centre = next((c for c in candidates if c[0] == must_include), None)

    if centre is not None and min_overlap_frac > 0:
        pool = [c for c in candidates
                if c[0] == must_include or _overlap_frac(c[2], centre[2]) >= min_overlap_frac]
    else:
        pool = list(candidates)

    remaining = sorted(pool, key=lambda c: c[1], reverse=True)
    picked = []

    def _take(cand):
        nonlocal remaining
        picked.append((cand[0], cand[2]))
        # c[0] != cand[0] on its own, so the pick always leaves the pool even at min_gap=0
        remaining = [c for c in remaining
                     if c[0] != cand[0] and abs(c[0] - cand[0]) >= min_gap]

    forced = []
    if centre is not None:
        forced.append(centre)
    if edge_frac > 0:
        forced.extend(_edge_candidates(pool, edge_frac, edge_score_frac))

    for cand in forced:
        if len(picked) >= n_anchors:
            break
        # a forced pick still has to clear min_gap against the ones before it: on a
        # short window the centre and an end band can land on the same few slices
        if any(c[0] == cand[0] for c in remaining):
            _take(cand)

    while remaining and len(picked) < n_anchors:
        _take(remaining[0])
    return picked
