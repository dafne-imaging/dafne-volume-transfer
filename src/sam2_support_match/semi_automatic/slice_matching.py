import itertools

import numpy as np
import torch
from collections import defaultdict
import torch.nn.functional as F

from sam2_support_match.preprocessing import body_mask_2d
from sam2_support_match.utils import _to_grid

def descriptor_from_feat(feat: torch.Tensor, mask: np.ndarray = None,
                         soft: bool = False) -> torch.Tensor:
    """[C,h,w] feature, mask [H,W] or None (whole slice) -> [C] L2-normalized mean of the
    covered cells; falls back to the single best cell when the resize to the feature grid
    leaves none. soft: read mask as a [0,1] weight map instead of thresholding at 0.5."""
    C, h, w = feat.shape
    flat = feat.reshape(C, h * w)
    if mask is None:
        return F.normalize(flat.mean(dim=1), dim=0)

    if not soft:
        g = _to_grid(mask, h, w, feat.device).reshape(-1)
        m = g > 0.5
        if not m.any():
            m = torch.zeros_like(g, dtype=torch.bool)
            m[int(g.argmax())] = True
        return F.normalize(flat[:, m].mean(dim=1), dim=0)

    weight = np.asarray(mask, dtype=np.float32)
    g = _to_grid(weight, h, w, feat.device).reshape(-1).clamp(min=0)
    total = g.sum()
    if total <= 0:
        # bilinear resampling can sample straight past a small region: keep its centre cell
        wy, wx = np.nonzero(weight > 0)
        if wy.size == 0:
            return F.normalize(flat.mean(dim=1), dim=0)
        gy = min(h - 1, int(wy.mean() * h / weight.shape[0]))
        gx = min(w - 1, int(wx.mean() * w / weight.shape[1]))
        g = torch.zeros_like(g)
        g[gy * w + gx] = 1.0
        total = g.sum()
    return F.normalize(flat @ (g / total).to(flat.dtype), dim=0)


def slice_descriptor(seg, frame_u8: np.ndarray, mask: np.ndarray = None,
                     soft: bool = False) -> torch.Tensor:
    """[H,W] uint8 slice -> [C] descriptor vector (see descriptor_from_feat)."""
    return descriptor_from_feat(seg.encoder_f_extractor(frame_u8), mask, soft=soft)


def build_slice_bags(seg,
                    supp_slices: dict,
                    supp_mask_slices: dict,
                    pool_region: dict = None,
                    soft: bool = False) -> dict:
    """Return: dict[roi_name][slice_idx] -> descriptor, one vector per (roi, support
    slice). pool_region overrides the per-slice mask as the pooling region (None ->
    build_roi_region_masks; soft -> build_roi_pool_weights)."""
    per_slice = defaultdict(list)
    for roi_name, mask_slices in supp_mask_slices.items():
        for idx, mask in mask_slices.items():
            pool = pool_region.get(roi_name) if pool_region is not None else mask
            per_slice[idx].append((roi_name, pool))

    bags = defaultdict(dict)
    for idx, entries in per_slice.items():
        feat = seg.encoder_f_extractor(supp_slices[idx])
        for roi_name, pool in entries:
            bags[roi_name][idx] = descriptor_from_feat(feat, pool, soft=soft)
        del feat
    return bags


def build_roi_region_masks(supp_mask_slices: dict) -> dict:
    """Union of each roi's mask over z: dict[roi_name -> [H,W] bool]."""
    region = {}
    for roi_name, mask_slices in supp_mask_slices.items():
        union = None
        for mask_matrix in mask_slices.values():
            mb = np.asanyarray(mask_matrix) > 0
            union = mb if union is None else (union | mb)
        if union is not None:
            region[roi_name] = union
    return region


def build_roi_occupancy(supp_mask_slices: dict) -> dict:
    """dict[roi_name -> [H,W] float in [0,1]]: how often each pixel belongs to the roi,
    over the slices it appears on."""
    occupancy = {}
    for roi_name, mask_slices in supp_mask_slices.items():
        if not mask_slices:
            continue
        acc = None
        for mask_matrix in mask_slices.values():
            mb = (np.asanyarray(mask_matrix) > 0).astype(np.float32)
            acc = mb if acc is None else acc + mb
        occupancy[roi_name] = acc / float(len(mask_slices))
    return occupancy


def build_roi_pool_weights(supp_mask_slices: dict, gamma: float = 1.0,
                           min_frac: float = 0.05) -> dict:
    """
    dict[roi_name -> [H,W] float in [0,1]], the region each roi is pooled over: its
    exclusive core (own occupancy minus the strongest rival's), so neighbouring rois whose
    unions overlap along z don't end up pooled over the same patch. gamma > 1 sharpens the
    weight further onto the core; falls back to the plain union when the core holds too
    little of the roi's own mass (min_frac).
    """
    occupancy = build_roi_occupancy(supp_mask_slices)
    union = build_roi_region_masks(supp_mask_slices)
    weights = {}
    for roi_name, own in occupancy.items():
        rival = None
        for other_name, other in occupancy.items():
            if other_name != roi_name:
                rival = other if rival is None else np.maximum(rival, other)
        w = own if rival is None else np.clip(own - rival, 0.0, 1.0)
        if gamma != 1.0:
            w = w ** gamma
        fallback = union.get(roi_name)
        if w.sum() < min_frac * own.sum() and fallback is not None:
            w = fallback.astype(np.float32)
        weights[roi_name] = w
    return weights


def body_geometry(mask: np.ndarray) -> tuple:
    """[H,W] bool body mask -> (cy, cx, scale=sqrt(area)), or None if empty.
       Return the mean of ys and xs of non-zero pixels coordinates
    """
    ys, xs = np.nonzero(mask) # two array: non-zero pixel indexes
    if ys.size == 0:
        return None
    return float(ys.mean()), float(xs.mean()), float(np.sqrt(ys.size))


def normalized_geometry(geo: tuple, shape: tuple) -> tuple:
    """geo in pixels -> centroid in [0,1], scale as a share of the frame."""
    H, W = shape
    cy, cx, scale = geo
    return cy / float(H), cx / float(W), scale / float(np.sqrt(H * W))


def build_body_geometry(slices: dict, body_thresh: float = 10.0, body_min_px: int = 50,
                        idxs=None) -> dict:
    """dict[slice_idx -> (cy,cx,scale)] over idxs (or every slice); no-body slices dropped.
       For each slice generate body_mask_2d and compute ys and xs as mean of non-zero pixel
       coordinates
    """
    out = {}
    for idx in (sorted(slices) if idxs is None else sorted(idxs)):
        geo = body_geometry(body_mask_2d(slices[idx], body_thresh, body_min_px))
        if geo is not None:
            out[idx] = geo
    return out


def roi_reference_centroid(geo: dict, mask_slices: dict) -> tuple:
    """(cy, cx) median body centroid over the slices the roi appears on, or None.
        Take all slices in mask_slices and extract [(y1, x1), (y2, x2), ...] if is in geo
        Return: median float of all slice coordinates
    """
    rows = [geo[i][:2] for i in mask_slices if i in geo]
    if not rows:
        return None
    cy, cx = np.median(np.asarray(rows, dtype=np.float64), axis=0)
    return float(cy), float(cx)


def align_region(region: np.ndarray, src_centroid: tuple, dst_centroid: tuple,
                 scale: float = 1.0, out_shape: tuple = None) -> np.ndarray:
    """Resample region (drawn around src_centroid) into a destination slice's frame via a
    similarity transform (translation + uniform scale, nearest neighbour); unchanged when
    either centroid is missing."""
    if region is None or src_centroid is None or dst_centroid is None:
        return region
    region = np.asarray(region)
    H, W = out_shape if out_shape is not None else region.shape
    sy, sx = src_centroid
    dy, dx = dst_centroid
    k = 1.0 / scale if scale else 1.0                  # destination pixel -> source pixel
    yi = np.round((np.arange(H) - dy) * k + sy).astype(int)
    xi = np.round((np.arange(W) - dx) * k + sx).astype(int)
    oy = (yi >= 0) & (yi < region.shape[0])
    ox = (xi >= 0) & (xi < region.shape[1])
    out = np.zeros((H, W), dtype=region.dtype)
    out[np.ix_(oy, ox)] = region[np.ix_(yi[oy], xi[ox])]
    return out


# drop outliers paris
def monotone_pairs(pairs: list) -> list:
    """Largest subset of pairs that rises in support_idx as query_idx rises, sorted by
    query_idx (longest increasing subsequence)."""
    seen, uniq = set(), []
    for q_idx, s_idx in sorted(pairs, key=lambda p: p[0]):
        if q_idx not in seen:
            seen.add(q_idx)
            uniq.append((q_idx, s_idx))
    if len(uniq) < 2:
        return uniq

    n = len(uniq)
    best = [1] * n
    prev = [-1] * n
    for j in range(n):
        for i in range(j):
            if uniq[i][1] < uniq[j][1] and best[i] + 1 > best[j]:
                best[j], prev[j] = best[i] + 1, i

    out, k = [], int(np.argmax(best))
    while k != -1:
        out.append(uniq[k])
        k = prev[k]
    return out[::-1]

# find best transformation for supp_idx = a * query_idx + b
def fit_z_map(pairs: list, a_bounds: tuple = (0.5, 2.0), min_span: int = 3) -> tuple:
    """(a, b) with support_idx = a*query_idx + b."""
    if not pairs:
        raise ValueError("fit_z_map needs at least one (query_idx, support_idx) pair")

    pairs = monotone_pairs(pairs)
    # get query and support couple selected by the user
    # (e.g couple0: (1, 10), couple1: (2, 15), couple2: (4, 25))
    # with q the first element of of tuple and s the second one
    # q and s factored as an array
    q = np.asarray([p[0] for p in pairs], dtype=np.float64)
    s = np.asarray([p[1] for p in pairs], dtype=np.float64)

    if len(pairs) < 2 or (q.max() - q.min()) < min_span:
        # too few, or all bunched together: the slope such points imply is noise
        return 1.0, float(np.median(s - q))

    # compute the slope (a): take all the couple indexes, compute the slope for each couple 
    # combination (e.g (0,1), (0, 2), (1, 2)) and then compute the median of all slopes
    slopes = [(s[j] - s[i]) / (q[j] - q[i])
              for i, j in itertools.combinations(range(len(q)), 2) if q[j] != q[i]]
    a = float(np.clip(np.median(slopes), *a_bounds))
    return a, float(np.median(s - a * q)) #return b as median value


def query_window_from_z_map(a: float, b: float, s_lo: int, s_hi: int,
                            q_bounds: tuple) -> tuple:
    """(a, b) from fit_z_map, (s_lo, s_hi) support range -> (lo, hi) query range, clipped
    to q_bounds."""
    q_lo, q_hi = sorted(((s_lo - b) / a, (s_hi - b) / a))
    lo = int(max(q_bounds[0], min(q_bounds[1], round(q_lo))))
    hi = int(max(q_bounds[0], min(q_bounds[1], round(q_hi))))
    return (lo, hi) if lo <= hi else (hi, lo)


def query_support_slice_similarity(seg, query_frame: np.ndarray, slice_bags: dict,
                                   roi_name: str, region_mask: np.ndarray = None,
                                   idx_range: tuple = None, soft: bool = False) -> list:
    """list[(support_slice_idx, cosine_sim)], best first. idx_range restricts which
    support slices are searched (None = all)."""
    idx_to_vec = slice_bags.get(roi_name, {})
    idxs = sorted(idx_to_vec)
    if idx_range is not None:
        lo, hi = idx_range
        idxs = [i for i in idxs if lo <= i <= hi]
    if not idxs:
        return []
    q = slice_descriptor(seg, query_frame, mask=region_mask, soft=soft)
    V = torch.stack([idx_to_vec[i] for i in idxs])
    sims = (V @ q).tolist()
    return sorted(zip(idxs, sims), key=lambda t: t[1], reverse=True)
