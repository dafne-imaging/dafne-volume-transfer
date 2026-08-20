import itertools
from collections import defaultdict
from typing import Callable, Optional

import numpy as np
import torch
from skimage.measure import label as cc_label

from dafne_sam2.automatic.backbone import SAM2Segmenter, mask_to_box
from dafne_sam2.automatic.checkpoints import resolve_checkpoint
from dafne_sam2.matching import (
    build_multiclass_bags, multiclass_score_maps, multiclass_masks,
    pick_anchors, estimate_z_window, _overlap_frac, _positional_channels,
)
from dafne_sam2.metrics import dice
from dafne_sam2.preprocessing import body_mask_2d, body_z_extent
from dafne_sam2.utils import _to_grid, leg_crop_boxes


def _crop_slices(slices: dict[int, np.ndarray], box: tuple) -> dict[int, np.ndarray]:
    """Crop every slice image to box (y0,y1,x0,x1)."""
    y0, y1, x0, x1 = box
    return {idx: img[y0:y1, x0:x1] for idx, img in slices.items()}


def _crop_masks(masks: dict[str, dict[int, np.ndarray]], box: tuple) -> dict[str, dict[int, np.ndarray]]:
    """Crop every roi mask (all slices) to box (y0,y1,x0,x1)."""
    y0, y1, x0, x1 = box
    return {roi: {idx: m[y0:y1, x0:x1] for idx, m in slice_masks.items()}
            for roi, slice_masks in masks.items()}


def estimate_windows(support_slices: dict, support_masks: dict, query_slices: dict,
                     body_thresh: float = 10.0, body_min_px: int = 50) -> dict[str, tuple[int, int]]:
    """
    GT-free starting suggestion for `window` in find_prompts: each roi's support slice
    range, mapped through body-silhouette extent onto the query's own extent.
    Return: dict[roi_name -> (lo, hi)].
    """
    supp_body_z = body_z_extent(support_slices, body_thresh, body_min_px)
    query_body_z = body_z_extent(query_slices, body_thresh, body_min_px)
    sorted_idxs = sorted(query_slices)
    lo_bound, hi_bound = sorted_idxs[0], sorted_idxs[-1]

    out = {}
    for roi_name, mask_slices in support_masks.items():
        lo, hi = estimate_z_window(mask_slices, supp_body_z, query_body_z)
        lo_i = max(lo_bound, min(hi_bound, round(lo)))
        hi_i = max(lo_bound, min(hi_bound, round(hi)))
        out[roi_name] = (lo_i, hi_i) if lo_i <= hi_i else (hi_i, lo_i)
    return out


def _windows_from_ranges(window: dict, support_masks: dict, sorted_idxs: list) -> tuple[dict, dict]:
    """window -> (per-roi set of in-range slices, per-roi centre slice = pick_anchors' must_include)."""
    if not window:
        raise ValueError("window is empty: no roi to match")

    unknown = sorted(set(window) - set(support_masks))
    if unknown:
        raise ValueError(f"window given for roi(s) absent from support_masks: {unknown}")

    windows, centres = {}, {}
    for roi_name, (lo, hi) in window.items():
        in_range = [idx for idx in sorted_idxs if lo <= idx <= hi]
        if not in_range:
            continue
        windows[roi_name] = set(in_range)
        mid = (lo + hi) / 2.0
        centres[roi_name] = min(in_range, key=lambda idx: abs(idx - mid))
    return windows, centres


def find_prompts(support_slices: dict[int, np.ndarray],
                 query_slices: dict[int, np.ndarray],
                 support_masks: dict[str, dict[int, np.ndarray]],
                 window: dict[str, tuple[int, int]],
                 checkpoint: str | None = None,
                 model_cfg: str | None = None,
                 device: str = "auto",
                 chunk_size: int = 8,
                 thr_hi: float = 0.85,
                 thr_lo: float = 0.3,
                 body_thresh: float = 10.0,
                 body_min_px: int = 50,
                 score_thresh: float = 0.0,
                 n_anchors: int = 5,
                 anchor_min_gap: int = 2,
                 score_mode: str = 'sum_margin',
                 return_windows: bool = False,
                 seg: SAM2Segmenter | None = None,
                 debug_sink: dict | None = None,
                 query_masks: dict[str, dict[int, np.ndarray]] | None = None,
                 ) -> dict[str, dict[int, np.ndarray]]:
    """
    Matching only, no SAM2 propagation: bag from support, per-slice candidate mask on
    query, keep the best anchors (up to n_anchors per roi). Pass seg to reuse a built
    segmenter. window (person-confirmed, see estimate_windows), not the score, decides
    WHERE a roi sits along z -- appearance alone cannot tell an organ from a look-alike
    elsewhere in the volume, and a wrong centre poisons every anchor after it. Scoring
    only ranks slices inside the range and finds the mask in-plane.
    return_windows: also return dict[roi_name -> (lo, hi)] for propagate()'s z_bounds.
    debug_sink/query_masks: see debugging.dump_run; query_masks is GT for evaluation
    only, never reaches matching.
    """
    if seg is None:
        checkpoint, model_cfg = resolve_checkpoint(checkpoint, model_cfg)
        seg = SAM2Segmenter(checkpoint, model_cfg, device=device)

    sorted_idxs = sorted(query_slices)
    windows, centres = _windows_from_ranges(window, support_masks, sorted_idxs)

    bags = build_multiclass_bags(seg, support_slices, support_masks,
                                 thr_hi=thr_hi, thr_lo=thr_lo,
                                 body_thresh=body_thresh, body_min_px=body_min_px)

    # only slices some roi's window covers get encoded at all
    needed = sorted(set().union(*windows.values()))
    feats = seg.encoder_frames_batched([query_slices[idx] for idx in needed],
                                       chunk_size=chunk_size)

    candidates = defaultdict(list)
    for idx, feat in zip(needed, feats):
        frame_u8 = query_slices[idx]
        body = body_mask_2d(frame_u8, body_thresh, body_min_px)
        # same 3 positional channels build_multiclass_bags appends support-side,
        # so the query feature stays dimension-matched against the bags
        C, h, w = feat.shape
        body_grid = _to_grid(body, h, w, feat.device)
        pos = _positional_channels(body_grid)
        feat = torch.cat([feat, pos.to(feat.dtype)], dim=0)
        score_maps = multiclass_score_maps(feat, bags)
        masks = multiclass_masks(score_maps, body, frame_u8.shape, score_thresh,
                                 score_mode=score_mode)
        for roi_name, (score, mask) in masks.items():
            if idx in windows.get(roi_name, ()):
                candidates[roi_name].append((idx, score, mask))

    prompts = {roi_name: dict(pick_anchors(cands, n_anchors=n_anchors, min_gap=anchor_min_gap,
                                           must_include=centres[roi_name]))
               for roi_name, cands in candidates.items() if cands}

    if debug_sink is not None:
        for roi_name, cands in candidates.items():
            centre_idx = centres[roi_name]
            centre_mask = next((m for i, _s, m in cands if i == centre_idx), None)
            gt_roi = (query_masks or {}).get(roi_name, {})
            # last two fields None without query_masks -- keeps older dumps' [0..3]
            # indexing readable
            cand_rows = []
            for i, s, m in cands:
                gt = gt_roi.get(i)
                cand_rows.append((i, s, int(m.sum()),
                                  _overlap_frac(m, centre_mask) if centre_mask is not None else None,
                                  dice(m, gt) if query_masks is not None else None,
                                  int(gt.sum()) if gt is not None else (0 if query_masks is not None else None)))
            debug_sink[roi_name] = {
                "window": tuple(window.get(roi_name, ())),
                "centre": centre_idx,
                "candidates": cand_rows,
                "anchors": sorted(prompts.get(roi_name, {})),
                "anchor_dice": ({i: dice(m, gt_roi.get(i, np.zeros_like(m)))
                                 for i, m in sorted(prompts.get(roi_name, {}).items())}
                                if query_masks is not None else None),
            }

    if not return_windows:
        return prompts
    bounds = {roi_name: (min(windows[roi_name]), max(windows[roi_name])) for roi_name in prompts}
    return prompts, bounds


def _drop_duplicate_blobs(out: dict, logits_by_roi: dict, idx: int, pos: int,
                          roi_names: list, dominance: float) -> None:
    """
    In place: a blob covered >= dominance by another roi's blob is a copy, not a shared
    border -- drop it whole (by mean SAM2 logit over the contested pixels), otherwise
    per-pixel arbitration below would leave a rim around the winner. Runs before
    _resolve_overlaps' per-pixel pass.
    """
    comps = {r: cc_label(out[r][idx]) for r in roi_names}
    drops = []
    for ra, rb in itertools.combinations(roi_names, 2):
        la, lb = comps[ra], comps[rb]
        both = (la > 0) & (lb > 0)
        if not both.any():
            continue
        for ia, ib in set(zip(la[both].tolist(), lb[both].tolist())):
            blob_a, blob_b = la == ia, lb == ib
            inter = blob_a & blob_b
            n = int(inter.sum())
            frac_a, frac_b = n / int(blob_a.sum()), n / int(blob_b.sum())
            if max(frac_a, frac_b) < dominance:
                continue  # thin shared border, not a duplicate claim
            score_a = float(logits_by_roi[ra][pos][inter].mean())
            score_b = float(logits_by_roi[rb][pos][inter].mean())
            # only the blob that is itself mostly duplicated can be dropped: a big blob
            # overlapping a small one keeps its own uncontested territory either way
            if score_a >= score_b and frac_b >= dominance:
                drops.append((rb, blob_b))
            elif score_b > score_a and frac_a >= dominance:
                drops.append((ra, blob_a))
    for roi_name, blob in drops:
        out[roi_name][idx] = out[roi_name][idx] & ~blob


def _resolve_overlaps(out: dict, logits_by_roi: dict, pos_of: dict, sorted_idxs: list,
                      dominance: float = 0.5) -> None:
    """In place: any pixel claimed by 2+ rois goes to whichever has the higher raw SAM2
    logit there. Runs _drop_duplicate_blobs first (dominance is its threshold)."""
    roi_names = list(out)
    if len(roi_names) < 2:
        return
    for idx in sorted_idxs:
        pos = pos_of[idx]
        if sum(out[r][idx].any() for r in roi_names) < 2:
            continue
        _drop_duplicate_blobs(out, logits_by_roi, idx, pos, roi_names, dominance)

        stacked_masks = np.stack([out[r][idx] for r in roi_names])
        overlap = stacked_masks.sum(axis=0) > 1
        if not overlap.any():
            continue
        stacked_logits = np.stack([logits_by_roi[r][pos] for r in roi_names])
        stacked_logits = np.where(stacked_masks, stacked_logits, -np.inf)
        winner = stacked_logits.argmax(axis=0)
        for i, r in enumerate(roi_names):
            out[r][idx] = out[r][idx] & (~overlap | (winner == i))


def _fill_gaps(out: dict, sorted_idxs: list) -> None:
    """In place: an interior slice left empty for a roi, with both neighbours non-empty,
    is filled with their union -- covers SAM2's single-frame confidence dropout (most
    visible under joint_propagate). Multi-slice absences are left alone -- the roi is
    genuinely ending there."""
    for slices in out.values():
        for pos in range(1, len(sorted_idxs) - 1):
            idx = sorted_idxs[pos]
            if slices[idx].any():
                continue
            prev_m, next_m = slices[sorted_idxs[pos - 1]], slices[sorted_idxs[pos + 1]]
            if prev_m.any() and next_m.any():
                slices[idx] = prev_m | next_m


def propagate(query_slices: dict[int, np.ndarray],
              prompts: dict[str, dict[int, np.ndarray]],
              checkpoint: str | None = None,
              model_cfg: str | None = None,
              device: str = "auto",
              z_bounds: dict[str, tuple[int, int]] | None = None,
              seg: SAM2Segmenter | None = None,
              prompt_kind: str | dict[str, str] = 'mask',
              resolve_overlaps: bool = False,
              joint_propagate: bool = False,
              fill_gaps: bool = False,
              refine_mask_prompt: bool = True,
              progress_callback: Optional[Callable[[int, int], None]] = None
              ) -> dict[str, dict[int, np.ndarray]]:
    """
    Propagation only, matching-agnostic: prompts can come from find_prompts, an external
    tool, or manual annotation. One SAM2 pass per roi, or one shared session (joint_propagate).
    z_bounds blanks a roi outside its expected range -- SAM2 tracks but cannot detect an
    organ ending, and past the last real slice the mask hops onto a neighbouring structure
    and keeps growing.
    prompt_kind='box': prompt with the anchor's bounding box instead of its mask -- more
    honest when the pseudo-label mask itself is unreliable (small/paired structures).
    refine_mask_prompt (independent-session mode only, prompt_kind='mask'): see
    backbone.segment_volume_mask -- True (default) pairs each mask anchor with a
    corrective point so SAM2 actually re-derives it; False trusts every anchor mask as-is.
    Opt-in fixes for trackers claiming each other's territory, weakest to strongest:
    resolve_overlaps (arbitrate contested pixels by raw SAM2 confidence, no-op where only
    one tracker ever reaches a pixel), joint_propagate (shared session discourages drift
    in the first place, combinable with resolve_overlaps), fill_gaps (patch an isolated
    single-slice dropout, applied last).
    progress_callback(rois_done, rois_total): called after each roi's SAM2 pass completes
    (independent-session mode) or once before/after the single shared pass (joint_propagate,
    where per-roi granularity isn't available -- one shared SAM2 session tracks every roi at
    once).
    """
    if seg is None:
        checkpoint, model_cfg = resolve_checkpoint(checkpoint, model_cfg)
        seg = SAM2Segmenter(checkpoint, model_cfg, device=device)

    sorted_idxs = sorted(query_slices)
    vol_u8 = np.stack([query_slices[idx] for idx in sorted_idxs])
    pos_of = {idx: pos for pos, idx in enumerate(sorted_idxs)}
    kind_of = (defaultdict(lambda: 'mask', prompt_kind) if isinstance(prompt_kind, dict)
              else defaultdict(lambda: prompt_kind))

    for roi_name, roi_prompts in prompts.items():
        bad = [idx for idx in roi_prompts if idx not in pos_of]
        if bad:
            raise ValueError(f"prompt slice(s) {bad} for {roi_name!r} not in query_slices")

    if joint_propagate:
        local_prompts, kinds = {}, {}
        for roi_name, roi_prompts in prompts.items():
            local = {pos_of[idx]: mask for idx, mask in roi_prompts.items()}
            kinds[roi_name] = kind_of[roi_name]
            if kind_of[roi_name] == 'box':
                boxes = {pos: mask_to_box(mask) for pos, mask in local.items()}
                local_prompts[roi_name] = {pos: box for pos, box in boxes.items() if box is not None}
            else:
                local_prompts[roi_name] = local

        if progress_callback is not None:
            progress_callback(0, 1)
        result = seg.segment_volume_joint(vol_u8, local_prompts, kinds, return_logits=resolve_overlaps)
        if progress_callback is not None:
            progress_callback(1, 1)
        propagated_by_roi, logits_by_roi = result if resolve_overlaps else (result, None)

        out = {}
        for roi_name, propagated in propagated_by_roi.items():
            lo, hi = (z_bounds or {}).get(roi_name, (sorted_idxs[0], sorted_idxs[-1]))
            out[roi_name] = {idx: (propagated[pos_of[idx]].astype(bool) if lo <= idx <= hi
                                   else np.zeros(vol_u8.shape[1:], dtype=bool))
                             for idx in sorted_idxs}
        if resolve_overlaps:
            _resolve_overlaps(out, logits_by_roi, pos_of, sorted_idxs)
        if fill_gaps:
            _fill_gaps(out, sorted_idxs)
        return out

    out = {}
    logits_by_roi = {} if resolve_overlaps else None
    n_rois = len(prompts)
    if progress_callback is not None:
        progress_callback(0, n_rois)
    for i, (roi_name, roi_prompts) in enumerate(prompts.items()):
        bad = [idx for idx in roi_prompts if idx not in pos_of]
        if bad:
            raise ValueError(f"prompt slice(s) {bad} for {roi_name!r} not in query_slices")
        local = {pos_of[idx]: mask for idx, mask in roi_prompts.items()}
        if kind_of[roi_name] == 'box':
            boxes = {pos: mask_to_box(mask) for pos, mask in local.items()}
            boxes = {pos: box for pos, box in boxes.items() if box is not None}
            result = seg.segment_volume_box(vol_u8, boxes, return_logits=resolve_overlaps)
        else:
            result = seg.segment_volume_mask(vol_u8, local, return_logits=resolve_overlaps,
                                             refine_mask_prompt=refine_mask_prompt)
        if resolve_overlaps:
            propagated, logits_by_roi[roi_name] = result
        else:
            propagated = result
        lo, hi = (z_bounds or {}).get(roi_name, (sorted_idxs[0], sorted_idxs[-1]))
        out[roi_name] = {idx: (propagated[pos_of[idx]].astype(bool) if lo <= idx <= hi
                               else np.zeros(vol_u8.shape[1:], dtype=bool))
                         for idx in sorted_idxs}
        if progress_callback is not None:
            progress_callback(i + 1, n_rois)

    if resolve_overlaps:
        _resolve_overlaps(out, logits_by_roi, pos_of, sorted_idxs)
    if fill_gaps:
        _fill_gaps(out, sorted_idxs)

    return out


def match_support_query(support_slices: dict[int, np.ndarray],
                        query_slices: dict[int, np.ndarray],
                        support_masks: dict[str, dict[int, np.ndarray]],
                        window: dict[str, tuple[int, int]],
                        checkpoint: str | None = None,
                        model_cfg: str | None = None,
                        device: str = "auto",
                        chunk_size: int = 8,
                        thr_hi: float = 0.7,
                        thr_lo: float = 0.3,
                        body_thresh: float = 10.0,
                        body_min_px: int = 50,
                        score_thresh: float = 0.0,
                        n_anchors: int = 5,
                        anchor_min_gap: int = 2,
                        score_mode: str = 'sum_margin',
                        split_legs: bool = False,
                        prompt_kind: str | dict[str, str] = 'mask',
                        resolve_overlaps: bool = False,
                        joint_propagate: bool = False,
                        fill_gaps: bool = False) -> dict[str, dict[int, np.ndarray]]:
    """
    Convenience wrapper: find_prompts() then propagate(), sharing one segmenter build.
    split_legs: run both steps per leg (own crop box, isolated propagation), paste back
    full-size; window is a z range so the in-plane crop does not affect it.
    """
    checkpoint, model_cfg = resolve_checkpoint(checkpoint, model_cfg)
    seg = SAM2Segmenter(checkpoint, model_cfg, device=device)

    if split_legs:
        supp_vol = np.stack([support_slices[idx] for idx in sorted(support_slices)])
        query_vol = np.stack([query_slices[idx] for idx in sorted(query_slices)])
        supp_boxes = leg_crop_boxes(supp_vol, body_thresh, body_min_px)
        query_boxes = leg_crop_boxes(query_vol, body_thresh, body_min_px)
        sides = set(supp_boxes) & set(query_boxes)  # own box per volume, not shared

        out: dict[str, dict[int, np.ndarray]] = {}
        for side in sides:  # each side matched+propagated independently, no mirroring
            side_support_slices = _crop_slices(support_slices, supp_boxes[side])
            side_support_masks = _crop_masks(support_masks, supp_boxes[side])
            side_query_slices = _crop_slices(query_slices, query_boxes[side])

            side_prompts, side_bounds = find_prompts(
                side_support_slices, side_query_slices, side_support_masks, window,
                chunk_size=chunk_size, thr_hi=thr_hi, thr_lo=thr_lo,
                body_thresh=body_thresh, body_min_px=body_min_px,
                score_thresh=score_thresh, n_anchors=n_anchors,
                anchor_min_gap=anchor_min_gap, score_mode=score_mode,
                return_windows=True, seg=seg)
            side_out = propagate(side_query_slices, side_prompts, z_bounds=side_bounds, seg=seg,
                                 prompt_kind=prompt_kind, resolve_overlaps=resolve_overlaps,
                                 joint_propagate=joint_propagate, fill_gaps=fill_gaps)

            y0, y1, x0, x1 = query_boxes[side]
            for roi_name, slice_masks in side_out.items():
                # full-size canvas, lazily created; each side pastes into its own box
                full = out.setdefault(roi_name, {
                    idx: np.zeros(query_slices[idx].shape, dtype=bool) for idx in query_slices})
                for idx, m in slice_masks.items():
                    full[idx][y0:y1, x0:x1] = m
        return out

    prompts, bounds = find_prompts(support_slices, query_slices, support_masks, window,
                                   chunk_size=chunk_size, thr_hi=thr_hi, thr_lo=thr_lo,
                                   body_thresh=body_thresh, body_min_px=body_min_px,
                                   score_thresh=score_thresh, n_anchors=n_anchors,
                                   anchor_min_gap=anchor_min_gap, score_mode=score_mode,
                                   return_windows=True, seg=seg)
    return propagate(query_slices, prompts, z_bounds=bounds, seg=seg, prompt_kind=prompt_kind,
                     resolve_overlaps=resolve_overlaps, joint_propagate=joint_propagate,
                     fill_gaps=fill_gaps)
