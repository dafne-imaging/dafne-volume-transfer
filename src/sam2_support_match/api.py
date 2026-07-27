from collections import defaultdict

import numpy as np

from sam2_support_match.backbone import MedSAM2Segmenter
from sam2_support_match.checkpoints import resolve_checkpoint
from sam2_support_match.matching import build_multiclass_bags, multiclass_score_maps, multiclass_masks, pick_anchors
from sam2_support_match.preprocessing import body_mask_2d
from sam2_support_match.utils import leg_crop_boxes


def _crop_slices(slices: dict[int, np.ndarray], box: tuple) -> dict[int, np.ndarray]:
    """Crop every slice image to box (y0,y1,x0,x1)."""
    y0, y1, x0, x1 = box
    return {idx: img[y0:y1, x0:x1] for idx, img in slices.items()}


def _crop_masks(masks: dict[str, dict[int, np.ndarray]], box: tuple) -> dict[str, dict[int, np.ndarray]]:
    """Crop every roi mask (all slices) to box (y0,y1,x0,x1)."""
    y0, y1, x0, x1 = box
    return {roi: {idx: m[y0:y1, x0:x1] for idx, m in slice_masks.items()}
            for roi, slice_masks in masks.items()}


def find_prompts(support_slices: dict[int, np.ndarray],
                 query_slices: dict[int, np.ndarray],
                 support_masks: dict[str, dict[int, np.ndarray]],
                 checkpoint: str | None = None,
                 model_cfg: str | None = None,
                 device: str = "cuda",
                 chunk_size: int = 8,
                 thr_hi: float = 0.7,
                 thr_lo: float = 0.3,
                 body_thresh: float = 10.0,
                 body_min_px: int = 50,
                 score_thresh: float = 0.0,
                 n_anchors: int = 3,
                 anchor_min_gap: int = 3,
                 seg: MedSAM2Segmenter | None = None) -> dict[str, dict[int, np.ndarray]]:
    """
    Input: support_slices dict[slice_idx, image], query_slices dict[slice_idx, image],
           support_masks dict[roi_name, dict[slice_idx, mask]]
    Return: dict[roi_name -> dict[anchor_slice_idx -> mask]], SPARSE: only the
            up to n_anchors picked slices per class, not every query slice
    Automatching only (no SAM2 propagation): bag from support, per-slice
    candidate mask on query, keep best anchors. Pass seg to reuse an already
    built segmenter, otherwise one is built from checkpoint/model_cfg.
    """
    if seg is None:
        checkpoint, model_cfg = resolve_checkpoint(checkpoint, model_cfg)
        seg = MedSAM2Segmenter(checkpoint, model_cfg, device=device)

    bags = build_multiclass_bags(seg, support_slices, support_masks,
                                 thr_hi=thr_hi, thr_lo=thr_lo,
                                 body_thresh=body_thresh, body_min_px=body_min_px)

    sorted_idxs = sorted(query_slices)
    feats = seg.encoder_frames_batched([query_slices[idx] for idx in sorted_idxs],
                                       chunk_size=chunk_size)

    # per class, collect every slice's candidate (score, mask), then pick up to
    # n_anchors of them apart by anchor_min_gap
    candidates = defaultdict(list)
    for idx, feat in zip(sorted_idxs, feats):
        frame_u8 = query_slices[idx]
        score_maps = multiclass_score_maps(feat, bags)
        body = body_mask_2d(frame_u8, body_thresh, body_min_px)
        masks = multiclass_masks(score_maps, body, frame_u8.shape, score_thresh)
        for roi_name, (score, mask) in masks.items():
            candidates[roi_name].append((idx, score, mask))

    return {roi_name: dict(pick_anchors(cands, n_anchors=n_anchors, min_gap=anchor_min_gap))
            for roi_name, cands in candidates.items()}


def propagate(query_slices: dict[int, np.ndarray],
             prompts: dict[str, dict[int, np.ndarray]],
             checkpoint: str | None = None,
             model_cfg: str | None = None,
             device: str = "cuda",
             seg: MedSAM2Segmenter | None = None) -> dict[str, dict[int, np.ndarray]]:
    """
    Input: query_slices dict[slice_idx, image], prompts dict[roi_name -> dict[slice_idx -> seed mask]]
    Return: dict[roi_name -> dict[slice_idx -> mask]] for every query slice
    Propagation only, matching-agnostic: seed masks can come from find_prompts,
    an external ROI tool, or manual annotation. One SAM2 video-propagation pass
    per roi_name. Pass seg to reuse an already built segmenter.
    """
    if seg is None:
        checkpoint, model_cfg = resolve_checkpoint(checkpoint, model_cfg)
        seg = MedSAM2Segmenter(checkpoint, model_cfg, device=device)

    sorted_idxs = sorted(query_slices)
    vol_u8 = np.stack([query_slices[idx] for idx in sorted_idxs])
    pos_of = {idx: pos for pos, idx in enumerate(sorted_idxs)}

    out = {}
    for roi_name, roi_prompts in prompts.items():
        bad = [idx for idx in roi_prompts if idx not in pos_of]
        if bad:
            raise ValueError(f"prompt slice(s) {bad} for {roi_name!r} not in query_slices")
        local = {pos_of[idx]: mask for idx, mask in roi_prompts.items()}
        propagated = seg.segment_volume_mask(vol_u8, local)
        out[roi_name] = {idx: propagated[pos_of[idx]].astype(bool) for idx in sorted_idxs}

    return out


def match_support_query(support_slices: dict[int, np.ndarray],
                        query_slices: dict[int, np.ndarray],
                        support_masks: dict[str, dict[int, np.ndarray]],
                        checkpoint: str | None = None,
                        model_cfg: str | None = None,
                        device: str = "cuda",
                        chunk_size: int = 8,
                        thr_hi: float = 0.7,
                        thr_lo: float = 0.3,
                        body_thresh: float = 10.0,
                        body_min_px: int = 50,
                        score_thresh: float = 0.0,
                        n_anchors: int = 3,
                        anchor_min_gap: int = 3,
                        split_legs: bool = False) -> dict[str, dict[int, np.ndarray]]:
    """
    Input: support_slices dict[slice_idx, image], query_slices dict[slice_idx, image],
           support_masks dict[roi_name, dict[slice_idx, mask]]
    Return: dict[roi_name -> dict[slice_idx -> mask]] for every query slice
    One-shot convenience wrapper: find_prompts() then propagate(), sharing one
    segmenter build. split_legs: run BOTH steps per leg (own crop box each,
    propagation stays isolated per leg), paste results back full-size.
    """
    checkpoint, model_cfg = resolve_checkpoint(checkpoint, model_cfg)
    seg = MedSAM2Segmenter(checkpoint, model_cfg, device=device)

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

            side_prompts = find_prompts(side_support_slices, side_query_slices, side_support_masks,
                                        chunk_size=chunk_size, thr_hi=thr_hi, thr_lo=thr_lo,
                                        body_thresh=body_thresh, body_min_px=body_min_px,
                                        score_thresh=score_thresh, n_anchors=n_anchors,
                                        anchor_min_gap=anchor_min_gap, seg=seg)
            side_out = propagate(side_query_slices, side_prompts, seg=seg)

            y0, y1, x0, x1 = query_boxes[side]
            for roi_name, slice_masks in side_out.items():
                # full-size canvas, lazily created; each side pastes into its own box
                full = out.setdefault(roi_name, {
                    idx: np.zeros(query_slices[idx].shape, dtype=bool) for idx in query_slices})
                for idx, m in slice_masks.items():
                    full[idx][y0:y1, x0:x1] = m
        return out

    prompts = find_prompts(support_slices, query_slices, support_masks,
                           chunk_size=chunk_size, thr_hi=thr_hi, thr_lo=thr_lo,
                           body_thresh=body_thresh, body_min_px=body_min_px,
                           score_thresh=score_thresh, n_anchors=n_anchors,
                           anchor_min_gap=anchor_min_gap, seg=seg)
    return propagate(query_slices, prompts, seg=seg)
