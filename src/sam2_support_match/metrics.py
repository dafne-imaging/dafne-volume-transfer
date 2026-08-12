import numpy as np


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    total = int(p.sum()) + int(g.sum())
    if total == 0:
        return 1.0
    return 2.0 * float(np.logical_and(p, g).sum()) / total


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """|A∩B|/|A∪B|. Both empty -> 1.0, same reasoning as dice()."""
    p, g = pred.astype(bool), gt.astype(bool)
    union = int(np.logical_or(p, g).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum()) / union


def _slice_pair(roi_pred: dict, roi_gt: dict, idx: int, shape: tuple):
    empty = np.zeros(shape, dtype=bool)
    return roi_pred.get(idx, empty), roi_gt.get(idx, empty)


def _shape_of(*roi_dicts) -> tuple | None:
    for d in roi_dicts:
        for m in d.values():
            return m.shape
    return None


def evaluate_roi(roi_pred: dict, roi_gt: dict, idxs=None) -> dict:
    """
    Input: roi_pred/roi_gt dict[slice_idx -> bool mask] for ONE roi; idxs restricts to
           those slice indices (e.g. the roi's z window), default: every slice either
           side mentions.
    Return: {"dice", "iou", "pred_px", "gt_px", "n_slices", "per_slice": {idx: dice}}.
            dice/iou are volumetric -- counts pooled over all slices before the ratio,
            not a mean of per-slice values, which over-weights near-empty slices.
    """
    shape = _shape_of(roi_pred, roi_gt)
    if shape is None:
        return {"dice": 1.0, "iou": 1.0, "pred_px": 0, "gt_px": 0, "n_slices": 0, "per_slice": {}}
    keys = sorted(set(roi_pred) | set(roi_gt)) if idxs is None else sorted(idxs)

    inter = pred_px = gt_px = 0
    per_slice = {}
    for idx in keys:
        p, g = _slice_pair(roi_pred, roi_gt, idx, shape)
        inter += int(np.logical_and(p, g).sum())
        pred_px += int(p.sum())
        gt_px += int(g.sum())
        per_slice[idx] = dice(p, g)

    total = pred_px + gt_px
    union = pred_px + gt_px - inter
    return {
        "dice": 1.0 if total == 0 else 2.0 * inter / total,
        "iou": 1.0 if union == 0 else inter / union,
        "pred_px": pred_px,
        "gt_px": gt_px,
        "n_slices": len(keys),
        "per_slice": per_slice,
    }


def evaluate(pred: dict, gt: dict, windows: dict | None = None) -> dict:
    """
    Input: pred/gt dict[roi_name -> dict[slice_idx -> bool mask]] (propagate() output,
           preprocessing.labels_to_masks(query GT)); windows dict[roi_name -> (lo, hi)]
           scores each roi only inside the range it was asked to segment. Rois absent
           from pred are skipped -- never segmented, so scoring them all-miss would
           report a failure where there was no attempt.
    Return: dict[roi_name -> evaluate_roi(...)], plus "_mean_dice"/"_mean_iou" over the
            scored rois (leading underscore keeps them out of the roi namespace).
    """
    out = {}
    for roi_name, roi_pred in pred.items():
        idxs = None
        if windows and roi_name in windows:
            lo, hi = windows[roi_name]
            idxs = range(lo, hi + 1)
        out[roi_name] = evaluate_roi(roi_pred, gt.get(roi_name, {}), idxs=idxs)
    if out:
        mean_dice = float(np.mean([r["dice"] for r in out.values()]))
        mean_iou = float(np.mean([r["iou"] for r in out.values()]))
        out["_mean_dice"], out["_mean_iou"] = mean_dice, mean_iou
    return out


def format_report(scores: dict) -> str:
    """One line per roi, sorted worst-dice first -- the interesting end when comparing
    runs. Meant for a terminal print / GUI status line, not for parsing."""
    rois = {k: v for k, v in scores.items() if not k.startswith("_")}
    lines = []
    for roi_name, s in sorted(rois.items(), key=lambda kv: kv[1]["dice"]):
        lines.append(f"{roi_name}: dice={s['dice']:.3f} iou={s['iou']:.3f} "
                     f"pred={s['pred_px']}px gt={s['gt_px']}px")
    if "_mean_dice" in scores:
        lines.append(f"mean dice={scores['_mean_dice']:.3f} iou={scores['_mean_iou']:.3f}")
    return "\n".join(lines)
