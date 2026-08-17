import numpy as np
import torch
from monai.metrics import (compute_average_surface_distance, compute_dice,
                           compute_hausdorff_distance, compute_iou)

_MONAI_KW = {"include_background": True, "ignore_empty": False}


def _as_bchw(*arrays: np.ndarray) -> tuple:
    """[*spatial] bool arrays -> float tensors shaped [1, 1, *spatial]."""
    return tuple(torch.as_tensor(np.ascontiguousarray(a, dtype=bool))[None, None].float()
                 for a in arrays)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _as_bchw(pred, gt)
    return float(compute_dice(p, g, **_MONAI_KW).item())


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """|A∩B|/|A∪B|. Both empty -> 1.0, same reasoning as dice()."""
    p, g = _as_bchw(pred, gt)
    return float(compute_iou(p, g, **_MONAI_KW).item())


def surface_metrics(pred: np.ndarray, gt: np.ndarray, spacing_zyx: tuple) -> dict:
    """
    Input: pred/gt [Z,H,W] bool over the SAME slice range, spacing in mm as (z, y, x).
    Return: {"hd95", "assd"} in mm, symmetric -- nan if either side is empty, since a
            distance to an absent surface is undefined and both 0 and the volume diagonal
            would misreport it.
    With thick slices (z spacing of several mm) both numbers are dominated by the
    through-plane term: read them against the slice thickness, not as isotropic mm.
    """
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() or not g.any():
        return {"hd95": float("nan"), "assd": float("nan")}
    yp, yg = _as_bchw(p, g)
    hd95 = compute_hausdorff_distance(yp, yg, include_background=True, percentile=95,
                                      spacing=spacing_zyx)
    assd = compute_average_surface_distance(yp, yg, include_background=True, symmetric=True,
                                            spacing=spacing_zyx)
    return {"hd95": float(hd95.item()), "assd": float(assd.item())}


def volume_metrics(pred: np.ndarray, gt: np.ndarray, spacing_zyx: tuple) -> dict:
    """Voxel counts as mL plus the signed error (pred - gt), what a Bland-Altman plot
    needs. Only as trustworthy as the header spacing."""
    ml = float(np.prod(spacing_zyx)) / 1000.0
    vp = int(pred.astype(bool).sum()) * ml
    vg = int(gt.astype(bool).sum()) * ml
    return {"vol_pred_ml": vp, "vol_gt_ml": vg, "vol_err_ml": vp - vg,
            "vol_err_pct": float("nan") if vg == 0 else 100.0 * (vp - vg) / vg}


def _shape_of(*roi_dicts) -> tuple | None:
    for d in roi_dicts:
        for m in d.values():
            return m.shape
    return None


def _stack(roi_masks: dict, keys: list, shape: tuple) -> np.ndarray:
    """dict[slice_idx -> mask] -> [len(keys), H, W] bool, missing slices left empty."""
    empty = np.zeros(shape, dtype=bool)
    return np.stack([roi_masks.get(idx, empty) for idx in keys]) if keys else \
        np.zeros((0,) + shape, dtype=bool)


def evaluate_roi(roi_pred: dict, roi_gt: dict, idxs=None, spacing_zyx: tuple | None = None) -> dict:
    """
    Input: roi_pred/roi_gt dict[slice_idx -> bool mask] for ONE roi; idxs restricts to
           those slice indices (e.g. the roi's z window), default: every slice either
           side mentions. spacing_zyx (mm) additionally reports hd95/assd/volume.
    Return: {"dice", "iou", "pred_px", "gt_px", "n_slices", "per_slice": {idx: dice}},
            plus the surface/volume keys when spacing_zyx is given.
            dice/iou are volumetric -- the whole stack is scored in one go, not averaged
            over slices, which would over-weight near-empty ones.
    """
    shape = _shape_of(roi_pred, roi_gt)
    if shape is None:
        return {"dice": 1.0, "iou": 1.0, "pred_px": 0, "gt_px": 0, "n_slices": 0, "per_slice": {}}
    keys = sorted(set(roi_pred) | set(roi_gt)) if idxs is None else sorted(idxs)

    pred3d, gt3d = _stack(roi_pred, keys, shape), _stack(roi_gt, keys, shape)
    yp, yg = _as_bchw(pred3d, gt3d)
    # per slice: the same call with the slice axis moved into the batch dimension
    sp, sg = yp[0].transpose(0, 1), yg[0].transpose(0, 1)
    per_slice_dice = compute_dice(sp, sg, **_MONAI_KW).flatten().tolist() if keys else []

    out = {
        "dice": float(compute_dice(yp, yg, **_MONAI_KW).item()) if keys else 1.0,
        "iou": float(compute_iou(yp, yg, **_MONAI_KW).item()) if keys else 1.0,
        "pred_px": int(pred3d.sum()),
        "gt_px": int(gt3d.sum()),
        "n_slices": len(keys),
        "per_slice": dict(zip(keys, per_slice_dice)),
    }
    if spacing_zyx is not None:
        out.update(surface_metrics(pred3d, gt3d, spacing_zyx))
        out.update(volume_metrics(pred3d, gt3d, spacing_zyx))
    return out


def evaluate(pred: dict, gt: dict, windows: dict | None = None,
             spacing_zyx: tuple | None = None) -> dict:
    """
    Input: pred/gt dict[roi_name -> dict[slice_idx -> bool mask]] (propagate() output,
           preprocessing.labels_to_masks(query GT)); windows dict[roi_name -> (lo, hi)]
           scores each roi only inside the range it was asked to segment. Rois absent
           from pred are skipped -- never segmented, so scoring them all-miss would
           report a failure where there was no attempt.
    Return: dict[roi_name -> evaluate_roi(...)]
    """
    out = {}
    for roi_name, roi_pred in pred.items():
        idxs = None
        if windows and roi_name in windows:
            lo, hi = windows[roi_name]
            idxs = range(lo, hi + 1)
        out[roi_name] = evaluate_roi(roi_pred, gt.get(roi_name, {}), idxs=idxs,
                                     spacing_zyx=spacing_zyx)
    if out:
        mean_dice = float(np.mean([r["dice"] for r in out.values()]))
        mean_iou = float(np.mean([r["iou"] for r in out.values()]))
        out["_mean_dice"], out["_mean_iou"] = mean_dice, mean_iou
    return out


def format_report(scores: dict) -> str:
    rois = {k: v for k, v in scores.items() if not k.startswith("_")}
    lines = []
    for roi_name, s in sorted(rois.items(), key=lambda kv: kv[1]["dice"]):
        line = (f"{roi_name}: dice={s['dice']:.3f} iou={s['iou']:.3f} "
                f"pred={s['pred_px']}px gt={s['gt_px']}px")
        if "hd95" in s:
            line += f" hd95={s['hd95']:.1f}mm assd={s['assd']:.1f}mm dV={s['vol_err_ml']:+.1f}mL"
        lines.append(line)
    if "_mean_dice" in scores:
        lines.append(f"mean dice={scores['_mean_dice']:.3f} iou={scores['_mean_iou']:.3f}")
    return "\n".join(lines)
