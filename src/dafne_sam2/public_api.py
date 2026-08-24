"""
Small, GUI-independent public surface for external callers: load the shared SAM2 model,
refine a single mask (SAM_refine), transfer + refine a support volume's masks onto one query
slice (transfer_slice), or propagate anchors across a whole query volume (SAM_propagate).

For the interactive semi-automatic workflow, create_session_from_volumes builds a
SliceMatchSession. The caller drives suggest_support() and accept_candidate(), then hands
session.anchors() and session.z_bounds() to SAM_propagate.

Each function accepts an optional `seg` (SAM2Segmenter). Pass one in to reuse a loaded
model across calls; leave it None for a self-contained call that loads and releases its own.

Each function also accepts an optional `progress_callback`. For SAM_refine/transfer_slice/
SAM_propagate it's `(current, total=100)`, percentage-scale (see each function's docstring
for what its stages are); for load_segmenter it's `(current_bytes, total_bytes)`, matching
download_checkpoint's own byte-count contract (there's no "execution" beyond the download
to report progress on).
"""

import os
import traceback
from collections import defaultdict
from typing import Callable, Optional

import numpy as np
from appdirs import user_cache_dir

from dafne_sam2.automatic.api import propagate
from dafne_sam2.automatic.backbone import SAM2Segmenter, mask_to_box
from dafne_sam2.automatic.checkpoints import resolve_model, CHECKPOINT_MODELS
from dafne_sam2.preprocessing import volume_to_slices, volume_to_uint8

from dafne_sam2.semi_automatic.api import create_session_from_volumes
from dafne_sam2.semi_automatic.api import SliceMatchConfig, SliceMatchSession
from dafne_sam2.semi_automatic.api import MatchCandidate, AcceptedMatch

# Mirrors gui/config.py's defaults, duplicated here (rather than imported) so this module
# stays usable without qtpy/gui installed.
_CKPT_DIR = user_cache_dir("dafne_sam2")
_CKPT_NAME = "sam2.1_tiny"

AVAILABLE_MODELS = list(CHECKPOINT_MODELS.keys())

ProgressCallback = Callable[[int, int], None]

__all__ = [
    "AVAILABLE_MODELS",
    "load_segmenter",
    "SAM_refine",
    "transfer_slice",
    "SAM_propagate",
    "SliceMatchSession",
    "SliceMatchConfig",
    "MatchCandidate",
    "AcceptedMatch",
    "create_session_from_volumes",
]


def _report(callback: Optional[ProgressCallback], current: int, total: int = 100) -> None:
    """Percentage-scale progress: current/total out of 100, unless overridden. No-op if
    callback is None."""
    if callback is not None:
        callback(current, total)


def _scaled(callback: Optional[ProgressCallback], lo: int, hi: int) -> Optional[ProgressCallback]:
    """Wrap a percentage-scale progress_callback so a sub-task's own (current, total) --
    e.g. download_checkpoint's bytes, or propagate()'s rois-done count -- maps onto
    [lo, hi] of the outer 0..100 scale instead of its own native range. None in, None out."""
    if callback is None:
        return None

    def inner(current: int, total: int) -> None:
        frac = (current / total) if total else 1.0
        _report(callback, round(lo + frac * (hi - lo)))

    return inner


def _default_segmenter(progress_callback: Optional[ProgressCallback] = None,
                       lo: int = 0, hi: int = 100) -> SAM2Segmenter:
    """Build a SAM2Segmenter for CHECKPOINT_MODELS[_CKPT_NAME], downloading it into
    _CKPT_DIR first if it's missing on disk (see gui/automatic_panel._get_seg, same default
    model, without the Qt dependency). progress_callback/lo/hi: see _scaled -- only fires
    while an actual download happens (download_checkpoint's own contract); a cached
    checkpoint loads silently."""
    checkpoint, model_cfg = resolve_model(_CKPT_NAME, _CKPT_DIR,
                                          progress_callback=_scaled(progress_callback, lo, hi))
    device = os.environ.get("DAFNE_SAM2_DEVICE", "auto")
    return SAM2Segmenter(checkpoint, model_cfg, device=device)


def load_segmenter(checkpoint_dir: str,
                   checkpoint_name: str = _CKPT_NAME,
                   device: str = "auto",
                   progress_callback: Optional[ProgressCallback] = None
                   ) -> SAM2Segmenter:
    """
    Load a SAM2Segmenter, downloading its checkpoint into checkpoint_dir first if it
    isn't there yet. Modeled after dafne.utils.sam_mask_refine.load_sam: for a caller (e.g.
    dafne) that manages its own model directory instead of this package's default cache dir,
    and wants download progress to drive its own UI.

    checkpoint_dir: folder the checkpoint is (or will be) stored in -- e.g. dafne's
    GlobalConfig['MODEL_PATH'].
    progress_callback(current_bytes, total_bytes): called as the checkpoint downloads;
    not called at all if it's already on disk. total_bytes is 0 if neither the server nor
    CHECKPOINT_MODELS reports a size. See automatic.checkpoints.download_checkpoint.
    checkpoint_name: key into automatic.checkpoints.CHECKPOINT_MODELS -- selects which SAM2
    model to load (its weights AND the hydra config they must be paired with); see that dict
    for every model this package can pick from (sam2 / sam2.1, tiny through large).
    device: see automatic.device_utils.pick_device ('auto', 'cpu', 'mps', 'cuda:N', ...).
    """
    checkpoint, model_cfg = resolve_model(checkpoint_name, checkpoint_dir, progress_callback=progress_callback)
    return SAM2Segmenter(checkpoint, model_cfg, device=device)


def _slice_to_uint8(frame: np.ndarray) -> np.ndarray:
    """[H,W] any dtype -> [H,W] uint8 (identity if already uint8)."""
    if frame.dtype == np.uint8:
        return frame
    return _normalize_to_u8(frame)


def _volume_to_uint8(vol: np.ndarray) -> np.ndarray:
    """[Z,H,W] any dtype -> [Z,H,W] uint8 (identity if already uint8)."""
    if vol.dtype == np.uint8:
        return vol
    return _normalize_to_u8(vol)


def _mask_volume_to_slices(vol: np.ndarray) -> dict[int, np.ndarray]:
    """[Z,H,W] binary -> dict[z -> mask [H,W] bool], slices with no positive pixels
    omitted (matches preprocessing.labels_to_masks)."""
    vol = np.asarray(vol) > 0
    return {z: vol[z] for z in range(vol.shape[0]) if vol[z].any()}

def _normalize_to_u8(vol: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to uint8; a constant array maps to all zeros."""
    vol = np.asarray(vol)
    max_val = float(vol.max())
    min_val = float(vol.min())
    range_value = max_val - min_val
    if range_value == 0:
        return np.zeros(vol.shape, dtype=np.uint8)
    normalized = (vol.astype(np.float32) - min_val) * (255.0 / range_value)
    return np.clip(normalized, 0, 255).astype(np.uint8)

def _refine_mask(seg: SAM2Segmenter, frame_u8: np.ndarray, mask: np.ndarray,
                 prompt_kind: str = 'mask') -> np.ndarray:
    """Core of SAM_refine: frame_u8 already uint8, mask already bool. Runs the initial mask
    through SAM2 as a single-frame session (backbone.segment_volume_mask/segment_volume_box)
    -- the mask decoder's own refinement of the prompt, not a multi-frame track.
    prompt_kind='box': prompt with the mask's bounding box (mask_to_box) instead of the mask
    itself -- more honest when the mask is unreliable (small/paired structures), same
    trade-off as automatic.api.propagate's prompt_kind. Empty mask with prompt_kind='box'
    returns an all-False mask (no box to prompt with)."""
    frame_u8 = _normalize_to_u8(frame_u8)
    if prompt_kind == 'box':
        box = mask_to_box(mask)
        if box is None:
            return np.zeros_like(mask, dtype=bool)
        out = seg.segment_volume_box(frame_u8[None], {0: box})
    elif prompt_kind == 'mask':
        out = seg.segment_volume_mask(frame_u8[None], {0: mask})
    else:
        raise ValueError(f"unsupported prompt_kind: {prompt_kind!r}")
    return out[0].astype(bool)


def SAM_refine(image: np.ndarray, mask: np.ndarray, seg: SAM2Segmenter | None = None,
               prompt_kind: str = 'mask',
               progress_callback: Optional[ProgressCallback] = None) -> np.ndarray:
    """
    Refine an initial binary mask on a single 2D image using SAM2.
    image: [H,W] grayscale, any dtype. mask: [H,W] binary initial (pseudo-)segmentation.
    prompt_kind: 'mask' (default) prompts SAM2 with the mask itself; 'box' prompts with its
    bounding box instead -- see _refine_mask.
    progress_callback(current, total=100): percentage through the call. If seg is None, most
    of the range (0-70%) is the one-off checkpoint download/load (see _default_segmenter);
    with a seg passed in, or once loaded, that portion completes instantly.
    Return: [H,W] bool refined mask.
    """
    owns_seg = seg is None
    _report(progress_callback, 0)
    seg = seg or _default_segmenter(progress_callback, 0, 70)
    _report(progress_callback, 70)
    try:
        frame_u8 = _slice_to_uint8(image)
        _report(progress_callback, 80)
        out = _refine_mask(seg, frame_u8, np.asarray(mask) > 0, prompt_kind=prompt_kind)
        _report(progress_callback, 100)
        return out
    except Exception as e:
        print("Error in SAM_refine:", traceback.format_exc())
        raise
    finally:
        if owns_seg:
            seg.release()


def transfer_slice(image: np.ndarray, support: np.ndarray, support_masks: dict[str, np.ndarray],
                   seg: SAM2Segmenter | None = None,
                   prompt_kind: str | dict[str, str] = 'mask',
                   progress_callback: Optional[ProgressCallback] = None) -> dict[str, np.ndarray]:
    """
    Segment a single query image from one annotated support volume, one ROI at a time:
    for each roi in support_masks, find the best-matching support slice for `image`
    (semi_automatic.session.SliceMatchSession, same machinery as the GUI's Slice-match
    route, one-shot instead of the GUI's multi-slice loop), transfer its mask onto `image`
    via SAM2 appearance matching, then refine it (SAM_refine).

    image: [H,W] query slice, any dtype.
    support: [H,W,Z] support volume, any dtype.
    support_masks: dict[roi_name -> [H,W,Z] binary mask, aligned with `support`].
    prompt_kind: 'mask' (default), 'box', or a per-roi dict of either -- see SAM_refine /
    automatic.api.propagate's prompt_kind.
    progress_callback(current, total=100): percentage through the call. If seg is None, the
    checkpoint download/load (see _default_segmenter) takes 0-30%; setup (encoding the
    support volume, building the matching session) takes it to 40%; the remaining 40-100% is
    divided evenly across each roi's match-and-refine step.
    Return: dict[roi_name -> [H,W] bool mask], rois with no match on `image` omitted.
    """
    owns_seg = seg is None
    _report(progress_callback, 0)
    seg = seg or _default_segmenter(progress_callback, 0, 30)
    _report(progress_callback, 30)
    kind_of = (defaultdict(lambda: 'mask', prompt_kind) if isinstance(prompt_kind, dict)
              else defaultdict(lambda: prompt_kind))
    try:
        support = np.moveaxis(np.asarray(support), -1, 0)
        support_masks = {roi: np.moveaxis(np.asarray(vol), -1, 0) for roi, vol in support_masks.items()}
        support_slices = volume_to_slices(_volume_to_uint8(support))
        support_mask_slices = {roi: _mask_volume_to_slices(vol) for roi, vol in support_masks.items()}
        image_u8 = _slice_to_uint8(image)
        session = SliceMatchSession(seg, support_slices, support_mask_slices, {0: image_u8})
        _report(progress_callback, 40)

        rois = [roi for roi, mask_slices in support_mask_slices.items() if mask_slices]
        out = {}
        for i, roi_name in enumerate(rois):
            ranked = session.suggest_support(roi_name, 0, top_k=1)
            if ranked:
                s_idx = ranked[0].support_index
                found = session.anchor_from_pair(roi_name, 0, s_idx)
                if found is not None:
                    _score, mask = found
                    out[roi_name] = _refine_mask(seg, image_u8, mask, prompt_kind=kind_of[roi_name])
            _report(progress_callback, 40 + round(60 * (i + 1) / len(rois)) if rois else 100)
        return out
    finally:
        if owns_seg:
            seg.release()


def SAM_propagate(image: np.ndarray, masks: dict[str, dict[int, np.ndarray]],
                  seg: SAM2Segmenter | None = None,
                  z_bounds: dict[str, tuple[int, int]] | None = None,
                  prompt_kind: str | dict[str, str] = 'mask',
                  resolve_overlaps: bool = False,
                  joint_propagate: bool = False,
                  fill_gaps: bool = False,
                  refine_mask_prompt: bool = True,
                  progress_callback: Optional[ProgressCallback] = None
                  ) -> dict[str, dict[int, np.ndarray]]:
    """
    Propagate existing per-slice masks to the rest of the volume with SAM2
    (automatic.api.propagate, see there for the optional-fix parameters below).

    image: [Z,H,W] query volume, any dtype.
    masks: dict[roi_name -> dict[slice_idx -> [H,W] binary]], the anchors to propagate from.
    refine_mask_prompt (prompt_kind='mask', independent-session mode only): True (default)
    pairs each anchor mask with a corrective point so SAM2 actually re-derives it instead
    of echoing it back unchanged; False trusts every anchor mask as-is -- see
    automatic.api.propagate / automatic.backbone.segment_volume_mask.
    progress_callback(current, total=100): percentage through the call. If seg is None, the
    checkpoint download/load (see _default_segmenter) takes 0-20%; the remaining 20-100% is
    driven by automatic.api.propagate's own progress -- one step per roi's SAM2 pass in
    independent-session mode, or a single start/end step under joint_propagate (one shared
    session, no per-roi granularity there).
    Return: dict[roi_name -> dict[slice_idx -> [H,W] bool]] covering every slice in `image`.
    """
    owns_seg = seg is None
    _report(progress_callback, 0)
    seg = seg or _default_segmenter(progress_callback, 0, 20)
    _report(progress_callback, 20)
    try:
        query_slices = volume_to_slices(_volume_to_uint8(image))
        return propagate(query_slices, masks, seg=seg, z_bounds=z_bounds, prompt_kind=prompt_kind,
                         resolve_overlaps=resolve_overlaps, joint_propagate=joint_propagate,
                         fill_gaps=fill_gaps, refine_mask_prompt=refine_mask_prompt,
                         progress_callback=_scaled(progress_callback, 20, 100))
    finally:
        if owns_seg:
            seg.release()
