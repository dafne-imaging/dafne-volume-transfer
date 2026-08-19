"""
Small, GUI-independent public surface for external callers: refine a single mask
(SAM_refine), transfer + refine a support volume's masks onto one query slice
(transfer_slice, mimicking the GUI's Slice-match route), and propagate an anchored set of
masks across a whole query volume (SAM_propagate, automatic.api.propagate under the hood).

Each function accepts an optional `seg` (SAM2Segmenter). Pass one in to reuse a loaded
model across calls; leave it None for a self-contained call that loads and releases its own.
"""

import os
from collections import defaultdict
from typing import Callable, Optional

import numpy as np
from appdirs import user_cache_dir

from dafne_sam2.automatic.api import propagate
from dafne_sam2.automatic.backbone import SAM2Segmenter, mask_to_box
from dafne_sam2.automatic.checkpoints import resolve_model, CHECKPOINT_MODELS
from dafne_sam2.preprocessing import volume_to_slices, volume_to_uint8
from dafne_sam2.semi_automatic.slice_api import SliceMatchSession

# Mirrors gui/config.py's defaults, duplicated here (rather than imported) so this module
# stays usable without qtpy/gui installed.
_CKPT_DIR = user_cache_dir("dafne_sam2")
_CKPT_NAME = "sam2.1_tiny"

AVAILABLE_MODELS = list(CHECKPOINT_MODELS.keys())

def _default_segmenter() -> SAM2Segmenter:
    """Build a SAM2Segmenter for CHECKPOINT_MODELS[_CKPT_NAME], downloading it into
    _CKPT_DIR first if it's missing on disk (see gui/automatic_panel._get_seg, same default
    model, without the Qt dependency)."""
    checkpoint, model_cfg = resolve_model(_CKPT_NAME, _CKPT_DIR)
    device = os.environ.get("DAFNE_SAM2_DEVICE", "auto")
    return SAM2Segmenter(checkpoint, model_cfg, device=device)


def load_segmenter(checkpoint_dir: str,
                   progress_callback: Optional[Callable[[int, int], None]] = None,
                   checkpoint_name: str = _CKPT_NAME,
                   device: str = "auto") -> SAM2Segmenter:
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
    return frame if frame.dtype == np.uint8 else volume_to_uint8(frame[None])[0]


def _volume_to_uint8(vol: np.ndarray) -> np.ndarray:
    """[Z,H,W] any dtype -> [Z,H,W] uint8 (identity if already uint8)."""
    return vol if vol.dtype == np.uint8 else volume_to_uint8(vol)


def _mask_volume_to_slices(vol: np.ndarray) -> dict[int, np.ndarray]:
    """[Z,H,W] binary -> dict[z -> mask [H,W] bool], slices with no positive pixels
    omitted (matches preprocessing.labels_to_masks)."""
    vol = np.asarray(vol) > 0
    return {z: vol[z] for z in range(vol.shape[0]) if vol[z].any()}


def _refine_mask(seg: SAM2Segmenter, frame_u8: np.ndarray, mask: np.ndarray,
                 prompt_kind: str = 'mask') -> np.ndarray:
    """Core of SAM_refine: frame_u8 already uint8, mask already bool. Runs the initial mask
    through SAM2 as a single-frame session (backbone.segment_volume_mask/segment_volume_box)
    -- the mask decoder's own refinement of the prompt, not a multi-frame track.
    prompt_kind='box': prompt with the mask's bounding box (mask_to_box) instead of the mask
    itself -- more honest when the mask is unreliable (small/paired structures), same
    trade-off as automatic.api.propagate's prompt_kind. Empty mask with prompt_kind='box'
    returns an all-False mask (no box to prompt with)."""
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
               prompt_kind: str = 'mask') -> np.ndarray:
    """
    Refine an initial binary mask on a single 2D image using SAM2.
    image: [H,W] grayscale, any dtype. mask: [H,W] binary initial (pseudo-)segmentation.
    prompt_kind: 'mask' (default) prompts SAM2 with the mask itself; 'box' prompts with its
    bounding box instead -- see _refine_mask.
    Return: [H,W] bool refined mask.
    """
    owns_seg = seg is None
    seg = seg or _default_segmenter()
    try:
        return _refine_mask(seg, _slice_to_uint8(image), np.asarray(mask) > 0, prompt_kind=prompt_kind)
    finally:
        if owns_seg:
            seg.release()


def transfer_slice(image: np.ndarray, support: np.ndarray, support_masks: dict[str, np.ndarray],
                   seg: SAM2Segmenter | None = None,
                   prompt_kind: str | dict[str, str] = 'mask') -> dict[str, np.ndarray]:
    """
    Segment a single query image from one annotated support volume, one ROI at a time:
    for each roi in support_masks, find the best-matching support slice for `image`
    (semi_automatic.slice_api.SliceMatchSession, same machinery as the GUI's Slice-match
    route, one-shot instead of the GUI's multi-slice loop), transfer its mask onto `image`
    via SAM2 appearance matching, then refine it (SAM_refine).

    image: [H,W] query slice, any dtype.
    support: [Z,H,W] support volume, any dtype.
    support_masks: dict[roi_name -> [Z,H,W] binary mask, aligned with `support`].
    prompt_kind: 'mask' (default), 'box', or a per-roi dict of either -- see SAM_refine /
    automatic.api.propagate's prompt_kind.
    Return: dict[roi_name -> [H,W] bool mask], rois with no match on `image` omitted.
    """
    owns_seg = seg is None
    seg = seg or _default_segmenter()
    kind_of = (defaultdict(lambda: 'mask', prompt_kind) if isinstance(prompt_kind, dict)
              else defaultdict(lambda: prompt_kind))
    try:
        support_slices = volume_to_slices(_volume_to_uint8(support))
        support_mask_slices = {roi: _mask_volume_to_slices(vol) for roi, vol in support_masks.items()}
        image_u8 = _slice_to_uint8(image)
        session = SliceMatchSession(seg, support_slices, support_mask_slices, {0: image_u8})

        out = {}
        for roi_name, mask_slices in support_mask_slices.items():
            if not mask_slices:
                continue
            ranked = session.suggest_support(roi_name, 0, top_k=1)
            if not ranked:
                continue
            s_idx = ranked[0][0]
            found = session.anchor_from_pair(roi_name, 0, s_idx)
            if found is None:
                continue
            _score, mask = found
            out[roi_name] = _refine_mask(seg, image_u8, mask, prompt_kind=kind_of[roi_name])
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
                  fill_gaps: bool = False) -> dict[str, dict[int, np.ndarray]]:
    """
    Propagate existing per-slice masks to the rest of the volume with SAM2
    (automatic.api.propagate, see there for the optional-fix parameters below).

    image: [Z,H,W] query volume, any dtype.
    masks: dict[roi_name -> dict[slice_idx -> [H,W] binary]], the anchors to propagate from.
    Return: dict[roi_name -> dict[slice_idx -> [H,W] bool]] covering every slice in `image`.
    """
    owns_seg = seg is None
    seg = seg or _default_segmenter()
    try:
        query_slices = volume_to_slices(_volume_to_uint8(image))
        return propagate(query_slices, masks, seg=seg, z_bounds=z_bounds, prompt_kind=prompt_kind,
                         resolve_overlaps=resolve_overlaps, joint_propagate=joint_propagate,
                         fill_gaps=fill_gaps)
    finally:
        if owns_seg:
            seg.release()
