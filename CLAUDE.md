# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

One-shot support/query volume matching and segmentation on SAM2 embeddings, for MRI/CT
(PerSAM-style bag-of-vectors matching). Given one annotated "support" volume (with per-ROI
label masks) and an unannotated "query" volume of the same body region, the pipeline finds
where each ROI sits in the query and segments it there using SAM2's video predictor as the
tracker (via `automatic/video_predictor_npz.py`, a small local subclass that feeds it an
already-preprocessed tensor instead of a video path — see that module's docstring; the `sam2`
pip package itself is unmodified/upstream). Any checkpoint in
`automatic/checkpoints.CHECKPOINT_MODELS` (sam2/sam2.1, tiny through large) can be selected;
`gui/config.py` defaults to `sam2.1_tiny`.

## Commands

```bash
pip install -e ".[dev]"     # install with test deps
pip install -e ".[napari]"  # optional napari integration
pytest                      # run tests
python app.py               # launch the Qt GUI (entry point)
```

There is no lint/format command configured in this repo.

### Runtime configuration (env vars, see `gui/config.py` / `automatic/checkpoints.py`)

- `DAFNE_SAM2_CHECKPOINT` — path to the SAM2 `.pt` checkpoint (auto-downloaded to
  the platform cache dir, via `appdirs`, on first GUI run if missing).
- `DAFNE_SAM2_MODEL_CFG` — hydra config name, resolved relative to the installed
  `sam2` package (e.g. `configs/sam2.1/sam2.1_hiera_t.yaml`); must be the entry paired with
  the checkpoint above in `CHECKPOINT_MODELS` (`automatic/checkpoints.resolve_model` picks
  both together by name, avoiding a mismatched pair).
- `DAFNE_SAM2_DEVICE` — `auto` (default), `cpu`, `mps`, `cuda:N`; see
  `automatic/device_utils.pick_device`.
- `DAFNE_SAM2_DEBUG_DIR` — when set, the GUI's Segment run dumps per-ROI matching
  candidates/scores/dice via `automatic/debugging.dump_run`.

## Architecture

### Two independent matching routes, one shared propagation step

The core pipeline always ends the same way — SAM2 video-tracker propagation from a handful
of per-slice "anchor" prompts (`automatic/api.propagate`) — but there are two ways to get
those anchors, corresponding to the GUI's two sidebar tabs:

1. **Extent route** (`gui/automatic_panel.py`, `automatic/api.py`): the user confirms a
   z-range ("window") per ROI (optionally seeded by `automatic/api.estimate_windows`, a
   GT-free body-extent transfer). `find_prompts` then builds appearance "bags" from the
   support ROI masks (`matching.build_multiclass_bags`), scores every query slice inside
   the window against all ROI bags at once (`matching.multiclass_score_maps` — winner-take-
   all margin scoring, not raw cosine sim), extracts per-slice candidate masks
   (`matching.multiclass_masks`), and picks up to `n_anchors` well-spread anchors per ROI
   (`matching.pick_anchors`). Fully automatic once the window is confirmed.

2. **Slice-match route** (`gui/match_panel.py`, `semi_automatic/slice_api.py`,
   `semi_automatic/slice_matching.py`): a `SliceMatchSession` proposes, one query slice at
   a time, the best-matching support slice for a given ROI; the user confirms a pair
   (`add_pair`), which both produces one anchor mask AND refines an affine z-map
   (`support_idx = a*query_idx + b`) used to narrow future search ranges and predict the
   ROI's full query extent (`query_window_from_z_map`). More human-in-the-loop, useful when
   the extent route's body-extent transfer is unreliable (small/paired structures).

Both routes ultimately produce `prompts: dict[roi_name -> dict[slice_idx -> mask]]` fed to
`automatic/api.propagate`, which runs one SAM2 tracking pass (or one shared multi-object
pass, see below) and returns `dict[roi_name -> dict[slice_idx -> mask]]` over the whole
query volume.

### Matching internals (`matching.py`)

- Appearance vectors are SAM2 encoder features (`backbone.SAM2Segmenter._fuse_fpn_levels`,
  concatenating all FPN levels resampled onto the finest grid) concatenated with 3
  position channels (`_positional_channels`: radius + angle from the slice's own body
  centroid) — position disambiguates ROIs where appearance alone degrades (e.g. a
  muscle's tapering end).
- Per-ROI "bags" are sets of these vectors sampled from confidently-labeled support pixels
  (`thr_hi`); a shared background bag comes from pixels confidently outside every ROI on
  that slice, gated to the body silhouette (`preprocessing.body_mask_2d`).
- Scoring is *margin*-based: each ROI's similarity minus the best rival ROI's similarity,
  not raw cosine similarity — this is what makes scoring winner-take-all across ROIs.
- `pick_anchors` always forces a centre anchor (the user-confirmed window midpoint) plus
  anchors near each window edge, then fills in by score — greedy-by-score alone tends to
  cluster in the mid-belly since scores peak there.
- Body geometry (`preprocessing.body_mask_2d`, `_two_legs_cc`/`leg_crop_boxes` in `utils.py`)
  is used throughout to gate candidate regions and, for `split_legs=True`, to crop and
  match/propagate each leg independently before pasting back — needed because a naive
  whole-slice match can confuse left/right paired structures.

### Propagation internals (`automatic/api.py`, `automatic/backbone.py`)

- `propagate()` is matching-agnostic: it just needs `prompts` (mask or box per slice, per
  ROI) and optional `z_bounds` per ROI (SAM2 tracks forever but can't detect where an organ
  actually ends — bounds blank the mask outside the confirmed/predicted range).
- Two propagation modes: one independent SAM2 session per ROI (default), or
  `joint_propagate=True` — all ROIs share one session as distinct SAM2 object ids, so the
  tracker's memory attention sees every ROI every frame and is discouraged from drifting
  onto territory another ROI already holds.
- `resolve_overlaps=True` arbitrates any pixel two ROIs both claim by raw SAM2 logit
  confidence (`api._resolve_overlaps`), first dropping whole duplicate blobs
  (`api._drop_duplicate_blobs`) rather than leaving a contested rim.
- `fill_gaps=True` patches an isolated single-slice dropout (both neighbour slices non-empty,
  this one empty) — a known SAM2 failure mode, most visible under joint propagation.
- `prompt_kind` (`'mask'` default or `'box'` per ROI) controls whether SAM2 is prompted
  with the pseudo-label mask or just its bounding box (`backbone.mask_to_box`) — box is
  more honest when the mask itself is unreliable (small/paired structures like kidneys).
- All three fixes above are opt-in and off by default (preserve prior behavior); they stack
  weakest-to-strongest as: `resolve_overlaps` → `joint_propagate` → `fill_gaps`.

### Data format

Support/query volumes are loaded from `.npz` files (`gui/slice_pane.load_case`) as
`(volume [Z,H,W], label_volume [Z,H,W] int or None, roi_names list[str])`. ROI label `i`
(1-indexed) in the label volume corresponds to `roi_names[i-1]`.
`preprocessing.labels_to_masks`/`masks_to_labels` convert between that label-volume form and
the `dict[roi_name -> dict[slice_idx -> mask]]` form used throughout matching/propagation.
Query GT (if the query `.npz` has its own labels) is used only for scoring after a run
(`metrics.evaluate`), positionally matched against the support's `roi_names` — never fed
into matching itself.

### Public API for external callers (`src/dafne_sam2/public_api.py`)

GUI-independent (no `qtpy`/`gui` import) entry points for embedding this package elsewhere
(e.g. dafne):

- `SAM_refine(image, mask, seg=None, prompt_kind='mask') -> mask` — refine one 2D
  pseudo-label mask through SAM2 as a single-frame mask/box-prompted session (the mask
  decoder's own refinement of the prompt, not a multi-frame track).
- `transfer_slice(image, support, support_masks, seg=None, prompt_kind='mask') -> dict[roi_name -> mask]`
  — one-shot counterpart to the GUI's Slice-match route: for each ROI independently, find
  the best-matching support slice for `image` (`SliceMatchSession`, one query slice instead
  of the GUI's multi-slice loop), transfer its mask via SAM2 appearance matching, then
  refine it (`SAM_refine`). `support_masks` here is `dict[roi_name -> [Z,H,W] binary]`
  (whole-volume masks aligned with `support`), not the per-slice-dict form used elsewhere.
- `SAM_propagate(image, masks, seg=None, ...) -> dict[roi_name -> dict[slice_idx -> mask]]`
  — thin wrapper over `automatic/api.propagate` for a raw `[Z,H,W]` volume input.
- `load_segmenter(checkpoint_dir, progress_callback=None, checkpoint_name='sam2.1_tiny', device='auto') -> SAM2Segmenter`
  — modeled on `dafne.utils.sam_mask_refine.load_sam`: lets an external caller manage its
  own model directory and download-progress UI instead of this package's default cache dir.
  `checkpoint_name` is a key into `CHECKPOINT_MODELS` (see above) — selecting a model by
  name always pairs its weights with the correct hydra config.

All four accept an optional `seg: SAM2Segmenter`. Passed in, it's reused as-is (caller
owns its lifecycle); left `None`, each call builds its own default segmenter
(`CHECKPOINT_MODELS["sam2.1_tiny"]`, downloaded into the `appdirs` cache dir if needed) and
releases it before returning — so a call with no `seg` is self-contained but reloads
weights every time.

### GUI structure (`gui/`)

`MainWindow` (`main_window.py`) composes mixins, each owning one sidebar concern:
`IOPanelMixin` (load/export), `WindowPanelMixin` (extent-route window state — see its own
file), `AutomaticPanelMixin` (Extent-route run: builds/releases the `SAM2Segmenter`,
calls `find_prompts` + `propagate`), `MatchPanelMixin` (Slice-match route, drives a
`SliceMatchSession`). `SlicePane`/`slice_pane.py` renders one volume with slice scrubbing
and overlay marks. The `SAM2Segmenter` (GPU-resident SAM2 model) is built lazily on
first Segment/match action and released (`_release_seg`) after every run and on window
close — it is not meant to sit resident on the GPU between actions.
