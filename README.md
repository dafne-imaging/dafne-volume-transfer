# dafne-sam2

MRI segmentation tools built on SAM2. The package supports automatic support/query
matching as well as a semi-automatic workflow where an operator confirms a small
number of matches before propagation.

It can be used through the included Qt GUI or imported as a Python library by
another application.

## Installation

Install the project with the GUI dependencies:

```bash
pip install -e ".[gui]"
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

## GUI

Run the application from the repository root:

```bash
python app.py
```

## Python API

External integrations should use `dafne_sam2.public_api`. The matching and
backbone modules are implementation details and are not intended as integration
entry points.

```python
from dafne_sam2.public_api import (
    SAM_propagate,
    SAM_refine,
    SliceMatchConfig,
    create_session_from_volumes,
    load_segmenter,
    transfer_slice,
)
```

Load a model once and reuse it across calls:

```python
seg = load_segmenter(checkpoint_dir="./models", device="auto")
```

`SAM_refine(image, mask, seg=seg)` refines a binary mask on one 2D slice.
`transfer_slice(image, support, support_masks, seg=seg)` matches and refines
the masks from an annotated support volume on one query slice.
`SAM_propagate(image, masks, seg=seg)` propagates existing masks through a
query volume.

### Semi-automatic matching

The semi-automatic API is centred on `SliceMatchSession`. It proposes support
slices, leaves the selection to the caller, and turns accepted matches into
anchors for propagation.

```python
session = create_session_from_volumes(
    seg=seg,
    support_volume=support_volume,
    support_labels=support_labels,
    roi_names=["roi_1", "roi_2"],
    query_volume=query_volume,
    config=SliceMatchConfig(),
)

candidates = session.suggest_support("roi_1", q_idx=50, top_k=3)
# Show candidates to the user and pass back the selected one.
accepted = session.accept_candidate(candidates[0])

result = SAM_propagate(
    image=query_volume,
    masks=session.anchors(),
    z_bounds=session.z_bounds(),
    seg=seg,
    refine_mask_prompt=False,
)
```

The usual loop is:

```text
suggest_support -> user selection -> accept_candidate -> next_query_slice
```

Once enough anchors have been accepted, pass `session.anchors()` and
`session.z_bounds()` to `SAM_propagate`.

## Data conventions

Volumes and label maps used by the session and by `SAM_propagate` have shape
`[Z, H, W]`. Support labels are integer maps: label `1` corresponds to the
first item in `roi_names`, label `2` to the second, and so on. Returned masks
are boolean arrays.

`transfer_slice` currently expects its support volume and masks in `[H, W, Z]`
format.

## Project layout

```
src/dafne_sam2/
  public_api.py       Stable entry point for external callers
  automatic/          Automatic matching and propagation
  semi_automatic/     Interactive matching session
  matching.py         Shared matching logic
gui/                  Qt interface
app.py                GUI entry point
```
