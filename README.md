# dafne-sam2

One-shot support/query matching and segmentation on SAM2 embeddings, for MRI (PerSAM-style bag-of-vectors matching).

## Structure

```
src/dafne_sam2/
  automatic/         automatic pipeline (api.py, backbone.py, checkpoints.py, ...)
  semi_automatic/     semi-automatic pipeline (slice_api.py, slice_matching.py)
  matching.py         shared multiclass scoring
  preprocessing.py, utils.py, metrics.py   shared utilities
gui/                  Qt interface
app.py                GUI entry point
```

## Install

```bash
pip install -e .
```

## Usage

```bash
python app.py
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
