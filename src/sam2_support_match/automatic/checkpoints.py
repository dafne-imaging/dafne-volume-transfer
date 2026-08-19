import os
import urllib.request
from pathlib import Path
from typing import Callable, Optional

_ENV_CKPT = "SAM2_SUPPORT_MATCH_CHECKPOINT"
_ENV_CFG = "SAM2_SUPPORT_MATCH_MODEL_CFG"

_SAM2_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2"

# Every SAM2 checkpoint Meta publishes -- weights + the hydra config (resolved relative to
# the installed `sam2` pip package, see resolve_checkpoint) it must be paired with.
# automatic.video_predictor_npz.build_sam2_video_predictor_npz overrides each config's
# image_size (and the memory-attention feat_sizes derived from it) to
# preprocessing.IMG_SIZE, so every entry here works regardless of the resolution its own
# yaml file happens to default to -- no per-resolution config variants needed.
CHECKPOINT_MODELS = {
    "sam2.1_tiny": {
        "file_name": "sam2.1_hiera_tiny.pt",
        "url": f"{_SAM2_BASE_URL}/092824/sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "size": 156008466,
    },
    "sam2.1_small": {
        "file_name": "sam2.1_hiera_small.pt",
        "url": f"{_SAM2_BASE_URL}/092824/sam2.1_hiera_small.pt",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "size": 184416285,
    },
    "sam2.1_base_plus": {
        "file_name": "sam2.1_hiera_base_plus.pt",
        "url": f"{_SAM2_BASE_URL}/092824/sam2.1_hiera_base_plus.pt",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "size": 323606802,
    },
    "sam2.1_large": {
        "file_name": "sam2.1_hiera_large.pt",
        "url": f"{_SAM2_BASE_URL}/092824/sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "size": 898083611,
    },
    "sam2_tiny": {
        "file_name": "sam2_hiera_tiny.pt",
        "url": f"{_SAM2_BASE_URL}/072824/sam2_hiera_tiny.pt",
        "config": "configs/sam2/sam2_hiera_t.yaml",
        "size": 155906050,
    },
    "sam2_small": {
        "file_name": "sam2_hiera_small.pt",
        "url": f"{_SAM2_BASE_URL}/072824/sam2_hiera_small.pt",
        "config": "configs/sam2/sam2_hiera_s.yaml",
        "size": 184309650,
    },
    "sam2_base_plus": {
        "file_name": "sam2_hiera_base_plus.pt",
        "url": f"{_SAM2_BASE_URL}/072824/sam2_hiera_base_plus.pt",
        "config": "configs/sam2/sam2_hiera_b+.yaml",
        "size": 323493298,
    },
    "sam2_large": {
        "file_name": "sam2_hiera_large.pt",
        "url": f"{_SAM2_BASE_URL}/072824/sam2_hiera_large.pt",
        "config": "configs/sam2/sam2_hiera_l.yaml",
        "size": 897952466,
    },
}

_DOWNLOAD_BLOCK_SIZE = 1024 * 1024  # 1 MB


def download_checkpoint(name: str, dest_dir: str,
                        progress_callback: Optional[Callable[[int, int], None]] = None) -> str:
    """
    Input: name (key into CHECKPOINT_MODELS), dest_dir (folder to save into),
           progress_callback(current_bytes, total_bytes) -- called after every downloaded
           chunk (total_bytes is 0 if neither the server's Content-Length header nor
           CHECKPOINT_MODELS[name]['size'] is available). Not called at all if the
           checkpoint is already present (nothing to download).
    Return: path to the downloaded .pt file
    Downloads the checkpoint if not already present at dest_dir; skips if it is. Downloads to
    a .part file first and renames on completion, so an interrupted download can't leave a
    truncated file mistaken for a valid checkpoint next time; raises if the download
    completes with a different byte count than CHECKPOINT_MODELS[name]['size'].
    """
    if name not in CHECKPOINT_MODELS:
        raise ValueError(f"unknown checkpoint name {name!r}, choose from {list(CHECKPOINT_MODELS)}")

    details = CHECKPOINT_MODELS[name]
    dest = Path(dest_dir) / details["file_name"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file():
        return str(dest)

    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(details["url"]) as response, open(tmp, "wb") as f:
        total = int(response.headers.get("Content-Length", 0)) or details.get("size", 0)
        current = 0
        if progress_callback is not None:
            progress_callback(current, total)
        while True:
            chunk = response.read(_DOWNLOAD_BLOCK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            current += len(chunk)
            if progress_callback is not None:
                progress_callback(current, total)

    expected = details.get("size")
    if expected and current != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"download of {name!r} incomplete: got {current} bytes, expected {expected}")
    tmp.rename(dest)

    return str(dest)


def resolve_checkpoint(checkpoint: str | None = None, model_cfg: str | None = None) -> tuple[str, str]:
    """
    Input: checkpoint (.pt path or None), model_cfg (hydra config name understood by
           automatic.video_predictor_npz.build_sam2_video_predictor_npz, e.g.
           "configs/sam2.1/sam2.1_hiera_t.yaml", resolved relative to the installed sam2
           package, not the current directory)
    Return: (checkpoint, model_cfg) validated
    Falls back to env vars when an argument is None; raises if checkpoint is missing
    on disk or model_cfg doesn't exist inside the sam2 package.
    """
    checkpoint = checkpoint or os.environ.get(_ENV_CKPT)
    model_cfg = model_cfg or os.environ.get(_ENV_CFG)

    if not checkpoint:
        raise ValueError(f"no checkpoint given and {_ENV_CKPT} not set")
    if not model_cfg:
        raise ValueError(f"no model_cfg given and {_ENV_CFG} not set")

    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    import sam2
    sam2_pkg_dir = Path(sam2.__file__).parent
    if not (sam2_pkg_dir / model_cfg).is_file():
        raise FileNotFoundError(f"model_cfg not found in sam2 package ({sam2_pkg_dir}): {model_cfg}")

    return checkpoint, model_cfg


def resolve_model(name: str, checkpoint_dir: str,
                  progress_callback: Optional[Callable[[int, int], None]] = None) -> tuple[str, str]:
    """
    Input: name (key into CHECKPOINT_MODELS -- selects which SAM2 model), checkpoint_dir
           (folder holding/receiving the checkpoint), progress_callback (see
           download_checkpoint).
    Return: (checkpoint_path, model_cfg), both validated (resolve_checkpoint).
    The name-based counterpart to resolve_checkpoint's raw-path validation: picking a model
    by its CHECKPOINT_MODELS key always pairs the right weights with the right config,
    where two independently-set paths (checkpoint + model_cfg) could drift out of sync.
    Downloads the checkpoint into checkpoint_dir first if it isn't there yet.
    """
    if name not in CHECKPOINT_MODELS:
        raise ValueError(f"unknown model name {name!r}, choose from {list(CHECKPOINT_MODELS)}")
    checkpoint_path = download_checkpoint(name, checkpoint_dir, progress_callback=progress_callback)
    return resolve_checkpoint(checkpoint_path, CHECKPOINT_MODELS[name]["config"])
