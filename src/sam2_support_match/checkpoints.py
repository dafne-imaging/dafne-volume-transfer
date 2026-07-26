import os
import urllib.request
from pathlib import Path

_ENV_CKPT = "SAM2_SUPPORT_MATCH_CHECKPOINT"
_ENV_CFG = "SAM2_SUPPORT_MATCH_MODEL_CFG"

_SAM2_TINY_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
_MEDSAM2_URL = "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"

CHECKPOINT_URLS = {
    "sam2_tiny": _SAM2_TINY_URL,
    "medsam2": _MEDSAM2_URL,
}


def download_checkpoint(name: str, dest_dir: str) -> str:
    """
    Input: name ('sam2_tiny' or 'medsam2'), dest_dir (folder to save into)
    Return: path to the downloaded .pt file
    Downloads the checkpoint if not already present at dest_dir; skips if it is.
    """
    if name not in CHECKPOINT_URLS:
        raise ValueError(f"unknown checkpoint name {name!r}, choose from {list(CHECKPOINT_URLS)}")

    url = CHECKPOINT_URLS[name]
    dest = Path(dest_dir) / Path(url).name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not dest.is_file():
        urllib.request.urlretrieve(url, dest)

    return str(dest)


def resolve_checkpoint(checkpoint: str | None = None, model_cfg: str | None = None) -> tuple[str, str]:
    """
    Input: checkpoint (.pt path or None), model_cfg (yaml path or None)
    Return: (checkpoint, model_cfg) as validated existing paths
    Falls back to env vars when an argument is None; raises if a path is still
    missing or doesn't exist on disk.
    """
    checkpoint = checkpoint or os.environ.get(_ENV_CKPT)
    model_cfg = model_cfg or os.environ.get(_ENV_CFG)

    if not checkpoint:
        raise ValueError(f"no checkpoint given and {_ENV_CKPT} not set")
    if not model_cfg:
        raise ValueError(f"no model_cfg given and {_ENV_CFG} not set")

    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not Path(model_cfg).is_file():
        raise FileNotFoundError(f"model_cfg not found: {model_cfg}")

    return checkpoint, model_cfg
