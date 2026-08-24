import torch
import warnings

def empty_cache(device_type: str) -> None:
    """
    Free cuda cache between stages
    """
    if device_type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == 'mps':
        mps = getattr(torch, 'mps', None)
        if mps is not None and hasattr(mps, 'empty_cache'):
            mps.empty_cache()


def pick_device(requested: str = "auto") -> str:
    """
    Input: requested -- 'auto', 'cuda', 'cuda:N', 'mps' or 'cpu'
    Return: a device string torch can actually use here ('auto' -> cuda, else mps,
            else cpu). An unavailable device falls back to cpu with a warning.
    """
    req = requested.lower()

    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return requested
        warnings.warn(
            f"device={requested!r} requested but torch.cuda.is_available() is False -- "
            "running on cpu (much slower). If this machine has an NVIDIA GPU, check "
            "`nvidia-smi`: a 'Driver/library version mismatch' means the loaded kernel "
            "module and the NVML userspace disagree, and a reboot fixes it.",
            stacklevel=2)
        return "cpu"

    if req == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        warnings.warn(f"device={requested!r} requested but MPS is unavailable -- running on cpu.",
                      stacklevel=2)
        return "cpu"

    return req