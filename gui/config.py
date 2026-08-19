import os

from appdirs import user_cache_dir

from sam2_support_match.automatic.checkpoints import CHECKPOINT_MODELS

from gui import REPO

_CKPT_DIR = user_cache_dir("sam2_support_match")
# finetuned MedSAM2 weights tried and dropped: features less informative for this task
# than plain SAM2; see automatic.checkpoints.CHECKPOINT_MODELS for every other choice.
_CKPT_NAME = "sam2.1_tiny"
_ckpt_details = CHECKPOINT_MODELS[_CKPT_NAME]
os.environ.setdefault("SAM2_SUPPORT_MATCH_CHECKPOINT",
                       os.path.join(_CKPT_DIR, _ckpt_details["file_name"]))
os.environ.setdefault("SAM2_SUPPORT_MATCH_MODEL_CFG", _ckpt_details["config"])

DEMO_SUPPORT = os.path.join(REPO, "examples/data/chaos/chaos_1.npz")
DEMO_QUERY = os.path.join(REPO, "examples/data/amos/amos_0507.npz")
