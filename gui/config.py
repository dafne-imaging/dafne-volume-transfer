import os

from sam2_support_match.automatic.checkpoints import CHECKPOINT_URLS

from gui import REPO

_CKPT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "sam2_support_match")
_CKPT_NAME = "sam2_tiny"  # MedSAM2 tried and dropped: features less informative for this task
os.environ.setdefault("SAM2_SUPPORT_MATCH_CHECKPOINT",
                       os.path.join(_CKPT_DIR, os.path.basename(CHECKPOINT_URLS[_CKPT_NAME])))
os.environ.setdefault("SAM2_SUPPORT_MATCH_MODEL_CFG", "configs/sam2.1_hiera_t512.yaml")

DEMO_SUPPORT = os.path.join(REPO, "examples/data/chaos/chaos_1.npz")
DEMO_QUERY = os.path.join(REPO, "examples/data/amos/amos_0507.npz")
