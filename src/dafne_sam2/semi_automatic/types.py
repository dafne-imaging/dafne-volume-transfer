from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SliceMatchConfig:
    thr_hi: float = 0.7
    thr_lo: float = 0.3
    body_thresh: float = 10.0
    body_min_px: int = 50
    score_thresh: float = 0.0
    score_mode: str = "sum_margin"
    search_radius: int = 8
    min_gap: int = 3
    enhanced: bool = False
    align: bool | None = None
    soft_pool: bool | None = None
    pool_gamma: float = 1.0


@dataclass(frozen=True, slots=True)
class MatchCandidate:
    roi_name: str
    query_index: int
    support_index: int
    similarity: float


@dataclass(frozen=True, slots=True)
class AcceptedMatch:
    candidate: MatchCandidate
    score: float
    mask: np.ndarray
