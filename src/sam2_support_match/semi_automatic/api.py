"""Public API for the semi-automatic matching workflow."""

from dataclasses import asdict

from sam2_support_match.preprocessing import (
    labels_to_masks,
    volume_to_slices,
    volume_to_uint8,
)
from sam2_support_match.semi_automatic.session import SliceMatchSession, collect_anchors
from .types import AcceptedMatch, MatchCandidate, SliceMatchConfig


# CREATE A NEW SEGMENT SESSION
def create_session_from_volumes(
    seg,
    support_volume,
    support_labels,
    roi_names: list[str],
    query_volume,
    config: SliceMatchConfig | None = None,
) -> SliceMatchSession:
    """Build a matching session from raw volumes and a support label map.
       1. support normalization and query uint8 conversion
       2. support volume labels conversion in a dictionary (as dafne convention)
       3. create a slice match session
    """
    options = asdict(config or SliceMatchConfig())
    return SliceMatchSession(
        seg,
        volume_to_slices(volume_to_uint8(support_volume)),
        labels_to_masks(support_labels, roi_names),
        volume_to_slices(volume_to_uint8(query_volume)),
        **options,
    )


__all__ = [
    "SliceMatchSession",
    "SliceMatchConfig",
    "MatchCandidate",
    "AcceptedMatch",
    "create_session_from_volumes",
    "collect_anchors",
]
