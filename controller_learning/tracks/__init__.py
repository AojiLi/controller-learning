"""Procedural track geometry and benchmark pools."""

from controller_learning.tracks.generator import (
    TrackCandidate,
    TrackGenerationError,
    TrackGenerationSpec,
    generate_track_candidate,
    pack_track,
)
from controller_learning.tracks.geometry import (
    closed_polyline_self_intersections,
    cross_2d,
    minimum_nonlocal_clearance,
    point_segment_distance,
    segment_distance,
    segments_intersect,
    signed_area,
)
from controller_learning.tracks.types import (
    Track,
    TrackBatch,
    TrackCapacity,
    TrackSchemaError,
    stack_tracks,
    track_array_bytes,
)
from controller_learning.tracks.validator import (
    TrackValidationSpec,
    ValidationResult,
    validate_track_candidate,
)

__all__ = [
    "Track",
    "TrackBatch",
    "TrackCandidate",
    "TrackCapacity",
    "TrackGenerationError",
    "TrackGenerationSpec",
    "TrackSchemaError",
    "TrackValidationSpec",
    "ValidationResult",
    "closed_polyline_self_intersections",
    "cross_2d",
    "generate_track_candidate",
    "minimum_nonlocal_clearance",
    "pack_track",
    "point_segment_distance",
    "segment_distance",
    "segments_intersect",
    "signed_area",
    "stack_tracks",
    "track_array_bytes",
    "validate_track_candidate",
]
