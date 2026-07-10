"""Procedural track geometry and benchmark pools."""

from controller_learning.tracks.assets import (
    TRACK_ASSET_SCHEMA_VERSION,
    TrackAssetError,
    TrackAssetManifest,
    TrackAssetRecord,
    load_manifest_track_batch,
    load_track_asset_manifest,
    load_track_batch_npz,
    save_track_batch_npz,
    validate_track_batch,
    write_track_asset_manifest,
)
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
from controller_learning.tracks.hashing import (
    track_batch_geometry_sha256,
    track_geometry_sha256,
)
from controller_learning.tracks.level0 import (
    DEFAULT_LEVEL0_CAPACITY,
    LEVEL0_TRACK_SEED,
    Level0TrackSpec,
    build_level0_candidate,
    build_level0_track,
)
from controller_learning.tracks.pool import (
    TrackPool,
    TrackPoolSplit,
    gather_track_batch,
    masked_replace_track_batch,
    track_pool_indices,
)
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
    validation_spec_from_project,
)
from controller_learning.tracks.types import (
    Track,
    TrackBatch,
    TrackCapacity,
    TrackSchemaError,
    stack_tracks,
    track_array_bytes,
    track_from_batch_row,
)
from controller_learning.tracks.validator import (
    TrackValidationSpec,
    ValidationResult,
    validate_track_candidate,
)

__all__ = [
    "DEFAULT_LEVEL0_CAPACITY",
    "LEVEL0_TRACK_SEED",
    "TRACK_ASSET_SCHEMA_VERSION",
    "Level0TrackSpec",
    "Track",
    "TrackAssetError",
    "TrackAssetManifest",
    "TrackAssetRecord",
    "TrackBatch",
    "TrackCandidate",
    "TrackCapacity",
    "TrackGenerationError",
    "TrackGenerationSpec",
    "TrackPool",
    "TrackPoolSplit",
    "TrackSchemaError",
    "TrackValidationSpec",
    "ValidationResult",
    "build_level0_candidate",
    "build_level0_track",
    "closed_polyline_self_intersections",
    "cross_2d",
    "gather_track_batch",
    "generate_track_candidate",
    "generation_spec_from_project",
    "load_manifest_track_batch",
    "load_track_asset_manifest",
    "load_track_batch_npz",
    "masked_replace_track_batch",
    "minimum_nonlocal_clearance",
    "pack_track",
    "point_segment_distance",
    "save_track_batch_npz",
    "segment_distance",
    "segments_intersect",
    "signed_area",
    "stack_tracks",
    "track_array_bytes",
    "track_batch_geometry_sha256",
    "track_capacity_from_project",
    "track_from_batch_row",
    "track_geometry_sha256",
    "track_pool_indices",
    "validate_track_batch",
    "validate_track_candidate",
    "validation_spec_from_project",
    "write_track_asset_manifest",
]
