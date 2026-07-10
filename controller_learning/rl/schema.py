"""Lightweight version identifiers shared by optional M7 runtime modules."""

from typing import Final

LOCAL_TRACK_FEATURE_SCHEMA_VERSION: Final = 1
PUBLIC_REWARD_SCHEMA_VERSION: Final = "controller-learning.m7-public-reward.v1"

__all__ = [
    "LOCAL_TRACK_FEATURE_SCHEMA_VERSION",
    "PUBLIC_REWARD_SCHEMA_VERSION",
]
