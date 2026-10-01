"""Data loading, leakage-audited splitting, and feature engineering."""

from .features import FeatureBuilder, FeatureSpec, featurize, group_healthy
from .split import (
    ChunkSplit,
    assert_no_leakage,
    label_distribution,
    naive_window_split,
    split_runs,
    window_runs,
)

__all__ = [
    "split_runs",
    "window_runs",
    "assert_no_leakage",
    "naive_window_split",
    "label_distribution",
    "ChunkSplit",
    "FeatureSpec",
    "FeatureBuilder",
    "featurize",
    "group_healthy",
]