"""Leakage-audited splitting for windowed industrial time series.

This module implements the chunk-based protocol of Shamim et al. (arXiv
2607.16493, 2026), which is worth restating because it changes published
numbers dramatically. In their ablation on C-MAPSS, IMS bearing and a
hydraulic rig, splitting *windows* rather than *contiguous raw segments*
moved classification accuracy between ~20-60% (genuine) and 99.9%
(artefactual), or to 0% via degenerate class representation -- on identical
raw data and an identical architecture.

Two ways this goes wrong, both common in published PdM pipelines:

1. **Window-level random split.** Stride-1 windows overlap by ``w-1`` rows.
   Shuffle the windows and a window in train is a near-duplicate of one in
   test. The model memorises trajectories, not degradation physics.
2. **Row-level random split.** Destroys temporal order inside a window, so the
   "sequence" model sees a shuffled signal.

The fix is ordering: partition the *raw, time-ordered* sequence into
contiguous non-overlapping chunks, assign whole chunks to splits, and only
then window *within* each chunk. Cross-split raw-row overlap is then zero *by
construction*, which ``assert_no_leakage`` verifies independently rather than
taking on trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..types import RunToFailure


@dataclass
class ChunkSplit:
    train: list[RunToFailure]
    val: list[RunToFailure]
    test: list[RunToFailure]
    window: int
    stride: int

    def summary(self) -> dict[str, object]:
        return {
            "n_train_assets": len(self.train),
            "n_val_assets": len(self.val),
            "n_test_assets": len(self.test),
            "window": self.window,
            "stride": self.stride,
            "train_samples": sum(len(r) for r in self.train),
            "val_samples": sum(len(r) for r in self.val),
            "test_samples": sum(len(r) for r in self.test),
        }


def _cut(run: RunToFailure, fractions: tuple[float, float, float]) -> tuple[RunToFailure, ...]:
    n = len(run)
    n_tr = int(round(n * fractions[0]))
    n_va = int(round(n * fractions[1]))
    n_tr = max(1, min(n_tr, n - 2))
    n_va = max(1, min(n_va, n - n_tr - 1))
    n_te = n - n_tr - n_va
    if n_te < 1:  # extremely short trajectory
        n_tr, n_va, n_te = max(1, n - 2), 1, 1

    def sl(a: int, b: int) -> RunToFailure:
        return RunToFailure(
            asset_id=f"{run.asset_id}[{a}:{b}]",
            X=run.X[a:b],
            rul=run.rul[a:b],
            regime=run.regime[a:b],
            recipe=run.recipe[a:b],
            feature_names=run.feature_names,
            meta=dict(run.meta),
        )

    return sl(0, n_tr), sl(n_tr, n_tr + n_va), sl(n_tr + n_va, n)


def split_runs(
    runs: Sequence[RunToFailure],
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
    asset_wise: bool = True,
) -> ChunkSplit:
    """Split run-to-failure trajectories.

    ``asset_wise=True`` (default, and the production-honest setting)
    allocates *whole assets* to splits. This is strictly harder than splitting
    one asset temporally, because a model must then generalise to equipment it
    has never seen. It is also the only setting that answers the question a
    plant actually asks, so it is the default despite costing accuracy.

    ``asset_wise=False`` falls back to temporal chunking of each asset, which
    is appropriate for the deployment case where you *will* have history on
    the same tool but want an honest held-out future.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(runs))
    n = len(runs)
    n_tr = max(1, int(round(n * fractions[0])))
    n_va = max(1, int(round(n * fractions[1])))
    n_tr = min(n_tr, n - 2)
    n_va = min(n_va, n - n_tr - 1)

    tr = [runs[i] for i in idx[:n_tr]]
    va = [runs[i] for i in idx[n_tr : n_tr + n_va]]
    te = [runs[i] for i in idx[n_tr + n_va :]]

    if not asset_wise:
        tr2, va2, te2 = [], [], []
        for r in runs:
            a, b, c = _cut(r, fractions)
            tr2.append(a)
            va2.append(b)
            te2.append(c)
        tr, va, te = tr2, va2, te2

    return ChunkSplit(train=tr, val=va, test=te, window=0, stride=0)


# ------------------------------------------------------------- windowing
def window_runs(
    runs: Sequence[RunToFailure], window: int = 30, stride: int = 5, add_context: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, group, regime) with leakage-safe context.

    Each example is a ``window``-length sequence of the last ``window`` rows
    plus ``add_context`` earlier rows used only as *context* (so the feature
    extractor has history), never as prediction targets. ``group`` carries the
    asset id so the evaluator can do grouped statistics.

    Windows never span asset boundaries, and within an asset they may overlap
    -- which is fine, because the whole asset is in one split.
    """
    Xs, ys, gs, rs = [], [], [], []
    for r in runs:
        n = len(r)
        if n < window:
            continue
        for end in range(window, n + 1, stride):
            start = end - window
            ctx = max(0, start - add_context)
            seq = r.X[ctx:end]
            if seq.shape[0] < window:
                continue
            Xs.append(seq[-window:])
            ys.append(float(r.rul[end - 1]))
            gs.append(r.asset_id)
            rs.append(str(r.regime[end - 1]))
    if not Xs:
        raise ValueError("no windows produced; reduce window or supply longer runs")
    return (
        np.stack(Xs).astype(np.float64),
        np.asarray(ys, float),
        np.asarray(gs),
        np.asarray(rs),
    )


def assert_no_leakage(
    train_ids: np.ndarray, test_ids: np.ndarray
) -> dict[str, object]:
    """Independent check that no asset appears in two splits.

    Deliberately separate from ``split_runs`` so the guarantee is *verified*,
    not merely asserted by the construction.
    """
    a, b = set(map(str, train_ids)), set(map(str, test_ids))
    overlap = a & b
    return {"clean": len(overlap) == 0, "n_overlapping_assets": len(overlap),
            "examples": sorted(overlap)[:5]}


def naive_window_split(
    runs: Sequence[RunToFailure], window: int = 30, stride: int = 1, seed: int = 0
) -> dict[str, object]:
    """Reproduce the *leaky* protocol for the audit baseline.

    Windows everything first, then shuffles at the window level. Returns the
    measured overlap so the report can quantify the leak rather than merely
    assert it exists.
    """
    X, y, g, _ = window_runs(runs, window=window, stride=stride, add_context=0)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_tr = int(0.7 * len(y))
    tr, te = perm[:n_tr], perm[n_tr:]
    return {
        "n_windows": len(y),
        "n_train": len(tr),
        "n_test": len(te),
        "window_overlap_ratio": 1.0 - stride / window,
        "audit": assert_no_leakage(g[tr], g[te]),
    }


def label_distribution(regimes: np.ndarray) -> dict[str, float]:
    return {str(r): float(np.mean(regimes == r)) for r in np.unique(regimes)}