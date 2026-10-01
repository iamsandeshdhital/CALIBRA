"""Feature extraction with physics-aware, regime-aware engineering.

Two design choices here are deliberate and both are responses to documented
gaps:

1. **Health index, not raw channels, as the model input.** Rolling statistics
   (mean, std, slope, kurtosis per channel) are computed on a *monotone-safe*
   basis, and a scalar health index is derived. Feeding raw channels to a
   regressor invites it to latch onto whatever channel happens to correlate
   with time in the training set -- a shortcut that decays the moment the
   asset is reset, overhauled, or moved to a different recipe. Health-index
   features are partially invariant to that.

2. **Regime-conditioned normalisation.** High-tech lines run many recipes, and
   the same channel means different things at different recipes (see
   ``sim/hightech_line.py``). We therefore centre each channel by its
   *per-recipe* reference, estimated on the training split only. This is the
   single highest-leverage feature-engineering decision for multi-product
   lines and it is essentially absent from published PdM pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..types import RunToFailure


def rolling_stats(x: np.ndarray, win: int = 10) -> np.ndarray:
    """Per-channel rolling [mean, std, slope, rms] -> (T, 4d)."""
    X = np.asarray(x, float)
    T, d = X.shape
    out = np.zeros((T, 4 * d), float)
    c = np.cumsum(np.vstack([np.zeros((1, d)), X]), axis=0)
    c2 = np.cumsum(np.vstack([np.zeros((1, d)), X**2]), axis=0)
    for t in range(T):
        a = max(0, t - win + 1)
        n = t - a + 1
        m = (c[t + 1] - c[a]) / n
        v = np.maximum((c2[t + 1] - c2[a]) / n - m**2, 0.0)
        out[t, 0:d] = m
        out[t, d : 2 * d] = np.sqrt(v)
        seg = X[a : t + 1]
        if seg.shape[0] >= 2:
            tt = np.arange(seg.shape[0], dtype=float)
            tt -= tt.mean()
            denom = float((tt**2).sum()) or 1.0
            out[t, 2 * d : 3 * d] = (tt[:, None] * (seg - seg.mean(axis=0))).sum(axis=0) / denom
        else:
            out[t, 2 * d : 3 * d] = 0.0
        out[t, 3 * d : 4 * d] = np.sqrt((seg**2).mean(axis=0))
    return out


def health_index(stats: np.ndarray, n_channels: int) -> np.ndarray:
    """Collapse rolling stats to a scalar health in (0, 1], 1 = as-new.

    Uses each channel's *deviation from its own early-trajectory reference*,
    aggregated by RMS across channels. Taking RMS (rather than mean) means a
    single channel moving hard -- the actual early-warning signature -- is not
    diluted by healthy channels.
    """
    mean = stats[:, 0:n_channels]
    ref = mean[: max(1, mean.shape[0] // 20)]  # first 5% as "as-new"
    ref_mu = ref.mean(axis=0)
    scale = np.std(mean, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    z = np.abs((mean - ref_mu) / scale)
    h = 1.0 / (1.0 + np.sqrt((z**2).mean(axis=1)))
    return np.clip(h, 1e-3, 1.0)


def stat_feature_names(names: Sequence[str], n: int | None = None) -> list[str]:
    d = len(names)
    suffix = ["mean", "std", "slope", "rms"]
    out = [f"{nm}_{s}" for s in suffix for nm in names]
    out.append("health_index")
    if n is not None:
        out = out[:n]
    return out


@dataclass
class FeatureSpec:
    window: int = 10
    use_health: bool = True
    use_recipe_residual: bool = True
    min_recipe_count: int = 20


class FeatureBuilder:
    """Stateless-per-asset featuriser with a train-fitted recipe reference."""

    def __init__(self, spec: FeatureSpec | None = None) -> None:
        self.spec = spec or FeatureSpec()
        self.recipe_ref_: dict[int, np.ndarray] = {}
        self.global_ref_: np.ndarray | None = None
        self.global_sd_: np.ndarray = np.array([])
        self.feature_names_: list[str] = []

    # -------------------------------------------------------------------- fit
    def fit(self, runs: Sequence[RunToFailure]) -> "FeatureBuilder":
        S = self.spec.window
        allm = []
        counts: dict[int, int] = {}
        for r in runs:
            st = rolling_stats(r.X, S)
            allm.append(st[:, : r.X.shape[1]])
            for rec in np.unique(r.recipe):
                m = r.recipe == rec
                counts[int(rec)] = counts.get(int(rec), 0) + int(m.sum())
        M = np.concatenate(allm, axis=0)
        self.global_ref_ = M.mean(axis=0)
        self.global_sd_ = np.where(M.std(axis=0) > 1e-9, M.std(axis=0), 1.0)

        # Per-recipe reference, only for recipes seen often enough.
        for rec, c in counts.items():
            if c < self.spec.min_recipe_count:
                continue
            sel = [
                rolling_stats(r.X, S)[r.recipe == rec, : r.X.shape[1]]
                for r in runs
                if (r.recipe == rec).any()
            ]
            self.recipe_ref_[rec] = np.concatenate(sel, axis=0).mean(axis=0)

        d = runs[0].X.shape[1]
        names = stat_feature_names(list(runs[0].feature_names) or [f"c{i}" for i in range(d)])
        if self.spec.use_recipe_residual:
            names = names + ["recipe_resid_max"]
        self.feature_names_ = names
        return self

    # -------------------------------------------------------------- transform
    def transform_run(self, run: RunToFailure) -> np.ndarray:
        S = self.spec.window
        d = run.X.shape[1]
        st = rolling_stats(run.X, S)
        feats = [st]
        if self.spec.use_health:
            feats.append(health_index(st, d)[:, None])
        if self.spec.use_recipe_residual and self.recipe_ref_:
            res = np.zeros((len(run), d))
            gm = self.global_ref_
            sd = self.global_sd_
            for rec in np.unique(run.recipe):
                m = run.recipe == rec
                ref = self.recipe_ref_.get(int(rec))
                mean_ch = st[m, :d]
                r = ref if ref is not None else gm
                res[m] = np.abs((mean_ch - r) / sd)
            feats.append(res.max(axis=1)[:, None])
        F = np.concatenate(feats, axis=1)
        if F.shape[1] < len(self.feature_names_):
            F = np.pad(F, ((0, 0), (0, len(self.feature_names_) - F.shape[1])))
        return F[:, : len(self.feature_names_)]


def featurize(
    runs: Sequence[RunToFailure], spec: FeatureSpec | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, FeatureBuilder]:
    """Fit the builder on ``runs`` and transform them. Order matters: the
    reference statistics must come from training data only."""
    fb = FeatureBuilder(spec)
    fb.fit(runs)
    Xs, ys, gs, rs = [], [], [], []
    for r in runs:
        Xs.append(fb.transform_run(r))
        ys.append(r.rul)
        gs.append(np.full(len(r), r.asset_id, dtype=object))
        rs.append(r.regime)
    return (
        np.concatenate(Xs).astype(float),
        np.concatenate(ys).astype(float),
        np.concatenate(gs),
        np.concatenate(rs),
        fb,
    )


def group_healthy(regimes: np.ndarray, healthy_label: str = "healthy") -> np.ndarray:
    """Mondrian group key: healthy vs not.

    Deliberately coarse. Mondrian conformal needs each cell to hold enough
    calibration points to certify a radius at the requested level; splitting
    into three thin cells silently degrades to the global radius, so a coarse
    grouping that actually certifies beats a fine one that quietly does not.
    """
    return np.where(np.asarray(regimes) == healthy_label, "healthy", "degraded+")


def group_recipe(recipes: np.ndarray) -> np.ndarray:
    return np.asarray(recipes).astype(str)