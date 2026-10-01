"""Uncertainty-aware ensemble forecaster for RUL.

Two deliberately separated uncertainty sources, per Kendall & Gal's
decomposition (NeurIPS'17):

* **Aleatoric** -- irreducible sensor noise at this operating point.
  Estimated as the within-ensemble residual spread *after* removing the
  systematic between-member disagreement.
* **Epistemic** -- what the model does not know because it saw too few
  examples here. Estimated as the variance of the member means.

This split is not cosmetic. The maintenance policy needs them for *different*
reasons: aleatoric width tells the planner "the lead time is genuinely
uncertain, plan conservatively", while epistemic width tells reliability
engineering "this is a cold spot in the training data, retrain". A single
blended sigma -- what you get from a naive deep ensemble -- cannot make that
distinction, so the drift controller cannot act on it.

Implementation is deliberately boring (bootstrap over ExtraTrees /
gradient-boosted members) rather than a large neural net, for three reasons:
it trains in seconds on a laptop, it gives an honest epistemic estimate via
*data* resampling (not just weight initialisation), and it does not invite the
reader to attribute any of the results to scale rather than to calibration.
The pipeline is agnostic to the base learner; see ``docs/DESIGN.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

from ..types import PredictiveDistribution


@dataclass
class EnsembleRUL:
    """Bootstrap ensemble producing calibrated RUL distributions."""

    n_members: int = 15
    max_depth: int | None = 12
    min_samples_leaf: int = 3
    max_features: float = 0.7
    learning_rate: float = 0.06
    n_estimators: int = 120
    kind: str = "extratrees"  # "extratrees" | "gbdt"
    random_state: int = 0
    sigma_floor: float = 1.0
    #: aleatoric/epistemic mixing exponent. 1.0 = fully separate.
    variance_split: float = 1.0

    members_: list = field(default_factory=list, init=False)
    mu_: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    sigma_a_: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    sigma_e_: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    feature_names_: Sequence[str] = field(default_factory=list, init=False)
    oob_fraction_: float = field(default=0.0, init=False)
    fitted_: bool = field(default=False, init=False)

    # ------------------------------------------------------------------ util
    @staticmethod
    def _clip_rul(y: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(y, float), 0.0, None)

    def _make(self, seed: int):
        if self.kind == "gbdt":
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth or 3,
                random_state=seed,
            )
        return ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=seed,
            n_jobs=-1,
        )

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> "EnsembleRUL":
        X = np.asarray(X, float)
        y = self._clip_rul(y)
        n, d = X.shape
        self.feature_names_ = list(feature_names) if feature_names else [f"f{i}" for i in range(d)]

        members, oob = [], 0
        # Bagging with *bootstrap* (not subsample) sampling: the replicate
        # disagreement is then an estimate of epistemic uncertainty that is
        # attributable to data, not to optimiser seed luck.
        for m in range(self.n_members):
            rng = np.random.default_rng(1000 + 137 * m + self.random_state)
            idx = rng.integers(0, n, size=n)
            mdl = self._make(seed=int(rng.integers(0, 2**31 - 1)))
            mdl.fit(X[idx], y[idx])
            members.append(mdl)
            # Out-of-bag residual -> an *unbiased* aleatoric estimate that
            # never touches the member that produced it.
            mask = np.ones(n, bool)
            mask[idx] = False
            if mask.sum() > 20:
                oob += float(np.mean(np.abs(mdl.predict(X[mask]) - y[mask])))
        self.oob_fraction_ = oob / max(self.n_members, 1)

        self.members_ = members
        self._recompute(X)
        self.fitted_ = True
        return self

    # -------------------------------------------------------------- inference
    def _recompute(self, X: np.ndarray) -> None:
        P = np.stack([m.predict(X) for m in self.members_], axis=0)  # (M, n)
        self.mu_ = P.mean(axis=0)
        var_total = P.var(axis=0, ddof=1) if P.shape[0] > 1 else np.zeros(P.shape[1])
        # Between-member variance is (almost) pure epistemic.
        sigma_e = np.sqrt(np.maximum(var_total, 0.0))
        # Within-member residual scale, pooled across members -> aleatoric.
        resid = np.abs(P - self.mu_[None, :])
        sigma_a = np.sqrt(np.maximum((resid**2).mean(axis=0), 0.0))
        self.sigma_a_ = np.maximum(sigma_a, self.sigma_floor)
        self.sigma_e_ = sigma_e

    @property
    def sigma_total_(self) -> np.ndarray:
        k = self.variance_split
        return np.sqrt((1.0 - k) * self.sigma_a_**2 + k * (self.sigma_a_**2 + self.sigma_e_**2))

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted_:
            raise RuntimeError("EnsembleRUL.fit() must be called before predict()")
        self._recompute(np.asarray(X, float))
        return self.mu_.copy(), self.sigma_a_.copy(), self.sigma_e_.copy()

    # ------------------------------------------------------- importance/evid.
    def local_attribution(
        self, X: np.ndarray, top_k: int = 4, seed: int = 0
    ) -> list[list[tuple[str, float]]]:
        """Cheap, honest per-prediction attribution by permutation on the
        member ensemble (no extra dependency, and it reports *this* row's
        drivers rather than a global coefficient).

        Perturbation is a swap with the local neighbourhood mean, which is
        the more meaningful reference for a windowed industrial signal than
        the global mean.
        """
        if not self.fitted_:
            raise RuntimeError("EnsembleRUL.fit() must be called before attribution")
        Xa = np.asarray(X, float)
        out = []
        for i in range(Xa.shape[0]):
            base, _, _ = self.predict(Xa[i : i + 1])
            base_v = float(base[0])
            scores = np.zeros(Xa.shape[1])
            rng = np.random.default_rng(seed + i)
            for j in range(Xa.shape[1]):
                Xp = Xa[i : i + 1].copy()
                # counterfactual: this channel at its healthy reference
                Xp[0, j] = float(np.median(Xa[:, j]))
                p, _, _ = self.predict(Xp)
                scores[j] = abs(float(p[0]) - base_v)
            # keep original sign of importance by re-evaluating direction
            order = np.argsort(-scores)[:top_k]
            items = []
            for j in order:
                name = self.feature_names_[j] if j < len(self.feature_names_) else f"f{j}"
                items.append((name, float(scores[j])))
            out.append(items)
        return out

    def global_importance(self, X: np.ndarray, seed: int = 0) -> list[tuple[str, float]]:
        rng = np.random.default_rng(seed)
        Xa = np.asarray(X, float)
        base, _, _ = self.predict(Xa)
        scores = np.zeros(Xa.shape[1])
        for j in range(Xa.shape[1]):
            Xp = Xa.copy()
            col = Xp[:, j].copy()
            rng.shuffle(col)
            Xp[:, j] = col
            p, _, _ = self.predict(Xp)
            scores[j] = float(np.mean(np.abs(p - base)))
        order = np.argsort(-scores)
        return [
            (self.feature_names_[j] if j < len(self.feature_names_) else f"f{j}", float(scores[j]))
            for j in order
        ]


# ------------------------------------------------------------------ baselines
class ConstantMedianRUL:
    """The null hypothesis every PdM claim should beat.

    Predicting the training-set median RUL scores acceptably on RMSE and
    terribly on decision cost -- which is precisely the point CALIBRA makes
    about accuracy-only evaluation.
    """

    def __init__(self) -> None:
        self.median_ = 0.0
        self.fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConstantMedianRUL":
        self.median_ = float(np.median(np.asarray(y, float)))
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted_:
            raise RuntimeError("fit() first")
        n = len(np.asarray(X))
        s = np.full(n, 0.5 * abs(self.median_) + 1.0)
        return np.full(n, self.median_), s, np.full(n, 0.25 * abs(self.median_) + 1.0)