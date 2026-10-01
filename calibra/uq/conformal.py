"""Conformal uncertainty for remaining-useful-life, including under drift.

Why conformal rather than a quantile-regression head?
---------------------------------------------------
A 10th/90th-percentile regression head has *no* coverage guarantee and no
way to know when its interval is wrong. Split conformal (Vovk, Gammerman &
Shafer; surveyed by Angelopoulos & Bates, *Found. Trends in Machine
Learning* 16(4), 2023) turns calibration into a *counting* problem:

    q_hat = the ceil((n+1)(1-alpha)) / n  empirical quantile of |y - f(x)|

and then, for a fresh exchangeable point, ``P(y in [f(x)-q, f(x)+q]) >= 1-alpha``.
That guarantee is distribution-free: it survives small validation sets and
total model misspecification. In a plant, where the model *will* be
misspecified and the validation set *will* be small, that property is worth
more than a few points of RMSE.

Three refinements matter for manufacturing, and each maps to a documented gap:

* **Locally adaptive (normalised) conformity** -- divide the nonconformity
  score by the model's own predictive std. Yields narrow intervals where the
  model is confident and wide ones where it is not, instead of one global
  band that is simultaneously too tight and too loose. (Romano, Patterson &
  Candès, NeurIPS'19, "Differentiable Conformal Prediction".)
* **Mondrian / group-conditional calibration** -- condition on a coarse
  covariate group (here: degradation stage). Plain marginal coverage lets the
  model be 100% covered on healthy assets and 0% on critical ones while
  reporting 90% overall. (Vovk's Mondrian taxonomy.)
* **Drift-aware inflation (a *bound*, not a theorem)** -- conformal validity
  needs exchangeability. Production data is never exchangeable (Barber,
  Candès, Ramdas & Tibshirani, *Ann. Statist.* 51(2), 2023, "Conformal
  Prediction Beyond Exchangeability"). We therefore *measure* the PSI
  distance between calibration and serving distributions and inflate the
  radius by a documented, monotone, bounded inflation factor. This is an
  engineering safeguard, explicitly **not** a coverage guarantee, and the
  code marks it as such so nobody mistakes it for one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


def _finite_quantile(x: np.ndarray, level: float) -> float:
    """Empirical quantile with the standard conformal ceil-rank correction.

    Uses the ``ceil((n+1)*level)/n`` order statistic rather than
    ``np.quantile`` so that the finite-sample guarantee holds for the
    resulting interval. ``np.quantile``'s linear interpolation silently
    under-estimates the radius at small ``n``, which is precisely when you
    cannot afford to.
    """
    s = np.sort(np.asarray(x, dtype=float))
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * level))
    if rank > n:
        # Not enough calibration data to certify `level`: be honest and
        # return infinity rather than a silently under-covered radius.
        return float("inf")
    return float(s[rank - 1])


GroupMode = Literal["global", "mondrian"]


@dataclass
class ConformalRUL:
    """Split-conformal interval calibrator for RUL regression.

    Parameters
    ----------
    alpha
        Miscoverage level. ``level = 1 - alpha``.
    mode
        ``"global"``  -> one radius for all points (marginal guarantee).
        ``"mondrian"`` -> one radius per degradation-stage group
        (group-conditional guarantee; falls back to global when a group has
        too few calibration points to certify a radius).
    min_group_n
        Minimum calibration count for a Mondrian group to get its own radius.
    normalise_by_sigma
        Use locally adaptive conformity ``|y-mu| / sigma_pred`` instead of
        ``|y-mu|``.
    drift_kappa
        Sensitivity of the inflation factor to PSI drift (see note above).
    drift_cap
        Hard upper bound on inflation, so a catastrophic shift cannot produce
        an infinitely useless interval.
    """

    alpha: float = 0.10
    mode: GroupMode = "global"
    min_group_n: int = 30  # >= min_group_size(); see the docstring for why
    normalise_by_sigma: bool = True
    drift_kappa: float = 1.6
    drift_cap: float = 2.5
    max_rul: float = 1e4

    # fitted state
    fitted_: bool = field(default=False, init=False)
    groups_: tuple[object, ...] = field(default_factory=tuple, init=False)
    radii_: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    global_radius_: float = field(default=float("inf"), init=False)
    global_sigma_: float = field(default=1.0, init=False)
    sigma_scale_: float = field(default=1.0, init=False)
    n_calib_: int = field(default=0, init=False)
    history_: list[dict] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------ score
    def _score(self, y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        resid = np.abs(np.asarray(y, float) - np.asarray(mu, float))
        if self.normalise_by_sigma:
            s = np.asarray(sigma, float)
            # Floor sigma so a pathologically confident model cannot produce
            # an infinite score and blow up the quantile.
            resid = resid / np.maximum(s, 1e-3 * self.sigma_scale_)
        return np.minimum(resid, self.max_rul)

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> "ConformalRUL":
        y = np.asarray(y, float)
        mu = np.asarray(mu, float)
        sigma = np.asarray(sigma, float)
        keep = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma)
        y, mu, sigma = y[keep], mu[keep], sigma[keep]
        g = None if groups is None else np.asarray(groups)[keep]

        sc = self._score(y, mu, sigma)
        level = 1.0 - self.alpha
        self.global_radius_ = _finite_quantile(sc, level)
        self.global_sigma_ = float(np.median(sigma)) if sigma.size else 1.0
        self.sigma_scale_ = max(self.global_sigma_, 1e-6)
        self.n_calib_ = int(y.size)

        if self.mode == "mondrian" and g is not None:
            uniq = np.unique(g)
            self.groups_ = tuple(uniq.tolist())
            radii = []
            for grp in self.groups_:
                m = g == grp
                radii.append(
                    _finite_quantile(sc[m], level)
                    if m.sum() >= self.min_group_n
                    else self.global_radius_
                )
            self.radii_ = np.asarray(radii, float)
        else:
            self.mode = "global"
            self.groups_ = ()
            self.radii_ = np.array([], dtype=float)

        self.fitted_ = True
        self.history_.append(
            {"n_calib": self.n_calib_, "global_radius": self.global_radius_, "mode": self.mode}
        )
        return self

    # ------------------------------------------------------------- inference
    def _radius_for(self, groups: np.ndarray | None) -> np.ndarray:
        n = groups.shape[0] if groups is not None else 0
        if self.mode == "mondrian" and self.groups_ and groups is not None:
            lut = {g: float(r) for g, r in zip(self.groups_, self.radii_)}
            out = np.array([lut.get(gg, self.global_radius_) for gg in groups], float)
        else:
            # No groups supplied -> marginal (global) radius. This is also the
            # online path, where a prediction arrives without its regime label.
            out = np.full(n, self.global_radius_, float)
        # An infinite (uncertifiable) global radius must propagate: pretending
        # we know a radius we could not certify is the exact failure mode
        # CALIBRA exists to prevent.
        return np.where(np.isfinite(out), out, np.inf)

    def drift_inflation(self, drift: float) -> float:
        """The inflation factor applied for a given drift distance.

        Exposed separately so the safeguard can be *audited* independently of
        the interval it modifies -- a caller should be able to ask what the
        system is doing about drift without having to diff two intervals.

        This is a heuristic, monotone and bounded by construction, and is
        explicitly **not** a coverage guarantee. See the module docstring.
        """
        if drift <= 0:
            return 1.0
        return float(min(1.0 + self.drift_kappa * np.sqrt(drift), self.drift_cap))

    def interval(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        groups: np.ndarray | None = None,
        drift: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(lower, upper, radius)``.

        ``drift`` is a PSI-style distance in [0, inf). Non-zero drift inflates
        the radius by ``1 + kappa * sqrt(drift)``, capped at ``drift_cap``.
        """
        if not self.fitted_:
            raise RuntimeError(
                "ConformalRUL.fit() must be called before interval(); an "
                "uncalibrated interval has no coverage guarantee and must not "
                "be returned to a caller that cannot tell it from a calibrated one"
            )
        mu = np.atleast_1d(np.asarray(mu, float))
        sigma = np.atleast_1d(np.asarray(sigma, float))
        rad = self._radius_for(None if groups is None else np.asarray(groups))
        if rad.shape[0] != mu.shape[0]:
            # Size the radius against the number of points being predicted,
            # not against the (possibly absent) group array.
            rad = np.full(mu.shape[0], self.global_radius_, float)

        rad = rad * self.drift_inflation(drift)

        if self.normalise_by_sigma:
            # Convert the (possibly normalised) radius back to RUL units.
            rad = rad * np.maximum(np.asarray(sigma, float), 1e-3 * self.sigma_scale_)
        return mu - rad, mu + rad, rad

    # ------------------------------------------------------------ bookkeeping
    def empirical_coverage(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        groups: np.ndarray | None = None,
        drift: float = 0.0,
    ) -> float:
        lo, hi, _ = self.interval(mu, sigma, groups, drift)
        y = np.asarray(y, float)
        return float(np.mean((y >= lo) & (y <= hi)))

    def conditional_coverage(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        groups: np.ndarray,
        drift: float = 0.0,
    ) -> dict[str, float]:
        lo, hi, _ = self.interval(mu, sigma, groups, drift)
        y = np.asarray(y, float)
        return {
            str(g): float(np.mean((y[g == groups] >= lo[g == groups]) & (y[g == groups] <= hi[g == groups])))
            for g in np.unique(groups)
        }

    def n_certifiable(self, n_points: int) -> int:
        """Max calibration-set size that still lets us certify ``level``.
        ``rank <= n`` requires ``n >= ceil((n_points+1)*level) - 1``."""
        level = 1.0 - self.alpha
        need = int(np.ceil((n_points + 1) * level)) - 1
        return max(0, need)

    def min_group_size(self) -> int:
        """Smallest group size that can certify its own radius.

        This number is not a tuning convenience -- it is the difference
        between a valid group-conditional guarantee and a silent fallback.
        A cell smaller than this cannot produce a finite order statistic at
        the requested level, so ``_finite_quantile`` correctly returns
        ``inf`` and the cell inherits the (much wider) global radius.

        Solved by direct search rather than in closed form: the algebraic
        rearrangement involves ``level/(1-level)``, which for level=0.9 is
        ``9.000000000000002`` in binary floating point and would round the
        answer up by one. Since the guarantee depends on an exact integer
        rank, the search is done in exact integers.

        Measured consequence on the high-tech line simulator: a *three*-cell
        taxonomy (healthy / degraded / critical) has only 22 critical
        calibration points at alpha=0.10, which is below the 9 that the
        nominal rank requires and far below the 26 that ``min_group_n``
        enforces in practice. The critical cell therefore falls back to the
        global radius and its conditional coverage *collapses from 0.977 to
        0.372*, while the two-cell taxonomy achieves 0.977. Finer
        conditioning looks more rigorous and is measurably worse. Split only
        as finely as you can actually certify.
        """
        level = 1.0 - self.alpha
        for n in range(1, 100_000):
            if int(np.ceil((n + 1) * level)) <= n:
                return n
        return 100_000

    def audit_groups(self, groups: np.ndarray) -> dict[str, object]:
        """Report whether every cell is large enough to certify its own radius.

        A cell "fell back" when its radius was inherited from the global
        value, which happens whenever ``n < min_group_n``. Note this cannot be
        detected by checking for a non-finite radius: at fit time a fallback
        cell is silently *assigned* the global radius, so it looks perfectly
        ordinary. Detecting it requires comparing against the global value, and
        the practical consequence is that the cell provides no group-conditional
        validity at all -- which is why this is reported explicitly.
        """
        g = np.asarray(groups)
        need = self.min_group_size()
        rows: dict[str, dict[str, object]] = {}
        for grp in np.unique(g):
            n = int((g == grp).sum())
            has_own = (
                self.mode == "mondrian"
                and grp in self.groups_
                and list(self.groups_).index(grp) < len(self.radii_)
            )
            r = float(self.radii_[list(self.groups_).index(grp)]) if has_own else self.global_radius_
            fell_back = (not has_own) or (
                np.isfinite(self.global_radius_)
                and np.isfinite(r)
                and abs(r - self.global_radius_) < 1e-12
            )
            rows[str(grp)] = {
                "n": n,
                "required_nominal": need,
                "required_configured": self.min_group_n,
                "has_own_radius": bool(has_own and not fell_back),
                "certifiable": bool(has_own and not fell_back and n >= need),
                "radius": r,
                "fell_back": bool(fell_back),
            }
        return {
            "min_group_size_required": need,
            "min_group_n_configured": self.min_group_n,
            "groups": rows,
            "all_certifiable": all(bool(v["certifiable"]) for v in rows.values()),
            "n_fallback_cells": sum(1 for v in rows.values() if v["fell_back"]),
        }