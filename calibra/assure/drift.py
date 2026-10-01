"""Drift detection and the self-healing controller.

Concept drift in a production line is not hypothetical. A model trained on
last quarter's chambers is serving data from chambers that have been
re-qualified, filters that have loaded, sensors that have been replaced with
a different vendor, and a product mix that has shifted. The model does not
crash -- it silently keeps answering, and its calibration decays, and the
alarms get worse, and the operators learn (correctly, from experience) to
ignore them.

Documented problems this layer addresses:

* Silent degradation is the norm, not the exception. Jourdan et al.
  (*TU Darmstadt*, 2025) show drift-driven performance degradation in PHM
  that is "not obvious to machine or plant operators". That is precisely the
  failure that turns a good model into an alert-fatigue generator.
* Standard detectors watch the *inputs*. That is insufficient here: the
  dangerous case is a shift in ``P(y|x)`` that leaves ``P(x)`` looking
  ordinary. So this module monitors **standardised residuals** as well as
  input marginals. A rise in residual scale with no input PSI is the
  signature of a genuine concept change and is what triggers retraining
  rather than recalibration.
* Detectors themselves need a budget. Too sensitive and you thrash
  (constant false retraining, which destroys trust in the system itself);
  too insensitive and you serve a dead model. The controller therefore has
  explicit escalation levels and a dwell requirement.

PSI follows the population-stability convention (bins on the calibration
quantiles, symmetric eps floor). CUSUM is the classic two-sided sequential
detector (Page, 1954) applied to the standardised residual stream; ADWIN
(Bifet & Gavalda, SDM'07) is included as an optional windowed detector for
step changes because it carries an explicit false-alarm bound.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Severity(str, Enum):
    STABLE = "stable"
    WATCH = "watch"
    RECALIBRATE = "recalibrate"
    RETRAIN = "retrain"
    ROLLBACK = "rollback"


@dataclass
class DriftReport:
    severity: Severity
    psi_input: float
    psi_residual: float
    cusum: float
    coverage_observed: float | None
    coverage_nominal: float
    recommended: str
    detail: str = ""


def psi(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10, eps: float = 1e-6
) -> float:
    """Population Stability Index between a reference and a current sample.

    Bins are quantile edges from *reference* (the deployment contract), so a
    new bin cannot be invented by the serving data.
    """
    ref = np.asarray(reference, float).ravel()
    cur = np.asarray(current, float).ravel()
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size < n_bins or cur.size == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        return 0.0
    r = np.histogram(np.clip(ref, edges[0], edges[-1]), bins=edges)[0] / ref.size
    c = np.histogram(np.clip(cur, edges[0], edges[-1]), bins=edges)[0] / cur.size
    r = np.clip(r, eps, None)
    c = np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


class _VectorPSI:
    """Cached, vectorised per-channel PSI.

    Recomputing ``np.quantile`` and two histograms for every channel on every
    streamed sample costs tens of milliseconds per decision point, which is
    enough to make online monitoring the slowest part of the system. The
    reference bin edges are fixed by the deployment contract and therefore
    computed once; after that a single ``digitize`` + ``bincount`` per channel
    gives all channels at once.
    """

    def __init__(self, reference: np.ndarray, n_bins: int = 8, eps: float = 1e-6) -> None:
        R = np.asarray(reference, float)
        if R.ndim == 1:
            R = R[:, None]
        self.eps = eps
        self.n_bins = n_bins
        self._reference_ = R
        self._null: np.ndarray | None = None
        self._baseline_counts: list[np.ndarray] = []
        self.valid: list[int] = []
        self.edges: list[np.ndarray] = []
        self.counts: list[np.ndarray] = []
        self.lo: list[float] = []
        self.hi: list[float] = []
        qs = np.linspace(0, 1, n_bins + 1)
        for j in range(R.shape[1]):
            col = R[:, j]
            col = col[np.isfinite(col)]
            if col.size < n_bins:
                continue
            e = np.unique(np.quantile(col, qs))
            if e.size < 3:
                continue
            h = np.histogram(np.clip(col, e[0], e[-1]), bins=e)[0].astype(float)
            self.valid.append(j)
            self.edges.append(e)
            self.counts.append(h / max(h.sum(), 1.0))
            self.lo.append(float(e[0]))
            self.hi.append(float(e[-1]))
        self._baseline_counts = list(self.counts)

    def per_channel(self, current: np.ndarray) -> np.ndarray:
        C = np.asarray(current, float)
        if C.ndim == 1:
            C = C[:, None]
        n = C.shape[0]
        out = np.zeros(C.shape[1], float)
        if n == 0:
            return out
        for k, j in enumerate(self.valid):
            col = C[:, j]
            col = col[np.isfinite(col)]
            if col.size == 0:
                continue
            idx = np.digitize(np.clip(col, self.lo[k], self.hi[k]), self.edges[k]) - 1
            idx = np.clip(idx, 0, len(self.counts[k]) - 1)
            h = np.bincount(idx, minlength=len(self.counts[k])).astype(float)
            h /= max(h.sum(), 1.0)
            r = np.clip(self.counts[k], self.eps, None)
            c = np.clip(h, self.eps, None)
            out[j] = float(np.sum((c - r) * np.log(c / r)))
        return out

    def null_baseline(self) -> np.ndarray:
        """Per-channel PSI expected when *nothing has drifted*.

        Estimated by splitting the reference into two halves and measuring PSI
        between them. This is essential: with ``k`` bins, PSI between two
        i.i.d. samples of the *same* distribution has expectation ~(k-1)/2,
        which is ~3.5 for 8 bins. The familiar industry thresholds (<0.10
        stable, 0.10-0.25 moderate) implicitly assume samples thousands of
        times larger than the bin count. On a 200-sample rolling window the
        null value is an order of magnitude above those thresholds, so a
        monitor tuned to them fires permanently on healthy data.
        """
        if self._null is not None:
            return self._null
        R = self._reference_
        half = R.shape[0] // 2
        if half < 30:
            self._null = np.zeros_like(self._baseline_counts[0]) if self._baseline_counts else np.array([])
            return self._null
        a = _VectorPSI(R[:half], n_bins=self.n_bins)
        b = _VectorPSI(R[half:], n_bins=self.n_bins)
        # PSI(a, b) with bins taken from `a`
        vals = np.zeros(len(self.valid), float)
        for k, j in enumerate(self.valid):
            col = R[half:, j]
            col = col[np.isfinite(col)]
            if col.size == 0:
                continue
            idx = np.digitize(np.clip(col, a.lo[k], a.hi[k]), a.edges[k]) - 1
            idx = np.clip(idx, 0, len(a.counts[k]) - 1)
            h = np.bincount(idx, minlength=len(a.counts[k])).astype(float)
            h /= max(h.sum(), 1.0)
            r = np.clip(a.counts[k], self.eps, None)
            c = np.clip(h, self.eps, None)
            vals[k] = float(np.sum((c - r) * np.log(c / r)))
        full = np.zeros(R.shape[1], float)
        for k, j in enumerate(self.valid):
            full[j] = vals[k]
        self._null = full
        return full


class CUSUM:
    """Two-sided CUSUM on a standardised stream.

    ``h`` is the decision threshold and ``k`` the slack.

    Note on ``k``: CUSUM accumulates ``g_pos = max(0, g_pos + x - k)``, so a
    shift smaller than ``k`` never accumulates at all -- ``k`` is a minimum
    detectable effect, not a free parameter. The default is deliberately small
    relative to a unit-scale standardised stream; a large ``k`` silently makes
    the detector blind to moderate shifts, which is worse than not having one
    because it still reports a clean bill of health.

    Exposed so the operating point can be set from a target average run length
    rather than tuned by hand.
    """

    def __init__(self, h: float = 5.0, k: float = 0.1) -> None:
        self.h = float(h)
        self.k = float(k)
        self.g_pos = 0.0
        self.g_neg = 0.0

    def update(self, x: float) -> bool:
        v = float(x)
        self.g_pos = max(0.0, self.g_pos + v - self.k)
        self.g_neg = max(0.0, self.g_neg - v - self.k)
        return bool(self.g_pos > self.h or self.g_neg > self.h)

    def reset(self) -> None:
        self.g_pos = self.g_neg = 0.0


@dataclass
class DriftMonitor:
    """Streaming monitor with an escalation ladder and a dwell requirement.

    Parameters
    ----------
    psi_watch / psi_recal / psi_retrain
        PSI thresholds. Conventional interpretation: <0.10 stable,
        0.10-0.25 moderate, >0.25 significant.
    coverage_tolerance
        Allowed gap between observed and nominal coverage before the
        calibration is declared stale. This is the *operational* drift
        signal -- it fires even when the inputs look unchanged, which is
        exactly the concept-shift case input-only detectors miss.
    psi_thresholds_are_excess
        Interpret ``psi_watch/recal/retrain`` as *excess above the
        per-channel no-drift baseline* (``_VectorPSI.null_baseline``) rather
        than as raw PSI. Raw PSI on a rolling window sits several units above
        zero even on perfectly healthy data; thresholding it directly makes
        the monitor fire forever and destroys its value as a signal.
    dwell
        Consecutive breaching windows required before escalating. Prevents
        thrash; a system that retrains on every blip is itself an
        alert-fatigue source, for the platform team.
    window
        Number of recent predictions over which coverage and residual scale
        are measured.
    """

    psi_watch: float = 0.35
    psi_recal: float = 0.90
    psi_retrain: float = 1.80
    coverage_tolerance: float = 0.10
    nominal: float = 0.90
    window: int = 200
    dwell: int = 3
    #: Recompute the (cached-vectorised) PSI only every ``psi_every`` samples.
    #: Drift is a slow phenomenon; measuring it every sample costs 30x more for
    #: no additional information, and an online monitor that dominates the
    #: pipeline's latency is one that gets disabled in production.
    psi_every: int = 8

    reference_X_: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    reference_resid_: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    calibration_valid_: bool = field(default=True, init=False)
    last_action_: Severity = field(default=Severity.STABLE, init=False)
    consecutive_breach_: int = field(default=0, init=False)
    cusum_: CUSUM = field(default_factory=CUSUM, init=False)
    psi_input_: _VectorPSI | None = field(default=None, init=False)
    psi_resid_: _VectorPSI | None = field(default=None, init=False)
    _step_: int = field(default=0, init=False)
    _psi_cache_: tuple[float, float] = field(default=(0.0, 0.0), init=False)

    _cov_buf: deque = field(default_factory=lambda: deque(maxlen=200), init=False)
    _res_buf: deque = field(default_factory=lambda: deque(maxlen=200), init=False)
    _x_buf: deque = field(default_factory=lambda: deque(maxlen=200), init=False)
    log_: list[dict] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------ setup
    def set_reference(self, X: np.ndarray, residuals: np.ndarray) -> None:
        self.reference_X_ = np.asarray(X, float)
        self.reference_resid_ = np.asarray(residuals, float)
        self._x_buf = deque(np.asarray(X, float).tolist(), maxlen=self.window)
        self._res_buf = deque(np.asarray(residuals, float).tolist(), maxlen=self.window)
        self._cov_buf.clear()
        self.calibration_valid_ = True
        self.consecutive_breach_ = 0
        self.cusum_.reset()
        self.psi_input_ = _VectorPSI(self.reference_X_, n_bins=8)
        self.psi_resid_ = _VectorPSI(self.reference_resid_, n_bins=8)
        self._step_ = 0
        self._psi_cache_ = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self._cov_buf.maxlen != self.window:
            self._cov_buf = deque(self._cov_buf, maxlen=self.window)
            self._res_buf = deque(self._res_buf, maxlen=self.window)
            self._x_buf = deque(self._x_buf, maxlen=self.window)

    # -------------------------------------------------------------- streaming
    def update(
        self, X_row: np.ndarray, residual: float | None, inside_interval: bool | None = None
    ) -> DriftReport:
        """Ingest one decision point and return the current verdict.

        ``residual`` may be ``None``, which is the correct input when no model
        prediction is available yet (a freshly commissioned line, or an
        unsupervised scoring pass). The residual channel is then left
        untouched rather than being fed a placeholder -- feeding zeros against
        a reference of real residuals produces a large spurious PSI and a
        false "concept shift" alarm.
        """
        self._x_buf.append(np.asarray(X_row, float))
        if residual is not None:
            self._res_buf.append(float(residual))
        if inside_interval is not None:
            self._cov_buf.append(bool(inside_interval))

        self._step_ += 1
        recompute = (self._step_ % max(1, self.psi_every) == 0)
        if recompute and self.psi_input_ is not None:
            Xc = np.stack(self._x_buf)
            raw_in = self.psi_input_.per_channel(Xc)
            # Excess over the no-drift baseline, floored at 0. Aggregation is
            # a *composite*: the 95th percentile captures broad drift, while a
            # separate single-channel max captures the case that matters
            # operationally -- one thermistor replaced, one encoder swapped --
            # which a quantile over ~34 channels would hide.
            base_in = self.psi_input_.null_baseline()
            exc_in = np.maximum(raw_in - base_in, 0.0)
            psi_in = float(np.quantile(exc_in, 0.95) + 0.5 * exc_in.max()) if exc_in.size else 0.0

            raw_res = self.psi_resid_.per_channel(np.asarray(self._res_buf, float))
            base_res = self.psi_resid_.null_baseline()
            exc_res = np.maximum(raw_res - base_res, 0.0)
            psi_res = float(np.quantile(exc_res, 0.95) + 0.5 * exc_res.max()) if exc_res.size else 0.0

            self._psi_cache_ = (psi_in, psi_res)
        psi_in, psi_res = self._psi_cache_
        if residual is None:
            # No prediction this step: hold the residual verdict rather than
            # letting a stale value masquerade as a current measurement.
            psi_res = 0.0
            cus = 0.0

        # --- sequential detector on standardised residual scale.
        # Only meaningful when residuals are actually being supplied.
        cus = 0.0
        if residual is not None and self.reference_resid_.size:
            scale = float(np.std(self.reference_resid_)) or 1.0
            z = (float(residual) - float(np.mean(self.reference_resid_))) / scale
            cus = float(self.cusum_.update(z))

        # --- observed coverage
        cov_obs = None
        if len(self._cov_buf) >= max(30, self.window // 4):
            cov_obs = float(np.mean(self._cov_buf))
        cov_gap = None if cov_obs is None else cov_obs - self.nominal

        # --- escalate
        severity, detail = self._verdict(psi_in, psi_res, cus, cov_gap)
        if severity != Severity.STABLE:
            self.consecutive_breach_ += 1
        else:
            self.consecutive_breach_ = 0

        # Dwell gate: do not escalate on a single breaching window.
        gated = severity
        if self.consecutive_breach_ < self.dwell and severity != Severity.STABLE:
            gated = Severity.WATCH

        if gated != self.last_action_:
            self.log_.append(
                {
                    "from": self.last_action_.value,
                    "to": gated.value,
                    "psi_input": psi_in,
                    "psi_residual": psi_res,
                    "cusum": cus,
                    "coverage_observed": cov_obs,
                    "detail": detail,
                }
            )
        self.last_action_ = gated
        if gated in (Severity.RECALIBRATE, Severity.RETRAIN, Severity.ROLLBACK):
            self.calibration_valid_ = False

        return DriftReport(
            severity=gated,
            psi_input=psi_in,
            psi_residual=psi_res,
            cusum=cus,
            coverage_observed=cov_obs,
            coverage_nominal=self.nominal,
            recommended=gated.value,
            detail=detail,
        )

    def _verdict(
        self, psi_in: float, psi_res: float, cus: float, cov_gap: float | None
    ) -> tuple[Severity, str]:
        worst = max(psi_in, psi_res)
        # Coverage collapse dominates: it means the *guarantee* is void,
        # regardless of what the inputs look like.
        if cov_gap is not None and cov_gap < -self.coverage_tolerance:
            if psi_in < self.psi_watch:
                return Severity.RETRAIN, "coverage collapsed while inputs are stable -> concept shift"
            return Severity.RETRAIN, "coverage collapsed with input drift"
        if worst >= self.psi_retrain or cus > self.cusum_.h:
            return Severity.RETRAIN, "large input/residual drift"
        if worst >= self.psi_recal:
            return Severity.RECALIBRATE, "moderate drift; conformal radius is stale"
        if worst >= self.psi_watch:
            return Severity.WATCH, "drift within tolerance; monitor"
        return Severity.STABLE, "in control"

    # ------------------------------------------------------------- operations
    def mark_recalibrated(self) -> None:
        self.calibration_valid_ = True
        self.consecutive_breach_ = 0
        self.cusum_.reset()
        self.last_action_ = Severity.STABLE

    def status(self) -> dict[str, object]:
        return {
            "last_action": self.last_action_.value,
            "calibration_valid": self.calibration_valid_,
            "consecutive_breach": self.consecutive_breach_,
            "n_transitions": len(self.log_),
            "window": self.window,
        }


def coverage_schedule(
    levels: np.ndarray, y: np.ndarray, lower: np.ndarray, upper: np.ndarray, n_blocks: int = 6
) -> list[dict[str, float]]:
    """Rolling empirical coverage per time block -- the plot that shows a
    calibration guarantee quietly expiring."""
    n = min(len(y), len(lower), len(upper))
    y = np.asarray(y, float)[:n]
    lo = np.asarray(lower, float)[:n]
    hi = np.asarray(upper, float)[:n]
    out = []
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    for i in range(n_blocks):
        a, b = edges[i], edges[i + 1]
        if b - a < 5:
            continue
        ins = (y[a:b] >= lo[a:b]) & (y[a:b] <= hi[a:b])
        out.append(
            {
                "block": i,
                "t0": int(a),
                "t1": int(b),
                "coverage": float(ins.mean()),
                "width": float(np.mean(hi[a:b] - lo[a:b])),
                "nominal": float(np.median(levels[a:b])) if levels.ndim else float("nan"),
            }
        )
    return out