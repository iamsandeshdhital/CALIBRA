"""Core datatypes for CALIBRA.

Deliberately dependency-light (numpy only) so the decision/UQ layers can be
embedded in an edge agent or audited independently of the modelling stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

import numpy as np


class Action(str, Enum):
    """Maintenance action ladder, ordered by increasing intrusiveness.

    The ladder encodes a real constraint of high-tech manufacturing: the
    further you climb, the higher the *discretionary* cost and the more
    scarce the resource (metrology capacity, spare-part lead time, line
    downtime). CALIBRA never proposes an action without pricing the whole
    ladder, because in fab/battery/precision-machining lines the cheapest
    decision is almost never the most conservative one.
    """

    RUN = "run"  # do nothing, keep producing
    WATCH = "watch"  # tighten sampling, no physical intervention
    INSPECT = "inspect"  # metrology re-check / AOI re-scan (minutes)
    RECIPE = "recipe_tune"  # in-line process parameter correction (free-ish)
    RECOVER = "recover"  # PM swap during planned window (spares + labour)
    STOP = "stop"  # controlled line stop / lot hold (very expensive)


ACTION_ORDER: tuple[Action, ...] = (
    Action.RUN,
    Action.WATCH,
    Action.INSPECT,
    Action.RECIPE,
    Action.RECOVER,
    Action.STOP,
)

#: Index -> Action. Use for matrix construction and report ordering.
ACTION_LIST: tuple[Action, ...] = ACTION_ORDER


class Regime(str, Enum):
    """Ground-truth machine state, i.e. the label we are trying to avoid
    having to hand-label in a real plant."""

    HEALTHY = "healthy"  # within spec
    DEGRADED = "degraded"  # measurable drift, still producing
    CRITICAL = "critical"  # defect will escape to the next station / field


REGIME_ORDER: tuple[str, ...] = (
    Regime.HEALTHY.value,
    Regime.DEGRADED.value,
    Regime.CRITICAL.value,
)


class Health(str, Enum):
    """Self-assessed trustworthiness of a single prediction.

    This is CALIBRA's answer to the *selective prediction* literature
    (Geifman & El-Yaniv, ICML'19): rather than silently emitting a wrong
    number, the system emits a calibrated interval plus an explicit
    self-assessment, and abstains when the interval is too wide to act on.
    """

    TRUSTED = "trusted"  # interval narrow enough to commit to an action
    WIDE = "wide"  # usable only for WATCH
    ABSTAIN = "abstain"  # do not act; escalate to data/relabelling review


@dataclass(frozen=True)
class PredictiveDistribution:
    """A calibrated predictive distribution over Remaining Useful Life.

    Attributes
    ----------
    mean, aleatoric, epistemic, std
        Point estimate and its two uncertainty components, following the
        aleatoric/epistemic decomposition of Kendall & Gal (NeurIPS'17).
        ``std = sqrt(aleatoric**2 + epistemic**2)``.
    lower, upper, level
        Conformal interval at nominal coverage ``level`` (0-1). If the model
        has never seen a failure, these come from split conformal, whose
        guarantee is distribution-free (Angelopoulos & Bates, 2023).
    """

    mean: float
    aleatoric: float
    epistemic: float
    lower: float
    upper: float
    level: float

    @property
    def std(self) -> float:
        return float(np.hypot(self.aleatoric, self.epistemic))

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)

    def probability_below(self, threshold: float) -> float:
        """P(RUL < threshold) under the Gaussian approximation.

        Only used for reporting and for outcome-binning; the *decisions*
        are made from the conformal set, not from this plug-in tail, so a
        mis-calibrated tail cannot by itself trigger a line stop.
        """
        s = max(self.std, 1e-6)
        from math import erf, sqrt

        z = (threshold - self.mean) / (s * np.sqrt(2.0))
        return float(0.5 * (1.0 + erf(z)))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mean, max(self.std, 1e-6), size=n)


@dataclass(frozen=True)
class CostMatrix:
    """Cost of taking action ``a`` when reality is ``regime``.

    Units are currency per event (any consistent unit; the demo uses EUR).
    The off-diagonal entries are what actually happen in a plant:
    ``c_FN`` is scrap + customer penalty, ``c_FP`` is metrology + eng time.
    """

    matrix: Mapping[str, Mapping[str, float]]

    def get(self, action: Action | str, regime: str) -> float:
        a = action.value if isinstance(action, Action) else str(action)
        return float(self.matrix[a][regime])

    def row(self, action: Action | str) -> np.ndarray:
        a = action.value if isinstance(action, Action) else str(action)
        return np.array([self.matrix[a][r] for r in REGIME_ORDER], dtype=float)

    def as_array(self) -> np.ndarray:
        return np.stack([self.row(a) for a in ACTION_LIST])

    @property
    def cost_miss(self) -> float:
        """Cost of the best possible action failing to prevent a critical
        outcome (DAG) vs. the cheapest detection action. This ratio, not the
        FPR, is what should drive alert economics."""
        return float(np.min(self.row(Action.RUN)[REGIME_ORDER.index(Regime.CRITICAL.value) :]))


@dataclass(frozen=True)
class Evidence:
    """Per-prediction evidence bundle handed to the maintenance engineer.

    CALIBRA's operating principle: an operator who cannot see *why* an
    alarm fired will not act on it, and an alarm that is not acted on has
    negative expected value (it burns trust). Hence every alert ships with
    its evidence and its expected cost, not just a number.
    """

    top_features: Sequence[tuple[str, float]]
    n_calibration_points: int
    calibration_level: float
    drift_score: float
    coverage_recent: float | None
    expected_cost: Mapping[str, float]
    chosen_action: Action
    health: Health
    rationale: str = ""


@dataclass
class RunToFailure:
    """A single asset trajectory plus ground truth used only for scoring.

    ``X`` is ``(T, d)`` raw sensor telemetry, ``rul`` is ``(T,)`` cycles-to-
    failure, ``regime`` is ``(T,)`` ground-truth state, ``recipe`` is
    ``(T,)`` which process recipe was running (high-tech lines are
    multi-product, which breaks stationarity -- see NEED.md S4).
    """

    asset_id: str
    X: np.ndarray
    rul: np.ndarray
    regime: np.ndarray
    recipe: np.ndarray
    feature_names: Sequence[str] = field(default_factory=list)
    meta: Mapping[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.X.shape[0])