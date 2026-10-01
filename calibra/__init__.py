"""CALIBRA -- Cost-Aware, Leakage-audited, Interpretable Bayesian Reliability Advisor.

An industrial-AI reference implementation for high-tech manufacturing lines
(fab, battery/EV cell, precision machining, photonics).

The problem it targets, in one line: predictive-maintenance models are
trained and scored on the wrong objective. They are optimised for point
accuracy (RMSE) and evaluated there, but a plant does not fail because an RUL
estimate was off by 12 cycles -- it fails because the model was
*overconfident*, *never abstained*, *silently drifted*, and *could not say
what an alert would cost or whether anyone would act on it*. CALIBRA replaces
accuracy with a calibrated decision contract:

    calibrated interval -> regime probabilities -> expected-cost minimisation
    -> abstention gate -> conformal risk budget -> streaming drift control

Layer map
---------
```
types      Action ladder, CostMatrix, PredictiveDistribution, RunToFailure
sim        High-tech line simulator (label imbalance, sensor faults, drift)
data       Leakage-audited splits, physics-aware features, real-dataset loaders
models     Bootstrap ensemble with aleatoric/epistemic decomposition
uq         Split + Mondrian conformal prediction, drift-aware inflation
decision   Cost matrices, expected-cost policy, conformal risk control
assure     PSI / CUSUM residual monitor, severity escalation, rollback
evidence   Evidence cards (attribution, analog, cost table, self-assessment)
metrics    CRPS, Winkler, PIT calibration, ECA, Trust-Adjusted Recall
pipeline   CalibraRunner -- the six stages wired together
```
"""

__version__ = "1.0.0"

from .types import (
    ACTION_LIST,
    REGIME_ORDER,
    Action,
    CostMatrix,
    Evidence,
    Health,
    PredictiveDistribution,
    Regime,
    RunToFailure,
)
from .pipeline import CalibraConfig, CalibraRunner, format_report

__all__ = [
    "ACTION_LIST",
    "REGIME_ORDER",
    "Action",
    "CostMatrix",
    "Evidence",
    "Health",
    "PredictiveDistribution",
    "Regime",
    "RunToFailure",
    "CalibraConfig",
    "CalibraRunner",
    "format_report",
    "__version__",
]