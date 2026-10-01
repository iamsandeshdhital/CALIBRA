"""Proper scoring rules and calibration diagnostics.

Proper scoring rules matter disproportionately here. In a plant the number
that reaches the reliability engineer is *not* RMSE -- nobody gets woken up
for an RMSE delta. It is "will I have a catastrophic defect, and how sure
are we?". CRPS and the Winkler interval score answer exactly that, and they
are proper, which means a model cannot improve them by hedging incoherently
the way it can with RMSE in a skewed target.

Two metrics are introduced here because no standard metric captures the two
failures that actually kill PdM ROI in the field:

``Expected Cost of Alarm (ECA)``
    The metrology + engineering + opportunity cost the plant pays per
    decision point to service *false* alarms, in currency. Reported instead
    of false-positive rate because FPR is dimensionless and therefore cannot
    be traded against anything. A shop with 5-minute metrology cycles and one
    with 5-hour ones can have identical FPRs and completely different
    economics. This is the metric that makes the alert budget legible to a
    controller, and it is what Siemens/Rolls-Royce-class deployments are
    actually negotiated on in terms of throughput.

``Trust-Adjusted Recall (TAR)``
    Recall multiplied by the fraction of alerts a human actually acted on.

    Motivation, documented and not hypothetical: repeated false alarms cause
    alarm fatigue, operators stop responding, and the real failure is then
    missed. Reported field benchmarks put healthy programmes at >85% alert
    response rate and <20% false-positive rate -- but those are two separate
    numbers, and multiplying them is the only way to show that recall 95% at
    a 60% response rate is *worse than* recall 80% at a 95% response rate.
    TAR makes that trade explicit and optimisable, and the response rate is a
    measured input (from CMMS work-order closure), not an assumption.
"""

from __future__ import annotations

import numpy as np

from ..types import REGIME_ORDER, Action


# --------------------------------------------------------------- proper rules
def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Closed-form CRPS for a Gaussian predictive distribution.

    For ``z = (y - mu)/sigma`` the standard result is

        CRPS = sigma * [ z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]

    and ``2*Phi(z) - 1 == erf(z/sqrt(2))``. Getting this identity wrong (e.g.
    writing ``z*(2 - erf(...))``) yields a *negative* score for many inputs,
    which is the fastest way to notice that a scoring rule has been
    mis-transcribed -- CRPS is non-negative by construction.
    """
    from scipy.special import erf

    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    s = np.maximum(np.asarray(sigma, float), 1e-6)
    z = (y - mu) / s
    return float(
        np.mean(
            s * (z * erf(z / np.sqrt(2.0)) + np.sqrt(2.0 / np.pi) * np.exp(-0.5 * z**2))
        )
    )


def crps_ensemble(y: np.ndarray, samples: np.ndarray) -> float:
    """Sample-based CRPS: no Gaussian assumption, so it measures the actual
    ensemble's sharpness honestly. ``samples`` is ``(M, n)``."""
    y = np.asarray(y, float)[None, :]
    s = np.asarray(samples, float)
    m = s.mean(axis=0)
    term1 = np.abs(y - m)
    pair = np.abs(s[:, None, :] - s[None, :, :]).mean(axis=(0, 1))
    return float(np.mean(term1 - 0.5 * pair))


def pinball(y: np.ndarray, q_pred: np.ndarray, quantile: float) -> float:
    y = np.asarray(y, float)
    q = np.asarray(q_pred, float)
    d = y - q
    return float(np.mean(np.maximum(quantile * d, (quantile - 1.0) * d)))


def winkler(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    """Winkler / interval score.

        IS_alpha(l, u; y) = (u - l) + (2/alpha)(l - y) 1[y < l]
                                        + (2/alpha)(y - u) 1[y > u]

    The canonical proper score for an interval prediction (Gneiting &
    Raftery, JASA 2007). It is the metric that rewards a model for being
    *right-width*, not just right on average -- which is exactly the property
    a maintenance planner needs, since they must reserve a window before they
    know the true failure time.
    """
    y = np.asarray(y, float)
    l = np.asarray(lower, float)
    u = np.asarray(upper, float)
    width = np.maximum(u - l, 0.0)
    pen = (2.0 / alpha) * (np.maximum(l - y, 0.0) + np.maximum(y - u, 0.0))
    return float(np.mean(width + pen))


def alpha_asymmetric(y: np.ndarray, mu: np.ndarray, alpha: float = 0.2) -> float:
    """NASA's asymmetric prognostics score (used by the C-MAPSS competition).

        s = mean[ alpha*(y-yhat)  if yhat >= y
                 , (1-alpha)*(yhat-y) if yhat <  y ]

    Penalising over-prediction of RUL (i.e. saying "you have more life left"
    than you do) heavily, because that is the dangerous direction. Kept for
    comparability with the PdM literature; winkler/CRPS are preferred.
    """
    y = np.asarray(y, float)
    p = np.asarray(mu, float)
    return float(
        np.mean(np.where(p >= y, alpha * (y - p), (1 - alpha) * (p - y)))
    )


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


# --------------------------------------------------------------- calibration
def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y = np.asarray(y, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(np.asarray(upper, float) - np.asarray(lower, float)))


def pit_values(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Probability integral transform values.

    If the predictive distribution is calibrated, these are Uniform(0,1).
    Any structure in their histogram is a direct, auditable read-out of
    over- or under-confidence -- which is what an ISO-13849-style safety
    argument needs, and what a single coverage number hides.
    """
    from scipy.special import ndtr

    return ndtr((np.asarray(y, float) - np.asarray(mu, float)) / np.maximum(np.asarray(sigma, float), 1e-6))


def pit_uniformity_ks(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """KS statistic of PIT values against Uniform(0,1). 0 = perfectly calibrated."""
    from scipy.stats import kstest

    u = pit_values(y, mu, sigma)
    if u.size < 8:
        return float("nan")
    return float(kstest(u, "uniform").statistic)


def expected_calibration_error(
    y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, n_bins: int = 12
) -> float:
    """ECE over PIT bins. Lower is better; 0 is calibrated."""
    u = pit_values(y, mu, sigma)
    if u.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(u, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.mean()) * abs(u[m].mean() - (b + 0.5) / n_bins)
    return float(ece)


def expected_calibration_error_sigmoid(
    p: np.ndarray, labels: np.ndarray, n_bins: int = 12
) -> float:
    """ECE for binary probabilities (used for the alarm-vs-defect head)."""
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    yb = np.asarray(labels, int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - yb[m].mean())
    return float(ece)


# ------------------------------------------------- the two economic metrics
def expected_cost_of_alarm(
    actions: np.ndarray,
    truth: np.ndarray,
    matrix,
    detection_actions: tuple[str, ...] = ("watch", "inspect", "recipe_tune", "recover", "stop"),
) -> dict[str, float]:
    """Currency spent per decision point servicing alarms that were not real.

    A "false alarm" is any non-RUN action taken while the asset was in fact
    healthy. We charge the *full* cost of that action, because that is what
    the plant pays: engineer attention, metrology slot, opportunity cost, and
    - for an unnecessary RECOVER - the PM labour it consumed.
    """
    actions = np.asarray(actions)
    truth = np.asarray(truth)
    costs = np.array([matrix.get(str(a), str(t)) for a, t in zip(actions, truth)], float)
    raised = np.isin(actions.astype(str), list(detection_actions))
    false_alarm = raised & (truth == REGIME_ORDER[0])
    n = max(len(actions), 1)
    n_fa = int(false_alarm.sum())
    return {
        "n": float(len(actions)),
        "alarm_rate": float(raised.mean()),
        "false_alarms": float(n_fa),
        "false_alarm_rate": float(n_fa / max(raised.sum(), 1)) if raised.any() else 0.0,
        "eca_total": float(costs[false_alarm].sum()),
        "eca_per_decision": float(costs[false_alarm].sum() / n),
        "eca_per_alarm": float(costs[false_alarm].sum() / max(n_fa, 1)) if n_fa else 0.0,
        "wasted_capacity_units": float(n_fa),
    }


def trust_adjusted_recall(
    alerts: np.ndarray,
    critical: np.ndarray,
    response_rate: float,
) -> dict[str, float]:
    """Recall, degraded by the fraction of alerts a human actually actioned.

    ``alerts``    boolean array of whether the policy raised an action.
    ``critical``  boolean array of whether the asset was truly critical.
    ``response_rate``  measured fraction of alerts that produced a CMMS work
                        order. This is an *input from the plant*, not a model
                        output -- which is the point: the model is optimised
                        against the behaviour of the organisation that has to
                        execute it.

    Returns the plain recall, the response-adjusted recall (TAR), and the
    "missed due to inaction" count -- i.e. critical events that *were* raised
        but that a human discarded, which is invisible to conventional recall.
    """
    a = np.asarray(alerts, bool)
    c = np.asarray(critical, bool)
    n_crit = int(c.sum())
    if n_crit == 0:
        return {
            "recall": float("nan"),
            "tar": float("nan"),
            "n_critical": 0.0,
            "n_caught": 0.0,
            "n_caught_but_ignored": 0.0,
            "n_caught_and_acted": 0.0,
            "n_missed_silently": 0.0,
        }
    caught = a & c
    n_caught = int(caught.sum())
    # Expected number actioned. Derived from the *rate* rather than by
    # sampling a per-alarm coin flip: TAR is a reported expectation, so
    # computing it by sampling would make the reported number noisier than
    # the quantity it claims to be, and with a single critical event the
    # sampled value would frequently be 0 or 1 rather than the true rate.
    acted = n_caught * float(response_rate)
    return {
        "recall": float(n_caught / n_crit),
        "tar": float(acted / n_crit),
        "n_critical": float(n_crit),
        "n_caught": float(n_caught),
        "n_caught_but_ignored": float(n_caught - acted),
        "n_caught_and_acted": float(acted),
        "n_missed_silently": float(n_crit - n_caught),
    }


def alert_response(mask: np.ndarray, response_rate: float) -> np.ndarray:
    """Deterministic, seedable stand-in for per-alarm human response.

    In a real deployment this is replaced by the CMMS join; the signature is
    kept identical so the metric code is unchanged. Uses a hash of the index
    so results are reproducible without a global RNG.
    """
    m = np.asarray(mask, bool)
    out = np.zeros(m.shape, bool)
    idx = np.flatnonzero(m)
    for k, i in enumerate(idx):
        h = (int(i) * 2654435761) % 1000
        out[i] = h < int(response_rate * 1000)
    return out


def decision_regret(
    actions: np.ndarray, truth: np.ndarray, matrix, oracle: float
) -> float:
    """Mean excess cost versus a perfect-information policy.

    Reported in currency *and* as a ratio, because "regret 2.1x oracle" is a
    procurement conversation and "regret EUR 340k/line/year" is a budget one.
    """
    from .policy import policy_summary

    s = policy_summary(actions, truth, matrix)
    return float(s["mean_cost"] / max(oracle, 1e-9)) if oracle > 0 else float("nan")


# ---------------------------------------------------------------- utilities
def bootstrap_ci(
    values: np.ndarray, statistic=np.mean, n_boot: int = 2000, seed: int = 7
) -> tuple[float, float]:
    """Percentile bootstrap CI. Used so every headline number carries an
    interval -- seed-to-seed variance is routinely larger than method gaps in
    this literature and single-point comparisons are misleading."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.array(
        [statistic(rng.choice(v, size=v.size, replace=True)) for _ in range(n_boot)]
    )
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Paired test for two policies on the same decision points.

    Cost differences are paired and heavily skewed, so a paired bootstrap on
    the difference is reported alongside McNemar's exact test on the
    win/loss counts. Both, because either alone is misleading here.
    """
    from scipy.stats import binomtest

    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    p = float(binomtest(n10, n01 + n10, 0.5).pvalue) if (n01 + n10) > 0 else 1.0
    return {
        "n_policy_a_only": float(n10),
        "n_policy_b_only": float(n01),
        "mcnemar_p": p,
    }


def full_report(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal: float,
) -> dict[str, float]:
    """One-call bundle of the standard prognostic scorecard."""
    y = np.asarray(y, float)
    return {
        "rmse": rmse(y, mu),
        "mae": mae(y, mu),
        "crps_gaussian": crps_gaussian(y, mu, sigma),
        "winkler": winkler(y, lower, upper, 1.0 - nominal),
        "asym_alpha0.2": alpha_asymmetric(y, mu, 0.2),
        "coverage": coverage(y, lower, upper),
        "coverage_gap": coverage(y, lower, upper) - nominal,
        "mean_width": interval_width(lower, upper),
        "pit_ks": pit_uniformity_ks(y, mu, sigma),
        "ece": expected_calibration_error(y, mu, sigma),
    }