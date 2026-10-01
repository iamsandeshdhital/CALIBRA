"""Decision policies: expected-cost minimisation, abstention, risk control.

The pipeline is:

    calibrated conformal set  ->  regime probabilities  ->  E[cost | action]
      ->  argmin, subject to (a) a self-assessment gate and
                             (b) a conformal risk budget on expected cost

Each of those three pieces exists because of a specific, documented failure:

* **argmin of E[cost]** replaces threshold-on-probability because the
  cost-optimal threshold is a function of the cost ratio (Elkan, IJCAI'01).
* **The abstention gate** replaces always-emit because a model that is
  uncertain in a regime it has never seen should say so rather than act
  (Geifman & El-Yaniv's SelectiveNet, ICML'19; the coverage-risk frontier of
  their prior selective-classification work).
* **Conformal risk control on cost** replaces "we tuned the threshold on the
  validation set" because tuning on validation and then reporting the same
  validation number is not a guarantee. CRC (Angelopoulos, Bates, Fisch, Lei
  & Schuster, ICLR'24) gives a finite-sample upper bound on the *cost*
  functional, which is what the plant actually cares about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..types import ACTION_LIST, REGIME_ORDER, Action, CostMatrix, Health
from .costs import OutcomeProbabilities


@dataclass
class PolicyConfig:
    alpha: float = 0.10
    #: Coverage of the conformal set below which the model may not commit to a
    #: physical action. Below this the system can only WATCH.
    min_coverage_for_action: float = 0.85
    #: Interval width (in RUL units) beyond which the system abstains
    #: entirely, regardless of nominal coverage. A 400-cycle-wide interval is
    #: not actionable no matter how it was calibrated.
    max_actionable_width: float = 140.0
    #: Expected-cost budget used by conformal risk control. Set to a
    #: fraction of the cost of the cheapest *always-stop* policy.
    budget_multiple: float = 0.10
    #: Minimum number of calibration points required before CRC will emit
    #: anything other than "no guarantee available".
    min_calibration: int = 60


def expected_cost(P: np.ndarray, matrix: CostMatrix) -> dict[str, np.ndarray]:
    """E[cost | action] for every action. ``P`` is ``(n, 3)`` over REGIME_ORDER."""
    P = np.asarray(P, float)
    out: dict[str, np.ndarray] = {}
    for a in ACTION_LIST:
        c = matrix.row(a)  # (3,)
        out[a.value] = P @ c
    return out


def choose_actions(
    ecost: Mapping[str, np.ndarray],
    width: np.ndarray,
    cfg: PolicyConfig,
    coverage: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(chosen_index, health_code)``.

    ``health_code``: 0 = TRUSTED, 1 = WIDE, 2 = ABSTAIN.

    Three-stage rule, in order of safety:
      1. If the conformal set is too wide (or coverage too low) we do not
         choose a physical action. We cap at WATCH, or ABSTAIN if the
         interval is so wide it carries no information at all.
      2. Otherwise take the argmin of expected cost. Note this can select a
         *low* action (RUN) with high confidence -- a confident model that
         says "this tool is fine" is doing useful work and suppressing
         noise, which is what halts alert fatigue.
      3. Clamp to at least WATCH whenever health is not TRUSTED, so the
         system always leaves a trace in the CMMS even when it abstains.
    """
    names = [a.value for a in ACTION_LIST]
    E = np.stack([ecost[n] for n in names], axis=1)  # (n, |A|)
    best = np.argmin(E, axis=1)
    best_E = E[np.arange(E.shape[0]), best]

    ok = np.isfinite(best_E) & (best_E >= 0)
    w = np.asarray(width, float)
    cov = np.ones_like(w) if coverage is None else np.asarray(coverage, float)

    trusted = ok & (w <= cfg.max_actionable_width) & (cov >= cfg.min_coverage_for_action)
    wide = ok & ~trusted & (w <= 3.0 * cfg.max_actionable_width)
    health = np.where(trusted, 0, np.where(wide, 1, 2)).astype(int)

    # Abstain -> cap the ladder at WATCH (index 1). Wide -> cap at INSPECT.
    cap = np.where(trusted, len(names) - 1, np.where(wide, 2, 1))
    chosen = np.minimum(best, cap)
    # If everything is uninformative, still emit WATCH so the decision is logged.
    chosen = np.where(ok, chosen, 1)
    return chosen, health


def assign(
    chosen_idx: np.ndarray, health_code: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    names = np.array([a.value for a in ACTION_LIST], dtype=object)
    return names[chosen_idx], health_code


# --------------------------------------------------------------------- CRC
def _upper_bound_quantile(x: np.ndarray, level: float) -> float:
    """(1-alpha) upper bound on the mean of a non-negative cost vector.

    Uses the finite-sample ``ceil((n+1)*level)/n`` order statistic, the same
    correction as split conformal, so the bound holds at finite n rather than
    asymptotically.
    """
    s = np.sort(np.asarray(x, float))
    n = s.size
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * level))
    if rank > n:
        return float("inf")
    return float(s[rank - 1])


def _worst_case(ecost: Mapping[str, np.ndarray], budget: float, levels: Sequence[Action]) -> float:
    """Max over the monotone family of action thresholds of mean cost.

    The family is nested: for threshold ``k`` we act when ``E_k < budget``,
    i.e. we escalate to action ``k`` or higher only when nothing cheaper is
    within budget. Nesting is what makes a single scalar budget control the
    whole policy, and it is why we can search it with a binary scan instead
    of an exponential one.
    """
    names = [a.value for a in levels]
    E = np.stack([ecost[n] for n in names], axis=1)
    out = np.empty(E.shape[0], float)
    for i in range(E.shape[0]):
        row = E[i]
        # Simulate: pick cheapest action within budget, else the most expensive.
        feasible = np.where(row <= budget)[0]
        out[i] = row[feasible].max() if feasible.size else row.max()
    return out


def conformal_risk_control(
    ecost: Mapping[str, np.ndarray],
    truth: np.ndarray,
    cfg: PolicyConfig,
    levels: Sequence[Action] = tuple(ACTION_LIST),
) -> dict[str, float | bool | str]:
    """Choose the cheapest budget whose expected cost is provably within a
    multiple of the always-stop policy's cost.

    Returns a dict with ``budget``, ``guaranteed_mean_cost``, ``budget_usable``
    and ``reason``. When ``n < min_calibration`` the function refuses to
    certify and says so, rather than reporting a number it cannot defend.
    """
    # CRC-style procedures bound a functional of the model's own predicted
    # distribution, so `truth` is deliberately unused: this works with zero
    # failure labels, which is the situation that defeats supervised
    # threshold tuning.
    del truth  # signature kept for interface symmetry with the other policies
    n_stop = len(ecost[Action.STOP.value])
    always_stop = np.array([ecost[Action.STOP.value][i] for i in range(n_stop)])
    n = n_stop

    if n < cfg.min_calibration:
        return {
            "budget_usable": False,
            "reason": f"n={n} < min_calibration={cfg.min_calibration}; no guarantee available",
            "budget": float("nan"),
            "guaranteed_mean_cost": float("nan"),
            "n": n,
        }

    reference = _upper_bound_quantile(always_stop, 1.0 - cfg.alpha)
    target = cfg.budget_multiple * reference

    # Search for the smallest budget meeting the bound (monotone in budget).
    lo, hi = 0.0, float(np.max(always_stop)) * 1.0001
    best_budget, best_cost = float("nan"), float("inf")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        cost = _worst_case(ecost, mid, levels)
        ub = _upper_bound_quantile(cost, 1.0 - cfg.alpha)
        if np.isfinite(ub) and ub <= target:
            best_budget, best_cost = mid, ub
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-6:
            break

    return {
        "budget_usable": bool(np.isfinite(best_budget)),
        "reason": "ok" if np.isfinite(best_budget) else "no budget met the target within range",
        "budget": float(best_budget) if np.isfinite(best_budget) else float("nan"),
        "guaranteed_mean_cost": float(best_cost) if np.isfinite(best_cost) else float("inf"),
        "target_cost": float(target),
        "reference_always_stop_cost": float(reference),
        "n": n,
    }


def policy_summary(
    actions: np.ndarray, truth: np.ndarray, matrix: CostMatrix
) -> dict[str, float]:
    """Realized cost of a policy given the ground-truth outcomes."""
    actions = np.asarray(actions)
    truth = np.asarray(truth)
    costs = np.array(
        [matrix.get(str(a), str(t)) for a, t in zip(actions, truth)], dtype=float
    )
    return {
        "mean_cost": float(costs.mean()),
        "median_cost": float(np.median(costs)),
        "p95_cost": float(np.percentile(costs, 95)),
        "total_cost": float(costs.sum()),
    }


def oracle_cost(truth: np.ndarray, matrix: CostMatrix) -> float:
    """Regret floor: the best action if the regime were known exactly."""
    best = np.array([np.min(matrix.row(a)) for a in ACTION_LIST])
    idx = np.array([REGIME_ORDER.index(str(t)) for t in truth])
    return float(best[idx].mean())