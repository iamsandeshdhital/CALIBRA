"""Evidence cards: why an alarm fired, and what it will cost you.

Design rationale (this is the part most PdM deployments omit, and it is the
part that determines whether the output is used):

An operator who receives "RUL = 41" and nothing else has no way to judge it,
so they either over-trust it (and get burned once) or under-trust it (and
ignore it always). Both outcomes are worse than having no model. What makes
a model actionable is a *triangulated card*:

  1. **A calibrated interval, not a point.** "41 ± 60 cycles" tells the
     planner whether the maintenance window even fits before the next
     scheduled stop. A point estimate cannot.
  2. **Local attribution.** Which channels drove *this* call.
  3. **A historical analog.** The nearest past trajectory from the same asset
     class, and what actually happened to it. This converts an abstract
     probability into a precedent the engineer has already seen.
  4. **The cost table.** What each rung of the ladder costs in this instance,
     so the operator can disagree with the recommendation on grounds the
     system has made explicit.
  5. **Self-assessment.** Is the model calibrated and in-control *right now*?
     A drifted model should not be presenting a confident number.

The whole card is a single dict so it can be rendered in a UI, logged to a
CMMS, or pushed to a chat/alerting integration without transformation.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np

from ..types import REGIME_ORDER, Action, CostMatrix, Evidence, Health, PredictiveDistribution


def nearest_analog(
    query: np.ndarray, library: np.ndarray, library_rul: np.ndarray, asset_class: str | None = None
) -> dict[str, float | str | None]:
    """Find the most similar past trajectory and report what became of it.

    Similarity is standardised Euclidean distance over the current feature
    vector, which is crude but auditable and cheap -- and for an evidence card
    shown to a human, a simple explainable distance beats an opaque embedding.
    """
    if library.shape[0] == 0:
        return {"analog_distance": float("inf"), "analog_rul": None, "analog_class": asset_class}
    q = np.asarray(query, float).ravel()
    L = np.asarray(library, float)
    mu = L.mean(axis=0)
    sd = np.where(L.std(axis=0) > 1e-9, L.std(axis=0), 1.0)
    d = np.sqrt((((L - mu) / sd) ** 2 - ((q - mu) / sd) ** 2).sum(axis=1))
    i = int(np.argmin(d))
    return {
        "analog_distance": float(d[i]),
        "analog_rul": float(np.asarray(library_rul, float)[i]),
        "analog_class": asset_class,
    }


def build_card(
    i: int,
    dist: PredictiveDistribution,
    action: Action,
    health: Health,
    attribution: Sequence[tuple[str, float]],
    probs: np.ndarray,
    matrix: CostMatrix,
    ecost: dict[str, float],
    n_calibration: int,
    drift_score: float,
    coverage_recent: float | None,
    analog: dict[str, float | str | None] | None = None,
) -> Evidence:
    """Assemble one evidence card.

    Accepts the action as either an :class:`Action` or its string value. The
    online path resolves actions to plain strings (that is what gets written to
    a CMMS), so the card builder must not assume the enum -- and normalising
    here rather than at each call site means a caller cannot produce a card
    that crashes only on the first alarm.
    """
    action = action if isinstance(action, Action) else Action(str(action))
    health = health if isinstance(health, Health) else Health(str(health))

    # Costs may arrive as scalars or as length-1 arrays (the batch pipeline
    # passes per-point arrays). Collapse anything array-like to a float so the
    # card formats identically in both the batch and the online path.
    flat_cost: dict[str, float] = {}
    for k, v in dict(ecost).items():
        arr = np.asarray(v, float).reshape(-1)
        flat_cost[k] = float(arr[0]) if arr.size else float("nan")
    ranked = sorted(REGIME_ORDER, key=lambda r: -probs[REGIME_ORDER.index(r)])
    margin = float(probs[REGIME_ORDER.index(ranked[0])] - probs[REGIME_ORDER.index(ranked[1])])

    lines = [
        f"RUL {dist.mean:.1f} cycles (90% CI {dist.lower:.0f} to {dist.upper:.0f}, "
        f"width {dist.width:.0f}).",
        f"Most likely state: {ranked[0]} (p={probs[REGIME_ORDER.index(ranked[0])]:.2f}, "
        f"margin {margin:.2f} over {ranked[1]}).",
    ]
    if attribution:
        top = ", ".join(f"{n} ({v:.1f})" for n, v in attribution[:3])
        lines.append(f"Driven by: {top}.")
    if analog and analog.get("analog_rul") is not None:
        lines.append(
            f"Closest historical trajectory had RUL {float(analog['analog_rul']):.0f} "
            f"(distance {float(analog['analog_distance']):.2f})."
        )
    lines.append(
        f"Recommendation: {action.value.upper()} "
        f"(expected cost {flat_cost.get(action.value, float('nan')):,.0f})."
    )
    if health is Health.ABSTAIN:
        lines.append(
            "Model self-assessment: ABSTAIN - interval too wide to commit to a physical "
            "action. Escalate to engineering review; do not close as false alarm."
        )
    elif health is Health.WIDE:
        lines.append(
            "Model self-assessment: WIDE - usable for tighter sampling only; a physical "
            "action is not supported by the current evidence."
        )
    if coverage_recent is not None:
        lines.append(f"Recent empirical coverage: {coverage_recent:.3f} (nominal {dist.level:.2f}).")
    if drift_score > 0.10:
        lines.append(
            f"Drift warning: PSI {drift_score:.3f}. Treat the interval as indicative only."
        )
    lines.append(f"Calibration set: {n_calibration} points, conformal level {dist.level:.2f}.")

    return Evidence(
        top_features=tuple(attribution[:4]),
        n_calibration_points=int(n_calibration),
        calibration_level=float(dist.level),
        drift_score=float(drift_score),
        coverage_recent=None if coverage_recent is None else float(coverage_recent),
        expected_cost=flat_cost,
        chosen_action=action,
        health=health,
        rationale=" ".join(lines),
    )


def card_to_dict(card: Evidence) -> dict:
    d = asdict(card)
    d["chosen_action"] = card.chosen_action.value
    d["health"] = card.health.value
    d["top_features"] = [[n, float(v)] for n, v in card.top_features]
    d["expected_cost"] = {k: float(v) for k, v in card.expected_cost.items()}
    return d


def batch_cards(cards: Sequence[Evidence]) -> list[dict]:
    return [card_to_dict(c) for c in cards]