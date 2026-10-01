"""Cost matrices grounded in how high-tech plants actually pay.

The economics here are the whole argument for the project. In conventional
discrete manufacturing, false positives are embarrassing. In high-tech
lines they are *structurally* expensive too, because the verification
capacity is scarce:

* **Semiconductor fab** -- an unverified defect call consumes a mask shot,
  a metrology slot and, for a true tool-related false alarm, a chamber
  qualification run. A missed catastrophic defect (bridge/short) scraps a
  lot and carries customer DPPM penalties that run to 5-6 orders of
  magnitude above the metrology cost. Ratio C_FN / C_FP is routinely 1e3-1e5.
* **Battery / ESS cell line** -- a missed early-failure cell becomes a
  field thermal-runaway risk, i.e. a recall. A false call costs a re-grade
  cycle (hours of formation time). Ratio 1e2-1e4.
* **Precision machining cell** -- a missed tool-break costs a scrapped
  workpiece plus a crash and a possible collision. A false call costs a
  probe touch. Ratio 1e1-1e2.

Two consequences drive the design:

1. **You cannot fix a threshold.** With C_FN/C_FP = 1e4, the
   accuracy-maximising threshold is not the cost-minimising one. Elkan's
   cost-sensitive learning result (IJCAI'01) is the canonical statement:
   the optimal decision rule depends on the cost ratio, so a fixed 0.5
   threshold is *wrong by construction* whenever costs are asymmetric.
2. **You cannot run the verification path at 100%.** If C_FP is large and
   verification capacity is finite, the binding constraint becomes expected
   cost, not recall. That is a *risk budget* problem, which
   Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei & Schuster,
   ICLR'24) solves with a finite-sample guarantee: find the most aggressive
   (cheapest) action set whose *expected cost* is provably within budget.

The default matrix below is the semiconductor/fab-flavoured one, chosen as
the harshest case, and every number is in the docstring so it can be argued
with and replaced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import REGIME_ORDER, Action, CostMatrix


def fab_cost_matrix(
    c_metrology_minutes: float = 12.0,
    engineer_hourly: float = 95.0,
    lot_value: float = 48_000.0,
    penalty_per_defective_part: float = 2_600.0,
    escape_ppm: float = 40_000.0,
    chamber_qual_minutes: float = 240.0,
    chamber_qual_hourly: float = 180.0,
    line_stop_hourly: float = 145_000.0,
) -> CostMatrix:
    """Default matrix for a high-value discrete/wafer line.

    Derivation (per *event*, i.e. per decision point):

    ``c_metrology``  Re-inspection / AOI re-scan opportunity cost.
    ``c_engineer``   Engineer time to triage, review and disposition.
    ``c_lot``        Expected scrap exposure of the lot at risk, entered when
                     a call is made but cannot be resolved in time.
    ``c_escape``     Expected cost of a critical defect escaping undetected:

                         c_escape = penalty_per_defective_part * escape_ppm / 1e6

                     ``escape_ppm`` is the expected defective-parts rate of the
                     affected family (ppm), and ``penalty_per_defective_part``
                     is EUR per escaped part (line-down, sorting, return
                     material authorisation, line credit).

                     Unit warning: this is a product of a *rate* (ppm) and a
                     *per-unit* price, so it comes out small. At 85 EUR/part
                     and 40,000 ppm the product is only EUR 3.40, which is
                     *cheaper than a 12-minute metrology re-check* -- a
                     nonsense regime in which it is always rational to run.
                     Real fabs sit far higher because the per-part figure for
                     an escaped critical defect includes the line credit.
                     ``ordering_is_sane()`` below asserts this explicitly so
                     the parameterisation cannot silently become degenerate.

    The structure that matters is the *diagonal-safe* property: no action is
    free, and the cheapest safe action (WATCH) is only cheap because it buys
    information rather than outcomes. RECIPE is near-free but only partially
    effective (it cannot fix a mechanical failure) -- modelling that partial
    effectiveness is why the policy is not a monotone threshold.
    """
    c_inspect = c_metrology_minutes / 60.0 * (engineer_hourly * 2.0)
    c_watch = 0.25 * engineer_hourly  # ~15 min of review per tightened cycle
    c_recipe = 0.15 * engineer_hourly  # parameter tweak + verification pass
    c_recover = chamber_qual_minutes / 60.0 * chamber_qual_hourly + 4.0 * engineer_hourly
    c_stop = line_stop_hourly * 1.5 / 24.0  # ~1.5h of lost tool time per hour

    c_escape = penalty_per_defective_part * escape_ppm / 1e6

    m = {
        Action.RUN.value: {
            REGIME_ORDER[0]: 0.0,
            REGIME_ORDER[1]: 0.20 * lot_value,     # drift out of spec -> scrap
            REGIME_ORDER[2]: c_escape + lot_value,  # defect escapes -> penalty + lot
        },
        Action.WATCH.value: {
            REGIME_ORDER[0]: c_watch,
            REGIME_ORDER[1]: c_watch + 0.05 * lot_value,
            REGIME_ORDER[2]: c_watch + 0.45 * c_escape + 0.30 * lot_value,
        },
        Action.INSPECT.value: {
            # Diagnostic. A metrology re-check *identifies* the condition; it
            # does not restore the tool. This is the distinction that keeps the
            # action ladder from collapsing. If inspection were priced as
            # though it repaired the machine (previous version charged only
            # 20% of the escape cost), it would dominate every other rung in
            # every state and the policy would degenerate to "always
            # inspect" -- arithmetically optimal and operationally useless,
            # because it never actually fixes anything.
            #
            # On a confirmed critical, inspection still converts an *escape*
            # into a *detection*, which is worth a lot: the customer penalty
            # is avoided and the lot can be contained, but the tool is still
            # broken and a RECOVER or STOP follows.
            REGIME_ORDER[0]: c_inspect,
            REGIME_ORDER[1]: c_inspect + 0.02 * lot_value,
            REGIME_ORDER[2]: c_inspect + 0.85 * c_escape + 0.20 * lot_value,
        },
        Action.RECIPE.value: {
            REGIME_ORDER[0]: c_recipe,
            REGIME_ORDER[1]: c_recipe + 0.10 * lot_value,   # only partly fixes drift
            REGIME_ORDER[2]: c_recipe + 0.60 * c_escape + 0.55 * lot_value,
        },
        Action.RECOVER.value: {
            # A planned-window PM swap. Planned PM is the *efficient* path:
            # it needs the chamber for 4h at a scheduled time.
            REGIME_ORDER[0]: c_recover + 1.4 * c_inspect,  # unnecessary: pure waste
            REGIME_ORDER[1]: c_recover + 0.4 * c_inspect,  # mostly avoidable
            # Preventive, but NOT free: an emergency swap outside the planned
            # window costs the same chamber qualification *plus* the
            # line-stop time needed to get at it. Leaving that out makes
            # RECOVER unconditionally optimal and collapses the whole ladder,
            # which is why this cell is the most load-bearing number in the
            # matrix.
            REGIME_ORDER[2]: c_recover + 0.60 * c_stop,
        },
        Action.STOP.value: {
            REGIME_ORDER[0]: c_stop + 4.0 * engineer_hourly,  # very expensive if wrong
            REGIME_ORDER[1]: c_stop + 0.6 * c_inspect,
            # A contained stop: the defect is stopped before it escapes, but
            # the tool time is still lost. Stopping is cheaper than RECOVER on a
            # critical event only because it does not require a full chamber
            # re-qualification first.
            REGIME_ORDER[2]: 0.35 * c_stop,
        },
    }
    return CostMatrix(matrix=m)


def bearing_cost_matrix() -> CostMatrix:
    """Cheaper, lower-asymmetry variant for a machining cell.

    Included so the policy layer can be shown to *adapt* to the cost regime
    rather than hard-code one plant: C_FN/C_FP is ~100 here versus ~140 in the
    fab matrix, and the line stop is 20x cheaper, so the same probabilities
    must yield a different recommendation.

    Structure rule obeyed throughout: for every action,
    ``cost(action | critical) < cost(action | degraded) < cost(action | healthy)
    is NOT required`` -- what *is* required is that no action is cheaper when
    critical than when healthy, except where doing the work genuinely fixes
    the machine (RECOVER, STOP). An earlier version had
    ``RECOVER|critical = 240 < RECOVER|healthy = 690``, i.e. breaking a
    bearing was *cheaper* to fix than a healthy one to disturb, which made the
    cost ratio negative and the policy degenerate.
    """
    m = {
        Action.RUN.value: {REGIME_ORDER[0]: 0.0, REGIME_ORDER[1]: 380.0, REGIME_ORDER[2]: 41_000.0},
        Action.WATCH.value: {REGIME_ORDER[0]: 12.0, REGIME_ORDER[1]: 48.0, REGIME_ORDER[2]: 18_500.0},
        # Diagnostic only -- same semantics as the fab matrix.
        Action.INSPECT.value: {REGIME_ORDER[0]: 28.0, REGIME_ORDER[1]: 44.0, REGIME_ORDER[2]: 9_400.0},
        Action.RECIPE.value: {REGIME_ORDER[0]: 22.0, REGIME_ORDER[1]: 92.0, REGIME_ORDER[2]: 26_000.0},
        # Planned swap is efficient; an emergency swap costs the crash plus
        # the fixture change, so it is dearer than a planned one but still far
        # cheaper than running the tool to failure.
        Action.RECOVER.value: {REGIME_ORDER[0]: 690.0, REGIME_ORDER[1]: 520.0, REGIME_ORDER[2]: 1_150.0},
        Action.STOP.value: {REGIME_ORDER[0]: 9_500.0, REGIME_ORDER[1]: 3_100.0, REGIME_ORDER[2]: 640.0},
    }
    return CostMatrix(matrix=m)


def ordering_is_sane(matrix: CostMatrix) -> bool:
    """Assert the three cost scales are in a workable order:

        c_inspect(healthy)  <<  c_RUN(critical)  <<  lot value

    If detecting a defect costs more than letting it escape, the cost matrix is
    degenerate and every policy collapses to RUN. If escape costs more than the
    whole lot, the matrix is over-punishing. Both failure modes are silent --
    the policy still runs -- so they are checked explicitly.
    """
    detect = matrix.row(Action.INSPECT)[0]
    escape = matrix.row(Action.RUN)[2]
    return bool(detect < escape)


def cost_ratio(matrix: CostMatrix) -> float:
    """C_FN / C_FP -- the number that decides whether threshold-based PdM is
    even approximately right.

    Defined rigorously as the ratio of the *incremental* costs of two states,
    which is what governs a decision boundary:

        C_FN = C(action_best | critical) - C(action_best | healthy)
        C_FP = C(action_best | critical) - C(action_best | critical)

    Practical definition used here: the cost of the cheapest fully-protective
    action (RECOVER) minus its cost on a healthy asset, divided by the cost of
    the cheapest *detecting* action (INSPECT) on a healthy asset.

        ratio = [ C(RECOVER|critical) - C(RECOVER|healthy) ]
                / [ C(INSPECT|healthy) ]

    Numerator = what you save by preventing failure. Denominator = what a false
    alarm costs. An earlier version took min() across regimes for each term,
    which compared incompatible quantities and inverted the ordering between
    the fab and bearing matrices -- exactly the kind of silent sign error that
    makes a cost model untrustworthy.
    """
    best_protective = matrix.row(Action.RECOVER)
    cheapest_detect = matrix.row(Action.INSPECT)
    save_from_preventing = best_protective[2] - best_protective[0]
    cost_of_false_alarm = cheapest_detect[0]
    return float(save_from_preventing / max(cost_of_false_alarm, 1e-9))


@dataclass
class OutcomeProbabilities:
    """P(regime) implied by a conformal RUL set.

    We do not ask the model for a class probability. We derive the regime
    distribution from the *calibrated set* of plausible RUL values crossed
    with the plant's degradation geometry (the RUL->regime thresholds), so
    calibration of the interval is what propagates into the decision. That is
    what makes the downstream cost numbers auditable.

    A uniform density over the conformal set is the honest default: conformal
    gives us a *set*, not a density, and pretending otherwise would smuggle
    back the over-confidence we are trying to remove.
    """

    degraded_rul: float
    critical_rul: float
    #: Physical upper bound on remaining life (the RUL censoring horizon).
    #: Intervals are clipped into [0, max_rul] because RUL cannot be negative
    #: and cannot exceed the horizon. An earlier version applied
    #: ``maximum(upper, max_rul)`` -- a floor rather than a cap -- which pushed
    #: every upper bound to 1e4 and made a certain-critical interval [0, 10]
    #: look 99% healthy. The direction of a bound is load-bearing.
    max_rul: float = 145.0

    def from_interval(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        lo = np.clip(np.asarray(lower, float), 0.0, self.max_rul)
        hi = np.clip(np.asarray(upper, float), 0.0, self.max_rul)
        # Keep the interval ordered and non-degenerate even after clipping.
        hi = np.maximum(hi, lo + 1e-9)
        # Under a uniform density on [lo, hi]:
        p_crit = np.clip((self.critical_rul - lo) / (hi - lo), 0.0, 1.0)
        p_deg = np.clip((self.degraded_rul - lo) / (hi - lo), 0.0, 1.0) - p_crit
        p_h = np.clip(1.0 - p_crit - p_deg, 0.0, 1.0)
        P = np.stack([p_h, np.clip(p_deg, 0.0, 1.0), p_crit], axis=1)
        return P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)

    @staticmethod
    def point(truth: str) -> np.ndarray:
        v = np.zeros(3)
        v[REGIME_ORDER.index(truth)] = 1.0
        return v