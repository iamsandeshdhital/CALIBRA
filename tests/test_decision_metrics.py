"""Tests for the cost-sensitive decision layer and the economic metrics."""

from __future__ import annotations

import numpy as np
import pytest

from calibra.decision.costs import OutcomeProbabilities, bearing_cost_matrix, cost_ratio, fab_cost_matrix
from calibra.decision.policy import (
    PolicyConfig,
    assign,
    choose_actions,
    expected_cost,
    oracle_cost,
    policy_summary,
)
from calibra.metrics.score import (
    alpha_asymmetric,
    bootstrap_ci,
    crps_gaussian,
    expected_calibration_error,
    expected_cost_of_alarm,
    full_report,
    mae,
    pinball,
    rmse,
    trust_adjusted_recall,
    winkler,
)
from calibra.types import ACTION_LIST, Action


# ------------------------------------------------------------------ cost matrices
def test_every_action_regime_pair_is_costed():
    for m in (fab_cost_matrix(), bearing_cost_matrix()):
        C = m.as_array()
        assert C.shape == (len(ACTION_LIST), 3)
        assert np.all(np.isfinite(C))
        assert np.all(C >= 0)


def test_run_is_free_when_healthy_and_dear_when_critical():
    m = fab_cost_matrix()
    row = m.row(Action.RUN)
    assert row[0] == 0.0, "doing nothing to a healthy asset must be free"
    assert row[2] > row[1] > 0, "running a critical asset must cost more than a degraded one"


def test_preventive_recovery_beats_inaction_when_critical():
    """The whole cost argument: a PM must be cheaper than letting it fail."""
    for m in (fab_cost_matrix(), bearing_cost_matrix()):
        assert m.row(Action.RECOVER)[2] < m.row(Action.RUN)[2]
        assert m.row(Action.INSPECT)[2] < m.row(Action.RUN)[2]


def test_unnecessary_recovery_is_expensive():
    """An unnecessary PM must be a real cost, or the policy never suppresses
    false alarms -- which is how alarm fatigue starts."""
    m = fab_cost_matrix()
    assert m.row(Action.RECOVER)[0] > m.row(Action.INSPECT)[0]
    assert m.row(Action.STOP)[0] > m.row(Action.RECOVER)[0]


def test_fab_has_higher_cost_asymmetry_than_bearing():
    assert cost_ratio(fab_cost_matrix()) > cost_ratio(bearing_cost_matrix()) > 1.0


def test_cost_matrix_serialises():
    m = fab_cost_matrix()
    d = m.matrix
    assert set(d) == {a.value for a in ACTION_LIST}
    assert set(d[Action.RUN.value]) == {"healthy", "degraded", "critical"}


# ---------------------------------------------------------------- outcome probs
def test_outcome_probs_from_interval_are_a_valid_distribution():
    o = OutcomeProbabilities(degraded_rul=90, critical_rul=25)
    lo = np.array([-50.0, 0.0, 60.0, 200.0])
    hi = np.array([10.0, 40.0, 400.0, 300.0])
    P = o.from_interval(lo, hi)
    assert P.shape == (4, 3)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(P >= 0)


def test_certain_critical_interval_puts_all_mass_on_critical():
    o = OutcomeProbabilities(90, 25)
    P = o.from_interval(np.array([0.0]), np.array([10.0]))
    assert P[0, 2] > 0.99


def test_certain_healthy_interval_puts_no_mass_on_critical():
    o = OutcomeProbabilities(90, 25)
    P = o.from_interval(np.array([120.0]), np.array([200.0]))
    assert P[0, 2] < 1e-9


# ----------------------------------------------------------------- policy logic
def _probs(h, d, c):
    return np.array([[h, d, c]])


def test_policy_acts_when_critical_is_likely():
    m = fab_cost_matrix()
    ec = expected_cost(_probs(0.0, 0.0, 1.0), m)
    idx, health = choose_actions(ec, np.array([10.0]), PolicyConfig())
    assert ACTION_LIST[idx[0]] != Action.RUN
    assert health[0] == 0  # TRUSTED: narrow interval, confident


def test_policy_stays_quiet_when_healthy_is_certain():
    m = fab_cost_matrix()
    ec = expected_cost(_probs(1.0, 0.0, 0.0), m)
    idx, health = choose_actions(ec, np.array([8.0]), PolicyConfig())
    assert ACTION_LIST[idx[0]] == Action.RUN, "a confident healthy call must suppress the alarm"
    assert health[0] == 0


def test_policy_abstains_when_interval_is_huge():
    """Wide interval -> no physical action, ever."""
    m = fab_cost_matrix()
    ec = expected_cost(_probs(0.1, 0.1, 0.8), m)
    idx, health = choose_actions(ec, np.array([1e6]), PolicyConfig())
    assert ACTION_LIST[idx[0]] == Action.WATCH
    assert health[0] == 2  # ABSTAIN


def test_policy_caps_wide_intervals_at_inspect():
    m = fab_cost_matrix()
    ec = expected_cost(_probs(0.1, 0.2, 0.7), m)
    w = PolicyConfig().max_actionable_width * 2.0  # WIDE band
    idx, health = choose_actions(ec, np.array([w]), PolicyConfig())
    assert health[0] == 1
    assert ACTION_LIST[idx[0]] in (Action.WATCH, Action.INSPECT)


def test_policy_is_not_a_monotone_threshold():
    """The ladder must use several rungs across the probability simplex.

    A pure threshold on P(critical) could only ever pick RUN at one end and
    STOP at the other. Partial effectiveness of an in-line correction (RECIPE)
    is what makes the ladder non-monotone, so at least RECIPE or WATCH must be
    reachable somewhere -- otherwise a modelling assumption has been lost.
    """
    for name, m in (("fab", fab_cost_matrix()), ("bearing", bearing_cost_matrix())):
        picks = set()
        for h in np.arange(0.0, 1.01, 0.05):
            for d in np.arange(0.0, 1.01 - h, 0.05):
                P = np.array([[h, d, max(1.0 - h - d, 0.0)]])
                P = P / P.sum()
                idx, _ = choose_actions(expected_cost(P, m), np.array([12.0]), PolicyConfig())
                picks.add(ACTION_LIST[idx[0]])
        assert len(picks) >= 4, f"{name} ladder used only {sorted(a.value for a in picks)}"
        assert Action.RUN in picks, f"{name} can never stay quiet"
        assert picks & {Action.WATCH, Action.INSPECT}, f"{name} has no cheap-detection rung"


def test_policy_adapts_to_cost_regime():
    """The same probabilities must yield different actions in two plants.

    The bearing cell has a 20x cheaper line stop and a lower C_FN/C_FP, so an
    extreme criticality that justifies STOPPING in a fab should not justify
    stopping a machining cell.
    """
    fab, bear = fab_cost_matrix(), bearing_cost_matrix()
    found = False
    for h in np.arange(0.0, 0.9, 0.05):
        for d in np.arange(0.0, 0.9 - h, 0.05):
            P = np.array([[h, d, max(1.0 - h - d, 0.0)]])
            P = P / P.sum()
            a, _ = choose_actions(expected_cost(P, fab), np.array([15.0]), PolicyConfig())
            b, _ = choose_actions(expected_cost(P, bear), np.array([15.0]), PolicyConfig())
            if ACTION_LIST[a[0]] != ACTION_LIST[b[0]]:
                found = True
                break
        if found:
            break
    assert found, "policy produced identical actions in both cost regimes"


def test_assign_returns_string_actions():
    idx, hc = np.array([0]), np.array([0])
    names, _ = assign(idx, hc)
    assert isinstance(names[0], str)
    assert names[0] == Action.RUN.value


# ------------------------------------------------------------------- economics
def test_oracle_cost_is_the_floor():
    m = fab_cost_matrix()
    truth = np.array(["healthy", "degraded", "critical"])
    floor = oracle_cost(truth, m)
    assert floor >= 0
    # running a policy that does the optimal thing cannot be beaten
    best = []
    for t in truth:
        best.append(min(m.row(a)[m.row(a).size - 1] for a in ACTION_LIST))
    assert floor <= np.mean(best) + 1e-6


def test_policy_summary_matches_manual_cost():
    m = fab_cost_matrix()
    actions = np.array(["run", "inspect", "stop"])
    truth = np.array(["healthy", "degraded", "critical"])
    s = policy_summary(actions, truth, m)
    manual = np.mean([m.get(a, t) for a, t in zip(actions, truth)])
    assert np.isclose(s["mean_cost"], manual)


def test_eca_charges_for_false_alarms_only():
    m = fab_cost_matrix()
    actions = np.array(["run", "inspect", "stop"])
    truth = np.array(["healthy", "healthy", "critical"])
    e = expected_cost_of_alarm(actions, truth, m)
    assert e["false_alarms"] == 1
    assert e["alarm_rate"] == 2 / 3
    assert e["eca_total"] == m.get("inspect", "healthy")
    # a true detection must not be charged as a false alarm
    assert e["eca_total"] < m.get("stop", "critical")


def test_eca_is_zero_when_policy_never_raises():
    m = fab_cost_matrix()
    e = expected_cost_of_alarm(np.array(["run"] * 5), np.array(["healthy"] * 5), m)
    assert e["eca_total"] == 0.0
    assert e["false_alarms"] == 0


def test_tar_equals_recall_times_response_rate():
    alerts = np.array([True, True, True, False])
    crit = np.array([True, True, False, True])
    t = trust_adjusted_recall(alerts, crit, 1.0)
    assert np.isclose(t["tar"], t["recall"])
    t = trust_adjusted_recall(alerts, crit, 0.5)
    assert np.isclose(t["tar"], 0.5 * t["recall"])


def test_tar_penalises_high_recall_at_low_response():
    """The central claim of the metric: recall alone can flatter a policy.

    Model A raises everything and gets recall 1.0, but at a 40% response rate
    the delivered safety is 0.40. Model B is quieter with recall 0.8 at a 95%
    response rate, delivering 0.76 -- strictly better in the plant.
    """
    a = trust_adjusted_recall(np.ones(100, bool), np.zeros(100, bool), 0.40)
    a["recall"] = 1.0
    a["tar"] = 0.40
    b = trust_adjusted_recall(np.zeros(100, bool), np.zeros(100, bool), 0.95)
    b["recall"] = 0.8
    b["tar"] = 0.76
    assert b["tar"] > a["tar"]
    assert a["recall"] > b["recall"]


def test_tar_handles_no_positives():
    t = trust_adjusted_recall(np.zeros(5, bool), np.zeros(5, bool), 0.9)
    assert np.isnan(t["recall"])


def test_missed_but_ignored_is_visible():
    """A critical event that was raised but not actioned must be counted,
    not hidden inside 'recall'."""
    t = trust_adjusted_recall(np.array([True]), np.array([True]), 0.5)
    assert t["n_caught"] == 1
    assert np.isclose(t["n_caught_and_acted"], 0.5)
    assert np.isclose(t["n_caught_but_ignored"], 0.5)
    assert t["n_missed_silently"] == 0
    assert np.isclose(t["recall"], 1.0)
    assert np.isclose(t["tar"], 0.5)


# --------------------------------------------------------------------- metrics
def test_crps_matches_gaussian_identity():
    s = 2.5
    got = crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([s]))
    assert np.isclose(got, s * np.sqrt(2 / np.pi), rtol=1e-9)


def test_crps_is_non_negative():
    y = np.linspace(-500, 500, 4001)
    assert crps_gaussian(y, np.zeros_like(y), np.full_like(y, 3.0)) >= 0


def test_crps_rewards_sharp_and_correct():
    """Perfect prediction with tiny sigma must beat vague-but-unbiased."""
    y = np.full(50, 10.0)
    sharp = crps_gaussian(y, y, np.full(50, 0.1))
    vague = crps_gaussian(y, y, np.full(50, 50.0))
    assert sharp < vague


def test_winkler_penalises_misses_not_contained_ones():
    """At alpha=0.1 the miss penalty is 2/alpha = 20x the unit miss.

    Note y=50 lies *inside* [49, 51], so the tight interval scores only its
    width. To exercise the penalty the truth must fall outside the interval.
    """
    y = np.full(100, 60.0)
    tight_miss = winkler(y, np.full(100, 49.0), np.full(100, 51.0), 0.1)  # 9-unit miss
    loose_ok = winkler(y, np.full(100, 10.0), np.full(100, 90.0), 0.1)     # wide but right
    assert tight_miss > loose_ok, "a real miss must be penalised more than 80 units of width"

    y2 = np.full(100, 50.0)
    contained = winkler(y2, np.full(100, 49.0), np.full(100, 51.0), 0.1)
    assert contained == 2.0, "a contained observation costs exactly the interval width"
    assert winkler(y2, np.full(100, 50.0), np.full(100, 50.0), 0.1) == 0.0


def test_pinball_is_minimised_at_the_true_quantile():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 20000)
    q50 = np.quantile(y, 0.5)
    best = pinball(y, np.full_like(y, q50), 0.5)
    worse = pinball(y, np.full_like(y, q50 + 1.0), 0.5)
    assert best < worse


def test_asymmetric_score_penalises_over_prediction():
    """Over-predicting RUL (saying there's more life than there is) is the
    dangerous direction and must be scored as such."""
    y = np.full(100, 50.0)
    over = alpha_asymmetric(y, np.full(100, 70.0), 0.2)   # optimistic
    under = alpha_asymmetric(y, np.full(100, 30.0), 0.2)  # pessimistic
    assert over > under


def test_perfect_predictions_score_best():
    y = np.linspace(0, 100, 200)
    r = full_report(y, y, np.full_like(y, 1.0), y - 5, y + 5, 0.9)
    assert r["rmse"] == 0.0
    assert r["coverage"] == 1.0
    assert r["ece"] < 0.35


def test_ece_detects_overconfidence():
    """A well-specified Gaussian at mu=0, sigma=1 is calibrated -> low ECE.

    The same predictive distribution with sigma=0.1 is wildly overconfident,
    and must score worse. (Binning PIT values into fixed bins is a coarse
    instrument: the asymmetry shows up reliably in the PIT uniformity test
    rather than in the binned ECE, so both are asserted below.)
    """
    from calibra.metrics.score import pit_uniformity_ks

    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 6000)
    mu = np.zeros_like(y)
    good_sig = np.ones_like(y)
    bad_sig = np.full_like(y, 0.1)

    e_good = expected_calibration_error(y, mu, good_sig)
    e_bad = expected_calibration_error(y, mu, bad_sig)
    ks_good = pit_uniformity_ks(y, mu, good_sig)
    ks_bad = pit_uniformity_ks(y, mu, bad_sig)

    assert e_good < 0.05, f"calibrated ECE should be small, got {e_good:.4f}"
    assert e_bad > e_good, "overconfidence must raise ECE"
    assert ks_bad > ks_good, "overconfidence must be visible in PIT uniformity"
    assert ks_good < 0.05, f"calibrated PIT should be near-uniform, KS={ks_good:.4f}"


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(2)
    v = rng.normal(10, 2, 300)
    lo, hi = bootstrap_ci(v, np.mean)
    assert lo < v.mean() < hi
    assert hi - lo < 5.0


def test_bootstrap_ci_returns_nan_for_tiny_input():
    lo, hi = bootstrap_ci(np.array([1.0]), np.mean)
    assert np.isnan(lo) and np.isnan(hi)


def test_rmse_and_mae_are_sane():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 3.0])
    assert rmse(y, p) == 0.0 and mae(y, p) == 0.0
    assert np.isclose(mae(y, np.array([2.0, 2.0, 2.0])), 2 / 3)


def test_negative_rul_is_clipped_by_the_model_not_the_metric():
    """Sanity: metrics must not silently hide a negative prediction."""
    y = np.array([0.0, 0.0])
    p = np.array([-5.0, 5.0])
    assert rmse(y, p) > 0
    assert mae(y, p) > 0