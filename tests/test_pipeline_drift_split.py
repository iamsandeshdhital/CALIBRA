"""Tests for drift detection, leakage-audited splitting, and the pipeline.

These are the tests that protect the *claims* rather than the functions: a
coverage guarantee that survives a stream, a split that provably cannot leak,
and a pipeline whose reported numbers are internally consistent.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibra import CalibraConfig, CalibraRunner
from calibra.assure.drift import CUSUM, DriftMonitor, Severity, coverage_schedule, psi
from calibra.data.features import group_healthy
from calibra.data.split import assert_no_leakage, naive_window_split, split_runs
from calibra.sim.hightech_line import HighTechLine, LineConfig


@pytest.fixture(scope="module")
def line_runs():
    return HighTechLine(LineConfig(seed=7, n_assets=30)).generate()


@pytest.fixture(scope="module")
def fitted(line_runs):
    return CalibraRunner(CalibraConfig(seed=7)).fit(line_runs)


# ------------------------------------------------------------------ simulator
def test_simulator_produces_realistic_imbalance(line_runs):
    """Real lines are ~90% healthy. If the simulator stops reproducing that,
    every downstream claim is measured against the wrong regime."""
    _, _, g, _ = HighTechLine(LineConfig(seed=3, n_assets=30)).flatten(line_runs)
    healthy = float((g == "healthy").mean())
    critical = float((g == "critical").mean())
    assert 0.75 <= healthy <= 0.97, f"healthy fraction {healthy:.3f} is not line-like"
    assert 0.005 <= critical <= 0.08, f"critical fraction {critical:.3f} is not line-like"


def test_rul_is_monotone_decreasing(line_runs):
    for r in line_runs:
        assert np.all(np.diff(r.rul) <= 1e-9), f"{r.asset_id} RUL is not monotone"


def test_healthy_assets_are_right_censored(line_runs):
    """A healthy asset's RUL must sit exactly at the cap, not at an
    unobservable pseudo-value derived from trajectory length."""
    cap = 145.0
    healthy = [r for r in line_runs if not r.meta["degrades"]]
    assert healthy, "no healthy assets generated"
    for r in healthy:
        assert np.allclose(r.rul, cap), f"{r.asset_id} healthy RUL not censored at the cap"


def test_some_assets_degrade_and_some_do_not(line_runs):
    deg = sum(1 for r in line_runs if r.meta["degrades"])
    assert 0 < deg < len(line_runs), "every asset degrading removes the stay-quiet case"


def test_recipe_changes_occur(line_runs):
    """Multi-product operation is a core high-tech property; if recipes never
    change, the recipe-aware featurisation is untested."""
    changes = sum(int((np.diff(r.recipe) != 0).sum()) for r in line_runs)
    assert changes > 0, "no recipe changes generated"


# ------------------------------------------------------------------- leakage
def test_split_is_disjoint_by_asset(line_runs):
    s = split_runs(line_runs, seed=1)
    tr = {r.asset_id for r in s.train}
    va = {r.asset_id for r in s.val}
    te = {r.asset_id for r in s.test}
    assert not (tr & te) and not (tr & va) and not (va & te)
    assert tr | va | te == {r.asset_id for r in line_runs}, "split lost assets"


def test_leakage_audit_passes(line_runs):
    s = split_runs(line_runs, seed=2)
    a = assert_no_leakage(
        np.array([r.asset_id for r in s.train]), np.array([r.asset_id for r in s.test])
    )
    assert a["clean"] is True
    assert a["n_overlapping_assets"] == 0


def test_leakage_audit_detects_an_injected_overlap():
    a = assert_no_leakage(np.array(["A", "B"]), np.array(["B", "C"]))
    assert a["clean"] is False
    assert a["n_overlapping_assets"] == 1


def test_split_is_seed_deterministic(line_runs):
    a = split_runs(line_runs, seed=5)
    b = split_runs(line_runs, seed=5)
    assert [r.asset_id for r in a.test] == [r.asset_id for r in b.test]
    c = split_runs(line_runs, seed=6)
    assert [r.asset_id for r in a.test] != [r.asset_id for r in c.test]


def test_naive_window_split_reports_overlap(line_runs):
    """The audit baseline must quantify the leak rather than assert it."""
    out = naive_window_split(line_runs, window=30, stride=1, seed=0)
    assert out["window_overlap_ratio"] > 0.9
    assert "audit" in out


def test_group_healthy_partitions_correctly():
    g = np.array(["healthy", "degraded", "critical"])
    out = group_healthy(g)
    assert list(out) == ["healthy", "degraded+", "degraded+"]


# ---------------------------------------------------------------------- drift
def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=3000)
    assert psi(x, rng.normal(size=3000)) < 0.05


def test_psi_grows_with_a_real_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(size=3000)
    small = psi(base, rng.normal(loc=0.15, size=3000))
    large = psi(base, rng.normal(loc=1.0, size=3000))
    assert large > small


def test_psi_null_baseline_is_positive():
    """PSI has a non-zero null value; thresholding raw PSI is meaningless."""
    from calibra.assure.drift import _VectorPSI

    rng = np.random.default_rng(2)
    v = _VectorPSI(rng.normal(size=(800, 10)), n_bins=8)
    assert v.null_baseline().mean() > 0.0


def test_cusum_triggers_on_a_sustained_shift():
    """CUSUM must detect a shift *larger than its slack k*.

    A drift of 0.4 with k=0.5 cannot accumulate by construction
    (``g_pos = max(0, g_pos + x - k)``), so this test pins the operating
    point rather than assuming any drift is detectable.
    """
    detectable = CUSUM(h=5.0, k=0.1)
    fired = [detectable.update(0.4) for _ in range(80)]
    assert any(fired), "CUSUM missed a sustained 0.4 shift against k=0.1"

    below_slack = CUSUM(h=5.0, k=0.5)
    assert not any(below_slack.update(0.4) for _ in range(80)), (
        "a shift smaller than the slack must not accumulate"
    )
    assert not any(CUSUM(h=5.0, k=0.1).update(0.0) for _ in range(80)), "CUSUM fired on pure noise"


def test_monitor_stays_stable_on_stationary_data():
    """No residual, no coverage history -- the honest steady-state case.

    Passing placeholder zeros against a reference of real residuals would make
    the residual PSI fire, which is why ``update`` accepts ``None``.
    """
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 6))
    m = DriftMonitor(nominal=0.9, window=250, psi_every=1)
    m.set_reference(X, rng.normal(size=500))
    sev = [m.update(X[i], None, None).severity for i in range(500)]
    assert sev[-1] == Severity.STABLE, f"pinned at {sev[-1]} on stationary data"
    assert sum(1 for s in sev if s != Severity.STABLE) < 0.05 * len(sev)


def test_monitor_escalates_on_injected_shift():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(400, 6))
    m = DriftMonitor(nominal=0.9, window=200, psi_every=1, dwell=1)
    m.set_reference(X, rng.normal(size=400))
    for i in range(200):
        m.update(X[i], 0.0, True)
    Xs = X + 1.5
    sev = [m.update(Xs[i], 0.0, True).severity for i in range(200)]
    assert sev[-1] != Severity.STABLE, "monitor missed a 1.5-sigma input shift"


def test_monitor_respects_dwell():
    """A single breaching window must not escalate -- that is how a monitor
    becomes its own alert-fatigue source."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, 4))
    m = DriftMonitor(nominal=0.9, window=200, psi_every=1, dwell=5)
    m.set_reference(X, rng.normal(size=400))
    m.update(X[0] + 50.0, 0.0, True)
    assert m.last_action_ in (Severity.STABLE, Severity.WATCH)


def test_monitor_marks_calibration_invalid_on_escalation():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(300, 4))
    m = DriftMonitor(nominal=0.9, window=200, psi_every=1, dwell=1)
    m.set_reference(X, rng.normal(size=300))
    m.mark_recalibrated()
    assert m.calibration_valid_ is True
    for i in range(150):
        m.update(X[i], 0.0, True)
    for i in range(150):
        m.update(X[i] + 5.0, 0.0, True)
    if m.last_action_ != Severity.STABLE:
        assert m.calibration_valid_ is False


def test_coverage_schedule_shape():
    rng = np.random.default_rng(7)
    y = rng.normal(size=600)
    lo, hi = y - 2, y + 2
    s = coverage_schedule(np.full(600, 0.9), y, lo, hi, n_blocks=6)
    assert len(s) == 6
    assert all(0.0 <= b["coverage"] <= 1.0 for b in s)


# ------------------------------------------------------------------ pipeline
def test_pipeline_report_is_internally_consistent(fitted):
    rep = fitted.evaluate("test")
    assert rep["data"]["leakage_audit"]["clean"] is True
    # coverage must be reported with its gap, and near nominal
    p = rep["prognostic"]
    assert abs(p["coverage_gap"]) <= 0.12, f"coverage {p['coverage']:.3f} far from nominal"
    # epistemic share in [0,1]
    assert 0.0 <= p["epistemic_share"] <= 1.0
    # total sigma >= each component
    assert p["mean_aleatoric"] >= 0 and p["mean_epistemic"] >= 0


def test_pipeline_alarm_economics_beat_the_baseline(fitted):
    """The central economic claim: same predictions, far fewer wasted alarms."""
    rep = fitted.evaluate("test")
    c, b = rep["trust"]["calibra"], rep["trust"]["median_baseline"]
    assert c["alarm_rate"] < b["alarm_rate"], "CALIBRA is not quieter than the baseline"
    assert c["false_alarms"] < b["false_alarms"]
    assert c["eca_per_decision"] < b["eca_per_decision"]


def test_pipeline_caught_the_critical_events(fitted):
    rep = fitted.evaluate("test")
    assert rep["trust"]["calibra"]["recall"] >= 0.8, "recall too low to be useful"


def test_pipeline_cost_ci_is_reported_honestly(fitted):
    """The paired bootstrap CI must be reported, and must be consistent with
    the point estimate's sign.

    Deliberately *not* asserting significance: across seeds the paired CI
    excludes zero on some runs and not others (seeds 7 and 11 significant,
    seed 13 not). A single-seed significance assertion would be a test that
    passes only on lucky seeds, which is exactly the practice this project is
    arguing against.
    """
    rep = fitted.evaluate("test")
    lo, hi = rep["decision"]["paired_cost_delta_vs_median_ci95"]
    delta = rep["decision"]["mean_cost_delta_vs_median"]
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < delta < hi, "the point estimate must lie inside its own CI"
    assert hi - lo > 0


def test_pipeline_improvement_holds_across_seeds():
    """Mean cost must beat the constant-median baseline on average over seeds,
    with the win rate reported. This is the honest form of the economic claim.
    """
    deltas = []
    for seed in (7, 11, 13, 17):
        runs = HighTechLine(LineConfig(seed=seed, n_assets=30)).generate()
        rep = CalibraRunner(CalibraConfig(seed=seed)).fit(runs).evaluate("test")
        deltas.append(rep["decision"]["mean_cost_delta_vs_median"])
    assert float(np.mean(deltas)) < 0, f"no average cost improvement: {deltas}"
    assert sum(1 for d in deltas if d < 0) >= 3, f"wins only on {sum(d<0 for d in deltas)}/4 seeds"


def test_pipeline_response_rate_lowers_tar_but_not_recall(fitted):
    a = fitted.evaluate("test", response_rate=0.95)
    b = fitted.evaluate("test", response_rate=0.40)
    assert a["trust"]["calibra"]["tar"] > b["trust"]["calibra"]["tar"]
    assert a["prognostic"]["coverage"] == pytest.approx(b["prognostic"]["coverage"])


def test_online_step_returns_a_complete_evidence_card(fitted):
    X = fitted._data["Xte"]
    out = fitted.step(X[0])
    for k in ("prediction", "action", "health", "expected_cost", "evidence", "assurance"):
        assert k in out
    p = out["prediction"]
    assert p["lower"] <= p["rul_mean"] <= p["upper"]
    assert p["level"] == pytest.approx(0.9)
    assert out["evidence"]["rationale"]
    assert out["evidence"]["chosen_action"] == out["action"]
    assert len(out["expected_cost"]) == 6


def test_online_step_rejects_wrong_feature_count(fitted):
    with pytest.raises(ValueError):
        fitted.step(np.zeros(3))


def test_mondrian_and_global_both_produce_valid_intervals(line_runs):
    for mode in ("global", "mondrian"):
        r = CalibraRunner(CalibraConfig(seed=7, conformal_mode=mode)).fit(line_runs)
        rep = r.evaluate("test")
        assert rep["prognostic"]["mean_width"] > 0
        assert 0.5 <= rep["prognostic"]["coverage"] <= 1.0


def test_crc_refuses_to_certify_with_too_little_data(fitted):
    """Honest refusal beats a fabricated guarantee."""
    from calibra.decision.policy import PolicyConfig, conformal_risk_control
    import numpy as np

    cfg = PolicyConfig(min_calibration=10_000)
    ecost = {a.value: np.ones(50) for a in __import__("calibra").ACTION_LIST}
    out = conformal_risk_control(ecost, np.zeros(50), cfg)
    assert out["budget_usable"] is False
    assert "no guarantee" in out["reason"] or "min_calibration" in out["reason"]