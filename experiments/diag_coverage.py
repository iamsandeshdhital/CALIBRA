"""Diagnostic: does group-conditional calibration fix the critical-cell failure?

CALIBRA's Mondrian mode uses two cells (healthy | degraded+). This script
checks whether a three-cell taxonomy (healthy | degraded | critical) closes
the conditional-coverage gap on the critical cell, and what it costs in
interval width.

Run:  python experiments/diag_coverage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibra import CalibraConfig, CalibraRunner
from calibra.sim.hightech_line import HighTechLine, LineConfig
from calibra.uq.conformal import ConformalRUL


def three_way(regimes: np.ndarray) -> np.ndarray:
    r = np.asarray(regimes)
    return np.where(r == "healthy", "healthy", np.where(r == "degraded", "degraded", "critical"))


def main() -> None:
    line = HighTechLine(LineConfig(seed=7, n_assets=34))
    runs = line.generate()
    r = CalibraRunner(CalibraConfig(seed=7, conformal_mode="mondrian")).fit(runs)

    c = r.conformal_
    print("=== fitted Mondrian (2-cell) ===")
    print(f"global radius : {c.global_radius_:.3f}")
    print(f"groups        : {c.groups_}")
    print(f"radii         : {np.round(c.radii_, 3)}")

    Xva, yva, rva = r._data["Xva"], r._data["yva"], r._data["rva"]
    Xte, yte, rte = r._data["Xte"], r._data["yte"], r._data["rte"]
    muv, sigv = r._predict_distributions(Xva, None)[0], r._predict_distributions(Xva, None)[3]
    mute, sigte = r._predict_distributions(Xte, None)[0], r._predict_distributions(Xte, None)[3]

    print("\n=== two-cell conditional coverage (test) ===")
    for label, mask in (
        ("healthy", rte == "healthy"),
        ("degraded+", rte != "healthy"),
        ("critical", yte <= 25),
    ):
        if mask.sum():
            lo, hi, _ = c.interval(mute, sigte, np.where(rte == "healthy", "healthy", "degraded+"))
            ins = (yte[mask] >= lo[mask]) & (yte[mask] <= hi[mask])
            print(f"  {label:<10s} {ins.mean():.3f}   (n={int(mask.sum())})")

    print("\n=== three-cell Mondrian ===")
    g_va, g_te = three_way(rva), three_way(rte)
    sizes = {g: int((g_va == g).sum()) for g in np.unique(g_va)}
    print("calibration group sizes:", sizes)
    c3 = ConformalRUL(alpha=0.10, mode="mondrian", min_group_n=30, normalise_by_sigma=True)
    c3.fit(yva, muv, sigv, g_va)
    print("three-cell radii:", {g: round(float(v), 3) for g, v in zip(c3.groups_, c3.radii_)})

    lo3, hi3, _ = c3.interval(mute, sigte, g_te)
    ins3 = (yte >= lo3) & (yte <= hi3)
    for g in ("healthy", "degraded", "critical"):
        m = rte == g
        if m.sum():
            print(f"  {g:<10s} coverage={ins3[m].mean():.3f}  (n={int(m.sum())})")
    print(f"  marginal  coverage={ins3.mean():.3f}")
    print(f"  mean width={float(np.mean(hi3 - lo3)):.2f}  (two-cell: 26.10)")

    print("\n=== cost under each calibration ===")
    from calibra.data.features import group_healthy
    from calibra.decision.costs import OutcomeProbabilities, fab_cost_matrix
    from calibra.decision.policy import PolicyConfig, choose_actions, assign, policy_summary

    mtx = fab_cost_matrix()
    out = OutcomeProbabilities(90.0, 25.0)
    for name, (lo_, hi_, g) in {
        "two-cell": c.interval(mute, sigte, group_healthy(rte)),
        "three-cell": (lo3, hi3, g_te),
    }.items():
        P = out.from_interval(lo_, hi_)
        ec = {a.value: P @ mtx.row(a) for a in mtx.row.__self__ and __import__("calibra").ACTION_LIST}
        idx, hc = choose_actions(ec, hi_ - lo_, PolicyConfig())
        acts, _ = assign(idx, hc)
        s = policy_summary(acts, rte, mtx)
        n_alarm = sum(1 for a in acts if a != "run")
        print(
            f"  {name:<11s} mean_cost={s['mean_cost']:8.1f}  alarms={n_alarm:4d}  "
            f"critical_caught={int(((np.array(acts) != 'run') & (rte == 'critical')).sum())}/{int((rte == 'critical').sum())}"
        )


if __name__ == "__main__":
    main()