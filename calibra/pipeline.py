"""End-to-end CALIBRA pipeline: ingest -> forecast -> calibrate -> decide -> assure.

This is the object a plant would actually run. The public entry point is
``CalibraRunner``, which wires the six stages together and produces a single
audit report.

    raw telemetry
      -> leakage-audited split          (data.split)
      -> recipe-aware featurisation     (data.features)
      -> ensemble RUL + UQ decomposition (models.ensemble)
      -> split-conformal calibration    (uq.conformal)
      -> cost-sensitive decision + CRC  (decision.policy)
      -> streaming drift monitor        (assure.drift)
      -> evidence cards                 (evidence.card)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .assure.drift import DriftMonitor, Severity, coverage_schedule
from .data.features import FeatureSpec, group_healthy
from .data.split import assert_no_leakage, split_runs
from .decision.costs import CostMatrix, OutcomeProbabilities, cost_ratio, fab_cost_matrix
from .decision.policy import (
    PolicyConfig,
    assign,
    choose_actions,
    conformal_risk_control,
    expected_cost,
    oracle_cost,
    policy_summary,
)
from .evidence.card import build_card
from .metrics.score import (
    bootstrap_ci,
    expected_cost_of_alarm,
    full_report,
    trust_adjusted_recall,
)
from .models.ensemble import ConstantMedianRUL, EnsembleRUL
from .types import REGIME_ORDER, Action, Health, PredictiveDistribution, RunToFailure
from .uq.conformal import ConformalRUL


@dataclass
class CalibraConfig:
    """One config object so an experiment is fully specified by a serialisable
    dict -- no hidden globals, no implicit defaults buried in functions."""

    window: int = 12
    stat_window: int = 10
    n_members: int = 15
    conformal_alpha: float = 0.10
    conformal_mode: str = "mondrian"  # "global" | "mondrian"
    normalise_by_sigma: bool = True
    drift_kappa: float = 1.6
    drift_cap: float = 2.5

    degraded_rul: float = 90.0
    critical_rul: float = 25.0
    #: RUL censoring horizon. Must match the data generator's cap: clipping
    #: intervals into a different range than the labels were produced in
    #: silently rescales the regime probabilities.
    rul_cap: float = 145.0

    cost_matrix: str = "fab"  # "fab" | "bearing"
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    #: Human alert-response rate, measured from CMMS work-order closure.
    #: 0.85 is the field-healthy benchmark reported by reliability
    #: programmes; lower values model a fatigued team.
    response_rate: float = 0.85

    seed: int = 7
    asset_wise_split: bool = True

    def matrix(self) -> CostMatrix:
        if self.cost_matrix == "bearing":
            from .decision.costs import bearing_cost_matrix

            return bearing_cost_matrix()
        return fab_cost_matrix()


@dataclass
class StageResult:
    name: str
    metrics: dict[str, float]
    detail: dict[str, object] = field(default_factory=dict)


class CalibraRunner:
    """Fit, calibrate, evaluate and stream. Reusable across datasets."""

    def __init__(self, cfg: CalibraConfig | None = None) -> None:
        self.cfg = cfg or CalibraConfig()
        self.matrix_ = self.cfg.matrix()
        self.model_: EnsembleRUL | None = None
        self.conformal_: ConformalRUL | None = None
        self.monitor_: DriftMonitor | None = None
        self.report_: dict[str, object] = {}

    # ------------------------------------------------------------------ train
    def fit(self, runs: Sequence[RunToFailure]) -> "CalibraRunner":
        cfg = self.cfg
        split = split_runs(
            runs, seed=cfg.seed, asset_wise=cfg.asset_wise_split
        )
        leak = assert_no_leakage(
            np.array([r.asset_id for r in split.train]),
            np.array([r.asset_id for r in split.test]),
        )

        from .data.features import featurize

        spec = FeatureSpec(window=cfg.stat_window)
        Xtr, ytr, gtr, rtr, fb = featurize(split.train, spec)
        Xva, yva, gva, rva, _ = featurize(split.val, spec)
        Xte, yte, gte, rte, _ = featurize(split.test, spec)

        # Truncate feature rows until the rolling statistics are warm.
        w = cfg.stat_window
        Xtr, ytr, gtr, rtr = Xtr[w:], ytr[w:], gtr[w:], rtr[w:]
        Xva, yva, gva, rva = Xva[w:], yva[w:], gva[w:], rva[w:]
        Xte, yte, gte, rte = Xte[w:], yte[w:], gte[w:], rte[w:]

        model = EnsembleRUL(n_members=cfg.n_members, random_state=cfg.seed)
        model.fit(Xtr, ytr, fb.feature_names_)
        self.model_ = model
        self.feature_names_ = fb.feature_names_

        # --- calibrate on validation. Conformal's whole point is that this
        # split is *only* used to fit one radius, never to fit the model.
        mu_va, sa_va, se_va = model.predict(Xva)
        sig_va = np.sqrt(sa_va**2 + se_va**2)
        conf = ConformalRUL(
            alpha=cfg.conformal_alpha,
            mode=cfg.conformal_mode,
            normalise_by_sigma=cfg.normalise_by_sigma,
            drift_kappa=cfg.drift_kappa,
            drift_cap=cfg.drift_cap,
        )
        conf.fit(yva, mu_va, sig_va, group_healthy(rva))
        self.conformal_ = conf

        # --- set the drift reference to the calibration distribution
        resid_va = yva - mu_va
        self.monitor_ = DriftMonitor(
            nominal=1.0 - cfg.conformal_alpha, window=max(60, min(400, len(yva) // 2))
        )
        self.monitor_.set_reference(Xva, resid_va)

        self._data = {
            "split": split,
            "leak": leak,
            "Xtr": Xtr, "ytr": ytr, "rtr": rtr,
            "Xva": Xva, "yva": yva, "rva": rva,
            "Xte": Xte, "yte": yte, "rte": rte,
            "feature_names": fb.feature_names_,
        }
        return self

    # ------------------------------------------------------------- inference
    def _predict_distributions(
        self, X: np.ndarray, groups: np.ndarray, drift: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mu, sa, se = self.model_.predict(X)
        sig = np.sqrt(sa**2 + se**2)
        lo, hi, rad = self.conformal_.interval(mu, sig, groups, drift=drift)
        return mu, sa, se, sig, lo, hi

    # ------------------------------------------------------------- evaluation
    def evaluate(self, split_name: str = "test", response_rate: float | None = None) -> dict[str, object]:
        cfg = self.cfg
        # "test"/"val" are the public names; internally they are short keys.
        suffix = {"test": "te", "val": "va", "train": "tr"}[split_name]
        d = self._data[f"X{suffix}"]
        y = self._data[f"y{suffix}"]
        truth = self._data[f"r{suffix}"]
        groups = group_healthy(truth)

        mu, sa, se, sig, lo, hi = self._predict_distributions(d, groups)

        # ---- Stage: prognostic accuracy + calibration
        nominal = 1.0 - cfg.conformal_alpha
        score = full_report(y, mu, sig, lo, hi, nominal)
        score["mean_aleatoric"] = float(sa.mean())
        score["mean_epistemic"] = float(se.mean())
        score["epistemic_share"] = float((se**2).sum() / max((sig**2).sum(), 1e-9))
        cov_by_group = self.conformal_.conditional_coverage(y, mu, sig, groups)

        # ---- Stage: decision
        P = OutcomeProbabilities(cfg.degraded_rul, cfg.critical_rul, cfg.rul_cap).from_interval(lo, hi)
        ecost = expected_cost(P, self.matrix_)
        width = hi - lo
        inside = (y >= lo) & (y <= hi)
        chosen_idx, health_code = choose_actions(
            ecost, width, cfg.policy, coverage=np.where(inside, 1.0, 0.0)
        )
        actions, _ = assign(chosen_idx, health_code)
        summary = policy_summary(actions, truth, self.matrix_)
        oracle = oracle_cost(truth, self.matrix_)

        alerts = np.asarray([a != Action.RUN.value for a in actions])
        critical = truth == REGIME_ORDER[2]
        rr = cfg.response_rate if response_rate is None else response_rate
        eca = expected_cost_of_alarm(actions, truth, self.matrix_)
        tar = trust_adjusted_recall(alerts, critical, rr)

        crc = conformal_risk_control(ecost, truth, cfg.policy)

        # ---- Stage: streaming assurance
        assert self.monitor_ is not None
        monitor = DriftMonitor(
            psi_watch=self.monitor_.psi_watch,
            psi_recal=self.monitor_.psi_recal,
            psi_retrain=self.monitor_.psi_retrain,
            coverage_tolerance=self.monitor_.coverage_tolerance,
            nominal=self.monitor_.nominal,
            window=self.monitor_.window,
            dwell=self.monitor_.dwell,
        )
        monitor.set_reference(self._data["Xva"], self._data["yva"] - self._predict_distributions(
            self._data["Xva"], group_healthy(self._data["rva"])
        )[0])
        actions_seen: list[str] = []
        for i in range(len(y)):
            rep = monitor.update(d[i], float(y[i] - mu[i]), bool(inside[i]))
            actions_seen.append(rep.recommended)
        n_escalate = int(sum(1 for a in actions_seen if a != Severity.STABLE.value))
        self.monitor_ = monitor

        # ---- Stage: baselines on the same split
        base = ConstantMedianRUL().fit(self._data["Xtr"], self._data["ytr"])
        bmu, bsa, bse = base.predict(d)
        bsig = np.sqrt(bsa**2 + bse**2)
        blo, bhi, _ = self.conformal_.interval(bmu, bsig, groups)
        base_score = full_report(y, bmu, bsig, blo, bhi, nominal)
        bP = OutcomeProbabilities(cfg.degraded_rul, cfg.critical_rul, cfg.rul_cap).from_interval(blo, bhi)
        becost = expected_cost(bP, self.matrix_)
        b_idx, b_health = choose_actions(
            becost, bhi - blo, cfg.policy, coverage=np.where((y >= blo) & (y <= bhi), 1.0, 0.0)
        )
        b_actions, _ = assign(b_idx, b_health)
        b_summary = policy_summary(b_actions, truth, self.matrix_)
        b_eca = expected_cost_of_alarm(b_actions, truth, self.matrix_)
        b_alerts = np.asarray([a != Action.RUN.value for a in b_actions])
        b_tar = trust_adjusted_recall(b_alerts, critical, rr)

        cost_diff = np.array(
            [self.matrix_.get(str(a), str(t)) for a, t in zip(actions, truth)], float
        ) - np.array(
            [self.matrix_.get(str(a), str(t)) for a, t in zip(b_actions, truth)], float
        )
        ci_lo, ci_hi = bootstrap_ci(cost_diff, np.mean)

        report = {
            "config": {
                "cost_matrix": cfg.cost_matrix,
                "cost_ratio_CFN_over_CFP": cost_ratio(self.matrix_),
                "conformal_alpha": cfg.conformal_alpha,
                "mode": self.conformal_.mode,
                "n_calibration": self.conformal_.n_calib_,
                "calibration_radius": self.conformal_.global_radius_,
                "response_rate": rr,
                "seed": cfg.seed,
            },
            "data": {
                "leakage_audit": self._data["leak"],
                "split": self._data["split"].summary(),
                "test_regime_distribution": {
                    r: float(np.mean(truth == r)) for r in REGIME_ORDER
                },
            },
            "prognostic": score,
            "calibration_by_group": cov_by_group,
            "decision": {
                **summary,
                "oracle_mean_cost": oracle,
                "regret_ratio": summary["mean_cost"] / max(oracle, 1e-9),
                "baselines_median_policy": b_summary,
                "regret_ratio_vs_median_baseline": b_summary["mean_cost"] / max(oracle, 1e-9),
                "action_mix": {
                    a.value: int(np.sum(np.asarray(actions, str) == a.value)) for a in Action
                },
                "health_mix": {
                    Health.TRUSTED.value: int(np.sum(health_code == 0)),
                    Health.WIDE.value: int(np.sum(health_code == 1)),
                    Health.ABSTAIN.value: int(np.sum(health_code == 2)),
                },
                "conformal_risk_control": crc,
                "paired_cost_delta_vs_median_ci95": [ci_lo, ci_hi],
                "mean_cost_delta_vs_median": float(cost_diff.mean()),
            },
            "trust": {
                "calibra": {**eca, **tar},
                "median_baseline": {**b_eca, **b_tar},
                "tar_penalty_median_baseline": b_tar["tar"] - b_tar["recall"],
                "tar_penalty_calibra": tar["tar"] - tar["recall"],
            },
            "assurance": {
                "n_escalations": n_escalate,
                "escalation_rate": n_escalate / max(len(y), 1),
                "final_severity": monitor.last_action_.value,
                "transitions": monitor.status(),
                "coverage_schedule": coverage_schedule(
                    np.full(len(y), nominal), y, lo, hi, n_blocks=6
                ),
            },
            "baselines": {
                "median_policy": base_score,
            },
            "uqcoupling": {
                "epistemic_share": score["epistemic_share"],
                "coverage_gap": score["coverage_gap"],
                "comment": "epistemic share should rise in OOD regions; "
                           "coverage gap near zero is the contract",
            },
        }
        self.report_ = report
        return report

    # ------------------------------------------------------------- streaming
    def step(self, x_row: np.ndarray, regime_hint: str | None = None) -> dict[str, object]:
        """One online decision point, with drift checking and an evidence card.

        ``regime_hint`` is only for offline replay/evaluation; in production
        it is unavailable and the calibration still holds.
        """
        assert self.model_ is not None and self.conformal_ is not None and self.monitor_ is not None
        x = np.asarray(x_row, float).reshape(1, -1)
        if x.shape[1] != len(self.feature_names_):
            raise ValueError(
                f"expected {len(self.feature_names_)} features, got {x.shape[1]}"
            )

        mu, sa, se, sig, lo, hi = self._predict_distributions(x, None)
        P = OutcomeProbabilities(
            self.cfg.degraded_rul, self.cfg.critical_rul, self.cfg.rul_cap
        ).from_interval(lo, hi)
        ecost = expected_cost(P, self.matrix_)
        width = hi - lo
        idx, hcode = choose_actions(ecost, width, self.cfg.policy)
        names, _ = assign(idx, hcode)
        action_name = names[0]
        action = action_name if isinstance(action_name, Action) else Action(str(action_name))
        health = [Health.TRUSTED, Health.WIDE, Health.ABSTAIN][int(hcode[0])]

        dist = PredictiveDistribution(
            mean=float(mu[0]),
            aleatoric=float(sa[0]),
            epistemic=float(se[0]),
            lower=float(lo[0]),
            upper=float(hi[0]),
            level=1.0 - self.cfg.conformal_alpha,
        )

        attribution = self.model_.local_attribution(x, top_k=4, seed=self.cfg.seed)
        card = build_card(
            i=0,
            dist=dist,
            action=action,
            health=health,
            attribution=attribution[0],
            probs=P[0],
            matrix=self.matrix_,
            ecost=ecost,
            n_calibration=self.conformal_.n_calib_,
            drift_score=float(self.monitor_.last_action_ != Severity.STABLE),
            coverage_recent=None,
        )

        from .evidence.card import card_to_dict

        return {
            "prediction": {
                "rul_mean": dist.mean,
                "lower": dist.lower,
                "upper": dist.upper,
                "level": dist.level,
                "std": dist.std,
                "p_degraded": float(P[0][1]),
                "p_critical": float(P[0][2]),
            },
            "action": action.value,
            "health": health.value,
            "expected_cost": {k: float(v[0]) for k, v in ecost.items()},
            "evidence": card_to_dict(card),
            "assurance": self.monitor_.status(),
        }


def format_report(rep: dict[str, object]) -> str:
    """Human-readable digest for a terminal or a CI log."""
    L: list[str] = []
    add = L.append
    add("=" * 72)
    add("CALIBRA REPORT")
    add("=" * 72)

    c = rep["config"]
    add(f"cost matrix      : {c['cost_matrix']}  (C_FN/C_FP = {c['cost_ratio_CFN_over_CFP']:.3g})")
    add(f"conformal        : alpha={c['conformal_alpha']}  mode={c['mode']}  "
        f"n_calib={c['n_calibration']}  radius={c['calibration_radius']:.3f}")
    add(f"alert response   : {c['response_rate']:.2f}")

    d = rep["data"]
    la = d["leakage_audit"]
    add(f"leakage audit    : clean={la['clean']}  (overlapping assets={la['n_overlapping_assets']})")
    add(f"test regimes     : " + ", ".join(
        f"{k}={v:.3%}" for k, v in d["test_regime_distribution"].items()))

    p = rep["prognostic"]
    add("-" * 72)
    add(f"RMSE {p['rmse']:.2f}   MAE {p['mae']:.2f}   CRPS {p['crps_gaussian']:.2f}")
    add(f"coverage {p['coverage']:.3f} (target {1-c['conformal_alpha']:.2f}, "
        f"gap {p['coverage_gap']:+.3f})   mean width {p['mean_width']:.1f}")
    add(f"Winkler {p['winkler']:.2f}   PIT-KS {p['pit_ks']:.3f}   ECE {p['ece']:.3f}")
    add(f"aleatoric {p['mean_aleatoric']:.2f}  epistemic {p['mean_epistemic']:.2f}  "
        f"epistemic share {p['epistemic_share']:.2%}")

    add("-" * 72)
    add("conditional coverage (Mondrian):")
    for g, v in rep["calibration_by_group"].items():
        add(f"  {g:<14s} {v:.3f}")

    dec = rep["decision"]
    add("-" * 72)
    add(f"mean cost/decision {dec['mean_cost']:,.1f}   oracle {dec['oracle_mean_cost']:,.1f}   "
        f"regret {dec['regret_ratio']:.2f}x")
    add(f"median-baseline     {dec['baselines_median_policy']['mean_cost']:,.1f}   "
        f"regret {dec['regret_ratio_vs_median_baseline']:.2f}x")
    lo, hi = dec["paired_cost_delta_vs_median_ci95"]
    add(f"paired cost delta vs median: {dec['mean_cost_delta_vs_median']:,.1f} "
        f"[{lo:,.1f}, {hi:,.1f}]")
    add(f"action mix          : " + ", ".join(f"{k}={v}" for k, v in dec["action_mix"].items()))
    add(f"self-assessment     : " + ", ".join(f"{k}={v}" for k, v in dec["health_mix"].items()))
    crc = dec["conformal_risk_control"]
    add(f"conformal risk ctrl : usable={crc['budget_usable']}  budget="
        f"{crc.get('budget', float('nan')):,.1f}  guaranteed={crc.get('guaranteed_mean_cost', float('nan')):,.1f}")

    add("-" * 72)
    t = rep["trust"]
    for who in ("calibra", "median_baseline"):
        s = t[who]
        add(f"{who:<17s}: alarm_rate={s['alarm_rate']:.3f}  false_alarms={int(s['false_alarms']):5d}  "
            f"ECA/dec={s['eca_per_decision']:10,.1f}  recall={s['recall']:.3f}  TAR={s['tar']:.3f}")

    a = rep["assurance"]
    add("-" * 72)
    add(f"drift escalations  : {a['n_escalations']} ({a['escalation_rate']:.2%}), "
        f"final={a['final_severity']}")
    add("rolling coverage:")
    for b in a["coverage_schedule"]:
        add(f"  block {b['block']}  cov={b['coverage']:.3f}  width={b['width']:.1f}")
    add("=" * 72)
    return "\n".join(L)