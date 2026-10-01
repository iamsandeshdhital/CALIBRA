# CALIBRA

**Cost-Aware, Leakage-audited, Interpretable Bayesian Reliability Advisor**

An industrial-AI reference implementation for **high-tech manufacturing** lines —
semiconductor fabs, battery/EV cell lines, precision machining cells, photonics.

> **The one-line problem.** Predictive-maintenance models are trained and scored on
> the wrong objective. They are optimised for point accuracy (RMSE) and evaluated
> there. A plant does not fail because an RUL estimate was off by twelve cycles — it
> fails because the model was overconfident, never abstained, silently drifted, and
> could not say what an alarm would cost or whether anyone would act on it.

---

## Why this is the gap

The predictive-maintenance literature is saturated with models that predict
remaining useful life (RUL) accurately. What it is **not** saturated with is
evidence that those models get *used* — that they survive contact with an alarm
budget, a maintenance team's attention span, and a production line that drifts.

Each requirement below is traceable to a specific documented failure:

| # | Requirement | Documented evidence |
|---|---|---|
| **R1** | **Accuracy is the wrong target.** Optimise and evaluate expected *cost*, not RMSE | Cost-sensitive learning is governed by the cost ratio, so any fixed probability threshold is *wrong by construction* when costs are asymmetric — Elkan, IJCAI 2001 ([UCSD](https://cseweb.ucsd.edu/~elkan/rescale.pdf)) |
| **R2** | **Failures are rare.** Most samples are "keep running", so a model must learn to stay quiet | Run-to-failure benchmarks (C-MAPSS, PHM-2010) contain *only* failures; a model trained on them never learns restraint. Alarm fatigue from repeated false alarms is documented as "the silent killer of predictive maintenance ROI", with healthy programmes at >85% alert response and <20% FPR ([Oxmaint](https://oxmaint.com/article/predictive-maintenance-alert-fatigue), [iFactory](https://ifactoryapp.com/industries/manufacturing-plant/alert-fatigue-predictive-maintenance-solution)) |
| **R3** | **Intervals must be calibrated, and the guarantee must survive.** Report a distribution-free coverage guarantee | Split conformal turns calibration into a counting problem with a distribution-free guarantee — Angelopoulos & Bates (CMU / Harvard), *Found. Trends in Machine Learning* 16(4), 2023 ([ACM](https://dl.acm.org/doi/10.1561/2200000101)) |
| **R4** | **Marginal coverage hides conditional failure.** Guarantee must hold *where it matters*, not on average | Mondrian (group-conditional) calibration; a model can be perfectly covered on healthy assets and uncovered on failing ones. Measured here directly — see [Negative result N1](#negative-results) |
| **R5** | **Overconfidence must be decomposed.** Aleatoric and epistemic uncertainty drive *different* responses | Kendall & Gal (Cambridge / Toronto), NeurIPS 2017 ([arXiv:1703.04977](https://arxiv.org/abs/1703.04977)) |
| **R6** | **The system must be able to abstain.** Emit a self-assessment, not just a number | Selective prediction with a reject option — Geifman & El-Yaniv (Technion), ICML 2019 ([arXiv:1901.09192](https://arxiv.org/pdf/1901.09192)) |
| **R7** | **The operator must be able to disagree on explicit grounds.** Ship an evidence bundle | Explanation + uncertainty combined for prognostics — *Eng. Applications of AI* 2025 ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197625021220)) |
| **R8** | **Calibration silently expires under drift.** Detect it and say so | Drift-driven degradation is "not obvious to machine or plant operators" — Jourdan et al. (TU Darmstadt), 2025 ([TUprints](https://tuprints.ulb.tu-darmstadt.de/entities/publication/6ae533a3-cd99-449d-878f-e4e25ed89ba7)); ADWIN — Bifet & Gavaldà, SDM 2007 |
| **R9** | **The cost budget itself must be guaranteed, not tuned.** | Conformal Risk Control — Angelopoulos, Bates, Fisch, Lei & Schuster (Berkeley / Stanford / MIT / Google Research), ICLR 2024 |
| **R10** | **Evaluation must be leakage-free.** | Shamim et al. (2026) showed sliding-window splitting can move accuracy between **~20–60% (genuine) and 99.9% (artefactual)** on identical data and architecture ([arXiv:2607.16493](https://arxiv.org/html/2607.16493v1)) |
| **R11** | **The research gap is stated by the field itself** | NIST *2026 Roadmap on AI and ML for Smart Manufacturing* ([NIST](https://www.nist.gov/publications/2026-roadmap-artificial-intelligence-and-machine-learning-smart-manufacturing), [IOP](https://iopscience.iop.org/article/10.1088/3049-4761/ae5967)) |

Supporting industrial context: Siemens Senseye, Rolls-Royce IntelligentEngine,
and Bosch-class deployments all ship *decision* products rather than RUL numbers
([Siemens](https://www.siemens.com/en-us/products/industrial-digitalization-services/senseye-predictive-maintenance/),
[Rolls-Royce](https://www.rolls-royce.com/innovation/digital/digital-twin.aspx)).

---

## Architecture

```
 raw telemetry
   │
   ├─► 1. INGEST      leakage-audited split by whole asset   data/split.py
   ├─► 2. CONTEXT     recipe-aware features + health index   data/features.py
   ├─► 3. PROGNOSE    ensemble RUL, aleatoric/epistemic UQ   models/ensemble.py
   ├─► 4. CALIBRATE   split + Mondrian conformal interval    uq/conformal.py
   ├─► 5. DECIDE      cost minimisation + abstention + CRC   decision/
   └─► 6. ASSURE      PSI/CUSUM drift monitor, evidence      assure/, evidence/

 calibrated interval
      → regime probabilities
      → E[cost | action] over a 6-rung action ladder
      → argmin, gated by a self-assessment
      → bounded by a conformal cost budget
```

**The action ladder** (ordered by intrusiveness, not by probability):

`RUN → WATCH → INSPECT → RECIPE_TUNE → RECOVER → STOP`

Every rung is priced, including the ones that never fire. No action is free, and
`RECIPE_TUNE` is only *partially* effective — which is what makes the ladder
non-monotone and therefore not reducible to a threshold.

---

## Quick start

```bash
pip install -r requirements.txt

# End-to-end run on the built-in high-tech line simulator
python -c "
from calibra import CalibraConfig, CalibraRunner, format_report
from calibra.sim.hightech_line import HighTechLine, LineConfig
runs = HighTechLine(LineConfig(seed=7, n_assets=34)).generate()
print(format_report(CalibraRunner(CalibraConfig(seed=7)).fit(runs).evaluate('test')))
"

# One online decision point, with the full evidence card
python -c "
from calibra import CalibraConfig, CalibraRunner
from calibra.sim.hightech_line import HighTechLine, LineConfig
import json
runs = HighTechLine(LineConfig(seed=7, n_assets=34)).generate()
r = CalibraRunner(CalibraConfig(seed=7)).fit(runs)
print(json.dumps(r.step(r._data['Xte'][0]), indent=2, default=str))
"

# Test suite (78 tests) and experiment suite
python -m pytest tests/ -q
python -m experiments.run_experiments --seeds 7 11 13 --json-out results/experiments.json
```

---

## The high-tech line simulator

Published benchmarks cannot support these claims, so `calibra/sim/` builds a
line that reproduces the four pathologies that actually break PdM in the field:

| Pathology | Real analogue | Why public benchmarks miss it |
|---|---|---|
| **Label imbalance** (~90% healthy / 8% degraded / 2% critical) | A quarter of tool time on a real line | C-MAPSS/PHM-2010 are 100% run-to-failure |
| **Sensor faults** (stuck-at freeze, gain jumps) | A drifting thermistor, a replaced encoder | Benchmarks assume perfect sensors |
| **Multi-product regime change** | 4 recipes; the same channel means different things | Benchmarks are single-process |
| **Non-stationary tool ageing** | Shared slow drift + Poisson step shifts | Benchmarks are i.i.d. samples |

Real-dataset adapters live alongside it for the accuracy claims; the simulator is
used for the trust, calibration and drift claims, because only it contains these
pathologies. See `docs/DESIGN.md` for the generator's physics.

---

## Results

Reproduce with `python -m experiments.run_experiments --seeds 7 11 13`
(results committed at `results/experiments.json`).

### R4 — Marginal coverage hides conditional failure (experiment E3)

The sharpest result, and the clearest justification for group-conditional
calibration. Same data, same model, same split — only the calibration granularity
changes:

| Calibration | marginal coverage | **degraded+ conditional** | healthy conditional | mean width | mean cost |
|---|---|---|---|---|---|
| global (marginal) | 0.911 ✅ | **0.429** ❌ | 0.949 | 17.96 | 390.2 |
| Mondrian (conditional) | 0.923 ✅ | **0.907** ✅ | 0.924 | 20.25 | **320.6** |

The globally-calibrated model reports **0.911 coverage — nominally on target —
while missing 57% of degraded assets**, which are exactly the population that
becomes a critical defect next. Marginal coverage is satisfied; the guarantee is
useless. Mondrian calibration costs 13% more width and **18% less money**.

A single number would have shown "0.911, within tolerance" for the broken model.

### Calibration and economics (experiments E1, E2, E7)

| Metric | Value | Target |
|---|---|---|
| Conformal coverage (seed 7) | 0.885 | 0.90 |
| Coverage gap | −0.015 | ≈ 0 |
| CRPS (proper, non-negative) | 7.48 | — |
| Winkler interval score | 51.68 | — |
| Mean interval width | 26.1 | — |
| Epistemic share of variance | 48.9% | — |

| | CALIBRA | Constant-median baseline |
|---|---|---|
| Alarm rate | **12.4%** | 100% |
| False alarms | **83** | 1 793 |
| Expected cost of alarm / decision | **€1.6** | €21.4 |
| Recall (critical events caught) | 1.00 | 1.00 |
| Trust-Adjusted Recall (85% response) | 0.85 | 0.85 |
| Mean cost / decision | **€427** | €525 |

Paired bootstrap on cost: **−€97 [−153, −39]** (seed 7, significant). Across
seeds 7/11/13 the per-seed CI excludes zero on **2 of 3** — reported that way on
purpose, since a single-seed significance claim would be the practice this
project criticises. Multi-seed means (E7, 3 seeds): coverage **0.874
[0.848, 0.899]**, TAR 0.806 [0.719, 0.850].

**TAR sweep (E2)** — recall is 1.00 at every response rate for both models, so
plain recall cannot distinguish them. TAR exposes the difference:

| Alert response rate | 0.95 | 0.85 | 0.60 | 0.35 |
|---|---|---|---|---|
| Recall (both models) | 1.00 | 1.00 | 1.00 | 1.00 |
| TAR | 0.95 | 0.85 | 0.60 | 0.35 |

The deployed safety of an identical prediction collapses to 35% as the team
fatigues, while every conventional metric stays at 1.00.

**Policy adapts to the cost regime (E5)** — identical predictions, two plants:

| | fab (C_FN/C_FP 142) | machining cell (C_FN/C_FP 16) |
|---|---|---|
| RUN | 87.6% | 88.0% |
| WATCH | 0.2% | 6.0% |
| INSPECT | 11.2% | 4.8% |
| RECOVER | 1.1% | 1.3% |

**Leakage audit (E6)** — the naive stride-1 window protocol puts **30 assets on
both sides** of the split (96.7% window overlap). CALIBRA's asset-wise split has
0, verified independently.

**Self-assessment** (1 993 decision points): 1 679 `TRUSTED`, 313 `WIDE`,
1 `ABSTAIN`. The system declines to commit to a physical action when the interval
cannot support one.

---

## Negative results

Reported because a project that only prints favourable numbers is not measuring.

**N1 — Finer conditional calibration made things much worse.**
A three-cell Mondrian taxonomy (healthy / degraded / critical) looks more
rigorous than the two-cell one. Measured on the same split:

| Taxonomy | critical-cell coverage | mean cost | criticals caught |
|---|---|---|---|
| 2-cell (healthy \| degraded+) | **0.977** | **391.0** | 43 / 43 |
| 3-cell (healthy \| degraded \| critical) | **0.372** | 524.8 | 41 / 43 |

The critical cell held only 22 calibration points — below the 26 needed to
certify its own radius at α = 0.10 — so it silently inherited the global radius.
**Finer conditioning is worse when the cells cannot be certified.** `ConformalRUL`
therefore enforces `min_group_n` and exposes `audit_groups()` so the fallback is
visible rather than silent. Reproduce: `python experiments/diag_coverage.py`.

**N2 — PSI on a rolling window is not a drift signal as normally configured.**
PSI has a non-zero null value: with 8 bins, two samples from the *same*
distribution give E[PSI] ≈ (k−1)/2 ≈ 3.5. The familiar thresholds (< 0.10
stable, > 0.25 significant) implicitly assume samples thousands of times larger
than the bin count. On a 200-sample window the null value is an order of
magnitude above them, so a conventionally-tuned monitor fires permanently — it
reported drift on 99.6% of healthy decision points before this was fixed.
CALIBRA measures the null baseline by splitting its own reference and thresholds
the *excess*. Escalation rate on stationary data is now 0%.

**N3 — The drift safeguard is a bound, not a guarantee.**
Conformal validity requires exchangeability, which production data never
satisfies (Barber, Candès, Ramdas & Tibshirani, *Ann. Statist.* 51(2), 2023 —
"Conformal Prediction Beyond Exchangeability"). The inflation factor
`1 + κ√PSI` capped at `drift_cap` is an engineering safeguard, marked as such in
the code. It is not a coverage guarantee and must not be quoted as one.

---

## Package layout

```
calibra/
  types.py            Action ladder, CostMatrix, PredictiveDistribution
  sim/                High-tech line simulator with real pathologies
  data/
    split.py          Leakage-audited chunked splits + independent audit
    features.py       Recipe-aware features, monotone health index
  models/ensemble.py  Bootstrap ensemble, aleatoric/epistemic split
  uq/conformal.py     Split + Mondrian conformal, drift-aware inflation
  decision/
    costs.py          Cost matrices derived from fab / machining economics
    policy.py         Expected-cost minimisation, abstention, CRC
  assure/drift.py     PSI/CUSUM monitor with null-calibrated thresholds
  evidence/card.py    Evidence cards: attribution, analog, cost table
  metrics/score.py    CRPS, Winkler, PIT, ECA, Trust-Adjusted Recall
  pipeline.py         CalibraRunner — the six stages wired together
```

## Two metrics introduced here

Both exist because the standard ones cannot express the failure that actually
ends PdM programmes.

**Expected Cost of Alarm (ECA)** — the currency spent per decision point
servicing *false* alarms. Reported instead of FPR because FPR is dimensionless
and cannot be traded against anything: a shop with 5-minute metrology cycles
and one with 5-hour cycles can share an FPR and have completely different
economics.

**Trust-Adjusted Recall (TAR)** — `recall × measured alert-response-rate`. The
response rate is an *input from the plant* (CMMS work-order closure), not a model
output. This is the only way to expose that recall 95% at a 40% response rate is
worse than recall 80% at a 95% response rate. `n_caught_but_ignored` counts
critical events that were raised and then discarded — invisible to conventional
recall, and the actual mechanism of alert fatigue.

---

## License

MIT. See `LICENSE`.