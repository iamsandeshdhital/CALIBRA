# NEED.md — Requirements, and the papers that establish them

The method for this project: no requirement is included unless it is traceable to
a documented failure in the literature, a standards/roadmap statement from a
national metrology institute, or a reproduced failure observed in this codebase.
Where a requirement was derived from a bug found during development, that is
stated explicitly and the bug is documented — those turned out to be the most
load-bearing requirements of all.

---

## R1 — Evaluate expected cost, not point accuracy

**Requirement.** The primary metric must be the expected monetary cost of the
policy, with the cost ratio stated explicitly. RMSE may be reported but must not
be the objective.

**Evidence.**

- **Elkan, C.** "The Foundations of Cost-Sensitive Learning." *IJCAI 2001*
  (UC San Diego). The optimal decision rule is a function of the cost ratio; with
  asymmetric costs, a fixed threshold is wrong by construction.
  <https://cseweb.ucsd.edu/~elkan/rescale.pdf>
- **High-tech asymmetry is structural, not incidental.** In a fab, a false call
  consumes a mask shot and a metrology slot; a missed catastrophic defect scraps
  a lot plus customer DPPM penalty. Ratios of 10³–10⁵ are routine. In battery
  lines, a missed early-failure cell is a field thermal-runaway risk — a recall —
  while a false call costs a re-grade cycle.

**Observed in this codebase.** The first cost matrix priced `INSPECT` as though it
repaired the machine. It then dominated every state, and the policy degenerated
to "always inspect" — arithmetically optimal and operationally useless. See
`docs/DESIGN.md` §5.

---

## R2 — The system must be able to stay quiet

**Requirement.** Evaluate alarm *rate* and false-alarm cost explicitly, and treat
recall without an alarm budget as an incomplete claim.

**Evidence.**

- Run-to-failure benchmarks (NASA C-MAPSS, PHM-2010, IMS bearing) contain **only**
  failures. Every trajectory ends in one. A model trained and evaluated on these
  never learns restraint, because there is no "healthy, keep running" class to
  learn. Real lines are the opposite: failures are 1–3% of samples.
- **Alarm fatigue is documented as the mechanism that ends PdM programmes.**
  Healthy programmes report alert response >85% and false-positive rate <20%, and
  even those are quoted as two separate numbers:
  <https://oxmaint.com/article/predictive-maintenance-alert-fatigue>,
  <https://ifactoryapp.com/industries/manufacturing-plant/alert-fatigue-predictive-maintenance-solution>
- A model that is never wrong, because it never abstains, produces negative
  expected value: each false alarm burns engineer attention and metrology
  capacity, and the real failure is then missed because the team has learned to
  discount the channel.

**Observed in this codebase.** The constant-median baseline achieves recall 1.00
and TAR 0.85 — identical to CALIBRA — while raising an alarm on **100%** of
decision points and generating 1 793 false alarms against CALIBRA's 83. Identical
headline recall, 21× the economic cost.

---

## R3 — Intervals must carry a distribution-free guarantee

**Requirement.** Emit a calibrated interval with a stated finite-sample guarantee,
not a point estimate and not an unvalidated quantile head.

**Evidence.**

- **Angelopoulos, A.N. & Bates, S.** "Conformal Prediction: A Gentle
  Introduction." *Foundations and Trends in Machine Learning* 16(4):494–591, 2023
  (CMU / Harvard). Split conformal turns calibration into a counting problem with
  a distribution-free coverage guarantee.
  <https://dl.acm.org/doi/10.1561/2200000101>
- **Vovk, V., Gammerman, A. & Shafer, G.** *Algorithmic Learning in a Random
  World* (Springer, 2005) — the original formulation.
- **Romano, Y., Patterson, E. & Candès, E.** "Differentiable Conformal
  Prediction." *NeurIPS 2019* — the normalised/localised variant used here.
- **Why it matters operationally.** Barber, Candès, Ramdas & Tibshirani, "Conformal
  Prediction Beyond Exchangeability," *Ann. Statist.* 51(2), 2023, establishes that
  the guarantee rests on exchangeability — a property production data does not
  have. See R8 and limitation N3.

---

## R4 — Coverage must be conditional where it matters

**Requirement.** Report group-conditional coverage, not only marginal. Enforce a
minimum cell size so that finer conditioning cannot silently degrade.

**Evidence.**

- **Mondrian / group-conditional conformal prediction** (Vovk's taxonomy): the
  taxonomy choice is the primary lever on conditional validity, and marginal
  coverage can be satisfied while specific groups are uncovered.
- **Shamim, M.M. et al.** "Leakage-Robust Evaluation and Data-Scale Sensitivity
  of Attention-Enhanced Multi-Task Learning for Joint Fault Diagnosis and
  Remaining Useful Life Estimation." arXiv:2607.16493, 2026. Also documents that
  task-label *provenance* — measured vs derived proxy — predicts stability under
  data scarcity. Class labels derived from position in a run-to-failure
  trajectory are proxies, and should be treated as such.

**Negative result N1 (measured here).** A three-cell taxonomy was *worse*:

| Taxonomy | critical-cell coverage | mean cost |
|---|---|---|
| 2-cell | **0.977** | **391.0** |
| 3-cell | 0.372 | 524.8 |

The critical cell had 22 calibration points, below the 26 needed to certify its
own radius at α = 0.10, so it silently inherited the global radius. Reproduce with
`python experiments/diag_coverage.py`.

---

## R5 — Separate aleatoric from epistemic uncertainty

**Requirement.** Report the two components separately, and use them for different
purposes.

**Evidence.**

- **Kendall, A. & Gal, Y.** "What Uncertainties Do We Need in Bayesian Deep
  Learning for Computer Vision?" *NeurIPS 2017* (Cambridge / Toronto).
  <https://arxiv.org/abs/1703.04977>
- Aleatoric = irreducible noise; epistemic = what the model does not know. They
  imply different responses: plan conservatively vs retrain / gather data. A
  blended sigma cannot drive both.

---

## R6 — Abstain rather than guess

**Requirement.** Every prediction carries a self-assessment
(`TRUSTED` / `WIDE` / `ABSTAIN`), and a physical action is gated on it.

**Evidence.**

- **Geifman, Y. & El-Yaniv, R.** "SelectiveNet: A Deep Neural Network with an
  Integrated Reject Option." *ICML 2019* (Technion). Selective classification
  optimises prediction and coverage jointly.
  <https://arxiv.org/pdf/1901.09192>
- Prior work by the same authors establishes the coverage–risk frontier: for a
  fixed coverage, a minimised risk.

---

## R7 — Ship evidence, not just a number

**Requirement.** Every alert carries local attribution, a cost table, and a
self-assessment.

**Evidence.**

- **"Explainable remaining useful life uncertainty prediction method for
  degradation estimation."** *Engineering Applications of Artificial Intelligence*,
  2025. Combines uncertainty quantification with model explanation — the two are
  treated as needing to be delivered together, not separately.
  <https://www.sciencedirect.com/science/article/abs/pii/S0952197625021220>
- **Kendall & Gal (2017)** and **Geifman & El-Yaniv (2019)**, as above.
- Practical reading: an operator cannot adjudicate an unexplained number, so they
  either over-trust it (once burned) or ignore it (permanently). Both destroy ROI.

---

## R8 — Detect silent drift and expire the guarantee honestly

**Requirement.** Monitor inputs **and** residuals, with null-calibrated
thresholds, and mark the calibration invalid when drift exceeds tolerance.

**Evidence.**

- **Jourdan, N. et al.** "Addressing Concept Drift in Machine Learning-Based
  Predictive Maintenance." TU Darmstadt, 2025. Concept drift causes performance
  degradation that is "not obvious to machine or plant operators" — precisely the
  failure that converts a good model into an alarm-fatigue generator.
  <https://tuprints.ulb.tu-darmstadt.de/entities/publication/6ae533a3-cd99-449d-878f-e4e25ed89ba7>
- **Bifet, A. & Gavaldà, R.** "Learning from Time-Changing Data with Adaptive
  Windowing." *SDM 2007* — ADWIN, with an explicit false-alarm bound.
  <https://epubs.siam.org/doi/10.1137/1.9781611972771.42>
- **Barber, Candès, Ramdas & Tibshirani**, *Ann. Statist.* 51(2), 2023 — why
  exchangeability fails in production and what can still be said.
- **PSI null value.** With `k` bins, E[PSI] between two i.i.d. samples of the same
  distribution ≈ (k−1)/2 ≈ 3.5 at k=8. Standard thresholds assume far larger
  samples. **Observed here:** a conventionally-tuned monitor fired on **99.6%** of
  healthy decision points. See negative result N2.

---

## R9 — The cost budget must be guaranteed, not tuned

**Requirement.** Bound expected cost with a finite-sample statement; refuse to
certify when data is insufficient.

**Evidence.**

- **Angelopoulos, A.N., Bates, S., Fisch, A., Lei, L. & Schuster, T.** "Conformal
  Risk Control." *ICLR 2024* (Berkeley / Stanford / MIT / Google Research).
- Tuning a threshold on validation and then reporting the validation number is not
  a guarantee — it is the standard error of a hyperparameter sweep. CRC bounds a
  **cost** functional and, because it operates on the model's own predicted
  distribution, works with **zero failure labels**.

---

## R10 — Evaluation must be leakage-audited

**Requirement.** Split by contiguous chunks before windowing, and verify the
property independently.

**Evidence.**

- **Shamim, M.M. et al.** arXiv:2607.16493, 2026. Window-level random splitting
  under small stride moved classification accuracy between ~20–60% (genuine) and
  **99.9%** (artefactual) on identical data and architecture; row-level random
  splitting collapsed it to 0% via degenerate class representation.
  <https://arxiv.org/html/2607.16493v1>
- **Kapoor & Narayanan (2023)** and **Sasse et al. (2023)** on leakage in
  sequential/ML pipelines generally, as cited in that work.
- Related shortcoming documented there: single random splits give unreliable
  estimates on small samples, and unreported seed variance inflates effect sizes.
  CALIBRA therefore reports multi-seed means with bootstrap intervals.

---

## R11 — Multi-product operation must be handled explicitly

**Requirement.** Condition normalisation on process recipe, estimated on training
data only.

**Evidence.**

- High-tech lines are **multi-product**: fabs run dozens of recipes, battery lines
  switch cell formats, machining cells mix part families. The same channel means
  different things at each.
- NIST's 2026 roadmap identifies human-AI teaming, method integration, and
  industrial adoption as open problems rather than solved ones:
  <https://www.nist.gov/publications/2026-roadmap-artificial-intelligence-and-machine-learning-smart-manufacturing>
- **Observed here:** per-asset random drift coefficients, drawn as if each tool
  aged independently, produced offsets larger than the degradation signal and
  pinned PSI at ~9. Tool ageing is a property of the *population*; treating it as
  per-asset is a modelling error with a large downstream cost.

---

## Industry context (not a requirement, but the target)

Deployed industrial products in this space ship decisions, not RUL numbers:

- **Siemens Senseye** — condition-based maintenance, behavioural models.
  <https://www.siemens.com/en-us/products/industrial-digitalization-services/senseye-predictive-maintenance/>
- **Rolls-Royce IntelligentEngine** — engine digital twins for maintenance.
  <https://www.rolls-royce.com/innovation/digital/digital-twin.aspx>
- **Bosch / GE-class** deployments are negotiated on throughput and avoided
  downtime, not on RMSE.

This is corroboration that the decision layer is the product. CALIBRA's
contribution is to make that layer *measurable* — coverage, conditional validity,
cost, and alert economics reported together, with an honest statement of when it
does not know.