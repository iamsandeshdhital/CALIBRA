# CALIBRA — Design

How each stage works, and why it is built that way. Every design choice below
traces to a documented failure mode; the citations are collected in
`../README.md`.

---

## 1. Data and the leakage-audited split

### The problem

Sliding windows over a sensor trace overlap: window `i` spans rows `[i, i+w)` and
window `i+1` spans `[i+1, i+1+w)`, sharing `w-1` of `w` rows. Splitting *windows*
at random therefore places near-duplicates on both sides of the boundary.

Shamim et al. (arXiv:2607.16493, 2026) measured the damage on C-MAPSS, IMS
bearing and a hydraulic rig, holding architecture and raw data fixed:

| Splitting protocol | Classification accuracy |
|---|---|
| Chunk-based (correct) | ~20–60% (genuine) |
| Window-level random, stride 1 | **99.9% (artefactual)** |
| Degenerate class representation | **0%** |

Both extremes are wrong in opposite directions, and neither is visible without
an audit.

### The fix

`data/split.py` orders the operations:

1. Partition the raw, time-ordered sequence into **contiguous, non-overlapping
   chunks**.
2. Assign **whole chunks** to train / val / test.
3. Generate windows **only within** a chunk.

Cross-split raw-row overlap is then zero *by construction*.

`assert_no_leakage()` is a separate function that re-checks the property from the
emitted splits, rather than trusting the construction. `CalibraConfig.asset_wise`
defaults to `True`, which allocates whole **assets** — strictly harder than
temporal chunking, because the model must generalise to equipment it has never
seen. It is the default because it is the only setting that answers the question
a plant asks, even though it costs accuracy.

---

## 2. Features

`data/features.py` produces, per channel: rolling **mean, std, slope, RMS** over a
10-sample window, plus two derived signals.

### Health index

Rolling statistics are collapsed to a scalar health in `(0, 1]` where 1 = as-new:

```
h[t] = 1 / (1 + RMS_j( z_j(t) ))
z_j  = (mean_j(t) − ref_j) / scale_j
```

aggregated with **RMS rather than mean** across channels. RMS means one channel
moving hard — which is the actual early-warning signature — is not diluted by
seven healthy channels.

The reference `ref_j` is the asset's own first 5% of samples, so the index is
partially invariant to the absolute scale of a channel.

### Recipe-aware residuals

High-tech lines run many recipes, and the same channel means different things at
each: a roughing recipe runs hotter and softer than a finishing recipe. Features
are therefore centred on a **per-recipe reference estimated on the training split
only**, plus `recipe_resid_max` = the largest per-channel deviation from that
recipe's reference. This is the single highest-leverage feature decision for
multi-product lines and it is essentially absent from published PdM pipelines.

---

## 3. Uncertainty: two sources, kept separate

Following Kendall & Gal (NeurIPS 2017), total predictive variance is decomposed:

```
Var_total  =  E[ Var_within_member ]  +  Var[ member_mean ]
               └──── aleatoric ────┘      └── epistemic ──┘
```

- **Aleatoric** — irreducible sensor noise at this operating point. Estimated
  from the within-ensemble spread of residuals around the ensemble mean.
- **Epistemic** — what the model does not know because it saw too few examples
  here. Estimated as the variance of member means, with members built by
  **bootstrap resampling** so the disagreement is attributable to *data*, not to
  optimiser seed luck.

The split is not cosmetic. Aleatoric width tells a planner "the lead time is
genuinely uncertain, plan conservatively". Epistemic width tells reliability
engineering "this is a cold spot, retrain". A single blended sigma cannot make
that distinction, so the drift controller cannot act on it.

Members are bootstrap `ExtraTreesRegressor`s rather than a deep network, for
three reasons: it trains in seconds on a laptop; bootstrap resampling gives an
honest epistemic estimate; and it removes any temptation to attribute the
results to scale rather than to calibration.

---

## 4. Conformal calibration

### Why conformal

A quantile-regression head has no coverage guarantee and no way to know when its
interval is wrong. Split conformal (Vovk et al.; surveyed by Angelopoulos &
Bates, *Found. Trends in ML* 16(4), 2023) turns calibration into a counting
problem:

```
q̂ = the ceil((n+1)(1−α))/n empirical quantile of the nonconformity scores
P( y ∈ [f(x) − q̂, f(x) + q̂] ) ≥ 1 − α
```

Distribution-free. In a plant — where the model *will* be misspecified and the
validation set *will* be small — that property is worth more than a few points
of RMSE.

### Three refinements

**Locally adaptive conformity.** Scores are `|y − μ| / σ_pred` rather than
`|y − μ|`. This yields narrow intervals where the model is confident and wide
ones where it is not, instead of a single band that is simultaneously too tight
and too loose. σ is floored so a pathologically confident model cannot produce an
infinite score.

**Mondrian / group-conditional calibration.** Radius is fitted per degradation
group. Plain marginal coverage lets a model be 100% covered on healthy assets
and 0% on critical ones while reporting 90% overall.

**Drift-aware inflation.** See §6.

### Refusing to certify

`_finite_quantile` uses the `ceil((n+1)(1−α))/n` order statistic rather than
`np.quantile`, because linear interpolation silently under-estimates the radius
at small `n` — precisely when we cannot afford to. When `n` is too small to
certify `1−α`, it returns `inf` rather than a plausible-looking number, and that
propagates to the interval.

This is not defensive decoration. It is the mechanism behind negative result **N1**
in the README: a three-cell Mondrian taxonomy had 22 points in its critical cell,
below the threshold, so it inherited the global radius and its conditional
coverage collapsed from 0.977 to 0.372. `min_group_n` prevents this, and
`audit_groups()` makes any fallback visible.

`min_group_size()` solves `ceil((n+1)·(1−α)) ≤ n` by **direct integer search**,
not by the algebraic rearrangement `level/(1−level)` — which is
`9.000000000000002` in binary floating point at α = 0.10 and would round the
answer up by one. Since the guarantee rests on an exact integer rank, it is
computed in integers.

---

## 5. The decision layer

### From calibrated interval to regime probabilities

CALIBRA never asks the model for a class probability. The conformal set is
crossed with the plant's degradation geometry:

```
P(critical) = clip((critical_rul − lo) / (hi − lo), 0, 1)
P(degraded) = clip((degraded_rul  − lo) / (hi − lo), 0, 1) − P(critical)
```

under a **uniform density over the conformal set**. Uniform is the honest
default: conformal gives us a *set*, not a density, and pretending otherwise
would smuggle back exactly the over-confidence being removed.

The practical consequence is that **calibration propagates into the cost
numbers**, which is what makes them auditable.

Note `max_rul` is a *cap*, not a floor. An earlier version applied
`maximum(upper, max_rul)`, which pushed every upper bound to the horizon and made
a certain-critical interval `[0, 10]` look 99% healthy. The direction of a bound is
load-bearing.

### Expected-cost minimisation

```
a* = argmin_a  E[ cost(a, regime) ] = argmin_a  P(regime) · c_a
```

replacing threshold-on-probability. Per Elkan (IJCAI 2001), the optimal rule
depends on the cost ratio, so a fixed 0.5 threshold is *wrong by construction*
whenever costs are asymmetric.

### Abstention gate

```
TRUSTED  if width ≤ max_actionable_width and coverage ≥ min_coverage  → argmin over all rungs
WIDE     otherwise                                            → capped at INSPECT
ABSTAIN  width > 3 × max_actionable_width                     → capped at WATCH
```

Note a confident "this tool is fine" produces `RUN` — a model suppressing noise
is doing useful work, and that is what halts alert fatigue.

### Conformal Risk Control

Tuning a threshold on the validation set and then reporting a validation number is
not a guarantee. CRC (Angelopoulos, Bates, Fisch, Lei & Schuster, ICLR 2024)
bounds a **cost** functional with a finite-sample guarantee. Because it controls
a functional of the model's own predicted distribution, it works with **zero
failure labels** — the situation that defeats supervised threshold tuning.

It refuses to certify when `n < min_calibration`, and says so.

### Cost matrices

Derived from fab economics rather than asserted, and parameterised so they can be
argued with:

```
c_inspect = metrology_minutes/60 × (engineer_hourly × 2)
c_recover = chamber_qual_hours × chamber_rate + 4 × engineer_hourly
c_stop    = line_stop_hourly × 1.5 / 24
c_escape  = penalty_per_defective_part × escape_ppm / 1e6
```

Three structural rules, each of which was violated at least once during
development and is now asserted in tests:

1. **INSPECT is diagnostic, not preventive.** It *identifies* the condition; it
   does not restore the tool. Pricing it as though it repairs the machine made it
   dominate every state, and the policy degenerated to "always inspect" —
   arithmetically optimal, operationally useless.
2. **`cost(action | critical) ≥ cost(action | healthy)`** for every action except
   those that genuinely fix the machine. A version with
   `RECOVER|critical < RECOVER|healthy` made breaking a tool *cheaper* to fix
   than disturbing a healthy one, giving a **negative** cost ratio.
3. **`ordering_is_sane()`**: `c_inspect(healthy) < c_RUN(critical)`. If detecting
   a defect costs more than letting it escape, every policy collapses to RUN.

---

## 6. Assurance

### PSI has a non-zero null

With `k` bins, PSI between two i.i.d. samples of the **same** distribution has
expectation ≈ `(k−1)/2` — about **3.5** for 8 bins. The familiar thresholds
(< 0.10 stable, 0.10–0.25 moderate, > 0.25 significant) implicitly assume samples
thousands of times larger than the bin count.

On a 200-sample rolling window the null value is an order of magnitude above
those thresholds. A conventionally-tuned monitor therefore fires permanently: the
first version reported drift on **99.6%** of healthy decision points.

CALIBRA measures the null by **splitting its own reference in half** and
thresholding the *excess*. Escalation on stationary data is now 0%.

### Aggregate as quantile + single-channel max

`0.5·max + quantile_0.95` of per-channel excess. The quantile captures broad
drift and is stable; a separate single-channel max catches the operationally
important case of **one** thermistor replaced or one encoder swapped, which a
quantile over ~34 channels would hide. A pure `max` is unstable (multiple
comparisons) and flickers between channels on noise.

### Residuals, not just inputs

Standard detectors watch `P(x)`. The dangerous case is a shift in `P(y|x)` that
leaves `P(x)` looking ordinary. So the monitor tracks **standardised residuals**
too. Rising residual scale with flat input PSI is the signature of a genuine
concept change, and it triggers *retraining* rather than recalibration.

`update()` accepts `residual=None` for the case where no prediction exists yet.
Feeding placeholder zeros against a reference of real residuals produces a large
spurious PSI and a false "concept shift" alarm — which is exactly what happened
before this was fixed.

### Dwell

Consecutive breaching windows are required before escalation. A system that
retrains on every blip is itself an alert-fatigue source, just for the platform
team.

---

## 7. Evidence cards

An operator who receives "RUL = 41" and nothing else has no way to judge it, so
they either over-trust it (and get burned once) or under-trust it (and ignore it
always). Both are worse than having no model. Each card triangulates:

1. **A calibrated interval**, not a point — "41 ± 60" tells the planner whether
   the maintenance window fits before the next scheduled stop.
2. **Local attribution** — which channels drove *this* call, by permutation
   against the local neighbourhood median. Cheaper and more explainable than
   SHAP for an evidence card a human will read.
3. **Expected cost per rung** — so the operator can disagree on grounds the
   system has made explicit.
4. **Self-assessment** — is the model calibrated and in-control *right now*? A
   drifted model should not be presenting a confident number.
5. **Calibration provenance** — how many points, at what level.

---

## 8. The simulator's physics

```
x[t] = recipe_offset[recipe[t]] + health(t) ⊙ sig_weight
        + shared_ageing(t) + step_random_walk(t) + sensor_fault(t) + noise
```

- `health(t)` is a **monotone** power-law wear law, built by interpolation over
  control points. An earlier version smoothed with
  `np.convolve(..., mode='same')` using a kernel as long as the signal — which is a
  half-signal shift that silently destroyed the degradation geometry and
  inflated the critical label fraction to 43%.
- **Right-censoring.** Healthy assets get `RUL = rul_cap`, not a fabricated value.
  The earlier version used `T + max_len`, which depends on trajectory length — an
  **unobservable** variable. That is irreducible label noise, and because a
  held-out asset draws a different value than the calibration set, it voids any
  coverage guarantee. This was the actual cause of a measured 0.595 coverage
  against a 0.90 target.
- **Shared, not per-asset, ageing.** Tool ageing is a property of the tool
  population. Drawing the coefficient per asset, integrated over ~400 samples,
  produced an offset larger than the entire degradation signal, destroyed the
  learnable signal, and pinned PSI at ~9 so the monitor was permanently at
  "retrain".
- The observable `health` signal is derived from the **capped** wear law so the
  sensor trace and the label agree; using the uncapped RUL would let the horizon
  leak into the sensors and the model would learn trajectory length instead of
  degradation.

---

## Known limitations

Stated plainly, because the negative results section only covers what was measured.

1. **The accuracy claims rest on the simulator, not a fab.** Real adapters exist
   in `data/` but the published numbers are from the built-in line. Simulation
   fidelity is a modelling assumption, not a result.
2. **The cost matrices are illustrative.** They are internally coherent and
   ordering-checked, but a real deployment needs the plant's own numbers. The
   policy adapts to whatever matrix it is given — that is the testable claim, not
   the specific EUR values.
3. **Drift inflation is a heuristic bound**, not a guarantee. Marked as such.
4. **CRC is implemented but not exercised end-to-end** in the committed run; it
   correctly refuses to certify at the sizes available here.
5. **Sensor faults are injected, not diagnosed.** CALIBRA is made robust to them
   (they appear as epistemic width and drive abstention) but does not identify
   which sensor is faulty. That is a natural next module.
6. **`asset_wise=True` splits by asset**, so per-asset calibration groups are
   thin by construction. A plant with many more assets would do better here.