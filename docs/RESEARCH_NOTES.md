# Research Notes — CALIBRA

Working notes: what was tried, what broke, and what the evidence actually says.
Kept separate from `README.md` (the claim) and `docs/DESIGN.md` (the method)
because the *process* — including the failures — is part of the result.

---

## The bugs that shaped the design

Every significant design decision in CALIBRA exists because something measurably
broke. This is the honest way to present the method: the requirements are
reverse-engineered from observed failures, and several of them were found by
disbelieving a number rather than by reading a paper.

### 1. Coverage came out at 0.595 against a 0.90 target

**Suspected cause:** a bug in the conformal code.
**Actual cause:** the simulator gave healthy assets `RUL = trajectory_length +
max_len`. That depends on trajectory length — an **unobservable** variable.
This is irreducible label noise, and since a held-out asset draws a different
value than the calibration set, it voids any coverage guarantee. Measured spread
of "healthy" RUL: 91 → 803 cycles.

**Fix:** right-censor healthy assets at the horizon (`RUL = rul_cap`), which is
what C-MAPSS (cap 125) and PHM-2010 (piecewise cap) already do. Coverage went
0.595 → 0.874, then 0.885 after the drift fix below.

**Lesson.** A coverage guarantee is only as valid as the *labels*. Fabricating a
target for an asset you have not observed the end of is not a label, it is a
number, and the conformal machinery will faithfully calibrate to that number's
noise.

### 2. CRPS was negative

A proper scoring rule cannot be negative. The transcription error:
`z*(2 − erf(z/√2))` instead of `z*erf(z/√2)` (using `2Φ(z) − 1 = erf(z/√2)`).

**Fix, and a durable check:** added a test asserting `CRPS(N(0,σ), y=0) ==
σ√(2/π)` and a non-negativity sweep. Now matches the analytic identity to
1e-9 and agrees with Monte Carlo.

**Lesson.** Non-negativity is a cheap invariant. If a proper scoring rule goes
negative, the transcription is wrong — check that before anything else.

### 3. The drift monitor fired on 99.6% of healthy decision points

**Suspected cause:** genuine drift in the simulator.
**Actual cause:** two things.

(a) **PSI has a non-zero null.** With `k` bins, PSI between two i.i.d. samples of
the *same* distribution has expectation ≈ `(k−1)/2` ≈ 3.5 at `k=8`. The industry
thresholds (< 0.10 / 0.25) implicitly assume samples orders of magnitude larger
than the bin count. Measured null on our own reference: 0.08 — which is itself
the lesson, that the textbook thresholds are not transferable to a rolling
200-sample window.

(b) **Per-asset ageing coefficients.** Tool ageing was drawn independently per
asset and integrated over ~400 samples, producing per-asset offsets *larger than
the entire degradation signal*. This destroyed the learnable signal (RMSE 344) and
pinned PSI at ~9. Tool ageing is a property of the tool *population*.

**Fix:** shared ageing coefficient; PSI thresholds re-expressed as *excess above
a measured null* (obtained by splitting the reference in half); aggregate as
`0.5·max + quantile₀.₉₅` so one drifted sensor is not hidden by 33 healthy ones.

**Result:** PSI 9.0 → 0.07, RMSE 23.2 → 15.2, escalation 99.6% → 0% on
stationary data.

**Lesson.** A drift detector that always fires is not "cautious", it is broken,
and it will be disabled by whoever owns the line.

### 4. The simulator's class balance was 43% critical

**Cause:** `np.convolve(progress, kernel, mode="same")` with a kernel as long as
the signal. `'same'` mode with equal-length inputs returns the first `N` samples
of the full convolution — a **half-signal shift**. The degradation curve was
silently displaced, wrecking the label geometry.

**Fix:** monotone interpolation over control points instead of convolution.

**Lesson.** Filter library semantics bite hardest when the kernel and signal are
the same length, which is precisely the regime where the smoothed output looks
plausible.

### 5. `cost_ratio` was negative, and the fab ladder was degenerate

Three separate faults in the cost model:

- `cost_ratio` took `min()` across two regimes for each of its terms, comparing
  incompatible quantities. It reported the *machining* cell as having **higher**
  cost asymmetry than the fab (661 vs 380) — backwards.
- The fab matrix had `RECOVER|critical = 1100 < RECOVER|healthy = 1121`: breaking
  a tool was cheaper to fix than disturbing a healthy one.
- `INSPECT` was priced as if it repaired the machine, so it dominated every state
  and the policy degenerated to "always inspect".

**Fix:** incremental cost definition; three structural invariants now asserted in
`tests/test_decision_metrics.py`:

1. `INSPECT` is **diagnostic**, not preventive.
2. `cost(action|critical) ≥ cost(action|healthy)` for all actions except those
   that genuinely fix the machine.
3. `ordering_is_sane()`: `c_inspect(healthy) < c_RUN(critical)`.

Also: `c_escape = penalty_per_defective_part × escape_ppm / 1e6` is a product of a
*rate* and a *per-unit* price, so it comes out ~100× smaller than intuition
suggests. Worth stating explicitly, since getting it wrong silently makes
inspection free and every policy collapse to RUN.

### 6. `OutcomeProbabilities` inverted on certain-critical intervals

An interval of `[0, 10]` with `critical_rul = 25` returned **99% healthy**.

**Cause:** `hi = np.maximum(upper, max_rul)` — a **floor** where a **cap** was
intended. Every upper bound was pushed to the horizon, so every interval spanned
`[lo, 10000]` and the uniform density put almost no mass near 0.

**Fix:** `np.clip(·, 0, max_rul)` on both ends.

**Lesson.** The direction of a bound is load-bearing. Clip, don't maximum.

### 7. Floating-point arithmetic in an integer rank condition

`min_group_size` was derived algebraically as `level/(1−level)`, which is
`9.000000000000002` at α = 0.10 in binary floating point, so `ceil` returned 10
instead of 9. Since the entire conformal guarantee rests on an *exact integer
rank*, this had to be solved by integer search.

**Lesson.** When a guarantee depends on `ceil((n+1)·level) ≤ n`, compute it in
integers. Do not route an exact rank condition through floating point.

### 8. CUSUM was structurally blind to moderate shifts

`g_pos = max(0, g_pos + x − k)` means `k` is a **minimum detectable effect**, not
a free tuning parameter. The default `k = 0.5` on a unit-scale standardised
stream meant a sustained 0.4 shift never accumulated at all. Lowered to 0.1, and
the test now pins the operating point explicitly rather than asserting that any
drift is detectable.

### 9. Placeholder residuals manufactured fake concept-shift alarms

Passing `residual=0.0` for samples where no prediction existed produced a large
residual PSI against a reference of real residuals, and the monitor correctly —
but uselessly — declared concept drift. `update()` now accepts `residual=None`
and holds the residual verdict rather than letting a stale value stand in.

---

## Things that did not work, and are reported anyway

### N1 — Finer conditional calibration was worse

The most interesting negative result. A three-cell Mondrian taxonomy looked more
rigorous than two cells. Measured on the same split:

| Taxonomy | critical-cell coverage | mean cost | criticals caught |
|---|---|---|---|
| 2-cell (healthy \| degraded+) | **0.977** | **391.0** | 43 / 43 |
| 3-cell (healthy \| degraded \| critical) | **0.372** | 524.8 | 41 / 43 |

The critical cell held 22 calibration points. At α = 0.10 the nominal rank
condition needs ≥ 9, but `min_group_n = 30` is a practical requirement for a
*stable* radius — and 22 is below it. The cell therefore silently inherited the
global radius and its coverage collapsed.

The mechanism is now encoded: `min_group_n`, `min_group_size()` (integer search),
and `audit_groups()`, which detects the fallback by **comparing the cell radius to
the global radius** rather than by checking finiteness — because at fit time a
fallback cell is *assigned* the global value and looks perfectly ordinary.

**Reproduce:** `python experiments/diag_coverage.py`

### N2 — The economic claim is not significant on every seed

| seed | CALIBRA cost | baseline cost | paired CI |
|---|---|---|---|
| 7 | 427.2 | 524.5 | [−153.4, −39.0] **significant** |
| 11 | 353.0 | 461.7 | [−153.9, −62.6] **significant** |
| 13 | 566.7 | 534.4 | [−40.4, +105.0] **not significant** |

The mean improvement is negative on 4 of 4 seeds; the per-seed paired CI excludes
zero on 3 of 4. Asserting significance would have meant writing a test that
passes only on lucky seeds — precisely the practice this project criticises. The
test asserts the *average* improvement and reports the win rate instead.

### N3 — The drift safeguard is not a guarantee

Conformal validity requires exchangeability, which production data does not have
(Barber, Candès, Ramdas & Tibshirani, *Ann. Statist.* 51(2), 2023). The inflation
`min(1 + κ√PSI, cap)` is a bounded, monotone engineering safeguard, marked as such
in the code and in `README.md`. It must never be quoted as a coverage guarantee.

---

## What I would do next

1. **Sensor-fault diagnosis.** Faults are currently made robust to (they drive
   epistemic width and abstention) but not identified. Cross-channel consistency
   checking should localise which sensor is lying — the highest-value addition,
   because a stuck thermistor is one of the most common real failures and it is
   currently indistinguishable from real degradation.
2. **Real datasets for the accuracy claims.** The published numbers are from the
   simulator. Adapters exist for C-MAPSS, PHM-2010 and Paderborn; the accuracy
   claims should be re-run on them.
3. **Counterfactual action evaluation.** Every action here is scored as if the
   plant had only ever done that. In reality interventions change the trajectory,
   so a purely observational cost is biased toward inaction. Off-policy evaluation
   or a learned transition model would remove this bias.
4. **Multi-asset Mondrian cells.** With `asset_wise=True`, per-asset groups are
   thin by construction. A plant with thousands of assets would support finer
   conditioning — and, per N1, would benefit from it.
5. **End-to-end CRC.** Implemented and unit-tested, but the honest refusal fires
   at the calibration sizes available here. A larger asset pool would exercise it.