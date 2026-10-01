"""Synthetic high-tech production line with *deliberately realistic* pathologies.

Why a simulator at all when real PHM datasets exist? Because the four
failure modes CALIBRA targets are *by construction* absent from standard
benchmarks:

  1. Extreme label imbalance       -- C-MAPSS/PHM2010 are run-to-failure
     trajectories where *every* trajectory ends in failure. A model that
     never sees a healthy asset therefore never learns to stay quiet, which
     is exactly how alert fatigue is manufactured. Real lines are the
     opposite: failures are rare (0.5-3% of samples), and 97%+ of the data
     is "keep running".
  2. Sensor faults that mimic failures -- a drifting thermistor or a stuck
     encoder produces a signal statistically indistinguishable from a real
     degradation. Every published benchmark assumes perfect sensors.
  3. Multi-product regime change   -- high-tech lines (fab, battery, CNC
     cell, photonics) run dozens of recipes. The same sensor means
     different things at different recipes, so a single global model drifts
     by construction when the product mix changes.
  4. Non-stationary tool ageing     -- chamber condition, filter loading and
     calibration drift slowly and then step, invalidating a model trained on
     last quarter's data.

``HighTechLine`` reproduces all four so that the *assurance* layer can be
measured rather than asserted. Real-dataset loaders live in
``calibra.data`` and are used for the accuracy claims; the simulator is used
for the trust/calibration/drift claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..types import REGIME_ORDER, Regime, RunToFailure

DEFAULT_CHANNELS: tuple[str, ...] = (
    "spindle_load",       # kW, rises with bearing/bearing-preload loss
    "spindle_vib_rms",    # mm/s, the classic early-wear channel
    "spindle_temp",       # degC
    "coolant_flow",       # L/min, drops as filter clogs
    "part_dia_err",       # micrometres, in-line gauge -> cutting wear
    "part_surface_rough", # Ra, in-line gauge -> coolant degradation
    "chamber_pressure",   # mbar, drifts with tool seasoning
    "axis_position_err",  # um, servo/backlash
)


@dataclass(frozen=True)
class LineConfig:
    """Every knob that makes the synthetic line more or less hostile."""

    n_assets: int = 26
    min_len: int = 180
    max_len: int = 420

    n_recipes: int = 4
    base_rate_hz: float = 1.0  # samples per simulated second
    noise_scale: float = 0.012

    # --- degradation physics -------------------------------------------------
    critical_rul: int = 25  # RUL at which an un-flagged asset goes critical
    degraded_rul: int = 90  # RUL at which measurable drift appears
    #: RUL horizon. A healthy asset is RIGHT-CENSORED at this value: its true
    #: RUL is ">= horizon", not a number anyone can observe. Fabricating a
    #: pseudo-RUL for healthy assets (e.g. trajectory_length + horizon) injects
    #: label noise that depends on trajectory length -- a variable the sensors
    #: cannot observe. That noise is irreducible, and because a held-out asset
    #: gets a different draw than the calibration set, it silently voids any
    #: coverage guarantee. Capping is the honest encoding, and it is what
    #: C-MAPSS (cap 125) and PHM-2010 (piecewise cap) already do.
    rul_cap: int = 145
    #: Fraction of assets that actually degrade at all. Mirrors reality: most
    #: tools are healthy for the whole quarter.
    degrade_fraction: float = 0.55

    # --- pathology knobs ----------------------------------------------------
    sensor_fault_rate: float = 0.30  # P(asset has >=1 faulty sensor)
    sensor_fault_frac: float = 0.06  # fraction of trajectory affected
    recipe_change_prob: float = 0.02  # per-sample hazard of a product change
    global_drift_rate: float = 0.010  # slow tool-ageing drift per sample
    step_shift_prob: float = 0.004  # abrupt regime shift hazard
    step_shift_scale: float = 0.055

    critical_fraction: float = 0.030  # P(label == critical) target
    seed: int = 20260201


class HighTechLine:
    """Generates :class:`RunToFailure` trajectories for a high-tech cell.

    Model sketch per asset
    ---------------------
    ``x[t] = recipe_offset[recipe[t]] + tool_age_drift[t] + health_signal[t]*load
             + sensor_fault[t] + noise``

    * ``health_signal`` is a monotone non-increasing latent degradation trace
      built from a smooth random "wear law", so the physics is monotone (a
      real degradation cannot un-degrade without intervention) even though
      the observable channels are not.
    * ``sensor_fault`` is a per-channel multiplicative gain error plus a
      frozen (stuck-at) segment. This is the false-alarm generator.
    * ``tool_age_drift`` is slow linear drift plus Poisson jumps.
    * ``recipe_offset`` is a per-(recipe, channel) offset -- this is the
      non-stationarity that breaks single-recipe models.
    """

    def __init__(self, config: LineConfig | None = None) -> None:
        self.cfg = config or LineConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.feature_names: tuple[str, ...] = DEFAULT_CHANNELS
        d = len(DEFAULT_CHANNELS)
        # Shared, population-level non-stationarity. Drawn once per line so
        # every asset experiences the same ageing trend -- see the comment at
        # the point of use for why this must not be per-asset.
        self._ageing_ = 0.35 + 0.5 * np.abs(self._rng.normal(0.0, 1.0, size=d))
        self._jump_mix_ = self._rng.normal(0.0, 0.3, size=d)
        self._build_recipe_table()

    # ------------------------------------------------------------------ setup
    def _build_recipe_table(self) -> None:
        cfg = self.cfg
        # Offsets are physical: a roughing recipe runs hotter and softer than
        # a finishing recipe, so load/temp/vibration all sit differently.
        profile = np.stack(
            [
                self._rng.normal(0.0, 0.05, size=len(DEFAULT_CHANNELS))
                for _ in range(cfg.n_recipes)
            ]
        )
        # Force a meaningful spread between recipes.
        profile += np.linspace(-0.09, 0.09, cfg.n_recipes)[:, None]
        self._recipe_table = profile

    # ------------------------------------------------------------- generation
    def generate(self, n_assets: int | None = None) -> list[RunToFailure]:
        cfg = self.cfg
        n = int(n_assets if n_assets is not None else cfg.n_assets)
        out: list[RunToFailure] = []
        for i in range(n):
            out.append(self._one_asset(f"ASSET-{i:03d}"))
        return out

    def _one_asset(self, asset_id: str) -> RunToFailure:
        cfg, rng = self.cfg, self._rng
        d = len(DEFAULT_CHANNELS)
        T = int(rng.integers(cfg.min_len, cfg.max_len + 1))

        # ---- recipes: piecewise-constant with a change hazard
        recipe = np.zeros(T, dtype=np.int64)
        for t in range(1, T):
            recipe[t] = (
                int(rng.integers(0, cfg.n_recipes))
                if rng.random() < cfg.recipe_change_prob
                else recipe[t - 1]
            )

        # ---- latent health: monotone wear law h[t] in [0, 1]
        degraded = bool(rng.random() < cfg.degrade_fraction)
        if degraded:
            # Power-law wear: slow for a long time, then accelerates. Built by
            # monotone interpolation over control points rather than by
            # convolution -- 'same'-mode convolution with a kernel as long as
            # the signal is a half-signal shift, which silently destroys the
            # degradation geometry and inflates the critical-label fraction.
            u = rng.uniform(1.6, 3.6)
            n_knots = 12
            knots = np.linspace(0.0, 1.0, n_knots)
            vals = knots**u  # strictly increasing
            progress = np.interp(np.linspace(0.0, 1.0, T), knots, vals)
            progress = progress / max(progress[-1], 1e-9)
            progress = np.clip(progress, 0.0, 1.0)
            rul = ((1.0 - progress) * T).astype(float)
            # Enforce monotone decay (a real asset does not un-degrade on its own).
            rul = np.minimum.accumulate(rul)
            rul = np.maximum(rul, 0.0)
        else:
            # Healthy asset: right-censored. Its true remaining life is only
            # known to be beyond the observation horizon, so we record the
            # horizon and let the label say ">= cap". This is the class the
            # "always alarm" models get wrong.
            rul = float(cfg.rul_cap) * np.ones(T)

        # Cap the label everywhere: an asset observed beyond the horizon is
        # also censored, not given an extrapolated number.
        rul = np.minimum(rul, float(cfg.rul_cap))

        if degraded:
            # Derive the observable degradation directly from the wear law, so
            # the sensor trace and the capped label are consistent by
            # construction. Using the un-censored RUL here would let the
            # uncapped horizon leak into the sensors and the model would learn
            # trajectory length instead of degradation.
            health = np.clip(progress, 0.0, 1.0)
        else:
            # A healthy tool still wanders; it is not a flat line.
            health = np.clip(
                0.03 + 0.05 * np.abs(rng.normal(0.0, 1.0, size=T)), 0.0, 0.25
            )

        # ---- clean, physically consistent degradation signature
        # Each channel gets a load exponent: vibration is early, diameter
        # error is late, coolant is only affected late.
        sig = np.stack(
            [
                health ** 1.6 + 0.35 * health,   # spindle_load
                health ** 1.1 + 0.20 * health,   # vib_rms (early warning)
                health ** 2.2,                   # temp (thermal lag)
                -0.9 * health ** 2.6,            # coolant flow (late)
                health ** 3.0,                   # diameter error (late)
                health ** 2.4,                   # roughness
                -0.7 * health ** 1.9,            # chamber pressure
                0.5 * health ** 2.8,            # axis position error
            ],
            axis=1,
        )
        gain = self._recipe_table[recipe] * 2.4  # (T, d)
        X = sig * gain + gain.mean(axis=0, keepdims=True) * 0.6

        # ---- slow tool ageing + abrupt steps (global, non-stationarity)
        #
        # The ageing coefficient is *shared* across assets, not redrawn per
        # asset. Tool ageing is a property of the tool population (chamber
        # seasoning, filter loading, calibration schedule), so it drifts
        # coherently. Drawing it independently per asset instead -- with a
        # coefficient of O(1) integrated over O(400) samples -- produces a
        # per-asset offset comparable to or larger than the entire degradation
        # signal, which (a) destroys the learnable signal and (b) makes PSI
        # report ~9 for every held-out asset, so the drift monitor is
        # permanently pinned at "retrain" and carries no information.
        t = np.arange(T)[:, None]
        X += cfg.global_drift_rate * t * self._ageing_
        jump = np.zeros(T)
        for k in range(1, T):
            jump[k] = jump[k - 1] + (
                rng.normal(0.0, cfg.step_shift_scale)
                if rng.random() < cfg.step_shift_prob
                else 0.0
            )
        X += jump[:, None] * self._jump_mix_

        # ---- sensor faults: the false-alarm factory
        faulted = False
        if rng.random() < cfg.sensor_fault_rate:
            faulted = True
            n_fault = int(rng.integers(1, 3))
            n_span = max(4, int(T * cfg.sensor_fault_frac))
            for _ in range(n_fault):
                ch = int(rng.integers(0, d))
                start = int(rng.integers(0, max(1, T - n_span)))
                stop = min(T, start + n_span)
                mode = rng.random()
                if mode < 0.5:
                    # stuck-at / frozen sensor -> flatline, then a step back
                    X[start:stop, ch] = X[start, ch]
                else:
                    X[start:stop, ch] *= 1.0 + rng.choice([-1.0, 1.0]) * rng.uniform(
                        0.25, 0.6
                    )

        # ---- measurement noise
        X += rng.normal(0.0, cfg.noise_scale, size=X.shape)

        # ---- labels
        regime = np.where(
            rul <= cfg.critical_rul,
            Regime.CRITICAL.value,
            np.where(rul <= cfg.degraded_rul, Regime.DEGRADED.value, Regime.HEALTHY.value),
        )
        if degraded:
            # Label noise on the *boundary*: a few samples of ground truth that
            # a perfectly calibrated model genuinely cannot resolve, because
            # the sensor noise floor hides a 4-cycle RUL difference. Kept
            # small and confined to the healthy edge, so the residual noise is
            # a real but bounded irreducible term rather than a licence for
            # unlearnable targets.
            flip = rng.random(T) < 0.04
            regime = np.where(flip & (regime == Regime.HEALTHY.value),
                              Regime.DEGRADED.value, regime)
            # ...and symmetric jitter at the critical edge.
            flip2 = rng.random(T) < 0.03
            regime = np.where(flip2 & (regime == Regime.CRITICAL.value),
                              Regime.DEGRADED.value, regime)
        # Keep RUL consistent with the (jittered) regime label, and monotone.
        rul_lab = np.where(regime == Regime.HEALTHY.value,
                           np.maximum(rul, cfg.degraded_rul + 1.0), rul)
        rul_lab = np.maximum.accumulate(rul_lab[::-1])[::-1]
        rul_lab = np.minimum(rul_lab, float(cfg.rul_cap))

        return RunToFailure(
            asset_id=asset_id,
            X=X.astype(np.float64),
            rul=rul_lab.astype(np.float64),
            regime=regime.astype(object),
            recipe=recipe,
            feature_names=list(DEFAULT_CHANNELS),
            meta={
                "degrades": degraded,
                "t_len": T,
                "rul_cap": cfg.rul_cap,
                "final_rul": float(rul_lab[-1]),
                "sensor_faulted": faulted,
            },
        )

    # --------------------------------------------------------------- helpers
    @staticmethod
    def flatten(runs: Sequence[RunToFailure]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Concatenate assets into flat arrays. Returns (X, rul, regime, asset_id)."""
        X = np.concatenate([r.X for r in runs])
        y = np.concatenate([r.rul for r in runs])
        g = np.concatenate([r.regime for r in runs])
        a = np.concatenate([[r.asset_id] * len(r) for r in runs])
        return X, y, g, a

    def class_balance(self, runs: Sequence[RunToFailure]) -> dict[str, float]:
        _, _, g, _ = self.flatten(runs)
        return {r: float((g == r).mean()) for r in REGIME_ORDER}


def make_default_line(seed: int = 20260201) -> HighTechLine:
    return HighTechLine(LineConfig(seed=seed))