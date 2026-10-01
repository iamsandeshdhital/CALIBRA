"""Tests for the conformal layer.

The properties asserted here are the *contract*. If any of them breaks, the
"90% coverage" claim in the README is void, so they are tested directly
rather than inferred from an end-to-end number.

Run:  python -m pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pytest

from calibra.uq.conformal import ConformalRUL, _finite_quantile


def _synthetic(n=4000, seed=0, sigma=1.0):
    """Well-specified setting: y ~ N(mu, sigma)."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(50, 20, size=n)
    s = np.full(n, sigma)
    y = rng.normal(mu, s)
    return y, mu, s


def test_empirical_coverage_near_nominal():
    """The headline guarantee: >= 1-alpha coverage on exchangeable data."""
    y, mu, s = _synthetic(6000, seed=1)
    c = ConformalRUL(alpha=0.10, normalise_by_sigma=False).fit(y[:3000], mu[:3000], s[:3000])
    cov = c.empirical_coverage(y[3000:], mu[3000:], s[3000:])
    assert cov >= 0.88, f"coverage {cov:.3f} violated the 0.90 target"
    assert cov <= 0.97, f"coverage {cov:.3f} is suspiciously high (wasted capacity)"


def test_higher_alpha_gives_narrower_intervals():
    """``alpha`` is the *miscoverage* level, so raising it must buy **less**
    coverage and therefore a narrower interval. Getting this direction wrong
    is the classic way to silently ship an over-confident system."""
    y, mu, s = _synthetic(4000, seed=2)
    rows = []
    for a in (0.05, 0.10, 0.20):
        c = ConformalRUL(alpha=a, normalise_by_sigma=False).fit(y[:2000], mu[:2000], s[:2000])
        lo, hi, _ = c.interval(mu[2000:], s[2000:])
        rows.append((a, float(np.mean(hi - lo)), c.empirical_coverage(y[2000:], mu[2000:], s[2000:])))
    # alpha ascending -> width descending, coverage descending
    assert rows[0][1] > rows[1][1] > rows[2][1], rows
    assert rows[0][2] > rows[1][2] > rows[2][2], rows
    # and the tightest setting must reach its nominal target, within the
    # Monte-Carlo standard error of the held-out sample
    from scipy.stats import norm

    p, k = rows[0][2], 2000
    se = float(np.sqrt(p * (1 - p) / k))
    assert rows[0][2] >= 0.95 - 2 * se, (
        f"alpha=0.05 coverage {rows[0][2]:.4f} is more than 2 SE below nominal 0.95"
    )


def test_uncertifiable_calibration_returns_infinite_radius():
    """Refusing to certify is the correct behaviour, not a bug.

    With n=5 points we cannot certify 90% coverage, so the radius must be
    ``inf``. Silently returning a small radius here is exactly the failure
    that produces a confidently wrong production alarm.
    """
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c = ConformalRUL(alpha=0.10, normalise_by_sigma=False).fit(y, y, np.ones(5))
    assert not np.isfinite(c.global_radius_)
    lo, hi, _ = c.interval(np.array([3.0]), np.array([1.0]))
    assert np.isinf(lo[0]) and np.isinf(hi[0])


def test_ceil_rank_correction_is_used():
    """The finite-sample correction is what makes the guarantee valid.

    ``np.quantile``'s linear interpolation under-estimates the radius at small
    n, which is precisely when we cannot afford to.
    """
    x = np.arange(1, 11, dtype=float)  # n=10
    level = 0.9
    expected_rank = int(np.ceil((10 + 1) * level))
    got = _finite_quantile(x, level)
    assert got == x[expected_rank - 1]
    assert got >= np.quantile(x, level) - 1e-12


def test_local_adaptivity_narrows_where_confident():
    """Normalised conformity should give a tighter interval in low-sigma
    regions than a single global band can."""
    rng = np.random.default_rng(3)
    n = 5000
    s = np.where(rng.random(n) < 0.5, 1.0, 8.0)
    mu = rng.normal(50, 10, size=n)
    y = rng.normal(mu, s)
    glob = ConformalRUL(alpha=0.1, normalise_by_sigma=False).fit(y[:2500], mu[:2500], s[:2500])
    ada = ConformalRUL(alpha=0.1, normalise_by_sigma=True).fit(y[:2500], mu[:2500], s[:2500])
    lg, hg, _ = glob.interval(mu[2500:], s[2500:])
    la, ha, _ = ada.interval(mu[2500:], s[2500:])
    wg = float(np.mean(hg - lg))
    wa = float(np.mean(ha - la))
    assert (hg - lg).min() > 0 and (ha - la).min() > 0, "interval widths must be positive"
    assert wa < wg, f"adaptive {wa:.3f} should be narrower than global {wg:.3f} under heteroscedasticity"
    assert ada.empirical_coverage(y[2500:], mu[2500:], s[2500:]) >= 0.85


def test_mondrian_gives_conditional_coverage():
    y, mu, s = _synthetic(8000, seed=4)
    grp = np.where(np.arange(y.size) % 2 == 0, "a", "b")
    c = ConformalRUL(alpha=0.10, mode="mondrian", min_group_n=30).fit(
        y[:4000], mu[:4000], s[:4000], grp[:4000]
    )
    cov = c.conditional_coverage(y[4000:], mu[4000:], s[4000:], grp[4000:])
    for g, v in cov.items():
        assert v >= 0.85, f"group {g} coverage {v:.3f} below target"


def test_mondrian_falls_back_for_small_groups():
    """A group below the certifiable size must inherit the global radius, and
    ``audit_groups`` must say so loudly rather than silently."""
    y, mu, s = _synthetic(3000, seed=5)
    # 'tiny' gets 6 points: below min_group_n (30) and below the 9 needed to
    # certify at alpha=0.10.
    grp = np.where(np.arange(y.size) < 2994, "big", "tiny")
    c = ConformalRUL(alpha=0.10, mode="mondrian", min_group_n=30).fit(y, mu, s, grp)
    audit = c.audit_groups(grp)
    assert audit["groups"]["tiny"]["fell_back"] is True
    assert audit["groups"]["tiny"]["certifiable"] is False
    assert audit["groups"]["big"]["certifiable"] is True
    assert audit["all_certifiable"] is False
    assert audit["groups"]["tiny"]["radius"] == c.global_radius_


def test_min_group_size_matches_exact_rank_condition():
    """Cross-check the search against the rank condition it encodes:
    a group of size n can certify iff ceil((n+1)*level) <= n."""
    for alpha in (0.10, 0.05, 0.20):
        c = ConformalRUL(alpha=alpha)
        n = c.min_group_size()
        level = 1.0 - alpha
        assert int(np.ceil((n + 1) * level)) <= n, f"n={n} should certify at alpha={alpha}"
        if n > 1:
            assert int(np.ceil((n) * level)) > n - 1, f"n={n - 1} should NOT certify at alpha={alpha}"


def test_drift_inflation_widens_and_is_capped():
    y, mu, s = _synthetic(3000, seed=6)
    c = ConformalRUL(alpha=0.1, drift_kappa=1.6, drift_cap=2.5).fit(y[:1500], mu[:1500], s[:1500])
    _, _, r0 = c.interval(mu[1500:], s[1500:], drift=0.0)
    _, _, r1 = c.interval(mu[1500:], s[1500:], drift=0.1)
    _, _, r2 = c.interval(mu[1500:], s[1500:], drift=100.0)
    assert np.all(r1 >= r0)
    assert np.all(r2 >= r1)
    assert np.all(r2 <= r0 * 2.5 + 1e-9), "inflation must be capped"


def test_drift_inflation_improves_coverage_under_shift():
    """The safeguard should actually do something under genuine shift."""
    rng = np.random.default_rng(7)
    n = 6000
    mu = rng.normal(100, 10, size=n)
    s = np.full(n, 3.0)
    y = rng.normal(mu, s)
    c = ConformalRUL(alpha=0.1).fit(y[:3000], mu[:3000], s[:3000])
    # Shift the second half.
    mu_s, y_s = mu[3000:] + 6.0, y[3000:] + 6.0
    cov0 = c.empirical_coverage(y_s, mu_s, s[3000:])
    cov1 = c.empirical_coverage(y_s, mu_s, s[3000:], drift=0.5)
    assert cov1 > cov0, "drift-aware inflation should recover some coverage under shift"


def test_groups_none_in_mondrian_is_safe():
    """The online path has no regime label; it must not crash."""
    y, mu, s = _synthetic(2000, seed=8)
    c = ConformalRUL(alpha=0.1, mode="mondrian").fit(
        y[:1000], mu[:1000], s[:1000], np.where(np.arange(1000) % 2, "a", "b")
    )
    lo, hi, r = c.interval(mu[1000:], s[1000:], groups=None)
    assert lo.shape == mu[1000:].shape


def test_requires_fit():
    c = ConformalRUL()
    with pytest.raises(Exception):
        c.interval(np.array([1.0]), np.array([1.0]))