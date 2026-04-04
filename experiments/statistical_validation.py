"""
experiments/statistical_validation.py — Rigorous statistical validation of Config H.

Tests:
  1. One-sample t-test (H0: mean R = 0)
  2. Bootstrap confidence intervals (10k resamples)
  3. Monte Carlo permutation test (10k shuffles)
  4. Walk-forward robustness (leave-one-year-out)
  5. Drawdown analysis (duration, recovery factor, bootstrap)
  6. Multiple testing adjustment (Bonferroni / Holm-Bonferroni)
  7. Regime robustness (first half vs second half)

Config H: sq_short=0.80, block_pm_shorts=True, tp1_trim_pct=0.50,
          tp2_trim_pct=0.50, ny_tp_mult=1.0, trail_nth_swing=3

Usage: python experiments/statistical_validation.py
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)

from experiments.multi_level_tp import (
    prepare_liquidity_data,
    run_backtest_multi_tp,
)
from experiments.validate_improvements import (
    load_all,
    compute_metrics,
    walk_forward_metrics,
    print_metrics,
)

SEPARATOR = "=" * 110
THIN_SEP = "-" * 110

# ======================================================================
# Helper: compute PF from R array
# ======================================================================
def _pf(r: np.ndarray) -> float:
    wins = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    return wins / losses if losses > 0 else 999.0


def _max_dd(r: np.ndarray) -> float:
    cumr = np.cumsum(r)
    peak = np.maximum.accumulate(cumr)
    dd = peak - cumr
    return dd.max() if len(dd) > 0 else 0.0


def _ppdd(r: np.ndarray) -> float:
    total = r.sum()
    mdd = _max_dd(r)
    return total / mdd if mdd > 0 else 999.0


# ======================================================================
# Test 1: One-sample t-test
# ======================================================================
def test_1_ttest(r: np.ndarray) -> dict:
    """One-sample t-test: H0 = mean per-trade R is 0 (no edge)."""
    n = len(r)
    mean_r = r.mean()
    se = r.std(ddof=1) / np.sqrt(n)

    t_stat, p_value = stats.ttest_1samp(r, 0.0)

    # Confidence intervals
    ci_95 = stats.t.interval(0.95, df=n - 1, loc=mean_r, scale=se)
    ci_99 = stats.t.interval(0.99, df=n - 1, loc=mean_r, scale=se)

    return {
        "n": n,
        "mean_r": mean_r,
        "std_r": r.std(ddof=1),
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_95": ci_95,
        "ci_99": ci_99,
    }


# ======================================================================
# Test 2: Bootstrap confidence intervals
# ======================================================================
def test_2_bootstrap(r: np.ndarray, n_boot: int = 10_000, seed: int = 42) -> dict:
    """Bootstrap 10k resamples. CI for total R, PPDD, PF."""
    rng = np.random.default_rng(seed)
    n = len(r)

    boot_total_r = np.empty(n_boot)
    boot_ppdd = np.empty(n_boot)
    boot_pf = np.empty(n_boot)

    for b in range(n_boot):
        sample = rng.choice(r, size=n, replace=True)
        boot_total_r[b] = sample.sum()
        boot_ppdd[b] = _ppdd(sample)
        boot_pf[b] = _pf(sample)

    # 95% CI (percentile method)
    ci_total_r = (np.percentile(boot_total_r, 2.5), np.percentile(boot_total_r, 97.5))
    ci_ppdd = (np.percentile(boot_ppdd, 2.5), np.percentile(boot_ppdd, 97.5))
    ci_pf = (np.percentile(boot_pf, 2.5), np.percentile(boot_pf, 97.5))

    prob_profitable = (boot_total_r > 0).mean()

    return {
        "boot_total_r_mean": boot_total_r.mean(),
        "boot_total_r_median": np.median(boot_total_r),
        "ci_total_r_95": ci_total_r,
        "boot_ppdd_mean": boot_ppdd.mean(),
        "ci_ppdd_95": ci_ppdd,
        "boot_pf_mean": boot_pf.mean(),
        "ci_pf_95": ci_pf,
        "prob_profitable": prob_profitable,
        "boot_total_r": boot_total_r,
    }


# ======================================================================
# Test 3: Monte Carlo permutation test
# ======================================================================
def test_3_montecarlo(r: np.ndarray, n_perm: int = 10_000, seed: int = 42) -> dict:
    """Shuffle R series 10k times, compute fraction with total R >= observed."""
    rng = np.random.default_rng(seed)
    observed_total = r.sum()

    perm_totals = np.empty(n_perm)
    for p in range(n_perm):
        # Randomly flip signs (permutation test for mean = 0)
        signs = rng.choice([-1, 1], size=len(r))
        shuffled = r * signs
        perm_totals[p] = shuffled.sum()

    p_value = (perm_totals >= observed_total).mean()
    pctl_5 = np.percentile(perm_totals, 5)
    pctl_95 = np.percentile(perm_totals, 95)

    return {
        "observed_total": observed_total,
        "p_value": p_value,
        "perm_mean": perm_totals.mean(),
        "perm_std": perm_totals.std(),
        "perm_pctl_5": pctl_5,
        "perm_pctl_95": pctl_95,
        "perm_totals": perm_totals,
    }


# ======================================================================
# Test 4: Walk-forward robustness
# ======================================================================
def test_4_walkforward(trades_list: list[dict]) -> dict:
    """Per-year analysis + leave-one-year-out."""
    df = pd.DataFrame(trades_list)
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year
    r = df["r"].values
    total_r = r.sum()

    # Per-year breakdown
    yearly = []
    for year, grp in df.groupby("year"):
        yr = grp["r"].values
        wins = yr[yr > 0].sum()
        losses = abs(yr[yr < 0].sum())
        pf = wins / losses if losses > 0 else 999.0
        wr = (yr > 0).mean() * 100
        yearly.append({
            "year": year,
            "n": len(grp),
            "R": yr.sum(),
            "WR": wr,
            "PF": pf,
        })

    # Leave-one-year-out
    loyo = []
    for year, grp in df.groupby("year"):
        remaining = df[df["year"] != year]["r"].values
        remaining_r = remaining.sum()
        loyo.append({
            "year_removed": year,
            "remaining_n": len(remaining),
            "remaining_R": remaining_r,
            "remaining_PF": _pf(remaining),
            "remaining_PPDD": _ppdd(remaining),
        })

    # Best year identification
    yearly_sorted = sorted(yearly, key=lambda x: x["R"], reverse=True)
    best_year = yearly_sorted[0]
    r_without_best = total_r - best_year["R"]

    return {
        "yearly": yearly,
        "loyo": loyo,
        "best_year": best_year,
        "r_without_best": r_without_best,
        "still_profitable_without_best": r_without_best > 0,
    }


# ======================================================================
# Test 5: Drawdown analysis
# ======================================================================
def test_5_drawdown(r: np.ndarray, n_boot: int = 10_000, seed: int = 42) -> dict:
    """Max drawdown duration, recovery factor, bootstrap DD distribution."""
    cumr = np.cumsum(r)
    peak = np.maximum.accumulate(cumr)
    dd = peak - cumr

    max_dd = dd.max()
    total_r = r.sum()
    recovery_factor = total_r / max_dd if max_dd > 0 else 999.0

    # Max drawdown duration (trades, not time)
    in_dd = False
    current_duration = 0
    max_duration = 0
    max_dd_start = 0
    dd_start = 0

    for i in range(len(cumr)):
        if dd[i] > 0:
            if not in_dd:
                in_dd = True
                dd_start = i
            current_duration = i - dd_start + 1
            if current_duration > max_duration:
                max_duration = current_duration
                max_dd_start = dd_start
        else:
            in_dd = False
            current_duration = 0

    # Bootstrap: distribution of max drawdowns
    rng = np.random.default_rng(seed)
    boot_max_dd = np.empty(n_boot)

    for b in range(n_boot):
        sample = rng.choice(r, size=len(r), replace=True)
        boot_max_dd[b] = _max_dd(sample)

    prob_worse_dd = (boot_max_dd >= max_dd).mean()
    dd_95_pctl = np.percentile(boot_max_dd, 95)

    return {
        "max_dd": max_dd,
        "recovery_factor": recovery_factor,
        "max_dd_duration_trades": max_duration,
        "max_dd_start_trade": max_dd_start,
        "prob_worse_dd": prob_worse_dd,
        "boot_dd_mean": boot_max_dd.mean(),
        "boot_dd_median": np.median(boot_max_dd),
        "boot_dd_95_pctl": dd_95_pctl,
    }


# ======================================================================
# Test 6: Multiple testing adjustment
# ======================================================================
def test_6_multiple_testing(p_value_raw: float, n_configs_tested: int = 75) -> dict:
    """Apply Bonferroni and Holm-Bonferroni correction.

    Estimated configs tested:
      - ny_tp_mult: 9 values (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
      - trim split: ~7 values
      - trail: 3 values (1, 2, 3)
      - A2a-e experiments: ~5 major configs each with ~3-5 sub-variants
      - Total estimate: ~50-100, use midpoint 75
    """
    # Bonferroni: multiply p-value by number of tests
    bonferroni_p = min(1.0, p_value_raw * n_configs_tested)

    # Holm-Bonferroni: for a single test being checked, the correction
    # is the same as Bonferroni when we only have one p-value.
    # In practice, if this were the best of N, the correction is:
    # p_adjusted = p_raw * N (same as Bonferroni for the minimum p-value)
    holm_p = bonferroni_p  # For the best p-value among N tests

    # Sidak correction: 1 - (1 - p)^N
    sidak_p = 1.0 - (1.0 - p_value_raw) ** n_configs_tested

    return {
        "raw_p": p_value_raw,
        "n_configs": n_configs_tested,
        "bonferroni_p": bonferroni_p,
        "holm_p": holm_p,
        "sidak_p": sidak_p,
        "survives_bonferroni_05": bonferroni_p < 0.05,
        "survives_bonferroni_01": bonferroni_p < 0.01,
        "survives_sidak_05": sidak_p < 0.05,
        "survives_sidak_01": sidak_p < 0.01,
    }


# ======================================================================
# Test 7: Regime robustness
# ======================================================================
def test_7_regime(trades_list: list[dict]) -> dict:
    """Split into 2 halves by date + identify bull/bear years."""
    df = pd.DataFrame(trades_list)
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year

    years = sorted(df["year"].unique())
    mid = len(years) // 2
    first_half_years = set(years[:mid])
    second_half_years = set(years[mid:])

    df1 = df[df["year"].isin(first_half_years)]
    df2 = df[df["year"].isin(second_half_years)]

    r1 = df1["r"].values
    r2 = df2["r"].values

    # Bear market years for NQ (rough classification)
    # 2018: correction/volatile, 2020: COVID crash, 2022: bear market
    bear_years = {2018, 2020, 2022}
    bull_years = set(years) - bear_years

    df_bear = df[df["year"].isin(bear_years)]
    df_bull = df[df["year"].isin(bull_years)]
    r_bear = df_bear["r"].values if len(df_bear) > 0 else np.array([])
    r_bull = df_bull["r"].values if len(df_bull) > 0 else np.array([])

    # Two-sample t-test: is performance different between halves?
    if len(r1) > 1 and len(r2) > 1:
        t_stat_halves, p_halves = stats.ttest_ind(r1, r2)
    else:
        t_stat_halves, p_halves = np.nan, np.nan

    # Bull vs bear
    if len(r_bear) > 1 and len(r_bull) > 1:
        t_stat_regime, p_regime = stats.ttest_ind(r_bull, r_bear)
    else:
        t_stat_regime, p_regime = np.nan, np.nan

    return {
        "first_half": {
            "years": sorted(first_half_years),
            "n": len(r1),
            "R": r1.sum(),
            "PF": _pf(r1),
            "PPDD": _ppdd(r1),
            "avgR": r1.mean() if len(r1) > 0 else 0,
        },
        "second_half": {
            "years": sorted(second_half_years),
            "n": len(r2),
            "R": r2.sum(),
            "PF": _pf(r2),
            "PPDD": _ppdd(r2),
            "avgR": r2.mean() if len(r2) > 0 else 0,
        },
        "halves_t_stat": t_stat_halves,
        "halves_p_value": p_halves,
        "bull_years": sorted(bull_years),
        "bear_years": sorted(bear_years),
        "bull": {
            "n": len(r_bull),
            "R": r_bull.sum() if len(r_bull) > 0 else 0,
            "PF": _pf(r_bull) if len(r_bull) > 0 else 0,
            "avgR": r_bull.mean() if len(r_bull) > 0 else 0,
        },
        "bear": {
            "n": len(r_bear),
            "R": r_bear.sum() if len(r_bear) > 0 else 0,
            "PF": _pf(r_bear) if len(r_bear) > 0 else 0,
            "avgR": r_bear.mean() if len(r_bear) > 0 else 0,
        },
        "regime_t_stat": t_stat_regime,
        "regime_p_value": p_regime,
    }


# ======================================================================
# Print helpers
# ======================================================================
def print_test_1(res: dict):
    print(SEPARATOR)
    print("TEST 1: One-Sample t-Test  (H0: mean per-trade R = 0)")
    print(SEPARATOR)
    print(f"  Trades:        {res['n']}")
    print(f"  Mean R:        {res['mean_r']:+.4f}")
    print(f"  Std R:         {res['std_r']:.4f}")
    print(f"  Std Error:     {res['se']:.4f}")
    print(f"  t-statistic:   {res['t_stat']:.4f}")
    print(f"  p-value:       {res['p_value']:.2e}")
    print(f"  95% CI:        [{res['ci_95'][0]:+.4f}, {res['ci_95'][1]:+.4f}]")
    print(f"  99% CI:        [{res['ci_99'][0]:+.4f}, {res['ci_99'][1]:+.4f}]")
    print()
    if res["p_value"] < 0.001:
        print("  VERDICT: *** Highly significant (p < 0.001). Reject H0.")
        print("           The strategy has a statistically significant positive edge.")
    elif res["p_value"] < 0.01:
        print("  VERDICT: ** Significant (p < 0.01). Reject H0.")
    elif res["p_value"] < 0.05:
        print("  VERDICT: * Significant (p < 0.05). Reject H0.")
    else:
        print("  VERDICT: NOT significant (p >= 0.05). Cannot reject H0.")
    print()
    print(f"  Interpretation: With 95% confidence, the true mean R per trade")
    print(f"  lies between {res['ci_95'][0]:+.4f} and {res['ci_95'][1]:+.4f}.")
    print(f"  Over {res['n']} trades, that projects to [{res['ci_95'][0]*res['n']:+.1f}R, {res['ci_95'][1]*res['n']:+.1f}R] total.")


def print_test_2(res: dict):
    print()
    print(SEPARATOR)
    print("TEST 2: Bootstrap Confidence Intervals  (10,000 resamples)")
    print(SEPARATOR)
    print(f"  Total R:")
    print(f"    Mean:        {res['boot_total_r_mean']:+.1f}R")
    print(f"    Median:      {res['boot_total_r_median']:+.1f}R")
    print(f"    95% CI:      [{res['ci_total_r_95'][0]:+.1f}R, {res['ci_total_r_95'][1]:+.1f}R]")
    print(f"  PPDD:")
    print(f"    Mean:        {res['boot_ppdd_mean']:.2f}")
    print(f"    95% CI:      [{res['ci_ppdd_95'][0]:.2f}, {res['ci_ppdd_95'][1]:.2f}]")
    print(f"  Profit Factor:")
    print(f"    Mean:        {res['boot_pf_mean']:.2f}")
    print(f"    95% CI:      [{res['ci_pf_95'][0]:.2f}, {res['ci_pf_95'][1]:.2f}]")
    print()
    print(f"  P(strategy is net profitable) = {res['prob_profitable']*100:.1f}%")
    print()
    if res["prob_profitable"] > 0.99:
        print("  VERDICT: >99% probability of being net profitable.")
    elif res["prob_profitable"] > 0.95:
        print("  VERDICT: >95% probability of being net profitable.")
    else:
        print(f"  VERDICT: {res['prob_profitable']*100:.1f}% probability of being profitable.")
    print(f"  Even in the worst 2.5% of bootstraps, total R = {res['ci_total_r_95'][0]:+.1f}R")


def print_test_3(res: dict):
    print()
    print(SEPARATOR)
    print("TEST 3: Monte Carlo Permutation Test  (10,000 sign-flipped shuffles)")
    print(SEPARATOR)
    print(f"  Observed total R:      {res['observed_total']:+.1f}R")
    print(f"  Permutation mean:      {res['perm_mean']:+.1f}R")
    print(f"  Permutation std:       {res['perm_std']:.1f}R")
    print(f"  5th percentile:        {res['perm_pctl_5']:+.1f}R")
    print(f"  95th percentile:       {res['perm_pctl_95']:+.1f}R")
    print(f"  p-value (fraction >= observed): {res['p_value']:.6f}")
    print()
    if res["p_value"] == 0.0:
        print("  VERDICT: p < 0.0001 (0 out of 10,000 random shuffles matched the observed R).")
        print("           This result is extremely unlikely to occur by chance.")
    elif res["p_value"] < 0.001:
        print(f"  VERDICT: p = {res['p_value']:.4f}. Highly significant.")
    elif res["p_value"] < 0.05:
        print(f"  VERDICT: p = {res['p_value']:.4f}. Significant at 5% level.")
    else:
        print(f"  VERDICT: p = {res['p_value']:.4f}. NOT significant.")


def print_test_4(res: dict):
    print()
    print(SEPARATOR)
    print("TEST 4: Walk-Forward Robustness  (per-year + leave-one-year-out)")
    print(SEPARATOR)

    # Per-year table
    print(f"\n  {'Year':>6} | {'Trades':>6} | {'R':>9} | {'WR':>6} | {'PF':>6} | {'Verdict':>12}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*9}-+-{'-'*6}-+-{'-'*6}-+-{'-'*12}")
    n_positive = 0
    for y in res["yearly"]:
        verdict = "POSITIVE" if y["R"] > 0 else "NEGATIVE"
        if y["R"] > 0:
            n_positive += 1
        print(f"  {y['year']:>6} | {y['n']:>6} | {y['R']:+9.1f} | {y['WR']:5.1f}% | {y['PF']:6.2f} | {verdict:>12}")
    total_years = len(res["yearly"])
    print(f"\n  Positive years: {n_positive}/{total_years}")

    # Best year analysis
    best = res["best_year"]
    print(f"\n  Best year: {best['year']} ({best['R']:+.1f}R)")
    print(f"  Total R without best year: {res['r_without_best']:+.1f}R")
    print(f"  Still profitable without best year: {'YES' if res['still_profitable_without_best'] else 'NO'}")

    # Leave-one-year-out
    print(f"\n  Leave-One-Year-Out Analysis:")
    print(f"  {'Removed':>8} | {'Remaining n':>11} | {'Remaining R':>11} | {'PF':>6} | {'PPDD':>7}")
    print(f"  {'-'*8}-+-{'-'*11}-+-{'-'*11}-+-{'-'*6}-+-{'-'*7}")
    all_positive = True
    for lo in res["loyo"]:
        marker = " <-- BEST YEAR" if lo["year_removed"] == best["year"] else ""
        if lo["remaining_R"] <= 0:
            all_positive = False
        print(f"  {lo['year_removed']:>8} | {lo['remaining_n']:>11} | {lo['remaining_R']:+11.1f} | {lo['remaining_PF']:6.2f} | {lo['remaining_PPDD']:7.2f}{marker}")

    print()
    if all_positive:
        print("  VERDICT: Strategy is profitable in ALL leave-one-year-out scenarios.")
        print("           No single year is driving the result.")
    else:
        print("  WARNING: Removing some years makes the strategy unprofitable.")
        print("           Result may be driven by specific years.")


def print_test_5(res: dict):
    print()
    print(SEPARATOR)
    print("TEST 5: Drawdown Analysis")
    print(SEPARATOR)
    print(f"  Maximum drawdown:              {res['max_dd']:.1f}R")
    print(f"  Recovery factor (totalR/MaxDD): {res['recovery_factor']:.2f}")
    print(f"  Max drawdown duration:          {res['max_dd_duration_trades']} trades")
    print(f"    (starting at trade #{res['max_dd_start_trade']})")
    print(f"  Bootstrap MaxDD distribution (10k resamples):")
    print(f"    Mean MaxDD:                  {res['boot_dd_mean']:.1f}R")
    print(f"    Median MaxDD:                {res['boot_dd_median']:.1f}R")
    print(f"    95th percentile:             {res['boot_dd_95_pctl']:.1f}R")
    print(f"  P(experiencing MaxDD >= {res['max_dd']:.1f}R): {res['prob_worse_dd']*100:.1f}%")
    print()
    if res["recovery_factor"] > 5:
        print(f"  VERDICT: Recovery factor {res['recovery_factor']:.1f}x is excellent (>5x).")
    elif res["recovery_factor"] > 3:
        print(f"  VERDICT: Recovery factor {res['recovery_factor']:.1f}x is good (>3x).")
    elif res["recovery_factor"] > 2:
        print(f"  VERDICT: Recovery factor {res['recovery_factor']:.1f}x is acceptable (>2x).")
    else:
        print(f"  VERDICT: Recovery factor {res['recovery_factor']:.1f}x is low (<2x). Concerning.")


def print_test_6(res: dict):
    print()
    print(SEPARATOR)
    print("TEST 6: Multiple Testing Adjustment")
    print(SEPARATOR)
    print(f"  Estimated configs tested:      ~{res['n_configs']}")
    print(f"  Raw p-value (from t-test):     {res['raw_p']:.2e}")
    print(f"  Bonferroni-adjusted p-value:   {res['bonferroni_p']:.2e}")
    print(f"  Sidak-adjusted p-value:        {res['sidak_p']:.2e}")
    print()
    print(f"  Survives Bonferroni at alpha=0.05: {'YES' if res['survives_bonferroni_05'] else 'NO'}")
    print(f"  Survives Bonferroni at alpha=0.01: {'YES' if res['survives_bonferroni_01'] else 'NO'}")
    print(f"  Survives Sidak at alpha=0.05:      {'YES' if res['survives_sidak_05'] else 'NO'}")
    print(f"  Survives Sidak at alpha=0.01:      {'YES' if res['survives_sidak_01'] else 'NO'}")
    print()
    if res["survives_bonferroni_05"]:
        print("  VERDICT: Config H survives multiple testing correction at 5% significance.")
        if res["survives_bonferroni_01"]:
            print("           Also survives at 1% significance. Robust result.")
    else:
        print("  VERDICT: Config H does NOT survive Bonferroni correction.")
        print("           The apparent edge may be an artifact of optimization over ~75 configs.")


def print_test_7(res: dict):
    print()
    print(SEPARATOR)
    print("TEST 7: Regime Robustness")
    print(SEPARATOR)

    h1 = res["first_half"]
    h2 = res["second_half"]
    print(f"  First half  ({h1['years'][0]}-{h1['years'][-1]}): {h1['n']:4d} trades, R={h1['R']:+.1f}, PF={h1['PF']:.2f}, PPDD={h1['PPDD']:.2f}, avgR={h1['avgR']:+.4f}")
    print(f"  Second half ({h2['years'][0]}-{h2['years'][-1]}): {h2['n']:4d} trades, R={h2['R']:+.1f}, PF={h2['PF']:.2f}, PPDD={h2['PPDD']:.2f}, avgR={h2['avgR']:+.4f}")
    print(f"  t-stat (halves differ):  {res['halves_t_stat']:.3f}")
    print(f"  p-value (halves differ): {res['halves_p_value']:.4f}")

    both_profitable = h1["R"] > 0 and h2["R"] > 0
    print(f"\n  Both halves profitable: {'YES' if both_profitable else 'NO'}")
    if res["halves_p_value"] > 0.05:
        print("  No statistically significant difference between halves (p > 0.05).")
        print("  --> Strategy performance is CONSISTENT across time periods.")
    else:
        print("  Statistically significant difference between halves (p < 0.05).")
        print("  --> Strategy performance VARIES across time periods.")

    bull_yrs_str = ", ".join(str(int(y)) for y in res["bull_years"])
    bear_yrs_str = ", ".join(str(int(y)) for y in res["bear_years"])
    print(f"\n  Bull years ({bull_yrs_str}): {res['bull']['n']} trades, R={res['bull']['R']:+.1f}, PF={res['bull']['PF']:.2f}, avgR={res['bull']['avgR']:+.4f}")
    print(f"  Bear years ({bear_yrs_str}): {res['bear']['n']} trades, R={res['bear']['R']:+.1f}, PF={res['bear']['PF']:.2f}, avgR={res['bear']['avgR']:+.4f}")
    if not np.isnan(res["regime_p_value"]):
        print(f"  t-stat (bull vs bear):   {res['regime_t_stat']:.3f}")
        print(f"  p-value (bull vs bear):  {res['regime_p_value']:.4f}")
    bull_pos = res["bull"]["R"] > 0
    bear_pos = res["bear"]["R"] > 0
    print(f"\n  Profitable in bull markets: {'YES' if bull_pos else 'NO'}")
    print(f"  Profitable in bear markets: {'YES' if bear_pos else 'NO'}")
    if bull_pos and bear_pos:
        print("  --> Strategy works in BOTH bull and bear regimes. Good robustness.")
    else:
        print("  --> Strategy has regime dependency. Investigate further.")


# ======================================================================
# Overall summary
# ======================================================================
def print_overall_summary(
    res1: dict, res2: dict, res3: dict, res4: dict,
    res5: dict, res6: dict, res7: dict,
):
    print()
    print(SEPARATOR)
    print("OVERALL STATISTICAL SUMMARY")
    print(SEPARATOR)

    # Scorecard
    checks = []

    # T-test
    t_pass = res1["p_value"] < 0.05
    checks.append(("t-test p < 0.05", t_pass, f"p={res1['p_value']:.2e}"))

    # Bootstrap profitable
    boot_pass = res2["prob_profitable"] > 0.95
    checks.append(("Bootstrap P(profitable) > 95%", boot_pass,
                    f"{res2['prob_profitable']*100:.1f}%"))

    # Monte Carlo
    mc_pass = res3["p_value"] < 0.05
    checks.append(("Monte Carlo permutation p < 0.05", mc_pass,
                    f"p={res3['p_value']:.6f}"))

    # Walk-forward: profitable without best year
    wf_pass = res4["still_profitable_without_best"]
    checks.append(("Profitable without best year", wf_pass,
                    f"R_remaining={res4['r_without_best']:+.1f}"))

    # Positive years
    n_pos = sum(1 for y in res4["yearly"] if y["R"] > 0)
    n_total = len(res4["yearly"])
    yr_pass = n_pos >= n_total * 0.7
    checks.append((f">=70% years positive ({n_pos}/{n_total})", yr_pass,
                    f"{n_pos}/{n_total}"))

    # Recovery factor
    rf_pass = res5["recovery_factor"] > 3.0
    checks.append(("Recovery factor > 3.0", rf_pass,
                    f"RF={res5['recovery_factor']:.2f}"))

    # Multiple testing
    mt_pass = res6["survives_bonferroni_05"]
    checks.append(("Survives Bonferroni correction (alpha=0.05)", mt_pass,
                    f"adj_p={res6['bonferroni_p']:.2e}"))

    # Regime robustness
    both_halves = res7["first_half"]["R"] > 0 and res7["second_half"]["R"] > 0
    checks.append(("Both time-halves profitable", both_halves,
                    f"H1={res7['first_half']['R']:+.1f}R, H2={res7['second_half']['R']:+.1f}R"))

    bull_bear = res7["bull"]["R"] > 0 and res7["bear"]["R"] > 0
    checks.append(("Profitable in both bull & bear", bull_bear,
                    f"Bull={res7['bull']['R']:+.1f}R, Bear={res7['bear']['R']:+.1f}R"))

    n_pass = sum(1 for _, p, _ in checks if p)
    n_checks = len(checks)

    print(f"\n  SCORECARD: {n_pass}/{n_checks} checks passed\n")
    for label, passed, detail in checks:
        marker = "PASS" if passed else "FAIL"
        sym = "[+]" if passed else "[-]"
        print(f"  {sym} {label:50s} {marker:6s}  ({detail})")

    print(f"\n  {THIN_SEP}")
    if n_pass >= n_checks - 1:
        print("  FINAL VERDICT: Config H shows STRONG statistical evidence of a real edge.")
    elif n_pass >= n_checks - 3:
        print("  FINAL VERDICT: Config H shows MODERATE statistical evidence. Some concerns remain.")
    else:
        print("  FINAL VERDICT: Config H has WEAK statistical support. Likely overfit or fragile.")
    print(f"  {THIN_SEP}")


# ======================================================================
# Main
# ======================================================================
def main():
    t_start = _time.perf_counter()

    print(SEPARATOR)
    print("STATISTICAL VALIDATION — Config H")
    print("sq_short=0.80, block_pm_shorts=True, tp1_trim=0.50, tp2_trim=0.50,")
    print("ny_tp_mult=1.0, trail_nth_swing=3 (50/50/0 split, raw IRL, EOD close)")
    print(SEPARATOR)

    # ---- Load data ----
    print("\n[1/7] Loading data and running Config H backtest...")
    t0 = _time.perf_counter()
    d = load_all()
    d_extra = prepare_liquidity_data(d)

    # Run Config H backtest
    trades = run_backtest_multi_tp(
        d, d_extra,
        sq_short=0.80,
        block_pm_shorts=True,
        tp1_trim_pct=0.50,
        tp2_trim_pct=0.50,
        be_after_tp1=False,
        be_after_tp2=True,
        trail_nth_swing=3,
        ny_tp_mult=1.0,
    )

    m = compute_metrics(trades)
    print(f"\n  Config H backtest complete in {_time.perf_counter() - t0:.1f}s")
    print_metrics("Config H", m)

    r = np.array([t["r"] for t in trades])
    print(f"  Trade count: {len(r)}, total R: {r.sum():+.1f}")

    # ---- Test 1: t-test ----
    print(f"\n[2/7] Running t-test...")
    t0 = _time.perf_counter()
    res1 = test_1_ttest(r)
    print_test_1(res1)
    print(f"  (computed in {_time.perf_counter() - t0:.3f}s)")

    # ---- Test 2: Bootstrap ----
    print(f"\n[3/7] Running bootstrap (10,000 resamples)...")
    t0 = _time.perf_counter()
    res2 = test_2_bootstrap(r, n_boot=10_000)
    print_test_2(res2)
    print(f"  (computed in {_time.perf_counter() - t0:.1f}s)")

    # ---- Test 3: Monte Carlo ----
    print(f"\n[4/7] Running Monte Carlo permutation test (10,000 shuffles)...")
    t0 = _time.perf_counter()
    res3 = test_3_montecarlo(r, n_perm=10_000)
    print_test_3(res3)
    print(f"  (computed in {_time.perf_counter() - t0:.1f}s)")

    # ---- Test 4: Walk-forward ----
    print(f"\n[5/7] Running walk-forward robustness analysis...")
    t0 = _time.perf_counter()
    res4 = test_4_walkforward(trades)
    print_test_4(res4)
    print(f"  (computed in {_time.perf_counter() - t0:.3f}s)")

    # ---- Test 5: Drawdown ----
    print(f"\n[6/7] Running drawdown analysis...")
    t0 = _time.perf_counter()
    res5 = test_5_drawdown(r)
    print_test_5(res5)
    print(f"  (computed in {_time.perf_counter() - t0:.1f}s)")

    # ---- Test 6: Multiple testing ----
    print(f"\n[7/7] Running multiple testing adjustment...")
    t0 = _time.perf_counter()
    # Use t-test p-value as the raw p-value, estimate 75 configs tested
    res6 = test_6_multiple_testing(res1["p_value"], n_configs_tested=75)
    print_test_6(res6)
    print(f"  (computed in {_time.perf_counter() - t0:.3f}s)")

    # ---- Test 7: Regime ----
    # (numbered as bonus, but included in overall summary)
    print(f"\n[BONUS] Running regime robustness analysis...")
    t0 = _time.perf_counter()
    res7 = test_7_regime(trades)
    print_test_7(res7)
    print(f"  (computed in {_time.perf_counter() - t0:.3f}s)")

    # ---- Overall Summary ----
    print_overall_summary(res1, res2, res3, res4, res5, res6, res7)

    total_elapsed = _time.perf_counter() - t_start
    print(f"\n  Total elapsed: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
