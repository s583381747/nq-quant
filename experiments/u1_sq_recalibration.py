"""
experiments/u1_sq_recalibration.py — SQ Filter Recalibration for 1c (1m) Signals
=================================================================================

Problem: SQ filter was calibrated for 5m candles. 1m candles are naturally smaller,
so SQ scores are systematically lower → 74% of 1c signals get killed.

Key insight: size_sc = stop_dist / (1.5 * ATR_5m) ≈ 14 / 45 ≈ 0.31 for 1c,
vs ≈ 30 / 45 ≈ 0.67 for 5m. The size component is the primary killer.

Steps:
  1. Analyze SQ score distribution for all 1c signals reaching the SQ check
  2. Break down which SQ component kills most signals
  3. Test variants: lower threshold, 1m ATR normalization, remove size, no SQ
  4. Walk-forward top configs
  5. Check if more 1c trades degrade 5m performance (blocking)

Usage: python experiments/u1_sq_recalibration.py
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)

from experiments.validate_improvements import (
    load_all,
    _find_nth_swing,
    _compute_grade_fast,
    compute_metrics,
    walk_forward_metrics,
    print_metrics,
)
from experiments.multi_level_tp import (
    prepare_liquidity_data,
    _find_nth_swing_price,
    build_liquidity_ladder_long,
    build_liquidity_ladder_short,
)
from experiments.pure_liquidity_tp import run_backtest_pure_liquidity
from experiments.a2c_stop_widening_engine import widen_stop_array
from experiments.b1c_fvg_zone_1m_entry import (
    load_1m_data,
    precompute_1m_features,
    build_5m_fvg_zones,
    build_time_mappings,
    scan_1m_rejections,
    filter_signals,
    simulate_1m_outcomes,
    combine_5m_and_1m_trades,
)

SEP = "=" * 120
THIN = "-" * 80


# ======================================================================
# Run 5m Config H++ baseline (stop×0.85)
# ======================================================================
def run_5m_baseline(d: dict, d_extra: dict) -> tuple[list[dict], dict]:
    """Run Config H+ (5m) with stop tightening applied."""
    original_stop = d["model_stop_arr"].copy()
    tightened_stop = widen_stop_array(
        original_stop, d["entry_price_arr"], d["sig_dir"],
        d["sig_mask"], 0.85,
    )
    d["model_stop_arr"] = tightened_stop

    trades, diag = run_backtest_pure_liquidity(
        d, d_extra,
        sq_long=0.68, sq_short=0.80,
        min_stop_atr=1.7,
        block_pm_shorts=True,
        tp_strategy="v1",
        tp1_trim_pct=0.50,
        be_after_tp1=False,
        be_after_tp2=False,
    )
    d["model_stop_arr"] = original_stop
    m = compute_metrics(trades)
    return trades, m


# ======================================================================
# Compute SQ components for all 1c signals (post-hoc, not filtering)
# ======================================================================
def compute_sq_components_1c(
    signals: list[dict],
    d: dict,
    map_1m_to_prev_5m: np.ndarray,
    df_1m: pd.DataFrame,
    feat_1m: dict,
) -> list[dict]:
    """
    For each 1c signal, compute all 4 SQ components and the composite score.
    Also compute alternative size_sc using 1m ATR.

    Returns list of dicts with SQ info appended to each signal.
    """
    params = d["params"]
    atr_arr_5m = d["atr_arr"]
    fluency_5m = d["fluency_arr"]
    sq_params = params.get("signal_quality", {})

    o_1m = feat_1m["o"]
    c_1m = feat_1m["c"]
    br_1m = feat_1m["body_ratio"]

    w_size = sq_params.get("w_size", 0.3)
    w_disp = sq_params.get("w_disp", 0.3)
    w_flu = sq_params.get("w_flu", 0.2)
    w_pa = sq_params.get("w_pa", 0.2)

    result = []
    for sig in signals:
        j = sig["signal_1m_idx"]
        prev_5m = sig["prev_5m_idx"]
        atr_5m_val = atr_arr_5m[prev_5m] if not np.isnan(atr_arr_5m[prev_5m]) else 30.0
        atr_1m_val = sig.get("atr_1m", 6.0)

        # size_sc (original: uses 5m ATR)
        size_sc_5m = min(1.0, sig["stop_dist"] / (atr_5m_val * 1.5)) if atr_5m_val > 0 else 0.5

        # size_sc (recalibrated: uses 1m ATR)
        size_sc_1m = min(1.0, sig["stop_dist"] / (atr_1m_val * 1.5)) if atr_1m_val > 0 else 0.5

        # disp_sc: body ratio of 1m rejection candle
        disp_sc = br_1m[j]

        # flu_sc: 5m fluency from cache
        flu_val = fluency_5m[prev_5m] if not np.isnan(fluency_5m[prev_5m]) else 0.5
        flu_sc = min(1.0, max(0.0, flu_val))

        # pa_sc: directional consistency in recent 1m bars
        window = 6
        if j >= window:
            dirs = np.sign(c_1m[j - window:j] - o_1m[j - window:j])
            alt = np.sum(dirs[1:] != dirs[:-1]) / (window - 1)
            pa_sc = 1.0 - alt
        else:
            pa_sc = 0.5

        # Composite scores
        sq_original = w_size * size_sc_5m + w_disp * disp_sc + w_flu * flu_sc + w_pa * pa_sc
        sq_1m_atr = w_size * size_sc_1m + w_disp * disp_sc + w_flu * flu_sc + w_pa * pa_sc
        sq_no_size = (disp_sc + flu_sc + pa_sc) / 3.0

        entry = sig.copy()
        entry["size_sc_5m"] = size_sc_5m
        entry["size_sc_1m"] = size_sc_1m
        entry["disp_sc"] = disp_sc
        entry["flu_sc"] = flu_sc
        entry["pa_sc"] = pa_sc
        entry["sq_original"] = sq_original
        entry["sq_1m_atr"] = sq_1m_atr
        entry["sq_no_size"] = sq_no_size
        entry["atr_5m"] = atr_5m_val
        entry["atr_1m"] = atr_1m_val
        result.append(entry)

    return result


# ======================================================================
# Generate 1c signals with custom SQ threshold (post-hoc filtering)
# ======================================================================
def generate_1c_trades_custom_sq(
    raw_signals: list[dict],
    sq_enriched: list[dict],
    d: dict,
    d_extra: dict,
    df_1m: pd.DataFrame,
    feat_1m: dict,
    map_1m_to_prev_5m: np.ndarray,
    *,
    sq_field: str = "sq_original",  # which SQ score to use
    sq_long: float = 0.68,
    sq_short: float = 0.80,
    block_pm_shorts: bool = True,
    min_stop_atr_mult: float = 1.0,
    min_stop_pts_abs: float = 5.0,
    label: str = "",
) -> list[dict]:
    """
    Filter signals using pre-computed SQ scores (post-hoc) and simulate outcomes.

    This avoids re-running the full filter_signals which recomputes SQ internally.
    Instead we use the sq_enriched data and apply the threshold ourselves.
    """
    params = d["params"]
    atr_arr_5m = d["atr_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    fluency_5m = d["fluency_arr"]

    o_1m = feat_1m["o"]
    c_1m = feat_1m["c"]
    atr_1m = feat_1m["atr"]

    et_1m = df_1m.index.tz_convert("US/Eastern")
    et_frac_1m = (et_1m.hour + et_1m.minute / 60.0).values

    session_regime = params.get("session_regime", {})

    filtered = []
    stats = {"total": 0, "fail_obs": 0, "fail_news": 0, "fail_session": 0,
             "fail_pm": 0, "fail_bias": 0, "fail_min_stop": 0, "fail_lunch": 0,
             "fail_sq": 0, "pass": 0}

    for enr in sq_enriched:
        stats["total"] += 1
        j = enr["signal_1m_idx"]
        direction = enr["direction"]
        prev_5m = enr["prev_5m_idx"]
        et_frac = et_frac_1m[j]

        # Observation period
        if 9.5 <= et_frac <= 10.0:
            stats["fail_obs"] += 1
            continue

        # News blackout
        if news_blackout_arr is not None and news_blackout_arr[prev_5m]:
            stats["fail_news"] += 1
            continue

        # Session filter: NY only
        if not (10.0 <= et_frac < 16.0):
            stats["fail_session"] += 1
            continue

        # PM shorts block
        if block_pm_shorts and direction == -1 and et_frac >= 14.0:
            stats["fail_pm"] += 1
            continue

        # Bias alignment
        bd = bias_dir_arr[prev_5m]
        if direction == -np.sign(bd) and bd != 0:
            stats["fail_bias"] += 1
            continue

        # min_stop filter (1m ATR)
        stop_dist = enr["stop_dist"]
        if min_stop_pts_abs > 0 and stop_dist < min_stop_pts_abs:
            stats["fail_min_stop"] += 1
            continue
        atr_ref = enr.get("atr_1m", 6.0)
        if min_stop_atr_mult > 0 and atr_ref > 0 and (stop_dist / atr_ref) < min_stop_atr_mult:
            stats["fail_min_stop"] += 1
            continue

        # Session regime: lunch dead zone
        if session_regime.get("enabled", False):
            lunch_s = session_regime.get("lunch_start", 12.5)
            lunch_e = session_regime.get("lunch_end", 13.0)
            lunch_mult = session_regime.get("lunch_mult", 0.0)
            if lunch_s <= et_frac < lunch_e and lunch_mult == 0.0:
                stats["fail_lunch"] += 1
                continue

        # SQ filter using pre-computed score
        sq_val = enr[sq_field]
        eff_thresh = sq_long if direction == 1 else sq_short
        if sq_val < eff_thresh:
            stats["fail_sq"] += 1
            continue

        stats["pass"] += 1
        sig_out = enr.copy()
        sig_out["et_frac"] = et_frac
        sig_out["atr_5m"] = atr_arr_5m[prev_5m] if not np.isnan(atr_arr_5m[prev_5m]) else 30.0
        sig_out["regime"] = regime_arr[prev_5m]
        sig_out["bias_dir"] = bd
        filtered.append(sig_out)

    if not filtered:
        return []

    trades = simulate_1m_outcomes(
        filtered, d, d_extra, df_1m, feat_1m, map_1m_to_prev_5m,
    )
    for t in trades:
        t["source"] = "1m"

    return trades


# ======================================================================
# Combine with 5m baseline (one position at a time)
# ======================================================================
def combine_with_priority(
    trades_5m: list[dict],
    trades_1c: list[dict],
) -> list[dict]:
    """Combine 5m and 1c trades. One position at a time, 5m priority."""
    for t in trades_5m:
        t["source"] = "5m"
    all_trades = sorted(trades_5m + trades_1c, key=lambda t: t["entry_time"])
    combined = []
    last_exit = pd.Timestamp.min.tz_localize("UTC")
    for t in all_trades:
        entry_t = t["entry_time"]
        if hasattr(entry_t, "tz") and entry_t.tz is None:
            entry_t = entry_t.tz_localize("UTC")
        if entry_t <= last_exit:
            continue
        combined.append(t)
        exit_t = t["exit_time"]
        if hasattr(exit_t, "tz") and exit_t.tz is None:
            exit_t = exit_t.tz_localize("UTC")
        last_exit = exit_t
    return combined


# ======================================================================
# STEP 1: Distribution analysis
# ======================================================================
def step1_distribution_analysis(sq_enriched: list[dict]):
    """Analyze SQ score distribution for all 1c signals that reach the SQ check."""
    print(f"\n{SEP}")
    print("STEP 1: SQ SCORE DISTRIBUTION FOR 1c SIGNALS")
    print(SEP)

    n = len(sq_enriched)
    sq_orig = np.array([s["sq_original"] for s in sq_enriched])
    sq_1m = np.array([s["sq_1m_atr"] for s in sq_enriched])
    sq_nosize = np.array([s["sq_no_size"] for s in sq_enriched])

    print(f"\n  Total 1c signals reaching SQ check: {n}")

    print(f"\n  {'SQ Variant':25s} | {'Mean':>6s} | {'Median':>6s} | {'P25':>6s} | {'P75':>6s} | {'Min':>6s} | {'Max':>6s}")
    print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for name, arr in [("Original (5m ATR)", sq_orig), ("1m ATR recalib", sq_1m), ("No size (disp+flu+pa)", sq_nosize)]:
        print(f"  {name:25s} | {np.mean(arr):>6.3f} | {np.median(arr):>6.3f} | "
              f"{np.percentile(arr, 25):>6.3f} | {np.percentile(arr, 75):>6.3f} | "
              f"{np.min(arr):>6.3f} | {np.max(arr):>6.3f}")

    # Pass rates at various thresholds
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.68]
    print(f"\n  Pass rate at various thresholds:")
    print(f"  {'Threshold':>10s} | {'Original':>10s} | {'1m ATR':>10s} | {'No size':>10s}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for th in thresholds:
        p_orig = 100 * np.mean(sq_orig >= th)
        p_1m = 100 * np.mean(sq_1m >= th)
        p_ns = 100 * np.mean(sq_nosize >= th)
        print(f"  {th:>10.2f} | {p_orig:>9.1f}% | {p_1m:>9.1f}% | {p_ns:>9.1f}%")

    # By direction
    longs = [s for s in sq_enriched if s["direction"] == 1]
    shorts = [s for s in sq_enriched if s["direction"] == -1]
    print(f"\n  By direction:")
    for label, subset in [("Longs", longs), ("Shorts", shorts)]:
        if not subset:
            continue
        sq_o = np.array([s["sq_original"] for s in subset])
        sq_m = np.array([s["sq_1m_atr"] for s in subset])
        print(f"    {label}: n={len(subset)}, orig_mean={np.mean(sq_o):.3f}, "
              f"1m_mean={np.mean(sq_m):.3f}")
        eff_thresh = 0.68 if label == "Longs" else 0.80
        p_o = 100 * np.mean(sq_o >= eff_thresh)
        p_m = 100 * np.mean(sq_m >= eff_thresh)
        print(f"    Pass@{eff_thresh}: orig={p_o:.1f}%, 1m_atr={p_m:.1f}%")


# ======================================================================
# STEP 2: Component breakdown
# ======================================================================
def step2_component_breakdown(sq_enriched: list[dict]):
    """Analyze which SQ component kills most 1c signals."""
    print(f"\n{SEP}")
    print("STEP 2: SQ COMPONENT BREAKDOWN FOR 1c SIGNALS")
    print(SEP)

    n = len(sq_enriched)
    size_5m = np.array([s["size_sc_5m"] for s in sq_enriched])
    size_1m = np.array([s["size_sc_1m"] for s in sq_enriched])
    disp = np.array([s["disp_sc"] for s in sq_enriched])
    flu = np.array([s["flu_sc"] for s in sq_enriched])
    pa = np.array([s["pa_sc"] for s in sq_enriched])

    print(f"\n  Component statistics (n={n}):")
    print(f"  {'Component':20s} | {'Mean':>6s} | {'Median':>6s} | {'P25':>6s} | {'P75':>6s} | {'Weight':>6s} | {'Wtd Mean':>8s}")
    print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")

    for name, arr, w in [
        ("size_sc (5m ATR)", size_5m, 0.30),
        ("size_sc (1m ATR)", size_1m, 0.30),
        ("disp_sc", disp, 0.30),
        ("flu_sc", flu, 0.20),
        ("pa_sc", pa, 0.20),
    ]:
        print(f"  {name:20s} | {np.mean(arr):>6.3f} | {np.median(arr):>6.3f} | "
              f"{np.percentile(arr, 25):>6.3f} | {np.percentile(arr, 75):>6.3f} | "
              f"{w:>6.2f} | {w*np.mean(arr):>8.4f}")

    # Weighted contribution to SQ score
    print(f"\n  Weighted contribution to final SQ (original formula):")
    print(f"    size_sc(5m): {0.30 * np.mean(size_5m):.4f}  ({100 * 0.30 * np.mean(size_5m) / np.mean([s['sq_original'] for s in sq_enriched]):.1f}%)")
    print(f"    disp_sc:     {0.30 * np.mean(disp):.4f}  ({100 * 0.30 * np.mean(disp) / np.mean([s['sq_original'] for s in sq_enriched]):.1f}%)")
    print(f"    flu_sc:      {0.20 * np.mean(flu):.4f}  ({100 * 0.20 * np.mean(flu) / np.mean([s['sq_original'] for s in sq_enriched]):.1f}%)")
    print(f"    pa_sc:       {0.20 * np.mean(pa):.4f}  ({100 * 0.20 * np.mean(pa) / np.mean([s['sq_original'] for s in sq_enriched]):.1f}%)")

    # What if size used 1m ATR?
    sq_orig_mean = np.mean([s["sq_original"] for s in sq_enriched])
    sq_1m_mean = np.mean([s["sq_1m_atr"] for s in sq_enriched])
    delta = sq_1m_mean - sq_orig_mean
    print(f"\n  Effect of switching to 1m ATR for size:")
    print(f"    SQ mean (original): {sq_orig_mean:.4f}")
    print(f"    SQ mean (1m ATR):   {sq_1m_mean:.4f}")
    print(f"    Delta:              {delta:+.4f}")
    print(f"    size_sc mean: 5m={np.mean(size_5m):.3f} → 1m={np.mean(size_1m):.3f} (delta={np.mean(size_1m)-np.mean(size_5m):+.3f})")

    # ATR comparison
    atr_5m_vals = np.array([s["atr_5m"] for s in sq_enriched])
    atr_1m_vals = np.array([s["atr_1m"] for s in sq_enriched])
    stop_vals = np.array([s["stop_dist"] for s in sq_enriched])
    print(f"\n  ATR comparison at signal points:")
    print(f"    5m ATR: mean={np.mean(atr_5m_vals):.1f}, median={np.median(atr_5m_vals):.1f}")
    print(f"    1m ATR: mean={np.mean(atr_1m_vals):.1f}, median={np.median(atr_1m_vals):.1f}")
    print(f"    Ratio 5m/1m: {np.mean(atr_5m_vals)/np.mean(atr_1m_vals):.1f}x")
    print(f"    Stop dist: mean={np.mean(stop_vals):.1f}, median={np.median(stop_vals):.1f}")
    print(f"    Stop/ATR_5m: {np.mean(stop_vals/atr_5m_vals):.3f}")
    print(f"    Stop/ATR_1m: {np.mean(stop_vals/atr_1m_vals):.3f}")


# ======================================================================
# STEP 3+4: Test SQ variants (combined with 5m)
# ======================================================================
def step3_test_variants(
    sq_enriched: list[dict],
    raw_signals: list[dict],
    trades_5m: list[dict],
    d: dict,
    d_extra: dict,
    df_1m: pd.DataFrame,
    feat_1m: dict,
    map_1m_to_prev_5m: np.ndarray,
    baseline_m: dict,
) -> dict:
    """
    Test SQ variants for 1c and combine with 5m.

    Variants:
      A. Lower SQ threshold for 1c (sweep 0.40-0.68)
      B. Recalibrate size using 1m ATR (threshold sweep)
      C. Remove size component (disp+flu+pa only, threshold sweep)
      D. No SQ (baseline from b1c_deploy Config C)
    """
    print(f"\n{SEP}")
    print("STEP 3: TEST SQ VARIANTS FOR 1c (COMBINED WITH 5m)")
    print(SEP)

    results = {}

    # ========== VARIANT A: Lower SQ threshold (original formula) ==========
    print(f"\n{THIN}")
    print("VARIANT A: Lower SQ threshold for 1c only (original formula)")
    print(THIN)

    for sq_th in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.68]:
        label = f"A: sq_1c={sq_th:.2f}"
        trades_1c = generate_1c_trades_custom_sq(
            raw_signals, sq_enriched, d, d_extra, df_1m, feat_1m, map_1m_to_prev_5m,
            sq_field="sq_original",
            sq_long=sq_th,
            sq_short=max(sq_th, 0.80),  # shorts always stricter
            block_pm_shorts=True,
            label=label,
        )
        combined = combine_with_priority(
            [t.copy() for t in trades_5m],
            [t.copy() for t in trades_1c],
        )
        m = compute_metrics(combined)
        n5 = sum(1 for t in combined if t.get("source") == "5m")
        n1c = sum(1 for t in combined if t.get("source") == "1m")
        m_1c = compute_metrics(trades_1c) if trades_1c else {"trades": 0, "R": 0.0, "avgR": 0.0, "WR": 0.0}
        results[label] = {"m": m, "n5": n5, "n1c": n1c, "m_1c": m_1c, "trades": combined, "trades_1c": trades_1c}

    # ========== VARIANT B: 1m ATR for size component (threshold sweep) ==========
    print(f"\n{THIN}")
    print("VARIANT B: Recalibrate size_sc using 1m ATR")
    print(THIN)

    for sq_th in [0.50, 0.55, 0.60, 0.65, 0.68]:
        label = f"B: 1m_ATR sq={sq_th:.2f}"
        trades_1c = generate_1c_trades_custom_sq(
            raw_signals, sq_enriched, d, d_extra, df_1m, feat_1m, map_1m_to_prev_5m,
            sq_field="sq_1m_atr",
            sq_long=sq_th,
            sq_short=max(sq_th, 0.80),
            block_pm_shorts=True,
            label=label,
        )
        combined = combine_with_priority(
            [t.copy() for t in trades_5m],
            [t.copy() for t in trades_1c],
        )
        m = compute_metrics(combined)
        n5 = sum(1 for t in combined if t.get("source") == "5m")
        n1c = sum(1 for t in combined if t.get("source") == "1m")
        m_1c = compute_metrics(trades_1c) if trades_1c else {"trades": 0, "R": 0.0, "avgR": 0.0, "WR": 0.0}
        results[label] = {"m": m, "n5": n5, "n1c": n1c, "m_1c": m_1c, "trades": combined, "trades_1c": trades_1c}

    # ========== VARIANT C: Remove size component (disp+flu+pa only) ==========
    print(f"\n{THIN}")
    print("VARIANT C: No size component — SQ = (disp + flu + pa) / 3")
    print(THIN)

    for sq_th in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        label = f"C: no_size sq={sq_th:.2f}"
        trades_1c = generate_1c_trades_custom_sq(
            raw_signals, sq_enriched, d, d_extra, df_1m, feat_1m, map_1m_to_prev_5m,
            sq_field="sq_no_size",
            sq_long=sq_th,
            sq_short=max(sq_th, 0.80),
            block_pm_shorts=True,
            label=label,
        )
        combined = combine_with_priority(
            [t.copy() for t in trades_5m],
            [t.copy() for t in trades_1c],
        )
        m = compute_metrics(combined)
        n5 = sum(1 for t in combined if t.get("source") == "5m")
        n1c = sum(1 for t in combined if t.get("source") == "1m")
        m_1c = compute_metrics(trades_1c) if trades_1c else {"trades": 0, "R": 0.0, "avgR": 0.0, "WR": 0.0}
        results[label] = {"m": m, "n5": n5, "n1c": n1c, "m_1c": m_1c, "trades": combined, "trades_1c": trades_1c}

    # ========== VARIANT D: No SQ for 1c ==========
    print(f"\n{THIN}")
    print("VARIANT D: No SQ filter for 1c")
    print(THIN)

    label = "D: no_sq"
    trades_1c = generate_1c_trades_custom_sq(
        raw_signals, sq_enriched, d, d_extra, df_1m, feat_1m, map_1m_to_prev_5m,
        sq_field="sq_original",
        sq_long=-999.0,  # pass everything
        sq_short=-999.0,
        block_pm_shorts=True,
        label=label,
    )
    combined = combine_with_priority(
        [t.copy() for t in trades_5m],
        [t.copy() for t in trades_1c],
    )
    m = compute_metrics(combined)
    n5 = sum(1 for t in combined if t.get("source") == "5m")
    n1c = sum(1 for t in combined if t.get("source") == "1m")
    m_1c = compute_metrics(trades_1c) if trades_1c else {"trades": 0, "R": 0.0, "avgR": 0.0, "WR": 0.0}
    results[label] = {"m": m, "n5": n5, "n1c": n1c, "m_1c": m_1c, "trades": combined, "trades_1c": trades_1c}

    # ========== RESULTS TABLE ==========
    print(f"\n{SEP}")
    print("RESULTS TABLE (all variants, combined 5m+1c)")
    print(SEP)

    print(f"\n  {'Config':30s} | {'Total':>5s} | {'5m':>4s} | {'1c':>4s} | {'R':>9s} | {'PPDD':>7s} | {'PF':>6s} | {'WR':>6s} | {'MaxDD':>6s} | {'1c_R':>7s} | {'1c_avgR':>8s} | {'1c_WR':>6s}")
    print(f"  {'-'*30}-+-{'-'*5}-+-{'-'*4}-+-{'-'*4}-+-{'-'*9}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}")

    # Baseline row
    print(f"  {'5m only (H++ baseline)':30s} | {baseline_m['trades']:>5d} | {baseline_m['trades']:>4d} | {'0':>4s} | "
          f"{baseline_m['R']:>+9.1f} | {baseline_m['PPDD']:>7.2f} | {baseline_m['PF']:>6.2f} | {baseline_m['WR']:>5.1f}% | "
          f"{baseline_m['MaxDD']:>6.1f} | {'—':>7s} | {'—':>8s} | {'—':>6s}")

    for label in sorted(results.keys()):
        r = results[label]
        m = r["m"]
        m_1c = r["m_1c"]
        print(f"  {label:30s} | {m['trades']:>5d} | {r['n5']:>4d} | {r['n1c']:>4d} | "
              f"{m['R']:>+9.1f} | {m['PPDD']:>7.2f} | {m['PF']:>6.2f} | {m['WR']:>5.1f}% | "
              f"{m['MaxDD']:>6.1f} | {m_1c['R']:>+7.1f} | {m_1c['avgR']:>+8.4f} | {m_1c['WR']:>5.1f}%")

    return results


# ======================================================================
# STEP 5: Walk-forward for top configs
# ======================================================================
def step5_walk_forward(results: dict, baseline_m: dict):
    """Walk-forward validation for top 3 configs by PPDD."""
    print(f"\n{SEP}")
    print("STEP 5: WALK-FORWARD VALIDATION (Top Configs)")
    print(SEP)

    # Sort by PPDD descending
    sorted_configs = sorted(results.items(), key=lambda x: x[1]["m"]["PPDD"], reverse=True)
    top_n = min(5, len(sorted_configs))

    for rank, (label, data) in enumerate(sorted_configs[:top_n], 1):
        m = data["m"]
        combined = data["trades"]
        print(f"\n  #{rank}: {label}")
        print(f"    Combined: {m['trades']}t, R={m['R']:+.1f}, PPDD={m['PPDD']:.2f}, PF={m['PF']:.2f}, MaxDD={m['MaxDD']:.1f}")
        print(f"    vs baseline: dR={m['R']-baseline_m['R']:+.1f}, dPPDD={m['PPDD']-baseline_m['PPDD']:+.2f}")

        wf = walk_forward_metrics(combined)
        if wf:
            neg_years = sum(1 for yr in wf if yr["R"] < 0)
            print(f"\n    {'Year':>6s} | {'N':>4s} | {'5m':>4s} | {'1c':>4s} | {'R':>8s} | {'WR':>6s} | {'PF':>6s} | {'PPDD':>7s}")
            print(f"    {'-'*6}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")
            for yr in wf:
                y = yr["year"]
                src_5m = sum(1 for t in combined
                             if t.get("source") == "5m"
                             and pd.Timestamp(t["entry_time"]).year == y)
                src_1c = sum(1 for t in combined
                             if t.get("source") == "1m"
                             and pd.Timestamp(t["entry_time"]).year == y)
                print(f"    {y:>6d} | {yr['n']:>4d} | {src_5m:>4d} | {src_1c:>4d} | "
                      f"{yr['R']:>+8.1f} | {yr['WR']:>5.1f}% | {yr['PF']:>6.2f} | {yr['PPDD']:>7.2f}")
            print(f"    Negative years: {neg_years}/{len(wf)}")


# ======================================================================
# STEP 6: Check 1c blocking of 5m trades
# ======================================================================
def step6_blocking_analysis(results: dict, trades_5m: list[dict], baseline_m: dict):
    """Analyze whether 1c trades block 5m signals."""
    print(f"\n{SEP}")
    print("STEP 6: 1c BLOCKING ANALYSIS — Do 1c trades prevent 5m entries?")
    print(SEP)

    n_5m_baseline = baseline_m["trades"]

    print(f"\n  5m baseline trades: {n_5m_baseline}")
    print(f"\n  {'Config':30s} | {'5m in combo':>11s} | {'5m blocked':>10s} | {'1c added':>8s} | {'Net trades':>10s} | {'Net R':>8s}")
    print(f"  {'-'*30}-+-{'-'*11}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}")

    sorted_configs = sorted(results.items(), key=lambda x: x[1]["m"]["PPDD"], reverse=True)
    for label, data in sorted_configs[:8]:
        n5_combo = data["n5"]
        n1c = data["n1c"]
        blocked_5m = n_5m_baseline - n5_combo
        net_trades = n1c - blocked_5m
        net_r = data["m"]["R"] - baseline_m["R"]
        print(f"  {label:30s} | {n5_combo:>11d} | {blocked_5m:>10d} | {n1c:>8d} | {net_trades:>+10d} | {net_r:>+8.1f}")

    # For top config, analyze which 5m trades get blocked
    best_label = sorted_configs[0][0]
    best_data = sorted_configs[0]
    best_combined = best_data[1]["trades"]
    best_1c = best_data[1]["trades_1c"]

    print(f"\n  Detailed blocking analysis for best config: {best_label}")

    # Find 5m trades that were blocked
    combo_5m_times = set()
    for t in best_combined:
        if t.get("source") == "5m":
            combo_5m_times.add(t["entry_time"])

    blocked_trades = [t for t in trades_5m if t["entry_time"] not in combo_5m_times]
    if blocked_trades:
        bl_r = np.array([t["r"] for t in blocked_trades])
        print(f"    Blocked 5m trades: {len(blocked_trades)}")
        print(f"    Blocked 5m R: {bl_r.sum():+.1f} (avgR={bl_r.mean():+.4f}, WR={100*(bl_r>0).mean():.1f}%)")
        print(f"    1c trades added: {len(best_1c)}, R={sum(t['r'] for t in best_1c):+.1f}")
        print(f"    Net R impact: {sum(t['r'] for t in best_1c) - bl_r.sum():+.1f}")
    else:
        print(f"    No 5m trades blocked!")


# ======================================================================
# Main
# ======================================================================
def main():
    t_start = _time.perf_counter()

    print(SEP)
    print("U1: SQ FILTER RECALIBRATION FOR 1c SIGNALS")
    print("  Problem: SQ was calibrated for 5m candles; 1m candles → lower SQ → over-filtering")
    print("  Current: SQ kills 290/393 signals (74%) — the biggest bottleneck")
    print(SEP)

    # ================================================================
    # LOAD DATA
    # ================================================================
    print("\n[LOAD] Loading 5m data...")
    d = load_all()
    d_extra = prepare_liquidity_data(d)

    print("\n[LOAD] Loading 1m data...")
    df_1m = load_1m_data()
    feat_1m = precompute_1m_features(df_1m)

    nq_5m = d["nq"]
    atr_5m = d["atr_arr"]

    # Report ATR stats
    atr_1m_vals = feat_1m["atr"]
    valid_atr_1m = atr_1m_vals[~np.isnan(atr_1m_vals)]
    valid_atr_5m = atr_5m[~np.isnan(atr_5m)]
    print(f"  1m ATR: mean={np.mean(valid_atr_1m):.2f}, median={np.median(valid_atr_1m):.2f}")
    print(f"  5m ATR: mean={np.mean(valid_atr_5m):.2f}, median={np.median(valid_atr_5m):.2f}")

    # ================================================================
    # BUILD FVG ZONES AND MAPPINGS
    # ================================================================
    print("\n[BUILD] 5m FVG zones + time mappings...")
    fvg_zones = build_5m_fvg_zones(nq_5m, atr_5m, min_fvg_atr_mult=0.3)
    ts_5m_ns, ts_1m_ns, map_5m_to_1m, map_1m_to_prev_5m = build_time_mappings(nq_5m, df_1m)

    # ================================================================
    # 5m BASELINE (Config H++)
    # ================================================================
    print(f"\n{SEP}")
    print("5m BASELINE: Config H++ (stop*0.85)")
    print(SEP)

    trades_5m, baseline_m = run_5m_baseline(d, d_extra)
    print_metrics("5m H++ baseline", baseline_m)

    # ================================================================
    # SCAN ALL 1c RAW SIGNALS
    # ================================================================
    print(f"\n{SEP}")
    print("SCANNING 1c RAW SIGNALS (no SQ filter)")
    print(SEP)

    raw_signals = scan_1m_rejections(
        df_1m, feat_1m, fvg_zones,
        ts_5m_ns, ts_1m_ns, map_5m_to_1m, map_1m_to_prev_5m, d,
        min_body_ratio=0.50,
        min_stop_pts=0.0,
        tighten_factor=0.85,
        max_fvg_age_bars=200,
        signal_cooldown_1m=6,
    )

    print(f"  Total raw 1c signals: {len(raw_signals)}")

    # ================================================================
    # COMPUTE SQ FOR ALL SIGNALS (pre-filter for non-SQ filters)
    # ================================================================
    # First, apply all non-SQ filters to get the ~393 that reach SQ
    print(f"\n[FILTER] Applying non-SQ filters to identify signals reaching SQ check...")

    params = d["params"]
    atr_arr_5m = d["atr_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]

    et_1m = df_1m.index.tz_convert("US/Eastern")
    et_frac_1m = (et_1m.hour + et_1m.minute / 60.0).values
    session_regime = params.get("session_regime", {})

    pre_sq_signals = []
    pre_sq_stats = {"total": 0, "fail_obs": 0, "fail_news": 0, "fail_session": 0,
                    "fail_pm": 0, "fail_bias": 0, "fail_min_stop": 0, "fail_lunch": 0,
                    "reach_sq": 0}

    for sig in raw_signals:
        pre_sq_stats["total"] += 1
        j = sig["signal_1m_idx"]
        direction = sig["direction"]
        prev_5m = sig["prev_5m_idx"]
        et_frac = et_frac_1m[j]

        if 9.5 <= et_frac <= 10.0:
            pre_sq_stats["fail_obs"] += 1
            continue
        if news_blackout_arr is not None and news_blackout_arr[prev_5m]:
            pre_sq_stats["fail_news"] += 1
            continue
        if not (10.0 <= et_frac < 16.0):
            pre_sq_stats["fail_session"] += 1
            continue
        if direction == -1 and et_frac >= 14.0:
            pre_sq_stats["fail_pm"] += 1
            continue
        bd = bias_dir_arr[prev_5m]
        if direction == -np.sign(bd) and bd != 0:
            pre_sq_stats["fail_bias"] += 1
            continue

        # min_stop filter (1m ATR)
        stop_dist = sig["stop_dist"]
        if stop_dist < 5.0:
            pre_sq_stats["fail_min_stop"] += 1
            continue
        atr_ref = sig.get("atr_1m", 6.0)
        if atr_ref > 0 and (stop_dist / atr_ref) < 1.0:
            pre_sq_stats["fail_min_stop"] += 1
            continue

        if session_regime.get("enabled", False):
            lunch_s = session_regime.get("lunch_start", 12.5)
            lunch_e = session_regime.get("lunch_end", 13.0)
            lunch_mult = session_regime.get("lunch_mult", 0.0)
            if lunch_s <= et_frac < lunch_e and lunch_mult == 0.0:
                pre_sq_stats["fail_lunch"] += 1
                continue

        pre_sq_stats["reach_sq"] += 1
        pre_sq_signals.append(sig)

    print(f"  Pre-SQ filter stats:")
    for k, v in pre_sq_stats.items():
        print(f"    {k}: {v}")

    # ================================================================
    # COMPUTE SQ COMPONENTS
    # ================================================================
    print(f"\n[SQ] Computing SQ components for {len(pre_sq_signals)} signals...")
    sq_enriched = compute_sq_components_1c(
        pre_sq_signals, d, map_1m_to_prev_5m, df_1m, feat_1m
    )

    # ================================================================
    # STEP 1: Distribution analysis
    # ================================================================
    step1_distribution_analysis(sq_enriched)

    # Also compare vs 5m SQ distribution
    print(f"\n  5m SQ distribution for comparison:")
    sq_5m = d["signal_quality"]
    sq_5m_valid = sq_5m[~np.isnan(sq_5m)]
    if len(sq_5m_valid) > 0:
        print(f"    n={len(sq_5m_valid)}, mean={np.mean(sq_5m_valid):.3f}, "
              f"median={np.median(sq_5m_valid):.3f}, P25={np.percentile(sq_5m_valid, 25):.3f}, "
              f"P75={np.percentile(sq_5m_valid, 75):.3f}")
        for th in [0.50, 0.60, 0.68]:
            p = 100 * np.mean(sq_5m_valid >= th)
            print(f"    5m pass@{th}: {p:.1f}%")

    # ================================================================
    # STEP 2: Component breakdown
    # ================================================================
    step2_component_breakdown(sq_enriched)

    # ================================================================
    # STEP 3+4: Test variants
    # ================================================================
    results = step3_test_variants(
        sq_enriched, raw_signals, trades_5m, d, d_extra,
        df_1m, feat_1m, map_1m_to_prev_5m, baseline_m,
    )

    # ================================================================
    # STEP 5: Walk-forward for top configs
    # ================================================================
    step5_walk_forward(results, baseline_m)

    # ================================================================
    # STEP 6: Blocking analysis
    # ================================================================
    step6_blocking_analysis(results, trades_5m, baseline_m)

    # ================================================================
    # GRAND SUMMARY
    # ================================================================
    print(f"\n{SEP}")
    print("GRAND SUMMARY")
    print(SEP)

    # Find best by PPDD
    best_label = max(results.keys(), key=lambda k: results[k]["m"]["PPDD"])
    best = results[best_label]
    bm = best["m"]

    print(f"\n  5m Baseline: {baseline_m['trades']}t, R={baseline_m['R']:+.1f}, PPDD={baseline_m['PPDD']:.2f}, PF={baseline_m['PF']:.2f}")
    print(f"\n  Best config: {best_label}")
    print(f"    Combined: {bm['trades']}t, R={bm['R']:+.1f}, PPDD={bm['PPDD']:.2f}, PF={bm['PF']:.2f}, MaxDD={bm['MaxDD']:.1f}")
    print(f"    vs baseline: dR={bm['R']-baseline_m['R']:+.1f}, dPPDD={bm['PPDD']-baseline_m['PPDD']:+.2f}")
    print(f"    1c trades: {best['n1c']}, 1c R={best['m_1c']['R']:+.1f}, 1c avgR={best['m_1c']['avgR']:+.4f}")
    print(f"    5m trades in combo: {best['n5']} (blocked: {baseline_m['trades'] - best['n5']})")

    # Top 3 by different criteria
    print(f"\n  Top 3 by PPDD:")
    for rank, (lbl, data) in enumerate(sorted(results.items(), key=lambda x: x[1]["m"]["PPDD"], reverse=True)[:3], 1):
        m = data["m"]
        print(f"    #{rank}: {lbl:30s} | {m['trades']}t R={m['R']:+.1f} PPDD={m['PPDD']:.2f} PF={m['PF']:.2f}")

    print(f"\n  Top 3 by R:")
    for rank, (lbl, data) in enumerate(sorted(results.items(), key=lambda x: x[1]["m"]["R"], reverse=True)[:3], 1):
        m = data["m"]
        print(f"    #{rank}: {lbl:30s} | {m['trades']}t R={m['R']:+.1f} PPDD={m['PPDD']:.2f} PF={m['PF']:.2f}")

    elapsed = _time.perf_counter() - t_start
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"\n{SEP}")
    print("DONE")
    print(SEP)


if __name__ == "__main__":
    main()
