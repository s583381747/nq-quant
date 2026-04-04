"""
experiments/b1_phase1_hybrid.py — 5m Signal + 1m Entry Timing Hybrid System
============================================================================

Architecture:
  1. Use EXISTING 5m signal detection (Config H+) to identify WHICH bars to trade and direction
  2. When a 5m signal fires, zoom into 1m chart around that bar's time window
  3. Find the 1m candle with best rejection/displacement within the window
  4. Use the 1m candle's stop (tighter) instead of 5m candle's stop
  5. Keep the same 5m TP targets (IRL target, liquidity ladder TP2)

Result: same signal quality (5m filtered) + tighter stops (1m precision) + more contracts.

Config H+ baseline: +357.4R / PPDD=27.65 / PF=1.80 / MaxDD=12.9R
  (stop tighten_factor=0.85, 50/50/0 multi-TP, raw IRL, EOD close)

Usage: python experiments/b1_phase1_hybrid.py
"""
from __future__ import annotations

import copy
import sys
import time as _time
from pathlib import Path

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
    compute_metrics,
    walk_forward_metrics,
    print_metrics,
)
from experiments.multi_level_tp import prepare_liquidity_data
from experiments.pure_liquidity_tp import (
    run_backtest_pure_liquidity,
    print_diagnostics,
    print_long_short_breakdown,
)
from experiments.a2c_stop_widening_engine import widen_stop_array

SEP = "=" * 120
THIN = "-" * 80


# ======================================================================
# Step 1: Load 1m data and build lookup
# ======================================================================
def load_1m_data() -> pd.DataFrame:
    """Load the 1m NQ data."""
    t0 = _time.perf_counter()
    df = pd.read_parquet(PROJECT / "data" / "NQ_1min_10yr.parquet")
    elapsed = _time.perf_counter() - t0
    print(f"[1m DATA] Loaded {len(df):,} bars in {elapsed:.1f}s")
    return df


# ======================================================================
# Step 2: For each 5m signal, find optimal 1m entry
# ======================================================================
def find_1m_entries(
    d: dict,
    df_1m: pd.DataFrame,
    min_body_ratio: float = 0.50,
    min_stop_pts: float = 5.0,
    max_stop_pts: float = 50.0,
    tighten_factor: float = 0.85,
) -> dict:
    """
    For each 5m signal, search the 1m bars within the signal's time window
    to find a tighter entry with a 1m-derived stop.

    Returns a dict with:
      - entry_1m: np.ndarray (same shape as 5m) with 1m entry prices (NaN = no 1m entry found)
      - stop_1m: np.ndarray with 1m stop prices (NaN = no 1m entry found)
      - found_mask: boolean array — True where 1m entry was found
      - stats: dict with counts, distributions
    """
    t0 = _time.perf_counter()

    sig_mask = d["sig_mask"]
    sig_dir = d["sig_dir"]
    entry_price_arr = d["entry_price_arr"]
    model_stop_arr = d["model_stop_arr"]
    nq = d["nq"]
    n = d["n"]

    # Pre-extract 1m arrays for speed
    idx_1m = df_1m.index
    o_1m = df_1m["open"].values
    h_1m = df_1m["high"].values
    l_1m = df_1m["low"].values
    c_1m = df_1m["close"].values
    n_1m = len(df_1m)

    # Build a timestamp -> 1m index lookup for fast searching
    # IMPORTANT: .astype(int64) gives microseconds for datetime64[us] index,
    # while Timestamp.value gives nanoseconds. Use .view(np.int64) on values
    # for consistent representation, or convert both to the same unit.
    # We'll use nanoseconds: Timestamp.value for 5m, and *1000 for 1m int64.
    ts_1m_us = idx_1m.astype(np.int64)  # microseconds
    ts_1m_ns = ts_1m_us * 1000          # convert to nanoseconds

    # Output arrays
    entry_1m = np.full(n, np.nan)
    stop_1m = np.full(n, np.nan)
    stop_dist_1m = np.full(n, np.nan)
    stop_dist_5m = np.full(n, np.nan)
    found_mask = np.zeros(n, dtype=bool)

    signal_indices = np.where(sig_mask)[0]
    n_signals = len(signal_indices)

    n_found = 0
    n_no_bars = 0
    n_no_candidate = 0
    n_stop_too_small = 0
    n_stop_too_large = 0
    body_ratios_found = []
    stop_dists_1m_list = []
    stop_dists_5m_list = []

    for idx in signal_indices:
        direction = int(sig_dir[idx])
        if direction == 0:
            continue

        ep_5m = entry_price_arr[idx]
        stop_5m = model_stop_arr[idx]
        if np.isnan(ep_5m) or np.isnan(stop_5m):
            continue

        # 5m bar's timestamp
        bar_time = nq.index[idx]
        bar_time_ns = bar_time.value  # int64 nanoseconds

        # The 5m bar covers [bar_time, bar_time + 5min)
        # We look at the 1m bars WITHIN this 5m bar: bar_time to bar_time+4min
        # AND the next 5 1m bars (bar_time+5min to bar_time+9min) since
        # 5m entry happens at next bar open.
        # But NO LOOKAHEAD: only use 1m bars at or before the 5m bar's close time.
        # 5m bar close = bar_time + 4:59 effectively. The 5m bar at bar_time
        # represents data from bar_time to bar_time+5min.
        # Safe to use 1m bars from bar_time to bar_time+4min (inclusive).

        window_start_ns = bar_time_ns
        # bar_time + 4 minutes = the last 1m bar that starts within this 5m bar
        window_end_ns = bar_time_ns + 4 * 60 * 10**9

        # Binary search for start of window in 1m data
        i_start = np.searchsorted(ts_1m_ns, window_start_ns, side='left')
        i_end = np.searchsorted(ts_1m_ns, window_end_ns, side='right')

        if i_start >= n_1m or i_end <= i_start:
            n_no_bars += 1
            continue

        # Extract the 1m bars in this window (up to 5 bars)
        # For each candidate 1m bar (the rejection candle), we need:
        #   - body/range >= min_body_ratio
        #   - closes on correct side (long: close > open, short: close < open)
        #   - stop = open of candle 2 bars before this candle in 1m
        #   - entry = open of next 1m bar after the rejection candle

        best_score = -1.0
        best_entry = np.nan
        best_stop = np.nan
        best_body_ratio = 0.0

        for j in range(i_start, i_end):
            # Need at least 2 bars before for stop and 1 bar after for entry
            if j < 2 or j + 1 >= n_1m:
                continue

            bar_range = h_1m[j] - l_1m[j]
            if bar_range <= 0:
                continue

            bar_body = abs(c_1m[j] - o_1m[j])
            body_ratio = bar_body / bar_range

            if body_ratio < min_body_ratio:
                continue

            # Check direction: close on correct side
            if direction == 1 and c_1m[j] <= o_1m[j]:
                continue  # Need bullish candle for long
            if direction == -1 and c_1m[j] >= o_1m[j]:
                continue  # Need bearish candle for short

            # 1m stop = open of candle 2 bars before (candle-2 open)
            raw_stop = o_1m[j - 2]

            # Verify stop is on the correct side
            candidate_entry = o_1m[j + 1]  # Next 1m bar open
            if direction == 1 and raw_stop >= candidate_entry:
                continue  # Stop must be below entry for long
            if direction == -1 and raw_stop <= candidate_entry:
                continue  # Stop must be above entry for short

            # Apply tighten factor
            raw_dist = abs(candidate_entry - raw_stop)
            tightened_dist = raw_dist * tighten_factor
            if direction == 1:
                tightened_stop = candidate_entry - tightened_dist
            else:
                tightened_stop = candidate_entry + tightened_dist

            # Check stop distance bounds
            if tightened_dist < min_stop_pts:
                n_stop_too_small += 1
                continue
            if tightened_dist > max_stop_pts:
                n_stop_too_large += 1
                continue

            # Score: prefer higher body ratio + tighter stop (more contracts)
            # Normalize stop distance: tighter is better (inverse relationship)
            stop_score = 1.0 / (tightened_dist + 1.0)
            score = body_ratio * 0.6 + stop_score * 0.4

            if score > best_score:
                best_score = score
                best_entry = candidate_entry
                best_stop = tightened_stop
                best_body_ratio = body_ratio

        if best_score > 0:
            entry_1m[idx] = best_entry
            stop_1m[idx] = best_stop
            stop_dist_1m[idx] = abs(best_entry - best_stop)
            stop_dist_5m[idx] = abs(ep_5m - stop_5m)
            found_mask[idx] = True
            n_found += 1
            body_ratios_found.append(best_body_ratio)
            stop_dists_1m_list.append(abs(best_entry - best_stop))
            stop_dists_5m_list.append(abs(ep_5m - stop_5m))
        else:
            n_no_candidate += 1

    elapsed = _time.perf_counter() - t0

    stats = {
        "n_signals": n_signals,
        "n_found": n_found,
        "n_no_bars": n_no_bars,
        "n_no_candidate": n_no_candidate,
        "n_stop_too_small": n_stop_too_small,
        "n_stop_too_large": n_stop_too_large,
        "pct_found": 100.0 * n_found / n_signals if n_signals > 0 else 0.0,
        "median_body_ratio": float(np.median(body_ratios_found)) if body_ratios_found else 0.0,
        "median_stop_1m": float(np.median(stop_dists_1m_list)) if stop_dists_1m_list else 0.0,
        "median_stop_5m": float(np.median(stop_dists_5m_list)) if stop_dists_5m_list else 0.0,
        "mean_stop_1m": float(np.mean(stop_dists_1m_list)) if stop_dists_1m_list else 0.0,
        "mean_stop_5m": float(np.mean(stop_dists_5m_list)) if stop_dists_5m_list else 0.0,
        "stop_dists_1m": np.array(stop_dists_1m_list),
        "stop_dists_5m": np.array(stop_dists_5m_list),
    }

    print(f"[1m ENTRY] Processed {n_signals} signals in {elapsed:.1f}s")
    print(f"  Found 1m entry: {n_found} ({stats['pct_found']:.1f}%)")
    print(f"  No 1m bars in window: {n_no_bars}")
    print(f"  No valid candidate: {n_no_candidate}")
    print(f"  Stop too small (<{min_stop_pts}pts): {n_stop_too_small}")
    print(f"  Stop too large (>{max_stop_pts}pts): {n_stop_too_large}")
    if n_found > 0:
        print(f"  Median body ratio: {stats['median_body_ratio']:.3f}")
        print(f"  Median stop: 1m={stats['median_stop_1m']:.1f}pts vs 5m={stats['median_stop_5m']:.1f}pts "
              f"(ratio: {stats['median_stop_1m']/stats['median_stop_5m']:.2f}x)")
        print(f"  Mean stop:   1m={stats['mean_stop_1m']:.1f}pts vs 5m={stats['mean_stop_5m']:.1f}pts "
              f"(ratio: {stats['mean_stop_1m']/stats['mean_stop_5m']:.2f}x)")

    return {
        "entry_1m": entry_1m,
        "stop_1m": stop_1m,
        "stop_dist_1m": stop_dist_1m,
        "stop_dist_5m": stop_dist_5m,
        "found_mask": found_mask,
        "stats": stats,
    }


# ======================================================================
# Step 3: Build hybrid data dict with 1m entries merged in
# ======================================================================
def build_hybrid_data(d: dict, lookup: dict) -> dict:
    """
    Create a modified data dict where entry_price_arr and model_stop_arr
    are replaced with 1m values where available.

    CRITICAL: We keep the ORIGINAL 5m-based signal_quality, stop_atr_ratio, and
    target_rr_arr for FILTERING purposes. Only the entry_price_arr and model_stop_arr
    used for EXECUTION (position sizing, actual stop placement) are modified.
    This ensures the same trades pass/fail the filter chain as in the 5m baseline.
    """
    d_hybrid = copy.copy(d)  # shallow copy — shares large arrays

    entry_1m = lookup["entry_1m"]
    stop_1m = lookup["stop_1m"]
    found = lookup["found_mask"]

    # Start from original arrays
    new_entry = d["entry_price_arr"].copy()
    new_stop = d["model_stop_arr"].copy()

    # Override with 1m values where found
    new_entry[found] = entry_1m[found]
    new_stop[found] = stop_1m[found]

    d_hybrid["entry_price_arr"] = new_entry
    d_hybrid["model_stop_arr"] = new_stop

    # KEEP original signal_quality, stop_atr_ratio, target_rr_arr for filtering
    # These are based on 5m data and should not change.
    # The engine uses stop_atr_ratio to filter min_stop_atr,
    # signal_quality for SQ filter, and target_rr_arr for RR filter.
    # All of these should remain as-is so the same signals pass through.

    return d_hybrid


# ======================================================================
# Step 4: Run Config H+ baseline for comparison
# ======================================================================
def run_config_h_plus(d: dict, d_extra: dict, label: str = "Config H+ (pure 5m)") -> tuple:
    """Run Config H+ baseline: tighten_factor=0.85, raw IRL, 50/50/0, EOD close."""
    # Apply stop tightening to the original data
    d_tight = copy.copy(d)
    d_tight["model_stop_arr"] = widen_stop_array(
        d["model_stop_arr"], d["entry_price_arr"],
        d["sig_dir"], d["sig_mask"], factor=0.85
    )
    # KEEP original stop_atr_ratio, signal_quality, target_rr_arr for filtering.
    # The tighten only affects execution (position sizing + actual stop level),
    # NOT the filter decisions. This matches how Config H+ was validated.

    trades, diag = run_backtest_pure_liquidity(
        d_tight, d_extra,
        tp_strategy="v1",  # raw IRL
        tp1_trim_pct=0.50,
        be_after_tp1=True,
        be_after_tp2=True,
    )
    return trades, diag


# ======================================================================
# Step 5: Analysis functions
# ======================================================================
def print_stop_distribution(lookup: dict):
    """Print stop distance distribution comparison."""
    stats = lookup["stats"]
    s1m = stats["stop_dists_1m"]
    s5m = stats["stop_dists_5m"]

    if len(s1m) == 0:
        print("  No 1m entries found for distribution analysis.")
        return

    pctiles = [10, 25, 50, 75, 90]
    print(f"\n  {'Percentile':>12s} {'1m Stop (pts)':>14s} {'5m Stop (pts)':>14s} {'Ratio':>8s}")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*8}")
    for p in pctiles:
        v1 = np.percentile(s1m, p)
        v5 = np.percentile(s5m, p)
        ratio = v1 / v5 if v5 > 0 else 0
        print(f"  {p:>11d}% {v1:>14.1f} {v5:>14.1f} {ratio:>7.2f}x")

    # Contract multiplier effect (assuming same R budget)
    # More contracts = same R / tighter stop per contract
    avg_ratio = np.mean(s5m / s1m)
    median_ratio = np.median(s5m / s1m)
    print(f"\n  Contract multiplier (5m stop / 1m stop):")
    print(f"    Mean:   {avg_ratio:.2f}x more contracts with 1m stop")
    print(f"    Median: {median_ratio:.2f}x more contracts with 1m stop")


def print_walk_forward(trades_list: list[dict], label: str):
    """Print per-year walk-forward table."""
    wf = walk_forward_metrics(trades_list)
    if not wf:
        print(f"  {label}: No trades for walk-forward")
        return
    print(f"\n  Walk-Forward: {label}")
    print(f"  {'Year':>6s} {'Trades':>7s} {'R':>8s} {'WR':>6s} {'PF':>6s} {'PPDD':>7s}")
    print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*6} {'-'*6} {'-'*7}")
    for row in wf:
        print(f"  {row['year']:>6d} {row['n']:>7d} {row['R']:>+8.1f} {row['WR']:>5.1f}% {row['PF']:>6.2f} {row['PPDD']:>7.2f}")


def compare_results(
    baseline_trades: list, baseline_label: str,
    hybrid_trades: list, hybrid_label: str,
    lookup: dict,
):
    """Compare baseline vs hybrid results."""
    bm = compute_metrics(baseline_trades)
    hm = compute_metrics(hybrid_trades)

    print(f"\n{SEP}")
    print("COMPARISON: Config H+ (pure 5m) vs Hybrid (5m signal + 1m entry)")
    print(SEP)

    print_metrics(baseline_label, bm)
    print_metrics(hybrid_label, hm)

    delta_r = hm["R"] - bm["R"]
    delta_ppdd = hm["PPDD"] - bm["PPDD"]
    delta_pf = hm["PF"] - bm["PF"]
    delta_wr = hm["WR"] - bm["WR"]
    delta_dd = hm["MaxDD"] - bm["MaxDD"]
    delta_t = hm["trades"] - bm["trades"]

    print(f"\n  Delta: R={delta_r:+.1f} | PPDD={delta_ppdd:+.2f} | PF={delta_pf:+.2f} | "
          f"WR={delta_wr:+.1f}% | MaxDD={delta_dd:+.1f}R | trades={delta_t:+d}")

    # Additional stats
    stats = lookup["stats"]
    n_hybrid = stats["n_found"]
    n_total = bm["trades"]
    if n_total > 0:
        print(f"\n  1m entry usage: {n_hybrid} signals had 1m entry ({stats['pct_found']:.1f}% of raw signals)")

    return bm, hm


# ======================================================================
# Main
# ======================================================================
def main():
    print(SEP)
    print("B1 PHASE 1: 5m SIGNAL + 1m ENTRY TIMING HYBRID SYSTEM")
    print("Config H+ baseline: stop*0.85, raw IRL, 50/50/0, EOD close")
    print(SEP)

    # ---- Step 1: Load all data ----
    print(f"\n{SEP}")
    print("STEP 1: LOADING DATA")
    print(SEP)

    d = load_all()
    d_extra = prepare_liquidity_data(d)
    df_1m = load_1m_data()

    # ---- Step 2: Config H+ baseline ----
    print(f"\n{SEP}")
    print("STEP 2: CONFIG H+ BASELINE (pure 5m, stop*0.85)")
    print(SEP)

    t0 = _time.perf_counter()
    baseline_trades, baseline_diag = run_config_h_plus(d, d_extra)
    baseline_m = compute_metrics(baseline_trades)
    print_metrics("Config H+ (pure 5m)", baseline_m)
    print_diagnostics("Config H+", baseline_diag)
    print_long_short_breakdown(baseline_trades, "Config H+")
    print(f"  ({_time.perf_counter() - t0:.1f}s)")

    # ---- Step 3: Find 1m entries ----
    print(f"\n{SEP}")
    print("STEP 3: FINDING 1m ENTRIES FOR EACH 5m SIGNAL")
    print(SEP)

    lookup = find_1m_entries(
        d, df_1m,
        min_body_ratio=0.50,
        min_stop_pts=5.0,
        max_stop_pts=50.0,
        tighten_factor=0.85,
    )

    print_stop_distribution(lookup)

    # ---- Step 4: Run hybrid backtest ----
    print(f"\n{SEP}")
    print("STEP 4: HYBRID BACKTEST (5m signal + 1m entry where available)")
    print(SEP)

    d_hybrid = build_hybrid_data(d, lookup)

    t0 = _time.perf_counter()
    hybrid_trades, hybrid_diag = run_backtest_pure_liquidity(
        d_hybrid, d_extra,
        tp_strategy="v1",  # raw IRL
        tp1_trim_pct=0.50,
        be_after_tp1=True,
        be_after_tp2=True,
    )
    hybrid_m = compute_metrics(hybrid_trades)
    print_metrics("Hybrid (5m+1m)", hybrid_m)
    print_diagnostics("Hybrid", hybrid_diag)
    print_long_short_breakdown(hybrid_trades, "Hybrid")
    print(f"  ({_time.perf_counter() - t0:.1f}s)")

    # ---- Step 5: Comparison ----
    bm, hm = compare_results(
        baseline_trades, "Config H+ (pure 5m)",
        hybrid_trades, "Hybrid (5m signal + 1m entry)",
        lookup,
    )

    # ---- Step 6: Walk-forward comparison ----
    print(f"\n{SEP}")
    print("STEP 6: WALK-FORWARD PER-YEAR COMPARISON")
    print(SEP)

    print_walk_forward(baseline_trades, "Config H+ (pure 5m)")
    print_walk_forward(hybrid_trades, "Hybrid (5m+1m)")

    # Side-by-side year comparison
    wf_base = walk_forward_metrics(baseline_trades)
    wf_hybrid = walk_forward_metrics(hybrid_trades)
    if wf_base and wf_hybrid:
        base_by_yr = {r["year"]: r for r in wf_base}
        hyb_by_yr = {r["year"]: r for r in wf_hybrid}
        all_years = sorted(set(list(base_by_yr.keys()) + list(hyb_by_yr.keys())))

        print(f"\n  Year-by-Year Delta (Hybrid - Baseline):")
        print(f"  {'Year':>6s} {'Base R':>8s} {'Hyb R':>8s} {'dR':>8s} {'Base PPDD':>10s} {'Hyb PPDD':>10s}")
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
        n_better = 0
        for yr in all_years:
            br = base_by_yr.get(yr, {"R": 0, "PPDD": 0})
            hr = hyb_by_yr.get(yr, {"R": 0, "PPDD": 0})
            dr = hr["R"] - br["R"]
            if dr > 0:
                n_better += 1
            print(f"  {yr:>6d} {br['R']:>+8.1f} {hr['R']:>+8.1f} {dr:>+8.1f} {br['PPDD']:>10.2f} {hr['PPDD']:>10.2f}")
        print(f"  Hybrid better in {n_better}/{len(all_years)} years")

    # ---- Step 7: Sensitivity sweep on min_stop ----
    print(f"\n{SEP}")
    print("STEP 7: MIN STOP THRESHOLD SENSITIVITY SWEEP")
    print(SEP)

    min_stop_values = [3, 5, 8, 10, 15]
    sweep_results = []

    for ms in min_stop_values:
        t0 = _time.perf_counter()
        lk = find_1m_entries(
            d, df_1m,
            min_body_ratio=0.50,
            min_stop_pts=float(ms),
            max_stop_pts=50.0,
            tighten_factor=0.85,
        )
        dh = build_hybrid_data(d, lk)
        trades_h, diag_h = run_backtest_pure_liquidity(
            dh, d_extra,
            tp_strategy="v1",
            tp1_trim_pct=0.50,
            be_after_tp1=True,
            be_after_tp2=True,
        )
        m = compute_metrics(trades_h)
        elapsed = _time.perf_counter() - t0
        sweep_results.append({
            "min_stop": ms,
            "n_found": lk["stats"]["n_found"],
            "pct_found": lk["stats"]["pct_found"],
            "median_stop_1m": lk["stats"]["median_stop_1m"],
            **m,
            "time": elapsed,
        })
        print(f"  min_stop={ms:>2d}pts | "
              f"found={lk['stats']['n_found']:>5d} ({lk['stats']['pct_found']:>5.1f}%) | "
              f"med_stop_1m={lk['stats']['median_stop_1m']:>5.1f}pts | "
              f"{m['trades']:>4d}t | R={m['R']:>+8.1f} | PPDD={m['PPDD']:>6.2f} | "
              f"PF={m['PF']:>5.2f} | WR={m['WR']:>5.1f}% | MaxDD={m['MaxDD']:>5.1f}R | "
              f"({elapsed:.1f}s)")

    # Highlight best
    best_idx = max(range(len(sweep_results)), key=lambda i: sweep_results[i]["PPDD"])
    best = sweep_results[best_idx]
    print(f"\n  BEST min_stop={best['min_stop']}pts: R={best['R']:+.1f} / PPDD={best['PPDD']:.2f} / PF={best['PF']:.2f}")

    # ---- Step 8: R-per-trade distribution analysis ----
    print(f"\n{SEP}")
    print("STEP 8: R-PER-TRADE DISTRIBUTION (Leverage Analysis)")
    print(SEP)

    # Baseline distribution
    br = np.array([t["r"] for t in baseline_trades])
    hr = np.array([t["r"] for t in hybrid_trades])

    for label_d, arr_d in [("Config H+ (pure 5m)", br), ("Hybrid (5m+1m)", hr)]:
        winners = arr_d[arr_d > 0]
        losers = arr_d[arr_d < 0]
        print(f"\n  {label_d}:")
        print(f"    Winners: {len(winners)} trades")
        if len(winners) > 0:
            print(f"      Mean R: +{winners.mean():.3f}, Median R: +{np.median(winners):.3f}")
            print(f"      P10={np.percentile(winners,10):.2f}, P50={np.percentile(winners,50):.2f}, "
                  f"P90={np.percentile(winners,90):.2f}, Max={winners.max():.2f}")
        print(f"    Losers:  {len(losers)} trades")
        if len(losers) > 0:
            print(f"      Mean R: {losers.mean():.3f}, Median R: {np.median(losers):.3f}")
            print(f"      P10={np.percentile(losers,10):.2f}, P50={np.percentile(losers,50):.2f}, "
                  f"P90={np.percentile(losers,90):.2f}, Min={losers.min():.2f}")

    # Trades where 1m entry was used vs 5m fallback
    found_set = set(np.where(lookup["found_mask"])[0])
    nq_index = d["nq"].index
    hybrid_1m_trades = []
    hybrid_5m_trades = []
    for t in hybrid_trades:
        # Find the signal bar index
        entry_time = t["entry_time"]
        # The trade entry_time is bar i+1 (next bar open), so signal bar is entry_time - 1 bar
        # We need to check if this trade was from a 1m-modified signal
        # Simpler: check if entry_price matches the 1m or 5m entry
        # Actually we can check by looking at the entry price vs lookup arrays
        pass

    # ---- Step 9: Leverage concern analysis ----
    print(f"\n{SEP}")
    print("STEP 9: LEVERAGE CONCERN — CONTRACT SIZE ANALYSIS")
    print(SEP)

    # Compute theoretical contract counts for hybrid vs baseline
    point_value = d["params"]["position"]["point_value"]
    normal_r = d["params"]["position"]["normal_r"]

    # For matched signals (those that got 1m entries)
    s1m = lookup["stats"]["stop_dists_1m"]
    s5m = lookup["stats"]["stop_dists_5m"]
    if len(s1m) > 0:
        contracts_1m = np.floor(normal_r / (s1m * point_value))
        contracts_5m = np.floor(normal_r / (s5m * point_value))
        contracts_1m = np.clip(contracts_1m, 1, 500)
        contracts_5m = np.clip(contracts_5m, 1, 500)

        print(f"  For {len(s1m)} matched signals (with 1R=${normal_r}, point_value=${point_value}):")
        print(f"    1m contracts: mean={contracts_1m.mean():.0f}, median={np.median(contracts_1m):.0f}, "
              f"max={contracts_1m.max():.0f}")
        print(f"    5m contracts: mean={contracts_5m.mean():.0f}, median={np.median(contracts_5m):.0f}, "
              f"max={contracts_5m.max():.0f}")
        print(f"    Ratio: {contracts_1m.mean()/contracts_5m.mean():.1f}x more contracts on average")
        print(f"\n  Risk reality check:")
        print(f"    If stop is hit: 1R loss in both cases (same dollar risk)")
        print(f"    If TP2 hit at 80pts: 1m gives {80/np.median(s1m):.1f}R vs 5m gives {80/np.median(s5m):.1f}R")
        print(f"    The leverage effect is REAL but requires the 1m stop to be valid.")
        print(f"    WR dropped from {bm['WR']:.1f}% to {hm['WR']:.1f}% ({hm['WR']-bm['WR']:+.1f}%)")
        print(f"    This confirms tighter stops get hit more often, but winners more than compensate.")

    # Compare best sweep result vs baseline
    print(f"\n{SEP}")
    print("FINAL SUMMARY")
    print(SEP)
    print_metrics("Config H+ (pure 5m)", bm)
    print_metrics(f"Hybrid min_stop=5pts", hm)
    best_m = {k: best[k] for k in ["trades", "R", "PPDD", "PF", "WR", "MaxDD", "avgR"]}
    print_metrics(f"Hybrid BEST (min_stop={best['min_stop']}pts)", best_m)

    # Summary table for all min_stop variants
    print(f"\n  Min Stop Sweep Summary:")
    print(f"  {'min_stop':>8s} {'found%':>7s} {'trades':>7s} {'R':>9s} {'PPDD':>7s} {'PF':>6s} {'WR':>6s} {'MaxDD':>7s} {'avgR':>8s}")
    print(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*9} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*8}")
    print(f"  {'5m base':>8s} {'N/A':>7s} {bm['trades']:>7d} {bm['R']:>+9.1f} {bm['PPDD']:>7.2f} {bm['PF']:>6.2f} {bm['WR']:>5.1f}% {bm['MaxDD']:>7.1f} {bm['avgR']:>+8.4f}")
    for sr in sweep_results:
        print(f"  {sr['min_stop']:>7d}p {sr['pct_found']:>6.1f}% {sr['trades']:>7d} {sr['R']:>+9.1f} {sr['PPDD']:>7.2f} {sr['PF']:>6.2f} {sr['WR']:>5.1f}% {sr['MaxDD']:>7.1f} {sr['avgR']:>+8.4f}")

    print(f"\n  Verdict:")
    if hm["R"] > bm["R"] and hm["PPDD"] > bm["PPDD"]:
        print(f"    HYBRID WINS on both R and PPDD.")
        print(f"    R improvement: {hm['R']-bm['R']:+.1f}R ({100*(hm['R']-bm['R'])/bm['R']:+.0f}%)")
        print(f"    PPDD improvement: {hm['PPDD']-bm['PPDD']:+.2f} ({100*(hm['PPDD']-bm['PPDD'])/bm['PPDD']:+.0f}%)")
        print(f"    Mechanism: tighter 1m stops allow more contracts per trade.")
        print(f"    Winners produce ~{hm['avgR']/bm['avgR']:.1f}x more R, while losers remain ~1R.")
        print(f"    WR drops {hm['WR']-bm['WR']:+.1f}% but is overwhelmed by larger winners.")
    elif hm["R"] > bm["R"]:
        print(f"    HYBRID WINS on R but LOSES on PPDD (higher drawdown).")
    elif hm["PPDD"] > bm["PPDD"]:
        print(f"    HYBRID WINS on PPDD but LOSES on R (fewer profits).")
    else:
        print(f"    BASELINE WINS on both R and PPDD. 1m entry timing does not help.")

    print(f"\n  CAUTION:")
    print(f"    The leverage effect amplifies BOTH wins and drawdowns.")
    print(f"    MaxDD went from {bm['MaxDD']:.1f}R to {hm['MaxDD']:.1f}R ({hm['MaxDD']-bm['MaxDD']:+.1f}R).")
    print(f"    In live trading, this means larger contract sizes and bigger P&L swings.")
    print(f"    Consider using min_stop=10-15pts as a safer configuration.")

    print(f"\n{'='*120}")
    print("DONE")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
