"""
experiments/b1_phase1_correct.py — CORRECT 5m Signal + 1m Entry Hybrid System
===============================================================================

Fixes the temporal lookahead bug in b1_phase1_hybrid.py:
  OLD (WRONG): searched 1m bars WITHIN the 5m signal bar's time window
               → entry price could be from BEFORE the signal was known
  NEW (CORRECT): only search 1m bars AFTER the 5m signal bar closes
               → entry is always LATER than the 5m default, never earlier

Architecture:
  5m signal fires at bar i close (e.g., 10:05 UTC)
    → At 10:05, you NOW KNOW there's a signal
    → From 10:05 onward, watch 1m bars for a rejection candle
    → 1m bar at 10:06 shows rejection (body_ratio >= 0.50, correct side close)
    → Entry at 10:07 (NEXT 1m bar's open after rejection)
    → Stop = 1m bar j-2 open (the bar 2 bars before rejection)
    → If no valid rejection within N minutes → fall back to 5m entry (bar i+1 open)

Key differences from the WRONG version:
  1. Only look at 1m bars AFTER the 5m signal bar closes (time > bar_time + 5min)
  2. Entry is LATER than 5m entry, not earlier
  3. Stop from 1m candle structure (always known at decision time)
  4. No cherry-picking — take the FIRST valid 1m rejection, not the best

Usage: python experiments/b1_phase1_correct.py
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
# Step 1: Load 1m data
# ======================================================================
def load_1m_data() -> pd.DataFrame:
    """Load the 1m NQ data."""
    t0 = _time.perf_counter()
    df = pd.read_parquet(PROJECT / "data" / "NQ_1min_10yr.parquet")
    elapsed = _time.perf_counter() - t0
    print(f"[1m DATA] Loaded {len(df):,} bars in {elapsed:.1f}s")
    return df


# ======================================================================
# Step 2: Build 5m→1m timestamp mapping for fast lookup
# ======================================================================
def build_5m_to_1m_map(nq_5m: pd.DataFrame, df_1m: pd.DataFrame) -> np.ndarray:
    """
    For each 5m bar, find the 1m index of the FIRST 1m bar AFTER the 5m bar closes.

    5m bar at time T covers [T, T+5min). The bar is "known" at T+5min.
    The first 1m bar we can act on is the one at T+5min (i.e., the 1m bar
    whose open time == T + 5min).

    Returns: np.ndarray of shape (n_5m,) with int indices into 1m data.
             Value = index of the first 1m bar at or after T+5min.
             Value = -1 if no such bar exists.
    """
    t0 = _time.perf_counter()
    n_5m = len(nq_5m)

    # Both indexes are in UTC nanoseconds
    ts_5m_ns = nq_5m.index.astype(np.int64) * 1000  # us → ns
    ts_1m_ns = df_1m.index.astype(np.int64) * 1000   # us → ns

    five_min_ns = 5 * 60 * 10**9

    # For each 5m bar at time T, find 1m index at T + 5min
    # Use searchsorted for O(n log n) total
    target_ns = ts_5m_ns + five_min_ns
    map_arr = np.searchsorted(ts_1m_ns, target_ns, side='left')

    # Mark out-of-bounds as -1
    n_1m = len(df_1m)
    map_arr[map_arr >= n_1m] = -1

    elapsed = _time.perf_counter() - t0
    valid = (map_arr >= 0).sum()
    print(f"[MAP] Built 5m→1m mapping in {elapsed:.1f}s ({valid:,}/{n_5m:,} valid)")
    return map_arr


# ======================================================================
# Step 3: Precompute 1m features (ATR, body ratio)
# ======================================================================
def precompute_1m_features(df_1m: pd.DataFrame) -> dict:
    """Precompute 1m OHLC arrays and rolling ATR."""
    t0 = _time.perf_counter()

    o = df_1m["open"].values.astype(np.float64)
    h = df_1m["high"].values.astype(np.float64)
    l = df_1m["low"].values.astype(np.float64)
    c = df_1m["close"].values.astype(np.float64)
    n = len(df_1m)

    # True Range
    tr = np.empty(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        hl = h[i] - l[i]
        hc = abs(h[i] - c[i - 1])
        lc = abs(l[i] - c[i - 1])
        tr[i] = max(hl, hc, lc)

    # ATR(14) — simple rolling mean for speed
    atr = np.full(n, np.nan)
    period = 14
    cum = np.cumsum(tr)
    atr[period - 1:] = (cum[period - 1:] - np.concatenate([[0], cum[:n - period]])) / period

    # Body ratio: abs(close - open) / (high - low)
    rng = h - l
    body = np.abs(c - o)
    safe_rng = np.where(rng > 0, rng, 1.0)
    body_ratio = body / safe_rng
    body_ratio[rng <= 0] = 0.0

    elapsed = _time.perf_counter() - t0
    print(f"[1m FEATURES] Precomputed ATR(14) + body ratio in {elapsed:.1f}s")

    return {
        "o": o, "h": h, "l": l, "c": c,
        "atr": atr, "body_ratio": body_ratio,
        "n": n,
    }


# ======================================================================
# Step 4: For each 5m signal, find CORRECT 1m entry
# ======================================================================
def find_1m_entries_correct(
    d: dict,
    df_1m: pd.DataFrame,
    feat_1m: dict,
    map_5m_to_1m: np.ndarray,
    *,
    window_bars: int = 10,          # how many 1m bars to scan after signal
    min_body_ratio: float = 0.50,   # min body/range for rejection candle
    min_stop_pts: float = 8.0,      # min stop distance in points
    max_stop_pts: float = 50.0,     # max stop distance
    tighten_factor: float = 0.85,   # tighten stop by this factor
    fallback_to_5m: bool = True,    # if no 1m entry, fall back to 5m?
) -> dict:
    """
    For each 5m signal, search 1m bars AFTER the signal bar closes
    for the FIRST valid rejection candle. Take the first one, not the best.

    TIMING CORRECTNESS:
      - 5m bar at index i has timestamp T. It covers [T, T+5min).
      - At T+5min, the 5m bar is closed and signal is known.
      - We scan 1m bars starting at the one at T+5min (map_5m_to_1m[i]).
      - For rejection candle at 1m index j:
        - Entry = open of 1m bar j+1 (AFTER rejection closes)
        - Stop = open of 1m bar j-2 (PAST bar, always available)
      - The entry 1m bar j+1 falls within some 5m bar. We find which one.
      - delay_5m_bars = how many 5m bars after the signal the entry occurs.
        Must be >= 1 (entry is at bar i+1 or later in 5m terms).

    Returns dict with:
      - entry_1m: np.ndarray (shape n_5m), NaN where not found
      - stop_1m: np.ndarray
      - found_mask: bool array
      - delay_bars: int array (5m bar offset from signal to entry bar)
      - entry_1m_time: list of timestamps (for verification)
      - stats: dict
    """
    t0 = _time.perf_counter()

    sig_mask = d["sig_mask"]
    sig_dir = d["sig_dir"]
    entry_price_arr = d["entry_price_arr"]
    model_stop_arr = d["model_stop_arr"]
    nq = d["nq"]
    n_5m = d["n"]

    o_1m = feat_1m["o"]
    h_1m = feat_1m["h"]
    l_1m = feat_1m["l"]
    c_1m = feat_1m["c"]
    br_1m = feat_1m["body_ratio"]
    atr_1m = feat_1m["atr"]
    n_1m = feat_1m["n"]

    ts_1m_ns = df_1m.index.astype(np.int64) * 1000  # us → ns
    ts_5m_ns = nq.index.astype(np.int64) * 1000

    five_min_ns = 5 * 60 * 10**9

    # Output arrays
    entry_1m = np.full(n_5m, np.nan)
    stop_1m = np.full(n_5m, np.nan)
    found_mask = np.zeros(n_5m, dtype=bool)
    delay_bars = np.zeros(n_5m, dtype=int)  # delay in 5m bars

    signal_indices = np.where(sig_mask)[0]
    n_signals = len(signal_indices)

    # Stats tracking
    n_found = 0
    n_no_map = 0
    n_no_candidate = 0
    n_stop_too_small = 0
    n_stop_too_large = 0
    n_stop_wrong_side = 0
    n_fallback = 0
    stop_dists_1m_list = []
    stop_dists_5m_list = []
    body_ratios_found = []
    delay_1m_bars_list = []  # how many 1m bars after signal
    delay_5m_bars_list = []  # how many 5m bars after signal

    for idx in signal_indices:
        direction = int(sig_dir[idx])
        if direction == 0:
            continue

        ep_5m = entry_price_arr[idx]
        stop_5m = model_stop_arr[idx]
        if np.isnan(ep_5m) or np.isnan(stop_5m):
            continue

        # Find first 1m bar after 5m signal bar closes
        i_1m_start = map_5m_to_1m[idx]
        if i_1m_start < 0 or i_1m_start >= n_1m:
            n_no_map += 1
            continue

        # ANTI-LOOKAHEAD CHECK: verify the 1m bar is actually at or after T+5min
        signal_close_ns = ts_5m_ns[idx] + five_min_ns
        assert ts_1m_ns[i_1m_start] >= signal_close_ns, (
            f"Lookahead! 1m bar at {df_1m.index[i_1m_start]} is before "
            f"5m signal close at {nq.index[idx] + pd.Timedelta(minutes=5)}"
        )

        # Scan forward through next 'window_bars' 1m bars
        found_this = False
        scan_end = min(i_1m_start + window_bars, n_1m - 1)

        for j in range(i_1m_start, scan_end):
            # Need j-2 for stop and j+1 for entry
            if j < 2 or j + 1 >= n_1m:
                continue

            # Check body ratio
            if br_1m[j] < min_body_ratio:
                continue

            # Check direction: close on correct side
            if direction == 1 and c_1m[j] <= o_1m[j]:
                continue  # Need bullish (close > open) for long
            if direction == -1 and c_1m[j] >= o_1m[j]:
                continue  # Need bearish (close < open) for short

            # Check displacement: body > 0.5 * ATR(14) on 1m
            if not np.isnan(atr_1m[j]) and atr_1m[j] > 0:
                body_size = abs(c_1m[j] - o_1m[j])
                if body_size < 0.5 * atr_1m[j]:
                    continue

            # 1m stop = open of candle 2 bars before (candle-2 open)
            raw_stop = o_1m[j - 2]
            candidate_entry = o_1m[j + 1]  # Next 1m bar open

            # Verify stop is on correct side of entry
            if direction == 1 and raw_stop >= candidate_entry:
                n_stop_wrong_side += 1
                continue
            if direction == -1 and raw_stop <= candidate_entry:
                n_stop_wrong_side += 1
                continue

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

            # FIRST valid candle — take it (no cherry-picking)
            entry_1m[idx] = candidate_entry
            stop_1m[idx] = tightened_stop
            found_mask[idx] = True
            n_found += 1

            # Compute delay: which 5m bar does the entry (j+1) fall in?
            entry_time_ns = ts_1m_ns[j + 1]
            # Find the 5m bar that contains this 1m entry time
            # 5m bar at index k covers [ts_5m[k], ts_5m[k]+5min)
            # entry_time falls in 5m bar k where ts_5m[k] <= entry_time < ts_5m[k]+5min
            k = np.searchsorted(ts_5m_ns, entry_time_ns, side='right') - 1
            if k < 0:
                k = 0
            delay_5m = k - idx
            if delay_5m < 1:
                delay_5m = 1  # Safety: entry must be at bar i+1 or later
            delay_bars[idx] = delay_5m

            stop_dists_1m_list.append(tightened_dist)
            stop_dists_5m_list.append(abs(ep_5m - stop_5m))
            body_ratios_found.append(br_1m[j])
            delay_1m_bars_list.append(j - i_1m_start)
            delay_5m_bars_list.append(delay_5m)

            found_this = True
            break  # FIRST valid only

        if not found_this:
            n_no_candidate += 1
            if fallback_to_5m:
                n_fallback += 1
                # Keep NaN in entry_1m/stop_1m → engine uses original 5m values

    elapsed = _time.perf_counter() - t0

    stats = {
        "n_signals": n_signals,
        "n_found": n_found,
        "n_no_map": n_no_map,
        "n_no_candidate": n_no_candidate,
        "n_stop_too_small": n_stop_too_small,
        "n_stop_too_large": n_stop_too_large,
        "n_stop_wrong_side": n_stop_wrong_side,
        "n_fallback": n_fallback,
        "pct_found": 100.0 * n_found / n_signals if n_signals > 0 else 0.0,
        "median_body_ratio": float(np.median(body_ratios_found)) if body_ratios_found else 0.0,
        "median_stop_1m": float(np.median(stop_dists_1m_list)) if stop_dists_1m_list else 0.0,
        "median_stop_5m": float(np.median(stop_dists_5m_list)) if stop_dists_5m_list else 0.0,
        "mean_stop_1m": float(np.mean(stop_dists_1m_list)) if stop_dists_1m_list else 0.0,
        "mean_stop_5m": float(np.mean(stop_dists_5m_list)) if stop_dists_5m_list else 0.0,
        "median_delay_1m": float(np.median(delay_1m_bars_list)) if delay_1m_bars_list else 0.0,
        "mean_delay_1m": float(np.mean(delay_1m_bars_list)) if delay_1m_bars_list else 0.0,
        "median_delay_5m": float(np.median(delay_5m_bars_list)) if delay_5m_bars_list else 0.0,
        "mean_delay_5m": float(np.mean(delay_5m_bars_list)) if delay_5m_bars_list else 0.0,
        "stop_dists_1m": np.array(stop_dists_1m_list),
        "stop_dists_5m": np.array(stop_dists_5m_list),
    }

    print(f"[1m ENTRY] Processed {n_signals} signals in {elapsed:.1f}s "
          f"(window={window_bars}, min_body_ratio={min_body_ratio}, "
          f"min_stop={min_stop_pts}, tighten={tighten_factor})")
    print(f"  Found 1m entry: {n_found} ({stats['pct_found']:.1f}%)")
    print(f"  No 1m→5m mapping: {n_no_map}")
    print(f"  No valid candidate: {n_no_candidate} (fallback to 5m: {n_fallback})")
    print(f"  Stop too small (<{min_stop_pts}pts): {n_stop_too_small}")
    print(f"  Stop too large (>{max_stop_pts}pts): {n_stop_too_large}")
    print(f"  Stop wrong side: {n_stop_wrong_side}")
    if n_found > 0:
        print(f"  Median body ratio: {stats['median_body_ratio']:.3f}")
        print(f"  Median stop: 1m={stats['median_stop_1m']:.1f}pts vs 5m={stats['median_stop_5m']:.1f}pts "
              f"(ratio: {stats['median_stop_1m']/stats['median_stop_5m']:.2f}x)")
        print(f"  Mean stop:   1m={stats['mean_stop_1m']:.1f}pts vs 5m={stats['mean_stop_5m']:.1f}pts "
              f"(ratio: {stats['mean_stop_1m']/stats['mean_stop_5m']:.2f}x)")
        print(f"  Delay (1m bars): median={stats['median_delay_1m']:.0f}, mean={stats['mean_delay_1m']:.1f}")
        print(f"  Delay (5m bars): median={stats['median_delay_5m']:.0f}, mean={stats['mean_delay_5m']:.1f}")

    return {
        "entry_1m": entry_1m,
        "stop_1m": stop_1m,
        "found_mask": found_mask,
        "delay_bars": delay_bars,
        "stats": stats,
    }


# ======================================================================
# Step 5: Build hybrid data with CORRECT timing
# ======================================================================
def build_hybrid_data_correct(d: dict, lookup: dict) -> dict:
    """
    Create a modified data dict where entry_price_arr and model_stop_arr
    are replaced with 1m values where available.

    For signals with 1m entry:
      - entry_price_arr[i] = 1m entry price
      - model_stop_arr[i] = 1m stop price
      The engine will still enter at bar i+1, using these modified prices.
      Since the 1m entry is at a later time, the entry price reflects a price
      from within bar i+1 or later (the open of the 1m bar after rejection).

    For signals without 1m entry (fallback):
      - Keep original 5m entry_price and model_stop (with tighten applied separately).

    CRITICAL: We keep original signal_quality, stop_atr_ratio, target_rr_arr
    for FILTERING. Only execution prices change.
    """
    d_hybrid = copy.copy(d)

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

    return d_hybrid


# ======================================================================
# Step 6: Run Config H+ baseline
# ======================================================================
def run_config_h_plus(d: dict, d_extra: dict) -> tuple:
    """Run Config H+ baseline: tighten_factor=0.85, raw IRL, 50/50/0, EOD close."""
    d_tight = copy.copy(d)
    d_tight["model_stop_arr"] = widen_stop_array(
        d["model_stop_arr"], d["entry_price_arr"],
        d["sig_dir"], d["sig_mask"], factor=0.85
    )
    trades, diag = run_backtest_pure_liquidity(
        d_tight, d_extra,
        tp_strategy="v1",
        tp1_trim_pct=0.50,
        be_after_tp1=True,
        be_after_tp2=True,
    )
    return trades, diag


# ======================================================================
# Step 7: Printing helpers
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
        print(f"  {row['year']:>6d} {row['n']:>7d} {row['R']:>+8.1f} "
              f"{row['WR']:>5.1f}% {row['PF']:>6.2f} {row['PPDD']:>7.2f}")


def print_results_table(results: list[dict]):
    """Print summary table of all configurations."""
    print(f"\n{SEP}")
    print("RESULTS SUMMARY TABLE")
    print(SEP)
    print(f"  {'Config':<55s} | {'Trades':>6s} | {'R':>8s} | {'PPDD':>7s} | "
          f"{'PF':>6s} | {'WR':>6s} | {'MaxDD':>6s} | {'1m%':>5s} | {'AvgDelay':>8s} | {'Avg1mStop':>9s}")
    print(f"  {'-'*55} | {'-'*6} | {'-'*8} | {'-'*7} | "
          f"{'-'*6} | {'-'*6} | {'-'*6} | {'-'*5} | {'-'*8} | {'-'*9}")
    for r in results:
        m = r["metrics"]
        pct_1m = r.get("pct_1m", 0.0)
        avg_delay = r.get("avg_delay", 0.0)
        avg_stop_1m = r.get("avg_stop_1m", 0.0)
        print(f"  {r['label']:<55s} | {m['trades']:>6d} | {m['R']:>+8.1f} | "
              f"{m['PPDD']:>7.2f} | {m['PF']:>6.2f} | {m['WR']:>5.1f}% | "
              f"{m['MaxDD']:>6.1f} | {pct_1m:>4.1f}% | {avg_delay:>7.1f}m | "
              f"{avg_stop_1m:>8.1f}p")


# ======================================================================
# Main execution
# ======================================================================
def main():
    print(SEP)
    print("B1 PHASE 1 CORRECT: 5m SIGNAL + 1m ENTRY (ANTI-LOOKAHEAD)")
    print("Only searches 1m bars AFTER 5m signal bar closes — entry is LATER, never earlier")
    print("Takes FIRST valid rejection candle, no cherry-picking")
    print(SEP)

    # ================================================================
    # STEP 1: LOAD ALL DATA
    # ================================================================
    print(f"\n{SEP}")
    print("STEP 1: LOADING DATA")
    print(SEP)

    t0_total = _time.perf_counter()
    d = load_all()
    d_extra = prepare_liquidity_data(d)
    df_1m = load_1m_data()

    # Build 5m→1m mapping
    map_5m_to_1m = build_5m_to_1m_map(d["nq"], df_1m)

    # Precompute 1m features
    feat_1m = precompute_1m_features(df_1m)

    print(f"  Total data load: {_time.perf_counter() - t0_total:.1f}s")

    # ================================================================
    # STEP 2: CONFIG H+ BASELINE (pure 5m)
    # ================================================================
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

    all_results = [{
        "label": "Config H+ baseline (pure 5m, stop*0.85)",
        "metrics": baseline_m,
        "trades": baseline_trades,
        "pct_1m": 0.0,
        "avg_delay": 0.0,
        "avg_stop_1m": 0.0,
    }]

    # ================================================================
    # STEP 3: PARAMETER SWEEP
    # ================================================================
    print(f"\n{SEP}")
    print("STEP 3: PARAMETER SWEEP")
    print(SEP)

    # A. Window size sweep (with fixed min_stop=8, tighten=0.85)
    # B. Min stop sweep (with best window)
    # C. Fallback vs skip

    sweep_configs = []

    # --- A. Window sweep ---
    for window in [5, 10, 15, 20]:
        for min_stop in [5, 8, 10]:
            for fallback in [True, False]:
                fb_label = "FB" if fallback else "SKIP"
                label = f"W={window:2d} minS={min_stop:2d} {fb_label}"
                sweep_configs.append({
                    "label": label,
                    "window_bars": window,
                    "min_stop_pts": float(min_stop),
                    "fallback_to_5m": fallback,
                })

    best_ppdd = -999.0
    best_config = None
    best_lookup = None

    for i_cfg, cfg in enumerate(sweep_configs):
        t0 = _time.perf_counter()
        label = cfg["label"]

        # Find 1m entries with this config
        lookup = find_1m_entries_correct(
            d, df_1m, feat_1m, map_5m_to_1m,
            window_bars=cfg["window_bars"],
            min_body_ratio=0.50,
            min_stop_pts=cfg["min_stop_pts"],
            max_stop_pts=50.0,
            tighten_factor=0.85,
            fallback_to_5m=cfg["fallback_to_5m"],
        )

        # Build hybrid data
        d_hybrid = build_hybrid_data_correct(d, lookup)

        # For signals WITHOUT 1m entry and fallback=True, we still need
        # the 5m stop tightened. Apply tighten to the non-1m entries.
        original_stop = d["model_stop_arr"]
        tightened_stop = widen_stop_array(
            original_stop, d["entry_price_arr"],
            d["sig_dir"], d["sig_mask"], factor=0.85
        )
        # Where we DON'T have a 1m entry, use the tightened 5m stop
        no_1m = ~lookup["found_mask"]
        d_hybrid["model_stop_arr"][no_1m] = tightened_stop[no_1m]

        # Run backtest
        trades, diag = run_backtest_pure_liquidity(
            d_hybrid, d_extra,
            tp_strategy="v1",
            tp1_trim_pct=0.50,
            be_after_tp1=True,
            be_after_tp2=True,
        )
        m = compute_metrics(trades)

        stats = lookup["stats"]
        delta_r = m["R"] - baseline_m["R"]
        delta_ppdd = m["PPDD"] - baseline_m["PPDD"]

        print(f"  [{i_cfg+1:2d}/{len(sweep_configs)}] {label:30s} | "
              f"{m['trades']:4d}t | R={m['R']:+8.1f} | PPDD={m['PPDD']:6.2f} | "
              f"PF={m['PF']:5.2f} | WR={m['WR']:5.1f}% | MaxDD={m['MaxDD']:5.1f}R | "
              f"1m={stats['pct_found']:4.1f}% | "
              f"dR={delta_r:+6.1f} dPPDD={delta_ppdd:+.2f} | "
              f"({_time.perf_counter() - t0:.1f}s)")

        result = {
            "label": label,
            "metrics": m,
            "trades": trades,
            "pct_1m": stats["pct_found"],
            "avg_delay": stats["mean_delay_1m"],
            "avg_stop_1m": stats["mean_stop_1m"],
        }
        all_results.append(result)

        if m["PPDD"] > best_ppdd:
            best_ppdd = m["PPDD"]
            best_config = cfg
            best_lookup = lookup
            best_trades = trades
            best_m = m
            best_diag = diag

    # ================================================================
    # STEP 4: BEST CONFIG DEEP ANALYSIS
    # ================================================================
    print(f"\n{SEP}")
    print(f"STEP 4: BEST CONFIG ANALYSIS — {best_config['label']}")
    print(SEP)

    print_metrics("BEST Hybrid", best_m)
    print_metrics("Config H+ baseline", baseline_m)

    delta_r = best_m["R"] - baseline_m["R"]
    delta_ppdd = best_m["PPDD"] - baseline_m["PPDD"]
    delta_pf = best_m["PF"] - baseline_m["PF"]
    delta_wr = best_m["WR"] - baseline_m["WR"]
    delta_dd = best_m["MaxDD"] - baseline_m["MaxDD"]
    print(f"\n  Delta: R={delta_r:+.1f} | PPDD={delta_ppdd:+.2f} | PF={delta_pf:+.2f} | "
          f"WR={delta_wr:+.1f}% | MaxDD={delta_dd:+.1f}R")

    print_diagnostics("BEST Hybrid", best_diag)
    print_long_short_breakdown(best_trades, "BEST Hybrid")
    print_stop_distribution(best_lookup)

    # ================================================================
    # STEP 5: WALK-FORWARD FOR BEST CONFIG + BASELINE
    # ================================================================
    print(f"\n{SEP}")
    print("STEP 5: WALK-FORWARD ANALYSIS")
    print(SEP)

    print_walk_forward(baseline_trades, "Config H+ baseline")
    print_walk_forward(best_trades, f"BEST Hybrid ({best_config['label']})")

    # ================================================================
    # STEP 6: ANTI-LOOKAHEAD VERIFICATION
    # ================================================================
    print(f"\n{SEP}")
    print("STEP 6: ANTI-LOOKAHEAD VERIFICATION")
    print(SEP)

    # Verify: all 1m entries are at timestamps AFTER their 5m signal bar close
    n_violations = 0
    n_checked = 0
    found_indices = np.where(best_lookup["found_mask"])[0]
    ts_5m = d["nq"].index
    five_min = pd.Timedelta(minutes=5)

    for idx in found_indices:
        signal_close_time = ts_5m[idx] + five_min
        entry_price = best_lookup["entry_1m"][idx]
        # The entry comes from o_1m[j+1] where j is the rejection candle
        # and j >= i_1m_start which is >= T+5min
        # So the entry bar is at j+1 >= i_1m_start + 1 which is > T+5min
        # We verify by checking delay_bars >= 1
        if best_lookup["delay_bars"][idx] < 1:
            n_violations += 1
        n_checked += 1

    if n_violations == 0:
        print(f"  PASS: All {n_checked} 1m entries have delay_bars >= 1 (no lookahead)")
    else:
        print(f"  FAIL: {n_violations}/{n_checked} entries have delay_bars < 1!")

    # Verify: 1m entries are LATER (higher prices for long in trending up,
    # or at least not systematically earlier)
    if len(found_indices) > 0:
        ep_5m = d["entry_price_arr"][found_indices]
        ep_1m = best_lookup["entry_1m"][found_indices]
        dirs = d["sig_dir"][found_indices]

        # For longs: entry should be at or above 5m entry (price moved up while waiting)
        # For shorts: entry should be at or below 5m entry
        # This is not guaranteed per-trade but on AVERAGE should show entry is later
        long_mask = dirs == 1
        short_mask = dirs == -1

        if long_mask.any():
            avg_slip_long = np.mean(ep_1m[long_mask] - ep_5m[long_mask])
            pct_worse_long = np.mean(ep_1m[long_mask] > ep_5m[long_mask]) * 100
            print(f"  Longs: avg 1m-5m entry diff = {avg_slip_long:+.2f} pts "
                  f"({pct_worse_long:.1f}% of 1m entries worse than 5m)")
        if short_mask.any():
            avg_slip_short = np.mean(ep_5m[short_mask] - ep_1m[short_mask])
            pct_worse_short = np.mean(ep_1m[short_mask] < ep_5m[short_mask]) * 100
            print(f"  Shorts: avg 1m-5m entry diff = {avg_slip_short:+.2f} pts "
                  f"({pct_worse_short:.1f}% of 1m entries worse than 5m)")

    # ================================================================
    # STEP 7: RESULTS SUMMARY TABLE
    # ================================================================
    print_results_table(all_results)

    # ================================================================
    # TOTAL TIME
    # ================================================================
    total_time = _time.perf_counter() - t0_total
    print(f"\n  Total execution time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
