"""
experiments/audit_u2_hostile.py — HOSTILE ADVERSARIAL AUDIT of U2 Limit Order FVG System
========================================================================================

Claims: +3,144R / PPDD=164 / PF=2.45 from 4,180 trades.
avgR +0.75 per trade. This is 2.2x better per-trade edge than Config H++ AND 3.8x more trades.

This audit runs diagnostic checks D1-D10 to find what's wrong.

CRITICAL TESTS:
  D2: FVG zone creation timing (same-bar fill lookahead)
  D3: Multi-day holding / overnight positions
  D4: Trade duration distribution
  D5: Fill rate sensitivity (penetration requirement)
  D6: Largest trades deep dive
  D7: Year-by-year consistency
  D8: Shorts-only check
  D9: RANDOM ENTRY COMPARISON (the killer test)
  D10: R multiplier / leverage effect analysis

Usage: python experiments/audit_u2_hostile.py
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from collections import Counter
from copy import deepcopy

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
    _compute_grade_fast,
    compute_metrics,
    walk_forward_metrics,
    print_metrics,
)
from experiments.multi_level_tp import prepare_liquidity_data
from experiments.pure_liquidity_tp import _compute_tp_v1_raw_irl
from experiments.a2c_stop_widening_engine import widen_stop_array
from experiments.u2_limit_order_fvg import (
    run_limit_order_backtest,
    FVGZone,
)
from features.fvg import detect_fvg

SEP = "=" * 120
THIN = "-" * 80


def main():
    t_start = _time.perf_counter()

    print(SEP)
    print("HOSTILE ADVERSARIAL AUDIT — U2 Limit Order FVG System")
    print("Checking every possible source of inflated R")
    print(SEP)

    # ================================================================
    # LOAD DATA
    # ================================================================
    print("\n[LOAD] Loading data...")
    d = load_all()
    d_extra = prepare_liquidity_data(d)
    nq = d["nq"]
    n = d["n"]
    params = d["params"]
    print(f"  {n} bars, {nq.index[0]} to {nq.index[-1]}")

    # Determine approximate years
    date_range = (nq.index[-1] - nq.index[0]).days / 365.25
    print(f"  ~{date_range:.1f} years of data")

    # ================================================================
    # RUN THE SYSTEM — use the config that supposedly gets +3,144R
    # We'll try several configs and find the best one
    # ================================================================
    print(f"\n{SEP}")
    print("[STEP 1] Run U2 Limit Order Backtest — Best Config Search")
    print(SEP)

    # Try the configs most likely to produce the claimed results
    configs_to_try = [
        ("A1_B1_C3_D1", "A1", "B1", 200, 0.3),
        ("A1_B1_C4_D1", "A1", "B1", 500, 0.3),
        ("A2_B1_C3_D1", "A2", "B1", 200, 0.3),
        ("A2_B1_C4_D1", "A2", "B1", 500, 0.3),
        ("A1_B2_C3_D1", "A1", "B2", 200, 0.3),
        ("A1_B2_C4_D1", "A1", "B2", 500, 0.3),
        ("A1_B1_C2_D1", "A1", "B1", 100, 0.3),
    ]

    best_cfg_name = None
    best_trades = None
    best_stats = None
    best_m = None
    best_r = -9999

    print(f"  {'Config':20s} | {'Trades':>6s} | {'R':>9s} | {'PPDD':>7s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>8s} | {'AvgStop':>7s}")
    print(f"  {THIN}")

    for cfg_name, a, b, c, d_val in configs_to_try:
        trades_l, stats_l = run_limit_order_backtest(
            d, d_extra,
            stop_strategy=a,
            min_stop_mode=b,
            max_fvg_age=c,
            fvg_size_mult=d_val,
            block_pm_shorts=True,
        )
        m = compute_metrics(trades_l)
        avg_stop = np.mean([abs(t["entry_price"] - t["stop_price"]) for t in trades_l]) if trades_l else 0
        print(f"  {cfg_name:20s} | {m['trades']:>6d} | {m['R']:>+9.1f} | {m['PPDD']:>7.2f} | "
              f"{m['PF']:>6.2f} | {m['WR']:>5.1f}% | {m['avgR']:>+8.4f} | {avg_stop:>7.1f}")

        if m['R'] > best_r:
            best_r = m['R']
            best_cfg_name = cfg_name
            best_trades = trades_l
            best_stats = stats_l
            best_m = m

    print(f"\n  >>> Using best config: {best_cfg_name} with R={best_m['R']:+.1f}, {best_m['trades']} trades")
    trades = best_trades
    stats = best_stats

    if not trades:
        print("  NO TRADES — audit cannot proceed.")
        return

    # Convert trades to DataFrame for analysis
    df_t = pd.DataFrame(trades)

    # ================================================================
    # D2: FVG ZONE CREATION TIMING (LOOKAHEAD CHECK)
    # ================================================================
    print(f"\n{SEP}")
    print("[D2] FVG Zone Creation Timing — Same-Bar Fill Check")
    print(SEP)

    # The FVG detection uses shift(1) in detect_fvg.
    # In the backtest, zone.birth_bar = i when bull_mask[i] fires.
    # The fill check requires zone.birth_bar < i (strict less than).
    # So birth_bar is set to the bar where the shifted signal appears.
    #
    # FVG at candles c1=j-1, c2=j, c3=j+1 → signal at c2 → shifted to c3 = j+1
    # So bull_mask[j+1] = True → zone.birth_bar = j+1
    # Fill check: zone.birth_bar >= i → skip. So fill can happen at i = j+2 (earliest).
    # That means the FVG is filled 1 bar AFTER it becomes known. CORRECT.
    #
    # But wait — let's verify empirically by checking fill_wait_bars

    print(f"\n  Zone stats:")
    print(f"    Created: {stats['zones_created']}")
    print(f"    Filled:  {stats['zones_filled']} ({stats['fill_rate_pct']:.1f}%)")
    print(f"    Avg fill wait: {stats['avg_fill_wait_bars']:.1f} bars ({stats['avg_fill_wait_bars']*5:.0f} min)")
    print(f"    Median fill wait: {stats['median_fill_wait_bars']:.1f} bars ({stats['median_fill_wait_bars']*5:.0f} min)")

    # Check: are there any fills at wait=0? That would be same-bar-as-creation.
    # We need to re-run with instrumented code to get per-trade fill_wait
    # Instead, let's compute from trades: entry_time vs fvg creation time
    # We don't have zone birth time in trades, but we have fill_wait from stats
    if stats['median_fill_wait_bars'] < 1:
        print("  >>> WARNING: Median fill wait < 1 bar! Possible same-bar fill!")
    else:
        print(f"  Fill wait looks OK (median >= 1 bar)")

    # Double-check: reconstruct and count wait=1 fills (borderline)
    fvg_df = detect_fvg(nq)
    bull_mask_arr = fvg_df["fvg_bull"].values
    bear_mask_arr = fvg_df["fvg_bear"].values

    # Count how many FVG signals appear
    total_fvg_signals = int(bull_mask_arr.sum()) + int(bear_mask_arr.sum())
    print(f"  Total raw FVG signals (after shift): {total_fvg_signals}")

    # ================================================================
    # D3: MULTI-DAY HOLDING — OVERNIGHT POSITIONS
    # ================================================================
    print(f"\n{SEP}")
    print("[D3] Multi-Day Holding / Overnight Positions")
    print(SEP)

    df_t["entry_ts"] = pd.to_datetime(df_t["entry_time"])
    df_t["exit_ts"] = pd.to_datetime(df_t["exit_time"])
    df_t["duration_min"] = (df_t["exit_ts"] - df_t["entry_ts"]).dt.total_seconds() / 60.0
    df_t["entry_date"] = df_t["entry_ts"].dt.date
    df_t["exit_date"] = df_t["exit_ts"].dt.date
    df_t["is_overnight"] = df_t["entry_date"] != df_t["exit_date"]

    overnight_count = df_t["is_overnight"].sum()
    overnight_r = df_t.loc[df_t["is_overnight"], "r"].sum() if overnight_count > 0 else 0.0

    print(f"  Total trades: {len(df_t)}")
    print(f"  Overnight (entry date != exit date): {overnight_count} ({100*overnight_count/len(df_t):.1f}%)")
    if overnight_count > 0:
        print(f"    Overnight R contribution: {overnight_r:+.1f}")
        print(f"    >>> WARNING: {overnight_count} overnight positions! EOD close should prevent this.")
        # Show some examples
        print(f"    Sample overnight trades:")
        ov = df_t[df_t["is_overnight"]].head(10)
        for _, row in ov.iterrows():
            print(f"      Entry: {row['entry_ts']} -> Exit: {row['exit_ts']} "
                  f"(dur={row['duration_min']:.0f}min, R={row['r']:+.3f}, reason={row['reason']})")
    else:
        print(f"  PASS: No overnight positions detected.")

    print(f"\n  Trade duration stats:")
    print(f"    Min:    {df_t['duration_min'].min():.0f} min")
    print(f"    P5:     {df_t['duration_min'].quantile(0.05):.0f} min")
    print(f"    P25:    {df_t['duration_min'].quantile(0.25):.0f} min")
    print(f"    Median: {df_t['duration_min'].median():.0f} min")
    print(f"    P75:    {df_t['duration_min'].quantile(0.75):.0f} min")
    print(f"    P95:    {df_t['duration_min'].quantile(0.95):.0f} min")
    print(f"    Max:    {df_t['duration_min'].max():.0f} min")

    max_dur_idx = df_t['duration_min'].idxmax()
    longest = df_t.loc[max_dur_idx]
    print(f"\n  Longest trade: Entry={longest['entry_ts']}, Exit={longest['exit_ts']}, "
          f"dur={longest['duration_min']:.0f}min, R={longest['r']:+.3f}, reason={longest['reason']}")

    # ================================================================
    # D4: TRADE DURATION DISTRIBUTION & EOD CLOSE
    # ================================================================
    print(f"\n{SEP}")
    print("[D4] Trade Duration Distribution & EOD Close Analysis")
    print(SEP)

    # Trades entering very late (after 15:30) and exiting at EOD
    if 'entry_ts' in df_t.columns:
        entry_et = df_t["entry_ts"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern") if df_t["entry_ts"].dt.tz is None else df_t["entry_ts"].dt.tz_convert("US/Eastern")
        df_t["entry_hour_et"] = entry_et.dt.hour + entry_et.dt.minute / 60.0
    else:
        df_t["entry_hour_et"] = 12.0  # fallback

    late_entries = df_t[df_t["entry_hour_et"] >= 15.5]
    print(f"  Entries after 15:30 ET: {len(late_entries)} ({100*len(late_entries)/len(df_t):.1f}%)")
    if len(late_entries) > 0:
        late_r = late_entries["r"].sum()
        print(f"    R from late entries: {late_r:+.1f}")
        print(f"    Avg R: {late_entries['r'].mean():+.4f}")

    # Duration buckets
    duration_buckets = [
        ("<5min", 0, 5),
        ("5-15min", 5, 15),
        ("15-30min", 15, 30),
        ("30-60min", 30, 60),
        ("1-2h", 60, 120),
        ("2-4h", 120, 240),
        ("4h+", 240, 99999),
    ]
    print(f"\n  {'Duration':12s} | {'Count':>6s} | {'%':>6s} | {'R':>9s} | {'avgR':>8s}")
    print(f"  {THIN}")
    for label, lo, hi in duration_buckets:
        mask = (df_t["duration_min"] >= lo) & (df_t["duration_min"] < hi)
        cnt = mask.sum()
        r_sum = df_t.loc[mask, "r"].sum()
        avg_r = df_t.loc[mask, "r"].mean() if cnt > 0 else 0.0
        pct = 100 * cnt / len(df_t) if len(df_t) > 0 else 0
        print(f"  {label:12s} | {cnt:>6d} | {pct:>5.1f}% | {r_sum:>+9.1f} | {avg_r:>+8.4f}")

    # Exit reason distribution
    print(f"\n  Exit reason distribution:")
    reasons = Counter(t["reason"] for t in trades)
    total_t = len(trades)
    print(f"  {'Reason':15s} | {'Count':>6s} | {'%':>6s} | {'R':>9s} | {'avgR':>8s}")
    print(f"  {THIN}")
    for reason, cnt in reasons.most_common():
        r_sum = sum(t["r"] for t in trades if t["reason"] == reason)
        avg_r = r_sum / cnt if cnt > 0 else 0.0
        print(f"  {reason:15s} | {cnt:>6d} | {100*cnt/total_t:>5.1f}% | {r_sum:>+9.1f} | {avg_r:>+8.4f}")

    # ================================================================
    # D5: FILL RATE SENSITIVITY — PENETRATION REQUIREMENT
    # ================================================================
    print(f"\n{SEP}")
    print("[D5] Fill Rate Sensitivity — Penetration Requirements")
    print("  Testing: what if limit fill requires price to go THROUGH the limit?")
    print(SEP)

    # We'll re-implement a simplified fill check with penetration requirements
    # by post-filtering trades based on bar data
    # For each trade, check if the fill bar's low actually went below entry - N ticks

    atr_arr = d["atr_arr"]
    o_arr = d["o"]
    h_arr = d["h"]
    l_arr = d["l"]
    c_arr = d["c"]

    # We need the fill bar index for each trade. We can find it from entry_time.
    nq_idx = nq.index
    trade_fill_bars = []
    for t in trades:
        entry_time = t["entry_time"]
        # Find the bar index
        try:
            if hasattr(entry_time, 'tz') and entry_time.tz is not None:
                bar_idx = nq_idx.get_loc(entry_time)
            else:
                bar_idx = nq_idx.get_loc(pd.Timestamp(entry_time, tz='UTC'))
        except KeyError:
            bar_idx = nq_idx.get_indexer([entry_time], method='nearest')[0]
        trade_fill_bars.append(bar_idx)

    df_t["fill_bar_idx"] = trade_fill_bars

    # For longs: fill at fvg_top. Price must go below fvg_top for fill.
    # Penetration test: bar_low <= entry_price - penetration
    penetrations = [0.0, 0.25, 0.50, 1.0, 2.0, 4.0]

    print(f"\n  {'Penetration':>12s} | {'Surviving':>9s} | {'R':>9s} | {'avgR':>8s} | {'PPDD':>7s} | {'PF':>6s}")
    print(f"  {THIN}")

    for pen in penetrations:
        surviving = []
        for j, t in enumerate(trades):
            bar_i = trade_fill_bars[j]
            if bar_i < 0 or bar_i >= n:
                continue
            entry_p = t["entry_price"]
            if t["dir"] == 1:
                # Long: fill at limit = entry_p. Need low <= entry_p - pen
                if l_arr[bar_i] <= entry_p - pen:
                    surviving.append(t)
            else:
                # Short: fill at limit = entry_p. Need high >= entry_p + pen
                if h_arr[bar_i] >= entry_p + pen:
                    surviving.append(t)

        m_pen = compute_metrics(surviving)
        pct_surviving = 100 * len(surviving) / len(trades) if trades else 0
        print(f"  {pen:>11.2f}pt | {len(surviving):>5d} ({pct_surviving:>4.1f}%) | "
              f"{m_pen['R']:>+9.1f} | {m_pen['avgR']:>+8.4f} | {m_pen['PPDD']:>7.2f} | {m_pen['PF']:>6.2f}")

    # ================================================================
    # D6: LARGEST TRADES DEEP DIVE
    # ================================================================
    print(f"\n{SEP}")
    print("[D6] Top 20 Trades by R — Deep Dive")
    print(SEP)

    sorted_trades = sorted(trades, key=lambda t: t["r"], reverse=True)
    top20 = sorted_trades[:20]

    print(f"\n  {'#':>3s} | {'R':>8s} | {'Dir':>4s} | {'Entry':>20s} | {'Exit':>20s} | {'Dur':>6s} | "
          f"{'EntryP':>10s} | {'Stop':>10s} | {'TP1':>10s} | {'TP2':>10s} | {'ExitP':>10s} | {'StopDist':>8s} | {'Reason':>10s}")
    print(f"  {'-'*3}-+-{'-'*8}-+-{'-'*4}-+-{'-'*20}-+-{'-'*20}-+-{'-'*6}-+-"
          f"{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")

    for rank, t in enumerate(top20, 1):
        entry_ts = pd.Timestamp(t["entry_time"])
        exit_ts = pd.Timestamp(t["exit_time"])
        dur_min = (exit_ts - entry_ts).total_seconds() / 60.0
        stop_dist = abs(t["entry_price"] - t["stop_price"])
        dir_str = "LONG" if t["dir"] == 1 else "SHRT"
        print(f"  {rank:>3d} | {t['r']:>+8.3f} | {dir_str:>4s} | {str(entry_ts)[:19]:>20s} | {str(exit_ts)[:19]:>20s} | "
              f"{dur_min:>5.0f}m | {t['entry_price']:>10.2f} | {t['stop_price']:>10.2f} | {t['tp1_price']:>10.2f} | "
              f"{t['tp2_price']:>10.2f} | {t['exit_price']:>10.2f} | {stop_dist:>8.1f} | {t['reason']:>10s}")

    # Sanity check: are there absurd R values?
    r_vals = np.array([t["r"] for t in trades])
    print(f"\n  R distribution:")
    print(f"    max={r_vals.max():.2f}, p99={np.percentile(r_vals,99):.2f}, p95={np.percentile(r_vals,95):.2f}")
    print(f"    p50={np.median(r_vals):.2f}")
    print(f"    p5={np.percentile(r_vals,5):.2f}, p1={np.percentile(r_vals,1):.2f}, min={r_vals.min():.2f}")

    # Check: trades with R > 5 — how much total R do they contribute?
    extreme = r_vals[r_vals > 5.0]
    print(f"    Trades with R > 5: {len(extreme)} ({100*len(extreme)/len(r_vals):.1f}%)")
    print(f"      Their total R: {extreme.sum():+.1f} ({100*extreme.sum()/r_vals.sum():.1f}% of total)")

    extreme2 = r_vals[r_vals > 3.0]
    print(f"    Trades with R > 3: {len(extreme2)} ({100*len(extreme2)/len(r_vals):.1f}%)")
    print(f"      Their total R: {extreme2.sum():+.1f} ({100*extreme2.sum()/r_vals.sum():.1f}% of total)")

    # Biggest single-trade R vs stop distance
    top5_trades = sorted_trades[:5]
    print(f"\n  Top 5 trades — stop distance analysis:")
    for rank, t in enumerate(top5_trades, 1):
        sd = abs(t["entry_price"] - t["stop_price"])
        atr_at_entry = atr_arr[trade_fill_bars[trades.index(t)]] if trades.index(t) < len(trade_fill_bars) else 30.0
        print(f"    #{rank}: R={t['r']:+.3f}, stop={sd:.1f}pts ({sd/atr_at_entry:.2f}*ATR), "
              f"dir={'LONG' if t['dir']==1 else 'SHORT'}, reason={t['reason']}")

    # ================================================================
    # D7: YEAR-BY-YEAR CONSISTENCY
    # ================================================================
    print(f"\n{SEP}")
    print("[D7] Year-by-Year Consistency Check")
    print(SEP)

    wf = walk_forward_metrics(trades)
    if wf:
        neg_years = sum(1 for yr in wf if yr["R"] < 0)
        print(f"\n  {'Year':>6s} | {'N':>5s} | {'R':>9s} | {'avgR':>8s} | {'WR':>6s} | {'PF':>6s} | {'PPDD':>7s}")
        print(f"  {THIN}")
        for yr in wf:
            avg_r_yr = yr["R"] / yr["n"] if yr["n"] > 0 else 0
            print(f"  {yr['year']:>6d} | {yr['n']:>5d} | {yr['R']:>+9.1f} | {avg_r_yr:>+8.4f} | "
                  f"{yr['WR']:>5.1f}% | {yr['PF']:>6.2f} | {yr['PPDD']:>7.2f}")
        print(f"\n  Negative years: {neg_years}/{len(wf)}")

        # Calculate coefficient of variation of yearly R
        yearly_r = np.array([yr["R"] for yr in wf])
        yearly_trades = np.array([yr["n"] for yr in wf])
        print(f"  Yearly R: mean={yearly_r.mean():.1f}, std={yearly_r.std():.1f}, CV={yearly_r.std()/yearly_r.mean():.2f}")
        print(f"  Yearly trades: mean={yearly_trades.mean():.0f}, std={yearly_trades.std():.0f}")

        # Check if most R comes from high-ATR years (COVID)
        print(f"\n  Market regime check:")
        for yr in wf:
            # Get ATR for this year
            yr_mask = pd.to_datetime(df_t["entry_time"]).dt.year == yr["year"]
            yr_bars = df_t.loc[yr_mask, "fill_bar_idx"].values
            yr_atr = np.nanmean(atr_arr[yr_bars]) if len(yr_bars) > 0 else 0
            yr_stop = np.mean([abs(t["entry_price"] - t["stop_price"]) for t in trades
                              if pd.Timestamp(t["entry_time"]).year == yr["year"]])
            print(f"    {yr['year']}: ATR={yr_atr:.1f}, avgStop={yr_stop:.1f}pts, R={yr['R']:+.1f}")

    # ================================================================
    # D8: SHORTS-ONLY CHECK
    # ================================================================
    print(f"\n{SEP}")
    print("[D8] Direction Breakdown — Shorts Check")
    print(SEP)

    longs = [t for t in trades if t["dir"] == 1]
    shorts = [t for t in trades if t["dir"] == -1]

    if longs:
        m_long = compute_metrics(longs)
        print(f"  LONGS:  {m_long['trades']}t | R={m_long['R']:+.1f} | avgR={m_long['avgR']:+.4f} | "
              f"WR={m_long['WR']:.1f}% | PF={m_long['PF']:.2f} | PPDD={m_long['PPDD']:.2f}")
    if shorts:
        m_short = compute_metrics(shorts)
        print(f"  SHORTS: {m_short['trades']}t | R={m_short['R']:+.1f} | avgR={m_short['avgR']:+.4f} | "
              f"WR={m_short['WR']:.1f}% | PF={m_short['PF']:.2f} | PPDD={m_short['PPDD']:.2f}")

    # What if we block all shorts?
    m_longs_only = compute_metrics(longs)
    print(f"\n  Longs-only R: {m_longs_only['R']:+.1f}")
    print(f"  Shorts R: {m_short['R'] if shorts else 0:+.1f}")
    print(f"  Shorts are {'destructive' if shorts and m_short['R'] < 0 else 'constructive'}")

    # ================================================================
    # D9: RANDOM ENTRY COMPARISON (THE KILLER TEST)
    # ================================================================
    print(f"\n{SEP}")
    print("[D9] RANDOM ENTRY COMPARISON — The Killer Test")
    print("  This separates leverage effect from genuine FVG edge.")
    print("  Generating random entries with SAME stop distances and TP logic.")
    print(SEP)

    # Collect statistics from real trades to replicate
    real_stop_dists = [abs(t["entry_price"] - t["stop_price"]) for t in trades]
    real_entry_bars = [trade_fill_bars[i] for i in range(len(trades))]
    real_dirs = [t["dir"] for t in trades]

    point_value = params["position"]["point_value"]
    commission_per_side = params["backtest"]["commission_per_side_micro"]
    normal_r = params["position"]["normal_r"]
    reduced_r = params["position"]["reduced_r"]
    a_plus_mult = params["grading"]["a_plus_size_mult"]

    # Strategy: for each real trade, we generate a random entry within the same session
    # window, with the SAME stop distance and SAME direction, and use the SAME TP logic.
    np.random.seed(42)
    n_random_trials = 5

    print(f"\n  Running {n_random_trials} random trials with {len(trades)} entries each...")

    random_results = []
    for trial in range(n_random_trials):
        np.random.seed(42 + trial)
        random_trades = []

        for idx, t in enumerate(trades):
            real_bar = real_entry_bars[idx]
            stop_dist = real_stop_dists[idx]
            direction = real_dirs[idx]

            # Random offset: shift entry bar by random amount within +/- 50 bars
            # (keep same session window by checking et_frac)
            offset = np.random.randint(-50, 51)
            rand_bar = real_bar + offset
            rand_bar = max(50, min(rand_bar, n - 100))

            # Use the random bar's price as entry
            entry_p = c_arr[rand_bar]  # entry at close of random bar

            if direction == 1:
                stop = entry_p - stop_dist
            else:
                stop = entry_p + stop_dist

            # Use same TP computation
            atr_val = atr_arr[rand_bar] if not np.isnan(atr_arr[rand_bar]) else 30.0

            if direction == 1:
                # Use _compute_tp_v1_raw_irl if possible
                irl_target = d_extra.get("reg_high_arr", h_arr)[rand_bar] if "reg_high_arr" in d_extra else entry_p + stop_dist * 2.0
                try:
                    tp1, tp2 = _compute_tp_v1_raw_irl(entry_p, stop, irl_target, rand_bar, d_extra, atr_val)
                except Exception:
                    tp1 = entry_p + stop_dist * 1.5
                    tp2 = entry_p + stop_dist * 3.0
            else:
                short_rr = params.get("dual_mode", {}).get("short_rr", 0.625)
                tp1 = entry_p - stop_dist * short_rr
                tp2 = 0.0

            # Simulate the trade forward
            contracts = max(1, int(normal_r / (stop_dist * point_value))) if stop_dist > 0 else 0
            if contracts <= 0:
                continue

            # Walk forward from entry to find exit
            trade_r = _simulate_trade_forward(
                rand_bar, direction, entry_p, stop, tp1, tp2,
                h_arr, l_arr, c_arr, n, d["et_frac_arr"],
                contracts, stop_dist, point_value, commission_per_side,
                is_multi_tp=(direction == 1),
                tp1_trim_pct=0.50,
            )
            random_trades.append({"r": trade_r, "dir": direction})

        m_rand = compute_metrics(random_trades)
        random_results.append(m_rand)
        print(f"    Trial {trial+1}: {m_rand['trades']}t | R={m_rand['R']:+.1f} | "
              f"avgR={m_rand['avgR']:+.4f} | WR={m_rand['WR']:.1f}% | PF={m_rand['PF']:.2f}")

    avg_random_r = np.mean([m["R"] for m in random_results])
    avg_random_avgr = np.mean([m["avgR"] for m in random_results])
    print(f"\n  Average random R: {avg_random_r:+.1f} (real: {best_m['R']:+.1f})")
    print(f"  Average random avgR: {avg_random_avgr:+.4f} (real: {best_m['avgR']:+.4f})")
    edge_ratio = best_m['R'] / avg_random_r if avg_random_r != 0 else float('inf')
    print(f"  Edge ratio (real/random): {edge_ratio:.2f}x")

    if avg_random_r > best_m['R'] * 0.5:
        print(f"  >>> CRITICAL: Random entries achieve >{50}% of real R!")
        print(f"  >>> This suggests the R is from leverage/stop structure, NOT from FVG edge!")
    elif avg_random_r > best_m['R'] * 0.25:
        print(f"  >>> WARNING: Random achieves >25% of real R. Significant leverage component.")
    else:
        print(f"  Random is <25% of real R. Genuine edge appears present.")

    # ================================================================
    # D10: R MULTIPLIER / LEVERAGE EFFECT ANALYSIS
    # ================================================================
    print(f"\n{SEP}")
    print("[D10] R Multiplier / Leverage Effect Analysis")
    print(SEP)

    stop_dists = np.array([abs(t["entry_price"] - t["stop_price"]) for t in trades])
    entry_prices = np.array([t["entry_price"] for t in trades])
    r_vals = np.array([t["r"] for t in trades])

    # Stop distance distribution
    print(f"\n  Stop distance distribution:")
    print(f"    Mean:   {stop_dists.mean():.1f} pts")
    print(f"    Median: {np.median(stop_dists):.1f} pts")
    print(f"    P5:     {np.percentile(stop_dists, 5):.1f} pts")
    print(f"    P95:    {np.percentile(stop_dists, 95):.1f} pts")
    print(f"    Min:    {stop_dists.min():.1f} pts")
    print(f"    Max:    {stop_dists.max():.1f} pts")

    # Contracts per trade
    risk_per_contract = stop_dists * point_value
    contracts = np.maximum(1, np.floor(normal_r / risk_per_contract)).astype(int)
    print(f"\n  Contracts per trade:")
    print(f"    Mean:   {contracts.mean():.1f}")
    print(f"    Median: {np.median(contracts):.0f}")
    print(f"    Max:    {contracts.max()}")
    print(f"    Min:    {contracts.min()}")

    # The key insight: smaller stops = more contracts = more leverage
    # Bucket by stop distance
    stop_buckets = [
        ("<5pts", 0, 5),
        ("5-10pts", 5, 10),
        ("10-15pts", 10, 15),
        ("15-20pts", 15, 20),
        ("20-30pts", 20, 30),
        ("30-50pts", 30, 50),
        ("50+pts", 50, 9999),
    ]

    print(f"\n  R by stop distance bucket:")
    print(f"  {'Bucket':12s} | {'Count':>6s} | {'%':>5s} | {'R':>9s} | {'avgR':>8s} | {'WR':>6s} | {'avgContr':>8s}")
    print(f"  {THIN}")
    for label, lo, hi in stop_buckets:
        mask = (stop_dists >= lo) & (stop_dists < hi)
        cnt = mask.sum()
        r_sum = r_vals[mask].sum() if cnt > 0 else 0
        avg_r = r_vals[mask].mean() if cnt > 0 else 0
        wr = 100 * (r_vals[mask] > 0).mean() if cnt > 0 else 0
        avg_c = contracts[mask].mean() if cnt > 0 else 0
        pct = 100 * cnt / len(trades)
        print(f"  {label:12s} | {cnt:>6d} | {pct:>4.1f}% | {r_sum:>+9.1f} | {avg_r:>+8.4f} | {wr:>5.1f}% | {avg_c:>8.1f}")

    # The real test: if we normalize stops to 20pts (median), how much R do we get?
    # This removes the leverage effect
    print(f"\n  NORMALIZED STOP TEST:")
    print(f"  What if all stops were 20pts? (removes leverage effect)")
    norm_stop = 20.0
    norm_contracts = max(1, int(normal_r / (norm_stop * point_value)))
    norm_r_total = 0.0
    for t in trades:
        # Recompute R with normalized stop
        sd_orig = abs(t["entry_price"] - t["stop_price"])
        if sd_orig > 0:
            # Scale: R = pnl / risk. If we change stop, risk changes, but pnl stays same IF trade still works.
            # Actually, different stop means different exit. So we can't simply rescale.
            # But we CAN ask: "what if the STOP was the same but contracts were fixed?"
            # Normalized R = (exit_price - entry_price) / norm_stop for longs
            if t["dir"] == 1:
                pts_gained = t["exit_price"] - t["entry_price"]
            else:
                pts_gained = t["entry_price"] - t["exit_price"]
            norm_r = (pts_gained * point_value * norm_contracts - commission_per_side * 2 * norm_contracts) / (norm_stop * point_value * norm_contracts)
            norm_r_total += norm_r

    print(f"  Normalized total R (20pt stops, {norm_contracts} contracts): {norm_r_total:+.1f}")
    print(f"  Original total R: {best_m['R']:+.1f}")
    print(f"  Ratio: {norm_r_total / best_m['R']:.2f}x" if best_m['R'] != 0 else "  N/A")

    # ================================================================
    # D_EXTRA: INTRA-BAR PRICE SEQUENCE CHECK (D1)
    # ================================================================
    print(f"\n{SEP}")
    print("[D1_EXTRA] Intra-Bar Price Sequence — TP Hit on Same Bar as Fill")
    print(SEP)

    # For each trade, check if TP1 was hit on the SAME bar as the fill
    same_bar_tp1 = 0
    same_bar_details = []
    for idx, t in enumerate(trades):
        bar_i = trade_fill_bars[idx]
        if bar_i < 0 or bar_i >= n:
            continue
        if t["dir"] == 1:
            # Long: TP1 hit if high >= tp1 on fill bar
            if h_arr[bar_i] >= t["tp1_price"] and t["tp1_price"] > 0:
                same_bar_tp1 += 1
                same_bar_details.append(t)
        else:
            # Short: TP1 hit if low <= tp1 on fill bar
            if l_arr[bar_i] <= t["tp1_price"] and t["tp1_price"] > 0:
                same_bar_tp1 += 1
                same_bar_details.append(t)

    print(f"  Trades where TP1 is reachable on SAME BAR as fill: {same_bar_tp1} ({100*same_bar_tp1/len(trades):.1f}%)")
    if same_bar_tp1 > 0:
        same_bar_r = sum(t["r"] for t in same_bar_details)
        print(f"  R from these trades: {same_bar_r:+.1f}")
        print(f"  >>> These trades have AMBIGUOUS intra-bar sequencing.")
        print(f"  >>> On a 5m bar, we don't know if fill happened before or after TP1 price was touched.")

    # Also check: stop reachable on fill bar (but code already handles this with the skip)
    # Let's verify
    same_bar_stop = 0
    for idx, t in enumerate(trades):
        bar_i = trade_fill_bars[idx]
        if bar_i < 0 or bar_i >= n:
            continue
        if t["dir"] == 1:
            if l_arr[bar_i] <= t["stop_price"]:
                same_bar_stop += 1
        else:
            if h_arr[bar_i] >= t["stop_price"]:
                same_bar_stop += 1

    print(f"  Trades where STOP is reachable on fill bar: {same_bar_stop}")
    if same_bar_stop > 0:
        print(f"  >>> BUG: {same_bar_stop} trades have stop within fill bar range but weren't skipped!")

    # ================================================================
    # GRAND SUMMARY — VERDICT
    # ================================================================
    print(f"\n{SEP}")
    print("GRAND SUMMARY — HOSTILE AUDIT VERDICT")
    print(SEP)

    print(f"\n  Claimed: +3,144R / PPDD=164 / PF=2.45 from 4,180 trades")
    print(f"  Reproduced: R={best_m['R']:+.1f} / PPDD={best_m['PPDD']:.2f} / PF={best_m['PF']:.2f} from {best_m['trades']} trades")
    print(f"  Config: {best_cfg_name}")

    print(f"\n  FINDINGS:")
    print(f"    D1 (intra-bar): Same-bar TP1 reachable on {same_bar_tp1} trades ({100*same_bar_tp1/len(trades):.1f}%)")
    print(f"    D2 (timing): Median fill wait = {stats['median_fill_wait_bars']:.1f} bars (1+ is OK)")
    print(f"    D3 (overnight): {overnight_count} overnight trades")
    print(f"    D5 (fill sensitivity): Check table above for R degradation with penetration")
    print(f"    D7 (year consistency): {sum(1 for yr in wf if yr['R'] < 0)}/{len(wf)} negative years" if wf else "    D7: No data")
    print(f"    D9 (random): avg random R = {avg_random_r:+.1f} vs real {best_m['R']:+.1f} (ratio: {edge_ratio:.2f}x)")
    print(f"    D10 (leverage): avg stop = {stop_dists.mean():.1f}pts, avg contracts = {contracts.mean():.0f}")

    elapsed = _time.perf_counter() - t_start
    print(f"\n  Audit runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"\n{SEP}")
    print("AUDIT COMPLETE")
    print(SEP)


def _simulate_trade_forward(
    entry_bar: int, direction: int, entry_p: float, stop: float,
    tp1: float, tp2: float,
    h_arr: np.ndarray, l_arr: np.ndarray, c_arr: np.ndarray,
    n: int, et_frac_arr: np.ndarray,
    contracts: int, stop_dist: float, point_value: float,
    commission_per_side: float,
    is_multi_tp: bool = True,
    tp1_trim_pct: float = 0.50,
    max_bars: int = 200,
) -> float:
    """Simulate a single trade forward from entry_bar. Returns R multiple."""
    trim_stage = 0
    trim1_contracts = 0
    tp1_pnl_pts = 0.0
    remaining = contracts
    tp1_hit_bar = -1
    slippage = 0.25  # 1 tick

    for i in range(entry_bar + 1, min(entry_bar + max_bars, n)):
        # EOD close
        if et_frac_arr[i] >= 15.917:
            exit_p = c_arr[i] - slippage if direction == 1 else c_arr[i] + slippage
            pnl_pts = (exit_p - entry_p) if direction == 1 else (entry_p - exit_p)
            total_pnl = pnl_pts * point_value * remaining
            if trim_stage >= 1:
                total_pnl += tp1_pnl_pts * point_value * trim1_contracts
            total_pnl -= commission_per_side * 2 * contracts
            total_risk = stop_dist * point_value * contracts
            return total_pnl / total_risk if total_risk > 0 else 0.0

        if direction == 1:
            # Check stop
            if l_arr[i] <= stop:
                exit_p = stop - slippage
                pnl_pts = exit_p - entry_p
                total_pnl = pnl_pts * point_value * remaining
                if trim_stage >= 1:
                    total_pnl += tp1_pnl_pts * point_value * trim1_contracts
                total_pnl -= commission_per_side * 2 * contracts
                total_risk = stop_dist * point_value * contracts
                return total_pnl / total_risk if total_risk > 0 else 0.0

            if is_multi_tp:
                # TP1
                if trim_stage == 0 and h_arr[i] >= tp1:
                    tc1 = max(1, int(np.ceil(contracts * tp1_trim_pct)))
                    trim1_contracts = tc1
                    remaining = contracts - tc1
                    tp1_pnl_pts = tp1 - entry_p
                    trim_stage = 1
                    tp1_hit_bar = i
                    if remaining <= 0:
                        total_pnl = tp1_pnl_pts * point_value * contracts - commission_per_side * 2 * contracts
                        total_risk = stop_dist * point_value * contracts
                        return total_pnl / total_risk if total_risk > 0 else 0.0
                # TP2
                if trim_stage == 1 and i > tp1_hit_bar and tp2 > 0 and h_arr[i] >= tp2:
                    tp2_pnl_pts = tp2 - entry_p
                    total_pnl = tp1_pnl_pts * point_value * trim1_contracts + tp2_pnl_pts * point_value * remaining
                    total_pnl -= commission_per_side * 2 * contracts
                    total_risk = stop_dist * point_value * contracts
                    return total_pnl / total_risk if total_risk > 0 else 0.0
            else:
                # Single TP (shorts)
                if trim_stage == 0 and l_arr[i] <= tp1:
                    pnl_pts = entry_p - tp1
                    total_pnl = pnl_pts * point_value * contracts - commission_per_side * 2 * contracts
                    total_risk = stop_dist * point_value * contracts
                    return total_pnl / total_risk if total_risk > 0 else 0.0

        else:  # Short
            # Check stop
            if h_arr[i] >= stop:
                exit_p = stop + slippage
                pnl_pts = entry_p - exit_p
                total_pnl = pnl_pts * point_value * remaining - commission_per_side * 2 * contracts
                total_risk = stop_dist * point_value * contracts
                return total_pnl / total_risk if total_risk > 0 else 0.0

            # TP1 for shorts
            if trim_stage == 0 and l_arr[i] <= tp1:
                pnl_pts = entry_p - tp1
                total_pnl = pnl_pts * point_value * contracts - commission_per_side * 2 * contracts
                total_risk = stop_dist * point_value * contracts
                return total_pnl / total_risk if total_risk > 0 else 0.0

    # Timed out — close at last bar
    last_i = min(entry_bar + max_bars - 1, n - 1)
    exit_p = c_arr[last_i]
    if direction == 1:
        pnl_pts = exit_p - entry_p
    else:
        pnl_pts = entry_p - exit_p
    total_pnl = pnl_pts * point_value * remaining
    if trim_stage >= 1:
        total_pnl += tp1_pnl_pts * point_value * trim1_contracts
    total_pnl -= commission_per_side * 2 * contracts
    total_risk = stop_dist * point_value * contracts
    return total_pnl / total_risk if total_risk > 0 else 0.0


if __name__ == "__main__":
    main()
