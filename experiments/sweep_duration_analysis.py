"""
experiments/sweep_duration_analysis.py — Sweep Duration vs Trade Quality
========================================================================

Hypothesis: shorter sweep duration (V-reversal, 1-2 bars) = more likely
a stop hunt = better PF.

For each breakdown at overnight low (longs), measure how many consecutive
bars price stayed below ON_low AFTER the breakdown bar.
  - 1 bar  = V-reversal (just the breakdown bar itself)
  - 2-3    = quick recovery
  - 5+     = sustained breakdown

Then run chain backtest, match trades to their nearest prior breakdown,
tag each trade with sweep_duration, and split by bins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.chain_engine import load_all, detect_breakdowns, run_chain_backtest, compute_metrics


def measure_sweep_duration(c: np.ndarray, on_lo: np.ndarray, breakdowns: list[dict]) -> dict[int, int]:
    """For each breakdown bar_idx, count consecutive bars where close < ON_low.

    The breakdown bar itself counts as bar 1.
    Returns {bar_idx: duration}.
    """
    n = len(c)
    result = {}
    for bd in breakdowns:
        idx = bd["bar_idx"]
        level = on_lo[idx]
        if np.isnan(level) or level <= 0:
            result[idx] = 1
            continue

        duration = 1  # the breakdown bar itself
        for j in range(idx + 1, n):
            if c[j] < level:
                duration += 1
            else:
                break
        result[idx] = duration
    return result


def match_trade_to_breakdown(
    trade_entry_time: pd.Timestamp,
    nq_index: pd.DatetimeIndex,
    breakdown_bars: list[int],
) -> int | None:
    """Find the nearest breakdown bar_idx that is BEFORE the trade entry bar."""
    # Convert entry_time to bar index
    try:
        entry_bar = nq_index.get_loc(trade_entry_time)
    except KeyError:
        # Nearest match
        entry_bar = nq_index.searchsorted(trade_entry_time)

    best = None
    best_dist = 999999
    for bd_bar in breakdown_bars:
        if bd_bar <= entry_bar:
            dist = entry_bar - bd_bar
            if dist < best_dist:
                best_dist = dist
                best = bd_bar
    return best


def main():
    print("=" * 100)
    print("SWEEP DURATION ANALYSIS — V-Reversal Hypothesis")
    print("=" * 100)

    d = load_all()
    nq = d["nq"]
    c = d["c"]
    on_lo = d["on_lo"]
    on_hi = d["on_hi"]

    # --- Step 1: Detect breakdowns at ON_low (for longs) and ON_high (for shorts) ---
    from experiments.chain_engine import detect_breakdowns
    import yaml
    params = d["params"]
    min_depth = params.get("chain", {}).get("min_depth_pts", 1.0)

    bd_low = detect_breakdowns(d["h"], d["l"], c, on_lo, "low", min_depth_pts=min_depth)
    bd_high = detect_breakdowns(d["h"], d["l"], c, on_hi, "high", min_depth_pts=min_depth)

    print(f"\n[INFO] Breakdowns detected: {len(bd_low)} at ON_low (long setups), {len(bd_high)} at ON_high (short setups)")

    # --- Step 2: Measure sweep duration for ON_low breakdowns ---
    sweep_dur_low = measure_sweep_duration(c, on_lo, bd_low)
    sweep_dur_high = measure_sweep_duration_high(c, on_hi, bd_high)

    # Print distribution
    durs_low = list(sweep_dur_low.values())
    durs_high = list(sweep_dur_high.values())
    print(f"\n[ON_LOW SWEEP DURATION DISTRIBUTION]")
    print(f"  1 bar (V-reversal):  {sum(1 for x in durs_low if x == 1):5d}  ({sum(1 for x in durs_low if x == 1)/len(durs_low)*100:.1f}%)")
    print(f"  2-3 bars (quick):    {sum(1 for x in durs_low if 2 <= x <= 3):5d}  ({sum(1 for x in durs_low if 2 <= x <= 3)/len(durs_low)*100:.1f}%)")
    print(f"  4-5 bars:            {sum(1 for x in durs_low if 4 <= x <= 5):5d}  ({sum(1 for x in durs_low if 4 <= x <= 5)/len(durs_low)*100:.1f}%)")
    print(f"  6+ bars (sustained): {sum(1 for x in durs_low if x >= 6):5d}  ({sum(1 for x in durs_low if x >= 6)/len(durs_low)*100:.1f}%)")

    print(f"\n[ON_HIGH SWEEP DURATION DISTRIBUTION]")
    print(f"  1 bar (V-reversal):  {sum(1 for x in durs_high if x == 1):5d}  ({sum(1 for x in durs_high if x == 1)/len(durs_high)*100:.1f}%)")
    print(f"  2-3 bars (quick):    {sum(1 for x in durs_high if 2 <= x <= 3):5d}  ({sum(1 for x in durs_high if 2 <= x <= 3)/len(durs_high)*100:.1f}%)")
    print(f"  4-5 bars:            {sum(1 for x in durs_high if 4 <= x <= 5):5d}  ({sum(1 for x in durs_high if 4 <= x <= 5)/len(durs_high)*100:.1f}%)")
    print(f"  6+ bars (sustained): {sum(1 for x in durs_high if x >= 6):5d}  ({sum(1 for x in durs_high if x >= 6)/len(durs_high)*100:.1f}%)")

    # --- Step 3: Run chain backtest (ON only, standard config) ---
    trades, stats = run_chain_backtest(d, breakdown_sources=[
        (on_lo, "low"),
        (on_hi, "high"),
    ])
    m_all = compute_metrics(trades)
    print(f"\n[BASELINE] All trades: {m_all['trades']}t | R={m_all['R']:+.1f} | PF={m_all['PF']:.2f} | PPDD={m_all['PPDD']:.2f} | WR={m_all['WR']:.1f}%")

    # --- Step 4: Match trades to breakdowns, tag with sweep_duration ---
    # Combine all breakdown bar indices with their durations
    all_bd_bars_low = sorted(sweep_dur_low.keys())
    all_bd_bars_high = sorted(sweep_dur_high.keys())

    nq_index = nq.index

    tagged_trades = []
    unmatched = 0
    for t in trades:
        direction = t["dir"]
        entry_time = t["entry_time"]

        # Match to correct breakdown type
        if direction == 1:
            bd_bars = all_bd_bars_low
            dur_map = sweep_dur_low
        else:
            bd_bars = all_bd_bars_high
            dur_map = sweep_dur_high

        matched_bar = match_trade_to_breakdown(entry_time, nq_index, bd_bars)

        if matched_bar is not None and matched_bar in dur_map:
            t_copy = dict(t)
            t_copy["sweep_duration"] = dur_map[matched_bar]
            t_copy["breakdown_bar"] = matched_bar
            tagged_trades.append(t_copy)
        else:
            unmatched += 1

    print(f"[MATCH] Tagged {len(tagged_trades)} trades, {unmatched} unmatched")

    # --- Step 5: Split by sweep_duration bins and compute metrics ---
    bins = [
        ("1 bar (V-reversal)", lambda d: d == 1),
        ("2-3 bars (quick)",   lambda d: 2 <= d <= 3),
        ("4-6 bars",           lambda d: 4 <= d <= 6),
        ("7-10 bars",          lambda d: 7 <= d <= 10),
        ("11+ bars (sustained)", lambda d: d >= 11),
    ]

    print(f"\n{'='*100}")
    print("SWEEP DURATION vs TRADE QUALITY — ALL TRADES (Longs + Shorts)")
    print(f"{'='*100}")
    print(f"  {'Bin':30s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'PPDD':>7s} | {'WR':>6s} | {'AvgR':>7s} | {'DD':>6s}")
    print(f"  {'-'*30}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}")

    for label, filt in bins:
        subset = [t for t in tagged_trades if filt(t["sweep_duration"])]
        if len(subset) < 5:
            print(f"  {label:30s} | {len(subset):5d} | too few")
            continue
        m = compute_metrics(subset)
        print(f"  {label:30s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['PPDD']:6.2f} | {m['WR']:5.1f}% | {m['avgR']:+.4f} | {m['MaxDD']:5.1f}R")

    # --- Split LONGS only ---
    longs_tagged = [t for t in tagged_trades if t["dir"] == 1]
    print(f"\n{'='*100}")
    print("SWEEP DURATION vs TRADE QUALITY — LONGS ONLY (ON_low breakdowns)")
    print(f"{'='*100}")
    print(f"  {'Bin':30s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'PPDD':>7s} | {'WR':>6s} | {'AvgR':>7s} | {'DD':>6s}")
    print(f"  {'-'*30}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}")

    for label, filt in bins:
        subset = [t for t in longs_tagged if filt(t["sweep_duration"])]
        if len(subset) < 5:
            print(f"  {label:30s} | {len(subset):5d} | too few")
            continue
        m = compute_metrics(subset)
        print(f"  {label:30s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['PPDD']:6.2f} | {m['WR']:5.1f}% | {m['avgR']:+.4f} | {m['MaxDD']:5.1f}R")

    # --- Split SHORTS only ---
    shorts_tagged = [t for t in tagged_trades if t["dir"] == -1]
    print(f"\n{'='*100}")
    print("SWEEP DURATION vs TRADE QUALITY — SHORTS ONLY (ON_high breakdowns)")
    print(f"{'='*100}")
    print(f"  {'Bin':30s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'PPDD':>7s} | {'WR':>6s} | {'AvgR':>7s} | {'DD':>6s}")
    print(f"  {'-'*30}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}")

    for label, filt in bins:
        subset = [t for t in shorts_tagged if filt(t["sweep_duration"])]
        if len(subset) < 5:
            print(f"  {label:30s} | {len(subset):5d} | too few")
            continue
        m = compute_metrics(subset)
        print(f"  {label:30s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['PPDD']:6.2f} | {m['WR']:5.1f}% | {m['avgR']:+.4f} | {m['MaxDD']:5.1f}R")

    # --- Granular 1-bar breakdown: exact duration ---
    print(f"\n{'='*100}")
    print("GRANULAR: EXACT SWEEP DURATION (1-15+) — ALL TRADES")
    print(f"{'='*100}")
    print(f"  {'Duration':15s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'AvgR':>7s}")
    print(f"  {'-'*15}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")

    for dur in range(1, 16):
        subset = [t for t in tagged_trades if t["sweep_duration"] == dur]
        if len(subset) < 5:
            if len(subset) > 0:
                print(f"  {str(dur)+' bars':15s} | {len(subset):5d} | (too few)")
            continue
        m = compute_metrics(subset)
        print(f"  {str(dur)+' bars':15s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['WR']:5.1f}% | {m['avgR']:+.4f}")

    # 16+ lumped
    subset = [t for t in tagged_trades if t["sweep_duration"] >= 16]
    if len(subset) >= 5:
        m = compute_metrics(subset)
        print(f"  {'16+ bars':15s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['WR']:5.1f}% | {m['avgR']:+.4f}")

    # --- Median sweep duration for winners vs losers ---
    winners = [t for t in tagged_trades if t["r"] > 0]
    losers = [t for t in tagged_trades if t["r"] < 0]
    if winners and losers:
        w_dur = np.median([t["sweep_duration"] for t in winners])
        l_dur = np.median([t["sweep_duration"] for t in losers])
        print(f"\n[INSIGHT] Median sweep duration — Winners: {w_dur:.1f} bars, Losers: {l_dur:.1f} bars")

    print(f"\n{'='*100}")
    print("DONE")
    print(f"{'='*100}")


def measure_sweep_duration_high(c: np.ndarray, on_hi: np.ndarray, breakdowns: list[dict]) -> dict[int, int]:
    """For each ON_high breakdown, count consecutive bars where close > ON_high."""
    n = len(c)
    result = {}
    for bd in breakdowns:
        idx = bd["bar_idx"]
        level = on_hi[idx]
        if np.isnan(level) or level <= 0:
            result[idx] = 1
            continue

        duration = 1  # the breakdown bar itself
        for j in range(idx + 1, n):
            if c[j] > level:
                duration += 1
            else:
                break
        result[idx] = duration
    return result


if __name__ == "__main__":
    main()
