"""
experiments/sweep_bar_character_analysis.py — Sweep Bar Character vs Trade Quality
==================================================================================

For each LONG trade from the chain engine:
  1. Find the breakdown bar (detect_breakdowns on ON_lo)
  2. Measure sweep bar characteristics:
     - lower_wick / range  (wick ratio)
     - body / range  (body ratio)
     - close position: (close - low) / (high - low)
  3. Bin trades by these features, compute PF/R/WR per bin

Key question: Does the CHARACTER of the sweep (breakdown) bar predict trade quality?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.chain_engine import load_all, run_chain_backtest, compute_metrics, detect_breakdowns


def main():
    print("=" * 100)
    print("SWEEP BAR CHARACTER ANALYSIS — Does breakdown bar shape predict trade quality?")
    print("=" * 100)

    # ---- Load data and run baseline backtest ----
    d = load_all()
    nq = d["nq"]
    n = d["n"]
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]

    # Default ON-only config
    trades, stats = run_chain_backtest(d)
    longs = [t for t in trades if t["dir"] == 1]

    m_all = compute_metrics(trades)
    m_longs = compute_metrics(longs)
    print(f"\n  Baseline: {m_all['trades']}t total, {len(longs)} longs")
    print(f"  All:   R={m_all['R']:+.1f}  PF={m_all['PF']:.2f}  PPDD={m_all['PPDD']:.2f}  WR={m_all['WR']:.1f}%")
    print(f"  Longs: R={m_longs['R']:+.1f}  PF={m_longs['PF']:.2f}  PPDD={m_longs['PPDD']:.2f}  WR={m_longs['WR']:.1f}%")

    # ---- Re-detect breakdowns on ON_lo (for longs: breakdown of low = stop hunt below) ----
    on_lo = d["on_lo"]
    params = d["params"]
    min_depth = params.get("chain", {}).get("min_depth_pts", 1.0)
    bd_events = detect_breakdowns(h, l, c, on_lo, "low", min_depth_pts=min_depth)
    bd_bars = np.array([ev["bar_idx"] for ev in bd_events])
    print(f"\n  Detected {len(bd_bars)} low-breakdown events at ON_lo")

    # ---- Map each long trade to its nearest preceding breakdown bar ----
    # Build index mapping from nq.index timestamp -> integer position
    nq_index = nq.index
    # Ensure both are comparable (both tz-aware or both naive)
    # nq.index is tz-aware (UTC). trade entry_time should also be tz-aware.

    enriched = []
    n_matched = 0
    n_unmatched = 0

    for t in longs:
        entry_ts = t["entry_time"]

        # Find integer bar index of entry
        try:
            entry_idx = nq_index.get_loc(entry_ts)
        except KeyError:
            # Try nearest
            entry_idx = nq_index.get_indexer([entry_ts], method="nearest")[0]
            if entry_idx < 0:
                n_unmatched += 1
                continue

        # Find the most recent breakdown bar before (or at) entry_idx
        # bd_bars is sorted ascending
        candidates = bd_bars[bd_bars <= entry_idx]
        if len(candidates) == 0:
            n_unmatched += 1
            continue

        bd_idx = candidates[-1]  # most recent breakdown before entry

        # Only consider if breakdown is within 200 bars of entry (max_fvg_age default)
        if entry_idx - bd_idx > 200:
            n_unmatched += 1
            continue

        # ---- Measure sweep bar characteristics ----
        bar_o = o[bd_idx]
        bar_h = h[bd_idx]
        bar_l = l[bd_idx]
        bar_c = c[bd_idx]
        bar_range = bar_h - bar_l

        if bar_range <= 0:
            n_unmatched += 1
            continue

        # For a breakdown of lows (bearish bar breaking below support):
        # The bar closed BELOW the level, so it's a bearish candle typically
        # lower_wick = low to min(open, close)
        lower_wick = min(bar_o, bar_c) - bar_l
        upper_wick = bar_h - max(bar_o, bar_c)
        body = abs(bar_c - bar_o)

        wick_ratio = lower_wick / bar_range
        body_ratio = body / bar_range
        close_position = (bar_c - bar_l) / bar_range  # 0=closed at low, 1=closed at high

        enriched.append({
            **t,
            "bd_bar_idx": bd_idx,
            "entry_bar_idx": entry_idx,
            "bars_bd_to_entry": entry_idx - bd_idx,
            "sweep_wick_ratio": wick_ratio,
            "sweep_body_ratio": body_ratio,
            "sweep_close_position": close_position,
            "sweep_lower_wick": lower_wick,
            "sweep_upper_wick": upper_wick,
            "sweep_body": body,
            "sweep_range": bar_range,
        })
        n_matched += 1

    print(f"  Matched {n_matched} longs to breakdown bars, {n_unmatched} unmatched")

    if n_matched < 30:
        print("  Too few matched trades for meaningful analysis. Exiting.")
        return

    df = pd.DataFrame(enriched)

    # ---- ANALYSIS 1: Wick Ratio Bins ----
    wick_bins = [0, 0.10, 0.30, 0.50, 1.0]
    wick_labels = ["0-0.10", "0.10-0.30", "0.30-0.50", "0.50-1.00"]
    df["wick_bin"] = pd.cut(df["sweep_wick_ratio"], bins=wick_bins, labels=wick_labels, include_lowest=True)

    print(f"\n{'='*100}")
    print("ANALYSIS 1: SWEEP BAR LOWER WICK RATIO (lower_wick / range)")
    print(f"  Low wick = bar closed near its low (strong bearish close)")
    print(f"  High wick = bar swept low but closed back up (rejection wick)")
    print(f"{'='*100}")
    print(f"  {'Wick Ratio Bin':20s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>7s} | {'PPDD':>7s}")
    print(f"  {'-'*20}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}")

    for label in wick_labels:
        subset = df[df["wick_bin"] == label]
        if len(subset) < 5:
            print(f"  {label:20s} | {len(subset):5d} | too few")
            continue
        tlist = subset.to_dict("records")
        m = compute_metrics(tlist)
        print(f"  {label:20s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['WR']:5.1f}% | {m['avgR']:+.4f} | {m['PPDD']:6.2f}")

    # ---- ANALYSIS 2: Body Ratio Bins ----
    body_bins = [0, 0.30, 0.50, 0.70, 1.0]
    body_labels = ["0-0.30", "0.30-0.50", "0.50-0.70", "0.70-1.00"]
    df["body_bin"] = pd.cut(df["sweep_body_ratio"], bins=body_bins, labels=body_labels, include_lowest=True)

    print(f"\n{'='*100}")
    print("ANALYSIS 2: SWEEP BAR BODY RATIO (body / range)")
    print(f"  High body = decisive candle with little wick")
    print(f"  Low body = indecisive candle, lots of wicks")
    print(f"{'='*100}")
    print(f"  {'Body Ratio Bin':20s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>7s} | {'PPDD':>7s}")
    print(f"  {'-'*20}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}")

    for label in body_labels:
        subset = df[df["body_bin"] == label]
        if len(subset) < 5:
            print(f"  {label:20s} | {len(subset):5d} | too few")
            continue
        tlist = subset.to_dict("records")
        m = compute_metrics(tlist)
        print(f"  {label:20s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['WR']:5.1f}% | {m['avgR']:+.4f} | {m['PPDD']:6.2f}")

    # ---- ANALYSIS 3: Close Position Bins ----
    close_bins = [0, 0.20, 0.40, 0.60, 0.80, 1.0]
    close_labels = ["0-0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", "0.80-1.00"]
    df["close_bin"] = pd.cut(df["sweep_close_position"], bins=close_bins, labels=close_labels, include_lowest=True)

    print(f"\n{'='*100}")
    print("ANALYSIS 3: SWEEP BAR CLOSE POSITION ((close - low) / (high - low))")
    print(f"  Low = closed near bottom of bar (bearish commitment)")
    print(f"  High = closed near top of bar (rejection / reversal start)")
    print(f"{'='*100}")
    print(f"  {'Close Position Bin':20s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>7s} | {'PPDD':>7s}")
    print(f"  {'-'*20}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}")

    for label in close_labels:
        subset = df[df["close_bin"] == label]
        if len(subset) < 5:
            print(f"  {label:20s} | {len(subset):5d} | too few")
            continue
        tlist = subset.to_dict("records")
        m = compute_metrics(tlist)
        print(f"  {label:20s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['WR']:5.1f}% | {m['avgR']:+.4f} | {m['PPDD']:6.2f}")

    # ---- ANALYSIS 4: 2D Cross — Wick × Body ----
    print(f"\n{'='*100}")
    print("ANALYSIS 4: 2D CROSS — Wick Ratio × Body Ratio")
    print(f"{'='*100}")
    print(f"  {'Wick':12s} × {'Body':12s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>7s}")
    print(f"  {'-'*12}-x-{'-'*12}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")

    for wl in wick_labels:
        for bl in body_labels:
            subset = df[(df["wick_bin"] == wl) & (df["body_bin"] == bl)]
            if len(subset) < 5:
                continue
            tlist = subset.to_dict("records")
            m = compute_metrics(tlist)
            print(f"  {wl:12s} x {bl:12s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['WR']:5.1f}% | {m['avgR']:+.4f}")

    # ---- ANALYSIS 5: Bars between breakdown and entry ----
    dist_bins = [0, 5, 15, 30, 60, 200]
    dist_labels = ["0-5", "5-15", "15-30", "30-60", "60-200"]
    df["dist_bin"] = pd.cut(df["bars_bd_to_entry"], bins=dist_bins, labels=dist_labels, include_lowest=True)

    print(f"\n{'='*100}")
    print("ANALYSIS 5: BARS FROM BREAKDOWN TO ENTRY")
    print(f"  How quickly after breakdown does entry happen?")
    print(f"{'='*100}")
    print(f"  {'Bars BD→Entry':20s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>7s} | {'PPDD':>7s}")
    print(f"  {'-'*20}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}")

    for label in dist_labels:
        subset = df[df["dist_bin"] == label]
        if len(subset) < 5:
            print(f"  {label:20s} | {len(subset):5d} | too few")
            continue
        tlist = subset.to_dict("records")
        m = compute_metrics(tlist)
        print(f"  {label:20s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['WR']:5.1f}% | {m['avgR']:+.4f} | {m['PPDD']:6.2f}")

    # ---- ANALYSIS 6: Sweep Range (ATR-normalized) ----
    atr_at_bd = np.array([d["atr_arr"][int(row["bd_bar_idx"])] for _, row in df.iterrows()])
    atr_at_bd = np.where(np.isnan(atr_at_bd) | (atr_at_bd <= 0), 30.0, atr_at_bd)
    df["sweep_range_atr"] = df["sweep_range"].values / atr_at_bd

    range_bins = [0, 0.5, 1.0, 1.5, 2.0, 10.0]
    range_labels = ["0-0.5x", "0.5-1.0x", "1.0-1.5x", "1.5-2.0x", "2.0x+"]
    df["range_bin"] = pd.cut(df["sweep_range_atr"], bins=range_bins, labels=range_labels, include_lowest=True)

    print(f"\n{'='*100}")
    print("ANALYSIS 6: SWEEP BAR RANGE (normalized by ATR)")
    print(f"  How large is the breakdown bar relative to recent volatility?")
    print(f"{'='*100}")
    print(f"  {'Range / ATR':20s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>7s} | {'PPDD':>7s}")
    print(f"  {'-'*20}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}")

    for label in range_labels:
        subset = df[df["range_bin"] == label]
        if len(subset) < 5:
            print(f"  {label:20s} | {len(subset):5d} | too few")
            continue
        tlist = subset.to_dict("records")
        m = compute_metrics(tlist)
        print(f"  {label:20s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['WR']:5.1f}% | {m['avgR']:+.4f} | {m['PPDD']:6.2f}")

    # ---- Summary statistics ----
    print(f"\n{'='*100}")
    print("DISTRIBUTION SUMMARY")
    print(f"{'='*100}")
    for col in ["sweep_wick_ratio", "sweep_body_ratio", "sweep_close_position", "sweep_range_atr", "bars_bd_to_entry"]:
        vals = df[col].dropna()
        print(f"  {col:30s}: mean={vals.mean():.3f}  median={vals.median():.3f}  "
              f"std={vals.std():.3f}  min={vals.min():.3f}  max={vals.max():.3f}")

    print(f"\n{'='*100}")
    print("DONE")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
