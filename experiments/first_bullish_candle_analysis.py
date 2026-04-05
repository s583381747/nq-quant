"""
experiments/first_bullish_candle_analysis.py
============================================
Hypothesis: A STRONG first bullish candle after overnight-low breakdown
signals institutional buying and predicts better trade outcomes.

Steps:
1. Load data via chain_engine.load_all()
2. Detect breakdowns at overnight low (longs only)
3. For each breakdown bar, find the FIRST bullish candle (close > open)
4. Measure: body size, body/range ratio, body as ATR multiple
5. Run chain backtest (ON low longs only), match trades to breakdowns
6. Tag each long trade with first_bullish_candle features
7. Split by body/ATR ratio bins: [0-0.3, 0.3-0.6, 0.6-1.0, 1.0+]
8. Compute PF/R/WR for each bin
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.chain_engine import load_all, detect_breakdowns, run_chain_backtest, compute_metrics


def find_first_bullish_candle(o, h, l, c, atr_arr, breakdown_bar: int, max_look: int = 30):
    """Find the first bullish candle (close > open) after a breakdown bar.
    Returns dict with features or None if not found within max_look bars.
    """
    n = len(o)
    for i in range(breakdown_bar + 1, min(breakdown_bar + max_look + 1, n)):
        if c[i] > o[i]:
            body = c[i] - o[i]
            rng = h[i] - l[i]
            body_ratio = body / rng if rng > 0 else 0.0
            atr_val = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
            body_atr = body / atr_val if atr_val > 0 else 0.0
            return {
                "bar_idx": i,
                "bars_after_bd": i - breakdown_bar,
                "body": body,
                "body_ratio": body_ratio,
                "body_atr": body_atr,
                "range": rng,
            }
    return None


def main():
    print("=" * 100)
    print("FIRST BULLISH CANDLE AFTER OVERNIGHT-LOW BREAKDOWN — Quality Analysis")
    print("=" * 100)

    # 1. Load data
    d = load_all()
    nq = d["nq"]
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    atr_arr = d["atr_arr"]
    n = d["n"]

    # 2. Detect breakdowns at overnight low (longs)
    params = d["params"]
    min_depth = params.get("chain", {}).get("min_depth_pts", 1.0)
    bds = detect_breakdowns(h, l, c, d["on_lo"], "low", min_depth_pts=min_depth)
    print(f"\n[INFO] Overnight-low breakdowns detected: {len(bds)}")

    # 3. For each breakdown, find the first bullish candle
    bd_features = {}  # bar_idx -> features dict
    for bd in bds:
        feat = find_first_bullish_candle(o, h, l, c, atr_arr, bd["bar_idx"], max_look=30)
        if feat is not None:
            bd_features[bd["bar_idx"]] = feat

    print(f"[INFO] Breakdowns with first bullish candle found: {len(bd_features)} / {len(bds)}")

    # Distribution of first bullish candle features
    if bd_features:
        body_atrs = [f["body_atr"] for f in bd_features.values()]
        body_ratios = [f["body_ratio"] for f in bd_features.values()]
        bars_after = [f["bars_after_bd"] for f in bd_features.values()]
        print(f"\n[DIST] First bullish candle body/ATR: "
              f"mean={np.mean(body_atrs):.3f}, median={np.median(body_atrs):.3f}, "
              f"std={np.std(body_atrs):.3f}")
        print(f"[DIST] First bullish candle body/range: "
              f"mean={np.mean(body_ratios):.3f}, median={np.median(body_ratios):.3f}")
        print(f"[DIST] Bars after breakdown: "
              f"mean={np.mean(bars_after):.1f}, median={np.median(bars_after):.1f}")

    # 4. Run chain backtest — ON low longs only
    trades, stats = run_chain_backtest(
        d,
        breakdown_sources=[(d["on_lo"], "low")],
    )
    print(f"\n[BACKTEST] Total trades: {len(trades)}")

    longs = [t for t in trades if t["dir"] == 1]
    shorts = [t for t in trades if t["dir"] == -1]
    print(f"[BACKTEST] Long trades: {len(longs)}, Short trades: {len(shorts)}")

    if not longs:
        print("[ERROR] No long trades found. Exiting.")
        return

    # Overall long metrics
    m_all = compute_metrics(longs)
    print(f"\n[BASELINE] All longs: {m_all['trades']}t | R={m_all['R']:+.1f} | "
          f"PF={m_all['PF']:.2f} | WR={m_all['WR']:.1f}% | PPDD={m_all['PPDD']:.2f} | MaxDD={m_all['MaxDD']:.1f}R")

    # 5. Match each long trade to its most recent breakdown
    # Build sorted list of breakdown bar indices
    bd_bar_indices = sorted(bd_features.keys())
    bd_bar_arr = np.array(bd_bar_indices)

    tagged_trades = []
    matched = 0
    unmatched = 0

    for t in longs:
        entry_ts = t["entry_time"]
        entry_idx = nq.index.get_loc(entry_ts)
        if isinstance(entry_idx, slice):
            entry_idx = entry_idx.start

        # Find most recent breakdown bar before entry
        candidates = bd_bar_arr[bd_bar_arr < entry_idx]
        if len(candidates) == 0:
            unmatched += 1
            continue

        nearest_bd = candidates[-1]  # most recent
        feat = bd_features[nearest_bd]

        # Sanity: breakdown shouldn't be too far from entry (max 200 bars)
        if entry_idx - nearest_bd > 200:
            unmatched += 1
            continue

        t_tagged = dict(t)
        t_tagged["bd_bar"] = nearest_bd
        t_tagged["first_bull_body_atr"] = feat["body_atr"]
        t_tagged["first_bull_body_ratio"] = feat["body_ratio"]
        t_tagged["first_bull_body"] = feat["body"]
        t_tagged["first_bull_bars_after"] = feat["bars_after_bd"]
        tagged_trades.append(t_tagged)
        matched += 1

    print(f"\n[MATCH] Matched: {matched}, Unmatched: {unmatched}")

    if not tagged_trades:
        print("[ERROR] No trades could be matched to breakdowns. Exiting.")
        return

    # 6. Split by first candle body/ATR ratio bins
    bins = [
        ("0.00 - 0.30", 0.0, 0.3),
        ("0.30 - 0.60", 0.3, 0.6),
        ("0.60 - 1.00", 0.6, 1.0),
        ("1.00+       ", 1.0, float("inf")),
    ]

    print(f"\n{'='*100}")
    print("RESULTS: Long Trade Performance by First Bullish Candle Body/ATR")
    print(f"{'='*100}")
    print(f"\n  {'Bin':16s} | {'Trades':>6s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'PPDD':>7s} | {'MaxDD':>6s} | {'avgR':>7s} | {'AvgBodyATR':>10s}")
    print(f"  {'-'*16}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}-+-{'-'*10}")

    for label, lo_val, hi_val in bins:
        subset = [t for t in tagged_trades if lo_val <= t["first_bull_body_atr"] < hi_val]
        if not subset:
            print(f"  {label:16s} | {0:6d} | {'n/a':>8s}")
            continue
        m = compute_metrics(subset)
        avg_ba = np.mean([t["first_bull_body_atr"] for t in subset])
        print(f"  {label:16s} | {m['trades']:6d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['WR']:5.1f}% | {m['PPDD']:6.2f} | {m['MaxDD']:5.1f}R | {m['avgR']:+.4f} | {avg_ba:10.3f}")

    # 7. Also split by body/range ratio for cross-check
    bins_br = [
        ("BR 0.00-0.40", 0.0, 0.4),
        ("BR 0.40-0.60", 0.4, 0.6),
        ("BR 0.60-0.80", 0.6, 0.8),
        ("BR 0.80-1.00", 0.8, 1.0),
    ]

    print(f"\n{'='*100}")
    print("CROSS-CHECK: Long Trade Performance by First Bullish Candle Body/Range Ratio")
    print(f"{'='*100}")
    print(f"\n  {'Bin':16s} | {'Trades':>6s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'PPDD':>7s} | {'MaxDD':>6s} | {'avgR':>7s}")
    print(f"  {'-'*16}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")

    for label, lo_val, hi_val in bins_br:
        subset = [t for t in tagged_trades if lo_val <= t["first_bull_body_ratio"] < hi_val]
        if not subset:
            print(f"  {label:16s} | {0:6d} | {'n/a':>8s}")
            continue
        m = compute_metrics(subset)
        print(f"  {label:16s} | {m['trades']:6d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['WR']:5.1f}% | {m['PPDD']:6.2f} | {m['MaxDD']:5.1f}R | {m['avgR']:+.4f}")

    # 8. Also check: bars_after_bd (how quickly the first bullish candle appears)
    bins_delay = [
        ("1 bar after   ", 1, 2),
        ("2-3 bars after", 2, 4),
        ("4-6 bars after", 4, 7),
        ("7+ bars after ", 7, 31),
    ]

    print(f"\n{'='*100}")
    print("CROSS-CHECK: Long Trade Performance by Bars Until First Bullish Candle")
    print(f"{'='*100}")
    print(f"\n  {'Bin':16s} | {'Trades':>6s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'PPDD':>7s} | {'MaxDD':>6s} | {'avgR':>7s}")
    print(f"  {'-'*16}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")

    for label, lo_val, hi_val in bins_delay:
        subset = [t for t in tagged_trades if lo_val <= t["first_bull_bars_after"] < hi_val]
        if not subset:
            print(f"  {label:16s} | {0:6d} | {'n/a':>8s}")
            continue
        m = compute_metrics(subset)
        print(f"  {label:16s} | {m['trades']:6d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['WR']:5.1f}% | {m['PPDD']:6.2f} | {m['MaxDD']:5.1f}R | {m['avgR']:+.4f}")

    # 9. Interaction: body/ATR × delay
    print(f"\n{'='*100}")
    print("INTERACTION: Body/ATR (strong vs weak) × Delay (fast vs slow)")
    print(f"{'='*100}")

    median_ba = np.median([t["first_bull_body_atr"] for t in tagged_trades])
    median_delay = np.median([t["first_bull_bars_after"] for t in tagged_trades])
    print(f"  Median body/ATR = {median_ba:.3f}, Median delay = {median_delay:.1f} bars")

    combos = [
        ("Strong+Fast", lambda t: t["first_bull_body_atr"] >= median_ba and t["first_bull_bars_after"] <= median_delay),
        ("Strong+Slow", lambda t: t["first_bull_body_atr"] >= median_ba and t["first_bull_bars_after"] > median_delay),
        ("Weak+Fast",   lambda t: t["first_bull_body_atr"] < median_ba and t["first_bull_bars_after"] <= median_delay),
        ("Weak+Slow",   lambda t: t["first_bull_body_atr"] < median_ba and t["first_bull_bars_after"] > median_delay),
    ]

    print(f"\n  {'Combo':16s} | {'Trades':>6s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'PPDD':>7s} | {'MaxDD':>6s} | {'avgR':>7s}")
    print(f"  {'-'*16}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")

    for label, filt in combos:
        subset = [t for t in tagged_trades if filt(t)]
        if not subset:
            print(f"  {label:16s} | {0:6d} | {'n/a':>8s}")
            continue
        m = compute_metrics(subset)
        print(f"  {label:16s} | {m['trades']:6d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['WR']:5.1f}% | {m['PPDD']:6.2f} | {m['MaxDD']:5.1f}R | {m['avgR']:+.4f}")

    print(f"\n{'='*100}")
    print("CONCLUSION")
    print(f"{'='*100}")
    print("  If PF and WR increase monotonically with body/ATR bins,")
    print("  the hypothesis is CONFIRMED: strong first bullish candle = institutional buying.")
    print("  Actionable: filter/weight entries by first bullish candle strength.")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
