"""
experiments/bars_to_fill_analysis.py — FVG Formation Speed Analysis
====================================================================

Question: Do FVGs that form QUICKLY after breakdown perform better?

Approach:
  1. Run chain backtest with default ON-only config
  2. Re-detect breakdowns and FVGs to get birth_bar + breakdown_bar
  3. For each trade, match to its source FVG zone by entry price
  4. Compute: bars_bd_to_fvg = birth_bar - breakdown_bar
  5. Split by bins and compute PF/R per bin
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.chain_engine import (
    load_all, run_chain_backtest, compute_metrics, pr,
    detect_breakdowns, find_fvg_not_mss, ChainZone,
)


def main():
    print("=" * 120)
    print("FVG FORMATION SPEED ANALYSIS — Do quick FVGs outperform slow ones?")
    print("=" * 120)

    # -------------------------------------------------------------------
    # Step 1: Load data + run backtest
    # -------------------------------------------------------------------
    d = load_all()
    trades, stats = run_chain_backtest(d)  # Default ON-only config
    m = compute_metrics(trades)
    print(f"\n[BASELINE] ON-only chain backtest:")
    pr("ON-only baseline", m)
    print(f"  Breakdowns={stats['breakdowns']}, FVGs found={stats['fvgs_found']}, MSS rejected={stats['fvgs_mss_rejected']}")

    # -------------------------------------------------------------------
    # Step 2: Re-detect breakdowns + FVG zones (to get birth_bar + bd_bar)
    # -------------------------------------------------------------------
    print(f"\n[DETECT] Re-detecting breakdowns and FVGs...")

    n = d["n"]
    h, l, c, o = d["h"], d["l"], d["c"], d["o"]
    atr_arr = d["atr_arr"]
    nq = d["nq"]
    params = d["params"]
    fvg_df = d["fvg_df"]

    fvg_bm = fvg_df["fvg_bull"].values
    fvg_bt = fvg_df["fvg_bull_top"].values
    fvg_bb = fvg_df["fvg_bull_bottom"].values
    fvg_em = fvg_df["fvg_bear"].values
    fvg_et = fvg_df["fvg_bear_top"].values
    fvg_eb = fvg_df["fvg_bear_bottom"].values

    swing_high_mask = d["swing_high_mask"]
    swing_low_mask = d["swing_low_mask"]
    sh_prices = d["swing_high_price_at_mask"]
    sl_prices = d["swing_low_price_at_mask"]

    # Breakdown sources: ON only (default)
    breakdown_sources = [
        (d["on_lo"], "low"),
        (d["on_hi"], "high"),
    ]

    min_depth_pts = params.get("chain", {}).get("min_depth_pts", 1.0)
    max_wait_bars = 30
    min_fvg_size_atr = 0.3

    # Detect all breakdowns
    all_bds_raw = []
    for level_arr, level_type in breakdown_sources:
        bds = detect_breakdowns(h, l, c, level_arr, level_type, min_depth_pts=min_depth_pts)
        all_bds_raw.extend(bds)

    all_bds_raw.sort(key=lambda x: x["bar_idx"])
    all_bds = []
    last_bar = -10
    for bd in all_bds_raw:
        if bd["bar_idx"] - last_bar >= 3:
            all_bds.append(bd)
            last_bar = bd["bar_idx"]

    print(f"  Total breakdowns (deduped): {len(all_bds)}")

    # For each breakdown, find FVG zone and record (bd_bar, birth_bar, zone_top, zone_bottom)
    zone_records = []
    for bd in all_bds:
        zone = find_fvg_not_mss(
            fvg_bm, fvg_bt, fvg_bb, fvg_em, fvg_et, fvg_eb,
            h, l, c, o, atr_arr,
            swing_high_mask, swing_low_mask, sh_prices, sl_prices,
            bd["bar_idx"], bd["level_type"],
            max_wait_bars, min_fvg_size_atr,
        )
        if zone is not None:
            zone_records.append({
                "bd_bar": bd["bar_idx"],
                "birth_bar": zone.birth_bar,
                "direction": zone.direction,
                "top": zone.top,
                "bottom": zone.bottom,
                "bars_bd_to_fvg": zone.birth_bar - bd["bar_idx"],
            })

    print(f"  Zones found: {len(zone_records)}")

    # -------------------------------------------------------------------
    # Step 3: Match each trade to its source zone
    # -------------------------------------------------------------------
    print(f"\n[MATCH] Matching {len(trades)} trades to {len(zone_records)} zones...")

    # Build a mapping from nq.index to bar index for fast lookup
    ts_to_bar = {ts: idx for idx, ts in enumerate(nq.index)}

    matched_trades = []
    unmatched = 0

    for t in trades:
        entry_time = t["entry_time"]
        entry_bar = ts_to_bar.get(entry_time, None)
        if entry_bar is None:
            unmatched += 1
            continue

        entry_price = t["entry_price"]
        direction = t["dir"]

        # Find the zone that matches this trade:
        # - zone.direction == trade direction
        # - entry price matches zone top (long) or zone bottom (short)
        # - zone birth_bar < entry_bar (zone existed before entry)
        best_match = None
        best_dist = float("inf")

        for zr in zone_records:
            if zr["direction"] != direction:
                continue
            if zr["birth_bar"] >= entry_bar:
                continue

            # Entry price should match zone edge
            if direction == 1:
                price_diff = abs(entry_price - zr["top"])
            else:
                price_diff = abs(entry_price - zr["bottom"])

            if price_diff < 1.0:  # tolerance: 1 point
                # Prefer the zone closest in time (most recent birth_bar before entry)
                time_dist = entry_bar - zr["birth_bar"]
                if time_dist < best_dist:
                    best_dist = time_dist
                    best_match = zr

        if best_match is not None:
            matched_trades.append({
                **t,
                "bd_bar": best_match["bd_bar"],
                "birth_bar": best_match["birth_bar"],
                "entry_bar": entry_bar,
                "bars_bd_to_fvg": best_match["bars_bd_to_fvg"],
                "bars_fvg_to_fill": entry_bar - best_match["birth_bar"],
            })
        else:
            unmatched += 1

    print(f"  Matched: {len(matched_trades)}, Unmatched: {unmatched}")

    if len(matched_trades) < 50:
        print("\n  ERROR: Too few matched trades for meaningful analysis. Aborting.")
        return

    # -------------------------------------------------------------------
    # Step 4: Analyze bars_bd_to_fvg (BD -> FVG speed)
    # -------------------------------------------------------------------
    df = pd.DataFrame(matched_trades)

    print(f"\n{'='*120}")
    print("ANALYSIS 1: Bars from Breakdown -> FVG Formation (bars_bd_to_fvg)")
    print("Question: Do FVGs that form QUICKLY after breakdown perform better?")
    print(f"{'='*120}")

    # Distribution overview
    print(f"\n  bars_bd_to_fvg distribution:")
    print(f"    min={df['bars_bd_to_fvg'].min()}, max={df['bars_bd_to_fvg'].max()}, "
          f"mean={df['bars_bd_to_fvg'].mean():.1f}, median={df['bars_bd_to_fvg'].median():.0f}")

    # Bin analysis
    bins_bd = [
        ("1-3 bars",   1,  3),
        ("4-6 bars",   4,  6),
        ("7-10 bars",  7, 10),
        ("11-20 bars", 11, 20),
        ("21+ bars",   21, 999),
    ]

    print(f"\n  {'Bin':15s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>7s} | {'PPDD':>7s} | {'DD':>6s}")
    print(f"  {'-'*15}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}")

    for label, lo, hi in bins_bd:
        mask = (df["bars_bd_to_fvg"] >= lo) & (df["bars_bd_to_fvg"] <= hi)
        subset = df[mask].to_dict("records")
        if len(subset) < 5:
            print(f"  {label:15s} | {len(subset):5d} | too few")
            continue
        met = compute_metrics(subset)
        print(f"  {label:15s} | {met['trades']:5d} | {met['R']:+8.1f} | {met['PF']:5.2f} | {met['WR']:5.1f}% | {met['avgR']:+7.4f} | {met['PPDD']:6.2f} | {met['MaxDD']:5.1f}R")

    # -------------------------------------------------------------------
    # Step 5: Analyze bars_fvg_to_fill (FVG -> Entry speed)
    # -------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("ANALYSIS 2: Bars from FVG Birth -> Entry Fill (bars_fvg_to_fill)")
    print("Question: Do FVGs that fill quickly vs slowly perform differently?")
    print(f"{'='*120}")

    print(f"\n  bars_fvg_to_fill distribution:")
    print(f"    min={df['bars_fvg_to_fill'].min()}, max={df['bars_fvg_to_fill'].max()}, "
          f"mean={df['bars_fvg_to_fill'].mean():.1f}, median={df['bars_fvg_to_fill'].median():.0f}")

    bins_fill = [
        ("1-5 bars",    1,   5),
        ("6-15 bars",   6,  15),
        ("16-30 bars", 16,  30),
        ("31-60 bars", 31,  60),
        ("61+ bars",   61, 999),
    ]

    print(f"\n  {'Bin':15s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>7s} | {'PPDD':>7s} | {'DD':>6s}")
    print(f"  {'-'*15}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}")

    for label, lo, hi in bins_fill:
        mask = (df["bars_fvg_to_fill"] >= lo) & (df["bars_fvg_to_fill"] <= hi)
        subset = df[mask].to_dict("records")
        if len(subset) < 5:
            print(f"  {label:15s} | {len(subset):5d} | too few")
            continue
        met = compute_metrics(subset)
        print(f"  {label:15s} | {met['trades']:5d} | {met['R']:+8.1f} | {met['PF']:5.2f} | {met['WR']:5.1f}% | {met['avgR']:+7.4f} | {met['PPDD']:6.02f} | {met['MaxDD']:5.1f}R")

    # -------------------------------------------------------------------
    # Step 6: Combined 2D analysis (BD->FVG speed × FVG->Fill speed)
    # -------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("ANALYSIS 3: Combined — BD->FVG speed × FVG->Fill speed")
    print(f"{'='*120}")

    # Simpler: fast BD->FVG (<=6) vs slow (>6), fast fill (<=15) vs slow (>15)
    bd_fast = df["bars_bd_to_fvg"] <= 6
    fill_fast = df["bars_fvg_to_fill"] <= 15

    combos = [
        ("Fast BD + Fast Fill", bd_fast & fill_fast),
        ("Fast BD + Slow Fill", bd_fast & ~fill_fast),
        ("Slow BD + Fast Fill", ~bd_fast & fill_fast),
        ("Slow BD + Slow Fill", ~bd_fast & ~fill_fast),
    ]

    print(f"\n  {'Combo':25s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>7s} | {'PPDD':>7s}")
    print(f"  {'-'*25}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}")

    for label, mask in combos:
        subset = df[mask].to_dict("records")
        if len(subset) < 5:
            print(f"  {label:25s} | {len(subset):5d} | too few")
            continue
        met = compute_metrics(subset)
        print(f"  {label:25s} | {met['trades']:5d} | {met['R']:+8.1f} | {met['PF']:5.2f} | {met['WR']:5.1f}% | {met['avgR']:+7.4f} | {met['PPDD']:6.02f}")

    # -------------------------------------------------------------------
    # Step 7: Long vs Short breakdown
    # -------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("ANALYSIS 4: BD->FVG Speed by Direction (Long vs Short)")
    print(f"{'='*120}")

    for dir_label, dir_val in [("LONG", 1), ("SHORT", -1)]:
        dfd = df[df["dir"] == dir_val]
        if len(dfd) < 10:
            print(f"\n  {dir_label}: too few trades ({len(dfd)})")
            continue
        print(f"\n  {dir_label} ({len(dfd)} trades):")
        print(f"    {'Bin':15s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>7s}")
        print(f"    {'-'*15}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")

        for label, lo, hi in bins_bd:
            mask = (dfd["bars_bd_to_fvg"] >= lo) & (dfd["bars_bd_to_fvg"] <= hi)
            subset = dfd[mask].to_dict("records")
            if len(subset) < 5:
                print(f"    {label:15s} | {len(subset):5d} | too few")
                continue
            met = compute_metrics(subset)
            print(f"    {label:15s} | {met['trades']:5d} | {met['R']:+8.1f} | {met['PF']:5.2f} | {met['WR']:5.1f}% | {met['avgR']:+7.4f}")

    # -------------------------------------------------------------------
    # Step 8: Total chain delay (BD -> Entry)
    # -------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("ANALYSIS 5: Total Chain Delay — Breakdown -> Entry (bars)")
    print(f"{'='*120}")

    df["total_chain_bars"] = df["entry_bar"] - df["bd_bar"]
    print(f"\n  total_chain_bars distribution:")
    print(f"    min={df['total_chain_bars'].min()}, max={df['total_chain_bars'].max()}, "
          f"mean={df['total_chain_bars'].mean():.1f}, median={df['total_chain_bars'].median():.0f}")

    bins_total = [
        ("2-10 bars",   2,  10),
        ("11-25 bars", 11,  25),
        ("26-50 bars", 26,  50),
        ("51-100 bars", 51, 100),
        ("101+ bars",  101, 9999),
    ]

    print(f"\n  {'Bin':15s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>7s} | {'PPDD':>7s} | {'DD':>6s}")
    print(f"  {'-'*15}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}")

    for label, lo, hi in bins_total:
        mask = (df["total_chain_bars"] >= lo) & (df["total_chain_bars"] <= hi)
        subset = df[mask].to_dict("records")
        if len(subset) < 5:
            print(f"  {label:15s} | {len(subset):5d} | too few")
            continue
        met = compute_metrics(subset)
        print(f"  {label:15s} | {met['trades']:5d} | {met['R']:+8.1f} | {met['PF']:5.2f} | {met['WR']:5.1f}% | {met['avgR']:+7.4f} | {met['PPDD']:6.02f} | {met['MaxDD']:5.1f}R")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")

    # Compute correlation
    r_vals = df["r"].values
    bd_to_fvg = df["bars_bd_to_fvg"].values
    fvg_to_fill = df["bars_fvg_to_fill"].values
    total_chain = df["total_chain_bars"].values

    corr_bd = np.corrcoef(bd_to_fvg, r_vals)[0, 1]
    corr_fill = np.corrcoef(fvg_to_fill, r_vals)[0, 1]
    corr_total = np.corrcoef(total_chain, r_vals)[0, 1]

    print(f"\n  Correlation with R:")
    print(f"    bars_bd_to_fvg   vs R: {corr_bd:+.4f}")
    print(f"    bars_fvg_to_fill vs R: {corr_fill:+.4f}")
    print(f"    total_chain_bars vs R: {corr_total:+.4f}")

    # Win rate by speed
    fast_fvg = df[df["bars_bd_to_fvg"] <= 6]
    slow_fvg = df[df["bars_bd_to_fvg"] > 6]
    print(f"\n  Quick FVG (BD->FVG <=6):  {len(fast_fvg)} trades, WR={(fast_fvg['r']>0).mean()*100:.1f}%, "
          f"avgR={fast_fvg['r'].mean():+.4f}, totalR={fast_fvg['r'].sum():+.1f}")
    print(f"  Slow FVG  (BD->FVG > 6):  {len(slow_fvg)} trades, WR={(slow_fvg['r']>0).mean()*100:.1f}%, "
          f"avgR={slow_fvg['r'].mean():+.4f}, totalR={slow_fvg['r'].sum():+.1f}")

    print(f"\n  Answer: ", end="")
    if fast_fvg["r"].mean() > slow_fvg["r"].mean() + 0.05:
        print("YES — Quick FVGs after breakdown significantly outperform slow ones.")
    elif slow_fvg["r"].mean() > fast_fvg["r"].mean() + 0.05:
        print("NO — Slow FVGs actually outperform quick ones.")
    else:
        print("INCONCLUSIVE — No significant difference between quick and slow FVG formation.")

    print()


if __name__ == "__main__":
    main()
