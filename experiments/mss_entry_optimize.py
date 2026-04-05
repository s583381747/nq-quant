"""
experiments/mss_entry_optimize.py -- MSS Entry Timing Optimization
===================================================================
Analyze the sequence: Breakdown -> MSS -> FVG -> Fill
Find: when does MSS happen relative to FVG? Can we optimize entry timing?
"""
from __future__ import annotations
import sys, copy
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.breakdown_chain_research import detect_breakdowns, find_fvg_after_breakdown
from experiments.sweep_research import compute_pdhl
from experiments.mss_research import detect_mss_between
from experiments.shorts_runner_test import run_with_symmetric_management
from experiments.u2_clean import load_all, compute_metrics
from features.swing import compute_swing_levels


def main():
    d = load_all()
    nq, n = d["nq"], d["n"]
    h, l, c, o = d["h"], d["l"], d["c"], d["o"]
    atr_arr = d["atr_arr"]
    params = d["params"]
    sc = pd.read_parquet(PROJECT / "data" / "cache_session_levels_10yr_v2.parquet")
    pdhl = compute_pdhl(nq)

    rb = params["swing"]["right_bars"]
    swings = compute_swing_levels(nq, {"left_bars": params["swing"]["left_bars"], "right_bars": rb})
    swing_hi_shifted = swings["swing_high"].shift(rb, fill_value=False).values
    swing_lo_shifted = swings["swing_low"].shift(rb, fill_value=False).values
    raw_sh, raw_sl = swings["swing_high"].values, swings["swing_low"].values
    sh_prices = np.full(n, np.nan)
    sl_prices = np.full(n, np.nan)
    for j in range(n):
        if raw_sh[j] and j + rb < n: sh_prices[j + rb] = h[j]
        if raw_sl[j] and j + rb < n: sl_prices[j + rb] = l[j]

    on_lo = detect_breakdowns(h, l, c, sc["overnight_low"].values, "low", min_depth_pts=1.0)
    for bd in on_lo: bd["level_type"] = "low"
    on_hi = detect_breakdowns(h, l, c, sc["overnight_high"].values, "high", min_depth_pts=1.0)
    for bd in on_hi: bd["level_type"] = "high"
    all_bds = on_lo + on_hi

    fvg_df = d["fvg_df"]
    fvg_bm = fvg_df["fvg_bull"].values
    fvg_bt = fvg_df["fvg_bull_top"].values
    fvg_bb = fvg_df["fvg_bull_bottom"].values
    fvg_em = fvg_df["fvg_bear"].values
    fvg_et = fvg_df["fvg_bear_top"].values
    fvg_eb = fvg_df["fvg_bear_bottom"].values

    print("=" * 100)
    print("MSS ENTRY TIMING OPTIMIZATION")
    print("=" * 100)

    # ================================================================
    # Analyze sequence timing
    # ================================================================
    events = []
    for bd in all_bds:
        direction = 1 if bd["level_type"] == "low" else -1
        bd_bar = bd["bar_idx"]

        fvg = find_fvg_after_breakdown(fvg_bm, fvg_bt, fvg_bb, fvg_em, fvg_et, fvg_eb,
            atr_arr, bd_bar, bd["level_type"], 30, 0.0, 99.0, 0.3, h, l, c, o)
        if fvg is None:
            continue
        fvg_bar = fvg["fvg_bar"]

        mss = detect_mss_between(h, l, c, o, swing_hi_shifted, swing_lo_shifted,
            sh_prices, sl_prices, bd_bar, bd_bar + 50, direction)
        mss_bar = mss["mss_bar"] if mss["mss_found"] else -1

        # Does the FVG displacement candle itself break the swing?
        disp_bar = fvg_bar - 1
        fvg_is_mss = False
        if direction == 1 and disp_bar > 0:
            for j in range(bd_bar, max(0, bd_bar - 100), -1):
                if swing_hi_shifted[j]:
                    p = sh_prices[j]
                    if not np.isnan(p) and p > 0 and h[disp_bar] > p:
                        fvg_is_mss = True
                        break
        elif direction == -1 and disp_bar > 0:
            for j in range(bd_bar, max(0, bd_bar - 100), -1):
                if swing_lo_shifted[j]:
                    p = sl_prices[j]
                    if not np.isnan(p) and p > 0 and l[disp_bar] < p:
                        fvg_is_mss = True
                        break

        if not mss["mss_found"]:
            seq = "no_mss"
        elif abs(mss_bar - fvg_bar) <= 1:
            seq = "simultaneous"
        elif mss_bar < fvg_bar:
            seq = "mss_then_fvg"
        else:
            seq = "fvg_then_mss"

        events.append({
            "dir": direction, "bd_bar": bd_bar, "fvg_bar": fvg_bar,
            "mss_bar": mss_bar, "mss_found": mss["mss_found"],
            "sequence": seq, "fvg_is_mss": fvg_is_mss,
            "bars_bd_to_fvg": fvg_bar - bd_bar,
        })

    df = pd.DataFrame(events)

    print(f"\nSequence distribution ({len(df)} pairs):")
    for seq, grp in df.groupby("sequence"):
        print(f"  {seq:25s}: {len(grp):5d} ({len(grp)/len(df)*100:.1f}%)")

    print(f"\nFVG displacement IS the MSS:")
    print(f"  Yes: {df['fvg_is_mss'].sum()} ({df['fvg_is_mss'].mean()*100:.1f}%)")
    print(f"  No:  {(~df['fvg_is_mss']).sum()}")

    print(f"\nBD->FVG timing:")
    no_mss = df[df["sequence"] == "no_mss"]
    has_mss = df[df["mss_found"]]
    print(f"  No MSS:  BD->FVG avg {no_mss['bars_bd_to_fvg'].mean():.1f} bars (med {no_mss['bars_bd_to_fvg'].median():.0f})")
    print(f"  Has MSS: BD->FVG avg {has_mss['bars_bd_to_fvg'].mean():.1f} bars (med {has_mss['bars_bd_to_fvg'].median():.0f})")

    # ================================================================
    # Backtest different entry filters
    # ================================================================
    print(f"\n{'='*100}")
    print("BACKTEST: Entry Timing Filters (Fixed 1R TP)")
    print(f"{'='*100}")

    d_fixed = copy.copy(d)
    d_fixed["irl_high_arr"] = np.full(n, np.nan)
    d_fixed["irl_low_arr"] = np.full(n, np.nan)

    # Build filtered breakdown lists
    bd_no_mss = []
    bd_fvg_is_mss = []
    bd_fvg_not_mss = []
    bd_early = []       # FVG within 5 bars
    bd_medium = []      # FVG 5-15 bars
    bd_late = []        # FVG 15+ bars
    bd_simultaneous = []
    bd_mss_then_fvg = []

    for bd in all_bds:
        direction = 1 if bd["level_type"] == "low" else -1
        bd_bar = bd["bar_idx"]
        fvg = find_fvg_after_breakdown(fvg_bm, fvg_bt, fvg_bb, fvg_em, fvg_et, fvg_eb,
            atr_arr, bd_bar, bd["level_type"], 30, 0.0, 99.0, 0.3, h, l, c, o)
        if fvg is None:
            continue
        fvg_bar = fvg["fvg_bar"]
        bars_to_fvg = fvg_bar - bd_bar

        mss = detect_mss_between(h, l, c, o, swing_hi_shifted, swing_lo_shifted,
            sh_prices, sl_prices, bd_bar, bd_bar + 50, direction)
        mss_bar = mss["mss_bar"] if mss["mss_found"] else -1

        disp_bar = fvg_bar - 1
        is_fvg_mss = False
        if direction == 1 and disp_bar > 0:
            for j in range(bd_bar, max(0, bd_bar - 100), -1):
                if swing_hi_shifted[j]:
                    p = sh_prices[j]
                    if not np.isnan(p) and p > 0 and h[disp_bar] > p:
                        is_fvg_mss = True
                        break
        elif direction == -1 and disp_bar > 0:
            for j in range(bd_bar, max(0, bd_bar - 100), -1):
                if swing_lo_shifted[j]:
                    p = sl_prices[j]
                    if not np.isnan(p) and p > 0 and l[disp_bar] < p:
                        is_fvg_mss = True
                        break

        if not mss["mss_found"]:
            bd_no_mss.append(bd)
        elif abs(mss_bar - fvg_bar) <= 1:
            bd_simultaneous.append(bd)
        elif mss_bar < fvg_bar:
            bd_mss_then_fvg.append(bd)

        if is_fvg_mss:
            bd_fvg_is_mss.append(bd)
        else:
            bd_fvg_not_mss.append(bd)

        if bars_to_fvg <= 5:
            bd_early.append(bd)
        elif bars_to_fvg <= 15:
            bd_medium.append(bd)
        else:
            bd_late.append(bd)

    configs = [
        ("Baseline (all)", all_bds),
        ("No MSS", bd_no_mss),
        ("FVG IS the MSS (simultaneous)", bd_fvg_is_mss),
        ("FVG is NOT MSS", bd_fvg_not_mss),
        ("MSS before FVG (late entry)", bd_mss_then_fvg),
        ("MSS simultaneous with FVG", bd_simultaneous),
        ("Early FVG (<=5 bars after BD)", bd_early),
        ("Medium FVG (5-15 bars)", bd_medium),
        ("Late FVG (15+ bars)", bd_late),
        ("No MSS + early FVG", [bd for bd in bd_no_mss if bd in bd_early]),
    ]

    # Also build: no_mss AND early
    no_mss_early = []
    for bd in all_bds:
        direction = 1 if bd["level_type"] == "low" else -1
        bd_bar = bd["bar_idx"]
        fvg = find_fvg_after_breakdown(fvg_bm, fvg_bt, fvg_bb, fvg_em, fvg_et, fvg_eb,
            atr_arr, bd_bar, bd["level_type"], 30, 0.0, 99.0, 0.3, h, l, c, o)
        if fvg is None: continue
        bars_to_fvg = fvg["fvg_bar"] - bd_bar
        mss = detect_mss_between(h, l, c, o, swing_hi_shifted, swing_lo_shifted,
            sh_prices, sl_prices, bd_bar, bd_bar + 50, direction)
        if not mss["mss_found"] and bars_to_fvg <= 5:
            no_mss_early.append(bd)

    configs.append(("No MSS + early FVG (<=5)", no_mss_early))

    print(f"\n  {'Config':45s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'PPDD':>7s} | {'DD':>6s} | {'L:PF':>6s} | {'S:PF':>6s}")
    print(f"  {'-'*45}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    for label, bds in configs:
        if len(bds) < 10:
            print(f"  {label:45s} | {len(bds):5d} | too few")
            continue
        trades = run_with_symmetric_management(d_fixed, bds, tp_mult=0.5)
        m = compute_metrics(trades)
        longs = [t for t in trades if t["dir"] == 1]
        shorts = [t for t in trades if t["dir"] == -1]
        ml = compute_metrics(longs) if longs else {"PF": 0}
        ms = compute_metrics(shorts) if shorts else {"PF": 0}
        print(f"  {label:45s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['PPDD']:6.2f} | {m['MaxDD']:5.1f}R | {ml['PF']:5.2f} | {ms['PF']:5.2f}")


if __name__ == "__main__":
    main()
