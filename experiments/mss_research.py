"""
experiments/mss_research.py -- Market Structure Shift Research
==============================================================

MSS = after breakdown at significant level, price breaks a swing point
in the REVERSAL direction, confirming the stop hunt led to a real reversal.

For bull setup (breakdown of low level):
  1. Price breaks below overnight low (breakdown)
  2. Then price breaks ABOVE a recent swing high (MSS = bullish structure shift)
  3. Then bull FVG forms
  4. Limit entry at FVG

Question: does requiring MSS improve PF? Or does it just cut trades?

Also test: what if the FVG itself IS the MSS (the displacement that creates the FVG
also breaks a swing high)?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import copy

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.breakdown_chain_research import detect_breakdowns, find_fvg_after_breakdown
from experiments.sweep_research import compute_pdhl
from experiments.shorts_runner_test import run_with_symmetric_management
from experiments.u2_clean import load_all, compute_metrics, pr
from features.swing import compute_swing_levels


def detect_mss_between(h, l, c, o, swing_hi_shifted, swing_lo_shifted,
                        sh_prices, sl_prices,
                        start_bar, end_bar, direction):
    """Detect if MSS occurred between start_bar and end_bar.

    For bull MSS (direction=1): price breaks above a recent swing high
    For bear MSS (direction=-1): price breaks below a recent swing low

    Returns dict with mss_found, mss_bar, swing_price_broken.
    """
    result = {"mss_found": False, "mss_bar": -1, "swing_broken": 0.0, "mss_displacement": 0.0}

    if direction == 1:
        # Find the most recent swing high BEFORE start_bar
        recent_sh = 0.0
        for j in range(start_bar, max(0, start_bar - 100), -1):
            if swing_hi_shifted[j]:
                p = sh_prices[j]
                if not np.isnan(p) and p > 0:
                    recent_sh = p
                    break
        if recent_sh <= 0:
            return result

        # Check if price breaks above this swing high between start and end
        for j in range(start_bar + 1, min(end_bar + 1, len(h))):
            if h[j] > recent_sh:
                result["mss_found"] = True
                result["mss_bar"] = j
                result["swing_broken"] = recent_sh
                result["mss_displacement"] = h[j] - recent_sh
                break

    else:  # bear MSS
        recent_sl = 0.0
        for j in range(start_bar, max(0, start_bar - 100), -1):
            if swing_lo_shifted[j]:
                p = sl_prices[j]
                if not np.isnan(p) and p > 0:
                    recent_sl = p
                    break
        if recent_sl <= 0:
            return result

        for j in range(start_bar + 1, min(end_bar + 1, len(l))):
            if l[j] < recent_sl:
                result["mss_found"] = True
                result["mss_bar"] = j
                result["swing_broken"] = recent_sl
                result["mss_displacement"] = recent_sl - l[j]
                break

    return result


def main():
    d_orig = load_all()
    nq, n = d_orig["nq"], d_orig["n"]
    h, l, c, o = d_orig["h"], d_orig["l"], d_orig["c"], d_orig["o"]
    atr_arr = d_orig["atr_arr"]
    params = d_orig["params"]
    session_cache = pd.read_parquet(PROJECT / "data" / "cache_session_levels_10yr_v2.parquet")
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

    # Breakdowns
    on_lo = detect_breakdowns(h, l, c, session_cache["overnight_low"].values, "low", min_depth_pts=1.0)
    for bd in on_lo: bd["level_type"] = "low"
    on_hi = detect_breakdowns(h, l, c, session_cache["overnight_high"].values, "high", min_depth_pts=1.0)
    for bd in on_hi: bd["level_type"] = "high"
    all_bds = on_lo + on_hi

    # FVG detection
    fvg_df = d_orig["fvg_df"]
    fvg_bm = fvg_df["fvg_bull"].values
    fvg_bt = fvg_df["fvg_bull_top"].values
    fvg_bb = fvg_df["fvg_bull_bottom"].values
    fvg_em = fvg_df["fvg_bear"].values
    fvg_et = fvg_df["fvg_bear_top"].values
    fvg_eb = fvg_df["fvg_bear_bottom"].values

    print("=" * 120)
    print("MSS (Market Structure Shift) RESEARCH")
    print("=" * 120)

    # ================================================================
    # STEP 1: For each breakdown->FVG pair, check if MSS occurred between them
    # ================================================================
    print("\n[ANALYZE] Checking MSS between breakdown and FVG formation...")

    mss_stats = {"total": 0, "mss_before_fvg": 0, "mss_at_fvg": 0, "no_mss": 0}
    bd_with_mss = []
    bd_without_mss = []

    for bd in all_bds:
        direction = 1 if bd["level_type"] == "low" else -1
        fvg = find_fvg_after_breakdown(fvg_bm, fvg_bt, fvg_bb, fvg_em, fvg_et, fvg_eb,
            atr_arr, bd["bar_idx"], bd["level_type"], 30, 0.0, 99.0, 0.3, h, l, c, o)
        if fvg is None:
            continue

        mss_stats["total"] += 1

        # Check MSS between breakdown and FVG formation
        mss = detect_mss_between(h, l, c, o, swing_hi_shifted, swing_lo_shifted,
                                  sh_prices, sl_prices,
                                  bd["bar_idx"], fvg["fvg_bar"], direction)

        bd_copy = bd.copy()
        bd_copy["_has_mss"] = mss["mss_found"]

        if mss["mss_found"]:
            if mss["mss_bar"] < fvg["fvg_bar"]:
                mss_stats["mss_before_fvg"] += 1
            else:
                mss_stats["mss_at_fvg"] += 1
            bd_with_mss.append(bd_copy)
        else:
            mss_stats["no_mss"] += 1
            bd_without_mss.append(bd_copy)

    print(f"  Total breakdown->FVG pairs: {mss_stats['total']}")
    print(f"  MSS before FVG:  {mss_stats['mss_before_fvg']} ({mss_stats['mss_before_fvg']/mss_stats['total']*100:.1f}%)")
    print(f"  MSS at FVG bar:  {mss_stats['mss_at_fvg']} ({mss_stats['mss_at_fvg']/mss_stats['total']*100:.1f}%)")
    print(f"  No MSS:          {mss_stats['no_mss']} ({mss_stats['no_mss']/mss_stats['total']*100:.1f}%)")

    # ================================================================
    # STEP 2: Run backtest with MSS filter vs without
    # ================================================================
    print(f"\n{'='*120}")
    print("BACKTEST: MSS filter effect")
    print(f"{'='*120}")

    # Use fixed 1R TP (user's chosen baseline)
    d_fixed = copy.copy(d_orig)
    d_fixed["irl_high_arr"] = np.full(n, np.nan)
    d_fixed["irl_low_arr"] = np.full(n, np.nan)
    tp_mult_1r = 0.5  # fixed 1R = sd*2 * 0.5 = sd

    # Baseline: all breakdowns, no MSS filter
    trades_all = run_with_symmetric_management(d_fixed, all_bds, tp_mult=tp_mult_1r)
    m_all = compute_metrics(trades_all)

    # With MSS only
    trades_mss = run_with_symmetric_management(d_fixed, bd_with_mss, tp_mult=tp_mult_1r)
    m_mss = compute_metrics(trades_mss)

    # Without MSS
    trades_no_mss = run_with_symmetric_management(d_fixed, bd_without_mss, tp_mult=tp_mult_1r)
    m_no_mss = compute_metrics(trades_no_mss)

    print(f"\n  {'Config':40s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'PPDD':>7s} | {'DD':>6s}")
    print(f"  {'-'*40}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}")
    for label, m in [("Baseline (all, no MSS filter)", m_all),
                      ("MSS confirmed only", m_mss),
                      ("No MSS (MSS not found)", m_no_mss)]:
        tpd = m["trades"] / (252 * 10.5)
        print(f"  {label:40s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['PPDD']:6.2f} | {m['MaxDD']:5.1f}R")

    # Long/short split
    print(f"\n  --- Long/Short split ---")
    for label, trades in [("All", trades_all), ("MSS only", trades_mss), ("No MSS", trades_no_mss)]:
        longs = [t for t in trades if t["dir"] == 1]
        shorts = [t for t in trades if t["dir"] == -1]
        ml = compute_metrics(longs) if longs else {"trades": 0, "PF": 0, "R": 0}
        ms = compute_metrics(shorts) if shorts else {"trades": 0, "PF": 0, "R": 0}
        print(f"  {label:15s} L: {ml['trades']:4d}t PF={ml['PF']:5.2f} R={ml['R']:+8.1f} | S: {ms['trades']:4d}t PF={ms['PF']:5.2f} R={ms['R']:+8.1f}")

    # ================================================================
    # STEP 3: Try wider MSS window (maybe MSS happens during FVG, not before)
    # ================================================================
    print(f"\n{'='*120}")
    print("MSS WINDOW SWEEP: How many bars to look for MSS after breakdown?")
    print(f"{'='*120}")

    for mss_window in [5, 10, 15, 20, 30, 50]:
        bd_mss_w = []
        bd_nomss_w = []
        for bd in all_bds:
            direction = 1 if bd["level_type"] == "low" else -1
            end_bar = min(bd["bar_idx"] + mss_window, n - 1)
            mss = detect_mss_between(h, l, c, o, swing_hi_shifted, swing_lo_shifted,
                                      sh_prices, sl_prices,
                                      bd["bar_idx"], end_bar, direction)
            if mss["mss_found"]:
                bd_mss_w.append(bd)
            else:
                bd_nomss_w.append(bd)

        pct = len(bd_mss_w) / (len(bd_mss_w) + len(bd_nomss_w)) * 100 if (len(bd_mss_w) + len(bd_nomss_w)) > 0 else 0

        trades_w = run_with_symmetric_management(d_fixed, bd_mss_w, tp_mult=tp_mult_1r)
        trades_nw = run_with_symmetric_management(d_fixed, bd_nomss_w, tp_mult=tp_mult_1r)
        m_w = compute_metrics(trades_w)
        m_nw = compute_metrics(trades_nw)

        print(f"  Window {mss_window:3d} bars | MSS: {pct:5.1f}% | "
              f"MSS trades: {m_w['trades']:5d}t PF={m_w['PF']:5.2f} R={m_w['R']:+8.1f} | "
              f"No-MSS: {m_nw['trades']:5d}t PF={m_nw['PF']:5.2f} R={m_nw['R']:+8.1f}")

    # ================================================================
    # STEP 4: HTF MSS (use larger swing points)
    # ================================================================
    print(f"\n{'='*120}")
    print("HTF MSS: Use larger swing params for MSS detection")
    print(f"{'='*120}")

    for left, right in [(6, 2), (12, 3), (24, 6)]:
        htf_swings = compute_swing_levels(nq, {"left_bars": left, "right_bars": right})
        htf_hi_shifted = htf_swings["swing_high"].shift(right, fill_value=False).values
        htf_lo_shifted = htf_swings["swing_low"].shift(right, fill_value=False).values
        htf_sh_prices = np.full(n, np.nan)
        htf_sl_prices = np.full(n, np.nan)
        raw_htf_sh = htf_swings["swing_high"].values
        raw_htf_sl = htf_swings["swing_low"].values
        for j in range(n):
            if raw_htf_sh[j] and j + right < n: htf_sh_prices[j + right] = h[j]
            if raw_htf_sl[j] and j + right < n: htf_sl_prices[j + right] = l[j]

        bd_mss_htf = []
        bd_nomss_htf = []
        for bd in all_bds:
            direction = 1 if bd["level_type"] == "low" else -1
            fvg = find_fvg_after_breakdown(fvg_bm, fvg_bt, fvg_bb, fvg_em, fvg_et, fvg_eb,
                atr_arr, bd["bar_idx"], bd["level_type"], 30, 0.0, 99.0, 0.3, h, l, c, o)
            if fvg is None: continue
            end_bar = fvg["fvg_bar"]
            mss = detect_mss_between(h, l, c, o, htf_hi_shifted, htf_lo_shifted,
                                      htf_sh_prices, htf_sl_prices,
                                      bd["bar_idx"], end_bar, direction)
            if mss["mss_found"]:
                bd_mss_htf.append(bd)
            else:
                bd_nomss_htf.append(bd)

        pct = len(bd_mss_htf) / (len(bd_mss_htf) + len(bd_nomss_htf)) * 100

        trades_htf = run_with_symmetric_management(d_fixed, bd_mss_htf, tp_mult=tp_mult_1r)
        trades_no_htf = run_with_symmetric_management(d_fixed, bd_nomss_htf, tp_mult=tp_mult_1r)
        m_htf = compute_metrics(trades_htf)
        m_no_htf = compute_metrics(trades_no_htf)

        print(f"  L={left:2d} R={right:1d} | MSS: {pct:5.1f}% | "
              f"MSS: {m_htf['trades']:5d}t PF={m_htf['PF']:5.2f} R={m_htf['R']:+8.1f} | "
              f"No-MSS: {m_no_htf['trades']:5d}t PF={m_no_htf['PF']:5.2f} R={m_no_htf['R']:+8.1f}")

    print(f"\n{'='*120}")
    print("VERDICT: Does MSS improve the chain strategy?")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
