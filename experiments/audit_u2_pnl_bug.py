"""
Quantify the PnL triple-counting bug in U2 limit order system.

When exit_reason="tp2" and multi_tp is true:
  exit_contracts is set to pos_contracts (ALL contracts)
  But the PnL calculation then adds:
    1. tp1_pnl * trim1_contracts  (correct: TP1 portion)
    2. tp2_pnl * trim2_contracts  (correct: TP2 portion)
    3. (tp2-entry) * exit_contracts = (tp2-entry) * ALL_contracts  (BUG: extra full position)

The runner line (3) should contribute ZERO for TP2 exits because pos_remaining_contracts=0.
But exit_contracts was overwritten to pos_contracts.

This script recalculates R for all trades with the bug fixed.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import logging
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

from experiments.validate_improvements import load_all, compute_metrics, walk_forward_metrics
from experiments.multi_level_tp import prepare_liquidity_data
from experiments.u2_limit_order_fvg import run_limit_order_backtest

SEP = "=" * 120


def main():
    print(SEP)
    print("PNL BUG QUANTIFICATION — U2 Limit Order System")
    print(SEP)

    d = load_all()
    d_extra = prepare_liquidity_data(d)
    params = d["params"]
    point_value = params["position"]["point_value"]
    commission_per_side = params["backtest"]["commission_per_side_micro"]

    # Run with config that claims +3144R
    trades, stats = run_limit_order_backtest(
        d, d_extra,
        stop_strategy="A1", min_stop_mode="B1",
        max_fvg_age=100, fvg_size_mult=0.3,
        block_pm_shorts=True,
    )

    m_buggy = compute_metrics(trades)
    print(f"\nBUGGY results: {m_buggy['trades']}t | R={m_buggy['R']:+.1f} | PPDD={m_buggy['PPDD']:.2f} | PF={m_buggy['PF']:.2f}")

    # Now recalculate R for each trade, fixing the bug
    corrected_trades = []
    total_overcount = 0.0
    tp2_trades_found = 0

    for t in trades:
        new_t = t.copy()

        if t["reason"] == "tp2" and t["trim_stage"] == 2 and t["dir"] == 1:
            tp2_trades_found += 1
            # Reconstruct the correct PnL
            # We know: 50/50 split. tp1_trim_pct = 0.50
            entry_p = t["entry_price"]
            stop_p = t["stop_price"]
            tp1_p = t["tp1_price"]
            tp2_p = t["tp2_price"]
            stop_dist = abs(entry_p - stop_p)

            # Reconstruct contracts
            # The original code uses grade-based R amount, which we don't have here
            # But we have pnl_dollars, so we can work backwards
            # Actually, let's just fix the R directly.

            # Correct PnL for 50/50 split:
            # total_contracts = C
            # trim1 = ceil(C * 0.50)
            # trim2 = C - trim1
            # correct_pnl = (tp1 - entry) * pv * trim1 + (tp2 - entry) * pv * trim2 - comm * 2 * C
            # total_risk = stop_dist * pv * C
            # correct_r = correct_pnl / total_risk

            # From buggy values:
            # buggy_pnl = (tp1-entry)*pv*trim1 + (tp2-entry)*pv*trim2 + (tp2-entry)*pv*C - comm*2*C
            # buggy_risk = stop_dist * pv * C
            # buggy_r = t["r"]

            # The overcount = (tp2 - entry) * pv * C
            # buggy_risk = stop_dist * pv * C
            # overcount_r = (tp2 - entry) * pv * C / (stop_dist * pv * C) = (tp2 - entry) / stop_dist

            overcount_r = (tp2_p - entry_p) / stop_dist if stop_dist > 0 else 0
            correct_r = t["r"] - overcount_r

            total_overcount += overcount_r
            new_t["r"] = correct_r
            new_t["r_buggy"] = t["r"]
            new_t["overcount_r"] = overcount_r

        corrected_trades.append(new_t)

    m_corrected = compute_metrics(corrected_trades)
    print(f"\nCORRECTED results: {m_corrected['trades']}t | R={m_corrected['R']:+.1f} | PPDD={m_corrected['PPDD']:.2f} | PF={m_corrected['PF']:.2f}")
    print(f"\nTP2 trades: {tp2_trades_found}")
    print(f"Total R overcount: {total_overcount:+.1f}")
    print(f"R reduction: {m_buggy['R'] - m_corrected['R']:.1f} ({100*(m_buggy['R'] - m_corrected['R'])/m_buggy['R']:.1f}%)")

    # Also fix for stop/be_sweep/eod exits AFTER TP1 trim
    # When trim_stage >= 1 and exit is stop/be_sweep/eod_close:
    #   exit_contracts = pos_remaining_contracts (line 340, correct!)
    #   pnl_runner = (exit_price - entry) * pv * remaining_contracts
    #   BUT also adds tp1_pnl * trim1_contracts
    # This SHOULD be correct for stop/eod exits because remaining = original - trimmed
    # Let's verify by checking if exit_contracts was correct in those cases

    # For stop exits with trim_stage>=1:
    # exit_contracts = pos_remaining_contracts (set at line 340)
    # For stop, exit_contracts is NOT overwritten. So runner uses correct remaining.
    # Total = tp1_pnl * trim1 + (stop - entry) * remaining
    # This is correct.

    # So the bug ONLY affects tp2 exits (and tp1 exits with multi-tp where all contracts go to TP1)
    # For tp1 exit with multi_tp (line 378: exit_contracts = pos_contracts),
    # but pnl_pts_runner = (tp1 - entry) and trim_stage = 1
    # PnL = tp1_pnl * trim1 + (tp1 - entry) * pos_contracts
    # This ALSO double counts! tp1_pnl = tp1 - entry, so:
    # total = (tp1-entry)*trim1 + (tp1-entry)*ALL = (tp1-entry)*(trim1+ALL)
    # Should be: (tp1-entry)*ALL
    # Overcount = (tp1-entry)*trim1
    # But wait, this only happens when pos_remaining_contracts <= 0 after TP1 trim.
    # That means trim1 = ALL contracts (ceil(C*0.50) = C when C=1).
    # So trim1 = 1, ALL = 1, overcount = (tp1-entry)*1
    # And correct = (tp1-entry)*1
    # So buggy = 2x correct for these single-contract TP1 exits.

    # Let's also fix TP1 multi-tp exits
    corrected_trades2 = []
    tp1_overcount = 0.0
    tp1_multi_found = 0

    for t in corrected_trades:
        new_t = t.copy()
        if t["reason"] == "tp1" and t.get("trim_stage", 0) >= 1 and t["dir"] == 1:
            # This is a multi-tp trade where TP1 exit took all contracts
            # Correct PnL = (tp1 - entry) * ALL - comm
            # Buggy PnL also includes tp1_pnl * trim1 from the trim stage
            # But trim1 = ALL (since remaining=0), so:
            # buggy = (tp1-entry)*ALL + (tp1-entry)*ALL - comm = 2*(tp1-entry)*ALL - comm
            # correct = (tp1-entry)*ALL - comm

            entry_p = t["entry_price"]
            tp1_p = t["tp1_price"]
            stop_dist = abs(entry_p - t["stop_price"])

            overcount_r = (tp1_p - entry_p) / stop_dist if stop_dist > 0 else 0
            # But wait — this only applies if pos_remaining_contracts was 0
            # which means ceil(contracts * 0.50) = contracts, i.e. contracts = 1
            # For contracts=1, trim1=1, remaining=0 -> full TP1 exit
            # The overcount is trim1 portion PLUS the runner portion
            # Actually let me think again...

            # When tp1_trim_pct=0.50 and contracts=1:
            # trim1 = ceil(1 * 0.50) = 1
            # remaining = 1 - 1 = 0 -> exit
            # exit_contracts = pos_contracts = 1
            # PnL = tp1_pnl * 1 + (tp1-entry) * 1 = 2 * (tp1-entry)
            # But correct = (tp1-entry) * 1
            # So overcount = (tp1-entry)

            # For contracts > 1 (e.g., 2):
            # trim1 = ceil(2*0.50) = 1
            # remaining = 2-1 = 1 > 0, NO EXIT at TP1 stage
            # So this path only triggers for contracts=1

            tp1_multi_found += 1
            new_t["r"] = t["r"] - overcount_r
            tp1_overcount += overcount_r

        corrected_trades2.append(new_t)

    m_corrected2 = compute_metrics(corrected_trades2)
    print(f"\nFULLY CORRECTED (TP2+TP1 bug): {m_corrected2['trades']}t | R={m_corrected2['R']:+.1f} | PPDD={m_corrected2['PPDD']:.2f} | PF={m_corrected2['PF']:.2f}")
    print(f"TP1 multi-tp exits fixed: {tp1_multi_found}")
    print(f"TP1 overcount: {tp1_overcount:+.1f}")
    print(f"Total overcount (TP2+TP1): {total_overcount + tp1_overcount:+.1f}")

    # Walk-forward for corrected
    print(f"\nWalk-forward (fully corrected):")
    wf = walk_forward_metrics(corrected_trades2)
    if wf:
        neg = sum(1 for yr in wf if yr["R"] < 0)
        for yr in wf:
            print(f"  {yr['year']}: {yr['n']}t | R={yr['R']:+.1f} | WR={yr['WR']:.1f}% | PF={yr['PF']:.2f}")
        print(f"  Negative years: {neg}/{len(wf)}")

    # Also run the A1_B1_C4_D1 config (best)
    print(f"\n{'='*80}")
    print("ALSO: Best config A1_B1_C4_D1")
    print(f"{'='*80}")

    trades2, stats2 = run_limit_order_backtest(
        d, d_extra,
        stop_strategy="A1", min_stop_mode="B1",
        max_fvg_age=500, fvg_size_mult=0.3,
        block_pm_shorts=True,
    )

    m2_buggy = compute_metrics(trades2)
    print(f"BUGGY: {m2_buggy['trades']}t | R={m2_buggy['R']:+.1f}")

    corrected2 = []
    overcount2 = 0.0
    for t in trades2:
        new_t = t.copy()
        if t["reason"] == "tp2" and t["trim_stage"] == 2 and t["dir"] == 1:
            entry_p = t["entry_price"]
            tp2_p = t["tp2_price"]
            stop_dist = abs(entry_p - t["stop_price"])
            oc = (tp2_p - entry_p) / stop_dist if stop_dist > 0 else 0
            new_t["r"] = t["r"] - oc
            overcount2 += oc
        elif t["reason"] == "tp1" and t.get("trim_stage", 0) >= 1 and t["dir"] == 1:
            entry_p = t["entry_price"]
            tp1_p = t["tp1_price"]
            stop_dist = abs(entry_p - t["stop_price"])
            oc = (tp1_p - entry_p) / stop_dist if stop_dist > 0 else 0
            new_t["r"] = t["r"] - oc
            overcount2 += oc
        corrected2.append(new_t)

    m2_corrected = compute_metrics(corrected2)
    print(f"CORRECTED: {m2_corrected['trades']}t | R={m2_corrected['R']:+.1f} | PPDD={m2_corrected['PPDD']:.2f} | PF={m2_corrected['PF']:.2f}")
    print(f"Overcount: {overcount2:+.1f} ({100*overcount2/m2_buggy['R']:.1f}%)")

    # Direction breakdown corrected
    longs_c = [t for t in corrected2 if t["dir"] == 1]
    shorts_c = [t for t in corrected2 if t["dir"] == -1]
    m_lc = compute_metrics(longs_c)
    m_sc = compute_metrics(shorts_c) if shorts_c else None
    print(f"\nCorrected longs: {m_lc['trades']}t | R={m_lc['R']:+.1f} | avgR={m_lc['avgR']:+.4f}")
    if m_sc:
        print(f"Corrected shorts: {m_sc['trades']}t | R={m_sc['R']:+.1f} | avgR={m_sc['avgR']:+.4f}")

    # Walk-forward corrected
    print(f"\nWalk-forward (corrected):")
    wf2 = walk_forward_metrics(corrected2)
    if wf2:
        neg = sum(1 for yr in wf2 if yr["R"] < 0)
        for yr in wf2:
            avg_r = yr['R'] / yr['n'] if yr['n'] > 0 else 0
            print(f"  {yr['year']}: {yr['n']}t | R={yr['R']:+.1f} | avgR={avg_r:+.4f} | WR={yr['WR']:.1f}% | PF={yr['PF']:.2f}")
        print(f"  Negative years: {neg}/{len(wf2)}")


if __name__ == "__main__":
    main()
