"""
Verify the PnL bug by tracing a single TP2 trade step by step.
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

from experiments.validate_improvements import load_all, compute_metrics
from experiments.multi_level_tp import prepare_liquidity_data
from experiments.u2_limit_order_fvg import run_limit_order_backtest

SEP = "=" * 100


def main():
    print(SEP)
    print("SINGLE TRADE VERIFICATION — PnL Bug")
    print(SEP)

    d = load_all()
    d_extra = prepare_liquidity_data(d)
    params = d["params"]
    point_value = params["position"]["point_value"]  # $2 (micro)
    commission = params["backtest"]["commission_per_side_micro"]  # $0.62

    trades, _ = run_limit_order_backtest(
        d, d_extra,
        stop_strategy="A1", min_stop_mode="B1",
        max_fvg_age=100, fvg_size_mult=0.3,
        block_pm_shorts=True,
    )

    # Find a TP2 trade with clear numbers
    tp2_trades = [t for t in trades if t["reason"] == "tp2" and t["trim_stage"] == 2 and t["dir"] == 1]
    print(f"\nFound {len(tp2_trades)} TP2 trades")

    # Pick the first one with reasonable numbers
    for t in tp2_trades[:5]:
        entry = t["entry_price"]
        stop = t["stop_price"]
        tp1 = t["tp1_price"]
        tp2 = t["tp2_price"]
        exit_p = t["exit_price"]
        stop_dist = abs(entry - stop)

        # Reconstruct contracts
        # Grade affects R amount; we'll work from pnl_dollars
        pnl_dollars = t["pnl_dollars"]
        r_reported = t["r"]

        print(f"\n{'='*80}")
        print(f"Trade: {t['entry_time']} -> {t['exit_time']}")
        print(f"  Entry:     {entry:.2f}")
        print(f"  Stop:      {stop:.2f} (dist={stop_dist:.2f})")
        print(f"  TP1:       {tp1:.2f} (dist from entry={tp1-entry:.2f})")
        print(f"  TP2:       {tp2:.2f} (dist from entry={tp2-entry:.2f})")
        print(f"  Exit:      {exit_p:.2f}")
        print(f"  PnL:       ${pnl_dollars:.2f}")
        print(f"  R:         {r_reported:.4f}")
        print(f"  Grade:     {t['grade']}")

        # Back out contracts from total_risk = stop_dist * pv * contracts
        # r_mult = pnl / total_risk
        # total_risk = pnl / r_mult
        if r_reported != 0:
            total_risk = pnl_dollars / r_reported
            contracts = total_risk / (stop_dist * point_value)
            print(f"  Implied contracts: {contracts:.1f}")
            contracts = int(round(contracts))
        else:
            contracts = 1

        # 50/50 split
        trim1 = max(1, int(np.ceil(contracts * 0.50)))
        trim2 = contracts - trim1
        print(f"  Contracts: {contracts} (trim1={trim1}, trim2={trim2})")

        # BUGGY PnL calculation (as in code):
        tp1_pnl_pts = tp1 - entry
        tp2_pnl_pts = tp2 - entry
        runner_pnl_pts = exit_p - entry  # exit_p = tp2 for tp2 exits

        buggy_pnl = (tp1_pnl_pts * point_value * trim1 +
                     tp2_pnl_pts * point_value * trim2 +
                     runner_pnl_pts * point_value * contracts -  # BUG: ALL contracts
                     commission * 2 * contracts)
        buggy_risk = stop_dist * point_value * contracts
        buggy_r = buggy_pnl / buggy_risk if buggy_risk > 0 else 0

        print(f"\n  BUGGY PnL breakdown:")
        print(f"    TP1 portion: {tp1_pnl_pts:.2f}pts * ${point_value} * {trim1}c = ${tp1_pnl_pts * point_value * trim1:.2f}")
        print(f"    TP2 portion: {tp2_pnl_pts:.2f}pts * ${point_value} * {trim2}c = ${tp2_pnl_pts * point_value * trim2:.2f}")
        print(f"    Runner port: {runner_pnl_pts:.2f}pts * ${point_value} * {contracts}c = ${runner_pnl_pts * point_value * contracts:.2f} <<< BUG")
        print(f"    Commission:  -${commission * 2 * contracts:.2f}")
        print(f"    Total PnL:   ${buggy_pnl:.2f}")
        print(f"    Total risk:  ${buggy_risk:.2f}")
        print(f"    Buggy R:     {buggy_r:.4f} (reported: {r_reported:.4f})")

        # CORRECT PnL calculation:
        correct_pnl = (tp1_pnl_pts * point_value * trim1 +
                       tp2_pnl_pts * point_value * trim2 +
                       0 -  # remaining = 0, runner contributes nothing
                       commission * 2 * contracts)
        correct_r = correct_pnl / buggy_risk if buggy_risk > 0 else 0

        print(f"\n  CORRECT PnL breakdown:")
        print(f"    TP1 portion: {tp1_pnl_pts:.2f}pts * ${point_value} * {trim1}c = ${tp1_pnl_pts * point_value * trim1:.2f}")
        print(f"    TP2 portion: {tp2_pnl_pts:.2f}pts * ${point_value} * {trim2}c = ${tp2_pnl_pts * point_value * trim2:.2f}")
        print(f"    Runner:      $0.00 (remaining_contracts = 0)")
        print(f"    Commission:  -${commission * 2 * contracts:.2f}")
        print(f"    Total PnL:   ${correct_pnl:.2f}")
        print(f"    Correct R:   {correct_r:.4f}")

        overcount = buggy_pnl - correct_pnl
        overcount_r = buggy_r - correct_r
        print(f"\n  OVERCOUNT: ${overcount:.2f} ({overcount_r:.4f}R)")
        print(f"  Buggy/Correct ratio: {buggy_r/correct_r:.2f}x" if correct_r != 0 else "  Correct R is 0!")


if __name__ == "__main__":
    main()
