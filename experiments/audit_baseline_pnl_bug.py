"""
Check if Config H++ baseline ALSO has the TP2 PnL triple-counting bug.
If yes, both systems are inflated and the comparison is apples-to-apples.
If no, U2 is uniquely broken.
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
from experiments.pure_liquidity_tp import run_backtest_pure_liquidity
from experiments.a2c_stop_widening_engine import widen_stop_array

SEP = "=" * 100


def main():
    print(SEP)
    print("BASELINE CONFIG H++ — PNL BUG CHECK")
    print(SEP)

    d = load_all()
    d_extra = prepare_liquidity_data(d)
    params = d["params"]
    point_value = params["position"]["point_value"]

    # Tighten stops (same as U2 baseline)
    original_stop = d["model_stop_arr"].copy()
    tightened = widen_stop_array(
        original_stop, d["entry_price_arr"], d["sig_dir"],
        d["sig_mask"], 0.85)
    d["model_stop_arr"] = tightened

    trades, diag = run_backtest_pure_liquidity(
        d, d_extra,
        sq_long=0.68, sq_short=0.80, min_stop_atr=1.7,
        block_pm_shorts=True, tp_strategy="v1",
        tp1_trim_pct=0.50, be_after_tp1=False, be_after_tp2=False,
    )
    d["model_stop_arr"] = original_stop

    m_buggy = compute_metrics(trades)
    print(f"\nBuggy baseline: {m_buggy['trades']}t | R={m_buggy['R']:+.1f} | PPDD={m_buggy['PPDD']:.2f} | PF={m_buggy['PF']:.2f}")

    # Count TP2 exits
    tp2_trades = [t for t in trades if t.get("reason") == "tp2" and t.get("trim_stage", 0) == 2 and t.get("dir", 0) == 1]
    tp1_multi = [t for t in trades if t.get("reason") == "tp1" and t.get("trim_stage", 0) >= 1 and t.get("dir", 0) == 1]
    print(f"TP2 exits (multi-tp longs): {len(tp2_trades)}")
    print(f"TP1 full-exit (multi-tp longs): {len(tp1_multi)}")

    # Fix the bug
    corrected = []
    overcount = 0.0
    for t in trades:
        new_t = t.copy()
        if t.get("reason") == "tp2" and t.get("trim_stage", 0) == 2 and t.get("dir", 0) == 1:
            entry_p = t["entry_price"]
            tp2_p = t["tp2_price"]
            stop_dist = abs(entry_p - t["stop_price"])
            oc = (tp2_p - entry_p) / stop_dist if stop_dist > 0 else 0
            new_t["r"] = t["r"] - oc
            overcount += oc
        elif t.get("reason") == "tp1" and t.get("trim_stage", 0) >= 1 and t.get("dir", 0) == 1:
            entry_p = t["entry_price"]
            tp1_p = t["tp1_price"]
            stop_dist = abs(entry_p - t["stop_price"])
            oc = (tp1_p - entry_p) / stop_dist if stop_dist > 0 else 0
            new_t["r"] = t["r"] - oc
            overcount += oc
        corrected.append(new_t)

    m_corrected = compute_metrics(corrected)
    print(f"\nCorrected baseline: {m_corrected['trades']}t | R={m_corrected['R']:+.1f} | PPDD={m_corrected['PPDD']:.2f} | PF={m_corrected['PF']:.2f}")
    print(f"Overcount: {overcount:+.1f} ({100*overcount/m_buggy['R']:.1f}% of buggy R)")

    print(f"\nSUMMARY:")
    print(f"  Baseline buggy:    R={m_buggy['R']:+.1f}")
    print(f"  Baseline corrected: R={m_corrected['R']:+.1f}")
    print(f"  Bug impact:        {m_buggy['R'] - m_corrected['R']:.1f}R ({100*(m_buggy['R'] - m_corrected['R'])/m_buggy['R']:.1f}%)")

    # Walk-forward
    print(f"\nWalk-forward corrected baseline:")
    wf = walk_forward_metrics(corrected)
    if wf:
        for yr in wf:
            avg_r = yr['R'] / yr['n'] if yr['n'] > 0 else 0
            print(f"  {yr['year']}: {yr['n']}t | R={yr['R']:+.1f} | avgR={avg_r:+.4f} | WR={yr['WR']:.1f}%")
        neg = sum(1 for yr in wf if yr['R'] < 0)
        print(f"  Negative years: {neg}/{len(wf)}")


if __name__ == "__main__":
    main()
