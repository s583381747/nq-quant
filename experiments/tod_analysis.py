"""
Time-of-Day Analysis for Chain Engine
======================================
Question: Is there a time-of-day effect? ICT says NY AM (10:00-12:00) is the "kill zone."

Bins:
  10:00-10:30, 10:30-11:00, 11:00-12:00, 12:00-13:00 (lunch),
  13:00-14:00, 14:00-15:00, 15:00-16:00

Also: AM (10:00-12:00) vs PM (13:00-16:00)
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd
from experiments.chain_engine import load_all, run_chain_backtest, compute_metrics


def assign_bin(et_hour: float) -> str:
    """Assign a time-of-day bin based on ET fractional hour."""
    if 10.0 <= et_hour < 10.5:
        return "10:00-10:30"
    elif 10.5 <= et_hour < 11.0:
        return "10:30-11:00"
    elif 11.0 <= et_hour < 12.0:
        return "11:00-12:00"
    elif 12.0 <= et_hour < 13.0:
        return "12:00-13:00"
    elif 13.0 <= et_hour < 14.0:
        return "13:00-14:00"
    elif 14.0 <= et_hour < 15.0:
        return "14:00-15:00"
    elif 15.0 <= et_hour < 16.0:
        return "15:00-16:00"
    else:
        return "other"


def assign_session(et_hour: float) -> str:
    if 10.0 <= et_hour < 12.0:
        return "AM (10-12)"
    elif 13.0 <= et_hour < 16.0:
        return "PM (13-16)"
    elif 12.0 <= et_hour < 13.0:
        return "LUNCH (12-13)"
    else:
        return "other"


def fmt_metrics(label: str, trades: list[dict], max_label: int = 20) -> str:
    """Format metrics for a group of trades."""
    if not trades:
        return f"  {label:<{max_label}s} |     0t |       -- |    -- |    -- |    --"
    m = compute_metrics(trades)
    r_arr = np.array([t["r"] for t in trades])
    wr = (r_arr > 0).mean() * 100
    avg_win = r_arr[r_arr > 0].mean() if (r_arr > 0).any() else 0
    avg_loss = r_arr[r_arr < 0].mean() if (r_arr < 0).any() else 0
    return (
        f"  {label:<{max_label}s} | {m['trades']:5d}t | "
        f"R={m['R']:+8.1f} | PF={m['PF']:5.2f} | WR={m['WR']:5.1f}% | "
        f"DD={m['MaxDD']:5.1f}R | PPDD={m['PPDD']:6.2f} | "
        f"avgR={m['avgR']:+.4f} | avgW={avg_win:+.3f} | avgL={avg_loss:+.3f}"
    )


def main():
    print("=" * 130)
    print("TIME-OF-DAY ANALYSIS — Chain Engine (ON-only default config)")
    print("=" * 130)

    # 1. Load data + run backtest
    d = load_all()
    trades, stats = run_chain_backtest(d)  # default = ON-only
    print(f"\nTotal trades: {len(trades)}")
    m_all = compute_metrics(trades)
    print(f"Baseline: R={m_all['R']:+.1f} | PF={m_all['PF']:.2f} | WR={m_all['WR']:.1f}% | PPDD={m_all['PPDD']:.2f} | DD={m_all['MaxDD']:.1f}R")

    # 2. Convert entry_time to ET and extract fractional hour
    for t in trades:
        et = t["entry_time"].tz_convert("US/Eastern")
        t["et_hour"] = et.hour + et.minute / 60.0
        t["et_bin"] = assign_bin(t["et_hour"])
        t["et_session"] = assign_session(t["et_hour"])
        t["entry_et"] = et

    # 3. Granular bins
    bin_order = [
        "10:00-10:30", "10:30-11:00", "11:00-12:00",
        "12:00-13:00", "13:00-14:00", "14:00-15:00", "15:00-16:00"
    ]

    LBL = 16
    print(f"\n{'='*130}")
    print("GRANULAR TIME-OF-DAY BINS")
    print(f"{'='*130}")
    print(f"  {'Bin':<{LBL}s} | {'N':>5s}t | {'R':>8s} | {'PF':>5s} | {'WR':>5s}  | {'DD':>5s}  | {'PPDD':>6s}  | {'avgR':>8s} | {'avgW':>6s}  | {'avgL':>6s}")
    print(f"  {'-'*LBL}-+-{'-'*6}-+-{'-'*9}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}")

    for b in bin_order:
        group = [t for t in trades if t["et_bin"] == b]
        print(fmt_metrics(b, group, LBL))

    # Check "other"
    other = [t for t in trades if t["et_bin"] == "other"]
    if other:
        print(fmt_metrics("other", other, LBL))

    # 4. AM vs Lunch vs PM
    session_order = ["AM (10-12)", "LUNCH (12-13)", "PM (13-16)"]
    print(f"\n{'='*130}")
    print("AM vs LUNCH vs PM SESSION SPLIT")
    print(f"{'='*130}")
    print(f"  {'Session':<{LBL}s} | {'N':>5s}t | {'R':>8s} | {'PF':>5s} | {'WR':>5s}  | {'DD':>5s}  | {'PPDD':>6s}  | {'avgR':>8s} | {'avgW':>6s}  | {'avgL':>6s}")
    print(f"  {'-'*LBL}-+-{'-'*6}-+-{'-'*9}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}")

    for s in session_order:
        group = [t for t in trades if t["et_session"] == s]
        print(fmt_metrics(s, group, LBL))

    # 5. Long vs Short within AM/PM
    print(f"\n{'='*130}")
    print("DIRECTION x SESSION")
    print(f"{'='*130}")
    LBL2 = 22
    print(f"  {'Group':<{LBL2}s} | {'N':>5s}t | {'R':>8s} | {'PF':>5s} | {'WR':>5s}  | {'DD':>5s}  | {'PPDD':>6s}  | {'avgR':>8s} | {'avgW':>6s}  | {'avgL':>6s}")
    print(f"  {'-'*LBL2}-+-{'-'*6}-+-{'-'*9}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}")

    for s in session_order:
        for d_name, d_val in [("Long", 1), ("Short", -1)]:
            group = [t for t in trades if t["et_session"] == s and t["dir"] == d_val]
            label = f"{s} {d_name}"
            print(fmt_metrics(label, group, LBL2))

    # 6. Year-over-year consistency for AM vs PM
    print(f"\n{'='*130}")
    print("YEARLY CONSISTENCY: AM (10-12) vs PM (13-16)")
    print(f"{'='*130}")

    am_trades = [t for t in trades if t["et_session"] == "AM (10-12)"]
    pm_trades = [t for t in trades if t["et_session"] == "PM (13-16)"]

    years = sorted(set(t["entry_et"].year for t in trades))
    print(f"\n  {'Year':<6s} | {'AM trades':>9s} {'AM R':>8s} {'AM PF':>7s} {'AM WR':>7s} | {'PM trades':>9s} {'PM R':>8s} {'PM PF':>7s} {'PM WR':>7s}")
    print(f"  {'-'*6}-+-{'-'*33}-+-{'-'*33}")

    for y in years:
        am_y = [t for t in am_trades if t["entry_et"].year == y]
        pm_y = [t for t in pm_trades if t["entry_et"].year == y]
        am_m = compute_metrics(am_y) if am_y else {"trades": 0, "R": 0.0, "PF": 0.0, "WR": 0.0}
        pm_m = compute_metrics(pm_y) if pm_y else {"trades": 0, "R": 0.0, "PF": 0.0, "WR": 0.0}
        print(f"  {y:<6d} | {am_m['trades']:9d} {am_m['R']:+8.1f} {am_m['PF']:7.2f} {am_m['WR']:6.1f}% | {pm_m['trades']:9d} {pm_m['R']:+8.1f} {pm_m['PF']:7.2f} {pm_m['WR']:6.1f}%")

    # 7. Summary conclusion
    print(f"\n{'='*130}")
    print("SUMMARY")
    print(f"{'='*130}")
    am_m = compute_metrics(am_trades)
    pm_m = compute_metrics(pm_trades)
    lunch_m = compute_metrics([t for t in trades if t["et_session"] == "LUNCH (12-13)"])

    print(f"  AM  (10:00-12:00): {am_m['trades']}t, R={am_m['R']:+.1f}, PF={am_m['PF']:.2f}, WR={am_m['WR']:.1f}%, PPDD={am_m['PPDD']:.2f}")
    print(f"  LUNCH (12-13):     {lunch_m['trades']}t, R={lunch_m['R']:+.1f}, PF={lunch_m['PF']:.2f}, WR={lunch_m['WR']:.1f}%, PPDD={lunch_m['PPDD']:.2f}")
    print(f"  PM  (13:00-16:00): {pm_m['trades']}t, R={pm_m['R']:+.1f}, PF={pm_m['PF']:.2f}, WR={pm_m['WR']:.1f}%, PPDD={pm_m['PPDD']:.2f}")

    if am_m['PF'] > pm_m['PF'] and am_m['PPDD'] > pm_m['PPDD']:
        print(f"\n  >> AM KILL ZONE CONFIRMED: AM has higher PF ({am_m['PF']:.2f} vs {pm_m['PF']:.2f}) and PPDD ({am_m['PPDD']:.2f} vs {pm_m['PPDD']:.2f})")
    elif pm_m['PF'] > am_m['PF'] and pm_m['PPDD'] > am_m['PPDD']:
        print(f"\n  >> PM is BETTER: PM has higher PF ({pm_m['PF']:.2f} vs {am_m['PF']:.2f}) and PPDD ({pm_m['PPDD']:.2f} vs {am_m['PPDD']:.2f})")
    else:
        print(f"\n  >> MIXED: AM PF={am_m['PF']:.2f}, PPDD={am_m['PPDD']:.2f} vs PM PF={pm_m['PF']:.2f}, PPDD={pm_m['PPDD']:.2f}")

    # R per trade comparison
    am_rpt = am_m['R'] / am_m['trades'] if am_m['trades'] > 0 else 0
    pm_rpt = pm_m['R'] / pm_m['trades'] if pm_m['trades'] > 0 else 0
    print(f"  AM R/trade: {am_rpt:+.4f} | PM R/trade: {pm_rpt:+.4f}")

    print()


if __name__ == "__main__":
    main()
