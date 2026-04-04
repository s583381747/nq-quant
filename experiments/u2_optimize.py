"""
experiments/u2_optimize.py — U2 optimization: 0-for-X sweep + daily range filter
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.u2_clean import load_all, run_u2_backtest, compute_metrics, walk_forward


def main():
    d = load_all()

    # ================================================================
    # STEP 1: 0-for-X sweep
    # ================================================================
    print("=" * 100)
    print("STEP 1: 0-FOR-X SWEEP (base: A2 sz>0.3 age<500 min>5pt, long-only)")
    print("=" * 100)

    results_0forx = {}
    for max_cl in [2, 3, 4, 5, 10, 999]:
        d_mod = dict(d)
        params_mod = dict(d["params"])
        risk_mod = dict(params_mod["risk"])
        risk_mod["max_consecutive_losses"] = max_cl
        params_mod["risk"] = risk_mod
        d_mod["params"] = params_mod

        trades, _ = run_u2_backtest(d_mod, stop_strategy="A2", fvg_size_mult=0.3,
                                     max_fvg_age=500, min_stop_pts=5.0)
        longs = [t for t in trades if t["dir"] == 1]
        m = compute_metrics(longs)
        label = f"0-for-{max_cl}" if max_cl < 999 else "NO LIMIT"
        tpd = m["trades"] / (252 * 10.5)
        pf_ok = " *" if m["PF"] >= 1.8 else ""
        print(f"  {label:12s}: {m['trades']:5d}t {tpd:.2f}/d PF={m['PF']:.2f}{pf_ok} R={m['R']:+8.1f} PPDD={m['PPDD']:6.2f} MaxDD={m['MaxDD']:5.1f}R WR={m['WR']:.1f}%")
        results_0forx[max_cl] = (m, longs)

    # ================================================================
    # STEP 2: Daily range filter (prev-day, no lookahead)
    # ================================================================
    print("\n" + "=" * 100)
    print("STEP 2: PREV-DAY RANGE FILTER (applied on top of each 0-for-X)")
    print("=" * 100)

    # Build prev-day range
    nq_h, nq_l = d["h"], d["l"]
    et_frac, dates = d["et_frac_arr"], d["dates"]
    daily_range = {}
    for i in range(d["n"]):
        date = dates[i]
        ef = et_frac[i]
        if 10.0 <= ef < 16.0:
            if date not in daily_range:
                daily_range[date] = {"h": nq_h[i], "l": nq_l[i]}
            else:
                daily_range[date]["h"] = max(daily_range[date]["h"], nq_h[i])
                daily_range[date]["l"] = min(daily_range[date]["l"], nq_l[i])

    sorted_dates = sorted(daily_range.keys())
    prev_day_range = {}
    for idx, date in enumerate(sorted_dates):
        if idx > 0:
            prev = sorted_dates[idx - 1]
            prev_day_range[date] = daily_range[prev]["h"] - daily_range[prev]["l"]

    # Sweep combinations
    print(f"\n  {'Config':40s} | {'Trades':>6} | {'t/day':>5} | {'PF':>5} | {'R':>8} | {'PPDD':>6} | {'MaxDD':>5} | {'WR':>5}")
    print("  " + "-" * 95)

    all_combos = []
    for max_cl in [2, 3, 4, 999]:
        _, longs = results_0forx[max_cl]
        cl_label = f"0f{max_cl}" if max_cl < 999 else "noLim"

        for min_range in [0, 50, 75, 100, 125]:
            if min_range == 0:
                filtered = longs
            else:
                filtered = []
                for t in longs:
                    date = pd.to_datetime(t["entry_time"]).date()
                    if date in prev_day_range and prev_day_range[date] >= min_range:
                        filtered.append(t)
                    elif date not in prev_day_range:
                        filtered.append(t)

            m = compute_metrics(filtered)
            if m["trades"] < 30:
                continue
            tpd = m["trades"] / (252 * 10.5)
            label = f"{cl_label} prevRange>={min_range}pt"
            pf_ok = " *" if m["PF"] >= 1.8 else ""
            freq_ok = " +" if tpd >= 0.5 else ""
            print(f"  {label:40s} | {m['trades']:6d} | {tpd:5.2f}{freq_ok} | {m['PF']:5.2f}{pf_ok} | {m['R']:+8.1f} | {m['PPDD']:6.2f} | {m['MaxDD']:5.1f}R | {m['WR']:5.1f}%")
            all_combos.append((label, m, filtered))

    # ================================================================
    # STEP 3: Walk-forward for PF>=1.8 + freq>=0.5
    # ================================================================
    print("\n" + "=" * 100)
    print("STEP 3: WALK-FORWARD FOR BEST CONFIGS (PF>=1.8 AND freq>=0.5/day)")
    print("=" * 100)

    targets = [(l, m, t) for l, m, t in all_combos
               if m["PF"] >= 1.8 and m["trades"] / (252 * 10.5) >= 0.5]
    targets.sort(key=lambda x: x[1]["PPDD"], reverse=True)

    if not targets:
        print("  No config meets both targets. Showing best PF configs:")
        targets = sorted(all_combos, key=lambda x: x[1]["PF"], reverse=True)[:3]

    for label, m, trades in targets[:5]:
        tpd = m["trades"] / (252 * 10.5)
        print(f"\n  {label}: {m['trades']}t {tpd:.2f}/d PF={m['PF']:.2f} R={m['R']:+.1f} PPDD={m['PPDD']:.2f} MaxDD={m['MaxDD']:.1f}R")
        wf = walk_forward(trades)
        neg = 0
        for yr in wf:
            flag = " NEG" if yr["R"] < 0 else ""
            if yr["R"] < 0:
                neg += 1
            print(f"    {yr['year']}: {yr['n']:3d}t  R={yr['R']:+7.1f}  WR={yr['WR']:5.1f}%  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}{flag}")
        print(f"    Negative years: {neg}/{len(wf)}")

        # Narrow-day PF check
        df_t = pd.DataFrame(trades)
        trade_ranges = []
        for t in trades:
            date = pd.to_datetime(t["entry_time"]).date()
            if date in daily_range:
                dr = daily_range[date]
                trade_ranges.append(dr["h"] - dr["l"])
            else:
                trade_ranges.append(np.nan)
        df_t["daily_range"] = trade_ranges
        narrow = df_t[df_t["daily_range"] < 100]
        if len(narrow) > 0:
            nw = narrow["r"].values
            nw_wins = nw[nw > 0].sum()
            nw_losses = abs(nw[nw < 0].sum())
            nw_pf = nw_wins / nw_losses if nw_losses > 0 else 999
            print(f"    Narrow-day (<100pt) PF: {nw_pf:.2f} ({len(narrow)}t)")

    # ================================================================
    # FINAL COMPARISON
    # ================================================================
    print("\n" + "=" * 100)
    print("FINAL COMPARISON TABLE")
    print("=" * 100)

    from experiments.pure_scalp_validation import load_all as load_f3, run_backtest as run_f3, compute_metrics as cm_f3
    d_f3 = load_f3()
    f3 = cm_f3(run_f3(d_f3))

    print(f"\n  {'Config':45s} | {'t':>5} | {'t/d':>4} | {'PF':>5} | {'R':>8} | {'PPDD':>6} | {'DD':>5} | {'WR':>5}")
    print("  " + "-" * 95)

    f3_tpd = f3["trades"] / (252 * 10.5)
    print(f"  {'F3 baseline':45s} | {f3['trades']:5d} | {f3_tpd:.2f} | {f3['PF']:5.2f} | {f3['R']:+8.1f} | {f3['PPDD']:6.2f} | {f3['MaxDD']:5.1f} | {f3['WR']:5.1f}%")

    base_m = results_0forx[2][0]
    base_tpd = base_m["trades"] / (252 * 10.5)
    print(f"  {'U2L base (0-for-2)':45s} | {base_m['trades']:5d} | {base_tpd:.2f} | {base_m['PF']:5.2f} | {base_m['R']:+8.1f} | {base_m['PPDD']:6.2f} | {base_m['MaxDD']:5.1f} | {base_m['WR']:5.1f}%")

    for label, m, _ in targets[:5]:
        tpd = m["trades"] / (252 * 10.5)
        print(f"  {label:45s} | {m['trades']:5d} | {tpd:.2f} | {m['PF']:5.2f} | {m['R']:+8.1f} | {m['PPDD']:6.2f} | {m['MaxDD']:5.1f} | {m['WR']:5.1f}%")

    print("\n  TARGET: PF >= 1.80 AND freq >= 0.50/day")
    n_target = len(targets)
    print(f"  Configs meeting both: {n_target}")


if __name__ == "__main__":
    main()
