"""U2 MaxDD reduction without sacrificing PF/PPDD.

Approach: instead of risk-scaling during DD (which cuts R),
find STRUCTURAL changes that improve the loss distribution itself.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from experiments.u2_clean import load_all, run_u2_backtest, compute_metrics, walk_forward


def metrics_from_trades(trades):
    if not trades:
        return {"trades": 0, "R": 0, "PF": 0, "PPDD": 0, "MaxDD": 0, "WR": 0, "avgR": 0}
    r = np.array([t["r"] for t in trades])
    total = r.sum()
    wr = (r > 0).mean() * 100
    w = r[r > 0].sum()
    lo = abs(r[r < 0].sum())
    pf = w / lo if lo > 0 else 999
    cr = np.cumsum(r)
    pk = np.maximum.accumulate(cr)
    dd = pk - cr
    mdd = dd.max() if len(dd) > 0 else 0
    ppdd = total / mdd if mdd > 0 else 999
    return {"trades": len(r), "R": round(total, 1), "PF": round(pf, 2),
            "PPDD": round(ppdd, 2), "MaxDD": round(mdd, 1), "WR": round(wr, 1),
            "avgR": round(total / len(r), 4)}


def pr(label, m):
    tpd = m["trades"] / (252 * 10.5) if m["trades"] > 0 else 0
    print(f"  {label:55s} | {m['trades']:5d}t {tpd:.2f}/d | PF={m['PF']:5.2f} | R={m['R']:+8.1f} | PPDD={m['PPDD']:6.2f} | DD={m['MaxDD']:5.1f}R | WR={m['WR']:5.1f}%")


def main():
    d = load_all()

    # Baseline
    bl_trades, _ = run_u2_backtest(d, stop_strategy="A2", fvg_size_mult=0.3,
                                    max_fvg_age=500, min_stop_pts=5.0)
    bl_longs = [t for t in bl_trades if t["dir"] == 1]
    bl = metrics_from_trades(bl_longs)
    print("=" * 120)
    print("STRUCTURAL DD REDUCTION (preserve PF/PPDD)")
    print("=" * 120)
    pr("BASELINE: A2 sz>0.3 age<500 min>5pt", bl)

    # ================================================================
    # AXIS 1: Min stop points (higher = less leveraged)
    # ================================================================
    print("\n--- AXIS 1: MIN STOP POINTS ---")
    print("  (Higher min stop -> fewer leveraged trades, lower R variance)")
    for ms in [5, 6, 7, 8, 10, 12, 15, 20]:
        t, _ = run_u2_backtest(d, stop_strategy="A2", fvg_size_mult=0.3,
                                max_fvg_age=500, min_stop_pts=float(ms))
        lo = [x for x in t if x["dir"] == 1]
        m = metrics_from_trades(lo)
        pr(f"min_stop={ms}pt", m)

    # ================================================================
    # AXIS 2: FVG size filter (larger FVGs = more reliable)
    # ================================================================
    print("\n--- AXIS 2: FVG SIZE FILTER ---")
    for sz in [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        t, _ = run_u2_backtest(d, stop_strategy="A2", fvg_size_mult=sz,
                                max_fvg_age=500, min_stop_pts=5.0)
        lo = [x for x in t if x["dir"] == 1]
        m = metrics_from_trades(lo)
        pr(f"fvg_size>{sz}xATR", m)

    # ================================================================
    # AXIS 3: Stop buffer (A2 buffer pct)
    # ================================================================
    print("\n--- AXIS 3: STOP BUFFER (A2 buffer %) ---")
    for buf in [0.0, 0.10, 0.15, 0.20, 0.30, 0.50]:
        t, _ = run_u2_backtest(d, stop_strategy="A2", stop_buffer_pct=buf,
                                fvg_size_mult=0.3, max_fvg_age=500, min_stop_pts=5.0)
        lo = [x for x in t if x["dir"] == 1]
        m = metrics_from_trades(lo)
        pr(f"buffer={int(buf*100)}%", m)

    # ================================================================
    # AXIS 4: Tighten factor (stop tightening)
    # ================================================================
    print("\n--- AXIS 4: STOP TIGHTEN FACTOR ---")
    for tf in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
        t, _ = run_u2_backtest(d, stop_strategy="A2", fvg_size_mult=0.3,
                                max_fvg_age=500, min_stop_pts=5.0, tighten_factor=tf)
        lo = [x for x in t if x["dir"] == 1]
        m = metrics_from_trades(lo)
        pr(f"tighten={tf:.2f}", m)

    # ================================================================
    # AXIS 5: Max FVG age
    # ================================================================
    print("\n--- AXIS 5: MAX FVG AGE ---")
    for age in [50, 100, 150, 200, 300, 500, 1000]:
        t, _ = run_u2_backtest(d, stop_strategy="A2", fvg_size_mult=0.3,
                                max_fvg_age=age, min_stop_pts=5.0)
        lo = [x for x in t if x["dir"] == 1]
        m = metrics_from_trades(lo)
        pr(f"max_age={age}", m)

    # ================================================================
    # AXIS 6: TP multiplier (affects trim/runner behavior)
    # ================================================================
    print("\n--- AXIS 6: TP MULTIPLIER ---")
    for tp in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        t, _ = run_u2_backtest(d, stop_strategy="A2", fvg_size_mult=0.3,
                                max_fvg_age=500, min_stop_pts=5.0, tp_mult=tp)
        lo = [x for x in t if x["dir"] == 1]
        m = metrics_from_trades(lo)
        pr(f"tp_mult={tp:.1f}", m)

    # ================================================================
    # AXIS 7: Trim pct
    # ================================================================
    print("\n--- AXIS 7: TRIM PCT ---")
    for tp in [0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 0.75, 1.0]:
        t, _ = run_u2_backtest(d, stop_strategy="A2", fvg_size_mult=0.3,
                                max_fvg_age=500, min_stop_pts=5.0, trim_pct=tp)
        lo = [x for x in t if x["dir"] == 1]
        m = metrics_from_trades(lo)
        pr(f"trim={int(tp*100)}%", m)

    # ================================================================
    # COMBOS: Find best MaxDD with PF>=1.8 and PPDD>=34
    # ================================================================
    print("\n" + "=" * 120)
    print("COMBO SWEEP: Find lowest MaxDD with PF>=1.80 AND PPDD>=34")
    print("=" * 120)

    combos = []
    for ms in [5, 7, 8, 10]:
        for sz in [0.3, 0.5]:
            for age in [100, 200, 500]:
                for buf in [0.15, 0.30]:
                    for tf in [0.80, 0.85, 0.90]:
                        t, _ = run_u2_backtest(d, stop_strategy="A2",
                            fvg_size_mult=sz, max_fvg_age=age,
                            min_stop_pts=float(ms), stop_buffer_pct=buf,
                            tighten_factor=tf)
                        lo = [x for x in t if x["dir"] == 1]
                        m = metrics_from_trades(lo)
                        tpd = m["trades"] / (252 * 10.5)
                        if m["PF"] >= 1.80 and tpd >= 0.5 and m["trades"] >= 100:
                            label = f"ms{ms} sz{sz} age{age} buf{int(buf*100)} tf{tf}"
                            combos.append((label, m, lo))

    # Sort by MaxDD ascending (lowest DD first)
    combos.sort(key=lambda x: x[1]["MaxDD"])
    print(f"\nTop 20 lowest MaxDD configs (PF>=1.8 + freq>=0.5/day):")
    for label, m, _ in combos[:20]:
        tpd = m["trades"] / (252 * 10.5)
        print(f"  {label:40s} | {m['trades']:5d}t {tpd:.2f}/d | PF={m['PF']:5.2f} | R={m['R']:+8.1f} | PPDD={m['PPDD']:6.2f} | DD={m['MaxDD']:5.1f}R")

    # Also sort by PPDD descending
    combos_ppdd = sorted(combos, key=lambda x: x[1]["PPDD"], reverse=True)
    print(f"\nTop 10 highest PPDD configs (PF>=1.8 + freq>=0.5/day):")
    for label, m, _ in combos_ppdd[:10]:
        tpd = m["trades"] / (252 * 10.5)
        print(f"  {label:40s} | {m['trades']:5d}t {tpd:.2f}/d | PF={m['PF']:5.2f} | R={m['R']:+8.1f} | PPDD={m['PPDD']:6.2f} | DD={m['MaxDD']:5.1f}R")

    # Walk-forward for lowest-DD config
    if combos:
        print(f"\n--- WALK-FORWARD: Lowest MaxDD config ---")
        label, m, trades = combos[0]
        tpd = m["trades"] / (252 * 10.5)
        print(f"  {label}: {m['trades']}t {tpd:.2f}/d PF={m['PF']:.2f} R={m['R']:+.1f} PPDD={m['PPDD']:.2f} MaxDD={m['MaxDD']:.1f}R")
        wf = walk_forward(trades)
        neg = 0
        for yr in wf:
            flag = " NEG" if yr["R"] < 0 else ""
            if yr["R"] < 0: neg += 1
            print(f"    {yr['year']}: {yr['n']:3d}t  R={yr['R']:+7.1f}  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}{flag}")
        print(f"    Negative years: {neg}/{len(wf)}")

        # Also WF for highest PPDD
        print(f"\n--- WALK-FORWARD: Highest PPDD config ---")
        label2, m2, trades2 = combos_ppdd[0]
        tpd2 = m2["trades"] / (252 * 10.5)
        print(f"  {label2}: {m2['trades']}t {tpd2:.2f}/d PF={m2['PF']:.2f} R={m2['R']:+.1f} PPDD={m2['PPDD']:.2f} MaxDD={m2['MaxDD']:.1f}R")
        wf2 = walk_forward(trades2)
        neg2 = 0
        for yr in wf2:
            flag = " NEG" if yr["R"] < 0 else ""
            if yr["R"] < 0: neg2 += 1
            print(f"    {yr['year']}: {yr['n']:3d}t  R={yr['R']:+7.1f}  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}{flag}")
        print(f"    Negative years: {neg2}/{len(wf2)}")


if __name__ == "__main__":
    main()
