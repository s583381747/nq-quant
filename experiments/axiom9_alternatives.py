"""
experiments/axiom9_alternatives.py -- Test structural alternatives to Axiom 9 tick data

Variants:
  V1. Current worst-case (BE=true, worst_case_trim_be=true)  -- baseline lower bound
  V2. Current optimistic (BE=true, worst_case_trim_be=false) -- baseline upper bound
  V3. No BE (BE=false)                                        -- eliminates Axiom 9 entirely
  V4. No BE + no trim (pure runner trail)                    -- alternative #21 pure form
  V5. No BE + 50% trim                                        -- alternative 50% trim
  V6. No BE + TP 1.5R                                         -- different TP distance
  V7. No BE + TP 2R                                           -- further TP

If V3/V4/V5 >= V1 (worst-case baseline), the Axiom 9 ambiguity is
structurally eliminated and no tick data is needed.
"""
from __future__ import annotations
import sys, time as _time
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.chain_engine import load_all, compute_metrics
from experiments.unified_engine_1m import run_hybrid_1m


def main():
    print("=" * 100)
    print("AXIOM 9 STRUCTURAL ALTERNATIVES — No tick data needed?")
    print("=" * 100)

    t0 = _time.time()
    d5 = load_all()
    nq1 = pd.read_parquet(PROJECT / "data" / "NQ_1min_10yr.parquet")
    print(f"Data loaded in {_time.time()-t0:.1f}s  |  5m: {d5['n']:,}  1m: {len(nq1):,}")

    # All variants use trend_r_mult=0.5 (production setting)
    base = dict(trend_r_mult=0.5)

    variants = [
        ("V1 Worst-case (BE, Axiom 9 forced)",
            {**base, "use_be": True,  "worst_case_trim_be": True,  "trim_pct": 0.25, "fixed_tp_r": 1.0}),
        ("V2 Optimistic (BE, no Axiom 9)",
            {**base, "use_be": True,  "worst_case_trim_be": False, "trim_pct": 0.25, "fixed_tp_r": 1.0}),
        ("V3 No BE (trail only from trim)",
            {**base, "use_be": False, "worst_case_trim_be": False, "trim_pct": 0.25, "fixed_tp_r": 1.0}),
        ("V4 No BE + no trim (pure trail)",
            {**base, "use_be": False, "worst_case_trim_be": False, "trim_pct": 0.0,  "fixed_tp_r": 1.0}),
        ("V5 No BE + 50% trim",
            {**base, "use_be": False, "worst_case_trim_be": False, "trim_pct": 0.50, "fixed_tp_r": 1.0}),
        ("V6 No BE + TP 1.5R",
            {**base, "use_be": False, "worst_case_trim_be": False, "trim_pct": 0.25, "fixed_tp_r": 1.5}),
        ("V7 No BE + TP 2R",
            {**base, "use_be": False, "worst_case_trim_be": False, "trim_pct": 0.25, "fixed_tp_r": 2.0}),
    ]

    results = {}
    for name, kw in variants:
        t1 = _time.time()
        trades, _ = run_hybrid_1m(d5, nq1, **kw)
        m = compute_metrics(trades)
        results[name] = (trades, m)
        print(f"  {name:40s}  {_time.time()-t1:.1f}s  "
              f"{m['trades']}t  R={m['R']:+.1f}  PF={m['PF']:.2f}  "
              f"PPDD={m['PPDD']:.1f}  MaxDD={m['MaxDD']:.1f}R")

    # --- Summary table ---
    print(f"\n{'='*100}")
    print(f"{'Variant':42s} {'Trades':>7s} {'R':>10s} {'PF':>7s} {'PPDD':>7s} {'MaxDD':>7s} {'avgR':>8s}")
    print(f"{'-'*100}")
    for name, _ in variants:
        _, m = results[name]
        print(f"{name:42s} {m['trades']:7d} {m['R']:+10.1f} {m['PF']:7.2f} "
              f"{m['PPDD']:7.1f} {m['MaxDD']:7.1f} {m['avgR']:+8.3f}")

    # --- Walk-forward for top 3 candidates (V1, V3, V4 or V5) ---
    for name in [v[0] for v in variants[:5]]:
        trades, _ = results[name]
        df = pd.DataFrame(trades)
        if len(df) == 0:
            continue
        df["year"] = pd.to_datetime(df["entry_time"]).dt.year
        print(f"\n  {name} walk-forward:")
        neg = 0
        for yr, grp in df.groupby("year"):
            r = grp["r"].values
            w, lo = r[r > 0].sum(), abs(r[r < 0].sum())
            pf = w / lo if lo > 0 else 999
            flag = " NEG" if r.sum() < 0 else ""
            if r.sum() < 0: neg += 1
            print(f"    {yr}: {len(grp):4d}t  R={r.sum():+7.1f}  PF={pf:.2f}{flag}")
        print(f"    Negative years: {neg}")

    # --- Exit reason breakdown for V3 (key variant) ---
    print(f"\n  V3 exit reasons:")
    trades, _ = results[variants[2][0]]
    df = pd.DataFrame(trades)
    for reason, grp in df.groupby("reason"):
        print(f"    {reason:20s}: {len(grp):5d}t  R={grp['r'].sum():+8.1f}  avgR={grp['r'].mean():+.3f}")

    # --- Verdict ---
    v1_r = results[variants[0][0]][1]["R"]
    v1_pf = results[variants[0][0]][1]["PF"]
    v3_r = results[variants[2][0]][1]["R"]
    v3_pf = results[variants[2][0]][1]["PF"]
    v4_r = results[variants[3][0]][1]["R"]
    v4_pf = results[variants[3][0]][1]["PF"]

    print(f"\n{'='*100}")
    print("VERDICT")
    print(f"{'='*100}")
    print(f"  V1 Worst-case baseline: R={v1_r:+.1f}  PF={v1_pf:.2f}")
    print(f"  V3 No BE:               R={v3_r:+.1f}  PF={v3_pf:.2f}  "
          f"({'WIN' if v3_pf >= v1_pf else 'LOSS'}: {v3_r - v1_r:+.1f}R, {v3_pf - v1_pf:+.2f} PF)")
    print(f"  V4 No BE no trim:       R={v4_r:+.1f}  PF={v4_pf:.2f}  "
          f"({'WIN' if v4_pf >= v1_pf else 'LOSS'}: {v4_r - v1_r:+.1f}R, {v4_pf - v1_pf:+.2f} PF)")

    if v3_pf >= v1_pf or v4_pf >= v1_pf:
        print(f"\n  ✅ STRUCTURAL WIN: Axiom 9 eliminated, no tick data needed")
        print(f"     Best variant beats worst-case baseline.")
    else:
        print(f"\n  ❌ No structural win. Worst-case baseline (BE, Axiom 9 forced) is best.")
        print(f"     Next step: try Brownian bridge EV or buy 3-month tick data sample.")

    print(f"\nTotal: {_time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
