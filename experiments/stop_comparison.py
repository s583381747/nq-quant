"""
experiments/stop_comparison.py -- Zone-based vs Candle-based stop
Runs 6 combos: {chain, trend, all} × {zone, candle}, worst-case Axiom 9.
Single process, one data load, iterate variants.
"""
from __future__ import annotations
import sys, time as _time
from pathlib import Path
import numpy as np, pandas as pd, yaml

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.chain_engine import load_all, compute_metrics, pr
from experiments.unified_engine_1m import run_hybrid_1m


def main():
    print("=" * 100)
    print("STOP COMPARISON: Zone vs Candle x Chain vs Trend  (worst-case Axiom 9)")
    print("=" * 100)

    t0 = _time.time()
    d5 = load_all()
    nq1 = pd.read_parquet(PROJECT / "data" / "NQ_1min_10yr.parquet")
    print(f"Data loaded in {_time.time() - t0:.1f}s  |  5m: {d5['n']:,}  1m: {len(nq1):,}")

    common = dict(trend_r_mult=0.5, worst_case_trim_be=True)

    variants = [
        ("Chain+Zone",   {**common, "stop_mode": "zone",   "tier_filter": 1}),
        ("Chain+Candle", {**common, "stop_mode": "candle", "tier_filter": 1}),
        ("Trend+Zone",   {**common, "stop_mode": "zone",   "tier_filter": 2}),
        ("Trend+Candle", {**common, "stop_mode": "candle", "tier_filter": 2}),
        ("ALL+Zone",     {**common, "stop_mode": "zone",   "tier_filter": 0}),
        ("ALL+Candle",   {**common, "stop_mode": "candle", "tier_filter": 0}),
    ]

    results = {}
    for label, kw in variants:
        t1 = _time.time()
        trades, stats = run_hybrid_1m(d5, nq1, **kw)
        m = compute_metrics(trades)
        results[label] = (trades, m)
        print(f"  {label:20s}  {_time.time()-t1:.1f}s  {m['trades']}t  R={m['R']:+.1f}  PF={m['PF']:.2f}")

    # ---- Summary table ----
    print(f"\n{'='*100}")
    print(f"{'Config':20s} {'Trades':>7s} {'R':>10s} {'PF':>7s} {'PPDD':>7s} {'MaxDD':>7s} {'AvgStop':>8s} {'MedStop':>8s}")
    print(f"{'-'*100}")
    for label, _ in variants:
        trades, m = results[label]
        df = pd.DataFrame(trades)
        avg_sd = df["stop_dist_pts"].mean() if len(df) > 0 else 0
        med_sd = df["stop_dist_pts"].median() if len(df) > 0 else 0
        print(f"{label:20s} {m['trades']:7d} {m['R']:+10.1f} {m['PF']:7.2f} {m['PPDD']:7.1f} {m.get('MaxDD',0):7.1f} {avg_sd:7.1f}pt {med_sd:7.1f}pt")

    # ---- Walk-forward per combo ----
    for label in ["Chain+Zone", "Chain+Candle", "Trend+Zone", "Trend+Candle"]:
        trades, m = results[label]
        df = pd.DataFrame(trades)
        if len(df) == 0:
            continue
        df["year"] = pd.to_datetime(df["entry_time"]).dt.year
        print(f"\n  {label} walk-forward:")
        neg = 0
        for yr, grp in df.groupby("year"):
            r = grp["r"].values
            w, lo = r[r > 0].sum(), abs(r[r < 0].sum())
            pf = w / lo if lo > 0 else 999
            flag = " NEG" if r.sum() < 0 else ""
            if r.sum() < 0: neg += 1
            print(f"    {yr}: {len(grp):3d}t  R={r.sum():+7.1f}  PF={pf:.2f}{flag}")
        print(f"    Negative years: {neg}")

    # ---- Exit reasons ----
    for label in ["Chain+Zone", "Chain+Candle", "Trend+Zone", "Trend+Candle"]:
        trades, _ = results[label]
        df = pd.DataFrame(trades)
        if len(df) == 0:
            continue
        print(f"\n  {label} exits:")
        for reason, grp in df.groupby("reason"):
            print(f"    {reason:20s}: {len(grp):4d}t  R={grp['r'].sum():+.1f}")

    print(f"\nTotal: {_time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
