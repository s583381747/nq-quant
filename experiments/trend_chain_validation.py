"""
experiments/trend_chain_validation.py -- Validate trend chain + combined strategy
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.trend_fvg_research import run_trend_fvg_backtest
from experiments.chain_engine import load_all, compute_metrics, run_chain_backtest


def bootstrap_pf(r_arr, n_boot=10000):
    pfs = []
    for _ in range(n_boot):
        s = np.random.choice(r_arr, size=len(r_arr), replace=True)
        w, lo = s[s > 0].sum(), abs(s[s < 0].sum())
        pfs.append(min(w / lo if lo > 0 else 5, 10))
    return np.percentile(pfs, [2.5, 50, 97.5])


def bootstrap_pf_diff(r1, r2, n_boot=10000):
    diffs = []
    for _ in range(n_boot):
        s1 = np.random.choice(r1, size=len(r1), replace=True)
        s2 = np.random.choice(r2, size=len(r2), replace=True)
        w1, l1 = s1[s1 > 0].sum(), abs(s1[s1 < 0].sum())
        w2, l2 = s2[s2 > 0].sum(), abs(s2[s2 < 0].sum())
        diffs.append(min(w1/l1 if l1 > 0 else 5, 10) - min(w2/l2 if l2 > 0 else 5, 10))
    return np.percentile(diffs, [2.5, 50, 97.5])


def main():
    d = load_all()
    nq_idx = d["nq"].index
    on_hi, on_lo = d["on_hi"], d["on_lo"]
    bias_dir_arr = d["bias_dir_arr"]

    trades = run_trend_fvg_backtest(d, include_breakdown_fvgs=False)

    for t in trades:
        idx = nq_idx.get_loc(t["entry_time"])
        direction = t["dir"]
        on_h, on_l = on_hi[idx], on_lo[idx]
        on_range = on_h - on_l if not (np.isnan(on_h) or np.isnan(on_l)) else 0
        on_pos = (t["entry_price"] - on_l) / on_range if on_range > 0 else 0.5
        in_discount = on_pos < 0.5
        bias_aligned = (direction == np.sign(bias_dir_arr[idx])) if bias_dir_arr[idx] != 0 else False

        t["trend_chain"] = False
        if direction == 1 and in_discount and bias_aligned:
            t["trend_chain"] = True
        elif direction == -1 and (not in_discount) and bias_aligned:
            t["trend_chain"] = True
        t["pd_only"] = (direction == 1 and in_discount) or (direction == -1 and not in_discount)

    df = pd.DataFrame(trades)
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year
    trend_chain = df[df["trend_chain"]]
    pd_only = df[df["pd_only"]]

    np.random.seed(42)

    print("=" * 120)
    print("TREND CHAIN VALIDATION")
    print("=" * 120)

    # ================================================================
    # Bootstrap
    # ================================================================
    print("\n" + "#" * 80)
    print("TEST 1: BOOTSTRAP CI")
    print("#" * 80)

    groups = [
        ("ALL trend", df["r"].values),
        ("PD only (disc L + prem S)", pd_only["r"].values),
        ("Trend chain (PD + bias)", trend_chain["r"].values),
        ("Excluded (not PD)", df[~df["pd_only"]]["r"].values),
    ]

    print(f"\n  {'Group':40s} | {'N':>5s} | {'PF':>6s} | {'2.5%':>6s} | {'50%':>6s} | {'97.5%':>6s} | {'PF>1':>5s}")
    print(f"  {'-'*40}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}")
    for name, r in groups:
        if len(r) < 20:
            continue
        w, lo = r[r > 0].sum(), abs(r[r < 0].sum())
        pf = w / lo if lo > 0 else 999
        ci = bootstrap_pf(r)
        sig = "YES" if ci[0] > 1 else "NO"
        print(f"  {name:40s} | {len(r):5d} | {pf:5.2f} | {ci[0]:5.2f} | {ci[1]:5.2f} | {ci[2]:5.2f} | {sig:>5s}")

    # Diff test
    r_tc = trend_chain["r"].values
    r_ex = df[~df["trend_chain"]]["r"].values
    if len(r_tc) >= 20 and len(r_ex) >= 20:
        ci_diff = bootstrap_pf_diff(r_tc, r_ex)
        print(f"\n  Trend chain vs excluded PF diff: [{ci_diff[0]:+.2f}, {ci_diff[1]:+.2f}, {ci_diff[2]:+.2f}]  "
              f"Sig: {'YES' if ci_diff[0] > 0 else 'NO'}")

    # ================================================================
    # Temporal split
    # ================================================================
    print("\n" + "#" * 80)
    print("TEST 2: TEMPORAL SPLIT")
    print("#" * 80)

    years = sorted(df["year"].unique())
    mid = len(years) // 2
    train_y = set(years[:mid])
    test_y = set(years[mid:])
    print(f"\n  Train: {min(train_y)}-{max(train_y)}, Test: {min(test_y)}-{max(test_y)}")

    for name, sub in [("ALL trend", df), ("PD only", pd_only), ("Trend chain (PD+bias)", trend_chain)]:
        tr = sub[sub["year"].isin(train_y)]
        te = sub[sub["year"].isin(test_y)]
        if len(tr) < 20 or len(te) < 20:
            print(f"  {name:35s} | too few")
            continue
        m_tr = compute_metrics(tr.to_dict("records"))
        m_te = compute_metrics(te.to_dict("records"))
        holds = "HOLDS" if m_te["PF"] > 1.2 else "WEAK" if m_te["PF"] > 1.0 else "FAILS"
        print(f"  {name:35s} | train PF={m_tr['PF']:5.2f} ({len(tr):4d}t) -> test PF={m_te['PF']:5.2f} ({len(te):4d}t)  {holds}")

    # ================================================================
    # Year-by-year
    # ================================================================
    print("\n" + "#" * 80)
    print("TEST 3: YEAR-BY-YEAR (Trend chain: PD + bias)")
    print("#" * 80)

    neg = 0
    for yr in sorted(trend_chain["year"].unique()):
        grp = trend_chain[trend_chain["year"] == yr]
        r = grp["r"].values
        total_r = r.sum()
        w, lo = r[r > 0].sum(), abs(r[r < 0].sum())
        pf = w / lo if lo > 0 else 999
        flag = " NEG" if total_r < 0 else ""
        if total_r < 0:
            neg += 1
        nl = len(grp[grp["dir"] == 1])
        ns = len(grp[grp["dir"] == -1])
        print(f"  {yr}: {len(grp):3d}t (L:{nl:3d} S:{ns:3d})  R={total_r:+7.1f}  PF={pf:5.2f}{flag}")
    print(f"  Negative years: {neg}/{len(trend_chain['year'].unique())}")

    # ================================================================
    # Combined: chain (1.0x) + trend chain (0.5x)
    # ================================================================
    print("\n" + "#" * 80)
    print("TEST 4: COMBINED STRATEGY")
    print("#" * 80)

    chain_trades, _ = run_chain_backtest(d)
    trend_chain_list = trend_chain.to_dict("records")

    all_t = [(t["entry_time"], t["r"], "chain") for t in chain_trades]
    all_t += [(t["entry_time"], t["r"] * 0.5, "trend") for t in trend_chain_list]
    all_t.sort(key=lambda x: x[0])

    r_comb = np.array([t[1] for t in all_t])
    total_r = r_comb.sum()
    wins, losses = r_comb[r_comb > 0].sum(), abs(r_comb[r_comb < 0].sum())
    pf = wins / losses if losses > 0 else 999
    cumr = np.cumsum(r_comb)
    peak = np.maximum.accumulate(cumr)
    dd = (peak - cumr).max()
    ppdd = total_r / dd if dd > 0 else 999
    n_chain = sum(1 for t in all_t if t[2] == "chain")
    n_trend = sum(1 for t in all_t if t[2] == "trend")
    tpd = len(all_t) / (252 * 10.5)

    print(f"\n  Chain:     {n_chain} trades (1.0x R)")
    print(f"  Trend:     {n_trend} trades (0.5x R)")
    print(f"  Total:     {len(all_t)} trades ({tpd:.2f}/day)")
    print(f"  R:         {total_r:+.1f}")
    print(f"  PF:        {pf:.2f}")
    print(f"  PPDD:      {ppdd:.2f}")
    print(f"  MaxDD:     {dd:.1f}R")

    # Year-by-year combined
    df_comb = pd.DataFrame({
        "time": [t[0] for t in all_t],
        "r": [t[1] for t in all_t],
        "type": [t[2] for t in all_t],
    })
    df_comb["year"] = pd.to_datetime(df_comb["time"]).dt.year

    print(f"\n  Year-by-year:")
    neg_c = 0
    for yr, grp in df_comb.groupby("year"):
        r = grp["r"].values
        total = r.sum()
        w, lo = r[r > 0].sum(), abs(r[r < 0].sum())
        pf_y = w / lo if lo > 0 else 999
        nc = (grp["type"] == "chain").sum()
        nt = (grp["type"] == "trend").sum()
        flag = " NEG" if total < 0 else ""
        if total < 0:
            neg_c += 1
        print(f"    {yr}: {len(grp):3d}t (C:{nc:3d} T:{nt:3d})  R={total:+7.1f}  PF={pf_y:5.2f}{flag}")
    print(f"    Negative years: {neg_c}")

    # Comparison
    m_ch = compute_metrics(chain_trades)
    print(f"\n  COMPARISON:")
    print(f"    Chain alone:  {m_ch['trades']:4d}t  R={m_ch['R']:+8.1f}  PF={m_ch['PF']:5.2f}  PPDD={m_ch['PPDD']:6.2f}  DD={m_ch['MaxDD']:5.1f}R  {m_ch['trades']/(252*10.5):.2f}/d")
    print(f"    Combined:     {len(all_t):4d}t  R={total_r:+8.1f}  PF={pf:5.2f}  PPDD={ppdd:6.2f}  DD={dd:5.1f}R  {tpd:.2f}/d")
    print(f"    U2 baseline:  2589t  R=  +791.5  PF= 1.59  PPDD= 28.35  DD= 27.9R  0.98/d")


if __name__ == "__main__":
    main()
