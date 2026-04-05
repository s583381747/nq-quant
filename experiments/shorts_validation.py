"""
experiments/shorts_validation.py -- Statistical validation of symmetric shorts
=============================================================================

Validate that the new shorts-with-runners finding is statistically robust
before building production engine.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.shorts_runner_test import run_with_symmetric_management
from experiments.breakdown_chain_research import detect_breakdowns
from experiments.sweep_research import compute_pdhl
from experiments.u2_clean import load_all, compute_metrics, pr


def bootstrap_pf(r_arr, n_boot=10000):
    pfs = []
    n = len(r_arr)
    for _ in range(n_boot):
        sample = np.random.choice(r_arr, size=n, replace=True)
        wins = sample[sample > 0].sum()
        losses = abs(sample[sample < 0].sum())
        pf = min(wins / losses if losses > 0 else 5, 10)
        pfs.append(pf)
    return np.percentile(pfs, [2.5, 50, 97.5])


def bootstrap_pf_diff(r1, r2, n_boot=10000):
    diffs = []
    for _ in range(n_boot):
        s1 = np.random.choice(r1, size=len(r1), replace=True)
        s2 = np.random.choice(r2, size=len(r2), replace=True)
        w1, l1 = s1[s1 > 0].sum(), abs(s1[s1 < 0].sum())
        w2, l2 = s2[s2 > 0].sum(), abs(s2[s2 < 0].sum())
        pf1 = min(w1 / l1 if l1 > 0 else 5, 10)
        pf2 = min(w2 / l2 if l2 > 0 else 5, 10)
        diffs.append(pf1 - pf2)
    return np.percentile(diffs, [2.5, 50, 97.5])


def main():
    print("=" * 120)
    print("STATISTICAL VALIDATION: Symmetric Shorts + Chain Strategy")
    print("=" * 120)

    d = load_all()
    h, l, c, o = d["h"], d["l"], d["c"], d["o"]
    session_cache = pd.read_parquet(PROJECT / "data" / "cache_session_levels_10yr_v2.parquet")
    pdhl = compute_pdhl(d["nq"])

    # Build breakdowns
    pdl_bds = detect_breakdowns(h, l, c, pdhl["pdl"].values, "low", min_depth_pts=1.0)
    for bd in pdl_bds: bd["level_type"] = "low"

    pdh_bds = detect_breakdowns(h, l, c, pdhl["pdh"].values, "high", min_depth_pts=1.0)
    for bd in pdh_bds: bd["level_type"] = "high"

    on_lo = detect_breakdowns(h, l, c, session_cache["overnight_low"].values, "low", min_depth_pts=1.0)
    for bd in on_lo: bd["level_type"] = "low"

    on_hi = detect_breakdowns(h, l, c, session_cache["overnight_high"].values, "high", min_depth_pts=1.0)
    for bd in on_hi: bd["level_type"] = "high"

    # Best config: ON_low + ON_high
    best_bds = on_lo + on_hi
    trades = run_with_symmetric_management(d, best_bds)

    longs = [t for t in trades if t["dir"] == 1]
    shorts = [t for t in trades if t["dir"] == -1]
    df = pd.DataFrame(trades)
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year

    np.random.seed(42)

    # ================================================================
    # TEST 1: Bootstrap CI
    # ================================================================
    print("\n" + "#" * 80)
    print("TEST 1: BOOTSTRAP CI (ON_low + ON_high)")
    print("#" * 80)

    groups = [
        ("ALL", np.array([t["r"] for t in trades])),
        ("LONGS", np.array([t["r"] for t in longs])),
        ("SHORTS", np.array([t["r"] for t in shorts])),
    ]

    # Also test PDL+PDH
    pdl_pdh_bds = pdl_bds + pdh_bds
    trades_pp = run_with_symmetric_management(d, pdl_pdh_bds)
    longs_pp = [t for t in trades_pp if t["dir"] == 1]
    shorts_pp = [t for t in trades_pp if t["dir"] == -1]
    groups.append(("PDL+PDH ALL", np.array([t["r"] for t in trades_pp])))
    groups.append(("PDL+PDH LONGS", np.array([t["r"] for t in longs_pp])))
    groups.append(("PDL+PDH SHORTS", np.array([t["r"] for t in shorts_pp])))

    print(f"\n  {'Group':30s} | {'N':>5s} | {'PF':>6s} | {'2.5%':>6s} | {'50%':>6s} | {'97.5%':>6s} | {'PF>1?':>6s}")
    print(f"  {'-'*30}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    for name, r_arr in groups:
        if len(r_arr) < 20:
            print(f"  {name:30s} | {len(r_arr):5d} | too few")
            continue
        w = r_arr[r_arr > 0].sum()
        lo = abs(r_arr[r_arr < 0].sum())
        pf = w / lo if lo > 0 else 999
        ci = bootstrap_pf(r_arr)
        sig = "YES" if ci[0] > 1.0 else "NO"
        print(f"  {name:30s} | {len(r_arr):5d} | {pf:5.2f} | {ci[0]:5.2f} | {ci[1]:5.2f} | {ci[2]:5.2f} | {sig:>6s}")

    # ================================================================
    # TEST 2: Shorts PF difference vs breakeven
    # ================================================================
    print("\n" + "#" * 80)
    print("TEST 2: SHORTS vs LONGS PF DIFFERENCE")
    print("#" * 80)

    r_l = np.array([t["r"] for t in longs])
    r_s = np.array([t["r"] for t in shorts])
    ci = bootstrap_pf_diff(r_l, r_s)
    print(f"\n  ON: Longs PF - Shorts PF:")
    print(f"  95% CI: [{ci[0]:+.2f}, {ci[1]:+.2f}, {ci[2]:+.2f}]")
    print(f"  Longs significantly better: {'YES' if ci[0] > 0 else 'NO'}")

    # Are shorts significantly > 1.0?
    ci_s = bootstrap_pf(r_s)
    print(f"\n  Shorts PF 95% CI: [{ci_s[0]:.2f}, {ci_s[1]:.2f}, {ci_s[2]:.2f}]")
    print(f"  Shorts PF > 1.0: {'YES' if ci_s[0] > 1.0 else 'NO'}")

    # ================================================================
    # TEST 3: Temporal split
    # ================================================================
    print("\n" + "#" * 80)
    print("TEST 3: TEMPORAL SPLIT")
    print("#" * 80)

    years = sorted(df["year"].unique())
    mid = len(years) // 2
    train_years = set(years[:mid])
    test_years = set(years[mid:])
    print(f"\n  Train: {min(train_years)}-{max(train_years)}")
    print(f"  Test:  {min(test_years)}-{max(test_years)}")

    for tag, subset in [("ALL", trades), ("LONGS", longs), ("SHORTS", shorts)]:
        df_sub = pd.DataFrame(subset)
        df_sub["year"] = pd.to_datetime(df_sub["entry_time"]).dt.year
        tr = df_sub[df_sub["year"].isin(train_years)]
        te = df_sub[df_sub["year"].isin(test_years)]
        if len(tr) < 20 or len(te) < 20:
            print(f"  {tag:10s} | too few")
            continue
        m_tr = compute_metrics(tr.to_dict("records"))
        m_te = compute_metrics(te.to_dict("records"))
        holds = "HOLDS" if m_te["PF"] > 1.2 else "WEAK" if m_te["PF"] > 1.0 else "FAILS"
        print(f"  {tag:10s} | train: PF={m_tr['PF']:5.2f} ({len(tr):4d}t) | test: PF={m_te['PF']:5.2f} ({len(te):4d}t) | {holds}")

    # Also for PDL+PDH
    print("\n  PDL+PDH:")
    for tag, subset in [("ALL", trades_pp), ("LONGS", longs_pp), ("SHORTS", shorts_pp)]:
        df_sub = pd.DataFrame(subset)
        df_sub["year"] = pd.to_datetime(df_sub["entry_time"]).dt.year
        tr = df_sub[df_sub["year"].isin(train_years)]
        te = df_sub[df_sub["year"].isin(test_years)]
        if len(tr) < 20 or len(te) < 20:
            print(f"  {tag:10s} | too few")
            continue
        m_tr = compute_metrics(tr.to_dict("records"))
        m_te = compute_metrics(te.to_dict("records"))
        holds = "HOLDS" if m_te["PF"] > 1.2 else "WEAK" if m_te["PF"] > 1.0 else "FAILS"
        print(f"  {tag:10s} | train: PF={m_tr['PF']:5.2f} ({len(tr):4d}t) | test: PF={m_te['PF']:5.2f} ({len(te):4d}t) | {holds}")

    # ================================================================
    # TEST 4: Year-by-year
    # ================================================================
    print("\n" + "#" * 80)
    print("TEST 4: YEAR-BY-YEAR (ON_low + ON_high)")
    print("#" * 80)

    for tag, subset in [("LONGS", longs), ("SHORTS", shorts), ("ALL", trades)]:
        df_sub = pd.DataFrame(subset)
        df_sub["year"] = pd.to_datetime(df_sub["entry_time"]).dt.year
        print(f"\n  --- {tag} ---")
        neg = 0
        for yr in sorted(df_sub["year"].unique()):
            grp = df_sub[df_sub["year"] == yr]
            if len(grp) < 3:
                continue
            r = grp["r"].values
            total_r = r.sum()
            w = r[r > 0].sum()
            lo = abs(r[r < 0].sum())
            pf = w / lo if lo > 0 else 999
            flag = " NEG" if total_r < 0 else ""
            if total_r < 0:
                neg += 1
            print(f"    {yr}: {len(grp):3d}t  R={total_r:+7.1f}  PF={pf:5.2f}{flag}")
        print(f"    Negative years: {neg}/{len(df_sub['year'].unique())}")

    # ================================================================
    # VERDICT
    # ================================================================
    print("\n" + "=" * 120)
    print("VERDICT")
    print("=" * 120)

    m_all = compute_metrics(trades)
    m_l = compute_metrics(longs)
    m_s = compute_metrics(shorts)

    print(f"\n  ON_low + ON_high (best config):")
    pr("  LONGS", m_l)
    pr("  SHORTS", m_s)
    pr("  COMBINED", m_all)

    m_pp = compute_metrics(trades_pp)
    print(f"\n  PDL + PDH:")
    pr("  COMBINED", m_pp)

    print(f"\n  U2 baseline (longs only): 2589t R=+791.5 PF=1.59 PPDD=28.35 DD=27.9R")

    print("\n  If all tests pass -> proceed to production engine (chain_engine.py)")
    print("=" * 120)


if __name__ == "__main__":
    main()
