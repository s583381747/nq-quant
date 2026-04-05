"""
experiments/sprint8_validation.py -- Statistical Validation of Sprint 8 Findings
================================================================================

Phase 1: Before implementing in engine, validate that findings are statistically
significant and temporally stable.

Test 1: Bootstrap confidence intervals on key PF values
Test 2: PF difference bootstrap (breakdown vs bounce, disp good vs bad)
Test 3: Temporal split (first half vs second half of years)
Test 4: Year-by-year stability
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.ict_context_diagnostic import run_u2_with_context
from experiments.u2_clean import load_all, compute_metrics
from experiments.sweep_research import compute_pdhl, compute_htf_swings, compute_htf_swings_4h
from experiments.level_reaction_research import detect_level_touches, link_trades_to_reactions


def bootstrap_pf(r_arr, n_boot=10000):
    """Bootstrap PF confidence interval."""
    pfs = []
    n = len(r_arr)
    for _ in range(n_boot):
        sample = np.random.choice(r_arr, size=n, replace=True)
        wins = sample[sample > 0].sum()
        losses = abs(sample[sample < 0].sum())
        pf = wins / losses if losses > 0 else 5.0
        pfs.append(min(pf, 10.0))  # cap for stability
    pfs = np.array(pfs)
    return np.percentile(pfs, [2.5, 50, 97.5])


def bootstrap_pf_diff(r1, r2, n_boot=10000):
    """Bootstrap the PF difference between two groups."""
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
    print("PHASE 1: STATISTICAL VALIDATION -- Sprint 8 Findings")
    print("=" * 120)

    # Load + compute
    d = load_all()
    nq = d["nq"]
    n = d["n"]
    h, l, c, o = d["h"], d["l"], d["c"], d["o"]
    atr = d["atr_arr"]

    session_cache = pd.read_parquet(PROJECT / "data" / "cache_session_levels_10yr_v2.parquet")
    pdhl = compute_pdhl(nq)
    htf_1h = compute_htf_swings(nq, left=12, right=3)
    htf_4h = compute_htf_swings_4h(nq, left=48, right=12)

    level_defs = {
        "pdl": (pdhl["pdl"].values, "low"),
        "overnight_low": (session_cache["overnight_low"].values, "low"),
        "asia_low": (session_cache["asia_low"].values, "low"),
        "london_low": (session_cache["london_low"].values, "low"),
        "htf_1h_low": (htf_1h["htf_swing_low_price"].values, "low"),
        "htf_4h_low": (htf_4h["htf4h_swing_low_price"].values, "low"),
    }
    all_reactions = {}
    for lname, (arr, ltype) in level_defs.items():
        all_reactions[lname] = detect_level_touches(
            h, l, c, o, atr, arr, ltype,
            touch_threshold_pts=2.0, reaction_window=6,
        )

    trades = run_u2_with_context(d,
        stop_strategy="A2", fvg_size_mult=0.3, max_fvg_age=200,
        min_stop_pts=5.0, tighten_factor=0.85, tp_mult=0.35, nth_swing=5,
    )
    longs = [t for t in trades if t["dir"] == 1]
    longs = link_trades_to_reactions(longs, all_reactions, max_lookback_bars=60)
    df = pd.DataFrame(longs)
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year

    np.random.seed(42)

    # ==============================================================
    # TEST 1: Bootstrap confidence intervals
    # ==============================================================
    print("\n" + "#" * 80)
    print("TEST 1: BOOTSTRAP CONFIDENCE INTERVALS (10000 resamples)")
    print("Question: Is each group's PF significantly > 1.0?")
    print("#" * 80)

    groups = [
        ("ALL longs (baseline)", df["r"].values),
        ("Breakdown reaction", df[df["reaction_class"] == "breakdown"]["r"].values),
        ("Strong bounce reaction", df[df["reaction_class"] == "strong_bounce"]["r"].values),
        ("disp 0.3-1.5", df[(df["disp_atr_mult"] >= 0.3) & (df["disp_atr_mult"] <= 1.5)]["r"].values),
        ("disp > 1.5 (the bad ones)", df[df["disp_atr_mult"] > 1.5]["r"].values),
    ]

    # Add PDL breakdown if available
    if "react_pdl" in df.columns:
        pdl_bd = df[df["react_pdl"] == "breakdown"]["r"].values
        groups.append(("PDL breakdown", pdl_bd))
        pdl_any = df[df["react_pdl"] != "none"]["r"].values
        groups.append(("PDL any reaction", pdl_any))
        combo = df[
            (df["disp_atr_mult"] >= 0.3) & (df["disp_atr_mult"] <= 1.5) &
            (df["react_pdl"] != "none")
        ]["r"].values
        groups.append(("disp + PDL reaction", combo))

    print(f"\n  {'Group':45s} | {'N':>5s} | {'PF':>6s} | {'2.5%':>6s} | {'50%':>6s} | {'97.5%':>6s} | {'PF>1?':>6s}")
    print(f"  {'-'*45}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    for name, r_arr in groups:
        if len(r_arr) < 20:
            print(f"  {name:45s} | {len(r_arr):5d} |    -- |    -- |    -- |    -- | too few")
            continue
        wins = r_arr[r_arr > 0].sum()
        losses = abs(r_arr[r_arr < 0].sum())
        pf = wins / losses if losses > 0 else 999
        ci = bootstrap_pf(r_arr)
        sig = "YES" if ci[0] > 1.0 else "NO"
        print(f"  {name:45s} | {len(r_arr):5d} | {pf:5.2f} | {ci[0]:5.2f} | {ci[1]:5.2f} | {ci[2]:5.2f} | {sig:>6s}")

    # ==============================================================
    # TEST 2: PF difference bootstrap
    # ==============================================================
    print("\n" + "#" * 80)
    print("TEST 2: PF DIFFERENCE BOOTSTRAP")
    print("Question: Is group A significantly BETTER than group B?")
    print("#" * 80)

    comparisons = []

    r_bd = df[df["reaction_class"] == "breakdown"]["r"].values
    r_sb = df[df["reaction_class"] == "strong_bounce"]["r"].values
    if len(r_bd) >= 20 and len(r_sb) >= 20:
        comparisons.append(("Breakdown vs Strong bounce", r_bd, r_sb))

    r_good_disp = df[(df["disp_atr_mult"] >= 0.3) & (df["disp_atr_mult"] <= 1.5)]["r"].values
    r_bad_disp = df[df["disp_atr_mult"] > 1.5]["r"].values
    if len(r_good_disp) >= 20 and len(r_bad_disp) >= 20:
        comparisons.append(("disp[0.3-1.5] vs disp[>1.5]", r_good_disp, r_bad_disp))

    if "react_pdl" in df.columns:
        r_pdl = df[df["react_pdl"] != "none"]["r"].values
        r_no_pdl = df[df["react_pdl"] == "none"]["r"].values
        if len(r_pdl) >= 20 and len(r_no_pdl) >= 20:
            comparisons.append(("PDL reaction vs no PDL", r_pdl, r_no_pdl))

    for name, r1, r2 in comparisons:
        ci = bootstrap_pf_diff(r1, r2)
        sig = "YES" if ci[0] > 0 else "NO"
        print(f"\n  {name}:")
        print(f"    N: {len(r1)} vs {len(r2)}")
        print(f"    PF diff 95% CI: [{ci[0]:+.2f}, {ci[1]:+.2f}, {ci[2]:+.2f}]")
        print(f"    Significant (CI > 0): {sig}")

    # ==============================================================
    # TEST 3: Temporal split
    # ==============================================================
    print("\n" + "#" * 80)
    print("TEST 3: TEMPORAL SPLIT (first half vs second half)")
    print("Question: Do findings hold out-of-sample?")
    print("#" * 80)

    years = sorted(df["year"].unique())
    mid = len(years) // 2
    train_years = set(years[:mid])
    test_years = set(years[mid:])
    print(f"\n  Train: {min(train_years)}-{max(train_years)} ({len(train_years)} years)")
    print(f"  Test:  {min(test_years)}-{max(test_years)} ({len(test_years)} years)")

    df_train = df[df["year"].isin(train_years)]
    df_test = df[df["year"].isin(test_years)]

    filters = [
        ("Baseline (all longs)", pd.Series(True, index=df.index)),
        ("disp 0.3-1.5", (df["disp_atr_mult"] >= 0.3) & (df["disp_atr_mult"] <= 1.5)),
        ("Breakdown reaction", df["reaction_class"] == "breakdown"),
        ("Strong bounce reaction", df["reaction_class"] == "strong_bounce"),
    ]
    if "react_pdl" in df.columns:
        filters.append(("PDL breakdown", df["react_pdl"] == "breakdown"))
        filters.append(("PDL any reaction", df["react_pdl"] != "none"))
        filters.append(("disp + PDL reaction",
            (df["disp_atr_mult"] >= 0.3) & (df["disp_atr_mult"] <= 1.5) &
            (df["react_pdl"] != "none")))

    print(f"\n  {'Filter':45s} | {'Train PF':>10s} | {'Train N':>8s} | {'Test PF':>10s} | {'Test N':>8s} | Result")
    print(f"  {'-'*45}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+--------")

    for name, mask in filters:
        tr = df_train[mask.reindex(df_train.index, fill_value=False)]
        te = df_test[mask.reindex(df_test.index, fill_value=False)]
        if len(tr) < 20 or len(te) < 20:
            print(f"  {name:45s} | {'--':>10s} | {len(tr):8d} | {'--':>10s} | {len(te):8d} | too few")
            continue
        m_tr = compute_metrics(tr.to_dict("records"))
        m_te = compute_metrics(te.to_dict("records"))
        holds = "HOLDS" if m_te["PF"] > 1.2 and m_te["PF"] > m_tr["PF"] * 0.6 else "WEAK" if m_te["PF"] > 1.0 else "FAILS"
        print(f"  {name:45s} | {m_tr['PF']:10.2f} | {len(tr):8d} | {m_te['PF']:10.2f} | {len(te):8d} | {holds}")

    # ==============================================================
    # TEST 4: Year-by-year stability
    # ==============================================================
    print("\n" + "#" * 80)
    print("TEST 4: YEAR-BY-YEAR STABILITY")
    print("#" * 80)

    key_filters = [
        ("disp 0.3-1.5", (df["disp_atr_mult"] >= 0.3) & (df["disp_atr_mult"] <= 1.5)),
        ("Breakdown reaction", df["reaction_class"] == "breakdown"),
    ]
    if "react_pdl" in df.columns:
        key_filters.append(("disp + PDL reaction",
            (df["disp_atr_mult"] >= 0.3) & (df["disp_atr_mult"] <= 1.5) &
            (df["react_pdl"] != "none")))

    for filter_name, mask in key_filters:
        sub = df[mask]
        print(f"\n  --- {filter_name} ({len(sub)} trades) ---")
        neg = 0
        for yr in sorted(sub["year"].unique()):
            yr_df = sub[sub["year"] == yr]
            if len(yr_df) < 3:
                continue
            r = yr_df["r"].values
            total_r = r.sum()
            wins = r[r > 0].sum()
            losses = abs(r[r < 0].sum())
            pf = wins / losses if losses > 0 else 999
            flag = " NEG" if total_r < 0 else ""
            if total_r < 0:
                neg += 1
            print(f"    {yr}: {len(yr_df):3d}t  R={total_r:+7.1f}  PF={pf:5.2f}{flag}")
        print(f"    Negative years: {neg}/{len(sub['year'].unique())}")

    # ==============================================================
    # VERDICT
    # ==============================================================
    print("\n" + "=" * 120)
    print("VERDICT")
    print("=" * 120)
    print("Check above:")
    print("  1. Bootstrap: Is PF 95% CI above 1.0 for key groups?")
    print("  2. Difference: Is breakdown vs bounce difference significant?")
    print("  3. Temporal: Does PF hold in test period (second half)?")
    print("  4. Year-by-year: How many negative years?")
    print("  -> If all pass: PROCEED to engine implementation")
    print("  -> If any fail: STOP and investigate before implementing")
    print("=" * 120)


if __name__ == "__main__":
    main()
