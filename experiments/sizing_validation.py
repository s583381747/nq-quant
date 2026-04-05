"""
experiments/sizing_validation.py -- Validate sizing signals before implementation
================================================================================

3 signals to validate:
  1. Sweep bar range >= 1.3 ATR → 1.5x R
  2. Entry below ON low → 1.5x R
  3. AM shorts (10-12 ET) → 0.5x R

Tests: Bootstrap CI, temporal split, year-by-year
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.chain_engine import load_all, run_chain_backtest, compute_metrics, detect_breakdowns


def bootstrap_pf(r_arr, n_boot=10000):
    pfs = []
    n = len(r_arr)
    for _ in range(n_boot):
        sample = np.random.choice(r_arr, size=n, replace=True)
        w = sample[sample > 0].sum()
        lo = abs(sample[sample < 0].sum())
        pfs.append(min(w / lo if lo > 0 else 5, 10))
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
    d = load_all()
    nq, n = d["nq"], d["n"]
    h, l, c, o = d["h"], d["l"], d["c"], d["o"]
    atr_arr = d["atr_arr"]
    on_lo, on_hi = d["on_lo"], d["on_hi"]
    et_frac_arr = d["et_frac_arr"]
    nq_idx = nq.index

    trades, _ = run_chain_backtest(d)
    df = pd.DataFrame(trades)

    # Tag each trade with the 3 signals
    bd_lo = detect_breakdowns(h, l, c, on_lo, "low", min_depth_pts=1.0)
    bd_hi = detect_breakdowns(h, l, c, on_hi, "high", min_depth_pts=1.0)
    bd_bars_lo = sorted([bd["bar_idx"] for bd in bd_lo])
    bd_bars_hi = sorted([bd["bar_idx"] for bd in bd_hi])

    for _, t in df.iterrows():
        idx = nq_idx.get_loc(t["entry_time"])
        direction = t["dir"]

        # Find matching breakdown
        bd_bars = bd_bars_lo if direction == 1 else bd_bars_hi
        bd_bar = -1
        for b in reversed(bd_bars):
            if b < idx:
                bd_bar = b
                break

        # Signal 1: Sweep bar range
        if bd_bar >= 0:
            atr = atr_arr[bd_bar] if not np.isnan(atr_arr[bd_bar]) else 30.0
            sweep_range = h[bd_bar] - l[bd_bar]
            df.at[_, "sweep_range_atr"] = sweep_range / atr if atr > 0 else 0
        else:
            df.at[_, "sweep_range_atr"] = 0

        # Signal 2: Entry below ON low (longs) / above ON high (shorts)
        if direction == 1:
            on_level = on_lo[idx]
            df.at[_, "in_discount"] = t["entry_price"] < on_level if not np.isnan(on_level) else False
        else:
            on_level = on_hi[idx]
            df.at[_, "in_discount"] = t["entry_price"] > on_level if not np.isnan(on_level) else False

        # Signal 3: AM short
        et = nq_idx[idx].tz_convert("US/Eastern") if hasattr(nq_idx[idx], "tz_convert") else nq_idx[idx]
        hour = et.hour + et.minute / 60
        df.at[_, "is_am_short"] = (direction == -1) and (10.0 <= hour < 12.0)
        df.at[_, "et_hour"] = hour

    # Derived columns
    df["big_sweep"] = df["sweep_range_atr"] >= 1.3
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year

    np.random.seed(42)

    print("=" * 120)
    print("SIZING SIGNAL VALIDATION")
    print("=" * 120)

    m_base = compute_metrics(trades)
    print(f"\nBaseline: {m_base['trades']}t  R={m_base['R']:+.1f}  PF={m_base['PF']:.2f}  PPDD={m_base['PPDD']:.2f}")

    # ================================================================
    # SIGNAL 1: Sweep bar range
    # ================================================================
    print(f"\n{'#'*80}")
    print("SIGNAL 1: Sweep bar range (big >= 1.3 ATR vs small)")
    print(f"{'#'*80}")

    big = df[df["big_sweep"]]
    small = df[~df["big_sweep"]]
    r_big = big["r"].values
    r_small = small["r"].values

    m_big = compute_metrics(big.to_dict("records"))
    m_small = compute_metrics(small.to_dict("records"))
    print(f"\n  Big sweep:   {m_big['trades']:4d}t  PF={m_big['PF']:.2f}  R={m_big['R']:+.1f}  avgR={m_big['avgR']:+.3f}")
    print(f"  Small sweep: {m_small['trades']:4d}t  PF={m_small['PF']:.2f}  R={m_small['R']:+.1f}  avgR={m_small['avgR']:+.3f}")

    # Bootstrap
    ci_big = bootstrap_pf(r_big)
    ci_small = bootstrap_pf(r_small)
    ci_diff = bootstrap_pf_diff(r_big, r_small)
    print(f"\n  Bootstrap PF CI:")
    print(f"    Big:   [{ci_big[0]:.2f}, {ci_big[1]:.2f}, {ci_big[2]:.2f}]  PF>1: {'YES' if ci_big[0] > 1 else 'NO'}")
    print(f"    Small: [{ci_small[0]:.2f}, {ci_small[1]:.2f}, {ci_small[2]:.2f}]  PF>1: {'YES' if ci_small[0] > 1 else 'NO'}")
    print(f"    Diff:  [{ci_diff[0]:+.2f}, {ci_diff[1]:+.2f}, {ci_diff[2]:+.2f}]  Big>Small: {'YES' if ci_diff[0] > 0 else 'NO'}")

    # Temporal split
    years = sorted(df["year"].unique())
    mid = len(years) // 2
    train_y = set(years[:mid])
    test_y = set(years[mid:])

    for tag, sub in [("Big sweep", big), ("Small sweep", small)]:
        tr = sub[sub["year"].isin(train_y)]
        te = sub[sub["year"].isin(test_y)]
        if len(tr) >= 10 and len(te) >= 10:
            m_tr = compute_metrics(tr.to_dict("records"))
            m_te = compute_metrics(te.to_dict("records"))
            holds = "HOLDS" if m_te["PF"] > 1.2 else "WEAK" if m_te["PF"] > 1.0 else "FAILS"
            print(f"  Temporal: {tag:15s}  train PF={m_tr['PF']:.2f} ({len(tr)}t) -> test PF={m_te['PF']:.2f} ({len(te)}t)  {holds}")

    # ================================================================
    # SIGNAL 2: In discount zone (below ON level)
    # ================================================================
    print(f"\n{'#'*80}")
    print("SIGNAL 2: Entry in discount zone (below ON low for longs / above ON high for shorts)")
    print(f"{'#'*80}")

    discount = df[df["in_discount"] == True]
    premium = df[df["in_discount"] == False]
    r_disc = discount["r"].values
    r_prem = premium["r"].values

    m_disc = compute_metrics(discount.to_dict("records"))
    m_prem = compute_metrics(premium.to_dict("records"))
    print(f"\n  Discount: {m_disc['trades']:4d}t  PF={m_disc['PF']:.2f}  R={m_disc['R']:+.1f}  avgR={m_disc['avgR']:+.3f}")
    print(f"  Premium:  {m_prem['trades']:4d}t  PF={m_prem['PF']:.2f}  R={m_prem['R']:+.1f}  avgR={m_prem['avgR']:+.3f}")

    ci_disc = bootstrap_pf(r_disc)
    ci_prem = bootstrap_pf(r_prem)
    ci_diff2 = bootstrap_pf_diff(r_disc, r_prem)
    print(f"\n  Bootstrap PF CI:")
    print(f"    Discount: [{ci_disc[0]:.2f}, {ci_disc[1]:.2f}, {ci_disc[2]:.2f}]  PF>1: {'YES' if ci_disc[0] > 1 else 'NO'}")
    print(f"    Premium:  [{ci_prem[0]:.2f}, {ci_prem[1]:.2f}, {ci_prem[2]:.2f}]  PF>1: {'YES' if ci_prem[0] > 1 else 'NO'}")
    print(f"    Diff:     [{ci_diff2[0]:+.2f}, {ci_diff2[1]:+.2f}, {ci_diff2[2]:+.2f}]  Disc>Prem: {'YES' if ci_diff2[0] > 0 else 'NO'}")

    for tag, sub in [("Discount", discount), ("Premium", premium)]:
        tr = sub[sub["year"].isin(train_y)]
        te = sub[sub["year"].isin(test_y)]
        if len(tr) >= 10 and len(te) >= 10:
            m_tr = compute_metrics(tr.to_dict("records"))
            m_te = compute_metrics(te.to_dict("records"))
            holds = "HOLDS" if m_te["PF"] > 1.2 else "WEAK" if m_te["PF"] > 1.0 else "FAILS"
            print(f"  Temporal: {tag:15s}  train PF={m_tr['PF']:.2f} ({len(tr)}t) -> test PF={m_te['PF']:.2f} ({len(te)}t)  {holds}")

    # ================================================================
    # SIGNAL 3: AM shorts
    # ================================================================
    print(f"\n{'#'*80}")
    print("SIGNAL 3: AM shorts (10-12 ET, direction=-1)")
    print(f"{'#'*80}")

    am_shorts = df[df["is_am_short"] == True]
    other = df[df["is_am_short"] == False]
    r_ams = am_shorts["r"].values
    r_other = other["r"].values

    m_ams = compute_metrics(am_shorts.to_dict("records"))
    m_other = compute_metrics(other.to_dict("records"))
    print(f"\n  AM shorts: {m_ams['trades']:4d}t  PF={m_ams['PF']:.2f}  R={m_ams['R']:+.1f}  avgR={m_ams['avgR']:+.3f}")
    print(f"  Others:    {m_other['trades']:4d}t  PF={m_other['PF']:.2f}  R={m_other['R']:+.1f}  avgR={m_other['avgR']:+.3f}")

    if len(r_ams) >= 10:
        ci_ams = bootstrap_pf(r_ams)
        print(f"\n  Bootstrap PF CI AM shorts: [{ci_ams[0]:.2f}, {ci_ams[1]:.2f}, {ci_ams[2]:.2f}]  PF>1: {'YES' if ci_ams[0] > 1 else 'NO'}")

    # Temporal for AM shorts
    for tag, sub in [("AM shorts", am_shorts), ("Others", other)]:
        tr = sub[sub["year"].isin(train_y)]
        te = sub[sub["year"].isin(test_y)]
        if len(tr) >= 5 and len(te) >= 5:
            m_tr = compute_metrics(tr.to_dict("records"))
            m_te = compute_metrics(te.to_dict("records"))
            holds = "HOLDS" if m_te["PF"] > 1.2 else "WEAK" if m_te["PF"] > 1.0 else "FAILS"
            print(f"  Temporal: {tag:15s}  train PF={m_tr['PF']:.2f} ({len(tr)}t) -> test PF={m_te['PF']:.2f} ({len(te)}t)  {holds}")

    # Year-by-year for AM shorts
    print(f"\n  AM shorts year-by-year:")
    neg = 0
    for yr in sorted(am_shorts["year"].unique()):
        grp = am_shorts[am_shorts["year"] == yr]
        if len(grp) < 2: continue
        r = grp["r"].values
        total_r = r.sum()
        w = r[r > 0].sum()
        lo = abs(r[r < 0].sum())
        pf = w / lo if lo > 0 else 999
        flag = " NEG" if total_r < 0 else ""
        if total_r < 0: neg += 1
        print(f"    {yr}: {len(grp):2d}t  R={total_r:+6.1f}  PF={pf:5.2f}{flag}")
    print(f"    Negative years: {neg}")

    # ================================================================
    # COMBINED: Simulate sizing
    # ================================================================
    print(f"\n{'#'*80}")
    print("COMBINED: Simulate sizing effect")
    print(f"{'#'*80}")

    # Base: all trades at 1.0x
    base_r = df["r"].sum()
    print(f"\n  Base (all 1.0x):  R={base_r:+.1f}")

    # With sizing: multiply R by size factor
    df["size_mult"] = 1.0
    # Big sweep → 1.5x
    df.loc[df["big_sweep"], "size_mult"] = df.loc[df["big_sweep"], "size_mult"] * 1.5
    # Discount → 1.5x
    df.loc[df["in_discount"] == True, "size_mult"] = df.loc[df["in_discount"] == True, "size_mult"] * 1.5
    # AM short → 0.5x
    df.loc[df["is_am_short"] == True, "size_mult"] = df.loc[df["is_am_short"] == True, "size_mult"] * 0.5

    df["sized_r"] = df["r"] * df["size_mult"]
    sized_total = df["sized_r"].sum()

    print(f"  Sized:            R={sized_total:+.1f}  (delta={sized_total - base_r:+.1f})")

    # Sized PF
    sized_wins = df.loc[df["sized_r"] > 0, "sized_r"].sum()
    sized_losses = abs(df.loc[df["sized_r"] < 0, "sized_r"].sum())
    sized_pf = sized_wins / sized_losses if sized_losses > 0 else 999
    print(f"  Sized PF: {sized_pf:.2f}  (base PF: {m_base['PF']:.2f})")

    # Size distribution
    print(f"\n  Size multiplier distribution:")
    for mult in sorted(df["size_mult"].unique()):
        cnt = (df["size_mult"] == mult).sum()
        sub_r = df[df["size_mult"] == mult]["r"].sum()
        print(f"    {mult:.2f}x: {cnt:4d} trades  R={sub_r:+.1f}")

    # ================================================================
    # VERDICT
    # ================================================================
    print(f"\n{'='*120}")
    print("VERDICT")
    print(f"{'='*120}")
    print("  Signal 1 (Sweep range): Check bootstrap diff CI > 0 and temporal split")
    print("  Signal 2 (Discount):    Check bootstrap diff CI > 0 and temporal split")
    print("  Signal 3 (AM shorts):   Check bootstrap PF CI includes 1.0 and temporal")
    print("  Combined sizing:        Does sized R > base R?")


if __name__ == "__main__":
    main()
