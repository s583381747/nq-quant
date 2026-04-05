"""
experiments/p1_pd_depth_sizing.py — Premium/Discount Depth as Sizing Signal
============================================================================

Research question: Should "deep discount longs" get more R than "shallow discount longs"?

Method:
  1. Run trend-only backtest (exclude breakdown FVGs)
  2. For each trade, compute ON position = (entry - ON_low) / (ON_high - ON_low)
  3. For longs: discount_depth = 0.5 - on_position (higher = deeper in discount)
  4. For shorts: premium_depth = on_position - 0.5 (higher = deeper in premium)
  5. Only look at PD + bias aligned trades
  6. Split by depth bins, compute PF/R per bin
  7. Test monotonicity: deeper depth → better PF?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.trend_fvg_research import run_trend_fvg_backtest
from experiments.chain_engine import load_all, compute_metrics


def main():
    print("=" * 120)
    print("P1: PREMIUM/DISCOUNT DEPTH AS SIZING SIGNAL")
    print("=" * 120)

    d = load_all()
    nq = d["nq"]
    on_hi = d["on_hi"]
    on_lo = d["on_lo"]
    bias_dir_arr = d["bias_dir_arr"]

    # Build bar index lookup: timestamp -> bar index
    idx_map = {ts: i for i, ts in enumerate(nq.index)}

    # --------------------------------------------------------------------------
    # Run trend-only backtest (no breakdown FVGs)
    # --------------------------------------------------------------------------
    print("\n[RUN] Running trend-only backtest (include_breakdown_fvgs=False)...")
    trades = run_trend_fvg_backtest(d, include_breakdown_fvgs=False)
    print(f"[RUN] Got {len(trades)} trend trades")
    m_all = compute_metrics(trades)
    print(f"  Baseline: {m_all['trades']}t  R={m_all['R']:+.1f}  PF={m_all['PF']:.2f}  PPDD={m_all['PPDD']:.2f}  MaxDD={m_all['MaxDD']:.1f}R")

    # --------------------------------------------------------------------------
    # Compute PD depth for each trade
    # --------------------------------------------------------------------------
    records = []
    for t in trades:
        entry_ts = t["entry_time"]
        bar_idx = idx_map.get(entry_ts)
        if bar_idx is None:
            continue

        onh = on_hi[bar_idx]
        onl = on_lo[bar_idx]
        if np.isnan(onh) or np.isnan(onl) or onh <= onl or onh <= 0:
            continue

        on_range = onh - onl
        on_position = (t["entry_price"] - onl) / on_range  # 0=ON low, 1=ON high

        direction = t["dir"]
        bias = bias_dir_arr[bar_idx]

        # PD + bias aligned check
        if direction == 1:
            # Long: want discount (on_position < 0.5) and bullish bias
            discount_depth = 0.5 - on_position  # positive = in discount, higher = deeper
            pd_depth = discount_depth
            pd_aligned = (on_position < 0.5) and (bias > 0)
        else:
            # Short: want premium (on_position > 0.5) and bearish bias
            premium_depth = on_position - 0.5  # positive = in premium, higher = deeper
            pd_depth = premium_depth
            pd_aligned = (on_position > 0.5) and (bias < 0)

        records.append({
            "entry_time": entry_ts,
            "dir": direction,
            "r": t["r"],
            "reason": t["reason"],
            "grade": t["grade"],
            "entry_price": t["entry_price"],
            "on_hi": onh,
            "on_lo": onl,
            "on_range": on_range,
            "on_position": on_position,
            "pd_depth": pd_depth,
            "pd_aligned": pd_aligned,
            "bias": bias,
            "bar_idx": bar_idx,
        })

    df = pd.DataFrame(records)
    print(f"\n  Trades with valid ON data: {len(df)} / {len(trades)}")

    # --------------------------------------------------------------------------
    # Section 1: PD alignment breakdown
    # --------------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("SECTION 1: PD ALIGNMENT BREAKDOWN")
    print(f"{'='*120}")

    for label, mask in [
        ("All trades", df.index == df.index),
        ("PD aligned", df["pd_aligned"]),
        ("PD NOT aligned", ~df["pd_aligned"]),
    ]:
        sub = df[mask]
        if len(sub) < 10:
            print(f"  {label:30s} | {len(sub):5d}t | too few")
            continue
        r_arr = sub["r"].values
        total_r = r_arr.sum()
        wins = r_arr[r_arr > 0].sum()
        losses = abs(r_arr[r_arr < 0].sum())
        pf = wins / losses if losses > 0 else 999
        wr = (r_arr > 0).mean() * 100
        avg_r = total_r / len(r_arr)
        cumr = np.cumsum(r_arr)
        peak = np.maximum.accumulate(cumr)
        dd = (peak - cumr).max()
        ppdd = total_r / dd if dd > 0 else 999
        print(f"  {label:30s} | {len(sub):5d}t | R={total_r:+8.1f} | PF={pf:5.2f} | WR={wr:5.1f}% | avgR={avg_r:+.4f} | PPDD={ppdd:.2f} | DD={dd:.1f}R")

    # Same for longs / shorts separately
    for dir_label, dir_val in [("LONGS", 1), ("SHORTS", -1)]:
        ddir = df[df["dir"] == dir_val]
        print(f"\n  --- {dir_label} ---")
        for label, mask in [
            ("All", ddir.index == ddir.index),
            ("PD aligned", ddir["pd_aligned"]),
            ("PD NOT aligned", ~ddir["pd_aligned"]),
        ]:
            sub = ddir[mask]
            if len(sub) < 10:
                print(f"    {label:28s} | {len(sub):5d}t | too few")
                continue
            r_arr = sub["r"].values
            total_r = r_arr.sum()
            wins = r_arr[r_arr > 0].sum()
            losses = abs(r_arr[r_arr < 0].sum())
            pf = wins / losses if losses > 0 else 999
            wr = (r_arr > 0).mean() * 100
            avg_r = total_r / len(r_arr)
            print(f"    {label:28s} | {len(sub):5d}t | R={total_r:+8.1f} | PF={pf:5.2f} | WR={wr:5.1f}% | avgR={avg_r:+.4f}")

    # --------------------------------------------------------------------------
    # Section 2: PD Depth bins (PD-aligned trades only)
    # --------------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("SECTION 2: PD DEPTH BINS (PD-aligned trades only)")
    print("=" * 120)

    aligned = df[df["pd_aligned"]].copy()
    if len(aligned) < 20:
        print("  Too few PD-aligned trades to bin.")
        return

    # Depth bins: negative depth = wrong side, 0 = equilibrium, positive = aligned deep
    # Since pd_aligned filter already ensures pd_depth direction is correct for longs/shorts,
    # we may still have negative pd_depth values for some edge cases.
    print(f"\n  PD depth stats (PD-aligned subset, n={len(aligned)}):")
    print(f"    min={aligned['pd_depth'].min():.4f}  median={aligned['pd_depth'].median():.4f}  "
          f"mean={aligned['pd_depth'].mean():.4f}  max={aligned['pd_depth'].max():.4f}")

    # Quintile bins
    print(f"\n  --- Quintile bins (PD-aligned) ---")
    try:
        aligned["depth_q"] = pd.qcut(aligned["pd_depth"], q=5, duplicates="drop")
    except ValueError:
        aligned["depth_q"] = pd.qcut(aligned["pd_depth"], q=3, duplicates="drop")

    print(f"\n  {'Depth Bin':35s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>8s} | {'avgW':>7s} | {'avgL':>7s}")
    print(f"  {'-'*35}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}")

    pf_by_bin = []
    for bin_label, grp in aligned.groupby("depth_q", observed=True):
        r_arr = grp["r"].values
        total_r = r_arr.sum()
        wins = r_arr[r_arr > 0]
        losses = r_arr[r_arr < 0]
        w_sum = wins.sum()
        l_sum = abs(losses.sum())
        pf = w_sum / l_sum if l_sum > 0 else 999
        wr = (r_arr > 0).mean() * 100
        avg_r = total_r / len(r_arr)
        avg_w = wins.mean() if len(wins) > 0 else 0
        avg_l = losses.mean() if len(losses) > 0 else 0
        pf_by_bin.append((str(bin_label), pf, total_r, len(grp)))
        print(f"  {str(bin_label):35s} | {len(grp):5d} | {total_r:+8.1f} | {pf:5.2f} | {wr:5.1f}% | {avg_r:+.4f} | {avg_w:+.4f} | {avg_l:+.4f}")

    # Monotonicity check
    pfs = [x[1] for x in pf_by_bin]
    is_monotonic_inc = all(pfs[i] <= pfs[i+1] for i in range(len(pfs)-1))
    is_monotonic_dec = all(pfs[i] >= pfs[i+1] for i in range(len(pfs)-1))
    print(f"\n  Monotonicity: PF increasing with depth? {is_monotonic_inc}")
    print(f"  Monotonicity: PF decreasing with depth? {is_monotonic_dec}")
    if not is_monotonic_inc and not is_monotonic_dec:
        print("  -> Non-monotonic: no clear linear relationship between depth and PF")

    # --------------------------------------------------------------------------
    # Section 2b: Fixed depth thresholds
    # --------------------------------------------------------------------------
    print(f"\n  --- Fixed depth thresholds (PD-aligned) ---")
    thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    print(f"\n  {'Depth >=':12s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>8s} | {'PPDD':>6s}")
    print(f"  {'-'*12}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}")

    for thresh in thresholds:
        sub = aligned[aligned["pd_depth"] >= thresh]
        if len(sub) < 10:
            print(f"  >= {thresh:.2f}       | {len(sub):5d} | too few")
            continue
        r_arr = sub["r"].values
        total_r = r_arr.sum()
        w = r_arr[r_arr > 0].sum()
        lo = abs(r_arr[r_arr < 0].sum())
        pf = w / lo if lo > 0 else 999
        wr = (r_arr > 0).mean() * 100
        avg_r = total_r / len(r_arr)
        cumr = np.cumsum(r_arr)
        peak = np.maximum.accumulate(cumr)
        dd = (peak - cumr).max()
        ppdd = total_r / dd if dd > 0 else 999
        print(f"  >= {thresh:.2f}       | {len(sub):5d} | {total_r:+8.1f} | {pf:5.2f} | {wr:5.1f}% | {avg_r:+.4f} | {ppdd:5.2f}")

    # --------------------------------------------------------------------------
    # Section 3: Longs vs Shorts depth analysis
    # --------------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("SECTION 3: DEPTH BINS BY DIRECTION (PD-aligned)")
    print("=" * 120)

    for dir_label, dir_val in [("LONGS (discount depth)", 1), ("SHORTS (premium depth)", -1)]:
        sub_dir = aligned[aligned["dir"] == dir_val].copy()
        if len(sub_dir) < 20:
            print(f"\n  {dir_label}: {len(sub_dir)} trades, too few for bins")
            continue

        print(f"\n  --- {dir_label} (n={len(sub_dir)}) ---")
        print(f"    Depth stats: min={sub_dir['pd_depth'].min():.4f}  median={sub_dir['pd_depth'].median():.4f}  "
              f"mean={sub_dir['pd_depth'].mean():.4f}  max={sub_dir['pd_depth'].max():.4f}")

        try:
            sub_dir["dq"] = pd.qcut(sub_dir["pd_depth"], q=4, duplicates="drop")
        except ValueError:
            sub_dir["dq"] = pd.qcut(sub_dir["pd_depth"], q=3, duplicates="drop")

        print(f"\n    {'Depth Bin':35s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>8s}")
        print(f"    {'-'*35}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")

        for bin_label, grp in sub_dir.groupby("dq", observed=True):
            r_arr = grp["r"].values
            total_r = r_arr.sum()
            w = r_arr[r_arr > 0].sum()
            lo = abs(r_arr[r_arr < 0].sum())
            pf = w / lo if lo > 0 else 999
            wr = (r_arr > 0).mean() * 100
            avg_r = total_r / len(r_arr)
            print(f"    {str(bin_label):35s} | {len(grp):5d} | {total_r:+8.1f} | {pf:5.2f} | {wr:5.1f}% | {avg_r:+.4f}")

    # --------------------------------------------------------------------------
    # Section 4: ON range normalization check
    # --------------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("SECTION 4: ON RANGE SIZE INTERACTION")
    print("=" * 120)

    aligned["on_range_q"] = pd.qcut(aligned["on_range"], q=3, labels=["Narrow", "Mid", "Wide"], duplicates="drop")
    aligned["deep"] = aligned["pd_depth"] >= aligned["pd_depth"].median()

    print(f"\n  {'ON Range':10s} | {'Deep':6s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'avgR':>8s}")
    print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}")

    for rng_label, rng_grp in aligned.groupby("on_range_q", observed=True):
        for deep_label, deep_grp in rng_grp.groupby("deep"):
            sub = deep_grp
            if len(sub) < 10:
                continue
            r_arr = sub["r"].values
            total_r = r_arr.sum()
            w = r_arr[r_arr > 0].sum()
            lo = abs(r_arr[r_arr < 0].sum())
            pf = w / lo if lo > 0 else 999
            avg_r = total_r / len(r_arr)
            dl = "Deep" if deep_label else "Shallow"
            print(f"  {str(rng_label):10s} | {dl:6s} | {len(sub):5d} | {total_r:+8.1f} | {pf:5.2f} | {avg_r:+.4f}")

    # --------------------------------------------------------------------------
    # Section 5: Simulated sizing test
    # --------------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("SECTION 5: SIMULATED DEPTH-BASED SIZING")
    print("=" * 120)
    print("  Compare: flat 1.0R vs depth-scaled R (deeper = more R)")

    # Flat baseline: all PD-aligned at 1.0R
    baseline_r = aligned["r"].values
    baseline_total = baseline_r.sum()
    baseline_cumr = np.cumsum(baseline_r)
    baseline_peak = np.maximum.accumulate(baseline_cumr)
    baseline_dd = (baseline_peak - baseline_cumr).max()
    baseline_pf_val = baseline_r[baseline_r > 0].sum() / abs(baseline_r[baseline_r < 0].sum()) if abs(baseline_r[baseline_r < 0].sum()) > 0 else 999
    baseline_ppdd = baseline_total / baseline_dd if baseline_dd > 0 else 999

    print(f"\n  Flat 1.0R:  {len(aligned)}t  R={baseline_total:+.1f}  PF={baseline_pf_val:.2f}  PPDD={baseline_ppdd:.2f}  DD={baseline_dd:.1f}R")

    # Depth-scaled sizing schemes
    schemes = [
        ("Linear 0.5-1.5R", lambda d: 0.5 + d * 2.0),   # depth 0 -> 0.5R, depth 0.5 -> 1.5R
        ("Linear 0.5-2.0R", lambda d: 0.5 + d * 3.0),   # depth 0 -> 0.5R, depth 0.5 -> 2.0R
        ("Step: <0.15=0.5R, >=0.15=1.0R", lambda d: 1.0 if d >= 0.15 else 0.5),
        ("Step: <0.10=0.5R, >=0.10=1.0R, >=0.25=1.5R",
         lambda d: 1.5 if d >= 0.25 else (1.0 if d >= 0.10 else 0.5)),
        ("Step: <0.15=skip, >=0.15=1.0R",
         lambda d: 1.0 if d >= 0.15 else 0.0),  # filter out shallow
        ("Step: <0.10=skip, >=0.10=1.0R",
         lambda d: 1.0 if d >= 0.10 else 0.0),
    ]

    print(f"\n  {'Scheme':45s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'PPDD':>6s} | {'DD':>6s} | {'vs flat':>8s}")
    print(f"  {'-'*45}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")

    for scheme_name, size_fn in schemes:
        depths = aligned["pd_depth"].values
        r_vals = aligned["r"].values
        scaled_r = np.array([r * size_fn(d) for r, d in zip(r_vals, depths)])

        # Remove zero-sized (filtered out)
        mask = np.array([size_fn(d) > 0 for d in depths])
        scaled_r = scaled_r[mask]

        if len(scaled_r) < 10:
            print(f"  {scheme_name:45s} | {len(scaled_r):5d} | too few")
            continue

        total_r = scaled_r.sum()
        w = scaled_r[scaled_r > 0].sum()
        lo = abs(scaled_r[scaled_r < 0].sum())
        pf = w / lo if lo > 0 else 999
        cumr = np.cumsum(scaled_r)
        peak = np.maximum.accumulate(cumr)
        dd = (peak - cumr).max()
        ppdd = total_r / dd if dd > 0 else 999
        delta = ppdd - baseline_ppdd
        print(f"  {scheme_name:45s} | {sum(mask):5d} | {total_r:+8.1f} | {pf:5.2f} | {ppdd:5.2f} | {dd:5.1f}R | {delta:+.2f}")

    # --------------------------------------------------------------------------
    # Section 6: ON position distribution
    # --------------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("SECTION 6: ON POSITION DISTRIBUTION (all trades)")
    print("=" * 120)

    print(f"\n  {'ON Position Bin':20s} | {'Longs':>6s} | {'L_PF':>6s} | {'L_R':>8s} | {'Shorts':>6s} | {'S_PF':>6s} | {'S_R':>8s}")
    print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")

    on_bins = [(0.0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.0)]
    for lo_bound, hi_bound in on_bins:
        label = f"{lo_bound:.2f}-{hi_bound:.2f}"
        for dir_val, dir_name in [(1, "L"), (-1, "S")]:
            sub = df[(df["on_position"] >= lo_bound) & (df["on_position"] < hi_bound) & (df["dir"] == dir_val)]
            if dir_val == 1:
                l_sub = sub
            else:
                s_sub = sub

        l_n = len(l_sub)
        s_n = len(s_sub)
        l_r = l_sub["r"].sum() if l_n > 0 else 0
        s_r = s_sub["r"].sum() if s_n > 0 else 0
        l_pf = l_sub["r"][l_sub["r"] > 0].sum() / abs(l_sub["r"][l_sub["r"] < 0].sum()) if l_n > 5 and abs(l_sub["r"][l_sub["r"] < 0].sum()) > 0 else 0
        s_pf = s_sub["r"][s_sub["r"] > 0].sum() / abs(s_sub["r"][s_sub["r"] < 0].sum()) if s_n > 5 and abs(s_sub["r"][s_sub["r"] < 0].sum()) > 0 else 0

        print(f"  {label:20s} | {l_n:6d} | {l_pf:5.2f} | {l_r:+8.1f} | {s_n:6d} | {s_pf:5.2f} | {s_r:+8.1f}")

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("SUMMARY")
    print("=" * 120)
    print("""
  Key questions answered:
  1. Does PD alignment help? (PD-aligned vs not)
  2. Does deeper depth = better PF? (monotonic bins)
  3. Does depth-based sizing improve PPDD vs flat sizing?
  4. Where in the ON range are longs/shorts most profitable?
""")


if __name__ == "__main__":
    main()
