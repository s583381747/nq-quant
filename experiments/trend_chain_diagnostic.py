"""
experiments/trend_chain_diagnostic.py -- Trend FVG Chain Analysis
=================================================================

Same methodology as breakdown chain: what CONTEXT makes a trend FVG good?

For breakdown entries, the chain was: Level breakdown -> FVG -> entry
For trend entries, find the equivalent chain: ??? -> FVG -> entry

Variables to analyze on the 2343 trend-only trades:
  1. Direction (long vs short)
  2. NOT-MSS filter effect
  3. Regime/bias alignment
  4. Recent trend strength (how strong is the trend before FVG?)
  5. Pullback depth (how far did price pull back before FVG zone fill?)
  6. FVG position in daily range (premium vs discount)
  7. Time of day
  8. Displacement quality of the FVG
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.trend_fvg_research import run_trend_fvg_backtest
from experiments.chain_engine import load_all, compute_metrics, pr


def main():
    d = load_all()
    nq, n = d["nq"], d["n"]
    h, l, c, o = d["h"], d["l"], d["c"], d["o"]
    atr_arr = d["atr_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    on_hi, on_lo = d["on_hi"], d["on_lo"]
    et_frac_arr = d["et_frac_arr"]
    nq_idx = nq.index
    fvg_df = d["fvg_df"]

    # Run trend-only trades (excluding breakdown FVGs)
    trades = run_trend_fvg_backtest(d, include_breakdown_fvgs=False)
    print(f"Trend-only trades: {len(trades)}")

    # Tag each trade with contextual features
    for t in trades:
        idx = nq_idx.get_loc(t["entry_time"])
        atr = atr_arr[idx] if not np.isnan(atr_arr[idx]) else 30.0
        direction = t["dir"]

        # 1. Regime and bias at entry
        t["bias"] = bias_dir_arr[idx]
        t["regime"] = regime_arr[idx]
        t["bias_aligned"] = (direction == np.sign(bias_dir_arr[idx])) if bias_dir_arr[idx] != 0 else False

        # 2. Recent trend strength: count directional bars in last 10 bars
        lookback = 10
        start = max(0, idx - lookback)
        if direction == 1:
            bullish_bars = sum(1 for j in range(start, idx) if c[j] > o[j])
            t["trend_strength"] = bullish_bars / lookback
        else:
            bearish_bars = sum(1 for j in range(start, idx) if c[j] < o[j])
            t["trend_strength"] = bearish_bars / lookback

        # 3. Pullback depth: how far from recent high/low is the entry?
        recent_high = np.max(h[max(0, idx-20):idx+1])
        recent_low = np.min(l[max(0, idx-20):idx+1])
        recent_range = recent_high - recent_low
        if direction == 1 and recent_range > 0:
            t["pullback_pct"] = (recent_high - t["entry_price"]) / recent_range
        elif direction == -1 and recent_range > 0:
            t["pullback_pct"] = (t["entry_price"] - recent_low) / recent_range
        else:
            t["pullback_pct"] = 0.5

        # 4. FVG position relative to daily range (ON high/low)
        on_h = on_hi[idx]
        on_l = on_lo[idx]
        on_range = on_h - on_l if not (np.isnan(on_h) or np.isnan(on_l)) else 0
        if on_range > 0:
            t["on_position"] = (t["entry_price"] - on_l) / on_range  # 0=at ON low, 1=at ON high
        else:
            t["on_position"] = 0.5

        # 5. Is entry in discount (below 50%) or premium (above 50%)?
        t["in_discount"] = t["on_position"] < 0.5

        # 6. Time of day
        et = nq_idx[idx]
        if hasattr(et, "tz_convert"):
            et = et.tz_convert("US/Eastern")
        t["hour"] = et.hour + et.minute / 60.0
        t["is_am"] = 10.0 <= t["hour"] < 12.0

        # 7. Displacement quality (FVG creation candle body/ATR)
        # Approximate: look at the bar before entry bar's zone birth
        # Simpler: use stop distance as proxy for FVG quality
        t["stop_dist_atr"] = t["stop_dist_pts"] / atr if atr > 0 else 0

    df = pd.DataFrame(trades)
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year

    print("=" * 120)
    print("TREND FVG CHAIN DIAGNOSTIC")
    print("=" * 120)

    m_all = compute_metrics(trades)
    pr("ALL trend trades", m_all)

    # ================================================================
    # Split by each variable
    # ================================================================

    def split_show(title, col, bins=None, labels=None, min_n=30):
        print(f"\n  --- {title} ---")
        if bins:
            df["_grp"] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
        else:
            df["_grp"] = df[col]
        for grp, sub in df.groupby("_grp", observed=True):
            if len(sub) < min_n:
                continue
            m = compute_metrics(sub.to_dict("records"))
            tpd = m["trades"] / (252 * 10.5)
            print(f"  {str(grp):30s} | {m['trades']:5d}t | R={m['R']:+8.1f} | PF={m['PF']:5.2f} | "
                  f"PPDD={m['PPDD']:6.2f} | WR={m['WR']:5.1f}% | DD={m['MaxDD']:5.1f}R | {tpd:.2f}/d")

    # Direction
    split_show("Direction", "dir")

    # Bias aligned
    split_show("Bias aligned with direction", "bias_aligned")

    # Regime
    split_show("Regime", "regime",
        bins=[-0.01, 0.5, 1.0, 2.0, 99], labels=["low<0.5", "med 0.5-1", "high 1-2", "very high>2"])

    # Trend strength (directional bars in last 10)
    split_show("Recent trend strength", "trend_strength",
        bins=[-0.01, 0.3, 0.5, 0.7, 1.01], labels=["weak<30%", "mixed 30-50%", "strong 50-70%", "very strong>70%"])

    # Pullback depth
    split_show("Pullback depth (from recent extreme)", "pullback_pct",
        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["shallow<20%", "moderate 20-40%", "deep 40-60%", "very deep 60-80%", "extreme>80%"])

    # ON range position
    split_show("Position in ON range", "on_position",
        bins=[-99, 0, 0.3, 0.5, 0.7, 1.0, 99],
        labels=["below ON", "low 0-30%", "mid-low 30-50%", "mid-high 50-70%", "high 70-100%", "above ON"])

    # Premium vs Discount for longs
    longs = df[df["dir"] == 1]
    shorts = df[df["dir"] == -1]
    print(f"\n  --- Longs: Premium vs Discount ---")
    for label, sub in [("Discount (below 50% ON range)", longs[longs["in_discount"]]),
                        ("Premium (above 50% ON range)", longs[~longs["in_discount"]])]:
        if len(sub) < 30:
            continue
        m = compute_metrics(sub.to_dict("records"))
        print(f"  {label:45s} | {m['trades']:5d}t | PF={m['PF']:5.2f} | R={m['R']:+8.1f}")

    # Time of day
    split_show("Time of day", "hour",
        bins=[10, 11, 12, 13, 14, 15, 16], labels=["10-11", "11-12", "12-13", "13-14", "14-15", "15-16"])

    # AM vs PM
    split_show("AM (10-12) vs PM (12-16)", "is_am")

    # AM longs vs AM shorts
    print(f"\n  --- AM direction split ---")
    am = df[df["is_am"]]
    for label, sub in [("AM longs", am[am["dir"] == 1]), ("AM shorts", am[am["dir"] == -1])]:
        if len(sub) < 20:
            continue
        m = compute_metrics(sub.to_dict("records"))
        print(f"  {label:30s} | {m['trades']:5d}t | PF={m['PF']:5.2f} | R={m['R']:+8.1f}")

    # Stop distance / FVG size
    split_show("Stop dist (ATR)", "stop_dist_atr",
        bins=[0, 0.3, 0.5, 0.8, 1.5, 99], labels=["tiny<0.3", "small 0.3-0.5", "med 0.5-0.8", "large 0.8-1.5", "huge>1.5"])

    # ================================================================
    # Multi-factor: best combo
    # ================================================================
    print(f"\n{'='*120}")
    print("MULTI-FACTOR COMBOS")
    print(f"{'='*120}")

    combos = [
        ("Baseline", pd.Series(True, index=df.index)),
        ("Bias aligned only", df["bias_aligned"] == True),
        ("Regime >= 1.0", df["regime"] >= 1.0),
        ("Trend strength >= 0.5", df["trend_strength"] >= 0.5),
        ("Pullback 20-60%", (df["pullback_pct"] >= 0.2) & (df["pullback_pct"] <= 0.6)),
        ("Longs in discount", (df["dir"] == 1) & (df["in_discount"])),
        ("Shorts in premium", (df["dir"] == -1) & (~df["in_discount"])),
        ("Discount longs + premium shorts",
         ((df["dir"] == 1) & (df["in_discount"])) | ((df["dir"] == -1) & (~df["in_discount"]))),
        ("Bias + regime >= 1", (df["bias_aligned"]) & (df["regime"] >= 1.0)),
        ("Bias + trend >= 0.5", (df["bias_aligned"]) & (df["trend_strength"] >= 0.5)),
        ("Bias + discount/premium",
         (df["bias_aligned"]) & (((df["dir"] == 1) & (df["in_discount"])) | ((df["dir"] == -1) & (~df["in_discount"])))),
        ("Full: bias + regime + PD",
         (df["bias_aligned"]) & (df["regime"] >= 1.0) &
         (((df["dir"] == 1) & (df["in_discount"])) | ((df["dir"] == -1) & (~df["in_discount"])))),
    ]

    print(f"\n  {'Combo':50s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'PPDD':>7s} | {'DD':>6s} | {'/day':>5s}")
    print(f"  {'-'*50}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*5}")

    for label, mask in combos:
        sub = df[mask]
        if len(sub) < 30:
            print(f"  {label:50s} | {len(sub):5d} | too few")
            continue
        m = compute_metrics(sub.to_dict("records"))
        tpd = m["trades"] / (252 * 10.5)
        print(f"  {label:50s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['PPDD']:6.2f} | {m['MaxDD']:5.1f}R | {tpd:.2f}")


if __name__ == "__main__":
    main()
