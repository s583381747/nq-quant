"""
ninjatrader/deep_analysis.py — Deep analysis: 2063 bar-by-bar trades vs 534 reference trades.

Answers the question: Are the extra ~1500 trades valuable?

Usage:
    python ninjatrader/deep_analysis.py
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent


def load_trades():
    """Load both trade sets."""
    eng_path = PROJECT / "ninjatrader" / "bar_by_bar_trades.csv"
    ref_path = PROJECT / "ninjatrader" / "python_trades_545.csv"

    eng = pd.read_csv(eng_path, parse_dates=["entry_time", "exit_time"])
    ref = pd.read_csv(ref_path, parse_dates=["entry_time", "exit_time"])

    # Ensure UTC
    eng["entry_time"] = pd.to_datetime(eng["entry_time"], utc=True)
    eng["exit_time"] = pd.to_datetime(eng["exit_time"], utc=True)
    ref["entry_time"] = pd.to_datetime(ref["entry_time"], utc=True)
    ref["exit_time"] = pd.to_datetime(ref["exit_time"], utc=True)

    # Derived columns
    for df in [eng, ref]:
        df["year"] = df["entry_time"].dt.year
        df["month"] = df["entry_time"].dt.to_period("M")
        df["weekday"] = df["entry_time"].dt.day_name()

    return eng, ref


# ============================================================
# METRICS FUNCTIONS
# ============================================================

def compute_metrics(trades_r: np.ndarray, label: str, years_span: float = 10.0) -> dict:
    """Compute full metrics suite for a series of R values."""
    n = len(trades_r)
    if n == 0:
        return {"label": label, "n": 0}

    total_r = trades_r.sum()
    avg_r = trades_r.mean()
    std_r = trades_r.std() if n > 1 else 0.0
    wins = (trades_r > 0).sum()
    losses = (trades_r < 0).sum()
    win_rate = wins / n if n > 0 else 0.0

    # Max drawdown in R
    cum_r = np.cumsum(trades_r)
    running_max = np.maximum.accumulate(cum_r)
    drawdowns = cum_r - running_max
    max_dd = drawdowns.min()  # most negative
    max_dd_idx = drawdowns.argmin()

    # DD start (peak before trough)
    peak_idx = np.argmax(cum_r[:max_dd_idx + 1]) if max_dd_idx > 0 else 0

    # Recovery: find when cum_r >= running_max[peak_idx] after trough
    recovery_idx = None
    if max_dd < 0:
        peak_val = cum_r[peak_idx]
        for j in range(max_dd_idx, n):
            if cum_r[j] >= peak_val:
                recovery_idx = j
                break

    ppdd = total_r / abs(max_dd) if max_dd != 0 else float("inf")

    # Profit factor
    gross_profit = trades_r[trades_r > 0].sum() if wins > 0 else 0.0
    gross_loss = abs(trades_r[trades_r < 0].sum()) if losses > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe-like (mean R / std R)
    sharpe = avg_r / std_r if std_r > 0 else 0.0

    # Max consecutive losses
    max_consec_loss = 0
    current_streak = 0
    for r in trades_r:
        if r < 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    # Max consecutive wins
    max_consec_win = 0
    current_streak = 0
    for r in trades_r:
        if r > 0:
            current_streak += 1
            max_consec_win = max(max_consec_win, current_streak)
        else:
            current_streak = 0

    trades_per_year = n / years_span if years_span > 0 else n

    return {
        "label": label,
        "n": n,
        "total_r": total_r,
        "avg_r": avg_r,
        "std_r": std_r,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "max_dd_r": max_dd,
        "max_dd_peak_idx": peak_idx,
        "max_dd_trough_idx": max_dd_idx,
        "recovery_idx": recovery_idx,
        "ppdd": ppdd,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_consec_losses": max_consec_loss,
        "max_consec_wins": max_consec_win,
        "trades_per_year": trades_per_year,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "cum_r": np.cumsum(trades_r),
    }


def print_metrics_comparison(m1: dict, m2: dict):
    """Print side-by-side comparison of two metric sets."""
    print(f"\n{'Metric':<28} {'Bar-by-Bar (2063)':>20} {'Reference (534)':>20} {'Delta':>12}")
    print("-" * 84)

    rows = [
        ("Trades", f"{m1['n']}", f"{m2['n']}", f"{m1['n'] - m2['n']:+d}"),
        ("Total R", f"{m1['total_r']:+.2f}", f"{m2['total_r']:+.2f}", f"{m1['total_r'] - m2['total_r']:+.2f}"),
        ("Avg R/trade", f"{m1['avg_r']:+.4f}", f"{m2['avg_r']:+.4f}", f"{m1['avg_r'] - m2['avg_r']:+.4f}"),
        ("Std R/trade", f"{m1['std_r']:.4f}", f"{m2['std_r']:.4f}", ""),
        ("Win Rate", f"{m1['win_rate']:.1%}", f"{m2['win_rate']:.1%}", f"{(m1['win_rate'] - m2['win_rate'])*100:+.1f}pp"),
        ("Wins / Losses", f"{m1['wins']}/{m1['losses']}", f"{m2['wins']}/{m2['losses']}", ""),
        ("Max Drawdown (R)", f"{m1['max_dd_r']:+.2f}", f"{m2['max_dd_r']:+.2f}", f"{m1['max_dd_r'] - m2['max_dd_r']:+.2f}"),
        ("PPDD (R/DD)", f"{m1['ppdd']:.2f}", f"{m2['ppdd']:.2f}", f"{m1['ppdd'] - m2['ppdd']:+.2f}"),
        ("Profit Factor", f"{m1['profit_factor']:.3f}", f"{m2['profit_factor']:.3f}", f"{m1['profit_factor'] - m2['profit_factor']:+.3f}"),
        ("Sharpe (R)", f"{m1['sharpe']:.4f}", f"{m2['sharpe']:.4f}", f"{m1['sharpe'] - m2['sharpe']:+.4f}"),
        ("Max Consec Losses", f"{m1['max_consec_losses']}", f"{m2['max_consec_losses']}", ""),
        ("Max Consec Wins", f"{m1['max_consec_wins']}", f"{m2['max_consec_wins']}", ""),
        ("Trades/Year", f"{m1['trades_per_year']:.1f}", f"{m2['trades_per_year']:.1f}", ""),
        ("Gross Profit (R)", f"{m1['gross_profit']:+.2f}", f"{m2['gross_profit']:+.2f}", ""),
        ("Gross Loss (R)", f"{m1['gross_loss']:.2f}", f"{m2['gross_loss']:.2f}", ""),
    ]

    for name, v1, v2, delta in rows:
        print(f"  {name:<26} {v1:>20} {v2:>20} {delta:>12}")


# ============================================================
# MATCHING
# ============================================================

def match_trades(eng: pd.DataFrame, ref: pd.DataFrame, tolerance_minutes: int = 6):
    """Match engine trades to reference trades by entry_time (+-tolerance) + same direction."""
    tolerance = timedelta(minutes=tolerance_minutes)
    matched_eng = set()
    matched_ref = set()
    matches = []

    for ri in ref.index:
        ref_t = ref.at[ri, "entry_time"]
        ref_dir = ref.at[ri, "dir"]

        best_ei = None
        best_dt = timedelta(days=999)

        for ei in eng.index:
            if ei in matched_eng:
                continue
            eng_t = eng.at[ei, "entry_time"]
            eng_dir = eng.at[ei, "dir"]

            if eng_dir != ref_dir:
                continue

            dt = abs(eng_t - ref_t)
            if dt <= tolerance and dt < best_dt:
                best_dt = dt
                best_ei = ei

        if best_ei is not None:
            matched_eng.add(best_ei)
            matched_ref.add(ri)
            matches.append((ri, best_ei, best_dt))

    group_a_idx = [ei for _, ei, _ in matches]
    group_b_idx = [i for i in eng.index if i not in matched_eng]

    return matches, group_a_idx, group_b_idx


# ============================================================
# SECTION 1: RISK-ADJUSTED COMPARISON
# ============================================================

def section_1(eng: pd.DataFrame, ref: pd.DataFrame):
    print("\n" + "=" * 84)
    print("SECTION 1: RISK-ADJUSTED COMPARISON")
    print("=" * 84)

    years_span = (eng["entry_time"].max() - eng["entry_time"].min()).days / 365.25

    m_eng = compute_metrics(eng["r"].values, "Bar-by-Bar", years_span)
    m_ref = compute_metrics(ref["r"].values, "Reference", years_span)

    print_metrics_comparison(m_eng, m_ref)

    return m_eng, m_ref


# ============================================================
# SECTION 2: ARE THE EXTRA TRADES VALUABLE?
# ============================================================

def section_2(eng: pd.DataFrame, ref: pd.DataFrame, matches, group_a_idx, group_b_idx):
    print("\n" + "=" * 84)
    print("SECTION 2: ARE THE EXTRA ~1500 TRADES VALUABLE?")
    print("=" * 84)

    group_a = eng.loc[group_a_idx].copy()
    group_b = eng.loc[group_b_idx].copy()

    print(f"\n  Group A (matched reference): {len(group_a)} trades")
    print(f"  Group B (engine-only extra):  {len(group_b)} trades")

    # -- Group A summary --
    if len(group_a) > 0:
        a_r = group_a["r"].values
        print(f"\n  --- Group A (matched) ---")
        print(f"    Total R: {a_r.sum():+.2f}")
        print(f"    Avg R:   {a_r.mean():+.4f}")
        print(f"    Win Rate: {(a_r > 0).mean():.1%}")

    # -- Group B overall --
    b_r = group_b["r"].values
    print(f"\n  --- Group B (extra trades) ---")
    print(f"    Total R:  {b_r.sum():+.2f}")
    print(f"    Avg R:    {b_r.mean():+.4f}")
    print(f"    Win Rate: {(b_r > 0).mean():.1%}")
    print(f"    Net: {'POSITIVE (valuable)' if b_r.sum() > 0 else 'NEGATIVE (harmful)'}")

    # -- Group B by year --
    print(f"\n  Group B breakdown by YEAR:")
    print(f"    {'Year':<6} {'Count':>6} {'Total R':>10} {'Avg R':>10} {'WinRate':>10}")
    print("    " + "-" * 46)
    for year in sorted(group_b["year"].unique()):
        mask = group_b["year"] == year
        yr_r = group_b.loc[mask, "r"]
        wr = (yr_r > 0).mean()
        print(f"    {year:<6} {mask.sum():>6} {yr_r.sum():>+10.2f} {yr_r.mean():>+10.4f} {wr:>10.1%}")

    # -- Group B by signal_type --
    print(f"\n  Group B breakdown by SIGNAL TYPE:")
    print(f"    {'Type':<8} {'Count':>6} {'Total R':>10} {'Avg R':>10} {'WinRate':>10}")
    print("    " + "-" * 48)
    for st in sorted(group_b["type"].unique()):
        mask = group_b["type"] == st
        st_r = group_b.loc[mask, "r"]
        wr = (st_r > 0).mean()
        print(f"    {st:<8} {mask.sum():>6} {st_r.sum():>+10.2f} {st_r.mean():>+10.4f} {wr:>10.1%}")

    # -- Group B by grade --
    print(f"\n  Group B breakdown by GRADE:")
    print(f"    {'Grade':<8} {'Count':>6} {'Total R':>10} {'Avg R':>10} {'WinRate':>10}")
    print("    " + "-" * 48)
    for g in sorted(group_b["grade"].unique()):
        mask = group_b["grade"] == g
        g_r = group_b.loc[mask, "r"]
        wr = (g_r > 0).mean()
        print(f"    {g:<8} {mask.sum():>6} {g_r.sum():>+10.2f} {g_r.mean():>+10.4f} {wr:>10.1%}")

    # -- Group B by exit_reason --
    print(f"\n  Group B breakdown by EXIT REASON:")
    print(f"    {'Reason':<16} {'Count':>6} {'Total R':>10} {'Avg R':>10} {'WinRate':>10}")
    print("    " + "-" * 56)
    for r in sorted(group_b["reason"].unique()):
        mask = group_b["reason"] == r
        r_r = group_b.loc[mask, "r"]
        wr = (r_r > 0).mean()
        print(f"    {r:<16} {mask.sum():>6} {r_r.sum():>+10.2f} {r_r.mean():>+10.4f} {wr:>10.1%}")

    # -- Consistency check: is Group B profitable in most years? --
    yearly_positive = 0
    yearly_total = 0
    for year in sorted(group_b["year"].unique()):
        mask = group_b["year"] == year
        yr_sum = group_b.loc[mask, "r"].sum()
        yearly_total += 1
        if yr_sum > 0:
            yearly_positive += 1

    print(f"\n  Consistency: Group B profitable in {yearly_positive}/{yearly_total} years ({100*yearly_positive/yearly_total:.0f}%)")

    return group_a, group_b


# ============================================================
# SECTION 3: FILTER IMPACT ANALYSIS
# ============================================================

def section_3(eng: pd.DataFrame, ref: pd.DataFrame):
    print("\n" + "=" * 84)
    print("SECTION 3: FILTER IMPACT ANALYSIS")
    print("=" * 84)

    # Compare filter pass rates implied by trade characteristics
    print("\n  Engine vs Reference trade composition:")

    # By type
    print(f"\n  Signal Type Distribution:")
    print(f"    {'Type':<8} {'Eng#':>6} {'Eng%':>8} {'Ref#':>6} {'Ref%':>8}")
    print("    " + "-" * 36)
    all_types = sorted(set(eng["type"].unique()) | set(ref["type"].unique()))
    for t in all_types:
        ec = (eng["type"] == t).sum()
        rc = (ref["type"] == t).sum()
        print(f"    {t:<8} {ec:>6} {100*ec/len(eng):>7.1f}% {rc:>6} {100*rc/len(ref):>7.1f}%")

    # By grade
    print(f"\n  Grade Distribution:")
    print(f"    {'Grade':<8} {'Eng#':>6} {'Eng%':>8} {'Ref#':>6} {'Ref%':>8}")
    print("    " + "-" * 36)
    all_grades = sorted(set(eng["grade"].unique()) | set(ref["grade"].unique()))
    for g in all_grades:
        ec = (eng["grade"] == g).sum()
        rc = (ref["grade"] == g).sum()
        print(f"    {g:<8} {ec:>6} {100*ec/len(eng):>7.1f}% {rc:>6} {100*rc/len(ref):>7.1f}%")

    # By direction
    print(f"\n  Direction Distribution:")
    for d, label in [(1, "Long"), (-1, "Short")]:
        ec = (eng["dir"] == d).sum()
        rc = (ref["dir"] == d).sum()
        print(f"    {label:<8} eng={ec:>5} ({100*ec/len(eng):.1f}%)  ref={rc:>4} ({100*rc/len(ref):.1f}%)")

    # The engine produces many more MSS signals (55% vs 23% in reference)
    # The reference is much heavier on trend signals (77% vs 44%)
    eng_mss_pct = (eng["type"] == "mss").sum() / len(eng) * 100
    ref_mss_pct = (ref["type"] == "mss").sum() / len(ref) * 100
    print(f"\n  KEY INSIGHT: Engine MSS share = {eng_mss_pct:.1f}%, Reference MSS share = {ref_mss_pct:.1f}%")
    print(f"  The bar-by-bar engine is much more aggressive at detecting MSS signals.")

    eng_c_pct = (eng["grade"] == "C").sum() / len(eng) * 100
    ref_c_pct = (ref["grade"] == "C").sum() / len(ref) * 100
    print(f"\n  Engine C-grade share = {eng_c_pct:.1f}%, Reference C-grade share = {ref_c_pct:.1f}%")
    # C-grade skip param
    print(f"  (C-grades may be filtered in reference; engine may not enforce c_skip consistently)")

    # Trades per session day analysis
    print(f"\n  Daily Trade Frequency Analysis:")
    eng_dates = eng["entry_time"].dt.date
    ref_dates = ref["entry_time"].dt.date
    eng_daily = eng_dates.value_counts()
    ref_daily = ref_dates.value_counts()
    print(f"    Engine: avg={eng_daily.mean():.2f} trades/day, max={eng_daily.max()}, "
          f"days with >2 trades: {(eng_daily > 2).sum()}")
    print(f"    Ref:    avg={ref_daily.mean():.2f} trades/day, max={ref_daily.max()}, "
          f"days with >2 trades: {(ref_daily > 2).sum()}")


# ============================================================
# SECTION 4: EQUITY CURVE COMPARISON
# ============================================================

def section_4(eng: pd.DataFrame, ref: pd.DataFrame, m_eng: dict, m_ref: dict):
    print("\n" + "=" * 84)
    print("SECTION 4: EQUITY CURVE COMPARISON")
    print("=" * 84)

    for label, df, m in [("Bar-by-Bar", eng, m_eng), ("Reference", ref, m_ref)]:
        cum_r = m["cum_r"]
        peak_idx = m["max_dd_peak_idx"]
        trough_idx = m["max_dd_trough_idx"]
        recovery_idx = m["recovery_idx"]

        peak_time = df["entry_time"].iloc[peak_idx] if peak_idx < len(df) else "N/A"
        trough_time = df["entry_time"].iloc[trough_idx] if trough_idx < len(df) else "N/A"
        rec_time = df["entry_time"].iloc[recovery_idx] if recovery_idx is not None and recovery_idx < len(df) else "Never"

        # DD duration in trades
        dd_trades = trough_idx - peak_idx
        rec_trades = (recovery_idx - trough_idx) if recovery_idx is not None else "N/A"

        print(f"\n  --- {label} ---")
        print(f"    Max Drawdown: {m['max_dd_r']:+.2f} R")
        print(f"    DD Peak:      trade #{peak_idx} ({str(peak_time)[:19]})")
        print(f"    DD Trough:    trade #{trough_idx} ({str(trough_time)[:19]})")
        print(f"    DD Duration:  {dd_trades} trades")
        print(f"    Recovery at:  trade #{recovery_idx if recovery_idx is not None else 'N/A'} ({str(rec_time)[:19]})")
        print(f"    Recovery len: {rec_trades} trades")

        # Per year
        print(f"\n    Yearly R:")
        print(f"    {'Year':<6} {'Trades':>7} {'R':>10} {'WinRate':>10} {'CumR':>10}")
        print("    " + "-" * 47)
        cumulative = 0.0
        for year in sorted(df["year"].unique()):
            mask = df["year"] == year
            yr_r = df.loc[mask, "r"]
            yr_sum = yr_r.sum()
            cumulative += yr_sum
            wr = (yr_r > 0).mean()
            print(f"    {year:<6} {mask.sum():>7} {yr_sum:>+10.2f} {wr:>10.1%} {cumulative:>+10.2f}")

    # Monthly R distribution
    print(f"\n  Monthly R Distribution:")
    print(f"    {'':>10} {'Eng Mean':>10} {'Eng Std':>10} {'Ref Mean':>10} {'Ref Std':>10}")
    eng_monthly = eng.groupby("month")["r"].sum()
    ref_monthly = ref.groupby("month")["r"].sum()
    print(f"    {'Monthly R':>10} {eng_monthly.mean():>+10.2f} {eng_monthly.std():>10.2f} "
          f"{ref_monthly.mean():>+10.2f} {ref_monthly.std():>10.2f}")

    # Worst/best months
    print(f"\n    Worst month (eng): {eng_monthly.idxmin()} = {eng_monthly.min():+.2f} R")
    print(f"    Best month  (eng): {eng_monthly.idxmax()} = {eng_monthly.max():+.2f} R")
    print(f"    Worst month (ref): {ref_monthly.idxmin()} = {ref_monthly.min():+.2f} R")
    print(f"    Best month  (ref): {ref_monthly.idxmax()} = {ref_monthly.max():+.2f} R")

    # Negative months
    eng_neg = (eng_monthly < 0).sum()
    ref_neg = (ref_monthly < 0).sum()
    print(f"\n    Negative months: eng={eng_neg}/{len(eng_monthly)} ({100*eng_neg/len(eng_monthly):.0f}%), "
          f"ref={ref_neg}/{len(ref_monthly)} ({100*ref_neg/len(ref_monthly):.0f}%)")


# ============================================================
# SECTION 5: COULD "MORE TRADES" ACTUALLY WORK?
# ============================================================

def section_5(eng: pd.DataFrame, ref: pd.DataFrame, group_b: pd.DataFrame, m_eng: dict, m_ref: dict):
    print("\n" + "=" * 84)
    print("SECTION 5: COULD 'MORE TRADES' ACTUALLY WORK?")
    print("=" * 84)

    years_span = (eng["entry_time"].max() - eng["entry_time"].min()).days / 365.25
    b_r = group_b["r"].values
    b_total = b_r.sum()
    b_avg = b_r.mean() if len(b_r) > 0 else 0.0

    print(f"\n  Extra trades (Group B): {len(group_b)} trades, total R = {b_total:+.2f}, avg R = {b_avg:+.4f}")
    print(f"  Reference:              {len(ref)} trades, total R = {m_ref['total_r']:+.2f}, avg R = {m_ref['avg_r']:+.4f}")

    # What if we keep ALL 2063?
    print(f"\n  --- Scenario: Keep ALL 2063 bar-by-bar trades ---")
    print(f"    Total R:          {m_eng['total_r']:+.2f}")
    print(f"    Max DD:           {m_eng['max_dd_r']:+.2f}")
    print(f"    PPDD:             {m_eng['ppdd']:.2f}")
    print(f"    Profit Factor:    {m_eng['profit_factor']:.3f}")
    print(f"    Sharpe:           {m_eng['sharpe']:.4f}")
    print(f"    Trades/year:      {m_eng['trades_per_year']:.1f}")

    print(f"\n  --- For comparison: Reference 534 trades ---")
    print(f"    Total R:          {m_ref['total_r']:+.2f}")
    print(f"    Max DD:           {m_ref['max_dd_r']:+.2f}")
    print(f"    PPDD:             {m_ref['ppdd']:.2f}")
    print(f"    Profit Factor:    {m_ref['profit_factor']:.3f}")
    print(f"    Sharpe:           {m_ref['sharpe']:.4f}")
    print(f"    Trades/year:      {m_ref['trades_per_year']:.1f}")

    # Is equity curve smoother?
    eng_cum = m_eng["cum_r"]
    ref_cum = m_ref["cum_r"]

    # Rolling 50-trade max DD for smoothness
    def rolling_max_dd(cum_r, window=50):
        """Max drawdown within rolling windows."""
        dds = []
        for i in range(window, len(cum_r)):
            segment = cum_r[i-window:i+1]
            seg_peak = np.maximum.accumulate(segment)
            seg_dd = (segment - seg_peak).min()
            dds.append(seg_dd)
        return np.array(dds)

    if len(eng_cum) >= 50:
        eng_rolling_dd = rolling_max_dd(eng_cum, 50)
        print(f"\n    Eng 50-trade rolling DD: mean={eng_rolling_dd.mean():.2f}, worst={eng_rolling_dd.min():.2f}")
    if len(ref_cum) >= 50:
        ref_rolling_dd = rolling_max_dd(ref_cum, 50)
        print(f"    Ref 50-trade rolling DD: mean={ref_rolling_dd.mean():.2f}, worst={ref_rolling_dd.min():.2f}")

    # What threshold of per-trade avg R makes "more trades" worth it?
    print(f"\n  --- Threshold Analysis: When is 'more trades' worth it? ---")
    print(f"    Baseline: {len(ref)} trades at avg R = {m_ref['avg_r']:+.4f} => {m_ref['total_r']:+.2f} R")
    print()
    print(f"    To match reference total R ({m_ref['total_r']:+.2f}) with {len(eng)} trades:")
    needed_avg = m_ref["total_r"] / len(eng)
    print(f"      Need avg R >= {needed_avg:+.4f} per trade (currently {m_eng['avg_r']:+.4f})")
    print(f"      Status: {'ACHIEVED' if m_eng['avg_r'] >= needed_avg else 'NOT MET'}")

    # Break-even analysis for extra trades
    print(f"\n    For Group B extra trades to be worth adding:")
    print(f"      They need avg R > 0 (currently {b_avg:+.4f})")
    print(f"      Status: {'POSITIVE (worth keeping)' if b_avg > 0 else 'NEGATIVE (harmful, should be filtered)'}")

    # What if we filter Group B to only keep certain signal types / grades?
    print(f"\n  --- Group B Filtered Sub-strategies ---")
    filters = [
        ("B only: trend", group_b["type"] == "trend"),
        ("B only: mss", group_b["type"] == "mss"),
        ("B only: A+ grade", group_b["grade"] == "A+"),
        ("B only: B+ grade", group_b["grade"] == "B+"),
        ("B only: C grade", group_b["grade"] == "C"),
        ("B only: A+ or B+", group_b["grade"].isin(["A+", "B+"])),
        ("B only: trend+A+/B+", (group_b["type"] == "trend") & group_b["grade"].isin(["A+", "B+"])),
        ("B only: mss+A+/B+", (group_b["type"] == "mss") & group_b["grade"].isin(["A+", "B+"])),
    ]

    print(f"    {'Filter':<22} {'Count':>6} {'Total R':>10} {'Avg R':>10} {'WR':>8} {'PF':>8}")
    print("    " + "-" * 68)
    for name, mask in filters:
        subset = group_b.loc[mask]
        if len(subset) == 0:
            print(f"    {name:<22} {'---':>6}")
            continue
        sr = subset["r"].values
        total = sr.sum()
        avg = sr.mean()
        wr = (sr > 0).mean()
        gp = sr[sr > 0].sum() if (sr > 0).any() else 0
        gl = abs(sr[sr < 0].sum()) if (sr < 0).any() else 0
        pf = gp / gl if gl > 0 else float("inf")
        print(f"    {name:<22} {len(subset):>6} {total:>+10.2f} {avg:>+10.4f} {wr:>7.1%} {pf:>8.3f}")

    # Hybrid strategy: Ref + best of Group B
    print(f"\n  --- Hybrid Strategy Ideas ---")
    # What if we combine reference trades with the best Group B subset?
    for name, mask in filters:
        subset = group_b.loc[mask]
        if len(subset) == 0:
            continue
        sr = subset["r"].values
        if sr.sum() > 0:
            # Merge with reference (simplified: just add R values)
            combined_r = np.concatenate([ref["r"].values, sr])
            c_total = combined_r.sum()
            c_avg = combined_r.mean()
            c_cum = np.cumsum(combined_r)
            c_peak = np.maximum.accumulate(c_cum)
            c_dd = (c_cum - c_peak).min()
            c_ppdd = c_total / abs(c_dd) if c_dd < 0 else float("inf")
            gp = combined_r[combined_r > 0].sum()
            gl = abs(combined_r[combined_r < 0].sum())
            c_pf = gp / gl if gl > 0 else float("inf")
            print(f"    Ref + {name[8:]:<16}: {len(combined_r):>5} trades, "
                  f"R={c_total:>+8.2f}, PPDD={c_ppdd:.2f}, PF={c_pf:.3f}")

    # What R/trade threshold for "more trades" viability
    print(f"\n  --- Viability Thresholds ---")
    print(f"    Commission + slippage cost per round-trip in R terms:")
    # Rough estimate: commission ~$1.24 round-trip micro, slippage ~1 tick = $0.50
    # At $1000 risk (1R), that's ~0.002R per trade
    # But also consider opportunity cost of being in a losing trade
    print(f"      ~0.002R per trade (commission+slippage for micro)")
    print(f"      Current extra trades avg R: {b_avg:+.4f}")
    print(f"      After costs: ~{b_avg - 0.002:+.4f} per trade")
    print(f"      Breakeven point: avg R > ~0.002 per trade")
    print(f"      Current status: {'VIABLE' if b_avg > 0.002 else 'MARGINAL/NOT VIABLE'}")

    # At what per-trade avg R does PPDD of 2063 trades beat PPDD of 534?
    # PPDD = totalR / |maxDD|
    # For 2063 trades: PPDD = 2063 * avgR / |maxDD|
    # For 534 trades: PPDD = 156.6 / |refMaxDD|
    ref_ppdd = m_ref["ppdd"]
    eng_max_dd = abs(m_eng["max_dd_r"])
    # Need: 2063 * x / eng_max_dd > ref_ppdd
    # x > ref_ppdd * eng_max_dd / 2063
    needed_avg_for_ppdd = ref_ppdd * eng_max_dd / len(eng) if len(eng) > 0 else 0
    print(f"\n    To beat reference PPDD ({ref_ppdd:.2f}):")
    print(f"      Need avg R > {needed_avg_for_ppdd:+.4f} per trade (at current DD)")
    print(f"      Current: {m_eng['avg_r']:+.4f}")
    print(f"      Verdict: {'PPDD IS BETTER' if m_eng['ppdd'] > ref_ppdd else 'PPDD IS WORSE'}")


# ============================================================
# SECTION 6: DIRECTION + SESSION + WEEKDAY ANALYSIS
# ============================================================

def section_6(eng: pd.DataFrame, group_b: pd.DataFrame):
    print("\n" + "=" * 84)
    print("SECTION 6: ADDITIONAL BREAKDOWNS (GROUP B)")
    print("=" * 84)

    # By direction
    print(f"\n  Group B by DIRECTION:")
    for d, label in [(1, "Long"), (-1, "Short")]:
        mask = group_b["dir"] == d
        if mask.sum() == 0:
            continue
        dr = group_b.loc[mask, "r"]
        print(f"    {label:<6}: {mask.sum():>5} trades, R={dr.sum():>+8.2f}, "
              f"avg={dr.mean():>+.4f}, WR={((dr > 0).mean()):>.1%}")

    # By weekday
    print(f"\n  Group B by WEEKDAY:")
    print(f"    {'Day':<12} {'Count':>6} {'Total R':>10} {'Avg R':>10} {'WR':>8}")
    print("    " + "-" * 50)
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        mask = group_b["weekday"] == day
        if mask.sum() == 0:
            continue
        dr = group_b.loc[mask, "r"]
        wr = (dr > 0).mean()
        print(f"    {day:<12} {mask.sum():>6} {dr.sum():>+10.2f} {dr.mean():>+10.4f} {wr:>7.1%}")

    # By hour of day (ET)
    print(f"\n  Group B by HOUR (UTC):")
    eng_hour = group_b["entry_time"].dt.hour
    print(f"    {'Hour':>6} {'Count':>6} {'Total R':>10} {'Avg R':>10} {'WR':>8}")
    print("    " + "-" * 44)
    for h in sorted(eng_hour.unique()):
        mask = eng_hour == h
        dr = group_b.loc[mask, "r"]
        wr = (dr > 0).mean()
        print(f"    {h:>6} {mask.sum():>6} {dr.sum():>+10.2f} {dr.mean():>+10.4f} {wr:>7.1%}")

    # trimmed vs not trimmed
    if "trimmed" in group_b.columns:
        print(f"\n  Group B by TRIMMED status:")
        for t_val in [True, False]:
            mask = group_b["trimmed"] == t_val
            if mask.sum() == 0:
                continue
            dr = group_b.loc[mask, "r"]
            print(f"    trimmed={t_val}: {mask.sum():>5} trades, R={dr.sum():>+8.2f}, avg={dr.mean():>+.4f}")


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("*" * 84)
    print("  DEEP ANALYSIS: 2063 BAR-BY-BAR TRADES vs 534 REFERENCE TRADES")
    print("*" * 84)

    eng, ref = load_trades()
    print(f"\n  Loaded: {len(eng)} engine trades, {len(ref)} reference trades")

    # Section 1: Risk-adjusted comparison
    m_eng, m_ref = section_1(eng, ref)

    # Match trades
    print("\n  Matching trades (+-6min + same direction)...")
    matches, group_a_idx, group_b_idx = match_trades(eng, ref)
    print(f"  Matched: {len(matches)}, Group A (matched): {len(group_a_idx)}, Group B (extra): {len(group_b_idx)}")

    group_b = eng.loc[group_b_idx].copy()

    # Section 2: Are extra trades valuable?
    group_a, group_b = section_2(eng, ref, matches, group_a_idx, group_b_idx)

    # Section 3: Filter impact
    section_3(eng, ref)

    # Section 4: Equity curve comparison
    section_4(eng, ref, m_eng, m_ref)

    # Section 5: Could more trades work?
    section_5(eng, ref, group_b, m_eng, m_ref)

    # Section 6: Additional breakdowns
    section_6(eng, group_b)

    # ========== FINAL VERDICT ==========
    print("\n" + "=" * 84)
    print("  FINAL VERDICT")
    print("=" * 84)

    b_r = group_b["r"].values
    b_total = b_r.sum()
    b_avg = b_r.mean()
    eng_total = m_eng["total_r"]
    ref_total = m_ref["total_r"]

    print(f"""
  The bar-by-bar engine produces {len(eng)} trades vs {len(ref)} reference trades.
  That's {len(eng) - len(ref):+d} extra trades ({100*(len(eng) - len(ref))/len(ref):.0f}% more).

  KEY NUMBERS:
    Total R:        Engine={eng_total:+.2f}  vs  Ref={ref_total:+.2f}  (delta={eng_total-ref_total:+.2f})
    Avg R/trade:    Engine={m_eng['avg_r']:+.4f}  vs  Ref={m_ref['avg_r']:+.4f}
    Max Drawdown:   Engine={m_eng['max_dd_r']:+.2f}  vs  Ref={m_ref['max_dd_r']:+.2f}
    PPDD:           Engine={m_eng['ppdd']:.2f}  vs  Ref={m_ref['ppdd']:.2f}
    Profit Factor:  Engine={m_eng['profit_factor']:.3f}  vs  Ref={m_ref['profit_factor']:.3f}
    Sharpe:         Engine={m_eng['sharpe']:.4f}  vs  Ref={m_ref['sharpe']:.4f}

  EXTRA TRADES (Group B: {len(group_b)} trades):
    Total R: {b_total:+.2f}, Avg R: {b_avg:+.4f}
    Are they net positive? {'YES' if b_total > 0 else 'NO'}

  CONCLUSION:""")

    if m_eng["ppdd"] > m_ref["ppdd"] and b_total > 0:
        print(f"    The extra trades ARE valuable. More trades with lower per-trade edge")
        print(f"    produces BETTER risk-adjusted returns (higher PPDD).")
        print(f"    The equity curve is smoother due to more frequent positive trades.")
    elif b_total > 0 and m_eng["ppdd"] < m_ref["ppdd"]:
        print(f"    The extra trades have positive R but WORSE PPDD. The larger")
        print(f"    drawdown from additional trades is not fully compensated by extra R.")
        print(f"    Consider filtering Group B to keep only the best subsets.")
    elif b_total <= 0:
        print(f"    The extra trades are NET NEGATIVE. They dilute the edge and")
        print(f"    increase drawdown. The bar-by-bar engine's filters need tightening.")
    else:
        print(f"    Mixed results. Further investigation needed.")

    print()


if __name__ == "__main__":
    main()
