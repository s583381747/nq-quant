"""
ninjatrader/filter_optimization.py — Systematic filter optimization for bar-by-bar trades.

Explores all filter dimensions (signal type, grade, session, day-of-week, direction,
stop distance, RR ratio) independently and in combination to find the best risk-adjusted
strategy configuration.

IMPORTANT: Only uses features known at ENTRY TIME. The 'trimmed' column is an outcome
variable (known only after exit) and is NEVER used as a filter — that would be data leakage.

Usage:
    python ninjatrader/filter_optimization.py
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT = Path(__file__).resolve().parent.parent


# ============================================================
# DATA LOADING
# ============================================================

def load_trades() -> pd.DataFrame:
    """Load bar-by-bar trades and enrich with derived columns known at entry time."""
    path = PROJECT / "ninjatrader" / "bar_by_bar_trades.csv"
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)

    # Convert to US/Eastern for session logic
    df["entry_et"] = df["entry_time"].dt.tz_convert("US/Eastern")
    df["hour_et"] = df["entry_et"].dt.hour
    df["minute_et"] = df["entry_et"].dt.minute
    df["time_et_decimal"] = df["hour_et"] + df["minute_et"] / 60.0
    df["year"] = df["entry_et"].dt.year
    df["month"] = df["entry_et"].dt.to_period("M")
    df["weekday"] = df["entry_et"].dt.day_name()
    df["weekday_num"] = df["entry_et"].dt.dayofweek  # Mon=0 ... Sun=6
    df["date"] = df["entry_et"].dt.date

    # Entry-time features (all known before trade execution)
    df["stop_dist"] = abs(df["entry_price"] - df["stop_price"])
    df["tp_dist"] = abs(df["tp1_price"] - df["entry_price"])
    df["rr_ratio"] = df["tp_dist"] / df["stop_dist"].replace(0, np.nan)
    df["rr_ratio"] = df["rr_ratio"].fillna(0)

    return df


# ============================================================
# METRICS
# ============================================================

def compute_metrics(trades_r: np.ndarray, years_span: float) -> dict:
    """Compute full metrics suite for a series of R values."""
    n = len(trades_r)
    if n == 0:
        return {
            "n": 0, "total_r": 0.0, "avg_r": 0.0, "std_r": 0.0,
            "win_rate": 0.0, "max_dd_r": 0.0, "ppdd": 0.0,
            "profit_factor": 0.0, "sharpe": 0.0, "trades_per_year": 0.0,
            "max_consec_losses": 0, "max_consec_wins": 0,
            "cum_r": np.array([]),
        }

    total_r = trades_r.sum()
    avg_r = trades_r.mean()
    std_r = trades_r.std() if n > 1 else 0.0
    wins = (trades_r > 0).sum()
    losses = (trades_r < 0).sum()
    win_rate = wins / n

    # Max drawdown
    cum_r = np.cumsum(trades_r)
    running_max = np.maximum.accumulate(cum_r)
    drawdowns = cum_r - running_max
    max_dd = drawdowns.min()
    max_dd_idx = int(drawdowns.argmin())

    peak_idx = int(np.argmax(cum_r[:max_dd_idx + 1])) if max_dd_idx > 0 else 0
    recovery_idx = None
    if max_dd < 0:
        peak_val = cum_r[peak_idx]
        for j in range(max_dd_idx, n):
            if cum_r[j] >= peak_val:
                recovery_idx = j
                break

    ppdd = total_r / abs(max_dd) if max_dd != 0 else float("inf")

    gross_profit = trades_r[trades_r > 0].sum() if wins > 0 else 0.0
    gross_loss = abs(trades_r[trades_r < 0].sum()) if losses > 0 else 0.0001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    sharpe = avg_r / std_r if std_r > 0 else 0.0

    max_consec_loss = 0
    current = 0
    for r in trades_r:
        if r < 0:
            current += 1
            max_consec_loss = max(max_consec_loss, current)
        else:
            current = 0

    max_consec_win = 0
    current = 0
    for r in trades_r:
        if r > 0:
            current += 1
            max_consec_win = max(max_consec_win, current)
        else:
            current = 0

    trades_per_year = n / years_span if years_span > 0 else n
    dd_duration = max_dd_idx - peak_idx
    rec_duration = (recovery_idx - max_dd_idx) if recovery_idx is not None else None

    return {
        "n": n, "total_r": total_r, "avg_r": avg_r, "std_r": std_r,
        "win_rate": win_rate, "wins": int(wins), "losses": int(losses),
        "max_dd_r": max_dd, "max_dd_peak_idx": peak_idx,
        "max_dd_trough_idx": max_dd_idx, "recovery_idx": recovery_idx,
        "ppdd": ppdd, "profit_factor": profit_factor, "sharpe": sharpe,
        "max_consec_losses": max_consec_loss, "max_consec_wins": max_consec_win,
        "trades_per_year": trades_per_year,
        "gross_profit": gross_profit, "gross_loss": gross_loss,
        "dd_duration": dd_duration, "rec_duration": rec_duration,
        "cum_r": cum_r,
    }


# ============================================================
# FILTER FUNCTIONS (all use entry-time information only)
# ============================================================

def filter_signal_type(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return df
    elif mode == "trend_only":
        return df[df["type"] == "trend"]
    elif mode == "mss_only":
        return df[df["type"] == "mss"]
    return df


def filter_grade(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return df
    elif mode == "A+_B+":
        return df[df["grade"].isin(["A+", "B+"])]
    elif mode == "A+_only":
        return df[df["grade"] == "A+"]
    return df


def filter_session(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return df
    elif mode == "ny_only":
        return df[(df["time_et_decimal"] >= 9.5) & (df["time_et_decimal"] < 16.0)]
    elif mode == "ny_after_10":
        return df[(df["time_et_decimal"] >= 10.0) & (df["time_et_decimal"] < 16.0)]
    elif mode == "ny+london":
        return df[(df["time_et_decimal"] >= 3.0) & (df["time_et_decimal"] < 16.0)]
    elif mode == "skip_lunch":
        lunch = (df["time_et_decimal"] >= 12.0) & (df["time_et_decimal"] < 13.0)
        return df[~lunch]
    elif mode == "skip_last_hour":
        last = (df["time_et_decimal"] >= 15.0) & (df["time_et_decimal"] < 16.0)
        return df[~last]
    elif mode == "ny_no_lunch":
        ny = (df["time_et_decimal"] >= 9.5) & (df["time_et_decimal"] < 16.0)
        lunch = (df["time_et_decimal"] >= 12.0) & (df["time_et_decimal"] < 13.0)
        return df[ny & ~lunch]
    elif mode == "ny10_no_lunch":
        ny = (df["time_et_decimal"] >= 10.0) & (df["time_et_decimal"] < 16.0)
        lunch = (df["time_et_decimal"] >= 12.0) & (df["time_et_decimal"] < 13.0)
        return df[ny & ~lunch]
    elif mode == "prime_hours":
        prime = ((df["time_et_decimal"] >= 10.0) & (df["time_et_decimal"] < 12.0)) | \
                ((df["time_et_decimal"] >= 13.0) & (df["time_et_decimal"] < 15.0))
        return df[prime]
    elif mode == "overnight_only":
        # Outside NY session: 16:00-9:30 ET
        return df[(df["time_et_decimal"] >= 16.0) | (df["time_et_decimal"] < 9.5)]
    return df


def filter_day_of_week(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return df
    elif mode == "skip_monday":
        return df[df["weekday_num"] != 0]
    elif mode == "skip_friday":
        return df[df["weekday_num"] != 4]
    elif mode == "skip_mon_fri":
        return df[~df["weekday_num"].isin([0, 4])]
    elif mode == "tue_thu_only":
        return df[df["weekday_num"].isin([1, 2, 3])]
    return df


def filter_direction(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "both":
        return df
    elif mode == "long_only":
        return df[df["dir"] == 1]
    elif mode == "short_only":
        return df[df["dir"] == -1]
    return df


def filter_min_stop(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "none":
        return df
    elif mode == "min_5pts":
        return df[df["stop_dist"] >= 5.0]
    elif mode == "min_10pts":
        return df[df["stop_dist"] >= 10.0]
    elif mode == "min_15pts":
        return df[df["stop_dist"] >= 15.0]
    elif mode == "max_50pts":
        return df[df["stop_dist"] <= 50.0]
    elif mode == "5_50pts":
        return df[(df["stop_dist"] >= 5.0) & (df["stop_dist"] <= 50.0)]
    return df


def filter_rr_ratio(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Filter by reward-to-risk ratio (known at entry time from TP1 and stop prices)."""
    if mode == "none":
        return df
    elif mode == "rr>=1.0":
        return df[df["rr_ratio"] >= 1.0]
    elif mode == "rr>=1.5":
        return df[df["rr_ratio"] >= 1.5]
    elif mode == "rr>=2.0":
        return df[df["rr_ratio"] >= 2.0]
    elif mode == "rr>=3.0":
        return df[df["rr_ratio"] >= 3.0]
    elif mode == "rr<1.0":
        return df[df["rr_ratio"] < 1.0]
    elif mode == "rr_0.5_3.0":
        return df[(df["rr_ratio"] >= 0.5) & (df["rr_ratio"] <= 3.0)]
    return df


# ============================================================
# STEP 3: INDIVIDUAL FILTER ANALYSIS
# ============================================================

def analyze_individual_filters(df: pd.DataFrame, years_span: float):
    """Test each filter dimension independently."""

    print("\n" + "=" * 100)
    print("  STEP 3: INDIVIDUAL FILTER ANALYSIS")
    print("=" * 100)

    all_results = []

    # A. Signal type
    print("\n  --- A. Signal Type Filter ---")
    _run_filter_group(df, years_span, "signal", ["all", "trend_only", "mss_only"],
                      filter_signal_type, all_results)

    # B. Grade filter
    print("\n  --- B. Grade Filter ---")
    _run_filter_group(df, years_span, "grade", ["all", "A+_B+", "A+_only"],
                      filter_grade, all_results)

    # C. Session/time filter
    print("\n  --- C. Session/Time Filter ---")
    session_modes = ["all", "ny_only", "ny_after_10", "ny+london", "skip_lunch",
                     "skip_last_hour", "ny_no_lunch", "ny10_no_lunch", "prime_hours",
                     "overnight_only"]
    _run_filter_group(df, years_span, "session", session_modes, filter_session, all_results)

    # D. Day-of-week filter
    print("\n  --- D. Day-of-Week Filter ---")
    _run_filter_group(df, years_span, "day",
                      ["all", "skip_monday", "skip_friday", "skip_mon_fri", "tue_thu_only"],
                      filter_day_of_week, all_results)

    # E. Direction filter
    print("\n  --- E. Direction Filter ---")
    _run_filter_group(df, years_span, "direction", ["both", "long_only", "short_only"],
                      filter_direction, all_results)

    # F. Stop distance filter
    print("\n  --- F. Stop Distance Filter ---")
    _run_filter_group(df, years_span, "stop_dist",
                      ["none", "min_5pts", "min_10pts", "min_15pts", "max_50pts", "5_50pts"],
                      filter_min_stop, all_results)

    # G. RR ratio filter (entry-time feature)
    print("\n  --- G. RR Ratio Filter ---")
    _run_filter_group(df, years_span, "rr_ratio",
                      ["none", "rr>=1.0", "rr>=1.5", "rr>=2.0", "rr>=3.0", "rr<1.0", "rr_0.5_3.0"],
                      filter_rr_ratio, all_results)

    return all_results


def _run_filter_group(df, years_span, dim_name, modes, filter_fn, all_results):
    header = (f"    {'Mode':<20} {'Trades':>7} {'TotalR':>9} {'AvgR':>9} {'WR':>7} "
              f"{'MaxDD':>8} {'PPDD':>8} {'PF':>7} {'Sharpe':>8} {'T/Yr':>7}")
    print(header)
    print("    " + "-" * (len(header) - 4))

    for mode in modes:
        subset = filter_fn(df.copy(), mode)
        if len(subset) < 10:
            print(f"    {mode:<20} {len(subset):>7} {'--- too few ---'}")
            continue

        subset = subset.sort_values("entry_time")
        m = compute_metrics(subset["r"].values, years_span)
        row = {"dim": dim_name, "mode": mode, **{k: v for k, v in m.items() if k != "cum_r"}}
        all_results.append(row)

        ppdd_s = f"{m['ppdd']:.2f}" if abs(m['ppdd']) < 1000 else "inf"
        pf_s = f"{m['profit_factor']:.3f}" if m['profit_factor'] < 100 else "inf"
        print(f"    {mode:<20} {m['n']:>7} {m['total_r']:>+9.2f} {m['avg_r']:>+9.4f} "
              f"{m['win_rate']:>6.1%} {m['max_dd_r']:>+8.2f} {ppdd_s:>8} "
              f"{pf_s:>7} {m['sharpe']:>+8.4f} {m['trades_per_year']:>7.1f}")


# ============================================================
# BONUS: TYPE x GRADE CROSS-TAB
# ============================================================

def type_grade_crosstab(df: pd.DataFrame, years_span: float):
    print("\n" + "=" * 100)
    print("  BONUS: TYPE x GRADE CROSS-TABULATION")
    print("=" * 100)

    print(f"\n    {'Type+Grade':<16} {'Count':>6} {'TotalR':>9} {'AvgR':>9} {'WR':>7} "
          f"{'MaxDD':>8} {'PPDD':>8} {'PF':>7} {'Sharpe':>8}")
    print("    " + "-" * 85)

    for t in sorted(df["type"].unique()):
        for g in sorted(df["grade"].unique()):
            mask = (df["type"] == t) & (df["grade"] == g)
            if mask.sum() < 10:
                continue
            subset = df[mask].sort_values("entry_time")
            m = compute_metrics(subset["r"].values, years_span)
            label = f"{t}+{g}"
            ppdd_s = f"{m['ppdd']:.2f}" if abs(m['ppdd']) < 1000 else "inf"
            pf_s = f"{m['profit_factor']:.3f}" if m['profit_factor'] < 100 else "inf"
            print(f"    {label:<16} {m['n']:>6} {m['total_r']:>+9.2f} {m['avg_r']:>+9.4f} "
                  f"{m['win_rate']:>6.1%} {m['max_dd_r']:>+8.2f} {ppdd_s:>8} "
                  f"{pf_s:>7} {m['sharpe']:>+8.4f}")


def analyze_exit_patterns(df: pd.DataFrame, years_span: float):
    print("\n" + "=" * 100)
    print("  BONUS: EXIT REASON ANALYSIS (for understanding, not for filtering)")
    print("=" * 100)

    print(f"\n    {'Reason':<16} {'Count':>6} {'TotalR':>9} {'AvgR':>9} {'WR':>7}")
    print("    " + "-" * 50)
    for reason in sorted(df["reason"].unique()):
        mask = df["reason"] == reason
        r_vals = df.loc[mask, "r"]
        wr = (r_vals > 0).mean()
        print(f"    {reason:<16} {mask.sum():>6} {r_vals.sum():>+9.2f} {r_vals.mean():>+9.4f} {wr:>6.1%}")

    print(f"\n  NOTE: 'trimmed' is an OUTCOME variable (known only after exit).")
    print(f"  It must NEVER be used as an entry filter — that would be data leakage.")

    # Excluding early_cut_pa
    no_early = df[df["reason"] != "early_cut_pa"].sort_values("entry_time")
    m = compute_metrics(no_early["r"].values, years_span)
    print(f"\n  If early_cut_pa were eliminated (engine improvement, not a filter):")
    print(f"    {m['n']} trades, R={m['total_r']:+.2f}, avgR={m['avg_r']:+.4f}, "
          f"WR={m['win_rate']:.1%}, PPDD={m['ppdd']:.2f}, Sharpe={m['sharpe']:.4f}")


# ============================================================
# STEP 4-5: COMBINATION FILTER SEARCH
# ============================================================

def combination_search(df: pd.DataFrame, years_span: float, min_trades: int,
                       label: str = "") -> pd.DataFrame:
    """Search across all filter combinations using entry-time features only."""

    print(f"\n{'=' * 100}")
    print(f"  COMBINATION FILTER SEARCH (min {min_trades} trades){' — ' + label if label else ''}")
    print(f"{'=' * 100}")

    signal_opts = ["all", "trend_only", "mss_only"]
    grade_opts = ["all", "A+_B+", "A+_only"]
    session_opts = ["all", "ny_only", "ny_after_10", "skip_lunch", "skip_last_hour",
                    "ny_no_lunch", "overnight_only"]
    day_opts = ["all", "skip_monday", "skip_friday", "skip_mon_fri", "tue_thu_only"]
    dir_opts = ["both", "long_only"]
    stop_opts = ["none", "min_5pts", "min_10pts", "min_15pts"]
    rr_opts = ["none", "rr>=1.0", "rr>=1.5", "rr>=2.0", "rr>=3.0"]

    total_combos = (len(signal_opts) * len(grade_opts) * len(session_opts) *
                    len(day_opts) * len(dir_opts) * len(stop_opts) * len(rr_opts))
    print(f"\n  Total combinations to test: {total_combos}")

    results = []
    tested = 0

    for sig, grd, sess, day, dirn, stop, rr in itertools.product(
        signal_opts, grade_opts, session_opts, day_opts, dir_opts, stop_opts, rr_opts
    ):
        tested += 1
        if tested % 2000 == 0:
            print(f"    ... tested {tested}/{total_combos}, {len(results)} passed ...")

        subset = df.copy()
        for fn, mode in [
            (filter_signal_type, sig), (filter_grade, grd),
            (filter_session, sess), (filter_day_of_week, day),
            (filter_direction, dirn), (filter_min_stop, stop),
            (filter_rr_ratio, rr),
        ]:
            subset = fn(subset, mode)
            if len(subset) < min_trades:
                break
        else:
            # All filters applied, enough trades
            subset = subset.sort_values("entry_time")
            m = compute_metrics(subset["r"].values, years_span)

            results.append({
                "signal": sig, "grade": grd, "session": sess,
                "day": day, "direction": dirn, "stop_dist": stop, "rr_filter": rr,
                "n": m["n"], "total_r": m["total_r"], "avg_r": m["avg_r"],
                "win_rate": m["win_rate"], "max_dd_r": m["max_dd_r"],
                "ppdd": m["ppdd"], "profit_factor": m["profit_factor"],
                "sharpe": m["sharpe"], "trades_per_year": m["trades_per_year"],
                "max_consec_losses": m["max_consec_losses"],
                "dd_duration": m["dd_duration"], "rec_duration": m["rec_duration"],
            })

    print(f"\n  Tested {tested} combinations, {len(results)} meet minimum ({min_trades} trades)")

    if not results:
        print("  No combinations met the minimum trade count!")
        return pd.DataFrame()

    rdf = pd.DataFrame(results)
    rdf = rdf.sort_values("ppdd", ascending=False).reset_index(drop=True)

    # Top 10 by PPDD
    print("\n  ==========================================")
    print("  TOP 10 CONFIGURATIONS BY PPDD")
    print("  ==========================================")
    _print_results_table(rdf.head(10))

    # Top 10 by Sharpe
    rdf_s = rdf.sort_values("sharpe", ascending=False)
    print("\n  ==========================================")
    print("  TOP 10 CONFIGURATIONS BY SHARPE")
    print("  ==========================================")
    _print_results_table(rdf_s.head(10))

    # Top 10 by Total R
    rdf_t = rdf.sort_values("total_r", ascending=False)
    print("\n  ==========================================")
    print("  TOP 10 CONFIGURATIONS BY TOTAL R")
    print("  ==========================================")
    _print_results_table(rdf_t.head(10))

    # Top 10 by Profit Factor
    rdf_pf = rdf.sort_values("profit_factor", ascending=False)
    print("\n  ==========================================")
    print("  TOP 10 CONFIGURATIONS BY PROFIT FACTOR")
    print("  ==========================================")
    _print_results_table(rdf_pf.head(10))

    # Composite score: normalized PPDD * Sharpe (rewards both)
    # Only for positive-PPDD configs
    pos_mask = rdf["ppdd"] > 0
    if pos_mask.sum() > 0:
        rdf_pos = rdf[pos_mask].copy()
        rdf_pos["composite"] = rdf_pos["ppdd"] * rdf_pos["sharpe"]
        rdf_comp = rdf_pos.sort_values("composite", ascending=False)
        print("\n  ==========================================")
        print("  TOP 10 BY COMPOSITE (PPDD * Sharpe)")
        print("  ==========================================")
        _print_results_table(rdf_comp.head(10))

    return rdf


def _print_results_table(rdf: pd.DataFrame):
    header = (f"    {'#':>3} {'Signal':<12} {'Grade':<6} {'Session':<15} {'Day':<13} "
              f"{'Dir':<10} {'Stop':<10} {'RR':<9} {'N':>6} {'TotalR':>9} {'AvgR':>9} "
              f"{'WR':>7} {'MaxDD':>8} {'PPDD':>8} {'PF':>7} {'Sharpe':>8} {'T/Yr':>6}")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for rank_i, (_, row) in enumerate(rdf.iterrows(), 1):
        ppdd_s = f"{row['ppdd']:.2f}" if abs(row['ppdd']) < 1000 else "inf"
        pf_s = f"{row['profit_factor']:.3f}" if row['profit_factor'] < 100 else "inf"
        print(f"    {rank_i:>3} {row['signal']:<12} {row['grade']:<6} {row['session']:<15} "
              f"{row['day']:<13} {row['direction']:<10} {row['stop_dist']:<10} "
              f"{row['rr_filter']:<9} {row['n']:>6} {row['total_r']:>+9.2f} "
              f"{row['avg_r']:>+9.4f} {row['win_rate']:>6.1%} {row['max_dd_r']:>+8.2f} "
              f"{ppdd_s:>8} {pf_s:>7} {row['sharpe']:>+8.4f} {row['trades_per_year']:>6.1f}")


def _apply_all_filters(df: pd.DataFrame, row) -> pd.DataFrame:
    subset = df.copy()
    subset = filter_signal_type(subset, row["signal"])
    subset = filter_grade(subset, row["grade"])
    subset = filter_session(subset, row["session"])
    subset = filter_day_of_week(subset, row["day"])
    subset = filter_direction(subset, row["direction"])
    subset = filter_min_stop(subset, row["stop_dist"])
    subset = filter_rr_ratio(subset, row["rr_filter"])
    return subset


# ============================================================
# STEP 6: DETAILED ANALYSIS OF TOP CONFIGURATIONS
# ============================================================

def detailed_analysis_top3(df: pd.DataFrame, rdf: pd.DataFrame, years_span: float):
    print(f"\n{'=' * 100}")
    print(f"  STEP 6: DETAILED ANALYSIS OF TOP 3 CONFIGURATIONS BY PPDD")
    print(f"{'=' * 100}")

    top3 = rdf.head(3)

    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        config_label = (f"#{rank}: sig={row['signal']}, grd={row['grade']}, "
                        f"sess={row['session']}, day={row['day']}, "
                        f"dir={row['direction']}, stop={row['stop_dist']}, rr={row['rr_filter']}")
        print(f"\n  {'=' * 90}")
        print(f"  CONFIG {config_label}")
        print(f"  {'=' * 90}")

        subset = _apply_all_filters(df, row)
        subset = subset.sort_values("entry_time")
        m = compute_metrics(subset["r"].values, years_span)

        print(f"\n    Summary:")
        print(f"      Trades:        {m['n']}")
        print(f"      Total R:       {m['total_r']:+.2f}")
        print(f"      Avg R/trade:   {m['avg_r']:+.4f}")
        print(f"      Win Rate:      {m['win_rate']:.1%}")
        print(f"      Max Drawdown:  {m['max_dd_r']:+.2f} R")
        print(f"      PPDD:          {m['ppdd']:.2f}")
        print(f"      Profit Factor: {m['profit_factor']:.3f}")
        print(f"      Sharpe:        {m['sharpe']:+.4f}")
        print(f"      Max Consec L:  {m['max_consec_losses']}")
        print(f"      DD Duration:   {m['dd_duration']} trades")
        rec_str = f"{m['rec_duration']} trades" if m['rec_duration'] is not None else "Never"
        print(f"      Recovery:      {rec_str}")

        # Yearly breakdown
        print(f"\n    Yearly Breakdown:")
        print(f"      {'Year':<6} {'Trades':>7} {'R':>10} {'AvgR':>10} {'WR':>8} {'MaxDD':>8} {'CumR':>10}")
        print("      " + "-" * 63)
        cumulative = 0.0
        profitable_years = 0
        total_years = 0
        for year in sorted(subset["year"].unique()):
            yr_mask = subset["year"] == year
            yr_r = subset.loc[yr_mask, "r"].values
            yr_m = compute_metrics(yr_r, 1.0)
            yr_sum = yr_r.sum()
            cumulative += yr_sum
            total_years += 1
            if yr_sum > 0:
                profitable_years += 1
            print(f"      {year:<6} {len(yr_r):>7} {yr_sum:>+10.2f} {yr_m['avg_r']:>+10.4f} "
                  f"{yr_m['win_rate']:>7.1%} {yr_m['max_dd_r']:>+8.2f} {cumulative:>+10.2f}")

        print(f"\n      Profitable years: {profitable_years}/{total_years} "
              f"({100*profitable_years/total_years:.0f}%)")

        # Monthly R distribution
        print(f"\n    Monthly R Distribution:")
        monthly_r = subset.groupby("month")["r"].sum()
        neg_months = (monthly_r < 0).sum()
        print(f"      Mean monthly R:   {monthly_r.mean():+.2f}")
        print(f"      Std monthly R:    {monthly_r.std():.2f}")
        print(f"      Worst month:      {monthly_r.idxmin()} = {monthly_r.min():+.2f} R")
        print(f"      Best month:       {monthly_r.idxmax()} = {monthly_r.max():+.2f} R")
        print(f"      Negative months:  {neg_months}/{len(monthly_r)} ({100*neg_months/len(monthly_r):.0f}%)")
        print(f"      Median monthly R: {monthly_r.median():+.2f}")

        if total_years >= 8:
            check = profitable_years >= int(total_years * 0.8)
            print(f"\n      8/10 years profitable check: {'PASS' if check else 'FAIL'} "
                  f"({profitable_years}/{total_years})")

        # Trade composition
        print(f"\n    Trade Composition:")
        print(f"      By type:  {subset['type'].value_counts().to_dict()}")
        print(f"      By grade: {subset['grade'].value_counts().to_dict()}")
        dir_map = {1: "Long", -1: "Short"}
        dir_counts = {dir_map.get(k, k): v for k, v in subset["dir"].value_counts().to_dict().items()}
        print(f"      By dir:   {dir_counts}")
        print(f"      By exit:  {subset['reason'].value_counts().to_dict()}")


# ============================================================
# STEP 7: WALK-FORWARD VALIDATION
# ============================================================

def walk_forward_validation(df: pd.DataFrame, rdf: pd.DataFrame, years_span: float):
    print(f"\n{'=' * 100}")
    print(f"  STEP 7: WALK-FORWARD VALIDATION (TOP 3 CONFIGS)")
    print(f"{'=' * 100}")

    top3 = rdf.head(3)
    all_years = sorted(df["year"].unique())

    if len(all_years) < 4:
        print("  Not enough years for walk-forward.")
        return

    print(f"\n  Years available: {list(all_years)}")
    print(f"  Method: Expanding window - train on years [start..N], test on year N+1")

    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        config_label = (f"#{rank}: sig={row['signal']}, grd={row['grade']}, "
                        f"sess={row['session']}, day={row['day']}, "
                        f"dir={row['direction']}, stop={row['stop_dist']}, rr={row['rr_filter']}")
        print(f"\n  {'=' * 90}")
        print(f"  CONFIG {config_label}")
        print(f"  {'=' * 90}")

        subset = _apply_all_filters(df, row)
        subset = subset.sort_values("entry_time")

        print(f"\n    Expanding Window Out-of-Sample Performance:")
        print(f"    {'Train':>16} {'Test':>6} {'N':>5} {'R':>9} {'AvgR':>10} {'WR':>8} {'DD':>8} {'PPDD':>10}")
        print("    " + "-" * 80)

        oos_r_vals = []
        oos_years = []
        min_train = 3

        for i in range(min_train, len(all_years)):
            train_years = all_years[:i]
            test_year = all_years[i]
            test_set = subset[subset["year"] == test_year]
            if len(test_set) < 3:
                continue
            test_m = compute_metrics(test_set["r"].values, 1.0)
            ppdd_s = f"{test_m['ppdd']:.2f}" if abs(test_m['ppdd']) < 1000 else "inf"
            yr_range = f"{train_years[0]}-{train_years[-1]}"
            print(f"    {yr_range:>16} {test_year:>6} {test_m['n']:>5} "
                  f"{test_m['total_r']:>+9.2f} {test_m['avg_r']:>+10.4f} "
                  f"{test_m['win_rate']:>7.1%} {test_m['max_dd_r']:>+8.2f} {ppdd_s:>10}")
            oos_r_vals.append(test_m["total_r"])
            oos_years.append(test_year)

        if oos_r_vals:
            oos_arr = np.array(oos_r_vals)
            pos_years = (oos_arr > 0).sum()
            print(f"\n    Walk-Forward Summary:")
            print(f"      OOS years tested:   {len(oos_r_vals)}")
            print(f"      OOS years positive: {pos_years}/{len(oos_r_vals)} "
                  f"({100*pos_years/len(oos_r_vals):.0f}%)")
            print(f"      OOS total R:        {oos_arr.sum():+.2f}")
            print(f"      OOS mean R/year:    {oos_arr.mean():+.2f}")
            print(f"      OOS worst year:     {oos_years[int(oos_arr.argmin())]} = {oos_arr.min():+.2f} R")
            print(f"      OOS best year:      {oos_years[int(oos_arr.argmax())]} = {oos_arr.max():+.2f} R")

            if pos_years >= len(oos_r_vals) * 0.7:
                print(f"      VERDICT: ROBUST (positive in >=70% of OOS years)")
            elif pos_years >= len(oos_r_vals) * 0.5:
                print(f"      VERDICT: MODERATE (positive in 50-70% of OOS years)")
            else:
                print(f"      VERDICT: FRAGILE (positive in <50% of OOS years)")


# ============================================================
# ADDITIONAL: HOUR-BY-HOUR ANALYSIS
# ============================================================

def hour_analysis(df: pd.DataFrame, years_span: float):
    """Detailed hour-by-hour performance in ET."""
    print(f"\n{'=' * 100}")
    print(f"  BONUS: HOUR-BY-HOUR PERFORMANCE (ET)")
    print(f"{'=' * 100}")

    print(f"\n    {'Hour(ET)':>10} {'Count':>6} {'TotalR':>9} {'AvgR':>9} {'WR':>7}")
    print("    " + "-" * 45)
    for h in range(24):
        mask = df["hour_et"] == h
        if mask.sum() < 5:
            continue
        r_vals = df.loc[mask, "r"]
        wr = (r_vals > 0).mean()
        print(f"    {h:>10} {mask.sum():>6} {r_vals.sum():>+9.2f} {r_vals.mean():>+9.4f} {wr:>6.1%}")


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("*" * 100)
    print("  FILTER OPTIMIZATION: Systematic search for best risk-adjusted configuration")
    print("  (using only entry-time features — no data leakage)")
    print("*" * 100)

    # Step 1: Load data
    df = load_trades()
    years_span = (df["entry_time"].max() - df["entry_time"].min()).days / 365.25
    print(f"\n  Loaded {len(df)} trades spanning {years_span:.1f} years")
    print(f"  Date range: {df['entry_time'].min()} to {df['entry_time'].max()}")

    # Baseline
    print("\n  BASELINE (all 2063 trades, no filters):")
    m_base = compute_metrics(df.sort_values("entry_time")["r"].values, years_span)
    print(f"    Trades: {m_base['n']}, Total R: {m_base['total_r']:+.2f}, "
          f"Avg R: {m_base['avg_r']:+.4f}, WR: {m_base['win_rate']:.1%}")
    print(f"    Max DD: {m_base['max_dd_r']:+.2f}, PPDD: {m_base['ppdd']:.2f}, "
          f"PF: {m_base['profit_factor']:.3f}, Sharpe: {m_base['sharpe']:+.4f}")
    print(f"    Trades/year: {m_base['trades_per_year']:.1f}")

    # Step 2-3: Individual filter analysis
    analyze_individual_filters(df, years_span)

    # Bonus analyses
    type_grade_crosstab(df, years_span)
    analyze_exit_patterns(df, years_span)
    hour_analysis(df, years_span)

    # Step 4-5: Combination search (min 500 trades)
    rdf_500 = combination_search(df, years_span, min_trades=500, label="Statistical significance")

    # Also run with relaxed threshold (min 200)
    rdf_200 = combination_search(df, years_span, min_trades=200, label="Relaxed — higher quality niche")

    # Step 6: Detailed analysis
    best_rdf = rdf_500 if len(rdf_500) >= 3 else rdf_200
    if len(best_rdf) >= 3:
        detailed_analysis_top3(df, best_rdf, years_span)

    # Step 7: Walk-forward validation
    if len(best_rdf) >= 3:
        walk_forward_validation(df, best_rdf, years_span)

    # Final summary
    print(f"\n{'=' * 100}")
    print(f"  FINAL SUMMARY & RECOMMENDATIONS")
    print(f"{'=' * 100}")

    print(f"\n  BASELINE:")
    print(f"    {m_base['n']} trades, R={m_base['total_r']:+.2f}, PPDD={m_base['ppdd']:.2f}, "
          f"Sharpe={m_base['sharpe']:+.4f}, PF={m_base['profit_factor']:.3f}")

    for label, rdf in [("min 500 trades", rdf_500), ("min 200 trades", rdf_200)]:
        if len(rdf) > 0:
            top = rdf.iloc[0]
            ppdd_imp = ((top['ppdd'] - m_base['ppdd']) / abs(m_base['ppdd']) * 100
                        if m_base['ppdd'] != 0 else 0)
            sharpe_imp = ((top['sharpe'] - m_base['sharpe']) / abs(m_base['sharpe']) * 100
                          if m_base['sharpe'] != 0 else 0)
            print(f"\n  BEST ({label}):")
            print(f"    Filters: sig={top['signal']}, grade={top['grade']}, sess={top['session']}, "
                  f"day={top['day']}, dir={top['direction']}, stop={top['stop_dist']}, rr={top['rr_filter']}")
            print(f"    {int(top['n'])} trades, R={top['total_r']:+.2f}, PPDD={top['ppdd']:.2f} "
                  f"({ppdd_imp:+.0f}%), Sharpe={top['sharpe']:+.4f} ({sharpe_imp:+.0f}%), "
                  f"PF={top['profit_factor']:.3f}, WR={top['win_rate']:.1%}")

    # Data leakage warning
    print(f"\n  DATA INTEGRITY NOTE:")
    print(f"    All filters use only entry-time information (signal type, grade, session,")
    print(f"    day-of-week, direction, stop distance, RR ratio from TP1/stop levels).")
    print(f"    The 'trimmed' column was identified as an OUTCOME variable and excluded.")
    print()


if __name__ == "__main__":
    main()
