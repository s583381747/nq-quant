"""
experiments/sweep_fvg_mult.py — Sweep min_fvg_atr_mult to find optimal trade-count vs PPDD trade-off.

Uses the v4 cache (data/cache_signals_10yr_v4.parquet) which has ALL signals (mult=0.3).
For each mult value, filters signals post-hoc: skip where abs(entry-stop)/ATR < mult.
Runs full backtest engine with current filters (session skip, SQ 0.68/0.82, etc.).

IMPORTANT: The engine has its own `regime.min_stop_atr_mult` (default 1.5) which also
filters by stop_dist/ATR. To properly sweep, we set that to 0.0 and use our post-hoc
filter as the SOLE FVG size gate. This way the sweep actually covers the full range.

Also tests: different mult for TREND vs MSS signals (split sweep).

Usage: python experiments/sweep_fvg_mult.py
"""

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import logging
logging.basicConfig(
    level=logging.WARNING,  # Suppress INFO noise during sweep
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("sweep_fvg_mult")
logger.setLevel(logging.INFO)

DATA = PROJECT / "data"


# ============================================================
# UTILITIES
# ============================================================

def load_common_data():
    """Load all data needed for backtests (cached across runs)."""
    logger.info("Loading common data...")
    t0 = _time.perf_counter()
    nq = pd.read_parquet(DATA / "NQ_5m_10yr.parquet")
    es = pd.read_parquet(DATA / "ES_5m_10yr.parquet")
    bias = pd.read_parquet(DATA / "cache_bias_10yr_v2.parquet")
    regime = pd.read_parquet(DATA / "cache_regime_10yr_v2.parquet")
    sig_v4 = pd.read_parquet(DATA / "cache_signals_10yr_v4.parquet")
    atr_cache = pd.read_parquet(DATA / "cache_atr_flu_10yr_v2.parquet")

    with open(PROJECT / "config" / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # Compute SMT once
    from features.smt import compute_smt
    smt = compute_smt(nq, es, {
        "swing": {"left_bars": 3, "right_bars": 1},
        "smt": {"sweep_lookback": 15, "time_tolerance": 1},
    })

    elapsed = _time.perf_counter() - t0
    logger.info("Common data loaded in %.1fs", elapsed)
    return nq, es, bias, regime, sig_v4, atr_cache, params, smt


def apply_smt_gate(sig_cache: pd.DataFrame, smt: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply SMT gate for MSS signals + kill MSS overnight."""
    ss = sig_cache.copy()
    mm = ss["signal"].astype(bool) & (ss["signal_type"] == "mss")
    mi = ss.index[mm]
    ss.loc[mi, "signal"] = False
    ss.loc[mi, "signal_dir"] = 0
    ss["has_smt"] = False

    c = mi.intersection(smt.index)
    if len(c) > 0:
        md = sig_cache.loc[c, "signal_dir"].values
        ok = ((md == 1) & smt.loc[c, "smt_bull"].values.astype(bool)) | \
             ((md == -1) & smt.loc[c, "smt_bear"].values.astype(bool))
        g = c[ok]
        ss.loc[g, "signal"] = sig_cache.loc[g, "signal"]
        ss.loc[g, "signal_dir"] = sig_cache.loc[g, "signal_dir"]
        ss.loc[g, "has_smt"] = True

    # Kill MSS in overnight (16:00-03:00 ET)
    rem = ss["signal"].astype(bool) & (ss["signal_type"] == "mss")
    mi2 = ss.index[rem]
    if len(mi2) > 0:
        et = mi2.tz_convert("US/Eastern")
        ef = et.hour + et.minute / 60.0
        kill = (ef >= 16.0) | (ef < 3.0)
        if kill.any():
            ss.loc[mi2[kill], ["signal", "signal_dir"]] = [False, 0]

    return ss


def filter_by_fvg_mult(
    sig_df: pd.DataFrame,
    atr_series: pd.Series,
    mult_trend: float,
    mult_mss: float,
) -> pd.DataFrame:
    """Filter signals where stop_dist/ATR < mult. Separate mults for trend/mss."""
    out = sig_df.copy()
    mask = out["signal"].astype(bool)
    if not mask.any():
        return out

    sig_idx = out.index[mask]
    entry_p = out.loc[sig_idx, "entry_price"].values
    stop_p = out.loc[sig_idx, "model_stop"].values
    stop_dist = np.abs(entry_p - stop_p)
    atr_vals = atr_series.reindex(sig_idx).values
    ratio = np.where(atr_vals > 0, stop_dist / atr_vals, 0.0)

    sig_types = out.loc[sig_idx, "signal_type"].values
    is_mss = sig_types == "mss"

    # Apply different mults
    threshold = np.where(is_mss, mult_mss, mult_trend)
    kill = ratio < threshold

    kill_idx = sig_idx[kill]
    out.loc[kill_idx, "signal"] = False
    out.loc[kill_idx, "signal_dir"] = 0

    return out


def run_backtest_fast(
    nq: pd.DataFrame,
    sig_df: pd.DataFrame,
    bias: pd.DataFrame,
    regime: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Run backtest with given signal df and params.

    NOTE: We disable the engine's regime.min_stop_atr_mult (set to 0) so that
    the post-hoc FVG mult filter is the sole size gate. Otherwise the engine's
    built-in filter at 1.5 would dominate and make the sweep ineffective.
    """
    import copy
    p = copy.deepcopy(params)
    # Disable engine's own stop-size filter — our post-hoc filter replaces it
    p["regime"]["min_stop_atr_mult"] = 0.0

    class Dummy:
        def predict(self, d):
            return np.ones(d.num_row(), dtype=np.float32)

    dummy_X = pd.DataFrame({"d": np.zeros(len(nq))}, index=nq.index)

    from backtest.engine import run_backtest as _run_bt
    trades = _run_bt(nq, sig_df, bias, regime["regime"], Dummy(), dummy_X, p, threshold=0.0)
    return trades


def compute_metrics(trades: pd.DataFrame, label: str = "") -> dict:
    """Compute standard metrics from a trades DataFrame."""
    if len(trades) == 0:
        return {"label": label, "trades": 0, "R": 0.0, "PPDD": 0.0, "PF": 0.0,
                "WR": 0.0, "MaxDD": 0.0, "AvgR": 0.0}

    r_col = "r_multiple" if "r_multiple" in trades.columns else "r"
    r_vals = trades[r_col].values
    total_r = float(r_vals.sum())
    wr = float((r_vals > 0).mean() * 100)
    avg_r = float(r_vals.mean())

    # Max drawdown (in R)
    cumr = np.cumsum(r_vals)
    peak = np.maximum.accumulate(cumr)
    dd = peak - cumr
    max_dd = float(dd.max())

    # Profit factor
    wins = r_vals[r_vals > 0].sum()
    losses = abs(r_vals[r_vals < 0].sum())
    pf = float(wins / losses) if losses > 0 else 999.0

    # PPDD (peak profit / peak drawdown)
    ppdd = float(total_r / max_dd) if max_dd > 0 else 999.0

    return {
        "label": label,
        "trades": len(trades),
        "R": round(total_r, 1),
        "PPDD": round(ppdd, 2),
        "PF": round(pf, 2),
        "WR": round(wr, 1),
        "MaxDD": round(max_dd, 1),
        "AvgR": round(avg_r, 3),
    }


def print_table(rows: list[dict], title: str = ""):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 100}")
    if title:
        print(f"{title:^100}")
    print(f"{'=' * 100}")
    header = (f"{'Config':<30} {'Trades':>7} {'R':>8} {'PPDD':>7} {'PF':>6} "
              f"{'WR%':>6} {'MaxDD':>7} {'AvgR':>7}")
    print(header)
    print("-" * 100)
    for m in rows:
        row = (f"{m['label']:<30} {m['trades']:>7d} {m['R']:>8.1f} {m['PPDD']:>7.2f} "
               f"{m['PF']:>6.2f} {m['WR']:>5.1f}% {m['MaxDD']:>7.1f} {m['AvgR']:>7.3f}")
        print(row)
    print("=" * 100)


def walk_forward_validation(
    nq: pd.DataFrame,
    sig_filtered: pd.DataFrame,
    bias: pd.DataFrame,
    regime: pd.DataFrame,
    params: dict,
    oos_years: int = 1,
) -> list[dict]:
    """Expanding window walk-forward. Train=expanding, OOS=1 year."""
    et_index = nq.index.tz_convert("US/Eastern")
    years = sorted(set(et_index.year))

    # Need at least 2 years for train + OOS
    if len(years) < 3:
        return []

    results = []
    # Start OOS from year 3 (first 2 years = min training)
    for oos_start_idx in range(2, len(years)):
        oos_year = years[oos_start_idx]
        oos_start = pd.Timestamp(f"{oos_year}-01-01", tz="US/Eastern")
        oos_end = pd.Timestamp(f"{oos_year + oos_years}-01-01", tz="US/Eastern")

        if oos_end > et_index[-1]:
            oos_end = et_index[-1]

        # OOS mask
        oos_mask = (nq.index >= oos_start.tz_convert("UTC")) & (nq.index < oos_end.tz_convert("UTC"))
        if oos_mask.sum() == 0:
            continue

        # Create OOS-only signal df (blank signals outside OOS)
        sig_oos = sig_filtered.copy()
        sig_oos.loc[~oos_mask, "signal"] = False
        sig_oos.loc[~oos_mask, "signal_dir"] = 0

        trades = run_backtest_fast(nq, sig_oos, bias, regime, params)

        if len(trades) > 0:
            # Filter to OOS trades only
            oos_trades = trades[
                (pd.to_datetime(trades["entry_time"]) >= oos_start.tz_convert("UTC")) &
                (pd.to_datetime(trades["entry_time"]) < oos_end.tz_convert("UTC"))
            ]
        else:
            oos_trades = trades

        r_col = "r_multiple" if "r_multiple" in oos_trades.columns else "r"
        if len(oos_trades) > 0:
            r_vals = oos_trades[r_col].values
            wr = float((r_vals > 0).mean() * 100)
            total_r = float(r_vals.sum())
        else:
            wr = 0.0
            total_r = 0.0

        results.append({
            "oos_year": oos_year,
            "trades": len(oos_trades),
            "wr": round(wr, 1),
            "R": round(total_r, 1),
        })

    return results


# ============================================================
# SWEEP 1: Uniform mult for all signals
# ============================================================
def sweep_uniform(nq, bias, regime, sig_gated, atr_series, params):
    """Sweep min_fvg_atr_mult uniformly for all signal types."""
    logger.info("=== SWEEP 1: Uniform mult ===")
    mults = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0]
    results = []

    for mult in mults:
        t0 = _time.perf_counter()
        sig_f = filter_by_fvg_mult(sig_gated, atr_series, mult, mult)
        n_sig = int(sig_f["signal"].sum())

        trades = run_backtest_fast(nq, sig_f, bias, regime, params)
        m = compute_metrics(trades, f"mult={mult:.1f}")
        m["mult"] = mult
        m["raw_signals"] = n_sig
        results.append(m)

        elapsed = _time.perf_counter() - t0
        logger.info("  mult=%.1f: %d trades, R=%.1f, PPDD=%.2f (%.1fs)",
                     mult, m["trades"], m["R"], m["PPDD"], elapsed)

    return results


# ============================================================
# SWEEP 2: Split mult (different for trend vs MSS)
# ============================================================
def sweep_split(nq, bias, regime, sig_gated, atr_series, params):
    """Sweep different mults for trend vs MSS signals."""
    logger.info("=== SWEEP 2: Split mult (trend vs MSS) ===")

    # Sweep trend mult range that covers the interesting zone from uniform sweep
    # MSS uses IFVG which are generally smaller, so sweep lower range
    trend_mults = [0.6, 0.8, 1.0, 1.2, 1.5]
    mss_mults = [0.3, 0.5, 0.8, 1.0, 1.3]
    results = []

    for tm in trend_mults:
        for mm in mss_mults:
            t0 = _time.perf_counter()
            sig_f = filter_by_fvg_mult(sig_gated, atr_series, tm, mm)
            n_sig = int(sig_f["signal"].sum())

            trades = run_backtest_fast(nq, sig_f, bias, regime, params)
            m = compute_metrics(trades, f"T={tm:.1f}/M={mm:.1f}")
            m["trend_mult"] = tm
            m["mss_mult"] = mm
            m["raw_signals"] = n_sig
            results.append(m)

            elapsed = _time.perf_counter() - t0
            logger.info("  T=%.1f/M=%.1f: %d trades, R=%.1f, PPDD=%.2f (%.1fs)",
                         tm, mm, m["trades"], m["R"], m["PPDD"], elapsed)

    return results


# ============================================================
# WALK-FORWARD for top configs
# ============================================================
def run_wf_for_top_configs(top_configs, nq, bias, regime, sig_gated, atr_series, params):
    """Run walk-forward validation for top N configs."""
    logger.info("=== WALK-FORWARD VALIDATION ===")
    wf_results = []

    for cfg in top_configs:
        label = cfg["label"]
        # Parse mult from label
        if "/" in label:
            # Split: T=x/M=y
            parts = label.split("/")
            tm = float(parts[0].split("=")[1])
            mm = float(parts[1].split("=")[1])
        else:
            # Uniform: mult=x
            m_val = float(label.split("=")[1])
            tm = m_val
            mm = m_val

        t0 = _time.perf_counter()
        sig_f = filter_by_fvg_mult(sig_gated, atr_series, tm, mm)
        wf = walk_forward_validation(nq, sig_f, bias, regime, params)

        avg_wr = np.mean([w["wr"] for w in wf if w["trades"] > 0]) if wf else 0
        avg_r = np.mean([w["R"] for w in wf if w["trades"] > 0]) if wf else 0
        total_oos_trades = sum(w["trades"] for w in wf)
        pos_years = sum(1 for w in wf if w["R"] > 0)

        elapsed = _time.perf_counter() - t0
        logger.info("  %s: OOS trades=%d, avg WR=%.1f%%, avg R=%.1f, pos years=%d/%d (%.1fs)",
                     label, total_oos_trades, avg_wr, avg_r, pos_years, len(wf), elapsed)

        wf_results.append({
            "label": label,
            "oos_trades": total_oos_trades,
            "avg_wr": round(avg_wr, 1),
            "avg_yearly_r": round(avg_r, 1),
            "pos_years": pos_years,
            "total_years": len(wf),
            "yearly_detail": wf,
        })

    return wf_results


# ============================================================
# MAIN
# ============================================================
def main():
    t_total = _time.perf_counter()

    # Load data
    nq, es, bias, regime, sig_v4, atr_cache, params, smt = load_common_data()
    atr_series = atr_cache["atr"]

    # Apply SMT gate (shared across all sweeps)
    logger.info("Applying SMT gate...")
    sig_gated = apply_smt_gate(sig_v4, smt, params)
    n_gated = int(sig_gated["signal"].sum())
    logger.info("After SMT gate: %d signals", n_gated)

    # ========================================
    # SWEEP 1: Uniform mult
    # ========================================
    uniform_results = sweep_uniform(nq, bias, regime, sig_gated, atr_series, params)

    print_table(uniform_results, "SWEEP 1: UNIFORM min_fvg_atr_mult")

    # Find sweet spot: highest PPDD with trades > 700
    qualifying = [r for r in uniform_results if r["trades"] > 700]
    if qualifying:
        best_ppdd = max(qualifying, key=lambda x: x["PPDD"])
        print(f"\n>>> SWEET SPOT (trades > 700): mult={best_ppdd['mult']:.1f}, "
              f"trades={best_ppdd['trades']}, PPDD={best_ppdd['PPDD']:.2f}, "
              f"R={best_ppdd['R']:.1f}, PF={best_ppdd['PF']:.2f}")
    else:
        print("\n>>> No config with trades > 700 found.")

    # ========================================
    # SWEEP 2: Split mult (trend vs MSS)
    # ========================================
    split_results = sweep_split(nq, bias, regime, sig_gated, atr_series, params)

    # Sort by PPDD descending
    split_sorted = sorted(split_results, key=lambda x: x["PPDD"], reverse=True)

    # Print top 15
    print_table(split_sorted[:15], "SWEEP 2: SPLIT MULT (top 15 by PPDD)")

    # Find sweet spot for split: highest PPDD with trades > 600
    qualifying_split = [r for r in split_results if r["trades"] > 600]
    if qualifying_split:
        best_split = max(qualifying_split, key=lambda x: x["PPDD"])
        print(f"\n>>> SPLIT SWEET SPOT (trades > 600): T={best_split['trend_mult']:.1f}/"
              f"M={best_split['mss_mult']:.1f}, trades={best_split['trades']}, "
              f"PPDD={best_split['PPDD']:.2f}, R={best_split['R']:.1f}, PF={best_split['PF']:.2f}")

    # ========================================
    # WALK-FORWARD for top 3 configs by PPDD (trades > 600)
    # ========================================
    # Gather all results, filter trades > 600, sort by PPDD
    all_results = uniform_results + split_results
    qualifying_all = [r for r in all_results if r["trades"] > 600]
    # Deduplicate by label
    seen = set()
    unique_qualifying = []
    for r in sorted(qualifying_all, key=lambda x: x["PPDD"], reverse=True):
        if r["label"] not in seen:
            seen.add(r["label"])
            unique_qualifying.append(r)
    top3 = unique_qualifying[:3]

    if top3:
        print(f"\n{'=' * 100}")
        print(f"{'WALK-FORWARD VALIDATION — TOP 3 CONFIGS (trades > 600)':^100}")
        print(f"{'=' * 100}")
        for r in top3:
            print(f"  {r['label']}: trades={r['trades']}, PPDD={r['PPDD']:.2f}, R={r['R']:.1f}")

        wf_results = run_wf_for_top_configs(top3, nq, bias, regime, sig_gated, atr_series, params)

        print(f"\n{'=' * 100}")
        print(f"{'WALK-FORWARD RESULTS':^100}")
        print(f"{'=' * 100}")
        header = f"{'Config':<30} {'OOS Trades':>10} {'Avg WR%':>8} {'Avg Yearly R':>13} {'Pos/Total':>10}"
        print(header)
        print("-" * 100)
        for wf in wf_results:
            print(f"{wf['label']:<30} {wf['oos_trades']:>10d} {wf['avg_wr']:>7.1f}% "
                  f"{wf['avg_yearly_r']:>12.1f} {wf['pos_years']:>4d}/{wf['total_years']:<4d}")

        # Print yearly detail for each
        for wf in wf_results:
            print(f"\n  {wf['label']} — Year-by-year OOS:")
            for yr in wf["yearly_detail"]:
                print(f"    {yr['oos_year']}: trades={yr['trades']:>4d}, WR={yr['wr']:>5.1f}%, R={yr['R']:>+7.1f}")
        print("=" * 100)

    # ========================================
    # SUMMARY
    # ========================================
    elapsed = _time.perf_counter() - t_total
    print(f"\n{'=' * 100}")
    print(f"{'FINAL SUMMARY':^100}")
    print(f"{'=' * 100}")

    # Best uniform
    if qualifying:
        print(f"\nBest UNIFORM (trades > 700):")
        print(f"  mult={best_ppdd['mult']:.1f}: trades={best_ppdd['trades']}, "
              f"R={best_ppdd['R']:.1f}, PPDD={best_ppdd['PPDD']:.2f}, "
              f"PF={best_ppdd['PF']:.2f}, WR={best_ppdd['WR']:.1f}%, "
              f"MaxDD={best_ppdd['MaxDD']:.1f}")

    # Best split
    if qualifying_split:
        print(f"\nBest SPLIT (trades > 600):")
        print(f"  T={best_split['trend_mult']:.1f}/M={best_split['mss_mult']:.1f}: "
              f"trades={best_split['trades']}, R={best_split['R']:.1f}, "
              f"PPDD={best_split['PPDD']:.2f}, PF={best_split['PF']:.2f}, "
              f"WR={best_split['WR']:.1f}%, MaxDD={best_split['MaxDD']:.1f}")

    # ========================================
    # SWEEP 3: Refinement around top zone (1.4-2.0, step 0.1)
    # ========================================
    logger.info("=== SWEEP 3: Refinement 1.4-2.0 ===")
    fine_mults = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    fine_results = []
    for mult in fine_mults:
        t0 = _time.perf_counter()
        sig_f = filter_by_fvg_mult(sig_gated, atr_series, mult, mult)
        trades = run_backtest_fast(nq, sig_f, bias, regime, params)
        m = compute_metrics(trades, f"mult={mult:.1f}")
        m["mult"] = mult
        fine_results.append(m)
        elapsed_iter = _time.perf_counter() - t0
        logger.info("  mult=%.1f: %d trades, R=%.1f, PPDD=%.2f (%.1fs)",
                     mult, m["trades"], m["R"], m["PPDD"], elapsed_iter)
    print_table(fine_results, "SWEEP 3: REFINEMENT (1.4-2.0)")

    # ========================================
    # SWEEP 4: Split with higher trend mults
    # ========================================
    logger.info("=== SWEEP 4: Split with trend >= 1.5 ===")
    hi_trend = [1.5, 1.7, 2.0]
    lo_mss = [0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.7]
    split4_results = []
    for tm in hi_trend:
        for mm in lo_mss:
            t0 = _time.perf_counter()
            sig_f = filter_by_fvg_mult(sig_gated, atr_series, tm, mm)
            trades = run_backtest_fast(nq, sig_f, bias, regime, params)
            m = compute_metrics(trades, f"T={tm:.1f}/M={mm:.1f}")
            m["trend_mult"] = tm
            m["mss_mult"] = mm
            split4_results.append(m)
            elapsed_iter = _time.perf_counter() - t0
            logger.info("  T=%.1f/M=%.1f: %d trades, R=%.1f, PPDD=%.2f (%.1fs)",
                         tm, mm, m["trades"], m["R"], m["PPDD"], elapsed_iter)
    split4_sorted = sorted(split4_results, key=lambda x: x["PPDD"], reverse=True)
    print_table(split4_sorted, "SWEEP 4: HIGH-TREND SPLIT (sorted by PPDD)")

    # Find best split4 with trades > 600
    q_split4 = [r for r in split4_results if r["trades"] > 600]
    if q_split4:
        best4 = max(q_split4, key=lambda x: x["PPDD"])
        print(f"\n>>> SPLIT4 SWEET SPOT (trades > 600): T={best4['trend_mult']:.1f}/"
              f"M={best4['mss_mult']:.1f}, trades={best4['trades']}, "
              f"PPDD={best4['PPDD']:.2f}, R={best4['R']:.1f}")

    # Recommendation
    print(f"\nRecommendation: compare the best uniform vs best split above.")
    print(f"If PPDD is similar, prefer the simpler uniform config.")
    print(f"If split gives >10% PPDD lift, adopt separate mults.")

    elapsed = _time.perf_counter() - t_total
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
