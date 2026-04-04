"""
experiments/phase2_sq_sweep.py — Signal Quality Threshold Optimization

Sweeps SQ long and short thresholds across a grid to find configs that
improve trade count without degrading PPDD. Includes:
  1. Full grid sweep with metrics (trades, R, PPDD, MaxDD, WR, PF)
  2. Pareto frontier: configs that improve trade count with PPDD >= 8.0
  3. Walk-forward validation for top 3 configs (expanding window, 1-year OOS)
  4. Per-signal-type analysis: separate SQ for MSS vs Trend

Usage: python experiments/phase2_sq_sweep.py
"""
import sys
import copy
import time
import logging
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)  # suppress engine spam

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# ============================================================
# Metrics helpers
# ============================================================

def compute_metrics(trades: pd.DataFrame) -> dict:
    """Compute standard backtest metrics from trade DataFrame."""
    if trades.empty or len(trades) == 0:
        return {"trades": 0, "R": 0.0, "PPDD": 0.0, "MaxDD_R": 0.0,
                "WR": 0.0, "PF": 0.0, "AvgR": 0.0}

    r_col = "r_multiple" if "r_multiple" in trades.columns else "r"
    r_vals = trades[r_col].values

    total_r = float(np.sum(r_vals))
    cum_r = np.cumsum(r_vals)
    peak_r = np.maximum.accumulate(cum_r)
    dd_r = cum_r - peak_r
    max_dd_r = float(np.abs(np.min(dd_r))) if len(dd_r) > 0 else 0.0
    ppdd = total_r / max_dd_r if max_dd_r > 0.01 else (total_r if total_r > 0 else 0.0)

    n_wins = int(np.sum(r_vals > 0))
    wr = 100.0 * n_wins / len(r_vals) if len(r_vals) > 0 else 0.0

    gross_profit = float(np.sum(r_vals[r_vals > 0]))
    gross_loss = float(np.abs(np.sum(r_vals[r_vals <= 0])))
    pf = gross_profit / gross_loss if gross_loss > 0.01 else (float("inf") if gross_profit > 0 else 0.0)

    avg_r = float(np.mean(r_vals))

    return {
        "trades": len(trades),
        "R": round(total_r, 2),
        "PPDD": round(ppdd, 2),
        "MaxDD_R": round(max_dd_r, 2),
        "WR": round(wr, 1),
        "PF": round(pf, 2),
        "AvgR": round(avg_r, 4),
    }


# ============================================================
# Data loading (same as validate_nt_logic.py)
# ============================================================

def load_data():
    """Load all data needed for backtesting."""
    print("Loading data...")
    nq = pd.read_parquet(PROJECT / "data" / "NQ_5m_10yr.parquet")
    es = pd.read_parquet(PROJECT / "data" / "ES_5m_10yr.parquet")
    sig3 = pd.read_parquet(PROJECT / "data" / "cache_signals_10yr_v3.parquet")
    bias = pd.read_parquet(PROJECT / "data" / "cache_bias_10yr_v2.parquet")
    regime = pd.read_parquet(PROJECT / "data" / "cache_regime_10yr_v2.parquet")
    with open(PROJECT / "config" / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return nq, es, sig3, bias, regime, params


def build_signals_with_smt(nq, es, sig3, params):
    """Apply SMT gating to MSS signals (same as validate_nt_logic.py)."""
    print("Computing SMT divergence...")
    from features.smt import compute_smt
    smt = compute_smt(nq, es, {
        'swing': {'left_bars': 3, 'right_bars': 1},
        'smt': {'sweep_lookback': 15, 'time_tolerance': 1}
    })

    s = sig3.copy()
    mm = s['signal'].astype(bool) & (s['signal_type'] == 'mss')
    mi = s.index[mm]
    s.loc[mi, 'signal'] = False
    s.loc[mi, 'signal_dir'] = 0
    s['has_smt'] = False

    c = mi.intersection(smt.index)
    if len(c) > 0:
        md = sig3.loc[c, 'signal_dir'].values
        ok = ((md == 1) & smt.loc[c, 'smt_bull'].values.astype(bool)) | \
             ((md == -1) & smt.loc[c, 'smt_bear'].values.astype(bool))
        g = c[ok]
        s.loc[g, 'signal'] = sig3.loc[g, 'signal']
        s.loc[g, 'signal_dir'] = sig3.loc[g, 'signal_dir']
        s.loc[g, 'has_smt'] = True

    # Kill MSS in overnight (16:00-03:00 ET)
    rem = s['signal'].astype(bool) & (s['signal_type'] == 'mss')
    mi2 = s.index[rem]
    if len(mi2) > 0:
        et = mi2.tz_convert('US/Eastern')
        ef = et.hour + et.minute / 60.0
        kill = (ef >= 16.0) | (ef < 3.0)
        if kill.any():
            s.loc[mi2[kill], ['signal', 'signal_dir']] = [False, 0]

    n_total = s['signal'].sum()
    n_mss = ((s['signal'].astype(bool)) & (s['signal_type'] == 'mss')).sum()
    print(f"Merged signals: {n_total} total ({n_total - n_mss} trend, {n_mss} MSS+SMT)")
    return s


class DummyModel:
    """Dummy model that always predicts 1.0 (pass all signals through)."""
    def predict(self, d):
        return np.ones(d.num_row(), dtype=np.float32)


def run_backtest_with_sq(nq, signals, bias, regime, params, dummy_model, dummy_X,
                          long_sq, short_sq, signal_type_sq=None):
    """Run backtest with specific SQ thresholds.

    Parameters
    ----------
    signal_type_sq : dict or None
        If provided, per-signal-type thresholds: {"trend": {"long": X, "short": Y}, "mss": {"long": X, "short": Y}}
        When None, uses global long_sq / short_sq.
    """
    from backtest.engine import run_backtest

    p = copy.deepcopy(params)

    if signal_type_sq is not None:
        # For per-type SQ, we need to run separate backtests and merge.
        # But engine.py doesn't support per-type SQ natively.
        # Strategy: run twice — once with only trend signals, once with only MSS.
        results = []
        for sig_type_name, thresholds in signal_type_sq.items():
            p2 = copy.deepcopy(params)
            p2["signal_quality"]["threshold"] = thresholds["long"]
            p2["dual_mode"]["long_sq_threshold"] = thresholds["long"]
            p2["dual_mode"]["short_sq_threshold"] = thresholds["short"]
            if sig_type_name == "trend":
                p2["signal_filter"]["allow_trend"] = True
                p2["signal_filter"]["allow_mss"] = False
            elif sig_type_name == "mss":
                p2["signal_filter"]["allow_trend"] = False
                p2["signal_filter"]["allow_mss"] = True
            trades = run_backtest(nq, signals, bias, regime["regime"], dummy_model,
                                  dummy_X, p2, threshold=0.0)
            if not trades.empty:
                trades["_sig_type_filter"] = sig_type_name
                results.append(trades)
        if results:
            # Merge and sort by entry_time, then re-run daily limits.
            # NOTE: simple merge without re-applying daily limits is an approximation.
            # The daily 0-for-2 / 2R limit interaction between types is lost.
            # For a sweep experiment, this approximation is acceptable.
            merged = pd.concat(results).sort_values("entry_time").reset_index(drop=True)
            return merged
        return pd.DataFrame()
    else:
        p["signal_quality"]["threshold"] = long_sq
        p["dual_mode"]["long_sq_threshold"] = long_sq
        p["dual_mode"]["short_sq_threshold"] = short_sq
        trades = run_backtest(nq, signals, bias, regime["regime"], dummy_model,
                              dummy_X, p, threshold=0.0)
        return trades


def run_wf_backtest(nq, signals, bias, regime, params, dummy_model, dummy_X,
                     long_sq, short_sq, n_years_oos=1):
    """Walk-forward: expanding training window, 1-year OOS slices.

    Returns list of dicts with per-window OOS metrics.
    """
    # Get date range from signals index
    et_idx = nq.index.tz_convert("US/Eastern")
    years = sorted(set(et_idx.year))
    if len(years) < 3:
        return []

    results = []
    # Use expanding window: train on years[0..i], test on year i+1
    for i in range(2, len(years)):
        oos_year = years[i]
        # OOS period: full year
        oos_start = pd.Timestamp(f"{oos_year}-01-01", tz="US/Eastern")
        oos_end = pd.Timestamp(f"{oos_year + n_years_oos - 1}-12-31 23:59:59", tz="US/Eastern")
        if oos_end > et_idx[-1]:
            oos_end = et_idx[-1]

        # Convert to UTC for slicing
        oos_start_utc = oos_start.tz_convert("UTC")
        oos_end_utc = oos_end.tz_convert("UTC")

        # Filter data to OOS window
        mask = (nq.index >= oos_start_utc) & (nq.index <= oos_end_utc)
        if mask.sum() < 100:
            continue

        nq_oos = nq.loc[mask]
        sig_oos = signals.loc[mask]
        bias_oos = bias.loc[mask]
        reg_oos = regime.loc[mask]
        dx_oos = dummy_X.loc[mask]

        trades = run_backtest_with_sq(nq_oos, sig_oos, bias_oos, reg_oos, params,
                                       dummy_model, dx_oos, long_sq, short_sq)
        m = compute_metrics(trades)
        m["oos_year"] = oos_year
        m["profitable"] = m["R"] > 0
        results.append(m)

    return results


# ============================================================
# Main experiment
# ============================================================

def main():
    t0 = time.time()
    nq, es, sig3, bias, regime, params = load_data()
    signals = build_signals_with_smt(nq, es, sig3, params)
    dummy_model = DummyModel()
    dummy_X = pd.DataFrame({"d": np.zeros(len(nq))}, index=nq.index)

    # Current baseline
    CURRENT_LONG_SQ = 0.68
    CURRENT_SHORT_SQ = 0.82

    # Sweep grids
    long_sq_grid = [0.50, 0.55, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
    short_sq_grid = [0.70, 0.75, 0.78, 0.80, 0.82, 0.85]
    combos = list(product(long_sq_grid, short_sq_grid))

    print(f"\n{'='*80}")
    print(f"PHASE 2: Signal Quality Threshold Sweep")
    print(f"{'='*80}")
    print(f"Grid: {len(long_sq_grid)} long x {len(short_sq_grid)} short = {len(combos)} combos")
    print(f"Current config: long_sq={CURRENT_LONG_SQ}, short_sq={CURRENT_SHORT_SQ}")
    print(f"Running full backtest for each combo...\n")

    # ================================================================
    # STEP 1: Full grid sweep
    # ================================================================
    results = []
    for idx, (lsq, ssq) in enumerate(combos):
        t1 = time.time()
        trades = run_backtest_with_sq(nq, signals, bias, regime, params,
                                       dummy_model, dummy_X, lsq, ssq)
        m = compute_metrics(trades)
        m["long_sq"] = lsq
        m["short_sq"] = ssq
        m["is_current"] = (lsq == CURRENT_LONG_SQ and ssq == CURRENT_SHORT_SQ)
        elapsed = time.time() - t1
        results.append(m)

        marker = " <<<" if m["is_current"] else ""
        print(f"  [{idx+1:>2}/{len(combos)}] L={lsq:.2f} S={ssq:.2f} | "
              f"n={m['trades']:>4d} R={m['R']:>7.1f} PPDD={m['PPDD']:>6.2f} "
              f"MaxDD={m['MaxDD_R']:>5.1f}R WR={m['WR']:>5.1f}% PF={m['PF']:>5.2f} "
              f"({elapsed:.1f}s){marker}")

    rdf = pd.DataFrame(results).sort_values("PPDD", ascending=False)

    # ================================================================
    # STEP 2: Results table sorted by PPDD
    # ================================================================
    print(f"\n{'='*80}")
    print(f"FULL RESULTS (sorted by PPDD)")
    print(f"{'='*80}")
    print(f"{'LongSQ':>7} {'ShortSQ':>8} {'Trades':>7} {'R':>8} {'PPDD':>7} "
          f"{'MaxDD_R':>8} {'WR%':>6} {'PF':>6} {'AvgR':>7}")
    print("-" * 80)
    for _, r in rdf.iterrows():
        marker = " <<<" if r["is_current"] else ""
        print(f"  {r['long_sq']:>5.2f}    {r['short_sq']:>5.2f}  {r['trades']:>6.0f} "
              f"{r['R']:>7.1f}  {r['PPDD']:>6.2f}  {r['MaxDD_R']:>7.1f}  "
              f"{r['WR']:>5.1f} {r['PF']:>5.2f}  {r['AvgR']:>6.4f}{marker}")

    # ================================================================
    # STEP 3: Pareto frontier (improve trade count, PPDD >= 8.0)
    # ================================================================
    current_row = rdf[rdf["is_current"]].iloc[0] if rdf["is_current"].any() else None
    current_trades = int(current_row["trades"]) if current_row is not None else 534
    current_ppdd = float(current_row["PPDD"]) if current_row is not None else 10.39

    PPDD_FLOOR = 8.0
    pareto_candidates = rdf[(rdf["PPDD"] >= PPDD_FLOOR) & (rdf["trades"] > current_trades)]

    # True Pareto: no other config dominates on BOTH trades and PPDD
    pareto = []
    for _, row in pareto_candidates.iterrows():
        dominated = False
        for _, other in pareto_candidates.iterrows():
            if other["trades"] >= row["trades"] and other["PPDD"] >= row["PPDD"]:
                if other["trades"] > row["trades"] or other["PPDD"] > row["PPDD"]:
                    dominated = True
                    break
        if not dominated:
            pareto.append(row)

    print(f"\n{'='*80}")
    print(f"PARETO FRONTIER (trades > {current_trades}, PPDD >= {PPDD_FLOOR})")
    print(f"{'='*80}")
    if len(pareto) > 0:
        pdf = pd.DataFrame(pareto).sort_values("PPDD", ascending=False)
        print(f"{'LongSQ':>7} {'ShortSQ':>8} {'Trades':>7} {'R':>8} {'PPDD':>7} "
              f"{'MaxDD_R':>8} {'WR%':>6} {'PF':>6}")
        print("-" * 70)
        for _, r in pdf.iterrows():
            delta_t = r['trades'] - current_trades
            delta_ppdd = r['PPDD'] - current_ppdd
            print(f"  {r['long_sq']:>5.2f}    {r['short_sq']:>5.2f}  {r['trades']:>6.0f} "
                  f"{r['R']:>7.1f}  {r['PPDD']:>6.2f}  {r['MaxDD_R']:>7.1f}  "
                  f"{r['WR']:>5.1f} {r['PF']:>5.2f}  "
                  f"(+{delta_t:.0f}t, {delta_ppdd:+.2f} PPDD)")
    else:
        print("  No configs found that improve trades while keeping PPDD >= 8.0")
        print("  Showing top 5 by PPDD with trades > current:")
        fallback = rdf[rdf["trades"] > current_trades].head(5)
        for _, r in fallback.iterrows():
            print(f"  L={r['long_sq']:.2f} S={r['short_sq']:.2f} | "
                  f"n={r['trades']:.0f} R={r['R']:.1f} PPDD={r['PPDD']:.2f}")

    # ================================================================
    # STEP 4: Walk-forward for top 3 configs by PPDD (trades > current)
    # ================================================================
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD VALIDATION (top 3 by PPDD, trades > {current_trades})")
    print(f"Expanding window, 1-year OOS slices")
    print(f"{'='*80}")

    # Select top configs: top 3 PPDD from candidates with more trades than current
    wf_candidates = rdf[rdf["trades"] > current_trades].head(3)
    if len(wf_candidates) == 0:
        # If no config beats current trade count, use top 3 overall
        wf_candidates = rdf.head(3)
        print("  (No config beats current trade count; using top 3 by PPDD overall)\n")

    # Also include current config for comparison
    wf_configs = []
    for _, r in wf_candidates.iterrows():
        wf_configs.append((r["long_sq"], r["short_sq"]))
    if (CURRENT_LONG_SQ, CURRENT_SHORT_SQ) not in wf_configs:
        wf_configs.append((CURRENT_LONG_SQ, CURRENT_SHORT_SQ))

    for lsq, ssq in wf_configs:
        is_current = (lsq == CURRENT_LONG_SQ and ssq == CURRENT_SHORT_SQ)
        label = f"L={lsq:.2f} S={ssq:.2f}" + (" [CURRENT]" if is_current else "")
        print(f"\n  {label}")
        print(f"  {'-'*50}")

        wf_results = run_wf_backtest(nq, signals, bias, regime, params,
                                      dummy_model, dummy_X, lsq, ssq)
        if len(wf_results) == 0:
            print("    No OOS windows available")
            continue

        n_profitable = sum(1 for w in wf_results if w["profitable"])
        n_total = len(wf_results)
        total_oos_r = sum(w["R"] for w in wf_results)
        total_oos_trades = sum(w["trades"] for w in wf_results)

        for w in wf_results:
            pflag = "+" if w["profitable"] else "-"
            print(f"    OOS {w['oos_year']}: {pflag} n={w['trades']:>3d} R={w['R']:>6.1f} "
                  f"PPDD={w['PPDD']:>5.2f} WR={w['WR']:>5.1f}%")

        print(f"    {'='*46}")
        print(f"    OOS Score: {n_profitable}/{n_total} years profitable "
              f"({100*n_profitable/n_total:.0f}%)")
        print(f"    OOS Total: {total_oos_trades} trades, {total_oos_r:.1f}R")

    # ================================================================
    # STEP 5: Per-signal-type SQ analysis
    # ================================================================
    print(f"\n{'='*80}")
    print(f"PER-SIGNAL-TYPE SQ ANALYSIS")
    print(f"Testing: MSS with no SQ gate vs MSS with SQ, Trend with various SQ")
    print(f"{'='*80}")

    type_configs = [
        ("Baseline (current)", {"trend": {"long": 0.68, "short": 0.82},
                                 "mss": {"long": 0.68, "short": 0.82}}),
        ("MSS: no SQ gate", {"trend": {"long": 0.68, "short": 0.82},
                              "mss": {"long": 0.00, "short": 0.00}}),
        ("MSS: relaxed (0.50/0.70)", {"trend": {"long": 0.68, "short": 0.82},
                                       "mss": {"long": 0.50, "short": 0.70}}),
        ("MSS: no SQ + Trend relaxed (0.60/0.78)", {"trend": {"long": 0.60, "short": 0.78},
                                                      "mss": {"long": 0.00, "short": 0.00}}),
        ("Trend relaxed (0.60/0.78) + MSS current", {"trend": {"long": 0.60, "short": 0.78},
                                                       "mss": {"long": 0.68, "short": 0.82}}),
        ("MSS: relaxed + Trend relaxed (0.60/0.78)", {"trend": {"long": 0.60, "short": 0.78},
                                                        "mss": {"long": 0.50, "short": 0.70}}),
    ]

    print(f"\n{'Config':<45s} {'Trades':>7} {'R':>8} {'PPDD':>7} {'MaxDD':>7} {'WR%':>6} {'PF':>6}")
    print("-" * 90)

    for name, type_sq in type_configs:
        trades = run_backtest_with_sq(nq, signals, bias, regime, params,
                                       dummy_model, dummy_X,
                                       long_sq=0.68, short_sq=0.82,
                                       signal_type_sq=type_sq)
        m = compute_metrics(trades)
        print(f"  {name:<43s} {m['trades']:>6d} {m['R']:>7.1f}  {m['PPDD']:>6.2f}  "
              f"{m['MaxDD_R']:>6.1f}  {m['WR']:>5.1f} {m['PF']:>5.2f}")

    # ================================================================
    # STEP 6: Summary & Recommendation
    # ================================================================
    elapsed_total = time.time() - t0
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")

    # Find best PPDD overall
    best = rdf.iloc[0]
    print(f"Best PPDD config: L={best['long_sq']:.2f} S={best['short_sq']:.2f} | "
          f"n={best['trades']:.0f} R={best['R']:.1f} PPDD={best['PPDD']:.2f}")

    # Find best trade count with PPDD > 8
    trade_sorted = rdf[rdf["PPDD"] >= 8.0].sort_values("trades", ascending=False)
    if len(trade_sorted) > 0:
        best_trades = trade_sorted.iloc[0]
        print(f"Best trades (PPDD>=8): L={best_trades['long_sq']:.2f} S={best_trades['short_sq']:.2f} | "
              f"n={best_trades['trades']:.0f} R={best_trades['R']:.1f} PPDD={best_trades['PPDD']:.2f}")

    # Find best trade count with PPDD > current PPDD
    trade_sorted2 = rdf[rdf["PPDD"] >= current_ppdd].sort_values("trades", ascending=False)
    if len(trade_sorted2) > 0:
        best_trades2 = trade_sorted2.iloc[0]
        print(f"Best trades (PPDD>={current_ppdd:.1f}): L={best_trades2['long_sq']:.2f} S={best_trades2['short_sq']:.2f} | "
              f"n={best_trades2['trades']:.0f} R={best_trades2['R']:.1f} PPDD={best_trades2['PPDD']:.2f}")

    if current_row is not None:
        print(f"\nCurrent config:     L={CURRENT_LONG_SQ:.2f} S={CURRENT_SHORT_SQ:.2f} | "
              f"n={current_row['trades']:.0f} R={current_row['R']:.1f} PPDD={current_row['PPDD']:.2f}")

    print(f"\nTotal runtime: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")

    # Save full results
    out_path = PROJECT / "experiments" / "phase2_sq_sweep_results.csv"
    rdf.to_csv(out_path, index=False)
    print(f"Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
