"""
experiments/option_b_xgb_test.py -- Test XGBoost V3 on Option B configuration.

Option B: alt_dir < 0.334 (medium filter), ~675 trades pool
Goal: XGBoost selects top trades from pool, improve WR from ~41% to 50%+

Runs engine at threshold 0.0, 0.40, 0.45, 0.50, 0.55, 0.60
Compares: n, WR, PnL(flat), PPDD (PnL per peak drawdown), PF
"""

import sys
import logging
import time as _time
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
# Enable INFO for our pipeline only
logging.getLogger("models.train_xgb").setLevel(logging.INFO)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from models.train_xgb import build_feature_matrix_v2, train, _load_params, _SAVED_DIR
from models.evaluate import evaluate, print_report
from backtest.engine import run_backtest
from features.entry_signals import detect_all_signals
from features.bias import compute_daily_bias, compute_regime
from features.sessions import compute_session_levels, compute_orm, label_sessions

_DATA_DIR = _PROJECT_ROOT / "data"


def compute_ppdd(trades: pd.DataFrame) -> float:
    """PnL per Peak Drawdown (PPDD). Higher = better risk-adjusted return."""
    if trades.empty:
        return 0.0
    equity = trades["pnl_dollars"].cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = abs(dd.min())
    total_pnl = trades["pnl_dollars"].sum()
    if max_dd < 1.0:
        return total_pnl  # no drawdown
    return total_pnl / max_dd


def compute_pf(trades: pd.DataFrame) -> float:
    """Profit Factor."""
    if trades.empty:
        return 0.0
    gross_profit = trades.loc[trades["pnl_dollars"] > 0, "pnl_dollars"].sum()
    gross_loss = abs(trades.loc[trades["pnl_dollars"] <= 0, "pnl_dollars"].sum())
    if gross_loss < 1.0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def main():
    t_total = _time.perf_counter()

    # ---- Load and validate params ----
    params = _load_params()
    # Ensure Option B config
    params["pa_quality"]["alt_dir_threshold"] = 0.334
    params["grading"]["c_skip"] = False

    print("=" * 70)
    print("OPTION B: XGBoost V3 Threshold Test")
    print(f"  alt_dir_threshold = {params['pa_quality']['alt_dir_threshold']}")
    print(f"  c_skip = {params['grading']['c_skip']}")
    print("=" * 70)

    # ================================================================
    # STEP 1: Build feature matrix (V3 with PA features)
    # ================================================================
    print("\n[STEP 1] Building V3 feature matrix...")
    t0 = _time.perf_counter()
    X, y, signals_df = build_feature_matrix_v2(timeframe="5m")
    print(f"  Done in {_time.perf_counter() - t0:.1f}s")
    print(f"  X shape: {X.shape}")
    print(f"  Labels: 1={int(y.sum())}, 0={int((y == 0).sum())}")
    print(f"  Positive rate: {100 * y.mean():.2f}%")
    print(f"  Date range: {X.index.min()} to {X.index.max()}")

    # ================================================================
    # STEP 2: Train XGBoost V3
    # ================================================================
    print("\n[STEP 2] Training XGBoost V3...")
    t0 = _time.perf_counter()
    model = train(X, y)
    split = train._last_split
    print(f"  Done in {_time.perf_counter() - t0:.1f}s")

    # Quick model eval
    print("\n[STEP 2b] Model evaluation on val/test...")
    val_results = evaluate(model, split["X_val"], split["y_val"], save_plots=False)
    test_results = evaluate(model, split["X_test"], split["y_test"], save_plots=False)

    print("\n  Validation (2025-Q1):")
    print_report(val_results)
    print("\n  Test (2025-Q2+):")
    print_report(test_results)

    # ================================================================
    # STEP 3: Prepare full data for backtest
    # ================================================================
    print("\n[STEP 3] Loading full data for backtest...")
    t0 = _time.perf_counter()

    df_5m = pd.read_parquet(_DATA_DIR / "NQ_5m.parquet")
    df_1h = pd.read_parquet(_DATA_DIR / "NQ_1H.parquet")
    df_4h = pd.read_parquet(_DATA_DIR / "NQ_4H.parquet")

    # Compute bias + regime on full data
    sessions = label_sessions(df_5m, params)
    session_levels = compute_session_levels(df_5m, params)
    orm_data = compute_orm(df_5m, params)
    bias_data = compute_daily_bias(df_5m, session_levels, orm_data, df_4h, df_1h, params)
    regime = compute_regime(df_5m, df_4h, bias_data, params)

    print(f"  Done in {_time.perf_counter() - t0:.1f}s")
    print(f"  5m bars: {len(df_5m)} ({df_5m.index[0]} to {df_5m.index[-1]})")

    # ================================================================
    # STEP 4: Run backtest at multiple thresholds
    # ================================================================
    print("\n[STEP 4] Running backtests at multiple thresholds...")

    thresholds = [0.00, 0.40, 0.45, 0.50, 0.55, 0.60]
    results = {}

    for th in thresholds:
        t0 = _time.perf_counter()
        trades = run_backtest(
            df_5m, signals_df, bias_data, regime,
            model, X, params, threshold=th,
        )
        elapsed = _time.perf_counter() - t0

        if len(trades) > 0:
            n = len(trades)
            wr = (trades["pnl_dollars"] > 0).mean()
            total_pnl = trades["pnl_dollars"].sum()
            avg_r = trades["r_multiple"].mean()
            ppdd = compute_ppdd(trades)
            pf = compute_pf(trades)

            # Max drawdown
            equity = trades["pnl_dollars"].cumsum()
            max_dd = abs((equity - equity.cummax()).min())

            results[th] = {
                "n": n,
                "wr": wr,
                "pnl": total_pnl,
                "avg_r": avg_r,
                "ppdd": ppdd,
                "pf": pf,
                "max_dd": max_dd,
                "elapsed": elapsed,
                "trades": trades,
            }
            print(f"  threshold={th:.2f}: n={n:>4}, WR={100*wr:.1f}%, "
                  f"PnL=${total_pnl:>10,.0f}, PPDD={ppdd:.2f}, PF={pf:.2f} ({elapsed:.1f}s)")
        else:
            results[th] = {
                "n": 0, "wr": 0, "pnl": 0, "avg_r": 0,
                "ppdd": 0, "pf": 0, "max_dd": 0, "elapsed": elapsed,
                "trades": pd.DataFrame(),
            }
            print(f"  threshold={th:.2f}: 0 trades ({elapsed:.1f}s)")

    # ================================================================
    # FINAL REPORT
    # ================================================================
    print("\n" + "=" * 70)
    print("OPTION B RESULTS (alt<0.334)")
    print("=" * 70)
    print(f"\n{'threshold':>10} {'n':>6} {'WR%':>7} {'PnL':>12} {'MaxDD':>10} {'PPDD':>8} {'PF':>7} {'AvgR':>8}")
    print("-" * 75)

    best_ppdd = -float("inf")
    best_th = 0.0

    for th in thresholds:
        r = results[th]
        marker = ""
        if r["ppdd"] > best_ppdd and r["n"] >= 5:
            best_ppdd = r["ppdd"]
            best_th = th
        print(f"{th:>10.2f} {r['n']:>6} {100*r['wr']:>6.1f}% "
              f"${r['pnl']:>11,.0f} ${r['max_dd']:>9,.0f} "
              f"{r['ppdd']:>8.2f} {r['pf']:>7.2f} {r['avg_r']:>8.3f}")

    # Mark best
    print(f"\nBest threshold (by PPDD, n>=5): {best_th:.2f}")
    print(f"  PPDD={best_ppdd:.2f}")

    # ---- Detailed breakdown for threshold=0 (baseline) and best ----
    for label, th in [("BASELINE (no XGB filter)", 0.00), ("BEST THRESHOLD", best_th)]:
        r = results[th]
        trades = r["trades"]
        if trades.empty:
            continue

        print(f"\n--- {label} (threshold={th:.2f}) ---")
        print(f"  Trades: {r['n']}, WR: {100*r['wr']:.1f}%, PnL: ${r['pnl']:,.0f}")
        print(f"  Max DD: ${r['max_dd']:,.0f}, PPDD: {r['ppdd']:.2f}, PF: {r['pf']:.2f}")

        # By signal type
        by_type = trades.groupby("signal_type").agg(
            n=("pnl_dollars", "count"),
            wr=("pnl_dollars", lambda x: (x > 0).mean()),
            pnl=("pnl_dollars", "sum"),
        )
        print(f"\n  By signal type:")
        for st, row in by_type.iterrows():
            print(f"    {st}: n={int(row['n'])}, WR={100*row['wr']:.1f}%, PnL=${row['pnl']:,.0f}")

        # By direction
        by_dir = trades.groupby("direction").agg(
            n=("pnl_dollars", "count"),
            wr=("pnl_dollars", lambda x: (x > 0).mean()),
            pnl=("pnl_dollars", "sum"),
        )
        print(f"\n  By direction:")
        for d, row in by_dir.iterrows():
            d_str = "LONG" if d == 1 else "SHORT"
            print(f"    {d_str}: n={int(row['n'])}, WR={100*row['wr']:.1f}%, PnL=${row['pnl']:,.0f}")

        # By exit reason
        by_exit = trades["exit_reason"].value_counts()
        print(f"\n  By exit reason:")
        for reason, count in by_exit.items():
            pnl = trades.loc[trades["exit_reason"] == reason, "pnl_dollars"].sum()
            print(f"    {reason}: n={count}, PnL=${pnl:,.0f}")

        # Yearly breakdown
        trades_c = trades.copy()
        if trades_c["entry_time"].dt.tz is not None:
            trades_c["year"] = trades_c["entry_time"].dt.tz_convert("US/Eastern").dt.year
        else:
            trades_c["year"] = trades_c["entry_time"].dt.year
        by_year = trades_c.groupby("year").agg(
            n=("pnl_dollars", "count"),
            wr=("pnl_dollars", lambda x: (x > 0).mean()),
            pnl=("pnl_dollars", "sum"),
        )
        print(f"\n  By year:")
        for yr, row in by_year.iterrows():
            print(f"    {yr}: n={int(row['n'])}, WR={100*row['wr']:.1f}%, PnL=${row['pnl']:,.0f}")

    elapsed_total = _time.perf_counter() - t_total
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
