"""
run_v2_pipeline.py -- End-to-end V2 pipeline:
  1. Build features + detect signals + compute bias
  2. Label signal bars (liquidity-based)
  3. Train XGBoost V2
  4. Backtest on test period (2025-Q2+)
  5. Generate report
"""

import logging
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.train_xgb import build_feature_matrix_v2, train, _load_params, _SAVED_DIR
from models.evaluate import evaluate, print_report
from backtest.engine import run_backtest
from backtest.report import generate_report
from features.entry_signals import detect_all_signals
from features.bias import compute_daily_bias, compute_regime
from features.sessions import compute_session_levels, compute_orm, label_sessions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_DATA_DIR = _PROJECT_ROOT / "data"


def main():
    t_total = _time.perf_counter()
    params = _load_params()

    # ================================================================
    # STEP 1: Build feature matrix (signals + bias + labels)
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 1: BUILD V2 FEATURE MATRIX")
    print("=" * 70)

    X, y, signals_df = build_feature_matrix_v2(timeframe="5m")

    print(f"\nV2 Feature Matrix:")
    print(f"  Shape: {X.shape}")
    print(f"  Labels: 1={int(y.sum())}, 0={int((y==0).sum())}")
    print(f"  Positive rate: {100*y.mean():.2f}%")
    print(f"  Date range: {X.index.min()} to {X.index.max()}")
    print(f"  Feature columns ({len(X.columns)}):")
    for i, col in enumerate(X.columns):
        print(f"    {i+1:3d}. {col}")

    # ================================================================
    # STEP 2: Train XGBoost V2
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 2: TRAIN XGBOOST V2")
    print("=" * 70)

    model = train(X, y)
    split = train._last_split

    # ================================================================
    # STEP 3: Evaluate model
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 70)

    print("\n--- Test Set (2025-Q2+) ---")
    test_results = evaluate(model, split["X_test"], split["y_test"])
    print_report(test_results)

    print("\n--- Validation Set (2025-Q1) ---")
    val_results = evaluate(model, split["X_val"], split["y_val"], save_plots=False)
    print_report(val_results)

    # ================================================================
    # STEP 4: Backtest on test period
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 4: BACKTEST ON TEST PERIOD (2025-Q2+)")
    print("=" * 70)

    # Load full 5m data and prepare for backtest
    logger.info("Loading 5m data for backtest...")
    df_5m = pd.read_parquet(_DATA_DIR / "NQ_5m.parquet")
    df_1h = pd.read_parquet(_DATA_DIR / "NQ_1H.parquet")
    df_4h = pd.read_parquet(_DATA_DIR / "NQ_4H.parquet")

    # Filter to test period: 2025-04-01 onwards
    # But we need signals/bias computed on the full dataset for continuity
    # signals_df is already on the full dataset from build_feature_matrix_v2
    # We need bias + regime on the full dataset too

    # Compute bias + regime on full 5m
    logger.info("Computing bias + regime for backtest period...")
    sessions = label_sessions(df_5m, params)
    session_levels = compute_session_levels(df_5m, params)
    orm_data = compute_orm(df_5m, params)

    bias_data = compute_daily_bias(df_5m, session_levels, orm_data, df_4h, df_1h, params)
    regime = compute_regime(df_5m, df_4h, bias_data, params)

    # Build full feature matrix for model predictions (need all bars)
    # We need features_X that covers the test period signal bars
    # Reuse features from build_feature_matrix_v2 -- they cover all bars before filtering
    # But build_feature_matrix_v2 only returns filtered signal bars in X
    # We need the full features for prediction. Let's build a lighter version.

    logger.info("Building feature matrix for prediction on test period...")
    # The simplest approach: re-extract features for signal bars in test period
    # from the X matrix (which has signal bars with valid labels)
    # For signal bars WITHOUT labels (e.g., NaN targets), we also need features

    # Actually, we already have X which has all labeled signal bars.
    # For backtest, we just need features for signal bars in test period.
    # Let's use X directly -- it covers signal bars with valid labels.

    # Slice to test period
    test_start = pd.Timestamp("2025-04-01", tz="UTC")
    test_df_5m = df_5m.loc[df_5m.index >= test_start].copy()
    test_signals = signals_df.loc[signals_df.index >= test_start].copy()
    test_bias = bias_data.loc[bias_data.index >= test_start].copy()
    test_regime = regime.loc[regime.index >= test_start].copy()

    # Features for test period signal bars (from X)
    test_X = X.loc[X.index >= test_start].copy()

    logger.info("Test period: %d bars, %d signals, %d labeled features",
                len(test_df_5m), int(test_signals["signal"].sum()), len(test_X))

    # Run backtest at multiple thresholds
    best_threshold = 0.50
    best_pnl = -float("inf")
    threshold_results = {}

    for threshold in [0.40, 0.45, 0.50, 0.55, 0.60]:
        logger.info("Running backtest at threshold %.2f...", threshold)
        trades = run_backtest(
            test_df_5m, test_signals, test_bias, test_regime,
            model, test_X, params, threshold=threshold,
        )
        if len(trades) > 0:
            total_pnl = trades["pnl_dollars"].sum()
            n_trades = len(trades)
            win_rate = (trades["pnl_dollars"] > 0).mean()
            avg_r = trades["r_multiple"].mean()
            threshold_results[threshold] = {
                "n_trades": n_trades,
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "avg_r": avg_r,
            }
            if total_pnl > best_pnl:
                best_pnl = total_pnl
                best_threshold = threshold

            print(f"\n  Threshold {threshold:.2f}: {n_trades} trades, "
                  f"WR={100*win_rate:.1f}%, Avg R={avg_r:.3f}, PnL=${total_pnl:,.2f}")
        else:
            print(f"\n  Threshold {threshold:.2f}: 0 trades")
            threshold_results[threshold] = {
                "n_trades": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "avg_r": 0,
            }

    # ================================================================
    # STEP 5: Generate report for best threshold
    # ================================================================
    print("\n" + "=" * 70)
    print(f"STEP 5: FINAL REPORT (threshold={best_threshold:.2f})")
    print("=" * 70)

    final_trades = run_backtest(
        test_df_5m, test_signals, test_bias, test_regime,
        model, test_X, params, threshold=best_threshold,
    )

    report = generate_report(final_trades, params, save_html=True)
    print(report)

    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed = _time.perf_counter() - t_total
    print("\n" + "=" * 70)
    print("FULL V2 PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Model saved: {_SAVED_DIR / 'xgb_v2.json'}")

    print("\nThreshold scan:")
    print(f"  {'Thresh':>7} {'Trades':>7} {'Win%':>7} {'Avg R':>8} {'PnL':>12}")
    print("  " + "-" * 48)
    for t, r in sorted(threshold_results.items()):
        marker = " <-- best" if t == best_threshold else ""
        print(f"  {t:>7.2f} {r['n_trades']:>7} {100*r['win_rate']:>6.1f}% "
              f"{r['avg_r']:>8.3f} ${r['total_pnl']:>11,.2f}{marker}")


if __name__ == "__main__":
    main()
