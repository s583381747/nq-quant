"""
models/train_xgb.py -- XGBoost V2 training pipeline for NQ quantitative system.

V2 changes vs V1:
  - Entry candidates = signal bars only (from entry_signals.detect_all_signals)
    NOT every NY bar. This drops candidates from ~41K to ~2-5K quality setups.
  - New features: bias_direction, bias_confidence, bias_aligned, regime,
    signal_type (0=trend 1=mss), setup_rr from entry_signals.
  - Labels: liquidity-based labeling using signal_dir and model_stop from
    entry_signals (not candle direction).
  - All existing features (displacement, fluency, swings, sessions, FVG, HTF,
    time, price) are preserved.

References: CLAUDE.md sections 3, 4, 14, 17, 19
"""

from __future__ import annotations

import logging
import sys
import time as _time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from features.displacement import (
    compute_atr,
    compute_fluency,
    detect_bad_candles,
    detect_displacement,
)
from features.fvg import compute_active_fvgs, detect_fvg
from features.sessions import compute_orm, compute_session_levels, label_sessions
from features.swing import compute_swing_levels
from features.labeler import label_liquidity_based
from features.entry_signals import detect_all_signals
from features.bias import compute_daily_bias, compute_regime

logger = logging.getLogger(__name__)

_CONFIG_PATH = _PROJECT_ROOT / "config" / "params.yaml"
_DATA_DIR = _PROJECT_ROOT / "data"
_SAVED_DIR = _PROJECT_ROOT / "models" / "saved"


def _load_params(config_path: Path = _CONFIG_PATH) -> dict:
    """Load tunable parameters from params.yaml."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# build_feature_matrix_v2 -- signal-bar-only pipeline
# ---------------------------------------------------------------------------

def build_feature_matrix_v2(
    timeframe: str = "5m",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build V2 feature matrix: only signal bars are candidates.

    Steps:
      1. Load data (5m + HTF)
      2. Detect entry signals (Trend + MSS)
      3. Compute bias + regime
      4. Build feature columns on ALL bars
      5. Label signal bars using liquidity-based labeler with signal-derived
         direction, model_stop, irl_target
      6. Filter to signal bars only, return (X, y, signals_df)

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.DataFrame]
        (X, y, signals_df) where:
          X = feature matrix (only signal bars)
          y = binary labels
          signals_df = full signal info for all bars (for backtest)
    """
    params = _load_params()
    t0 = _time.perf_counter()

    # ---- 1. Load data ----
    tf_file_map = {
        "1m": "NQ_1min.parquet",
        "5m": "NQ_5m.parquet",
        "15m": "NQ_15m.parquet",
        "1H": "NQ_1H.parquet",
        "4H": "NQ_4H.parquet",
        "1D": "NQ_1D.parquet",
    }
    logger.info("Loading %s data...", timeframe)
    df = pd.read_parquet(_DATA_DIR / tf_file_map[timeframe])
    logger.info("Loaded %d bars (%s to %s)", len(df), df.index[0], df.index[-1])

    logger.info("Loading HTF data (1H, 4H)...")
    df_1h = pd.read_parquet(_DATA_DIR / "NQ_1H.parquet")
    df_4h = pd.read_parquet(_DATA_DIR / "NQ_4H.parquet")

    # ---- 2. Detect entry signals ----
    logger.info("Detecting entry signals (Trend + MSS)...")
    t_sig = _time.perf_counter()
    signals_df = detect_all_signals(df, params)
    logger.info("Signal detection done in %.1fs", _time.perf_counter() - t_sig)

    signal_mask = signals_df["signal"].values.astype(bool)
    n_signals = int(signal_mask.sum())
    logger.info("Signal bars: %d / %d total (%.2f%%)",
                n_signals, len(df), 100.0 * n_signals / len(df))

    # ---- 3. Compute bias + regime ----
    logger.info("Computing session levels and ORM...")
    sessions = label_sessions(df, params)
    session_levels = compute_session_levels(df, params)
    orm_data = compute_orm(df, params)

    logger.info("Computing daily bias (HTF + overnight + ORM)...")
    t_bias = _time.perf_counter()
    bias_data = compute_daily_bias(df, session_levels, orm_data, df_4h, df_1h, params)
    logger.info("Bias computed in %.1fs", _time.perf_counter() - t_bias)

    logger.info("Computing regime...")
    regime = compute_regime(df, df_4h, bias_data, params)

    # ---- 4. Build feature columns on ALL bars ----
    features = pd.DataFrame(index=df.index)

    # 4a. ATR
    logger.info("Computing base features...")
    features["atr"] = compute_atr(df, period=14)

    # 4b. Displacement
    features["is_displacement"] = detect_displacement(df, params).astype(int)

    # 4c. Fluency
    features["fluency"] = compute_fluency(df, params)

    # 4d. Bad candles
    bad = detect_bad_candles(df, params)
    features["is_doji"] = bad["is_doji"].astype(int)
    features["is_long_wick"] = bad["is_long_wick"].astype(int)

    # 4e. Swing levels
    swing_params = {
        "left_bars": params["swing"]["left_bars"],
        "right_bars": params["swing"]["right_bars"],
    }
    swings = compute_swing_levels(df, swing_params)
    # FIX: shift(1) to account for right_bars lookahead (same as entry_signals.py and engine.py)
    sw_high = swings["swing_high_price"].shift(1).ffill()
    sw_low = swings["swing_low_price"].shift(1).ffill()
    features["dist_to_swing_high"] = df["close"] - sw_high
    features["dist_to_swing_low"] = df["close"] - sw_low
    features["swing_range"] = sw_high - sw_low

    # 4f. Session labels (one-hot)
    session_dummies = pd.get_dummies(sessions, prefix="session", dtype=int)
    for col in session_dummies.columns:
        features[col] = session_dummies[col]

    # 4g. Session levels (distances)
    for level_col in ["asia_high", "asia_low", "london_high", "london_low",
                       "overnight_high", "overnight_low", "ny_open"]:
        if level_col in session_levels.columns:
            features[f"dist_{level_col}"] = df["close"] - session_levels[level_col]

    # 4h. ORM
    features["dist_orm_high"] = df["close"] - orm_data["orm_high"]
    features["dist_orm_low"] = df["close"] - orm_data["orm_low"]
    features["orm_range"] = orm_data["orm_high"] - orm_data["orm_low"]

    # 4i. FVG features on working timeframe
    logger.info("Computing FVG features...")
    fvg_feats = compute_active_fvgs(df, params)
    features["nearest_bull_fvg_dist"] = fvg_feats["nearest_bull_fvg_dist"]
    features["nearest_bear_fvg_dist"] = fvg_feats["nearest_bear_fvg_dist"]
    features["num_active_bull_fvgs"] = fvg_feats["num_active_bull_fvgs"]
    features["num_active_bear_fvgs"] = fvg_feats["num_active_bear_fvgs"]
    features["nearest_fvg_size"] = fvg_feats["nearest_fvg_size"]

    # ---- HTF features (1H, 4H) aligned to working timeframe ----
    for tf_label, htf_df in [("1H", df_1h), ("4H", df_4h)]:
        logger.info("Computing HTF features for %s...", tf_label)
        htf_fvg = compute_active_fvgs(htf_df, params)
        htf_disp = detect_displacement(htf_df, params).astype(int)
        htf_fluency = compute_fluency(htf_df, params)
        htf_atr = compute_atr(htf_df, period=14)

        htf_feats = htf_fvg.copy()
        htf_feats["is_displacement"] = htf_disp
        htf_feats["fluency_score"] = htf_fluency
        htf_feats["atr"] = htf_atr

        # shift(1) + ffill for no lookahead (CLAUDE.md section 13)
        shifted = htf_feats.shift(1)
        aligned = shifted.reindex(df.index, method="ffill")

        for col in aligned.columns:
            features[f"htf_{tf_label}_{col}"] = aligned[col]

    # ---- Time features ----
    logger.info("Computing time features...")
    et_index = df.index.tz_convert("US/Eastern")
    features["hour_of_day"] = et_index.hour + et_index.minute / 60.0
    features["day_of_week"] = et_index.dayofweek
    features["is_monday"] = (et_index.dayofweek == 0).astype(int)
    features["is_friday"] = (et_index.dayofweek == 4).astype(int)

    # ---- Price features ----
    features["bar_body"] = (df["close"] - df["open"]).abs()
    features["bar_range"] = df["high"] - df["low"]
    safe_range = features["bar_range"].replace(0, np.nan)
    features["body_ratio"] = features["bar_body"] / safe_range
    features["returns_1"] = df["close"].pct_change(1)
    features["returns_5"] = df["close"].pct_change(5)

    # ---- 4j. NEW V2 features: bias, regime, signal info ----
    logger.info("Adding V2 features (bias, regime, signal)...")

    # Bias features
    features["bias_direction"] = bias_data["bias_direction"]
    features["bias_confidence"] = bias_data["bias_confidence"]
    features["regime"] = regime

    # Signal-derived features (only meaningful on signal bars, but fill for all)
    # signal_type: 0=trend, 1=mss, NaN=no signal
    signal_type_numeric = np.full(len(df), np.nan)
    for i in range(len(df)):
        if signals_df["signal"].iat[i]:
            signal_type_numeric[i] = 0.0 if signals_df["signal_type"].iat[i] == "trend" else 1.0
    features["signal_type"] = signal_type_numeric

    # setup_rr: R:R from entry_signals (entry to IRL / entry to stop)
    entry_p = signals_df["entry_price"].values
    stop_p = signals_df["model_stop"].values
    target_p = signals_df["irl_target"].values
    risk = np.abs(entry_p - stop_p)
    reward = np.abs(target_p - entry_p)
    rr = np.where(risk > 0, reward / risk, np.nan)
    features["setup_rr"] = rr

    # bias_aligned: does signal direction match bias direction?
    sig_dir = signals_df["signal_dir"].values.astype(float)
    bias_dir = bias_data["bias_direction"].values.astype(float)
    bias_aligned = np.where(
        (sig_dir != 0) & (bias_dir != 0),
        (np.sign(sig_dir) == np.sign(bias_dir)).astype(float),
        np.nan,
    )
    features["bias_aligned"] = bias_aligned

    # ---- 4k. NEW V3 features: PA quality (Phase A) ----
    logger.info("Adding V3 PA quality features (Phase A)...")
    from features.pa_quality import compute_all_pa_features
    pa_features = compute_all_pa_features(
        df, signals_df=signals_df, bias_data=bias_data, params=params
    )
    for col in pa_features.columns:
        features[col] = pa_features[col]
    logger.info("PA features added: %s", list(pa_features.columns))

    # ---- 5. Label signal bars using liquidity-based labeler ----
    logger.info("Labeling signal bars (liquidity-based)...")

    # Use signal_dir from entry_signals (not candle direction)
    signal_direction = pd.Series(
        signals_df["signal_dir"].values.astype(float),
        index=df.index,
    )
    # Only label signal bars
    signal_entry_mask = pd.Series(signal_mask, index=df.index)

    # For model_stop and irl_target, we pass them through the labeler
    # But labeler computes its own stops -- we want to use the entry_signals stops
    # So we'll do custom labeling here using signals_df data
    label_result = _label_signal_bars(df, signals_df, params)
    y = label_result["label"]

    # Also store the labeler's RR for feature
    # NOTE: label_rr removed — it comes from the labeler which uses forward-looking data.
    # setup_rr (from entry_signals) is the correct feature — uses only known info at entry time.

    # ---- 6. Filter to signal bars with valid labels ----
    valid_mask = y.notna()
    X = features.loc[valid_mask].copy()
    y_clean = y.loc[valid_mask].copy()

    # Fill remaining NaN in features
    X = X.ffill().fillna(0)

    elapsed = _time.perf_counter() - t0
    logger.info(
        "V2 Feature matrix built in %.1fs: X shape=%s, y shape=%s, "
        "positive rate=%.2f%%",
        elapsed, X.shape, y_clean.shape,
        100.0 * y_clean.mean() if len(y_clean) > 0 else 0,
    )
    logger.info("Feature columns (%d): %s", len(X.columns), list(X.columns))

    return X, y_clean, signals_df


def _label_signal_bars(
    df: pd.DataFrame,
    signals_df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Label signal bars using entry_signals model_stop and irl_target.

    For each signal bar:
      - Entry = close of signal bar (entry_price from signals_df)
      - SL = model_stop from signals_df
      - TP = irl_target from signals_df
      - Direction from signal_dir
      - Scan forward up to max_holding_bars
      - Label 1 if TP hit first, 0 if SL hit or timeout

    Returns DataFrame with label, holding_time, exit_type, rr_ratio columns.
    """
    cfg = params["labeling"]
    max_hold = cfg["max_holding_bars"]
    min_rr = cfg.get("min_rr", 1.0)

    n = len(df)
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values

    sig_mask = signals_df["signal"].values.astype(bool)
    sig_dir = signals_df["signal_dir"].values.astype(float)
    entry_price = signals_df["entry_price"].values
    model_stop = signals_df["model_stop"].values
    irl_target = signals_df["irl_target"].values

    entry_indices = np.where(sig_mask)[0]

    label_arr = np.full(n, np.nan)
    hold_arr = np.full(n, np.nan)
    exit_type_arr = np.empty(n, dtype=object)
    exit_type_arr[:] = ""
    rr_arr = np.full(n, np.nan)

    skipped_nan = 0
    skipped_rr = 0

    for idx in entry_indices:
        d = sig_dir[idx]
        ep = entry_price[idx]
        sl = model_stop[idx]
        tp = irl_target[idx]

        # Validate
        if np.isnan(ep) or np.isnan(sl) or np.isnan(tp) or d == 0:
            skipped_nan += 1
            continue

        # Compute R:R
        sl_dist = abs(ep - sl)
        tp_dist = abs(tp - ep)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0.0

        if rr < min_rr:
            skipped_rr += 1
            continue

        rr_arr[idx] = rr

        # Scan forward
        end_idx = min(idx + max_hold, n - 1)
        hit_label = 0.0
        hit_type = "timeout"
        hit_bars = max_hold

        for j in range(idx + 1, end_idx + 1):
            bars_elapsed = j - idx

            if d > 0:  # LONG
                tp_hit = high[j] >= tp
                sl_hit = low[j] <= sl
            else:  # SHORT
                tp_hit = low[j] <= tp
                sl_hit = high[j] >= sl

            if tp_hit and sl_hit:
                # Both hit same bar -- conservative: check open
                if d > 0:
                    if open_[j] >= tp:
                        hit_label, hit_type = 1.0, "tp"
                    else:
                        hit_label, hit_type = 0.0, "sl"
                else:
                    if open_[j] <= tp:
                        hit_label, hit_type = 1.0, "tp"
                    else:
                        hit_label, hit_type = 0.0, "sl"
                hit_bars = bars_elapsed
                break
            elif tp_hit:
                hit_label, hit_type = 1.0, "tp"
                hit_bars = bars_elapsed
                break
            elif sl_hit:
                hit_label, hit_type = 0.0, "sl"
                hit_bars = bars_elapsed
                break

        label_arr[idx] = hit_label
        hold_arr[idx] = hit_bars
        exit_type_arr[idx] = hit_type

    result = pd.DataFrame({
        "label": label_arr,
        "holding_time": hold_arr,
        "exit_type": exit_type_arr,
        "rr_ratio": rr_arr,
    }, index=df.index)

    entries = result["label"].dropna()
    if len(entries) > 0:
        n_tp = int((entries == 1.0).sum())
        n_sl = int((entries == 0.0).sum())
        logger.info(
            "_label_signal_bars: %d labeled — %d TP (%.1f%%), %d SL/timeout (%.1f%%), "
            "skipped: %d nan, %d bad_rr, mean RR: %.2f",
            len(entries), n_tp, 100 * n_tp / len(entries),
            n_sl, 100 * n_sl / len(entries),
            skipped_nan, skipped_rr,
            result["rr_ratio"].dropna().mean(),
        )

    return result


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def train(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
) -> xgb.Booster:
    """Train XGBoost V2 with time-based splits.

    Split strategy (temporal, no shuffling):
      - Train: 2022-01-01 through 2024-12-31
      - Validation: 2025-01-01 through 2025-03-31 (early stopping)
      - Test: 2025-04-01 onward

    Returns
    -------
    xgb.Booster
        Trained model (saved to models/saved/xgb_v2.json).
    """
    # ---- Time splits ----
    idx = X.index.tz_localize(None) if X.index.tz is not None else X.index

    train_end = pd.Timestamp("2024-12-31 23:59:59")
    val_end = pd.Timestamp("2025-03-31 23:59:59")

    train_mask = idx <= train_end
    val_mask = (idx > train_end) & (idx <= val_end)
    test_mask = idx > val_end

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    logger.info(
        "Split sizes -- train: %d, val: %d, test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    logger.info(
        "Label rates -- train: %.2f%%, val: %.2f%%, test: %.2f%%",
        100.0 * y_train.mean() if len(y_train) > 0 else 0,
        100.0 * y_val.mean() if len(y_val) > 0 else 0,
        100.0 * y_test.mean() if len(y_test) > 0 else 0,
    )

    # ---- XGBoost parameters ----
    if params is None:
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        spw = n_neg / max(n_pos, 1)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
            "learning_rate": 0.03,
            "min_child_weight": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "gamma": 2.0,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "scale_pos_weight": spw,
            "tree_method": "hist",
            "verbosity": 1,
        }
    logger.info("XGBoost params: %s", params)

    # ---- DMatrix ----
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X.columns))
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(X.columns))
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(X.columns))

    # ---- Train ----
    evals = [(dtrain, "train"), (dval, "val")]
    t0 = _time.perf_counter()

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    elapsed = _time.perf_counter() - t0
    logger.info("Training completed in %.1fs, best iteration: %d", elapsed, model.best_iteration)

    # ---- Save model ----
    _SAVED_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _SAVED_DIR / "xgb_v2.json"
    model.save_model(str(model_path))
    logger.info("Model saved to %s", model_path)

    # ---- Store splits for evaluation ----
    train._last_split = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "dtrain": dtrain,
        "dval": dval,
        "dtest": dtest,
    }

    return model


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )

    logger.info("=" * 70)
    logger.info("XGBoost V2 Training Pipeline -- NQ Quant")
    logger.info("=" * 70)

    # 1. Build feature matrix
    X, y, signals_df = build_feature_matrix_v2(timeframe="5m")

    print(f"\n{'='*70}")
    print("V2 FEATURE MATRIX SUMMARY")
    print(f"{'='*70}")
    print(f"Shape: {X.shape}")
    print(f"Label distribution: 1={int(y.sum())}, 0={int((y==0).sum())}")
    print(f"Positive rate: {100*y.mean():.2f}%")
    print(f"Date range: {X.index.min()} to {X.index.max()}")
    print(f"\nFeature columns ({len(X.columns)}):")
    for i, col in enumerate(X.columns):
        print(f"  {i+1:3d}. {col}")

    # 2. Train XGBoost V2
    print(f"\n{'='*70}")
    print("TRAINING XGBOOST V2")
    print(f"{'='*70}")
    model = train(X, y)

    # 3. Evaluate
    print(f"\n{'='*70}")
    print("TEST SET EVALUATION (2025-Q2+)")
    print(f"{'='*70}")

    from models.evaluate import evaluate, print_report

    split = train._last_split
    results = evaluate(model, split["X_test"], split["y_test"])
    print_report(results)

    # Validation set
    print(f"\n{'='*70}")
    print("VALIDATION SET EVALUATION (2025-Q1)")
    print(f"{'='*70}")
    val_results = evaluate(model, split["X_val"], split["y_val"], save_plots=False)
    print_report(val_results)

    print(f"\n{'='*70}")
    print("V2 PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Model saved to: {_SAVED_DIR / 'xgb_v2.json'}")
