"""
experiments/ml_liquidity_model.py — ML-based liquidity level hit probability model.

Can ML predict WHICH liquidity level is most likely to be reached?
If so, can we use that to select smarter TP2 targets?

Parts:
  1. Build training dataset: for each signal, enumerate all liquidity levels + features + labels
  2. Train XGBoost hit probability model (walk-forward)
  3. TP2 selection strategies via ML predictions
  4. Backtest comparison vs Config E baseline
  5. Analysis: simple rules that capture ML benefit

Usage: python experiments/ml_liquidity_model.py
"""
from __future__ import annotations

import logging
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

from features.swing import detect_swing_highs, detect_swing_lows
from experiments.validate_improvements import (
    load_all,
    _find_nth_swing,
    _compute_grade_fast,
    compute_metrics,
    walk_forward_metrics,
    print_metrics,
)
from experiments.multi_level_tp import (
    prepare_liquidity_data,
    run_backtest_multi_tp,
    build_liquidity_ladder_long,
    build_liquidity_ladder_short,
    _find_nth_swing_price,
    print_trim_stage_breakdown,
)

# ======================================================================
# Constants
# ======================================================================
MAX_FORWARD_BARS = 100       # 500 min on 5m data
SAMPLE_STRIDE = 3            # every 3rd signal for speed
SWING_HISTORY_WINDOW = 200   # bars lookback for swing collection

LEVEL_TYPE_MAP = {
    "swing_1": 0, "swing_2": 1, "swing_3": 2,
    "htf_swing": 3,
    "session_asia": 4, "session_london": 5, "session_overnight": 6,
    "irl_target": 7, "orm_level": 8,
}

FEATURE_COLS = [
    "distance_atr", "distance_pts", "distance_rr",
    "level_type_enc", "atr_14", "fluency",
    "hour_et", "day_of_week", "bias_aligned", "regime",
    "signal_type_enc", "stop_distance_atr",
    "entry_vs_overnight_pct", "signal_quality",
]


# ======================================================================
# Part 1: Build Training Dataset
# ======================================================================
def build_training_dataset(d: dict, d_extra: dict) -> pd.DataFrame:
    """For each sampled signal, enumerate all liquidity levels with features + labels."""
    t0 = _time.perf_counter()
    print("\n" + "=" * 100)
    print("PART 1: BUILDING TRAINING DATASET")
    print("=" * 100)

    nq = d["nq"]
    n = d["n"]
    o, h_arr, l_arr, c_arr = d["o"], d["h"], d["l"], d["c"]
    atr_arr = d["atr_arr"]
    fluency_arr = d["fluency_arr"]
    sig_mask = d["sig_mask"]
    sig_dir = d["sig_dir"]
    sig_type = d["sig_type"]
    entry_price_arr = d["entry_price_arr"]
    model_stop_arr = d["model_stop_arr"]
    irl_target_arr = d["irl_target_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    signal_quality = d["signal_quality"]
    et_frac_arr = d["et_frac_arr"]
    dow_arr = d["dow_arr"]
    dates = d["dates"]
    et_idx = d["et_idx"]

    # Session levels from d_extra
    sess_asia_high = d_extra["sess_asia_high"]
    sess_asia_low = d_extra["sess_asia_low"]
    sess_london_high = d_extra["sess_london_high"]
    sess_london_low = d_extra["sess_london_low"]
    sess_overnight_high = d_extra["sess_overnight_high"]
    sess_overnight_low = d_extra["sess_overnight_low"]
    htf_swing_high_price = d_extra["htf_swing_high_price"]
    htf_swing_low_price = d_extra["htf_swing_low_price"]
    reg_sh_mask = d_extra["reg_sh_mask"]
    reg_sl_mask = d_extra["reg_sl_mask"]
    reg_high_arr = d_extra["reg_high_arr"]
    reg_low_arr = d_extra["reg_low_arr"]

    # Build swing arrays for nth-swing lookup (shift(1) already done in d_extra)
    # Also need HTF swings with left=10, right=3
    print("  Computing HTF swing detection (left=10, right=3)...")
    htf_sh_bool = detect_swing_highs(nq["high"], left=10, right=3)
    htf_sl_bool = detect_swing_lows(nq["low"], left=10, right=3)
    htf_sh_shifted = htf_sh_bool.shift(1).fillna(False).astype(bool).values
    htf_sl_shifted = htf_sl_bool.shift(1).fillna(False).astype(bool).values
    htf_sh_prices = np.where(htf_sh_shifted, nq["high"].shift(1).values, np.nan)
    htf_sl_prices = np.where(htf_sl_shifted, nq["low"].shift(1).values, np.nan)

    # Regular swings for nth-swing collection
    reg_sh_shifted = d["swing_high_mask"]  # already shifted in load_all
    reg_sl_shifted = d["swing_low_mask"]

    # Numpy OHLC for forward simulation
    ohlc_high = h_arr.astype(np.float64)
    ohlc_low = l_arr.astype(np.float64)

    # Sample signals
    signal_indices = np.where(sig_mask)[0]
    sampled = signal_indices[::SAMPLE_STRIDE]
    print(f"  Sampled {len(sampled)} signals (stride={SAMPLE_STRIDE}) from {len(signal_indices)} total")

    all_records = []
    t_loop = _time.perf_counter()

    for count, idx in enumerate(sampled):
        if count % 5000 == 0 and count > 0:
            elapsed = _time.perf_counter() - t_loop
            rate = count / elapsed
            eta = (len(sampled) - count) / rate
            print(f"  Processed {count}/{len(sampled)} signals ({rate:.0f}/sec, ETA {eta:.0f}s)")

        direction = int(sig_dir[idx])
        if direction == 0:
            continue

        entry = float(entry_price_arr[idx])
        stop = float(model_stop_arr[idx])
        irl = float(irl_target_arr[idx])

        if np.isnan(entry) or np.isnan(stop):
            continue

        atr = float(atr_arr[idx]) if not np.isnan(atr_arr[idx]) else np.nan
        if np.isnan(atr) or atr < 0.1:
            continue

        stop_dist = abs(entry - stop)
        if stop_dist < 0.5:
            continue

        flu = float(fluency_arr[idx]) if not np.isnan(fluency_arr[idx]) else 0.5
        sq = float(signal_quality[idx]) if not np.isnan(signal_quality[idx]) else 0.5
        bias_d = float(bias_dir_arr[idx]) if not np.isnan(bias_dir_arr[idx]) else 0.0
        bias_aligned = 1.0 if (direction == np.sign(bias_d) and bias_d != 0) else 0.0
        regime = float(regime_arr[idx]) if not np.isnan(regime_arr[idx]) else 0.0
        hour_et = int(et_frac_arr[idx])
        dow = int(dow_arr[idx])
        sig_type_enc = 0.0 if str(sig_type[idx]) == "trend" else 1.0

        # Entry vs overnight range
        on_high = float(sess_overnight_high[idx]) if not np.isnan(sess_overnight_high[idx]) else np.nan
        on_low = float(sess_overnight_low[idx]) if not np.isnan(sess_overnight_low[idx]) else np.nan
        if not np.isnan(on_high) and not np.isnan(on_low) and (on_high - on_low) > 1.0:
            entry_vs_overnight = (entry - on_low) / (on_high - on_low)
        else:
            entry_vs_overnight = 0.5

        # ---- Collect liquidity levels in trade direction ----
        targets = {}

        if direction == 1:
            # Swing highs above entry
            for nth in [1, 2, 3]:
                val = _find_nth_swing_price(reg_sh_shifted, reg_high_arr, idx, nth)
                if not np.isnan(val) and val > entry + 1.0:
                    targets[f"swing_{nth}"] = (val, f"swing_{nth}")

            # HTF swing high
            for j in range(idx - 1, max(0, idx - SWING_HISTORY_WINDOW) - 1, -1):
                if htf_sh_shifted[j]:
                    val = nq["high"].values[j]
                    if val > entry + 1.0:
                        targets["htf_swing"] = (val, "htf_swing")
                    break

            # Session levels
            val = sess_asia_high[idx]
            if not np.isnan(val) and val > entry + 1.0:
                targets["session_asia"] = (val, "session_asia")
            val = sess_london_high[idx]
            if not np.isnan(val) and val > entry + 1.0:
                targets["session_london"] = (val, "session_london")
            val = sess_overnight_high[idx]
            if not np.isnan(val) and val > entry + 1.0:
                targets["session_overnight"] = (val, "session_overnight")

            # IRL target
            if not np.isnan(irl) and irl > entry + 1.0:
                targets["irl_target"] = (irl, "irl_target")

        else:  # SHORT
            # Swing lows below entry
            for nth in [1, 2, 3]:
                val = _find_nth_swing_price(reg_sl_shifted, reg_low_arr, idx, nth)
                if not np.isnan(val) and val < entry - 1.0:
                    targets[f"swing_{nth}"] = (val, f"swing_{nth}")

            # HTF swing low
            for j in range(idx - 1, max(0, idx - SWING_HISTORY_WINDOW) - 1, -1):
                if htf_sl_shifted[j]:
                    val = nq["low"].values[j]
                    if val < entry - 1.0:
                        targets["htf_swing"] = (val, "htf_swing")
                    break

            # Session levels
            val = sess_asia_low[idx]
            if not np.isnan(val) and val < entry - 1.0:
                targets["session_asia"] = (val, "session_asia")
            val = sess_london_low[idx]
            if not np.isnan(val) and val < entry - 1.0:
                targets["session_london"] = (val, "session_london")
            val = sess_overnight_low[idx]
            if not np.isnan(val) and val < entry - 1.0:
                targets["session_overnight"] = (val, "session_overnight")

            # IRL target
            if not np.isnan(irl) and irl < entry - 1.0:
                targets["irl_target"] = (irl, "irl_target")

        if len(targets) == 0:
            continue

        # ---- Forward simulate for each level ----
        end_sim = min(idx + 1 + MAX_FORWARD_BARS, n)
        window_high = ohlc_high[idx + 1:end_sim]
        window_low = ohlc_low[idx + 1:end_sim]

        if len(window_high) == 0:
            continue

        # Precompute cumulative MAE
        if direction == 1:
            cum_mae_arr = entry - np.minimum.accumulate(window_low)
        else:
            cum_mae_arr = np.maximum.accumulate(window_high) - entry

        ts = nq.index[idx]

        for level_key, (level_price, level_type) in targets.items():
            dist_pts = abs(level_price - entry)
            dist_atr = dist_pts / atr
            dist_rr = dist_pts / stop_dist if stop_dist > 0 else np.nan

            # Check hit
            if direction == 1:
                hit_mask = window_high >= level_price
            else:
                hit_mask = window_low <= level_price

            if hit_mask.any():
                bar_idx = np.argmax(hit_mask)
                hit = 1
                bars_to_hit = bar_idx + 1
                if direction == 1:
                    mae_before = float(max(0, entry - np.min(window_low[:bar_idx + 1])))
                else:
                    mae_before = float(max(0, np.max(window_high[:bar_idx + 1]) - entry))
            else:
                hit = 0
                bars_to_hit = np.nan
                mae_before = float(max(0, cum_mae_arr[-1])) if len(cum_mae_arr) > 0 else np.nan

            all_records.append({
                "timestamp": ts,
                "idx": idx,
                "direction": direction,
                "entry_price": entry,
                "stop_price": stop,
                "level_price": level_price,
                "level_type": level_type,
                "level_type_enc": LEVEL_TYPE_MAP.get(level_type, -1),
                # Features
                "distance_atr": dist_atr,
                "distance_pts": dist_pts,
                "distance_rr": dist_rr,
                "atr_14": atr,
                "fluency": flu,
                "signal_quality": sq,
                "hour_et": hour_et,
                "day_of_week": dow,
                "bias_aligned": bias_aligned,
                "regime": regime,
                "signal_type_enc": sig_type_enc,
                "stop_distance_atr": stop_dist / atr,
                "entry_vs_overnight_pct": entry_vs_overnight,
                # Label
                "hit": hit,
                "bars_to_hit": bars_to_hit,
                "mae_before_hit": mae_before,
            })

    df = pd.DataFrame(all_records)
    elapsed = _time.perf_counter() - t0
    print(f"\n  Dataset built: {len(df):,} rows from {len(sampled):,} signals in {elapsed:.1f}s")
    print(f"  Unique signals: {df['timestamp'].nunique():,}")
    print(f"  Hit rate overall: {df['hit'].mean()*100:.1f}%")
    print(f"\n  Level type distribution:")
    for lt in sorted(df["level_type"].unique()):
        sub = df[df["level_type"] == lt]
        print(f"    {lt:<20s} n={len(sub):>6,}  hit={sub['hit'].mean()*100:5.1f}%  med_dist_atr={sub['distance_atr'].median():.2f}")

    return df


# ======================================================================
# Part 2: Train XGBoost Hit Probability Model
# ======================================================================
def train_model(df: pd.DataFrame) -> tuple:
    """Train XGBoost with walk-forward: train 2016-2021, test 2022-2025."""
    print("\n" + "=" * 100)
    print("PART 2: TRAIN XGBOOST HIT PROBABILITY MODEL")
    print("=" * 100)

    # Add year column
    df = df.copy()
    df["year"] = pd.to_datetime(df["timestamp"]).dt.year

    # Walk-forward split
    train_mask = df["year"] <= 2021
    test_mask = df["year"] >= 2022

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    print(f"  Train: {len(train_df):,} rows ({train_df['year'].min()}-{train_df['year'].max()})")
    print(f"  Test:  {len(test_df):,} rows ({test_df['year'].min()}-{test_df['year'].max()})")
    print(f"  Train hit rate: {train_df['hit'].mean()*100:.1f}%")
    print(f"  Test hit rate:  {test_df['hit'].mean()*100:.1f}%")

    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df["hit"].values.astype(np.float32)
    X_test = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test = test_df["hit"].values.astype(np.float32)

    # Handle NaN in features
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    print("\n  Training XGBoost...")
    t0 = _time.perf_counter()

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1.0,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    elapsed = _time.perf_counter() - t0
    print(f"  Training done in {elapsed:.1f}s")

    # Predictions
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation
    auc = roc_auc_score(y_test, y_pred_prob)
    ap = average_precision_score(y_test, y_pred_prob)
    brier = brier_score_loss(y_test, y_pred_prob)

    print(f"\n  === TEST SET EVALUATION ===")
    print(f"  AUC:              {auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    print(f"  Brier Score:       {brier:.4f}")

    # Precision/Recall at various thresholds
    print(f"\n  Precision/Recall at various thresholds:")
    print(f"  {'Threshold':>10s} {'Precision':>10s} {'Recall':>10s} {'N_pred':>10s} {'True_hit%':>10s}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for thresh in [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        pred_pos = y_pred_prob >= thresh
        if pred_pos.sum() == 0:
            continue
        prec = y_test[pred_pos].mean()
        rec = y_test[pred_pos].sum() / y_test.sum() if y_test.sum() > 0 else 0
        print(f"  {thresh:>10.2f} {prec:>10.3f} {rec:>10.3f} {pred_pos.sum():>10,} {prec*100:>9.1f}%")

    # Calibration
    print(f"\n  Calibration (predicted vs actual):")
    try:
        frac_pos, mean_pred = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy='uniform')
        print(f"  {'Bin':>6s} {'Predicted':>10s} {'Actual':>10s} {'Diff':>8s}")
        for mp, fp in zip(mean_pred, frac_pos):
            print(f"  {mp:>10.3f} {mp:>10.3f} {fp:>10.3f} {fp-mp:>+8.3f}")
    except Exception as e:
        print(f"  Calibration failed: {e}")

    # Feature importance
    print(f"\n  Feature Importance:")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for rank, fi in enumerate(sorted_idx):
        print(f"    {rank+1:>2d}. {FEATURE_COLS[fi]:25s} importance={importances[fi]:.4f}")

    # Per-level-type AUC
    print(f"\n  Per-level-type AUC (test set):")
    for lt in sorted(test_df["level_type"].unique()):
        lt_mask = test_df["level_type"].values == lt
        if lt_mask.sum() < 50 or y_test[lt_mask].std() == 0:
            continue
        try:
            lt_auc = roc_auc_score(y_test[lt_mask], y_pred_prob[lt_mask])
            print(f"    {lt:<20s} AUC={lt_auc:.4f}  n={lt_mask.sum():,}  hit_rate={y_test[lt_mask].mean()*100:.1f}%")
        except ValueError:
            print(f"    {lt:<20s} AUC=N/A (single class)")

    # Store predictions on test set for Part 3
    test_df = test_df.copy()
    test_df["hit_prob"] = y_pred_prob

    return model, train_df, test_df


# ======================================================================
# Part 3: TP2 Selection via ML
# ======================================================================
def compute_ml_tp2_selections(test_df: pd.DataFrame) -> pd.DataFrame:
    """For each signal, compute ML-based TP2 using different strategies."""
    print("\n" + "=" * 100)
    print("PART 3: TP2 SELECTION VIA ML PREDICTIONS")
    print("=" * 100)

    # Group by timestamp (signal)
    signal_groups = test_df.groupby("timestamp")
    print(f"  Total signals with predictions: {len(signal_groups):,}")

    results = []
    for ts, grp in signal_groups:
        if len(grp) < 2:
            continue

        direction = grp["direction"].iloc[0]
        entry_price = grp["entry_price"].iloc[0]
        stop_price = grp["stop_price"].iloc[0]
        idx = int(grp["idx"].iloc[0])
        stop_dist = abs(entry_price - stop_price)

        # Exclude swing_1 from TP2 candidates (swing_1 is TP1)
        # TP2 candidates: everything except the closest swing
        # Sort by distance
        grp_sorted = grp.sort_values("distance_atr")

        # Strategy A: highest probability
        best_prob_row = grp_sorted.loc[grp_sorted["hit_prob"].idxmax()]
        tp2_a = float(best_prob_row["level_price"])

        # Strategy B: highest expected value (hit_prob * distance_rr)
        grp_sorted = grp_sorted.copy()
        grp_sorted["ev"] = grp_sorted["hit_prob"] * grp_sorted["distance_rr"]
        best_ev_row = grp_sorted.loc[grp_sorted["ev"].idxmax()]
        tp2_b = float(best_ev_row["level_price"])

        # Strategy C: farthest level with hit_prob > threshold
        for c_thresh in [0.60, 0.65, 0.70]:
            above_thresh = grp_sorted[grp_sorted["hit_prob"] >= c_thresh]
            if len(above_thresh) > 0:
                # Farthest
                farthest = above_thresh.sort_values("distance_atr", ascending=False).iloc[0]
                results.append({
                    "timestamp": ts,
                    "idx": idx,
                    "direction": direction,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    f"tp2_c_{int(c_thresh*100)}": float(farthest["level_price"]),
                    f"tp2_c_{int(c_thresh*100)}_type": farthest["level_type"],
                    f"tp2_c_{int(c_thresh*100)}_prob": float(farthest["hit_prob"]),
                    f"tp2_c_{int(c_thresh*100)}_rr": float(farthest["distance_rr"]),
                })

        # Find the "nearest" level (current baseline approach)
        nearest_row = grp_sorted.iloc[0]  # Smallest distance
        tp2_nearest = float(nearest_row["level_price"])

        # Find TP1 equivalent (nearest swing_1 or irl_target)
        tp1_candidates = grp_sorted[grp_sorted["level_type"].isin(["swing_1", "irl_target"])]
        if len(tp1_candidates) > 0:
            tp1_val = float(tp1_candidates.iloc[0]["level_price"])
        else:
            tp1_val = entry_price + direction * stop_dist

        # Filter TP2 candidates: must be beyond TP1
        if direction == 1:
            tp2_cands = grp_sorted[grp_sorted["level_price"] > tp1_val + 1.0]
        else:
            tp2_cands = grp_sorted[grp_sorted["level_price"] < tp1_val - 1.0]

        if len(tp2_cands) == 0:
            # Fallback
            tp2_a_filtered = tp1_val + direction * stop_dist * 1.5
            tp2_b_filtered = tp2_a_filtered
            tp2_nearest_filtered = tp2_a_filtered
        else:
            # Strategy A on filtered
            best_prob_row = tp2_cands.loc[tp2_cands["hit_prob"].idxmax()]
            tp2_a_filtered = float(best_prob_row["level_price"])

            # Strategy B on filtered
            tp2_cands_ev = tp2_cands.copy()
            tp2_cands_ev["ev"] = tp2_cands_ev["hit_prob"] * tp2_cands_ev["distance_rr"]
            best_ev_row = tp2_cands_ev.loc[tp2_cands_ev["ev"].idxmax()]
            tp2_b_filtered = float(best_ev_row["level_price"])

            # Nearest (baseline)
            if direction == 1:
                tp2_nearest_filtered = float(tp2_cands.sort_values("level_price").iloc[0]["level_price"])
            else:
                tp2_nearest_filtered = float(tp2_cands.sort_values("level_price", ascending=False).iloc[0]["level_price"])

        # Get details for analysis
        result_row = {
            "timestamp": ts,
            "idx": idx,
            "direction": direction,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "tp1": tp1_val,
            "tp2_nearest": tp2_nearest_filtered,
            "tp2_ml_prob": tp2_a_filtered,
            "tp2_ml_ev": tp2_b_filtered,
            "n_levels": len(grp),
            "n_tp2_candidates": len(tp2_cands) if 'tp2_cands' in dir() else 0,
        }

        # Strategy C thresholds on filtered candidates
        if len(tp2_cands) > 0:
            for c_thresh in [0.60, 0.65, 0.70]:
                above_thresh = tp2_cands[tp2_cands["hit_prob"] >= c_thresh]
                if len(above_thresh) > 0:
                    farthest = above_thresh.sort_values("distance_atr", ascending=False).iloc[0]
                    result_row[f"tp2_c{int(c_thresh*100)}"] = float(farthest["level_price"])
                else:
                    result_row[f"tp2_c{int(c_thresh*100)}"] = tp2_nearest_filtered
        else:
            for c_thresh in [0.60, 0.65, 0.70]:
                result_row[f"tp2_c{int(c_thresh*100)}"] = tp2_nearest_filtered

        results.append(result_row)

    # Flatten: some results may be dicts with different keys from the C threshold loop
    # Rebuild cleanly
    clean_results = []
    for r in results:
        if "tp2_nearest" in r:
            clean_results.append(r)

    tp2_df = pd.DataFrame(clean_results)
    print(f"  TP2 selections computed for {len(tp2_df):,} signals")

    # Compare distances
    if len(tp2_df) > 0:
        print(f"\n  TP2 Distance Comparison (ATR multiples from entry):")
        for col in ["tp2_nearest", "tp2_ml_prob", "tp2_ml_ev", "tp2_c60", "tp2_c65", "tp2_c70"]:
            if col not in tp2_df.columns:
                continue
            dists = np.abs(tp2_df[col] - tp2_df["entry_price"])
            atrs = tp2_df.merge(
                test_df[["timestamp", "atr_14"]].drop_duplicates("timestamp"),
                on="timestamp", how="left"
            )["atr_14"]
            dist_atr = dists / atrs
            valid = ~dist_atr.isna()
            if valid.sum() > 0:
                print(f"    {col:20s}: med={dist_atr[valid].median():.2f} ATR, mean={dist_atr[valid].mean():.2f} ATR")

        # How often do ML strategies differ from nearest?
        diff_prob = (tp2_df["tp2_ml_prob"] != tp2_df["tp2_nearest"]).mean()
        diff_ev = (tp2_df["tp2_ml_ev"] != tp2_df["tp2_nearest"]).mean()
        print(f"\n  ML vs Nearest disagreement rate:")
        print(f"    ML Prob strategy:  {diff_prob*100:.1f}% of signals choose a different TP2")
        print(f"    ML EV strategy:    {diff_ev*100:.1f}% of signals choose a different TP2")

    return tp2_df


# ======================================================================
# Part 4: Backtest Comparison
# ======================================================================
def run_backtest_with_ml_tp2(
    d: dict,
    d_extra: dict,
    tp2_lookup: dict,  # {idx: tp2_price}
    strategy_name: str = "ML",
    # Config E parameters
    sq_short: float = 0.80,
    block_pm_shorts: bool = True,
    ny_tp_mult: float = 3.0,
    trail_nth_swing: int = 3,
) -> list[dict]:
    """Run multi-TP backtest with ML-selected TP2.

    Overrides the default TP2 selection with ML prediction for signals
    that have an ML prediction available. Falls back to nearest-level for
    signals without ML predictions.
    """
    # We modify the build_liquidity_ladder functions by monkey-patching d_extra
    # with a custom TP2 override. The cleanest approach: run the standard
    # multi-TP backtest but intercept the TP2 computation.

    # Strategy: create a wrapper that patches run_backtest_multi_tp
    # The simplest way: precompute TP2 overrides and inject them

    # Actually, run_backtest_multi_tp calls build_liquidity_ladder_long/short
    # inside the main loop. Instead of modifying that function, let me create
    # a modified version that checks the lookup table.

    # First, let me just run the standard backtest to get TP1 values,
    # then override TP2 and re-simulate. But the backtest is integrated...

    # The cleanest approach: modify d_extra to include tp2 overrides
    # and create a patched version of build_liquidity_ladder.

    # Let's use a different approach: create modified build_liquidity functions
    # that return the ML TP2 when available.

    class MLLiquidityPatcher:
        """Patches liquidity ladder building with ML TP2."""
        def __init__(self, d_extra_orig, tp2_lk, direction_lk):
            self.d_extra = d_extra_orig
            self.tp2_lookup = tp2_lk
            self.direction_lookup = direction_lk

        def build_long(self, entry_price, stop_price, irl_target, idx, min_r_dist=0.5):
            tp1, tp2_default = build_liquidity_ladder_long(
                entry_price, stop_price, irl_target, idx, self.d_extra, min_r_dist)
            if idx in self.tp2_lookup:
                tp2_ml = self.tp2_lookup[idx]
                # Validate: TP2 must be above TP1 for longs
                if tp2_ml > tp1 + 1.0:
                    return tp1, tp2_ml
            return tp1, tp2_default

        def build_short(self, entry_price, stop_price, irl_target, idx, min_r_dist=0.5):
            tp1, tp2_default = build_liquidity_ladder_short(
                entry_price, stop_price, irl_target, idx, self.d_extra, min_r_dist)
            if idx in self.tp2_lookup:
                tp2_ml = self.tp2_lookup[idx]
                # Validate: TP2 must be below TP1 for shorts
                if tp2_ml < tp1 - 1.0:
                    return tp1, tp2_ml
            return tp1, tp2_default

    # Since we can't easily modify the inner loop of run_backtest_multi_tp,
    # we'll create a thin wrapper that copies-and-modifies the key TP2 logic.
    # For efficiency, let's use a simpler approach: run the backtest with a
    # monkey-patched module-level function.

    import experiments.multi_level_tp as mlt_module

    # Save originals
    orig_build_long = mlt_module.build_liquidity_ladder_long
    orig_build_short = mlt_module.build_liquidity_ladder_short

    # Create patched versions
    def patched_build_long(entry_price, stop_price, irl_target, idx, d_extra_arg, min_r_dist=0.5):
        tp1, tp2_default = orig_build_long(entry_price, stop_price, irl_target, idx, d_extra_arg, min_r_dist)
        if idx in tp2_lookup:
            tp2_ml = tp2_lookup[idx]
            if tp2_ml > tp1 + 1.0:
                return tp1, tp2_ml
        return tp1, tp2_default

    def patched_build_short(entry_price, stop_price, irl_target, idx, d_extra_arg, min_r_dist=0.5):
        tp1, tp2_default = orig_build_short(entry_price, stop_price, irl_target, idx, d_extra_arg, min_r_dist)
        if idx in tp2_lookup:
            tp2_ml = tp2_lookup[idx]
            if tp2_ml < tp1 - 1.0:
                return tp1, tp2_ml
        return tp1, tp2_default

    # Monkey-patch
    mlt_module.build_liquidity_ladder_long = patched_build_long
    mlt_module.build_liquidity_ladder_short = patched_build_short

    try:
        trades = run_backtest_multi_tp(
            d, d_extra,
            sq_short=sq_short,
            block_pm_shorts=block_pm_shorts,
            tp1_trim_pct=0.25,
            tp2_trim_pct=0.25,
            be_after_tp1=False,
            be_after_tp2=True,
            trail_nth_swing=trail_nth_swing,
            ny_tp_mult=ny_tp_mult,
        )
    finally:
        # Restore originals
        mlt_module.build_liquidity_ladder_long = orig_build_long
        mlt_module.build_liquidity_ladder_short = orig_build_short

    return trades


def run_backtests_comparison(d: dict, d_extra: dict, tp2_df: pd.DataFrame) -> dict:
    """Run backtest for each ML strategy and compare vs baseline."""
    print("\n" + "=" * 100)
    print("PART 4: BACKTEST COMPARISON")
    print("=" * 100)

    # Config E parameters
    config_e = dict(
        sq_short=0.80,
        block_pm_shorts=True,
        ny_tp_mult=3.0,
        trail_nth_swing=3,
    )

    # Test period only (2022-2025) for fair comparison with ML
    test_start = "2022-01-01"
    test_end = "2025-12-31"

    results_all = {}

    # ---- Baseline: Config E with nearest TP2 (standard multi-TP) ----
    print("\n  Running Config E baseline (nearest TP2)...")
    t0 = _time.perf_counter()
    baseline_trades = run_backtest_multi_tp(
        d, d_extra,
        tp1_trim_pct=0.25, tp2_trim_pct=0.25,
        be_after_tp1=False, be_after_tp2=True,
        start_date=test_start, end_date=test_end,
        **config_e,
    )
    baseline_m = compute_metrics(baseline_trades)
    print_metrics("Config E Baseline (nearest TP2)", baseline_m)
    print(f"  ({_time.perf_counter() - t0:.1f}s)")
    results_all["baseline"] = (baseline_trades, baseline_m)

    # Also full period baseline
    print("\n  Running Config E baseline (FULL period)...")
    t0 = _time.perf_counter()
    baseline_full = run_backtest_multi_tp(
        d, d_extra,
        tp1_trim_pct=0.25, tp2_trim_pct=0.25,
        be_after_tp1=False, be_after_tp2=True,
        **config_e,
    )
    baseline_full_m = compute_metrics(baseline_full)
    print_metrics("Config E Full Period", baseline_full_m)
    print(f"  ({_time.perf_counter() - t0:.1f}s)")
    results_all["baseline_full"] = (baseline_full, baseline_full_m)

    # ---- ML Strategies on test period ----
    strategies = {}

    if len(tp2_df) > 0:
        for strat_col, strat_name in [
            ("tp2_ml_prob", "ML Highest Prob"),
            ("tp2_ml_ev", "ML Highest EV"),
            ("tp2_c60", "ML Threshold 60%"),
            ("tp2_c65", "ML Threshold 65%"),
            ("tp2_c70", "ML Threshold 70%"),
        ]:
            if strat_col not in tp2_df.columns:
                continue
            valid = tp2_df[~tp2_df[strat_col].isna()]
            if len(valid) == 0:
                continue

            # Build lookup: idx -> tp2_price
            tp2_lookup = dict(zip(valid["idx"].astype(int), valid[strat_col].astype(float)))

            print(f"\n  Running {strat_name} (n={len(tp2_lookup):,} overrides)...")
            t0 = _time.perf_counter()
            trades = run_backtest_with_ml_tp2(
                d, d_extra, tp2_lookup, strat_name,
                **config_e,
            )
            # Filter to test period by year
            test_trades = [t for t in trades if t["entry_time"].year >= 2022]

            m = compute_metrics(test_trades)
            print_metrics(f"{strat_name} (test 2022+)", m)
            delta_r = m["R"] - baseline_m["R"]
            delta_ppdd = m["PPDD"] - baseline_m["PPDD"]
            print(f"    vs Baseline: dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}")
            print(f"  ({_time.perf_counter() - t0:.1f}s)")
            results_all[strat_name] = (test_trades, m)

    # ---- Summary table ----
    print("\n" + "=" * 100)
    print("BACKTEST SUMMARY (Test Period 2022+)")
    print("=" * 100)
    print(f"  {'Strategy':40s} | {'Trades':>6} | {'R':>9} | {'PPDD':>7} | {'PF':>6} | {'WR':>6} | {'MaxDD':>6}")
    print(f"  {'-'*40}-+-{'-'*6}-+-{'-'*9}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for label, (trades, m) in results_all.items():
        if label == "baseline_full":
            continue
        print(f"  {label:40s} | {m['trades']:6d} | {m['R']:+9.1f} | {m['PPDD']:7.2f} | {m['PF']:6.2f} | {m['WR']:5.1f}% | {m['MaxDD']:6.1f}")

    # ---- Walk-forward for baseline and best ML strategy ----
    if len(results_all) > 2:
        print("\n  Walk-Forward by Year:")
        # Find best ML strategy by PPDD
        ml_strategies = {k: v for k, v in results_all.items() if k not in ("baseline", "baseline_full")}
        if ml_strategies:
            best_ml_name = max(ml_strategies.keys(), key=lambda k: ml_strategies[k][1].get("PPDD", 0))
            best_ml_trades, best_ml_m = ml_strategies[best_ml_name]

            print(f"\n  Best ML Strategy: {best_ml_name}")
            wf_bl = walk_forward_metrics(baseline_trades)
            wf_ml = walk_forward_metrics(best_ml_trades)

            bl_dict = {w["year"]: w for w in wf_bl}
            ml_dict = {w["year"]: w for w in wf_ml}
            all_years = sorted(set(list(bl_dict.keys()) + list(ml_dict.keys())))

            print(f"  {'Year':>6} | {'--- Baseline ---':^28s} | {'--- ' + best_ml_name[:20] + ' ---':^28s}")
            print(f"  {'':>6} | {'n':>4} {'R':>7} {'PF':>5} {'PPDD':>6} | {'n':>4} {'R':>7} {'PF':>5} {'PPDD':>6}")
            print(f"  {'-'*6}-+-{'-'*28}-+-{'-'*28}")

            for y in all_years:
                bl = bl_dict.get(y, {"n": 0, "R": 0, "PF": 0, "PPDD": 0})
                ml = ml_dict.get(y, {"n": 0, "R": 0, "PF": 0, "PPDD": 0})
                marker = " *" if ml["R"] > bl["R"] else ""
                print(f"  {y:>6} | {bl['n']:4d} {bl['R']:+7.1f} {bl['PF']:5.2f} {bl['PPDD']:+6.2f}"
                      f" | {ml['n']:4d} {ml['R']:+7.1f} {ml['PF']:5.2f} {ml['PPDD']:+6.2f}{marker}")

    return results_all


# ======================================================================
# Part 5: Analysis
# ======================================================================
def analyze_results(test_df: pd.DataFrame, model: xgb.XGBClassifier) -> None:
    """Analyze what drives hit probability and find simple rules."""
    print("\n" + "=" * 100)
    print("PART 5: ANALYSIS — What drives hit probability?")
    print("=" * 100)

    # ---- Q1: Does distance_atr dominate? ----
    print("\n  Q1: Feature importance breakdown")
    importances = model.feature_importances_
    total_imp = importances.sum()
    sorted_idx = np.argsort(importances)[::-1]
    top_3_imp = sum(importances[sorted_idx[:3]]) / total_imp * 100
    print(f"  Top 3 features account for {top_3_imp:.1f}% of total importance")
    for rank, fi in enumerate(sorted_idx[:5]):
        print(f"    {rank+1}. {FEATURE_COLS[fi]:25s} {importances[fi]/total_imp*100:.1f}%")

    # ---- Q2: When do far targets work? ----
    print("\n  Q2: Hit rate by market condition for FAR targets (dist_atr > 2.0)")
    far = test_df[test_df["distance_atr"] > 2.0].copy()
    if len(far) > 100:
        # By regime
        for regime_val, regime_label in [(0.0, "Regime 0 (low)"), (1.0, "Regime 1 (high)")]:
            sub = far[far["regime"] == regime_val]
            if len(sub) > 50:
                print(f"    {regime_label}: hit_rate={sub['hit'].mean()*100:.1f}% (n={len(sub):,})")

        # By fluency
        high_flu = far[far["fluency"] > 0.7]
        low_flu = far[far["fluency"] <= 0.7]
        if len(high_flu) > 50:
            print(f"    High fluency (>0.7): hit_rate={high_flu['hit'].mean()*100:.1f}% (n={len(high_flu):,})")
        if len(low_flu) > 50:
            print(f"    Low fluency (<=0.7): hit_rate={low_flu['hit'].mean()*100:.1f}% (n={len(low_flu):,})")

        # By bias alignment
        aligned = far[far["bias_aligned"] == 1]
        not_aligned = far[far["bias_aligned"] == 0]
        if len(aligned) > 50:
            print(f"    Bias aligned:     hit_rate={aligned['hit'].mean()*100:.1f}% (n={len(aligned):,})")
        if len(not_aligned) > 50:
            print(f"    Bias not aligned: hit_rate={not_aligned['hit'].mean()*100:.1f}% (n={len(not_aligned):,})")

        # By hour
        print(f"\n    Far target hit rate by hour (ET):")
        for hour in range(10, 16):
            sub = far[far["hour_et"] == hour]
            if len(sub) > 30:
                print(f"      {hour:02d}:00  hit_rate={sub['hit'].mean()*100:.1f}%  n={len(sub):,}")

    # ---- Q3: Simple rule that captures ML benefit? ----
    print("\n  Q3: Simple rule-based alternatives")

    # Rule 1: Use far target when fluency > 0.7 AND regime == 1
    rule1 = (test_df["fluency"] > 0.7) & (test_df["regime"] >= 1.0)
    # Rule 2: Use far target when fluency > 0.7 AND bias_aligned
    rule2 = (test_df["fluency"] > 0.7) & (test_df["bias_aligned"] == 1.0)
    # Rule 3: Distance-adaptive threshold
    rule3 = test_df["distance_atr"] < 3.0  # Only target levels within 3 ATR

    for rule_name, rule_mask in [
        ("Rule 1: flu>0.7 AND regime=1", rule1),
        ("Rule 2: flu>0.7 AND bias aligned", rule2),
        ("Rule 3: distance < 3 ATR only", rule3),
    ]:
        if rule_mask.sum() > 100:
            sub = test_df[rule_mask]
            print(f"    {rule_name}")
            print(f"      N={rule_mask.sum():,}  hit_rate={sub['hit'].mean()*100:.1f}%  "
                  f"med_dist_atr={sub['distance_atr'].median():.2f}")

    # ---- Q4: ML vs simple distance-only model ----
    print("\n  Q4: ML vs simple distance-based prediction")
    y_test = test_df["hit"].values.astype(float)
    y_pred_ml = test_df["hit_prob"].values

    # Simple model: hit_prob = 1 / (1 + distance_atr)
    dist_atr = test_df["distance_atr"].values
    y_pred_simple = 1.0 / (1.0 + dist_atr)

    try:
        auc_ml = roc_auc_score(y_test, y_pred_ml)
        auc_simple = roc_auc_score(y_test, y_pred_simple)
        print(f"    ML model AUC:              {auc_ml:.4f}")
        print(f"    Simple 1/(1+dist) AUC:     {auc_simple:.4f}")
        print(f"    ML improvement:            {auc_ml - auc_simple:+.4f}")
    except ValueError as e:
        print(f"    AUC comparison failed: {e}")

    # ---- Q5: Interaction effects ----
    print("\n  Q5: Interaction effects (hit rate by distance x fluency)")
    dist_bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 999)]
    flu_bins = [(0, 0.5), (0.5, 0.7), (0.7, 1.0)]

    print(f"    {'Dist ATR':>12s}", end="")
    for fl, fh in flu_bins:
        print(f" | {'Flu '+str(fl)+'-'+str(fh):>14s}", end="")
    print()
    print(f"    {'-'*12}", end="")
    for _ in flu_bins:
        print(f"-+-{'-'*14}", end="")
    print()

    for dl, dh in dist_bins:
        dist_mask = (test_df["distance_atr"] >= dl) & (test_df["distance_atr"] < dh)
        label = f"{dl}-{dh}" if dh < 999 else f"{dl}+"
        print(f"    {label:>12s}", end="")
        for fl, fh in flu_bins:
            flu_mask = (test_df["fluency"] >= fl) & (test_df["fluency"] < fh)
            combined = dist_mask & flu_mask
            n_comb = combined.sum()
            if n_comb > 20:
                hr = test_df.loc[combined, "hit"].mean() * 100
                print(f" | {hr:5.1f}% n={n_comb:>5d}", end="")
            else:
                print(f" | {'---':>14s}", end="")
        print()


# ======================================================================
# Main
# ======================================================================
def main():
    t_start = _time.perf_counter()
    print("=" * 100)
    print("ML LIQUIDITY HIT PROBABILITY MODEL")
    print("Can ML predict which liquidity level will be reached?")
    print("=" * 100)

    # Load data
    d = load_all()
    d_extra = prepare_liquidity_data(d)

    # Part 1: Build training dataset
    df = build_training_dataset(d, d_extra)

    # Part 2: Train model
    model, train_df, test_df = train_model(df)

    # Part 3: ML TP2 selections
    tp2_df = compute_ml_tp2_selections(test_df)

    # Part 4: Backtest comparison
    results = run_backtests_comparison(d, d_extra, tp2_df)

    # Part 5: Analysis
    analyze_results(test_df, model)

    # ---- Final Summary ----
    print("\n" + "=" * 100)
    print("EXECUTION SUMMARY")
    print("=" * 100)
    total_time = _time.perf_counter() - t_start
    print(f"  Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Training data: {len(train_df):,} rows")
    print(f"  Test data:     {len(test_df):,} rows")
    print(f"  Signals with ML TP2: {len(tp2_df):,}")

    # Key findings
    print(f"\n  KEY FINDINGS:")
    if "baseline" in results:
        bl_m = results["baseline"][1]
        print(f"  Config E Baseline (test 2022+): R={bl_m['R']:+.1f}, PPDD={bl_m['PPDD']:.2f}, PF={bl_m['PF']:.2f}")

    best_name = None
    best_ppdd = -999
    for name, (trades, m) in results.items():
        if name in ("baseline", "baseline_full"):
            continue
        if m.get("PPDD", 0) > best_ppdd:
            best_ppdd = m["PPDD"]
            best_name = name

    if best_name:
        best_m = results[best_name][1]
        print(f"  Best ML Strategy ({best_name}): R={best_m['R']:+.1f}, PPDD={best_m['PPDD']:.2f}, PF={best_m['PF']:.2f}")
        if "baseline" in results:
            bl_m = results["baseline"][1]
            print(f"  Improvement: dR={best_m['R'] - bl_m['R']:+.1f}, dPPDD={best_m['PPDD'] - bl_m['PPDD']:+.2f}")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
