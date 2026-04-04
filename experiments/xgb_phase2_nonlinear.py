"""
experiments/xgb_phase2_nonlinear.py

Comprehensive XGBoost non-linear feature interaction analysis on the
signal feature database (15,894 signals x 58 columns).

Walk-forward time-series splits, feature importance, interaction analysis,
and practical trading simulation vs rule-based baseline.

References: CLAUDE.md sections 14, 17, 19
"""

from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_DATA_PATH = _PROJECT_ROOT / "data" / "signal_feature_database.parquet"

# ---------------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load signal feature database from parquet."""
    df = pd.read_parquet(_DATA_PATH)
    print(f"Loaded {len(df)} signals, {len(df.columns)} columns")
    print(f"Date range: {df['bar_time_utc'].min()} to {df['bar_time_utc'].max()}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Prepare feature matrix X, binary label y, outcome_r, passes_all_filters, and year series.

    Returns: (X, y, outcome_r, passes_filters, year)
    """
    # --- Extract time info before dropping ---
    year = pd.to_datetime(df["bar_time_utc"]).dt.year

    # --- Outcome columns (keep for later, not as features) ---
    outcome_r = df["outcome_r"].copy()
    passes_filters = df["passes_all_filters"].copy()
    outcome_label = df["outcome_label"].copy()

    # --- Binary label: 1 = TP hit, 0 = everything else ---
    y = (outcome_label == 1).astype(int)

    # --- Feature columns to keep (numeric) ---
    numeric_features = [
        "fvg_size_pts", "fvg_size_atr", "fvg_swept_liquidity", "fvg_sweep_score",
        "bar_body_ratio", "bar_body_atr", "bar_range_atr",
        "is_displaced", "fluency_score", "signal_quality",
        "atr_14", "atr_percentile",
        "hour_et", "day_of_week", "is_monday", "is_friday",
        "bias_confidence", "bias_aligned",
        "stop_distance_pts", "stop_distance_atr",
        "target_distance_pts", "target_rr",
        "pa_alt_dir_ratio", "is_orm_period",
        # Also include SMT features and bias_direction
        "bias_direction", "has_smt", "smt_bull", "smt_bear",
    ]

    X = pd.DataFrame(index=df.index)

    # Add numeric features
    for col in numeric_features:
        if col in df.columns:
            X[col] = df[col].values
        else:
            print(f"WARNING: column {col} not found in data, skipping")

    # --- Encode categoricals ---
    # signal_dir: already 1/-1
    X["signal_dir"] = df["signal_dir"].values

    # signal_type: trend=0, mss=1
    X["signal_type"] = (df["signal_type"] == "mss").astype(int).values

    # session: one-hot
    for sess in ["asia", "london", "ny", "other"]:
        X[f"session_{sess}"] = (df["session"] == sess).astype(int).values

    # sub_session: one-hot
    for sub in ["asia", "london", "ny_am", "ny_lunch", "ny_pm", "other"]:
        X[f"sub_{sub}"] = (df["sub_session"] == sub).astype(int).values

    # grade: ordinal (A+=2, B+=1, C=0)
    grade_map = {"A+": 2, "B+": 1, "C": 0}
    X["grade"] = df["grade"].map(grade_map).fillna(0).astype(int).values

    # regime: already numeric (0.0, 0.5, 1.0)
    X["regime"] = df["regime"].values

    # --- Convert booleans to int ---
    bool_cols = ["fvg_swept_liquidity", "is_displaced", "bias_aligned",
                 "is_monday", "is_friday", "is_orm_period",
                 "has_smt", "smt_bull", "smt_bear"]
    for col in bool_cols:
        if col in X.columns:
            X[col] = X[col].astype(int)

    # --- Fill NaN ---
    X = X.fillna(0)

    print(f"\nFeature matrix: {X.shape}")
    print(f"Binary label distribution: 1={int(y.sum())} ({100*y.mean():.1f}%), "
          f"0={int((y==0).sum())} ({100*(1-y.mean()):.1f}%)")
    print(f"Features ({len(X.columns)}): {list(X.columns)}")

    return X, y, outcome_r, passes_filters, year


# ---------------------------------------------------------------------------
# 2. Walk-forward time-series splits
# ---------------------------------------------------------------------------

def build_walk_forward_splits(
    X: pd.DataFrame, y: pd.Series, year: pd.Series,
) -> list[dict]:
    """Build expanding-window walk-forward splits.

    Folds:
      Fold 1: Train 2016-2019, Val 2020
      Fold 2: Train 2016-2020, Val 2021
      Fold 3: Train 2016-2021, Val 2022
      Fold 4: Train 2016-2022, Val 2023
      Fold 5: Train 2016-2023, Val 2024
      Fold 6: Train 2016-2024, Val 2025-2026
    """
    folds = []
    fold_configs = [
        (2016, 2019, 2020, 2020),
        (2016, 2020, 2021, 2021),
        (2016, 2021, 2022, 2022),
        (2016, 2022, 2023, 2023),
        (2016, 2023, 2024, 2024),
        (2016, 2024, 2025, 2026),
    ]

    for train_start, train_end, val_start, val_end in fold_configs:
        train_mask = (year >= train_start) & (year <= train_end)
        val_mask = (year >= val_start) & (year <= val_end)

        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        folds.append({
            "name": f"Train {train_start}-{train_end} / Val {val_start}-{val_end}",
            "train_idx": train_mask,
            "val_idx": val_mask,
        })

    return folds


# ---------------------------------------------------------------------------
# 3. Train XGBoost per fold
# ---------------------------------------------------------------------------

def train_fold(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    feature_names: list[str],
) -> tuple[xgb.Booster, dict]:
    """Train XGBoost on one fold with early stopping."""

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw = n_neg / max(n_pos, 1)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 1.0,
        "reg_alpha": 0.3,
        "reg_lambda": 2.0,
        "scale_pos_weight": spw,
        "tree_method": "hist",
        "verbosity": 0,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    return model, params


# ---------------------------------------------------------------------------
# 4. Evaluate fold
# ---------------------------------------------------------------------------

def evaluate_fold(
    model: xgb.Booster,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
    thresholds: list[float] = [0.40, 0.45, 0.50, 0.55, 0.60],
) -> dict:
    """Evaluate a single fold at multiple thresholds."""

    dval = xgb.DMatrix(X_val, feature_names=feature_names)
    y_prob = model.predict(dval)

    results = {
        "y_prob": y_prob,
        "y_true": y_val.values,
        "auc": roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.5,
        "logloss": log_loss(y_val, y_prob),
        "thresholds": {},
    }

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        n_pred_pos = int(y_pred.sum())
        n_actual_pos = int(y_val.sum())

        if n_pred_pos == 0:
            results["thresholds"][t] = {
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "accuracy": accuracy_score(y_val, y_pred),
                "n_pred_pos": 0, "n_actual_pos": n_actual_pos,
                "n_total": len(y_val),
            }
        else:
            results["thresholds"][t] = {
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred, zero_division=0),
                "f1": f1_score(y_val, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_val, y_pred),
                "n_pred_pos": n_pred_pos,
                "n_actual_pos": n_actual_pos,
                "n_total": len(y_val),
            }

    return results


# ---------------------------------------------------------------------------
# 5. Feature importance (gain-based)
# ---------------------------------------------------------------------------

def get_feature_importance(model: xgb.Booster) -> pd.DataFrame:
    """Get gain-based feature importance."""
    imp = model.get_score(importance_type="gain")
    df_imp = pd.DataFrame([
        {"feature": k, "gain": v} for k, v in imp.items()
    ]).sort_values("gain", ascending=False).reset_index(drop=True)
    return df_imp


# ---------------------------------------------------------------------------
# 6. Feature interaction analysis (tree-structure based)
# ---------------------------------------------------------------------------

def analyze_interactions(model: xgb.Booster, feature_names: list[str], top_n: int = 15) -> pd.DataFrame:
    """Analyze feature interactions by parsing tree splits.

    For each tree, look at parent-child split feature pairs.
    Features that frequently co-occur as parent-child in trees are
    likely forming important non-linear interactions.
    """
    trees_df = model.trees_to_dataframe()

    # Filter to non-leaf nodes
    splits = trees_df[trees_df["Feature"] != "Leaf"].copy()

    # Build parent-child pairs within each tree
    interaction_counts = {}
    interaction_gains = {}

    for tree_id in splits["Tree"].unique():
        tree_splits = splits[splits["Tree"] == tree_id].copy()

        # Map node ID to feature
        node_feature = dict(zip(tree_splits["ID"], tree_splits["Feature"]))
        node_gain = dict(zip(tree_splits["ID"], tree_splits["Gain"]))

        for _, row in tree_splits.iterrows():
            parent_feature = row["Feature"]
            parent_gain = row["Gain"]

            # Check children
            for child_col in ["Yes", "No"]:
                child_id = row[child_col]
                if child_id in node_feature:
                    child_feature = node_feature[child_id]
                    child_gain = node_gain[child_id]

                    if parent_feature != child_feature:
                        pair = tuple(sorted([parent_feature, child_feature]))
                        interaction_counts[pair] = interaction_counts.get(pair, 0) + 1
                        interaction_gains[pair] = interaction_gains.get(pair, 0.0) + parent_gain + child_gain

    # Build dataframe
    rows = []
    for pair, count in interaction_counts.items():
        rows.append({
            "feature_1": pair[0],
            "feature_2": pair[1],
            "co_occurrence": count,
            "combined_gain": interaction_gains[pair],
        })

    df_int = pd.DataFrame(rows).sort_values("combined_gain", ascending=False).reset_index(drop=True)
    return df_int.head(top_n)


# ---------------------------------------------------------------------------
# 7. Practical trading simulation
# ---------------------------------------------------------------------------

def simulate_trading(
    y_prob: np.ndarray,
    outcome_r: np.ndarray,
    passes_filters: np.ndarray,
    threshold: float,
    label: str = "",
) -> dict:
    """Simulate trading results for signals selected by XGB.

    Returns dict with trade count, total R, avg R, win rate, max DD, PPDD, PF.
    """
    results = {}

    # --- Scenario A: XGB + rule-based filters (intersection) ---
    mask_a = (y_prob >= threshold) & passes_filters
    r_a = outcome_r[mask_a]
    results["xgb_AND_filters"] = _compute_trade_stats(r_a, f"{label} XGB(>{threshold:.2f}) AND filters")

    # --- Scenario B: XGB only (replace rule-based filters) ---
    mask_b = (y_prob >= threshold)
    r_b = outcome_r[mask_b]
    results["xgb_only"] = _compute_trade_stats(r_b, f"{label} XGB(>{threshold:.2f}) only")

    # --- Baseline: rule-based filters only ---
    r_base = outcome_r[passes_filters]
    results["baseline"] = _compute_trade_stats(r_base, f"{label} Baseline (filters only)")

    return results


def _compute_trade_stats(r_values: np.ndarray, label: str) -> dict:
    """Compute comprehensive trade statistics from R-multiple array."""
    n = len(r_values)
    if n == 0:
        return {
            "label": label, "n_trades": 0, "total_r": 0.0, "avg_r": 0.0,
            "win_rate": 0.0, "max_dd_r": 0.0, "ppdd": 0.0, "pf": 0.0,
        }

    total_r = float(np.sum(r_values))
    avg_r = float(np.mean(r_values))
    wins = r_values[r_values > 0]
    losses = r_values[r_values <= 0]
    win_rate = len(wins) / n if n > 0 else 0.0
    pf = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else float("inf")

    # Max drawdown in R (equity curve based)
    cumr = np.cumsum(r_values)
    running_max = np.maximum.accumulate(cumr)
    drawdowns = cumr - running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    ppdd = total_r / abs(max_dd) if max_dd != 0 else float("inf")

    return {
        "label": label,
        "n_trades": n,
        "total_r": round(total_r, 2),
        "avg_r": round(avg_r, 4),
        "win_rate": round(win_rate * 100, 1),
        "max_dd_r": round(max_dd, 2),
        "ppdd": round(ppdd, 2),
        "pf": round(pf, 2),
    }


# ---------------------------------------------------------------------------
# 8. Printing utilities
# ---------------------------------------------------------------------------

def print_separator(title: str, char: str = "=", width: int = 90):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_fold_results(fold_name: str, eval_results: dict):
    """Print evaluation results for a single fold."""
    print(f"\n  --- {fold_name} ---")
    print(f"  AUC-ROC: {eval_results['auc']:.4f}  |  LogLoss: {eval_results['logloss']:.4f}")
    print(f"  {'Threshold':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>8s} "
          f"{'Acc':>8s} {'PredPos':>8s} {'ActPos':>8s} {'Total':>8s}")
    print(f"  {'-'*76}")

    for t, m in eval_results["thresholds"].items():
        print(f"  {t:>10.2f} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>8.3f} "
              f"{m['accuracy']:>8.3f} {m['n_pred_pos']:>8d} {m['n_actual_pos']:>8d} {m['n_total']:>8d}")


def print_trade_stats(stats: dict):
    """Print trading simulation stats."""
    s = stats
    if s["n_trades"] == 0:
        print(f"    {s['label']}: NO TRADES")
        return
    ppdd_str = f"{s['ppdd']:.2f}" if s['ppdd'] != float('inf') else "inf"
    pf_str = f"{s['pf']:.2f}" if s['pf'] != float('inf') else "inf"
    print(f"    {s['label']}")
    print(f"      Trades: {s['n_trades']:>5d}  |  TotalR: {s['total_r']:>8.2f}  |  "
          f"AvgR: {s['avg_r']:>7.4f}  |  WR: {s['win_rate']:>5.1f}%  |  "
          f"MaxDD: {s['max_dd_r']:>7.2f}R  |  PPDD: {ppdd_str:>7s}  |  PF: {pf_str:>6s}")


# ---------------------------------------------------------------------------
# 9. Additional analysis: depth sensitivity, hyperparameter robustness
# ---------------------------------------------------------------------------

def train_depth_sensitivity(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    feature_names: list[str],
) -> dict:
    """Train multiple models with different max_depth to check if non-linear interactions matter."""

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw = n_neg / max(n_pos, 1)

    depth_results = {}
    for depth in [1, 2, 3, 4, 5, 6, 8]:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": depth,
            "learning_rate": 0.05,
            "min_child_weight": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 1.0,
            "reg_alpha": 0.3,
            "reg_lambda": 2.0,
            "scale_pos_weight": spw,
            "tree_method": "hist",
            "verbosity": 0,
        }

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

        model = xgb.train(
            params, dtrain,
            num_boost_round=1500,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        y_prob = model.predict(dval)
        auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.5
        ll = log_loss(y_val, y_prob)
        y_pred_50 = (y_prob >= 0.50).astype(int)
        prec_50 = precision_score(y_val, y_pred_50, zero_division=0)

        depth_results[depth] = {
            "auc": auc,
            "logloss": ll,
            "precision_50": prec_50,
            "best_iter": model.best_iteration,
        }

    return depth_results


# ---------------------------------------------------------------------------
# 10. Calibration analysis
# ---------------------------------------------------------------------------

def calibration_analysis(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Compute calibration: how well do predicted probabilities match actual rates?"""
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:  # include right edge
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        n = int(mask.sum())
        if n > 0:
            actual_rate = float(y_true[mask].mean())
            mean_pred = float(y_prob[mask].mean())
        else:
            actual_rate = 0.0
            mean_pred = 0.0
        rows.append({
            "bin": f"{bins[i]:.2f}-{bins[i+1]:.2f}",
            "count": n,
            "mean_pred": round(mean_pred, 4),
            "actual_rate": round(actual_rate, 4),
            "calibration_gap": round(actual_rate - mean_pred, 4),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print_separator("XGBoost Phase 2: Non-Linear Feature Interaction Analysis")
    print("Signal Feature Database → Walk-Forward XGBoost → Interaction Discovery")
    print("No individual feature was significant (linear). Testing non-linear interactions.")

    # ---- 1. Load data ----
    print_separator("1. DATA LOADING", "-")
    df = load_data()

    # ---- 2. Prepare features ----
    print_separator("2. FEATURE PREPARATION", "-")
    X, y, outcome_r, passes_filters, year = prepare_features(df)

    feature_names = list(X.columns)

    # ---- 3. Walk-forward splits ----
    print_separator("3. WALK-FORWARD TIME-SERIES SPLITS", "-")
    folds = build_walk_forward_splits(X, y, year)

    for i, fold in enumerate(folds):
        n_train = fold["train_idx"].sum()
        n_val = fold["val_idx"].sum()
        y_train_rate = y[fold["train_idx"]].mean() * 100
        y_val_rate = y[fold["val_idx"]].mean() * 100
        print(f"  Fold {i+1}: {fold['name']}  "
              f"(train={n_train}, val={n_val}, "
              f"train_WR={y_train_rate:.1f}%, val_WR={y_val_rate:.1f}%)")

    # ---- 4. Train and evaluate each fold ----
    print_separator("4. PER-FOLD TRAINING & EVALUATION", "-")

    all_fold_results = []
    all_importances = []
    all_interactions = []
    all_y_prob_val = []
    all_y_true_val = []
    all_outcome_r_val = []
    all_passes_val = []

    for i, fold in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"  FOLD {i+1}: {fold['name']}")
        print(f"{'='*60}")

        train_mask = fold["train_idx"]
        val_mask = fold["val_idx"]

        X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
        X_val, y_val = X[val_mask].copy(), y[val_mask].copy()
        r_val = outcome_r[val_mask].values
        pf_val = passes_filters[val_mask].values

        # Train
        model, xgb_params = train_fold(X_train, y_train, X_val, y_val, feature_names)
        print(f"  Best iteration: {model.best_iteration}")

        # Evaluate
        eval_res = evaluate_fold(model, X_val, y_val, feature_names)
        print_fold_results(fold["name"], eval_res)
        all_fold_results.append(eval_res)

        # Accumulate for aggregate analysis
        all_y_prob_val.append(eval_res["y_prob"])
        all_y_true_val.append(eval_res["y_true"])
        all_outcome_r_val.append(r_val)
        all_passes_val.append(pf_val)

        # Feature importance
        imp = get_feature_importance(model)
        imp["fold"] = i + 1
        all_importances.append(imp)

        # Interactions
        interactions = analyze_interactions(model, feature_names, top_n=20)
        interactions["fold"] = i + 1
        all_interactions.append(interactions)

        # Trading simulation for this fold
        print(f"\n  --- Trading Simulation (Fold {i+1}) ---")
        for threshold in [0.45, 0.50, 0.55, 0.60]:
            sim = simulate_trading(eval_res["y_prob"], r_val, pf_val, threshold)
            print(f"\n    Threshold = {threshold:.2f}:")
            print_trade_stats(sim["baseline"])
            print_trade_stats(sim["xgb_AND_filters"])
            print_trade_stats(sim["xgb_only"])

    # ---- 5. Aggregate results ----
    print_separator("5. AGGREGATE RESULTS ACROSS ALL FOLDS")

    # Concatenate all validation predictions
    agg_y_prob = np.concatenate(all_y_prob_val)
    agg_y_true = np.concatenate(all_y_true_val)
    agg_r = np.concatenate(all_outcome_r_val)
    agg_pf = np.concatenate(all_passes_val)

    # AUC across all folds
    agg_auc = roc_auc_score(agg_y_true, agg_y_prob) if len(np.unique(agg_y_true)) > 1 else 0.5
    print(f"\n  Aggregate AUC-ROC (all folds): {agg_auc:.4f}")

    # Per-fold AUC summary
    print(f"\n  Per-fold AUC-ROC:")
    for i, res in enumerate(all_fold_results):
        print(f"    Fold {i+1}: {res['auc']:.4f}")
    print(f"    Mean:   {np.mean([r['auc'] for r in all_fold_results]):.4f}")
    print(f"    Std:    {np.std([r['auc'] for r in all_fold_results]):.4f}")

    # Aggregate threshold metrics
    print(f"\n  Aggregate Metrics (all folds combined):")
    print(f"  {'Threshold':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>8s} "
          f"{'PredPos':>8s} {'ActPos':>8s} {'Total':>8s}")
    print(f"  {'-'*68}")
    for t in [0.40, 0.45, 0.50, 0.55, 0.60]:
        y_pred = (agg_y_prob >= t).astype(int)
        n_pred = int(y_pred.sum())
        if n_pred > 0:
            prec = precision_score(agg_y_true, y_pred, zero_division=0)
            rec = recall_score(agg_y_true, y_pred, zero_division=0)
            f1 = f1_score(agg_y_true, y_pred, zero_division=0)
        else:
            prec = rec = f1 = 0.0
        print(f"  {t:>10.2f} {prec:>10.3f} {rec:>10.3f} {f1:>8.3f} "
              f"{n_pred:>8d} {int(agg_y_true.sum()):>8d} {len(agg_y_true):>8d}")

    # ---- 6. Aggregate feature importance ----
    print_separator("6. FEATURE IMPORTANCE (Gain-Based, Averaged Across Folds)")

    all_imp_df = pd.concat(all_importances, ignore_index=True)
    avg_imp = all_imp_df.groupby("feature")["gain"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    avg_imp = avg_imp.reset_index()

    print(f"\n  Top 20 Features by Average Gain:")
    print(f"  {'Rank':>4s} {'Feature':<30s} {'AvgGain':>10s} {'StdGain':>10s} {'InFolds':>8s}")
    print(f"  {'-'*66}")
    for i, row in avg_imp.head(20).iterrows():
        print(f"  {i+1:>4d} {row['feature']:<30s} {row['mean']:>10.2f} {row['std']:>10.2f} {int(row['count']):>8d}")

    # ---- 7. Aggregate interactions ----
    print_separator("7. TOP FEATURE INTERACTIONS (Parent-Child Co-occurrence in Trees)")

    all_int_df = pd.concat(all_interactions, ignore_index=True)
    avg_int = all_int_df.groupby(["feature_1", "feature_2"]).agg(
        total_cooccurrence=("co_occurrence", "sum"),
        total_gain=("combined_gain", "sum"),
        n_folds=("fold", "nunique"),
    ).sort_values("total_gain", ascending=False).reset_index()

    print(f"\n  Top 20 Feature Interactions by Combined Gain:")
    print(f"  {'Rank':>4s} {'Feature 1':<25s} {'Feature 2':<25s} "
          f"{'CoOccur':>8s} {'TotalGain':>12s} {'Folds':>6s}")
    print(f"  {'-'*84}")
    for i, row in avg_int.head(20).iterrows():
        print(f"  {i+1:>4d} {row['feature_1']:<25s} {row['feature_2']:<25s} "
              f"{int(row['total_cooccurrence']):>8d} {row['total_gain']:>12.1f} {int(row['n_folds']):>6d}")

    # ---- 8. Aggregate trading simulation ----
    print_separator("8. AGGREGATE TRADING SIMULATION (All Validation Folds Combined)")

    for threshold in [0.45, 0.50, 0.55, 0.60]:
        sim = simulate_trading(agg_y_prob, agg_r, agg_pf, threshold, label="[AGG]")
        print(f"\n  Threshold = {threshold:.2f}:")
        print_trade_stats(sim["baseline"])
        print_trade_stats(sim["xgb_AND_filters"])
        print_trade_stats(sim["xgb_only"])

    # ---- 9. Depth sensitivity (non-linearity test) ----
    print_separator("9. DEPTH SENSITIVITY ANALYSIS (Is Non-Linearity Helping?)")
    print("  Training on largest fold (2016-2024 train, 2025-2026 val) with varying max_depth")
    print("  If depth>1 significantly beats depth=1 (stump), non-linear interactions matter.")

    last_fold = folds[-1]
    X_train_last = X[last_fold["train_idx"]].copy()
    y_train_last = y[last_fold["train_idx"]].copy()
    X_val_last = X[last_fold["val_idx"]].copy()
    y_val_last = y[last_fold["val_idx"]].copy()

    depth_res = train_depth_sensitivity(X_train_last, y_train_last, X_val_last, y_val_last, feature_names)

    print(f"\n  {'Depth':>6s} {'AUC':>8s} {'LogLoss':>10s} {'Prec@0.50':>10s} {'BestIter':>10s}")
    print(f"  {'-'*48}")
    for depth, metrics in sorted(depth_res.items()):
        print(f"  {depth:>6d} {metrics['auc']:>8.4f} {metrics['logloss']:>10.4f} "
              f"{metrics['precision_50']:>10.3f} {metrics['best_iter']:>10d}")

    # Interpret
    auc_d1 = depth_res[1]["auc"]
    auc_best = max(v["auc"] for v in depth_res.values())
    best_depth = [k for k, v in depth_res.items() if v["auc"] == auc_best][0]
    auc_improvement = auc_best - auc_d1

    print(f"\n  Depth=1 (linear splits only) AUC: {auc_d1:.4f}")
    print(f"  Best AUC: {auc_best:.4f} at depth={best_depth}")
    print(f"  Non-linearity improvement: {auc_improvement:+.4f}")

    if auc_improvement > 0.01:
        print("  --> NON-LINEAR INTERACTIONS DETECTED: depth > 1 provides meaningful improvement")
    elif auc_improvement > 0.005:
        print("  --> MARGINAL non-linear interactions: small improvement from deeper trees")
    else:
        print("  --> NO significant non-linear interactions: stumps perform comparably")

    # ---- 10. Calibration ----
    print_separator("10. CALIBRATION ANALYSIS (Aggregate)")

    cal = calibration_analysis(agg_y_true, agg_y_prob, n_bins=10)
    print(f"\n  {'Bin':>12s} {'Count':>8s} {'MeanPred':>10s} {'ActualRate':>12s} {'Gap':>8s}")
    print(f"  {'-'*54}")
    for _, row in cal.iterrows():
        print(f"  {row['bin']:>12s} {row['count']:>8d} {row['mean_pred']:>10.4f} "
              f"{row['actual_rate']:>12.4f} {row['calibration_gap']:>+8.4f}")

    # ---- 11. Key findings summary ----
    print_separator("11. KEY FINDINGS SUMMARY")

    # Best achievable precision at reasonable threshold
    y_pred_55 = (agg_y_prob >= 0.55).astype(int)
    prec_55 = precision_score(agg_y_true, y_pred_55, zero_division=0)
    n_pred_55 = int(y_pred_55.sum())

    y_pred_50 = (agg_y_prob >= 0.50).astype(int)
    prec_50 = precision_score(agg_y_true, y_pred_50, zero_division=0)
    n_pred_50 = int(y_pred_50.sum())

    base_wr = agg_y_true.mean()

    print(f"""
  CLASSIFICATION PERFORMANCE:
    Aggregate AUC-ROC:       {agg_auc:.4f}  (0.50 = random, 1.00 = perfect)
    Baseline win rate:       {base_wr*100:.1f}%  (positive class prevalence)
    Precision @ 0.50:        {prec_50*100:.1f}%  (on {n_pred_50} predictions)
    Precision @ 0.55:        {prec_55*100:.1f}%  (on {n_pred_55} predictions)

  NON-LINEARITY:
    Depth=1 AUC:             {auc_d1:.4f}
    Best depth AUC:          {auc_best:.4f} (depth={best_depth})
    Interaction benefit:     {auc_improvement:+.4f}

  TOP 5 FEATURES:""")

    for i, row in avg_imp.head(5).iterrows():
        print(f"    {i+1}. {row['feature']} (avg gain={row['mean']:.1f})")

    print(f"\n  TOP 5 INTERACTIONS:")
    for i, row in avg_int.head(5).iterrows():
        print(f"    {i+1}. {row['feature_1']} x {row['feature_2']} "
              f"(co-occur={int(row['total_cooccurrence'])}, gain={row['total_gain']:.1f})")

    # Trading comparison
    sim_agg = simulate_trading(agg_y_prob, agg_r, agg_pf, 0.55, label="[FINAL]")
    base = sim_agg["baseline"]
    xgb_filt = sim_agg["xgb_AND_filters"]
    xgb_only = sim_agg["xgb_only"]

    print(f"""
  TRADING COMPARISON (threshold=0.55, all validation folds):
    Baseline (filters only):    {base['n_trades']} trades, {base['total_r']:.1f}R, WR={base['win_rate']}%, PF={base['pf']}, PPDD={base['ppdd']}
    XGB AND filters:            {xgb_filt['n_trades']} trades, {xgb_filt['total_r']:.1f}R, WR={xgb_filt['win_rate']}%, PF={xgb_filt['pf']}, PPDD={xgb_filt['ppdd']}
    XGB only (no filters):      {xgb_only['n_trades']} trades, {xgb_only['total_r']:.1f}R, WR={xgb_only['win_rate']}%, PF={xgb_only['pf']}, PPDD={xgb_only['ppdd']}
""")

    if xgb_filt["avg_r"] > base["avg_r"] and xgb_filt["n_trades"] > 0:
        print("  VERDICT: XGB adds value on top of rule-based filters (higher avg R)")
    elif xgb_only["avg_r"] > base["avg_r"] and xgb_only["n_trades"] > base["n_trades"]:
        print("  VERDICT: XGB can potentially replace some rule-based filters (better avg R, more trades)")
    else:
        print("  VERDICT: XGB does not improve over rule-based filters in trading performance")

    print(f"\n{'='*90}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
