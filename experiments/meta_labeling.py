"""
Meta-Labeling Classifier for NQ Quant System
=============================================
Lopez de Prado meta-labeling: given the primary model already generated a signal,
should we EXECUTE it?

Subset 1: Meta-label on traded signals (564 signals)
Subset 2: Expand the pool to ALL 15,894 signals

Walk-forward only, never shuffle time series.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss,
    classification_report
)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not available, using XGBoost only")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available")


# ─────────────────────────────────────────────────────────────
# 0. LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "signal_feature_database.parquet"
df_raw = pd.read_parquet(DATA_PATH)

# Create year column
df_raw["year"] = df_raw["bar_time_et"].dt.year

print("=" * 80)
print("META-LABELING CLASSIFIER — NQ QUANT")
print("=" * 80)
print(f"Total signals: {len(df_raw):,}")
print(f"Traded signals (passes_all_filters): {df_raw['passes_all_filters'].sum():,}")
print(f"Date range: {df_raw['bar_time_et'].min()} → {df_raw['bar_time_et'].max()}")
print()

# ─────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS
# ─────────────────────────────────────────────────────────────

# Outcome columns — NEVER use as features
OUTCOME_COLS = [
    "outcome_r", "outcome_label", "hit_tp", "hit_sl",
    "max_favorable_excursion", "max_adverse_excursion", "bars_to_outcome",
]

# Blocked columns — these encode the rule-based filter decisions
BLOCKED_COLS = [c for c in df_raw.columns if c.startswith("blocked_by_")]

# Meta / ID columns
META_COLS = [
    "bar_time_utc", "bar_time_et", "passes_all_filters",
    "entry_price", "model_stop", "irl_target",  # price levels, not features
    "year",
]

# String columns that need encoding
STRING_COLS = ["signal_type", "session", "sub_session", "grade"]

# All feature columns = everything not in exclude lists
EXCLUDE_COLS = set(OUTCOME_COLS + BLOCKED_COLS + META_COLS + STRING_COLS)
NUMERIC_FEATURES = [c for c in df_raw.select_dtypes(include=["number", "bool"]).columns
                    if c not in EXCLUDE_COLS]

print(f"Numeric features ({len(NUMERIC_FEATURES)}):")
for f in NUMERIC_FEATURES:
    print(f"  {f}")
print()


def prepare_features(df: pd.DataFrame, include_blocked: bool = False) -> pd.DataFrame:
    """Build feature matrix from raw dataframe."""
    feat = df[NUMERIC_FEATURES].copy()

    # Convert bools to int
    for c in feat.columns:
        if feat[c].dtype == bool:
            feat[c] = feat[c].astype(int)

    # Encode string columns
    for sc in STRING_COLS:
        dummies = pd.get_dummies(df[sc], prefix=sc, drop_first=True)
        for dc in dummies.columns:
            feat[dc] = dummies[dc].astype(int)

    # Optionally include blocked_by columns as features (for subset 2)
    if include_blocked:
        for bc in BLOCKED_COLS:
            feat[bc] = df[bc].astype(int)

    # Fill NaN
    feat = feat.fillna(0)

    return feat


def compute_metrics(outcome_r: pd.Series) -> dict:
    """Compute R, MaxDD (running), PPDD = totalR / MaxDD, PF."""
    total_r = outcome_r.sum()
    cum_r = outcome_r.cumsum()
    running_max = cum_r.cummax()
    drawdown = cum_r - running_max
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    gross_profit = outcome_r[outcome_r > 0].sum()
    gross_loss = abs(outcome_r[outcome_r <= 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    ppdd = total_r / max_dd if max_dd > 0 else float("inf")

    return {
        "n_trades": len(outcome_r),
        "total_R": round(total_r, 2),
        "avg_R": round(total_r / len(outcome_r), 4) if len(outcome_r) > 0 else 0,
        "win_rate": round((outcome_r > 0).mean() * 100, 1) if len(outcome_r) > 0 else 0,
        "max_DD": round(max_dd, 2),
        "PPDD": round(ppdd, 2),
        "PF": round(pf, 3),
    }


def print_metrics(label: str, metrics: dict):
    print(f"  {label:40s} | N={metrics['n_trades']:4d} | "
          f"R={metrics['total_R']:+8.2f} | avgR={metrics['avg_R']:+.4f} | "
          f"WR={metrics['win_rate']:5.1f}% | DD={metrics['max_DD']:6.2f} | "
          f"PPDD={metrics['PPDD']:6.2f} | PF={metrics['PF']:.3f}")


# ─────────────────────────────────────────────────────────────
# WALK-FORWARD ENGINE
# ─────────────────────────────────────────────────────────────

def walk_forward_split(df: pd.DataFrame, min_train_years: int = 3):
    """
    Walk-forward: train on all years up to year Y, test on year Y+1.
    Yields (train_mask, test_mask, test_year).
    """
    years = sorted(df["year"].unique())
    for i in range(min_train_years, len(years)):
        test_year = years[i]
        train_years = years[:i]
        train_mask = df["year"].isin(train_years)
        test_mask = df["year"] == test_year
        if train_mask.sum() > 10 and test_mask.sum() > 0:
            yield train_mask, test_mask, test_year


def train_and_predict(X_train, y_train, X_test, model_type="xgb"):
    """Train a model and return predicted probabilities on test set."""

    if model_type == "logistic":
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        model = LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced",
            penalty="l2", solver="lbfgs", random_state=42
        )
        model.fit(X_tr_s, y_train)
        proba = model.predict_proba(X_te_s)[:, 1]
        return proba, model

    elif model_type == "xgb" and HAS_XGB:
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=1.0,
            reg_lambda=5.0,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        return proba, model

    elif model_type == "lgb" and HAS_LGB:
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=1.0,
            reg_lambda=5.0,
            scale_pos_weight=scale_pos,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        return proba, model

    else:
        raise ValueError(f"Model type {model_type} not available")


# ─────────────────────────────────────────────────────────────
# 1. SUBSET 1: META-LABEL ON TRADED SIGNALS
# ─────────────────────────────────────────────────────────────

def run_subset1():
    print("\n" + "=" * 80)
    print("SUBSET 1: META-LABELING ON TRADED SIGNALS (passes_all_filters == True)")
    print("=" * 80)

    traded = df_raw[df_raw["passes_all_filters"] == True].copy()
    traded = traded.sort_values("bar_time_et").reset_index(drop=True)

    print(f"Traded signals: {len(traded)}")
    print(f"Win rate: {(traded['outcome_r'] > 0).mean():.1%}")
    print(f"Total R: {traded['outcome_r'].sum():.2f}")
    print()

    X = prepare_features(traded)
    y = (traded["outcome_r"] > 0).astype(int)
    feature_names = X.columns.tolist()

    # Baseline metrics
    baseline_metrics = compute_metrics(traded["outcome_r"])
    print("BASELINE (trade all 564):")
    print_metrics("All traded signals", baseline_metrics)
    print()

    # Run walk-forward for each model type
    model_types = ["logistic"]
    if HAS_XGB:
        model_types.append("xgb")
    if HAS_LGB:
        model_types.append("lgb")

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

    for mt in model_types:
        print(f"\n--- Model: {mt.upper()} ---")

        # Collect OOS predictions
        all_probas = pd.Series(dtype=float)
        all_indices = []
        all_feature_importances = []

        for train_mask, test_mask, test_year in walk_forward_split(traded):
            X_train = X.loc[train_mask]
            y_train = y.loc[train_mask]
            X_test = X.loc[test_mask]

            if len(X_train) < 20 or len(X_test) < 3:
                continue

            proba, model = train_and_predict(
                X_train.values, y_train.values, X_test.values, model_type=mt
            )
            proba_series = pd.Series(proba, index=X_test.index)
            all_probas = pd.concat([all_probas, proba_series])
            all_indices.extend(X_test.index.tolist())

            # Feature importance
            if mt == "logistic":
                imp = np.abs(model.coef_[0])
            elif mt == "xgb":
                imp = model.feature_importances_
            elif mt == "lgb":
                imp = model.feature_importances_
            all_feature_importances.append(imp)

        if len(all_probas) == 0:
            print("  No OOS predictions generated")
            continue

        # Align predictions with traded data
        oos_traded = traded.loc[all_indices].copy()
        oos_traded["meta_prob"] = all_probas.values
        oos_y = y.loc[all_indices]

        # Classification metrics
        oos_pred = (oos_traded["meta_prob"] > 0.5).astype(int)
        print(f"\n  OOS Classification Metrics (n={len(oos_y)}):")
        print(f"    AUC-ROC:     {roc_auc_score(oos_y, oos_traded['meta_prob']):.4f}")
        print(f"    Brier Score: {brier_score_loss(oos_y, oos_traded['meta_prob']):.4f}")
        print(f"    Accuracy:    {accuracy_score(oos_y, oos_pred):.4f}")
        print(f"    Precision:   {precision_score(oos_y, oos_pred, zero_division=0):.4f}")
        print(f"    Recall:      {recall_score(oos_y, oos_pred, zero_division=0):.4f}")
        print()

        # Trading metrics at each threshold
        print(f"  Trading Metrics by Threshold:")
        oos_baseline = compute_metrics(oos_traded["outcome_r"])
        print_metrics("OOS Baseline (all OOS trades)", oos_baseline)

        for thresh in thresholds:
            mask = oos_traded["meta_prob"] > thresh
            if mask.sum() < 3:
                print(f"  {'thresh=' + str(thresh):40s} | N={mask.sum():4d} | TOO FEW TRADES")
                continue
            filtered_r = oos_traded.loc[mask, "outcome_r"]
            m = compute_metrics(filtered_r)
            label = f"thresh={thresh:.2f}"
            print_metrics(label, m)

        # Per-year breakdown for best threshold candidate (0.50)
        print(f"\n  Per-Year Breakdown (thresh=0.50):")
        for year in sorted(oos_traded["year"].unique()):
            yr_mask = oos_traded["year"] == year
            yr_data = oos_traded[yr_mask]
            yr_all = compute_metrics(yr_data["outcome_r"])
            yr_filtered = yr_data[yr_data["meta_prob"] > 0.50]
            if len(yr_filtered) >= 1:
                yr_filt_m = compute_metrics(yr_filtered["outcome_r"])
                print(f"    {year}: Baseline N={yr_all['n_trades']}, R={yr_all['total_R']:+.2f} | "
                      f"Meta N={yr_filt_m['n_trades']}, R={yr_filt_m['total_R']:+.2f}, "
                      f"avgR={yr_filt_m['avg_R']:+.4f}")
            else:
                print(f"    {year}: Baseline N={yr_all['n_trades']}, R={yr_all['total_R']:+.2f} | "
                      f"Meta: 0 trades")

        # Feature importance
        if all_feature_importances:
            avg_imp = np.mean(all_feature_importances, axis=0)
            imp_df = pd.DataFrame({
                "feature": feature_names[:len(avg_imp)],
                "importance": avg_imp,
            }).sort_values("importance", ascending=False)

            print(f"\n  Top 15 Features ({mt.upper()}):")
            for _, row in imp_df.head(15).iterrows():
                bar = "#" * int(row["importance"] / imp_df["importance"].max() * 30)
                print(f"    {row['feature']:30s} {row['importance']:.4f}  {bar}")

        # Calibration check
        print(f"\n  Calibration Check ({mt.upper()}):")
        try:
            prob_true, prob_pred = calibration_curve(
                oos_y, oos_traded["meta_prob"], n_bins=5, strategy="uniform"
            )
            print(f"    {'Bin Range':20s} {'Predicted':>10s} {'Actual':>10s} {'Count':>8s}")
            bin_edges = np.linspace(0, 1, 6)
            for i in range(len(prob_true)):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                bin_mask = (oos_traded["meta_prob"] >= lo) & (oos_traded["meta_prob"] < hi)
                cnt = bin_mask.sum()
                print(f"    [{lo:.2f}, {hi:.2f})       {prob_pred[i]:10.3f} {prob_true[i]:10.3f} {cnt:8d}")
        except Exception as e:
            print(f"    Calibration failed: {e}")

    return oos_traded if "oos_traded" in dir() else None


# ─────────────────────────────────────────────────────────────
# 2. SUBSET 2: EXPAND THE POOL (ALL 15,894 SIGNALS)
# ─────────────────────────────────────────────────────────────

def run_subset2():
    print("\n" + "=" * 80)
    print("SUBSET 2: META-LABELING ON ALL SIGNALS (Expand the Pool)")
    print("=" * 80)

    all_signals = df_raw.copy()
    all_signals = all_signals.sort_values("bar_time_et").reset_index(drop=True)

    print(f"Total signals: {len(all_signals):,}")
    print(f"Overall win rate: {(all_signals['outcome_r'] > 0).mean():.1%}")
    print()

    # Features include blocked_by columns (model can learn which filters matter)
    X = prepare_features(all_signals, include_blocked=False)
    y = (all_signals["outcome_r"] > 0).astype(int)
    feature_names = X.columns.tolist()

    print(f"Features: {len(feature_names)}")
    print()

    # Baseline: current rule-based system
    traded_mask = all_signals["passes_all_filters"] == True
    baseline_r = all_signals.loc[traded_mask, "outcome_r"]
    baseline_metrics = compute_metrics(baseline_r)
    print("BASELINE (rule-based, 564 trades):")
    print_metrics("Rule-based system", baseline_metrics)
    print()

    model_types = ["logistic"]
    if HAS_XGB:
        model_types.append("xgb")
    if HAS_LGB:
        model_types.append("lgb")

    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

    for mt in model_types:
        print(f"\n--- Model: {mt.upper()} ---")

        all_probas = pd.Series(dtype=float)
        all_indices = []
        all_feature_importances = []

        for train_mask, test_mask, test_year in walk_forward_split(all_signals, min_train_years=3):
            X_train = X.loc[train_mask]
            y_train = y.loc[train_mask]
            X_test = X.loc[test_mask]

            if len(X_train) < 50 or len(X_test) < 10:
                continue

            proba, model = train_and_predict(
                X_train.values, y_train.values, X_test.values, model_type=mt
            )
            proba_series = pd.Series(proba, index=X_test.index)
            all_probas = pd.concat([all_probas, proba_series])
            all_indices.extend(X_test.index.tolist())

            if mt == "logistic":
                imp = np.abs(model.coef_[0])
            elif mt in ("xgb", "lgb"):
                imp = model.feature_importances_
            all_feature_importances.append(imp)

        if len(all_probas) == 0:
            print("  No OOS predictions generated")
            continue

        oos_all = all_signals.loc[all_indices].copy()
        oos_all["meta_prob"] = all_probas.values
        oos_y = y.loc[all_indices]

        # Classification metrics
        oos_pred = (oos_all["meta_prob"] > 0.5).astype(int)
        print(f"\n  OOS Classification Metrics (n={len(oos_y):,}):")
        print(f"    AUC-ROC:     {roc_auc_score(oos_y, oos_all['meta_prob']):.4f}")
        print(f"    Brier Score: {brier_score_loss(oos_y, oos_all['meta_prob']):.4f}")
        print(f"    Accuracy:    {accuracy_score(oos_y, oos_pred):.4f}")
        print()

        # Compare: OOS rule-based baseline vs ML-selected
        oos_traded_mask = oos_all["passes_all_filters"] == True
        oos_baseline = compute_metrics(oos_all.loc[oos_traded_mask, "outcome_r"])
        print(f"  OOS Rule-Based Baseline:")
        print_metrics("Rule-based (OOS period)", oos_baseline)

        print(f"\n  ML-Selected at Various Thresholds:")
        for thresh in thresholds:
            mask = oos_all["meta_prob"] > thresh
            if mask.sum() < 3:
                print(f"  {'thresh=' + str(thresh):40s} | N={mask.sum():4d} | TOO FEW TRADES")
                continue
            filtered_r = oos_all.loc[mask, "outcome_r"]
            m = compute_metrics(filtered_r)
            label = f"thresh={thresh:.2f}"
            print_metrics(label, m)

        # Key question: does ML find signals the rule-based system misses?
        print(f"\n  ML-Found Signals NOT in Rule-Based System (thresh=0.55):")
        ml_selected = oos_all["meta_prob"] > 0.55
        rule_selected = oos_all["passes_all_filters"] == True
        only_ml = ml_selected & ~rule_selected
        only_rules = rule_selected & ~ml_selected
        both = ml_selected & rule_selected
        neither = ~ml_selected & ~rule_selected

        print(f"    Both select:            {both.sum():5d} trades, "
              f"R={oos_all.loc[both, 'outcome_r'].sum():+8.2f}")
        print(f"    Only ML:                {only_ml.sum():5d} trades, "
              f"R={oos_all.loc[only_ml, 'outcome_r'].sum():+8.2f}")
        print(f"    Only rules:             {only_rules.sum():5d} trades, "
              f"R={oos_all.loc[only_rules, 'outcome_r'].sum():+8.2f}")
        print(f"    Neither:                {neither.sum():5d} trades")

        # Hybrid: combine rule-based AND ML
        hybrid_mask = rule_selected | (ml_selected & ~rule_selected)
        hybrid_metrics = compute_metrics(oos_all.loc[hybrid_mask, "outcome_r"])
        print(f"\n  Hybrid (rules OR ML@0.55):")
        print_metrics("Hybrid", hybrid_metrics)

        # Strict hybrid: rules AND ML
        strict_hybrid_mask = rule_selected & ml_selected
        if strict_hybrid_mask.sum() >= 3:
            strict_metrics = compute_metrics(oos_all.loc[strict_hybrid_mask, "outcome_r"])
            print(f"  Strict Hybrid (rules AND ML@0.55):")
            print_metrics("Strict Hybrid", strict_metrics)

        # Per-year breakdown
        print(f"\n  Per-Year: Rule-based vs ML@0.55 vs Strict Hybrid")
        for year in sorted(oos_all["year"].unique()):
            yr_mask = oos_all["year"] == year
            yr_data = oos_all[yr_mask]

            yr_rules = yr_data[yr_data["passes_all_filters"] == True]
            yr_ml = yr_data[yr_data["meta_prob"] > 0.55]
            yr_strict = yr_data[(yr_data["passes_all_filters"] == True) & (yr_data["meta_prob"] > 0.55)]

            r_rules = yr_rules["outcome_r"].sum() if len(yr_rules) > 0 else 0
            r_ml = yr_ml["outcome_r"].sum() if len(yr_ml) > 0 else 0
            r_strict = yr_strict["outcome_r"].sum() if len(yr_strict) > 0 else 0

            print(f"    {year}: Rules N={len(yr_rules):3d} R={r_rules:+7.2f} | "
                  f"ML N={len(yr_ml):4d} R={r_ml:+8.2f} | "
                  f"Strict N={len(yr_strict):3d} R={r_strict:+7.2f}")

        # Feature importance
        if all_feature_importances:
            avg_imp = np.mean(all_feature_importances, axis=0)
            imp_df = pd.DataFrame({
                "feature": feature_names[:len(avg_imp)],
                "importance": avg_imp,
            }).sort_values("importance", ascending=False)

            print(f"\n  Top 20 Features ({mt.upper()}):")
            for _, row in imp_df.head(20).iterrows():
                bar = "#" * int(row["importance"] / imp_df["importance"].max() * 30)
                print(f"    {row['feature']:30s} {row['importance']:.4f}  {bar}")

        # Calibration
        print(f"\n  Calibration Check ({mt.upper()}):")
        try:
            prob_true, prob_pred = calibration_curve(
                oos_y, oos_all["meta_prob"], n_bins=8, strategy="uniform"
            )
            print(f"    {'Bin Range':20s} {'Predicted':>10s} {'Actual':>10s} {'Count':>8s}")
            bin_edges = np.linspace(0, 1, 9)
            for i in range(len(prob_true)):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                bin_mask = (oos_all["meta_prob"] >= lo) & (oos_all["meta_prob"] < hi)
                cnt = bin_mask.sum()
                print(f"    [{lo:.2f}, {hi:.2f})       {prob_pred[i]:10.3f} {prob_true[i]:10.3f} {cnt:8d}")
        except Exception as e:
            print(f"    Calibration failed: {e}")


# ─────────────────────────────────────────────────────────────
# 3. SUBSET 2b: EXPAND POOL WITH BLOCKED_BY AS FEATURES
# ─────────────────────────────────────────────────────────────

def run_subset2b():
    print("\n" + "=" * 80)
    print("SUBSET 2b: ALL SIGNALS WITH BLOCKED_BY COLUMNS AS FEATURES")
    print("=" * 80)
    print("Can the model learn WHICH filters to override?")

    all_signals = df_raw.copy()
    all_signals = all_signals.sort_values("bar_time_et").reset_index(drop=True)

    X = prepare_features(all_signals, include_blocked=True)
    y = (all_signals["outcome_r"] > 0).astype(int)
    feature_names = X.columns.tolist()

    print(f"Features (incl blocked_by): {len(feature_names)}")

    if not HAS_XGB:
        print("XGBoost not available, skipping subset 2b")
        return

    mt = "xgb"
    all_probas = pd.Series(dtype=float)
    all_indices = []
    all_feature_importances = []

    for train_mask, test_mask, test_year in walk_forward_split(all_signals, min_train_years=3):
        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_test = X.loc[test_mask]

        if len(X_train) < 50 or len(X_test) < 10:
            continue

        proba, model = train_and_predict(
            X_train.values, y_train.values, X_test.values, model_type=mt
        )
        proba_series = pd.Series(proba, index=X_test.index)
        all_probas = pd.concat([all_probas, proba_series])
        all_indices.extend(X_test.index.tolist())
        all_feature_importances.append(model.feature_importances_)

    oos_all = all_signals.loc[all_indices].copy()
    oos_all["meta_prob"] = all_probas.values

    # Compare
    oos_traded_mask = oos_all["passes_all_filters"] == True
    oos_baseline = compute_metrics(oos_all.loc[oos_traded_mask, "outcome_r"])
    print(f"\n  OOS Rule-Based Baseline:")
    print_metrics("Rule-based (OOS)", oos_baseline)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    print(f"\n  ML-Selected (with blocked_by features):")
    for thresh in thresholds:
        mask = oos_all["meta_prob"] > thresh
        if mask.sum() < 3:
            continue
        m = compute_metrics(oos_all.loc[mask, "outcome_r"])
        print_metrics(f"thresh={thresh:.2f}", m)

    # Which blocked_by columns matter?
    if all_feature_importances:
        avg_imp = np.mean(all_feature_importances, axis=0)
        imp_df = pd.DataFrame({
            "feature": feature_names[:len(avg_imp)],
            "importance": avg_imp,
        }).sort_values("importance", ascending=False)

        print(f"\n  Blocked_by Feature Importances:")
        blocked_imp = imp_df[imp_df["feature"].str.startswith("blocked_by")]
        for _, row in blocked_imp.iterrows():
            bar = "#" * int(row["importance"] / imp_df["importance"].max() * 30)
            print(f"    {row['feature']:30s} {row['importance']:.4f}  {bar}")

        print(f"\n  Top 20 Features Overall:")
        for _, row in imp_df.head(20).iterrows():
            bar = "#" * int(row["importance"] / imp_df["importance"].max() * 30)
            print(f"    {row['feature']:30s} {row['importance']:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────
# 4. ENSEMBLE: COMBINE MULTIPLE MODELS
# ─────────────────────────────────────────────────────────────

def run_ensemble_subset1():
    """Ensemble of logistic + XGB on traded signals for robustness."""
    if not HAS_XGB:
        print("\nSkipping ensemble (XGBoost not available)")
        return

    print("\n" + "=" * 80)
    print("ENSEMBLE: LOGISTIC + XGB ON TRADED SIGNALS")
    print("=" * 80)

    traded = df_raw[df_raw["passes_all_filters"] == True].copy()
    traded = traded.sort_values("bar_time_et").reset_index(drop=True)

    X = prepare_features(traded)
    y = (traded["outcome_r"] > 0).astype(int)

    all_proba_lr = pd.Series(dtype=float)
    all_proba_xgb = pd.Series(dtype=float)
    all_indices = []

    for train_mask, test_mask, test_year in walk_forward_split(traded):
        X_train = X.loc[train_mask].values
        y_train = y.loc[train_mask].values
        X_test = X.loc[test_mask].values

        if len(X_train) < 20 or len(X_test) < 3:
            continue

        proba_lr, _ = train_and_predict(X_train, y_train, X_test, "logistic")
        proba_xgb, _ = train_and_predict(X_train, y_train, X_test, "xgb")

        idx = X.loc[test_mask].index
        all_proba_lr = pd.concat([all_proba_lr, pd.Series(proba_lr, index=idx)])
        all_proba_xgb = pd.concat([all_proba_xgb, pd.Series(proba_xgb, index=idx)])
        all_indices.extend(idx.tolist())

    if len(all_indices) == 0:
        print("No predictions generated")
        return

    oos = traded.loc[all_indices].copy()
    oos["prob_lr"] = all_proba_lr.values
    oos["prob_xgb"] = all_proba_xgb.values
    oos["prob_ensemble"] = 0.5 * oos["prob_lr"] + 0.5 * oos["prob_xgb"]

    # Agreement-based: only trade when BOTH models agree
    thresholds = [0.45, 0.50, 0.55, 0.60]
    print("\n  Ensemble Average:")
    oos_baseline = compute_metrics(oos["outcome_r"])
    print_metrics("Baseline (all OOS)", oos_baseline)

    for t in thresholds:
        mask = oos["prob_ensemble"] > t
        if mask.sum() >= 3:
            m = compute_metrics(oos.loc[mask, "outcome_r"])
            print_metrics(f"Ensemble avg > {t}", m)

    print("\n  Agreement (both > threshold):")
    for t in thresholds:
        mask = (oos["prob_lr"] > t) & (oos["prob_xgb"] > t)
        if mask.sum() >= 3:
            m = compute_metrics(oos.loc[mask, "outcome_r"])
            print_metrics(f"Both > {t}", m)

    # Correlation between models
    corr = oos[["prob_lr", "prob_xgb"]].corr().iloc[0, 1]
    print(f"\n  Correlation between LR and XGB probabilities: {corr:.4f}")


# ─────────────────────────────────────────────────────────────
# 5. DEEP DIVE: SIGNAL CHARACTERISTICS OF META-MODEL PICKS
# ─────────────────────────────────────────────────────────────

def signal_characteristics_analysis():
    """Analyze what kind of signals the meta-model selects vs rejects."""
    if not HAS_XGB:
        return

    print("\n" + "=" * 80)
    print("SIGNAL CHARACTERISTICS: META-SELECTED vs META-REJECTED")
    print("=" * 80)

    traded = df_raw[df_raw["passes_all_filters"] == True].copy()
    traded = traded.sort_values("bar_time_et").reset_index(drop=True)

    X = prepare_features(traded)
    y = (traded["outcome_r"] > 0).astype(int)

    # Collect OOS predictions
    all_probas = pd.Series(dtype=float)
    all_indices = []

    for train_mask, test_mask, test_year in walk_forward_split(traded):
        X_train = X.loc[train_mask].values
        y_train = y.loc[train_mask].values
        X_test = X.loc[test_mask].values

        if len(X_train) < 20 or len(X_test) < 3:
            continue

        proba, _ = train_and_predict(X_train, y_train, X_test, "xgb")
        idx = X.loc[test_mask].index
        all_probas = pd.concat([all_probas, pd.Series(proba, index=idx)])
        all_indices.extend(idx.tolist())

    oos = traded.loc[all_indices].copy()
    oos["meta_prob"] = all_probas.values

    selected = oos[oos["meta_prob"] > 0.55]
    rejected = oos[oos["meta_prob"] <= 0.55]

    compare_cols = [
        "fvg_size_atr", "fvg_sweep_score", "bar_body_ratio", "bar_body_atr",
        "fluency_score", "signal_quality", "atr_percentile", "stop_distance_atr",
        "target_rr", "pa_alt_dir_ratio", "bias_confidence", "hour_et", "day_of_week"
    ]

    print(f"\n  N selected: {len(selected)}, N rejected: {len(rejected)}")
    print(f"\n  {'Feature':30s} {'Selected':>12s} {'Rejected':>12s} {'Delta':>10s}")
    print("  " + "-" * 70)
    for col in compare_cols:
        if col in oos.columns:
            sel_mean = selected[col].mean()
            rej_mean = rejected[col].mean()
            delta = sel_mean - rej_mean
            print(f"  {col:30s} {sel_mean:12.4f} {rej_mean:12.4f} {delta:+10.4f}")

    # Hour distribution
    print(f"\n  Hour Distribution (ET):")
    print(f"  {'Hour':>6s} {'Selected':>10s} {'Rejected':>10s}")
    for h in range(6, 17):
        sel_n = (selected["hour_et"] == h).sum()
        rej_n = (rejected["hour_et"] == h).sum()
        sel_pct = sel_n / len(selected) * 100 if len(selected) > 0 else 0
        rej_pct = rej_n / len(rejected) * 100 if len(rejected) > 0 else 0
        print(f"  {h:6d} {sel_pct:9.1f}% {rej_pct:9.1f}%")

    # Day of week
    print(f"\n  Day of Week Distribution:")
    days = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    for d, name in days.items():
        sel_n = (selected["day_of_week"] == d).sum()
        rej_n = (rejected["day_of_week"] == d).sum()
        sel_pct = sel_n / len(selected) * 100 if len(selected) > 0 else 0
        rej_pct = rej_n / len(rejected) * 100 if len(rejected) > 0 else 0
        print(f"  {name:>6s} {sel_pct:9.1f}% {rej_pct:9.1f}%")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_subset1()
    run_subset2()
    run_subset2b()
    run_ensemble_subset1()
    signal_characteristics_analysis()

    print("\n" + "=" * 80)
    print("META-LABELING ANALYSIS COMPLETE")
    print("=" * 80)
