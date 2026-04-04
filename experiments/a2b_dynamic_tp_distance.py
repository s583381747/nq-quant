"""
experiments/a2b_dynamic_tp_distance.py — Dynamic TP Distance Analysis

Analyzes whether TP distance should vary by signal context using the
signal_feature_database (15,894 signals x 58 cols).

Key insight: TP distance in the real backtest controls TRIM POINT, not final exit.
After trim at TP1, the remaining position trails. So TP controls when you lock in
partial profit, not the total R. Higher TP = bigger trim R but lower trim hit rate.

Parts:
  1. MFE Analysis by Signal Features (all signals + traded-only)
  2. Optimal TP Distance by Group (E[R] sweep, traded signals focus)
  3. XGBoost TP Distance Regression Model (walk-forward)
  4. Rules Extraction (simple lookup table)
  5. Practical Backtest Test (using validate_improvements engine)

Usage: python experiments/a2b_dynamic_tp_distance.py
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

DATA = PROJECT / "data"


# ======================================================================
# Utility
# ======================================================================
def fmt_metrics(m: dict) -> str:
    return (f"trades={m['trades']}  R={m['R']:+.1f}  PPDD={m['PPDD']:.2f}  "
            f"PF={m['PF']:.2f}  WR={m['WR']:.1f}%  MaxDD={m['MaxDD']:.1f}")


def compute_metrics(r_arr: np.ndarray) -> dict:
    if len(r_arr) == 0:
        return {"trades": 0, "R": 0.0, "PPDD": 0.0, "PF": 0.0, "WR": 0.0, "MaxDD": 0.0, "avgR": 0.0}
    total_r = float(r_arr.sum())
    wr = float((r_arr > 0).mean() * 100)
    wins = float(r_arr[r_arr > 0].sum())
    losses = float(abs(r_arr[r_arr < 0].sum()))
    pf = wins / losses if losses > 0 else 999.0
    cumr = np.cumsum(r_arr)
    peak = np.maximum.accumulate(cumr)
    dd = peak - cumr
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0
    ppdd = total_r / max_dd if max_dd > 0 else 999.0
    return {
        "trades": len(r_arr),
        "R": round(total_r, 2),
        "PPDD": round(ppdd, 2),
        "PF": round(pf, 2),
        "WR": round(wr, 1),
        "MaxDD": round(max_dd, 1),
        "avgR": round(total_r / len(r_arr), 4),
    }


# ======================================================================
# Part 1: MFE Analysis by Signal Features
# ======================================================================
def part1_mfe_analysis(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("PART 1: MFE ANALYSIS BY SIGNAL FEATURES")
    print("=" * 80)

    # MFE is in R multiples in the database
    # Convert to ATR multiples: mfe_atr = mfe_R * stop_distance_atr
    df = df.copy()
    df["mfe_atr"] = df["max_favorable_excursion"] * df["stop_distance_atr"]
    df["mfe_pts"] = df["max_favorable_excursion"] * df["stop_distance_pts"]

    # Bins
    df["fluency_bin"] = pd.cut(df["fluency_score"],
                                bins=[-0.01, 0.5, 0.7, 1.01],
                                labels=["low(<0.5)", "mid(0.5-0.7)", "high(>0.7)"])
    df["atr_pctl_bin"] = pd.cut(df["atr_percentile"].fillna(0.5),
                                 bins=[-0.01, 0.33, 0.66, 1.01],
                                 labels=["low_vol", "mid_vol", "high_vol"])
    df["hour_bin"] = pd.cut(df["hour_et"],
                             bins=[9, 12, 14, 16],
                             labels=["10-12", "12-14", "14-16"],
                             right=False)

    for population, label in [(df, "ALL SIGNALS (n=15,894)"),
                               (df[df["passes_all_filters"]], "TRADED SIGNALS (passes_all_filters)")]:
        pop = population.copy()
        print(f"\n{'='*60}")
        print(f"  {label}: n={len(pop)}")
        print(f"{'='*60}")

        if len(pop) < 10:
            print("  Too few signals, skipping")
            continue

        print(f"\n  MFE (ATR multiples) stats:")
        print(f"    Mean:   {pop['mfe_atr'].mean():.3f}")
        print(f"    Median: {pop['mfe_atr'].median():.3f}")
        print(f"    P75:    {pop['mfe_atr'].quantile(0.75):.3f}")
        print(f"    P90:    {pop['mfe_atr'].quantile(0.90):.3f}")
        print(f"    P95:    {pop['mfe_atr'].quantile(0.95):.3f}")

        print(f"\n  MFE (R multiples) stats:")
        print(f"    Mean:   {pop['max_favorable_excursion'].mean():.3f}")
        print(f"    Median: {pop['max_favorable_excursion'].median():.3f}")
        print(f"    P75:    {pop['max_favorable_excursion'].quantile(0.75):.3f}")
        print(f"    P90:    {pop['max_favorable_excursion'].quantile(0.90):.3f}")

        group_dims = [
            ("signal_type", "signal_type"),
            ("grade", "grade"),
            ("fluency_bin", "fluency_bin"),
            ("hour_bin", "hour_bin"),
            ("bias_aligned", "bias_aligned"),
        ]

        for dim_name, col in group_dims:
            grp = pop.dropna(subset=[col]).groupby(col)["mfe_atr"]
            stats = grp.agg(["count", "median",
                             lambda x: x.quantile(0.75),
                             lambda x: x.quantile(0.90)])
            stats.columns = ["count", "median_mfe_atr", "p75_mfe_atr", "p90_mfe_atr"]

            # Also show MFE in R
            grp_r = pop.dropna(subset=[col]).groupby(col)["max_favorable_excursion"]
            stats_r = grp_r.agg(["median", lambda x: x.quantile(0.75)])
            stats_r.columns = ["median_mfe_R", "p75_mfe_R"]

            combined = stats.join(stats_r)
            combined = combined.round(3)
            print(f"\n  --- MFE by {dim_name} ---")
            print(f"  {combined.to_string()}")

        # Cross: signal_type x grade
        print(f"\n  --- MFE by signal_type x grade ---")
        cross = pop.groupby(["signal_type", "grade"]).agg(
            count=("mfe_atr", "count"),
            median_mfe_atr=("mfe_atr", "median"),
            p75_mfe_atr=("mfe_atr", lambda x: x.quantile(0.75)),
            median_mfe_R=("max_favorable_excursion", "median"),
            p75_mfe_R=("max_favorable_excursion", lambda x: x.quantile(0.75)),
            avg_outcome_r=("outcome_r", "mean"),
            tp_hit_rate=("hit_tp", "mean"),
        ).round(3)
        print(f"  {cross.to_string()}")

    return df


# ======================================================================
# Part 2: Optimal TP Distance by Group
# ======================================================================
def part2_optimal_tp_by_group(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("PART 2: OPTIMAL TP DISTANCE BY GROUP")
    print("=" * 80)
    print("NOTE: In the real backtest, TP1 is the TRIM point, not final exit.")
    print("After trim, remaining position trails with swing stops.")
    print("So 'optimal TP' here means 'optimal trim point'.")
    print()
    print("Model: At TP1 hit -> trim 50%, lock profit = 0.5 * tp_R")
    print("       Remaining 50% trails -> assume avg trail R from actual data")
    print("       At TP1 miss -> SL hit -> R = -1.0")

    tp_levels = np.arange(0.5, 10.5, 0.5)

    # For the trim+trail model we need to estimate trail R
    # From actual traded signals, compute average trail outcome for trimmed trades
    traded = df[df["passes_all_filters"]].copy()
    # MFE beyond TP tells us about trail potential
    # For now, use a simple model: trail R = MFE_R - tp_R (capped at 0)

    def compute_optimal_tp_trim_model(sub_df: pd.DataFrame, label: str) -> dict:
        """
        Trim+trail model:
        If MFE_R >= tp_R:
            - Trim 50% at tp_R -> trim_R = 0.5 * tp_R
            - Remaining 50% trails: assume BE at worst (trim already locked in)
            - Total R ~ 0.5 * tp_R + 0.5 * max(0, remaining_trail)
            - Simplified: total R ~ 0.5 * tp_R (conservative, assuming BE on remainder)
        If MFE_R < tp_R:
            - SL hit: R = -1.0
        """
        mfe_r = sub_df["max_favorable_excursion"].values
        valid = ~np.isnan(mfe_r)
        if valid.sum() < 10:
            return {"label": label, "n": int(valid.sum()), "results": []}

        mfe = mfe_r[valid]
        n = len(mfe)
        results = []

        for tp_r in tp_levels:
            hits = mfe >= tp_r
            hr = float(hits.mean())

            # Conservative model: trimmed half locks in tp_R, BE on remainder
            # R_if_hit = 0.5 * tp_R (trim profit only)
            # More realistic: use actual MFE beyond TP to estimate trail
            excess_mfe = np.maximum(mfe[hits] - tp_r, 0) if hits.any() else np.array([0])
            avg_trail_r = float(excess_mfe.mean()) if len(excess_mfe) > 0 else 0

            # Total R if hit: trim portion + trail portion (capped at 0 for BE stop)
            # Trim: 0.5 * tp_R
            # Trail: 0.5 * min(avg_trail_r, MFE_beyond_tp) -- but BE at worst = 0
            r_if_hit = 0.5 * tp_r + 0.5 * 0.0  # Conservative: BE
            r_if_hit_optimistic = 0.5 * tp_r + 0.5 * avg_trail_r  # With trail upside

            er_conservative = hr * r_if_hit - (1 - hr) * 1.0
            er_optimistic = hr * r_if_hit_optimistic - (1 - hr) * 1.0

            results.append({
                "tp_r": float(tp_r),
                "hit_rate": hr,
                "er_conservative": round(float(er_conservative), 4),
                "er_optimistic": round(float(er_optimistic), 4),
                "avg_trail_r": round(float(avg_trail_r), 3),
            })

        return {"label": label, "n": n, "results": results}

    # Analyze for different populations
    for pop_label, pop in [("ALL signals", df), ("TRADED signals", traded)]:
        print(f"\n{'='*60}")
        print(f"  {pop_label} (n={len(pop)})")
        print(f"{'='*60}")

        key_groups = {
            "ALL":       pop,
            "trend":     pop[pop["signal_type"] == "trend"],
            "mss":       pop[pop["signal_type"] == "mss"],
            "trend_A+":  pop[(pop["signal_type"] == "trend") & (pop["grade"] == "A+")],
            "trend_B+":  pop[(pop["signal_type"] == "trend") & (pop["grade"] == "B+")],
            "trend_C":   pop[(pop["signal_type"] == "trend") & (pop["grade"] == "C")],
            "mss_A+":    pop[(pop["signal_type"] == "mss") & (pop["grade"] == "A+")],
            "mss_B+":    pop[(pop["signal_type"] == "mss") & (pop["grade"] == "B+")],
            "mss_C":     pop[(pop["signal_type"] == "mss") & (pop["grade"] == "C")],
        }

        for name, sub in key_groups.items():
            res = compute_optimal_tp_trim_model(sub, name)
            if res["n"] < 10:
                continue

            # Find optimal (conservative and optimistic)
            best_cons = max(res["results"], key=lambda x: x["er_conservative"])
            best_opt = max(res["results"], key=lambda x: x["er_optimistic"])

            print(f"\n  {name} (n={res['n']}):")
            print(f"    Best conservative: TP={best_cons['tp_r']:.1f}R  E[R]={best_cons['er_conservative']:+.4f}  HR={best_cons['hit_rate']:.3f}")
            print(f"    Best optimistic:   TP={best_opt['tp_r']:.1f}R  E[R]={best_opt['er_optimistic']:+.4f}  HR={best_opt['hit_rate']:.3f}")

        # Show full sweep for key traded groups
        if pop_label == "TRADED signals":
            print(f"\n  --- Full TP Sweep (Trim+Trail Model) for traded signals ---")
            for name in ["ALL", "trend_A+", "trend_B+", "mss_B+", "mss_C"]:
                sub = key_groups.get(name, pd.DataFrame())
                if len(sub) < 10:
                    continue
                res = compute_optimal_tp_trim_model(sub, name)
                print(f"\n  {name} (n={res['n']}):")
                print(f"  {'TP_R':>6s}  {'HitRate':>8s}  {'E[R]_cons':>10s}  {'E[R]_opt':>10s}  {'AvgTrailR':>10s}")
                for row in res["results"]:
                    bc = max(res["results"], key=lambda x: x["er_conservative"])
                    bo = max(res["results"], key=lambda x: x["er_optimistic"])
                    marker = ""
                    if abs(row["tp_r"] - bc["tp_r"]) < 0.01:
                        marker += " <-CONS"
                    if abs(row["tp_r"] - bo["tp_r"]) < 0.01:
                        marker += " <-OPT"
                    print(f"  {row['tp_r']:6.1f}  {row['hit_rate']:8.3f}  "
                          f"{row['er_conservative']:+10.4f}  {row['er_optimistic']:+10.4f}  "
                          f"{row['avg_trail_r']:10.3f}{marker}")

    return pd.DataFrame()


# ======================================================================
# Part 3: XGBoost MFE Prediction Model
# ======================================================================
def part3_xgb_model(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("PART 3: XGBoost MFE Prediction Model (Walk-Forward)")
    print("=" * 80)

    try:
        import xgboost as xgb
        from sklearn.metrics import mean_absolute_error, r2_score
    except ImportError:
        print("  [SKIP] xgboost or sklearn not installed")
        return

    df = df.copy()
    df["signal_type_enc"] = (df["signal_type"] == "trend").astype(int)
    df["grade_enc"] = df["grade"].map({"A+": 2, "B+": 1, "C": 0}).fillna(0).astype(int)
    df["bias_aligned_enc"] = df["bias_aligned"].astype(int)
    df["regime_enc"] = df["regime"].fillna(0.5)
    df["has_smt_enc"] = df["has_smt"].astype(int)
    df["is_displaced_enc"] = df["is_displaced"].astype(int)

    feature_cols = [
        "signal_dir", "fluency_score", "signal_quality", "atr_14", "atr_percentile",
        "stop_distance_atr", "bar_body_ratio", "bar_body_atr", "bar_range_atr",
        "fvg_size_atr", "target_rr", "pa_alt_dir_ratio",
        "hour_et", "day_of_week", "bias_confidence",
        "signal_type_enc", "grade_enc", "bias_aligned_enc", "regime_enc",
        "has_smt_enc", "is_displaced_enc",
    ]

    target_col = "mfe_atr"

    valid = df[feature_cols + [target_col]].dropna()
    valid = valid[valid[target_col] >= 0]
    print(f"  Valid samples: {len(valid)} / {len(df)}")

    df_time = df.loc[valid.index].copy()
    df_time["year"] = pd.to_datetime(df_time["bar_time_et"]).dt.year

    train_mask = df_time["year"].isin(range(2016, 2022))
    val_mask = df_time["year"].isin([2022, 2023])
    test_mask = df_time["year"].isin([2024, 2025, 2026])

    X_train = valid.loc[train_mask[train_mask].index, feature_cols]
    y_train = valid.loc[train_mask[train_mask].index, target_col]
    X_val = valid.loc[val_mask[val_mask].index, feature_cols]
    y_val = valid.loc[val_mask[val_mask].index, target_col]
    X_test = valid.loc[test_mask[test_mask].index, feature_cols]
    y_test = valid.loc[test_mask[test_mask].index, target_col]

    print(f"  Train: {len(X_train)} (2016-2021), Val: {len(X_val)} (2022-2023), Test: {len(X_test)} (2024-2026)")

    if len(X_train) < 100 or len(X_val) < 50:
        print("  [SKIP] Insufficient data")
        return

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test) if len(X_test) > 0 else np.array([])

    print(f"\n  {'':10s}  {'MAE':>8s}  {'R2':>8s}")
    print(f"  {'Train':10s}  {mean_absolute_error(y_train, pred_train):8.3f}  {r2_score(y_train, pred_train):8.4f}")
    print(f"  {'Val':10s}  {mean_absolute_error(y_val, pred_val):8.3f}  {r2_score(y_val, pred_val):8.4f}")
    if len(X_test) > 0:
        print(f"  {'Test':10s}  {mean_absolute_error(y_test, pred_test):8.3f}  {r2_score(y_test, pred_test):8.4f}")

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\n  Top 10 feature importances:")
    for feat, imp in importances.head(10).items():
        print(f"    {feat:25s}  {imp:.4f}")

    # Practical check on validation set
    print(f"\n  Predicted MFE quartile analysis (validation):")
    val_df = df_time.loc[X_val.index].copy()
    val_df["pred_mfe"] = pred_val
    for q_label, q_lo, q_hi in [("Q1 (low)", 0, 0.25), ("Q2", 0.25, 0.5),
                                  ("Q3", 0.5, 0.75), ("Q4 (high)", 0.75, 1.0)]:
        lo = val_df["pred_mfe"].quantile(q_lo)
        hi = val_df["pred_mfe"].quantile(q_hi)
        mask = (val_df["pred_mfe"] >= lo) & (val_df["pred_mfe"] < hi + 0.001)
        sub = val_df.loc[mask]
        if len(sub) > 0:
            print(f"    {q_label:12s}  n={len(sub):4d}  pred_mfe={sub['pred_mfe'].mean():.2f}  "
                  f"actual_mfe={sub['mfe_atr'].mean():.2f}  outcome_r={sub['outcome_r'].mean():+.3f}  "
                  f"tp_hit={sub['hit_tp'].mean():.3f}")

    # Key conclusion
    print(f"\n  R2 on validation = {r2_score(y_val, pred_val):.4f}")
    print(f"  Interpretation: XGBoost can explain ~{r2_score(y_val, pred_val)*100:.1f}% of MFE variance.")
    print(f"  MFE is largely stochastic -- limited predictability from signal features alone.")


# ======================================================================
# Part 4: Rules Extraction
# ======================================================================
def part4_rules_extraction(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 80)
    print("PART 4: RULES EXTRACTION")
    print("=" * 80)

    # Focus on traded signals for practical rules
    traded = df[df["passes_all_filters"]].copy()
    if len(traded) < 20:
        print("  Too few traded signals for rules extraction")
        return {}

    tp_levels = np.arange(0.5, 10.5, 0.5)

    def find_best_tp_r(mfe_r_arr: np.ndarray, mode: str = "optimistic") -> tuple:
        """Find TP (in R) that maximizes E[R] under trim+trail model."""
        n = len(mfe_r_arr)
        if n < 5:
            return (np.nan, np.nan, np.nan)
        best_er = -999.0
        best_tp = 1.0
        best_hr = 0.0
        for tp_r in tp_levels:
            hits = mfe_r_arr >= tp_r
            hr = float(hits.mean())
            if mode == "optimistic":
                excess = np.maximum(mfe_r_arr[hits] - tp_r, 0).mean() if hits.any() else 0
                r_hit = 0.5 * tp_r + 0.5 * excess
            else:
                r_hit = 0.5 * tp_r  # Conservative: trail = BE
            er = hr * r_hit - (1 - hr) * 1.0
            if er > best_er:
                best_er = er
                best_tp = tp_r
                best_hr = hr
        return (best_tp, best_er, best_hr)

    # Rules by signal_type x grade
    print(f"\n  TRADED SIGNALS — Optimal TP by signal_type x grade")
    print(f"  {'Group':20s}  {'N':>5s}  {'Opt_TP_R':>9s}  {'E[R]':>8s}  {'HR':>6s}  {'MedianMFE_R':>12s}")
    print("  " + "-" * 70)

    simple_rules_r = {}
    for st in ["trend", "mss"]:
        for gr in ["A+", "B+", "C"]:
            sub = traded[(traded["signal_type"] == st) & (traded["grade"] == gr)]
            key = f"{st}_{gr}"
            if len(sub) < 5:
                simple_rules_r[key] = 2.0
                print(f"  {key:20s}  {len(sub):5d}  {'N/A':>9s}  {'N/A':>8s}  {'N/A':>6s}  {'N/A':>12s}  (fallback: 2.0R)")
                continue
            mfe_r = sub["max_favorable_excursion"].values
            valid = ~np.isnan(mfe_r)
            mfe = mfe_r[valid]
            tp, er, hr = find_best_tp_r(mfe, mode="optimistic")
            med_mfe = float(np.median(mfe))
            simple_rules_r[key] = float(tp) if not np.isnan(tp) else 2.0
            print(f"  {key:20s}  {len(sub):5d}  {tp:9.1f}  {er:+8.4f}  {hr:6.3f}  {med_mfe:12.3f}")

    # Now convert R-based rules to ny_tp_mult equivalents
    # In the backtest: TP_distance = ny_tp_mult * IRL_target_distance
    # And: target_rr = IRL_target_distance / stop_distance
    # So: TP_R = ny_tp_mult * target_rr
    # Therefore: ny_tp_mult = TP_R / target_rr
    print(f"\n  Converting TP_R rules to ny_tp_mult equivalents:")
    print(f"  (ny_tp_mult = optimal_TP_R / median_target_rr_for_group)")
    print()

    simple_rules_mult = {}
    for st in ["trend", "mss"]:
        for gr in ["A+", "B+", "C"]:
            key = f"{st}_{gr}"
            sub = traded[(traded["signal_type"] == st) & (traded["grade"] == gr)]
            if len(sub) < 5:
                simple_rules_mult[key] = 3.0
                continue
            med_target_rr = sub["target_rr"].median()
            tp_r = simple_rules_r[key]
            if med_target_rr > 0 and not np.isnan(tp_r):
                mult = tp_r / med_target_rr
                simple_rules_mult[key] = round(float(mult), 1)
                print(f"    {key:20s}  TP_R={tp_r:.1f}  target_rr_median={med_target_rr:.2f}  -> ny_tp_mult={mult:.1f}")
            else:
                simple_rules_mult[key] = 3.0
                print(f"    {key:20s}  fallback ny_tp_mult=3.0")

    # Additional analysis: by time of day
    print(f"\n  TRADED SIGNALS — Optimal TP by hour")
    for hr_label, hr_lo, hr_hi in [("AM(10-12)", 10, 12), ("Lunch(12-14)", 12, 14), ("PM(14-16)", 14, 16)]:
        sub = traded[(traded["hour_et"] >= hr_lo) & (traded["hour_et"] < hr_hi)]
        if len(sub) < 5:
            continue
        mfe_r = sub["max_favorable_excursion"].dropna().values
        tp, er, hr = find_best_tp_r(mfe_r, mode="optimistic")
        print(f"    {hr_label:15s}  n={len(sub):4d}  opt_TP={tp:.1f}R  E[R]={er:+.4f}  HR={hr:.3f}")

    # By bias alignment
    print(f"\n  TRADED SIGNALS — Optimal TP by bias alignment")
    for ba_label, ba in [("aligned", True), ("opposing", False)]:
        sub = traded[traded["bias_aligned"] == ba]
        if len(sub) < 5:
            continue
        mfe_r = sub["max_favorable_excursion"].dropna().values
        tp, er, hr = find_best_tp_r(mfe_r, mode="optimistic")
        print(f"    {ba_label:15s}  n={len(sub):4d}  opt_TP={tp:.1f}R  E[R]={er:+.4f}  HR={hr:.3f}")

    return simple_rules_mult


# ======================================================================
# Part 5: Practical Backtest Test
# ======================================================================
def part5_practical_test(df: pd.DataFrame, simple_rules_mult: dict):
    print("\n" + "=" * 80)
    print("PART 5: PRACTICAL BACKTEST TEST")
    print("=" * 80)
    print("Running the ACTUAL backtest engine with different TP configurations")
    print("Using validate_improvements.py infrastructure")

    from experiments.validate_improvements import load_all, run_backtest_improved, compute_metrics as vi_compute_metrics

    print("\nLoading full backtest data...")
    t0 = _time.perf_counter()
    d = load_all()
    print(f"  Loaded in {_time.perf_counter() - t0:.1f}s")

    # Baseline: current config (ny_tp_mult=3.0)
    print(f"\n--- Baseline: ny_tp_mult=3.0 (current config) ---")
    trades_baseline = run_backtest_improved(d)
    m_baseline = vi_compute_metrics(trades_baseline)
    print(f"  {fmt_metrics(m_baseline)}")

    # Now run with different fixed ny_tp_mult values
    # To do this, we modify the params in d before calling run_backtest_improved
    import copy
    import yaml

    def run_with_tp_mult(d_orig: dict, ny_mult: float) -> list:
        """Run backtest with a specific ny_tp_multiplier."""
        d_mod = dict(d_orig)  # Shallow copy
        params = copy.deepcopy(d_orig["params"])
        params["session_rules"]["ny_tp_multiplier"] = ny_mult
        d_mod["params"] = params
        return run_backtest_improved(d_mod)

    print(f"\n--- Fixed TP multiplier sweep ---")
    print(f"  {'ny_tp_mult':>12s}  {'Trades':>6s}  {'R':>8s}  {'PPDD':>7s}  {'PF':>6s}  {'WR':>6s}  {'MaxDD':>7s}  {'AvgR':>8s}")
    print("  " + "-" * 70)

    sweep_results = {}
    for mult in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]:
        trades = run_with_tp_mult(d, mult)
        m = vi_compute_metrics(trades)
        sweep_results[mult] = m
        marker = " <-- CURRENT" if mult == 3.0 else ""
        print(f"  {mult:12.1f}  {m['trades']:6d}  {m['R']:+8.1f}  {m['PPDD']:7.2f}  "
              f"{m['PF']:6.2f}  {m['WR']:5.1f}%  {m['MaxDD']:7.1f}  {m['avgR']:+8.4f}{marker}")

    # Find best
    best_mult = max(sweep_results.keys(), key=lambda k: sweep_results[k]["R"])
    best_ppdd = max(sweep_results.keys(), key=lambda k: sweep_results[k]["PPDD"])
    print(f"\n  Best by R:    ny_tp_mult={best_mult:.1f}  ({fmt_metrics(sweep_results[best_mult])})")
    print(f"  Best by PPDD: ny_tp_mult={best_ppdd:.1f}  ({fmt_metrics(sweep_results[best_ppdd])})")

    # Dynamic TP: modify backtest to use signal-context-dependent TP
    # This requires modifying the TP computation inside the backtest loop
    # We can do this by pre-computing TP multipliers per signal bar
    print(f"\n--- Dynamic TP by signal_type x grade ---")
    print(f"  Rules: {simple_rules_mult}")

    # Pre-compute per-bar TP multiplier
    n = d["n"]
    sig_mask = d["sig_mask"]
    sig_type = d["sig_type"]

    # We need to compute grade for each signal to assign the right mult
    # Replicate the grade computation from the backtest
    bias_dir_arr = d["bias_dir_arr"]
    sig_dir = d["sig_dir"]
    regime_arr = d["regime_arr"]

    dynamic_tp_mult_arr = np.full(n, 3.0)  # Default to 3.0
    signal_indices = np.where(sig_mask)[0]

    for idx in signal_indices:
        direction = int(sig_dir[idx])
        ba = 1.0 if (direction == np.sign(bias_dir_arr[idx]) and bias_dir_arr[idx] != 0) else 0.0
        regime = regime_arr[idx]
        if np.isnan(ba) or np.isnan(regime) or regime == 0.0:
            grade = "C"
        else:
            aligned = ba > 0.5
            full = regime >= 1.0
            if aligned and full:
                grade = "A+"
            elif aligned or full:
                grade = "B+"
            else:
                grade = "C"
        st = str(sig_type[idx])
        key = f"{st}_{grade}"
        dynamic_tp_mult_arr[idx] = simple_rules_mult.get(key, 3.0)

    # Run backtest with dynamic TP: modify the backtest engine to use per-signal TP mult
    # The cleanest way: override ny_tp_multiplier per signal in a modified run
    # We'll create a modified version that accepts an array

    def run_backtest_dynamic_tp(d: dict, tp_mult_arr: np.ndarray) -> list:
        """Modified backtest that uses per-signal TP multiplier instead of fixed."""
        import copy
        d_mod = dict(d)
        params = copy.deepcopy(d["params"])
        # Set a high fixed mult to prevent the fixed override from being too restrictive
        params["session_rules"]["ny_tp_multiplier"] = 3.0  # Will be overridden per-signal
        d_mod["params"] = params

        # We need to monkey-patch the TP calculation
        # Instead, we'll modify entry_price/irl_target to encode the desired TP
        # TP = entry + mult * (irl_target - entry) for longs
        # So new_irl_target = entry + (mult/3.0) * (irl_target - entry)
        # This way, when the engine applies ny_tp_mult=3.0, it gets:
        # TP = entry + 3.0 * (new_irl_target - entry) = entry + 3.0 * (mult/3.0) * (orig_target - entry)
        # = entry + mult * (orig_target - entry) = desired TP

        orig_irl = d["irl_target_arr"].copy()
        entry = d["entry_price_arr"]
        new_irl = orig_irl.copy()

        for idx in signal_indices:
            if np.isnan(entry[idx]) or np.isnan(orig_irl[idx]):
                continue
            desired_mult = tp_mult_arr[idx]
            # Scale: new_irl = entry + (desired_mult / 3.0) * (orig_irl - entry)
            new_irl[idx] = entry[idx] + (desired_mult / 3.0) * (orig_irl[idx] - entry[idx])

        d_mod["irl_target_arr"] = new_irl
        trades = run_backtest_improved(d_mod)
        d_mod["irl_target_arr"] = orig_irl  # Restore
        return trades

    trades_dynamic = run_backtest_dynamic_tp(d, dynamic_tp_mult_arr)
    m_dynamic = vi_compute_metrics(trades_dynamic)
    print(f"  Dynamic:  {fmt_metrics(m_dynamic)}")
    print(f"  Baseline: {fmt_metrics(m_baseline)}")
    delta_r = m_dynamic["R"] - m_baseline["R"]
    delta_ppdd = m_dynamic["PPDD"] - m_baseline["PPDD"]
    delta_pf = m_dynamic["PF"] - m_baseline["PF"]
    print(f"  Delta: R={delta_r:+.1f}  PPDD={delta_ppdd:+.2f}  PF={delta_pf:+.2f}")

    # Walk-forward: derive rules from 2016-2022, test on 2023-2026
    print(f"\n--- Walk-Forward: train rules on 2016-2022, test on 2023-2026 ---")
    traded_train = df[(df["passes_all_filters"]) & (pd.to_datetime(df["bar_time_et"]).dt.year <= 2022)]

    tp_levels = np.arange(0.5, 10.5, 0.5)
    wf_rules = {}
    for st in ["trend", "mss"]:
        for gr in ["A+", "B+", "C"]:
            key = f"{st}_{gr}"
            sub = traded_train[(traded_train["signal_type"] == st) & (traded_train["grade"] == gr)]
            if len(sub) < 5:
                wf_rules[key] = 3.0
                continue
            mfe_r = sub["max_favorable_excursion"].dropna().values
            best_er = -999
            best_tp = 2.0
            for tp_r in tp_levels:
                hits = mfe_r >= tp_r
                hr = float(hits.mean())
                excess = np.maximum(mfe_r[hits] - tp_r, 0).mean() if hits.any() else 0
                r_hit = 0.5 * tp_r + 0.5 * excess
                er = hr * r_hit - (1 - hr) * 1.0
                if er > best_er:
                    best_er = er
                    best_tp = float(tp_r)
            # Convert to mult
            med_rr = sub["target_rr"].median()
            if med_rr > 0:
                wf_rules[key] = round(best_tp / med_rr, 1)
            else:
                wf_rules[key] = 3.0

    print(f"  WF rules (2016-2022): {wf_rules}")

    # Apply WF rules
    wf_tp_mult_arr = np.full(n, 3.0)
    for idx in signal_indices:
        direction = int(sig_dir[idx])
        ba = 1.0 if (direction == np.sign(bias_dir_arr[idx]) and bias_dir_arr[idx] != 0) else 0.0
        regime = regime_arr[idx]
        if np.isnan(ba) or np.isnan(regime) or regime == 0.0:
            grade = "C"
        else:
            aligned = ba > 0.5
            full = regime >= 1.0
            if aligned and full:
                grade = "A+"
            elif aligned or full:
                grade = "B+"
            else:
                grade = "C"
        st = str(sig_type[idx])
        key = f"{st}_{grade}"
        wf_tp_mult_arr[idx] = wf_rules.get(key, 3.0)

    # Full period with WF rules
    trades_wf = run_backtest_dynamic_tp(d, wf_tp_mult_arr)
    m_wf = vi_compute_metrics(trades_wf)
    print(f"\n  Full period with WF rules:  {fmt_metrics(m_wf)}")
    print(f"  Baseline (fixed 3.0):       {fmt_metrics(m_baseline)}")

    # Test period only (2023+)
    trades_wf_test = [t for t in trades_wf if pd.to_datetime(t["entry_time"]).year >= 2023]
    trades_base_test = [t for t in trades_baseline if pd.to_datetime(t["entry_time"]).year >= 2023]

    if trades_wf_test:
        m_wf_test = vi_compute_metrics(trades_wf_test)
        m_base_test = vi_compute_metrics(trades_base_test)
        print(f"\n  Test period (2023-2026):")
        print(f"    WF Dynamic: {fmt_metrics(m_wf_test)}")
        print(f"    Baseline:   {fmt_metrics(m_base_test)}")
        print(f"    Delta: R={m_wf_test['R'] - m_base_test['R']:+.1f}  "
              f"PPDD={m_wf_test['PPDD'] - m_base_test['PPDD']:+.2f}  "
              f"PF={m_wf_test['PF'] - m_base_test['PF']:+.2f}")

    # Per-year breakdown
    print(f"\n  Per-Year Breakdown (WF Dynamic vs Baseline):")
    trades_wf_df = pd.DataFrame(trades_wf)
    trades_base_df = pd.DataFrame(trades_baseline)
    trades_wf_df["year"] = pd.to_datetime(trades_wf_df["entry_time"]).dt.year
    trades_base_df["year"] = pd.to_datetime(trades_base_df["entry_time"]).dt.year

    print(f"  {'Year':>6s}  {'Strategy':>12s}  {'N':>4s}  {'R':>8s}  {'PPDD':>7s}  {'PF':>6s}  {'WR':>6s}  {'MaxDD':>7s}")
    print("  " + "-" * 65)
    for year in sorted(set(trades_wf_df["year"].unique()) | set(trades_base_df["year"].unique())):
        for label, tdf in [("WF_Dynamic", trades_wf_df), ("Baseline", trades_base_df)]:
            yr = tdf[tdf["year"] == year]
            if len(yr) == 0:
                continue
            m = vi_compute_metrics(yr.to_dict("records"))
            print(f"  {year:6d}  {label:>12s}  {m['trades']:4d}  {m['R']:+8.1f}  "
                  f"{m['PPDD']:7.2f}  {m['PF']:6.2f}  {m['WR']:5.1f}%  {m['MaxDD']:7.1f}")
        print()

    # ================================================================
    # Part 5b: SMART Dynamic TP — use real backtest sweep per group
    # ================================================================
    print(f"\n{'='*80}")
    print(f"PART 5b: SMART DYNAMIC TP — Backtest-Derived Rules")
    print(f"{'='*80}")
    print(f"Instead of MFE-based rules (which underestimate due to trim+trail),")
    print(f"derive optimal TP mult per signal_type x grade from actual backtest results.")
    print(f"Strategy: run backtest with different TP mults, measure R per group,")
    print(f"then assign each group its own best multiplier.")

    # For each signal_type x grade group, find the TP mult that maximizes that group's R
    # We already have trades from sweep_results keyed by mult
    # But those are aggregate — we need per-trade group breakdown

    # Re-run a subset of TP mults and track per-trade metadata
    test_mults = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    group_results = {}  # {(mult, type, grade): metrics}

    for mult in test_mults:
        trades = run_with_tp_mult(d, mult)
        tdf = pd.DataFrame(trades)
        if len(tdf) == 0:
            continue
        for (st, gr), grp in tdf.groupby(["type", "grade"]):
            r_arr = grp["r"].values
            key = (mult, st, gr)
            group_results[key] = {
                "trades": len(r_arr),
                "R": float(r_arr.sum()),
                "avgR": float(r_arr.mean()),
                "WR": float((r_arr > 0).mean() * 100),
            }

    # Find best mult per group
    print(f"\n  Per-group optimal TP multiplier (from backtest):")
    print(f"  {'Group':20s}  {'BestMult':>9s}  {'R':>8s}  {'AvgR':>8s}  {'WR':>6s}  {'N':>5s}")
    print("  " + "-" * 65)

    smart_rules = {}
    for st in ["trend", "mss"]:
        for gr in ["A+", "B+", "C"]:
            group_key = f"{st}_{gr}"
            best_mult_for_group = 3.0
            best_r = -999
            best_info = None
            for mult in test_mults:
                key = (mult, st, gr)
                if key in group_results:
                    info = group_results[key]
                    if info["R"] > best_r:
                        best_r = info["R"]
                        best_mult_for_group = mult
                        best_info = info

            smart_rules[group_key] = best_mult_for_group
            if best_info:
                print(f"  {group_key:20s}  {best_mult_for_group:9.1f}  {best_info['R']:+8.1f}  "
                      f"{best_info['avgR']:+8.4f}  {best_info['WR']:5.1f}%  {best_info['trades']:5d}")
            else:
                print(f"  {group_key:20s}  {best_mult_for_group:9.1f}  (no data)")

    print(f"\n  Smart rules: {smart_rules}")

    # Build smart TP mult array
    smart_tp_mult_arr = np.full(n, 3.0)
    for idx in signal_indices:
        direction = int(sig_dir[idx])
        ba = 1.0 if (direction == np.sign(bias_dir_arr[idx]) and bias_dir_arr[idx] != 0) else 0.0
        regime = regime_arr[idx]
        if np.isnan(ba) or np.isnan(regime) or regime == 0.0:
            grade = "C"
        else:
            aligned = ba > 0.5
            full = regime >= 1.0
            if aligned and full:
                grade = "A+"
            elif aligned or full:
                grade = "B+"
            else:
                grade = "C"
        st = str(sig_type[idx])
        key = f"{st}_{grade}"
        smart_tp_mult_arr[idx] = smart_rules.get(key, 3.0)

    trades_smart = run_backtest_dynamic_tp(d, smart_tp_mult_arr)
    m_smart = vi_compute_metrics(trades_smart)
    print(f"\n  Smart Dynamic:  {fmt_metrics(m_smart)}")
    print(f"  Baseline 3.0:   {fmt_metrics(m_baseline)}")
    print(f"  Best fixed:     {fmt_metrics(sweep_results[best_ppdd])}")

    # Walk-forward for smart rules: derive from 2016-2022, test 2023+
    print(f"\n  Walk-Forward (smart rules trained on 2016-2022):")
    wf_group_results = {}
    for mult in test_mults:
        trades = run_with_tp_mult(d, mult)
        tdf = pd.DataFrame(trades)
        tdf["year"] = pd.to_datetime(tdf["entry_time"]).dt.year
        train_tdf = tdf[tdf["year"] <= 2022]
        for (st, gr), grp in train_tdf.groupby(["type", "grade"]):
            r_arr = grp["r"].values
            key = (mult, st, gr)
            wf_group_results[key] = {
                "trades": len(r_arr),
                "R": float(r_arr.sum()),
                "avgR": float(r_arr.mean()),
            }

    wf_smart_rules = {}
    print(f"\n  WF Smart rules (trained 2016-2022):")
    for st in ["trend", "mss"]:
        for gr in ["A+", "B+", "C"]:
            group_key = f"{st}_{gr}"
            best_mult_for_group = 3.0
            best_r = -999
            for mult in test_mults:
                key = (mult, st, gr)
                if key in wf_group_results:
                    if wf_group_results[key]["R"] > best_r:
                        best_r = wf_group_results[key]["R"]
                        best_mult_for_group = mult
            wf_smart_rules[group_key] = best_mult_for_group
            print(f"    {group_key}: {best_mult_for_group:.1f}")

    # Apply WF smart rules
    wf_smart_tp_arr = np.full(n, 3.0)
    for idx in signal_indices:
        direction = int(sig_dir[idx])
        ba = 1.0 if (direction == np.sign(bias_dir_arr[idx]) and bias_dir_arr[idx] != 0) else 0.0
        regime = regime_arr[idx]
        if np.isnan(ba) or np.isnan(regime) or regime == 0.0:
            grade = "C"
        else:
            aligned = ba > 0.5
            full = regime >= 1.0
            if aligned and full:
                grade = "A+"
            elif aligned or full:
                grade = "B+"
            else:
                grade = "C"
        st = str(sig_type[idx])
        key = f"{st}_{grade}"
        wf_smart_tp_arr[idx] = wf_smart_rules.get(key, 3.0)

    trades_wf_smart = run_backtest_dynamic_tp(d, wf_smart_tp_arr)
    m_wf_smart = vi_compute_metrics(trades_wf_smart)

    # Test period
    trades_wf_smart_test = [t for t in trades_wf_smart if pd.to_datetime(t["entry_time"]).year >= 2023]
    trades_base_test2 = [t for t in trades_baseline if pd.to_datetime(t["entry_time"]).year >= 2023]

    print(f"\n  Full period:")
    print(f"    WF Smart Dynamic: {fmt_metrics(m_wf_smart)}")
    print(f"    Baseline 3.0:     {fmt_metrics(m_baseline)}")

    if trades_wf_smart_test:
        m_wfs_test = vi_compute_metrics(trades_wf_smart_test)
        m_b_test = vi_compute_metrics(trades_base_test2)

        # Also run best fixed on test period
        trades_best_fixed_all = run_with_tp_mult(d, best_ppdd)
        trades_best_fixed_test = [t for t in trades_best_fixed_all if pd.to_datetime(t["entry_time"]).year >= 2023]
        m_bf_test = vi_compute_metrics(trades_best_fixed_test)

        print(f"\n  Test period (2023-2026):")
        print(f"    WF Smart Dynamic:  {fmt_metrics(m_wfs_test)}")
        print(f"    Baseline 3.0:      {fmt_metrics(m_b_test)}")
        print(f"    Best fixed ({best_ppdd:.0f}.0): {fmt_metrics(m_bf_test)}")

    # Per-year for smart dynamic
    print(f"\n  Per-Year: WF Smart Dynamic vs Baseline vs Fixed {best_ppdd:.0f}.0:")
    trades_smart_df = pd.DataFrame(trades_wf_smart)
    trades_smart_df["year"] = pd.to_datetime(trades_smart_df["entry_time"]).dt.year
    trades_bf_df = pd.DataFrame(trades_best_fixed_all)
    trades_bf_df["year"] = pd.to_datetime(trades_bf_df["entry_time"]).dt.year

    print(f"  {'Year':>6s}  {'Strategy':>14s}  {'N':>4s}  {'R':>8s}  {'PPDD':>7s}  {'PF':>6s}  {'WR':>6s}  {'MaxDD':>7s}")
    print("  " + "-" * 70)
    for year in sorted(set(trades_smart_df["year"].unique()) | set(trades_base_df["year"].unique())):
        for label, tdf in [("WF_Smart", trades_smart_df), ("Baseline_3.0", trades_base_df),
                            (f"Fixed_{best_ppdd:.0f}.0", trades_bf_df)]:
            yr = tdf[tdf["year"] == year]
            if len(yr) == 0:
                continue
            m = vi_compute_metrics(yr.to_dict("records"))
            print(f"  {year:6d}  {label:>14s}  {m['trades']:4d}  {m['R']:+8.1f}  "
                  f"{m['PPDD']:7.2f}  {m['PF']:6.2f}  {m['WR']:5.1f}%  {m['MaxDD']:7.1f}")
        print()

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'='*80}")
    print(f"FINAL ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"\n  Current config:  ny_tp_mult = 3.0")
    print(f"  Baseline result: {fmt_metrics(m_baseline)}")
    print()
    best_fixed = max(sweep_results.items(), key=lambda x: x[1]["PPDD"])
    print(f"  Best fixed TP mult (by PPDD): {best_fixed[0]:.1f}")
    print(f"  Best fixed result:            {fmt_metrics(best_fixed[1])}")
    print()
    print(f"  MFE-based Dynamic TP:         {fmt_metrics(m_dynamic)}  (FAILED: MFE model too conservative)")
    print(f"  Smart Dynamic TP (all data):  {fmt_metrics(m_smart)}")
    print(f"  Smart Dynamic TP (WF):        {fmt_metrics(m_wf_smart)}")
    print()

    # Compare best options
    candidates = [
        ("Baseline 3.0", m_baseline),
        (f"Fixed {best_fixed[0]:.1f}", best_fixed[1]),
        ("Smart Dynamic (WF)", m_wf_smart),
    ]
    print(f"  {'Strategy':>25s}  {'R':>8s}  {'PPDD':>7s}  {'PF':>6s}  {'WR':>6s}  {'MaxDD':>7s}")
    print("  " + "-" * 60)
    for name, m in candidates:
        print(f"  {name:>25s}  {m['R']:+8.1f}  {m['PPDD']:7.2f}  {m['PF']:6.2f}  {m['WR']:5.1f}%  {m['MaxDD']:7.1f}")

    # Verdict
    best_candidate = max(candidates, key=lambda x: x[1]["PPDD"])
    print(f"\n  VERDICT: Best strategy by PPDD is '{best_candidate[0]}'")
    if best_candidate[0] == "Smart Dynamic (WF)":
        print(f"  RECOMMENDATION: Use dynamic TP rules: {wf_smart_rules}")
    elif "Fixed" in best_candidate[0]:
        print(f"  RECOMMENDATION: Increase ny_tp_mult from 3.0 to {best_fixed[0]:.1f}")
        print(f"  Dynamic TP adds complexity without sufficient OOS improvement.")
    else:
        print(f"  RECOMMENDATION: Keep current ny_tp_mult=3.0")
        print(f"  Neither fixed alternatives nor dynamic TP beat current config on PPDD.")


# ======================================================================
# Supplementary: Current TP analysis
# ======================================================================
def analyze_current_tp_in_atr(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("SUPPLEMENTARY: What does ny_tp_mult=3.0 translate to in ATR?")
    print("=" * 80)

    traded = df[df["passes_all_filters"]].copy()
    if len(traded) == 0:
        print("  No traded signals")
        return

    traded["tp_distance_pts_3x"] = 3.0 * traded["target_distance_pts"]
    traded["tp_distance_atr_3x"] = traded["tp_distance_pts_3x"] / traded["atr_14"]
    traded["tp_distance_R_3x"] = 3.0 * traded["target_rr"]

    print(f"\n  Current TP (ny_tp_mult=3.0):")
    for col, label in [("tp_distance_R_3x", "TP distance (R)"),
                        ("tp_distance_atr_3x", "TP distance (ATR)")]:
        print(f"    {label}:")
        print(f"      Mean={traded[col].mean():.2f}  Median={traded[col].median():.2f}  "
              f"P25={traded[col].quantile(0.25):.2f}  P75={traded[col].quantile(0.75):.2f}")

    print(f"\n  By signal_type x grade:")
    for (st, gr), sub in traded.groupby(["signal_type", "grade"]):
        if len(sub) >= 3:
            print(f"    {st}_{gr:3s}: TP_R_median={sub['tp_distance_R_3x'].median():.2f}  "
                  f"TP_ATR_median={sub['tp_distance_atr_3x'].median():.2f}  "
                  f"MFE_R_median={sub['max_favorable_excursion'].median():.2f}  (n={len(sub)})")


# ======================================================================
# Main
# ======================================================================
def main():
    t0 = _time.perf_counter()

    print("Loading signal feature database...")
    df = pd.read_parquet(DATA / "signal_feature_database.parquet")
    print(f"  Loaded {len(df)} signals x {len(df.columns)} columns")

    # Compute MFE in ATR
    df["mfe_atr"] = df["max_favorable_excursion"] * df["stop_distance_atr"]

    # Supplementary
    analyze_current_tp_in_atr(df)

    # Part 1: MFE Analysis
    df = part1_mfe_analysis(df)

    # Part 2: Optimal TP by Group (trim+trail model)
    part2_optimal_tp_by_group(df)

    # Part 3: XGBoost MFE prediction
    part3_xgb_model(df)

    # Part 4: Rules Extraction
    simple_rules_mult = part4_rules_extraction(df)

    # Part 5: Practical Backtest
    part5_practical_test(df, simple_rules_mult)

    elapsed = _time.perf_counter() - t0
    print(f"\n\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
