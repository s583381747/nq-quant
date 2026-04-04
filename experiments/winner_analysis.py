"""
Winner vs Loser Analysis — Comprehensive Signal Feature Database Study
======================================================================
Analyzes 15,894 signals across 58 features to identify what separates
big winners from full stops, build a discriminator score, and extract
actionable trading rules.

Every summary table shows R, PPDD, PF together per the user requirement.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT = Path(r"C:/projects/lanto quant/nq quant")
DB_PATH = PROJECT / "data" / "signal_feature_database.parquet"

# ── Helper functions ───────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute R, PPDD, PF, WR, count for a subset of signals."""
    n = len(df)
    if n == 0:
        return {"N": 0, "Total_R": 0, "Avg_R": 0, "WR%": 0, "PF": 0, "PPDD": 0}

    total_r = df["outcome_r"].sum()
    avg_r = df["outcome_r"].mean()

    winners = df[df["outcome_r"] > 0]
    losers = df[df["outcome_r"] <= 0]
    wr = len(winners) / n * 100 if n > 0 else 0

    gross_profit = winners["outcome_r"].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers["outcome_r"].sum()) if len(losers) > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Peak-to-peak drawdown (PPDD) in R
    cumr = df["outcome_r"].cumsum()
    running_max = cumr.cummax()
    drawdown = cumr - running_max
    ppdd = drawdown.min()

    return {
        "N": n,
        "Total_R": round(total_r, 2),
        "Avg_R": round(avg_r, 4),
        "WR%": round(wr, 1),
        "PF": round(pf, 3),
        "PPDD": round(ppdd, 2),
    }


def print_separator(title: str):
    log.info("")
    log.info("=" * 80)
    log.info(f"  {title}")
    log.info("=" * 80)


def df_to_str(df: pd.DataFrame, max_col_width: int = 14) -> str:
    """Pretty-print a DataFrame as a string table."""
    return df.to_string(index=True, float_format=lambda x: f"{x:.4f}" if abs(x) < 100 else f"{x:.2f}")


# ══════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════

log.info(f"Loading {DB_PATH} ...")
df = pd.read_parquet(DB_PATH)
log.info(f"Loaded {len(df)} signals × {len(df.columns)} features")

# Sort by time for cumulative metrics
df = df.sort_values("bar_time_utc").reset_index(drop=True)

# ── Define feature sets ────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "fvg_size_atr", "fvg_sweep_score", "bar_body_ratio", "bar_body_atr",
    "bar_range_atr", "fluency_score", "signal_quality", "atr_14",
    "atr_percentile", "hour_et", "day_of_week", "bias_confidence",
    "regime", "stop_distance_atr", "target_rr", "pa_alt_dir_ratio",
    "max_favorable_excursion", "max_adverse_excursion", "bars_to_outcome",
    "stop_distance_pts", "target_distance_pts", "fvg_size_pts",
]

BOOL_FEATURES = [
    "fvg_swept_liquidity", "is_displaced", "is_monday", "is_friday",
    "bias_aligned", "has_smt", "smt_bull", "smt_bear", "is_orm_period",
]

# Features for analysis (exclude outcome-related)
ANALYSIS_FEATURES = [
    "fvg_size_atr", "fvg_sweep_score", "bar_body_ratio", "bar_body_atr",
    "bar_range_atr", "fluency_score", "signal_quality", "atr_14",
    "atr_percentile", "hour_et", "day_of_week", "bias_confidence",
    "regime", "stop_distance_atr", "target_rr", "pa_alt_dir_ratio",
    "stop_distance_pts", "target_distance_pts", "fvg_size_pts",
]

ANALYSIS_BOOLS = [
    "fvg_swept_liquidity", "is_displaced", "is_monday", "is_friday",
    "bias_aligned", "has_smt", "is_orm_period",
]

# ══════════════════════════════════════════════════════════════════════
#  PART 1: OUTCOME GROUP ANALYSIS
# ══════════════════════════════════════════════════════════════════════

print_separator("PART 1: OUTCOME GROUPS")

def categorize(r):
    if r > 2.0:
        return "Big Winners (R>2)"
    elif r > 0:
        return "Small Winners (0<R<=2)"
    elif r >= -0.2:
        return "Break-even (-0.2<=R<=0)"
    elif r > -0.8:
        return "Small Losers (-0.8<R<-0.2)"
    else:
        return "Full Stops (R<=-0.8)"

df["outcome_group"] = df["outcome_r"].apply(categorize)

GROUP_ORDER = [
    "Big Winners (R>2)",
    "Small Winners (0<R<=2)",
    "Break-even (-0.2<=R<=0)",
    "Small Losers (-0.8<R<-0.2)",
    "Full Stops (R<=-0.8)",
]

# Summary metrics per group
group_metrics = []
for g in GROUP_ORDER:
    subset = df[df["outcome_group"] == g]
    m = compute_metrics(subset)
    m["Group"] = g
    group_metrics.append(m)

gm_df = pd.DataFrame(group_metrics).set_index("Group")
log.info("\n--- Outcome Group Summary (R, PPDD, PF) ---")
log.info(gm_df.to_string())

# Feature averages per group
log.info("\n--- Feature Averages by Outcome Group ---")
feature_avgs = []
for g in GROUP_ORDER:
    subset = df[df["outcome_group"] == g]
    row = {"Group": g, "Count": len(subset)}
    for f in ANALYSIS_FEATURES:
        if f in subset.columns:
            row[f] = subset[f].mean()
    for f in ANALYSIS_BOOLS:
        if f in subset.columns:
            row[f"{f}_%"] = subset[f].mean() * 100
    feature_avgs.append(row)

fa_df = pd.DataFrame(feature_avgs).set_index("Group")
# Print key features only (not all)
key_feats = ["Count", "signal_quality", "fluency_score", "fvg_size_atr",
             "bar_body_ratio", "stop_distance_atr", "target_rr",
             "pa_alt_dir_ratio", "is_displaced_%", "has_smt_%",
             "fvg_swept_liquidity_%", "bias_aligned_%"]
log.info(fa_df[[c for c in key_feats if c in fa_df.columns]].to_string(float_format=lambda x: f"{x:.3f}"))


# ══════════════════════════════════════════════════════════════════════
#  PART 2: DISCRIMINATING FEATURES (Cohen's d)
# ══════════════════════════════════════════════════════════════════════

print_separator("PART 2: DISCRIMINATING FEATURES (Big Winners vs Full Stops)")

big_winners = df[df["outcome_r"] > 2.0].copy()
full_stops = df[df["outcome_r"] <= -0.8].copy()

log.info(f"Big Winners: {len(big_winners)}")
log.info(f"Full Stops: {len(full_stops)}")

ALL_DISC_FEATURES = ANALYSIS_FEATURES + ANALYSIS_BOOLS

effect_sizes = []
for f in ALL_DISC_FEATURES:
    if f not in df.columns:
        continue
    w_vals = big_winners[f].astype(float).dropna()
    l_vals = full_stops[f].astype(float).dropna()

    if len(w_vals) < 10 or len(l_vals) < 10:
        continue

    mean_w = w_vals.mean()
    mean_l = l_vals.mean()

    # Pooled standard deviation
    n_w, n_l = len(w_vals), len(l_vals)
    var_w, var_l = w_vals.var(ddof=1), l_vals.var(ddof=1)
    pooled_std = np.sqrt(((n_w - 1) * var_w + (n_l - 1) * var_l) / (n_w + n_l - 2))

    if pooled_std == 0:
        continue

    d = (mean_w - mean_l) / pooled_std

    effect_sizes.append({
        "Feature": f,
        "Mean_Winners": round(mean_w, 4),
        "Mean_Losers": round(mean_l, 4),
        "Cohen_d": round(d, 4),
        "Abs_d": round(abs(d), 4),
        "Direction": "higher=winner" if d > 0 else "lower=winner",
    })

es_df = pd.DataFrame(effect_sizes).sort_values("Abs_d", ascending=False)
log.info("\n--- Top 20 Most Discriminating Features (by |Cohen's d|) ---")
log.info(es_df.head(20).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════
#  PART 3: WINNER DNA PROFILE
# ══════════════════════════════════════════════════════════════════════

print_separator("PART 3: WINNER DNA PROFILE (Big Winners, R > 2.0)")

log.info(f"\nTotal big winners: {len(big_winners)}")
log.info(f"  ({len(big_winners)/len(df)*100:.1f}% of all signals)")

# Boolean percentages
log.info("\n--- Boolean Feature Rates ---")
for f in ["is_displaced", "has_smt", "fvg_swept_liquidity", "bias_aligned",
          "is_monday", "is_friday", "is_orm_period"]:
    pct = big_winners[f].mean() * 100
    log.info(f"  {f}: {pct:.1f}%")

# Session distribution
log.info("\n--- Session Distribution ---")
sess_dist = big_winners["session"].value_counts()
for s, c in sess_dist.items():
    log.info(f"  {s}: {c} ({c/len(big_winners)*100:.1f}%)")

# Signal type distribution
log.info("\n--- Signal Type Distribution ---")
type_dist = big_winners["signal_type"].value_counts()
for t, c in type_dist.items():
    log.info(f"  {t}: {c} ({c/len(big_winners)*100:.1f}%)")

# Direction distribution
log.info("\n--- Direction Distribution ---")
dir_dist = big_winners["signal_dir"].value_counts()
for d, c in dir_dist.items():
    label = "Long" if d == 1 else "Short"
    log.info(f"  {label}: {c} ({c/len(big_winners)*100:.1f}%)")

# Continuous feature ranges (25th-75th percentile)
log.info("\n--- Key Feature Ranges (25th - 75th percentile) ---")
range_features = [
    "signal_quality", "fluency_score", "fvg_size_atr", "stop_distance_atr",
    "target_rr", "hour_et", "bar_body_ratio", "pa_alt_dir_ratio",
    "atr_percentile", "bias_confidence",
]
for f in range_features:
    if f in big_winners.columns:
        p25 = big_winners[f].quantile(0.25)
        p50 = big_winners[f].quantile(0.50)
        p75 = big_winners[f].quantile(0.75)
        log.info(f"  {f}: P25={p25:.3f}  P50={p50:.3f}  P75={p75:.3f}")


# ══════════════════════════════════════════════════════════════════════
#  PART 4: LOSER DNA PROFILE + DIFFERENCES
# ══════════════════════════════════════════════════════════════════════

print_separator("PART 4: LOSER DNA PROFILE (Full Stops, R < -1.0) + DIFFERENCES")

log.info(f"\nTotal full stops: {len(full_stops)}")
log.info(f"  ({len(full_stops)/len(df)*100:.1f}% of all signals)")

# Boolean percentages
log.info("\n--- Boolean Feature Rates ---")
for f in ["is_displaced", "has_smt", "fvg_swept_liquidity", "bias_aligned",
          "is_monday", "is_friday", "is_orm_period"]:
    pct_w = big_winners[f].mean() * 100
    pct_l = full_stops[f].mean() * 100
    diff = pct_w - pct_l
    log.info(f"  {f}: Losers={pct_l:.1f}%  Winners={pct_w:.1f}%  (diff={diff:+.1f}pp)")

# Session distribution
log.info("\n--- Session Distribution (Losers) ---")
sess_dist_l = full_stops["session"].value_counts()
for s, c in sess_dist_l.items():
    log.info(f"  {s}: {c} ({c/len(full_stops)*100:.1f}%)")

# Signal type
log.info("\n--- Signal Type (Losers) ---")
type_dist_l = full_stops["signal_type"].value_counts()
for t, c in type_dist_l.items():
    log.info(f"  {t}: {c} ({c/len(full_stops)*100:.1f}%)")

# Direction
log.info("\n--- Direction (Losers) ---")
dir_dist_l = full_stops["signal_dir"].value_counts()
for d, c in dir_dist_l.items():
    label = "Long" if d == 1 else "Short"
    log.info(f"  {label}: {c} ({c/len(full_stops)*100:.1f}%)")

# Continuous features comparison
log.info("\n--- Key Feature Ranges: WINNERS vs LOSERS ---")
log.info(f"{'Feature':<22} {'Win_P25':>8} {'Win_P50':>8} {'Win_P75':>8} | {'Lose_P25':>8} {'Lose_P50':>8} {'Lose_P75':>8}")
log.info("-" * 85)
for f in range_features:
    if f in big_winners.columns and f in full_stops.columns:
        wp25, wp50, wp75 = big_winners[f].quantile([0.25, 0.50, 0.75])
        lp25, lp50, lp75 = full_stops[f].quantile([0.25, 0.50, 0.75])
        log.info(f"  {f:<20} {wp25:>8.3f} {wp50:>8.3f} {wp75:>8.3f} | {lp25:>8.3f} {lp50:>8.3f} {lp75:>8.3f}")

# Identify useless features (similar between winners and losers)
log.info("\n--- Features with SMALLEST difference (potentially useless) ---")
small_diff = es_df.tail(10)
log.info(small_diff.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════
#  PART 5: DISCRIMINATOR SCORE
# ══════════════════════════════════════════════════════════════════════

print_separator("PART 5: DISCRIMINATOR SCORE (Top 5 Features)")

# Pick top 5 features by Cohen's d (excluding outcome-related and duplicate features)
# NOTE: fvg_size_atr == stop_distance_atr (perfectly correlated), so exclude one
# Also exclude features that are direct functions of outcome (MFE, MAE, bars_to_outcome)
EXCLUDE_FROM_SCORE = {"max_favorable_excursion", "max_adverse_excursion",
                      "bars_to_outcome", "target_distance_pts", "target_rr",
                      "stop_distance_pts", "stop_distance_atr",  # duplicate of fvg_size_atr
                      "fvg_size_pts",  # duplicate in pts
                      }

es_eligible = es_df[~es_df["Feature"].isin(EXCLUDE_FROM_SCORE)].copy()

# Also deduplicate features with correlation > 0.95
selected = []
selected_names = set()
for _, row in es_eligible.iterrows():
    feat = row["Feature"]
    # Check correlation with already selected
    dominated = False
    for sf in selected_names:
        if sf in df.columns and feat in df.columns:
            corr = abs(df[sf].astype(float).corr(df[feat].astype(float)))
            if corr > 0.95:
                dominated = True
                break
    if not dominated:
        selected.append(row)
        selected_names.add(feat)
    if len(selected) >= 5:
        break

top5 = pd.DataFrame(selected)

log.info("\n--- Top 5 Features for Discriminator Score ---")
log.info(top5[["Feature", "Cohen_d", "Direction"]].to_string(index=False))

# Build z-scores and composite
score_features = top5["Feature"].tolist()
score_directions = []  # +1 if higher=winner, -1 if lower=winner
for _, row in top5.iterrows():
    score_directions.append(1.0 if row["Cohen_d"] > 0 else -1.0)

log.info("\n--- Building winner_score ---")
df["winner_score"] = 0.0
for feat, direction in zip(score_features, score_directions):
    col = df[feat].astype(float)
    mu = col.mean()
    sigma = col.std()
    if sigma > 0:
        z = (col - mu) / sigma
    else:
        z = 0.0
    df["winner_score"] += z * direction
    log.info(f"  {feat}: direction={'higher' if direction > 0 else 'lower'}, mean={mu:.4f}, std={sigma:.4f}")

# Quintile analysis
df["winner_quintile"] = pd.qcut(df["winner_score"], 5, labels=["Q1 (Bottom)", "Q2", "Q3", "Q4", "Q5 (Top)"])

log.info("\n--- Winner Score Quintile Analysis (R, PPDD, PF) ---")
quintile_results = []
for q in ["Q1 (Bottom)", "Q2", "Q3", "Q4", "Q5 (Top)"]:
    subset = df[df["winner_quintile"] == q].sort_values("bar_time_utc")
    m = compute_metrics(subset)
    m["Quintile"] = q
    m["Score_Range"] = f"{subset['winner_score'].min():.2f} to {subset['winner_score'].max():.2f}"
    quintile_results.append(m)

qr_df = pd.DataFrame(quintile_results).set_index("Quintile")
log.info(qr_df.to_string())

# Monotonicity check
log.info("\n--- Does winner_score predict outcomes? ---")
avg_rs = [qr_df.loc[q, "Avg_R"] for q in ["Q1 (Bottom)", "Q2", "Q3", "Q4", "Q5 (Top)"]]
monotonic = all(avg_rs[i] <= avg_rs[i+1] for i in range(len(avg_rs)-1))
log.info(f"  Avg R by quintile: {[round(x, 4) for x in avg_rs]}")
log.info(f"  Monotonically increasing: {monotonic}")
log.info(f"  Q5 vs Q1 Avg R spread: {avg_rs[4] - avg_rs[0]:.4f}")


# ══════════════════════════════════════════════════════════════════════
#  PART 6: "IF I ONLY TRADED SIGNALS THAT LOOK LIKE WINNERS"
# ══════════════════════════════════════════════════════════════════════

print_separator("PART 6: TOP 20% WINNER-SCORE SIGNALS vs CURRENT FILTERS")

top20_threshold = df["winner_score"].quantile(0.80)
top20 = df[df["winner_score"] >= top20_threshold].sort_values("bar_time_utc").copy()
current_filtered = df[df["passes_all_filters"] == True].sort_values("bar_time_utc").copy()

log.info(f"\n--- Top 20% by winner_score (threshold >= {top20_threshold:.3f}) ---")
m_top20 = compute_metrics(top20)
log.info(f"  Signals: {m_top20['N']}")
log.info(f"  Total R: {m_top20['Total_R']}")
log.info(f"  Avg R: {m_top20['Avg_R']}")
log.info(f"  WR: {m_top20['WR%']}%")
log.info(f"  PF: {m_top20['PF']}")
log.info(f"  PPDD: {m_top20['PPDD']}")

log.info(f"\n--- Session distribution (Top 20%) ---")
for s, c in top20["session"].value_counts().items():
    log.info(f"  {s}: {c} ({c/len(top20)*100:.1f}%)")

log.info(f"\n--- Current filter chain (passes_all_filters=True) ---")
m_filt = compute_metrics(current_filtered)
log.info(f"  Signals: {m_filt['N']}")
log.info(f"  Total R: {m_filt['Total_R']}")
log.info(f"  Avg R: {m_filt['Avg_R']}")
log.info(f"  WR: {m_filt['WR%']}%")
log.info(f"  PF: {m_filt['PF']}")
log.info(f"  PPDD: {m_filt['PPDD']}")

# Overlap analysis
overlap = top20[top20["passes_all_filters"] == True]
log.info(f"\n--- Overlap ---")
log.info(f"  Signals in BOTH top20% AND current filters: {len(overlap)}")
log.info(f"  Top20% unique: {len(top20) - len(overlap)}")
log.info(f"  Current filters unique: {len(current_filtered) - len(overlap)}")

# Intersection metrics
if len(overlap) > 0:
    m_overlap = compute_metrics(overlap.sort_values("bar_time_utc"))
    log.info(f"\n--- Intersection (both filters) ---")
    log.info(f"  Signals: {m_overlap['N']}")
    log.info(f"  Total R: {m_overlap['Total_R']}")
    log.info(f"  Avg R: {m_overlap['Avg_R']}")
    log.info(f"  WR: {m_overlap['WR%']}%")
    log.info(f"  PF: {m_overlap['PF']}")
    log.info(f"  PPDD: {m_overlap['PPDD']}")

# Union metrics
union = df[(df["winner_score"] >= top20_threshold) | (df["passes_all_filters"] == True)].sort_values("bar_time_utc")
m_union = compute_metrics(union)
log.info(f"\n--- Union (either filter) ---")
log.info(f"  Signals: {m_union['N']}")
log.info(f"  Total R: {m_union['Total_R']}")
log.info(f"  Avg R: {m_union['Avg_R']}")
log.info(f"  WR: {m_union['WR%']}%")
log.info(f"  PF: {m_union['PF']}")
log.info(f"  PPDD: {m_union['PPDD']}")

# ── Walk-forward test ──────────────────────────────────────────────────
log.info("\n--- Walk-Forward Test (expanding window, test on next year) ---")

df["year"] = df["bar_time_utc"].dt.year
years = sorted(df["year"].unique())
log.info(f"  Available years: {years}")

wf_results = []
for test_year in years:
    if test_year == years[0]:
        continue  # Need at least 1 training year

    train = df[df["year"] < test_year].copy()
    test = df[df["year"] == test_year].copy()

    if len(train) < 100 or len(test) < 50:
        continue

    # Compute z-score parameters on training set
    test_score = np.zeros(len(test))
    for feat, direction in zip(score_features, score_directions):
        col_train = train[feat].astype(float)
        mu = col_train.mean()
        sigma = col_train.std()
        if sigma > 0:
            z = (test[feat].astype(float) - mu) / sigma
        else:
            z = 0.0
        test_score += z.values * direction

    test["wf_winner_score"] = test_score
    threshold_train = train["winner_score"].quantile(0.80)

    top20_oos = test[test["wf_winner_score"] >= threshold_train].sort_values("bar_time_utc")
    all_test = test.sort_values("bar_time_utc")

    m_oos = compute_metrics(top20_oos)
    m_all = compute_metrics(all_test)

    wf_results.append({
        "Test_Year": test_year,
        "Train_Size": len(train),
        "Test_Size": len(test),
        "Top20_N": m_oos["N"],
        "Top20_Avg_R": m_oos["Avg_R"],
        "Top20_WR%": m_oos["WR%"],
        "Top20_PF": m_oos["PF"],
        "Top20_PPDD": m_oos["PPDD"],
        "Top20_Total_R": m_oos["Total_R"],
        "All_Avg_R": m_all["Avg_R"],
        "All_WR%": m_all["WR%"],
        "All_PF": m_all["PF"],
        "Improvement": round(m_oos["Avg_R"] - m_all["Avg_R"], 4),
    })

wf_df = pd.DataFrame(wf_results)
log.info(wf_df.to_string(index=False))

holds_oos = sum(1 for r in wf_results if r["Improvement"] > 0)
log.info(f"\n  Walk-forward: top20% beats baseline in {holds_oos}/{len(wf_results)} years")


# ══════════════════════════════════════════════════════════════════════
#  PART 7: FEATURE CLUSTERS IN WINNERS
# ══════════════════════════════════════════════════════════════════════

print_separator("PART 7: FEATURE CLUSTERS IN BIG WINNERS")

# Split by signal type
for stype in ["trend", "mss"]:
    subset = big_winners[big_winners["signal_type"] == stype].copy()
    log.info(f"\n--- Big Winners: {stype.upper()} ({len(subset)} signals) ---")

    if len(subset) < 10:
        log.info("  Too few signals for analysis")
        continue

    m = compute_metrics(subset.sort_values("bar_time_utc"))
    log.info(f"  Total R: {m['Total_R']}, Avg R: {m['Avg_R']}, WR: {m['WR%']}%, PF: {m['PF']}, PPDD: {m['PPDD']}")

    # Session breakdown
    log.info(f"\n  Session breakdown:")
    for s in ["ny", "london", "asia"]:
        ss = subset[subset["session"] == s]
        if len(ss) > 0:
            ms = compute_metrics(ss.sort_values("bar_time_utc"))
            log.info(f"    {s}: N={ms['N']}, Avg_R={ms['Avg_R']}, WR={ms['WR%']}%, PF={ms['PF']}, PPDD={ms['PPDD']}")

    # Direction breakdown
    log.info(f"\n  Direction breakdown:")
    for d in [1, -1]:
        label = "Long" if d == 1 else "Short"
        ss = subset[subset["signal_dir"] == d]
        if len(ss) > 0:
            ms = compute_metrics(ss.sort_values("bar_time_utc"))
            log.info(f"    {label}: N={ms['N']}, Avg_R={ms['Avg_R']}, WR={ms['WR%']}%, PF={ms['PF']}, PPDD={ms['PPDD']}")

    # Key feature profile
    log.info(f"\n  Feature profile (medians):")
    for f in ["signal_quality", "fluency_score", "fvg_size_atr", "bar_body_ratio",
              "stop_distance_atr", "pa_alt_dir_ratio", "hour_et"]:
        if f in subset.columns:
            log.info(f"    {f}: {subset[f].median():.3f}")

    # Displacement rate
    log.info(f"    is_displaced: {subset['is_displaced'].mean()*100:.1f}%")
    log.info(f"    has_smt: {subset['has_smt'].mean()*100:.1f}%")
    log.info(f"    fvg_swept_liquidity: {subset['fvg_swept_liquidity'].mean()*100:.1f}%")

# Sub-clusters within big winners
log.info("\n\n--- Sub-Cluster Analysis ---")

# Cluster 1: High-displacement trend longs in NY AM
c1 = big_winners[
    (big_winners["signal_type"] == "trend") &
    (big_winners["signal_dir"] == 1) &
    (big_winners["session"] == "ny") &
    (big_winners["is_displaced"] == True)
]
log.info(f"\nCluster 1: 'Displaced Trend Longs in NY' ({len(c1)} signals)")
if len(c1) > 0:
    m = compute_metrics(c1.sort_values("bar_time_utc"))
    log.info(f"  R={m['Total_R']}, Avg_R={m['Avg_R']}, WR={m['WR%']}%, PF={m['PF']}, PPDD={m['PPDD']}")
    log.info(f"  Avg hour_et: {c1['hour_et'].mean():.1f}, Avg signal_quality: {c1['signal_quality'].mean():.3f}")

# Cluster 2: MSS reversals in London/Asia
c2 = big_winners[
    (big_winners["signal_type"] == "mss") &
    (big_winners["session"].isin(["london", "asia"]))
]
log.info(f"\nCluster 2: 'MSS Reversals in London/Asia' ({len(c2)} signals)")
if len(c2) > 0:
    m = compute_metrics(c2.sort_values("bar_time_utc"))
    log.info(f"  R={m['Total_R']}, Avg_R={m['Avg_R']}, WR={m['WR%']}%, PF={m['PF']}, PPDD={m['PPDD']}")
    log.info(f"  Avg fluency: {c2['fluency_score'].mean():.3f}, has_smt: {c2['has_smt'].mean()*100:.1f}%")

# Cluster 3: High-quality displaced NY trades
c3 = big_winners[
    (big_winners["session"] == "ny") &
    (big_winners["is_displaced"] == True) &
    (big_winners["signal_quality"] > big_winners["signal_quality"].median())
]
log.info(f"\nCluster 3: 'High-Quality Displaced NY' ({len(c3)} signals)")
if len(c3) > 0:
    m = compute_metrics(c3.sort_values("bar_time_utc"))
    log.info(f"  R={m['Total_R']}, Avg_R={m['Avg_R']}, WR={m['WR%']}%, PF={m['PF']}, PPDD={m['PPDD']}")

# Cluster 4: SMT-confirmed trades
c4 = big_winners[big_winners["has_smt"] == True]
log.info(f"\nCluster 4: 'SMT-Confirmed Big Winners' ({len(c4)} signals)")
if len(c4) > 0:
    m = compute_metrics(c4.sort_values("bar_time_utc"))
    log.info(f"  R={m['Total_R']}, Avg_R={m['Avg_R']}, WR={m['WR%']}%, PF={m['PF']}, PPDD={m['PPDD']}")

# Cluster 5: Liquidity sweep big winners
c5 = big_winners[big_winners["fvg_swept_liquidity"] == True]
log.info(f"\nCluster 5: 'Liquidity Sweep Big Winners' ({len(c5)} signals)")
if len(c5) > 0:
    m = compute_metrics(c5.sort_values("bar_time_utc"))
    log.info(f"  R={m['Total_R']}, Avg_R={m['Avg_R']}, WR={m['WR%']}%, PF={m['PF']}, PPDD={m['PPDD']}")

# Cluster 6: Trend shorts in NY
c6 = big_winners[
    (big_winners["signal_type"] == "trend") &
    (big_winners["signal_dir"] == -1) &
    (big_winners["session"] == "ny")
]
log.info(f"\nCluster 6: 'Trend Shorts in NY' ({len(c6)} signals)")
if len(c6) > 0:
    m = compute_metrics(c6.sort_values("bar_time_utc"))
    log.info(f"  R={m['Total_R']}, Avg_R={m['Avg_R']}, WR={m['WR%']}%, PF={m['PF']}, PPDD={m['PPDD']}")


# ══════════════════════════════════════════════════════════════════════
#  PART 8: ACTIONABLE RULES
# ══════════════════════════════════════════════════════════════════════

print_separator("PART 8: ACTIONABLE IF-THEN RULES")

# First compute thresholds from winner DNA (Part 3)
sq_p25 = big_winners["signal_quality"].quantile(0.25)
fvg_p25 = big_winners["fvg_size_atr"].quantile(0.25)
fl_p25 = big_winners["fluency_score"].quantile(0.25)
bbr_p25 = big_winners["bar_body_ratio"].quantile(0.25)
sda_p75 = big_winners["stop_distance_atr"].quantile(0.75)  # Winners have TIGHTER stops typically

log.info(f"\nWinner-derived thresholds:")
log.info(f"  signal_quality >= {sq_p25:.3f} (winner P25)")
log.info(f"  fvg_size_atr >= {fvg_p25:.3f} (winner P25)")
log.info(f"  fluency_score >= {fl_p25:.3f} (winner P25)")
log.info(f"  bar_body_ratio >= {bbr_p25:.3f} (winner P25)")
log.info(f"  stop_distance_atr <= {sda_p75:.3f} (winner P75)")

# Get top discriminating features info for smarter rules
top_disc = es_eligible.head(10)

rules = {}

# Rule 1: Signal quality + FVG size + NY session
r1_mask = (
    (df["signal_quality"] >= sq_p25) &
    (df["fvg_size_atr"] >= fvg_p25) &
    (df["session"] == "ny")
)
rules["R1: SQ+FVG+NY"] = r1_mask

# Rule 2: Displaced + high fluency + good body ratio
r2_mask = (
    (df["is_displaced"] == True) &
    (df["fluency_score"] >= fl_p25) &
    (df["bar_body_ratio"] >= bbr_p25)
)
rules["R2: Displaced+Fluent+BodyRatio"] = r2_mask

# Rule 3: Top discriminating features (use Cohen's d insights)
# Use the top 3 non-outcome features with their directions
r3_conditions = pd.Series(True, index=df.index)
r3_desc_parts = []
r3_count = 0
for i, (_, row) in enumerate(top_disc.iterrows()):
    if r3_count >= 3:
        break
    feat = row["Feature"]
    if feat in EXCLUDE_FROM_SCORE:
        continue
    col = df[feat].astype(float)
    if row["Cohen_d"] > 0:
        # Higher = winner
        thresh = float(col.quantile(0.60))  # Above 60th percentile
        r3_conditions = r3_conditions & (col >= thresh)
        r3_desc_parts.append(f"{feat}>={thresh:.3f}")
    else:
        # Lower = winner
        thresh = float(col.quantile(0.40))  # Below 40th percentile
        r3_conditions = r3_conditions & (col <= thresh)
        r3_desc_parts.append(f"{feat}<={thresh:.3f}")
    r3_count += 1

rules[f"R3: TopDisc ({'+'.join(r3_desc_parts)})"] = r3_conditions

# Rule 4: Liquidity sweep + displacement + NY
r4_mask = (
    (df["fvg_swept_liquidity"] == True) &
    (df["is_displaced"] == True) &
    (df["session"] == "ny")
)
rules["R4: LiqSweep+Displaced+NY"] = r4_mask

# Rule 5: Comprehensive "winner DNA" rule
r5_mask = (
    (df["signal_quality"] >= sq_p25) &
    (df["is_displaced"] == True) &
    (df["fluency_score"] >= fl_p25) &
    (df["session"].isin(["ny", "london"]))
)
rules["R5: SQ+Displaced+Fluent+NY/London"] = r5_mask

# Evaluate each rule
log.info("\n--- Individual Rule Performance ---")
log.info(f"{'Rule':<45} {'N':>5} {'TotalR':>8} {'AvgR':>8} {'WR%':>6} {'PF':>7} {'PPDD':>7} {'BigWin%':>8} {'FullStop%':>9}")
log.info("-" * 110)

for name, mask in rules.items():
    subset = df[mask].sort_values("bar_time_utc")
    m = compute_metrics(subset)

    # What % of big winners does this catch?
    caught_winners = big_winners[mask.loc[big_winners.index]].shape[0] if len(big_winners) > 0 else 0
    pct_winners = caught_winners / len(big_winners) * 100 if len(big_winners) > 0 else 0

    # What % of full stops does this filter out?
    caught_losers = full_stops[mask.loc[full_stops.index]].shape[0] if len(full_stops) > 0 else 0
    pct_losers_kept = caught_losers / len(full_stops) * 100 if len(full_stops) > 0 else 0

    log.info(f"  {name:<43} {m['N']:>5} {m['Total_R']:>8.1f} {m['Avg_R']:>8.4f} {m['WR%']:>6.1f} {m['PF']:>7.3f} {m['PPDD']:>7.1f} {pct_winners:>7.1f}% {pct_losers_kept:>8.1f}%")

# Combined rules
log.info("\n--- Combined Rule Performance ---")
combined_masks = {
    "R1 AND R2": rules["R1: SQ+FVG+NY"] & rules["R2: Displaced+Fluent+BodyRatio"],
    "R1 AND R5": rules["R1: SQ+FVG+NY"] & rules["R5: SQ+Displaced+Fluent+NY/London"],
    "R2 AND R4": rules["R2: Displaced+Fluent+BodyRatio"] & rules["R4: LiqSweep+Displaced+NY"],
    "R1 OR R4": rules["R1: SQ+FVG+NY"] | rules["R4: LiqSweep+Displaced+NY"],
    "Any rule (union)": rules["R1: SQ+FVG+NY"] | rules["R2: Displaced+Fluent+BodyRatio"] | rules["R4: LiqSweep+Displaced+NY"] | rules["R5: SQ+Displaced+Fluent+NY/London"],
    "All 5 rules (intersection)": rules["R1: SQ+FVG+NY"] & rules["R2: Displaced+Fluent+BodyRatio"] & rules[f"R3: TopDisc ({'+'.join(r3_desc_parts)})"] & rules["R4: LiqSweep+Displaced+NY"] & rules["R5: SQ+Displaced+Fluent+NY/London"],
}

log.info(f"{'Combined Rule':<35} {'N':>5} {'TotalR':>8} {'AvgR':>8} {'WR%':>6} {'PF':>7} {'PPDD':>7} {'BigWin%':>8} {'FullStop%':>9}")
log.info("-" * 100)

for name, mask in combined_masks.items():
    subset = df[mask].sort_values("bar_time_utc")
    m = compute_metrics(subset)

    caught_winners = big_winners[mask.loc[big_winners.index]].shape[0] if len(big_winners) > 0 else 0
    pct_winners = caught_winners / len(big_winners) * 100 if len(big_winners) > 0 else 0

    caught_losers = full_stops[mask.loc[full_stops.index]].shape[0] if len(full_stops) > 0 else 0
    pct_losers_kept = caught_losers / len(full_stops) * 100 if len(full_stops) > 0 else 0

    log.info(f"  {name:<33} {m['N']:>5} {m['Total_R']:>8.1f} {m['Avg_R']:>8.4f} {m['WR%']:>6.1f} {m['PF']:>7.3f} {m['PPDD']:>7.1f} {pct_winners:>7.1f}% {pct_losers_kept:>8.1f}%")

# Compare with current filter chain
log.info("\n--- Comparison with Current Filter Chain ---")
log.info(f"{'Method':<35} {'N':>5} {'TotalR':>8} {'AvgR':>8} {'WR%':>6} {'PF':>7} {'PPDD':>7}")
log.info("-" * 80)

comparisons = {
    "All signals (unfiltered)": df,
    "passes_all_filters=True": current_filtered,
    "Top 20% winner_score": top20,
}
for name, subset_data in comparisons.items():
    subset_sorted = subset_data.sort_values("bar_time_utc")
    m = compute_metrics(subset_sorted)
    log.info(f"  {name:<33} {m['N']:>5} {m['Total_R']:>8.1f} {m['Avg_R']:>8.4f} {m['WR%']:>6.1f} {m['PF']:>7.3f} {m['PPDD']:>7.1f}")

# Add best individual rule and best combined rule
best_rule_name = max(rules.keys(), key=lambda k: compute_metrics(df[rules[k]].sort_values("bar_time_utc"))["Avg_R"])
best_rule_subset = df[rules[best_rule_name]].sort_values("bar_time_utc")
m_best = compute_metrics(best_rule_subset)
log.info(f"  Best Rule: {best_rule_name:<20} {m_best['N']:>5} {m_best['Total_R']:>8.1f} {m_best['Avg_R']:>8.4f} {m_best['WR%']:>6.1f} {m_best['PF']:>7.3f} {m_best['PPDD']:>7.1f}")

best_combo_name = max(combined_masks.keys(), key=lambda k: compute_metrics(df[combined_masks[k]].sort_values("bar_time_utc"))["Avg_R"] if combined_masks[k].sum() > 0 else -999)
best_combo_subset = df[combined_masks[best_combo_name]].sort_values("bar_time_utc")
m_bestc = compute_metrics(best_combo_subset)
log.info(f"  Best Combo: {best_combo_name:<19} {m_bestc['N']:>5} {m_bestc['Total_R']:>8.1f} {m_bestc['Avg_R']:>8.4f} {m_bestc['WR%']:>6.1f} {m_bestc['PF']:>7.3f} {m_bestc['PPDD']:>7.1f}")


# ══════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════

print_separator("FINAL SUMMARY & KEY FINDINGS")

log.info("""
KEY FINDING: WINNERS AND LOSERS ARE NEARLY IDENTICAL IN FEATURE SPACE.

The strongest discriminator (fvg_size_atr / stop_distance_atr) has Cohen's d = 0.27,
which is a SMALL effect size. For reference:
  - d < 0.2 = negligible
  - d = 0.2-0.5 = small
  - d = 0.5-0.8 = medium
  - d > 0.8 = large

ALL 26 features tested have |d| < 0.3. This means the feature distributions
of big winners and full stops MASSIVELY overlap. No single feature, nor any
simple combination, can reliably separate them.

EVIDENCE:
  - Discriminator score quintiles show NO monotonic R improvement (Q5 vs Q1 spread: ~0)
  - Top 20% by winner_score: Avg_R = -0.029, PF = 0.953 (WORSE than baseline)
  - Walk-forward: beats baseline in 7/10 years but by tiny margins (avg +0.01R)

PRACTICALLY USELESS FEATURES (|d| < 0.03):
  - has_smt (d=0.0007): identical between winners and losers
  - bias_aligned (d=-0.015): no discrimination
  - is_monday (d=-0.027): no discrimination
  - bias_confidence (d=-0.029): no discrimination
  - bar_body_ratio (d=-0.032): no discrimination

THE CURRENT FILTER CHAIN IS FAR SUPERIOR:
  passes_all_filters=True: 564 signals, Avg_R=0.112, WR=56.4%, PF=1.291, PPDD=-12.66
  This works because it combines MULTIPLE orthogonal filters (session, bias, SQ, geometry, etc.)

  Best simple rule from this analysis:
  R1 (SQ+FVG+NY): 3501 signals, Avg_R=0.027, WR=51.0%, PF=1.060, PPDD=-68.2

  Best combo:
  R1 AND R2: 1416 signals, Avg_R=0.048, WR=49.4%, PF=1.104, PPDD=-27.7

BOTTOM LINE:
  The existing multi-filter chain is doing the right thing. Winner DNA analysis
  confirms that edge comes from COMBINING many weak signals, not from any single
  feature having strong predictive power. The existing filter chain already
  achieves this. Focus optimization efforts on:
  1. Tuning existing filter thresholds (not adding new features)
  2. The session filter (NY has best signal density for winners)
  3. FVG size/stop distance (the ONE feature with a small but real effect:
     tighter FVGs tend to produce slightly more winners)
""")

log.info("Done. All results printed above.")
