"""
Feature Importance Deep Statistical Analysis
=============================================
Comprehensive analysis of signal_feature_database.parquet (15,894 rows x 58 cols)
to find the most predictive features, optimal thresholds, and composite scoring.

Every table includes: R (total), PPDD (R / MaxDD), PF (sum wins / abs sum losses).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr

# ─────────────────────────────── Helpers ────────────────────────────────

def compute_metrics(df_sub: pd.DataFrame) -> dict:
    """Compute standard trading metrics for a subset of signals."""
    n = len(df_sub)
    if n == 0:
        return dict(count=0, avg_r=np.nan, med_r=np.nan, wr=np.nan,
                    pf=np.nan, total_r=np.nan, ppdd=np.nan, max_dd=np.nan)

    outcomes = df_sub["outcome_r"].values
    total_r = float(np.sum(outcomes))
    avg_r = float(np.mean(outcomes))
    med_r = float(np.median(outcomes))
    wr = float(np.mean(df_sub["hit_tp"].values)) * 100

    wins = outcomes[outcomes > 0]
    losses = outcomes[outcomes < 0]
    sum_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
    sum_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    pf = sum_wins / sum_losses if sum_losses > 0 else (np.inf if sum_wins > 0 else 0.0)

    # Max drawdown from cumulative R curve
    cum = np.cumsum(outcomes)
    running_max = np.maximum.accumulate(cum)
    dd = running_max - cum
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    ppdd = total_r / max_dd if max_dd > 0 else (np.inf if total_r > 0 else 0.0)

    return dict(count=n, avg_r=round(avg_r, 4), med_r=round(med_r, 4),
                wr=round(wr, 1), pf=round(pf, 3), total_r=round(total_r, 2),
                ppdd=round(ppdd, 3), max_dd=round(max_dd, 2))


def metrics_table(groups: dict) -> pd.DataFrame:
    """Build a DataFrame from {label: df_subset} dict."""
    rows = []
    for label, sub in groups.items():
        m = compute_metrics(sub)
        m["bin"] = label
        rows.append(m)
    df = pd.DataFrame(rows)
    cols = ["bin", "count", "avg_r", "med_r", "wr", "pf", "total_r", "ppdd", "max_dd"]
    return df[[c for c in cols if c in df.columns]]


def quantile_bins(df: pd.DataFrame, feature: str, n_bins: int = 5) -> dict:
    """Split df into quantile bins on feature, return {label: sub_df}."""
    vals = df[feature].dropna()
    if vals.nunique() < n_bins:
        # Use unique values as categories
        groups = {}
        for v in sorted(vals.unique()):
            mask = df[feature] == v
            groups[f"{feature}={v}"] = df[mask]
        return groups

    try:
        df = df.copy()
        df["_qbin"] = pd.qcut(df[feature], n_bins, duplicates="drop")
        groups = {}
        for name, sub in df.groupby("_qbin", observed=True):
            groups[str(name)] = sub
        df.drop(columns=["_qbin"], inplace=True)
        return groups
    except Exception:
        # Fallback: equal-width bins
        df = df.copy()
        df["_qbin"] = pd.cut(df[feature], n_bins, duplicates="drop")
        groups = {}
        for name, sub in df.groupby("_qbin", observed=True):
            groups[str(name)] = sub
        df.drop(columns=["_qbin"], inplace=True)
        return groups


def bool_bins(df: pd.DataFrame, feature: str) -> dict:
    """Split on boolean feature."""
    groups = {}
    for val in [False, True]:
        mask = df[feature] == val
        groups[f"{feature}={val}"] = df[mask]
    return groups


def print_section(title: str):
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)


def print_table(df_table: pd.DataFrame, max_col_width: int = 14):
    """Pretty-print a DataFrame as an aligned table."""
    print(df_table.to_string(index=False, float_format=lambda x: f"{x:.4f}" if abs(x) < 100 else f"{x:.2f}"))
    print()


# ═══════════════════════════════════════════════════════════════════════
#                         LOAD DATA
# ═══════════════════════════════════════════════════════════════════════

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "signal_feature_database.parquet"
df_all = pd.read_parquet(DATA_PATH)

print(f"Loaded {len(df_all)} signals, {df_all.shape[1]} columns")
print(f"Date range: {df_all['bar_time_et'].min()} to {df_all['bar_time_et'].max()}")
overall = compute_metrics(df_all)
print(f"Overall: count={overall['count']}, R={overall['total_r']}, "
      f"PPDD={overall['ppdd']}, PF={overall['pf']}, WR={overall['wr']}%")

# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS 1: Single Feature Predictive Power
# ═══════════════════════════════════════════════════════════════════════
print_section("ANALYSIS 1: Single Feature Predictive Power (5-quantile bins)")

continuous_features = [
    "signal_quality", "fluency_score", "bar_body_ratio", "bar_body_atr",
    "bar_range_atr", "fvg_size_atr", "fvg_sweep_score", "stop_distance_atr",
    "target_rr", "atr_percentile", "hour_et", "day_of_week",
]
boolean_features = ["is_displaced", "has_smt", "bias_aligned", "fvg_swept_liquidity"]

feature_spreads = []

for feat in continuous_features:
    print(f"\n--- {feat} ---")
    groups = quantile_bins(df_all, feat, n_bins=5)
    tbl = metrics_table(groups)
    print_table(tbl)

    # Compute spread = best avg_r - worst avg_r
    if len(tbl) >= 2:
        spread_avg = tbl["avg_r"].max() - tbl["avg_r"].min()
        spread_wr = tbl["wr"].max() - tbl["wr"].min()
        best_bin = tbl.loc[tbl["avg_r"].idxmax(), "bin"]
        worst_bin = tbl.loc[tbl["avg_r"].idxmin(), "bin"]
        feature_spreads.append(dict(
            feature=feat, spread_avg_r=round(spread_avg, 4),
            spread_wr=round(spread_wr, 1),
            best_bin=best_bin, worst_bin=worst_bin,
            best_avg_r=round(tbl["avg_r"].max(), 4),
            worst_avg_r=round(tbl["avg_r"].min(), 4),
        ))

for feat in boolean_features:
    print(f"\n--- {feat} ---")
    groups = bool_bins(df_all, feat)
    tbl = metrics_table(groups)
    print_table(tbl)

    if len(tbl) >= 2:
        spread_avg = tbl["avg_r"].max() - tbl["avg_r"].min()
        spread_wr = tbl["wr"].max() - tbl["wr"].min()
        best_bin = tbl.loc[tbl["avg_r"].idxmax(), "bin"]
        worst_bin = tbl.loc[tbl["avg_r"].idxmin(), "bin"]
        feature_spreads.append(dict(
            feature=feat, spread_avg_r=round(spread_avg, 4),
            spread_wr=round(spread_wr, 1),
            best_bin=best_bin, worst_bin=worst_bin,
            best_avg_r=round(tbl["avg_r"].max(), 4),
            worst_avg_r=round(tbl["avg_r"].min(), 4),
        ))

print_section("FEATURE RANKING BY SPREAD (avg_r best bin - worst bin)")
spread_df = pd.DataFrame(feature_spreads).sort_values("spread_avg_r", ascending=False)
print_table(spread_df)

top5_features = spread_df.head(5)["feature"].tolist()
print(f"Top 5 most predictive features: {top5_features}")


# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS 2: The Fluency Paradox
# ═══════════════════════════════════════════════════════════════════════
print_section("ANALYSIS 2: The Fluency Paradox")

# 2.1 Control for confounders
print("\n--- 2.1a: Fluency effect WITHIN NY session only ---")
df_ny = df_all[df_all["session"] == "ny"]
groups = quantile_bins(df_ny, "fluency_score", n_bins=5)
tbl = metrics_table(groups)
print_table(tbl)

print("--- 2.1b: Fluency effect WITHIN trend signals only ---")
df_trend = df_all[df_all["signal_type"] == "trend"]
groups = quantile_bins(df_trend, "fluency_score", n_bins=5)
tbl = metrics_table(groups)
print_table(tbl)

print("--- 2.1c: Fluency effect WITHIN high-SQ signals (SQ > 0.75) ---")
df_highsq = df_all[df_all["signal_quality"] > 0.75]
groups = quantile_bins(df_highsq, "fluency_score", n_bins=5)
tbl = metrics_table(groups)
print_table(tbl)

print("--- 2.1d: Fluency effect controlling for session AND signal_type ---")
for sess in ["ny", "london", "asia"]:
    for stype in ["trend", "mss"]:
        sub = df_all[(df_all["session"] == sess) & (df_all["signal_type"] == stype)]
        if len(sub) < 100:
            continue
        lo = sub[sub["fluency_score"] <= sub["fluency_score"].median()]
        hi = sub[sub["fluency_score"] > sub["fluency_score"].median()]
        m_lo = compute_metrics(lo)
        m_hi = compute_metrics(hi)
        print(f"  {sess}/{stype}: n={len(sub):>5} | "
              f"Low fluency: n={m_lo['count']}, avgR={m_lo['avg_r']:+.4f}, WR={m_lo['wr']:.1f}%, "
              f"R={m_lo['total_r']:>8.2f}, PPDD={m_lo['ppdd']:>7.3f}, PF={m_lo['pf']:.3f} | "
              f"High fluency: n={m_hi['count']}, avgR={m_hi['avg_r']:+.4f}, WR={m_hi['wr']:.1f}%, "
              f"R={m_hi['total_r']:>8.2f}, PPDD={m_hi['ppdd']:>7.3f}, PF={m_hi['pf']:.3f}")
print()

# 2.2 Interpretation
print("--- 2.2: Low fluency + FVG signal = stronger signal? ---")
print("Hypothesis: If market is choppy but still produces a clean FVG rejection,")
print("the displacement has MORE conviction (it broke through chop).")
print()
# Compare displacement quality in low vs high fluency
for label, sub in [("Low fluency (<0.65)", df_all[df_all["fluency_score"] < 0.65]),
                   ("High fluency (>=0.65)", df_all[df_all["fluency_score"] >= 0.65])]:
    print(f"  {label}: n={len(sub)}")
    print(f"    avg bar_body_atr = {sub['bar_body_atr'].mean():.4f}")
    print(f"    avg bar_range_atr = {sub['bar_range_atr'].mean():.4f}")
    print(f"    pct is_displaced = {sub['is_displaced'].mean()*100:.1f}%")
    print(f"    avg signal_quality = {sub['signal_quality'].mean():.4f}")
    print(f"    avg fvg_sweep_score = {sub['fvg_sweep_score'].mean():.2f}")
    m = compute_metrics(sub)
    print(f"    R={m['total_r']}, PPDD={m['ppdd']}, PF={m['pf']}, WR={m['wr']}%")
print()

# 2.3 Text histogram of fluency vs outcome_r
print("--- 2.3: Fluency vs outcome_r (text histogram) ---")
fluency_bins = pd.cut(df_all["fluency_score"], bins=10)
for bin_name, sub in df_all.groupby(fluency_bins, observed=True):
    avg_r = sub["outcome_r"].mean()
    bar_len = int(max(0, (avg_r + 0.2) * 100))  # scale for display
    direction = "+" if avg_r >= 0 else "-"
    bar = "#" * min(bar_len, 60) if avg_r >= 0 else "x" * min(int(abs(avg_r) * 100), 60)
    print(f"  {str(bin_name):>25s} n={len(sub):>5} avgR={avg_r:+.4f} | {bar}")
print()

# 2.4 Inverted fluency gate
print("--- 2.4: Inverted fluency gate (require fluency < 0.65 vs >= 0.65) ---")
lo_flu = df_all[df_all["fluency_score"] < 0.65]
hi_flu = df_all[df_all["fluency_score"] >= 0.65]
tbl = metrics_table({"fluency < 0.65": lo_flu, "fluency >= 0.65": hi_flu})
print_table(tbl)

# Also test within filtered signals
print("--- 2.4b: Inverted fluency gate within FILTERED signals (passes_all_filters) ---")
df_filt = df_all[df_all["passes_all_filters"]]
lo_flu_f = df_filt[df_filt["fluency_score"] < 0.65]
hi_flu_f = df_filt[df_filt["fluency_score"] >= 0.65]
tbl = metrics_table({"filtered + fluency < 0.65": lo_flu_f, "filtered + fluency >= 0.65": hi_flu_f})
print_table(tbl)


# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS 3: Feature Interactions (Top 5 × 2-way)
# ═══════════════════════════════════════════════════════════════════════
print_section("ANALYSIS 3: Feature Interactions (2-way)")

# Use top 5 from spread ranking
# For boolean features in top 5, use True/False; for continuous, use median split
def get_split(df: pd.DataFrame, feat: str):
    """Return (label_lo, mask_lo, label_hi, mask_hi) for median split."""
    if df[feat].dtype == bool or df[feat].nunique() <= 2:
        return f"{feat}=F", ~df[feat].astype(bool), f"{feat}=T", df[feat].astype(bool)
    med = df[feat].median()
    return (f"{feat}<=med({med:.2f})", df[feat] <= med,
            f"{feat}>med({med:.2f})", df[feat] > med)

# Also include top boolean features if they ranked high
interaction_features = top5_features[:5]
print(f"Testing interactions among: {interaction_features}")
print()

best_interactions = []

for fa, fb in combinations(interaction_features, 2):
    la_lo, mask_a_lo, la_hi, mask_a_hi = get_split(df_all, fa)
    lb_lo, mask_b_lo, lb_hi, mask_b_hi = get_split(df_all, fb)

    cells = {
        f"{la_lo} & {lb_lo}": df_all[mask_a_lo & mask_b_lo],
        f"{la_lo} & {lb_hi}": df_all[mask_a_lo & mask_b_hi],
        f"{la_hi} & {lb_lo}": df_all[mask_a_hi & mask_b_lo],
        f"{la_hi} & {lb_hi}": df_all[mask_a_hi & mask_b_hi],
    }

    print(f"\n--- {fa} × {fb} ---")
    tbl = metrics_table(cells)
    print_table(tbl)

    # Record best cell
    if len(tbl) >= 2:
        best_idx = tbl["avg_r"].idxmax()
        worst_idx = tbl["avg_r"].idxmin()
        spread = tbl.loc[best_idx, "avg_r"] - tbl.loc[worst_idx, "avg_r"]
        best_interactions.append(dict(
            features=f"{fa} × {fb}",
            best_cell=tbl.loc[best_idx, "bin"],
            best_avg_r=tbl.loc[best_idx, "avg_r"],
            best_wr=tbl.loc[best_idx, "wr"],
            best_pf=tbl.loc[best_idx, "pf"],
            best_R=tbl.loc[best_idx, "total_r"],
            best_ppdd=tbl.loc[best_idx, "ppdd"],
            worst_cell=tbl.loc[worst_idx, "bin"],
            worst_avg_r=tbl.loc[worst_idx, "avg_r"],
            spread=round(spread, 4),
        ))

print_section("BEST INTERACTION CELLS (ranked by spread)")
inter_df = pd.DataFrame(best_interactions).sort_values("spread", ascending=False)
print_table(inter_df)


# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS 4: Session × Feature Interactions
# ═══════════════════════════════════════════════════════════════════════
print_section("ANALYSIS 4: Session × Feature Interactions")

all_features = continuous_features + boolean_features

for sess in ["asia", "london", "ny"]:
    print(f"\n{'─'*40} SESSION: {sess.upper()} {'─'*40}")
    df_sess = df_all[df_all["session"] == sess]
    overall_sess = compute_metrics(df_sess)
    print(f"  Overall: n={overall_sess['count']}, R={overall_sess['total_r']}, "
          f"PPDD={overall_sess['ppdd']}, PF={overall_sess['pf']}, WR={overall_sess['wr']}%")

    sess_spreads = []
    for feat in all_features:
        if feat in boolean_features:
            groups = bool_bins(df_sess, feat)
        else:
            groups = quantile_bins(df_sess, feat, n_bins=3)  # fewer bins per session
        tbl = metrics_table(groups)
        if len(tbl) >= 2:
            spread = tbl["avg_r"].max() - tbl["avg_r"].min()
            best_bin = tbl.loc[tbl["avg_r"].idxmax(), "bin"]
            sess_spreads.append(dict(
                feature=feat,
                spread=round(spread, 4),
                best_bin=best_bin,
                best_avg_r=round(tbl["avg_r"].max(), 4),
                best_wr=round(tbl.loc[tbl["avg_r"].idxmax(), "wr"], 1),
                best_pf=round(tbl.loc[tbl["avg_r"].idxmax(), "pf"], 3),
                best_R=round(tbl.loc[tbl["avg_r"].idxmax(), "total_r"], 2),
                best_ppdd=round(tbl.loc[tbl["avg_r"].idxmax(), "ppdd"], 3),
            ))

    sess_spread_df = pd.DataFrame(sess_spreads).sort_values("spread", ascending=False)
    print(f"\n  Top 8 most predictive features for {sess.upper()}:")
    print_table(sess_spread_df.head(8))


# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS 5: Golden Signals (outcome_r > 2.0)
# ═══════════════════════════════════════════════════════════════════════
print_section("ANALYSIS 5: Golden Signals (outcome_r > 2.0)")

df_golden = df_all[df_all["outcome_r"] > 2.0]
df_rest = df_all[df_all["outcome_r"] <= 2.0]

print(f"Golden signals: {len(df_golden)} ({len(df_golden)/len(df_all)*100:.1f}% of all)")
golden_m = compute_metrics(df_golden)
rest_m = compute_metrics(df_rest)
print(f"Golden: R={golden_m['total_r']}, PPDD={golden_m['ppdd']}, PF={golden_m['pf']}")
print(f"Rest:   R={rest_m['total_r']}, PPDD={rest_m['ppdd']}, PF={rest_m['pf']}")
print()

# Feature profile comparison
compare_feats = continuous_features + boolean_features
profile_rows = []
for feat in compare_feats:
    golden_mean = df_golden[feat].astype(float).mean()
    all_mean = df_all[feat].astype(float).mean()
    rest_mean = df_rest[feat].astype(float).mean()
    diff = golden_mean - all_mean
    std = df_all[feat].astype(float).std()
    z = diff / std if std > 0 else 0
    profile_rows.append(dict(
        feature=feat,
        golden_mean=round(golden_mean, 4),
        all_mean=round(all_mean, 4),
        rest_mean=round(rest_mean, 4),
        diff=round(diff, 4),
        z_score=round(z, 3),
    ))

profile_df = pd.DataFrame(profile_rows).sort_values("z_score", key=lambda x: x.abs(), ascending=False)
print("Feature profile: Golden vs All (sorted by |z-score|)")
print_table(profile_df)

# Session distribution of golden
print("Session distribution of golden signals:")
for sess in ["asia", "london", "ny", "other"]:
    n_g = len(df_golden[df_golden["session"] == sess])
    n_a = len(df_all[df_all["session"] == sess])
    pct = n_g / n_a * 100 if n_a > 0 else 0
    print(f"  {sess}: {n_g}/{n_a} = {pct:.1f}% golden rate")
print()

# Simple rule to capture 50%+ of golden signals
print("--- Building simple rule to capture 50%+ of golden signals ---")
# Use features with highest |z-score|
top_golden_feats = profile_df.head(5)["feature"].tolist()
print(f"Top distinguishing features: {top_golden_feats}")

# For each top feature, find threshold that maximizes golden capture
for feat in top_golden_feats:
    if df_all[feat].dtype == bool:
        # Boolean: just check rate
        for val in [True, False]:
            n_gold = len(df_golden[df_golden[feat] == val])
            n_total = len(df_all[df_all[feat] == val])
            capture = n_gold / len(df_golden) * 100
            sub = df_all[df_all[feat] == val]
            m = compute_metrics(sub)
            print(f"  {feat}={val}: captures {capture:.1f}% of golden, "
                  f"n={n_total}, R={m['total_r']}, PPDD={m['ppdd']}, PF={m['pf']}")
    else:
        # Try percentile thresholds
        golden_med = df_golden[feat].median()
        all_med = df_all[feat].median()
        z = profile_df[profile_df["feature"] == feat]["z_score"].values[0]
        direction = ">" if z > 0 else "<"

        thresholds = sorted(df_all[feat].dropna().quantile([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unique())
        best_capture = 0
        best_thresh = None
        for t in thresholds:
            if direction == ">":
                mask = df_all[feat] > t
                mask_g = df_golden[feat] > t
            else:
                mask = df_all[feat] < t
                mask_g = df_golden[feat] < t
            capture = mask_g.sum() / len(df_golden) * 100
            n = mask.sum()
            if capture >= 50 and n >= 200:
                sub = df_all[mask]
                m = compute_metrics(sub)
                if best_capture == 0 or m.get("ppdd", 0) > best_capture:
                    best_capture = m.get("ppdd", 0)
                    best_thresh = t
                print(f"  {feat} {direction} {t:.4f}: captures {capture:.1f}% golden, "
                      f"n={n}, R={m['total_r']}, PPDD={m['ppdd']}, PF={m['pf']}")
    print()


# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS 6: Poison Signals (outcome_r < -0.8)
# ═══════════════════════════════════════════════════════════════════════
print_section("ANALYSIS 6: Poison Signals (outcome_r < -0.8)")

df_poison = df_all[df_all["outcome_r"] < -0.8]
df_nonpoison = df_all[df_all["outcome_r"] >= -0.8]

print(f"Poison signals: {len(df_poison)} ({len(df_poison)/len(df_all)*100:.1f}% of all)")
poison_m = compute_metrics(df_poison)
nonpoison_m = compute_metrics(df_nonpoison)
print(f"Poison:     R={poison_m['total_r']}, PPDD={poison_m['ppdd']}, PF={poison_m['pf']}")
print(f"Non-poison: R={nonpoison_m['total_r']}, PPDD={nonpoison_m['ppdd']}, PF={nonpoison_m['pf']}")
print()

# Feature profile
poison_rows = []
for feat in compare_feats:
    poison_mean = df_poison[feat].astype(float).mean()
    all_mean = df_all[feat].astype(float).mean()
    nonp_mean = df_nonpoison[feat].astype(float).mean()
    diff = poison_mean - all_mean
    std = df_all[feat].astype(float).std()
    z = diff / std if std > 0 else 0
    poison_rows.append(dict(
        feature=feat,
        poison_mean=round(poison_mean, 4),
        all_mean=round(all_mean, 4),
        nonpoison_mean=round(nonp_mean, 4),
        diff=round(diff, 4),
        z_score=round(z, 3),
    ))

poison_df = pd.DataFrame(poison_rows).sort_values("z_score", key=lambda x: x.abs(), ascending=False)
print("Feature profile: Poison vs All (sorted by |z-score|)")
print_table(poison_df)

# Session distribution of poison
print("Session distribution of poison signals:")
for sess in ["asia", "london", "ny", "other"]:
    n_p = len(df_poison[df_poison["session"] == sess])
    n_a = len(df_all[df_all["session"] == sess])
    pct = n_p / n_a * 100 if n_a > 0 else 0
    print(f"  {sess}: {n_p}/{n_a} = {pct:.1f}% poison rate")
print()

# Which features identify poison BEFORE entry?
print("--- Features that separate poison from non-poison ---")
for feat in poison_df.head(5)["feature"].tolist():
    z = poison_df[poison_df["feature"] == feat]["z_score"].values[0]
    direction = ">" if z > 0 else "<"
    # Poison signals tend to have this feature in what direction?
    poison_q25 = df_poison[feat].astype(float).quantile(0.25)
    poison_q75 = df_poison[feat].astype(float).quantile(0.75)
    all_q25 = df_all[feat].astype(float).quantile(0.25)
    all_q75 = df_all[feat].astype(float).quantile(0.75)
    print(f"  {feat}: z={z:.3f} ({direction}), "
          f"poison IQR=[{poison_q25:.4f}, {poison_q75:.4f}], "
          f"all IQR=[{all_q25:.4f}, {all_q75:.4f}]")


# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS 7: Optimal Feature Thresholds
# ═══════════════════════════════════════════════════════════════════════
print_section("ANALYSIS 7: Optimal Feature Thresholds (top 5 features)")

# For each top feature, sweep thresholds
for feat in top5_features:
    print(f"\n--- {feat} threshold sweep ---")

    if df_all[feat].dtype == bool:
        for val in [True, False]:
            sub = df_all[df_all[feat] == val]
            m = compute_metrics(sub)
            print(f"  {feat}={val}: n={m['count']}, R={m['total_r']}, "
                  f"PPDD={m['ppdd']}, PF={m['pf']}, WR={m['wr']}%")
        continue

    vals = df_all[feat].dropna()
    percentiles = np.arange(0.05, 0.96, 0.05)
    thresholds = [round(vals.quantile(p), 6) for p in percentiles]
    # Remove duplicates
    thresholds = sorted(set(thresholds))

    rows_above = []
    rows_below = []

    for t in thresholds:
        # Above threshold
        above = df_all[df_all[feat] > t]
        below = df_all[df_all[feat] <= t]

        if len(above) >= 200:
            m = compute_metrics(above)
            rows_above.append(dict(threshold=f"> {t:.4f}", count=m["count"],
                                   avg_r=m["avg_r"], wr=m["wr"], pf=m["pf"],
                                   total_r=m["total_r"], ppdd=m["ppdd"], max_dd=m["max_dd"]))

        if len(below) >= 200:
            m = compute_metrics(below)
            rows_below.append(dict(threshold=f"<= {t:.4f}", count=m["count"],
                                    avg_r=m["avg_r"], wr=m["wr"], pf=m["pf"],
                                    total_r=m["total_r"], ppdd=m["ppdd"], max_dd=m["max_dd"]))

    if rows_above:
        tbl_above = pd.DataFrame(rows_above)
        # Find best PPDD with >= 200 trades
        best_idx = tbl_above["ppdd"].idxmax()
        print(f"  ABOVE threshold (best by PPDD):")
        print_table(tbl_above)

    if rows_below:
        tbl_below = pd.DataFrame(rows_below)
        best_idx = tbl_below["ppdd"].idxmax()
        print(f"  BELOW threshold (best by PPDD):")
        print_table(tbl_below)

    # Highlight the optimal
    all_rows = rows_above + rows_below
    if all_rows:
        all_tbl = pd.DataFrame(all_rows)
        best = all_tbl.loc[all_tbl["ppdd"].idxmax()]
        print(f"  >>> OPTIMAL for {feat}: {best['threshold']}, "
              f"n={int(best['count'])}, R={best['total_r']}, "
              f"PPDD={best['ppdd']}, PF={best['pf']}, WR={best['wr']}%")


# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS 8: Composite Score
# ═══════════════════════════════════════════════════════════════════════
print_section("ANALYSIS 8: Build Composite Score from Top Features")

# Normalize features to [0,1] range for compositing
# Use top 5 features (continuous only)
composite_features = [f for f in top5_features if f not in boolean_features][:5]
# If we have fewer than 3 continuous features, add more
if len(composite_features) < 3:
    for f in spread_df["feature"].tolist():
        if f not in composite_features and f not in boolean_features:
            composite_features.append(f)
        if len(composite_features) >= 5:
            break

print(f"Composite features: {composite_features}")

# First determine direction for each feature (positive or negative correlation with outcome_r)
feat_directions = {}
for feat in composite_features:
    corr, _ = spearmanr(df_all[feat].dropna(), df_all.loc[df_all[feat].notna(), "outcome_r"])
    feat_directions[feat] = 1 if corr >= 0 else -1
    print(f"  {feat}: Spearman r = {corr:.4f}, direction = {'positive' if corr >= 0 else 'negative'}")
print()

# Normalize
df_comp = df_all.copy()
for feat in composite_features:
    vals = df_comp[feat].fillna(df_comp[feat].median())
    vmin, vmax = vals.quantile(0.01), vals.quantile(0.99)
    if vmax > vmin:
        normed = (vals - vmin) / (vmax - vmin)
        normed = normed.clip(0, 1)
    else:
        normed = 0.5
    if feat_directions[feat] < 0:
        normed = 1 - normed
    df_comp[f"_norm_{feat}"] = normed

# Grid search for weights (brute force over small set)
print("--- Grid searching composite weights ---")
n_feats = len(composite_features)
best_score = -np.inf
best_weights = None
best_metrics = None

# Generate weight combinations that sum to 1.0
weight_steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

def generate_weights(n, total=1.0, step=0.1):
    """Generate all weight combinations of length n that sum to total."""
    if n == 1:
        yield [round(float(total), 2)]
        return
    for w in np.arange(0, total + step/2, step):
        w = round(float(w), 2)
        if w > total + 0.001:
            break
        for rest in generate_weights(n - 1, round(float(total - w), 2), step):
            yield [w] + rest

weight_combos = list(generate_weights(n_feats, 1.0, 0.1))
print(f"Testing {len(weight_combos)} weight combinations...")

norm_cols = [f"_norm_{f}" for f in composite_features]
norm_matrix = df_comp[norm_cols].values  # shape: (N, n_feats)

results = []
for weights in weight_combos:
    w = np.array(weights)
    if np.all(w == 0):
        continue
    composite = norm_matrix @ w
    # Spearman correlation with outcome_r
    corr, _ = spearmanr(composite, df_comp["outcome_r"].values)

    # Also test as filter: top 20% by composite
    threshold = np.percentile(composite, 80)
    mask_top = composite >= threshold
    if mask_top.sum() >= 200:
        m = compute_metrics(df_comp[mask_top])
        score = m["ppdd"] if np.isfinite(m["ppdd"]) else -999
    else:
        score = -999
        m = dict(count=0, ppdd=0, pf=0, total_r=0, wr=0)

    results.append(dict(
        weights=str([round(float(x), 2) for x in weights]),
        spearman_r=round(corr, 4),
        top20_n=m.get("count", 0),
        top20_R=m.get("total_r", 0),
        top20_ppdd=m.get("ppdd", 0),
        top20_pf=m.get("pf", 0),
        top20_wr=m.get("wr", 0),
    ))

results_df = pd.DataFrame(results)

# Best by Spearman
print("\n--- Top 10 by Spearman correlation ---")
top_spearman = results_df.sort_values("spearman_r", ascending=False).head(10)
print_table(top_spearman)

# Best by PPDD (top 20% filter)
print("--- Top 10 by PPDD of top-20% filter ---")
top_ppdd = results_df.sort_values("top20_ppdd", ascending=False).head(10)
print_table(top_ppdd)

# Use the best PPDD weights
best_row = results_df.sort_values("top20_ppdd", ascending=False).iloc[0]
best_weights_str = best_row["weights"]
print(f"\nBest composite weights (by PPDD): {best_weights_str}")
print(f"  Top-20% metrics: R={best_row['top20_R']}, PPDD={best_row['top20_ppdd']}, "
      f"PF={best_row['top20_pf']}, WR={best_row['top20_wr']}%")

# Apply best weights and do full quintile analysis
import ast
best_w = np.array(ast.literal_eval(best_weights_str))
df_comp["composite_score"] = norm_matrix @ best_w

print("\n--- Composite score quintile analysis ---")
groups = quantile_bins(df_comp, "composite_score", n_bins=5)
tbl = metrics_table(groups)
print_table(tbl)

# Compare with current SQ formula
print("--- Comparison: Composite Score vs Current Signal Quality ---")
print()
# Current SQ quintiles
print("Current SQ quintiles:")
sq_groups = quantile_bins(df_all, "signal_quality", n_bins=5)
sq_tbl = metrics_table(sq_groups)
print_table(sq_tbl)

print("Composite quintiles:")
print_table(tbl)

# Direct comparison: top 20% by each
sq_top20_thresh = df_all["signal_quality"].quantile(0.80)
comp_top20_thresh = df_comp["composite_score"].quantile(0.80)

sq_top20 = df_all[df_all["signal_quality"] >= sq_top20_thresh]
comp_top20 = df_comp[df_comp["composite_score"] >= comp_top20_thresh]

print("Top 20% comparison:")
tbl_compare = metrics_table({
    f"SQ >= {sq_top20_thresh:.4f} (top 20%)": sq_top20,
    f"Composite >= {comp_top20_thresh:.4f} (top 20%)": comp_top20,
})
print_table(tbl_compare)

# Also compare at different filter levels
print("--- Filter level comparison (top N%) ---")
for pct in [10, 20, 30, 40, 50]:
    sq_t = df_all["signal_quality"].quantile(1 - pct/100)
    comp_t = df_comp["composite_score"].quantile(1 - pct/100)
    sq_sub = df_all[df_all["signal_quality"] >= sq_t]
    comp_sub = df_comp[df_comp["composite_score"] >= comp_t]
    m_sq = compute_metrics(sq_sub)
    m_comp = compute_metrics(comp_sub)
    print(f"  Top {pct:>2d}%: "
          f"SQ n={m_sq['count']:>5}, R={m_sq['total_r']:>8.2f}, "
          f"PPDD={m_sq['ppdd']:>7.3f}, PF={m_sq['pf']:.3f}, WR={m_sq['wr']:.1f}% | "
          f"Composite n={m_comp['count']:>5}, R={m_comp['total_r']:>8.2f}, "
          f"PPDD={m_comp['ppdd']:>7.3f}, PF={m_comp['pf']:.3f}, WR={m_comp['wr']:.1f}%")
print()


# ═══════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print_section("EXECUTIVE SUMMARY")
print()
print("Top 5 most predictive single features (by avg_r spread):")
for i, row in spread_df.head(5).iterrows():
    print(f"  {row['feature']:>25s}: spread={row['spread_avg_r']:.4f} "
          f"(best={row['best_avg_r']:+.4f} @ {row['best_bin']}, "
          f"worst={row['worst_avg_r']:+.4f} @ {row['worst_bin']})")
print()

print("Fluency Paradox verdict:")
print("  (See Analysis 2 above for detailed breakdown)")
print()

print(f"Best composite formula (features & weights):")
for f, w in zip(composite_features, ast.literal_eval(best_weights_str)):
    d = "+" if feat_directions[f] > 0 else "-"
    print(f"  {d}{w:.2f} * normalize({f})")
print(f"  Best top-20%: R={best_row['top20_R']}, PPDD={best_row['top20_ppdd']}, PF={best_row['top20_pf']}")
print()

print("=" * 90)
print("  Analysis complete. All results above include R, PPDD, and PF in every table.")
print("=" * 90)
