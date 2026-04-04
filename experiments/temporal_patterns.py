"""
Temporal Patterns Analysis for NQ Signal Feature Database
=========================================================
Analyzes hour, day, sub-session, and calendar effects on signal quality.
Uses proper statistical tests with multiple-comparison corrections.
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────

def safe_pf(wins_r: float, losses_r: float) -> str:
    """Profit factor as string, handle zero denominator."""
    if losses_r == 0:
        return "inf" if wins_r > 0 else "n/a"
    return f"{abs(wins_r / losses_r):.2f}"


def compute_metrics(group: pd.DataFrame) -> dict:
    """Compute standard metrics for a group of signals."""
    n = len(group)
    if n == 0:
        return {"count": 0, "win_rate": np.nan, "avgR": np.nan, "sumR": 0,
                "maxDD_R": np.nan, "PPDD": np.nan, "PF": "n/a", "medR": np.nan}
    wins = (group["outcome_label"] == 1).sum()
    wr = wins / n
    avgR = group["outcome_r"].mean()
    sumR = group["outcome_r"].sum()
    medR = group["outcome_r"].median()

    # Compute max drawdown from cumulative R
    cum = group["outcome_r"].cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    maxDD = abs(dd.min()) if len(dd) > 0 else 0
    ppdd = sumR / maxDD if maxDD > 0 else (np.inf if sumR > 0 else 0)

    gross_wins = group.loc[group["outcome_r"] > 0, "outcome_r"].sum()
    gross_losses = group.loc[group["outcome_r"] < 0, "outcome_r"].sum()
    pf = safe_pf(gross_wins, gross_losses)

    return {"count": n, "win_rate": wr, "avgR": avgR, "sumR": sumR,
            "medR": medR, "maxDD_R": maxDD, "PPDD": ppdd, "PF": pf}


def print_table(rows: list[dict], title: str):
    """Print a formatted table."""
    if not rows:
        print(f"\n{title}: No data\n")
        return
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    cols = list(rows[0].keys())
    # Determine column widths
    widths = {}
    for c in cols:
        w = len(str(c))
        for r in rows:
            w = max(w, len(fmt_val(r[c])))
        widths[c] = w + 2

    header = "".join(str(c).rjust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = "".join(fmt_val(r[c]).rjust(widths[c]) for c in cols)
        print(line)
    print()


def fmt_val(v) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return "n/a"
        if abs(v) > 100:
            return f"{v:.1f}"
        return f"{v:.3f}"
    return str(v)


def bootstrap_mean_ci(data, n_boot=5000, ci=0.95, seed=42):
    """Bootstrap confidence interval for mean."""
    rng = np.random.RandomState(seed)
    means = np.array([data.sample(len(data), replace=True, random_state=rng.randint(0, 1e9)).mean()
                       for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi


def permutation_test(groups_r: list[np.ndarray], n_perm=10000, seed=42):
    """Permutation test: is there a significant difference in mean R across groups?"""
    rng = np.random.RandomState(seed)
    all_data = np.concatenate(groups_r)
    group_sizes = [len(g) for g in groups_r]
    observed_F = _F_stat(groups_r)

    count_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(all_data)
        perm_groups = []
        idx = 0
        for s in group_sizes:
            perm_groups.append(perm[idx:idx + s])
            idx += s
        if _F_stat(perm_groups) >= observed_F:
            count_ge += 1
    return count_ge / n_perm


def _F_stat(groups):
    """Compute F-statistic for one-way ANOVA."""
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    if ss_within == 0 or k <= 1:
        return 0
    return (ss_between / (k - 1)) / (ss_within / (n_total - k))


# ──────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────

print("Loading signal feature database...")
df = pd.read_parquet("data/signal_feature_database.parquet")
print(f"  Total signals: {len(df):,}")

filt = df[df["passes_all_filters"]].copy()
print(f"  Filtered (passes_all_filters): {len(filt):,}")
print(f"  Filtered sumR: {filt['outcome_r'].sum():.1f}R")
print(f"  Filtered avgR: {filt['outcome_r'].mean():.3f}R")

# Add derived time columns
df["date"] = pd.to_datetime(df["bar_time_et"]).dt.date
df["month"] = pd.to_datetime(df["bar_time_et"]).dt.month
df["year"] = pd.to_datetime(df["bar_time_et"]).dt.year
df["week_of_month"] = (pd.to_datetime(df["bar_time_et"]).dt.day - 1) // 7 + 1
df["day_name"] = pd.to_datetime(df["bar_time_et"]).dt.day_name()

filt = df[df["passes_all_filters"]].copy()

# ======================================================================
# 1. HOUR-OF-DAY ANALYSIS
# ======================================================================

print("\n" + "#" * 90)
print("#  1. HOUR-OF-DAY ANALYSIS")
print("#" * 90)

# --- All signals ---
rows = []
for h in sorted(df["hour_et"].unique()):
    g = df[df["hour_et"] == h]
    m = compute_metrics(g)
    rows.append({"hour": h, **m})
print_table(rows, "ALL SIGNALS by Hour (ET)")

# --- Filtered signals ---
rows_f = []
for h in sorted(filt["hour_et"].unique()):
    g = filt[filt["hour_et"] == h]
    m = compute_metrics(g)
    rows_f.append({"hour": h, **m})
print_table(rows_f, "FILTERED SIGNALS by Hour (ET)")

# --- Statistical test: Kruskal-Wallis on outcome_r by hour ---
# For all signals
hour_groups_all = [df.loc[df["hour_et"] == h, "outcome_r"].values
                    for h in sorted(df["hour_et"].unique())]
hour_groups_all = [g for g in hour_groups_all if len(g) >= 5]
kw_stat, kw_p = stats.kruskal(*hour_groups_all)
print(f"Kruskal-Wallis (ALL, outcome_r ~ hour): H={kw_stat:.2f}, p={kw_p:.6f}")

# For filtered signals
hour_groups_filt = [filt.loc[filt["hour_et"] == h, "outcome_r"].values
                     for h in sorted(filt["hour_et"].unique())]
hour_groups_filt = [g for g in hour_groups_filt if len(g) >= 5]
if len(hour_groups_filt) >= 2:
    kw_stat_f, kw_p_f = stats.kruskal(*hour_groups_filt)
    print(f"Kruskal-Wallis (FILTERED, outcome_r ~ hour): H={kw_stat_f:.2f}, p={kw_p_f:.6f}")

# Chi-squared on win rate by hour (filtered)
ct = pd.crosstab(filt["hour_et"], filt["outcome_label"].apply(lambda x: 1 if x == 1 else 0))
if ct.shape[0] >= 2 and ct.shape[1] >= 2:
    chi2, chi_p, dof, _ = stats.chi2_contingency(ct)
    print(f"Chi-squared (FILTERED, win_label ~ hour): chi2={chi2:.2f}, p={chi_p:.6f}, dof={dof}")

# Best/worst hours
print("\n--- Best & Worst Hours (FILTERED, by avgR) ---")
filt_hour_stats = []
for h in sorted(filt["hour_et"].unique()):
    g = filt[filt["hour_et"] == h]
    if len(g) >= 5:
        filt_hour_stats.append({"hour": h, "n": len(g), "avgR": g["outcome_r"].mean(),
                                 "sumR": g["outcome_r"].sum(), "wr": (g["outcome_label"] == 1).mean()})
filt_hour_stats.sort(key=lambda x: x["avgR"], reverse=True)
print(f"  BEST:  hour={filt_hour_stats[0]['hour']:2d} ET  avgR={filt_hour_stats[0]['avgR']:+.3f}  sumR={filt_hour_stats[0]['sumR']:+.1f}  n={filt_hour_stats[0]['n']}  wr={filt_hour_stats[0]['wr']:.1%}")
print(f"  WORST: hour={filt_hour_stats[-1]['hour']:2d} ET  avgR={filt_hour_stats[-1]['avgR']:+.3f}  sumR={filt_hour_stats[-1]['sumR']:+.1f}  n={filt_hour_stats[-1]['n']}  wr={filt_hour_stats[-1]['wr']:.1%}")


# ======================================================================
# 2. DAY-OF-WEEK ANALYSIS
# ======================================================================

print("\n" + "#" * 90)
print("#  2. DAY-OF-WEEK ANALYSIS")
print("#" * 90)

day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 6: "Sun"}

# --- All signals ---
rows = []
for d in sorted(df["day_of_week"].unique()):
    g = df[df["day_of_week"] == d]
    m = compute_metrics(g)
    rows.append({"day": day_names.get(d, str(d)), "dow": d, **m})
print_table(rows, "ALL SIGNALS by Day of Week")

# --- Filtered signals ---
rows_f = []
for d in sorted(filt["day_of_week"].unique()):
    g = filt[filt["day_of_week"] == d]
    m = compute_metrics(g)
    rows_f.append({"day": day_names.get(d, str(d)), "dow": d, **m})
print_table(rows_f, "FILTERED SIGNALS by Day of Week")

# Statistical tests
dow_groups_all = [df.loc[df["day_of_week"] == d, "outcome_r"].values
                   for d in sorted(df["day_of_week"].unique()) if len(df[df["day_of_week"] == d]) >= 5]
kw_stat_d, kw_p_d = stats.kruskal(*dow_groups_all)
print(f"Kruskal-Wallis (ALL, outcome_r ~ day_of_week): H={kw_stat_d:.2f}, p={kw_p_d:.6f}")

dow_groups_filt = [filt.loc[filt["day_of_week"] == d, "outcome_r"].values
                    for d in sorted(filt["day_of_week"].unique()) if len(filt[filt["day_of_week"] == d]) >= 5]
if len(dow_groups_filt) >= 2:
    kw_stat_df, kw_p_df = stats.kruskal(*dow_groups_filt)
    print(f"Kruskal-Wallis (FILTERED, outcome_r ~ day_of_week): H={kw_stat_df:.2f}, p={kw_p_df:.6f}")

# Pairwise comparisons (filtered): Mon/Fri vs Tue/Wed/Thu
mon_fri = filt[filt["day_of_week"].isin([0, 4])]["outcome_r"]
mid_week = filt[filt["day_of_week"].isin([1, 2, 3])]["outcome_r"]
u_stat, u_p = stats.mannwhitneyu(mon_fri, mid_week, alternative="two-sided")
print(f"\nMann-Whitney U (FILTERED): Mon+Fri vs Tue+Wed+Thu")
print(f"  Mon+Fri: n={len(mon_fri)}, avgR={mon_fri.mean():.3f}, sumR={mon_fri.sum():.1f}")
print(f"  Tue-Thu: n={len(mid_week)}, avgR={mid_week.mean():.3f}, sumR={mid_week.sum():.1f}")
print(f"  U={u_stat:.0f}, p={u_p:.4f}")

# Individual day comparisons with Bonferroni
print("\n--- Pairwise Day Comparisons (FILTERED, Bonferroni-corrected) ---")
day_vals = sorted(filt["day_of_week"].unique())
n_comparisons = len(list(combinations(day_vals, 2)))
for d1, d2 in combinations(day_vals, 2):
    g1 = filt.loc[filt["day_of_week"] == d1, "outcome_r"]
    g2 = filt.loc[filt["day_of_week"] == d2, "outcome_r"]
    if len(g1) >= 5 and len(g2) >= 5:
        u, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        adj_p = min(p * n_comparisons, 1.0)
        sig = "*" if adj_p < 0.05 else ""
        print(f"  {day_names.get(d1,'?')} vs {day_names.get(d2,'?')}: "
              f"avgR {g1.mean():+.3f} vs {g2.mean():+.3f}, "
              f"raw_p={p:.4f}, adj_p={adj_p:.4f} {sig}")


# ======================================================================
# 3. HOUR x DIRECTION CROSS
# ======================================================================

print("\n" + "#" * 90)
print("#  3. HOUR x DIRECTION CROSS (Filtered Signals)")
print("#" * 90)

rows = []
for h in sorted(filt["hour_et"].unique()):
    for d_name, d_val in [("Long", 1), ("Short", -1)]:
        g = filt[(filt["hour_et"] == h) & (filt["signal_dir"] == d_val)]
        if len(g) == 0:
            continue
        m = compute_metrics(g)
        rows.append({"hour": h, "dir": d_name, **m})
print_table(rows, "FILTERED: Hour x Direction")

# Identify hours with large long/short divergence
print("--- Hours with Long/Short divergence (|avgR_long - avgR_short| > 0.3) ---")
for h in sorted(filt["hour_et"].unique()):
    gl = filt[(filt["hour_et"] == h) & (filt["signal_dir"] == 1)]
    gs = filt[(filt["hour_et"] == h) & (filt["signal_dir"] == -1)]
    if len(gl) >= 3 and len(gs) >= 3:
        diff = gl["outcome_r"].mean() - gs["outcome_r"].mean()
        if abs(diff) > 0.3:
            print(f"  Hour {h:2d}: Long avgR={gl['outcome_r'].mean():+.3f} (n={len(gl)}), "
                  f"Short avgR={gs['outcome_r'].mean():+.3f} (n={len(gs)}), "
                  f"diff={diff:+.3f}")


# ======================================================================
# 4. HOUR x SIGNAL TYPE CROSS
# ======================================================================

print("\n" + "#" * 90)
print("#  4. HOUR x SIGNAL TYPE CROSS (Filtered Signals)")
print("#" * 90)

rows = []
for h in sorted(filt["hour_et"].unique()):
    for st in ["trend", "mss"]:
        g = filt[(filt["hour_et"] == h) & (filt["signal_type"] == st)]
        if len(g) == 0:
            continue
        m = compute_metrics(g)
        rows.append({"hour": h, "type": st, **m})
print_table(rows, "FILTERED: Hour x Signal Type")

# Aggregate: trend vs mss overall
for st in ["trend", "mss"]:
    g = filt[filt["signal_type"] == st]
    m = compute_metrics(g)
    print(f"  {st:5s}: n={m['count']}, avgR={m['avgR']:.3f}, sumR={m['sumR']:.1f}, "
          f"wr={m['win_rate']:.1%}, PF={m['PF']}, PPDD={m['PPDD']:.2f}")


# ======================================================================
# 5. SUB-SESSION GRANULARITY
# ======================================================================

print("\n" + "#" * 90)
print("#  5. SUB-SESSION GRANULARITY")
print("#" * 90)

# --- All signals ---
rows = []
for ss in ["asia", "london", "ny_am", "ny_lunch", "ny_pm", "other"]:
    g = df[df["sub_session"] == ss]
    if len(g) == 0:
        continue
    m = compute_metrics(g)
    rows.append({"sub_session": ss, **m})
print_table(rows, "ALL SIGNALS by Sub-Session")

# --- Filtered ---
rows_f = []
for ss in ["asia", "london", "ny_am", "ny_lunch", "ny_pm", "other"]:
    g = filt[filt["sub_session"] == ss]
    if len(g) == 0:
        continue
    m = compute_metrics(g)
    rows_f.append({"sub_session": ss, **m})
print_table(rows_f, "FILTERED SIGNALS by Sub-Session")

# Fine-grained NY breakdown: 10:00-10:30, 10:30-11:00, 11:00-12:00, 12:00-13:00, 13:00-14:00, 14:00-15:00, 15:00-16:00
filt_ny = filt[filt["session"] == "ny"].copy()
filt_ny["minute_et"] = pd.to_datetime(filt_ny["bar_time_et"]).dt.minute
filt_ny["half_hour"] = filt_ny["hour_et"].astype(str) + ":" + np.where(filt_ny["minute_et"] < 30, "00", "30")

print_table_rows = []
for hh in sorted(filt_ny["half_hour"].unique()):
    g = filt_ny[filt_ny["half_hour"] == hh]
    if len(g) < 3:
        continue
    m = compute_metrics(g)
    print_table_rows.append({"slot": hh, **m})
print_table(print_table_rows, "FILTERED NY Signals by Half-Hour Slot")

# Dead zone analysis: 12:00-13:00 vs rest
if len(filt_ny) > 0:
    lunch = filt_ny[filt_ny["hour_et"].isin([12])]
    non_lunch = filt_ny[~filt_ny["hour_et"].isin([12])]
    print("--- Dead Zone Check (12:xx vs rest of NY, FILTERED) ---")
    print(f"  12:xx    : n={len(lunch)}, avgR={lunch['outcome_r'].mean():.3f}" if len(lunch) > 0 else "  12:xx    : no signals")
    print(f"  Non-12:xx: n={len(non_lunch)}, avgR={non_lunch['outcome_r'].mean():.3f}")


# ======================================================================
# 6. INTRADAY EDGE DECAY
# ======================================================================

print("\n" + "#" * 90)
print("#  6. INTRADAY EDGE DECAY (Filtered Signals)")
print("#" * 90)

print("\n  Hour(ET) |  n  |  avgR  | sumR  | win_rate |  trend")
print("  " + "-" * 60)
prev_avg = None
for h in sorted(filt["hour_et"].unique()):
    g = filt[filt["hour_et"] == h]
    if len(g) < 3:
        continue
    avg = g["outcome_r"].mean()
    arrow = ""
    if prev_avg is not None:
        if avg > prev_avg + 0.05:
            arrow = " ^"
        elif avg < prev_avg - 0.05:
            arrow = " v"
        else:
            arrow = " ="
    prev_avg = avg
    wr = (g["outcome_label"] == 1).mean()
    print(f"    {h:2d}     | {len(g):3d} | {avg:+.3f} | {g['outcome_r'].sum():+.1f} | {wr:.1%}    |{arrow}")

# Spearman correlation: hour vs outcome_r (filtered, NY hours only)
filt_ny_only = filt[filt["session"] == "ny"]
if len(filt_ny_only) >= 10:
    rho, rho_p = stats.spearmanr(filt_ny_only["hour_et"], filt_ny_only["outcome_r"])
    print(f"\n  Spearman correlation (NY hours, hour vs outcome_r): rho={rho:.3f}, p={rho_p:.4f}")


# ======================================================================
# 7. WEEK-OF-MONTH / MONTH / CALENDAR EFFECTS
# ======================================================================

print("\n" + "#" * 90)
print("#  7. CALENDAR EFFECTS (Filtered Signals)")
print("#" * 90)

# Month of year
rows = []
month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
for mo in sorted(filt["month"].unique()):
    g = filt[filt["month"] == mo]
    m = compute_metrics(g)
    rows.append({"month": month_names.get(mo, str(mo)), **m})
print_table(rows, "FILTERED by Month")

# Week of month
rows = []
for w in sorted(filt["week_of_month"].unique()):
    g = filt[filt["week_of_month"] == w]
    m = compute_metrics(g)
    rows.append({"week": w, **m})
print_table(rows, "FILTERED by Week-of-Month")

# Kruskal-Wallis for month
month_groups = [filt.loc[filt["month"] == mo, "outcome_r"].values
                 for mo in sorted(filt["month"].unique())]
month_groups = [g for g in month_groups if len(g) >= 3]
if len(month_groups) >= 2:
    kw_mo, kw_mo_p = stats.kruskal(*month_groups)
    print(f"Kruskal-Wallis (FILTERED, outcome_r ~ month): H={kw_mo:.2f}, p={kw_mo_p:.6f}")

# Year over year
rows = []
for y in sorted(filt["year"].unique()):
    g = filt[filt["year"] == y]
    m = compute_metrics(g)
    rows.append({"year": y, **m})
print_table(rows, "FILTERED by Year")

# OpEx week detection (3rd Friday of month)
filt_dt = pd.to_datetime(filt["bar_time_et"])
filt["is_opex_week"] = False
for idx in filt.index:
    dt = filt_dt.loc[idx]
    # Find 3rd Friday: first day of month, find first Friday, add 14 days
    first_of_month = dt.replace(day=1)
    first_friday = first_of_month + pd.Timedelta(days=(4 - first_of_month.weekday()) % 7)
    third_friday = first_friday + pd.Timedelta(days=14)
    # OpEx week = Mon-Fri of that week
    opex_mon = third_friday - pd.Timedelta(days=4)
    opex_fri = third_friday
    if opex_mon.date() <= dt.date() <= opex_fri.date():
        filt.loc[idx, "is_opex_week"] = True

opex = filt[filt["is_opex_week"]]
non_opex = filt[~filt["is_opex_week"]]
print(f"\n--- OpEx Week Effect (FILTERED) ---")
print(f"  OpEx week:     n={len(opex)}, avgR={opex['outcome_r'].mean():.3f}, sumR={opex['outcome_r'].sum():.1f}" if len(opex) > 0 else "  OpEx week: no signals")
print(f"  Non-OpEx week: n={len(non_opex)}, avgR={non_opex['outcome_r'].mean():.3f}, sumR={non_opex['outcome_r'].sum():.1f}")
if len(opex) >= 5 and len(non_opex) >= 5:
    u, p = stats.mannwhitneyu(opex["outcome_r"], non_opex["outcome_r"], alternative="two-sided")
    print(f"  Mann-Whitney U={u:.0f}, p={p:.4f}")

# Month-start vs month-end (day 1-5 vs day 25-31)
filt["day_of_month"] = pd.to_datetime(filt["bar_time_et"]).dt.day
month_start = filt[filt["day_of_month"] <= 5]
month_end = filt[filt["day_of_month"] >= 25]
month_mid = filt[(filt["day_of_month"] > 5) & (filt["day_of_month"] < 25)]
print(f"\n--- Month Position Effect (FILTERED) ---")
print(f"  Start (day 1-5):   n={len(month_start)}, avgR={month_start['outcome_r'].mean():.3f}" if len(month_start) > 0 else "  Start: no signals")
print(f"  Mid (day 6-24):    n={len(month_mid)}, avgR={month_mid['outcome_r'].mean():.3f}" if len(month_mid) > 0 else "  Mid: no signals")
print(f"  End (day 25-31):   n={len(month_end)}, avgR={month_end['outcome_r'].mean():.3f}" if len(month_end) > 0 else "  End: no signals")


# ======================================================================
# 8. ACTIONABLE RECOMMENDATIONS + SIGNIFICANCE TESTS
# ======================================================================

print("\n" + "#" * 90)
print("#  8. ACTIONABLE RECOMMENDATIONS")
print("#" * 90)

recommendations = []

# 8a: Remove worst hours
print("\n--- 8a: Hour-based filters ---")
for h in sorted(filt["hour_et"].unique()):
    g = filt[filt["hour_et"] == h]
    if len(g) < 5:
        continue
    if g["outcome_r"].mean() < -0.1:
        excl = filt[filt["hour_et"] != h]
        delta_R = excl["outcome_r"].sum() - filt["outcome_r"].sum()
        delta_n = len(excl) - len(filt)
        # Bootstrap significance
        lo, hi = bootstrap_mean_ci(g["outcome_r"])
        sig = "YES" if hi < 0 else "no"
        rec = f"EXCLUDE hour {h}: avgR={g['outcome_r'].mean():+.3f} (n={len(g)}), " \
              f"95% CI=[{lo:+.3f}, {hi:+.3f}], sig={sig}, " \
              f"impact: {delta_R:+.1f}R, {delta_n:+d} trades"
        print(f"  {rec}")
        recommendations.append(rec)

# 8b: Direction x Hour filters
print("\n--- 8b: Direction x Hour filters ---")
for h in sorted(filt["hour_et"].unique()):
    for d_val, d_name in [(1, "Long"), (-1, "Short")]:
        g = filt[(filt["hour_et"] == h) & (filt["signal_dir"] == d_val)]
        if len(g) < 5:
            continue
        if g["outcome_r"].mean() < -0.2:
            excl = filt[~((filt["hour_et"] == h) & (filt["signal_dir"] == d_val))]
            delta_R = excl["outcome_r"].sum() - filt["outcome_r"].sum()
            lo, hi = bootstrap_mean_ci(g["outcome_r"])
            sig = "YES" if hi < 0 else "no"
            rec = f"EXCLUDE {d_name} at hour {h}: avgR={g['outcome_r'].mean():+.3f} (n={len(g)}), " \
                  f"95% CI=[{lo:+.3f}, {hi:+.3f}], sig={sig}, " \
                  f"impact: {delta_R:+.1f}R, {len(filt)-len(excl):d} trades removed"
            print(f"  {rec}")
            recommendations.append(rec)

# 8c: Day-of-week filter refinement
print("\n--- 8c: Day-of-week filter refinement ---")
for d in sorted(filt["day_of_week"].unique()):
    g = filt[filt["day_of_week"] == d]
    if len(g) < 10:
        continue
    rest = filt[filt["day_of_week"] != d]
    u, p = stats.mannwhitneyu(g["outcome_r"], rest["outcome_r"], alternative="two-sided")
    lo, hi = bootstrap_mean_ci(g["outcome_r"])
    print(f"  {day_names.get(d, '?'):3s}: avgR={g['outcome_r'].mean():+.3f}, "
          f"95% CI=[{lo:+.3f}, {hi:+.3f}], "
          f"vs rest p={p:.4f}, sumR={g['outcome_r'].sum():+.1f} (n={len(g)})")

# 8d: Signal type x time filters
print("\n--- 8d: Signal type time preferences ---")
for st in ["trend", "mss"]:
    g = filt[filt["signal_type"] == st]
    if len(g) < 10:
        continue
    # Best hours for this type
    best_h = None
    best_avg = -999
    worst_h = None
    worst_avg = 999
    for h in sorted(g["hour_et"].unique()):
        hg = g[g["hour_et"] == h]
        if len(hg) >= 5:
            avg = hg["outcome_r"].mean()
            if avg > best_avg:
                best_avg, best_h = avg, h
            if avg < worst_avg:
                worst_avg, worst_h = avg, h
    if best_h is not None:
        print(f"  {st}: best hour={best_h} (avgR={best_avg:+.3f}), worst hour={worst_h} (avgR={worst_avg:+.3f})")

# 8e: Cumulative impact of all recommendations
print("\n--- 8e: Combined filter impact estimate ---")
# Apply all hour exclusions with negative avgR
exclude_mask = pd.Series(False, index=filt.index)
excluded_details = []
for h in sorted(filt["hour_et"].unique()):
    g = filt[filt["hour_et"] == h]
    if len(g) >= 5 and g["outcome_r"].mean() < -0.15:
        lo, hi = bootstrap_mean_ci(g["outcome_r"])
        if hi < 0:  # Only if statistically significant
            exclude_mask |= (filt["hour_et"] == h)
            excluded_details.append(f"hour {h}")

# Also check direction x hour
for h in sorted(filt["hour_et"].unique()):
    for d_val, d_name in [(1, "Long"), (-1, "Short")]:
        g = filt[(filt["hour_et"] == h) & (filt["signal_dir"] == d_val)]
        if len(g) >= 5 and g["outcome_r"].mean() < -0.25:
            lo, hi = bootstrap_mean_ci(g["outcome_r"])
            if hi < 0:
                mask = (filt["hour_et"] == h) & (filt["signal_dir"] == d_val)
                if not exclude_mask[mask].all():  # Don't double-count
                    exclude_mask |= mask
                    excluded_details.append(f"{d_name} at hour {h}")

surviving = filt[~exclude_mask]
print(f"  Excluded filters: {', '.join(excluded_details) if excluded_details else 'none'}")
print(f"  Before: {len(filt)} trades, sumR={filt['outcome_r'].sum():.1f}, avgR={filt['outcome_r'].mean():.3f}")
print(f"  After:  {len(surviving)} trades, sumR={surviving['outcome_r'].sum():.1f}, avgR={surviving['outcome_r'].mean():.3f}")
print(f"  Delta:  {len(surviving) - len(filt):+d} trades, {surviving['outcome_r'].sum() - filt['outcome_r'].sum():+.1f}R")

# Final: Permutation test on the combined exclusion
if len(excluded_details) > 0 and len(surviving) >= 10:
    excluded_signals = filt[exclude_mask]
    # Permutation test: is the excluded group significantly worse?
    perm_p = permutation_test([surviving["outcome_r"].values, excluded_signals["outcome_r"].values],
                               n_perm=5000)
    print(f"  Permutation test (excluded vs surviving): p={perm_p:.4f}")


# ======================================================================
# SUMMARY
# ======================================================================

print("\n" + "#" * 90)
print("#  SUMMARY")
print("#" * 90)
print(f"""
Dataset: {len(df):,} total signals, {len(filt):,} pass all filters
Filtered performance: sumR={filt['outcome_r'].sum():.1f}, avgR={filt['outcome_r'].mean():.3f}

Key findings (see detailed tables above for full data):
""")

# Reprint key metrics
print("Filtered signals by hour (sorted by avgR):")
filt_hour_stats.sort(key=lambda x: x["avgR"], reverse=True)
for s in filt_hour_stats:
    bar = "+" * max(0, int(s["avgR"] * 10)) + "-" * max(0, int(-s["avgR"] * 10))
    print(f"  Hour {s['hour']:2d}: avgR={s['avgR']:+.3f}  sumR={s['sumR']:+6.1f}  n={s['n']:3d}  wr={s['wr']:.0%}  {bar}")

print("\nDone.")
