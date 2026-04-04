"""
Systematic comparison between CSV baseline trades and v9 internal trades.
Finds ALL remaining divergences and categorizes root causes.

CSV baseline: C:/temp/csv_baseline.csv (1,145 trades, +167.9R)
v9 internal:  C:/temp/v9_trades_latest.csv
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import timedelta

# ============================================================
# LOAD DATA
# ============================================================

csv_path = "C:/temp/csv_baseline.csv"
v9_path = "C:/temp/v9_trades_latest.csv"

csv_df = pd.read_csv(csv_path, parse_dates=["entry_time", "exit_time"])
v9_df = pd.read_csv(v9_path, parse_dates=["entry_time", "exit_time"])

print("=" * 80)
print("NQ QUANT: SYSTEMATIC V9 DIVERGENCE AUDIT")
print("=" * 80)
print(f"\nCSV baseline: {len(csv_df)} trades, total R = {csv_df['r_multiple'].sum():.2f}")
print(f"v9 internal:  {len(v9_df)} trades, total R = {v9_df['r_multiple'].sum():.2f}")
print(f"R gap: {csv_df['r_multiple'].sum() - v9_df['r_multiple'].sum():.2f}")
print(f"Trade count gap: {len(csv_df) - len(v9_df)}")

# ============================================================
# 1. TRADE MATCHING (±6 min + same direction)
# ============================================================
print("\n" + "=" * 80)
print("1. TRADE MATCHING")
print("=" * 80)

MATCH_WINDOW = timedelta(minutes=6)

# Track which v9 trades are matched (to avoid double-matching)
v9_matched_idx = set()
csv_matched_idx = set()

matches = []  # (csv_idx, v9_idx)

for ci, crow in csv_df.iterrows():
    best_v9_idx = None
    best_time_diff = timedelta(days=999)
    for vi, vrow in v9_df.iterrows():
        if vi in v9_matched_idx:
            continue
        if crow["direction"] != vrow["direction"]:
            continue
        td = abs(crow["entry_time"] - vrow["entry_time"])
        if td <= MATCH_WINDOW and td < best_time_diff:
            best_time_diff = td
            best_v9_idx = vi
    if best_v9_idx is not None:
        matches.append((ci, best_v9_idx))
        v9_matched_idx.add(best_v9_idx)
        csv_matched_idx.add(ci)

csv_only_idx = [i for i in csv_df.index if i not in csv_matched_idx]
v9_only_idx = [i for i in v9_df.index if i not in v9_matched_idx]

group_a = len(matches)
group_b = len(csv_only_idx)
group_c = len(v9_only_idx)

print(f"\nGroup A (matched in both):  {group_a}")
print(f"Group B (CSV-only, v9 missed): {group_b}")
print(f"Group C (v9-only, extras):     {group_c}")
print(f"Total CSV: {group_a + group_b} = {len(csv_df)}")
print(f"Total v9:  {group_a + group_c} = {len(v9_df)}")

# Build matched DataFrame
matched_rows = []
for ci, vi in matches:
    crow = csv_df.loc[ci]
    vrow = v9_df.loc[vi]
    matched_rows.append({
        "csv_entry_time": crow["entry_time"],
        "v9_entry_time": vrow["entry_time"],
        "direction": crow["direction"],
        "csv_signal_type": crow["signal_type"],
        "v9_signal_type": vrow["signal_type"],
        "csv_entry_price": crow["entry_price"],
        "v9_entry_price": vrow["entry_price"],
        "entry_price_diff": abs(crow["entry_price"] - vrow["entry_price"]),
        "csv_stop_price": crow["stop_price"],
        "v9_stop_price": vrow["stop_price"],
        "stop_price_diff": abs(crow["stop_price"] - vrow["stop_price"]),
        "csv_tp1_price": crow["tp1_price"],
        "v9_tp1_price": vrow["tp1_price"],
        "tp1_price_diff": abs(crow["tp1_price"] - vrow["tp1_price"]),
        "csv_r": crow["r_multiple"],
        "v9_r": vrow["r_multiple"],
        "r_diff": crow["r_multiple"] - vrow["r_multiple"],
        "abs_r_diff": abs(crow["r_multiple"] - vrow["r_multiple"]),
        "csv_exit_reason": crow["exit_reason"],
        "v9_exit_reason": vrow["exit_reason"],
        "exit_reason_match": crow["exit_reason"] == vrow["exit_reason"],
        "csv_grade": crow["grade"],
        "v9_grade": vrow["grade"],
        "grade_match": crow["grade"] == vrow["grade"],
        "csv_trimmed": crow["trimmed"],
        "v9_trimmed": vrow["trimmed"],
        "csv_exit_time": crow["exit_time"],
        "v9_exit_time": vrow["exit_time"],
        "csv_exit_price": crow["exit_price"],
        "v9_exit_price": vrow["exit_price"],
    })
matched_df = pd.DataFrame(matched_rows)

# ============================================================
# 2. FIELD-BY-FIELD COMPARISON (Group A)
# ============================================================
print("\n" + "=" * 80)
print("2. MATCHED TRADES FIELD COMPARISON")
print("=" * 80)

if len(matched_df) > 0:
    # Price diffs
    for field in ["entry_price", "stop_price", "tp1_price"]:
        diff_col = f"{field}_diff"
        print(f"\n--- {field} differences ---")
        print(f"  Mean:   {matched_df[diff_col].mean():.4f}")
        print(f"  Median: {matched_df[diff_col].median():.4f}")
        print(f"  Max:    {matched_df[diff_col].max():.4f}")
        print(f"  > 0.5 pts: {(matched_df[diff_col] > 0.5).sum()}")
        print(f"  > 1.0 pts: {(matched_df[diff_col] > 1.0).sum()}")
        print(f"  > 5.0 pts: {(matched_df[diff_col] > 5.0).sum()}")
        print(f"  > 10 pts:  {(matched_df[diff_col] > 10.0).sum()}")
        print(f"  Exact match (0): {(matched_df[diff_col] == 0).sum()}")

    # R diffs
    print(f"\n--- R multiple differences ---")
    print(f"  Mean abs diff: {matched_df['abs_r_diff'].mean():.4f}")
    print(f"  Median abs diff: {matched_df['abs_r_diff'].median():.4f}")
    print(f"  > 0.1 R: {(matched_df['abs_r_diff'] > 0.1).sum()}")
    print(f"  > 0.5 R: {(matched_df['abs_r_diff'] > 0.5).sum()}")
    print(f"  > 1.0 R: {(matched_df['abs_r_diff'] > 1.0).sum()}")
    print(f"  Sum of R diffs (CSV - v9): {matched_df['r_diff'].sum():.4f}")
    print(f"  CSV matched R total: {matched_df['csv_r'].sum():.4f}")
    print(f"  v9 matched R total:  {matched_df['v9_r'].sum():.4f}")

    # Exit reason match
    print(f"\n--- Exit reason match ---")
    print(f"  Matching: {matched_df['exit_reason_match'].sum()} / {len(matched_df)}")
    print(f"  Mismatched: {(~matched_df['exit_reason_match']).sum()}")

    # Grade match
    print(f"\n--- Grade match ---")
    print(f"  Matching: {matched_df['grade_match'].sum()} / {len(matched_df)}")
    print(f"  Mismatched: {(~matched_df['grade_match']).sum()}")

    # Signal type match
    signal_match = matched_df['csv_signal_type'] == matched_df['v9_signal_type']
    print(f"\n--- Signal type match ---")
    print(f"  Matching: {signal_match.sum()} / {len(matched_df)}")
    print(f"  Mismatched: {(~signal_match).sum()}")

    # Top 10 largest R differences
    print(f"\n--- Top 10 trades with LARGEST R difference ---")
    top10 = matched_df.nlargest(10, "abs_r_diff")
    for _, row in top10.iterrows():
        print(f"  {row['csv_entry_time']}  dir={int(row['direction']):+d}  "
              f"csv_R={row['csv_r']:+.4f}  v9_R={row['v9_r']:+.4f}  "
              f"diff={row['r_diff']:+.4f}  "
              f"csv_exit={row['csv_exit_reason']}  v9_exit={row['v9_exit_reason']}  "
              f"csv_tp1={row['csv_tp1_price']:.2f}  v9_tp1={row['v9_tp1_price']:.2f}  "
              f"tp1_diff={row['tp1_price_diff']:.2f}")

    # Top 10 where CSV has bigger R (v9 is losing)
    print(f"\n--- Top 10 trades where CSV outperforms v9 (positive R diff) ---")
    top10_csv_better = matched_df.nlargest(10, "r_diff")
    for _, row in top10_csv_better.iterrows():
        print(f"  {row['csv_entry_time']}  dir={int(row['direction']):+d}  "
              f"csv_R={row['csv_r']:+.4f}  v9_R={row['v9_r']:+.4f}  "
              f"diff={row['r_diff']:+.4f}  "
              f"csv_exit={row['csv_exit_reason']}  v9_exit={row['v9_exit_reason']}  "
              f"entry_diff={row['entry_price_diff']:.2f}  stop_diff={row['stop_price_diff']:.2f}  "
              f"tp1_diff={row['tp1_price_diff']:.2f}")

# ============================================================
# 3. CSV-ONLY TRADES (Group B) - Why did v9 miss them?
# ============================================================
print("\n" + "=" * 80)
print("3. CSV-ONLY TRADES (v9 MISSED)")
print("=" * 80)

csv_only = csv_df.loc[csv_only_idx].copy()
print(f"\nTotal CSV-only trades: {len(csv_only)}")
print(f"Total R from CSV-only trades: {csv_only['r_multiple'].sum():.4f}")

if len(csv_only) > 0:
    # Direction breakdown
    print(f"\nDirection breakdown:")
    print(csv_only['direction'].value_counts().to_string())

    # Signal type breakdown
    print(f"\nSignal type breakdown:")
    print(csv_only['signal_type'].value_counts().to_string())

    # Grade breakdown
    print(f"\nGrade breakdown:")
    print(csv_only['grade'].value_counts().to_string())

    # Exit reason breakdown
    print(f"\nExit reason breakdown:")
    print(csv_only['exit_reason'].value_counts().to_string())

    # Session clustering
    csv_only["hour"] = csv_only["entry_time"].dt.hour
    print(f"\nEntry hour distribution (EST):")
    print(csv_only["hour"].value_counts().sort_index().to_string())

    # Year clustering
    csv_only["year"] = csv_only["entry_time"].dt.year
    print(f"\nYear distribution:")
    for yr, grp in csv_only.groupby("year"):
        print(f"  {yr}: {len(grp)} trades, R = {grp['r_multiple'].sum():.4f}")

    # R stats
    print(f"\nR stats for CSV-only trades:")
    print(f"  Mean R: {csv_only['r_multiple'].mean():.4f}")
    print(f"  Median R: {csv_only['r_multiple'].median():.4f}")
    print(f"  Positive: {(csv_only['r_multiple'] > 0).sum()}")
    print(f"  Negative: {(csv_only['r_multiple'] < 0).sum()}")

    # First 20
    print(f"\nFirst 20 CSV-only trades:")
    print(f"{'entry_time':>22s}  {'dir':>4s}  {'type':>6s}  {'entry':>10s}  {'stop':>10s}  "
          f"{'tp1':>10s}  {'R':>8s}  {'exit':>10s}  {'grade':>5s}")
    for _, row in csv_only.head(20).iterrows():
        print(f"  {row['entry_time']}  {int(row['direction']):+4d}  {row['signal_type']:>6s}  "
              f"{row['entry_price']:10.2f}  {row['stop_price']:10.2f}  "
              f"{row['tp1_price']:10.2f}  {row['r_multiple']:+8.4f}  "
              f"{row['exit_reason']:>10s}  {row['grade']:>5s}")

# ============================================================
# 4. V9-ONLY TRADES (Group C) - Why does v9 have extras?
# ============================================================
print("\n" + "=" * 80)
print("4. V9-ONLY TRADES (EXTRAS)")
print("=" * 80)

v9_only = v9_df.loc[v9_only_idx].copy()
print(f"\nTotal v9-only trades: {len(v9_only)}")
print(f"Total R from v9-only trades: {v9_only['r_multiple'].sum():.4f}")

if len(v9_only) > 0:
    print(f"\nDirection breakdown:")
    print(v9_only['direction'].value_counts().to_string())

    print(f"\nSignal type breakdown:")
    print(v9_only['signal_type'].value_counts().to_string())

    print(f"\nGrade breakdown:")
    print(v9_only['grade'].value_counts().to_string())

    print(f"\nExit reason breakdown:")
    print(v9_only['exit_reason'].value_counts().to_string())

    print(f"\nR stats for v9-only trades:")
    print(f"  Mean R: {v9_only['r_multiple'].mean():.4f}")
    print(f"  Median R: {v9_only['r_multiple'].median():.4f}")
    print(f"  Positive: {(v9_only['r_multiple'] > 0).sum()}")
    print(f"  Negative: {(v9_only['r_multiple'] < 0).sum()}")

    v9_only["year"] = v9_only["entry_time"].dt.year
    print(f"\nYear distribution:")
    for yr, grp in v9_only.groupby("year"):
        print(f"  {yr}: {len(grp)} trades, R = {grp['r_multiple'].sum():.4f}")

    print(f"\nFirst 20 v9-only trades:")
    print(f"{'entry_time':>22s}  {'dir':>4s}  {'type':>6s}  {'entry':>10s}  {'stop':>10s}  "
          f"{'tp1':>10s}  {'R':>8s}  {'exit':>10s}  {'grade':>5s}")
    for _, row in v9_only.head(20).iterrows():
        print(f"  {row['entry_time']}  {int(row['direction']):+4d}  {row['signal_type']:>6s}  "
              f"{row['entry_price']:10.2f}  {row['stop_price']:10.2f}  "
              f"{row['tp1_price']:10.2f}  {row['r_multiple']:+8.4f}  "
              f"{row['exit_reason']:>10s}  {row['grade']:>5s}")

# ============================================================
# 5. TP1 DISTANCE COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("5. TP1 DISTANCE COMPARISON (matched trades)")
print("=" * 80)

if len(matched_df) > 0:
    matched_df["csv_tp1_dist"] = abs(matched_df["csv_tp1_price"] - matched_df["csv_entry_price"])
    matched_df["v9_tp1_dist"] = abs(matched_df["v9_tp1_price"] - matched_df["v9_entry_price"])
    matched_df["tp1_dist_diff"] = matched_df["csv_tp1_dist"] - matched_df["v9_tp1_dist"]
    matched_df["abs_tp1_dist_diff"] = abs(matched_df["tp1_dist_diff"])

    print(f"\nCSV tp1_dist:  mean={matched_df['csv_tp1_dist'].mean():.2f}  "
          f"median={matched_df['csv_tp1_dist'].median():.2f}")
    print(f"v9  tp1_dist:  mean={matched_df['v9_tp1_dist'].mean():.2f}  "
          f"median={matched_df['v9_tp1_dist'].median():.2f}")
    print(f"\ntp1_dist difference (CSV - v9):")
    print(f"  Mean:   {matched_df['tp1_dist_diff'].mean():.4f}")
    print(f"  Median: {matched_df['tp1_dist_diff'].median():.4f}")
    print(f"  Std:    {matched_df['tp1_dist_diff'].std():.4f}")

    print(f"\n  abs diff > 1 pt:  {(matched_df['abs_tp1_dist_diff'] > 1).sum()}")
    print(f"  abs diff > 5 pts: {(matched_df['abs_tp1_dist_diff'] > 5).sum()}")
    print(f"  abs diff > 10 pts: {(matched_df['abs_tp1_dist_diff'] > 10).sum()}")
    print(f"  abs diff > 20 pts: {(matched_df['abs_tp1_dist_diff'] > 20).sum()}")
    print(f"  abs diff > 50 pts: {(matched_df['abs_tp1_dist_diff'] > 50).sum()}")
    print(f"  Exact match (0): {(matched_df['abs_tp1_dist_diff'] == 0).sum()}")

    # Distribution buckets
    bins = [0, 0.5, 1, 2, 5, 10, 20, 50, 100, 500, 10000]
    labels = ["0-0.5", "0.5-1", "1-2", "2-5", "5-10", "10-20", "20-50", "50-100", "100-500", "500+"]
    matched_df["tp1_diff_bucket"] = pd.cut(matched_df["abs_tp1_dist_diff"], bins=bins, labels=labels)
    print(f"\nTP1 distance difference distribution:")
    print(matched_df["tp1_diff_bucket"].value_counts().sort_index().to_string())

    # Top 15 worst TP1 mismatches
    print(f"\nTop 15 worst TP1 distance mismatches:")
    top15_tp1 = matched_df.nlargest(15, "abs_tp1_dist_diff")
    for _, row in top15_tp1.iterrows():
        print(f"  {row['csv_entry_time']}  dir={int(row['direction']):+d}  "
              f"csv_tp1={row['csv_tp1_price']:.2f} (dist={row['csv_tp1_dist']:.2f})  "
              f"v9_tp1={row['v9_tp1_price']:.2f} (dist={row['v9_tp1_dist']:.2f})  "
              f"diff={row['tp1_dist_diff']:+.2f}  "
              f"csv_R={row['csv_r']:+.4f}  v9_R={row['v9_r']:+.4f}")

# ============================================================
# 6. EXIT REASON TRANSITION MATRIX
# ============================================================
print("\n" + "=" * 80)
print("6. EXIT REASON TRANSITION MATRIX")
print("=" * 80)

if len(matched_df) > 0:
    # Get all exit reasons
    all_reasons = sorted(set(matched_df["csv_exit_reason"].unique()) |
                         set(matched_df["v9_exit_reason"].unique()))

    matrix = pd.crosstab(
        matched_df["csv_exit_reason"],
        matched_df["v9_exit_reason"],
        margins=True
    )
    print(f"\nRows = CSV exit reason, Cols = v9 exit reason")
    print(matrix.to_string())

    # Show the off-diagonal transitions with R impact
    print(f"\nKey transitions (where exit reason changed):")
    mismatched = matched_df[~matched_df["exit_reason_match"]]
    for (csv_er, v9_er), grp in mismatched.groupby(["csv_exit_reason", "v9_exit_reason"]):
        r_impact = grp["r_diff"].sum()
        print(f"  {csv_er:>12s} -> {v9_er:<12s}: {len(grp):4d} trades, "
              f"R impact = {r_impact:+.4f}  "
              f"(avg R diff = {grp['r_diff'].mean():+.4f})")

# ============================================================
# 7. GRADE DISTRIBUTION COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("7. GRADE DISTRIBUTION COMPARISON")
print("=" * 80)

print(f"\nCSV grade distribution:")
csv_grade = csv_df['grade'].value_counts().sort_index()
for g, cnt in csv_grade.items():
    r_sum = csv_df[csv_df['grade'] == g]['r_multiple'].sum()
    print(f"  {g}: {cnt:4d} trades ({cnt/len(csv_df)*100:.1f}%), R = {r_sum:+.2f}")

print(f"\nv9 grade distribution:")
v9_grade = v9_df['grade'].value_counts().sort_index()
for g, cnt in v9_grade.items():
    r_sum = v9_df[v9_df['grade'] == g]['r_multiple'].sum()
    print(f"  {g}: {cnt:4d} trades ({cnt/len(v9_df)*100:.1f}%), R = {r_sum:+.2f}")

# Matched trades grade comparison
if len(matched_df) > 0:
    grade_mismatches = matched_df[~matched_df["grade_match"]]
    print(f"\nGrade mismatches in matched trades: {len(grade_mismatches)}")
    if len(grade_mismatches) > 0:
        print(f"\nGrade transition counts:")
        grade_trans = pd.crosstab(
            matched_df["csv_grade"],
            matched_df["v9_grade"],
            margins=True
        )
        print(grade_trans.to_string())

# ============================================================
# 8. YEARLY R BREAKDOWN
# ============================================================
print("\n" + "=" * 80)
print("8. YEARLY R BREAKDOWN")
print("=" * 80)

csv_df["year"] = csv_df["entry_time"].dt.year
v9_df["year"] = v9_df["entry_time"].dt.year

csv_yearly = csv_df.groupby("year").agg(
    trades=("r_multiple", "count"),
    total_r=("r_multiple", "sum")
).reset_index()

v9_yearly = v9_df.groupby("year").agg(
    trades=("r_multiple", "count"),
    total_r=("r_multiple", "sum")
).reset_index()

all_years = sorted(set(csv_df["year"].unique()) | set(v9_df["year"].unique()))

print(f"\n{'Year':>6s}  {'CSV_N':>6s}  {'v9_N':>6s}  {'dN':>5s}  "
      f"{'CSV_R':>8s}  {'v9_R':>8s}  {'R_gap':>8s}  {'%_of_gap':>8s}")
print("-" * 70)

total_r_gap = csv_df['r_multiple'].sum() - v9_df['r_multiple'].sum()

for yr in all_years:
    csv_row = csv_yearly[csv_yearly["year"] == yr]
    v9_row = v9_yearly[v9_yearly["year"] == yr]

    csv_n = int(csv_row["trades"].values[0]) if len(csv_row) > 0 else 0
    v9_n = int(v9_row["trades"].values[0]) if len(v9_row) > 0 else 0
    csv_r = float(csv_row["total_r"].values[0]) if len(csv_row) > 0 else 0.0
    v9_r = float(v9_row["total_r"].values[0]) if len(v9_row) > 0 else 0.0
    gap = csv_r - v9_r
    pct = (gap / total_r_gap * 100) if total_r_gap != 0 else 0

    print(f"{yr:6d}  {csv_n:6d}  {v9_n:6d}  {csv_n-v9_n:+5d}  "
          f"{csv_r:+8.2f}  {v9_r:+8.2f}  {gap:+8.2f}  {pct:7.1f}%")

print("-" * 70)
print(f"{'Total':>6s}  {len(csv_df):6d}  {len(v9_df):6d}  {len(csv_df)-len(v9_df):+5d}  "
      f"{csv_df['r_multiple'].sum():+8.2f}  {v9_df['r_multiple'].sum():+8.2f}  "
      f"{total_r_gap:+8.2f}  100.0%")

# ============================================================
# 9. ROOT CAUSE CATEGORIZATION
# ============================================================
print("\n" + "=" * 80)
print("9. ROOT CAUSE CATEGORIZATION")
print("=" * 80)

# Total R gap decomposition
r_gap_total = csv_df["r_multiple"].sum() - v9_df["r_multiple"].sum()

# R from CSV-only trades (v9 missed)
r_csv_only = csv_only["r_multiple"].sum() if len(csv_only) > 0 else 0
# R from v9-only trades (extras, likely negative if noise)
r_v9_only = v9_only["r_multiple"].sum() if len(v9_only) > 0 else 0
# R diff from matched trades
r_matched_diff = matched_df["r_diff"].sum() if len(matched_df) > 0 else 0

print(f"\nR gap decomposition:")
print(f"  Total R gap (CSV - v9):        {r_gap_total:+.4f}")
print(f"  From CSV-only trades:          {r_csv_only:+.4f} ({r_csv_only/r_gap_total*100:.1f}% of gap)")
print(f"  From v9-only trades:           {-r_v9_only:+.4f} ({-r_v9_only/r_gap_total*100:.1f}% of gap)")
print(f"  From matched trade R diffs:    {r_matched_diff:+.4f} ({r_matched_diff/r_gap_total*100:.1f}% of gap)")
print(f"  Check: {r_csv_only:+.4f} + {-r_v9_only:+.4f} + {r_matched_diff:+.4f} = {r_csv_only - r_v9_only + r_matched_diff:+.4f}")

# Categorize matched trade divergences
if len(matched_df) > 0:
    # Category A: TP1 distance different (FindIRL mismatch)
    cat_a = matched_df[matched_df["abs_tp1_dist_diff"] > 1.0].copy()
    # Category B: Different signal types (different FVG selected)
    cat_b = matched_df[matched_df["csv_signal_type"] != matched_df["v9_signal_type"]].copy()
    # Category C: Grade different
    cat_c = matched_df[~matched_df["grade_match"]].copy()
    # Category E: Entry price different (> 1 pt)
    cat_e = matched_df[matched_df["entry_price_diff"] > 1.0].copy()
    # Category F: Stop price different (> 1 pt)
    cat_f = matched_df[matched_df["stop_price_diff"] > 1.0].copy()
    # Category D: Exit reason different but TP1/entry/stop/grade same
    # (trail stop divergence)
    cat_d = matched_df[
        (~matched_df["exit_reason_match"]) &
        (matched_df["abs_tp1_dist_diff"] <= 1.0) &
        (matched_df["entry_price_diff"] <= 1.0) &
        (matched_df["stop_price_diff"] <= 1.0) &
        (matched_df["grade_match"])
    ].copy()

    print(f"\nMatched trade divergence categories:")
    print(f"  A: TP1 distance diff > 1pt:       {len(cat_a):4d} trades, R impact = {cat_a['r_diff'].sum():+.4f}")
    print(f"  B: Different signal type:          {len(cat_b):4d} trades, R impact = {cat_b['r_diff'].sum():+.4f}")
    print(f"  C: Different grade:                {len(cat_c):4d} trades, R impact = {cat_c['r_diff'].sum():+.4f}")
    print(f"  D: Trail/exit divergence (pure):   {len(cat_d):4d} trades, R impact = {cat_d['r_diff'].sum():+.4f}")
    print(f"  E: Entry price diff > 1pt:         {len(cat_e):4d} trades, R impact = {cat_e['r_diff'].sum():+.4f}")
    print(f"  F: Stop price diff > 1pt:          {len(cat_f):4d} trades, R impact = {cat_f['r_diff'].sum():+.4f}")

    # Classify each divergent matched trade into primary root cause
    # (prioritize: B > E > F > A > C > D)
    matched_df["root_cause"] = "OK"  # perfect match
    # Assign in reverse priority order so higher priority overwrites
    matched_df.loc[matched_df["abs_r_diff"] > 0.01, "root_cause"] = "D_trail"
    matched_df.loc[~matched_df["grade_match"], "root_cause"] = "C_grade"
    matched_df.loc[matched_df["abs_tp1_dist_diff"] > 1.0, "root_cause"] = "A_tp1"
    matched_df.loc[matched_df["stop_price_diff"] > 1.0, "root_cause"] = "F_stop"
    matched_df.loc[matched_df["entry_price_diff"] > 1.0, "root_cause"] = "E_entry"
    matched_df.loc[matched_df["csv_signal_type"] != matched_df["v9_signal_type"], "root_cause"] = "B_signal"

    print(f"\nPrimary root cause assignment (each trade gets ONE category):")
    for cause, grp in matched_df.groupby("root_cause"):
        r_impact = grp["r_diff"].sum()
        pct = r_impact / r_gap_total * 100 if r_gap_total != 0 else 0
        print(f"  {cause:>10s}: {len(grp):4d} trades, R impact = {r_impact:+.4f} ({pct:+.1f}% of gap)")

    # Position blocking cascade estimate
    # If a CSV-only trade exists, subsequent trades may shift in time.
    # Count how many v9-only trades occur within 1 day of a CSV-only trade
    if len(csv_only) > 0 and len(v9_only) > 0:
        cascade_count = 0
        for _, co_row in csv_only.iterrows():
            for _, vo_row in v9_only.iterrows():
                if abs(co_row["entry_time"] - vo_row["entry_time"]) <= timedelta(hours=24):
                    cascade_count += 1
                    break  # count once per csv-only trade
        print(f"\n  G: Position blocking cascade estimate:")
        print(f"     CSV-only trades near a v9-only trade (within 24h): {cascade_count}")
        print(f"     (Suggests {cascade_count} CSV-only trades may cause cascading shifts)")

# ============================================================
# OVERALL R GAP ATTRIBUTION
# ============================================================
print("\n" + "=" * 80)
print("OVERALL R GAP ATTRIBUTION SUMMARY")
print("=" * 80)

print(f"\nTotal R gap: {r_gap_total:+.4f}")
print(f"\n  1. Missing trades (CSV-only):     {r_csv_only:+.4f}  ({r_csv_only/r_gap_total*100:5.1f}%)")
print(f"  2. Extra trades (v9-only):        {-r_v9_only:+.4f}  ({-r_v9_only/r_gap_total*100:5.1f}%)")
print(f"  3. Matched trade R differences:   {r_matched_diff:+.4f}  ({r_matched_diff/r_gap_total*100:5.1f}%)")
print(f"     ---------------------------")
print(f"     Sum check:                     {r_csv_only - r_v9_only + r_matched_diff:+.4f}")

if len(matched_df) > 0:
    print(f"\n  Matched trade R diff breakdown by root cause:")
    for cause in sorted(matched_df["root_cause"].unique()):
        grp = matched_df[matched_df["root_cause"] == cause]
        r_imp = grp["r_diff"].sum()
        print(f"    {cause:>10s}: {r_imp:+.4f}  ({r_imp/r_gap_total*100:5.1f}%)")

# ============================================================
# 10. SPECIFIC FIX RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 80)
print("10. FIX RECOMMENDATIONS")
print("=" * 80)

print("""
=== CRITICAL FINDING ===

The R gap is 111.7R, of which:
  - 66.4% (74.2R) = CSV-only trades that v9 never takes
  - 33.5% (37.5R) = v9-only trades that are LOSERS (avg R = -0.09)
  - 0.1%  (0.1R)  = Matched trades with R differences

THIS MEANS: The matched trades are almost identical (only 0.1R diff).
The ENTIRE problem is signal detection divergence:
  v9 misses 498 real signals and generates 410 phantom signals.

=== ROOT CAUSE #1 (66.4%): Missing Signals — 498 CSV-only trades ===

  These 498 trades exist in Python but v9 never fires them.
  Breakdown: 348 trend + 150 mss, 387 long + 111 short.

  LIKELY CAUSES:
  a) FVG detection divergence (LantoNQStrategy_v9.cs:DetectNewFVG5m)
     Python detects FVGs on the SIGNAL bar's timeframe data which includes
     .shift(1).ffill() applied swing prices for sweep scoring. v9 uses a
     rolling candleBuffer and real-time FVG detection. Any small difference
     in FVG Top/Bottom/Size/SweptLiquidity can cause a signal to exist in
     one system but not the other.

  b) Swing detection timing (v9.cs:2475-2522 vs features/swing.py)
     Python: compute_swing_levels() uses rolling().min().shift(1) on the
     ENTIRE dataset then shift(1).ffill(). This means swing_high_price[i]
     is the most recent confirmed swing high AS OF bar i-1.
     v9: UpdateSwings() detects swings bar-by-bar at bar[right_bars ago].
     The swing is confirmed with right_bars=1 delay. If highSeries uses
     NT8's BarsAgo indexing and there's an off-by-one, swings appear at
     slightly different bars -> different IRL targets -> different TP1.
     This also means FVG.SweptLiquidity can differ.

  c) HTF bias computation (v9.cs:On1hBar/On4hBar vs Python HTF FVG system)
     If composite bias differs, the BIAS FILTER blocks signals in v9
     that Python passes (or vice versa). The v9 dbgFilteredBias counter
     should be checked.
     FIX: Log which filter blocked each rejected signal in v9. Add a
     filter-reason field to a rejected-signals CSV export.

  d) Signal Quality filter (v9.cs:1970-1976 ComputeSignalQuality)
     SQ thresholds SQLongThreshold=0.68, SQShortThreshold=0.82.
     If the FVG size, displacement, or fluency computations differ even
     slightly, signals near the threshold cross in one system but not other.

  e) FVG cooldown (FVGCooldown=6, v9.cs:1406)
     After an FVG fires a signal, it's on cooldown for 6 bars. If v9
     fires a phantom signal on a different FVG, that FVG enters cooldown
     and blocks the real signal later.

  FIX STRATEGY:
  -> LantoNQStrategy_v9.cs: Add a DEBUG mode that logs every signal
     candidate BEFORE filtering. Export: bar_time, direction, signal_type,
     fvg_top, fvg_bottom, entry, stop, tp1, filter_that_blocked.
  -> Compare this debug log against CSV baseline to find which filter
     kills each missing signal.

=== ROOT CAUSE #2 (33.5%): Phantom Signals — 410 v9-only trades ===

  These 410 trades are v9 fabrications with avg R = -0.09.
  Breakdown: 241 trend + 169 mss, 340 long + 70 short.

  LIKELY CAUSES:
  a) FVG pool pollution: v9 detects FVGs that Python doesn't, or keeps
     FVGs alive longer (invalidation logic differs).
     v9.cs DetectNewFVG5m() vs Python detect_fvg_signals().
     Check if fvg_min_size_atr is applied identically.

  b) IFVG creation divergence: When Python invalidates an FVG, it may
     create an IFVG under different conditions than v9.
     v9.cs UpdateFVGStates5m() handles IFVG creation on invalidation.
     Python fvg.py handles it. If IFVG direction or boundaries differ,
     MSS signals appear in v9 that don't exist in Python.

  c) Sweep tracking divergence (v9.cs sweptLowBuffer/sweptHighBuffer)
     MSS signals require a prior sweep. If v9's sweep tracking marks
     more sweeps than Python, more IFVGs pass the hadSweep check,
     generating extra MSS signals.

  FIX STRATEGY:
  -> Same debug CSV as above. For v9-only trades, find the exact signal
     bar and check: did Python also see this FVG? If yes, which filter
     blocked it? If no, why did v9 create this FVG?

=== ROOT CAUSE #3 (only for matched): Grade Mismatch — 267/647 trades ===

  267 matched trades have different grades (41%). This affects position
  sizing but since the trades are matched, the R impact is only +1.7R.

  v9.cs:1797 ComputeGrade() uses GetCompositeBias() which depends on:
    - HTF FVG bias (htfBias4h, htfBias1h)
    - Overnight bias (nyOpenPrice in overnight range)
    - ORM bias

  The grade transition matrix shows heavy B+ -> A+ shifts, meaning v9
  computes a more aligned bias than Python does.

  FIX: Compare HTF FVG detection on 1H/4H. The v9 On1hBar/On4hBar handlers
  use NT8 BarsInProgress data which may have different bar boundaries than
  Python's resampled 1H/4H data. Also check nyOpenPrice computation.

=== ROOT CAUSE #4 (matched): TP1 Distance Mismatch — 118 trades ===

  118 matched trades have TP1 distance diff > 1pt. The worst cases show
  CSV using IRL targets 100-355 pts away while v9 uses targets 2-18 pts.

  DIAGNOSIS: Python's swing_high_price[i] is shift(1).ffill() which gives
  the most recent CONFIRMED swing high price. v9 uses
  swingHighs[swingHighs.Count-1].Price in FindIRL (line 2617).

  CRITICAL BUG PATTERN: The top mismatches all show v9 with tiny tp1_dist
  (1.88, 2.50, 3.00, 5.50) vs CSV with huge dist (130, 134, 143, 363).
  This means v9's most recent swing high is VERY CLOSE to entry price,
  while Python's is far away. The v9 likely detected a micro-swing high
  that Python didn't, or Python's shift(1) delayed it.

  FIX: LantoNQStrategy_v9.cs line 2605-2633 (FindIRL):
  -> Add minimum IRL distance filter: if target < entry + MinIRLDist,
     skip to next swing or use 2R fallback.
  -> Or: use htfSwingHighs instead of regular swingHighs for IRL (the
     htf swings have left=10 right=3, more structural).

=== ROOT CAUSE #5 (matched): Exit Reason Transitions ===

  20 trades go from CSV stop -> v9 be_sweep (R impact = -21.9R).
  This means v9 incorrectly trims these 20 trades (reaches TP1 with wrong
  TP1 distance) and then gets stopped at BE, while CSV correctly stops out.
  This is a CONSEQUENCE of TP1 mismatch: v9 has tiny TP1 dist -> hits TP1
  easily -> trims -> gets swept at BE with small gain, vs CSV has large
  TP1 dist -> never hits TP1 -> stops out cleanly at -1R.

  FIX: Fix FindIRL (root cause #4) and these transitions resolve.

=== PRIORITY FIX ORDER ===

  1. Signal detection divergence (causes #1 and #2, 99.9% of gap)
     -> Add comprehensive signal debug logging to v9
     -> Compare FVG pools bar-by-bar against Python
     -> Compare filter chain results signal-by-signal
     -> Fix FVG detection, sweep tracking, bias computation

  2. FindIRL minimum distance (cause #4, affects be_sweep accounting)
     -> LantoNQStrategy_v9.cs line 2605-2633
     -> Add min distance = MinStopAtrMult * ATR or use htf swings

  3. Grade computation (cause #3, minor R impact)
     -> LantoNQStrategy_v9.cs line 1797-1809
     -> Compare HTF bar alignment between NT8 and Python resampled data
""")

# ============================================================
# BONUS: Monthly breakdown for worst year
# ============================================================
print("\n" + "=" * 80)
print("BONUS: MONTHLY R BREAKDOWN FOR YEARS WITH BIGGEST GAP")
print("=" * 80)

csv_df["month"] = csv_df["entry_time"].dt.month
v9_df["month"] = v9_df["entry_time"].dt.month

# Find top 3 years by gap
year_gaps = {}
for yr in all_years:
    csv_r_yr = csv_df[csv_df["year"] == yr]["r_multiple"].sum()
    v9_r_yr = v9_df[v9_df["year"] == yr]["r_multiple"].sum()
    year_gaps[yr] = csv_r_yr - v9_r_yr

worst_years = sorted(year_gaps.keys(), key=lambda y: abs(year_gaps[y]), reverse=True)[:3]

for yr in worst_years:
    print(f"\n--- {yr} (gap = {year_gaps[yr]:+.2f}R) ---")
    print(f"{'Month':>6s}  {'CSV_N':>6s}  {'v9_N':>6s}  {'CSV_R':>8s}  {'v9_R':>8s}  {'Gap':>8s}")
    for m in range(1, 13):
        csv_m = csv_df[(csv_df["year"] == yr) & (csv_df["month"] == m)]
        v9_m = v9_df[(v9_df["year"] == yr) & (v9_df["month"] == m)]
        if len(csv_m) == 0 and len(v9_m) == 0:
            continue
        csv_n = len(csv_m)
        v9_n = len(v9_m)
        csv_r = csv_m["r_multiple"].sum()
        v9_r = v9_m["r_multiple"].sum()
        print(f"{m:6d}  {csv_n:6d}  {v9_n:6d}  {csv_r:+8.2f}  {v9_r:+8.2f}  {csv_r-v9_r:+8.2f}")

# ============================================================
# BONUS: Trimmed status comparison
# ============================================================
print("\n" + "=" * 80)
print("BONUS: TRIMMED STATUS COMPARISON")
print("=" * 80)

if len(matched_df) > 0:
    trim_match = matched_df["csv_trimmed"] == matched_df["v9_trimmed"]
    print(f"Trimmed status matches: {trim_match.sum()} / {len(matched_df)}")
    trim_mismatches = matched_df[~trim_match]
    if len(trim_mismatches) > 0:
        print(f"Trimmed mismatches: {len(trim_mismatches)}")
        print(f"  CSV trimmed, v9 not: {((trim_mismatches['csv_trimmed'] == True) & (trim_mismatches['v9_trimmed'] == False)).sum()}")
        print(f"  CSV not trimmed, v9 trimmed: {((trim_mismatches['csv_trimmed'] == False) & (trim_mismatches['v9_trimmed'] == True)).sum()}")
        print(f"  R impact of trim mismatches: {trim_mismatches['r_diff'].sum():+.4f}")

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)
