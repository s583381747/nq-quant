"""
Diagnostic: Compare v7, v8.1, and Python reference trade CSVs.
Identify what went wrong between v7 (R=+12.4) and v8.1 (R=-35.4).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# ── Load data ──────────────────────────────────────────────────────
v7_path = r"C:\Users\58338\OneDrive\桌面\nt8_trades_20260402_001937.csv"
v81_path = r"C:\Users\58338\OneDrive\桌面\nt8_trades_20260402_012313.csv"
py_path = r"C:\projects\lanto quant\nq quant\ninjatrader\python_trades_545.csv"

v7 = pd.read_csv(v7_path, parse_dates=["entry_time", "exit_time"])
v81 = pd.read_csv(v81_path, parse_dates=["entry_time", "exit_time"])
py = pd.read_csv(py_path, parse_dates=["entry_time", "exit_time"])

# Normalize Python columns to match NT8 naming
py = py.rename(columns={"r": "r_multiple", "reason": "exit_reason", "dir": "direction", "type": "signal_type"})

# Python timestamps are UTC, NT8 timestamps are EST (UTC-5).
# Convert Python from UTC to EST for comparison.
if py["entry_time"].dt.tz is not None:
    py["entry_time"] = py["entry_time"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
    py["exit_time"] = py["exit_time"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
else:
    # Already naive — assume UTC, subtract 5 hours
    py["entry_time"] = py["entry_time"] - timedelta(hours=5)
    py["exit_time"] = py["exit_time"] - timedelta(hours=5)

# Add year column
for df, name in [(v7, "v7"), (v81, "v8.1"), (py, "python")]:
    df["year"] = df["entry_time"].dt.year

print("=" * 80)
print("TRADE CSV DIAGNOSTIC: v7 vs v8.1 vs Python")
print("=" * 80)

# ── 1. Basic counts ──────────────────────────────────────────────
print("\n" + "─" * 80)
print("1. BASIC COUNTS")
print("─" * 80)
print(f"  v7:     {len(v7):>5} trades, R = {v7['r_multiple'].sum():>+8.2f}")
print(f"  v8.1:   {len(v81):>5} trades, R = {v81['r_multiple'].sum():>+8.2f}")
print(f"  Python: {len(py):>5} trades, R = {py['r_multiple'].sum():>+8.2f}")
print(f"\n  Delta (v8.1 - v7): {len(v81) - len(v7):>+4} trades, R = {v81['r_multiple'].sum() - v7['r_multiple'].sum():>+8.2f}")


# ── Helper: match trades between two datasets ────────────────────
def match_trades(df_a, df_b, name_a, name_b, tolerance_minutes=6):
    """Match trades by entry_time within tolerance. Returns matched, only_a, only_b."""
    matched_a_idx = set()
    matched_b_idx = set()
    matches = []

    for i, row_a in df_a.iterrows():
        best_j = None
        best_dt = timedelta(minutes=tolerance_minutes + 1)
        for j, row_b in df_b.iterrows():
            if j in matched_b_idx:
                continue
            dt = abs(row_a["entry_time"] - row_b["entry_time"])
            if dt <= timedelta(minutes=tolerance_minutes) and dt < best_dt:
                # Also check direction matches
                if row_a["direction"] == row_b["direction"]:
                    best_j = j
                    best_dt = dt
        if best_j is not None:
            matched_a_idx.add(i)
            matched_b_idx.add(best_j)
            matches.append((i, best_j, best_dt))

    only_a = df_a.loc[~df_a.index.isin(matched_a_idx)].copy()
    only_b = df_b.loc[~df_b.index.isin(matched_b_idx)].copy()

    matched_a = df_a.loc[[m[0] for m in matches]].copy()
    matched_b = df_b.loc[[m[1] for m in matches]].copy()

    return matches, matched_a, matched_b, only_a, only_b


# ── 2. v7 vs v8.1 comparison ─────────────────────────────────────
print("\n" + "─" * 80)
print("2. v7 vs v8.1 TRADE MATCHING (tolerance: ±6 minutes)")
print("─" * 80)

matches_7_81, matched_v7, matched_v81, only_v7, only_v81 = match_trades(v7, v81, "v7", "v8.1")

print(f"  Common trades:        {len(matches_7_81):>5}")
print(f"  In v7 only (removed): {len(only_v7):>5}  (R = {only_v7['r_multiple'].sum():>+8.2f})")
print(f"  In v8.1 only (new):   {len(only_v81):>5}  (R = {only_v81['r_multiple'].sum():>+8.2f})")

# For common trades, compare R values
if len(matches_7_81) > 0:
    r_diff = matched_v81["r_multiple"].values - matched_v7["r_multiple"].values
    print(f"\n  Common trades R change:")
    print(f"    v7 common R sum:   {matched_v7['r_multiple'].sum():>+8.2f}")
    print(f"    v8.1 common R sum: {matched_v81['r_multiple'].sum():>+8.2f}")
    print(f"    R delta (common):  {r_diff.sum():>+8.2f}")
    print(f"    Mean R shift:      {r_diff.mean():>+8.4f}")
    print(f"    Trades with worse R: {(r_diff < -0.01).sum()}")
    print(f"    Trades with better R: {(r_diff > 0.01).sum()}")
    print(f"    Trades unchanged:    {((r_diff >= -0.01) & (r_diff <= 0.01)).sum()}")

print(f"\n  R DECOMPOSITION (v8.1 - v7 = {v81['r_multiple'].sum() - v7['r_multiple'].sum():>+.2f}):")
print(f"    From removed trades (v7 only):  {-only_v7['r_multiple'].sum():>+8.2f}  (removing positive = bad)")
print(f"    From new trades (v8.1 only):    {only_v81['r_multiple'].sum():>+8.2f}")
print(f"    From changed common trades:     {r_diff.sum():>+8.2f}")
print(f"    Total explained:                {only_v81['r_multiple'].sum() - only_v7['r_multiple'].sum() + r_diff.sum():>+8.2f}")


# ── 2b. Deep dive into new v8.1 trades ───────────────────────────
print("\n" + "─" * 80)
print("2b. NEW v8.1 TRADES (not in v7) — breakdown")
print("─" * 80)
if len(only_v81) > 0:
    print(f"\n  By signal_type:")
    for st, grp in only_v81.groupby("signal_type"):
        print(f"    {st:>6}: {len(grp):>4} trades, R = {grp['r_multiple'].sum():>+8.2f}, mean R = {grp['r_multiple'].mean():>+.4f}")

    print(f"\n  By grade:")
    for g, grp in only_v81.groupby("grade"):
        print(f"    {g:>3}: {len(grp):>4} trades, R = {grp['r_multiple'].sum():>+8.2f}, mean R = {grp['r_multiple'].mean():>+.4f}")

    print(f"\n  By exit_reason:")
    for er, grp in only_v81.groupby("exit_reason"):
        print(f"    {er:>15}: {len(grp):>4} trades, R = {grp['r_multiple'].sum():>+8.2f}")

    print(f"\n  By year:")
    for y, grp in only_v81.groupby("year"):
        print(f"    {y}: {len(grp):>4} trades, R = {grp['r_multiple'].sum():>+8.2f}")

    # Show worst new trades
    print(f"\n  10 WORST new v8.1 trades:")
    worst = only_v81.nsmallest(10, "r_multiple")
    for _, t in worst.iterrows():
        print(f"    {t['entry_time']}  {t['signal_type']:>5}  {t['grade']:>3}  R={t['r_multiple']:>+.4f}  exit={t['exit_reason']}  dir={t['direction']}")


# ── 2c. Removed trades (in v7 but not v8.1) ──────────────────────
print("\n" + "─" * 80)
print("2c. REMOVED TRADES (in v7 but NOT in v8.1)")
print("─" * 80)
if len(only_v7) > 0:
    print(f"\n  By signal_type:")
    for st, grp in only_v7.groupby("signal_type"):
        print(f"    {st:>6}: {len(grp):>4} trades, R = {grp['r_multiple'].sum():>+8.2f}")

    print(f"\n  By grade:")
    for g, grp in only_v7.groupby("grade"):
        print(f"    {g:>3}: {len(grp):>4} trades, R = {grp['r_multiple'].sum():>+8.2f}")

    print(f"\n  By year:")
    for y, grp in only_v7.groupby("year"):
        print(f"    {y}: {len(grp):>4} trades, R = {grp['r_multiple'].sum():>+8.2f}")

    # Show best removed trades (these were profitable, losing them is bad)
    print(f"\n  10 BEST removed v7 trades (losing these hurts):")
    best = only_v7.nlargest(10, "r_multiple")
    for _, t in best.iterrows():
        print(f"    {t['entry_time']}  {t['signal_type']:>5}  {t['grade']:>3}  R={t['r_multiple']:>+.4f}  exit={t['exit_reason']}  dir={t['direction']}")


# ── 2d. Common trades with changed outcomes ──────────────────────
print("\n" + "─" * 80)
print("2d. COMMON TRADES WITH CHANGED OUTCOMES")
print("─" * 80)
if len(matches_7_81) > 0:
    # Check exit reason changes
    exit_changed = 0
    grade_changed = 0
    exit_changes = {}
    for (i7, i81, dt) in matches_7_81:
        er7 = v7.loc[i7, "exit_reason"]
        er81 = v81.loc[i81, "exit_reason"]
        g7 = v7.loc[i7, "grade"]
        g81 = v81.loc[i81, "grade"]
        if er7 != er81:
            exit_changed += 1
            key = f"{er7} -> {er81}"
            if key not in exit_changes:
                exit_changes[key] = {"count": 0, "r_delta": 0}
            exit_changes[key]["count"] += 1
            exit_changes[key]["r_delta"] += v81.loc[i81, "r_multiple"] - v7.loc[i7, "r_multiple"]
        if g7 != g81:
            grade_changed += 1

    print(f"  Exit reason changed: {exit_changed} / {len(matches_7_81)} common trades")
    print(f"  Grade changed:       {grade_changed} / {len(matches_7_81)} common trades")

    if exit_changes:
        print(f"\n  Exit reason transition details:")
        for trans, info in sorted(exit_changes.items(), key=lambda x: x[1]["r_delta"]):
            print(f"    {trans:>35}: {info['count']:>4} trades, R delta = {info['r_delta']:>+8.2f}")


# ── 3. v8.1 vs Python ────────────────────────────────────────────
print("\n" + "=" * 80)
print("3. v8.1 vs PYTHON REFERENCE (tolerance: ±6 minutes)")
print("=" * 80)

matches_81_py, matched_v81_py, matched_py_81, only_v81_nopy, only_py = match_trades(v81, py, "v8.1", "python")

print(f"  Common trades:           {len(matches_81_py):>5}")
print(f"  In v8.1 only (extra):    {len(only_v81_nopy):>5}  (R = {only_v81_nopy['r_multiple'].sum():>+8.2f})")
print(f"  In Python only (missed): {len(only_py):>5}  (R = {only_py['r_multiple'].sum():>+8.2f})")

if len(only_py) > 0:
    print(f"\n  MISSED Python trades by signal_type:")
    for st, grp in only_py.groupby("signal_type"):
        print(f"    {st:>6}: {len(grp):>4} trades, R = {grp['r_multiple'].sum():>+8.2f}")

    print(f"\n  MISSED Python trades by year:")
    for y, grp in only_py.groupby("year"):
        print(f"    {y}: {len(grp):>4} trades, R = {grp['r_multiple'].sum():>+8.2f}")

    print(f"\n  MISSED Python trades by grade:")
    for g, grp in only_py.groupby("grade"):
        print(f"    {g:>3}: {len(grp):>4} trades, R = {grp['r_multiple'].sum():>+8.2f}")

if len(only_v81_nopy) > 0:
    print(f"\n  EXTRA v8.1 trades (not in Python) by signal_type:")
    for st, grp in only_v81_nopy.groupby("signal_type"):
        print(f"    {st:>6}: {len(grp):>4} trades, R = {grp['r_multiple'].sum():>+8.2f}")

    print(f"\n  EXTRA v8.1 trades by year:")
    for y, grp in only_v81_nopy.groupby("year"):
        print(f"    {y}: {len(grp):>4} trades, R = {grp['r_multiple'].sum():>+8.2f}")


# ── 4. Signal type breakdown ─────────────────────────────────────
print("\n" + "─" * 80)
print("4. SIGNAL TYPE BREAKDOWN")
print("─" * 80)
print(f"\n  {'Dataset':<10} {'Type':<8} {'Count':>6} {'R Sum':>10} {'Mean R':>10} {'Win%':>8}")
print(f"  {'-'*10} {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")

for name, df in [("v7", v7), ("v8.1", v81), ("Python", py)]:
    for st in sorted(df["signal_type"].unique()):
        grp = df[df["signal_type"] == st]
        wins = (grp["r_multiple"] > 0).sum()
        winpct = wins / len(grp) * 100 if len(grp) > 0 else 0
        print(f"  {name:<10} {st:<8} {len(grp):>6} {grp['r_multiple'].sum():>+10.2f} {grp['r_multiple'].mean():>+10.4f} {winpct:>7.1f}%")


# ── 5. Grade breakdown ───────────────────────────────────────────
print("\n" + "─" * 80)
print("5. GRADE BREAKDOWN")
print("─" * 80)
print(f"\n  {'Dataset':<10} {'Grade':<6} {'Count':>6} {'%':>7} {'R Sum':>10} {'Mean R':>10} {'Win%':>8}")
print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*7} {'-'*10} {'-'*10} {'-'*8}")

for name, df in [("v7", v7), ("v8.1", v81), ("Python", py)]:
    for g in ["A+", "B+", "C"]:
        grp = df[df["grade"] == g]
        if len(grp) == 0:
            continue
        pct = len(grp) / len(df) * 100
        wins = (grp["r_multiple"] > 0).sum()
        winpct = wins / len(grp) * 100
        print(f"  {name:<10} {g:<6} {len(grp):>6} {pct:>6.1f}% {grp['r_multiple'].sum():>+10.2f} {grp['r_multiple'].mean():>+10.4f} {winpct:>7.1f}%")


# ── 6. Exit reason breakdown ─────────────────────────────────────
print("\n" + "─" * 80)
print("6. EXIT REASON BREAKDOWN")
print("─" * 80)
print(f"\n  {'Dataset':<10} {'Reason':<18} {'Count':>6} {'%':>7} {'R Sum':>10} {'Mean R':>10}")
print(f"  {'-'*10} {'-'*18} {'-'*6} {'-'*7} {'-'*10} {'-'*10}")

all_reasons = set()
for df in [v7, v81, py]:
    all_reasons.update(df["exit_reason"].unique())

for name, df in [("v7", v7), ("v8.1", v81), ("Python", py)]:
    for er in sorted(all_reasons):
        grp = df[df["exit_reason"] == er]
        if len(grp) == 0:
            continue
        pct = len(grp) / len(df) * 100
        print(f"  {name:<10} {er:<18} {len(grp):>6} {pct:>6.1f}% {grp['r_multiple'].sum():>+10.2f} {grp['r_multiple'].mean():>+10.4f}")


# ── 7. Year-by-year comparison ───────────────────────────────────
print("\n" + "─" * 80)
print("7. YEAR-BY-YEAR COMPARISON")
print("─" * 80)

all_years = sorted(set(v7["year"].unique()) | set(v81["year"].unique()) | set(py["year"].unique()))

print(f"\n  {'Year':<6} {'v7 N':>6} {'v7 R':>8} {'v8.1 N':>7} {'v8.1 R':>8} {'Delta R':>8} {'Py N':>6} {'Py R':>8}")
print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")

for y in all_years:
    v7y = v7[v7["year"] == y]
    v81y = v81[v81["year"] == y]
    pyy = py[py["year"] == y]
    r7 = v7y["r_multiple"].sum() if len(v7y) > 0 else 0
    r81 = v81y["r_multiple"].sum() if len(v81y) > 0 else 0
    rpy = pyy["r_multiple"].sum() if len(pyy) > 0 else 0
    delta = r81 - r7
    marker = " <<<" if abs(delta) > 5 else (" <<" if abs(delta) > 3 else "")
    print(f"  {y:<6} {len(v7y):>6} {r7:>+8.2f} {len(v81y):>7} {r81:>+8.2f} {delta:>+8.2f} {len(pyy):>6} {rpy:>+8.2f}{marker}")


# ── 8. Worst year deep-dive ──────────────────────────────────────
print("\n" + "─" * 80)
print("8. DEEP DIVE: WORST YEAR(S) FOR v8.1 vs v7")
print("─" * 80)

year_deltas = {}
for y in all_years:
    v7y = v7[v7["year"] == y]
    v81y = v81[v81["year"] == y]
    year_deltas[y] = v81y["r_multiple"].sum() - v7y["r_multiple"].sum()

worst_years = sorted(year_deltas.items(), key=lambda x: x[1])[:3]

for wy, wd in worst_years:
    print(f"\n  === {wy} (R delta = {wd:>+.2f}) ===")
    v7y = v7[v7["year"] == wy]
    v81y = v81[v81["year"] == wy]

    print(f"  v7:   {len(v7y)} trades, R = {v7y['r_multiple'].sum():>+.2f}")
    print(f"  v8.1: {len(v81y)} trades, R = {v81y['r_multiple'].sum():>+.2f}")

    # Grade distribution comparison for this year
    print(f"\n  Grade distribution:")
    for g in ["A+", "B+", "C"]:
        n7 = len(v7y[v7y["grade"] == g])
        n81 = len(v81y[v81y["grade"] == g])
        r7g = v7y[v7y["grade"] == g]["r_multiple"].sum()
        r81g = v81y[v81y["grade"] == g]["r_multiple"].sum()
        print(f"    {g:>3}: v7={n7:>3} (R={r7g:>+.2f})  v8.1={n81:>3} (R={r81g:>+.2f})  delta={r81g-r7g:>+.2f}")

    # Signal type for this year
    print(f"\n  Signal type distribution:")
    for st in ["trend", "mss"]:
        n7 = len(v7y[v7y["signal_type"] == st])
        n81 = len(v81y[v81y["signal_type"] == st])
        r7s = v7y[v7y["signal_type"] == st]["r_multiple"].sum()
        r81s = v81y[v81y["signal_type"] == st]["r_multiple"].sum()
        print(f"    {st:>6}: v7={n7:>3} (R={r7s:>+.2f})  v8.1={n81:>3} (R={r81s:>+.2f})  delta={r81s-r7s:>+.2f}")

    # New trades in this year
    new_in_year = only_v81[only_v81["year"] == wy]
    removed_in_year = only_v7[only_v7["year"] == wy]
    print(f"\n  New v8.1 trades this year:     {len(new_in_year):>3} (R = {new_in_year['r_multiple'].sum():>+.2f})")
    print(f"  Removed v7 trades this year:   {len(removed_in_year):>3} (R = {removed_in_year['r_multiple'].sum():>+.2f})")


# ── 9. Win rate & loss magnitude comparison ──────────────────────
print("\n" + "─" * 80)
print("9. WIN RATE & LOSS MAGNITUDE")
print("─" * 80)

for name, df in [("v7", v7), ("v8.1", v81), ("Python", py)]:
    wins = df[df["r_multiple"] > 0]
    losses = df[df["r_multiple"] <= 0]
    print(f"\n  {name}:")
    print(f"    Trades: {len(df)}, Wins: {len(wins)} ({len(wins)/len(df)*100:.1f}%), Losses: {len(losses)} ({len(losses)/len(df)*100:.1f}%)")
    print(f"    Avg win:  {wins['r_multiple'].mean():>+.4f}")
    print(f"    Avg loss: {losses['r_multiple'].mean():>+.4f}")
    print(f"    Win R sum:  {wins['r_multiple'].sum():>+.2f}")
    print(f"    Loss R sum: {losses['r_multiple'].sum():>+.2f}")
    # Full stop-outs (R ~ -1.0)
    full_stops = df[df["r_multiple"] <= -0.9]
    print(f"    Full stop-outs (R <= -0.9): {len(full_stops)} trades, R = {full_stops['r_multiple'].sum():>+.2f}")


# ── 10. Trimmed trade analysis ───────────────────────────────────
print("\n" + "─" * 80)
print("10. TRIMMED TRADE ANALYSIS")
print("─" * 80)

for name, df in [("v7", v7), ("v8.1", v81), ("Python", py)]:
    trimmed = df[df["trimmed"] == True]
    untrimmed = df[df["trimmed"] == False]
    print(f"\n  {name}:")
    print(f"    Trimmed:   {len(trimmed):>4} ({len(trimmed)/len(df)*100:.1f}%), R = {trimmed['r_multiple'].sum():>+.2f}")
    print(f"    Untrimmed: {len(untrimmed):>4} ({len(untrimmed)/len(df)*100:.1f}%), R = {untrimmed['r_multiple'].sum():>+.2f}")


# ── 11. Summary diagnosis ────────────────────────────────────────
print("\n" + "=" * 80)
print("11. DIAGNOSIS SUMMARY")
print("=" * 80)

total_delta = v81["r_multiple"].sum() - v7["r_multiple"].sum()
print(f"\n  Total R regression: {total_delta:>+.2f}")
print(f"\n  Breakdown:")
print(f"    1. New trades added in v8.1:       {len(only_v81):>4} trades contributing R = {only_v81['r_multiple'].sum():>+.2f}")
print(f"    2. Trades removed from v7:         {len(only_v7):>4} trades that had R = {only_v7['r_multiple'].sum():>+.2f} (losing these costs {-only_v7['r_multiple'].sum():>+.2f})")
print(f"    3. Common trades with changed R:   {len(matches_7_81):>4} trades with R shift = {r_diff.sum():>+.2f}")

# Identify main culprit
components = {
    "New trades (net negative)": only_v81["r_multiple"].sum(),
    "Lost good trades (was positive)": -only_v7["r_multiple"].sum(),
    "Common trades R shift": r_diff.sum(),
}
print(f"\n  Culprit ranking (most negative first):")
for desc, val in sorted(components.items(), key=lambda x: x[1]):
    pct = val / total_delta * 100 if total_delta != 0 else 0
    print(f"    {desc:>40}: {val:>+8.2f}  ({pct:>5.1f}% of regression)")

# Check if C-grade explosion is a factor
c_v7 = v7[v7["grade"] == "C"]
c_v81 = v81[v81["grade"] == "C"]
print(f"\n  C-grade trade explosion?")
print(f"    v7:   {len(c_v7):>4} C-grades, R = {c_v7['r_multiple'].sum():>+.2f}")
print(f"    v8.1: {len(c_v81):>4} C-grades, R = {c_v81['r_multiple'].sum():>+.2f}")
print(f"    Delta: {len(c_v81) - len(c_v7):>+4} trades, R = {c_v81['r_multiple'].sum() - c_v7['r_multiple'].sum():>+.2f}")

# Check if stop rate increased
stop_v7 = v7[v7["exit_reason"] == "stop"]
stop_v81 = v81[v81["exit_reason"] == "stop"]
print(f"\n  Stop-loss rate:")
print(f"    v7:   {len(stop_v7):>4} ({len(stop_v7)/len(v7)*100:.1f}%), R = {stop_v7['r_multiple'].sum():>+.2f}")
print(f"    v8.1: {len(stop_v81):>4} ({len(stop_v81)/len(v81)*100:.1f}%), R = {stop_v81['r_multiple'].sum():>+.2f}")

print("\n" + "=" * 80)
print("END OF DIAGNOSTIC")
print("=" * 80)
