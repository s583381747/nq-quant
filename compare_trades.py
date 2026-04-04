"""Compare NT8 CSV trades vs Python reference trades."""
import pandas as pd
import numpy as np

# Load both files
nt8 = pd.read_csv(r"C:/Users/58338/OneDrive/桌面/nt8_csv_trades_20260402_015105.csv")
py = pd.read_csv(r"C:/projects/lanto quant/nq quant/ninjatrader/python_trades_545.csv")

# Normalize column names: Python uses 'r','reason','dir','type' vs NT8 'r_multiple','exit_reason','direction','signal_type'
py = py.rename(columns={"r": "r_multiple", "reason": "exit_reason", "dir": "direction", "type": "signal_type"})

# Parse entry_time
# NT8 times appear to be EST (no tz info)
nt8["entry_time"] = pd.to_datetime(nt8["entry_time"])
# Python times are UTC — convert to EST (drop tz for matching)
py["entry_time"] = pd.to_datetime(py["entry_time"], utc=True).dt.tz_convert("US/Eastern").dt.tz_localize(None)

# Also parse exit_time for completeness
nt8["exit_time"] = pd.to_datetime(nt8["exit_time"])
py["exit_time"] = pd.to_datetime(py["exit_time"], utc=True).dt.tz_convert("US/Eastern").dt.tz_localize(None)

print(f"NT8 trades: {len(nt8)}")
print(f"Python trades: {len(py)}")
print()

# --- 1. Match trades by entry_time (±6 min) and same direction ---
TOLERANCE = pd.Timedelta(minutes=6)

nt8["matched_py_idx"] = -1
py["matched_nt8_idx"] = -1

# Sort both by entry_time
nt8_sorted = nt8.sort_values("entry_time").reset_index(drop=True)
py_sorted = py.sort_values("entry_time").reset_index(drop=True)

# Greedy matching: for each NT8 trade, find closest Python trade within tolerance & same direction
py_used = set()
matches = []  # list of (nt8_idx, py_idx)

for ni, nrow in nt8_sorted.iterrows():
    best_pi = None
    best_diff = pd.Timedelta(days=999)
    for pi, prow in py_sorted.iterrows():
        if pi in py_used:
            continue
        if nrow["direction"] != prow["direction"]:
            continue
        diff = abs(nrow["entry_time"] - prow["entry_time"])
        if diff <= TOLERANCE and diff < best_diff:
            best_diff = diff
            best_pi = pi
    if best_pi is not None:
        matches.append((ni, best_pi))
        py_used.add(best_pi)

matched_nt8_idxs = {m[0] for m in matches}
matched_py_idxs = {m[1] for m in matches}

nt8_only_idxs = [i for i in range(len(nt8_sorted)) if i not in matched_nt8_idxs]
py_only_idxs = [i for i in range(len(py_sorted)) if i not in matched_py_idxs]

print(f"=== MATCHING (±6 min, same direction) ===")
print(f"Matched:     {len(matches)}")
print(f"NT8-only:    {len(nt8_only_idxs)}")
print(f"Python-only: {len(py_only_idxs)}")
print()

# --- 2. Total R ---
nt8_total_r = nt8["r_multiple"].sum()
py_total_r = py["r_multiple"].sum()
print(f"=== TOTAL R ===")
print(f"NT8:    {nt8_total_r:.4f}")
print(f"Python: {py_total_r:.4f}")
print(f"Diff:   {nt8_total_r - py_total_r:.4f}")
print()

# --- 3. Matched trades: R diff and exit_reason comparison ---
r_diffs = []
nt8_reasons_matched = []
py_reasons_matched = []
for ni, pi in matches:
    r_diff = nt8_sorted.loc[ni, "r_multiple"] - py_sorted.loc[pi, "r_multiple"]
    r_diffs.append(r_diff)
    nt8_reasons_matched.append(nt8_sorted.loc[ni, "exit_reason"])
    py_reasons_matched.append(py_sorted.loc[pi, "exit_reason"])

r_diffs = np.array(r_diffs)
print(f"=== MATCHED TRADE R COMPARISON ===")
print(f"Sum of R diffs (NT8 - Python): {r_diffs.sum():.4f}")
print(f"Mean abs R diff:               {np.abs(r_diffs).mean():.4f}")
print(f"Max abs R diff:                {np.abs(r_diffs).max():.4f}")
print(f"Trades with R diff > 0.05:     {(np.abs(r_diffs) > 0.05).sum()}")
print()

# Exit reason agreement
reason_agree = sum(1 for a, b in zip(nt8_reasons_matched, py_reasons_matched) if a == b)
print(f"Exit reason agreement: {reason_agree}/{len(matches)} ({100*reason_agree/len(matches):.1f}%)")
reason_disagree = [(ni, pi) for (ni, pi), a, b in zip(matches, nt8_reasons_matched, py_reasons_matched) if a != b]
if reason_disagree:
    print(f"\nDisagreed exit reasons (first 20):")
    print(f"  {'Entry Time':<22s} {'NT8 reason':<14s} {'PY reason':<14s} {'NT8 R':>8s} {'PY R':>8s}")
    for ni, pi in reason_disagree[:20]:
        print(f"  {str(nt8_sorted.loc[ni,'entry_time']):<22s} {nt8_sorted.loc[ni,'exit_reason']:<14s} {py_sorted.loc[pi,'exit_reason']:<14s} {nt8_sorted.loc[ni,'r_multiple']:>8.4f} {py_sorted.loc[pi,'r_multiple']:>8.4f}")
print()

# --- 4. R by exit_reason for both ---
print(f"=== R BY EXIT REASON ===")
nt8_by_reason = nt8.groupby("exit_reason")["r_multiple"].agg(["sum", "count"]).sort_values("sum", ascending=False)
py_by_reason = py.groupby("exit_reason")["r_multiple"].agg(["sum", "count"]).sort_values("sum", ascending=False)

print("NT8:")
for reason, row in nt8_by_reason.iterrows():
    print(f"  {reason:<14s}  count={int(row['count']):>4d}  R={row['sum']:>+9.4f}")
print(f"  {'TOTAL':<14s}  count={int(nt8_by_reason['count'].sum()):>4d}  R={nt8_by_reason['sum'].sum():>+9.4f}")

print("Python:")
for reason, row in py_by_reason.iterrows():
    print(f"  {reason:<14s}  count={int(row['count']):>4d}  R={row['sum']:>+9.4f}")
print(f"  {'TOTAL':<14s}  count={int(py_by_reason['count'].sum()):>4d}  R={py_by_reason['sum'].sum():>+9.4f}")
print()

# --- 5. Python-only trades (missing from NT8) ---
py_only = py_sorted.loc[py_only_idxs].copy()
print(f"=== PYTHON-ONLY TRADES ({len(py_only)}) ===")
print(f"  {'entry_time':<22s} {'dir':>4s} {'signal_type':<8s} {'entry_price':>12s} {'r_multiple':>10s} {'grade':<6s} {'exit_reason':<14s}")
for _, row in py_only.iterrows():
    d = "Long" if row["direction"] == 1 else "Short"
    print(f"  {str(row['entry_time']):<22s} {d:>4s} {row['signal_type']:<8s} {row['entry_price']:>12.2f} {row['r_multiple']:>+10.4f} {row['grade']:<6s} {row['exit_reason']:<14s}")
print(f"  Total R of missing trades: {py_only['r_multiple'].sum():.4f}")
print()

# --- 6. NT8-only trades (in NT8 but not Python) ---
nt8_only = nt8_sorted.loc[nt8_only_idxs].copy()
print(f"=== NT8-ONLY TRADES ({len(nt8_only)}) ===")
print(f"  {'entry_time':<22s} {'dir':>4s} {'signal_type':<8s} {'entry_price':>12s} {'r_multiple':>10s} {'grade':<6s} {'exit_reason':<14s}")
for _, row in nt8_only.iterrows():
    d = "Long" if row["direction"] == 1 else "Short"
    print(f"  {str(row['entry_time']):<22s} {d:>4s} {row['signal_type']:<8s} {row['entry_price']:>12.2f} {row['r_multiple']:>+10.4f} {row['grade']:<6s} {row['exit_reason']:<14s}")
print(f"  Total R of NT8-only trades: {nt8_only['r_multiple'].sum():.4f}")
