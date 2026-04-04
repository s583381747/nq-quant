"""
experiments/diagnose_v2.py
Deep dive on losing years + more aggressive fix exploration.

Key findings from v1:
- 5 losing years: 2015(trivial), 2016, 2017, 2018, 2024
- 2016/2017: very few trades (29/20), losses small ($55/$136)
- 2018: FLAT year, NY-Long loses $268, London-Short breakeven
- 2024: BULL year but both sessions losing, London-Short -$463 is main culprit
- Fix B (SMA200 adaptive) flipped 2018 to profit but not 2024
- ATR vol filter blocked nothing (threshold too aggressive)

New explorations:
1. What if we lower ATR vol threshold?
2. Monthly seasonality filter
3. Signal type analysis (trend vs MSS)
4. R:R ratio filter (are low RR trades killing us?)
5. Composite fix
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path("/Users/mac/project/nq quant")
sys.path.insert(0, str(_PROJECT_ROOT))

print("Loading data...")
df_5m = pd.read_parquet(_PROJECT_ROOT / "data" / "NQ_5m_10yr.parquet")
signals = pd.read_parquet(_PROJECT_ROOT / "data" / "cache_signals_10yr_v2.parquet")
bias = pd.read_parquet(_PROJECT_ROOT / "data" / "cache_bias_10yr_v2.parquet")

sig_active = signals[signals["signal"]].copy()
sig_active = sig_active.join(bias[["bias_direction"]], how="left")
sig_active = sig_active.join(df_5m[["open", "high", "low", "close"]], how="left")

from features.pa_quality import compute_alternating_dir_ratio
alt_dir = compute_alternating_dir_ratio(df_5m, window=6)
sig_active = sig_active.join(alt_dir.rename("alt_dir"), how="left")

et_idx = sig_active.index.tz_convert("US/Eastern")
et_frac = et_idx.hour + et_idx.minute / 60.0

def classify_session(frac):
    if 9.5 <= frac < 16.0:
        return "NY"
    elif 3.0 <= frac < 9.5:
        return "London"
    else:
        return "Asia"

sig_active["session"] = [classify_session(f) for f in et_frac]

# Apply base filters (same as engine)
bias_dir = sig_active["bias_direction"].fillna(0)
sig_dir = sig_active["signal_dir"]
bias_opposing = (sig_dir == -np.sign(bias_dir)) & (bias_dir != 0)
mask_bias = ~bias_opposing
mask_alt = sig_active["alt_dir"] < 0.334

mask_stop_valid = pd.Series(True, index=sig_active.index)
for idx in sig_active.index:
    row = sig_active.loc[idx]
    d = row["signal_dir"]
    ep = row["entry_price"]
    sp = row["model_stop"]
    tp = row["irl_target"]
    if pd.isna(ep) or pd.isna(sp) or pd.isna(tp):
        mask_stop_valid.loc[idx] = False
        continue
    if d == 1:
        if sp >= ep or tp <= ep:
            mask_stop_valid.loc[idx] = False
    elif d == -1:
        if sp <= ep or tp >= ep:
            mask_stop_valid.loc[idx] = False

mask_min_stop = (sig_active["entry_price"] - sig_active["model_stop"]).abs() >= 10

# Session-direction
mask_session_dir = (
    ((sig_active["session"] == "NY") & (sig_active["signal_dir"] == 1)) |
    ((sig_active["session"] == "London") & (sig_active["signal_dir"] == -1))
)

base_mask = mask_bias & mask_alt & mask_stop_valid & mask_min_stop & mask_session_dir
filtered = sig_active[base_mask].copy()

# Compute RR
filtered["stop_dist"] = (filtered["entry_price"] - filtered["model_stop"]).abs()
filtered["tp_dist"] = (filtered["irl_target"] - filtered["entry_price"]).abs()
filtered["rr"] = filtered["tp_dist"] / filtered["stop_dist"].replace(0, np.nan)

# ──────────────────────────────────────────────────────────────
# Forward sim function
# ──────────────────────────────────────────────────────────────
PV = 2.0
MAX_BARS = 30
COMMISSION_RT = 0.62 * 2

idx_5m = df_5m.index
open_5m = df_5m["open"].values
high_5m = df_5m["high"].values
low_5m = df_5m["low"].values
close_5m = df_5m["close"].values
idx_map = pd.Series(np.arange(len(idx_5m)), index=idx_5m)


def forward_sim(sigs_df, label=""):
    trades = []
    last_exit_bar = -1
    for ts, row in sigs_df.iterrows():
        bar_i = idx_map.get(ts, None)
        if bar_i is None:
            continue
        bar_i = int(bar_i)
        if bar_i <= last_exit_bar:
            continue

        direction = int(row["signal_dir"])
        entry_price = row["entry_price"]
        stop_price = row["model_stop"]
        tp_price = row["irl_target"]
        stop_dist = abs(entry_price - stop_price)
        if stop_dist < 1:
            continue

        exit_price = np.nan
        exit_bar = bar_i + MAX_BARS
        exit_reason = "timeout"

        for j in range(bar_i + 1, min(bar_i + 1 + MAX_BARS, len(idx_5m))):
            h = high_5m[j]
            l = low_5m[j]
            if direction == 1:
                if l <= stop_price:
                    exit_price = stop_price; exit_bar = j; exit_reason = "stop"; break
                if h >= tp_price:
                    exit_price = tp_price; exit_bar = j; exit_reason = "tp"; break
            else:
                if h >= stop_price:
                    exit_price = stop_price; exit_bar = j; exit_reason = "stop"; break
                if l <= tp_price:
                    exit_price = tp_price; exit_bar = j; exit_reason = "tp"; break

        if np.isnan(exit_price):
            exit_j = min(bar_i + MAX_BARS, len(idx_5m) - 1)
            exit_price = close_5m[exit_j]
            exit_bar = exit_j

        last_exit_bar = exit_bar
        pnl_pts = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
        pnl_dollars = pnl_pts * PV - COMMISSION_RT

        trades.append({
            "entry_time": ts,
            "direction": direction,
            "session": row.get("session", ""),
            "signal_type": row.get("signal_type", ""),
            "pnl_dollars": pnl_dollars,
            "pnl_pts": pnl_pts,
            "exit_reason": exit_reason,
            "stop_dist": stop_dist,
            "rr": row.get("rr", np.nan),
        })

    tdf = pd.DataFrame(trades)
    if len(tdf) > 0:
        tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
        tdf["year"] = tdf["entry_time"].dt.year
        tdf["month"] = tdf["entry_time"].dt.month
    return tdf


def yearly_report(tdf, label=""):
    if len(tdf) == 0:
        print(f"  {label}: 0 trades")
        return 0, 0, 0
    total_pnl = tdf["pnl_dollars"].sum()
    neg = 0
    for year in sorted(tdf["year"].unique()):
        sub = tdf[tdf["year"] == year]
        n = len(sub)
        wr = (sub["pnl_dollars"] > 0).mean() * 100
        pnl = sub["pnl_dollars"].sum()
        gp = sub.loc[sub["pnl_dollars"] > 0, "pnl_dollars"].sum()
        gl = abs(sub.loc[sub["pnl_dollars"] <= 0, "pnl_dollars"].sum())
        pf = gp / gl if gl > 0 else float("inf")
        status = "PROFIT" if pnl > 0 else "**LOSS**"
        if pnl <= 0:
            neg += 1
        print(f"    {year}: {n:4d} trades | WR {wr:5.1f}% | PnL ${pnl:>10,.0f} | PF {pf:5.2f} | {status}")

    ny = len(tdf["year"].unique())
    pos = ny - neg
    print(f"  >> {label}: {pos}/{ny} profitable, Total PnL ${total_pnl:>,.0f}, {len(tdf)} trades")
    return pos, ny, total_pnl


# ──────────────────────────────────────────────────────────────
# Baseline
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("BASELINE")
print("=" * 80)
tdf_base = forward_sim(filtered, "baseline")
base_pos, base_ny, base_pnl = yearly_report(tdf_base, "BASELINE")

# ──────────────────────────────────────────────────────────────
# Analysis 1: Signal type breakdown by year (trend vs MSS)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SIGNAL TYPE ANALYSIS (trend vs MSS)")
print("=" * 80)

for stype in ["trend", "mss"]:
    sub = tdf_base[tdf_base["signal_type"] == stype]
    print(f"\n  {stype.upper()}:")
    for year in sorted(sub["year"].unique()):
        ys = sub[sub["year"] == year]
        n = len(ys)
        if n == 0:
            continue
        wr = (ys["pnl_dollars"] > 0).mean() * 100
        pnl = ys["pnl_dollars"].sum()
        print(f"    {year}: {n:3d} trades, WR {wr:5.1f}%, PnL ${pnl:>8,.0f}")
    total = sub["pnl_dollars"].sum()
    print(f"    TOTAL: {len(sub)} trades, PnL ${total:>,.0f}")

# ──────────────────────────────────────────────────────────────
# Analysis 2: RR distribution of winners vs losers
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("R:R DISTRIBUTION")
print("=" * 80)

for yr in [2016, 2017, 2018, 2024]:
    sub = tdf_base[tdf_base["year"] == yr]
    if len(sub) == 0:
        continue
    wins = sub[sub["pnl_dollars"] > 0]
    losses = sub[sub["pnl_dollars"] <= 0]
    print(f"\n  {yr}: Win RR median={wins['rr'].median():.2f}, "
          f"Loss RR median={losses['rr'].median():.2f}, "
          f"Win stop_dist median={wins['stop_dist'].median():.0f}, "
          f"Loss stop_dist median={losses['stop_dist'].median():.0f}")

# ──────────────────────────────────────────────────────────────
# Analysis 3: Stop distance distribution
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STOP DISTANCE ANALYSIS")
print("=" * 80)

for bucket_label, lo, hi in [("10-20pt", 10, 20), ("20-40pt", 20, 40), ("40-80pt", 40, 80), ("80+pt", 80, 500)]:
    sub = tdf_base[(tdf_base["stop_dist"] >= lo) & (tdf_base["stop_dist"] < hi)]
    if len(sub) == 0:
        continue
    wr = (sub["pnl_dollars"] > 0).mean() * 100
    pnl = sub["pnl_dollars"].sum()
    avg_pnl = sub["pnl_dollars"].mean()
    print(f"  {bucket_label}: {len(sub):3d} trades, WR {wr:5.1f}%, PnL ${pnl:>8,.0f}, Avg ${avg_pnl:>6,.1f}")

# ──────────────────────────────────────────────────────────────
# Fix exploration: More filters
# ──────────────────────────────────────────────────────────────

# SMA200 regime
daily = df_5m["close"].resample("1D").last().dropna()
sma200 = daily.rolling(200).mean()
sma50 = daily.rolling(50).mean()
regime_daily = pd.DataFrame({
    "close": daily,
    "sma200": sma200,
    "sma50": sma50,
    "bull_200": (daily > sma200).astype(int),
    "bull_50": (daily > sma50).astype(int),
    "sma_cross": ((sma50 > sma200).astype(int)),  # golden cross
}, index=daily.index)
regime_daily.index = regime_daily.index.tz_localize(None)

# ──────────────────────────────────────────────────────────────
# Fix F: Minimum RR filter
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FIX F: Minimum R:R Filter")
print("=" * 80)

for min_rr in [0.8, 1.0, 1.2, 1.5]:
    sigs_f = filtered[filtered["rr"] >= min_rr]
    tdf_f = forward_sim(sigs_f)
    if len(tdf_f) == 0:
        print(f"\n  RR >= {min_rr}: 0 trades")
        continue
    total_pnl = tdf_f["pnl_dollars"].sum()
    neg = sum(1 for y in tdf_f["year"].unique() if tdf_f[tdf_f["year"] == y]["pnl_dollars"].sum() <= 0)
    ny = len(tdf_f["year"].unique())
    pos = ny - neg
    print(f"  RR >= {min_rr}: {pos}/{ny} profitable, {len(tdf_f)} trades, Total PnL ${total_pnl:>,.0f}")

# ──────────────────────────────────────────────────────────────
# Fix G: Max stop distance filter
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FIX G: Max Stop Distance Filter")
print("=" * 80)

for max_stop in [30, 40, 50, 60, 80]:
    sigs_g = filtered[filtered["stop_dist"] <= max_stop]
    tdf_g = forward_sim(sigs_g)
    if len(tdf_g) == 0:
        continue
    total_pnl = tdf_g["pnl_dollars"].sum()
    neg = sum(1 for y in tdf_g["year"].unique() if tdf_g[tdf_g["year"] == y]["pnl_dollars"].sum() <= 0)
    ny = len(tdf_g["year"].unique())
    pos = ny - neg
    print(f"  Stop <= {max_stop}pt: {pos}/{ny} profitable, {len(tdf_g)} trades, Total PnL ${total_pnl:>,.0f}")

# ──────────────────────────────────────────────────────────────
# Fix H: SMA200 adaptive + RR filter
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FIX H: SMA200 Adaptive + RR >= 1.0")
print("=" * 80)

# Get all signals passing base filters except session-direction
all_base = sig_active[mask_bias & mask_alt & mask_stop_valid & mask_min_stop].copy()
all_base["stop_dist"] = (all_base["entry_price"] - all_base["model_stop"]).abs()
all_base["tp_dist"] = (all_base["irl_target"] - all_base["entry_price"]).abs()
all_base["rr"] = all_base["tp_dist"] / all_base["stop_dist"].replace(0, np.nan)

et_all = all_base.index.tz_convert("US/Eastern")
all_base["et_frac"] = et_all.hour + et_all.minute / 60.0
all_base["session"] = [classify_session(f) for f in all_base["et_frac"]]

# Add SMA200 regime
all_base["entry_date"] = all_base.index.tz_localize(None).normalize()
all_base = all_base.merge(
    regime_daily[["bull_200", "sma_cross"]],
    left_on="entry_date",
    right_index=True,
    how="left"
)

# Adaptive direction with RR filter
for min_rr in [0.8, 1.0]:
    mask_h = (
        ((all_base["session"] == "NY") & (all_base["bull_200"] == 1) & (all_base["signal_dir"] == 1)) |
        ((all_base["session"] == "NY") & (all_base["bull_200"] == 0) & (all_base["signal_dir"] == -1)) |
        ((all_base["session"] == "London") & (all_base["signal_dir"] == -1))
    ) & (all_base["rr"] >= min_rr)

    sigs_h = all_base[mask_h]
    tdf_h = forward_sim(sigs_h)
    if len(tdf_h) == 0:
        continue

    print(f"\n  SMA200 adaptive + RR >= {min_rr}:")
    yearly_report(tdf_h, f"Fix H (RR>={min_rr})")

# ──────────────────────────────────────────────────────────────
# Fix I: Golden cross (SMA50 > SMA200) for NY direction
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FIX I: Golden Cross (SMA50 > SMA200) for NY Direction")
print("=" * 80)

mask_i = (
    ((all_base["session"] == "NY") & (all_base["sma_cross"] == 1) & (all_base["signal_dir"] == 1)) |
    ((all_base["session"] == "NY") & (all_base["sma_cross"] == 0) & (all_base["signal_dir"] == -1)) |
    ((all_base["session"] == "London") & (all_base["signal_dir"] == -1))
)

sigs_i = all_base[mask_i]
tdf_i = forward_sim(sigs_i)
if len(tdf_i) > 0:
    yearly_report(tdf_i, "Fix I (Golden Cross)")

# ──────────────────────────────────────────────────────────────
# Fix J: Both sessions adaptive (SMA200)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FIX J: Both Sessions Adaptive (Bull → Long everywhere, Bear → Short everywhere)")
print("=" * 80)

# In bull: NY-Long + London-Long
# In bear: NY-Short + London-Short
mask_j = (
    ((all_base["bull_200"] == 1) & (all_base["signal_dir"] == 1) & (all_base["session"].isin(["NY", "London"]))) |
    ((all_base["bull_200"] == 0) & (all_base["signal_dir"] == -1) & (all_base["session"].isin(["NY", "London"])))
)

sigs_j = all_base[mask_j]
tdf_j = forward_sim(sigs_j)
if len(tdf_j) > 0:
    yearly_report(tdf_j, "Fix J (Full Adaptive)")

# ──────────────────────────────────────────────────────────────
# Fix K: London adaptive too (Bull → London-Long, Bear → London-Short)
# Keep NY-Long always, but London follows regime
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FIX K: NY-Long always + London adaptive (Bull→Long, Bear→Short)")
print("=" * 80)

mask_k = (
    ((all_base["session"] == "NY") & (all_base["signal_dir"] == 1)) |
    ((all_base["session"] == "London") & (all_base["bull_200"] == 1) & (all_base["signal_dir"] == 1)) |
    ((all_base["session"] == "London") & (all_base["bull_200"] == 0) & (all_base["signal_dir"] == -1))
)

sigs_k = all_base[mask_k]
tdf_k = forward_sim(sigs_k)
if len(tdf_k) > 0:
    yearly_report(tdf_k, "Fix K (NY-Long + London adaptive)")

# ──────────────────────────────────────────────────────────────
# Fix L: Drop session-direction entirely (both directions in both sessions)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FIX L: No session-direction filter (both dirs in NY+London)")
print("=" * 80)

mask_l = all_base["session"].isin(["NY", "London"])
sigs_l = all_base[mask_l]
tdf_l = forward_sim(sigs_l)
if len(tdf_l) > 0:
    yearly_report(tdf_l, "Fix L (No session-dir)")

# ──────────────────────────────────────────────────────────────
# Fix M: Current baseline + stricter alt_dir (0.25 instead of 0.334)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FIX M: Stricter alt_dir threshold")
print("=" * 80)

for alt_thresh in [0.15, 0.20, 0.25]:
    mask_m_alt = sig_active["alt_dir"] < alt_thresh
    mask_m = mask_bias & mask_m_alt & mask_stop_valid & mask_min_stop & mask_session_dir
    sigs_m = sig_active[mask_m].copy()
    sigs_m["stop_dist"] = (sigs_m["entry_price"] - sigs_m["model_stop"]).abs()
    sigs_m["tp_dist"] = (sigs_m["irl_target"] - sigs_m["entry_price"]).abs()
    sigs_m["rr"] = sigs_m["tp_dist"] / sigs_m["stop_dist"].replace(0, np.nan)

    tdf_m = forward_sim(sigs_m)
    if len(tdf_m) == 0:
        print(f"\n  alt_dir < {alt_thresh}: 0 trades")
        continue
    total_pnl = tdf_m["pnl_dollars"].sum()
    neg = sum(1 for y in tdf_m["year"].unique() if tdf_m[tdf_m["year"] == y]["pnl_dollars"].sum() <= 0)
    ny = len(tdf_m["year"].unique())
    pos = ny - neg
    print(f"  alt_dir < {alt_thresh}: {pos}/{ny} profitable, {len(tdf_m)} trades, Total PnL ${total_pnl:>,.0f}")

# ──────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
