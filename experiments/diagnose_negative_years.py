"""
experiments/diagnose_negative_years.py
Comprehensive diagnosis of negative-PnL years in 10yr backtest.

Steps:
1. Forward-sim using cached signals with current filter config
2. Yearly breakdown
3. Per-year diagnosis (session, direction, NQ trend, monthly)
4. Regime-based fix exploration (SMA200 filter, vol regime, seasonal)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path("/Users/mac/project/nq quant")
sys.path.insert(0, str(_PROJECT_ROOT))

# ──────────────────────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────────────────────
print("Loading data...")
df_5m = pd.read_parquet(_PROJECT_ROOT / "data" / "NQ_5m_10yr.parquet")
signals = pd.read_parquet(_PROJECT_ROOT / "data" / "cache_signals_10yr_v2.parquet")
bias = pd.read_parquet(_PROJECT_ROOT / "data" / "cache_bias_10yr_v2.parquet")

# Merge
sig_active = signals[signals["signal"]].copy()
sig_active = sig_active.join(bias[["bias_direction"]], how="left")
sig_active = sig_active.join(df_5m[["open", "high", "low", "close"]], how="left")

print(f"  Total signals: {len(sig_active)}")

# ──────────────────────────────────────────────────────────────
# 2. Compute alt_dir on the fly (alternating direction ratio)
# ──────────────────────────────────────────────────────────────
print("Computing alt_dir...")
from features.pa_quality import compute_alternating_dir_ratio
alt_dir = compute_alternating_dir_ratio(df_5m, window=6)
sig_active = sig_active.join(alt_dir.rename("alt_dir"), how="left")

# ──────────────────────────────────────────────────────────────
# 3. Session classification (ET)
# ──────────────────────────────────────────────────────────────
et_idx = sig_active.index.tz_convert("US/Eastern")
et_hour = et_idx.hour
et_min = et_idx.minute
et_frac = et_hour + et_min / 60.0

def classify_session(frac):
    if 9.5 <= frac < 16.0:
        return "NY"
    elif 3.0 <= frac < 9.5:
        return "London"
    else:
        return "Asia"

sig_active["session"] = [classify_session(f) for f in et_frac]
sig_active["et_frac"] = et_frac

# ──────────────────────────────────────────────────────────────
# 4. Apply current filters
# ──────────────────────────────────────────────────────────────
print("Applying filters...")

# Filter 1: Session-direction rules
mask_session_dir = (
    ((sig_active["session"] == "NY") & (sig_active["signal_dir"] == 1)) |  # NY: Long only
    ((sig_active["session"] == "London") & (sig_active["signal_dir"] == -1))  # London: Short only
)
# Asia: skip entirely

# Filter 2: Bias -- only block opposing (neutral passes)
bias_dir = sig_active["bias_direction"].fillna(0)
sig_dir = sig_active["signal_dir"]
bias_opposing = (sig_dir == -np.sign(bias_dir)) & (bias_dir != 0)
mask_bias = ~bias_opposing

# Filter 3: alt_dir < 0.334
mask_alt = sig_active["alt_dir"] < 0.334

# Filter 4: Stop on correct side
mask_stop = pd.Series(True, index=sig_active.index)
for idx in sig_active.index:
    row = sig_active.loc[idx]
    d = row["signal_dir"]
    ep = row["entry_price"]
    sp = row["model_stop"]
    tp = row["irl_target"]
    if pd.isna(ep) or pd.isna(sp) or pd.isna(tp):
        mask_stop.loc[idx] = False
        continue
    if d == 1:
        if sp >= ep or tp <= ep:
            mask_stop.loc[idx] = False
    elif d == -1:
        if sp <= ep or tp >= ep:
            mask_stop.loc[idx] = False

# Filter 5: Min stop distance
mask_min_stop = (sig_active["entry_price"] - sig_active["model_stop"]).abs() >= 10

combined_mask = mask_session_dir & mask_bias & mask_alt & mask_stop & mask_min_stop
filtered = sig_active[combined_mask].copy()
print(f"  After filters: {len(filtered)} signals (from {len(sig_active)})")

# ──────────────────────────────────────────────────────────────
# 5. Forward sim: flat $2/pt, 30-bar holding, one-at-a-time
# ──────────────────────────────────────────────────────────────
print("Running forward simulation...")

PV = 2.0  # point value (MNQ micro)
MAX_BARS = 30
COMMISSION_RT = 0.62 * 2  # round-trip per contract

# Pre-fetch 5m arrays for fast lookup
idx_5m = df_5m.index
open_5m = df_5m["open"].values
high_5m = df_5m["high"].values
low_5m = df_5m["low"].values
close_5m = df_5m["close"].values

# Build index lookup
idx_map = pd.Series(np.arange(len(idx_5m)), index=idx_5m)

trades = []
last_exit_bar = -1

for ts, row in filtered.iterrows():
    bar_i = idx_map.get(ts, None)
    if bar_i is None:
        continue
    bar_i = int(bar_i)

    # One at a time: skip if we haven't exited previous trade
    if bar_i <= last_exit_bar:
        continue

    direction = int(row["signal_dir"])
    entry_price = row["entry_price"]
    stop_price = row["model_stop"]
    tp_price = row["irl_target"]
    stop_dist = abs(entry_price - stop_price)

    if stop_dist < 1:
        continue

    # Simulate forward up to MAX_BARS
    exit_price = np.nan
    exit_bar = bar_i + MAX_BARS
    exit_reason = "timeout"

    for j in range(bar_i + 1, min(bar_i + 1 + MAX_BARS, len(idx_5m))):
        h = high_5m[j]
        l = low_5m[j]

        if direction == 1:  # LONG
            if l <= stop_price:
                exit_price = stop_price
                exit_bar = j
                exit_reason = "stop"
                break
            if h >= tp_price:
                exit_price = tp_price
                exit_bar = j
                exit_reason = "tp"
                break
        else:  # SHORT
            if h >= stop_price:
                exit_price = stop_price
                exit_bar = j
                exit_reason = "stop"
                break
            if l <= tp_price:
                exit_price = tp_price
                exit_bar = j
                exit_reason = "tp"
                break

    if np.isnan(exit_price):
        # Timeout: exit at close of last bar
        exit_j = min(bar_i + MAX_BARS, len(idx_5m) - 1)
        exit_price = close_5m[exit_j]
        exit_bar = exit_j
        exit_reason = "timeout"

    last_exit_bar = exit_bar

    if direction == 1:
        pnl_pts = exit_price - entry_price
    else:
        pnl_pts = entry_price - exit_price

    pnl_dollars = pnl_pts * PV - COMMISSION_RT
    r_mult = pnl_pts / stop_dist if stop_dist > 0 else 0

    trades.append({
        "entry_time": ts,
        "exit_time": idx_5m[exit_bar],
        "direction": direction,
        "session": row["session"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "tp_price": tp_price,
        "stop_dist": stop_dist,
        "pnl_pts": pnl_pts,
        "pnl_dollars": pnl_dollars,
        "r_mult": r_mult,
        "exit_reason": exit_reason,
        "bias_direction": row["bias_direction"],
        "alt_dir": row["alt_dir"],
        "signal_type": row["signal_type"],
    })

tdf = pd.DataFrame(trades)
tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
tdf["year"] = tdf["entry_time"].dt.year
tdf["month"] = tdf["entry_time"].dt.month

print(f"  Total trades: {len(tdf)}")

# ──────────────────────────────────────────────────────────────
# 6. Yearly breakdown
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("YEARLY BREAKDOWN (Current Config: NY-Long + London-Short + alt<0.334 + bias filter)")
print("=" * 80)

def calc_pf(subset):
    gp = subset.loc[subset["pnl_dollars"] > 0, "pnl_dollars"].sum()
    gl = abs(subset.loc[subset["pnl_dollars"] <= 0, "pnl_dollars"].sum())
    return gp / gl if gl > 0 else float("inf")

def calc_avg_win_loss(subset):
    wins = subset[subset["pnl_dollars"] > 0]["pnl_dollars"]
    losses = subset[subset["pnl_dollars"] <= 0]["pnl_dollars"]
    avg_w = wins.mean() if len(wins) > 0 else 0
    avg_l = losses.mean() if len(losses) > 0 else 0
    return avg_w, avg_l

yearly_summary = []
for year in sorted(tdf["year"].unique()):
    sub = tdf[tdf["year"] == year]
    n = len(sub)
    wr = (sub["pnl_dollars"] > 0).mean() * 100
    pnl = sub["pnl_dollars"].sum()
    pf = calc_pf(sub)
    avg_w, avg_l = calc_avg_win_loss(sub)

    ny_long = sub[(sub["session"] == "NY") & (sub["direction"] == 1)]
    ldn_short = sub[(sub["session"] == "London") & (sub["direction"] == -1)]

    status = "PROFIT" if pnl > 0 else "**LOSS**"
    yearly_summary.append({
        "year": year, "trades": n, "wr": wr, "pnl": pnl, "pf": pf,
        "avg_w": avg_w, "avg_l": avg_l, "status": status,
        "ny_trades": len(ny_long), "ny_pnl": ny_long["pnl_dollars"].sum(),
        "ldn_trades": len(ldn_short), "ldn_pnl": ldn_short["pnl_dollars"].sum(),
    })

    print(f"  {year}: {n:4d} trades | WR {wr:5.1f}% | PnL ${pnl:>10,.0f} | PF {pf:5.2f} | "
          f"AvgW ${avg_w:>7,.0f} AvgL ${avg_l:>7,.0f} | {status}")

neg_years = [s for s in yearly_summary if s["pnl"] <= 0]
pos_years = [s for s in yearly_summary if s["pnl"] > 0]
print(f"\n  SUMMARY: {len(pos_years)} profitable / {len(neg_years)} losing out of {len(yearly_summary)} years")

# ──────────────────────────────────────────────────────────────
# 7. NQ annual performance (first/last close)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("NQ ANNUAL PERFORMANCE")
print("=" * 80)

nq_annual = {}
for year in sorted(tdf["year"].unique()):
    year_data = df_5m[df_5m.index.year == year]
    if len(year_data) == 0:
        continue
    first_close = year_data["close"].iloc[0]
    last_close = year_data["close"].iloc[-1]
    ret = (last_close - first_close) / first_close * 100
    nq_annual[year] = {"first": first_close, "last": last_close, "ret_pct": ret}
    trend = "BULL" if ret > 5 else ("BEAR" if ret < -5 else "FLAT")
    print(f"  {year}: {first_close:>10,.0f} → {last_close:>10,.0f}  ({ret:+6.1f}%) [{trend}]")

# ──────────────────────────────────────────────────────────────
# 8. Diagnose each negative year
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("NEGATIVE YEAR DIAGNOSIS")
print("=" * 80)

for s in neg_years:
    year = s["year"]
    sub = tdf[tdf["year"] == year]
    nq_ret = nq_annual.get(year, {}).get("ret_pct", 0)
    nq_trend = "BULL" if nq_ret > 5 else ("BEAR" if nq_ret < -5 else "FLAT")

    print(f"\n{'─' * 60}")
    print(f"  YEAR {year} — NQ {nq_ret:+.1f}% ({nq_trend})")
    print(f"  Total: {s['trades']} trades, WR {s['wr']:.1f}%, PnL ${s['pnl']:,.0f}")
    print(f"{'─' * 60}")

    # Session breakdown
    print(f"\n  Session breakdown:")
    for sess_name in ["NY", "London"]:
        if sess_name == "NY":
            ss = sub[(sub["session"] == "NY") & (sub["direction"] == 1)]
            label = "NY-Long"
        else:
            ss = sub[(sub["session"] == "London") & (sub["direction"] == -1)]
            label = "London-Short"

        if len(ss) == 0:
            print(f"    {label}: 0 trades")
            continue

        wr = (ss["pnl_dollars"] > 0).mean() * 100
        pnl = ss["pnl_dollars"].sum()
        pf = calc_pf(ss)
        avg_w, avg_l = calc_avg_win_loss(ss)

        print(f"    {label}: {len(ss):3d} trades, WR {wr:5.1f}%, PnL ${pnl:>8,.0f}, PF {pf:.2f}, "
              f"AvgW ${avg_w:>6,.0f} / AvgL ${avg_l:>6,.0f}")

    # Exit reason breakdown
    print(f"\n  Exit reasons:")
    for reason in sorted(sub["exit_reason"].unique()):
        rs = sub[sub["exit_reason"] == reason]
        wr = (rs["pnl_dollars"] > 0).mean() * 100
        print(f"    {reason:10s}: {len(rs):3d} trades, WR {wr:5.1f}%, PnL ${rs['pnl_dollars'].sum():>8,.0f}")

    # Monthly breakdown
    print(f"\n  Monthly breakdown:")
    for m in range(1, 13):
        ms = sub[sub["month"] == m]
        if len(ms) == 0:
            continue
        pnl = ms["pnl_dollars"].sum()
        wr = (ms["pnl_dollars"] > 0).mean() * 100
        bar = "+" * max(0, int(pnl / 20)) if pnl > 0 else "-" * max(0, int(-pnl / 20))
        print(f"    {m:2d}: {len(ms):2d} trades, WR {wr:5.1f}%, PnL ${pnl:>7,.0f}  {bar}")

# ──────────────────────────────────────────────────────────────
# 9. Cross-year pattern analysis
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CROSS-YEAR PATTERN ANALYSIS")
print("=" * 80)

# NY-Long vs London-Short by year
print("\n  NY-Long PnL by year vs NQ return:")
for year in sorted(tdf["year"].unique()):
    sub_ny = tdf[(tdf["year"] == year) & (tdf["session"] == "NY") & (tdf["direction"] == 1)]
    sub_ldn = tdf[(tdf["year"] == year) & (tdf["session"] == "London") & (tdf["direction"] == -1)]
    nq_ret = nq_annual.get(year, {}).get("ret_pct", 0)

    ny_pnl = sub_ny["pnl_dollars"].sum() if len(sub_ny) > 0 else 0
    ldn_pnl = sub_ldn["pnl_dollars"].sum() if len(sub_ldn) > 0 else 0
    ny_wr = (sub_ny["pnl_dollars"] > 0).mean() * 100 if len(sub_ny) > 0 else 0
    ldn_wr = (sub_ldn["pnl_dollars"] > 0).mean() * 100 if len(sub_ldn) > 0 else 0

    print(f"    {year}: NQ {nq_ret:+6.1f}% | NY-Long: {len(sub_ny):3d} trades ${ny_pnl:>8,.0f} WR {ny_wr:5.1f}% | "
          f"Ldn-Short: {len(sub_ldn):3d} trades ${ldn_pnl:>8,.0f} WR {ldn_wr:5.1f}%")

# ──────────────────────────────────────────────────────────────
# 10. Regime Filter Tests
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FIX EXPLORATION")
print("=" * 80)

# 10a: SMA200 regime filter on daily data
print("\n--- Fix A: SMA200 Daily Regime Filter ---")
print("  Logic: compute SMA(200) on daily close. If close > SMA200 → bull regime → allow NY-Long.")
print("         If close < SMA200 → bear regime → block NY-Long (only London-Short).")

# Build daily close from 5m
daily = df_5m["close"].resample("1D").last().dropna()
sma200 = daily.rolling(200).mean()

# Map regime to each signal bar
# Use daily date alignment
regime_daily = pd.DataFrame({
    "close": daily,
    "sma200": sma200,
    "regime_bull": (daily > sma200).astype(int)
}, index=daily.index)

# Assign regime to each trade based on entry date
tdf["entry_date"] = tdf["entry_time"].dt.tz_localize(None).dt.normalize()
regime_daily.index = regime_daily.index.tz_localize(None)

# Join
tdf = tdf.merge(
    regime_daily[["regime_bull"]],
    left_on="entry_date",
    right_index=True,
    how="left"
)

# Test: block NY-Long when regime_bull == 0 (below SMA200)
tdf_fix_a = tdf.copy()
blocked = (tdf_fix_a["session"] == "NY") & (tdf_fix_a["direction"] == 1) & (tdf_fix_a["regime_bull"] == 0)
tdf_fix_a_filtered = tdf_fix_a[~blocked]

print(f"\n  Blocked {blocked.sum()} NY-Long trades in bear regime")
print(f"  Remaining trades: {len(tdf_fix_a_filtered)}")

print("\n  Fix A yearly breakdown:")
fix_a_neg = 0
for year in sorted(tdf_fix_a_filtered["year"].unique()):
    sub = tdf_fix_a_filtered[tdf_fix_a_filtered["year"] == year]
    n = len(sub)
    wr = (sub["pnl_dollars"] > 0).mean() * 100
    pnl = sub["pnl_dollars"].sum()
    pf = calc_pf(sub)
    status = "PROFIT" if pnl > 0 else "**LOSS**"
    if pnl <= 0:
        fix_a_neg += 1
    nq_ret = nq_annual.get(year, {}).get("ret_pct", 0)
    print(f"    {year}: {n:4d} trades | WR {wr:5.1f}% | PnL ${pnl:>10,.0f} | PF {pf:5.2f} | NQ {nq_ret:+.1f}% | {status}")

total_pnl_a = tdf_fix_a_filtered["pnl_dollars"].sum()
fix_a_pos = len(tdf_fix_a_filtered["year"].unique()) - fix_a_neg
print(f"\n  Fix A RESULT: {fix_a_pos}/{len(tdf_fix_a_filtered['year'].unique())} profitable years, Total PnL ${total_pnl_a:,.0f}")

# 10b: SMA200 + allow SHORT in NY during bear regime
print("\n--- Fix B: SMA200 Adaptive (Bull → NY-Long, Bear → NY-Short) ---")
print("  Logic: In bear regime, flip NY from Long-only to Short-only")

# Start from original filtered signals that passed all filters except session-direction
# We need to rebuild with adaptive rules
# Reload
sig_for_fix_b = sig_active[mask_bias & mask_alt & mask_stop & mask_min_stop].copy()

# Assign regime
sig_for_fix_b_et = sig_for_fix_b.index.tz_convert("US/Eastern")
sig_for_fix_b["et_frac"] = sig_for_fix_b_et.hour + sig_for_fix_b_et.minute / 60.0
sig_for_fix_b["session_fix"] = [classify_session(f) for f in sig_for_fix_b["et_frac"]]

# Add SMA200 regime
sig_for_fix_b["entry_date"] = sig_for_fix_b.index.tz_localize(None).normalize()
sig_for_fix_b = sig_for_fix_b.merge(
    regime_daily[["regime_bull"]],
    left_on="entry_date",
    right_index=True,
    how="left"
)

# Adaptive session-direction:
# NY + Bull → Long only
# NY + Bear → Short only
# London → Short only (always)
# Asia → skip
mask_fix_b = (
    # NY Bull: Long only
    ((sig_for_fix_b["session_fix"] == "NY") & (sig_for_fix_b["regime_bull"] == 1) & (sig_for_fix_b["signal_dir"] == 1)) |
    # NY Bear: Short only
    ((sig_for_fix_b["session_fix"] == "NY") & (sig_for_fix_b["regime_bull"] == 0) & (sig_for_fix_b["signal_dir"] == -1)) |
    # London: Short only
    ((sig_for_fix_b["session_fix"] == "London") & (sig_for_fix_b["signal_dir"] == -1))
)

fix_b_filtered = sig_for_fix_b[mask_fix_b].copy()

# Forward sim for Fix B
trades_b = []
last_exit_bar_b = -1

for ts, row in fix_b_filtered.iterrows():
    bar_i = idx_map.get(ts, None)
    if bar_i is None:
        continue
    bar_i = int(bar_i)

    if bar_i <= last_exit_bar_b:
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
                exit_price = stop_price
                exit_bar = j
                exit_reason = "stop"
                break
            if h >= tp_price:
                exit_price = tp_price
                exit_bar = j
                exit_reason = "tp"
                break
        else:
            if h >= stop_price:
                exit_price = stop_price
                exit_bar = j
                exit_reason = "stop"
                break
            if l <= tp_price:
                exit_price = tp_price
                exit_bar = j
                exit_reason = "tp"
                break

    if np.isnan(exit_price):
        exit_j = min(bar_i + MAX_BARS, len(idx_5m) - 1)
        exit_price = close_5m[exit_j]
        exit_bar = exit_j

    last_exit_bar_b = exit_bar

    if direction == 1:
        pnl_pts = exit_price - entry_price
    else:
        pnl_pts = entry_price - exit_price

    pnl_dollars = pnl_pts * PV - COMMISSION_RT

    trades_b.append({
        "entry_time": ts,
        "direction": direction,
        "session": row["session_fix"],
        "pnl_dollars": pnl_dollars,
        "pnl_pts": pnl_pts,
        "exit_reason": exit_reason,
        "regime_bull": row["regime_bull"],
    })

tdf_b = pd.DataFrame(trades_b)
tdf_b["year"] = pd.to_datetime(tdf_b["entry_time"]).dt.year

print(f"\n  Total trades Fix B: {len(tdf_b)}")
print("\n  Fix B yearly breakdown:")
fix_b_neg = 0
for year in sorted(tdf_b["year"].unique()):
    sub = tdf_b[tdf_b["year"] == year]
    n = len(sub)
    wr = (sub["pnl_dollars"] > 0).mean() * 100
    pnl = sub["pnl_dollars"].sum()
    pf = calc_pf(sub)
    status = "PROFIT" if pnl > 0 else "**LOSS**"
    if pnl <= 0:
        fix_b_neg += 1
    nq_ret = nq_annual.get(year, {}).get("ret_pct", 0)

    # Break down by NY-Bear-Short vs rest
    ny_bear_short = sub[(sub["session"] == "NY") & (sub["direction"] == -1)]
    ny_bull_long = sub[(sub["session"] == "NY") & (sub["direction"] == 1)]
    ldn_short = sub[sub["session"] == "London"]

    extra = ""
    if len(ny_bear_short) > 0:
        extra = f" [NY-Bear-Short: {len(ny_bear_short)} trades ${ny_bear_short['pnl_dollars'].sum():>6,.0f}]"

    print(f"    {year}: {n:4d} trades | WR {wr:5.1f}% | PnL ${pnl:>10,.0f} | PF {pf:5.2f} | NQ {nq_ret:+.1f}% | {status}{extra}")

total_pnl_b = tdf_b["pnl_dollars"].sum()
fix_b_pos = len(tdf_b["year"].unique()) - fix_b_neg
print(f"\n  Fix B RESULT: {fix_b_pos}/{len(tdf_b['year'].unique())} profitable years, Total PnL ${total_pnl_b:,.0f}")

# 10c: Just skip NY-Long entirely, London-Short only
print("\n--- Fix C: London-Short Only (drop NY-Long entirely) ---")
tdf_fix_c = tdf[(tdf["session"] == "London") & (tdf["direction"] == -1)]

print(f"\n  Total trades: {len(tdf_fix_c)}")
print("\n  Fix C yearly breakdown:")
fix_c_neg = 0
for year in sorted(tdf_fix_c["year"].unique()):
    sub = tdf_fix_c[tdf_fix_c["year"] == year]
    n = len(sub)
    wr = (sub["pnl_dollars"] > 0).mean() * 100
    pnl = sub["pnl_dollars"].sum()
    pf = calc_pf(sub)
    status = "PROFIT" if pnl > 0 else "**LOSS**"
    if pnl <= 0:
        fix_c_neg += 1
    nq_ret = nq_annual.get(year, {}).get("ret_pct", 0)
    print(f"    {year}: {n:4d} trades | WR {wr:5.1f}% | PnL ${pnl:>10,.0f} | PF {pf:5.2f} | NQ {nq_ret:+.1f}% | {status}")

total_pnl_c = tdf_fix_c["pnl_dollars"].sum()
fix_c_pos = len(tdf_fix_c["year"].unique()) - fix_c_neg
print(f"\n  Fix C RESULT: {fix_c_pos}/{len(tdf_fix_c['year'].unique())} profitable years, Total PnL ${total_pnl_c:,.0f}")

# 10d: ATR-scaled volatility regime
print("\n--- Fix D: ATR Volatility Regime (high vol → tighter filters) ---")
print("  Logic: When ATR(14) on daily is > 1.5x its 60-day SMA, market is high-vol.")
print("         In high-vol: skip NY-Long (only London-Short).")

daily_hl = df_5m[["high", "low"]].resample("1D").agg({"high": "max", "low": "min"})
daily_atr = (daily_hl["high"] - daily_hl["low"]).rolling(14).mean()
atr_sma60 = daily_atr.rolling(60).mean()
high_vol = (daily_atr > 1.5 * atr_sma60).astype(int)

vol_regime = pd.DataFrame({"high_vol": high_vol}, index=daily_atr.index)
vol_regime.index = vol_regime.index.tz_localize(None)

tdf_for_d = tdf.copy()
tdf_for_d = tdf_for_d.merge(
    vol_regime,
    left_on="entry_date",
    right_index=True,
    how="left"
)

blocked_d = (tdf_for_d["session"] == "NY") & (tdf_for_d["direction"] == 1) & (tdf_for_d["high_vol"] == 1)
tdf_fix_d = tdf_for_d[~blocked_d]

print(f"\n  Blocked {blocked_d.sum()} NY-Long trades in high-vol regime")
print(f"  Remaining trades: {len(tdf_fix_d)}")

print("\n  Fix D yearly breakdown:")
fix_d_neg = 0
for year in sorted(tdf_fix_d["year"].unique()):
    sub = tdf_fix_d[tdf_fix_d["year"] == year]
    n = len(sub)
    wr = (sub["pnl_dollars"] > 0).mean() * 100
    pnl = sub["pnl_dollars"].sum()
    pf = calc_pf(sub)
    status = "PROFIT" if pnl > 0 else "**LOSS**"
    if pnl <= 0:
        fix_d_neg += 1
    nq_ret = nq_annual.get(year, {}).get("ret_pct", 0)
    print(f"    {year}: {n:4d} trades | WR {wr:5.1f}% | PnL ${pnl:>10,.0f} | PF {pf:5.2f} | NQ {nq_ret:+.1f}% | {status}")

total_pnl_d = tdf_fix_d["pnl_dollars"].sum()
fix_d_pos = len(tdf_fix_d["year"].unique()) - fix_d_neg
print(f"\n  Fix D RESULT: {fix_d_pos}/{len(tdf_fix_d['year'].unique())} profitable years, Total PnL ${total_pnl_d:,.0f}")

# 10e: Combined SMA200 + Vol regime
print("\n--- Fix E: SMA200 + ATR Vol Combined ---")
print("  Logic: Block NY-Long if EITHER below SMA200 OR high vol")

blocked_e = (tdf_for_d["session"] == "NY") & (tdf_for_d["direction"] == 1) & (
    (tdf_for_d["regime_bull"] == 0) | (tdf_for_d["high_vol"] == 1)
)
tdf_fix_e = tdf_for_d[~blocked_e]

print(f"\n  Blocked {blocked_e.sum()} NY-Long trades")
print(f"  Remaining trades: {len(tdf_fix_e)}")

print("\n  Fix E yearly breakdown:")
fix_e_neg = 0
for year in sorted(tdf_fix_e["year"].unique()):
    sub = tdf_fix_e[tdf_fix_e["year"] == year]
    n = len(sub)
    wr = (sub["pnl_dollars"] > 0).mean() * 100
    pnl = sub["pnl_dollars"].sum()
    pf = calc_pf(sub)
    status = "PROFIT" if pnl > 0 else "**LOSS**"
    if pnl <= 0:
        fix_e_neg += 1
    nq_ret = nq_annual.get(year, {}).get("ret_pct", 0)
    print(f"    {year}: {n:4d} trades | WR {wr:5.1f}% | PnL ${pnl:>10,.0f} | PF {pf:5.2f} | NQ {nq_ret:+.1f}% | {status}")

total_pnl_e = tdf_fix_e["pnl_dollars"].sum()
fix_e_pos = len(tdf_fix_e["year"].unique()) - fix_e_neg
print(f"\n  Fix E RESULT: {fix_e_pos}/{len(tdf_fix_e['year'].unique())} profitable years, Total PnL ${total_pnl_e:,.0f}")

# ──────────────────────────────────────────────────────────────
# 11. FINAL SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

original_years = len(tdf["year"].unique())
original_neg = sum(1 for y in tdf["year"].unique() if tdf[tdf["year"] == y]["pnl_dollars"].sum() <= 0)
original_pnl = tdf["pnl_dollars"].sum()

results = [
    ("BASELINE (current)", original_years - original_neg, original_years, original_pnl, len(tdf)),
    ("Fix A: SMA200 block NY-Long in bear", fix_a_pos, len(tdf_fix_a_filtered["year"].unique()), total_pnl_a, len(tdf_fix_a_filtered)),
    ("Fix B: SMA200 adaptive (bear→NY-Short)", fix_b_pos, len(tdf_b["year"].unique()), total_pnl_b, len(tdf_b)),
    ("Fix C: London-Short only", fix_c_pos, len(tdf_fix_c["year"].unique()), total_pnl_c, len(tdf_fix_c)),
    ("Fix D: ATR vol block NY-Long", fix_d_pos, len(tdf_fix_d["year"].unique()), total_pnl_d, len(tdf_fix_d)),
    ("Fix E: SMA200 + ATR vol combined", fix_e_pos, len(tdf_fix_e["year"].unique()), total_pnl_e, len(tdf_fix_e)),
]

print(f"\n  {'Strategy':<45s} {'Prof/Total':>10s} {'Total PnL':>12s} {'Trades':>8s}")
print(f"  {'─'*45} {'─'*10} {'─'*12} {'─'*8}")
for name, pos, total, pnl, ntrades in results:
    print(f"  {name:<45s} {pos:>3d}/{total:<3d}     ${pnl:>10,.0f}  {ntrades:>6d}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
