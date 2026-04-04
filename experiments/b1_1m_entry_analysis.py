"""
B1 — 1-Minute Entry Analysis: Exploration of tighter stops via 1m FVG signals.

Goal: Understand the potential of 1m entries vs 5m entries.
  - Part 1: Understand existing 1m pipeline
  - Part 2: 1m vs 5m stop distance comparison
  - Part 3: Signal count comparison
  - Part 4: Theoretical R improvement
  - Part 5: Quick 1m signal quality check (TP1 hit rate)
  - Part 6: Feasibility assessment

Uses 2023 data only (1 year) for manageable runtime.
"""

import sys; sys.path.insert(0, '.')
import logging; logging.basicConfig(level=logging.WARNING)
import pandas as pd, numpy as np, yaml, time
from collections import defaultdict

params = yaml.safe_load(open('config/params.yaml', encoding='utf-8'))

SEPARATOR = '=' * 70
THIN_SEP = '-' * 70


# ============================================================================
# DATA LOADING
# ============================================================================
print(f"\n{SEPARATOR}")
print("B1 — 1-MINUTE ENTRY ANALYSIS")
print(SEPARATOR)

print("\n[Loading data...]")
t0 = time.time()
df_1m_full = pd.read_parquet('data/NQ_1min_10yr.parquet')
df_5m_full = pd.read_parquet('data/NQ_5m_10yr.parquet')
print(f"  1m full: {len(df_1m_full):,} bars")
print(f"  5m full: {len(df_5m_full):,} bars")
print(f"  Load time: {time.time()-t0:.1f}s")

# Filter to 2023
et_1m = df_1m_full.index.tz_convert('US/Eastern')
et_5m = df_5m_full.index.tz_convert('US/Eastern')
df_1m = df_1m_full[et_1m.year == 2023].copy()
df_5m = df_5m_full[et_5m.year == 2023].copy()
print(f"  1m 2023: {len(df_1m):,} bars")
print(f"  5m 2023: {len(df_5m):,} bars")


# ============================================================================
# PART 1: Understanding the existing 1m pipeline
# ============================================================================
print(f"\n{SEPARATOR}")
print("PART 1: EXISTING 1m PIPELINE ANALYSIS")
print(SEPARATOR)
print("""
The existing 1m_signal_backtest_fast.py works as follows:
  - Loads ALL 10yr of 1m + 5m + 1H + 4H data
  - Computes bias on 5m (using sessions, ORM, daily bias, regime)
  - Processes year-by-year:
    1. detect_fvg() on 1m data — vectorized, fast
    2. compute_atr(), compute_fluency(), detect_displacement() on 1m
    3. compute_swing_levels() on 1m for stop fallback
    4. FVG pool with max-age pruning (500 bars = ~8 hours)
    5. Signal loop: check each bar for FVG test+rejection
    6. Bias filter, session filter, stop filter, RR filter
    7. Simulate on 1m bars (300 bar max hold)

Key differences from 5m:
  - FVG_MAX_AGE=500 (critical on 1m to keep pool manageable)
  - Stop = candle-2 open from 1m FVG (tighter than 5m)
  - IRL target = nearest 5m swing high (same HTF target)
  - Max hold = 300 1m bars (5 hours)

Performance notes from existing script:
  - Feature computation is vectorized and fast
  - Signal loop is O(n * active_pool_size) — FVG pruning keeps pool small
  - Year-by-year processing to manage memory
""")


# ============================================================================
# PART 2: 1m vs 5m STOP DISTANCE COMPARISON
# ============================================================================
print(f"\n{SEPARATOR}")
print("PART 2: 1m vs 5m STOP DISTANCE COMPARISON")
print(SEPARATOR)

from features.fvg import detect_fvg
from features.displacement import compute_atr, compute_fluency, detect_displacement
from features.swing import compute_swing_levels

# ---- 1m FVG detection + stop distances ----
print("\n[Computing 1m features...]")
t1 = time.time()
fvg_1m = detect_fvg(df_1m)
atr_1m = compute_atr(df_1m, period=14)
fluency_1m = compute_fluency(df_1m, params)
disp_1m = detect_displacement(df_1m, params)
print(f"  1m features: {time.time()-t1:.1f}s")

n_bull_1m = fvg_1m['fvg_bull'].sum()
n_bear_1m = fvg_1m['fvg_bear'].sum()
print(f"  1m FVGs: {n_bull_1m:,} bullish, {n_bear_1m:,} bearish, {n_bull_1m+n_bear_1m:,} total")

# ---- 5m FVG detection + stop distances ----
print("\n[Computing 5m features...]")
t2 = time.time()
fvg_5m = detect_fvg(df_5m)
atr_5m = compute_atr(df_5m, period=14)
fluency_5m = compute_fluency(df_5m, params)
disp_5m = detect_displacement(df_5m, params)
print(f"  5m features: {time.time()-t2:.1f}s")

n_bull_5m = fvg_5m['fvg_bull'].sum()
n_bear_5m = fvg_5m['fvg_bear'].sum()
print(f"  5m FVGs: {n_bull_5m:,} bullish, {n_bear_5m:,} bearish, {n_bull_5m+n_bear_5m:,} total")


# ---- Compute stop distances for 1m FVG signals ----
# Stop = candle-2 open (the candle BEFORE the FVG detection bar, since FVG is shifted by 1)
# For a bull FVG at bar i: candle-2 is at i-1 (before the shift), so after shift the
# FVG appears at i, meaning candle-2 was at i-2 originally. The open of candle-2 = open[i-1]
# (since the detection bar after shift references candle-2 at shift-1 position).
# Actually: detect_fvg shifts by 1. At row i with fvg_bull=True, the actual candle-2
# was at index i-1 (before shift it was computed at i-1, then rolled to i).
# So candle-2's open = df.open[i-1].

print("\n[Computing 1m stop distances...]")

open_1m = df_1m['open'].values
close_1m = df_1m['close'].values
high_1m = df_1m['high'].values
low_1m = df_1m['low'].values
atr_1m_vals = atr_1m.values

bull_idx_1m = np.where(fvg_1m['fvg_bull'].values)[0]
bear_idx_1m = np.where(fvg_1m['fvg_bear'].values)[0]

stops_1m_bull = []
stops_1m_bear = []

for i in bull_idx_1m:
    if i < 2: continue
    # Entry would be close of rejection candle (approximately close[i] for analysis)
    entry = close_1m[i]
    # Stop = open of candle-2 (the displacement candle, which is at i-1 after shift)
    stop = open_1m[i-1]
    sd = abs(entry - stop)
    if sd > 0 and not np.isnan(sd):
        stops_1m_bull.append(sd)

for i in bear_idx_1m:
    if i < 2: continue
    entry = close_1m[i]
    stop = open_1m[i-1]
    sd = abs(entry - stop)
    if sd > 0 and not np.isnan(sd):
        stops_1m_bear.append(sd)

stops_1m_all = stops_1m_bull + stops_1m_bear
stops_1m_arr = np.array(stops_1m_all)

# ---- 5m stop distances ----
print("[Computing 5m stop distances...]")

open_5m = df_5m['open'].values
close_5m = df_5m['close'].values

bull_idx_5m = np.where(fvg_5m['fvg_bull'].values)[0]
bear_idx_5m = np.where(fvg_5m['fvg_bear'].values)[0]

stops_5m_bull = []
stops_5m_bear = []

for i in bull_idx_5m:
    if i < 2: continue
    entry = close_5m[i]
    stop = open_5m[i-1]
    sd = abs(entry - stop)
    if sd > 0 and not np.isnan(sd):
        stops_5m_bull.append(sd)

for i in bear_idx_5m:
    if i < 2: continue
    entry = close_5m[i]
    stop = open_5m[i-1]
    sd = abs(entry - stop)
    if sd > 0 and not np.isnan(sd):
        stops_5m_bear.append(sd)

stops_5m_all = stops_5m_bull + stops_5m_bear
stops_5m_arr = np.array(stops_5m_all)

# ---- Report ----
def pct(arr, p):
    return np.percentile(arr, p)

print(f"\n{'':>20}{'1m FVGs':>15}{'5m FVGs':>15}{'Ratio 5m/1m':>15}")
print(THIN_SEP)
print(f"{'Total FVGs':>20}{len(stops_1m_arr):>15,}{len(stops_5m_arr):>15,}{'':>15}")
print(f"{'Bull FVGs':>20}{len(stops_1m_bull):>15,}{len(stops_5m_bull):>15,}{'':>15}")
print(f"{'Bear FVGs':>20}{len(stops_1m_bear):>15,}{len(stops_5m_bear):>15,}{'':>15}")
print()
print(f"{'STOP DISTANCE (pts)':>20}{'1m':>15}{'5m':>15}{'5m/1m':>15}")
print(THIN_SEP)

if len(stops_1m_arr) > 0 and len(stops_5m_arr) > 0:
    m1_p25, m1_p50, m1_p75, m1_mean = pct(stops_1m_arr, 25), pct(stops_1m_arr, 50), pct(stops_1m_arr, 75), stops_1m_arr.mean()
    m5_p25, m5_p50, m5_p75, m5_mean = pct(stops_5m_arr, 25), pct(stops_5m_arr, 50), pct(stops_5m_arr, 75), stops_5m_arr.mean()

    print(f"{'P25':>20}{m1_p25:>15.1f}{m5_p25:>15.1f}{m5_p25/m1_p25:>15.2f}x")
    print(f"{'Median (P50)':>20}{m1_p50:>15.1f}{m5_p50:>15.1f}{m5_p50/m1_p50:>15.2f}x")
    print(f"{'P75':>20}{m1_p75:>15.1f}{m5_p75:>15.1f}{m5_p75/m1_p75:>15.2f}x")
    print(f"{'Mean':>20}{m1_mean:>15.1f}{m5_mean:>15.1f}{m5_mean/m1_mean:>15.2f}x")
    print(f"{'P10':>20}{pct(stops_1m_arr, 10):>15.1f}{pct(stops_5m_arr, 10):>15.1f}{pct(stops_5m_arr, 10)/pct(stops_1m_arr, 10):>15.2f}x")
    print(f"{'P90':>20}{pct(stops_1m_arr, 90):>15.1f}{pct(stops_5m_arr, 90):>15.1f}{pct(stops_5m_arr, 90)/pct(stops_1m_arr, 90):>15.2f}x")

    print(f"\n  Key finding: 1m FVG stops are ~{m5_p50/m1_p50:.1f}x tighter at the median")
    print(f"  1m median stop: {m1_p50:.1f} pts vs 5m median stop: {m5_p50:.1f} pts")
else:
    print("  ERROR: No stop distances computed!")


# ============================================================================
# PART 3: SIGNAL COUNT COMPARISON
# ============================================================================
print(f"\n{SEPARATOR}")
print("PART 3: SIGNAL COUNT COMPARISON")
print(SEPARATOR)

# Count signals per day on 1m and 5m
# A "signal" = FVG birth event that could potentially generate a trade signal
# when price returns to test it. As a proxy, count raw FVG births.

et_1m_idx = df_1m.index.tz_convert('US/Eastern')
et_5m_idx = df_5m.index.tz_convert('US/Eastern')

fvg_1m_dates = et_1m_idx[fvg_1m['fvg_bull'].values | fvg_1m['fvg_bear'].values]
fvg_5m_dates = et_5m_idx[fvg_5m['fvg_bull'].values | fvg_5m['fvg_bear'].values]

fvg_1m_per_day = pd.Series(fvg_1m_dates).dt.date.value_counts()
fvg_5m_per_day = pd.Series(fvg_5m_dates).dt.date.value_counts()

print(f"\nRaw FVG births per day (2023):")
print(f"{'':>25}{'1m':>12}{'5m':>12}")
print(THIN_SEP)
print(f"{'Mean FVGs/day':>25}{fvg_1m_per_day.mean():>12.1f}{fvg_5m_per_day.mean():>12.1f}")
print(f"{'Median FVGs/day':>25}{fvg_1m_per_day.median():>12.1f}{fvg_5m_per_day.median():>12.1f}")
print(f"{'Max FVGs/day':>25}{fvg_1m_per_day.max():>12}{fvg_5m_per_day.max():>12}")
print(f"{'Trading days':>25}{len(fvg_1m_per_day):>12}{len(fvg_5m_per_day):>12}")

print(f"\n  Ratio: 1m produces ~{fvg_1m_per_day.mean()/fvg_5m_per_day.mean():.1f}x more FVGs than 5m per day")

# Now do signal overlap analysis:
# For each 5m FVG timestamp, check if there's a 1m FVG within +/- 5 minutes
print(f"\n[Signal overlap analysis...]")

fvg_1m_ts = fvg_1m_dates.sort_values()
fvg_5m_ts = fvg_5m_dates.sort_values()

# Convert to minutes from epoch for fast matching
fvg_1m_mins = (pd.DatetimeIndex(fvg_1m_ts).astype(np.int64) // 10**9 // 60).values
fvg_5m_mins = (pd.DatetimeIndex(fvg_5m_ts).astype(np.int64) // 10**9 // 60).values

overlap_count = 0
for m5 in fvg_5m_mins:
    # Check if any 1m FVG is within 5 minutes of this 5m FVG
    diffs = np.abs(fvg_1m_mins - m5)
    if np.any(diffs <= 5):
        overlap_count += 1

overlap_pct = 100 * overlap_count / len(fvg_5m_mins) if len(fvg_5m_mins) > 0 else 0
print(f"  5m FVGs with a 1m FVG within +/-5 min: {overlap_count}/{len(fvg_5m_mins)} ({overlap_pct:.1f}%)")
print(f"  Unique 1m FVGs (no 5m counterpart within 5 min): ~{len(fvg_1m_mins) - overlap_count}")

# Now simulate ACTUAL trade signal generation on 1m (simplified)
# Use the same logic as 1m_signal_backtest_fast.py but just count signals
print(f"\n[Simulating signal detection on 1m (2023)...]")

FVG_MAX_AGE = 500
entry_cfg = params.get('entry', {})
min_fvg_atr_mult = entry_cfg.get('min_fvg_atr_mult', 0.5)
rejection_body_ratio = entry_cfg.get('rejection_body_ratio', 0.55)
signal_cooldown_bars = entry_cfg.get('signal_cooldown_bars', 10)
fluency_thresh = params['fluency']['threshold']

t_sig = time.time()

n_1m = len(df_1m)
bull_mask_1m = fvg_1m['fvg_bull'].values
bear_mask_1m = fvg_1m['fvg_bear'].values
fvg_bull_top_1m = fvg_1m['fvg_bull_top'].values
fvg_bull_bot_1m = fvg_1m['fvg_bull_bottom'].values
fvg_bear_top_1m = fvg_1m['fvg_bear_top'].values
fvg_bear_bot_1m = fvg_1m['fvg_bear_bottom'].values
fvg_size_1m = fvg_1m['fvg_size'].values

body_1m = np.abs(close_1m - open_1m)
range_1m = high_1m - low_1m
safe_range_1m = np.where(range_1m == 0, np.nan, range_1m)
body_ratio_1m = body_1m / safe_range_1m
fluency_1m_vals = fluency_1m.values

# FVG pool: [idx, dir, top, bot, size, c2_open, status, last_sig_idx]
active_fvgs_1m = []
signals_1m = []  # (idx, direction, c2_open, stop_dist)

for i in range(n_1m):
    c2_open_val = open_1m[i-1] if i > 0 else open_1m[i]

    if bull_mask_1m[i]:
        sz = fvg_size_1m[i]
        if not np.isnan(sz) and sz > 0:
            active_fvgs_1m.append([i, 1, fvg_bull_top_1m[i], fvg_bull_bot_1m[i], sz, c2_open_val, 0, -999])

    if bear_mask_1m[i]:
        sz = fvg_size_1m[i]
        if not np.isnan(sz) and sz > 0:
            active_fvgs_1m.append([i, -1, fvg_bear_top_1m[i], fvg_bear_bot_1m[i], sz, c2_open_val, 0, -999])

    # Quality checks
    cur_br = body_ratio_1m[i] if not np.isnan(body_ratio_1m[i]) else 0.0
    cur_fluency = fluency_1m_vals[i] if not np.isnan(fluency_1m_vals[i]) else 0.0
    cur_atr = atr_1m_vals[i] if not np.isnan(atr_1m_vals[i]) else 0.0

    if cur_br >= rejection_body_ratio and cur_fluency >= fluency_thresh:
        best_dir = 0; best_score = -1.0; best_rec = None
        for rec in active_fvgs_1m:
            if rec[6] == 2: continue
            if rec[0] >= i: continue
            if cur_atr > 0 and rec[4] < min_fvg_atr_mult * cur_atr: continue
            if (i - rec[7]) < signal_cooldown_bars: continue

            d = rec[1]
            top, bot = rec[2], rec[3]

            if d == 1:
                entered = low_1m[i] <= top and high_1m[i] >= bot
                rejected = close_1m[i] > top
                if entered and rejected:
                    score = rec[4]
                    if score > best_score:
                        best_score = score; best_dir = 1; best_rec = rec
            elif d == -1:
                entered = high_1m[i] >= bot and low_1m[i] <= top
                rejected = close_1m[i] < bot
                if entered and rejected:
                    score = rec[4]
                    if score > best_score:
                        best_score = score; best_dir = -1; best_rec = rec

        if best_rec is not None:
            best_rec[7] = i
            entry_price = close_1m[i]
            stop_price = best_rec[5]
            sd = abs(entry_price - stop_price)
            signals_1m.append((i, best_dir, best_rec[5], sd))

    # Update FVG states + prune
    new_active = []
    for rec in active_fvgs_1m:
        if rec[6] == 2: continue
        if (i - rec[0]) > FVG_MAX_AGE: continue

        d = rec[1]
        top, bot = rec[2], rec[3]
        old_status = rec[6]

        if d == 1:
            if close_1m[i] < bot:
                rec[6] = 2; continue
            elif low_1m[i] <= top and high_1m[i] >= bot:
                if old_status == 0: rec[6] = 1
        else:
            if close_1m[i] > top:
                rec[6] = 2; continue
            elif high_1m[i] >= bot and low_1m[i] <= top:
                if old_status == 0: rec[6] = 1

        new_active.append(rec)

    active_fvgs_1m = new_active

sig_time_1m = time.time() - t_sig
print(f"  1m signal detection: {sig_time_1m:.1f}s for {n_1m:,} bars ({n_1m/sig_time_1m:,.0f} bars/s)")
print(f"  1m raw signals (pre-filter): {len(signals_1m):,}")

# Extract signal stop distances
sig_stops_1m = np.array([s[3] for s in signals_1m])
sig_stops_1m_valid = sig_stops_1m[sig_stops_1m > 0]

# ---- 5m signal detection (simplified, same logic) ----
print(f"\n[Simulating signal detection on 5m (2023)...]")
t_sig5 = time.time()

n_5m = len(df_5m)
bull_mask_5m = fvg_5m['fvg_bull'].values
bear_mask_5m = fvg_5m['fvg_bear'].values
fvg_bull_top_5m = fvg_5m['fvg_bull_top'].values
fvg_bull_bot_5m = fvg_5m['fvg_bull_bottom'].values
fvg_bear_top_5m = fvg_5m['fvg_bear_top'].values
fvg_bear_bot_5m = fvg_5m['fvg_bear_bottom'].values
fvg_size_5m_vals = fvg_5m['fvg_size'].values

body_ratio_5m = np.abs(close_5m - open_5m) / np.where(
    (df_5m['high'].values - df_5m['low'].values) == 0, np.nan,
    df_5m['high'].values - df_5m['low'].values
)
fluency_5m_vals = fluency_5m.values
atr_5m_vals = atr_5m.values
high_5m = df_5m['high'].values
low_5m = df_5m['low'].values

active_fvgs_5m = []
signals_5m = []

for i in range(n_5m):
    c2_open_val = open_5m[i-1] if i > 0 else open_5m[i]

    if bull_mask_5m[i]:
        sz = fvg_size_5m_vals[i]
        if not np.isnan(sz) and sz > 0:
            active_fvgs_5m.append([i, 1, fvg_bull_top_5m[i], fvg_bull_bot_5m[i], sz, c2_open_val, 0, -999])

    if bear_mask_5m[i]:
        sz = fvg_size_5m_vals[i]
        if not np.isnan(sz) and sz > 0:
            active_fvgs_5m.append([i, -1, fvg_bear_top_5m[i], fvg_bear_bot_5m[i], sz, c2_open_val, 0, -999])

    cur_br = body_ratio_5m[i] if not np.isnan(body_ratio_5m[i]) else 0.0
    cur_fluency = fluency_5m_vals[i] if not np.isnan(fluency_5m_vals[i]) else 0.0
    cur_atr = atr_5m_vals[i] if not np.isnan(atr_5m_vals[i]) else 0.0

    if cur_br >= rejection_body_ratio and cur_fluency >= fluency_thresh:
        best_dir = 0; best_score = -1.0; best_rec = None
        for rec in active_fvgs_5m:
            if rec[6] == 2: continue
            if rec[0] >= i: continue
            if cur_atr > 0 and rec[4] < min_fvg_atr_mult * cur_atr: continue
            if (i - rec[7]) < signal_cooldown_bars: continue

            d = rec[1]
            top, bot = rec[2], rec[3]

            if d == 1:
                entered = low_5m[i] <= top and high_5m[i] >= bot
                rejected = close_5m[i] > top
                if entered and rejected:
                    score = rec[4]
                    if score > best_score:
                        best_score = score; best_dir = 1; best_rec = rec
            elif d == -1:
                entered = high_5m[i] >= bot and low_5m[i] <= top
                rejected = close_5m[i] < bot
                if entered and rejected:
                    score = rec[4]
                    if score > best_score:
                        best_score = score; best_dir = -1; best_rec = rec

        if best_rec is not None:
            best_rec[7] = i
            entry_price = close_5m[i]
            stop_price = best_rec[5]
            sd = abs(entry_price - stop_price)
            signals_5m.append((i, best_dir, best_rec[5], sd))

    # Update FVG states
    new_active = []
    for rec in active_fvgs_5m:
        if rec[6] == 2: continue
        d = rec[1]
        top, bot = rec[2], rec[3]
        old_status = rec[6]

        if d == 1:
            if close_5m[i] < bot:
                rec[6] = 2; continue
            elif low_5m[i] <= top and high_5m[i] >= bot:
                if old_status == 0: rec[6] = 1
        else:
            if close_5m[i] > top:
                rec[6] = 2; continue
            elif high_5m[i] >= bot and low_5m[i] <= top:
                if old_status == 0: rec[6] = 1

        new_active.append(rec)

    active_fvgs_5m = new_active

sig_time_5m = time.time() - t_sig5
print(f"  5m signal detection: {sig_time_5m:.1f}s for {n_5m:,} bars ({n_5m/sig_time_5m:,.0f} bars/s)")
print(f"  5m raw signals (pre-filter): {len(signals_5m):,}")

sig_stops_5m = np.array([s[3] for s in signals_5m])
sig_stops_5m_valid = sig_stops_5m[sig_stops_5m > 0]

# ---- Report ----
# Signals per day
sig_1m_dates = pd.Series([et_1m_idx[s[0]] for s in signals_1m]).dt.date
sig_5m_dates = pd.Series([et_5m_idx[s[0]] for s in signals_5m]).dt.date
sig_1m_per_day = sig_1m_dates.value_counts()
sig_5m_per_day = sig_5m_dates.value_counts()

print(f"\n{'ACTUAL SIGNALS':>25}{'1m':>12}{'5m':>12}{'Ratio':>12}")
print(THIN_SEP)
print(f"{'Total signals':>25}{len(signals_1m):>12,}{len(signals_5m):>12,}{len(signals_1m)/max(len(signals_5m),1):>12.1f}x")
print(f"{'Signals/day (mean)':>25}{sig_1m_per_day.mean():>12.1f}{sig_5m_per_day.mean():>12.1f}{sig_1m_per_day.mean()/max(sig_5m_per_day.mean(),0.01):>12.1f}x")
print(f"{'Signals/day (median)':>25}{sig_1m_per_day.median():>12.1f}{sig_5m_per_day.median():>12.1f}{'':>12}")

if len(sig_stops_1m_valid) > 0 and len(sig_stops_5m_valid) > 0:
    print(f"\n{'SIGNAL STOP DISTANCE':>25}{'1m':>12}{'5m':>12}{'5m/1m':>12}")
    print(THIN_SEP)
    print(f"{'Median':>25}{np.median(sig_stops_1m_valid):>12.1f}{np.median(sig_stops_5m_valid):>12.1f}{np.median(sig_stops_5m_valid)/np.median(sig_stops_1m_valid):>12.2f}x")
    print(f"{'Mean':>25}{sig_stops_1m_valid.mean():>12.1f}{sig_stops_5m_valid.mean():>12.1f}{sig_stops_5m_valid.mean()/sig_stops_1m_valid.mean():>12.2f}x")
    print(f"{'P25':>25}{np.percentile(sig_stops_1m_valid,25):>12.1f}{np.percentile(sig_stops_5m_valid,25):>12.1f}{np.percentile(sig_stops_5m_valid,25)/np.percentile(sig_stops_1m_valid,25):>12.2f}x")
    print(f"{'P75':>25}{np.percentile(sig_stops_1m_valid,75):>12.1f}{np.percentile(sig_stops_5m_valid,75):>12.1f}{np.percentile(sig_stops_5m_valid,75)/np.percentile(sig_stops_1m_valid,75):>12.2f}x")


# ============================================================================
# PART 4: THEORETICAL R IMPROVEMENT
# ============================================================================
print(f"\n{SEPARATOR}")
print("PART 4: THEORETICAL R IMPROVEMENT")
print(SEPARATOR)

point_value = params['position']['point_value']  # $2 for MNQ
R_amount = params['position']['normal_r']  # $1000

print(f"\nAssumptions:")
print(f"  R amount: ${R_amount}")
print(f"  Point value: ${point_value} (MNQ)")
print(f"  contracts = ${R_amount} / (stop_pts * ${point_value})")

if len(sig_stops_1m_valid) > 0 and len(sig_stops_5m_valid) > 0:
    median_1m_stop = np.median(sig_stops_1m_valid)
    median_5m_stop = np.median(sig_stops_5m_valid)

    scenarios = [
        ("1m Median", median_1m_stop),
        ("1m P25 (tight)", np.percentile(sig_stops_1m_valid, 25)),
        ("1m P75 (wide)", np.percentile(sig_stops_1m_valid, 75)),
        ("5m Median", median_5m_stop),
        ("5m P25 (tight)", np.percentile(sig_stops_5m_valid, 25)),
    ]

    print(f"\n{'Scenario':>25}{'Stop(pts)':>12}{'Contracts':>12}{'$/pt move':>12}{'TP1 @ 20pts':>12}")
    print(THIN_SEP)
    for name, stop_pts in scenarios:
        contracts = max(1, int(R_amount / (stop_pts * point_value)))
        dollar_per_pt = contracts * point_value
        tp1_profit = 20 * dollar_per_pt  # 20 points TP1
        print(f"{name:>25}{stop_pts:>12.1f}{contracts:>12}{dollar_per_pt:>12.0f}{tp1_profit:>12,.0f}")

    print(f"\n  If TP1 is 20 pts for both:")
    c_1m = max(1, int(R_amount / (median_1m_stop * point_value)))
    c_5m = max(1, int(R_amount / (median_5m_stop * point_value)))
    multiplier = c_1m / c_5m if c_5m > 0 else 0
    print(f"    1m: {c_1m} contracts @ {median_1m_stop:.1f} pts stop → ${20 * c_1m * point_value:,.0f} at TP1")
    print(f"    5m: {c_5m} contracts @ {median_5m_stop:.1f} pts stop → ${20 * c_5m * point_value:,.0f} at TP1")
    print(f"    Profit multiplier: {multiplier:.1f}x more with 1m entries")
    print(f"\n  BUT: tighter stop → more stop-outs. Need to check win rate.")


# ============================================================================
# PART 5: QUICK 1m SIGNAL QUALITY CHECK
# ============================================================================
print(f"\n{SEPARATOR}")
print("PART 5: QUICK 1m SIGNAL QUALITY CHECK")
print(SEPARATOR)

# For each signal, simulate forward on 1m data:
#   - Does price hit TP1 (entry + 20 pts for long, entry - 20 pts for short) before SL?
#   - Max hold: 300 bars (5 hours)
# Also run same simulation for 5m signals on 5m bars.

print("\n[Simulating 1m signal outcomes...]")

TP1_POINTS = 20  # Fixed TP1 distance for comparison
MAX_HOLD_1M = 300  # 5 hours on 1m

results_1m = {'win': 0, 'loss': 0, 'timeout': 0, 'total_pnl': 0.0}
stop_dist_wins_1m = []
stop_dist_losses_1m = []

for (idx, sig_d, c2_open, sd) in signals_1m:
    if idx + 2 >= n_1m: continue
    if sd <= 0 or np.isnan(sd): continue

    # Basic session filter (NY only: 10:00-16:00 ET)
    et_hour = et_1m_idx[idx].hour
    et_min = et_1m_idx[idx].minute
    if not (10 <= et_hour < 16): continue
    if et_hour == 10 and et_min < 0: continue  # already covered by 10 <=

    ep = open_1m[idx + 1]  # entry on next bar open
    if np.isnan(ep): continue

    if sig_d == 1:
        sl = ep - sd  # long stop below
        tp = ep + TP1_POINTS
    else:
        sl = ep + sd   # short stop above
        tp = ep - TP1_POINTS

    # Simulate
    outcome = 'timeout'
    pnl = 0.0
    for j in range(idx + 2, min(idx + 2 + MAX_HOLD_1M, n_1m)):
        if sig_d == 1:
            if low_1m[j] <= sl:
                outcome = 'loss'; pnl = -sd * point_value; break
            if high_1m[j] >= tp:
                outcome = 'win'; pnl = TP1_POINTS * point_value; break
        else:
            if high_1m[j] >= sl:
                outcome = 'loss'; pnl = -sd * point_value; break
            if low_1m[j] <= tp:
                outcome = 'win'; pnl = TP1_POINTS * point_value; break

    if outcome == 'timeout':
        # Close at end of hold
        exit_price = close_1m[min(idx + 1 + MAX_HOLD_1M, n_1m - 1)]
        if sig_d == 1:
            pnl = (exit_price - ep) * point_value
        else:
            pnl = (ep - exit_price) * point_value

    results_1m[outcome] += 1
    results_1m['total_pnl'] += pnl
    if outcome == 'win':
        stop_dist_wins_1m.append(sd)
    elif outcome == 'loss':
        stop_dist_losses_1m.append(sd)

total_1m_tested = results_1m['win'] + results_1m['loss'] + results_1m['timeout']
wr_1m = 100 * results_1m['win'] / total_1m_tested if total_1m_tested > 0 else 0

# Same for 5m signals
print("[Simulating 5m signal outcomes...]")
MAX_HOLD_5M = 60  # 60 × 5m = 5 hours

results_5m_sim = {'win': 0, 'loss': 0, 'timeout': 0, 'total_pnl': 0.0}
stop_dist_wins_5m = []
stop_dist_losses_5m = []

for (idx, sig_d, c2_open, sd) in signals_5m:
    if idx + 2 >= n_5m: continue
    if sd <= 0 or np.isnan(sd): continue

    et_hour = et_5m_idx[idx].hour
    et_min = et_5m_idx[idx].minute
    if not (10 <= et_hour < 16): continue

    ep = open_5m[idx + 1]
    if np.isnan(ep): continue

    if sig_d == 1:
        sl = ep - sd
        tp = ep + TP1_POINTS
    else:
        sl = ep + sd
        tp = ep - TP1_POINTS

    outcome = 'timeout'
    pnl = 0.0
    for j in range(idx + 2, min(idx + 2 + MAX_HOLD_5M, n_5m)):
        if sig_d == 1:
            if low_5m[j] <= sl:
                outcome = 'loss'; pnl = -sd * point_value; break
            if high_5m[j] >= tp:
                outcome = 'win'; pnl = TP1_POINTS * point_value; break
        else:
            if high_5m[j] >= sl:
                outcome = 'loss'; pnl = -sd * point_value; break
            if low_5m[j] <= tp:
                outcome = 'win'; pnl = TP1_POINTS * point_value; break

    if outcome == 'timeout':
        exit_price = close_5m[min(idx + 1 + MAX_HOLD_5M, n_5m - 1)]
        if sig_d == 1:
            pnl = (exit_price - ep) * point_value
        else:
            pnl = (ep - exit_price) * point_value

    results_5m_sim[outcome] += 1
    results_5m_sim['total_pnl'] += pnl
    if outcome == 'win':
        stop_dist_wins_5m.append(sd)
    elif outcome == 'loss':
        stop_dist_losses_5m.append(sd)

total_5m_tested = results_5m_sim['win'] + results_5m_sim['loss'] + results_5m_sim['timeout']
wr_5m = 100 * results_5m_sim['win'] / total_5m_tested if total_5m_tested > 0 else 0

# ---- Report ----
print(f"\n{'OUTCOME (TP1=20pts)':>25}{'1m Signals':>15}{'5m Signals':>15}")
print(THIN_SEP)
print(f"{'Tested (NY session)':>25}{total_1m_tested:>15,}{total_5m_tested:>15,}")
print(f"{'Wins':>25}{results_1m['win']:>15,}{results_5m_sim['win']:>15,}")
print(f"{'Losses':>25}{results_1m['loss']:>15,}{results_5m_sim['loss']:>15,}")
print(f"{'Timeouts':>25}{results_1m['timeout']:>15,}{results_5m_sim['timeout']:>15,}")
print(f"{'Win Rate':>25}{wr_1m:>14.1f}%{wr_5m:>14.1f}%")
pnl_1m_str = f"${results_1m['total_pnl']:,.0f}"
pnl_5m_str = f"${results_5m_sim['total_pnl']:,.0f}"
print(f"{'Total PnL (1 contract)':>25}{pnl_1m_str:>15}{pnl_5m_str:>15}")

# Expected value per trade analysis
if total_1m_tested > 0 and total_5m_tested > 0:
    # For 1m: with tighter stop, more contracts per R
    avg_win_1m = TP1_POINTS * point_value
    avg_loss_1m = np.mean(stop_dist_losses_1m) * point_value if stop_dist_losses_1m else 0
    ev_1m_per_contract = (wr_1m/100) * avg_win_1m - ((100-wr_1m)/100) * avg_loss_1m

    avg_loss_5m = np.mean(stop_dist_losses_5m) * point_value if stop_dist_losses_5m else 0
    ev_5m_per_contract = (wr_5m/100) * TP1_POINTS * point_value - ((100-wr_5m)/100) * avg_loss_5m

    print(f"\n{'EV ANALYSIS':>25}{'1m':>15}{'5m':>15}")
    print(THIN_SEP)
    print(f"{'Avg win ($)':>25}{'$'+f'{avg_win_1m:.0f}':>15}{'$'+f'{TP1_POINTS * point_value:.0f}':>15}")
    print(f"{'Avg loss ($)':>25}{'$'+f'{avg_loss_1m:.0f}':>15}{'$'+f'{avg_loss_5m:.0f}':>15}")
    print(f"{'EV/trade (1 contract)':>25}{'$'+f'{ev_1m_per_contract:.1f}':>15}{'$'+f'{ev_5m_per_contract:.1f}':>15}")

    # Now with R-sized positions
    if len(sig_stops_1m_valid) > 0:
        med_stop_1m = np.median(sig_stops_1m_valid)
        contracts_1m = max(1, int(R_amount / (med_stop_1m * point_value)))
    else:
        contracts_1m = 1
    if len(sig_stops_5m_valid) > 0:
        med_stop_5m = np.median(sig_stops_5m_valid)
        contracts_5m = max(1, int(R_amount / (med_stop_5m * point_value)))
    else:
        contracts_5m = 1

    ev_1m_R = ev_1m_per_contract * contracts_1m
    ev_5m_R = ev_5m_per_contract * contracts_5m

    print(f"\n{'WITH R-SIZED POSITION':>25}{'1m':>15}{'5m':>15}")
    print(THIN_SEP)
    print(f"{'Median stop (pts)':>25}{med_stop_1m:>15.1f}{med_stop_5m:>15.1f}")
    print(f"{'Contracts':>25}{contracts_1m:>15}{contracts_5m:>15}")
    print(f"{'EV/trade ($)':>25}{'$'+f'{ev_1m_R:.0f}':>15}{'$'+f'{ev_5m_R:.0f}':>15}")
    print(f"{'EV multiplier':>25}{'':>15}{f'{ev_1m_R/ev_5m_R:.2f}x' if ev_5m_R != 0 else 'N/A':>15}")


# ============================================================================
# PART 5b: BREAKDOWN BY STOP DISTANCE BUCKET
# ============================================================================
print(f"\n{THIN_SEP}")
print("PART 5b: 1m WIN RATE BY STOP DISTANCE BUCKET")
print(THIN_SEP)

# Collect all 1m signal outcomes with stop distances
all_1m_outcomes = []
for (idx, sig_d, c2_open, sd) in signals_1m:
    if idx + 2 >= n_1m: continue
    if sd <= 0 or np.isnan(sd): continue
    et_hour = et_1m_idx[idx].hour
    if not (10 <= et_hour < 16): continue
    ep = open_1m[idx + 1]
    if np.isnan(ep): continue

    if sig_d == 1:
        sl = ep - sd; tp = ep + TP1_POINTS
    else:
        sl = ep + sd; tp = ep - TP1_POINTS

    outcome = 'timeout'
    for j in range(idx + 2, min(idx + 2 + MAX_HOLD_1M, n_1m)):
        if sig_d == 1:
            if low_1m[j] <= sl: outcome = 'loss'; break
            if high_1m[j] >= tp: outcome = 'win'; break
        else:
            if high_1m[j] >= sl: outcome = 'loss'; break
            if low_1m[j] <= tp: outcome = 'win'; break

    all_1m_outcomes.append({'sd': sd, 'outcome': outcome, 'dir': sig_d})

if all_1m_outcomes:
    out_df = pd.DataFrame(all_1m_outcomes)
    buckets = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 100), (100, 999)]

    print(f"\n{'Stop Range':>15}{'Count':>8}{'Wins':>8}{'Losses':>8}{'WR%':>8}{'RR at TP1':>10}")
    print(THIN_SEP)
    for lo, hi in buckets:
        mask = (out_df['sd'] >= lo) & (out_df['sd'] < hi)
        sub = out_df[mask]
        if len(sub) == 0: continue
        w = (sub['outcome'] == 'win').sum()
        l = (sub['outcome'] == 'loss').sum()
        n = len(sub)
        wr = 100 * w / n if n > 0 else 0
        avg_sd = sub['sd'].mean()
        rr = TP1_POINTS / avg_sd if avg_sd > 0 else 0
        print(f"  {lo:>3}-{hi:<3} pts {n:>8} {w:>8} {l:>8} {wr:>7.1f}% {rr:>9.1f}:1")

    # By direction
    print(f"\n{'BY DIRECTION':>15}{'Long':>12}{'Short':>12}")
    print(THIN_SEP)
    for d, label in [(1, 'Long'), (-1, 'Short')]:
        sub = out_df[out_df['dir'] == d]
        w = (sub['outcome'] == 'win').sum()
        n = len(sub)
        wr = 100 * w / n if n > 0 else 0
        print(f"  {'Count':>13} {(out_df['dir']==1).sum():>12} {(out_df['dir']==-1).sum():>12}")
    for d, label in [(1, 'Long'), (-1, 'Short')]:
        sub = out_df[out_df['dir'] == d]
        w = (sub['outcome'] == 'win').sum()
        n = len(sub)
        wr = 100 * w / n if n > 0 else 0
        if d == 1:
            wr_long = wr
        else:
            wr_short = wr
    print(f"  {'Win Rate':>13} {wr_long:>11.1f}% {wr_short:>11.1f}%")


# ============================================================================
# PART 6: FEASIBILITY ASSESSMENT
# ============================================================================
print(f"\n{SEPARATOR}")
print("PART 6: FEASIBILITY ASSESSMENT")
print(SEPARATOR)

# Runtime projections
print(f"\n--- Runtime ---")
print(f"  1m signal detection (2023, {n_1m:,} bars): {sig_time_1m:.1f}s")
print(f"  Projected 10yr (~3.5M bars): ~{sig_time_1m * 3515519 / n_1m:.0f}s ({sig_time_1m * 3515519 / n_1m / 60:.1f} min)")
print(f"  5m signal detection (2023, {n_5m:,} bars): {sig_time_5m:.1f}s")
print(f"  Projected 10yr (~711K bars): ~{sig_time_5m * 711141 / n_5m:.0f}s ({sig_time_5m * 711141 / n_5m / 60:.1f} min)")
print(f"  Speedup factor: 5m is ~{(sig_time_1m * 3515519 / n_1m) / max(sig_time_5m * 711141 / n_5m, 0.01):.1f}x faster than 1m")

# Feature computation time
feat_total_1m = time.time() - t1  # rough total from part 2
print(f"\n--- Feature Computation ---")
print(f"  1m features (fvg+atr+fluency+disp) for 2023: already computed above")
print(f"  Note: detect_fvg on 1m is ~5x slower than on 5m (5x more bars)")

# What's needed
print(f"\n--- What's Needed for 1m Config H Equivalent ---")
print(f"  1. 1m FVG detection: EXISTS (features/fvg.py works on any timeframe)")
print(f"  2. 1m signal loop with FVG pool: EXISTS (1m_signal_backtest_fast.py)")
print(f"  3. Bias from 5m/1H/4H: EXISTS (merge to 1m with ffill)")
print(f"  4. SQ scoring on 1m: NEEDS WORK (fluency/displacement on 1m, but SQ weights may need re-tuning)")
print(f"  5. Multi-TP on 1m: NEEDS WORK (TP targets from 5m swings, trail on 1m FVGs)")
print(f"  6. Stop ATR filter: NEEDS RE-CALIBRATION (min_stop_atr_mult was tuned for 5m)")
print(f"  7. Session regime: EXISTS (time-based, works on any timeframe)")
print(f"  8. Dual mode: NEEDS RE-CALIBRATION (short SQ threshold may differ on 1m)")

print(f"\n--- Existing Pipeline Usability ---")
print(f"  1m_signal_backtest_fast.py is ~80% of what's needed:")
print(f"    [x] FVG detection on 1m")
print(f"    [x] Signal loop with pool management")
print(f"    [x] Bias from 5m")
print(f"    [x] Session filter")
print(f"    [ ] Signal Quality (SQ) scoring")
print(f"    [ ] Multi-TP (only has simple 2R target)")
print(f"    [ ] PM shorts block")
print(f"    [ ] Session regime (lunch dead zone)")
print(f"    [ ] Dual mode (direction-specific SQ + TP)")
print(f"    [ ] ATR-relative min stop filter")
print(f"  Estimated work: 1-2 sessions to port Config H filters to 1m pipeline")


# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{SEPARATOR}")
print("SUMMARY")
print(SEPARATOR)

if len(sig_stops_1m_valid) > 0 and len(sig_stops_5m_valid) > 0:
    med_1m = np.median(sig_stops_1m_valid)
    med_5m = np.median(sig_stops_5m_valid)
    ratio = med_5m / med_1m

    print(f"""
  STOP DISTANCE:
    1m median: {med_1m:.1f} pts | 5m median: {med_5m:.1f} pts | Ratio: {ratio:.1f}x tighter

  SIGNAL VOLUME:
    1m: {len(signals_1m):,} signals | 5m: {len(signals_5m):,} signals | {len(signals_1m)/max(len(signals_5m),1):.1f}x more on 1m

  WIN RATE (TP1=20pts, unfiltered, NY session):
    1m: {wr_1m:.1f}% | 5m: {wr_5m:.1f}%

  THEORETICAL R IMPROVEMENT:
    At median stops with $1000R, MNQ:
    1m: {max(1, int(R_amount / (med_1m * point_value)))} contracts | 5m: {max(1, int(R_amount / (med_5m * point_value)))} contracts
    Profit multiplier at same TP: {max(1, int(R_amount / (med_1m * point_value))) / max(1, int(R_amount / (med_5m * point_value))):.1f}x

  RUNTIME:
    1m 10yr: ~{sig_time_1m * 3515519 / n_1m / 60:.0f} min | 5m 10yr: ~{sig_time_5m * 711141 / n_5m / 60:.0f} min

  FEASIBILITY:
    Pipeline exists at ~80%. Needs 1-2 sessions to port Config H.
    Primary risk: tighter stops may get clipped more often.
    Key metric to watch: net EV per trade with R-sized positions.
""")
else:
    print("  ERROR: Insufficient data to generate summary.")

print(f"Total analysis runtime: {time.time()-t0:.0f}s")
print(SEPARATOR)
