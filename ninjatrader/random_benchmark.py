"""
Random Benchmark: Is the edge in the SIGNAL or the TRADE MANAGEMENT?

Generate N random entry strategies with the SAME trade management as the real
FVG strategy, then compare. If random entries + our management produce similar
R, the edge is in the management, not the signal detection.

Reference results (from validate_nt_logic.py):
  Real strategy: ~534 trades, +156.63R, WR~45.9%, PPDD~10.39

Usage: python ninjatrader/random_benchmark.py
"""
import sys
import time
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT = Path(__file__).resolve().parent.parent

# ============================================================
# 1. LOAD BASE DATA
# ============================================================
print("=" * 70)
print("RANDOM BENCHMARK — Signal vs Trade Management Attribution")
print("=" * 70)
print("\n[1/6] Loading data...")
t0 = time.time()

nq = pd.read_parquet(PROJECT / "data" / "NQ_5m_10yr.parquet")
with open(PROJECT / "config" / "params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# Load real trades for reference distributions
real_trades_path = PROJECT / "ninjatrader" / "python_trades_545.csv"
if real_trades_path.exists():
    real_trades = pd.read_csv(real_trades_path)
    real_trades["entry_time"] = pd.to_datetime(real_trades["entry_time"])
else:
    print("ERROR: python_trades_545.csv not found. Run validate_nt_logic.py first.")
    sys.exit(1)

# Pre-compute arrays
from features.displacement import compute_atr
from features.swing import compute_swing_levels

et_idx = nq.index.tz_convert("US/Eastern")
o = nq["open"].values
h = nq["high"].values
l = nq["low"].values
c = nq["close"].values
n = len(nq)

atr_arr = compute_atr(nq).values
swing_p = {"left_bars": params["swing"]["left_bars"], "right_bars": params["swing"]["right_bars"]}
swings = compute_swing_levels(nq, swing_p)
swings["swing_high_price"] = swings["swing_high_price"].shift(1).ffill()
swings["swing_low_price"] = swings["swing_low_price"].shift(1).ffill()
swing_high_mask = swings["swing_high"].shift(1, fill_value=False).values
swing_low_mask = swings["swing_low"].shift(1, fill_value=False).values

# Session date tracking (same as engine.py)
dates = np.array([
    (et_idx[j] + pd.Timedelta(days=1)).date() if et_idx[j].hour >= 18 else et_idx[j].date()
    for j in range(n)
])

# ET fractional hours for session filtering
et_hours = np.array([et_idx[j].hour + et_idx[j].minute / 60.0 for j in range(n)])
et_dayofweek = np.array([et_idx[j].dayofweek for j in range(n)])

# News blackout
from features.news_filter import build_news_blackout_mask
news_path = PROJECT / "config" / "news_calendar.csv"
news_blackout_arr = None
if news_path.exists():
    news_bl = build_news_blackout_mask(nq.index, str(news_path),
        params["news"]["blackout_minutes_before"], params["news"]["cooldown_minutes_after"])
    news_blackout_arr = news_bl.values

print(f"  Data loaded: {n:,} bars, {len(real_trades)} real trades")
print(f"  Time: {time.time()-t0:.1f}s")

# ============================================================
# 2. EXTRACT REAL STRATEGY DISTRIBUTIONS
# ============================================================
print("\n[2/6] Extracting real strategy distributions...")

# Stop distance distribution (in ATR multiples for scale-invariance)
real_stop_dists = np.abs(real_trades["entry_price"].values - real_trades["stop_price"].values)
# Map real trade entry times to bar indices to get ATR at entry
real_entry_indices = []
for _, row in real_trades.iterrows():
    et = pd.Timestamp(row["entry_time"])
    if et.tzinfo is None:
        et = et.tz_localize("UTC")
    idx = nq.index.get_indexer([et], method="nearest")[0]
    real_entry_indices.append(idx)
real_entry_indices = np.array(real_entry_indices)

real_stop_atr_mults = []
for j, idx in enumerate(real_entry_indices):
    a = atr_arr[idx] if not np.isnan(atr_arr[idx]) else 30.0
    real_stop_atr_mults.append(real_stop_dists[j] / a if a > 0 else 1.0)
real_stop_atr_mults = np.array(real_stop_atr_mults)

# Direction ratio
real_long_ratio = (real_trades["dir"] == 1).mean()
real_short_ratio = 1.0 - real_long_ratio

# TP distance distribution (in ATR multiples)
real_tp_dists = np.abs(real_trades["tp1_price"].values - real_trades["entry_price"].values)
real_tp_atr_mults = []
for j, idx in enumerate(real_entry_indices):
    a = atr_arr[idx] if not np.isnan(atr_arr[idx]) else 30.0
    real_tp_atr_mults.append(real_tp_dists[j] / a if a > 0 else 2.0)
real_tp_atr_mults = np.array(real_tp_atr_mults)

# Session hour distribution (ET)
real_entry_et_hours = []
for idx in real_entry_indices:
    real_entry_et_hours.append(et_hours[idx])
real_entry_et_hours = np.array(real_entry_et_hours)

print(f"  Real strategy: {len(real_trades)} trades, R={real_trades['r'].sum():.2f}")
print(f"  Long/Short ratio: {real_long_ratio:.1%} / {real_short_ratio:.1%}")
print(f"  Stop dist (ATR mult): mean={real_stop_atr_mults.mean():.2f}, "
      f"std={real_stop_atr_mults.std():.2f}")
print(f"  TP dist (ATR mult): mean={real_tp_atr_mults.mean():.2f}, "
      f"std={real_tp_atr_mults.std():.2f}")

# ============================================================
# 3. BUILD ELIGIBLE BAR POOL
# ============================================================
print("\n[3/6] Building eligible bar pool (session + news filters)...")

# Eligible bars: NY session (10:00-16:00 ET), not news blackout, not weekends,
# skip lunch dead zone (12:30-13:00 ET as per session_regime)
eligible_mask = np.zeros(n, dtype=bool)
for i in range(n):
    ef = et_hours[i]
    # NY session after ORM (10:05 to 15:55 ET — leave room for trade management)
    if ef < 10.0 or ef >= 15.83:
        continue
    # Skip lunch dead zone
    if 12.5 <= ef < 13.0:
        continue
    # Skip weekends
    if et_dayofweek[i] >= 5:
        continue
    # Skip news blackout
    if news_blackout_arr is not None and news_blackout_arr[i]:
        continue
    # Must have valid ATR
    if np.isnan(atr_arr[i]):
        continue
    # Must have room for next bar (entry at i+1)
    if i + 1 >= n:
        continue
    eligible_mask[i] = True

eligible_indices = np.where(eligible_mask)[0]
print(f"  Eligible bars: {len(eligible_indices):,} / {n:,} ({100*len(eligible_indices)/n:.1f}%)")

# ============================================================
# 4. TRADE MANAGEMENT ENGINE (simplified, vectorizable per-trade)
# ============================================================
# Engine params
trim_params = params["trim"]
trail_params = params["trail"]
pos_params = params["position"]
bt_params = params["backtest"]
risk_params = params["risk"]

point_value = pos_params["point_value"]
commission_per_side = bt_params["commission_per_side_micro"]
slippage_ticks = bt_params["slippage_normal_ticks"]
slippage_points = slippage_ticks * 0.25
trim_pct = trim_params["pct"]
be_after_trim = trim_params["be_after_trim"]
nth_swing = trail_params["use_nth_swing"]
daily_max_loss_r = risk_params["daily_max_loss_r"]
max_consec_losses = risk_params["max_consecutive_losses"]


def _find_nth_swing(mask, prices, idx, n_val, direction):
    """Find nth most recent swing level before bar idx."""
    count = 0
    for j in range(idx - 1, -1, -1):
        if mask[j]:
            count += 1
            if count == n_val:
                return prices[j]
    return np.nan


def run_trades(entry_bars, directions, stop_atr_mults, tp_atr_mults,
               use_trim_pcts=None, max_bars_per_trade=100):
    """
    Run trade management for a list of entries.

    Parameters
    ----------
    entry_bars : array of bar indices where signal occurs (entry at bar+1)
    directions : array of +1 / -1
    stop_atr_mults : array of stop distance in ATR multiples
    tp_atr_mults : array of TP distance in ATR multiples
    use_trim_pcts : array of trim percentages (default: all use trim_pct from params)
    max_bars_per_trade : maximum bars to hold before forced exit

    Returns
    -------
    dict with total_r, trade_count, win_count, r_values, max_dd
    """
    if use_trim_pcts is None:
        use_trim_pcts = np.full(len(entry_bars), trim_pct)

    trades_r = []
    current_date = None
    daily_pnl_r = 0.0
    consecutive_losses = 0
    day_stopped = False

    # Sort entries by bar index to process chronologically
    order = np.argsort(entry_bars)
    next_available_bar = 0  # earliest bar we can enter (one-at-a-time)

    for oi in order:
        sig_bar = entry_bars[oi]

        if sig_bar < next_available_bar:
            continue  # blocked by previous position

        entry_idx = sig_bar + 1
        if entry_idx >= n:
            continue

        # Daily reset
        bar_date = dates[sig_bar]
        if bar_date != current_date:
            current_date = bar_date
            daily_pnl_r = 0.0
            consecutive_losses = 0
            day_stopped = False

        if day_stopped:
            continue

        direction = directions[oi]
        cur_atr = atr_arr[sig_bar]
        if np.isnan(cur_atr) or cur_atr <= 0:
            cur_atr = 30.0

        # Compute entry, stop, TP
        stop_dist = stop_atr_mults[oi] * cur_atr
        tp_dist = tp_atr_mults[oi] * cur_atr

        if stop_dist < 1.0 or tp_dist < 0.5:
            continue

        if direction == 1:
            actual_entry = o[entry_idx] + slippage_points
            stop_price = actual_entry - stop_dist
            tp1_price = actual_entry + tp_dist
        else:
            actual_entry = o[entry_idx] - slippage_points
            stop_price = actual_entry + stop_dist
            tp1_price = actual_entry - tp_dist

        # Position sizing (simplified: use 1 contract for R normalization)
        contracts = 1
        cur_trim_pct = use_trim_pcts[oi]

        # Bar-by-bar trade management
        pos_trimmed = False
        pos_be_stop = 0.0
        pos_trail_stop = 0.0
        pos_remaining = contracts
        exited = False
        exit_price = 0.0
        exit_reason = ""
        exit_contracts = contracts

        max_bar = min(entry_idx + max_bars_per_trade, n)

        for i in range(entry_idx, max_bar):
            # Early cut PA check (bars 2-4 after entry)
            bars_in_trade = i - entry_idx
            if not pos_trimmed and 2 <= bars_in_trade <= 4:
                pa_start = max(entry_idx, 0)
                pa_end = i + 1
                pa_range = h[pa_start:pa_end] - l[pa_start:pa_end]
                pa_body = np.abs(c[pa_start:pa_end] - o[pa_start:pa_end])
                safe_pa = np.where(pa_range > 0, pa_range, 1.0)
                pa_wick = 1.0 - (pa_body / safe_pa)
                avg_wick = float(np.mean(pa_wick))
                pa_dirs = np.sign(c[pa_start:pa_end] - o[pa_start:pa_end])
                favorable = (pa_dirs == direction).mean()
                disp = (c[i] - actual_entry) if direction == 1 else (actual_entry - c[i])
                ia = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                no_progress = disp < ia * 0.3
                bad_pa = avg_wick > 0.65 and favorable < 0.5
                if bad_pa and no_progress and bars_in_trade >= 3:
                    exit_price = o[i+1] if i+1 < n else c[i]
                    exit_reason = "early_cut_pa"
                    exited = True
                    break

            if direction == 1:  # LONG
                eff_stop = pos_trail_stop if pos_trimmed and pos_trail_stop > 0 else stop_price
                if pos_trimmed and be_after_trim and pos_be_stop > 0:
                    eff_stop = max(eff_stop, pos_be_stop)

                if l[i] <= eff_stop:
                    exit_price = eff_stop - slippage_points
                    exit_reason = "be_sweep" if (pos_trimmed and eff_stop >= actual_entry) else "stop"
                    exited = True
                    break

                if not pos_trimmed and h[i] >= tp1_price:
                    tc = max(1, int(contracts * cur_trim_pct))
                    pos_remaining = contracts - tc
                    pos_trimmed = True
                    pos_be_stop = actual_entry
                    if pos_remaining > 0:
                        pos_trail_stop = _find_nth_swing(swing_low_mask, nq["low"].values, i, nth_swing, 1)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop
                    if pos_remaining <= 0:
                        exit_price = tp1_price
                        exit_reason = "tp1"
                        exit_contracts = contracts
                        exited = True
                        break

                if pos_trimmed and not exited:
                    nt = _find_nth_swing(swing_low_mask, nq["low"].values, i, nth_swing, 1)
                    if not np.isnan(nt) and nt > pos_trail_stop:
                        pos_trail_stop = nt

            else:  # SHORT
                eff_stop = pos_trail_stop if pos_trimmed and pos_trail_stop > 0 else stop_price
                if pos_trimmed and be_after_trim and pos_be_stop > 0:
                    eff_stop = min(eff_stop, pos_be_stop)

                if h[i] >= eff_stop:
                    exit_price = eff_stop + slippage_points
                    exit_reason = "be_sweep" if (pos_trimmed and eff_stop <= actual_entry) else "stop"
                    exited = True
                    break

                if not pos_trimmed and l[i] <= tp1_price:
                    tc = max(1, int(contracts * cur_trim_pct))
                    pos_remaining = contracts - tc
                    pos_trimmed = True
                    pos_be_stop = actual_entry
                    if pos_remaining > 0:
                        pos_trail_stop = _find_nth_swing(swing_high_mask, nq["high"].values, i, nth_swing, -1)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop
                    if pos_remaining <= 0:
                        exit_price = tp1_price
                        exit_reason = "tp1"
                        exit_contracts = contracts
                        exited = True
                        break

                if pos_trimmed:
                    nt = _find_nth_swing(swing_high_mask, nq["high"].values, i, nth_swing, -1)
                    if not np.isnan(nt) and nt < pos_trail_stop:
                        pos_trail_stop = nt

        if not exited:
            # Force close at last bar in window
            exit_bar = min(entry_idx + max_bars_per_trade - 1, n - 1)
            exit_price = c[exit_bar]
            exit_reason = "timeout"
            exited = True

        # Compute R-multiple (same logic as engine.py)
        if direction == 1:
            pnl_pts = exit_price - actual_entry
        else:
            pnl_pts = actual_entry - exit_price

        if pos_trimmed and exit_reason != "tp1":
            trim_pnl_total = (tp1_price - actual_entry) if direction == 1 else (actual_entry - tp1_price)
            trim_c = contracts - exit_contracts
            # exit_contracts is pos_remaining for trimmed trades
            exit_contracts = pos_remaining
            total_pnl = trim_pnl_total * point_value * (contracts - exit_contracts) + pnl_pts * point_value * exit_contracts
            total_comm = commission_per_side * 2 * contracts
            total_pnl -= total_comm
        else:
            total_pnl = pnl_pts * point_value * exit_contracts
            total_comm = commission_per_side * 2 * exit_contracts
            total_pnl -= total_comm

        total_risk = stop_dist * point_value * contracts
        r_mult = total_pnl / total_risk if total_risk > 0 else 0.0

        trades_r.append(r_mult)

        # Update daily state
        daily_pnl_r += r_mult
        if exit_reason == "be_sweep" and pos_trimmed:
            pass
        elif r_mult < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0

        if consecutive_losses >= max_consec_losses:
            day_stopped = True
        if daily_pnl_r <= -daily_max_loss_r:
            day_stopped = True

        # Update next_available_bar: find exit bar
        exit_bar_idx = entry_idx
        for check in range(entry_idx, min(entry_idx + max_bars_per_trade, n)):
            exit_bar_idx = check
            if exited:
                break
        next_available_bar = exit_bar_idx + 1

    trades_r = np.array(trades_r) if trades_r else np.array([0.0])

    # Compute max drawdown in R
    cum_r = np.cumsum(trades_r)
    running_max = np.maximum.accumulate(cum_r)
    drawdowns = running_max - cum_r
    max_dd = drawdowns.max() if len(drawdowns) > 0 else 0.0

    return {
        "total_r": float(trades_r.sum()),
        "trade_count": len(trades_r),
        "win_count": int((trades_r > 0).sum()),
        "win_rate": float((trades_r > 0).mean()) if len(trades_r) > 0 else 0.0,
        "avg_r": float(trades_r.mean()) if len(trades_r) > 0 else 0.0,
        "max_dd": float(max_dd),
        "ppdd": float(trades_r.sum() / max_dd) if max_dd > 0 else 0.0,
        "profit_factor": float(
            trades_r[trades_r > 0].sum() / abs(trades_r[trades_r < 0].sum())
            if (trades_r < 0).any() and trades_r[trades_r > 0].sum() > 0 else 0.0
        ),
    }


# ============================================================
# 5. RUN RANDOM BENCHMARK (1000 iterations)
# ============================================================
N_RANDOM = 1000
TRADE_COUNTS_TO_TEST = [len(real_trades)]  # Match real strategy count
EXTRA_COUNTS = [250, 750, 1000, 1500, 2000]  # Also test different counts

print(f"\n[4/6] Running {N_RANDOM} random strategies with {len(real_trades)} trades each...")
print(f"  (same trade management: trim {trim_pct*100:.0f}%, BE after trim, "
      f"trail {nth_swing}nd swing, 0-for-2, 2R daily limit)")
t1 = time.time()

rng = np.random.default_rng(42)  # Reproducible

random_results = []
for iteration in range(N_RANDOM):
    if (iteration + 1) % 100 == 0:
        elapsed = time.time() - t1
        eta = elapsed / (iteration + 1) * (N_RANDOM - iteration - 1)
        print(f"  Iteration {iteration+1}/{N_RANDOM} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    num_trades = len(real_trades)

    # Sample random entry bars from eligible pool
    if num_trades > len(eligible_indices):
        # More trades than eligible bars — sample with replacement
        sampled = rng.choice(eligible_indices, size=num_trades, replace=True)
    else:
        sampled = rng.choice(eligible_indices, size=num_trades, replace=False)

    # Random directions matching real strategy's long/short ratio
    dirs = rng.choice([1, -1], size=num_trades,
                      p=[real_long_ratio, real_short_ratio])

    # Sample stop distances from real strategy's distribution
    stop_mults = rng.choice(real_stop_atr_mults, size=num_trades, replace=True)

    # Sample TP distances from real strategy's distribution
    tp_mults = rng.choice(real_tp_atr_mults, size=num_trades, replace=True)

    # Use same trim pct distribution as real strategy
    # Real strategy: shorts use 1.0 trim, longs use 0.5
    trim_pcts = np.where(dirs == -1, 1.0, trim_pct)

    result = run_trades(sampled, dirs, stop_mults, tp_mults, trim_pcts)
    random_results.append(result)

elapsed_random = time.time() - t1
print(f"  Completed {N_RANDOM} iterations in {elapsed_random:.1f}s "
      f"({elapsed_random/N_RANDOM*1000:.0f}ms/iteration)")

# ============================================================
# 6. STATISTICAL ANALYSIS
# ============================================================
print("\n[5/6] Statistical analysis...")

random_total_r = np.array([r["total_r"] for r in random_results])
random_avg_r = np.array([r["avg_r"] for r in random_results])
random_win_rates = np.array([r["win_rate"] for r in random_results])
random_max_dd = np.array([r["max_dd"] for r in random_results])
random_ppdd = np.array([r["ppdd"] for r in random_results])
random_pf = np.array([r["profit_factor"] for r in random_results])
random_trade_counts = np.array([r["trade_count"] for r in random_results])

# Real strategy metrics
real_total_r = real_trades["r"].sum()
real_avg_r = real_trades["r"].mean()
real_win_rate = (real_trades["r"] > 0).mean()
cum_r = np.cumsum(real_trades["r"].values)
running_max = np.maximum.accumulate(cum_r)
real_max_dd = (running_max - cum_r).max()
real_ppdd = real_total_r / real_max_dd if real_max_dd > 0 else 0.0
wins = real_trades["r"][real_trades["r"] > 0].sum()
losses = abs(real_trades["r"][real_trades["r"] < 0].sum())
real_pf = wins / losses if losses > 0 else 0.0

# Percentile rank
pctile_total_r = (random_total_r < real_total_r).mean() * 100
pctile_avg_r = (random_avg_r < real_avg_r).mean() * 100
pctile_wr = (random_win_rates < real_win_rate).mean() * 100
pctile_ppdd = (random_ppdd < real_ppdd).mean() * 100
pctile_pf = (random_pf < real_pf).mean() * 100

# p-value: probability that random >= real
p_value_total_r = (random_total_r >= real_total_r).mean()
p_value_avg_r = (random_avg_r >= real_avg_r).mean()

print()
print("=" * 70)
print("RANDOM BENCHMARK RESULTS")
print("=" * 70)

print(f"\n{'Metric':<22} {'Real Strategy':>15} {'Random Mean':>15} {'Random Std':>12} {'Percentile':>12} {'p-value':>10}")
print("-" * 86)
print(f"{'Total R':<22} {real_total_r:>15.2f} {random_total_r.mean():>15.2f} {random_total_r.std():>12.2f} {pctile_total_r:>11.1f}% {p_value_total_r:>10.4f}")
print(f"{'Avg R/trade':<22} {real_avg_r:>15.4f} {random_avg_r.mean():>15.4f} {random_avg_r.std():>12.4f} {pctile_avg_r:>11.1f}% {p_value_avg_r:>10.4f}")
print(f"{'Win Rate':<22} {real_win_rate:>15.1%} {random_win_rates.mean():>15.1%} {random_win_rates.std():>12.1%} {pctile_wr:>11.1f}%")
print(f"{'Max Drawdown (R)':<22} {real_max_dd:>15.2f} {random_max_dd.mean():>15.2f} {random_max_dd.std():>12.2f}")
print(f"{'PPDD':<22} {real_ppdd:>15.2f} {random_ppdd.mean():>15.2f} {random_ppdd.std():>12.2f} {pctile_ppdd:>11.1f}%")
print(f"{'Profit Factor':<22} {real_pf:>15.2f} {random_pf.mean():>15.2f} {random_pf.std():>12.2f} {pctile_pf:>11.1f}%")
print(f"{'Trade Count (actual)':<22} {len(real_trades):>15d} {random_trade_counts.mean():>15.0f} {random_trade_counts.std():>12.0f}")

print(f"\nRandom Total R distribution:")
print(f"  Min:    {random_total_r.min():>10.2f}")
print(f"  P5:     {np.percentile(random_total_r, 5):>10.2f}")
print(f"  P25:    {np.percentile(random_total_r, 25):>10.2f}")
print(f"  Median: {np.percentile(random_total_r, 50):>10.2f}")
print(f"  P75:    {np.percentile(random_total_r, 75):>10.2f}")
print(f"  P95:    {np.percentile(random_total_r, 95):>10.2f}")
print(f"  Max:    {random_total_r.max():>10.2f}")

# How many random strategies were profitable?
profitable_pct = (random_total_r > 0).mean() * 100
print(f"\n  Random strategies that were profitable: {profitable_pct:.1f}%")
print(f"  Random strategies that beat real:       {p_value_total_r*100:.1f}%")

# ============================================================
# 7. VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

if p_value_total_r < 0.01:
    verdict = "STRONG EDGE IN SIGNAL"
    detail = (f"The FVG signal detection contributes significant alpha beyond trade management.\n"
              f"  Only {p_value_total_r*100:.1f}% of random strategies matched or beat +{real_total_r:.1f}R.\n"
              f"  This is statistically significant at the 1% level (p < 0.01).")
elif p_value_total_r < 0.05:
    verdict = "MODERATE EDGE IN SIGNAL"
    detail = (f"The FVG signal detection likely contributes alpha, but evidence is moderate.\n"
              f"  {p_value_total_r*100:.1f}% of random strategies matched or beat +{real_total_r:.1f}R.\n"
              f"  Significant at the 5% level but not at 1%.")
elif p_value_total_r < 0.20:
    verdict = "WEAK EDGE IN SIGNAL"
    detail = (f"The signal detection shows some edge, but trade management contributes substantially.\n"
              f"  {p_value_total_r*100:.1f}% of random strategies matched or beat +{real_total_r:.1f}R.")
else:
    verdict = "EDGE IS PRIMARILY IN TRADE MANAGEMENT"
    detail = (f"Random entries with the same management produce comparable results.\n"
              f"  {p_value_total_r*100:.1f}% of random strategies matched or beat +{real_total_r:.1f}R.\n"
              f"  The FVG signal detection does NOT provide statistically significant alpha.")

print(f"\n  {verdict}")
print(f"\n  {detail}")

# ============================================================
# 8. ANTI-SIGNAL TEST
# ============================================================
print("\n" + "=" * 70)
print("ANTI-SIGNAL TEST (invert all real trade directions)")
print("=" * 70)

# Use the exact same entry bars and parameters as real trades, but flip direction
anti_entry_bars = real_entry_indices.copy()
anti_dirs = -real_trades["dir"].values  # Invert directions
anti_stop_mults = real_stop_atr_mults.copy()
anti_tp_mults = real_tp_atr_mults.copy()
# Anti-signal: flip trim pcts too (longs become shorts and vice versa)
anti_trim_pcts = np.where(anti_dirs == -1, 1.0, trim_pct)

anti_result = run_trades(anti_entry_bars, anti_dirs, anti_stop_mults, anti_tp_mults, anti_trim_pcts)

print(f"\n  Real strategy:   {len(real_trades)} trades, R={real_total_r:>+8.2f}, "
      f"WR={real_win_rate:.1%}, PPDD={real_ppdd:.2f}")
print(f"  Anti-signal:     {anti_result['trade_count']} trades, R={anti_result['total_r']:>+8.2f}, "
      f"WR={anti_result['win_rate']:.1%}, PPDD={anti_result['ppdd']:.2f}")
print(f"  Spread (real - anti): {real_total_r - anti_result['total_r']:>+.2f}R")

if anti_result["total_r"] < 0:
    print(f"\n  Anti-signal LOSES money -> The DIRECTIONAL edge is real.")
    print(f"  The strategy knows WHEN to go long vs short.")
elif anti_result["total_r"] > 0 and anti_result["total_r"] < real_total_r * 0.5:
    print(f"\n  Anti-signal is profitable but much weaker -> Partial directional edge.")
    print(f"  Some alpha from direction, some from management.")
else:
    print(f"\n  Anti-signal performs similarly -> Edge is NOT directional.")
    print(f"  Trade management drives returns regardless of direction.")

# ============================================================
# 9. TRADE COUNT SENSITIVITY
# ============================================================
print("\n" + "=" * 70)
print("TRADE COUNT SENSITIVITY (does more random trades = more R?)")
print("=" * 70)

N_SENSITIVITY = 200  # fewer iterations for speed
t2 = time.time()

sensitivity_results = {}
for count in EXTRA_COUNTS:
    results_at_count = []
    for it in range(N_SENSITIVITY):
        if count > len(eligible_indices):
            sampled = rng.choice(eligible_indices, size=count, replace=True)
        else:
            sampled = rng.choice(eligible_indices, size=count, replace=False)

        dirs = rng.choice([1, -1], size=count, p=[real_long_ratio, real_short_ratio])
        stop_mults = rng.choice(real_stop_atr_mults, size=count, replace=True)
        tp_mults = rng.choice(real_tp_atr_mults, size=count, replace=True)
        trim_pcts = np.where(dirs == -1, 1.0, trim_pct)

        result = run_trades(sampled, dirs, stop_mults, tp_mults, trim_pcts)
        results_at_count.append(result)

    total_rs = [r["total_r"] for r in results_at_count]
    avg_rs = [r["avg_r"] for r in results_at_count]
    actual_counts = [r["trade_count"] for r in results_at_count]
    sensitivity_results[count] = {
        "mean_total_r": np.mean(total_rs),
        "std_total_r": np.std(total_rs),
        "mean_avg_r": np.mean(avg_rs),
        "mean_actual_count": np.mean(actual_counts),
        "profitable_pct": 100 * np.mean([t > 0 for t in total_rs]),
    }

print(f"\n{'Target Trades':>14} {'Actual Trades':>14} {'Mean Total R':>14} {'Std R':>10} {'Mean R/trade':>14} {'Profitable%':>12}")
print("-" * 78)
# Add the main N_RANDOM result for the real trade count
print(f"{len(real_trades):>14d} {random_trade_counts.mean():>14.0f} {random_total_r.mean():>14.2f} {random_total_r.std():>10.2f} {random_avg_r.mean():>14.4f} {profitable_pct:>11.1f}%")
for count in EXTRA_COUNTS:
    sr = sensitivity_results[count]
    print(f"{count:>14d} {sr['mean_actual_count']:>14.0f} {sr['mean_total_r']:>14.2f} {sr['std_total_r']:>10.2f} {sr['mean_avg_r']:>14.4f} {sr['profitable_pct']:>11.1f}%")

elapsed_sens = time.time() - t2
print(f"\n  Sensitivity test: {elapsed_sens:.1f}s ({N_SENSITIVITY} iterations x {len(EXTRA_COUNTS)} counts)")

# Check if more trades = more R (inherent positive expectancy in management)
mean_rs_by_count = [(len(real_trades), random_total_r.mean())]
for count in EXTRA_COUNTS:
    mean_rs_by_count.append((count, sensitivity_results[count]["mean_total_r"]))
mean_rs_by_count.sort(key=lambda x: x[0])

print(f"\n  Trend analysis:")
increasing = all(mean_rs_by_count[i][1] <= mean_rs_by_count[i+1][1]
                 for i in range(len(mean_rs_by_count)-1))
if all(sr["mean_total_r"] > 0 for sr in sensitivity_results.values()) and random_total_r.mean() > 0:
    print(f"  -> Random entries ARE inherently profitable with this trade management.")
    print(f"  -> The management has built-in positive expectancy (trim/trail/BE).")
    if increasing:
        print(f"  -> More trades = more R (scales linearly) -> management edge confirmed.")
elif all(sr["mean_total_r"] <= 0 for sr in sensitivity_results.values()) and random_total_r.mean() <= 0:
    print(f"  -> Random entries are NOT profitable -> signal detection is critical.")
else:
    print(f"  -> Mixed results across trade counts -> both signal and management contribute.")

# ============================================================
# 10. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("EXECUTIVE SUMMARY")
print("=" * 70)
total_time = time.time() - t0
print(f"""
  Real FVG Strategy:       {len(real_trades)} trades, {real_total_r:>+.2f}R, avg={real_avg_r:>+.4f}R/trade
  Random Mean ({N_RANDOM}x):     {random_trade_counts.mean():.0f} trades, {random_total_r.mean():>+.2f}R, avg={random_avg_r.mean():>+.4f}R/trade
  Anti-Signal:             {anti_result['trade_count']} trades, {anti_result['total_r']:>+.2f}R, avg={anti_result['avg_r']:>+.4f}R/trade

  Signal Percentile:       {pctile_total_r:.1f}th (p={p_value_total_r:.4f})
  Signal - Random Spread:  {real_total_r - random_total_r.mean():>+.2f}R
  Real - Anti Spread:      {real_total_r - anti_result['total_r']:>+.2f}R

  Management profitable?   {'YES' if random_total_r.mean() > 0 else 'NO'} (random mean R = {random_total_r.mean():>+.2f})
  Signal has edge?          {'YES' if p_value_total_r < 0.05 else 'UNCLEAR' if p_value_total_r < 0.20 else 'NO'} (p = {p_value_total_r:.4f})
  Direction matters?        {'YES' if anti_result['total_r'] < real_total_r * 0.3 else 'PARTIAL' if anti_result['total_r'] < real_total_r else 'NO'}

  Total runtime: {total_time:.1f}s
""")
print("=" * 70)
