"""
1m Signal Detection Backtest — Optimized version.

Same signal logic as entry_signals.py but with FVG max-age pruning
to keep the active pool small on 1m data (~350K bars/year).

Key difference from 5m baseline:
  - Signals detected on 1m → earlier, more precise entries
  - Stop = candle2 open from 1m FVG (tighter)
  - IRL target = nearest 5m swing high (HTF target)
  - Simulation on 1m bars (300 bar max hold = 5 hours)
"""
import sys; sys.path.insert(0, '.')
import logging; logging.basicConfig(level=logging.WARNING)
import pandas as pd, numpy as np, yaml, time

# Max age for active FVGs in bars — 500 1m bars = ~8 hours.
# Any FVG older than this is pruned. On 5m this was never needed
# because pool stayed small, but on 1m the pool explodes.
FVG_MAX_AGE = 500

params = yaml.safe_load(open('config/params.yaml'))

# ---- Load data ----
print("Loading data...")
t_load = time.time()
df_1m = pd.read_parquet('data/NQ_1min_10yr.parquet')
df_5m = pd.read_parquet('data/NQ_5m_10yr.parquet')
df_1h = pd.read_parquet('data/NQ_1H_10yr.parquet')
df_4h = pd.read_parquet('data/NQ_4H_10yr.parquet')
print(f"Loaded in {time.time()-t_load:.0f}s")
print(f"1m: {len(df_1m):,} bars")
print(f"5m: {len(df_5m):,} bars")

# ---- Compute bias on 5m (reuse existing code) ----
from features.sessions import label_sessions, compute_session_levels, compute_orm
from features.bias import compute_daily_bias, compute_regime
from features.displacement import compute_atr, compute_fluency, detect_displacement
from features.swing import compute_swing_levels
from features.fvg import detect_fvg

print("Computing bias on 5m...")
t0 = time.time()
sessions_5m = label_sessions(df_5m, params)
session_levels_5m = compute_session_levels(df_5m, params)
orm_5m = compute_orm(df_5m, params)
bias_5m = compute_daily_bias(df_5m, session_levels_5m, orm_5m, df_4h, df_1h, params)
bias_dir_5m = bias_5m['bias_direction']
regime_5m = compute_regime(df_5m, df_4h, bias_5m, params)
print(f"Bias computed in {time.time()-t0:.0f}s")

# Compute 5m swings for IRL targets (shift to prevent lookahead)
swings_5m = compute_swing_levels(df_5m, params['swing'])
sw_high_5m = swings_5m['swing_high_price'].shift(1).ffill()
sw_low_5m = swings_5m['swing_low_price'].shift(1).ffill()

# ---- Entry params ----
entry_cfg = params.get('entry', {})
min_fvg_atr_mult = entry_cfg.get('min_fvg_atr_mult', 0.5)
rejection_body_ratio = entry_cfg.get('rejection_body_ratio', 0.55)
signal_cooldown_bars = entry_cfg.get('signal_cooldown_bars', 10)
fluency_thresh = params['fluency']['threshold']
sweep_lookback = entry_cfg.get('sweep_lookback', 20)
small_candle_mult = params['stop_loss']['small_candle_atr_mult']

# ---- Process year by year ----
print("Detecting signals on 1m data (year by year)...")

et_1m = df_1m.index.tz_convert('US/Eastern')
years = sorted(et_1m.year.unique())

all_trades = []

for year in years:
    t1 = time.time()
    year_mask = et_1m.year == year
    df_year = df_1m[year_mask]
    n_y = len(df_year)
    if n_y < 1000:
        continue

    # ---- Vectorized feature computation (fast) ----
    t_feat = time.time()
    fvg_df = detect_fvg(df_year)
    atr_s = compute_atr(df_year, period=14)
    fluency_s = compute_fluency(df_year, params)
    disp_s = detect_displacement(df_year, params)

    # Numpy arrays for fast access
    open_y = df_year['open'].values
    high_y = df_year['high'].values
    low_y = df_year['low'].values
    close_y = df_year['close'].values
    atr_y = atr_s.values
    fluency_y = fluency_s.values
    disp_y = disp_s.values

    bull_mask = fvg_df['fvg_bull'].values
    bear_mask = fvg_df['fvg_bear'].values
    fvg_bull_top = fvg_df['fvg_bull_top'].values
    fvg_bull_bot = fvg_df['fvg_bull_bottom'].values
    fvg_bear_top = fvg_df['fvg_bear_top'].values
    fvg_bear_bot = fvg_df['fvg_bear_bottom'].values
    fvg_size_arr = fvg_df['fvg_size'].values

    body_y = np.abs(close_y - open_y)
    range_y = high_y - low_y
    safe_range_y = np.where(range_y == 0, np.nan, range_y)
    body_ratio_y = body_y / safe_range_y

    # Swing levels for stop fallback (shift by 1 for lookahead prevention)
    swing_df_y = compute_swing_levels(df_year, params['swing'])
    sw_high_y_local = swing_df_y['swing_high_price'].shift(1).ffill().values
    sw_low_y_local = swing_df_y['swing_low_price'].shift(1).ffill().values

    # Sweep detection for MSS
    swept_low_y = np.zeros(n_y, dtype=bool)
    swept_high_y = np.zeros(n_y, dtype=bool)
    for i in range(1, n_y):
        if not np.isnan(sw_low_y_local[i-1]) and low_y[i] < sw_low_y_local[i-1]:
            swept_low_y[i] = True
        if not np.isnan(sw_high_y_local[i-1]) and high_y[i] > sw_high_y_local[i-1]:
            swept_high_y[i] = True

    # Align bias & 5m swing targets to 1m
    bias_y = bias_dir_5m.reindex(df_year.index, method='ffill').values
    sw_high_5m_y = sw_high_5m.reindex(df_year.index, method='ffill').values

    et_y = df_year.index.tz_convert('US/Eastern')

    feat_time = time.time() - t_feat

    # ---- Signal detection loop with FVG max-age pruning ----
    t_loop = time.time()

    # FVG pool: list of (idx, direction, top, bottom, size, candle2_open, status, is_ifvg, ifvg_dir, last_signal_idx)
    # Using tuples for speed, indices: 0=idx, 1=dir, 2=top, 3=bot, 4=size, 5=c2open, 6=status, 7=is_ifvg, 8=ifvg_dir, 9=last_sig
    # status: 0=untested, 1=tested_rejected, 2=invalidated
    active_fvgs = []  # for trend model
    active_ifvgs = []  # for MSS model

    sig_bars = []  # (bar_idx, direction, signal_type, fvg_c2_open)

    n_signals = 0

    for i in range(n_y):
        # ---- Birth: register new FVGs ----
        c2_open_val = open_y[i-1] if i > 0 else open_y[i]

        if bull_mask[i]:
            sz = fvg_size_arr[i]
            if not np.isnan(sz) and sz > 0:
                active_fvgs.append([i, 1, fvg_bull_top[i], fvg_bull_bot[i], sz, c2_open_val, 0, False, 0, -999])

        if bear_mask[i]:
            sz = fvg_size_arr[i]
            if not np.isnan(sz) and sz > 0:
                active_fvgs.append([i, -1, fvg_bear_top[i], fvg_bear_bot[i], sz, c2_open_val, 0, False, 0, -999])

        # ---- Quality pre-checks ----
        cur_br = body_ratio_y[i] if not np.isnan(body_ratio_y[i]) else 0.0
        cur_fluency = fluency_y[i] if not np.isnan(fluency_y[i]) else 0.0
        cur_atr = atr_y[i] if not np.isnan(atr_y[i]) else 0.0
        cur_disp = bool(disp_y[i])

        if cur_br >= rejection_body_ratio and cur_fluency >= fluency_thresh:
            # ---- Trend model: check active FVGs ----
            best_dir = 0; best_score = -1.0; best_rec = None
            for rec in active_fvgs:
                if rec[6] == 2: continue  # invalidated
                if rec[7]: continue  # skip IFVGs for trend model
                if rec[0] >= i: continue  # must age 1 bar
                if cur_atr > 0 and rec[4] < min_fvg_atr_mult * cur_atr: continue
                if (i - rec[9]) < signal_cooldown_bars: continue

                d = rec[1]
                top, bot = rec[2], rec[3]

                if d == 1:  # bull FVG
                    entered = low_y[i] <= top and high_y[i] >= bot
                    rejected = close_y[i] > top
                    if entered and rejected:
                        score = rec[4] + (100.0 if cur_disp else 0.0)
                        if score > best_score:
                            best_score = score; best_dir = 1; best_rec = rec
                elif d == -1:  # bear FVG
                    entered = high_y[i] >= bot and low_y[i] <= top
                    rejected = close_y[i] < bot
                    if entered and rejected:
                        score = rec[4] + (100.0 if cur_disp else 0.0)
                        if score > best_score:
                            best_score = score; best_dir = -1; best_rec = rec

            if best_rec is not None:
                best_rec[9] = i  # update last_signal_idx
                sig_bars.append((i, best_dir, 'trend', best_rec[5]))
                n_signals += 1

            # ---- MSS model: check active IFVGs ----
            best_dir_m = 0; best_score_m = -1.0; best_ifvg = None
            for ifvg in active_ifvgs:
                if ifvg[6] == 2: continue
                if ifvg[0] >= i: continue
                if cur_atr > 0 and ifvg[4] < min_fvg_atr_mult * cur_atr: continue
                if (i - ifvg[9]) < signal_cooldown_bars: continue

                ifvg_d = ifvg[8]  # ifvg direction
                top, bot = ifvg[2], ifvg[3]

                if ifvg_d == 1:
                    sweep_start = max(0, ifvg[0] - sweep_lookback)
                    if not np.any(swept_low_y[sweep_start:ifvg[0]+1]): continue
                    entered = low_y[i] <= top and high_y[i] >= bot
                    respected = close_y[i] > top
                    if entered and respected:
                        score = ifvg[4] + (100.0 if cur_disp else 0.0)
                        if score > best_score_m:
                            best_score_m = score; best_dir_m = 1; best_ifvg = ifvg
                elif ifvg_d == -1:
                    sweep_start = max(0, ifvg[0] - sweep_lookback)
                    if not np.any(swept_high_y[sweep_start:ifvg[0]+1]): continue
                    entered = high_y[i] >= bot and low_y[i] <= top
                    respected = close_y[i] < bot
                    if entered and respected:
                        score = ifvg[4] + (100.0 if cur_disp else 0.0)
                        if score > best_score_m:
                            best_score_m = score; best_dir_m = -1; best_ifvg = ifvg

            if best_ifvg is not None and best_rec is None:  # MSS only if no trend signal
                best_ifvg[9] = i
                sig_bars.append((i, best_dir_m, 'mss', best_ifvg[5]))
                n_signals += 1

        # ---- Update FVG states ----
        new_active = []
        new_ifvgs_born = []
        for rec in active_fvgs:
            if rec[6] == 2:
                continue  # already invalidated, drop
            # Age pruning
            if (i - rec[0]) > FVG_MAX_AGE:
                continue  # too old, prune

            d = rec[1]
            top, bot = rec[2], rec[3]
            old_status = rec[6]
            is_ifvg = rec[7]

            # Determine effective direction
            eff_d = rec[8] if is_ifvg else d

            if eff_d == 1:  # bull
                entered = low_y[i] <= top and high_y[i] >= bot
                if entered:
                    if close_y[i] < bot:
                        new_status = 2  # invalidated
                    elif close_y[i] >= top:
                        new_status = 1 if old_status == 0 else old_status
                    else:
                        new_status = 1 if old_status == 0 else old_status
                elif close_y[i] < bot:
                    new_status = 2
                else:
                    new_status = old_status
            else:  # bear
                entered = high_y[i] >= bot and low_y[i] <= top
                if entered:
                    if close_y[i] > top:
                        new_status = 2
                    elif close_y[i] <= bot:
                        new_status = 1 if old_status == 0 else old_status
                    else:
                        new_status = 1 if old_status == 0 else old_status
                elif close_y[i] > top:
                    new_status = 2
                else:
                    new_status = old_status

            if new_status == 2 and old_status != 2:
                # Invalidated → spawn IFVG
                if not is_ifvg:
                    ifvg_d = -1 if d == 1 else 1
                    new_ifvg = [i, d, top, bot, rec[4], close_y[i], 0, True, ifvg_d, -999]
                    new_ifvgs_born.append(new_ifvg)
                    active_ifvgs.append(new_ifvg)
                rec[6] = 2
                # Don't add to new_active
            else:
                rec[6] = new_status
                new_active.append(rec)

        active_fvgs = new_active + new_ifvgs_born

        # Update IFVG states
        surviving_ifvgs = []
        for ifvg in active_ifvgs:
            if ifvg[6] == 2: continue
            if (i - ifvg[0]) > FVG_MAX_AGE: continue

            ifvg_d = ifvg[8]
            top, bot = ifvg[2], ifvg[3]

            if ifvg_d == 1:
                if close_y[i] < bot:
                    ifvg[6] = 2; continue
            else:
                if close_y[i] > top:
                    ifvg[6] = 2; continue
            surviving_ifvgs.append(ifvg)
        active_ifvgs = surviving_ifvgs

    loop_time = time.time() - t_loop

    # ---- Convert signals to trades ----
    year_trades = 0
    for (idx, sig_d, sig_type, c2_open) in sig_bars:
        if sig_d != 1: continue  # long only for now

        # Bias filter
        b = bias_y[idx]
        if np.isnan(b) or sig_d != np.sign(b) or b == 0: continue

        # Entry price = next bar open
        if idx + 1 >= n_y: continue
        ep = open_y[idx + 1]

        # Stop = candle2 open from 1m FVG
        stop = c2_open
        if np.isnan(ep) or np.isnan(stop): continue
        sd = abs(ep - stop)
        cur_atr = atr_y[idx]
        if cur_atr <= 0 or sd < 1.0 * cur_atr or sd > 8.0 * cur_atr: continue
        if stop >= ep: continue

        # IRL target = 5m swing high
        tp1 = sw_high_5m_y[idx]
        if np.isnan(tp1) or tp1 <= ep:
            tp1 = ep + sd * 2  # fallback: 2R

        rr = abs(tp1 - ep) / sd
        if rr >= 3.0: continue

        # Session filter
        hour = et_y[idx].hour
        minute = et_y[idx].minute
        if not (3 <= hour < 16): continue
        if (hour == 9 and minute >= 30) or (hour == 10 and minute == 0): continue

        # Simulate on 1m (300 bars max hold)
        contracts = max(1, int(500 / (sd * 2)))
        trimmed = False; trim_pnl = 0; stop_live = stop
        rpnl = 0
        for j in range(idx + 2, min(idx + 302, n_y)):
            if low_y[j] <= stop_live:
                rpnl = (stop_live - ep) * 2 * contracts; break
            if not trimmed and high_y[j] >= tp1:
                tc = contracts // 2; trim_pnl = (tp1 - ep) * 2 * tc
                contracts -= tc; trimmed = True; stop_live = ep
        else:
            rpnl = (close_y[min(idx + 301, n_y - 1)] - ep) * 2 * contracts

        total_pnl = trim_pnl + rpnl
        all_trades.append({
            'pnl': total_pnl,
            'win': 1 if total_pnl > 0 else 0,
            'year': year,
            'sd': sd,
            'rr': rr,
            'type': sig_type,
        })
        year_trades += 1

    elapsed = time.time() - t1
    print(f"  {year}: {n_signals} signals, {year_trades} trades "
          f"(feat={feat_time:.0f}s, loop={loop_time:.0f}s, total={elapsed:.0f}s)")

# ---- Results ----
tdf = pd.DataFrame(all_trades)
if len(tdf) == 0:
    print("No trades!")
    sys.exit()

eq = tdf['pnl'].cumsum()
dd = (eq - eq.cummax()).min()
wr = tdf['win'].mean() * 100
n_years = 10.3
tpd = len(tdf) / (252 * n_years)

print(f"\n{'='*70}")
print(f"1M SIGNAL DETECTION RESULTS (10yr)")
print(f"{'='*70}")
print(f"Trades: {len(tdf)} ({tpd:.2f}/day)")
print(f"WR: {wr:.1f}%")
print(f"PnL: ${tdf['pnl'].sum():,.0f}")
print(f"Avg: ${tdf['pnl'].mean():.0f}")
print(f"DD: ${dd:,.0f}")
print(f"PPDD: {tdf['pnl'].sum()/abs(dd):.2f}")

# Compare with 5m baseline
print(f"\nFor comparison — 5m signal detection:")
print(f"  1,975 trades (0.76/day), 53.3% WR, $85,246, PPDD=6.63")

# Yearly breakdown
print(f"\n{'Year':>6} {'N':>5} {'WR':>5} {'PnL':>10} {'Avg':>6}")
for y in sorted(tdf['year'].unique()):
    sub = tdf[tdf['year']==y]
    print(f"  {y:>4} {len(sub):>5} {100*sub['win'].mean():>4.0f}% ${sub['pnl'].sum():>9,.0f} ${sub['pnl'].mean():>5.0f}")

# By signal type
print(f"\nBy type:")
for t in sorted(tdf['type'].unique()):
    sub = tdf[tdf['type']==t]
    print(f"  {t}: {len(sub)} trades, WR={100*sub['win'].mean():.0f}%, avg=${sub['pnl'].mean():.0f}")

# Stop distance stats
print(f"\nStop dist: mean={tdf['sd'].mean():.1f}, median={tdf['sd'].median():.1f}")

# Pool size diagnostic
print(f"\nFVG_MAX_AGE: {FVG_MAX_AGE} bars (~{FVG_MAX_AGE/60:.0f} hours)")
