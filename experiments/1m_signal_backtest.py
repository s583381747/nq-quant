import sys; sys.path.insert(0, '.')
import logging; logging.basicConfig(level=logging.WARNING)
import pandas as pd, numpy as np, yaml, time

params = yaml.safe_load(open('config/params.yaml'))

# Load data
df_1m = pd.read_parquet('data/NQ_1min_10yr.parquet')
df_5m = pd.read_parquet('data/NQ_5m_10yr.parquet')
df_1h = pd.read_parquet('data/NQ_1H_10yr.parquet')
df_4h = pd.read_parquet('data/NQ_4H_10yr.parquet')

print(f"1m: {len(df_1m):,} bars")
print(f"5m: {len(df_5m):,} bars")

# Compute bias on 5m (too slow on 1m for full 10yr)
from features.sessions import label_sessions, compute_session_levels, compute_orm
from features.bias import compute_daily_bias, compute_regime
from features.displacement import compute_atr as compute_atr_func
from features.swing import compute_swing_levels

print("Computing bias on 5m...")
t0 = time.time()
sessions_5m = label_sessions(df_5m, params)
session_levels_5m = compute_session_levels(df_5m, params)
orm_5m = compute_orm(df_5m, params)
bias_5m = compute_daily_bias(df_5m, session_levels_5m, orm_5m, df_4h, df_1h, params)
bias_dir_5m = bias_5m['bias_direction']
regime_5m = compute_regime(df_5m, df_4h, bias_5m, params)
print(f"Bias computed in {time.time()-t0:.0f}s")

# Compute 5m swings for IRL targets
swings_5m = compute_swing_levels(df_5m, params['swing'])

# Align bias to 1m via ffill
bias_dir_1m = bias_dir_5m.reindex(df_1m.index, method='ffill').values
regime_1m = regime_5m.reindex(df_1m.index, method='ffill').values

# Run signal detection on 1m — but only for a subset (too slow for full 3.5M)
# Process year by year
print("Detecting signals on 1m data (year by year)...")
from features.entry_signals import detect_all_signals

et_1m = df_1m.index.tz_convert('US/Eastern')
atr_1m = compute_atr_func(df_1m)

# For IRL target: align 5m swing_high_price to 1m
# FIX 4: Shift swing levels by 1 (5m) bar before reindexing to 1m
swing_high_1m = swings_5m['swing_high_price'].shift(1).ffill().reindex(df_1m.index, method='ffill').values
swing_low_1m = swings_5m['swing_low_price'].shift(1).ffill().reindex(df_1m.index, method='ffill').values

# Process in yearly chunks to manage memory
all_trades = []
n_1m = len(df_1m)
close_1m = df_1m['close'].values
open_1m = df_1m['open'].values
high_1m = df_1m['high'].values
low_1m = df_1m['low'].values

years = sorted(et_1m.year.unique())
for year in years:
    t1 = time.time()
    year_mask = et_1m.year == year
    df_year = df_1m[year_mask]
    if len(df_year) < 1000:
        continue

    sig_year = detect_all_signals(df_year, params)
    sig_mask = sig_year['signal'].values.astype(bool)
    sig_dir = sig_year['signal_dir'].values

    n_y = len(df_year)
    close_y = df_year['close'].values
    open_y = df_year['open'].values
    high_y = df_year['high'].values
    low_y = df_year['low'].values
    et_y = df_year.index.tz_convert('US/Eastern')

    # Get aligned bias for this year
    bias_y = bias_dir_5m.reindex(df_year.index, method='ffill').values
    atr_y = compute_atr_func(df_year).values

    # Get aligned 5m swing for IRL target
    sw_high_y = swings_5m['swing_high_price'].reindex(df_year.index, method='ffill').values

    year_trades = 0
    for i in range(n_y - 1):
        if not sig_mask[i]: continue
        d = int(sig_dir[i])
        if d != 1: continue  # long only
        if np.isnan(bias_y[i]) or d != np.sign(bias_y[i]) or bias_y[i] == 0: continue

        ep = sig_year['entry_price'].values[i]
        stop = sig_year['model_stop'].values[i]  # candle2 open on 1m
        if np.isnan(ep) or np.isnan(stop): continue
        sd = abs(ep - stop)
        cur_atr = atr_y[i]
        if cur_atr <= 0 or sd < 1.0 * cur_atr or sd > 8.0 * cur_atr: continue
        if stop >= ep: continue

        # IRL target = 5m swing high (HTF target, not 1m swing)
        tp1 = sw_high_y[i]
        if np.isnan(tp1) or tp1 <= ep:
            tp1 = ep + sd * 2  # fallback: 2R

        rr = abs(tp1 - ep) / sd
        if rr >= 3.0: continue

        hour = et_y[i].hour
        minute = et_y[i].minute
        if not (3 <= hour < 16): continue  # NY + London
        # FIX 3: ORM no-trade window (9:30-10:00 ET observation only)
        if (hour == 9 and minute >= 30) or (hour == 10 and minute == 0): continue

        # Simulate on 1m (300 bars = 5 hours max hold)
        contracts = max(1, int(500 / (sd * 2)))
        trimmed = False; trim_pnl = 0; stop_live = stop
        for j in range(i+1, min(i+301, n_y)):
            if low_y[j] <= stop_live:
                rpnl = (stop_live - ep) * 2 * contracts; break
            if not trimmed and high_y[j] >= tp1:
                tc = contracts // 2; trim_pnl = (tp1 - ep) * 2 * tc
                contracts -= tc; trimmed = True; stop_live = ep
        else:
            rpnl = (close_y[min(i+300, n_y-1)] - ep) * 2 * contracts

        total_pnl = trim_pnl + rpnl
        all_trades.append({
            'pnl': total_pnl, 'win': 1 if total_pnl > 0 else 0,
            'year': year, 'sd': sd, 'rr': rr,
            'type': str(sig_year['signal_type'].values[i]),
        })
        year_trades += 1

    elapsed = time.time() - t1
    print(f"  {year}: {sig_mask.sum()} signals, {year_trades} trades ({elapsed:.0f}s)")

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
for t in tdf['type'].unique():
    sub = tdf[tdf['type']==t]
    print(f"  {t}: {len(sub)} trades, WR={100*sub['win'].mean():.0f}%, avg=${sub['pnl'].mean():.0f}")

# Stop distance stats
print(f"\nStop dist: mean={tdf['sd'].mean():.1f}, median={tdf['sd'].median():.1f}")
