"""
Noise Test (Taguchi method) for NQ trading strategy robustness.

Concept: Add random Gaussian noise to OHLC execution data, keep signals fixed,
re-run the trade management engine, measure stability of results.

Tests trade management robustness — NOT signal detection robustness.
Signals, entries, stops, TPs are all pre-computed. Only the OHLC data used for
stop/TP checking in the bar-by-bar loop gets perturbed.

Baseline: 534 trades, +156.63R (from validate_nt_logic.py)

Noise levels:
  - 0.5 tick = 0.125 points (micro-noise)
  - 1.0 tick = 0.25 points (normal execution noise)
  - 2.0 ticks = 0.50 points (stress test)

Usage: python ninjatrader/noise_test.py
"""
import sys
import time
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT = Path(__file__).resolve().parent.parent

# ============================================================
# Load data (same as validate_nt_logic.py)
# ============================================================
print("Loading data...")
t0 = time.time()
nq = pd.read_parquet(PROJECT / "data" / "NQ_5m_10yr.parquet")
es = pd.read_parquet(PROJECT / "data" / "ES_5m_10yr.parquet")
sig3 = pd.read_parquet(PROJECT / "data" / "cache_signals_10yr_v3.parquet")
bias = pd.read_parquet(PROJECT / "data" / "cache_bias_10yr_v2.parquet")
regime = pd.read_parquet(PROJECT / "data" / "cache_regime_10yr_v2.parquet")
with open(PROJECT / "config" / "params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)
print(f"  Data loaded in {time.time()-t0:.1f}s")

# ============================================================
# Compute SMT divergence (same as validate_nt_logic.py)
# ============================================================
print("Computing SMT divergence...")
t0 = time.time()
from features.smt import compute_smt
smt = compute_smt(nq, es, {'swing': {'left_bars': 3, 'right_bars': 1},
                             'smt': {'sweep_lookback': 15, 'time_tolerance': 1}})
print(f"  SMT computed in {time.time()-t0:.1f}s")

# ============================================================
# Build merged signals: v3 cache + SMT gate for MSS
# (exact copy from validate_nt_logic.py)
# ============================================================
def make_sig(sig3_s, smt_s):
    s = sig3_s.copy()
    mm = s['signal'].astype(bool) & (s['signal_type'] == 'mss')
    mi = s.index[mm]
    s.loc[mi, 'signal'] = False; s.loc[mi, 'signal_dir'] = 0; s['has_smt'] = False
    c = mi.intersection(smt_s.index)
    if len(c) == 0: return s
    md = sig3_s.loc[c, 'signal_dir'].values
    ok = ((md == 1) & smt_s.loc[c, 'smt_bull'].values.astype(bool)) | \
         ((md == -1) & smt_s.loc[c, 'smt_bear'].values.astype(bool))
    g = c[ok]
    s.loc[g, 'signal'] = sig3_s.loc[g, 'signal']
    s.loc[g, 'signal_dir'] = sig3_s.loc[g, 'signal_dir']
    s.loc[g, 'has_smt'] = True
    # Kill MSS in overnight (16:00-03:00 ET)
    rem = s['signal'].astype(bool) & (s['signal_type'] == 'mss')
    mi2 = s.index[rem]
    if len(mi2) > 0:
        et = mi2.tz_convert('US/Eastern')
        ef = et.hour + et.minute / 60.0
        kill = (ef >= 16.0) | (ef < 3.0)
        if kill.any():
            s.loc[mi2[kill], ['signal', 'signal_dir']] = [False, 0]
    return s

ss = make_sig(sig3, smt)
print(f"Merged signals: {ss['signal'].sum()} total")

# ============================================================
# Pre-compute all arrays (done ONCE, shared across all iterations)
# ============================================================
print("Pre-computing features...")
t0 = time.time()
from features.displacement import compute_atr, compute_fluency
from features.swing import compute_swing_levels
from features.pa_quality import compute_alternating_dir_ratio

et_idx = nq.index.tz_convert("US/Eastern")
o_base = nq["open"].values.copy()
h_base = nq["high"].values.copy()
l_base = nq["low"].values.copy()
c_base = nq["close"].values.copy()
n = len(nq)

atr_arr = compute_atr(nq).values
fluency_arr = compute_fluency(nq, params).values
pa_alt_arr = compute_alternating_dir_ratio(nq, window=6).values

swing_p = {"left_bars": params["swing"]["left_bars"], "right_bars": params["swing"]["right_bars"]}
swings = compute_swing_levels(nq, swing_p)
swings["swing_high_price"] = swings["swing_high_price"].shift(1).ffill()
swings["swing_low_price"] = swings["swing_low_price"].shift(1).ffill()
swing_high_mask = swings["swing_high"].shift(1, fill_value=False).values
swing_low_mask = swings["swing_low"].shift(1, fill_value=False).values
# Pre-extract low/high values for swing lookups
swing_low_prices = nq["low"].values.copy()
swing_high_prices = nq["high"].values.copy()

sig_mask = ss["signal"].values.astype(bool)
sig_dir = ss["signal_dir"].values.astype(float)
sig_type = ss["signal_type"].values
has_smt_arr = ss["has_smt"].values.astype(bool)
entry_price_arr = ss["entry_price"].values
model_stop_arr = ss["model_stop"].values
irl_target_arr = ss["irl_target"].values
bias_dir_arr = bias["bias_direction"].values.astype(float)
bias_conf_arr = bias["bias_confidence"].values.astype(float)
regime_arr = regime["regime"].values.astype(float)

# News filter
from features.news_filter import build_news_blackout_mask
news_path = PROJECT / "config" / "news_calendar.csv"
news_blackout_arr = None
if news_path.exists():
    news_bl = build_news_blackout_mask(nq.index, str(news_path),
        params["news"]["blackout_minutes_before"], params["news"]["cooldown_minutes_after"])
    news_blackout_arr = news_bl.values

# Signal quality (same as engine.py)
sq_params_cfg = params.get("signal_quality", {})
sq_enabled = sq_params_cfg.get("enabled", False)
sq_threshold = sq_params_cfg.get("threshold", 0.66)
dual_mode = params.get("dual_mode", {})
dual_mode_enabled = dual_mode.get("enabled", False)
mss_mgmt = params.get("mss_management", {})
mss_mgmt_enabled = mss_mgmt.get("enabled", False)
smt_cfg = params.get("smt", {})

signal_quality = np.full(n, np.nan)
signal_indices = np.where(sig_mask)[0]
if sq_enabled and len(signal_indices) > 0:
    for idx in signal_indices:
        a = atr_arr[idx] if not np.isnan(atr_arr[idx]) else 10.0
        gap = abs(entry_price_arr[idx] - model_stop_arr[idx])
        size_sc = min(1.0, gap / (a * 1.5)) if a > 0 else 0.5
        body = abs(c_base[idx] - o_base[idx])
        rng = h_base[idx] - l_base[idx]
        disp_sc = body / rng if rng > 0 else 0.0
        flu_val = fluency_arr[idx]
        flu_sc = min(1.0, max(0.0, flu_val)) if not np.isnan(flu_val) else 0.5
        window = 6
        if idx >= window:
            dirs = np.sign(c_base[idx-window:idx] - o_base[idx-window:idx])
            alt = np.sum(dirs[1:] != dirs[:-1]) / (window - 1)
            pa_sc = 1.0 - alt
        else:
            pa_sc = 0.5
        signal_quality[idx] = (sq_params_cfg.get("w_size",0.3)*size_sc + sq_params_cfg.get("w_disp",0.3)*disp_sc +
                               sq_params_cfg.get("w_flu",0.2)*flu_sc + sq_params_cfg.get("w_pa",0.2)*pa_sc)

# Date tracking
dates = np.array([
    (et_idx[j] + pd.Timedelta(days=1)).date() if et_idx[j].hour >= 18 else et_idx[j].date()
    for j in range(n)
])

# Engine params
pos_params = params["position"]
risk_params = params["risk"]
bt_params = params["backtest"]
grading_params = params["grading"]
trim_params_cfg = params["trim"]
trail_params = params["trail"]
normal_r = pos_params["normal_r"]
reduced_r = pos_params["reduced_r"]
point_value = pos_params["point_value"]
daily_max_loss_r = risk_params["daily_max_loss_r"]
max_consec_losses = risk_params["max_consecutive_losses"]
commission_per_side = bt_params["commission_per_side_micro"]
slippage_ticks = bt_params["slippage_normal_ticks"]
slippage_points = slippage_ticks * 0.25
a_plus_mult = grading_params["a_plus_size_mult"]
b_plus_mult = grading_params["b_plus_size_mult"]
c_skip = grading_params["c_skip"]
trim_pct = trim_params_cfg["pct"]
be_after_trim = trim_params_cfg["be_after_trim"]
nth_swing = trail_params["use_nth_swing"]
session_filter = params.get("session_filter", {})
sf_enabled = session_filter.get("enabled", False)
skip_london = session_filter.get("skip_london", False)
skip_asia = session_filter.get("skip_asia", True)
session_rules = params.get("session_rules", {})
session_regime_cfg = params.get("session_regime", {})
dir_mgmt = params.get("direction_mgmt", {})
pa_threshold = params.get("pa_quality", {}).get("alt_dir_threshold", 0.334)
min_stop_atr = params.get("regime", {}).get("min_stop_atr_mult", 0.5)
bias_relax = params.get("bias_relaxation", {})

# Pre-compute ET hours/minutes for entry filter (avoid repeated tz_convert)
et_hours = np.array([et_idx[j].hour for j in range(n)], dtype=np.int8)
et_minutes = np.array([et_idx[j].minute for j in range(n)], dtype=np.int8)
et_fracs = et_hours.astype(np.float32) + et_minutes.astype(np.float32) / 60.0
et_dows = np.array([et_idx[j].dayofweek for j in range(n)], dtype=np.int8)

print(f"  Features pre-computed in {time.time()-t0:.1f}s")


# ============================================================
# Helper functions
# ============================================================
def _compute_grade(ba, bc, reg):
    if np.isnan(ba) or np.isnan(reg): return "C"
    if reg == 0.0: return "C"
    aligned = ba > 0.5
    full = reg >= 1.0
    if aligned and full: return "A+"
    if aligned or full: return "B+"
    return "C"


def _compute_contracts(r_amt, stop_dist, pv):
    if stop_dist <= 0 or pv <= 0: return 0
    return max(1, int(r_amt / (stop_dist * pv)))


def _find_nth_swing(mask, prices, idx, n_val, direction):
    count = 0
    for j in range(idx - 1, -1, -1):
        if mask[j]:
            count += 1
            if count == n_val: return prices[j]
    return np.nan


# ============================================================
# Core engine loop — takes OHLC arrays as parameters
# Returns: (trade_count, total_R, win_count, loss_count, max_dd, trade_rs)
# ============================================================
def run_engine_with_ohlc(o, h, l, c):
    """Run the full engine loop with given OHLC arrays.

    Entry filtering uses ORIGINAL signal caches (not affected by noise).
    Exit management (stop/TP checks) uses the provided OHLC arrays.
    Early-cut PA quality check also uses the provided OHLC arrays.
    """
    trades_r = []
    current_date = None
    daily_pnl_r = 0.0
    consecutive_losses = 0
    day_stopped = False
    in_position = False
    pos_direction = pos_entry_idx = 0
    pos_entry_price = pos_stop = pos_tp1 = 0.0
    pos_contracts = pos_remaining_contracts = 0
    pos_trimmed = False
    pos_be_stop = pos_trail_stop = 0.0
    pos_signal_type = pos_grade = ""
    pos_trim_pct_local = trim_pct

    for i in range(n):
        bar_date = dates[i]
        if bar_date != current_date:
            current_date = bar_date
            daily_pnl_r = 0.0
            consecutive_losses = 0
            day_stopped = False

        # ---- EXIT MANAGEMENT (uses noisy OHLC) ----
        if in_position:
            exited = False
            exit_reason = ""
            exit_price = 0.0
            exit_contracts = pos_remaining_contracts

            bars_in_trade = i - pos_entry_idx
            if not pos_trimmed and 2 <= bars_in_trade <= 4:
                pa_start = max(pos_entry_idx, 0)
                pa_end = i + 1
                pa_range = h[pa_start:pa_end] - l[pa_start:pa_end]
                pa_body = np.abs(c[pa_start:pa_end] - o[pa_start:pa_end])
                safe_pa = np.where(pa_range > 0, pa_range, 1.0)
                pa_wick = 1.0 - (pa_body / safe_pa)
                avg_wick = float(np.mean(pa_wick))
                pa_dirs = np.sign(c[pa_start:pa_end] - o[pa_start:pa_end])
                favorable = (pa_dirs == pos_direction).mean()
                disp = (c[i] - pos_entry_price) if pos_direction == 1 else (pos_entry_price - c[i])
                cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                no_progress = disp < cur_atr * 0.3
                bad_pa = avg_wick > 0.65 and favorable < 0.5
                if bad_pa and no_progress and bars_in_trade >= 3:
                    exit_price = o[i+1] if i+1 < n else c[i]
                    exit_reason = "early_cut_pa"
                    exited = True

            if not exited and pos_direction == 1:
                eff_stop = pos_trail_stop if pos_trimmed and pos_trail_stop > 0 else pos_stop
                if pos_trimmed and be_after_trim and pos_be_stop > 0:
                    eff_stop = max(eff_stop, pos_be_stop)
                if l[i] <= eff_stop:
                    exit_price = eff_stop - slippage_points
                    exit_reason = "be_sweep" if (pos_trimmed and eff_stop >= pos_entry_price) else "stop"
                    exited = True
                elif not pos_trimmed and h[i] >= pos_tp1:
                    tc = max(1, int(pos_contracts * pos_trim_pct_local))
                    pos_remaining_contracts = pos_contracts - tc
                    pos_trimmed = True
                    pos_be_stop = pos_entry_price
                    if pos_remaining_contracts > 0:
                        pos_trail_stop = _find_nth_swing(swing_low_mask, swing_low_prices, i, nth_swing, 1)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0: pos_trail_stop = pos_be_stop
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True
                if pos_trimmed and not exited:
                    nt = _find_nth_swing(swing_low_mask, swing_low_prices, i, nth_swing, 1)
                    if not np.isnan(nt) and nt > pos_trail_stop: pos_trail_stop = nt

            elif not exited:  # SHORT
                eff_stop = pos_trail_stop if pos_trimmed and pos_trail_stop > 0 else pos_stop
                if pos_trimmed and be_after_trim and pos_be_stop > 0:
                    eff_stop = min(eff_stop, pos_be_stop)
                if h[i] >= eff_stop:
                    exit_price = eff_stop + slippage_points
                    exit_reason = "be_sweep" if (pos_trimmed and eff_stop <= pos_entry_price) else "stop"
                    exited = True
                elif not pos_trimmed and l[i] <= pos_tp1:
                    tc = max(1, int(pos_contracts * pos_trim_pct_local))
                    pos_remaining_contracts = pos_contracts - tc
                    pos_trimmed = True
                    pos_be_stop = pos_entry_price
                    if pos_remaining_contracts > 0:
                        pos_trail_stop = _find_nth_swing(swing_high_mask, swing_high_prices, i, nth_swing, -1)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0: pos_trail_stop = pos_be_stop
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True
                if pos_trimmed and not exited:
                    nt = _find_nth_swing(swing_high_mask, swing_high_prices, i, nth_swing, -1)
                    if not np.isnan(nt) and nt < pos_trail_stop: pos_trail_stop = nt

            if exited:
                pnl_pts = (exit_price - pos_entry_price) if pos_direction == 1 else (pos_entry_price - exit_price)
                if pos_trimmed and exit_reason != "tp1":
                    trim_pnl_total = (pos_tp1 - pos_entry_price) if pos_direction == 1 else (pos_entry_price - pos_tp1)
                    trim_c = pos_contracts - exit_contracts
                    total_pnl = trim_pnl_total * point_value * trim_c + pnl_pts * point_value * exit_contracts
                    total_comm = commission_per_side * 2 * pos_contracts
                    total_pnl -= total_comm
                else:
                    total_pnl = pnl_pts * point_value * exit_contracts
                    total_comm = commission_per_side * 2 * exit_contracts
                    total_pnl -= total_comm
                stop_dist = abs(pos_entry_price - pos_stop)
                total_risk = stop_dist * point_value * pos_contracts
                r_mult = total_pnl / total_risk if total_risk > 0 else 0.0

                trades_r.append(r_mult)
                daily_pnl_r += r_mult
                if exit_reason == "be_sweep" and pos_trimmed: pass
                elif r_mult < 0: consecutive_losses += 1
                else: consecutive_losses = 0
                if consecutive_losses >= max_consec_losses: day_stopped = True
                if daily_pnl_r <= -daily_max_loss_r: day_stopped = True
                in_position = False

        # ---- ENTRY FILTERS (identical to baseline — uses original signal caches) ----
        if not in_position and news_blackout_arr is not None and news_blackout_arr[i]:
            continue

        if not in_position and not day_stopped:
            et_h = et_hours[i]
            et_m = et_minutes[i]
            if (et_h == 9 and et_m >= 30) or (et_h == 10 and et_m == 0):
                continue

        if not in_position and not day_stopped and sig_mask[i]:
            direction = int(sig_dir[i])
            if direction == 0: continue

            sig_f = params.get("signal_filter", {})
            if not sig_f.get("allow_mss", True) and str(sig_type[i]) == "mss": continue
            if not sig_f.get("allow_trend", True) and str(sig_type[i]) == "trend": continue

            entry_p = entry_price_arr[i]
            stop = model_stop_arr[i]
            tp1 = irl_target_arr[i]
            if np.isnan(entry_p) or np.isnan(stop) or np.isnan(tp1): continue
            if direction == 1 and (stop >= entry_p or tp1 <= entry_p): continue
            if direction == -1 and (stop <= entry_p or tp1 >= entry_p): continue

            # Bias filter
            bias_opposing = (direction == -np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0)
            if bias_opposing:
                is_mss_smt_sig = (smt_cfg.get("enabled", False) and has_smt_arr[i]
                                  and str(sig_type[i]) == "mss")
                if is_mss_smt_sig:
                    pass
                elif bias_relax.get("enabled", False) and direction == -1:
                    pass
                else:
                    continue

            # PA quality
            if pa_alt_arr[i] >= pa_threshold: continue

            # Session filter
            et_frac = et_fracs[i]
            is_mss_smt = (str(sig_type[i]) == "mss" and has_smt_arr[i]
                          and smt_cfg.get("enabled", False))
            mss_bypass = is_mss_smt and smt_cfg.get("bypass_session_filter", False)

            if sf_enabled and not mss_bypass:
                if 9.5 <= et_frac < 16.0:
                    ad = session_filter.get("ny_direction", 0)
                    if ad != 0 and direction != ad: continue
                elif 3.0 <= et_frac < 9.5:
                    if skip_london: continue
                    ad = session_filter.get("london_direction", 0)
                    if ad != 0 and direction != ad: continue
                else:
                    if skip_asia: continue
            elif not sf_enabled and not mss_bypass:
                if not (3.0 <= et_frac < 16.0): continue

            # Min stop
            stop_dist = abs(entry_p - stop)
            min_stop = min_stop_atr * atr_arr[i] if not np.isnan(atr_arr[i]) else 10.0
            if stop_dist < min_stop: continue

            # Signal quality
            if sq_enabled and not np.isnan(signal_quality[i]):
                eff_sq = sq_threshold
                if dual_mode_enabled and direction == -1:
                    eff_sq = dual_mode.get("short_sq_threshold", 0.80)
                if signal_quality[i] < eff_sq: continue

            if i + 1 >= n: continue

            # Grade
            ba = 1.0 if (direction == np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0) else 0.0
            grade = _compute_grade(ba, bias_conf_arr[i], regime_arr[i])
            if grade == "C" and c_skip: continue

            # Sizing
            is_reduced = (et_dows[i] in (0, 4)) or (regime_arr[i] < 1.0)
            base_r = reduced_r if is_reduced else normal_r
            if grade == "A+": r_amount = base_r * a_plus_mult
            elif grade == "B+": r_amount = base_r * b_plus_mult
            else: r_amount = base_r * 0.5

            # Session regime
            if session_regime_cfg.get("enabled", False):
                sr_lunch_s = session_regime_cfg.get("lunch_start", 12.0)
                sr_lunch_e = session_regime_cfg.get("lunch_end", 13.5)
                sr_am_end = session_regime_cfg.get("am_end", 12.0)
                sr_pm_start = session_regime_cfg.get("pm_start", 13.5)
                if et_frac < sr_am_end: sr_mult = session_regime_cfg.get("am_mult", 1.0)
                elif sr_lunch_s <= et_frac < sr_lunch_e: sr_mult = session_regime_cfg.get("lunch_mult", 0.5)
                elif et_frac >= sr_pm_start: sr_mult = session_regime_cfg.get("pm_mult", 0.75)
                else: sr_mult = 1.0
                r_amount *= sr_mult
                if r_amount <= 0: continue

            # Slippage
            actual_entry = entry_p + slippage_points if direction == 1 else entry_p - slippage_points
            stop_dist = abs(actual_entry - stop)
            if stop_dist < 1.0: continue

            contracts = _compute_contracts(r_amount, stop_dist, point_value)
            if contracts <= 0: continue

            # TP adjustments
            is_mss_signal = str(sig_type[i]) == "mss"
            if session_rules.get("enabled", False):
                ny_tp_mult = session_rules.get("ny_tp_multiplier", 1.0)
                if dir_mgmt.get("enabled", False):
                    actual_tp_mult = dir_mgmt.get("long_tp_mult", ny_tp_mult) if direction == 1 else dir_mgmt.get("short_tp_mult", 1.25)
                else:
                    actual_tp_mult = ny_tp_mult
                if mss_mgmt_enabled and is_mss_signal and direction == 1:
                    actual_tp_mult = mss_mgmt.get("long_tp_mult", actual_tp_mult)
                if 9.5 <= et_frac < 16.0:
                    tp_distance = (tp1 - actual_entry) if direction == 1 else (actual_entry - tp1)
                    tp1 = (actual_entry + tp_distance * actual_tp_mult) if direction == 1 else (actual_entry - tp_distance * actual_tp_mult)

            if dual_mode_enabled and direction == -1:
                short_rr = dual_mode.get("short_rr", 0.625)
                if mss_mgmt_enabled and is_mss_signal:
                    short_rr = mss_mgmt.get("short_rr", short_rr)
                tp1 = actual_entry - stop_dist * short_rr

            # Determine trim pct
            if mss_mgmt_enabled and is_mss_signal:
                sig_trim_pct = mss_mgmt.get("short_trim_pct", 1.0) if direction == -1 else mss_mgmt.get("long_trim_pct", trim_pct)
            elif dual_mode_enabled and direction == -1:
                sig_trim_pct = dual_mode.get("short_trim_pct", 1.0)
            elif dir_mgmt.get("enabled", False):
                sig_trim_pct = dir_mgmt.get("long_trim_pct", trim_pct) if direction == 1 else dir_mgmt.get("short_trim_pct", 1.0)
            else:
                sig_trim_pct = trim_pct

            # Enter
            in_position = True
            pos_direction = direction
            pos_entry_idx = i + 1
            pos_entry_price = actual_entry
            pos_stop = stop
            pos_tp1 = tp1
            pos_contracts = contracts
            pos_remaining_contracts = contracts
            pos_trimmed = False
            pos_be_stop = 0.0
            pos_trail_stop = 0.0
            pos_signal_type = str(sig_type[i])
            pos_grade = grade
            pos_trim_pct_local = sig_trim_pct

    # Force close open position at end
    if in_position:
        exit_price = c[-1]
        pnl_pts = (exit_price - pos_entry_price) if pos_direction == 1 else (pos_entry_price - exit_price)
        total_pnl = pnl_pts * point_value * pos_remaining_contracts
        if pos_trimmed:
            trim_c = pos_contracts - pos_remaining_contracts
            if pos_direction == 1:
                trim_pnl = (pos_tp1 - pos_entry_price) * point_value * trim_c
            else:
                trim_pnl = (pos_entry_price - pos_tp1) * point_value * trim_c
            total_pnl += trim_pnl
            total_pnl -= commission_per_side * 2 * pos_contracts
        else:
            total_pnl -= commission_per_side * 2 * pos_remaining_contracts
        stop_dist = abs(pos_entry_price - pos_stop)
        total_risk = stop_dist * point_value * pos_contracts
        r_mult = total_pnl / total_risk if total_risk > 0 else 0.0
        trades_r.append(r_mult)

    # Compute stats
    trades_arr = np.array(trades_r)
    trade_count = len(trades_arr)
    total_R = float(trades_arr.sum()) if trade_count > 0 else 0.0
    win_count = int((trades_arr > 0).sum()) if trade_count > 0 else 0
    win_rate = win_count / trade_count if trade_count > 0 else 0.0

    # Max drawdown in R
    if trade_count > 0:
        cum_r = np.cumsum(trades_arr)
        running_max = np.maximum.accumulate(cum_r)
        drawdowns = cum_r - running_max
        max_dd = float(drawdowns.min())
    else:
        max_dd = 0.0

    ppdd = abs(total_R / max_dd) if max_dd != 0 else float('inf')

    return trade_count, total_R, win_rate, max_dd, ppdd, trades_arr


def generate_noisy_ohlc(noise_std, rng):
    """Generate noisy OHLC arrays maintaining consistency.

    noise_std: standard deviation in NQ points
    rng: numpy random generator

    Returns: (o_noisy, h_noisy, l_noisy, c_noisy)
    """
    # Generate noise for each price
    noise_o = rng.normal(0, noise_std, n)
    noise_h = rng.normal(0, noise_std, n)
    noise_l = rng.normal(0, noise_std, n)
    noise_c = rng.normal(0, noise_std, n)

    # Apply noise
    o_noisy = o_base + noise_o
    c_noisy = c_base + noise_c

    # High must be >= max(open, close) of the bar
    h_raw = h_base + noise_h
    h_noisy = np.maximum(h_raw, np.maximum(o_noisy, c_noisy))

    # Low must be <= min(open, close) of the bar
    l_raw = l_base + noise_l
    l_noisy = np.minimum(l_raw, np.minimum(o_noisy, c_noisy))

    # Also ensure high >= low
    # (should always be true given the above, but defensive)
    mask_bad = h_noisy < l_noisy
    if mask_bad.any():
        mid = (h_noisy[mask_bad] + l_noisy[mask_bad]) / 2
        h_noisy[mask_bad] = mid + 0.25
        l_noisy[mask_bad] = mid - 0.25

    return o_noisy, h_noisy, l_noisy, c_noisy


# ============================================================
# BASELINE RUN (no noise)
# ============================================================
print("\n" + "=" * 70)
print("BASELINE RUN (no noise)")
print("=" * 70)
t0 = time.time()
bl_count, bl_R, bl_wr, bl_dd, bl_ppdd, bl_trades = run_engine_with_ohlc(o_base, h_base, l_base, c_base)
bl_time = time.time() - t0
print(f"  Trades: {bl_count}")
print(f"  Total R: {bl_R:.2f}")
print(f"  Win Rate: {bl_wr*100:.1f}%")
print(f"  Max DD: {bl_dd:.2f}R")
print(f"  PPDD: {bl_ppdd:.2f}")
print(f"  Time: {bl_time:.2f}s")


# ============================================================
# NOISE TEST ITERATIONS
# ============================================================
NOISE_LEVELS = [
    (0.125, "0.5 tick"),   # 0.5 * tick_size(0.25) = 0.125 pts
    (0.25,  "1.0 tick"),   # 1.0 * tick_size = 0.25 pts
    (0.50,  "2.0 ticks"),  # 2.0 * tick_size = 0.50 pts
]
N_ITERATIONS = 100

all_results = {}

for noise_std, noise_label in NOISE_LEVELS:
    print(f"\n{'='*70}")
    print(f"NOISE TEST: {noise_label} (std = {noise_std} pts) x {N_ITERATIONS} iterations")
    print(f"{'='*70}")

    results = {
        'total_R': [],
        'trade_count': [],
        'win_rate': [],
        'max_dd': [],
        'ppdd': [],
    }

    t0 = time.time()
    for it in range(N_ITERATIONS):
        rng = np.random.default_rng(seed=42000 + it)
        o_n, h_n, l_n, c_n = generate_noisy_ohlc(noise_std, rng)
        tc, tr, wr, dd, ppdd_val, _ = run_engine_with_ohlc(o_n, h_n, l_n, c_n)
        results['total_R'].append(tr)
        results['trade_count'].append(tc)
        results['win_rate'].append(wr)
        results['max_dd'].append(dd)
        results['ppdd'].append(ppdd_val)

        if (it + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (it + 1) / elapsed
            print(f"  [{it+1:3d}/{N_ITERATIONS}] {elapsed:.1f}s ({rate:.1f} iter/s)")

    total_time = time.time() - t0
    all_results[noise_label] = results

    # Stats
    R_arr = np.array(results['total_R'])
    tc_arr = np.array(results['trade_count'])
    wr_arr = np.array(results['win_rate'])
    dd_arr = np.array(results['max_dd'])
    ppdd_arr = np.array(results['ppdd'])

    print(f"\n  --- Total R ---")
    print(f"  Mean:   {R_arr.mean():.2f}  (baseline: {bl_R:.2f})")
    print(f"  Median: {np.median(R_arr):.2f}")
    print(f"  Std:    {R_arr.std():.2f}")
    print(f"  Min:    {R_arr.min():.2f}")
    print(f"  Max:    {R_arr.max():.2f}")
    print(f"  Range:  {R_arr.max()-R_arr.min():.2f}")
    pct_5 = np.percentile(R_arr, 5)
    pct_95 = np.percentile(R_arr, 95)
    print(f"  P5-P95: [{pct_5:.2f}, {pct_95:.2f}]")
    print(f"  % profitable (R>0): {100*(R_arr>0).mean():.0f}%")
    print(f"  % R > baseline/2:   {100*(R_arr > bl_R/2).mean():.0f}%")
    print(f"  % R within 20% of baseline: {100*(np.abs(R_arr - bl_R) < 0.2 * abs(bl_R)).mean():.0f}%")

    print(f"\n  --- Trade Count ---")
    print(f"  Mean:   {tc_arr.mean():.1f}  (baseline: {bl_count})")
    print(f"  Min:    {tc_arr.min()},  Max: {tc_arr.max()}")
    print(f"  Std:    {tc_arr.std():.1f}")

    print(f"\n  --- Win Rate ---")
    print(f"  Mean:   {wr_arr.mean()*100:.1f}%  (baseline: {bl_wr*100:.1f}%)")
    print(f"  Min:    {wr_arr.min()*100:.1f}%,  Max: {wr_arr.max()*100:.1f}%")
    print(f"  Std:    {wr_arr.std()*100:.1f}%")

    print(f"\n  --- Max Drawdown ---")
    print(f"  Mean:   {dd_arr.mean():.2f}R  (baseline: {bl_dd:.2f}R)")
    print(f"  Worst:  {dd_arr.min():.2f}R")
    print(f"  Best:   {dd_arr.max():.2f}R")

    print(f"\n  --- PPDD ---")
    ppdd_finite = ppdd_arr[np.isfinite(ppdd_arr)]
    if len(ppdd_finite) > 0:
        print(f"  Mean:   {ppdd_finite.mean():.2f}  (baseline: {bl_ppdd:.2f})")
        print(f"  Min:    {ppdd_finite.min():.2f}")
        print(f"  % PPDD > 5: {100*(ppdd_finite > 5).mean():.0f}%")
    else:
        print(f"  All runs had 0 drawdown (infinite PPDD)")

    print(f"\n  Total time: {total_time:.1f}s ({N_ITERATIONS/total_time:.1f} iter/s)")


# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n\n" + "=" * 100)
print("NOISE TEST SUMMARY (Taguchi Method)")
print("=" * 100)
print(f"\nBaseline: {bl_count} trades, {bl_R:.2f}R, WR={bl_wr*100:.1f}%, DD={bl_dd:.2f}R, PPDD={bl_ppdd:.2f}")
print(f"Iterations per noise level: {N_ITERATIONS}")

header = f"{'Noise Level':>14s} | {'Mean R':>8s} | {'Med R':>8s} | {'Std R':>7s} | {'R Range':>14s} | {'P5-P95':>16s} | {'%Prof':>5s} | {'Trades':>8s} | {'WR':>6s} | {'MeanDD':>7s} | {'WorstDD':>8s} | {'PPDD':>6s}"
print(f"\n{header}")
print("-" * len(header))

# Baseline row
print(f"{'Baseline':>14s} | {bl_R:>8.2f} | {bl_R:>8.2f} | {'---':>7s} | {'---':>14s} | {'---':>16s} | {'100%':>5s} | {bl_count:>8d} | {bl_wr*100:>5.1f}% | {bl_dd:>7.2f} | {bl_dd:>8.2f} | {bl_ppdd:>6.2f}")

for noise_std, noise_label in NOISE_LEVELS:
    r = all_results[noise_label]
    R_arr = np.array(r['total_R'])
    tc_arr = np.array(r['trade_count'])
    wr_arr = np.array(r['win_rate'])
    dd_arr = np.array(r['max_dd'])
    ppdd_arr = np.array(r['ppdd'])
    ppdd_fin = ppdd_arr[np.isfinite(ppdd_arr)]

    r_range = f"[{R_arr.min():.1f}, {R_arr.max():.1f}]"
    p5_95 = f"[{np.percentile(R_arr,5):.1f}, {np.percentile(R_arr,95):.1f}]"
    pct_prof = f"{100*(R_arr>0).mean():.0f}%"
    tc_str = f"{tc_arr.mean():.0f}+/-{tc_arr.std():.0f}"
    wr_str = f"{wr_arr.mean()*100:.1f}%"
    ppdd_str = f"{ppdd_fin.mean():.2f}" if len(ppdd_fin) > 0 else "inf"

    print(f"{noise_label:>14s} | {R_arr.mean():>8.2f} | {np.median(R_arr):>8.2f} | {R_arr.std():>7.2f} | {r_range:>14s} | {p5_95:>16s} | {pct_prof:>5s} | {tc_str:>8s} | {wr_str:>6s} | {dd_arr.mean():>7.2f} | {dd_arr.min():>8.2f} | {ppdd_str:>6s}")


# ============================================================
# STABILITY VERDICT
# ============================================================
print("\n\n" + "=" * 70)
print("STABILITY VERDICT")
print("=" * 70)

# Use 1-tick noise (middle level) as primary stability metric
mid_results = all_results["1.0 tick"]
mid_R = np.array(mid_results['total_R'])
mid_wr = np.array(mid_results['win_rate'])

# Coefficient of variation (lower = more stable)
cv = mid_R.std() / abs(mid_R.mean()) * 100 if mid_R.mean() != 0 else float('inf')
print(f"\n  Coefficient of Variation (1-tick noise): {cv:.1f}%")

if cv < 5:
    stability = "EXCELLENT"
    desc = "Results extremely stable under execution noise"
elif cv < 10:
    stability = "GOOD"
    desc = "Results stable with minor variance from execution noise"
elif cv < 20:
    stability = "ACCEPTABLE"
    desc = "Results somewhat sensitive to execution noise"
elif cv < 30:
    stability = "CONCERNING"
    desc = "Results sensitive to execution noise — may be overfit to exact prices"
else:
    stability = "POOR"
    desc = "Results highly unstable — strategy is fragile"

print(f"  Stability rating: {stability}")
print(f"  Assessment: {desc}")

# Additional checks
stress_R = np.array(all_results["2.0 ticks"]['total_R'])
pct_profitable_stress = 100 * (stress_R > 0).mean()
print(f"\n  Stress test (2-tick noise): {pct_profitable_stress:.0f}% of runs profitable")
print(f"  Worst case R (2-tick): {stress_R.min():.2f}  (baseline: {bl_R:.2f})")
print(f"  Degradation at 2-tick: {100*(1 - stress_R.mean()/bl_R):.1f}% loss of R vs baseline")

# R-sensitivity: how many R per point of noise std
r_per_noise = []
for noise_std, noise_label in NOISE_LEVELS:
    r_arr = np.array(all_results[noise_label]['total_R'])
    delta_r = bl_R - r_arr.mean()
    r_per_noise.append((noise_std, delta_r, noise_label))

print(f"\n  R-sensitivity to noise:")
for ns, dr, nl in r_per_noise:
    print(f"    {nl:>12s}: {dr:>+7.2f}R delta ({dr/bl_R*100:>+6.1f}% of baseline)")

print(f"\n{'='*70}")
print("NOISE TEST COMPLETE")
print(f"{'='*70}")
