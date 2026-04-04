"""
Step 1: Validate NT trade management matches Python engine EXACTLY.
Uses Python's own signal cache — only trade management logic differs.
If this matches, we know the C# trade management is correct.
"""
import sys, numpy as np, pandas as pd, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load everything
nq = pd.read_parquet("data/NQ_5m_10yr.parquet")
sig3 = pd.read_parquet("data/cache_signals_10yr_v3.parquet")
bias = pd.read_parquet("data/cache_bias_10yr_v2.parquet")
regime = pd.read_parquet("data/cache_regime_10yr_v2.parquet")
with open("config/params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# Trend-only (block MSS — no SMT in this test)
ss = sig3.copy()
mm = ss["signal"].astype(bool) & (ss["signal_type"] == "mss")
ss.loc[ss.index[mm], ["signal", "signal_dir"]] = [False, 0]

# ============================================================
# Run Python engine (ground truth)
# ============================================================
class Dummy:
    def predict(self, d): return np.ones(d.num_row(), dtype=np.float32)
dummy_X = pd.DataFrame({"d": np.zeros(len(nq))}, index=nq.index)
from backtest.engine import run_backtest
py_trades = run_backtest(nq, ss, bias, regime["regime"], Dummy(), dummy_X, params, threshold=0.0)

# ============================================================
# Run NT-style trade management (must match Python EXACTLY)
# ============================================================
from features.displacement import compute_atr, compute_fluency
from features.swing import compute_swing_levels
from features.pa_quality import compute_alternating_dir_ratio

et_idx = nq.index.tz_convert("US/Eastern")
o, h, l, c = nq["open"].values, nq["high"].values, nq["low"].values, nq["close"].values
n = len(nq)

# Pre-compute same arrays as engine.py
atr_arr = compute_atr(nq).values
fluency_arr = compute_fluency(nq, params).values
pa_alt_arr = compute_alternating_dir_ratio(nq, window=6).values

swing_p = {"left_bars": params["swing"]["left_bars"], "right_bars": params["swing"]["right_bars"]}
swings = compute_swing_levels(nq, swing_p)
swings["swing_high_price"] = swings["swing_high_price"].shift(1).ffill()
swings["swing_low_price"] = swings["swing_low_price"].shift(1).ffill()
swing_high_mask = swings["swing_high"].shift(1, fill_value=False).values
swing_low_mask = swings["swing_low"].shift(1, fill_value=False).values

sig_mask = ss["signal"].values.astype(bool)
sig_dir = ss["signal_dir"].values.astype(float)
sig_type = ss["signal_type"].values
entry_price_arr = ss["entry_price"].values
model_stop_arr = ss["model_stop"].values
irl_target_arr = ss["irl_target"].values
bias_dir_arr = bias["bias_direction"].values.astype(float)
bias_conf_arr = bias["bias_confidence"].values.astype(float)
regime_arr = regime["regime"].values.astype(float)

# News filter
from features.news_filter import build_news_blackout_mask
news_path = Path("config/news_calendar.csv")
news_blackout_arr = None
if news_path.exists():
    news_bl = build_news_blackout_mask(nq.index, str(news_path),
        params["news"]["blackout_minutes_before"], params["news"]["cooldown_minutes_after"])
    news_blackout_arr = news_bl.values

# Signal quality (same as engine.py)
sq_params = params.get("signal_quality", {})
sq_enabled = sq_params.get("enabled", False)
sq_threshold = sq_params.get("threshold", 0.66)
dual_mode = params.get("dual_mode", {})
dual_mode_enabled = dual_mode.get("enabled", False)
mss_mgmt = params.get("mss_management", {})
mss_mgmt_enabled = mss_mgmt.get("enabled", False)

signal_quality = np.full(n, np.nan)
signal_indices = np.where(sig_mask)[0]
if sq_enabled and len(signal_indices) > 0:
    for idx in signal_indices:
        a = atr_arr[idx] if not np.isnan(atr_arr[idx]) else 10.0
        gap = abs(entry_price_arr[idx] - model_stop_arr[idx])
        size_sc = min(1.0, gap / (a * 1.5)) if a > 0 else 0.5
        body = abs(c[idx] - o[idx])
        rng = h[idx] - l[idx]
        disp_sc = body / rng if rng > 0 else 0.0
        flu_val = fluency_arr[idx]
        flu_sc = min(1.0, max(0.0, flu_val)) if not np.isnan(flu_val) else 0.5
        window = 6
        if idx >= window:
            dirs = np.sign(c[idx-window:idx] - o[idx-window:idx])
            alt = np.sum(dirs[1:] != dirs[:-1]) / (window - 1)
            pa_sc = 1.0 - alt
        else:
            pa_sc = 0.5
        signal_quality[idx] = (sq_params.get("w_size",0.3)*size_sc + sq_params.get("w_disp",0.3)*disp_sc +
                               sq_params.get("w_flu",0.2)*flu_sc + sq_params.get("w_pa",0.2)*pa_sc)

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
trim_params = params["trim"]
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
trim_pct = trim_params["pct"]
be_after_trim = trim_params["be_after_trim"]
nth_swing = trail_params["use_nth_swing"]
session_filter = params.get("session_filter", {})
sf_enabled = session_filter.get("enabled", False)
skip_london = session_filter.get("skip_london", False)
skip_asia = session_filter.get("skip_asia", True)
session_rules = params.get("session_rules", {})
session_regime = params.get("session_regime", {})
dir_mgmt = params.get("direction_mgmt", {})
pa_threshold = params.get("pa_quality", {}).get("alt_dir_threshold", 0.334)
min_stop_atr = params.get("regime", {}).get("min_stop_atr_mult", 0.5)

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

# ---- NT-style backtest (replicate engine.py EXACTLY) ----
trades_nt = []
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
pos_bias_dir = pos_regime = pos_model_prob = 0.0
pos_trim_pct = trim_pct

for i in range(n):
    bar_date = dates[i]
    if bar_date != current_date:
        current_date = bar_date
        daily_pnl_r = 0.0
        consecutive_losses = 0
        day_stopped = False

    # ---- EXIT MANAGEMENT (same as engine.py) ----
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
                tc = max(1, int(pos_contracts * pos_trim_pct))
                pos_remaining_contracts = pos_contracts - tc
                pos_trimmed = True
                pos_be_stop = pos_entry_price
                if pos_remaining_contracts > 0:
                    pos_trail_stop = _find_nth_swing(swing_low_mask, nq["low"].values, i, nth_swing, 1)
                    if np.isnan(pos_trail_stop) or pos_trail_stop <= 0: pos_trail_stop = pos_be_stop
                if pos_remaining_contracts <= 0:
                    exit_price = pos_tp1
                    exit_reason = "tp1"
                    exit_contracts = pos_contracts
                    exited = True
            if pos_trimmed and not exited:
                nt = _find_nth_swing(swing_low_mask, nq["low"].values, i, nth_swing, 1)
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
                tc = max(1, int(pos_contracts * pos_trim_pct))
                pos_remaining_contracts = pos_contracts - tc
                pos_trimmed = True
                pos_be_stop = pos_entry_price
                if pos_remaining_contracts > 0:
                    pos_trail_stop = _find_nth_swing(swing_high_mask, nq["high"].values, i, nth_swing, -1)
                    if np.isnan(pos_trail_stop) or pos_trail_stop <= 0: pos_trail_stop = pos_be_stop
                if pos_remaining_contracts <= 0:
                    exit_price = pos_tp1
                    exit_reason = "tp1"
                    exit_contracts = pos_contracts
                    exited = True
            if pos_trimmed and not exited:
                nt = _find_nth_swing(swing_high_mask, nq["high"].values, i, nth_swing, -1)
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

            trades_nt.append({"r": r_mult, "reason": exit_reason, "dir": pos_direction,
                              "type": pos_signal_type, "trimmed": pos_trimmed})
            daily_pnl_r += r_mult
            if exit_reason == "be_sweep" and pos_trimmed: pass
            elif r_mult < 0: consecutive_losses += 1
            else: consecutive_losses = 0
            if consecutive_losses >= max_consec_losses: day_stopped = True
            if daily_pnl_r <= -daily_max_loss_r: day_stopped = True
            in_position = False

    # ---- ENTRY (same as engine.py) ----
    if not in_position and news_blackout_arr is not None and news_blackout_arr[i]:
        continue

    if not in_position and not day_stopped:
        et_h = et_idx[i].hour
        et_m = et_idx[i].minute
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

        # Bias
        bias_opposing = (direction == -np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0)
        if bias_opposing: continue

        # PA quality
        if pa_alt_arr[i] >= pa_threshold: continue

        # Session
        et_frac = et_idx[i].hour + et_idx[i].minute / 60.0
        is_mss_smt = False  # no SMT in this test
        mss_bypass = False
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

        # SQ
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
        is_reduced = (et_idx[i].dayofweek in (0, 4)) or (regime_arr[i] < 1.0)
        base_r = reduced_r if is_reduced else normal_r
        if grade == "A+": r_amount = base_r * a_plus_mult
        elif grade == "B+": r_amount = base_r * b_plus_mult
        else: r_amount = base_r * 0.5

        # Session regime
        if session_regime.get("enabled", False):
            sr_lunch_s = session_regime.get("lunch_start", 12.0)
            sr_lunch_e = session_regime.get("lunch_end", 13.5)
            sr_am_end = session_regime.get("am_end", 12.0)
            sr_pm_start = session_regime.get("pm_start", 13.5)
            if et_frac < sr_am_end: sr_mult = session_regime.get("am_mult", 1.0)
            elif sr_lunch_s <= et_frac < sr_lunch_e: sr_mult = session_regime.get("lunch_mult", 0.5)
            elif et_frac >= sr_pm_start: sr_mult = session_regime.get("pm_mult", 0.75)
            else: sr_mult = 1.0
            r_amount *= sr_mult
            if r_amount <= 0: continue

        # Slippage
        actual_entry = entry_p + slippage_points if direction == 1 else entry_p - slippage_points
        stop_dist = abs(actual_entry - stop)
        if stop_dist < 1.0: continue

        contracts = _compute_contracts(r_amount, stop_dist, point_value)
        if contracts <= 0: continue

        # TP
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
        pos_bias_dir = bias_dir_arr[i]
        pos_regime = regime_arr[i]
        pos_grade = grade

        if mss_mgmt_enabled and is_mss_signal:
            pos_trim_pct = mss_mgmt.get("short_trim_pct", 1.0) if direction == -1 else mss_mgmt.get("long_trim_pct", trim_pct)
        elif dual_mode_enabled and direction == -1:
            pos_trim_pct = dual_mode.get("short_trim_pct", 1.0)
        elif dir_mgmt.get("enabled", False):
            pos_trim_pct = dir_mgmt.get("long_trim_pct", trim_pct) if direction == 1 else dir_mgmt.get("short_trim_pct", 1.0)
        else:
            pos_trim_pct = trim_pct

# ============================================================
# Compare
# ============================================================
nt_df = pd.DataFrame(trades_nt)
print("=" * 70)
print("TRADE MANAGEMENT VALIDATION")
print("=" * 70)
print(f"\nPython engine:  {len(py_trades)} trades, R={py_trades['r_multiple'].sum():.2f}, WR={100*(py_trades['r_multiple']>0).mean():.1f}%")
print(f"NT validator:   {len(nt_df)} trades, R={nt_df['r'].sum():.2f}, WR={100*(nt_df['r']>0).mean():.1f}%")
print(f"\nPython exits:\n{py_trades.groupby('exit_reason')['r_multiple'].agg(['count','sum']).to_string()}")
print(f"\nNT exits:\n{nt_df.groupby('reason')['r'].agg(['count','sum']).to_string()}")
print(f"\nDifference: {len(nt_df) - len(py_trades)} trades, {nt_df['r'].sum() - py_trades['r_multiple'].sum():.2f} R")
if len(nt_df) == len(py_trades):
    r_diff = nt_df["r"].values - py_trades["r_multiple"].values
    print(f"Per-trade R diff: max={r_diff.max():.4f}, min={r_diff.min():.4f}, mean={r_diff.mean():.6f}")
    mismatches = np.abs(r_diff) > 0.01
    print(f"Trades with R diff > 0.01: {mismatches.sum()}")
print("=" * 70)
