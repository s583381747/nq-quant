"""
Export ALL signal data needed for NT8 to execute trades without its own signal detection.

Produces one row per ACCEPTED signal bar (after all engine filters: bias, session,
news, SQ, grade, daily limits, position blocking, etc.). Each row contains everything
NT8 needs to decide entry + manage the trade.

Uses the exact same logic as validate_nt_logic.py / engine.py to determine which
signals are valid — including stateful filters like daily loss limit, 0-for-2 rule,
and "one position at a time" blocking.

Output: ninjatrader/python_signals_v4_export.csv

Usage: python ninjatrader/export_signals_for_nt8.py
"""
import sys
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
nq = pd.read_parquet(PROJECT / "data" / "NQ_5m_10yr.parquet")
es = pd.read_parquet(PROJECT / "data" / "ES_5m_10yr.parquet")
sig3 = pd.read_parquet(PROJECT / "data" / "cache_signals_10yr_v4.parquet")
bias = pd.read_parquet(PROJECT / "data" / "cache_bias_10yr_v2.parquet")
regime = pd.read_parquet(PROJECT / "data" / "cache_regime_10yr_v2.parquet")
with open(PROJECT / "config" / "params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# ============================================================
# Load session levels + HTF swings (for multi-level TP)
# ============================================================
multi_tp_cfg = params.get("multi_tp", {})
multi_tp_enabled = multi_tp_cfg.get("enabled", False)

if multi_tp_enabled:
    print("Loading session levels for multi-TP liquidity ladder...")
    session_levels = pd.read_parquet(PROJECT / "data" / "cache_session_levels_10yr_v2.parquet")
    sess_asia_high = session_levels["asia_high"].values
    sess_asia_low = session_levels["asia_low"].values
    sess_london_high = session_levels["london_high"].values
    sess_london_low = session_levels["london_low"].values
    sess_overnight_high = session_levels["overnight_high"].values
    sess_overnight_low = session_levels["overnight_low"].values

    print("Computing HTF swings (left=10, right=3) for liquidity ladder...")
    from features.swing import compute_swing_levels as _compute_swing_levels_fn
    htf_swing_p = {"left_bars": 10, "right_bars": 3}
    swings_htf = _compute_swing_levels_fn(nq, htf_swing_p)
    swings_htf["swing_high_price"] = swings_htf["swing_high_price"].shift(1).ffill()
    swings_htf["swing_low_price"] = swings_htf["swing_low_price"].shift(1).ffill()
    htf_swing_high_price = swings_htf["swing_high_price"].values
    htf_swing_low_price = swings_htf["swing_low_price"].values
    htf_swing_high_mask = swings_htf["swing_high"].shift(1, fill_value=False).values
    htf_swing_low_mask = swings_htf["swing_low"].shift(1, fill_value=False).values

# ============================================================
# Compute SMT divergence (same as validate_nt_logic.py)
# ============================================================
print("Computing SMT divergence...")
from features.smt import compute_smt
smt = compute_smt(nq, es, {'swing': {'left_bars': 3, 'right_bars': 1},
                             'smt': {'sweep_lookback': 15, 'time_tolerance': 1}})

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
print(f"Merged signals: {ss['signal'].sum()} total ({(ss['signal_type'] == 'trend').sum()} trend raw, "
      f"{((ss['signal'].astype(bool)) & (ss['signal_type'] == 'mss')).sum()} MSS+SMT)")

# ============================================================
# Pre-compute arrays (same as validate_nt_logic.py / engine.py)
# ============================================================
print("Pre-computing features...")
from features.displacement import compute_atr, compute_fluency
from features.swing import compute_swing_levels
from features.pa_quality import compute_alternating_dir_ratio

et_idx = nq.index.tz_convert("US/Eastern")
o, h, l, c = nq["open"].values, nq["high"].values, nq["low"].values, nq["close"].values
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

sig_mask = ss["signal"].values.astype(bool)
sig_dir = ss["signal_dir"].values.astype(float)
sig_type = ss["signal_type"].values
has_smt_arr = ss["has_smt"].values.astype(bool)
entry_price_arr = ss["entry_price"].values
model_stop_arr = ss["model_stop"].values
irl_target_arr = ss["irl_target"].values
sweep_score_arr = ss["sweep_score"].values if "sweep_score" in ss.columns else np.zeros(n, dtype=np.int8)
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

# Signal quality (same logic as engine.py)
sq_params = params.get("signal_quality", {})
sq_enabled = sq_params.get("enabled", False)
sq_threshold = sq_params.get("threshold", 0.66)
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

# Date tracking (same as engine.py)
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
session_regime_cfg = params.get("session_regime", {})
dir_mgmt = params.get("direction_mgmt", {})
pa_threshold = params.get("pa_quality", {}).get("alt_dir_threshold", 0.334)
min_stop_atr = params.get("regime", {}).get("min_stop_atr_mult", 0.5)
bias_relax = params.get("bias_relaxation", {})


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


def _find_nth_swing_price(mask, prices, idx, n_val):
    """Find the nth swing point price looking backward from idx."""
    count = 0
    for j in range(idx - 1, -1, -1):
        if mask[j]:
            count += 1
            if count == n_val:
                return prices[j]
    return np.nan


def _build_liquidity_ladder_long(entry_price, stop_price, tp1, idx):
    """Build TP2 for a LONG trade using liquidity ladder.

    TP2 = closest level above TP1 from:
      - Session highs (Asia, London, overnight)
      - HTF swing high
      - 2nd/3rd regular swing high
    Fallback: TP2 = entry + 1.5 * (TP1 - entry)
    """
    candidates = []

    # Session highs
    for arr in [sess_asia_high, sess_london_high, sess_overnight_high]:
        val = arr[idx]
        if not np.isnan(val) and val > tp1 + 1.0:
            candidates.append(val)

    # HTF swing high
    htf_sh = htf_swing_high_price[idx]
    if not np.isnan(htf_sh) and htf_sh > tp1 + 1.0:
        candidates.append(htf_sh)

    # 2nd regular swing high
    sh2 = _find_nth_swing_price(swing_high_mask, nq["high"].values, idx, 2)
    if not np.isnan(sh2) and sh2 > tp1 + 1.0:
        candidates.append(sh2)

    # 3rd regular swing high
    sh3 = _find_nth_swing_price(swing_high_mask, nq["high"].values, idx, 3)
    if not np.isnan(sh3) and sh3 > tp1 + 1.0:
        candidates.append(sh3)

    if candidates:
        tp2 = min(candidates)
    else:
        # Fallback: TP2 = entry + 1.5 * (TP1 - entry)
        tp2 = entry_price + (tp1 - entry_price) * 1.5

    return tp2


# ============================================================
# Full stateful simulation — collect accepted signals AND trades
# (exact replica of validate_nt_logic.py entry + exit logic)
# ============================================================
print("Running full engine simulation to identify accepted signals...")

accepted_signals = []

# Backtest state (needed to know which signals are blocked by position/daily limits)
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
trade_count = 0

for i in range(n):
    bar_date = dates[i]
    if bar_date != current_date:
        current_date = bar_date
        daily_pnl_r = 0.0
        consecutive_losses = 0
        day_stopped = False

    # ---- EXIT MANAGEMENT (same as validate_nt_logic.py) ----
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

            trade_count += 1
            daily_pnl_r += r_mult
            if exit_reason == "be_sweep" and pos_trimmed: pass
            elif r_mult < 0: consecutive_losses += 1
            else: consecutive_losses = 0
            if consecutive_losses >= max_consec_losses: day_stopped = True
            if daily_pnl_r <= -daily_max_loss_r: day_stopped = True
            in_position = False

    # ---- ENTRY FILTERS (same as validate_nt_logic.py) ----
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

        # Bias filter (exempt MSS+SMT)
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
        et_frac = et_idx[i].hour + et_idx[i].minute / 60.0

        # PM shorts block: no short entries after cutoff hour
        pm_block_cfg = params.get("pm_shorts_block", {})
        if pm_block_cfg.get("enabled", False) and direction == -1:
            pm_cutoff = pm_block_cfg.get("cutoff_hour", 14.0)
            if et_frac >= pm_cutoff:
                continue
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
        is_reduced = (et_idx[i].dayofweek in (0, 4)) or (regime_arr[i] < 1.0)
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

        # TP adjustments (same as validate_nt_logic.py)
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

        # Determine trim pct for this signal
        if mss_mgmt_enabled and is_mss_signal:
            sig_trim_pct = mss_mgmt.get("short_trim_pct", 1.0) if direction == -1 else mss_mgmt.get("long_trim_pct", trim_pct)
        elif dual_mode_enabled and direction == -1:
            sig_trim_pct = dual_mode.get("short_trim_pct", 1.0)
        elif dir_mgmt.get("enabled", False):
            sig_trim_pct = dir_mgmt.get("long_trim_pct", trim_pct) if direction == 1 else dir_mgmt.get("short_trim_pct", 1.0)
        else:
            sig_trim_pct = trim_pct

        # ---- COMPUTE MULTI-TP (TP2) FOR LONGS ----
        # Multi-TP applies to longs always; shorts only if apply_to_shorts=True
        is_multi_tp_signal = multi_tp_enabled and (
            direction == 1 or multi_tp_cfg.get("apply_to_shorts", False)
        )

        tp2_val = np.nan
        if is_multi_tp_signal:
            tp2_val = _build_liquidity_ladder_long(actual_entry, stop, tp1, i)
            sig_tp1_trim = multi_tp_cfg.get("tp1_trim_pct", 0.25)
            sig_tp2_trim = multi_tp_cfg.get("tp2_trim_pct", 0.25)
            sig_trail_nth = multi_tp_cfg.get("trail_nth_swing", 3)
        else:
            sig_tp1_trim = sig_trim_pct
            sig_tp2_trim = 0.0
            sig_trail_nth = nth_swing

        # ---- RECORD ACCEPTED SIGNAL ----
        bar_time_utc = nq.index[i]
        bar_time_et = et_idx[i]

        accepted_signals.append({
            "bar_time": str(bar_time_utc),
            "bar_time_et": bar_time_et.strftime("%Y-%m-%d %H:%M:%S"),
            "signal_dir": direction,
            "signal_type": str(sig_type[i]),
            "entry_price": round(actual_entry, 2),
            "raw_entry_price": round(entry_p, 2),
            "model_stop": round(stop, 2),
            "irl_target": round(irl_target_arr[i], 2),
            "tp1_adjusted": round(tp1, 2),
            "tp2_adjusted": round(tp2_val, 2) if not np.isnan(tp2_val) else np.nan,
            "is_multi_tp": is_multi_tp_signal,
            "tp1_trim_pct": round(sig_tp1_trim, 2),
            "tp2_trim_pct": round(sig_tp2_trim, 2),
            "trail_nth_swing": sig_trail_nth,
            "has_smt": bool(has_smt_arr[i]),
            "sweep_score": int(sweep_score_arr[i]),
            "bias_direction": int(bias_dir_arr[i]) if not np.isnan(bias_dir_arr[i]) else 0,
            "regime": float(regime_arr[i]),
            "grade": grade,
            "fluency": round(float(fluency_arr[i]), 4) if not np.isnan(fluency_arr[i]) else 0.0,
            "atr": round(float(atr_arr[i]), 4) if not np.isnan(atr_arr[i]) else 0.0,
            "signal_quality": round(float(signal_quality[i]), 4) if not np.isnan(signal_quality[i]) else 0.0,
            "contracts": contracts,
            "r_amount": round(r_amount, 2),
            "stop_distance": round(stop_dist, 2),
            "trim_pct": round(sig_trim_pct, 2),
            "is_reduced_risk": is_reduced,
        })

        # ---- ENTER POSITION (to maintain state for blocking/exit tracking) ----
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

# Force close open position at end (for correct trade count)
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
    trade_count += 1

# ============================================================
# Export
# ============================================================
export_df = pd.DataFrame(accepted_signals)
output_path = PROJECT / "ninjatrader" / "python_signals_v4_export.csv"
export_df.to_csv(output_path, index=False)

# Copy to C:/temp/ for NT8
import shutil
nt8_copy = Path("C:/temp/python_signals_v4_export.csv")
shutil.copy2(output_path, nt8_copy)
print(f"Copied to {nt8_copy}")

# ============================================================
# Summary
# ============================================================
print()
print("=" * 70)
print("SIGNAL EXPORT SUMMARY")
print("=" * 70)
print(f"\nTotal accepted signals exported: {len(export_df)}")
print(f"Total trades from engine:        {trade_count}")
print(f"Match: {'YES' if len(export_df) == trade_count else 'NO — MISMATCH'}")

print(f"\nBy signal type:")
for st in export_df["signal_type"].unique():
    mask = export_df["signal_type"] == st
    print(f"  {st:6s}: {mask.sum():4d} signals")

print(f"\nBy grade:")
for g in ["A+", "B+", "C"]:
    mask = export_df["grade"] == g
    print(f"  {g:3s}: {mask.sum():4d} signals")

print(f"\nBy direction:")
long_n = (export_df["signal_dir"] == 1).sum()
short_n = (export_df["signal_dir"] == -1).sum()
print(f"  Long:  {long_n}")
print(f"  Short: {short_n}")

print(f"\nBy multi-TP mode:")
multi_tp_n = export_df["is_multi_tp"].sum()
single_tp_n = len(export_df) - multi_tp_n
print(f"  Multi-TP:  {multi_tp_n}")
print(f"  Single-TP: {single_tp_n}")
if multi_tp_n > 0:
    mtp_mask = export_df["is_multi_tp"]
    tp2_vals = export_df.loc[mtp_mask, "tp2_adjusted"]
    tp2_valid = tp2_vals.dropna()
    print(f"  TP2 valid: {len(tp2_valid)} / {multi_tp_n}")
    if len(tp2_valid) > 0:
        tp1_for_mtp = export_df.loc[mtp_mask, "tp1_adjusted"].loc[tp2_valid.index]
        entry_for_mtp = export_df.loc[mtp_mask, "entry_price"].loc[tp2_valid.index]
        tp2_dist = (tp2_valid.values - entry_for_mtp.values)
        tp1_dist = (tp1_for_mtp.values - entry_for_mtp.values)
        avg_tp2_dist = np.mean(tp2_dist)
        avg_tp1_dist = np.mean(tp1_dist)
        print(f"  Avg TP1 dist: {avg_tp1_dist:.1f} pts")
        print(f"  Avg TP2 dist: {avg_tp2_dist:.1f} pts")
        print(f"  Avg TP2/TP1:  {avg_tp2_dist/avg_tp1_dist:.2f}x" if avg_tp1_dist > 0 else "  Avg TP2/TP1:  N/A")

print(f"\nBy has_smt:")
smt_n = export_df["has_smt"].sum()
print(f"  SMT:    {smt_n}")
print(f"  No SMT: {len(export_df) - smt_n}")

print(f"\nBy session:")
et_times = pd.to_datetime(export_df["bar_time_et"])
et_hours = et_times.dt.hour + et_times.dt.minute / 60.0
session_labels = pd.Series("Unknown", index=export_df.index)
session_labels[(et_hours >= 9.5) & (et_hours < 16.0)] = "NY"
session_labels[(et_hours >= 3.0) & (et_hours < 9.5)] = "London"
session_labels[(et_hours >= 18.0) | (et_hours < 3.0)] = "Asia"
for sess in ["NY", "London", "Asia", "Unknown"]:
    cnt = (session_labels == sess).sum()
    if cnt > 0:
        print(f"  {sess:8s}: {cnt:4d} signals")

print(f"\nPrice stats:")
print(f"  Entry price range: {export_df['entry_price'].min():.2f} - {export_df['entry_price'].max():.2f}")
print(f"  Avg stop distance: {export_df['stop_distance'].mean():.2f} pts")
print(f"  Avg ATR:           {export_df['atr'].mean():.2f}")
print(f"  Avg SQ:            {export_df['signal_quality'].mean():.4f}")
print(f"  Avg fluency:       {export_df['fluency'].mean():.4f}")

# Estimate total expected R (simple: TP distance / stop distance per signal)
tp_dist = np.abs(export_df["tp1_adjusted"].values - export_df["entry_price"].values)
sl_dist = export_df["stop_distance"].values
rr_per_signal = np.where(sl_dist > 0, tp_dist / sl_dist, 0.0)
total_expected_r = rr_per_signal.sum()
avg_rr = rr_per_signal.mean()
print(f"\nExpected R (if all TP1 hit):")
print(f"  Total expected R:  {total_expected_r:.1f}R")
print(f"  Avg RR per signal: {avg_rr:.2f}")

print(f"\nDate range: {export_df['bar_time_et'].iloc[0]} to {export_df['bar_time_et'].iloc[-1]}")

print(f"\nOutput: {output_path}")
print(f"NT8 copy: {nt8_copy}")
print("=" * 70)
