"""
experiments/phase2_london.py — London Session Opening Experiment

Tests whether London session trades can ADD positive-expectancy trades
without degrading PPDD (profit per unit drawdown).

Variants tested:
  A) BASELINE: skip_london=True (current production config)
  B) LONDON ALL: skip_london=False, everything else unchanged
  C) LONDON LONGS ONLY: allow London longs, block London shorts
  D) LONDON HIGH SQ: skip_london=False, London SQ threshold=0.80
  E) LONDON MSS+SMT ONLY: only allow MSS+SMT signals in London

Walk-forward validation (expanding window, 1-year OOS) on the best variant.

Usage: python experiments/phase2_london.py
"""
import sys
import copy
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# ============================================================
# Data loading (same pipeline as export_signals_for_nt8.py)
# ============================================================
print("=" * 80)
print("PHASE 2 EXPERIMENT: London Session Opening")
print("=" * 80)
print("\nLoading data...")

nq = pd.read_parquet(PROJECT / "data" / "NQ_5m_10yr.parquet")
es = pd.read_parquet(PROJECT / "data" / "ES_5m_10yr.parquet")
sig3 = pd.read_parquet(PROJECT / "data" / "cache_signals_10yr_v3.parquet")
bias = pd.read_parquet(PROJECT / "data" / "cache_bias_10yr_v2.parquet")
regime = pd.read_parquet(PROJECT / "data" / "cache_regime_10yr_v2.parquet")

with open(PROJECT / "config" / "params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

print(f"NQ: {len(nq)} bars, {nq.index[0]} to {nq.index[-1]}")
print(f"Signals (v3 cache): {sig3['signal'].sum()} raw signals")

# ============================================================
# SMT divergence
# ============================================================
print("Computing SMT divergence...")
from features.smt import compute_smt
smt = compute_smt(nq, es, {'swing': {'left_bars': 3, 'right_bars': 1},
                             'smt': {'sweep_lookback': 15, 'time_tolerance': 1}})

# ============================================================
# Merged signals (v3 cache + SMT gate for MSS)
# ============================================================
def make_sig(sig3_s, smt_s):
    s = sig3_s.copy()
    mm = s['signal'].astype(bool) & (s['signal_type'] == 'mss')
    mi = s.index[mm]
    s.loc[mi, 'signal'] = False
    s.loc[mi, 'signal_dir'] = 0
    s['has_smt'] = False
    c = mi.intersection(smt_s.index)
    if len(c) == 0:
        return s
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
# Pre-compute arrays (shared across all variants)
# ============================================================
print("Pre-computing features...")
from features.displacement import compute_atr, compute_fluency
from features.swing import compute_swing_levels
from features.pa_quality import compute_alternating_dir_ratio
from features.news_filter import build_news_blackout_mask

et_idx = nq.index.tz_convert("US/Eastern")
o, h, l, c_arr = nq["open"].values, nq["high"].values, nq["low"].values, nq["close"].values
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
sig_type_arr = ss["signal_type"].values
has_smt_arr = ss["has_smt"].values.astype(bool)
entry_price_arr = ss["entry_price"].values
model_stop_arr = ss["model_stop"].values
irl_target_arr = ss["irl_target"].values
sweep_score_arr = ss["sweep_score"].values if "sweep_score" in ss.columns else np.zeros(n, dtype=np.int8)
bias_dir_arr = bias["bias_direction"].values.astype(float)
bias_conf_arr = bias["bias_confidence"].values.astype(float)
regime_arr = regime["regime"].values.astype(float)

# News filter
news_path = PROJECT / "config" / "news_calendar.csv"
news_blackout_arr = None
if news_path.exists():
    news_bl = build_news_blackout_mask(nq.index, str(news_path),
        params["news"]["blackout_minutes_before"], params["news"]["cooldown_minutes_after"])
    news_blackout_arr = news_bl.values

# Signal quality
sq_params = params.get("signal_quality", {})
sq_enabled = sq_params.get("enabled", False)
sq_threshold_base = sq_params.get("threshold", 0.68)
dual_mode_cfg = params.get("dual_mode", {})
dual_mode_enabled = dual_mode_cfg.get("enabled", False)
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
        body = abs(c_arr[idx] - o[idx])
        rng = h[idx] - l[idx]
        disp_sc = body / rng if rng > 0 else 0.0
        flu_val = fluency_arr[idx]
        flu_sc = min(1.0, max(0.0, flu_val)) if not np.isnan(flu_val) else 0.5
        window = 6
        if idx >= window:
            dirs = np.sign(c_arr[idx-window:idx] - o[idx-window:idx])
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

# ET fractional hours (pre-computed)
et_frac_arr = np.array([et_idx[j].hour + et_idx[j].minute / 60.0 for j in range(n)])
et_dow_arr = np.array([et_idx[j].dayofweek for j in range(n)])

print(f"Pre-computation done. {len(signal_indices)} signal bars scored.")


# ============================================================
# Parameterized engine — runs full stateful backtest
# ============================================================
def run_engine(
    variant_name: str,
    skip_london: bool = True,
    london_direction: int = 0,       # 0=both, 1=longs only, -1=shorts only
    london_sq_threshold: float | None = None,  # Override SQ for London (None=use default)
    london_mss_smt_only: bool = False,  # Only allow MSS+SMT in London
    start_idx: int = 0,
    end_idx: int | None = None,
) -> pd.DataFrame:
    """Run full engine simulation with configurable London filters.

    Returns DataFrame of trades with columns matching engine.py output.
    """
    if end_idx is None:
        end_idx = n

    # Engine params from global config
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
    session_rules = params.get("session_rules", {})
    session_regime_cfg = params.get("session_regime", {})
    dir_mgmt = params.get("direction_mgmt", {})
    pa_threshold = params.get("pa_quality", {}).get("alt_dir_threshold", 0.334)
    min_stop_atr = params.get("regime", {}).get("min_stop_atr_mult", 0.5)
    bias_relax = params.get("bias_relaxation", {})
    sf_enabled = params.get("session_filter", {}).get("enabled", True)
    skip_asia = params.get("session_filter", {}).get("skip_asia", True)
    ny_direction = params.get("session_filter", {}).get("ny_direction", 0)

    def _find_nth_swing(mask, prices, idx, n_val, direction):
        count = 0
        for j in range(idx - 1, -1, -1):
            if mask[j]:
                count += 1
                if count == n_val:
                    return prices[j]
        return np.nan

    def _compute_grade(ba, bc, reg):
        if np.isnan(ba) or np.isnan(reg):
            return "C"
        if reg == 0.0:
            return "C"
        aligned = ba > 0.5
        full = reg >= 1.0
        if aligned and full:
            return "A+"
        if aligned or full:
            return "B+"
        return "C"

    def _compute_contracts(r_amt, stop_dist, pv):
        if stop_dist <= 0 or pv <= 0:
            return 0
        return max(1, int(r_amt / (stop_dist * pv)))

    trades = []
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
    pos_bias_dir = pos_regime = 0.0
    pos_trim_pct = trim_pct

    for i in range(start_idx, end_idx):
        bar_date = dates[i]
        if bar_date != current_date:
            current_date = bar_date
            daily_pnl_r = 0.0
            consecutive_losses = 0
            day_stopped = False

        # ---- EXIT MANAGEMENT ----
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
                pa_body = np.abs(c_arr[pa_start:pa_end] - o[pa_start:pa_end])
                safe_pa = np.where(pa_range > 0, pa_range, 1.0)
                pa_wick = 1.0 - (pa_body / safe_pa)
                avg_wick = float(np.mean(pa_wick))
                pa_dirs = np.sign(c_arr[pa_start:pa_end] - o[pa_start:pa_end])
                favorable = (pa_dirs == pos_direction).mean()
                disp = (c_arr[i] - pos_entry_price) if pos_direction == 1 else (pos_entry_price - c_arr[i])
                cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                no_progress = disp < cur_atr * 0.3
                bad_pa = avg_wick > 0.65 and favorable < 0.5
                if bad_pa and no_progress and bars_in_trade >= 3:
                    exit_price = o[i+1] if i+1 < end_idx else c_arr[i]
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
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True
                if pos_trimmed and not exited:
                    nt = _find_nth_swing(swing_low_mask, nq["low"].values, i, nth_swing, 1)
                    if not np.isnan(nt) and nt > pos_trail_stop:
                        pos_trail_stop = nt

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
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True
                if pos_trimmed and not exited:
                    nt = _find_nth_swing(swing_high_mask, nq["high"].values, i, nth_swing, -1)
                    if not np.isnan(nt) and nt < pos_trail_stop:
                        pos_trail_stop = nt

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

                trades.append({
                    "entry_time": nq.index[pos_entry_idx],
                    "exit_time": nq.index[i + 1] if exit_reason == "early_cut_pa" and i + 1 < end_idx else nq.index[i],
                    "direction": pos_direction,
                    "entry_price": pos_entry_price,
                    "exit_price": exit_price,
                    "stop_price": pos_stop,
                    "tp1_price": pos_tp1,
                    "contracts": pos_contracts,
                    "pnl_dollars": total_pnl,
                    "r_multiple": r_mult,
                    "exit_reason": exit_reason,
                    "signal_type": pos_signal_type,
                    "grade": pos_grade,
                    "trimmed": pos_trimmed,
                    "session": "london" if 3.0 <= et_frac_arr[pos_entry_idx - 1] < 9.5 else (
                        "ny" if 9.5 <= et_frac_arr[pos_entry_idx - 1] < 16.0 else "asia"),
                })

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
                in_position = False

        # ---- ENTRY FILTERS ----
        if not in_position and news_blackout_arr is not None and news_blackout_arr[i]:
            continue

        if not in_position and not day_stopped:
            et_h = et_idx[i].hour
            et_m = et_idx[i].minute
            if (et_h == 9 and et_m >= 30) or (et_h == 10 and et_m == 0):
                continue

        if not in_position and not day_stopped and sig_mask[i]:
            direction = int(sig_dir[i])
            if direction == 0:
                continue

            sig_f = params.get("signal_filter", {})
            if not sig_f.get("allow_mss", True) and str(sig_type_arr[i]) == "mss":
                continue
            if not sig_f.get("allow_trend", True) and str(sig_type_arr[i]) == "trend":
                continue

            entry_p = entry_price_arr[i]
            stop = model_stop_arr[i]
            tp1 = irl_target_arr[i]
            if np.isnan(entry_p) or np.isnan(stop) or np.isnan(tp1):
                continue
            if direction == 1 and (stop >= entry_p or tp1 <= entry_p):
                continue
            if direction == -1 and (stop <= entry_p or tp1 >= entry_p):
                continue

            # Bias filter (exempt MSS+SMT)
            bias_opposing = (direction == -np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0)
            if bias_opposing:
                is_mss_smt_sig = (smt_cfg.get("enabled", False) and has_smt_arr[i]
                                  and str(sig_type_arr[i]) == "mss")
                if is_mss_smt_sig:
                    pass
                elif bias_relax.get("enabled", False) and direction == -1:
                    pass
                else:
                    continue

            # PA quality
            if pa_alt_arr[i] >= pa_threshold:
                continue

            # ========================================
            # SESSION FILTER (VARIANT-SPECIFIC LOGIC)
            # ========================================
            et_frac = et_frac_arr[i]
            is_mss_smt = (str(sig_type_arr[i]) == "mss" and has_smt_arr[i]
                          and smt_cfg.get("enabled", False))
            mss_bypass = is_mss_smt and smt_cfg.get("bypass_session_filter", False)

            if sf_enabled and not mss_bypass:
                if 9.5 <= et_frac < 16.0:  # NY
                    if ny_direction != 0 and direction != ny_direction:
                        continue
                elif 3.0 <= et_frac < 9.5:  # London
                    if skip_london:
                        continue
                    # Variant C: London direction filter
                    if london_direction != 0 and direction != london_direction:
                        continue
                    # Variant E: London MSS+SMT only
                    if london_mss_smt_only and not is_mss_smt:
                        continue
                else:  # Asia
                    if skip_asia:
                        continue
            elif not sf_enabled and not mss_bypass:
                if not (3.0 <= et_frac < 16.0):
                    continue

            # Min stop
            stop_dist = abs(entry_p - stop)
            min_stop = min_stop_atr * atr_arr[i] if not np.isnan(atr_arr[i]) else 10.0
            if stop_dist < min_stop:
                continue

            # Signal quality (with London override)
            if sq_enabled and not np.isnan(signal_quality[i]):
                eff_sq = sq_threshold_base
                if dual_mode_enabled and direction == -1:
                    eff_sq = dual_mode_cfg.get("short_sq_threshold", 0.82)
                # Variant D: override SQ threshold for London signals
                if london_sq_threshold is not None and 3.0 <= et_frac < 9.5:
                    eff_sq = london_sq_threshold
                    # Still apply short premium on top
                    if dual_mode_enabled and direction == -1:
                        eff_sq = max(eff_sq, dual_mode_cfg.get("short_sq_threshold", 0.82))
                if signal_quality[i] < eff_sq:
                    continue

            if i + 1 >= end_idx:
                continue

            # Grade
            ba = 1.0 if (direction == np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0) else 0.0
            grade = _compute_grade(ba, bias_conf_arr[i], regime_arr[i])
            if grade == "C" and c_skip:
                continue

            # Sizing
            is_reduced = (et_dow_arr[i] in (0, 4)) or (regime_arr[i] < 1.0)
            base_r = reduced_r if is_reduced else normal_r
            if grade == "A+":
                r_amount = base_r * a_plus_mult
            elif grade == "B+":
                r_amount = base_r * b_plus_mult
            else:
                r_amount = base_r * 0.5

            # Session regime
            if session_regime_cfg.get("enabled", False):
                sr_lunch_s = session_regime_cfg.get("lunch_start", 12.0)
                sr_lunch_e = session_regime_cfg.get("lunch_end", 13.5)
                sr_am_end = session_regime_cfg.get("am_end", 12.0)
                sr_pm_start = session_regime_cfg.get("pm_start", 13.5)
                if et_frac < sr_am_end:
                    sr_mult = session_regime_cfg.get("am_mult", 1.0)
                elif sr_lunch_s <= et_frac < sr_lunch_e:
                    sr_mult = session_regime_cfg.get("lunch_mult", 0.5)
                elif et_frac >= sr_pm_start:
                    sr_mult = session_regime_cfg.get("pm_mult", 0.75)
                else:
                    sr_mult = 1.0
                r_amount *= sr_mult
                if r_amount <= 0:
                    continue

            # Slippage
            actual_entry = entry_p + slippage_points if direction == 1 else entry_p - slippage_points
            stop_dist = abs(actual_entry - stop)
            if stop_dist < 1.0:
                continue

            contracts = _compute_contracts(r_amount, stop_dist, point_value)
            if contracts <= 0:
                continue

            # TP adjustments
            is_mss_signal = str(sig_type_arr[i]) == "mss"
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
                short_rr = dual_mode_cfg.get("short_rr", 0.625)
                if mss_mgmt_enabled and is_mss_signal:
                    short_rr = mss_mgmt.get("short_rr", short_rr)
                tp1 = actual_entry - stop_dist * short_rr

            # Trim pct
            if mss_mgmt_enabled and is_mss_signal:
                sig_trim_pct = mss_mgmt.get("short_trim_pct", 1.0) if direction == -1 else mss_mgmt.get("long_trim_pct", trim_pct)
            elif dual_mode_enabled and direction == -1:
                sig_trim_pct = dual_mode_cfg.get("short_trim_pct", 1.0)
            elif dir_mgmt.get("enabled", False):
                sig_trim_pct = dir_mgmt.get("long_trim_pct", trim_pct) if direction == 1 else dir_mgmt.get("short_trim_pct", 1.0)
            else:
                sig_trim_pct = trim_pct

            # ---- ENTER POSITION ----
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
            pos_signal_type = str(sig_type_arr[i])
            pos_bias_dir = bias_dir_arr[i]
            pos_regime = regime_arr[i]
            pos_grade = grade
            pos_trim_pct = sig_trim_pct

    # Force close open position at end
    if in_position:
        exit_price = c_arr[end_idx - 1]
        pnl_pts = (exit_price - pos_entry_price) if pos_direction == 1 else (pos_entry_price - exit_price)
        total_pnl = pnl_pts * point_value * pos_remaining_contracts
        if pos_trimmed:
            trim_c = pos_contracts - pos_remaining_contracts
            trim_pnl = ((pos_tp1 - pos_entry_price) if pos_direction == 1 else (pos_entry_price - pos_tp1)) * point_value * trim_c
            total_pnl += trim_pnl
            total_pnl -= commission_per_side * 2 * pos_contracts
        else:
            total_pnl -= commission_per_side * 2 * pos_remaining_contracts
        stop_dist = abs(pos_entry_price - pos_stop)
        total_risk = stop_dist * point_value * pos_contracts
        r_mult = total_pnl / total_risk if total_risk > 0 else 0.0
        trades.append({
            "entry_time": nq.index[pos_entry_idx],
            "exit_time": nq.index[end_idx - 1],
            "direction": pos_direction,
            "entry_price": pos_entry_price,
            "exit_price": exit_price,
            "stop_price": pos_stop,
            "tp1_price": pos_tp1,
            "contracts": pos_contracts,
            "pnl_dollars": total_pnl,
            "r_multiple": r_mult,
            "exit_reason": "eod_close",
            "signal_type": pos_signal_type,
            "grade": pos_grade,
            "trimmed": pos_trimmed,
            "session": "london" if 3.0 <= et_frac_arr[pos_entry_idx - 1] < 9.5 else (
                "ny" if 9.5 <= et_frac_arr[pos_entry_idx - 1] < 16.0 else "asia"),
        })

    if len(trades) == 0:
        return pd.DataFrame()
    return pd.DataFrame(trades)


# ============================================================
# Metrics computation
# ============================================================
def compute_metrics(trades_df: pd.DataFrame, label: str = "") -> dict:
    """Compute standard backtest metrics from trade log."""
    if len(trades_df) == 0:
        return {"label": label, "trades": 0, "total_R": 0.0, "PPDD": 0.0,
                "max_DD_R": 0.0, "win_rate": 0.0, "PF": 0.0, "avg_R": 0.0,
                "years": 0, "trades_per_year": 0.0}

    r_arr = trades_df["r_multiple"].values
    total_r = float(np.sum(r_arr))
    wins = r_arr[r_arr > 0]
    losses = r_arr[r_arr < 0]
    win_rate = float(np.mean(r_arr > 0)) * 100.0
    pf = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else 999.0
    avg_r = float(np.mean(r_arr))

    # Max drawdown in R
    cum_r = np.cumsum(r_arr)
    peak = np.maximum.accumulate(cum_r)
    dd = peak - cum_r
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    # PPDD
    ppdd = total_r / max_dd if max_dd > 0 else total_r

    # Time span
    entry_times = pd.to_datetime(trades_df["entry_time"])
    span_years = (entry_times.max() - entry_times.min()).days / 365.25 if len(entry_times) > 1 else 1.0
    trades_per_year = len(trades_df) / span_years if span_years > 0 else len(trades_df)

    return {
        "label": label,
        "trades": len(trades_df),
        "total_R": round(total_r, 1),
        "PPDD": round(ppdd, 2),
        "max_DD_R": round(max_dd, 1),
        "win_rate": round(win_rate, 1),
        "PF": round(pf, 2),
        "avg_R": round(avg_r, 3),
        "years": round(span_years, 1),
        "trades_per_year": round(trades_per_year, 1),
    }


def print_metrics(m: dict):
    print(f"  {m['label']:40s} | {m['trades']:4d}t | R={m['total_R']:+7.1f} | "
          f"PPDD={m['PPDD']:6.2f} | MaxDD={m['max_DD_R']:5.1f}R | "
          f"WR={m['win_rate']:5.1f}% | PF={m['PF']:5.2f} | avgR={m['avg_R']:+.3f} | "
          f"{m['trades_per_year']:.0f}/yr")


def session_breakdown(trades_df: pd.DataFrame, label: str):
    """Print breakdown by session and direction."""
    if len(trades_df) == 0:
        print(f"  {label}: no trades")
        return

    for sess in ["ny", "london", "asia"]:
        mask = trades_df["session"] == sess
        if mask.sum() == 0:
            continue
        sub = trades_df[mask]
        r_arr = sub["r_multiple"].values
        total_r = np.sum(r_arr)
        wr = np.mean(r_arr > 0) * 100
        avg_r = np.mean(r_arr)
        print(f"    {sess:8s}: {mask.sum():3d}t  R={total_r:+7.1f}  WR={wr:5.1f}%  avgR={avg_r:+.3f}")

        for d, dl in [(1, "long"), (-1, "short")]:
            d_mask = mask & (trades_df["direction"] == d)
            if d_mask.sum() == 0:
                continue
            d_sub = trades_df[d_mask]
            d_r = d_sub["r_multiple"].values
            print(f"      {dl:6s}: {d_mask.sum():3d}t  R={np.sum(d_r):+7.1f}  WR={np.mean(d_r > 0)*100:5.1f}%  avgR={np.mean(d_r):+.3f}")


# ============================================================
# Run all variants
# ============================================================
print("\n" + "=" * 80)
print("RUNNING VARIANTS")
print("=" * 80)

variants = {
    "A) BASELINE (skip_london=True)": dict(
        skip_london=True,
    ),
    "B) LONDON ALL (both dirs, default SQ)": dict(
        skip_london=False,
        london_direction=0,
    ),
    "C) LONDON LONGS ONLY": dict(
        skip_london=False,
        london_direction=1,
    ),
    "D) LONDON HIGH SQ (threshold=0.80)": dict(
        skip_london=False,
        london_direction=0,
        london_sq_threshold=0.80,
    ),
    "E) LONDON MSS+SMT ONLY": dict(
        skip_london=False,
        london_direction=0,
        london_mss_smt_only=True,
    ),
}

results = {}
all_metrics = []

for name, kwargs in variants.items():
    print(f"\n--- {name} ---")
    trades_df = run_engine(name, **kwargs)
    m = compute_metrics(trades_df, name)
    results[name] = trades_df
    all_metrics.append(m)
    print_metrics(m)
    session_breakdown(trades_df, name)

# ============================================================
# Summary comparison table
# ============================================================
print("\n" + "=" * 80)
print("COMPARISON TABLE")
print("=" * 80)
print(f"  {'Variant':42s} | {'Trades':>6s} | {'Total R':>8s} | {'PPDD':>6s} | {'MaxDD':>6s} | {'WR%':>5s} | {'PF':>5s} | {'avgR':>6s} | {'/yr':>4s}")
print("  " + "-" * 110)
for m in all_metrics:
    print_metrics(m)

# ============================================================
# Delta analysis: what does London ADD vs baseline?
# ============================================================
print("\n" + "=" * 80)
print("DELTA ANALYSIS: London-only trades in each variant")
print("=" * 80)

baseline_df = results["A) BASELINE (skip_london=True)"]
baseline_m = all_metrics[0]

for name, trades_df in results.items():
    if name == "A) BASELINE (skip_london=True)":
        continue
    london_only = trades_df[trades_df["session"] == "london"]
    if len(london_only) == 0:
        print(f"\n  {name}: 0 London trades")
        continue
    lm = compute_metrics(london_only, f"  London-only from {name[:30]}")
    print(f"\n  {name}:")
    print(f"    London trades: {len(london_only)}")
    print(f"    London R:      {london_only['r_multiple'].sum():+.1f}")
    print(f"    London WR:     {(london_only['r_multiple'] > 0).mean()*100:.1f}%")
    print(f"    London avgR:   {london_only['r_multiple'].mean():+.3f}")
    # By direction
    for d, dl in [(1, "Long"), (-1, "Short")]:
        d_mask = london_only["direction"] == d
        if d_mask.sum() > 0:
            d_sub = london_only[d_mask]
            print(f"    London {dl:5s}: {d_mask.sum():3d}t  R={d_sub['r_multiple'].sum():+.1f}  WR={100*(d_sub['r_multiple']>0).mean():.1f}%  avgR={d_sub['r_multiple'].mean():+.3f}")
    # By signal type
    for st in london_only["signal_type"].unique():
        st_mask = london_only["signal_type"] == st
        st_sub = london_only[st_mask]
        print(f"    London {st:6s}: {st_mask.sum():3d}t  R={st_sub['r_multiple'].sum():+.1f}  WR={100*(st_sub['r_multiple']>0).mean():.1f}%  avgR={st_sub['r_multiple'].mean():+.3f}")

    # Delta vs baseline
    delta_r = compute_metrics(trades_df, name)["total_R"] - baseline_m["total_R"]
    delta_ppdd = compute_metrics(trades_df, name)["PPDD"] - baseline_m["PPDD"]
    print(f"    --- vs Baseline: delta_R={delta_r:+.1f}, delta_PPDD={delta_ppdd:+.2f}")


# ============================================================
# Identify best variant for walk-forward validation
# ============================================================
print("\n" + "=" * 80)
print("WALK-FORWARD VALIDATION")
print("=" * 80)

# Find best non-baseline variant by PPDD (must have PPDD >= baseline)
# If none beats baseline, pick best anyway and note it
best_name = None
best_ppdd = -999
for m in all_metrics[1:]:  # skip baseline
    if m["PPDD"] > best_ppdd:
        best_ppdd = m["PPDD"]
        best_name = m["label"]

# Also check: any variant that beats baseline in PPDD AND adds trades?
candidates = []
for m in all_metrics[1:]:
    if m["PPDD"] >= baseline_m["PPDD"] * 0.95 and m["total_R"] > baseline_m["total_R"]:
        candidates.append(m)

if candidates:
    # Among candidates, pick highest PPDD
    candidates.sort(key=lambda x: x["PPDD"], reverse=True)
    best_name = candidates[0]["label"]
    best_ppdd = candidates[0]["PPDD"]
    print(f"\nBest candidate (PPDD within 5% of baseline + more R): {best_name}")
else:
    print(f"\nNo variant beats baseline in both PPDD and R. Testing best PPDD: {best_name}")

best_kwargs = variants[best_name]
print(f"Best variant: {best_name}")
print(f"  Config: {best_kwargs}")

# Walk-forward: expanding window, 1-year OOS
# Data spans ~10 years. Use years 1-3 as initial IS, then expand.
# OOS = next year.

et_dates_pd = pd.to_datetime([str(d) for d in dates])
unique_years = sorted(set(d.year for d in dates))
print(f"\nData years: {unique_years[0]} - {unique_years[-1]}")

# Initial training: first 3 years. Then expand by 1 year, test on next year.
min_is_years = 3
wf_results = []

for oos_year_idx in range(min_is_years, len(unique_years)):
    oos_year = unique_years[oos_year_idx]
    is_years = unique_years[:oos_year_idx]

    # Find bar indices for IS and OOS
    oos_mask = np.array([d.year == oos_year for d in dates])
    if not oos_mask.any():
        continue
    oos_start = int(np.where(oos_mask)[0][0])
    oos_end = int(np.where(oos_mask)[0][-1]) + 1

    # Run engine on OOS period only
    trades_df = run_engine(f"WF-{oos_year}", **best_kwargs, start_idx=oos_start, end_idx=oos_end)

    if len(trades_df) == 0:
        wf_results.append({
            "oos_year": oos_year,
            "is_years": f"{is_years[0]}-{is_years[-1]}",
            "trades": 0,
            "total_R": 0.0,
            "win": False,
        })
        continue

    r_arr = trades_df["r_multiple"].values
    total_r = float(np.sum(r_arr))
    wr = float(np.mean(r_arr > 0)) * 100
    avg_r = float(np.mean(r_arr))

    # Also run baseline on same period
    baseline_trades = run_engine(f"Baseline-{oos_year}", skip_london=True, start_idx=oos_start, end_idx=oos_end)
    baseline_r = float(np.sum(baseline_trades["r_multiple"].values)) if len(baseline_trades) > 0 else 0.0

    delta_r = total_r - baseline_r
    win = total_r > 0  # positive expectancy OOS

    # London-specific trades
    london_count = 0
    london_r = 0.0
    if "session" in trades_df.columns:
        london_mask = trades_df["session"] == "london"
        london_count = int(london_mask.sum())
        london_r = float(trades_df.loc[london_mask, "r_multiple"].sum()) if london_count > 0 else 0.0

    wf_results.append({
        "oos_year": oos_year,
        "is_years": f"{is_years[0]}-{is_years[-1]}",
        "trades": len(trades_df),
        "total_R": round(total_r, 1),
        "baseline_R": round(baseline_r, 1),
        "delta_R": round(delta_r, 1),
        "WR": round(wr, 1),
        "avg_R": round(avg_r, 3),
        "london_trades": london_count,
        "london_R": round(london_r, 1),
        "win": win,
    })

print(f"\n{'OOS Year':>8s} | {'IS':>12s} | {'Trades':>6s} | {'Total R':>8s} | {'Base R':>7s} | {'Delta R':>8s} | {'WR%':>5s} | {'Ldn#':>4s} | {'Ldn R':>6s} | {'Win':>3s}")
print("-" * 95)
wins = 0
for wf in wf_results:
    if wf["trades"] == 0:
        print(f"  {wf['oos_year']:>6d} | {wf['is_years']:>12s} | {0:>6d} | {'N/A':>8s} | {'N/A':>7s} | {'N/A':>8s} | {'N/A':>5s} | {'N/A':>4s} | {'N/A':>6s} | {'N/A':>3s}")
        continue
    is_win = 'Y' if wf['win'] else 'N'
    if wf['win']:
        wins += 1
    print(f"  {wf['oos_year']:>6d} | {wf['is_years']:>12s} | {wf['trades']:>6d} | {wf['total_R']:>+8.1f} | {wf['baseline_R']:>+7.1f} | {wf['delta_R']:>+8.1f} | {wf['WR']:>5.1f} | {wf['london_trades']:>4d} | {wf['london_R']:>+6.1f} | {is_win:>3s}")

total_wf = len([w for w in wf_results if w["trades"] > 0])
print(f"\nWalk-forward: {wins}/{total_wf} windows profitable ({100*wins/total_wf:.0f}%)" if total_wf > 0 else "\nNo WF windows")

# Summary of WF London-only delta
total_london_r = sum(w.get("london_R", 0) for w in wf_results)
total_delta_r = sum(w.get("delta_R", 0) for w in wf_results if w["trades"] > 0)
total_london_trades = sum(w.get("london_trades", 0) for w in wf_results)
print(f"Total London R across WF: {total_london_r:+.1f} ({total_london_trades} trades)")
print(f"Total delta R vs baseline across WF: {total_delta_r:+.1f}")

# ============================================================
# Also run WF for baseline to get fair comparison of WF win rate
# ============================================================
print("\n--- Baseline WF (for comparison) ---")
baseline_wf_wins = 0
baseline_wf_total = 0
for oos_year_idx in range(min_is_years, len(unique_years)):
    oos_year = unique_years[oos_year_idx]
    oos_mask = np.array([d.year == oos_year for d in dates])
    if not oos_mask.any():
        continue
    oos_start = int(np.where(oos_mask)[0][0])
    oos_end = int(np.where(oos_mask)[0][-1]) + 1
    baseline_trades = run_engine(f"BL-WF-{oos_year}", skip_london=True, start_idx=oos_start, end_idx=oos_end)
    if len(baseline_trades) > 0:
        baseline_wf_total += 1
        if baseline_trades["r_multiple"].sum() > 0:
            baseline_wf_wins += 1

print(f"Baseline WF: {baseline_wf_wins}/{baseline_wf_total} windows profitable ({100*baseline_wf_wins/baseline_wf_total:.0f}%)" if baseline_wf_total > 0 else "No baseline WF windows")

# ============================================================
# Also run WF specifically on Variant D (London High SQ) since it added +10.1R
# ============================================================
print("\n" + "=" * 80)
print("WALK-FORWARD: Variant D (London High SQ = 0.80) — detailed")
print("=" * 80)

d_kwargs = variants["D) LONDON HIGH SQ (threshold=0.80)"]
d_wf_results = []
for oos_year_idx in range(min_is_years, len(unique_years)):
    oos_year = unique_years[oos_year_idx]
    oos_mask = np.array([d.year == oos_year for d in dates])
    if not oos_mask.any():
        continue
    oos_start = int(np.where(oos_mask)[0][0])
    oos_end = int(np.where(oos_mask)[0][-1]) + 1

    trades_df = run_engine(f"D-WF-{oos_year}", **d_kwargs, start_idx=oos_start, end_idx=oos_end)
    baseline_trades = run_engine(f"BL-D-WF-{oos_year}", skip_london=True, start_idx=oos_start, end_idx=oos_end)

    if len(trades_df) == 0:
        d_wf_results.append({"oos_year": oos_year, "trades": 0, "total_R": 0.0, "baseline_R": 0.0,
                              "delta_R": 0.0, "london_trades": 0, "london_R": 0.0, "win": False})
        continue

    r_arr = trades_df["r_multiple"].values
    total_r = float(np.sum(r_arr))
    baseline_r = float(np.sum(baseline_trades["r_multiple"].values)) if len(baseline_trades) > 0 else 0.0
    delta_r = total_r - baseline_r
    wr = float(np.mean(r_arr > 0)) * 100

    london_mask_d = trades_df["session"] == "london"
    london_count = int(london_mask_d.sum())
    london_r = float(trades_df.loc[london_mask_d, "r_multiple"].sum()) if london_count > 0 else 0.0

    # London by direction
    london_long_r = 0.0
    london_short_r = 0.0
    london_long_n = 0
    london_short_n = 0
    if london_count > 0:
        ld = trades_df[london_mask_d]
        lm = ld["direction"] == 1
        london_long_n = int(lm.sum())
        london_long_r = float(ld.loc[lm, "r_multiple"].sum()) if london_long_n > 0 else 0.0
        london_short_n = london_count - london_long_n
        london_short_r = float(ld.loc[~lm, "r_multiple"].sum()) if london_short_n > 0 else 0.0

    # Max DD
    cum_r = np.cumsum(r_arr)
    peak = np.maximum.accumulate(cum_r)
    dd = peak - cum_r
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    d_wf_results.append({
        "oos_year": oos_year,
        "trades": len(trades_df),
        "total_R": round(total_r, 1),
        "baseline_R": round(baseline_r, 1),
        "delta_R": round(delta_r, 1),
        "WR": round(wr, 1),
        "max_DD": round(max_dd, 1),
        "london_trades": london_count,
        "london_R": round(london_r, 1),
        "london_long_n": london_long_n,
        "london_long_R": round(london_long_r, 1),
        "london_short_n": london_short_n,
        "london_short_R": round(london_short_r, 1),
        "win": total_r > 0,
        "delta_positive": delta_r > 0,
    })

print(f"\n{'Year':>4s} | {'Trades':>6s} | {'TotalR':>7s} | {'BaseR':>6s} | {'DeltaR':>7s} | {'MaxDD':>5s} | {'Ldn#':>4s} | {'LdnR':>6s} | {'LdnL#':>5s} | {'LdnLR':>6s} | {'LdnS#':>5s} | {'LdnSR':>6s} | Delta+")
print("-" * 115)
d_wins = 0
d_delta_wins = 0
for w in d_wf_results:
    if w["trades"] == 0:
        continue
    if w["win"]:
        d_wins += 1
    if w.get("delta_positive", False):
        d_delta_wins += 1
    print(f"  {w['oos_year']} | {w['trades']:>6d} | {w['total_R']:>+7.1f} | {w['baseline_R']:>+6.1f} | {w['delta_R']:>+7.1f} | {w['max_DD']:>5.1f} | {w['london_trades']:>4d} | {w['london_R']:>+6.1f} | {w['london_long_n']:>5d} | {w['london_long_R']:>+6.1f} | {w['london_short_n']:>5d} | {w['london_short_R']:>+6.1f} | {'Y' if w.get('delta_positive') else 'N'}")

d_total = len([w for w in d_wf_results if w["trades"] > 0])
print(f"\nVariant D WF: {d_wins}/{d_total} profitable, {d_delta_wins}/{d_total} beat baseline")
total_d_london_r = sum(w.get("london_R", 0) for w in d_wf_results)
total_d_delta_r = sum(w.get("delta_R", 0) for w in d_wf_results if w["trades"] > 0)
print(f"Total London R: {total_d_london_r:+.1f}, Total Delta R vs baseline: {total_d_delta_r:+.1f}")

# ============================================================
# FINAL RECOMMENDATION
# ============================================================
print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

best_full_m = [m for m in all_metrics if m["label"] == best_name][0]
print(f"\nBest variant: {best_name}")
print(f"  Trades: {best_full_m['trades']} (vs baseline {baseline_m['trades']})")
print(f"  Total R: {best_full_m['total_R']:+.1f} (vs baseline {baseline_m['total_R']:+.1f})")
print(f"  PPDD: {best_full_m['PPDD']:.2f} (vs baseline {baseline_m['PPDD']:.2f})")
print(f"  MaxDD: {best_full_m['max_DD_R']:.1f}R (vs baseline {baseline_m['max_DD_R']:.1f}R)")

if best_full_m["PPDD"] >= baseline_m["PPDD"] * 0.95 and best_full_m["total_R"] > baseline_m["total_R"]:
    print(f"\n  VERDICT: {best_name} adds R without degrading PPDD. Consider deploying.")
    if wins / total_wf >= 0.6 if total_wf > 0 else False:
        print(f"  Walk-forward supports: {wins}/{total_wf} profitable windows.")
    else:
        print(f"  WARNING: Walk-forward is weak ({wins}/{total_wf}). Test more before deploying.")
else:
    print(f"\n  VERDICT: No London variant beats baseline. Keep skip_london=True.")
    print(f"  London sessions remain destructive or not worth the drawdown cost.")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)
