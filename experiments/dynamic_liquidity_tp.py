"""
experiments/dynamic_liquidity_tp.py — Dynamic TP based on liquidity levels.

Instead of fixed IRL × 2.0 for longs, set TP at the NEAREST significant
liquidity level in the trade direction that's at least 1R away.

Liquidity level types (priority order):
  1. Session high/low (overnight, London, Asia) — strongest magnets
  2. Next swing high/low beyond the IRL target
  3. HTF swing levels (computed with larger left_bars, e.g., left=10)

Variants tested:
  A. Nearest session level (fallback to IRL × 2.0)
  B. Nearest HTF swing (left=10, fallback to IRL × 2.0)
  C. Best of session OR HTF swing (whichever closer, >= 1R)
  D. Adaptive multiplier based on nearest liquidity (capped at 3.0x)
  E. Baseline: current IRL × 2.0

Usage: python experiments/dynamic_liquidity_tp.py
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import logging
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    stream=sys.stdout)

DATA = PROJECT / "data"
CONFIG = PROJECT / "config" / "params.yaml"


# ======================================================================
# Data loading — extends validate_improvements.load_all()
# ======================================================================
def load_all():
    """Load all data needed for backtest + session levels + HTF swings."""
    t0 = _time.perf_counter()
    print("[LOAD] Loading data...")

    nq = pd.read_parquet(DATA / "NQ_5m_10yr.parquet")
    es = pd.read_parquet(DATA / "ES_5m_10yr.parquet")
    sig_v4 = pd.read_parquet(DATA / "cache_signals_10yr_v4.parquet")
    bias = pd.read_parquet(DATA / "cache_bias_10yr_v2.parquet")
    regime = pd.read_parquet(DATA / "cache_regime_10yr_v2.parquet")
    atr_cache = pd.read_parquet(DATA / "cache_atr_flu_10yr_v2.parquet")
    session_levels = pd.read_parquet(DATA / "cache_session_levels_10yr_v2.parquet")
    with open(CONFIG, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # SMT gate (same as validate_improvements.py)
    from features.smt import compute_smt
    smt = compute_smt(nq, es, {'swing': {'left_bars': 3, 'right_bars': 1},
                                'smt': {'sweep_lookback': 15, 'time_tolerance': 1}})

    ss = sig_v4.copy()
    mm = ss['signal'].astype(bool) & (ss['signal_type'] == 'mss')
    mi = ss.index[mm]
    ss.loc[mi, 'signal'] = False
    ss.loc[mi, 'signal_dir'] = 0
    ss['has_smt'] = False
    c_idx = mi.intersection(smt.index)
    if len(c_idx) > 0:
        md = sig_v4.loc[c_idx, 'signal_dir'].values
        ok = ((md == 1) & smt.loc[c_idx, 'smt_bull'].values.astype(bool)) | \
             ((md == -1) & smt.loc[c_idx, 'smt_bear'].values.astype(bool))
        g = c_idx[ok]
        ss.loc[g, 'signal'] = sig_v4.loc[g, 'signal']
        ss.loc[g, 'signal_dir'] = sig_v4.loc[g, 'signal_dir']
        ss.loc[g, 'has_smt'] = True

    # Kill MSS overnight
    rem = ss['signal'].astype(bool) & (ss['signal_type'] == 'mss')
    mi2 = ss.index[rem]
    if len(mi2) > 0:
        et = mi2.tz_convert('US/Eastern')
        ef = et.hour + et.minute / 60.0
        kill = (ef >= 16.0) | (ef < 3.0)
        if kill.any():
            ss.loc[mi2[kill], ['signal', 'signal_dir']] = [False, 0]

    n_gated = int(ss['signal'].sum())
    print(f"[LOAD] v4 signals after SMT gate: {n_gated}")

    # Pre-compute arrays
    from features.displacement import compute_atr, compute_fluency
    from features.swing import compute_swing_levels
    from features.pa_quality import compute_alternating_dir_ratio
    from features.news_filter import build_news_blackout_mask

    et_idx = nq.index.tz_convert("US/Eastern")
    o = nq["open"].values
    h = nq["high"].values
    l = nq["low"].values
    c = nq["close"].values
    n = len(nq)

    atr_arr = atr_cache["atr"].values
    fluency_arr = atr_cache["fluency"].values
    pa_alt_arr = compute_alternating_dir_ratio(nq, window=6).values

    # Standard swings (left=3, right=1)
    swing_p = {"left_bars": params["swing"]["left_bars"],
               "right_bars": params["swing"]["right_bars"]}
    swings = compute_swing_levels(nq, swing_p)
    swings["swing_high_price"] = swings["swing_high_price"].shift(1).ffill()
    swings["swing_low_price"] = swings["swing_low_price"].shift(1).ffill()
    swing_high_mask = swings["swing_high"].shift(1, fill_value=False).values
    swing_low_mask = swings["swing_low"].shift(1, fill_value=False).values

    # HTF swings (left=10, right=3) — bigger structure
    print("[LOAD] Computing HTF swings (left=10, right=3)...")
    htf_swing_p = {"left_bars": 10, "right_bars": 3}
    htf_swings = compute_swing_levels(nq, htf_swing_p)
    htf_swings["swing_high_price"] = htf_swings["swing_high_price"].shift(1).ffill()
    htf_swings["swing_low_price"] = htf_swings["swing_low_price"].shift(1).ffill()
    htf_swing_high_mask = htf_swings["swing_high"].shift(1, fill_value=False).values
    htf_swing_low_mask = htf_swings["swing_low"].shift(1, fill_value=False).values

    # Build swing price history arrays for finding 2nd/3rd nearest swings
    # swing_high_price_raw[i] = high[i] if swing_high at bar i, else NaN (shifted already)
    swing_high_raw = np.where(swing_high_mask, h, np.nan)
    swing_low_raw = np.where(swing_low_mask, l, np.nan)
    htf_swing_high_raw = np.where(htf_swing_high_mask, h, np.nan)
    htf_swing_low_raw = np.where(htf_swing_low_mask, l, np.nan)

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
    news_path = PROJECT / "config" / "news_calendar.csv"
    news_blackout_arr = None
    if news_path.exists():
        news_bl = build_news_blackout_mask(nq.index, str(news_path),
            params["news"]["blackout_minutes_before"],
            params["news"]["cooldown_minutes_after"])
        news_blackout_arr = news_bl.values

    # Signal quality
    sq_params = params.get("signal_quality", {})
    signal_quality = np.full(n, np.nan)
    signal_indices = np.where(sig_mask)[0]
    if len(signal_indices) > 0:
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
            signal_quality[idx] = (
                sq_params.get("w_size", 0.3) * size_sc
                + sq_params.get("w_disp", 0.3) * disp_sc
                + sq_params.get("w_flu", 0.2) * flu_sc
                + sq_params.get("w_pa", 0.2) * pa_sc
            )

    # Stop distance / ATR ratio
    stop_atr_ratio = np.full(n, np.nan)
    if len(signal_indices) > 0:
        for idx in signal_indices:
            a = atr_arr[idx] if not np.isnan(atr_arr[idx]) else 10.0
            if a > 0:
                stop_atr_ratio[idx] = abs(entry_price_arr[idx] - model_stop_arr[idx]) / a

    # Target RR ratio
    target_rr_arr = np.full(n, np.nan)
    if len(signal_indices) > 0:
        for idx in signal_indices:
            ep = entry_price_arr[idx]
            sp = model_stop_arr[idx]
            tp = irl_target_arr[idx]
            stop_d = abs(ep - sp)
            targ_d = abs(tp - ep)
            if stop_d > 0:
                target_rr_arr[idx] = targ_d / stop_d

    # Date/time arrays
    dates = np.array([
        (et_idx[j] + pd.Timedelta(days=1)).date() if et_idx[j].hour >= 18 else et_idx[j].date()
        for j in range(n)
    ])
    dow_arr = np.array([et_idx[j].dayofweek for j in range(n)])
    et_frac_arr = np.array([et_idx[j].hour + et_idx[j].minute / 60.0 for j in range(n)])

    ts_minutes = (nq.index.astype(np.int64) // (60 * 10**9)).values

    # Align session levels to nq index
    sess_aligned = session_levels.reindex(nq.index, method='ffill')
    sess_asia_high = sess_aligned["asia_high"].values
    sess_asia_low = sess_aligned["asia_low"].values
    sess_london_high = sess_aligned["london_high"].values
    sess_london_low = sess_aligned["london_low"].values
    sess_overnight_high = sess_aligned["overnight_high"].values
    sess_overnight_low = sess_aligned["overnight_low"].values

    elapsed = _time.perf_counter() - t0
    print(f"[LOAD] All data loaded in {elapsed:.1f}s (n={n} bars)")

    return {
        "nq": nq, "params": params, "n": n,
        "o": o, "h": h, "l": l, "c": c,
        "atr_arr": atr_arr, "fluency_arr": fluency_arr, "pa_alt_arr": pa_alt_arr,
        "swing_high_mask": swing_high_mask, "swing_low_mask": swing_low_mask,
        "sig_mask": sig_mask, "sig_dir": sig_dir, "sig_type": sig_type,
        "has_smt_arr": has_smt_arr,
        "entry_price_arr": entry_price_arr, "model_stop_arr": model_stop_arr,
        "irl_target_arr": irl_target_arr,
        "bias_dir_arr": bias_dir_arr, "bias_conf_arr": bias_conf_arr,
        "regime_arr": regime_arr,
        "news_blackout_arr": news_blackout_arr,
        "signal_quality": signal_quality, "stop_atr_ratio": stop_atr_ratio,
        "target_rr_arr": target_rr_arr,
        "dates": dates, "dow_arr": dow_arr, "et_frac_arr": et_frac_arr,
        "et_idx": et_idx, "ts_minutes": ts_minutes,
        # New: HTF swings
        "htf_swing_high_mask": htf_swing_high_mask,
        "htf_swing_low_mask": htf_swing_low_mask,
        "swing_high_raw": swing_high_raw,
        "swing_low_raw": swing_low_raw,
        "htf_swing_high_raw": htf_swing_high_raw,
        "htf_swing_low_raw": htf_swing_low_raw,
        # New: Session levels
        "sess_asia_high": sess_asia_high,
        "sess_asia_low": sess_asia_low,
        "sess_london_high": sess_london_high,
        "sess_london_low": sess_london_low,
        "sess_overnight_high": sess_overnight_high,
        "sess_overnight_low": sess_overnight_low,
    }


# ======================================================================
# Liquidity target helpers
# ======================================================================
def _find_nth_swing(mask, prices, idx, n_val, direction):
    """Find the nth most recent swing level (looking backward)."""
    count = 0
    for j in range(idx - 1, -1, -1):
        if mask[j]:
            count += 1
            if count == n_val:
                return prices[j]
    return np.nan


def _find_swing_levels_above(swing_raw, idx, entry_price, max_levels=3):
    """Find up to max_levels distinct swing high prices above entry, scanning backward."""
    found = []
    seen = set()
    for j in range(idx - 1, -1, -1):
        v = swing_raw[j]
        if np.isnan(v):
            continue
        if v > entry_price and v not in seen:
            found.append(v)
            seen.add(v)
            if len(found) >= max_levels:
                break
    found.sort()  # ascending: nearest first
    return found


def _find_swing_levels_below(swing_raw, idx, entry_price, max_levels=3):
    """Find up to max_levels distinct swing low prices below entry, scanning backward."""
    found = []
    seen = set()
    for j in range(idx - 1, -1, -1):
        v = swing_raw[j]
        if np.isnan(v):
            continue
        if v < entry_price and v not in seen:
            found.append(v)
            seen.add(v)
            if len(found) >= max_levels:
                break
    found.sort(reverse=True)  # descending: nearest (highest) first
    return found


def _nearest_session_level_above(i, entry_price, d):
    """Find nearest session level above entry price (for longs)."""
    candidates = []
    for key in ["sess_overnight_high", "sess_london_high", "sess_asia_high"]:
        v = d[key][i]
        if not np.isnan(v) and v > entry_price:
            candidates.append(v)
    if not candidates:
        return np.nan
    return min(candidates)  # nearest = smallest above entry


def _nearest_session_level_below(i, entry_price, d):
    """Find nearest session level below entry price (for shorts)."""
    candidates = []
    for key in ["sess_overnight_low", "sess_london_low", "sess_asia_low"]:
        v = d[key][i]
        if not np.isnan(v) and v < entry_price:
            candidates.append(v)
    if not candidates:
        return np.nan
    return max(candidates)  # nearest = largest below entry


def _nearest_htf_swing_above(i, entry_price, d):
    """Find nearest HTF swing high above entry (for longs)."""
    levels = _find_swing_levels_above(d["htf_swing_high_raw"], i, entry_price, max_levels=1)
    return levels[0] if levels else np.nan


def _nearest_htf_swing_below(i, entry_price, d):
    """Find nearest HTF swing low below entry (for shorts)."""
    levels = _find_swing_levels_below(d["htf_swing_low_raw"], i, entry_price, max_levels=1)
    return levels[0] if levels else np.nan


# ======================================================================
# Compute grade (same as validate_improvements.py)
# ======================================================================
def _compute_grade_fast(ba, regime):
    if np.isnan(ba) or np.isnan(regime) or regime == 0.0:
        return "C"
    aligned = ba > 0.5
    full = regime >= 1.0
    if aligned and full:
        return "A+"
    if aligned or full:
        return "B+"
    return "C"


# ======================================================================
# TP Strategy enum
# ======================================================================
TP_NEAREST_SESSION = "A"
TP_NEAREST_HTF_SWING = "B"
TP_BEST_OF_SESSION_OR_HTF = "C"
TP_ADAPTIVE_MULT = "D"
TP_BASELINE = "E"


# ======================================================================
# Backtest engine with dynamic TP
# ======================================================================
def run_backtest_dynamic_tp(
    d: dict,
    tp_strategy: str = TP_BASELINE,
    # Standard params (Config D)
    sq_long: float = 0.68,
    sq_short: float = 0.80,
    min_stop_atr: float = 1.7,
    block_pm_shorts: bool = True,
    # Date range
    start_date=None, end_date=None,
) -> list[dict]:
    """Backtest with dynamic TP selection for longs. Shorts keep dual_mode scalp."""

    params = d["params"]
    n = d["n"]
    o, h, l, c_arr = d["o"], d["h"], d["l"], d["c"]
    atr_arr = d["atr_arr"]
    sig_mask = d["sig_mask"]
    sig_dir = d["sig_dir"]
    sig_type = d["sig_type"]
    has_smt_arr = d["has_smt_arr"]
    entry_price_arr = d["entry_price_arr"]
    model_stop_arr = d["model_stop_arr"]
    irl_target_arr = d["irl_target_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    signal_quality = d["signal_quality"]
    stop_atr_ratio = d["stop_atr_ratio"]
    target_rr_arr = d["target_rr_arr"]
    dates = d["dates"]
    dow_arr = d["dow_arr"]
    et_frac_arr = d["et_frac_arr"]
    pa_alt_arr = d["pa_alt_arr"]
    swing_high_mask = d["swing_high_mask"]
    swing_low_mask = d["swing_low_mask"]
    nq = d["nq"]
    ts_minutes = d["ts_minutes"]

    # Config
    pos_params = params["position"]
    risk_params = params["risk"]
    bt_params = params["backtest"]
    grading_params = params["grading"]
    trim_params = params["trim"]
    trail_params = params["trail"]
    smt_cfg = params.get("smt", {})
    mss_mgmt = params.get("mss_management", {})
    mss_mgmt_enabled = mss_mgmt.get("enabled", False)
    dual_mode = params.get("dual_mode", {})
    session_rules = params.get("session_rules", {})
    session_regime = params.get("session_regime", {})
    dir_mgmt = params.get("direction_mgmt", {})
    bias_relax = params.get("bias_relaxation", {})
    pa_threshold = params.get("pa_quality", {}).get("alt_dir_threshold", 1.0)

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

    # Date range
    start_idx = 0
    end_idx = n
    if start_date is not None:
        start_d = pd.Timestamp(start_date).date()
        for j in range(n):
            if dates[j] >= start_d:
                start_idx = j
                break
    if end_date is not None:
        end_d = pd.Timestamp(end_date).date()
        for j in range(n - 1, -1, -1):
            if dates[j] <= end_d:
                end_idx = j + 1
                break

    # State
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

        # ---- EXIT MANAGEMENT (same as validate_improvements.py) ----
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
                stop_dist_exit = abs(pos_entry_price - pos_stop)
                total_risk = stop_dist_exit * point_value * pos_contracts
                r_mult = total_pnl / total_risk if total_risk > 0 else 0.0

                trades.append({
                    "entry_time": nq.index[pos_entry_idx],
                    "exit_time": nq.index[i + 1] if exit_reason == "early_cut_pa" and i + 1 < end_idx else nq.index[i],
                    "r": r_mult, "reason": exit_reason, "dir": pos_direction,
                    "type": pos_signal_type, "trimmed": pos_trimmed, "grade": pos_grade,
                    "entry_price": pos_entry_price, "exit_price": exit_price,
                    "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
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

        if in_position:
            continue
        if news_blackout_arr is not None and news_blackout_arr[i]:
            continue

        et_frac = et_frac_arr[i]
        if 9.5 <= et_frac <= 10.0:
            continue
        if day_stopped:
            continue
        if not sig_mask[i]:
            continue

        direction = int(sig_dir[i])
        if direction == 0:
            continue

        sig_f = params.get("signal_filter", {})
        if not sig_f.get("allow_mss", True) and str(sig_type[i]) == "mss":
            continue
        if not sig_f.get("allow_trend", True) and str(sig_type[i]) == "trend":
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

        # Block PM shorts (Config D)
        if block_pm_shorts:
            if et_frac >= 14.0 and direction == -1:
                continue

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
        if pa_alt_arr[i] >= pa_threshold:
            continue

        # Session filter
        is_mss_smt = (str(sig_type[i]) == "mss" and has_smt_arr[i]
                      and smt_cfg.get("enabled", False))
        mss_bypass = is_mss_smt and smt_cfg.get("bypass_session_filter", False)

        is_ny = (10.0 <= et_frac < 16.0)
        is_london = (3.0 <= et_frac < 9.5)
        is_asia = (et_frac >= 18.0) or (et_frac < 3.0)

        if not mss_bypass:
            if is_ny:
                pass
            elif is_london:
                continue
            elif is_asia:
                continue
            else:
                continue

        # Min stop ATR
        if not np.isnan(stop_atr_ratio[i]):
            if stop_atr_ratio[i] < min_stop_atr:
                continue
        else:
            stop_dist_check = abs(entry_p - stop)
            a_check = atr_arr[i] if not np.isnan(atr_arr[i]) else 10.0
            if a_check > 0 and (stop_dist_check / a_check) < min_stop_atr:
                continue

        # SQ filter
        if not np.isnan(signal_quality[i]):
            eff_sq = sq_long if direction == 1 else sq_short
            if signal_quality[i] < eff_sq:
                continue

        if i + 1 >= end_idx:
            continue

        # Grade
        ba = 1.0 if (direction == np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0) else 0.0
        grade = _compute_grade_fast(ba, regime_arr[i])
        if grade == "C" and c_skip:
            continue

        is_reduced = (dow_arr[i] in (0, 4)) or (regime_arr[i] < 1.0)
        base_r = reduced_r if is_reduced else normal_r
        if grade == "A+":
            r_amount = base_r * a_plus_mult
        elif grade == "B+":
            r_amount = base_r * b_plus_mult
        else:
            r_amount = base_r * 0.5

        # Session regime (lunch dead zone)
        if session_regime.get("enabled", False):
            sr_lunch_s = session_regime.get("lunch_start", 12.0)
            sr_lunch_e = session_regime.get("lunch_end", 13.0)
            sr_am_end = session_regime.get("am_end", 12.0)
            sr_pm_start = session_regime.get("pm_start", 13.0)
            if et_frac < sr_am_end:
                sr_mult = session_regime.get("am_mult", 1.0)
            elif sr_lunch_s <= et_frac < sr_lunch_e:
                sr_mult = session_regime.get("lunch_mult", 0.5)
            elif et_frac >= sr_pm_start:
                sr_mult = session_regime.get("pm_mult", 0.75)
            else:
                sr_mult = 1.0
            r_amount *= sr_mult
            if r_amount <= 0:
                continue

        # Slippage + entry
        actual_entry = entry_p + slippage_points if direction == 1 else entry_p - slippage_points
        stop_dist = abs(actual_entry - stop)
        if stop_dist < 1.0:
            continue

        contracts = max(1, int(r_amount / (stop_dist * point_value))) if stop_dist > 0 and point_value > 0 else 0
        if contracts <= 0:
            continue

        # ===============================================================
        # TP COMPUTATION — this is where the dynamic logic goes
        # ===============================================================
        is_mss_signal = str(sig_type[i]) == "mss"

        # --- SHORT TP: always use dual_mode scalp, unchanged ---
        dual_mode_enabled = dual_mode.get("enabled", False)
        if dual_mode_enabled and direction == -1:
            short_rr = dual_mode.get("short_rr", 0.625)
            if mss_mgmt_enabled and is_mss_signal:
                short_rr = mss_mgmt.get("short_rr", short_rr)
            tp1 = actual_entry - stop_dist * short_rr
        elif direction == -1:
            # Non-dual-mode short: use baseline TP
            if session_rules.get("enabled", False):
                ny_tp_mult = session_rules.get("ny_tp_multiplier", 1.0)
                tp_distance = actual_entry - tp1
                tp1 = actual_entry - tp_distance * ny_tp_mult
        else:
            # --- LONG TP: dynamic logic ---
            # First compute the baseline IRL x mult for fallback
            if session_rules.get("enabled", False):
                ny_tp_mult = session_rules.get("ny_tp_multiplier", 1.0)
                if dir_mgmt.get("enabled", False):
                    actual_tp_mult = dir_mgmt.get("long_tp_mult", ny_tp_mult)
                else:
                    actual_tp_mult = ny_tp_mult
                if mss_mgmt_enabled and is_mss_signal:
                    actual_tp_mult = mss_mgmt.get("long_tp_mult", actual_tp_mult)
            else:
                actual_tp_mult = 2.0
                if mss_mgmt_enabled and is_mss_signal:
                    actual_tp_mult = mss_mgmt.get("long_tp_mult", actual_tp_mult)

            irl_distance = abs(tp1 - actual_entry)
            baseline_tp = actual_entry + irl_distance * actual_tp_mult

            if tp_strategy == TP_BASELINE:
                # E. Baseline: IRL x 2.0 (or MSS 2.5)
                if 9.5 <= et_frac < 16.0:
                    tp1 = baseline_tp
                # else: keep tp1 as-is (non-NY)

            elif tp_strategy == TP_NEAREST_SESSION:
                # A. Nearest session level >= 1R away, else fallback
                sess_target = _nearest_session_level_above(i, actual_entry, d)
                if not np.isnan(sess_target) and (sess_target - actual_entry) >= stop_dist:
                    tp1 = sess_target
                else:
                    tp1 = baseline_tp

            elif tp_strategy == TP_NEAREST_HTF_SWING:
                # B. Nearest HTF swing high >= 1R away, else fallback
                htf_target = _nearest_htf_swing_above(i, actual_entry, d)
                if not np.isnan(htf_target) and (htf_target - actual_entry) >= stop_dist:
                    tp1 = htf_target
                else:
                    tp1 = baseline_tp

            elif tp_strategy == TP_BEST_OF_SESSION_OR_HTF:
                # C. Best of session OR HTF swing (closer but >= 1R)
                sess_target = _nearest_session_level_above(i, actual_entry, d)
                htf_target = _nearest_htf_swing_above(i, actual_entry, d)
                candidates = []
                if not np.isnan(sess_target) and (sess_target - actual_entry) >= stop_dist:
                    candidates.append(sess_target)
                if not np.isnan(htf_target) and (htf_target - actual_entry) >= stop_dist:
                    candidates.append(htf_target)
                if candidates:
                    tp1 = min(candidates)  # closest valid target
                else:
                    tp1 = baseline_tp

            elif tp_strategy == TP_ADAPTIVE_MULT:
                # D. Adaptive multiplier: min(3.0, actual_distance / irl_distance)
                sess_target = _nearest_session_level_above(i, actual_entry, d)
                htf_target = _nearest_htf_swing_above(i, actual_entry, d)
                candidates = []
                if not np.isnan(sess_target) and (sess_target - actual_entry) >= stop_dist:
                    candidates.append(sess_target)
                if not np.isnan(htf_target) and (htf_target - actual_entry) >= stop_dist:
                    candidates.append(htf_target)
                if candidates and irl_distance > 0:
                    best_target = min(candidates)
                    actual_distance = best_target - actual_entry
                    adaptive_mult = min(3.0, actual_distance / irl_distance)
                    adaptive_mult = max(1.0, adaptive_mult)  # at least 1.0x
                    tp1 = actual_entry + irl_distance * adaptive_mult
                else:
                    tp1 = baseline_tp

        # Enter position
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

    # Force close at end
    if in_position and pos_entry_idx < end_idx:
        last_i = end_idx - 1
        exit_price = c_arr[last_i]
        pnl_pts = (exit_price - pos_entry_price) if pos_direction == 1 else (pos_entry_price - exit_price)
        total_pnl = pnl_pts * point_value * pos_remaining_contracts
        if pos_trimmed:
            trim_c = pos_contracts - pos_remaining_contracts
            trim_pnl = ((pos_tp1 - pos_entry_price) if pos_direction == 1 else (pos_entry_price - pos_tp1)) * point_value * trim_c
            total_pnl += trim_pnl
            total_pnl -= commission_per_side * 2 * pos_contracts
        else:
            total_pnl -= commission_per_side * 2 * pos_remaining_contracts
        stop_dist_exit = abs(pos_entry_price - pos_stop)
        total_risk = stop_dist_exit * point_value * pos_contracts
        r_mult = total_pnl / total_risk if total_risk > 0 else 0.0
        trades.append({
            "entry_time": nq.index[pos_entry_idx],
            "exit_time": nq.index[last_i],
            "r": r_mult, "reason": "eod_close", "dir": pos_direction,
            "type": pos_signal_type, "trimmed": pos_trimmed, "grade": pos_grade,
            "entry_price": pos_entry_price, "exit_price": exit_price,
            "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
        })

    return trades


# ======================================================================
# Metrics (same as validate_improvements.py)
# ======================================================================
def compute_metrics(trades_list):
    if not trades_list:
        return {"trades": 0, "R": 0.0, "PPDD": 0.0, "PF": 0.0, "WR": 0.0, "MaxDD": 0.0, "avgR": 0.0}
    r = np.array([t["r"] for t in trades_list])
    total_r = r.sum()
    wr = (r > 0).mean() * 100
    wins = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = wins / losses if losses > 0 else 999.0
    cumr = np.cumsum(r)
    peak = np.maximum.accumulate(cumr)
    dd = peak - cumr
    max_dd = dd.max() if len(dd) > 0 else 0
    ppdd = total_r / max_dd if max_dd > 0 else 999.0
    return {
        "trades": len(trades_list),
        "R": round(total_r, 1),
        "PPDD": round(ppdd, 2),
        "PF": round(pf, 2),
        "WR": round(wr, 1),
        "MaxDD": round(max_dd, 1),
        "avgR": round(total_r / len(trades_list), 4),
    }


def walk_forward_metrics(trades_list):
    """Per-year breakdown."""
    if not trades_list:
        return []
    df = pd.DataFrame(trades_list)
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year
    results = []
    for year, grp in df.groupby("year"):
        r = grp["r"].values
        total_r = r.sum()
        wr = (r > 0).mean() * 100
        wins = r[r > 0].sum()
        losses = abs(r[r < 0].sum())
        pf = wins / losses if losses > 0 else 999.0
        cumr = np.cumsum(r)
        peak = np.maximum.accumulate(cumr)
        dd = (peak - cumr).max()
        ppdd = total_r / dd if dd > 0 else 999.0
        results.append({"year": year, "n": len(grp), "R": round(total_r, 1),
                        "WR": round(wr, 1), "PF": round(pf, 2), "PPDD": round(ppdd, 2),
                        "MaxDD": round(dd, 1)})
    return results


def print_metrics(label, m):
    print(f"  {label:45s} | {m['trades']:4d}t | R={m['R']:+8.1f} | PPDD={m['PPDD']:6.2f} | PF={m['PF']:5.2f} | WR={m['WR']:5.1f}% | MaxDD={m['MaxDD']:5.1f}R | avgR={m['avgR']:+.4f}")


def print_wf(label, wf_results):
    print(f"\n  Walk-Forward: {label}")
    print(f"  {'Year':>6s}  {'N':>5s}  {'R':>8s}  {'WR':>6s}  {'PF':>6s}  {'PPDD':>7s}  {'MaxDD':>6s}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}")
    wins_r = 0
    for row in wf_results:
        flag = "+" if row["R"] > 0 else "-"
        if row["R"] > 0:
            wins_r += 1
        print(f"  {row['year']:>6d}  {row['n']:>5d}  {row['R']:>+8.1f}  {row['WR']:>5.1f}%  {row['PF']:>6.2f}  {row['PPDD']:>7.2f}  {row['MaxDD']:>5.1f}R  {flag}")
    n_years = len(wf_results)
    print(f"  WF Score: {wins_r}/{n_years} years profitable")


def tp_analysis(trades_list, label):
    """Analyze TP hit rates and distances for long trades."""
    if not trades_list:
        return
    longs = [t for t in trades_list if t["dir"] == 1]
    shorts = [t for t in trades_list if t["dir"] == -1]
    long_r = np.array([t["r"] for t in longs]) if longs else np.array([])
    short_r = np.array([t["r"] for t in shorts]) if shorts else np.array([])

    print(f"\n  TP Analysis: {label}")
    print(f"    Longs:  {len(longs):4d}t  R={long_r.sum():+.1f}  avgR={long_r.mean():+.4f}  WR={((long_r > 0).mean()*100):.1f}%" if longs else "    Longs:  0t")
    print(f"    Shorts: {len(shorts):4d}t  R={short_r.sum():+.1f}  avgR={short_r.mean():+.4f}  WR={((short_r > 0).mean()*100):.1f}%" if shorts else "    Shorts: 0t")

    if longs:
        # Trim hit rates
        trimmed = sum(1 for t in longs if t["trimmed"])
        stopped = sum(1 for t in longs if t["reason"] == "stop")
        be_swept = sum(1 for t in longs if t["reason"] == "be_sweep")
        tp1_full = sum(1 for t in longs if t["reason"] == "tp1")
        early_cut = sum(1 for t in longs if t["reason"] == "early_cut_pa")
        print(f"    Long exits: trimmed={trimmed} stop={stopped} be_sweep={be_swept} tp1_full={tp1_full} early_cut={early_cut}")

        # TP distance distribution (as R-multiple)
        tp_distances = []
        for t in longs:
            sd = abs(t["entry_price"] - t["stop_price"])
            if sd > 0:
                tp_dist = abs(t["tp1_price"] - t["entry_price"]) / sd
                tp_distances.append(tp_dist)
        if tp_distances:
            td = np.array(tp_distances)
            print(f"    TP distance (R): mean={td.mean():.2f}  med={np.median(td):.2f}  p25={np.percentile(td,25):.2f}  p75={np.percentile(td,75):.2f}")


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 110)
    print("DYNAMIC LIQUIDITY TP EXPERIMENT")
    print("=" * 110)

    d = load_all()

    # ---- Run all 5 variants ----
    variants = [
        ("E. Baseline (IRL x 2.0)", TP_BASELINE),
        ("A. Nearest session level", TP_NEAREST_SESSION),
        ("B. Nearest HTF swing (L=10)", TP_NEAREST_HTF_SWING),
        ("C. Best of session OR HTF swing", TP_BEST_OF_SESSION_OR_HTF),
        ("D. Adaptive multiplier (cap 3.0x)", TP_ADAPTIVE_MULT),
    ]

    print("\n" + "=" * 110)
    print("ALL VARIANTS (Config D base: sq_short=0.80, block_pm_shorts=True)")
    print("=" * 110)

    all_results = {}
    all_trades = {}
    for label, strategy in variants:
        t0 = _time.perf_counter()
        trades = run_backtest_dynamic_tp(d, tp_strategy=strategy)
        m = compute_metrics(trades)
        elapsed = _time.perf_counter() - t0
        print_metrics(label, m)
        print(f"    (computed in {elapsed:.1f}s)")
        all_results[strategy] = m
        all_trades[strategy] = trades

    # ---- TP Analysis for each variant ----
    print("\n" + "=" * 110)
    print("TP ANALYSIS (Long trades)")
    print("=" * 110)
    for label, strategy in variants:
        tp_analysis(all_trades[strategy], label)

    # ---- Comparison table ----
    print("\n" + "=" * 110)
    print("COMPARISON vs BASELINE")
    print("=" * 110)
    base = all_results[TP_BASELINE]
    for label, strategy in variants:
        m = all_results[strategy]
        dr = m["R"] - base["R"]
        dp = m["PPDD"] - base["PPDD"]
        dpf = m["PF"] - base["PF"]
        dmdd = m["MaxDD"] - base["MaxDD"]
        print(f"  {label:45s} | dR={dr:+6.1f} | dPPDD={dp:+6.2f} | dPF={dpf:+5.2f} | dMaxDD={dmdd:+5.1f}")

    # ---- Walk-forward for top 2 variants (+ baseline for reference) ----
    print("\n" + "=" * 110)
    print("WALK-FORWARD ANALYSIS")
    print("=" * 110)

    # Show walk-forward for ALL variants
    wf_variants = list(variants)

    for label, strategy in wf_variants:
        wf = walk_forward_metrics(all_trades[strategy])
        print_wf(label, wf)

    # ---- Session level hit rate analysis ----
    print("\n" + "=" * 110)
    print("SESSION LEVEL AVAILABILITY (at signal bars for longs)")
    print("=" * 110)

    # Check how often session levels are available and in-direction for long signals
    sig_indices = np.where(d["sig_mask"])[0]
    long_sigs = [i for i in sig_indices if d["sig_dir"][i] == 1]

    n_total = len(long_sigs)
    n_sess_available = 0
    n_sess_1r = 0
    n_htf_available = 0
    n_htf_1r = 0
    sess_distances = []
    htf_distances = []

    for i in long_sigs:
        entry_p = d["entry_price_arr"][i]
        stop_p = d["model_stop_arr"][i]
        if np.isnan(entry_p) or np.isnan(stop_p):
            continue
        stop_dist = abs(entry_p - stop_p)
        if stop_dist < 1.0:
            continue

        sess = _nearest_session_level_above(i, entry_p, d)
        if not np.isnan(sess):
            n_sess_available += 1
            dist = sess - entry_p
            sess_distances.append(dist / stop_dist)  # in R-multiples
            if dist >= stop_dist:
                n_sess_1r += 1

        htf = _nearest_htf_swing_above(i, entry_p, d)
        if not np.isnan(htf):
            n_htf_available += 1
            dist = htf - entry_p
            htf_distances.append(dist / stop_dist)
            if dist >= stop_dist:
                n_htf_1r += 1

    print(f"  Total long signals: {n_total}")
    print(f"  Session level above entry: {n_sess_available} ({n_sess_available/max(n_total,1)*100:.1f}%)")
    print(f"  Session level >= 1R away:  {n_sess_1r} ({n_sess_1r/max(n_total,1)*100:.1f}%)")
    if sess_distances:
        sd = np.array(sess_distances)
        print(f"  Session distance (R): mean={sd.mean():.2f}  med={np.median(sd):.2f}  p25={np.percentile(sd,25):.2f}  p75={np.percentile(sd,75):.2f}")

    print(f"  HTF swing above entry:     {n_htf_available} ({n_htf_available/max(n_total,1)*100:.1f}%)")
    print(f"  HTF swing >= 1R away:      {n_htf_1r} ({n_htf_1r/max(n_total,1)*100:.1f}%)")
    if htf_distances:
        hd = np.array(htf_distances)
        print(f"  HTF distance (R):   mean={hd.mean():.2f}  med={np.median(hd):.2f}  p25={np.percentile(hd,25):.2f}  p75={np.percentile(hd,75):.2f}")

    print("\n" + "=" * 110)
    print("DONE")
    print("=" * 110)


if __name__ == "__main__":
    main()
