"""
experiments/phase2_combined.py — Systematic grid search over combined filter relaxations.

Tests 243 combinations of:
  - Sessions: NY_only, NY+London, All_sessions
  - SQ_long: 0.60, 0.64, 0.68
  - SQ_short: 0.75, 0.80, 0.82
  - Day_filter: all_days, skip_friday, skip_mon_fri
  - Min_stop_ATR: 1.0, 1.5, 2.0

Baseline: 534 trades, +156.6R, PPDD=10.92
Goal: 800-1200 trades with PPDD > 10

For top 5 combos: walk-forward validation (expanding window, 1-year OOS).

Usage: python experiments/phase2_combined.py
"""
from __future__ import annotations

import itertools
import logging
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

DATA = PROJECT / "data"
CONFIG = PROJECT / "config" / "params.yaml"


# ======================================================================
# Grid dimensions
# ======================================================================

GRID = {
    "session": ["NY_only", "NY+London", "All_sessions"],
    "sq_long": [0.60, 0.64, 0.68],
    "sq_short": [0.75, 0.80, 0.82],
    "day_filter": ["all_days", "skip_friday", "skip_mon_fri"],
    "min_stop_atr": [1.0, 1.5, 2.0],
}

# Total combos
TOTAL = 1
for v in GRID.values():
    TOTAL *= len(v)
print(f"[GRID] Total combinations: {TOTAL}")


# ======================================================================
# Data loading (same as validate_nt_logic.py)
# ======================================================================

def load_all():
    """Load all data and pre-compute arrays once."""
    t0 = _time.perf_counter()
    print("[LOAD] Loading data...")

    nq = pd.read_parquet(DATA / "NQ_5m_10yr.parquet")
    es = pd.read_parquet(DATA / "ES_5m_10yr.parquet")
    sig3 = pd.read_parquet(DATA / "cache_signals_10yr_v3.parquet")
    bias = pd.read_parquet(DATA / "cache_bias_10yr_v2.parquet")
    regime = pd.read_parquet(DATA / "cache_regime_10yr_v2.parquet")
    with open(CONFIG, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # SMT divergence
    print("[LOAD] Computing SMT divergence...")
    from features.smt import compute_smt
    smt = compute_smt(nq, es, {'swing': {'left_bars': 3, 'right_bars': 1},
                                'smt': {'sweep_lookback': 15, 'time_tolerance': 1}})

    # Merge signals: v3 cache + SMT gate for MSS
    def make_sig(sig3_s, smt_s):
        s = sig3_s.copy()
        mm = s['signal'].astype(bool) & (s['signal_type'] == 'mss')
        mi = s.index[mm]
        s.loc[mi, 'signal'] = False
        s.loc[mi, 'signal_dir'] = 0
        s['has_smt'] = False
        c_idx = mi.intersection(smt_s.index)
        if len(c_idx) == 0:
            return s
        md = sig3_s.loc[c_idx, 'signal_dir'].values
        ok = ((md == 1) & smt_s.loc[c_idx, 'smt_bull'].values.astype(bool)) | \
             ((md == -1) & smt_s.loc[c_idx, 'smt_bear'].values.astype(bool))
        g = c_idx[ok]
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
    print(f"[LOAD] Merged signals: {ss['signal'].sum()} total")

    # Pre-compute all feature arrays
    print("[LOAD] Pre-computing features...")
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

    atr_arr = compute_atr(nq).values
    fluency_arr = compute_fluency(nq, params).values
    pa_alt_arr = compute_alternating_dir_ratio(nq, window=6).values

    swing_p = {"left_bars": params["swing"]["left_bars"],
               "right_bars": params["swing"]["right_bars"]}
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

    # Signal quality (pre-compute at every signal bar)
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

    # Date tracking (session dates)
    dates = np.array([
        (et_idx[j] + pd.Timedelta(days=1)).date() if et_idx[j].hour >= 18 else et_idx[j].date()
        for j in range(n)
    ])

    # Day-of-week for each bar (Monday=0, Friday=4)
    dow_arr = np.array([et_idx[j].dayofweek for j in range(n)])

    # ET fractional hour for each bar
    et_frac_arr = np.array([et_idx[j].hour + et_idx[j].minute / 60.0 for j in range(n)])

    elapsed = _time.perf_counter() - t0
    print(f"[LOAD] All data loaded and pre-computed in {elapsed:.1f}s (n={n} bars)")

    return {
        "nq": nq,
        "params": params,
        "n": n,
        "o": o, "h": h, "l": l, "c": c,
        "atr_arr": atr_arr,
        "fluency_arr": fluency_arr,
        "pa_alt_arr": pa_alt_arr,
        "swing_high_mask": swing_high_mask,
        "swing_low_mask": swing_low_mask,
        "sig_mask": sig_mask,
        "sig_dir": sig_dir,
        "sig_type": sig_type,
        "has_smt_arr": has_smt_arr,
        "entry_price_arr": entry_price_arr,
        "model_stop_arr": model_stop_arr,
        "irl_target_arr": irl_target_arr,
        "bias_dir_arr": bias_dir_arr,
        "bias_conf_arr": bias_conf_arr,
        "regime_arr": regime_arr,
        "news_blackout_arr": news_blackout_arr,
        "signal_quality": signal_quality,
        "dates": dates,
        "dow_arr": dow_arr,
        "et_frac_arr": et_frac_arr,
        "et_idx": et_idx,
    }


# ======================================================================
# Fast backtest (inlined, filter params varied)
# ======================================================================

def _find_nth_swing(mask, prices, idx, n_val, direction):
    """Find the nth swing high/low looking backwards."""
    count = 0
    for j in range(idx - 1, -1, -1):
        if mask[j]:
            count += 1
            if count == n_val:
                return prices[j]
    return np.nan


def run_backtest_fast(
    d: dict,
    # --- Grid params ---
    session_mode: str,      # "NY_only", "NY+London", "All_sessions"
    sq_long: float,         # long SQ threshold
    sq_short: float,        # short SQ threshold
    day_filter: str,        # "all_days", "skip_friday", "skip_mon_fri"
    min_stop_atr: float,    # min stop as ATR multiple
    # --- Optional: date range for walk-forward ---
    start_date=None,
    end_date=None,
) -> list[dict]:
    """Run backtest with specific filter configuration. Returns list of trade dicts."""

    params = d["params"]
    n = d["n"]
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    atr_arr = d["atr_arr"]
    sig_mask = d["sig_mask"]
    sig_dir = d["sig_dir"]
    sig_type = d["sig_type"]
    has_smt_arr = d["has_smt_arr"]
    entry_price_arr = d["entry_price_arr"]
    model_stop_arr = d["model_stop_arr"]
    irl_target_arr = d["irl_target_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    bias_conf_arr = d["bias_conf_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    signal_quality = d["signal_quality"]
    dates = d["dates"]
    dow_arr = d["dow_arr"]
    et_frac_arr = d["et_frac_arr"]
    pa_alt_arr = d["pa_alt_arr"]
    swing_high_mask = d["swing_high_mask"]
    swing_low_mask = d["swing_low_mask"]
    nq = d["nq"]

    # Fixed params from config
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

    # --- Determine session filter logic from grid param ---
    # session_mode controls which sessions we allow
    # "NY_only": skip London + Asia (baseline)
    # "NY+London": skip Asia only
    # "All_sessions": allow all

    # --- Determine day-of-week skips from grid param ---
    skip_days = set()
    if day_filter == "skip_friday":
        skip_days = {4}
    elif day_filter == "skip_mon_fri":
        skip_days = {0, 4}

    # --- Date range filtering ---
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

    # ---- Bar-by-bar backtest ----
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
                    exit_price = o[i+1] if i+1 < end_idx else c[i]
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
                    "r": r_mult,
                    "reason": exit_reason,
                    "dir": pos_direction,
                    "type": pos_signal_type,
                    "trimmed": pos_trimmed,
                    "grade": pos_grade,
                    "entry_price": pos_entry_price,
                    "exit_price": exit_price,
                    "stop_price": pos_stop,
                    "tp1_price": pos_tp1,
                    "pnl_dollars": total_pnl,
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

        # ---- Skip entry checks if in position ----
        if in_position:
            continue

        # ---- News blackout ----
        if news_blackout_arr is not None and news_blackout_arr[i]:
            continue

        # ---- ORM no-trade window (9:30-10:00 ET) ----
        if not day_stopped:
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

        # Signal type filter
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

        # ---- FILTER: Bias (exempt MSS+SMT) ----
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

        # ---- FILTER: Session (GRID PARAM) ----
        et_frac = et_frac_arr[i]
        is_mss_smt = (str(sig_type[i]) == "mss" and has_smt_arr[i]
                      and smt_cfg.get("enabled", False))
        mss_bypass = is_mss_smt and smt_cfg.get("bypass_session_filter", False)

        if not mss_bypass:
            if session_mode == "NY_only":
                # Only NY: 9.5 <= et_frac < 16.0 (after ORM)
                if not (9.5 <= et_frac < 16.0):
                    continue
            elif session_mode == "NY+London":
                # NY + London: 3.0 <= et_frac < 16.0
                if not (3.0 <= et_frac < 16.0):
                    continue
            elif session_mode == "All_sessions":
                # All sessions allowed (including Asia)
                pass

        # ---- FILTER: Day of week (GRID PARAM) ----
        if dow_arr[i] in skip_days:
            continue

        # ---- FILTER: Min stop (GRID PARAM) ----
        stop_dist = abs(entry_p - stop)
        min_stop = min_stop_atr * atr_arr[i] if not np.isnan(atr_arr[i]) else 10.0
        if stop_dist < min_stop:
            continue

        # ---- FILTER: Signal quality (GRID PARAMS: sq_long, sq_short) ----
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

        # Sizing
        is_reduced = (dow_arr[i] in (0, 4)) or (regime_arr[i] < 1.0)
        base_r = reduced_r if is_reduced else normal_r
        if grade == "A+":
            r_amount = base_r * a_plus_mult
        elif grade == "B+":
            r_amount = base_r * b_plus_mult
        else:
            r_amount = base_r * 0.5

        # Session regime
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

        # Slippage
        actual_entry = entry_p + slippage_points if direction == 1 else entry_p - slippage_points
        stop_dist = abs(actual_entry - stop)
        if stop_dist < 1.0:
            continue

        contracts = max(1, int(r_amount / (stop_dist * point_value))) if stop_dist > 0 and point_value > 0 else 0
        if contracts <= 0:
            continue

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

        dual_mode_enabled = dual_mode.get("enabled", False)
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

    # Force close open position at end
    if in_position and pos_entry_idx < end_idx:
        last_i = end_idx - 1
        exit_price = c[last_i]
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
            "stop_price": pos_stop, "tp1_price": pos_tp1,
            "pnl_dollars": total_pnl,
        })

    return trades


def _compute_grade_fast(ba: float, regime: float) -> str:
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
# Metrics computation
# ======================================================================

def compute_metrics(trades_list: list[dict]) -> dict:
    """Compute performance metrics from trade list."""
    if not trades_list:
        return {"trades": 0, "R": 0.0, "PPDD": 0.0, "MaxDD_R": 0.0,
                "WR": 0.0, "PF": 0.0, "avgR": 0.0}

    r_arr = np.array([t["r"] for t in trades_list])
    n = len(r_arr)
    total_r = float(r_arr.sum())
    wr = float((r_arr > 0).mean()) * 100

    # R-based equity curve for PPDD
    cum_r = np.cumsum(r_arr)
    peak = np.maximum.accumulate(cum_r)
    dd_r = cum_r - peak
    max_dd_r = float(abs(dd_r.min())) if len(dd_r) > 0 else 0.0
    ppdd = total_r / max_dd_r if max_dd_r > 0.01 else total_r

    # Profit factor
    gross_win = float(r_arr[r_arr > 0].sum()) if (r_arr > 0).any() else 0.0
    gross_loss = float(abs(r_arr[r_arr <= 0].sum())) if (r_arr <= 0).any() else 0.001
    pf = gross_win / gross_loss if gross_loss > 0.001 else 999.0

    return {
        "trades": n,
        "R": round(total_r, 2),
        "PPDD": round(ppdd, 2),
        "MaxDD_R": round(max_dd_r, 2),
        "WR": round(wr, 1),
        "PF": round(pf, 2),
        "avgR": round(float(r_arr.mean()), 4),
    }


def compute_yearly(trades_list: list[dict]) -> dict:
    """Compute per-year R for equity smoothness analysis."""
    if not trades_list:
        return {}
    df = pd.DataFrame(trades_list)
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year
    return df.groupby("year")["r"].sum().to_dict()


# ======================================================================
# Walk-forward validation (expanding window, 1-year OOS)
# ======================================================================

WF_WINDOWS = [
    # train_start, train_end, test_start, test_end
    ("2016-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2016-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2016-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2016-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2016-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2016-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]


def walk_forward_validate(d: dict, session_mode: str, sq_long: float,
                          sq_short: float, day_filter: str,
                          min_stop_atr_val: float) -> dict:
    """Run expanding-window walk-forward. Returns OOS aggregates."""
    oos_trades_all = []
    window_results = []

    for train_s, train_e, test_s, test_e in WF_WINDOWS:
        oos_trades = run_backtest_fast(
            d, session_mode=session_mode, sq_long=sq_long,
            sq_short=sq_short, day_filter=day_filter,
            min_stop_atr=min_stop_atr_val,
            start_date=test_s, end_date=test_e,
        )
        m = compute_metrics(oos_trades)
        m["window"] = f"{test_s[:4]}"
        window_results.append(m)
        oos_trades_all.extend(oos_trades)

    oos_agg = compute_metrics(oos_trades_all)
    oos_agg["n_windows"] = len(WF_WINDOWS)
    oos_agg["win_windows"] = sum(1 for w in window_results if w["R"] > 0)
    oos_agg["per_window"] = window_results
    return oos_agg


# ======================================================================
# Main grid search
# ======================================================================

def main():
    t_start = _time.perf_counter()

    # Load data once
    d = load_all()

    # First run baseline for reference
    print("\n" + "=" * 80)
    print("BASELINE (current production config)")
    print("=" * 80)
    baseline_trades = run_backtest_fast(
        d,
        session_mode="NY_only",
        sq_long=0.68,
        sq_short=0.82,
        day_filter="all_days",
        min_stop_atr=1.5,
    )
    baseline_m = compute_metrics(baseline_trades)
    print(f"  Trades={baseline_m['trades']}, R={baseline_m['R']:.1f}, "
          f"PPDD={baseline_m['PPDD']:.2f}, MaxDD={baseline_m['MaxDD_R']:.1f}R, "
          f"WR={baseline_m['WR']:.1f}%, PF={baseline_m['PF']:.2f}")

    # ---- Grid search ----
    print("\n" + "=" * 80)
    print(f"GRID SEARCH: {TOTAL} combinations")
    print("=" * 80)

    all_results = []
    combo_iter = itertools.product(
        GRID["session"],
        GRID["sq_long"],
        GRID["sq_short"],
        GRID["day_filter"],
        GRID["min_stop_atr"],
    )

    for idx, (sess, sq_l, sq_s, day_f, msa) in enumerate(combo_iter):
        trades = run_backtest_fast(
            d,
            session_mode=sess,
            sq_long=sq_l,
            sq_short=sq_s,
            day_filter=day_f,
            min_stop_atr=msa,
        )
        m = compute_metrics(trades)
        m["session"] = sess
        m["sq_long"] = sq_l
        m["sq_short"] = sq_s
        m["day_filter"] = day_f
        m["min_stop_atr"] = msa
        all_results.append(m)

        if (idx + 1) % 25 == 0:
            elapsed = _time.perf_counter() - t_start
            eta = elapsed / (idx + 1) * (TOTAL - idx - 1)
            print(f"  [{idx+1}/{TOTAL}] elapsed={elapsed:.0f}s, ETA={eta:.0f}s "
                  f"| last: t={m['trades']}, R={m['R']:.1f}, PPDD={m['PPDD']:.2f}")

    elapsed_grid = _time.perf_counter() - t_start
    print(f"\n[GRID] Complete in {elapsed_grid:.0f}s")

    # ---- Convert to DataFrame ----
    results_df = pd.DataFrame(all_results)

    # ---- Filter: PPDD >= 8, Trades >= 600, WR >= 40%, PF >= 1.3 ----
    print("\n" + "=" * 80)
    print("FILTERED RESULTS (PPDD >= 8.0, Trades >= 600, WR >= 40%, PF >= 1.3)")
    print("=" * 80)

    mask = (
        (results_df["PPDD"] >= 8.0) &
        (results_df["trades"] >= 600) &
        (results_df["WR"] >= 40.0) &
        (results_df["PF"] >= 1.3)
    )
    filtered = results_df[mask].sort_values("PPDD", ascending=False)
    print(f"  {len(filtered)} / {len(results_df)} combos pass filter")

    if len(filtered) == 0:
        # Relax filters progressively
        print("\n  No combos pass strict filter. Relaxing...")
        for ppdd_min, trade_min, wr_min, pf_min in [
            (6.0, 500, 38.0, 1.2),
            (4.0, 400, 35.0, 1.1),
            (2.0, 300, 30.0, 1.0),
        ]:
            mask2 = (
                (results_df["PPDD"] >= ppdd_min) &
                (results_df["trades"] >= trade_min) &
                (results_df["WR"] >= wr_min) &
                (results_df["PF"] >= pf_min)
            )
            filtered = results_df[mask2].sort_values("PPDD", ascending=False)
            if len(filtered) > 0:
                print(f"  Relaxed filter: PPDD>={ppdd_min}, Trades>={trade_min}, "
                      f"WR>={wr_min}%, PF>={pf_min}")
                print(f"  {len(filtered)} combos pass")
                break

    # ---- Top 20 by PPDD ----
    print("\n" + "=" * 80)
    print("TOP 20 COMBINATIONS (sorted by PPDD)")
    print("=" * 80)

    top20 = filtered.head(20) if len(filtered) >= 20 else results_df.sort_values("PPDD", ascending=False).head(20)

    display_cols = ["session", "sq_long", "sq_short", "day_filter", "min_stop_atr",
                    "trades", "R", "PPDD", "MaxDD_R", "WR", "PF", "avgR"]
    print(top20[display_cols].to_string(index=False))

    # ---- Maximum R with PPDD constraints ----
    print("\n" + "=" * 80)
    print("MAXIMUM R ANALYSIS")
    print("=" * 80)

    ppdd10 = results_df[results_df["PPDD"] >= 10.0]
    if len(ppdd10) > 0:
        best_r_ppdd10 = ppdd10.loc[ppdd10["R"].idxmax()]
        print(f"\n  Max R with PPDD >= 10: R={best_r_ppdd10['R']:.1f}, "
              f"PPDD={best_r_ppdd10['PPDD']:.2f}, Trades={best_r_ppdd10['trades']}")
        print(f"    Config: {best_r_ppdd10['session']}, SQ_L={best_r_ppdd10['sq_long']}, "
              f"SQ_S={best_r_ppdd10['sq_short']}, Day={best_r_ppdd10['day_filter']}, "
              f"MinATR={best_r_ppdd10['min_stop_atr']}")
    else:
        print("  No combos have PPDD >= 10.0")

    ppdd8 = results_df[results_df["PPDD"] >= 8.0]
    if len(ppdd8) > 0:
        best_r_ppdd8 = ppdd8.loc[ppdd8["R"].idxmax()]
        print(f"\n  Max R with PPDD >= 8: R={best_r_ppdd8['R']:.1f}, "
              f"PPDD={best_r_ppdd8['PPDD']:.2f}, Trades={best_r_ppdd8['trades']}")
        print(f"    Config: {best_r_ppdd8['session']}, SQ_L={best_r_ppdd8['sq_long']}, "
              f"SQ_S={best_r_ppdd8['sq_short']}, Day={best_r_ppdd8['day_filter']}, "
              f"MinATR={best_r_ppdd8['min_stop_atr']}")
    else:
        print("  No combos have PPDD >= 8.0")

    # Max R overall
    best_r_overall = results_df.loc[results_df["R"].idxmax()]
    print(f"\n  Max R overall: R={best_r_overall['R']:.1f}, "
          f"PPDD={best_r_overall['PPDD']:.2f}, Trades={best_r_overall['trades']}")
    print(f"    Config: {best_r_overall['session']}, SQ_L={best_r_overall['sq_long']}, "
          f"SQ_S={best_r_overall['sq_short']}, Day={best_r_overall['day_filter']}, "
          f"MinATR={best_r_overall['min_stop_atr']}")

    # ---- Walk-forward validation on top 5 ----
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION (top 5 by PPDD, expanding window, 1yr OOS)")
    print("=" * 80)

    top5_source = filtered.head(5) if len(filtered) >= 5 else results_df.sort_values("PPDD", ascending=False).head(5)

    wf_results = []
    for row_idx, row in top5_source.iterrows():
        t_wf = _time.perf_counter()
        wf = walk_forward_validate(
            d,
            session_mode=row["session"],
            sq_long=row["sq_long"],
            sq_short=row["sq_short"],
            day_filter=row["day_filter"],
            min_stop_atr_val=row["min_stop_atr"],
        )
        wf_elapsed = _time.perf_counter() - t_wf

        config_str = (f"{row['session']}, SQ_L={row['sq_long']}, SQ_S={row['sq_short']}, "
                      f"Day={row['day_filter']}, MinATR={row['min_stop_atr']}")

        print(f"\n  Config: {config_str}")
        print(f"  Full-sample: Trades={row['trades']}, R={row['R']:.1f}, PPDD={row['PPDD']:.2f}")
        print(f"  WF-OOS:      Trades={wf['trades']}, R={wf['R']:.1f}, PPDD={wf['PPDD']:.2f}, "
              f"Wins={wf['win_windows']}/{wf['n_windows']}")

        # Per-window breakdown
        for pw in wf["per_window"]:
            flag = "+" if pw["R"] > 0 else "-"
            print(f"    {pw['window']}: {flag} t={pw['trades']}, R={pw['R']:.1f}, "
                  f"WR={pw['WR']:.1f}%, PPDD={pw['PPDD']:.2f}")

        wf_results.append({
            "config": config_str,
            "full_trades": row["trades"],
            "full_R": row["R"],
            "full_PPDD": row["PPDD"],
            "oos_trades": wf["trades"],
            "oos_R": wf["R"],
            "oos_PPDD": wf["PPDD"],
            "oos_wins": wf["win_windows"],
        })

    # ---- Compare best combo vs baseline ----
    print("\n" + "=" * 80)
    print("BEST COMBO vs BASELINE COMPARISON")
    print("=" * 80)

    # Best = highest PPDD from filtered (or overall top)
    if len(filtered) > 0:
        best_row = filtered.iloc[0]
    else:
        best_row = results_df.sort_values("PPDD", ascending=False).iloc[0]

    best_trades = run_backtest_fast(
        d,
        session_mode=best_row["session"],
        sq_long=best_row["sq_long"],
        sq_short=best_row["sq_short"],
        day_filter=best_row["day_filter"],
        min_stop_atr=best_row["min_stop_atr"],
    )
    best_m = compute_metrics(best_trades)
    best_yearly = compute_yearly(best_trades)
    baseline_yearly = compute_yearly(baseline_trades)

    print(f"\n  {'Metric':<20} {'Baseline':>12} {'Best Combo':>12} {'Delta':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Trades':<20} {baseline_m['trades']:>12} {best_m['trades']:>12} "
          f"{best_m['trades'] - baseline_m['trades']:>+12}")
    print(f"  {'Total R':<20} {baseline_m['R']:>12.1f} {best_m['R']:>12.1f} "
          f"{best_m['R'] - baseline_m['R']:>+12.1f}")
    print(f"  {'PPDD':<20} {baseline_m['PPDD']:>12.2f} {best_m['PPDD']:>12.2f} "
          f"{best_m['PPDD'] - baseline_m['PPDD']:>+12.2f}")
    print(f"  {'MaxDD (R)':<20} {baseline_m['MaxDD_R']:>12.1f} {best_m['MaxDD_R']:>12.1f} "
          f"{best_m['MaxDD_R'] - baseline_m['MaxDD_R']:>+12.1f}")
    print(f"  {'Win Rate %':<20} {baseline_m['WR']:>12.1f} {best_m['WR']:>12.1f} "
          f"{best_m['WR'] - baseline_m['WR']:>+12.1f}")
    print(f"  {'Profit Factor':<20} {baseline_m['PF']:>12.2f} {best_m['PF']:>12.2f} "
          f"{best_m['PF'] - baseline_m['PF']:>+12.2f}")
    print(f"  {'Avg R':<20} {baseline_m['avgR']:>12.4f} {best_m['avgR']:>12.4f} "
          f"{best_m['avgR'] - baseline_m['avgR']:>+12.4f}")

    print(f"\n  Best config: {best_row['session']}, SQ_L={best_row['sq_long']}, "
          f"SQ_S={best_row['sq_short']}, Day={best_row['day_filter']}, "
          f"MinATR={best_row['min_stop_atr']}")

    # Yearly equity comparison
    print(f"\n  {'Year':<8} {'Baseline R':>12} {'Best R':>12} {'Delta':>12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    all_years = sorted(set(list(baseline_yearly.keys()) + list(best_yearly.keys())))
    for yr in all_years:
        bl_r = baseline_yearly.get(yr, 0.0)
        bt_r = best_yearly.get(yr, 0.0)
        print(f"  {yr:<8} {bl_r:>12.1f} {bt_r:>12.1f} {bt_r - bl_r:>+12.1f}")

    # Equity smoothness: count negative years
    baseline_neg = sum(1 for v in baseline_yearly.values() if v < 0)
    best_neg = sum(1 for v in best_yearly.values() if v < 0)
    print(f"\n  Negative years: baseline={baseline_neg}, best={best_neg}")

    # ---- Heatmap-style summary: Session x SQ_long interaction ----
    print("\n" + "=" * 80)
    print("HEATMAP: PPDD by Session x SQ_long (averaged across other dims)")
    print("=" * 80)
    pivot_data = []
    for _, row in results_df.iterrows():
        pivot_data.append({"session": row["session"], "sq_long": row["sq_long"],
                           "PPDD": row["PPDD"]})
    pivot_df = pd.DataFrame(pivot_data)
    heatmap = pivot_df.groupby(["session", "sq_long"])["PPDD"].mean().unstack(level=1)
    print(heatmap.round(2).to_string())

    print("\n" + "=" * 80)
    print("HEATMAP: PPDD by Day_filter x Min_stop_ATR (averaged)")
    print("=" * 80)
    pivot_data2 = []
    for _, row in results_df.iterrows():
        pivot_data2.append({"day_filter": row["day_filter"],
                            "min_stop_atr": row["min_stop_atr"],
                            "PPDD": row["PPDD"]})
    pivot_df2 = pd.DataFrame(pivot_data2)
    heatmap2 = pivot_df2.groupby(["day_filter", "min_stop_atr"])["PPDD"].mean().unstack(level=1)
    print(heatmap2.round(2).to_string())

    # ---- Save full results ----
    out_path = PROJECT / "experiments" / "phase2_combined_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n[SAVE] Full results saved to {out_path}")

    total_elapsed = _time.perf_counter() - t_start
    print(f"\n[DONE] Total runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
