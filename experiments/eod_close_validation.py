"""
experiments/eod_close_validation.py — Mandatory EOD Close Validation

Hell audit discovered Config F (+624R) was a multi-day trend follower.
Same-day trades NET LOSE (-31.3R).

ALL prop firm backtests must now enforce 16:00 ET close (15:55 ET realistic).
This script re-runs ALL key configs with mandatory EOD close and compares.

Usage: python experiments/eod_close_validation.py
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", stream=sys.stdout)

from experiments.validate_improvements import (
    load_all,
    _find_nth_swing,
    _compute_grade_fast,
    compute_metrics,
    walk_forward_metrics,
    print_metrics,
)
from experiments.multi_level_tp import (
    prepare_liquidity_data,
    build_liquidity_ladder_long,
    build_liquidity_ladder_short,
    _find_nth_swing_price,
    print_trim_stage_breakdown,
)
from features.swing import compute_swing_levels


# ======================================================================
# Single-TP engine WITH mandatory EOD close
# ======================================================================
def run_backtest_improved_eod(
    d: dict,
    # Standard params
    sq_long: float = 0.68,
    sq_short: float = 0.82,
    min_stop_atr: float = 1.7,
    # Config flags
    block_pm_shorts: bool = True,
    # Trim / trail overrides
    trim_pct_override: float | None = None,   # None = use params.yaml
    trail_nth_override: int | None = None,     # None = use params.yaml
    # NY TP multiplier override
    ny_tp_mult_override: float | None = None,  # None = use params.yaml
    # Date range
    start_date=None, end_date=None,
) -> list[dict]:
    """Single-TP backtest with MANDATORY 15:55 ET EOD close."""

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
    trim_pct = trim_pct_override if trim_pct_override is not None else trim_params["pct"]
    be_after_trim = trim_params["be_after_trim"]
    nth_swing = trail_nth_override if trail_nth_override is not None else trail_params["use_nth_swing"]

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

        # ---- EXIT MANAGEMENT ----
        if in_position:
            exited = False
            exit_reason = ""
            exit_price = 0.0
            exit_contracts = pos_remaining_contracts

            # ========================================
            # EOD CLOSE: force close at 15:55 ET
            # ========================================
            if et_frac_arr[i] >= 15.917:  # 15:55 ET
                exit_price = c_arr[i]  # Close at current bar's close
                # Apply market-order slippage for EOD close
                if pos_direction == 1:
                    exit_price -= slippage_points
                else:
                    exit_price += slippage_points
                exit_reason = "eod_close"
                exited = True

            # Early cut on bad PA (only pre-trim, bars 2-4)
            if not exited:
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
                if pos_trimmed and exit_reason not in ("tp1", "eod_close"):
                    trim_pnl_total = (pos_tp1 - pos_entry_price) if pos_direction == 1 else (pos_entry_price - pos_tp1)
                    trim_c = pos_contracts - exit_contracts
                    total_pnl = trim_pnl_total * point_value * trim_c + pnl_pts * point_value * exit_contracts
                    total_comm = commission_per_side * 2 * pos_contracts
                    total_pnl -= total_comm
                elif pos_trimmed and exit_reason == "eod_close":
                    # EOD close after trim: account for trimmed + remaining
                    trim_pnl_total = (pos_tp1 - pos_entry_price) if pos_direction == 1 else (pos_entry_price - pos_tp1)
                    trim_c = pos_contracts - pos_remaining_contracts
                    total_pnl = trim_pnl_total * point_value * trim_c + pnl_pts * point_value * pos_remaining_contracts
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
                # EOD close is neutral — does NOT count as loss for 0-for-2
                if exit_reason == "eod_close":
                    pass  # neutral
                elif exit_reason == "be_sweep" and pos_trimmed:
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

        # Block PM shorts
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

        # Slippage + entry
        actual_entry = entry_p + slippage_points if direction == 1 else entry_p - slippage_points
        stop_dist = abs(actual_entry - stop)
        if stop_dist < 1.0:
            continue

        contracts = max(1, int(r_amount / (stop_dist * point_value))) if stop_dist > 0 and point_value > 0 else 0
        if contracts <= 0:
            continue

        # TP computation
        is_mss_signal = str(sig_type[i]) == "mss"
        if session_rules.get("enabled", False):
            if ny_tp_mult_override is not None:
                actual_tp_mult = ny_tp_mult_override
            else:
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

    # Force close at data end (shouldn't happen with EOD close, but safety)
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
            "r": r_mult, "reason": "data_end", "dir": pos_direction,
            "type": pos_signal_type, "trimmed": pos_trimmed, "grade": pos_grade,
            "entry_price": pos_entry_price, "exit_price": exit_price,
            "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
        })

    return trades


# ======================================================================
# Multi-TP engine WITH mandatory EOD close + phantom runner fix
# ======================================================================
def run_backtest_multi_tp_eod(
    d: dict,
    d_extra: dict,
    # Config D base
    sq_long: float = 0.68,
    sq_short: float = 0.80,
    min_stop_atr: float = 1.7,
    block_pm_shorts: bool = True,
    # Multi-TP params
    tp1_trim_pct: float = 0.25,
    tp2_trim_pct: float = 0.25,
    be_after_tp1: bool = False,
    be_after_tp2: bool = True,
    trail_nth_swing: int = 3,
    use_multi_tp_for_shorts: bool = False,
    # NY TP multiplier
    ny_tp_mult: float = 2.0,
    mss_long_tp_mult: float = 2.5,
    # Phantom runner fix: if True, tp2 takes ALL remaining (no phantom)
    fix_phantom_runner: bool = False,
    # Date range
    start_date=None, end_date=None,
) -> list[dict]:
    """Multi-TP backtest with MANDATORY 15:55 ET EOD close."""

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
    dates = d["dates"]
    dow_arr = d["dow_arr"]
    et_frac_arr = d["et_frac_arr"]
    pa_alt_arr = d["pa_alt_arr"]
    swing_high_mask = d["swing_high_mask"]
    swing_low_mask = d["swing_low_mask"]
    nq = d["nq"]

    pos_params = params["position"]
    risk_params = params["risk"]
    bt_params = params["backtest"]
    grading_params = params["grading"]
    trim_params = params["trim"]
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
    be_after_trim = trim_params["be_after_trim"]
    default_trim_pct = trim_params["pct"]

    low_arr = nq["low"].values
    high_arr = nq["high"].values

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
    pos_entry_price = pos_stop = 0.0
    pos_tp1 = pos_tp2 = 0.0
    pos_contracts = pos_remaining_contracts = 0
    pos_trim_stage = 0
    pos_be_stop = pos_trail_stop = 0.0
    pos_signal_type = pos_grade = ""
    pos_bias_dir = pos_regime = 0.0
    pos_is_multi_tp = False
    pos_trim1_contracts = 0
    pos_trim2_contracts = 0
    pos_tp1_pnl_pts = 0.0
    pos_tp2_pnl_pts = 0.0
    pos_single_trim_pct = default_trim_pct
    pos_single_trimmed = False

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

            # ========================================
            # EOD CLOSE: force close at 15:55 ET — BEFORE all other exit checks
            # ========================================
            if et_frac_arr[i] >= 15.917:  # 15:55 ET
                exit_price = c_arr[i]
                if pos_direction == 1:
                    exit_price -= slippage_points
                else:
                    exit_price += slippage_points
                exit_reason = "eod_close"
                exited = True

            # Early cut on bad PA
            bars_in_trade = i - pos_entry_idx
            if not exited and pos_trim_stage == 0 and not pos_single_trimmed and 2 <= bars_in_trade <= 4:
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

            # ==================================================================
            # MULTI-TP EXIT MANAGEMENT (LONGS)
            # ==================================================================
            if not exited and pos_is_multi_tp and pos_direction == 1:
                if pos_trim_stage >= 2:
                    eff_stop = pos_trail_stop if pos_trail_stop > 0 else pos_stop
                    if pos_be_stop > 0:
                        eff_stop = max(eff_stop, pos_be_stop)
                elif pos_trim_stage == 1 and be_after_tp1:
                    eff_stop = pos_stop
                    if pos_be_stop > 0:
                        eff_stop = max(eff_stop, pos_be_stop)
                else:
                    eff_stop = pos_stop

                if l[i] <= eff_stop:
                    exit_price = eff_stop - slippage_points
                    if pos_trim_stage > 0 and eff_stop >= pos_entry_price:
                        exit_reason = "be_sweep"
                    else:
                        exit_reason = "stop"
                    exited = True

                elif pos_trim_stage == 0 and h[i] >= pos_tp1:
                    tc1 = max(1, int(pos_contracts * tp1_trim_pct))
                    pos_trim1_contracts = tc1
                    pos_remaining_contracts = pos_contracts - tc1
                    pos_tp1_pnl_pts = pos_tp1 - pos_entry_price
                    pos_trim_stage = 1
                    if be_after_tp1:
                        pos_be_stop = pos_entry_price
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True

                if not exited and pos_trim_stage == 1 and h[i] >= pos_tp2:
                    if fix_phantom_runner:
                        # PHANTOM RUNNER FIX: take ALL remaining contracts at TP2
                        tc2 = pos_remaining_contracts
                    else:
                        tc2 = max(1, int(pos_contracts * tp2_trim_pct))
                        tc2 = min(tc2, pos_remaining_contracts)
                    pos_trim2_contracts = tc2
                    pos_remaining_contracts -= tc2
                    pos_tp2_pnl_pts = pos_tp2 - pos_entry_price
                    pos_trim_stage = 2
                    if be_after_tp2:
                        pos_be_stop = pos_entry_price
                    if pos_remaining_contracts > 0:
                        pos_trail_stop = _find_nth_swing(
                            swing_low_mask, low_arr, i, trail_nth_swing, 1)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_entry_price
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp2
                        exit_reason = "tp2"
                        exit_contracts = pos_contracts
                        exited = True

                if not exited and pos_trim_stage == 2:
                    nt = _find_nth_swing(swing_low_mask, low_arr, i, trail_nth_swing, 1)
                    if not np.isnan(nt) and nt > pos_trail_stop:
                        pos_trail_stop = nt

            # ==================================================================
            # MULTI-TP EXIT MANAGEMENT (SHORTS)
            # ==================================================================
            elif not exited and pos_is_multi_tp and pos_direction == -1:
                if pos_trim_stage >= 2:
                    eff_stop = pos_trail_stop if pos_trail_stop > 0 else pos_stop
                    if pos_be_stop > 0:
                        eff_stop = min(eff_stop, pos_be_stop)
                elif pos_trim_stage == 1 and be_after_tp1:
                    eff_stop = pos_stop
                    if pos_be_stop > 0:
                        eff_stop = min(eff_stop, pos_be_stop)
                else:
                    eff_stop = pos_stop

                if h[i] >= eff_stop:
                    exit_price = eff_stop + slippage_points
                    if pos_trim_stage > 0 and eff_stop <= pos_entry_price:
                        exit_reason = "be_sweep"
                    else:
                        exit_reason = "stop"
                    exited = True

                elif pos_trim_stage == 0 and l[i] <= pos_tp1:
                    tc1 = max(1, int(pos_contracts * tp1_trim_pct))
                    pos_trim1_contracts = tc1
                    pos_remaining_contracts = pos_contracts - tc1
                    pos_tp1_pnl_pts = pos_entry_price - pos_tp1
                    pos_trim_stage = 1
                    if be_after_tp1:
                        pos_be_stop = pos_entry_price
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True

                if not exited and pos_trim_stage == 1 and l[i] <= pos_tp2:
                    if fix_phantom_runner:
                        tc2 = pos_remaining_contracts
                    else:
                        tc2 = max(1, int(pos_contracts * tp2_trim_pct))
                        tc2 = min(tc2, pos_remaining_contracts)
                    pos_trim2_contracts = tc2
                    pos_remaining_contracts -= tc2
                    pos_tp2_pnl_pts = pos_entry_price - pos_tp2
                    pos_trim_stage = 2
                    if be_after_tp2:
                        pos_be_stop = pos_entry_price
                    if pos_remaining_contracts > 0:
                        pos_trail_stop = _find_nth_swing(
                            swing_high_mask, high_arr, i, trail_nth_swing, -1)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_entry_price
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp2
                        exit_reason = "tp2"
                        exit_contracts = pos_contracts
                        exited = True

                if not exited and pos_trim_stage == 2:
                    nt = _find_nth_swing(swing_high_mask, high_arr, i, trail_nth_swing, -1)
                    if not np.isnan(nt) and nt < pos_trail_stop:
                        pos_trail_stop = nt

            # ==================================================================
            # SINGLE-TP EXIT MANAGEMENT
            # ==================================================================
            elif not exited and not pos_is_multi_tp:
                if pos_direction == 1:
                    eff_stop = pos_trail_stop if pos_single_trimmed and pos_trail_stop > 0 else pos_stop
                    if pos_single_trimmed and be_after_trim and pos_be_stop > 0:
                        eff_stop = max(eff_stop, pos_be_stop)
                    if l[i] <= eff_stop:
                        exit_price = eff_stop - slippage_points
                        exit_reason = "be_sweep" if (pos_single_trimmed and eff_stop >= pos_entry_price) else "stop"
                        exited = True
                    elif not pos_single_trimmed and h[i] >= pos_tp1:
                        tc = max(1, int(pos_contracts * pos_single_trim_pct))
                        pos_remaining_contracts = pos_contracts - tc
                        pos_single_trimmed = True
                        pos_be_stop = pos_entry_price
                        if pos_remaining_contracts > 0:
                            pos_trail_stop = _find_nth_swing(swing_low_mask, low_arr, i, trail_nth_swing, 1)
                            if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                                pos_trail_stop = pos_be_stop
                        if pos_remaining_contracts <= 0:
                            exit_price = pos_tp1
                            exit_reason = "tp1"
                            exit_contracts = pos_contracts
                            exited = True
                    if pos_single_trimmed and not exited:
                        nt = _find_nth_swing(swing_low_mask, low_arr, i, trail_nth_swing, 1)
                        if not np.isnan(nt) and nt > pos_trail_stop:
                            pos_trail_stop = nt
                else:  # SHORT single-TP
                    eff_stop = pos_trail_stop if pos_single_trimmed and pos_trail_stop > 0 else pos_stop
                    if pos_single_trimmed and be_after_trim and pos_be_stop > 0:
                        eff_stop = min(eff_stop, pos_be_stop)
                    if h[i] >= eff_stop:
                        exit_price = eff_stop + slippage_points
                        exit_reason = "be_sweep" if (pos_single_trimmed and eff_stop <= pos_entry_price) else "stop"
                        exited = True
                    elif not pos_single_trimmed and l[i] <= pos_tp1:
                        tc = max(1, int(pos_contracts * pos_single_trim_pct))
                        pos_remaining_contracts = pos_contracts - tc
                        pos_single_trimmed = True
                        pos_be_stop = pos_entry_price
                        if pos_remaining_contracts > 0:
                            pos_trail_stop = _find_nth_swing(swing_high_mask, high_arr, i, trail_nth_swing, -1)
                            if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                                pos_trail_stop = pos_be_stop
                        if pos_remaining_contracts <= 0:
                            exit_price = pos_tp1
                            exit_reason = "tp1"
                            exit_contracts = pos_contracts
                            exited = True
                    if pos_single_trimmed and not exited:
                        nt = _find_nth_swing(swing_high_mask, high_arr, i, trail_nth_swing, -1)
                        if not np.isnan(nt) and nt < pos_trail_stop:
                            pos_trail_stop = nt

            # ==================================================================
            # PNL CALCULATION ON EXIT
            # ==================================================================
            if exited:
                pnl_pts_runner = (exit_price - pos_entry_price) if pos_direction == 1 else (pos_entry_price - exit_price)

                if pos_is_multi_tp:
                    total_pnl = 0.0
                    if pos_trim_stage >= 1:
                        total_pnl += pos_tp1_pnl_pts * point_value * pos_trim1_contracts
                    if pos_trim_stage >= 2:
                        total_pnl += pos_tp2_pnl_pts * point_value * pos_trim2_contracts
                    total_pnl += pnl_pts_runner * point_value * exit_contracts
                    total_comm = commission_per_side * 2 * pos_contracts
                    total_pnl -= total_comm
                elif pos_single_trimmed and exit_reason not in ("tp1",):
                    trim_pnl_total = (pos_tp1 - pos_entry_price) if pos_direction == 1 else (pos_entry_price - pos_tp1)
                    trim_c = pos_contracts - exit_contracts
                    total_pnl = trim_pnl_total * point_value * trim_c + pnl_pts_runner * point_value * exit_contracts
                    total_comm = commission_per_side * 2 * pos_contracts
                    total_pnl -= total_comm
                else:
                    total_pnl = pnl_pts_runner * point_value * exit_contracts
                    total_comm = commission_per_side * 2 * exit_contracts
                    total_pnl -= total_comm

                stop_dist_exit = abs(pos_entry_price - pos_stop)
                total_risk = stop_dist_exit * point_value * pos_contracts
                r_mult = total_pnl / total_risk if total_risk > 0 else 0.0

                any_trimmed = pos_trim_stage > 0 or pos_single_trimmed

                trades.append({
                    "entry_time": nq.index[pos_entry_idx],
                    "exit_time": nq.index[i + 1] if exit_reason == "early_cut_pa" and i + 1 < end_idx else nq.index[i],
                    "r": r_mult, "reason": exit_reason, "dir": pos_direction,
                    "type": pos_signal_type, "trimmed": any_trimmed, "grade": pos_grade,
                    "entry_price": pos_entry_price, "exit_price": exit_price,
                    "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
                    "trim_stage": pos_trim_stage,
                })
                daily_pnl_r += r_mult
                # EOD close is neutral for 0-for-2 rule
                if exit_reason == "eod_close":
                    pass
                elif exit_reason == "be_sweep" and any_trimmed:
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
        tp1_raw = irl_target_arr[i]
        if np.isnan(entry_p) or np.isnan(stop) or np.isnan(tp1_raw):
            continue
        if direction == 1 and (stop >= entry_p or tp1_raw <= entry_p):
            continue
        if direction == -1 and (stop <= entry_p or tp1_raw >= entry_p):
            continue

        if block_pm_shorts:
            if et_frac >= 14.0 and direction == -1:
                continue

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

        if pa_alt_arr[i] >= pa_threshold:
            continue

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

        if not np.isnan(stop_atr_ratio[i]):
            if stop_atr_ratio[i] < min_stop_atr:
                continue
        else:
            stop_dist_check = abs(entry_p - stop)
            a_check = atr_arr[i] if not np.isnan(atr_arr[i]) else 10.0
            if a_check > 0 and (stop_dist_check / a_check) < min_stop_atr:
                continue

        if not np.isnan(signal_quality[i]):
            eff_sq = sq_long if direction == 1 else sq_short
            if signal_quality[i] < eff_sq:
                continue

        if i + 1 >= end_idx:
            continue

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

        actual_entry = entry_p + slippage_points if direction == 1 else entry_p - slippage_points
        stop_dist = abs(actual_entry - stop)
        if stop_dist < 1.0:
            continue

        contracts = max(1, int(r_amount / (stop_dist * point_value))) if stop_dist > 0 and point_value > 0 else 0
        if contracts <= 0:
            continue

        is_mss_signal = str(sig_type[i]) == "mss"

        apply_multi = (direction == 1) or (direction == -1 and use_multi_tp_for_shorts)

        if apply_multi:
            tp1_base = tp1_raw
            if session_rules.get("enabled", False) and 9.5 <= et_frac < 16.0:
                actual_tp_mult = ny_tp_mult
                if mss_mgmt_enabled and is_mss_signal and direction == 1:
                    actual_tp_mult = mss_long_tp_mult
                tp_distance = (tp1_raw - actual_entry) if direction == 1 else (actual_entry - tp1_raw)
                tp1_base = (actual_entry + tp_distance * actual_tp_mult) if direction == 1 else (actual_entry - tp_distance * actual_tp_mult)

            if direction == 1:
                tp1, tp2 = build_liquidity_ladder_long(
                    actual_entry, stop, tp1_base, i, d_extra)
            else:
                tp1, tp2 = build_liquidity_ladder_short(
                    actual_entry, stop, tp1_base, i, d_extra)

            in_position = True
            pos_direction = direction
            pos_entry_idx = i + 1
            pos_entry_price = actual_entry
            pos_stop = stop
            pos_tp1 = tp1
            pos_tp2 = tp2
            pos_contracts = contracts
            pos_remaining_contracts = contracts
            pos_trim_stage = 0
            pos_be_stop = 0.0
            pos_trail_stop = 0.0
            pos_signal_type = str(sig_type[i])
            pos_bias_dir = bias_dir_arr[i]
            pos_regime = regime_arr[i]
            pos_grade = grade
            pos_is_multi_tp = True
            pos_trim1_contracts = 0
            pos_trim2_contracts = 0
            pos_tp1_pnl_pts = 0.0
            pos_tp2_pnl_pts = 0.0
            pos_single_trimmed = False
        else:
            tp1 = tp1_raw
            if session_rules.get("enabled", False):
                if dir_mgmt.get("enabled", False):
                    actual_tp_mult = dir_mgmt.get("short_tp_mult", 1.25)
                else:
                    actual_tp_mult = ny_tp_mult
                if 9.5 <= et_frac < 16.0:
                    tp_distance = actual_entry - tp1
                    tp1 = actual_entry - tp_distance * actual_tp_mult

            dual_mode_enabled = dual_mode.get("enabled", False)
            if dual_mode_enabled and direction == -1:
                short_rr = dual_mode.get("short_rr", 0.625)
                if mss_mgmt_enabled and is_mss_signal:
                    short_rr = mss_mgmt.get("short_rr", short_rr)
                tp1 = actual_entry - stop_dist * short_rr

            in_position = True
            pos_direction = direction
            pos_entry_idx = i + 1
            pos_entry_price = actual_entry
            pos_stop = stop
            pos_tp1 = tp1
            pos_tp2 = 0.0
            pos_contracts = contracts
            pos_remaining_contracts = contracts
            pos_trim_stage = 0
            pos_be_stop = 0.0
            pos_trail_stop = 0.0
            pos_signal_type = str(sig_type[i])
            pos_bias_dir = bias_dir_arr[i]
            pos_regime = regime_arr[i]
            pos_grade = grade
            pos_is_multi_tp = False
            pos_trim1_contracts = 0
            pos_trim2_contracts = 0
            pos_tp1_pnl_pts = 0.0
            pos_tp2_pnl_pts = 0.0
            pos_single_trimmed = False

            if mss_mgmt_enabled and is_mss_signal:
                pos_single_trim_pct = mss_mgmt.get("short_trim_pct", 1.0) if direction == -1 else mss_mgmt.get("long_trim_pct", default_trim_pct)
            elif dual_mode_enabled and direction == -1:
                pos_single_trim_pct = dual_mode.get("short_trim_pct", 1.0)
            elif dir_mgmt.get("enabled", False):
                pos_single_trim_pct = dir_mgmt.get("long_trim_pct", default_trim_pct) if direction == 1 else dir_mgmt.get("short_trim_pct", 1.0)
            else:
                pos_single_trim_pct = default_trim_pct

    # Force close at data end (safety — EOD should handle everything)
    if in_position and pos_entry_idx < end_idx:
        last_i = end_idx - 1
        exit_price = c_arr[last_i]
        pnl_pts = (exit_price - pos_entry_price) if pos_direction == 1 else (pos_entry_price - exit_price)
        if pos_is_multi_tp:
            total_pnl = 0.0
            if pos_trim_stage >= 1:
                total_pnl += pos_tp1_pnl_pts * point_value * pos_trim1_contracts
            if pos_trim_stage >= 2:
                total_pnl += pos_tp2_pnl_pts * point_value * pos_trim2_contracts
            total_pnl += pnl_pts * point_value * pos_remaining_contracts
            total_pnl -= commission_per_side * 2 * pos_contracts
        elif pos_single_trimmed:
            trim_c = pos_contracts - pos_remaining_contracts
            trim_pnl = ((pos_tp1 - pos_entry_price) if pos_direction == 1 else (pos_entry_price - pos_tp1)) * point_value * trim_c
            total_pnl = pnl_pts * point_value * pos_remaining_contracts + trim_pnl
            total_pnl -= commission_per_side * 2 * pos_contracts
        else:
            total_pnl = pnl_pts * point_value * pos_remaining_contracts
            total_pnl -= commission_per_side * 2 * pos_remaining_contracts
        stop_dist_exit = abs(pos_entry_price - pos_stop)
        total_risk = stop_dist_exit * point_value * pos_contracts
        r_mult = total_pnl / total_risk if total_risk > 0 else 0.0
        trades.append({
            "entry_time": nq.index[pos_entry_idx],
            "exit_time": nq.index[last_i],
            "r": r_mult, "reason": "data_end", "dir": pos_direction,
            "type": pos_signal_type, "trimmed": pos_trim_stage > 0 or pos_single_trimmed,
            "grade": pos_grade,
            "entry_price": pos_entry_price, "exit_price": exit_price,
            "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
            "trim_stage": pos_trim_stage,
        })

    return trades


# ======================================================================
# Analysis helpers
# ======================================================================
def eod_close_analysis(trades_list: list[dict], label: str) -> dict:
    """Analyze impact of EOD close on a trade list."""
    if not trades_list:
        return {}
    total = len(trades_list)
    eod_trades = [t for t in trades_list if t["reason"] == "eod_close"]
    n_eod = len(eod_trades)
    eod_pct = 100.0 * n_eod / total if total > 0 else 0
    eod_r = sum(t["r"] for t in eod_trades) if eod_trades else 0
    eod_avg_r = eod_r / n_eod if n_eod > 0 else 0
    non_eod = [t for t in trades_list if t["reason"] != "eod_close"]
    non_eod_r = sum(t["r"] for t in non_eod) if non_eod else 0

    # Breakdown of EOD trades by trim status
    eod_trimmed = [t for t in eod_trades if t.get("trimmed", False)]
    eod_untrimmed = [t for t in eod_trades if not t.get("trimmed", False)]
    eod_trimmed_r = sum(t["r"] for t in eod_trimmed) if eod_trimmed else 0
    eod_untrimmed_r = sum(t["r"] for t in eod_untrimmed) if eod_untrimmed else 0

    print(f"\n  EOD CLOSE ANALYSIS — {label}")
    print(f"    Total trades: {total}")
    print(f"    EOD-closed:   {n_eod} ({eod_pct:.1f}%)")
    print(f"    EOD total R:  {eod_r:+.1f}")
    print(f"    EOD avg R:    {eod_avg_r:+.4f}")
    print(f"    EOD trimmed:  {len(eod_trimmed)} trades, R={eod_trimmed_r:+.1f}")
    print(f"    EOD untrimmed:{len(eod_untrimmed)} trades, R={eod_untrimmed_r:+.1f}")
    print(f"    Non-EOD R:    {non_eod_r:+.1f}")
    print(f"    Same-day %:   {100 - eod_pct:.1f}% (completed within day)")

    return {
        "n_eod": n_eod, "eod_pct": eod_pct, "eod_r": eod_r,
        "eod_avg_r": eod_avg_r, "non_eod_r": non_eod_r,
    }


def print_comparison_table(results: list[tuple[str, dict, dict | None]]):
    """Print a formatted comparison table. Each entry: (label, metrics, eod_analysis)."""
    print("\n" + "=" * 130)
    print(f"  {'Config':<45s} | {'Trades':>6s} | {'R':>9s} | {'PPDD':>7s} | {'PF':>6s} | {'WR':>6s} | {'MaxDD':>7s} | {'EOD%':>5s} | {'EOD_R':>8s}")
    print("-" * 130)
    for label, m, eod_a in results:
        eod_pct = eod_a.get("eod_pct", 0) if eod_a else 0
        eod_r = eod_a.get("eod_r", 0) if eod_a else 0
        print(f"  {label:<45s} | {m['trades']:6d} | {m['R']:+9.1f} | {m['PPDD']:7.2f} | {m['PF']:6.2f} | {m['WR']:5.1f}% | {m['MaxDD']:7.1f} | {eod_pct:4.1f}% | {eod_r:+8.1f}")
    print("=" * 130)


def print_walk_forward(trades_list: list[dict], label: str):
    """Print per-year walk-forward breakdown."""
    wf = walk_forward_metrics(trades_list)
    if not wf:
        print(f"  {label}: no trades")
        return
    neg_years = sum(1 for y in wf if y["R"] < 0)
    total_years = len(wf)
    print(f"\n  Walk-Forward: {label} ({total_years} years, {neg_years} negative)")
    print(f"    {'Year':<6s} | {'N':>4s} | {'R':>8s} | {'PPDD':>7s} | {'PF':>6s} | {'WR':>6s}")
    print("    " + "-" * 50)
    for y in wf:
        print(f"    {y['year']:<6d} | {y['n']:4d} | {y['R']:+8.1f} | {y['PPDD']:7.2f} | {y['PF']:6.2f} | {y['WR']:5.1f}%")


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 130)
    print("EOD CLOSE VALIDATION — MANDATORY 15:55 ET CLOSE FOR ALL CONFIGS")
    print("Hell audit finding: Config F (+624R) was a multi-day trend follower.")
    print("Same-day trades NET LOSE (-31.3R). All prop firm backtests must enforce EOD close.")
    print("=" * 130)

    t_start = _time.perf_counter()
    d = load_all()
    d_extra = prepare_liquidity_data(d)

    all_results = []

    # ==================================================================
    # STEP 2A: Single-TP configs with EOD close
    # ==================================================================
    print("\n" + "=" * 130)
    print("STEP 2A: SINGLE-TP CONFIGS WITH MANDATORY EOD CLOSE")
    print("=" * 130)

    # Config 1: Config D baseline (sq_short=0.80, block_pm_shorts, trim=0.50, trail=2, ny_tp=2.0)
    print("\n--- Config 1: Config D baseline (trim=0.50, trail=2, ny_tp=2.0) ---")
    t0 = _time.perf_counter()
    c1_trades = run_backtest_improved_eod(
        d, sq_short=0.80, block_pm_shorts=True,
        trim_pct_override=0.50, trail_nth_override=2, ny_tp_mult_override=2.0)
    c1_m = compute_metrics(c1_trades)
    print_metrics("C1: Config D baseline EOD", c1_m)
    c1_eod = eod_close_analysis(c1_trades, "C1")
    all_results.append(("C1: ConfigD base (t50/tr2/tp2.0)", c1_m, c1_eod))
    print(f"  ({_time.perf_counter() - t0:.1f}s)")

    # Config 2: Config D + trim=0.25 + trail=3
    print("\n--- Config 2: Config D + trim=0.25 + trail=3 + ny_tp=2.0 ---")
    t0 = _time.perf_counter()
    c2_trades = run_backtest_improved_eod(
        d, sq_short=0.80, block_pm_shorts=True,
        trim_pct_override=0.25, trail_nth_override=3, ny_tp_mult_override=2.0)
    c2_m = compute_metrics(c2_trades)
    print_metrics("C2: trim=0.25, trail=3, tp=2.0", c2_m)
    c2_eod = eod_close_analysis(c2_trades, "C2")
    all_results.append(("C2: ConfigD t25/tr3/tp2.0", c2_m, c2_eod))
    print(f"  ({_time.perf_counter() - t0:.1f}s)")

    # Config 3: Config D + ny_tp=3.0
    print("\n--- Config 3: Config D + ny_tp=3.0 ---")
    t0 = _time.perf_counter()
    c3_trades = run_backtest_improved_eod(
        d, sq_short=0.80, block_pm_shorts=True,
        trim_pct_override=0.50, trail_nth_override=2, ny_tp_mult_override=3.0)
    c3_m = compute_metrics(c3_trades)
    print_metrics("C3: trim=0.50, trail=2, tp=3.0", c3_m)
    c3_eod = eod_close_analysis(c3_trades, "C3")
    all_results.append(("C3: ConfigD t50/tr2/tp3.0", c3_m, c3_eod))
    print(f"  ({_time.perf_counter() - t0:.1f}s)")

    # Config 4: Config D + trim=0.25 + trail=3 + ny_tp=3.0
    print("\n--- Config 4: Config D + trim=0.25 + trail=3 + ny_tp=3.0 ---")
    t0 = _time.perf_counter()
    c4_trades = run_backtest_improved_eod(
        d, sq_short=0.80, block_pm_shorts=True,
        trim_pct_override=0.25, trail_nth_override=3, ny_tp_mult_override=3.0)
    c4_m = compute_metrics(c4_trades)
    print_metrics("C4: trim=0.25, trail=3, tp=3.0", c4_m)
    c4_eod = eod_close_analysis(c4_trades, "C4")
    all_results.append(("C4: ConfigD t25/tr3/tp3.0", c4_m, c4_eod))
    print(f"  ({_time.perf_counter() - t0:.1f}s)")

    # ==================================================================
    # STEP 2B: Multi-TP configs with EOD close
    # ==================================================================
    print("\n" + "=" * 130)
    print("STEP 2B: MULTI-TP CONFIGS WITH MANDATORY EOD CLOSE")
    print("=" * 130)

    # Config 5: Multi-TP 25/25/50, ny_tp=3.0, trail=3 (Config E)
    print("\n--- Config 5: Multi-TP 25/25/50, ny_tp=3.0, trail=3 (Config E) ---")
    t0 = _time.perf_counter()
    c5_trades = run_backtest_multi_tp_eod(
        d, d_extra,
        tp1_trim_pct=0.25, tp2_trim_pct=0.25,
        be_after_tp1=False, be_after_tp2=True,
        trail_nth_swing=3, ny_tp_mult=3.0)
    c5_m = compute_metrics(c5_trades)
    print_metrics("C5: MultiTP 25/25/50 tp3.0 tr3", c5_m)
    c5_eod = eod_close_analysis(c5_trades, "C5")
    all_results.append(("C5: Multi 25/25/50 tp3.0 tr3", c5_m, c5_eod))
    print(f"  ({_time.perf_counter() - t0:.1f}s)")

    # Config 6: Multi-TP 50/50/0, ny_tp=3.0
    print("\n--- Config 6: Multi-TP 50/50/0, ny_tp=3.0 ---")
    t0 = _time.perf_counter()
    c6_trades = run_backtest_multi_tp_eod(
        d, d_extra,
        tp1_trim_pct=0.50, tp2_trim_pct=0.50,
        be_after_tp1=False, be_after_tp2=True,
        trail_nth_swing=3, ny_tp_mult=3.0)
    c6_m = compute_metrics(c6_trades)
    print_metrics("C6: MultiTP 50/50/0 tp3.0", c6_m)
    c6_eod = eod_close_analysis(c6_trades, "C6")
    all_results.append(("C6: Multi 50/50/0 tp3.0", c6_m, c6_eod))
    print(f"  ({_time.perf_counter() - t0:.1f}s)")

    # Config 7: Multi-TP 50/50/0, ny_tp=2.0
    print("\n--- Config 7: Multi-TP 50/50/0, ny_tp=2.0 ---")
    t0 = _time.perf_counter()
    c7_trades = run_backtest_multi_tp_eod(
        d, d_extra,
        tp1_trim_pct=0.50, tp2_trim_pct=0.50,
        be_after_tp1=False, be_after_tp2=True,
        trail_nth_swing=3, ny_tp_mult=2.0)
    c7_m = compute_metrics(c7_trades)
    print_metrics("C7: MultiTP 50/50/0 tp2.0", c7_m)
    c7_eod = eod_close_analysis(c7_trades, "C7")
    all_results.append(("C7: Multi 50/50/0 tp2.0", c7_m, c7_eod))
    print(f"  ({_time.perf_counter() - t0:.1f}s)")

    # Config 8: Multi-TP 50/50/0, ny_tp=1.5
    print("\n--- Config 8: Multi-TP 50/50/0, ny_tp=1.5 ---")
    t0 = _time.perf_counter()
    c8_trades = run_backtest_multi_tp_eod(
        d, d_extra,
        tp1_trim_pct=0.50, tp2_trim_pct=0.50,
        be_after_tp1=False, be_after_tp2=True,
        trail_nth_swing=3, ny_tp_mult=1.5)
    c8_m = compute_metrics(c8_trades)
    print_metrics("C8: MultiTP 50/50/0 tp1.5", c8_m)
    c8_eod = eod_close_analysis(c8_trades, "C8")
    all_results.append(("C8: Multi 50/50/0 tp1.5", c8_m, c8_eod))
    print(f"  ({_time.perf_counter() - t0:.1f}s)")

    # Config 9: Multi-TP 25/25/50, ny_tp=2.0 (conservative)
    print("\n--- Config 9: Multi-TP 25/25/50, ny_tp=2.0 (conservative) ---")
    t0 = _time.perf_counter()
    c9_trades = run_backtest_multi_tp_eod(
        d, d_extra,
        tp1_trim_pct=0.25, tp2_trim_pct=0.25,
        be_after_tp1=False, be_after_tp2=True,
        trail_nth_swing=3, ny_tp_mult=2.0)
    c9_m = compute_metrics(c9_trades)
    print_metrics("C9: MultiTP 25/25/50 tp2.0", c9_m)
    c9_eod = eod_close_analysis(c9_trades, "C9")
    all_results.append(("C9: Multi 25/25/50 tp2.0 conserv", c9_m, c9_eod))
    print(f"  ({_time.perf_counter() - t0:.1f}s)")

    # ==================================================================
    # STEP 3: Comparison table
    # ==================================================================
    print("\n" + "=" * 130)
    print("STEP 3: FULL COMPARISON TABLE (ALL CONFIGS WITH EOD CLOSE)")
    print("=" * 130)
    print_comparison_table(all_results)

    # ==================================================================
    # STEP 4: Walk-forward for top 3 by PPDD
    # ==================================================================
    print("\n" + "=" * 130)
    print("STEP 4: WALK-FORWARD FOR TOP 3 BY PPDD")
    print("=" * 130)

    # Sort by PPDD to find top 3
    sorted_results = sorted(all_results, key=lambda x: x[1]["PPDD"], reverse=True)
    top3_labels = [sr[0] for sr in sorted_results[:3]]
    print(f"  Top 3 by PPDD: {top3_labels}")

    # Map labels to trade lists
    all_trades = {
        "C1: ConfigD base (t50/tr2/tp2.0)": c1_trades,
        "C2: ConfigD t25/tr3/tp2.0": c2_trades,
        "C3: ConfigD t50/tr2/tp3.0": c3_trades,
        "C4: ConfigD t25/tr3/tp3.0": c4_trades,
        "C5: Multi 25/25/50 tp3.0 tr3": c5_trades,
        "C6: Multi 50/50/0 tp3.0": c6_trades,
        "C7: Multi 50/50/0 tp2.0": c7_trades,
        "C8: Multi 50/50/0 tp1.5": c8_trades,
        "C9: Multi 25/25/50 tp2.0 conserv": c9_trades,
    }

    for label, m, _ in sorted_results[:3]:
        print_walk_forward(all_trades[label], label)

    # ==================================================================
    # STEP 5: Phantom runner fix for 50/50/0 configs
    # ==================================================================
    print("\n" + "=" * 130)
    print("STEP 5: PHANTOM RUNNER FIX — 50/50/0 CONFIGS")
    print("Odd contracts: e.g. 3 contracts -> TP1 trim 1, TP2 trim 1 -> phantom runner of 1")
    print("Fix: TP2 takes ALL remaining contracts, no phantom runner")
    print("=" * 130)

    phantom_results = []

    # Config 6 fixed
    print("\n--- C6-fixed: 50/50/0 tp3.0 (phantom runner fix) ---")
    t0 = _time.perf_counter()
    c6f_trades = run_backtest_multi_tp_eod(
        d, d_extra,
        tp1_trim_pct=0.50, tp2_trim_pct=0.50,
        be_after_tp1=False, be_after_tp2=True,
        trail_nth_swing=3, ny_tp_mult=3.0,
        fix_phantom_runner=True)
    c6f_m = compute_metrics(c6f_trades)
    print_metrics("C6-fixed: 50/50/0 tp3.0 (no phantom)", c6f_m)
    c6f_eod = eod_close_analysis(c6f_trades, "C6-fixed")
    phantom_results.append(("C6-fixed: 50/50/0 tp3.0 no phantom", c6f_m, c6f_eod))
    print(f"  ({_time.perf_counter() - t0:.1f}s)")

    # Config 7 fixed
    print("\n--- C7-fixed: 50/50/0 tp2.0 (phantom runner fix) ---")
    t0 = _time.perf_counter()
    c7f_trades = run_backtest_multi_tp_eod(
        d, d_extra,
        tp1_trim_pct=0.50, tp2_trim_pct=0.50,
        be_after_tp1=False, be_after_tp2=True,
        trail_nth_swing=3, ny_tp_mult=2.0,
        fix_phantom_runner=True)
    c7f_m = compute_metrics(c7f_trades)
    print_metrics("C7-fixed: 50/50/0 tp2.0 (no phantom)", c7f_m)
    c7f_eod = eod_close_analysis(c7f_trades, "C7-fixed")
    phantom_results.append(("C7-fixed: 50/50/0 tp2.0 no phantom", c7f_m, c7f_eod))
    print(f"  ({_time.perf_counter() - t0:.1f}s)")

    # Config 8 fixed
    print("\n--- C8-fixed: 50/50/0 tp1.5 (phantom runner fix) ---")
    t0 = _time.perf_counter()
    c8f_trades = run_backtest_multi_tp_eod(
        d, d_extra,
        tp1_trim_pct=0.50, tp2_trim_pct=0.50,
        be_after_tp1=False, be_after_tp2=True,
        trail_nth_swing=3, ny_tp_mult=1.5,
        fix_phantom_runner=True)
    c8f_m = compute_metrics(c8f_trades)
    print_metrics("C8-fixed: 50/50/0 tp1.5 (no phantom)", c8f_m)
    c8f_eod = eod_close_analysis(c8f_trades, "C8-fixed")
    phantom_results.append(("C8-fixed: 50/50/0 tp1.5 no phantom", c8f_m, c8f_eod))
    print(f"  ({_time.perf_counter() - t0:.1f}s)")

    # Phantom runner comparison
    print("\n--- Phantom Runner Fix Comparison ---")
    print(f"  {'Config':<45s} | {'R':>9s} | {'PPDD':>7s} | {'PF':>6s} | {'Notes'}")
    print("  " + "-" * 90)
    for orig, fixed in [
        (("C6 unfixed", c6_m), ("C6 fixed", c6f_m)),
        (("C7 unfixed", c7_m), ("C7 fixed", c7f_m)),
        (("C8 unfixed", c8_m), ("C8 fixed", c8f_m)),
    ]:
        dr = fixed[1]["R"] - orig[1]["R"]
        dp = fixed[1]["PPDD"] - orig[1]["PPDD"]
        print(f"  {orig[0]:<45s} | {orig[1]['R']:+9.1f} | {orig[1]['PPDD']:7.2f} | {orig[1]['PF']:6.2f} |")
        print(f"  {fixed[0]:<45s} | {fixed[1]['R']:+9.1f} | {fixed[1]['PPDD']:7.2f} | {fixed[1]['PF']:6.2f} | dR={dr:+.1f}, dPPDD={dp:+.2f}")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("\n" + "=" * 130)
    print("FINAL SUMMARY")
    print("=" * 130)

    # Combine all results including phantom-fixed
    all_combined = all_results + phantom_results
    sorted_all = sorted(all_combined, key=lambda x: x[1]["PPDD"], reverse=True)

    print("\n  ALL CONFIGS RANKED BY PPDD:")
    print(f"  {'Rank':<5s} {'Config':<45s} | {'Trades':>6s} | {'R':>9s} | {'PPDD':>7s} | {'PF':>6s} | {'WR':>6s} | {'MaxDD':>7s}")
    print("  " + "-" * 100)
    for rank, (label, m, _) in enumerate(sorted_all, 1):
        profitable = "***" if m["R"] > 0 else "   "
        print(f"  {rank:<5d} {label:<45s} | {m['trades']:6d} | {m['R']:+9.1f} | {m['PPDD']:7.2f} | {m['PF']:6.2f} | {m['WR']:5.1f}% | {m['MaxDD']:7.1f} {profitable}")

    # Walk-forward for phantom-fixed configs
    print("\n  WALK-FORWARD FOR TOP 3 PHANTOM-FIXED CONFIGS:")
    for label, trades_data in [
        ("C8-fixed: 50/50/0 tp1.5 no phantom", c8f_trades),
        ("C7-fixed: 50/50/0 tp2.0 no phantom", c7f_trades),
        ("C6-fixed: 50/50/0 tp3.0 no phantom", c6f_trades),
    ]:
        print_walk_forward(trades_data, label)

    # Verdict
    profitable_configs = [(l, m) for l, m, _ in sorted_all if m["R"] > 0]
    if profitable_configs:
        print(f"\n  VERDICT: {len(profitable_configs)} / {len(sorted_all)} configs are PROFITABLE with EOD close.")
        best_label, best_m = profitable_configs[0]
        print(f"  BEST: {best_label}")
        print(f"         R={best_m['R']:+.1f} | PPDD={best_m['PPDD']:.2f} | PF={best_m['PF']:.2f} | WR={best_m['WR']:.1f}% | MaxDD={best_m['MaxDD']:.1f}R")
    else:
        print(f"\n  VERDICT: NO CONFIG IS PROFITABLE WITH EOD CLOSE.")
        print("  The strategy is fundamentally a multi-day trend follower, not an intraday system.")
        print("  This is a critical finding — the entire strategy premise needs to be reconsidered.")

    elapsed = _time.perf_counter() - t_start
    print(f"\n  Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
