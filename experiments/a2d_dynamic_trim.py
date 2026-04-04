"""
experiments/a2d_dynamic_trim.py — Dynamic trim allocation based on signal conviction.

Instead of fixed 25/25/50 for all signals, adjust the split based on signal characteristics:
  - High conviction (A+, aligned, morning, high fluency) -> 10/10/80 (big runner)
  - Low conviction (C grade, PM, low fluency) -> 40/40/20 (quick lock)
  - Medium -> 25/25/50 (current default)

Steps:
  1. Fixed trim split sweep on multi-TP engine
  2. Analyze which signals benefit from runner vs trim (by grade/type/hour/fluency/regime)
  3. Design conviction-based split rules from Step 2 findings
  4. Backtest conviction-based dynamic trim
  5. Compare dynamic vs best-fixed split

Usage: python experiments/a2d_dynamic_trim.py
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
    run_backtest_multi_tp,
)
from features.swing import compute_swing_levels


# ======================================================================
# ENHANCED Multi-TP backtest with per-signal trim + feature recording
# ======================================================================
def run_backtest_dynamic_trim(
    d: dict,
    d_extra: dict,
    # Config E base
    sq_long: float = 0.68,
    sq_short: float = 0.80,
    min_stop_atr: float = 1.7,
    block_pm_shorts: bool = True,
    # Multi-TP defaults
    default_tp1_trim: float = 0.25,
    default_tp2_trim: float = 0.25,
    be_after_tp1: bool = False,
    be_after_tp2: bool = True,
    trail_nth_swing: int = 3,
    use_multi_tp_for_shorts: bool = False,
    ny_tp_mult: float = 3.0,
    mss_long_tp_mult: float = 2.5,
    # DYNAMIC TRIM: per-signal override
    # trim_lookup: dict mapping conviction -> (tp1_trim, tp2_trim)
    # conviction_fn: callable(signal_features_dict) -> conviction_level_str
    trim_lookup: dict | None = None,
    conviction_fn=None,
    # Date range
    start_date=None, end_date=None,
) -> list[dict]:
    """Multi-TP backtest with per-signal dynamic trim allocation.

    If trim_lookup and conviction_fn are provided, each signal's trim split
    is determined by conviction_fn(features) -> key into trim_lookup.
    Otherwise uses default_tp1_trim / default_tp2_trim for all signals.
    """
    params = d["params"]
    n = d["n"]
    o, h, l, c_arr = d["o"], d["h"], d["l"], d["c"]
    atr_arr = d["atr_arr"]
    fluency_arr = d["fluency_arr"]
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
    default_single_trim_pct = trim_params["pct"]

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
    pos_single_trimmed = False
    pos_single_trim_pct = default_single_trim_pct
    # Per-signal trim splits
    pos_tp1_trim_pct = default_tp1_trim
    pos_tp2_trim_pct = default_tp2_trim
    # Feature recording
    pos_fluency = 0.0
    pos_hour_et = 0.0
    pos_conviction = ""
    pos_signal_idx = 0

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

            # Early cut on bad PA (pre-trim only)
            if pos_trim_stage == 0 and not pos_single_trimmed and 2 <= bars_in_trade <= 4:
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

            # MULTI-TP LONG
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
                    tc1 = max(1, int(pos_contracts * pos_tp1_trim_pct))
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
                    tc2 = max(1, int(pos_contracts * pos_tp2_trim_pct))
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

            # MULTI-TP SHORT
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
                    tc1 = max(1, int(pos_contracts * pos_tp1_trim_pct))
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
                    tc2 = max(1, int(pos_contracts * pos_tp2_trim_pct))
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

            # SINGLE-TP (shorts fallback)
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
                else:
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

            # PNL CALCULATION ON EXIT
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
                elif pos_single_trimmed and exit_reason != "tp1":
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
                    # Extra feature fields for analysis
                    "fluency": pos_fluency,
                    "hour_et": pos_hour_et,
                    "regime": pos_regime,
                    "conviction": pos_conviction,
                    "signal_idx": pos_signal_idx,
                    "tp1_trim_used": pos_tp1_trim_pct,
                    "tp2_trim_used": pos_tp2_trim_pct,
                })
                daily_pnl_r += r_mult
                if exit_reason == "be_sweep" and any_trimmed:
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

        actual_entry = entry_p + slippage_points if direction == 1 else entry_p - slippage_points
        stop_dist = abs(actual_entry - stop)
        if stop_dist < 1.0:
            continue

        contracts = max(1, int(r_amount / (stop_dist * point_value))) if stop_dist > 0 and point_value > 0 else 0
        if contracts <= 0:
            continue

        is_mss_signal = str(sig_type[i]) == "mss"
        apply_multi = (direction == 1) or (direction == -1 and use_multi_tp_for_shorts)

        # --- Capture signal features for dynamic trim ---
        sig_fluency = fluency_arr[i] if not np.isnan(fluency_arr[i]) else 0.5
        sig_hour_et = et_frac
        sig_regime = regime_arr[i]

        # Determine per-signal trim split
        if trim_lookup is not None and conviction_fn is not None and apply_multi:
            sig_features = {
                "grade": grade,
                "signal_type": str(sig_type[i]),
                "hour_et": sig_hour_et,
                "fluency": sig_fluency,
                "regime": sig_regime,
                "direction": direction,
            }
            conv_level = conviction_fn(sig_features)
            tp1_trim, tp2_trim = trim_lookup.get(conv_level, (default_tp1_trim, default_tp2_trim))
        else:
            conv_level = "default"
            tp1_trim = default_tp1_trim
            tp2_trim = default_tp2_trim

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
            pos_tp1_trim_pct = tp1_trim
            pos_tp2_trim_pct = tp2_trim
            pos_fluency = sig_fluency
            pos_hour_et = sig_hour_et
            pos_conviction = conv_level
            pos_signal_idx = i
        else:
            # SHORT with single-TP
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
            pos_tp1_trim_pct = default_tp1_trim
            pos_tp2_trim_pct = default_tp2_trim
            pos_fluency = sig_fluency
            pos_hour_et = sig_hour_et
            pos_conviction = conv_level
            pos_signal_idx = i

            if mss_mgmt_enabled and is_mss_signal:
                pos_single_trim_pct = mss_mgmt.get("short_trim_pct", 1.0) if direction == -1 else mss_mgmt.get("long_trim_pct", default_single_trim_pct)
            elif dual_mode_enabled and direction == -1:
                pos_single_trim_pct = dual_mode.get("short_trim_pct", 1.0)
            elif dir_mgmt.get("enabled", False):
                pos_single_trim_pct = dir_mgmt.get("long_trim_pct", default_single_trim_pct) if direction == 1 else dir_mgmt.get("short_trim_pct", 1.0)
            else:
                pos_single_trim_pct = default_single_trim_pct

    # Force close at end
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
            "r": r_mult, "reason": "eod_close", "dir": pos_direction,
            "type": pos_signal_type, "trimmed": pos_trim_stage > 0 or pos_single_trimmed,
            "grade": pos_grade,
            "entry_price": pos_entry_price, "exit_price": exit_price,
            "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
            "trim_stage": pos_trim_stage,
            "fluency": pos_fluency,
            "hour_et": pos_hour_et,
            "regime": pos_regime,
            "conviction": pos_conviction,
            "signal_idx": pos_signal_idx,
            "tp1_trim_used": pos_tp1_trim_pct,
            "tp2_trim_used": pos_tp2_trim_pct,
        })

    return trades


# ======================================================================
# Conviction functions
# ======================================================================
def conviction_3tier(features: dict) -> str:
    """3-tier conviction: high / mid / low."""
    grade = features["grade"]
    fluency = features["fluency"]
    regime = features["regime"]
    hour = features["hour_et"]

    is_am = 10.0 <= hour < 12.5
    high_flu = fluency >= 0.7
    full_regime = regime >= 1.0

    # High conviction: A+ or (B+ AND morning AND high fluency AND full regime)
    if grade == "A+" and full_regime and high_flu:
        return "high"
    if grade == "A+" and (full_regime or high_flu):
        return "high"
    if grade == "B+" and is_am and high_flu and full_regime:
        return "high"

    # Low conviction: C grade, OR (PM AND low fluency), OR partial regime + low fluency
    if grade == "C":
        return "low"
    if hour >= 13.0 and fluency < 0.55:
        return "low"
    if regime < 1.0 and fluency < 0.55:
        return "low"

    return "mid"


def conviction_2tier(features: dict) -> str:
    """Simple 2-tier: high vs low."""
    grade = features["grade"]
    fluency = features["fluency"]
    regime = features["regime"]
    hour = features["hour_et"]

    if grade in ("A+", "B+") and fluency >= 0.65 and regime >= 1.0:
        return "high"
    return "low"


# ======================================================================
# Step 2: Signal analysis functions
# ======================================================================
def analyze_signal_groups(trades: list[dict]) -> None:
    """Analyze which signal characteristics benefit from more runner vs more trim."""
    if not trades:
        print("  No trades to analyze.")
        return

    df = pd.DataFrame(trades)
    # Focus on longs with multi-TP
    longs = df[df["dir"] == 1].copy()
    if len(longs) == 0:
        print("  No long trades to analyze.")
        return

    print(f"\n  Total trades: {len(df)} (longs={len(longs)}, shorts={len(df)-len(longs)})")

    # Classify features
    longs["flu_group"] = np.where(longs["fluency"] >= 0.7, "high_flu", "low_flu")
    longs["hour_group"] = np.where(longs["hour_et"] < 12.5, "AM(10-12:30)", "PM(13-16)")
    longs["regime_group"] = np.where(longs["regime"] >= 1.0, "full_regime", "partial_regime")

    def group_stats(subdf, label):
        n_total = len(subdf)
        if n_total == 0:
            return
        r_arr = subdf["r"].values
        total_r = r_arr.sum()
        avg_r = r_arr.mean()
        wr = (r_arr > 0).mean() * 100

        # Trim stage distribution
        n_s0 = (subdf["trim_stage"] == 0).sum()
        n_s1 = (subdf["trim_stage"] == 1).sum()
        n_s2 = (subdf["trim_stage"] >= 2).sum()
        pct_tp1 = (n_s1 + n_s2) / n_total * 100
        pct_tp2 = n_s2 / n_total * 100

        # R from each stage
        r_s0 = subdf.loc[subdf["trim_stage"] == 0, "r"].sum()
        r_s1 = subdf.loc[subdf["trim_stage"] == 1, "r"].sum()
        r_s2 = subdf.loc[subdf["trim_stage"] >= 2, "r"].sum()

        # Runner contribution for S2 trades (where runner was active)
        s2_trades = subdf[subdf["trim_stage"] >= 2]
        if len(s2_trades) > 0:
            s2_avg_r = s2_trades["r"].mean()
        else:
            s2_avg_r = 0.0

        print(f"    {label:30s} | {n_total:4d}t | R={total_r:+7.1f} | avgR={avg_r:+.4f} | WR={wr:4.1f}%"
              f" | TP1%={pct_tp1:4.1f} | TP2%={pct_tp2:4.1f} | R(S0)={r_s0:+6.1f} | R(S1)={r_s1:+6.1f} | R(S2+)={r_s2:+6.1f}"
              f" | S2 avgR={s2_avg_r:+.3f}")

    # By grade
    print("\n  === BY GRADE (longs only) ===")
    for g in ["A+", "B+", "C"]:
        sub = longs[longs["grade"] == g]
        group_stats(sub, f"Grade={g}")

    # By signal type
    print("\n  === BY SIGNAL TYPE (longs only) ===")
    for st in longs["type"].unique():
        sub = longs[longs["type"] == st]
        group_stats(sub, f"Type={st}")

    # By hour
    print("\n  === BY HOUR GROUP (longs only) ===")
    for hg in ["AM(10-12:30)", "PM(13-16)"]:
        sub = longs[longs["hour_group"] == hg]
        group_stats(sub, f"Time={hg}")

    # By fluency
    print("\n  === BY FLUENCY (longs only) ===")
    for fg in ["high_flu", "low_flu"]:
        sub = longs[longs["flu_group"] == fg]
        group_stats(sub, f"Fluency={fg}")

    # By regime
    print("\n  === BY REGIME (longs only) ===")
    for rg in ["full_regime", "partial_regime"]:
        sub = longs[longs["regime_group"] == rg]
        group_stats(sub, f"Regime={rg}")

    # Cross: grade + fluency
    print("\n  === CROSS: GRADE x FLUENCY (longs only) ===")
    for g in ["A+", "B+", "C"]:
        for fg in ["high_flu", "low_flu"]:
            sub = longs[(longs["grade"] == g) & (longs["flu_group"] == fg)]
            if len(sub) > 0:
                group_stats(sub, f"{g} + {fg}")

    # Cross: grade + hour
    print("\n  === CROSS: GRADE x HOUR (longs only) ===")
    for g in ["A+", "B+", "C"]:
        for hg in ["AM(10-12:30)", "PM(13-16)"]:
            sub = longs[(longs["grade"] == g) & (longs["hour_group"] == hg)]
            if len(sub) > 0:
                group_stats(sub, f"{g} + {hg}")

    # Cross: regime + fluency
    print("\n  === CROSS: REGIME x FLUENCY (longs only) ===")
    for rg in ["full_regime", "partial_regime"]:
        for fg in ["high_flu", "low_flu"]:
            sub = longs[(longs["regime_group"] == rg) & (longs["flu_group"] == fg)]
            if len(sub) > 0:
                group_stats(sub, f"{rg} + {fg}")

    # Summary: which groups have highest TP2+ rate (= benefit most from runners)?
    print("\n  === RUNNER BENEFIT SUMMARY ===")
    print("  Groups sorted by TP2+ rate (higher = more benefit from runner allocation):")
    groups_data = []
    for g in ["A+", "B+", "C"]:
        for rg in ["full_regime", "partial_regime"]:
            for fg in ["high_flu", "low_flu"]:
                sub = longs[(longs["grade"] == g) & (longs["regime_group"] == rg) & (longs["flu_group"] == fg)]
                if len(sub) >= 5:
                    n_s2 = (sub["trim_stage"] >= 2).sum()
                    pct_tp2 = n_s2 / len(sub) * 100
                    avg_r = sub["r"].mean()
                    s2_avg_r = sub.loc[sub["trim_stage"] >= 2, "r"].mean() if n_s2 > 0 else 0.0
                    groups_data.append({
                        "label": f"{g}/{rg[:4]}/{fg[:4]}",
                        "n": len(sub),
                        "tp2_pct": pct_tp2,
                        "avg_r": avg_r,
                        "s2_avg_r": s2_avg_r,
                    })
    groups_data.sort(key=lambda x: x["tp2_pct"], reverse=True)
    for gd in groups_data:
        marker = " *** HIGH RUNNER" if gd["tp2_pct"] > 35 else (" * low runner" if gd["tp2_pct"] < 15 else "")
        print(f"    {gd['label']:25s} | n={gd['n']:4d} | TP2%={gd['tp2_pct']:5.1f}% | avgR={gd['avg_r']:+.4f} | S2 avgR={gd['s2_avg_r']:+.3f}{marker}")


# ======================================================================
# MAIN
# ======================================================================
def main():
    print("=" * 120)
    print("A2D: DYNAMIC TRIM ALLOCATION EXPERIMENT")
    print("Base: Config E (sq_short=0.80, block_pm_shorts=True, ny_tp_mult=3.0, trail_nth_swing=3)")
    print("=" * 120)

    d = load_all()
    d_extra = prepare_liquidity_data(d)

    CONFIG_E_KWARGS = dict(
        sq_short=0.80, block_pm_shorts=True,
        ny_tp_mult=3.0, trail_nth_swing=3,
        be_after_tp1=False, be_after_tp2=True,
    )

    # ==================================================================
    # STEP 1: FIXED TRIM SPLIT SWEEP
    # ==================================================================
    print("\n" + "=" * 120)
    print("STEP 1: FIXED TRIM SPLIT SWEEP ON MULTI-TP ENGINE")
    print("=" * 120)

    splits = [
        ("10/10/80 (max runner)", 0.10, 0.10),
        ("15/15/70",             0.15, 0.15),
        ("20/20/60",             0.20, 0.20),
        ("25/25/50 (Config E)",  0.25, 0.25),
        ("30/30/40",             0.30, 0.30),
        ("40/40/20",             0.40, 0.40),
        ("50/50/0 (no runner)",  0.50, 0.50),
    ]

    print(f"\n  {'Split':30s} | {'Trades':>6} | {'R':>9} | {'PPDD':>7} | {'PF':>6} | {'WR':>6} | {'MaxDD':>6} | {'avgR':>8}")
    print(f"  {'-'*30}-+-{'-'*6}-+-{'-'*9}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")

    step1_results = {}
    for label, tp1_pct, tp2_pct in splits:
        t0 = _time.perf_counter()
        trades = run_backtest_multi_tp(
            d, d_extra,
            tp1_trim_pct=tp1_pct, tp2_trim_pct=tp2_pct,
            **CONFIG_E_KWARGS,
        )
        m = compute_metrics(trades)
        elapsed = _time.perf_counter() - t0
        print(f"  {label:30s} | {m['trades']:6d} | {m['R']:+9.1f} | {m['PPDD']:7.2f} | {m['PF']:6.2f} | {m['WR']:5.1f}% | {m['MaxDD']:6.1f} | {m['avgR']:+.4f}  ({elapsed:.1f}s)")
        step1_results[label] = (m, trades)

    # Identify best fixed split by PPDD
    best_label = max(step1_results, key=lambda k: step1_results[k][0]["PPDD"])
    best_m = step1_results[best_label][0]
    print(f"\n  BEST FIXED SPLIT: {best_label} (R={best_m['R']:+.1f}, PPDD={best_m['PPDD']:.2f}, PF={best_m['PF']:.2f})")

    # ==================================================================
    # STEP 2: ANALYZE SIGNAL GROUPS (using Config E 25/25/50 baseline)
    # ==================================================================
    print("\n" + "=" * 120)
    print("STEP 2: SIGNAL GROUP ANALYSIS — Which signals benefit from more runner?")
    print("Running enhanced backtest with feature recording...")
    print("=" * 120)

    t0 = _time.perf_counter()
    baseline_trades = run_backtest_dynamic_trim(
        d, d_extra,
        default_tp1_trim=0.25, default_tp2_trim=0.25,
        **CONFIG_E_KWARGS,
    )
    baseline_m = compute_metrics(baseline_trades)
    print_metrics("Config E baseline (enhanced engine)", baseline_m)
    print(f"  (computed in {_time.perf_counter() - t0:.1f}s)")

    analyze_signal_groups(baseline_trades)

    # ==================================================================
    # STEP 3: DESIGN CONVICTION-BASED RULES
    # ==================================================================
    print("\n" + "=" * 120)
    print("STEP 3: CONVICTION-BASED SPLIT RULES")
    print("=" * 120)

    # The rules are based on Step 2 analysis (designed from the data patterns):
    # High conviction: big runner allocation (runner benefits most for high-quality signals that reach TP2)
    # Low conviction: lock in profits quickly (these signals rarely reach TP2)

    # 3-Tier system
    trim_3tier = {
        "high": (0.10, 0.10),  # 10/10/80 -> big runner
        "mid":  (0.25, 0.25),  # 25/25/50 -> default
        "low":  (0.40, 0.40),  # 40/40/20 -> quick lock
    }

    # 3-Tier aggressive
    trim_3tier_agg = {
        "high": (0.05, 0.05),  # 5/5/90 -> maximum runner
        "mid":  (0.20, 0.20),  # 20/20/60
        "low":  (0.50, 0.50),  # 50/50/0 -> full lock
    }

    # 2-Tier system
    trim_2tier = {
        "high": (0.10, 0.10),  # 10/10/80
        "low":  (0.35, 0.35),  # 35/35/30
    }

    # Show conviction distribution
    print("\n  Conviction distribution (3-tier):")
    conv_counts = {"high": 0, "mid": 0, "low": 0}
    longs_df = pd.DataFrame([t for t in baseline_trades if t["dir"] == 1])
    if len(longs_df) > 0:
        for _, row in longs_df.iterrows():
            f = {"grade": row["grade"], "signal_type": row["type"],
                 "hour_et": row["hour_et"], "fluency": row["fluency"],
                 "regime": row["regime"], "direction": 1}
            c = conviction_3tier(f)
            conv_counts[c] += 1
        for k, v in conv_counts.items():
            pct = v / len(longs_df) * 100
            print(f"    {k:8s}: {v:4d} longs ({pct:5.1f}%)")

    conv2_counts = {"high": 0, "low": 0}
    if len(longs_df) > 0:
        for _, row in longs_df.iterrows():
            f = {"grade": row["grade"], "signal_type": row["type"],
                 "hour_et": row["hour_et"], "fluency": row["fluency"],
                 "regime": row["regime"], "direction": 1}
            c = conviction_2tier(f)
            conv2_counts[c] += 1
        print("\n  Conviction distribution (2-tier):")
        for k, v in conv2_counts.items():
            pct = v / len(longs_df) * 100
            print(f"    {k:8s}: {v:4d} longs ({pct:5.1f}%)")

    # ==================================================================
    # STEP 4: BACKTEST CONVICTION-BASED DYNAMIC TRIM
    # ==================================================================
    print("\n" + "=" * 120)
    print("STEP 4: BACKTEST — FIXED vs DYNAMIC TRIM SPLITS")
    print("=" * 120)

    configs = []

    # A) Fixed 25/25/50 (Config E baseline)
    configs.append(("A) Fixed 25/25/50 (Config E)", 0.25, 0.25, None, None))

    # B) Best fixed from Step 1
    best_tp1 = [s for s in splits if s[0] == best_label][0][1]
    best_tp2 = [s for s in splits if s[0] == best_label][0][2]
    configs.append((f"B) Best fixed: {best_label}", best_tp1, best_tp2, None, None))

    # C) Dynamic 3-tier
    configs.append(("C) Dynamic 3-tier (10/25/40)", 0.25, 0.25, trim_3tier, conviction_3tier))

    # D) Dynamic 3-tier aggressive
    configs.append(("D) Dynamic 3-tier aggressive", 0.25, 0.25, trim_3tier_agg, conviction_3tier))

    # E) Dynamic 2-tier
    configs.append(("E) Dynamic 2-tier (10/35)", 0.25, 0.25, trim_2tier, conviction_2tier))

    # F) Fixed 10/10/80 (from single sweep)
    configs.append(("F) Fixed 10/10/80", 0.10, 0.10, None, None))

    # G) Fixed 15/15/70
    configs.append(("G) Fixed 15/15/70", 0.15, 0.15, None, None))

    print(f"\n  {'Config':45s} | {'Trades':>6} | {'R':>9} | {'PPDD':>7} | {'PF':>6} | {'WR':>6} | {'MaxDD':>6} | {'avgR':>8}")
    print(f"  {'-'*45}-+-{'-'*6}-+-{'-'*9}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")

    step4_results = {}
    for label, tp1_def, tp2_def, t_lookup, c_fn in configs:
        t0 = _time.perf_counter()
        trades = run_backtest_dynamic_trim(
            d, d_extra,
            default_tp1_trim=tp1_def,
            default_tp2_trim=tp2_def,
            trim_lookup=t_lookup,
            conviction_fn=c_fn,
            **CONFIG_E_KWARGS,
        )
        m = compute_metrics(trades)
        elapsed = _time.perf_counter() - t0
        print(f"  {label:45s} | {m['trades']:6d} | {m['R']:+9.1f} | {m['PPDD']:7.2f} | {m['PF']:6.2f} | {m['WR']:5.1f}% | {m['MaxDD']:6.1f} | {m['avgR']:+.4f}  ({elapsed:.1f}s)")
        step4_results[label] = (m, trades)

    # Conviction distribution for dynamic configs
    for label, _, _, t_lookup, c_fn in configs:
        if t_lookup is not None and c_fn is not None:
            trades = step4_results[label][1]
            longs_t = [t for t in trades if t["dir"] == 1]
            conv_dist = {}
            for t in longs_t:
                c = t.get("conviction", "default")
                conv_dist[c] = conv_dist.get(c, 0) + 1
            r_by_conv = {}
            for t in longs_t:
                c = t.get("conviction", "default")
                if c not in r_by_conv:
                    r_by_conv[c] = []
                r_by_conv[c].append(t["r"])
            print(f"\n  {label} — Conviction breakdown (longs):")
            for c_level in sorted(conv_dist.keys()):
                n_c = conv_dist[c_level]
                r_vals = np.array(r_by_conv[c_level])
                trim_used = t_lookup.get(c_level, (0.25, 0.25))
                print(f"    {c_level:8s}: {n_c:4d} trades, R={r_vals.sum():+7.1f}, avgR={r_vals.mean():+.4f}, "
                      f"WR={100*(r_vals>0).mean():.1f}%, trim={trim_used[0]:.0%}/{trim_used[1]:.0%}/{1-trim_used[0]-trim_used[1]:.0%}")

    # ==================================================================
    # STEP 5: IS DYNAMIC BETTER THAN BEST FIXED?
    # ==================================================================
    print("\n" + "=" * 120)
    print("STEP 5: WALK-FORWARD VALIDATION — DYNAMIC vs BEST FIXED vs CONFIG E")
    print("=" * 120)

    # Pick the best dynamic config by PPDD
    dynamic_configs = [(l, m, t) for l, (m, t) in step4_results.items()
                       if "Dynamic" in l or "dynamic" in l]
    if dynamic_configs:
        best_dyn = max(dynamic_configs, key=lambda x: x[1]["PPDD"])
        best_dyn_label = best_dyn[0]
    else:
        best_dyn_label = None

    # Walk-forward comparison
    configs_for_wf = [
        ("A) Config E (25/25/50)", step4_results[[k for k in step4_results if k.startswith("A)")][0]][1]),
    ]

    # Add best fixed from step 1 if different from Config E
    best_fixed_key = [k for k in step4_results if k.startswith("B)")][0]
    configs_for_wf.append(("B) Best Fixed", step4_results[best_fixed_key][1]))

    # Add best dynamic
    if best_dyn_label:
        configs_for_wf.append(("Best Dynamic", step4_results[best_dyn_label][1]))

    # Add all dynamic variants
    for label in step4_results:
        if "Dynamic" in label and label != best_dyn_label:
            configs_for_wf.append((label[:30], step4_results[label][1]))

    # Print walk-forward table
    all_wf = []
    for label, trades in configs_for_wf:
        wf = walk_forward_metrics(trades)
        all_wf.append((label, wf))

    if all_wf:
        # Header
        hdr = f"  {'Year':>6}"
        for label, _ in all_wf:
            hdr += f" | {label[:25]:^32s}"
        print(hdr)

        sub = f"  {'':>6}"
        for _ in all_wf:
            sub += f" | {'n':>4} {'R':>7} {'PF':>5} {'PPDD':>6}"
        print(sub)
        print("  " + "-" * (8 + 34 * len(all_wf)))

        # Build dicts
        wf_dicts = [{w["year"]: w for w in wf} for _, wf in all_wf]
        all_years = sorted(set(y for wd in wf_dicts for y in wd.keys()))

        # Track wins vs Config E (index 0)
        wins_vs_baseline = [0] * len(all_wf)

        for y in all_years:
            parts = [f"  {y:>4}"]
            bl_r = wf_dicts[0].get(y, {"R": 0})["R"]
            for vi, wd in enumerate(wf_dicts):
                yv = wd.get(y, {"n": 0, "R": 0, "WR": 0, "PF": 0, "PPDD": 0})
                marker = " *" if vi > 0 and yv["R"] > bl_r else ""
                if vi > 0 and yv["R"] > bl_r:
                    wins_vs_baseline[vi] += 1
                parts.append(f"{yv['n']:4d} {yv['R']:+7.1f} {yv['PF']:5.2f} {yv['PPDD']:+6.2f}{marker}")
            print(" | ".join(parts))

        print("  " + "-" * (8 + 34 * len(all_wf)))
        for vi, (label, _) in enumerate(all_wf):
            if vi > 0:
                print(f"  {label}: wins {wins_vs_baseline[vi]}/{len(all_years)} years vs Config E")

    # ==================================================================
    # FINAL VERDICT
    # ==================================================================
    print("\n" + "=" * 120)
    print("FINAL VERDICT")
    print("=" * 120)

    # Compare best dynamic vs best fixed
    cfg_e_m = step4_results[[k for k in step4_results if k.startswith("A)")][0]][0]
    best_fixed_m = step4_results[best_fixed_key][0]

    print(f"\n  Config E (25/25/50):       R={cfg_e_m['R']:+8.1f}  PPDD={cfg_e_m['PPDD']:6.2f}  PF={cfg_e_m['PF']:5.2f}  MaxDD={cfg_e_m['MaxDD']:.1f}")
    print(f"  Best Fixed ({best_label}): R={best_fixed_m['R']:+8.1f}  PPDD={best_fixed_m['PPDD']:6.2f}  PF={best_fixed_m['PF']:5.2f}  MaxDD={best_fixed_m['MaxDD']:.1f}")

    if best_dyn_label:
        best_dyn_m = step4_results[best_dyn_label][0]
        print(f"  Best Dynamic ({best_dyn_label[:30]}): R={best_dyn_m['R']:+8.1f}  PPDD={best_dyn_m['PPDD']:6.2f}  PF={best_dyn_m['PF']:5.2f}  MaxDD={best_dyn_m['MaxDD']:.1f}")

        delta_r_vs_fixed = best_dyn_m["R"] - best_fixed_m["R"]
        delta_ppdd_vs_fixed = best_dyn_m["PPDD"] - best_fixed_m["PPDD"]
        delta_r_vs_e = best_dyn_m["R"] - cfg_e_m["R"]
        delta_ppdd_vs_e = best_dyn_m["PPDD"] - cfg_e_m["PPDD"]

        print(f"\n  Dynamic vs Best Fixed:  dR={delta_r_vs_fixed:+.1f}, dPPDD={delta_ppdd_vs_fixed:+.2f}")
        print(f"  Dynamic vs Config E:    dR={delta_r_vs_e:+.1f}, dPPDD={delta_ppdd_vs_e:+.2f}")

        if delta_ppdd_vs_fixed > 0.5:
            print("\n  CONCLUSION: Dynamic trim BEATS best fixed split. Worth the complexity.")
        elif delta_ppdd_vs_fixed > -0.5:
            print("\n  CONCLUSION: Dynamic trim is MARGINAL vs best fixed. Added complexity not justified.")
        else:
            print("\n  CONCLUSION: Dynamic trim LOSES to best fixed. Use the simpler fixed split.")

        # Check if best fixed beats Config E
        delta_fixed_vs_e = best_fixed_m["PPDD"] - cfg_e_m["PPDD"]
        if delta_fixed_vs_e > 0.5:
            print(f"  NOTE: Best fixed ({best_label}) improves over Config E by dPPDD={delta_fixed_vs_e:+.2f}. Consider adopting it.")
    else:
        print("\n  No dynamic configs to compare.")

    print("\n" + "=" * 120)
    print("DONE")
    print("=" * 120)


if __name__ == "__main__":
    main()
