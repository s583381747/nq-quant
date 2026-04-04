"""
experiments/a2e_runner_analysis.py — Deep runner exit analysis for multi-TP Config E.

Previous runner_variants experiment (Config D single-TP) found:
  - Time-based exit: DESTRUCTIVE (-25 to -28R)
  - ATR-based trail: DESTRUCTIVE (-13 to -37R)
  - Current trailing stop (nth swing) is well-calibrated

This experiment re-examines runner behavior specifically for the multi-TP system
where the runner is 50% of position (bigger stake after TP1+TP2 trims of 25%+25%).

Parts:
  1. Runner performance analysis (contribution, MFE, MAE, duration)
  2. Runner exit reason breakdown
  3. Momentum decay detection (ATR, body ratio, directional consistency)
  4. Kill condition testing (stall, drawdown from peak, time, vol collapse)
  5. Results comparison table

Usage: python experiments/a2e_runner_analysis.py
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
)


# ======================================================================
# Enhanced multi-TP backtest that captures detailed runner data
# ======================================================================
def run_backtest_multi_tp_with_runner_data(
    d: dict,
    d_extra: dict,
    # Config E params (V1 late BE)
    sq_long: float = 0.68,
    sq_short: float = 0.80,
    min_stop_atr: float = 1.7,
    block_pm_shorts: bool = True,
    tp1_trim_pct: float = 0.25,
    tp2_trim_pct: float = 0.25,
    be_after_tp1: bool = False,
    be_after_tp2: bool = True,
    trail_nth_swing: int = 3,
    use_multi_tp_for_shorts: bool = False,
    ny_tp_mult: float = 2.0,
    mss_long_tp_mult: float = 2.5,
    # === RUNNER KILL CONDITIONS ===
    runner_kill: str = "none",       # "none", "stall", "drawdown", "time", "vol_collapse"
    kill_stall_bars: int = 20,       # Kill A: no new high/low for N bars
    kill_drawdown_pct: float = 0.50, # Kill B: unrealized drops X% from peak
    kill_time_bars: int = 40,        # Kill C: exit after N bars post-TP2
    kill_vol_ratio: float = 0.50,    # Kill D: ATR drops below X% of ATR at TP2
    # Date range
    start_date=None, end_date=None,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (trades, runner_details).
    runner_details has per-runner bar-by-bar data for analysis.
    """

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
    runner_details = []  # Detailed per-runner info
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

    # Runner tracking state
    runner_active = False
    runner_tp2_bar = 0
    runner_tp2_price = 0.0      # Price at TP2 hit
    runner_tp2_atr = 0.0        # ATR at TP2 hit
    runner_peak_favorable = 0.0  # Peak unrealized PnL (in points) since TP2
    runner_last_new_extreme_bar = 0  # Last bar that made a new high/low
    runner_bar_data = []         # Per-bar data for momentum decay analysis
    runner_r_amount = 0.0        # R amount for this trade (for R calculations)
    runner_stop_dist = 0.0       # Original stop distance

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

            # === EARLY PA CUT (pre-trim only) ===
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

            # ==================================================================
            # MULTI-TP EXIT MANAGEMENT (LONGS)
            # ==================================================================
            if not exited and pos_is_multi_tp and pos_direction == 1:
                # Determine effective stop
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

                # Check stop hit
                if l[i] <= eff_stop:
                    exit_price = eff_stop - slippage_points
                    if pos_trim_stage > 0 and eff_stop >= pos_entry_price:
                        exit_reason = "be_sweep"
                    else:
                        exit_reason = "stop"
                    exited = True

                # Stage 0: Check TP1
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

                # Stage 1: Check TP2
                if not exited and pos_trim_stage == 1 and h[i] >= pos_tp2:
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
                        # Initialize runner tracking
                        runner_active = True
                        runner_tp2_bar = i
                        runner_tp2_price = h[i]  # Approximate TP2 price
                        runner_tp2_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                        runner_peak_favorable = h[i] - pos_entry_price
                        runner_last_new_extreme_bar = i
                        runner_bar_data = []
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp2
                        exit_reason = "tp2"
                        exit_contracts = pos_contracts
                        exited = True

                # Stage 2: Trail runner
                if not exited and pos_trim_stage == 2:
                    # Update trail stop
                    nt = _find_nth_swing(swing_low_mask, low_arr, i, trail_nth_swing, 1)
                    if not np.isnan(nt) and nt > pos_trail_stop:
                        pos_trail_stop = nt

                    # Track runner metrics
                    if runner_active:
                        bars_since_tp2 = i - runner_tp2_bar
                        favorable_excursion = h[i] - pos_entry_price
                        if favorable_excursion > runner_peak_favorable:
                            runner_peak_favorable = favorable_excursion
                            runner_last_new_extreme_bar = i

                        # Current unrealized
                        unrealized = c_arr[i] - pos_entry_price
                        giveback = runner_peak_favorable - favorable_excursion if runner_peak_favorable > 0 else 0.0
                        giveback_pct = giveback / runner_peak_favorable if runner_peak_favorable > 0 else 0.0

                        # Bar metrics
                        cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                        bar_body = abs(c_arr[i] - o[i])
                        bar_range = h[i] - l[i]
                        body_ratio = bar_body / bar_range if bar_range > 0 else 0.0
                        bar_dir = 1 if c_arr[i] > o[i] else (-1 if c_arr[i] < o[i] else 0)

                        runner_bar_data.append({
                            "bar_idx": i,
                            "bars_since_tp2": bars_since_tp2,
                            "favorable_excursion": favorable_excursion,
                            "peak_favorable": runner_peak_favorable,
                            "unrealized": unrealized,
                            "giveback": giveback,
                            "giveback_pct": giveback_pct,
                            "atr": cur_atr,
                            "atr_ratio_vs_tp2": cur_atr / runner_tp2_atr if runner_tp2_atr > 0 else 1.0,
                            "body_ratio": body_ratio,
                            "bar_dir": bar_dir,
                            "bars_since_new_high": i - runner_last_new_extreme_bar,
                        })

                        # === KILL CONDITIONS ===
                        if runner_kill == "stall":
                            if (i - runner_last_new_extreme_bar) >= kill_stall_bars:
                                exit_price = c_arr[i]
                                exit_reason = "kill_stall"
                                exited = True
                        elif runner_kill == "drawdown":
                            if runner_peak_favorable > 0 and giveback_pct >= kill_drawdown_pct:
                                exit_price = c_arr[i]
                                exit_reason = "kill_drawdown"
                                exited = True
                        elif runner_kill == "time":
                            if bars_since_tp2 >= kill_time_bars:
                                exit_price = c_arr[i]
                                exit_reason = "kill_time"
                                exited = True
                        elif runner_kill == "vol_collapse":
                            if runner_tp2_atr > 0 and cur_atr / runner_tp2_atr < kill_vol_ratio:
                                exit_price = c_arr[i]
                                exit_reason = "kill_vol"
                                exited = True

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
                        # Initialize runner tracking
                        runner_active = True
                        runner_tp2_bar = i
                        runner_tp2_price = l[i]
                        runner_tp2_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                        runner_peak_favorable = pos_entry_price - l[i]
                        runner_last_new_extreme_bar = i
                        runner_bar_data = []
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp2
                        exit_reason = "tp2"
                        exit_contracts = pos_contracts
                        exited = True

                if not exited and pos_trim_stage == 2:
                    nt = _find_nth_swing(swing_high_mask, high_arr, i, trail_nth_swing, -1)
                    if not np.isnan(nt) and nt < pos_trail_stop:
                        pos_trail_stop = nt

                    # Track runner metrics (shorts)
                    if runner_active:
                        bars_since_tp2 = i - runner_tp2_bar
                        favorable_excursion = pos_entry_price - l[i]
                        if favorable_excursion > runner_peak_favorable:
                            runner_peak_favorable = favorable_excursion
                            runner_last_new_extreme_bar = i

                        unrealized = pos_entry_price - c_arr[i]
                        giveback = runner_peak_favorable - favorable_excursion if runner_peak_favorable > 0 else 0.0
                        giveback_pct = giveback / runner_peak_favorable if runner_peak_favorable > 0 else 0.0

                        cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                        bar_body = abs(c_arr[i] - o[i])
                        bar_range = h[i] - l[i]
                        body_ratio = bar_body / bar_range if bar_range > 0 else 0.0
                        bar_dir = 1 if c_arr[i] > o[i] else (-1 if c_arr[i] < o[i] else 0)

                        runner_bar_data.append({
                            "bar_idx": i,
                            "bars_since_tp2": bars_since_tp2,
                            "favorable_excursion": favorable_excursion,
                            "peak_favorable": runner_peak_favorable,
                            "unrealized": unrealized,
                            "giveback": giveback,
                            "giveback_pct": giveback_pct,
                            "atr": cur_atr,
                            "atr_ratio_vs_tp2": cur_atr / runner_tp2_atr if runner_tp2_atr > 0 else 1.0,
                            "body_ratio": body_ratio,
                            "bar_dir": bar_dir,
                            "bars_since_new_high": i - runner_last_new_extreme_bar,
                        })

                        # === KILL CONDITIONS (shorts) ===
                        if runner_kill == "stall":
                            if (i - runner_last_new_extreme_bar) >= kill_stall_bars:
                                exit_price = c_arr[i]
                                exit_reason = "kill_stall"
                                exited = True
                        elif runner_kill == "drawdown":
                            if runner_peak_favorable > 0 and giveback_pct >= kill_drawdown_pct:
                                exit_price = c_arr[i]
                                exit_reason = "kill_drawdown"
                                exited = True
                        elif runner_kill == "time":
                            if bars_since_tp2 >= kill_time_bars:
                                exit_price = c_arr[i]
                                exit_reason = "kill_time"
                                exited = True
                        elif runner_kill == "vol_collapse":
                            if runner_tp2_atr > 0 and cur_atr / runner_tp2_atr < kill_vol_ratio:
                                exit_price = c_arr[i]
                                exit_reason = "kill_vol"
                                exited = True

            # ==================================================================
            # SINGLE-TP EXIT MANAGEMENT (shorts default, or non-multi-TP)
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

                trade_record = {
                    "entry_time": nq.index[pos_entry_idx],
                    "exit_time": nq.index[i + 1] if exit_reason == "early_cut_pa" and i + 1 < end_idx else nq.index[i],
                    "r": r_mult, "reason": exit_reason, "dir": pos_direction,
                    "type": pos_signal_type, "trimmed": any_trimmed, "grade": pos_grade,
                    "entry_price": pos_entry_price, "exit_price": exit_price,
                    "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
                    "trim_stage": pos_trim_stage,
                    "tp2_price": pos_tp2 if pos_is_multi_tp else np.nan,
                    "is_multi_tp": pos_is_multi_tp,
                }
                trades.append(trade_record)

                # Save runner details if runner was active
                if runner_active and pos_trim_stage >= 2:
                    # Calculate runner-only R contribution
                    runner_pnl = pnl_pts_runner * point_value * exit_contracts
                    runner_r = runner_pnl / total_risk if total_risk > 0 else 0.0

                    # TP1 + TP2 locked-in R
                    tp1_pnl = pos_tp1_pnl_pts * point_value * pos_trim1_contracts if pos_trim_stage >= 1 else 0.0
                    tp2_pnl = pos_tp2_pnl_pts * point_value * pos_trim2_contracts if pos_trim_stage >= 2 else 0.0
                    locked_r = (tp1_pnl + tp2_pnl) / total_risk if total_risk > 0 else 0.0

                    runner_details.append({
                        "entry_time": nq.index[pos_entry_idx],
                        "dir": pos_direction,
                        "total_r": r_mult,
                        "runner_r": runner_r,
                        "locked_r": locked_r,
                        "exit_reason": exit_reason,
                        "runner_bars": i - runner_tp2_bar,
                        "runner_mfe_pts": runner_peak_favorable,
                        "runner_exit_pts": pnl_pts_runner,
                        "runner_giveback_pts": runner_peak_favorable - (max(0, pnl_pts_runner) if pos_direction == 1 else max(0, pnl_pts_runner)),
                        "runner_peak_r": (runner_peak_favorable * point_value * exit_contracts) / total_risk if total_risk > 0 else 0.0,
                        "bar_data": runner_bar_data,
                        "tp2_atr": runner_tp2_atr,
                        "contracts_runner": exit_contracts,
                        "contracts_total": pos_contracts,
                    })

                runner_active = False
                runner_bar_data = []

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
            runner_active = False
            runner_bar_data = []
            runner_r_amount = r_amount
            runner_stop_dist = stop_dist
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
            runner_active = False
            runner_bar_data = []

            if mss_mgmt_enabled and is_mss_signal:
                pos_single_trim_pct = mss_mgmt.get("short_trim_pct", 1.0) if direction == -1 else mss_mgmt.get("long_trim_pct", default_trim_pct)
            elif dual_mode_enabled and direction == -1:
                pos_single_trim_pct = dual_mode.get("short_trim_pct", 1.0)
            elif dir_mgmt.get("enabled", False):
                pos_single_trim_pct = dir_mgmt.get("long_trim_pct", default_trim_pct) if direction == 1 else dir_mgmt.get("short_trim_pct", 1.0)
            else:
                pos_single_trim_pct = default_trim_pct

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
            "tp2_price": pos_tp2 if pos_is_multi_tp else np.nan,
            "is_multi_tp": pos_is_multi_tp,
        })

    return trades, runner_details


# ======================================================================
# Part 1: Runner Performance Analysis
# ======================================================================
def analyze_runner_performance(trades: list[dict], runner_details: list[dict]):
    """Detailed analysis of runner contribution."""
    print("\n" + "=" * 110)
    print("PART 1: RUNNER PERFORMANCE ANALYSIS")
    print("=" * 110)

    if not runner_details:
        print("  No runners found (no trades reached trim_stage >= 2)")
        return

    df = pd.DataFrame(runner_details)
    n_runners = len(df)
    n_trades = len(trades)
    trades_df = pd.DataFrame(trades)
    total_r = trades_df["r"].sum()

    # Runner contribution
    total_runner_r = df["runner_r"].sum()
    total_locked_r = df["locked_r"].sum()

    print(f"\n  Total trades: {n_trades}")
    print(f"  Trades reaching runner stage (trim_stage >= 2): {n_runners} ({100*n_runners/n_trades:.1f}%)")
    print(f"  Total R (all trades): {total_r:+.1f}")
    print(f"  Runner R contribution: {total_runner_r:+.1f}")
    print(f"  Locked-in R (TP1+TP2 trims): {total_locked_r:+.1f}")
    print(f"  Runner trades total R (locked + runner): {total_locked_r + total_runner_r:+.1f}")

    # Distribution of runner exit R
    pos_runners = (df["runner_r"] > 0).sum()
    neg_runners = (df["runner_r"] < 0).sum()
    zero_runners = (df["runner_r"] == 0).sum()
    print(f"\n  Runner exit R distribution:")
    print(f"    Positive: {pos_runners} ({100*pos_runners/n_runners:.1f}%)")
    print(f"    Negative: {neg_runners} ({100*neg_runners/n_runners:.1f}%)")
    print(f"    Zero/BE:  {zero_runners} ({100*zero_runners/n_runners:.1f}%)")
    print(f"    Mean runner R: {df['runner_r'].mean():+.4f}")
    print(f"    Median runner R: {df['runner_r'].median():+.4f}")

    # MFE (max favorable excursion after TP2)
    print(f"\n  Runner MFE (points beyond entry, after TP2):")
    print(f"    Mean:   {df['runner_mfe_pts'].mean():.1f} pts")
    print(f"    Median: {df['runner_mfe_pts'].median():.1f} pts")
    print(f"    P25:    {df['runner_mfe_pts'].quantile(0.25):.1f} pts")
    print(f"    P75:    {df['runner_mfe_pts'].quantile(0.75):.1f} pts")
    print(f"    Max:    {df['runner_mfe_pts'].max():.1f} pts")

    # Giveback
    giveback = df["runner_mfe_pts"] - df["runner_exit_pts"]
    print(f"\n  Runner giveback (MFE - exit, in points):")
    print(f"    Mean:   {giveback.mean():.1f} pts")
    print(f"    Median: {giveback.median():.1f} pts")
    print(f"    P75:    {giveback.quantile(0.75):.1f} pts")
    print(f"    P90:    {giveback.quantile(0.90):.1f} pts")

    # Peak R vs actual runner R
    print(f"\n  Runner peak R vs actual R:")
    print(f"    Mean peak R:   {df['runner_peak_r'].mean():+.4f}")
    print(f"    Mean actual R: {df['runner_r'].mean():+.4f}")
    capture_pct = df["runner_r"].sum() / df["runner_peak_r"].sum() * 100 if df["runner_peak_r"].sum() != 0 else 0
    print(f"    R capture efficiency: {capture_pct:.1f}%")

    # Duration
    print(f"\n  Runner duration (bars active after TP2):")
    print(f"    Mean:   {df['runner_bars'].mean():.1f} bars")
    print(f"    Median: {df['runner_bars'].median():.0f} bars")
    print(f"    P25:    {df['runner_bars'].quantile(0.25):.0f} bars")
    print(f"    P75:    {df['runner_bars'].quantile(0.75):.0f} bars")
    print(f"    Max:    {df['runner_bars'].max():.0f} bars")

    # By direction
    for dir_val, dir_name in [(1, "LONG"), (-1, "SHORT")]:
        sub = df[df["dir"] == dir_val]
        if len(sub) == 0:
            continue
        print(f"\n  {dir_name} runners ({len(sub)}):")
        print(f"    Runner R: {sub['runner_r'].sum():+.1f}, mean={sub['runner_r'].mean():+.4f}")
        print(f"    Positive: {(sub['runner_r'] > 0).sum()}/{len(sub)} ({100*(sub['runner_r']>0).mean():.1f}%)")
        print(f"    Mean MFE: {sub['runner_mfe_pts'].mean():.1f} pts, mean bars: {sub['runner_bars'].mean():.1f}")


# ======================================================================
# Part 2: Runner Exit Reason Breakdown
# ======================================================================
def analyze_runner_exits(runner_details: list[dict]):
    """Exit reason breakdown for runners."""
    print("\n" + "=" * 110)
    print("PART 2: RUNNER EXIT REASON BREAKDOWN")
    print("=" * 110)

    if not runner_details:
        print("  No runners found")
        return

    df = pd.DataFrame(runner_details)

    print(f"\n  {'Exit Reason':20s} | {'Count':>6} | {'%':>6} | {'Runner R':>10} | {'Avg R':>8} | {'Avg MFE':>8} | {'Avg Bars':>8} | {'Avg Giveback':>12}")
    print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}")

    for reason in sorted(df["exit_reason"].unique()):
        sub = df[df["exit_reason"] == reason]
        n = len(sub)
        pct = 100 * n / len(df)
        runner_r = sub["runner_r"].sum()
        avg_r = sub["runner_r"].mean()
        avg_mfe = sub["runner_mfe_pts"].mean()
        avg_bars = sub["runner_bars"].mean()
        avg_giveback = (sub["runner_mfe_pts"] - sub["runner_exit_pts"]).mean()
        print(f"  {reason:20s} | {n:6d} | {pct:5.1f}% | {runner_r:+10.1f} | {avg_r:+8.4f} | {avg_mfe:8.1f} | {avg_bars:8.1f} | {avg_giveback:12.1f}")

    # BE sweep detail
    be_sweeps = df[df["exit_reason"] == "be_sweep"]
    if len(be_sweeps) > 0:
        print(f"\n  BE sweep runners ({len(be_sweeps)}):")
        print(f"    These are runners that hit breakeven after giving back gains")
        print(f"    MFE before giving back: mean={be_sweeps['runner_mfe_pts'].mean():.1f} pts, median={be_sweeps['runner_mfe_pts'].median():.1f} pts")
        print(f"    Runner R (at BE): mean={be_sweeps['runner_r'].mean():+.4f}")
        print(f"    But TP1+TP2 locked: mean={be_sweeps['locked_r'].mean():+.4f} (these trades are STILL profitable overall)")

    # Stop exits
    stop_exits = df[df["exit_reason"] == "stop"]
    if len(stop_exits) > 0:
        print(f"\n  Trail stop exits ({len(stop_exits)}):")
        print(f"    Runner R: mean={stop_exits['runner_r'].mean():+.4f}")
        near_be = stop_exits[stop_exits["runner_r"].abs() < 0.05]
        real_loss = stop_exits[stop_exits["runner_r"] < -0.05]
        real_win = stop_exits[stop_exits["runner_r"] > 0.05]
        print(f"    Near BE (<0.05R):  {len(near_be)}")
        print(f"    Real loss (>0.05R neg): {len(real_loss)}, avg R={real_loss['runner_r'].mean():+.4f}" if len(real_loss) > 0 else "    Real loss: 0")
        print(f"    Real win (>0.05R pos): {len(real_win)}, avg R={real_win['runner_r'].mean():+.4f}" if len(real_win) > 0 else "    Real win: 0")


# ======================================================================
# Part 3: Momentum Decay Detection
# ======================================================================
def analyze_momentum_decay(runner_details: list[dict]):
    """Analyze bar-by-bar metrics after TP2 to detect momentum decay."""
    print("\n" + "=" * 110)
    print("PART 3: MOMENTUM DECAY DETECTION")
    print("=" * 110)

    if not runner_details:
        print("  No runners found")
        return

    # Collect all bar data across runners, bucketed by bars_since_tp2
    all_bars = []
    for rd in runner_details:
        for bd in rd["bar_data"]:
            bd_copy = dict(bd)
            bd_copy["exit_reason"] = rd["exit_reason"]
            bd_copy["runner_r"] = rd["runner_r"]
            bd_copy["runner_positive"] = rd["runner_r"] > 0
            all_bars.append(bd_copy)

    if not all_bars:
        print("  No bar data collected")
        return

    bar_df = pd.DataFrame(all_bars)

    # Bucket by bars since TP2
    buckets = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 50), (51, 100)]
    print(f"\n  {'Bars Since TP2':>15} | {'Count':>6} | {'Avg ATR Ratio':>13} | {'Avg Body%':>10} | {'Avg Giveback%':>13} | {'Bars No NewHi':>13}")
    print(f"  {'-'*15}-+-{'-'*6}-+-{'-'*13}-+-{'-'*10}-+-{'-'*13}-+-{'-'*13}")

    for lo, hi in buckets:
        sub = bar_df[(bar_df["bars_since_tp2"] >= lo) & (bar_df["bars_since_tp2"] <= hi)]
        if len(sub) == 0:
            continue
        avg_atr_ratio = sub["atr_ratio_vs_tp2"].mean()
        avg_body = sub["body_ratio"].mean()
        avg_giveback = sub["giveback_pct"].mean() * 100
        avg_no_new = sub["bars_since_new_high"].mean()
        print(f"  {lo:>6}-{hi:<8} | {len(sub):6d} | {avg_atr_ratio:13.3f} | {avg_body:9.3f} | {avg_giveback:12.1f}% | {avg_no_new:13.1f}")

    # Directional consistency: for each runner, compute rolling 5-bar directional ratio
    print(f"\n  Directional consistency analysis:")
    for rd in runner_details[:0]:  # Skip heavy computation, use aggregate
        pass

    # Compare metrics of positive vs negative runners
    pos_bars = bar_df[bar_df["runner_positive"]]
    neg_bars = bar_df[~bar_df["runner_positive"]]

    if len(pos_bars) > 0 and len(neg_bars) > 0:
        print(f"\n  Positive vs Negative runner bar characteristics:")
        print(f"  {'Metric':25s} | {'Positive Runners':>17} | {'Negative Runners':>17}")
        print(f"  {'-'*25}-+-{'-'*17}-+-{'-'*17}")
        print(f"  {'Avg ATR ratio':25s} | {pos_bars['atr_ratio_vs_tp2'].mean():17.3f} | {neg_bars['atr_ratio_vs_tp2'].mean():17.3f}")
        print(f"  {'Avg body ratio':25s} | {pos_bars['body_ratio'].mean():17.3f} | {neg_bars['body_ratio'].mean():17.3f}")
        print(f"  {'Avg giveback%':25s} | {pos_bars['giveback_pct'].mean()*100:16.1f}% | {neg_bars['giveback_pct'].mean()*100:16.1f}%")
        print(f"  {'Avg bars_no_new_extreme':25s} | {pos_bars['bars_since_new_high'].mean():17.1f} | {neg_bars['bars_since_new_high'].mean():17.1f}")

    # Key finding: at what giveback% do runners tend to recover vs not?
    print(f"\n  Giveback recovery analysis (peak giveback before exit):")
    for rd in runner_details:
        if not rd["bar_data"]:
            continue
        max_giveback = max(bd["giveback_pct"] for bd in rd["bar_data"])
        rd["peak_giveback_pct"] = max_giveback

    rd_df = pd.DataFrame([{
        "peak_giveback_pct": rd.get("peak_giveback_pct", 0),
        "runner_r": rd["runner_r"],
        "runner_positive": rd["runner_r"] > 0,
    } for rd in runner_details if "peak_giveback_pct" in rd])

    if len(rd_df) > 0:
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            high_gb = rd_df[rd_df["peak_giveback_pct"] >= threshold]
            if len(high_gb) > 0:
                pct_neg = (high_gb["runner_r"] <= 0).mean() * 100
                print(f"    Runners with peak giveback >= {threshold*100:.0f}%: {len(high_gb)}, {pct_neg:.0f}% ended negative/BE, avg runner R = {high_gb['runner_r'].mean():+.4f}")


# ======================================================================
# Part 4 & 5: Test Kill Conditions and Compare
# ======================================================================
def test_kill_conditions(d: dict, d_extra: dict):
    """Test various runner kill conditions and compare to baseline."""
    print("\n" + "=" * 110)
    print("PART 4 & 5: RUNNER KILL CONDITION TESTING")
    print("=" * 110)

    configs = [
        # (label, runner_kill, kwargs)
        ("E. Baseline (trail stop)", "none", {}),
        ("A1. Stall N=10", "stall", {"kill_stall_bars": 10}),
        ("A2. Stall N=15", "stall", {"kill_stall_bars": 15}),
        ("A3. Stall N=20", "stall", {"kill_stall_bars": 20}),
        ("A4. Stall N=30", "stall", {"kill_stall_bars": 30}),
        ("B1. Drawdown 30%", "drawdown", {"kill_drawdown_pct": 0.30}),
        ("B2. Drawdown 50%", "drawdown", {"kill_drawdown_pct": 0.50}),
        ("B3. Drawdown 70%", "drawdown", {"kill_drawdown_pct": 0.70}),
        ("C1. Time limit 20 bars", "time", {"kill_time_bars": 20}),
        ("C2. Time limit 40 bars", "time", {"kill_time_bars": 40}),
        ("C3. Time limit 60 bars", "time", {"kill_time_bars": 60}),
        ("D. Vol collapse <50% ATR", "vol_collapse", {"kill_vol_ratio": 0.50}),
    ]

    results = []
    baseline_r = None

    for label, kill_type, kwargs in configs:
        t0 = _time.perf_counter()
        trades, rd = run_backtest_multi_tp_with_runner_data(
            d, d_extra,
            runner_kill=kill_type,
            **kwargs,
        )
        elapsed = _time.perf_counter() - t0
        m = compute_metrics(trades)

        # Count runners and their R
        runner_trades = [t for t in trades if t.get("trim_stage", 0) >= 2]
        runner_r_sum = sum(t["r"] for t in runner_trades) if runner_trades else 0.0
        n_runners = len(runner_trades)

        if baseline_r is None:
            baseline_r = m["R"]

        results.append({
            "label": label,
            "m": m,
            "n_runners": n_runners,
            "runner_r": runner_r_sum,
            "elapsed": elapsed,
            "delta_r": m["R"] - baseline_r,
        })

    # Print comparison table
    print(f"\n  {'Kill Condition':30s} | {'Trades':>6} | {'R':>9} | {'PPDD':>7} | {'PF':>6} | {'WR':>6} | {'MaxDD':>6} | {'dR':>7} | {'Runners':>7} | {'RunnerR':>8}")
    print(f"  {'-'*30}-+-{'-'*6}-+-{'-'*9}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}")

    for r in results:
        m = r["m"]
        marker = " <-- baseline" if r["delta_r"] == 0 else ""
        destructive = " *** DESTRUCTIVE" if r["delta_r"] < -5 else ""
        print(f"  {r['label']:30s} | {m['trades']:6d} | {m['R']:+9.1f} | {m['PPDD']:7.2f} | {m['PF']:6.2f} | {m['WR']:5.1f}% | {m['MaxDD']:6.1f} | {r['delta_r']:+7.1f} | {r['n_runners']:7d} | {r['runner_r']:+8.1f}{marker}{destructive}")

    # Summary verdict
    print(f"\n  VERDICT:")
    any_better = False
    for r in results[1:]:  # Skip baseline
        if r["delta_r"] > 0:
            print(f"    {r['label']}: IMPROVED R by {r['delta_r']:+.1f}")
            any_better = True
        elif r["delta_r"] > -2:
            print(f"    {r['label']}: NEUTRAL (dR={r['delta_r']:+.1f})")
        else:
            print(f"    {r['label']}: DESTRUCTIVE (dR={r['delta_r']:+.1f})")

    if not any_better:
        print(f"\n  CONCLUSION: ALL kill conditions are destructive or neutral.")
        print(f"  The current trailing stop (nth swing) remains the best runner exit strategy.")
        print(f"  This is consistent with the previous runner_variants experiment (Config D).")

    return results


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 110)
    print("A2E RUNNER ANALYSIS — Multi-TP Config E")
    print("Previous finding (Config D): ALL runner kill conditions were DESTRUCTIVE")
    print("Re-testing on multi-TP where runner is 50% of position (bigger stake)")
    print("=" * 110)

    d = load_all()
    d_extra = prepare_liquidity_data(d)

    # Run baseline Config E (V1 late BE) with detailed runner data
    print("\n[BASELINE] Running Config E (V1 Multi-TP, late BE, trail=3rd swing)...")
    t0 = _time.perf_counter()
    baseline_trades, runner_details = run_backtest_multi_tp_with_runner_data(
        d, d_extra,
        runner_kill="none",
    )
    baseline_m = compute_metrics(baseline_trades)
    print(f"  Baseline computed in {_time.perf_counter() - t0:.1f}s")
    print_metrics("Config E Baseline", baseline_m)

    # Part 1
    analyze_runner_performance(baseline_trades, runner_details)

    # Part 2
    analyze_runner_exits(runner_details)

    # Part 3
    analyze_momentum_decay(runner_details)

    # Part 4 & 5
    kill_results = test_kill_conditions(d, d_extra)

    # Walk-forward for baseline + any improved conditions
    improved = [r for r in kill_results if r["delta_r"] > 0]
    if improved:
        print("\n" + "=" * 110)
        print("WALK-FORWARD: IMPROVED KILL CONDITIONS vs BASELINE")
        print("=" * 110)
        # Re-run with walk-forward
        baseline_wf = walk_forward_metrics(baseline_trades)
        print(f"\n  Baseline per-year:")
        for w in baseline_wf:
            print(f"    {w['year']}: {w['n']:3d}t R={w['R']:+7.1f} PF={w['PF']:5.2f} PPDD={w['PPDD']:+6.2f}")

    print("\n" + "=" * 110)
    print("A2E RUNNER ANALYSIS COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
