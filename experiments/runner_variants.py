"""
experiments/runner_variants.py — Test runner management variants.

Variants:
  1. Two-stage trim (25% + 25%)
  2. Three-stage trim (33% + 33% + 33%)
  3. BE offset (+0.25R)
  4. ATR-based trailing stop (multipliers: 1.5, 2.0, 2.5, 3.0)
  5. Early PA cut threshold variants (disabled / tighten / loosen)
  6. Time-based runner exit (N = 20, 40, 60, 100 bars)

Base config: Config D — sq_short=0.80, block_pm_shorts=True

Usage: python experiments/runner_variants.py
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
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", stream=sys.stdout)

# Reuse load_all from validate_improvements
from experiments.validate_improvements import (
    load_all,
    compute_metrics,
    walk_forward_metrics,
    print_metrics,
    _find_nth_swing,
    _compute_grade_fast,
)


# ======================================================================
# Variant backtest engine
# ======================================================================
def run_backtest_variants(
    d: dict,
    # Standard Config D params
    sq_long: float = 0.68,
    sq_short: float = 0.80,
    min_stop_atr: float = 1.7,
    # Config D improvements (always on)
    block_pm_shorts: bool = True,
    # Variant selection
    variant: str = "baseline",
    # Variant-specific params
    variant_params: dict | None = None,
    # Date range
    start_date=None, end_date=None,
) -> list[dict]:
    """Backtest with runner management variants.

    variant options:
      "baseline"          — current Config D behavior
      "two_stage_trim"    — 25% + 25% trim
      "three_stage_trim"  — 33% + 33% + 34% trim
      "be_offset"         — BE + 0.25R offset after trim
      "atr_trail"         — ATR-based trailing stop (variant_params: atr_mult)
      "early_cut_disabled"— no early PA cut
      "early_cut_tight"   — wick>0.55, favorable<0.6, bars>=2
      "early_cut_loose"   — wick>0.75, favorable<0.4, bars>=4
      "time_exit"         — time-based runner exit (variant_params: max_bars)
    """

    if variant_params is None:
        variant_params = {}

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
    pos_stop_dist = 0.0  # store original stop distance for BE offset calc
    last_signal_ts_minutes = -999999

    # Multi-stage trim state
    pos_trim_stage = 0       # 0=not trimmed, 1=first trim done, 2=second trim done
    pos_tp2 = 0.0            # second take-profit target
    pos_tp3 = 0.0            # third take-profit target (three-stage)
    pos_total_trim_pnl = 0.0 # accumulated PnL from partial trims (in points * contracts)
    pos_total_trim_contracts = 0  # accumulated trimmed contracts

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

            # === EARLY PA CUT (variant-dependent) ===
            apply_early_cut = True
            early_wick_thresh = 0.65
            early_fav_thresh = 0.5
            early_min_bars = 3
            early_max_bars = 4

            if variant == "early_cut_disabled":
                apply_early_cut = False
            elif variant == "early_cut_tight":
                early_wick_thresh = 0.55
                early_fav_thresh = 0.6
                early_min_bars = 2
                early_max_bars = 4
            elif variant == "early_cut_loose":
                early_wick_thresh = 0.75
                early_fav_thresh = 0.4
                early_min_bars = 4
                early_max_bars = 6

            any_trim_done = pos_trimmed if variant not in ("two_stage_trim", "three_stage_trim") else (pos_trim_stage > 0)

            if apply_early_cut and not any_trim_done and early_min_bars <= bars_in_trade <= early_max_bars:
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
                bad_pa = avg_wick > early_wick_thresh and favorable < early_fav_thresh
                if bad_pa and no_progress and bars_in_trade >= early_min_bars:
                    exit_price = o[i+1] if i+1 < end_idx else c_arr[i]
                    exit_reason = "early_cut_pa"
                    exited = True

            # === TIME-BASED RUNNER EXIT ===
            if variant == "time_exit" and not exited and any_trim_done:
                max_runner_bars = variant_params.get("max_bars", 60)
                if bars_in_trade >= max_runner_bars:
                    exit_price = c_arr[i]
                    exit_reason = "time_exit"
                    exited = True

            # ==============================
            # LONG EXIT LOGIC
            # ==============================
            if not exited and pos_direction == 1:
                # Determine effective stop
                if variant in ("two_stage_trim", "three_stage_trim"):
                    # Multi-stage: use trail/BE after any trim
                    if pos_trim_stage > 0:
                        eff_stop = pos_trail_stop if pos_trail_stop > 0 else pos_stop
                        if be_after_trim and pos_be_stop > 0:
                            eff_stop = max(eff_stop, pos_be_stop)
                    else:
                        eff_stop = pos_stop
                else:
                    # Standard baseline / other variants
                    eff_stop = pos_trail_stop if pos_trimmed and pos_trail_stop > 0 else pos_stop
                    if pos_trimmed and be_after_trim and pos_be_stop > 0:
                        eff_stop = max(eff_stop, pos_be_stop)

                # Check stop hit
                if l[i] <= eff_stop:
                    exit_price = eff_stop - slippage_points
                    if any_trim_done and eff_stop >= pos_entry_price:
                        exit_reason = "be_sweep"
                    else:
                        exit_reason = "stop"
                    exited = True

                # === TRIM LOGIC (LONG) ===
                elif variant == "two_stage_trim":
                    if pos_trim_stage == 0 and h[i] >= pos_tp1:
                        # First trim: 25% at TP1
                        tc = max(1, int(pos_contracts * 0.25))
                        pos_remaining_contracts -= tc
                        pos_total_trim_contracts += tc
                        trim_pts = pos_tp1 - pos_entry_price
                        pos_total_trim_pnl += trim_pts * point_value * tc
                        pos_trim_stage = 1
                        # BE offset
                        if variant == "be_offset":
                            pos_be_stop = pos_entry_price + 0.25 * pos_stop_dist
                        else:
                            pos_be_stop = pos_entry_price
                        pos_trail_stop = _find_nth_swing(swing_low_mask, nq["low"].values, i, nth_swing, 1)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop
                        if pos_remaining_contracts <= 0:
                            exit_price = pos_tp1
                            exit_reason = "tp1"
                            exit_contracts = pos_contracts
                            exited = True
                    elif pos_trim_stage == 1 and not exited and h[i] >= pos_tp2:
                        # Second trim: 25% at TP2
                        tc = max(1, int(pos_contracts * 0.25))
                        tc = min(tc, pos_remaining_contracts)
                        pos_remaining_contracts -= tc
                        pos_total_trim_contracts += tc
                        trim_pts = pos_tp2 - pos_entry_price
                        pos_total_trim_pnl += trim_pts * point_value * tc
                        pos_trim_stage = 2
                        if pos_remaining_contracts <= 0:
                            exit_price = pos_tp2
                            exit_reason = "tp2"
                            exit_contracts = pos_contracts
                            exited = True

                elif variant == "three_stage_trim":
                    if pos_trim_stage == 0 and h[i] >= pos_tp1:
                        # First trim: 33% at TP1
                        tc = max(1, int(pos_contracts * 0.33))
                        pos_remaining_contracts -= tc
                        pos_total_trim_contracts += tc
                        trim_pts = pos_tp1 - pos_entry_price
                        pos_total_trim_pnl += trim_pts * point_value * tc
                        pos_trim_stage = 1
                        pos_be_stop = pos_entry_price
                        pos_trail_stop = _find_nth_swing(swing_low_mask, nq["low"].values, i, nth_swing, 1)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop
                        if pos_remaining_contracts <= 0:
                            exit_price = pos_tp1
                            exit_reason = "tp1"
                            exit_contracts = pos_contracts
                            exited = True
                    elif pos_trim_stage == 1 and not exited and h[i] >= pos_tp2:
                        # Second trim: 33% at TP2
                        tc = max(1, int(pos_contracts * 0.33))
                        tc = min(tc, pos_remaining_contracts)
                        pos_remaining_contracts -= tc
                        pos_total_trim_contracts += tc
                        trim_pts = pos_tp2 - pos_entry_price
                        pos_total_trim_pnl += trim_pts * point_value * tc
                        pos_trim_stage = 2
                        if pos_remaining_contracts <= 0:
                            exit_price = pos_tp2
                            exit_reason = "tp2"
                            exit_contracts = pos_contracts
                            exited = True

                elif not exited:
                    # Standard baseline trim (50% at TP1)
                    if not pos_trimmed and h[i] >= pos_tp1:
                        tc = max(1, int(pos_contracts * pos_trim_pct))
                        pos_remaining_contracts = pos_contracts - tc
                        pos_trimmed = True
                        # BE offset variant
                        if variant == "be_offset":
                            be_offset = variant_params.get("be_offset_mult", 0.25)
                            pos_be_stop = pos_entry_price + be_offset * pos_stop_dist
                        else:
                            pos_be_stop = pos_entry_price

                        if pos_remaining_contracts > 0:
                            if variant == "atr_trail":
                                atr_mult = variant_params.get("atr_mult", 2.0)
                                cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                                # Use highest close so far
                                highest_close = np.max(c_arr[pos_entry_idx:i+1])
                                pos_trail_stop = highest_close - atr_mult * cur_atr
                            else:
                                pos_trail_stop = _find_nth_swing(swing_low_mask, nq["low"].values, i, nth_swing, 1)
                            if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                                pos_trail_stop = pos_be_stop
                        if pos_remaining_contracts <= 0:
                            exit_price = pos_tp1
                            exit_reason = "tp1"
                            exit_contracts = pos_contracts
                            exited = True

                # Trail stop update (every bar after any trim)
                if not exited:
                    if variant in ("two_stage_trim", "three_stage_trim"):
                        if pos_trim_stage > 0:
                            if variant == "atr_trail":
                                atr_mult = variant_params.get("atr_mult", 2.0)
                                cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                                highest_close = np.max(c_arr[pos_entry_idx:i+1])
                                new_trail = highest_close - atr_mult * cur_atr
                                if new_trail > pos_trail_stop:
                                    pos_trail_stop = new_trail
                            else:
                                nt = _find_nth_swing(swing_low_mask, nq["low"].values, i, nth_swing, 1)
                                if not np.isnan(nt) and nt > pos_trail_stop:
                                    pos_trail_stop = nt
                    elif pos_trimmed:
                        if variant == "atr_trail":
                            atr_mult = variant_params.get("atr_mult", 2.0)
                            cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                            highest_close = np.max(c_arr[pos_entry_idx:i+1])
                            new_trail = highest_close - atr_mult * cur_atr
                            if new_trail > pos_trail_stop:
                                pos_trail_stop = new_trail
                        else:
                            nt = _find_nth_swing(swing_low_mask, nq["low"].values, i, nth_swing, 1)
                            if not np.isnan(nt) and nt > pos_trail_stop:
                                pos_trail_stop = nt

            # ==============================
            # SHORT EXIT LOGIC
            # ==============================
            elif not exited and pos_direction == -1:
                # Determine effective stop
                if variant in ("two_stage_trim", "three_stage_trim"):
                    if pos_trim_stage > 0:
                        eff_stop = pos_trail_stop if pos_trail_stop > 0 else pos_stop
                        if be_after_trim and pos_be_stop > 0:
                            eff_stop = min(eff_stop, pos_be_stop)
                    else:
                        eff_stop = pos_stop
                else:
                    eff_stop = pos_trail_stop if pos_trimmed and pos_trail_stop > 0 else pos_stop
                    if pos_trimmed and be_after_trim and pos_be_stop > 0:
                        eff_stop = min(eff_stop, pos_be_stop)

                # Check stop hit
                if h[i] >= eff_stop:
                    exit_price = eff_stop + slippage_points
                    if any_trim_done and eff_stop <= pos_entry_price:
                        exit_reason = "be_sweep"
                    else:
                        exit_reason = "stop"
                    exited = True

                # === TRIM LOGIC (SHORT) ===
                elif variant == "two_stage_trim":
                    if pos_trim_stage == 0 and l[i] <= pos_tp1:
                        tc = max(1, int(pos_contracts * 0.25))
                        pos_remaining_contracts -= tc
                        pos_total_trim_contracts += tc
                        trim_pts = pos_entry_price - pos_tp1
                        pos_total_trim_pnl += trim_pts * point_value * tc
                        pos_trim_stage = 1
                        pos_be_stop = pos_entry_price
                        pos_trail_stop = _find_nth_swing(swing_high_mask, nq["high"].values, i, nth_swing, -1)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop
                        if pos_remaining_contracts <= 0:
                            exit_price = pos_tp1
                            exit_reason = "tp1"
                            exit_contracts = pos_contracts
                            exited = True
                    elif pos_trim_stage == 1 and not exited and l[i] <= pos_tp2:
                        tc = max(1, int(pos_contracts * 0.25))
                        tc = min(tc, pos_remaining_contracts)
                        pos_remaining_contracts -= tc
                        pos_total_trim_contracts += tc
                        trim_pts = pos_entry_price - pos_tp2
                        pos_total_trim_pnl += trim_pts * point_value * tc
                        pos_trim_stage = 2
                        if pos_remaining_contracts <= 0:
                            exit_price = pos_tp2
                            exit_reason = "tp2"
                            exit_contracts = pos_contracts
                            exited = True

                elif variant == "three_stage_trim":
                    if pos_trim_stage == 0 and l[i] <= pos_tp1:
                        tc = max(1, int(pos_contracts * 0.33))
                        pos_remaining_contracts -= tc
                        pos_total_trim_contracts += tc
                        trim_pts = pos_entry_price - pos_tp1
                        pos_total_trim_pnl += trim_pts * point_value * tc
                        pos_trim_stage = 1
                        pos_be_stop = pos_entry_price
                        pos_trail_stop = _find_nth_swing(swing_high_mask, nq["high"].values, i, nth_swing, -1)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop
                        if pos_remaining_contracts <= 0:
                            exit_price = pos_tp1
                            exit_reason = "tp1"
                            exit_contracts = pos_contracts
                            exited = True
                    elif pos_trim_stage == 1 and not exited and l[i] <= pos_tp2:
                        tc = max(1, int(pos_contracts * 0.33))
                        tc = min(tc, pos_remaining_contracts)
                        pos_remaining_contracts -= tc
                        pos_total_trim_contracts += tc
                        trim_pts = pos_entry_price - pos_tp2
                        pos_total_trim_pnl += trim_pts * point_value * tc
                        pos_trim_stage = 2
                        if pos_remaining_contracts <= 0:
                            exit_price = pos_tp2
                            exit_reason = "tp2"
                            exit_contracts = pos_contracts
                            exited = True

                elif not exited:
                    # Standard baseline trim
                    if not pos_trimmed and l[i] <= pos_tp1:
                        tc = max(1, int(pos_contracts * pos_trim_pct))
                        pos_remaining_contracts = pos_contracts - tc
                        pos_trimmed = True
                        if variant == "be_offset":
                            be_offset = variant_params.get("be_offset_mult", 0.25)
                            pos_be_stop = pos_entry_price - be_offset * pos_stop_dist
                        else:
                            pos_be_stop = pos_entry_price

                        if pos_remaining_contracts > 0:
                            if variant == "atr_trail":
                                atr_mult = variant_params.get("atr_mult", 2.0)
                                cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                                lowest_close = np.min(c_arr[pos_entry_idx:i+1])
                                pos_trail_stop = lowest_close + atr_mult * cur_atr
                            else:
                                pos_trail_stop = _find_nth_swing(swing_high_mask, nq["high"].values, i, nth_swing, -1)
                            if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                                pos_trail_stop = pos_be_stop
                        if pos_remaining_contracts <= 0:
                            exit_price = pos_tp1
                            exit_reason = "tp1"
                            exit_contracts = pos_contracts
                            exited = True

                # Trail stop update (short)
                if not exited:
                    if variant in ("two_stage_trim", "three_stage_trim"):
                        if pos_trim_stage > 0:
                            if variant == "atr_trail":
                                atr_mult = variant_params.get("atr_mult", 2.0)
                                cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                                lowest_close = np.min(c_arr[pos_entry_idx:i+1])
                                new_trail = lowest_close + atr_mult * cur_atr
                                if new_trail < pos_trail_stop:
                                    pos_trail_stop = new_trail
                            else:
                                nt = _find_nth_swing(swing_high_mask, nq["high"].values, i, nth_swing, -1)
                                if not np.isnan(nt) and nt < pos_trail_stop:
                                    pos_trail_stop = nt
                    elif pos_trimmed:
                        if variant == "atr_trail":
                            atr_mult = variant_params.get("atr_mult", 2.0)
                            cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                            lowest_close = np.min(c_arr[pos_entry_idx:i+1])
                            new_trail = lowest_close + atr_mult * cur_atr
                            if new_trail < pos_trail_stop:
                                pos_trail_stop = new_trail
                        else:
                            nt = _find_nth_swing(swing_high_mask, nq["high"].values, i, nth_swing, -1)
                            if not np.isnan(nt) and nt < pos_trail_stop:
                                pos_trail_stop = nt

            # ==============================
            # PNL CALCULATION ON EXIT
            # ==============================
            if exited:
                pnl_pts = (exit_price - pos_entry_price) if pos_direction == 1 else (pos_entry_price - exit_price)

                if variant in ("two_stage_trim", "three_stage_trim"):
                    # Multi-stage: accumulated trim PnL + runner PnL
                    if pos_trim_stage > 0 and exit_reason not in ("tp1", "tp2"):
                        # Runner exit (stop/be_sweep/trail/time_exit/early_cut)
                        runner_pnl = pnl_pts * point_value * exit_contracts
                        total_pnl = pos_total_trim_pnl + runner_pnl
                        total_comm = commission_per_side * 2 * pos_contracts
                        total_pnl -= total_comm
                    elif exit_reason in ("tp1", "tp2"):
                        # All contracts exited at some TP level
                        total_pnl = pos_total_trim_pnl
                        # If there are still remaining contracts at exit, add their pnl
                        if exit_contracts > pos_total_trim_contracts:
                            remaining_c = exit_contracts - pos_total_trim_contracts
                            total_pnl += pnl_pts * point_value * remaining_c
                        total_comm = commission_per_side * 2 * pos_contracts
                        total_pnl -= total_comm
                    else:
                        # No trim happened (stop before TP1)
                        total_pnl = pnl_pts * point_value * exit_contracts
                        total_comm = commission_per_side * 2 * exit_contracts
                        total_pnl -= total_comm
                else:
                    # Standard baseline PnL
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
                    "type": pos_signal_type, "trimmed": any_trim_done, "grade": pos_grade,
                    "entry_price": pos_entry_price, "exit_price": exit_price,
                    "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
                })
                daily_pnl_r += r_mult
                if exit_reason == "be_sweep" and any_trim_done:
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

        # Block PM shorts (Config D always on)
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

        # Record last signal timestamp
        last_signal_ts_minutes = ts_minutes[i]

        # Determine trim pct
        if mss_mgmt_enabled and is_mss_signal:
            cur_trim_pct = mss_mgmt.get("short_trim_pct", 1.0) if direction == -1 else mss_mgmt.get("long_trim_pct", trim_pct)
        elif dual_mode_enabled and direction == -1:
            cur_trim_pct = dual_mode.get("short_trim_pct", 1.0)
        elif dir_mgmt.get("enabled", False):
            cur_trim_pct = dir_mgmt.get("long_trim_pct", trim_pct) if direction == 1 else dir_mgmt.get("short_trim_pct", 1.0)
        else:
            cur_trim_pct = trim_pct

        # Compute TP2/TP3 for multi-stage variants
        tp2_val = 0.0
        tp3_val = 0.0
        if variant == "two_stage_trim":
            # TP2 = TP1 + 1.0 * stop_distance (roughly 2R profit)
            if direction == 1:
                tp2_val = pos_tp1 if pos_tp1 != 0 else tp1
                tp2_val = tp1 + 1.0 * stop_dist
            else:
                tp2_val = tp1 - 1.0 * stop_dist
        elif variant == "three_stage_trim":
            tp1_dist = abs(tp1 - actual_entry)
            if direction == 1:
                tp2_val = tp1 + tp1_dist
            else:
                tp2_val = tp1 - tp1_dist

        # Enter position
        in_position = True
        pos_direction = direction
        pos_entry_idx = i + 1
        pos_entry_price = actual_entry
        pos_stop = stop
        pos_tp1 = tp1
        pos_tp2 = tp2_val
        pos_tp3 = tp3_val
        pos_contracts = contracts
        pos_remaining_contracts = contracts
        pos_trimmed = False
        pos_be_stop = 0.0
        pos_trail_stop = 0.0
        pos_signal_type = str(sig_type[i])
        pos_bias_dir = bias_dir_arr[i]
        pos_regime = regime_arr[i]
        pos_grade = grade
        pos_trim_pct = cur_trim_pct
        pos_stop_dist = stop_dist
        pos_trim_stage = 0
        pos_total_trim_pnl = 0.0
        pos_total_trim_contracts = 0

    # Force close at end
    if in_position and pos_entry_idx < end_idx:
        last_i = end_idx - 1
        exit_price = c_arr[last_i]
        pnl_pts = (exit_price - pos_entry_price) if pos_direction == 1 else (pos_entry_price - exit_price)

        any_trim_done = pos_trimmed if variant not in ("two_stage_trim", "three_stage_trim") else (pos_trim_stage > 0)

        if variant in ("two_stage_trim", "three_stage_trim") and pos_trim_stage > 0:
            runner_pnl = pnl_pts * point_value * pos_remaining_contracts
            total_pnl = pos_total_trim_pnl + runner_pnl
            total_pnl -= commission_per_side * 2 * pos_contracts
        elif pos_trimmed:
            trim_c = pos_contracts - pos_remaining_contracts
            trim_pnl_eod = ((pos_tp1 - pos_entry_price) if pos_direction == 1 else (pos_entry_price - pos_tp1)) * point_value * trim_c
            total_pnl = pnl_pts * point_value * pos_remaining_contracts + trim_pnl_eod
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
            "type": pos_signal_type, "trimmed": any_trim_done, "grade": pos_grade,
            "entry_price": pos_entry_price, "exit_price": exit_price,
            "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
        })

    return trades


# ======================================================================
# Walk-forward print helper
# ======================================================================
def print_walk_forward(label, trades_list):
    """Print per-year metrics."""
    wf = walk_forward_metrics(trades_list)
    if not wf:
        print(f"  {label}: no trades")
        return
    print(f"\n  Walk-Forward: {label}")
    print(f"  {'Year':>6} | {'n':>5} | {'R':>8} | {'WR':>6} | {'PF':>6} | {'PPDD':>7}")
    print(f"  {'-'*50}")
    neg_years = 0
    for w in wf:
        marker = " <<<" if w["R"] < 0 else ""
        if w["R"] < 0:
            neg_years += 1
        print(f"  {w['year']:>6} | {w['n']:5d} | {w['R']:+8.1f} | {w['WR']:5.1f}% | {w['PF']:5.2f} | {w['PPDD']:+7.2f}{marker}")
    print(f"  Negative years: {neg_years}/{len(wf)}")


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 110)
    print("RUNNER MANAGEMENT VARIANTS — Config D Base (sq_short=0.80, block_pm_shorts=True)")
    print("=" * 110)

    d = load_all()

    # ---- Baseline (Config D) ----
    print("\n" + "=" * 110)
    print("BASELINE — Config D")
    print("=" * 110)
    t0 = _time.perf_counter()
    baseline_trades = run_backtest_variants(d, variant="baseline")
    baseline = compute_metrics(baseline_trades)
    print_metrics("Config D BASELINE", baseline)
    print(f"  (computed in {_time.perf_counter() - t0:.1f}s)")

    # Collect all results for summary table
    all_results = [("BASELINE (Config D)", baseline, baseline_trades)]

    # ---- Variant 1: Two-stage trim ----
    print("\n" + "-" * 110)
    print("VARIANT 1: Two-stage trim (25% + 25%)")
    print("-" * 110)
    t0 = _time.perf_counter()
    v1_trades = run_backtest_variants(d, variant="two_stage_trim")
    v1 = compute_metrics(v1_trades)
    print_metrics("V1: Two-stage trim", v1)
    print(f"  Δ vs baseline: R={v1['R']-baseline['R']:+.1f}, PPDD={v1['PPDD']-baseline['PPDD']:+.2f}, PF={v1['PF']-baseline['PF']:+.2f}  ({_time.perf_counter()-t0:.1f}s)")
    all_results.append(("V1: Two-stage trim (25+25)", v1, v1_trades))

    # ---- Variant 2: Three-stage trim ----
    print("\n" + "-" * 110)
    print("VARIANT 2: Three-stage trim (33% + 33% + 34%)")
    print("-" * 110)
    t0 = _time.perf_counter()
    v2_trades = run_backtest_variants(d, variant="three_stage_trim")
    v2 = compute_metrics(v2_trades)
    print_metrics("V2: Three-stage trim", v2)
    print(f"  Δ vs baseline: R={v2['R']-baseline['R']:+.1f}, PPDD={v2['PPDD']-baseline['PPDD']:+.2f}, PF={v2['PF']-baseline['PF']:+.2f}  ({_time.perf_counter()-t0:.1f}s)")
    all_results.append(("V2: Three-stage trim (33+33+34)", v2, v2_trades))

    # ---- Variant 3: BE offset ----
    print("\n" + "-" * 110)
    print("VARIANT 3: BE offset (+0.25R)")
    print("-" * 110)
    for offset in [0.15, 0.25, 0.35, 0.50]:
        t0 = _time.perf_counter()
        v3_trades = run_backtest_variants(d, variant="be_offset", variant_params={"be_offset_mult": offset})
        v3 = compute_metrics(v3_trades)
        label = f"V3: BE+{offset:.2f}R"
        print_metrics(label, v3)
        print(f"  Δ vs baseline: R={v3['R']-baseline['R']:+.1f}, PPDD={v3['PPDD']-baseline['PPDD']:+.2f}, PF={v3['PF']-baseline['PF']:+.2f}  ({_time.perf_counter()-t0:.1f}s)")
        all_results.append((label, v3, v3_trades))

    # ---- Variant 4: ATR-based trailing stop ----
    print("\n" + "-" * 110)
    print("VARIANT 4: ATR-based trailing stop")
    print("-" * 110)
    for atr_m in [1.5, 2.0, 2.5, 3.0]:
        t0 = _time.perf_counter()
        v4_trades = run_backtest_variants(d, variant="atr_trail", variant_params={"atr_mult": atr_m})
        v4 = compute_metrics(v4_trades)
        label = f"V4: ATR trail x{atr_m:.1f}"
        print_metrics(label, v4)
        print(f"  Δ vs baseline: R={v4['R']-baseline['R']:+.1f}, PPDD={v4['PPDD']-baseline['PPDD']:+.2f}, PF={v4['PF']-baseline['PF']:+.2f}  ({_time.perf_counter()-t0:.1f}s)")
        all_results.append((label, v4, v4_trades))

    # ---- Variant 5: Early PA cut variants ----
    print("\n" + "-" * 110)
    print("VARIANT 5: Early PA cut thresholds")
    print("-" * 110)
    for var_name, var_label in [
        ("early_cut_disabled", "V5A: Early cut DISABLED"),
        ("early_cut_tight",    "V5B: Early cut TIGHT (0.55/0.6/2)"),
        ("early_cut_loose",    "V5C: Early cut LOOSE (0.75/0.4/4)"),
    ]:
        t0 = _time.perf_counter()
        v5_trades = run_backtest_variants(d, variant=var_name)
        v5 = compute_metrics(v5_trades)
        print_metrics(var_label, v5)
        print(f"  Δ vs baseline: R={v5['R']-baseline['R']:+.1f}, PPDD={v5['PPDD']-baseline['PPDD']:+.2f}, PF={v5['PF']-baseline['PF']:+.2f}  ({_time.perf_counter()-t0:.1f}s)")
        all_results.append((var_label, v5, v5_trades))

    # ---- Variant 6: Time-based runner exit ----
    print("\n" + "-" * 110)
    print("VARIANT 6: Time-based runner exit")
    print("-" * 110)
    for max_b in [20, 40, 60, 100]:
        t0 = _time.perf_counter()
        v6_trades = run_backtest_variants(d, variant="time_exit", variant_params={"max_bars": max_b})
        v6 = compute_metrics(v6_trades)
        label = f"V6: Time exit {max_b} bars"
        print_metrics(label, v6)
        print(f"  Δ vs baseline: R={v6['R']-baseline['R']:+.1f}, PPDD={v6['PPDD']-baseline['PPDD']:+.2f}, PF={v6['PF']-baseline['PF']:+.2f}  ({_time.perf_counter()-t0:.1f}s)")
        all_results.append((label, v6, v6_trades))

    # ======================================================================
    # SUMMARY TABLE
    # ======================================================================
    print("\n" + "=" * 110)
    print("SUMMARY TABLE — All Variants vs Config D Baseline")
    print("=" * 110)
    print(f"  {'Variant':<42s} | {'trades':>6} | {'R':>9} | {'PPDD':>7} | {'PF':>6} | {'WR':>6} | {'MaxDD':>6} | {'avgR':>8} | {'dR':>7} | {'dPPDD':>7}")
    print(f"  {'-'*118}")
    for label, m, _ in all_results:
        dr = m["R"] - baseline["R"]
        dppdd = m["PPDD"] - baseline["PPDD"]
        marker = " ***" if m["PPDD"] > baseline["PPDD"] and m["R"] > baseline["R"] else ""
        print(f"  {label:<42s} | {m['trades']:6d} | {m['R']:+9.1f} | {m['PPDD']:7.2f} | {m['PF']:6.2f} | {m['WR']:5.1f}% | {m['MaxDD']:6.1f} | {m['avgR']:+8.4f} | {dr:+7.1f} | {dppdd:+7.2f}{marker}")

    # ======================================================================
    # WALK-FORWARD — Top 3 by PPDD improvement
    # ======================================================================
    print("\n" + "=" * 110)
    print("WALK-FORWARD — Top variants by PPDD improvement")
    print("=" * 110)

    # Sort by PPDD (excluding baseline)
    ranked = sorted(all_results[1:], key=lambda x: x[1]["PPDD"], reverse=True)

    # Also show baseline WF
    print_walk_forward("BASELINE (Config D)", baseline_trades)

    for label, m, trades_list in ranked[:3]:
        print_walk_forward(label, trades_list)

    # Side-by-side WF comparison: baseline vs top 3
    print("\n" + "=" * 110)
    print("WALK-FORWARD COMPARISON — Baseline vs Top 3")
    print("=" * 110)

    wf_bl = walk_forward_metrics(baseline_trades)
    bl_by_year = {w["year"]: w for w in wf_bl}
    top3 = ranked[:3]

    all_years = sorted(set(w["year"] for w in wf_bl))
    header_labels = ["BASELINE"] + [f"#{i+1}" for i in range(len(top3))]
    print(f"\n  Top variants:")
    for i, (label, m, _) in enumerate(top3):
        print(f"    #{i+1}: {label} (R={m['R']:+.1f}, PPDD={m['PPDD']:.2f})")

    print(f"\n  {'Year':>6}", end="")
    for hl in header_labels:
        print(f" | {hl:>12s}", end="")
    print()
    print(f"  {'-'*80}")

    top3_wf = []
    for _, _, trades_list in top3:
        wf = walk_forward_metrics(trades_list)
        top3_wf.append({w["year"]: w for w in wf})

    for y in all_years:
        bl_r = bl_by_year.get(y, {}).get("R", 0)
        print(f"  {y:>6} | {bl_r:+12.1f}", end="")
        for wf_dict in top3_wf:
            yr_data = wf_dict.get(y, {})
            yr_r = yr_data.get("R", 0)
            print(f" | {yr_r:+12.1f}", end="")
        print()

    # Count wins per variant
    print(f"\n  Year-over-year wins vs baseline:")
    for i, (label, m, _) in enumerate(top3):
        wins = 0
        for y in all_years:
            bl_r = bl_by_year.get(y, {}).get("R", 0)
            vr = top3_wf[i].get(y, {}).get("R", 0)
            if vr > bl_r:
                wins += 1
        print(f"    #{i+1} {label}: {wins}/{len(all_years)} years")

    print("\n" + "=" * 110)
    print("DONE")
    print("=" * 110)


if __name__ == "__main__":
    main()
