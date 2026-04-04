"""
experiments/pure_liquidity_tp.py — Pure liquidity-level take-profit targets (no fixed multipliers).

Tests whether using ACTUAL liquidity levels (swings, session highs/lows) as TP targets
outperforms fixed multiplier-based targets.  Mandatory EOD close at 15:55 ET.

Variants:
  V1: Raw IRL (no multiplier) + next liquidity level
  V2: Raw swing-to-swing (1st swing TP1, 2nd swing TP2)
  V3: Session level as primary TP1
  V4: Hybrid — swing first, session second
  V5: Multiplier sweep on raw IRL (1.0, 1.25, 1.5, 2.0)
  V6: Distance-calibrated (closest level at 1.0/2.0 ATR)
  V7: Config G baseline (ny_tp_mult=1.5) for comparison

All use: 50/50/0 split, EOD close, sq_short=0.80, block_pm_shorts=True, phantom runner fix.
Shorts: dual_mode scalp (100% exit at 0.625R).

Usage: python experiments/pure_liquidity_tp.py
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
    _find_nth_swing_price,
    build_liquidity_ladder_long,
    build_liquidity_ladder_short,
)


# ======================================================================
# TP computation strategies for LONGS
# ======================================================================

def _compute_tp_v1_raw_irl(
    actual_entry: float, stop: float, irl_target: float,
    idx: int, d_extra: dict, atr_val: float,
) -> tuple[float, float]:
    """V1: Raw IRL (no multiplier) + next liquidity level above."""
    stop_dist = abs(actual_entry - stop)
    min_dist = stop_dist * 0.5

    tp1 = irl_target
    if tp1 - actual_entry < min_dist:
        tp1 = actual_entry + min_dist

    # TP2: next distinct liquidity level above TP1 (from ladder)
    candidates = []
    for arr in [d_extra["sess_asia_high"], d_extra["sess_london_high"],
                d_extra["sess_overnight_high"]]:
        val = arr[idx]
        if not np.isnan(val) and val > tp1 + 1.0:
            candidates.append(val)
    htf_sh = d_extra["htf_swing_high_price"][idx]
    if not np.isnan(htf_sh) and htf_sh > tp1 + 1.0:
        candidates.append(htf_sh)
    sh2 = _find_nth_swing_price(d_extra["reg_sh_mask"], d_extra["reg_high_arr"], idx, 2)
    if not np.isnan(sh2) and sh2 > tp1 + 1.0:
        candidates.append(sh2)
    sh3 = _find_nth_swing_price(d_extra["reg_sh_mask"], d_extra["reg_high_arr"], idx, 3)
    if not np.isnan(sh3) and sh3 > tp1 + 1.0:
        candidates.append(sh3)

    if candidates:
        tp2 = min(candidates)
    else:
        tp2 = actual_entry + (tp1 - actual_entry) * 1.5

    return tp1, tp2


def _compute_tp_v2_swing_to_swing(
    actual_entry: float, stop: float, irl_target: float,
    idx: int, d_extra: dict, atr_val: float,
) -> tuple[float, float]:
    """V2: 1st swing high = TP1, 2nd swing high = TP2."""
    stop_dist = abs(actual_entry - stop)
    min_dist = stop_dist * 0.5

    tp1 = irl_target  # Already nearest swing
    if tp1 - actual_entry < min_dist:
        tp1 = actual_entry + min_dist

    # 2nd swing high above entry
    sh2 = _find_nth_swing_price(d_extra["reg_sh_mask"], d_extra["reg_high_arr"], idx, 2)
    if not np.isnan(sh2) and sh2 > tp1 + 1.0:
        tp2 = sh2
    else:
        # Fallback: 3rd swing or extend
        sh3 = _find_nth_swing_price(d_extra["reg_sh_mask"], d_extra["reg_high_arr"], idx, 3)
        if not np.isnan(sh3) and sh3 > tp1 + 1.0:
            tp2 = sh3
        else:
            tp2 = actual_entry + (tp1 - actual_entry) * 1.5

    return tp1, tp2


def _compute_tp_v3_session_primary(
    actual_entry: float, stop: float, irl_target: float,
    idx: int, d_extra: dict, atr_val: float,
) -> tuple[float, float]:
    """V3: Session level as primary TP1, next session/swing as TP2."""
    stop_dist = abs(actual_entry - stop)
    min_dist = stop_dist * 0.5

    # Gather session levels above entry with min 0.5R distance
    sess_candidates = []
    for arr in [d_extra["sess_overnight_high"], d_extra["sess_london_high"],
                d_extra["sess_asia_high"]]:
        val = arr[idx]
        if not np.isnan(val) and val > actual_entry + min_dist:
            sess_candidates.append(val)

    if sess_candidates:
        tp1 = min(sess_candidates)  # Nearest session level
    else:
        # Fallback to IRL swing target
        tp1 = irl_target
        if tp1 - actual_entry < min_dist:
            tp1 = actual_entry + min_dist

    # TP2: next session level or swing above TP1
    tp2_candidates = []
    for arr in [d_extra["sess_overnight_high"], d_extra["sess_london_high"],
                d_extra["sess_asia_high"]]:
        val = arr[idx]
        if not np.isnan(val) and val > tp1 + 1.0:
            tp2_candidates.append(val)
    htf_sh = d_extra["htf_swing_high_price"][idx]
    if not np.isnan(htf_sh) and htf_sh > tp1 + 1.0:
        tp2_candidates.append(htf_sh)
    sh2 = _find_nth_swing_price(d_extra["reg_sh_mask"], d_extra["reg_high_arr"], idx, 2)
    if not np.isnan(sh2) and sh2 > tp1 + 1.0:
        tp2_candidates.append(sh2)

    if tp2_candidates:
        tp2 = min(tp2_candidates)
    else:
        tp2 = actual_entry + (tp1 - actual_entry) * 1.5

    return tp1, tp2


def _compute_tp_v4_hybrid(
    actual_entry: float, stop: float, irl_target: float,
    idx: int, d_extra: dict, atr_val: float,
) -> tuple[float, float]:
    """V4: Hybrid — swing first (raw IRL), session level second."""
    stop_dist = abs(actual_entry - stop)
    min_dist = stop_dist * 0.5

    # TP1 = nearest swing (raw IRL, high probability ~90%)
    tp1 = irl_target
    if tp1 - actual_entry < min_dist:
        tp1 = actual_entry + min_dist

    # TP2 = nearest session level above TP1
    sess_candidates = []
    for arr in [d_extra["sess_overnight_high"], d_extra["sess_london_high"],
                d_extra["sess_asia_high"]]:
        val = arr[idx]
        if not np.isnan(val) and val > tp1 + 1.0:
            sess_candidates.append(val)

    if sess_candidates:
        tp2 = min(sess_candidates)
    else:
        # Fallback: HTF swing or 2nd regular swing
        fallback = []
        htf_sh = d_extra["htf_swing_high_price"][idx]
        if not np.isnan(htf_sh) and htf_sh > tp1 + 1.0:
            fallback.append(htf_sh)
        sh2 = _find_nth_swing_price(d_extra["reg_sh_mask"], d_extra["reg_high_arr"], idx, 2)
        if not np.isnan(sh2) and sh2 > tp1 + 1.0:
            fallback.append(sh2)
        if fallback:
            tp2 = min(fallback)
        else:
            tp2 = actual_entry + (tp1 - actual_entry) * 1.5

    return tp1, tp2


def _compute_tp_v5_mult_sweep(
    actual_entry: float, stop: float, irl_target: float,
    idx: int, d_extra: dict, atr_val: float,
    mult: float = 1.0,
) -> tuple[float, float]:
    """V5: IRL * mult for TP1, next liquidity above for TP2."""
    stop_dist = abs(actual_entry - stop)
    min_dist = stop_dist * 0.5

    tp_distance = irl_target - actual_entry
    tp1 = actual_entry + tp_distance * mult
    if tp1 - actual_entry < min_dist:
        tp1 = actual_entry + min_dist

    # TP2: next liquidity level above TP1
    candidates = []
    for arr in [d_extra["sess_asia_high"], d_extra["sess_london_high"],
                d_extra["sess_overnight_high"]]:
        val = arr[idx]
        if not np.isnan(val) and val > tp1 + 1.0:
            candidates.append(val)
    htf_sh = d_extra["htf_swing_high_price"][idx]
    if not np.isnan(htf_sh) and htf_sh > tp1 + 1.0:
        candidates.append(htf_sh)
    sh2 = _find_nth_swing_price(d_extra["reg_sh_mask"], d_extra["reg_high_arr"], idx, 2)
    if not np.isnan(sh2) and sh2 > tp1 + 1.0:
        candidates.append(sh2)

    if candidates:
        tp2 = min(candidates)
    else:
        tp2 = actual_entry + (tp1 - actual_entry) * 1.5

    return tp1, tp2


def _compute_tp_v6_distance_calibrated(
    actual_entry: float, stop: float, irl_target: float,
    idx: int, d_extra: dict, atr_val: float,
) -> tuple[float, float]:
    """V6: Distance-calibrated — closest level at/above 1.0 ATR (TP1) and 2.0 ATR (TP2)."""
    stop_dist = abs(actual_entry - stop)
    atr = atr_val if not np.isnan(atr_val) and atr_val > 0 else 30.0

    target_dist_1 = atr * 1.0
    target_dist_2 = atr * 2.0

    # Gather ALL liquidity levels above entry
    all_levels = []
    # Session levels
    for arr in [d_extra["sess_asia_high"], d_extra["sess_london_high"],
                d_extra["sess_overnight_high"]]:
        val = arr[idx]
        if not np.isnan(val) and val > actual_entry + 1.0:
            all_levels.append(val)
    # HTF swing
    htf_sh = d_extra["htf_swing_high_price"][idx]
    if not np.isnan(htf_sh) and htf_sh > actual_entry + 1.0:
        all_levels.append(htf_sh)
    # Regular swings (1st, 2nd, 3rd)
    for n_val in [1, 2, 3]:
        sh = _find_nth_swing_price(d_extra["reg_sh_mask"], d_extra["reg_high_arr"], idx, n_val)
        if not np.isnan(sh) and sh > actual_entry + 1.0:
            all_levels.append(sh)
    # Raw IRL
    if not np.isnan(irl_target) and irl_target > actual_entry + 1.0:
        all_levels.append(irl_target)

    all_levels = sorted(set(all_levels))

    # TP1: closest level at or above 1.0 ATR
    tp1 = None
    for lv in all_levels:
        if lv - actual_entry >= target_dist_1:
            tp1 = lv
            break
    if tp1 is None:
        tp1 = actual_entry + target_dist_1

    # TP2: closest level at or above 2.0 ATR (and above TP1)
    tp2 = None
    for lv in all_levels:
        if lv - actual_entry >= target_dist_2 and lv > tp1 + 1.0:
            tp2 = lv
            break
    if tp2 is None:
        tp2 = actual_entry + target_dist_2

    return tp1, tp2


# ======================================================================
# Main backtest engine — pure liquidity TP
# ======================================================================
def run_backtest_pure_liquidity(
    d: dict,
    d_extra: dict,
    # Signal filters (Config D/G base)
    sq_long: float = 0.68,
    sq_short: float = 0.80,
    min_stop_atr: float = 1.7,
    block_pm_shorts: bool = True,
    # TP strategy selection
    tp_strategy: str = "v1",  # v1..v7
    # V5 specific: multiplier
    v5_mult: float = 1.0,
    # V7 specific: ny_tp_mult for Config G baseline
    v7_ny_tp_mult: float = 1.5,
    # Split: 50/50/0 (TP1=50%, TP2 takes ALL remaining => phantom runner fix)
    tp1_trim_pct: float = 0.50,
    # BE management
    be_after_tp1: bool = True,
    be_after_tp2: bool = True,
    # Trail (for safety — shouldn't trigger with 50/50/0 split)
    trail_nth_swing: int = 3,
    # Date range
    start_date=None, end_date=None,
) -> list[dict]:
    """Backtest with pure liquidity-level TP targets for longs, scalp for shorts."""

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

    # Config from params.yaml
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
    pos_trim_stage = 0  # 0=untrimmed, 1=TP1 trimmed, 2=TP2 trimmed (all out)
    pos_be_stop = pos_trail_stop = 0.0
    pos_signal_type = pos_grade = ""
    pos_bias_dir = pos_regime = 0.0
    pos_is_multi_tp = False
    pos_trim1_contracts = 0
    pos_trim2_contracts = 0
    pos_tp1_pnl_pts = 0.0
    pos_tp2_pnl_pts = 0.0
    # For single-TP (shorts)
    pos_single_trim_pct = default_trim_pct
    pos_single_trimmed = False

    # Diagnostics: track TP distances and hit info
    tp1_distances_pts = []
    tp2_distances_pts = []
    tp1_distances_atr = []
    tp2_distances_atr = []
    tp1_hit_count = 0
    tp2_hit_count = 0
    eod_close_count = 0
    long_trade_count = 0

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

            # --- MANDATORY EOD CLOSE at 15:55 ET ---
            if et_frac_arr[i] >= 15.917:  # 15:55 ET
                exit_price = c_arr[i] - slippage_points if pos_direction == 1 else c_arr[i] + slippage_points
                exit_reason = "eod_close"
                exited = True

            # --- Early cut on bad PA (pre-trim only) ---
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
                    exit_price = o[i + 1] if i + 1 < end_idx else c_arr[i]
                    exit_reason = "early_cut_pa"
                    exited = True

            # ==================================================================
            # MULTI-TP EXIT (LONGS only — 50/50/0 split)
            # ==================================================================
            if not exited and pos_is_multi_tp and pos_direction == 1:
                # Effective stop
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
                    tp1_hit_count += 1
                    if be_after_tp1:
                        pos_be_stop = pos_entry_price
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True

                # Stage 1: Check TP2 — take ALL remaining (phantom runner fix)
                if not exited and pos_trim_stage == 1 and h[i] >= pos_tp2:
                    tc2 = pos_remaining_contracts  # ALL remaining
                    pos_trim2_contracts = tc2
                    pos_remaining_contracts -= tc2
                    pos_tp2_pnl_pts = pos_tp2 - pos_entry_price
                    pos_trim_stage = 2
                    tp2_hit_count += 1
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

                # Stage 2: Trail runner (shouldn't happen with 50/50/0)
                if not exited and pos_trim_stage == 2:
                    nt = _find_nth_swing(swing_low_mask, low_arr, i, trail_nth_swing, 1)
                    if not np.isnan(nt) and nt > pos_trail_stop:
                        pos_trail_stop = nt

            # ==================================================================
            # SINGLE-TP EXIT (SHORTS — dual mode scalp)
            # ==================================================================
            elif not exited and not pos_is_multi_tp:
                if pos_direction == -1:
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
                    # FIX: use pos_remaining_contracts (0 for TP2 exits), not exit_contracts
                    total_pnl += pnl_pts_runner * point_value * pos_remaining_contracts
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

                # Track EOD close for longs
                if pos_direction == 1 and exit_reason == "eod_close":
                    eod_close_count += 1

                trades.append({
                    "entry_time": nq.index[pos_entry_idx],
                    "exit_time": nq.index[i + 1] if exit_reason == "early_cut_pa" and i + 1 < end_idx else nq.index[i],
                    "r": r_mult, "reason": exit_reason, "dir": pos_direction,
                    "type": pos_signal_type, "trimmed": any_trimmed, "grade": pos_grade,
                    "entry_price": pos_entry_price, "exit_price": exit_price,
                    "stop_price": pos_stop, "tp1_price": pos_tp1, "tp2_price": pos_tp2,
                    "pnl_dollars": total_pnl, "trim_stage": pos_trim_stage,
                })
                daily_pnl_r += r_mult
                if exit_reason == "be_sweep" and any_trimmed:
                    pass
                elif exit_reason == "eod_close":
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

        # ==================================================================
        # TP COMPUTATION — strategy-dependent for LONGS
        # ==================================================================
        is_mss_signal = str(sig_type[i]) == "mss"
        atr_val = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0

        if direction == 1:
            # LONG: use selected TP strategy
            if tp_strategy == "v1":
                tp1, tp2 = _compute_tp_v1_raw_irl(actual_entry, stop, tp1_raw, i, d_extra, atr_val)
            elif tp_strategy == "v2":
                tp1, tp2 = _compute_tp_v2_swing_to_swing(actual_entry, stop, tp1_raw, i, d_extra, atr_val)
            elif tp_strategy == "v3":
                tp1, tp2 = _compute_tp_v3_session_primary(actual_entry, stop, tp1_raw, i, d_extra, atr_val)
            elif tp_strategy == "v4":
                tp1, tp2 = _compute_tp_v4_hybrid(actual_entry, stop, tp1_raw, i, d_extra, atr_val)
            elif tp_strategy == "v5":
                tp1, tp2 = _compute_tp_v5_mult_sweep(actual_entry, stop, tp1_raw, i, d_extra, atr_val, mult=v5_mult)
            elif tp_strategy == "v6":
                tp1, tp2 = _compute_tp_v6_distance_calibrated(actual_entry, stop, tp1_raw, i, d_extra, atr_val)
            elif tp_strategy == "v7":
                # Config G baseline: apply ny_tp_mult to IRL, then build ladder
                tp1_base = tp1_raw
                if session_rules.get("enabled", False) and 9.5 <= et_frac < 16.0:
                    actual_tp_mult = v7_ny_tp_mult
                    if mss_mgmt_enabled and is_mss_signal:
                        actual_tp_mult = mss_mgmt.get("long_tp_mult", 2.5)
                    tp_distance = tp1_raw - actual_entry
                    tp1_base = actual_entry + tp_distance * actual_tp_mult
                tp1, tp2 = build_liquidity_ladder_long(actual_entry, stop, tp1_base, i, d_extra)
            else:
                raise ValueError(f"Unknown tp_strategy: {tp_strategy}")

            # Track TP distances
            tp1_dist = tp1 - actual_entry
            tp2_dist = tp2 - actual_entry
            tp1_distances_pts.append(tp1_dist)
            tp2_distances_pts.append(tp2_dist)
            tp1_distances_atr.append(tp1_dist / atr_val if atr_val > 0 else 0)
            tp2_distances_atr.append(tp2_dist / atr_val if atr_val > 0 else 0)
            long_trade_count += 1

            # Enter multi-TP position
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
            # SHORT: dual_mode scalp (100% exit at 0.625R)
            tp1 = tp1_raw
            if session_rules.get("enabled", False):
                if dir_mgmt.get("enabled", False):
                    actual_tp_mult = dir_mgmt.get("short_tp_mult", 1.25)
                else:
                    actual_tp_mult = v7_ny_tp_mult
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
                pos_single_trim_pct = mss_mgmt.get("short_trim_pct", 1.0)
            elif dual_mode_enabled:
                pos_single_trim_pct = dual_mode.get("short_trim_pct", 1.0)
            elif dir_mgmt.get("enabled", False):
                pos_single_trim_pct = dir_mgmt.get("short_trim_pct", 1.0)
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
            "stop_price": pos_stop, "tp1_price": pos_tp1, "tp2_price": pos_tp2,
            "pnl_dollars": total_pnl, "trim_stage": pos_trim_stage,
        })

    # Attach diagnostics
    diag = {
        "long_trades": long_trade_count,
        "tp1_hit": tp1_hit_count,
        "tp2_hit": tp2_hit_count,
        "eod_close": eod_close_count,
        "avg_tp1_pts": float(np.mean(tp1_distances_pts)) if tp1_distances_pts else 0.0,
        "avg_tp2_pts": float(np.mean(tp2_distances_pts)) if tp2_distances_pts else 0.0,
        "avg_tp1_atr": float(np.mean(tp1_distances_atr)) if tp1_distances_atr else 0.0,
        "avg_tp2_atr": float(np.mean(tp2_distances_atr)) if tp2_distances_atr else 0.0,
    }

    return trades, diag


# ======================================================================
# Diagnostics
# ======================================================================
def print_diagnostics(label: str, diag: dict):
    """Print TP distance and hit rate diagnostics."""
    lt = diag["long_trades"]
    if lt == 0:
        print(f"  {label}: No long trades")
        return
    tp1_hr = 100.0 * diag["tp1_hit"] / lt if lt > 0 else 0
    tp2_hr = 100.0 * diag["tp2_hit"] / lt if lt > 0 else 0
    eod_pct = 100.0 * diag["eod_close"] / lt if lt > 0 else 0
    print(f"    Longs: {lt} | TP1 hit: {diag['tp1_hit']} ({tp1_hr:.1f}%) | TP2 hit: {diag['tp2_hit']} ({tp2_hr:.1f}%) | EOD: {diag['eod_close']} ({eod_pct:.1f}%)")
    print(f"    Avg TP1: {diag['avg_tp1_pts']:.1f} pts ({diag['avg_tp1_atr']:.2f} ATR) | Avg TP2: {diag['avg_tp2_pts']:.1f} pts ({diag['avg_tp2_atr']:.2f} ATR)")


def print_long_short_breakdown(trades_list: list[dict], label: str):
    """Print long/short R breakdown and exit reasons for longs."""
    if not trades_list:
        return
    longs = [t for t in trades_list if t["dir"] == 1]
    shorts = [t for t in trades_list if t["dir"] == -1]
    if longs:
        lr = np.array([t["r"] for t in longs])
        print(f"    Longs:  {len(longs)}t, R={lr.sum():+.1f}, avgR={lr.mean():+.4f}, WR={100*(lr>0).mean():.1f}%")
    if shorts:
        sr = np.array([t["r"] for t in shorts])
        print(f"    Shorts: {len(shorts)}t, R={sr.sum():+.1f}, avgR={sr.mean():+.4f}, WR={100*(sr>0).mean():.1f}%")
    # Exit reason breakdown for longs
    if longs:
        reasons = {}
        for t in longs:
            r = t["reason"]
            if r not in reasons:
                reasons[r] = {"count": 0, "r_sum": 0.0}
            reasons[r]["count"] += 1
            reasons[r]["r_sum"] += t["r"]
        print(f"    Long exit reasons:")
        for reason in sorted(reasons.keys()):
            rd = reasons[reason]
            print(f"      {reason:15s}: {rd['count']:4d} trades, R={rd['r_sum']:+7.1f}")


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 120)
    print("PURE LIQUIDITY-LEVEL TP TARGETS — NO FIXED MULTIPLIERS")
    print("50/50/0 split | EOD close 15:55 ET | Shorts: dual_mode scalp 0.625R")
    print("=" * 120)

    d = load_all()
    d_extra = prepare_liquidity_data(d)

    all_results = {}

    # ==================================================================
    # V7: Config G baseline (ny_tp_mult=1.5) — reference
    # ==================================================================
    print("\n" + "=" * 120)
    print("V7: CONFIG G BASELINE (IRL * 1.5 + liquidity ladder, 50/50/0)")
    print("=" * 120)
    t0 = _time.perf_counter()
    v7_trades, v7_diag = run_backtest_pure_liquidity(
        d, d_extra, tp_strategy="v7", v7_ny_tp_mult=1.5,
        tp1_trim_pct=0.50, be_after_tp1=True, be_after_tp2=True,
    )
    v7_m = compute_metrics(v7_trades)
    print_metrics("V7 Config G baseline (1.5x)", v7_m)
    print_diagnostics("V7", v7_diag)
    print_long_short_breakdown(v7_trades, "V7")
    print(f"  ({_time.perf_counter() - t0:.1f}s)")
    all_results["V7 Config G (1.5x)"] = (v7_m, v7_trades, v7_diag)

    # ==================================================================
    # V1: Raw IRL (no multiplier) + next liquidity level
    # ==================================================================
    print("\n" + "=" * 120)
    print("V1: RAW IRL (no multiplier) + next liquidity level")
    print("=" * 120)
    t0 = _time.perf_counter()
    v1_trades, v1_diag = run_backtest_pure_liquidity(
        d, d_extra, tp_strategy="v1",
        tp1_trim_pct=0.50, be_after_tp1=True, be_after_tp2=True,
    )
    v1_m = compute_metrics(v1_trades)
    print_metrics("V1 Raw IRL + liquidity", v1_m)
    print_diagnostics("V1", v1_diag)
    print_long_short_breakdown(v1_trades, "V1")
    delta_r = v1_m["R"] - v7_m["R"]
    delta_ppdd = v1_m["PPDD"] - v7_m["PPDD"]
    print(f"    vs V7: dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}  ({_time.perf_counter() - t0:.1f}s)")
    all_results["V1 Raw IRL"] = (v1_m, v1_trades, v1_diag)

    # ==================================================================
    # V2: Raw swing-to-swing
    # ==================================================================
    print("\n" + "=" * 120)
    print("V2: RAW SWING-TO-SWING (1st swing TP1, 2nd swing TP2)")
    print("=" * 120)
    t0 = _time.perf_counter()
    v2_trades, v2_diag = run_backtest_pure_liquidity(
        d, d_extra, tp_strategy="v2",
        tp1_trim_pct=0.50, be_after_tp1=True, be_after_tp2=True,
    )
    v2_m = compute_metrics(v2_trades)
    print_metrics("V2 Swing-to-swing", v2_m)
    print_diagnostics("V2", v2_diag)
    print_long_short_breakdown(v2_trades, "V2")
    delta_r = v2_m["R"] - v7_m["R"]
    delta_ppdd = v2_m["PPDD"] - v7_m["PPDD"]
    print(f"    vs V7: dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}  ({_time.perf_counter() - t0:.1f}s)")
    all_results["V2 Swing-to-swing"] = (v2_m, v2_trades, v2_diag)

    # ==================================================================
    # V3: Session level as primary TP1
    # ==================================================================
    print("\n" + "=" * 120)
    print("V3: SESSION LEVEL PRIMARY (nearest session level TP1, next TP2)")
    print("=" * 120)
    t0 = _time.perf_counter()
    v3_trades, v3_diag = run_backtest_pure_liquidity(
        d, d_extra, tp_strategy="v3",
        tp1_trim_pct=0.50, be_after_tp1=True, be_after_tp2=True,
    )
    v3_m = compute_metrics(v3_trades)
    print_metrics("V3 Session primary", v3_m)
    print_diagnostics("V3", v3_diag)
    print_long_short_breakdown(v3_trades, "V3")
    delta_r = v3_m["R"] - v7_m["R"]
    delta_ppdd = v3_m["PPDD"] - v7_m["PPDD"]
    print(f"    vs V7: dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}  ({_time.perf_counter() - t0:.1f}s)")
    all_results["V3 Session primary"] = (v3_m, v3_trades, v3_diag)

    # ==================================================================
    # V4: Hybrid — swing first, session second
    # ==================================================================
    print("\n" + "=" * 120)
    print("V4: HYBRID (swing TP1, session level TP2)")
    print("=" * 120)
    t0 = _time.perf_counter()
    v4_trades, v4_diag = run_backtest_pure_liquidity(
        d, d_extra, tp_strategy="v4",
        tp1_trim_pct=0.50, be_after_tp1=True, be_after_tp2=True,
    )
    v4_m = compute_metrics(v4_trades)
    print_metrics("V4 Hybrid swing+session", v4_m)
    print_diagnostics("V4", v4_diag)
    print_long_short_breakdown(v4_trades, "V4")
    delta_r = v4_m["R"] - v7_m["R"]
    delta_ppdd = v4_m["PPDD"] - v7_m["PPDD"]
    print(f"    vs V7: dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}  ({_time.perf_counter() - t0:.1f}s)")
    all_results["V4 Hybrid"] = (v4_m, v4_trades, v4_diag)

    # ==================================================================
    # V5: Multiplier sweep (1.0, 1.25, 1.5, 2.0) for comparison
    # ==================================================================
    print("\n" + "=" * 120)
    print("V5: MULTIPLIER SWEEP on raw IRL (TP1 = IRL * mult, TP2 = next liquidity)")
    print("=" * 120)
    v5_results = {}
    for mult in [1.0, 1.25, 1.5, 2.0]:
        t0 = _time.perf_counter()
        trades, diag = run_backtest_pure_liquidity(
            d, d_extra, tp_strategy="v5", v5_mult=mult,
            tp1_trim_pct=0.50, be_after_tp1=True, be_after_tp2=True,
        )
        m = compute_metrics(trades)
        marker = " <-- Config G equiv" if mult == 1.5 else ""
        marker = " <-- same as V1" if mult == 1.0 else marker
        print_metrics(f"V5 mult={mult}{marker}", m)
        print_diagnostics(f"V5 m={mult}", diag)
        delta_r = m["R"] - v7_m["R"]
        delta_ppdd = m["PPDD"] - v7_m["PPDD"]
        print(f"    vs V7: dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}  ({_time.perf_counter() - t0:.1f}s)")
        v5_results[mult] = (m, trades, diag)
        all_results[f"V5 mult={mult}"] = (m, trades, diag)

    # ==================================================================
    # V6: Distance-calibrated (1.0 ATR / 2.0 ATR)
    # ==================================================================
    print("\n" + "=" * 120)
    print("V6: DISTANCE-CALIBRATED (closest level at 1.0 ATR / 2.0 ATR)")
    print("=" * 120)
    t0 = _time.perf_counter()
    v6_trades, v6_diag = run_backtest_pure_liquidity(
        d, d_extra, tp_strategy="v6",
        tp1_trim_pct=0.50, be_after_tp1=True, be_after_tp2=True,
    )
    v6_m = compute_metrics(v6_trades)
    print_metrics("V6 Distance-calibrated", v6_m)
    print_diagnostics("V6", v6_diag)
    print_long_short_breakdown(v6_trades, "V6")
    delta_r = v6_m["R"] - v7_m["R"]
    delta_ppdd = v6_m["PPDD"] - v7_m["PPDD"]
    print(f"    vs V7: dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}  ({_time.perf_counter() - t0:.1f}s)")
    all_results["V6 Distance-cal"] = (v6_m, v6_trades, v6_diag)

    # ==================================================================
    # SUMMARY TABLE
    # ==================================================================
    print("\n" + "=" * 120)
    print("SUMMARY TABLE — ALL VARIANTS")
    print("=" * 120)
    print(f"  {'Variant':35s} | {'Trades':>6} | {'R':>9} | {'PPDD':>7} | {'PF':>6} | {'WR':>6} | {'MaxDD':>6} | {'avgR':>8} | {'TP1pts':>7} | {'TP1atr':>6} | {'TP2pts':>7} | {'TP2atr':>6}")
    print(f"  {'-'*35}-+-{'-'*6}-+-{'-'*9}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}")

    # Sort by PPDD descending
    sorted_results = sorted(all_results.items(), key=lambda x: x[1][0]["PPDD"], reverse=True)
    for label, (m, trades, diag) in sorted_results:
        print(f"  {label:35s} | {m['trades']:6d} | {m['R']:+9.1f} | {m['PPDD']:7.2f} | {m['PF']:6.2f} | {m['WR']:5.1f}% | {m['MaxDD']:6.1f} | {m['avgR']:+.4f} | {diag['avg_tp1_pts']:7.1f} | {diag['avg_tp1_atr']:6.2f} | {diag['avg_tp2_pts']:7.1f} | {diag['avg_tp2_atr']:6.2f}")

    # ==================================================================
    # TP HIT RATE COMPARISON
    # ==================================================================
    print("\n" + "=" * 120)
    print("TP HIT RATE & EOD CLOSE COMPARISON")
    print("=" * 120)
    print(f"  {'Variant':35s} | {'Longs':>6} | {'TP1 Hit':>8} | {'TP2 Hit':>8} | {'EOD Close':>10} | {'Stop/Other':>10}")
    print(f"  {'-'*35}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")
    for label, (m, trades, diag) in sorted_results:
        lt = diag["long_trades"]
        tp1h = diag["tp1_hit"]
        tp2h = diag["tp2_hit"]
        eod = diag["eod_close"]
        other = lt - tp1h  # Trades that never hit TP1 (stopped or EOD)
        if lt > 0:
            print(f"  {label:35s} | {lt:6d} | {tp1h:4d} ({100*tp1h/lt:4.1f}%) | {tp2h:4d} ({100*tp2h/lt:4.1f}%) | {eod:5d} ({100*eod/lt:5.1f}%) | {other:5d} ({100*other/lt:5.1f}%)")

    # ==================================================================
    # WALK-FORWARD for TOP 3 by PPDD
    # ==================================================================
    print("\n" + "=" * 120)
    print("WALK-FORWARD: TOP 3 VARIANTS vs V7 BASELINE")
    print("=" * 120)

    # Get top 3 (excluding V7 itself)
    non_baseline = [(lab, (m, tr, di)) for lab, (m, tr, di) in sorted_results if "V7" not in lab]
    top3 = non_baseline[:3]

    # Print header
    hdr_parts = [f"{'Year':>6}"]
    hdr_parts.append(f"{'--- V7 BASELINE ---':^32s}")
    for label, _ in top3:
        short_label = label[:25]
        hdr_parts.append(f"{'--- ' + short_label + ' ---':^32s}")
    print(" | ".join(hdr_parts))

    sub_parts = [f"{'':>6}"]
    for _ in range(1 + len(top3)):
        sub_parts.append(f"{'n':>4} {'R':>7} {'PF':>5} {'PPDD':>6}")
    print(" | ".join(sub_parts))
    print("-" * (8 + 34 * (1 + len(top3))))

    wf_baseline = walk_forward_metrics(v7_trades)
    wf_top = []
    for label, (m, trades, diag) in top3:
        wf_top.append(walk_forward_metrics(trades))

    bl_dict = {w["year"]: w for w in wf_baseline}
    top_dicts = [{w["year"]: w for w in wf} for wf in wf_top]

    all_years = sorted(set(
        list(bl_dict.keys()) +
        [y for td in top_dicts for y in td.keys()]
    ))

    wins_count = [0] * len(top3)
    for y in all_years:
        bl = bl_dict.get(y, {"n": 0, "R": 0, "WR": 0, "PF": 0, "PPDD": 0})
        parts = [f"  {y:>4}"]
        parts.append(f"{bl['n']:4d} {bl['R']:+7.1f} {bl['PF']:5.2f} {bl['PPDD']:+6.2f}")
        for vi, (_, _) in enumerate(top3):
            td = top_dicts[vi]
            tv = td.get(y, {"n": 0, "R": 0, "WR": 0, "PF": 0, "PPDD": 0})
            marker = " *" if tv["R"] > bl["R"] else ""
            if tv["R"] > bl["R"]:
                wins_count[vi] += 1
            parts.append(f"{tv['n']:4d} {tv['R']:+7.1f} {tv['PF']:5.2f} {tv['PPDD']:+6.2f}{marker}")
        print(" | ".join(parts))

    print("-" * (8 + 34 * (1 + len(top3))))
    for vi, (label, _) in enumerate(top3):
        print(f"  {label}: wins {wins_count[vi]}/{len(all_years)} years vs V7 baseline")

    # ==================================================================
    # ANALYSIS SUMMARY
    # ==================================================================
    print("\n" + "=" * 120)
    print("ANALYSIS SUMMARY")
    print("=" * 120)

    # Best overall
    best_label = sorted_results[0][0]
    best_m = sorted_results[0][1][0]
    best_diag = sorted_results[0][1][2]
    print(f"\n  BEST BY PPDD: {best_label}")
    print(f"    R={best_m['R']:+.1f} | PPDD={best_m['PPDD']:.2f} | PF={best_m['PF']:.2f}")

    # Best by R
    best_r_label = max(sorted_results, key=lambda x: x[1][0]["R"])[0]
    best_r_m = max(sorted_results, key=lambda x: x[1][0]["R"])[1][0]
    print(f"\n  BEST BY TOTAL R: {best_r_label}")
    print(f"    R={best_r_m['R']:+.1f} | PPDD={best_r_m['PPDD']:.2f} | PF={best_r_m['PF']:.2f}")

    # Raw vs multiplier comparison
    print(f"\n  RAW IRL (V1) vs MULTIPLIER SWEEP:")
    v1_info = all_results.get("V1 Raw IRL", ({"R": 0, "PPDD": 0, "PF": 0}, [], {}))
    print(f"    V1 (raw, mult=1.0): R={v1_info[0]['R']:+.1f}, PPDD={v1_info[0]['PPDD']:.2f}, PF={v1_info[0]['PF']:.2f}")
    for mult in [1.25, 1.5, 2.0]:
        key = f"V5 mult={mult}"
        if key in all_results:
            mi = all_results[key][0]
            print(f"    V5 (mult={mult}):     R={mi['R']:+.1f}, PPDD={mi['PPDD']:.2f}, PF={mi['PF']:.2f}")

    print(f"\n  CONCLUSION: Is raw liquidity better than fixed multipliers?")
    v1_ppdd = all_results.get("V1 Raw IRL", ({"PPDD": 0}, [], {}))[0]["PPDD"]
    v7_ppdd = v7_m["PPDD"]
    if v1_ppdd > v7_ppdd:
        print(f"    YES — Raw IRL (PPDD={v1_ppdd:.2f}) beats Config G 1.5x (PPDD={v7_ppdd:.2f})")
    else:
        print(f"    NO  — Config G 1.5x (PPDD={v7_ppdd:.2f}) beats Raw IRL (PPDD={v1_ppdd:.2f})")
        # Check if any pure-liquidity variant beats V7
        pure_variants = [(lab, m_info) for lab, m_info in sorted_results
                         if "V7" not in lab and "V5" not in lab]
        best_pure = max(pure_variants, key=lambda x: x[1][0]["PPDD"]) if pure_variants else None
        if best_pure and best_pure[1][0]["PPDD"] > v7_ppdd:
            print(f"    BUT: {best_pure[0]} (PPDD={best_pure[1][0]['PPDD']:.2f}) is better")

    print("\n" + "=" * 120)
    print("DONE")
    print("=" * 120)


if __name__ == "__main__":
    main()
