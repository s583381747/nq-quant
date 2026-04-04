"""
experiments/a2a_exit_classifier.py — Adaptive Exit Strategy Classifier

Instead of ONE exit strategy for all signals, classify each signal into the
optimal exit template. For each traded signal, simulate 4 exit strategies
INDEPENDENTLY (no path dependency), label with the best, train XGBoost to
predict, and compare adaptive vs fixed strategies.

Exit Strategies:
  A: Scalp        — 100% trim at TP1 (IRL × 2.0), no runner
  B: Standard     — 50% trim at TP1 (IRL × 2.0), trail 2nd swing, BE after trim
  C: Runner       — 25% trim at TP1 (IRL × 3.0), trail 3rd swing, BE after trim
  D: Multi-TP     — 25/25/50 at TP1/TP2 liquidity ladder, late BE, trail 3rd swing

Usage: python experiments/a2a_exit_classifier.py
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)

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
# Step 1: Per-signal independent outcome simulation
# ======================================================================

def _simulate_single_trade(
    direction: int,
    entry_price: float,
    stop_price: float,
    tp1_price: float,
    tp2_price: float,  # only used for multi-TP (strategy D)
    trim_pct: float,
    be_after_trim: bool,
    trail_nth_swing: int,
    max_hold: int,
    # bar arrays starting from entry bar
    h_arr: np.ndarray,
    l_arr: np.ndarray,
    c_arr: np.ndarray,
    o_arr: np.ndarray,
    swing_high_mask: np.ndarray,
    swing_low_mask: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    atr_arr: np.ndarray,
    # offset: absolute index of entry bar (for swing lookback)
    abs_entry_idx: int,
    full_swing_high_mask: np.ndarray,
    full_swing_low_mask: np.ndarray,
    full_high_arr: np.ndarray,
    full_low_arr: np.ndarray,
    # multi-tp specific
    is_multi_tp: bool = False,
    tp1_trim_pct: float = 0.25,
    tp2_trim_pct: float = 0.25,
    be_after_tp2: bool = True,
    slippage_points: float = 0.25,
    point_value: float = 2.0,
    commission_per_side: float = 0.62,
) -> float:
    """Simulate a single trade with given exit strategy. Returns R-multiple."""
    n_bars = len(h_arr)
    if n_bars == 0:
        return 0.0

    stop_dist = abs(entry_price - stop_price)
    if stop_dist < 0.5:
        return 0.0

    contracts = 10  # arbitrary for R calculation, cancels out
    remaining = contracts

    trimmed = False
    trim_stage = 0  # for multi-TP: 0=untrimmed, 1=TP1, 2=TP2
    be_stop = 0.0
    trail_stop = 0.0
    tp1_pnl_pts = 0.0
    tp2_pnl_pts = 0.0
    trim1_c = 0
    trim2_c = 0

    for j in range(min(n_bars, max_hold)):
        abs_idx = abs_entry_idx + 1 + j  # absolute index in full arrays
        bar_h = h_arr[j]
        bar_l = l_arr[j]

        if is_multi_tp:
            # --- Multi-TP exit logic ---
            if direction == 1:
                # Effective stop
                if trim_stage >= 2:
                    eff_stop = trail_stop if trail_stop > 0 else stop_price
                    if be_stop > 0:
                        eff_stop = max(eff_stop, be_stop)
                elif trim_stage == 1 and not be_after_tp2:
                    # be_after_tp1 = not be_after_tp2 for our config
                    eff_stop = stop_price
                    if be_stop > 0:
                        eff_stop = max(eff_stop, be_stop)
                else:
                    eff_stop = stop_price

                # Stop hit
                if bar_l <= eff_stop:
                    exit_price = eff_stop - slippage_points
                    pnl = 0.0
                    if trim_stage >= 1:
                        pnl += tp1_pnl_pts * point_value * trim1_c
                    if trim_stage >= 2:
                        pnl += tp2_pnl_pts * point_value * trim2_c
                    pnl += (exit_price - entry_price) * point_value * remaining
                    pnl -= commission_per_side * 2 * contracts
                    return pnl / (stop_dist * point_value * contracts)

                # TP1
                if trim_stage == 0 and bar_h >= tp1_price:
                    tc1 = max(1, int(contracts * tp1_trim_pct))
                    trim1_c = tc1
                    remaining = contracts - tc1
                    tp1_pnl_pts = tp1_price - entry_price
                    trim_stage = 1
                    if remaining <= 0:
                        pnl = tp1_pnl_pts * point_value * contracts
                        pnl -= commission_per_side * 2 * contracts
                        return pnl / (stop_dist * point_value * contracts)

                # TP2
                if trim_stage == 1 and bar_h >= tp2_price:
                    tc2 = max(1, int(contracts * tp2_trim_pct))
                    tc2 = min(tc2, remaining)
                    trim2_c = tc2
                    remaining -= tc2
                    tp2_pnl_pts = tp2_price - entry_price
                    trim_stage = 2
                    if be_after_tp2:
                        be_stop = entry_price
                    if remaining > 0:
                        trail_stop = _find_nth_swing(
                            full_swing_low_mask, full_low_arr, abs_idx, trail_nth_swing, 1)
                        if np.isnan(trail_stop) or trail_stop <= 0:
                            trail_stop = entry_price
                    if remaining <= 0:
                        pnl = tp1_pnl_pts * point_value * trim1_c + tp2_pnl_pts * point_value * trim2_c
                        pnl -= commission_per_side * 2 * contracts
                        return pnl / (stop_dist * point_value * contracts)

                # Update trail
                if trim_stage >= 2 and remaining > 0:
                    nt = _find_nth_swing(
                        full_swing_low_mask, full_low_arr, abs_idx, trail_nth_swing, 1)
                    if not np.isnan(nt) and nt > trail_stop:
                        trail_stop = nt

            else:  # SHORT multi-TP
                if trim_stage >= 2:
                    eff_stop = trail_stop if trail_stop > 0 else stop_price
                    if be_stop > 0:
                        eff_stop = min(eff_stop, be_stop)
                elif trim_stage == 1 and not be_after_tp2:
                    eff_stop = stop_price
                    if be_stop > 0:
                        eff_stop = min(eff_stop, be_stop)
                else:
                    eff_stop = stop_price

                if bar_h >= eff_stop:
                    exit_price = eff_stop + slippage_points
                    pnl = 0.0
                    if trim_stage >= 1:
                        pnl += tp1_pnl_pts * point_value * trim1_c
                    if trim_stage >= 2:
                        pnl += tp2_pnl_pts * point_value * trim2_c
                    pnl += (entry_price - exit_price) * point_value * remaining
                    pnl -= commission_per_side * 2 * contracts
                    return pnl / (stop_dist * point_value * contracts)

                if trim_stage == 0 and bar_l <= tp1_price:
                    tc1 = max(1, int(contracts * tp1_trim_pct))
                    trim1_c = tc1
                    remaining = contracts - tc1
                    tp1_pnl_pts = entry_price - tp1_price
                    trim_stage = 1
                    if remaining <= 0:
                        pnl = tp1_pnl_pts * point_value * contracts
                        pnl -= commission_per_side * 2 * contracts
                        return pnl / (stop_dist * point_value * contracts)

                if trim_stage == 1 and bar_l <= tp2_price:
                    tc2 = max(1, int(contracts * tp2_trim_pct))
                    tc2 = min(tc2, remaining)
                    trim2_c = tc2
                    remaining -= tc2
                    tp2_pnl_pts = entry_price - tp2_price
                    trim_stage = 2
                    if be_after_tp2:
                        be_stop = entry_price
                    if remaining > 0:
                        trail_stop = _find_nth_swing(
                            full_swing_high_mask, full_high_arr, abs_idx, trail_nth_swing, -1)
                        if np.isnan(trail_stop) or trail_stop <= 0:
                            trail_stop = entry_price
                    if remaining <= 0:
                        pnl = tp1_pnl_pts * point_value * trim1_c + tp2_pnl_pts * point_value * trim2_c
                        pnl -= commission_per_side * 2 * contracts
                        return pnl / (stop_dist * point_value * contracts)

                if trim_stage >= 2 and remaining > 0:
                    nt = _find_nth_swing(
                        full_swing_high_mask, full_high_arr, abs_idx, trail_nth_swing, -1)
                    if not np.isnan(nt) and nt < trail_stop:
                        trail_stop = nt

        else:
            # --- Single-TP exit logic ---
            if direction == 1:
                eff_stop = trail_stop if trimmed and trail_stop > 0 else stop_price
                if trimmed and be_after_trim and be_stop > 0:
                    eff_stop = max(eff_stop, be_stop)
                if bar_l <= eff_stop:
                    exit_price = eff_stop - slippage_points
                    if trimmed:
                        trim_c = contracts - remaining
                        pnl = (tp1_price - entry_price) * point_value * trim_c
                        pnl += (exit_price - entry_price) * point_value * remaining
                        pnl -= commission_per_side * 2 * contracts
                    else:
                        pnl = (exit_price - entry_price) * point_value * contracts
                        pnl -= commission_per_side * 2 * contracts
                    return pnl / (stop_dist * point_value * contracts)

                if not trimmed and bar_h >= tp1_price:
                    tc = max(1, int(contracts * trim_pct))
                    remaining = contracts - tc
                    trimmed = True
                    be_stop = entry_price
                    if remaining <= 0:
                        pnl = (tp1_price - entry_price) * point_value * contracts
                        pnl -= commission_per_side * 2 * contracts
                        return pnl / (stop_dist * point_value * contracts)
                    trail_stop = _find_nth_swing(
                        full_swing_low_mask, full_low_arr, abs_idx, trail_nth_swing, 1)
                    if np.isnan(trail_stop) or trail_stop <= 0:
                        trail_stop = be_stop

                if trimmed and remaining > 0:
                    nt = _find_nth_swing(
                        full_swing_low_mask, full_low_arr, abs_idx, trail_nth_swing, 1)
                    if not np.isnan(nt) and nt > trail_stop:
                        trail_stop = nt

            else:  # SHORT
                eff_stop = trail_stop if trimmed and trail_stop > 0 else stop_price
                if trimmed and be_after_trim and be_stop > 0:
                    eff_stop = min(eff_stop, be_stop)
                if bar_h >= eff_stop:
                    exit_price = eff_stop + slippage_points
                    if trimmed:
                        trim_c = contracts - remaining
                        pnl = (entry_price - tp1_price) * point_value * trim_c
                        pnl += (entry_price - exit_price) * point_value * remaining
                        pnl -= commission_per_side * 2 * contracts
                    else:
                        pnl = (entry_price - exit_price) * point_value * contracts
                        pnl -= commission_per_side * 2 * contracts
                    return pnl / (stop_dist * point_value * contracts)

                if not trimmed and bar_l <= tp1_price:
                    tc = max(1, int(contracts * trim_pct))
                    remaining = contracts - tc
                    trimmed = True
                    be_stop = entry_price
                    if remaining <= 0:
                        pnl = (entry_price - tp1_price) * point_value * contracts
                        pnl -= commission_per_side * 2 * contracts
                        return pnl / (stop_dist * point_value * contracts)
                    trail_stop = _find_nth_swing(
                        full_swing_high_mask, full_high_arr, abs_idx, trail_nth_swing, -1)
                    if np.isnan(trail_stop) or trail_stop <= 0:
                        trail_stop = be_stop

                if trimmed and remaining > 0:
                    nt = _find_nth_swing(
                        full_swing_high_mask, full_high_arr, abs_idx, trail_nth_swing, -1)
                    if not np.isnan(nt) and nt < trail_stop:
                        trail_stop = nt

    # Max hold reached — exit at last bar close
    last_j = min(n_bars, max_hold) - 1
    if last_j < 0:
        return 0.0
    exit_price = c_arr[last_j]

    if is_multi_tp:
        pnl = 0.0
        if trim_stage >= 1:
            pnl += tp1_pnl_pts * point_value * trim1_c
        if trim_stage >= 2:
            pnl += tp2_pnl_pts * point_value * trim2_c
        if direction == 1:
            pnl += (exit_price - entry_price) * point_value * remaining
        else:
            pnl += (entry_price - exit_price) * point_value * remaining
        pnl -= commission_per_side * 2 * contracts
    elif trimmed:
        trim_c = contracts - remaining
        if direction == 1:
            pnl = (tp1_price - entry_price) * point_value * trim_c
            pnl += (exit_price - entry_price) * point_value * remaining
        else:
            pnl = (entry_price - tp1_price) * point_value * trim_c
            pnl += (entry_price - exit_price) * point_value * remaining
        pnl -= commission_per_side * 2 * contracts
    else:
        if direction == 1:
            pnl = (exit_price - entry_price) * point_value * contracts
        else:
            pnl = (entry_price - exit_price) * point_value * contracts
        pnl -= commission_per_side * 2 * contracts

    return pnl / (stop_dist * point_value * contracts)


def generate_per_signal_outcomes(d: dict, d_extra: dict) -> pd.DataFrame:
    """For each Config E filtered signal, simulate 4 exit strategies independently."""
    import yaml
    t0 = _time.perf_counter()

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
    bias_conf_arr = d["bias_conf_arr"]
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

    smt_cfg = params.get("smt", {})
    bias_relax = params.get("bias_relaxation", {})
    pa_threshold = params.get("pa_quality", {}).get("alt_dir_threshold", 1.0)
    session_rules = params.get("session_rules", {})
    session_regime = params.get("session_regime", {})
    dir_mgmt = params.get("direction_mgmt", {})
    mss_mgmt = params.get("mss_management", {})
    mss_mgmt_enabled = mss_mgmt.get("enabled", False)
    dual_mode = params.get("dual_mode", {})
    grading_params = params["grading"]
    c_skip = grading_params["c_skip"]
    bt_params = params["backtest"]
    slippage_ticks = bt_params["slippage_normal_ticks"]
    slippage_points = slippage_ticks * 0.25
    point_value = params["position"]["point_value"]
    commission_per_side = bt_params["commission_per_side_micro"]

    full_high = nq["high"].values
    full_low = nq["low"].values

    # Config E filter parameters
    sq_long = 0.68
    sq_short = 0.80
    min_stop_atr = 1.7
    block_pm_shorts = True

    # Session levels for liquidity
    sess_overnight_high = d_extra["sess_overnight_high"]
    sess_overnight_low = d_extra["sess_overnight_low"]

    MAX_HOLD = 100
    records = []

    for i in range(n):
        if news_blackout_arr is not None and news_blackout_arr[i]:
            continue
        et_frac = et_frac_arr[i]
        if 9.5 <= et_frac <= 10.0:
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
        tp_raw = irl_target_arr[i]
        if np.isnan(entry_p) or np.isnan(stop) or np.isnan(tp_raw):
            continue
        if direction == 1 and (stop >= entry_p or tp_raw <= entry_p):
            continue
        if direction == -1 and (stop <= entry_p or tp_raw >= entry_p):
            continue

        # Block PM shorts
        if block_pm_shorts and et_frac >= 14.0 and direction == -1:
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
        if not mss_bypass:
            if not is_ny:
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

        if i + 1 >= n:
            continue

        # Grade
        ba = 1.0 if (direction == np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0) else 0.0
        grade = _compute_grade_fast(ba, regime_arr[i])
        if grade == "C" and c_skip:
            continue

        # Slippage + entry
        actual_entry = entry_p + slippage_points if direction == 1 else entry_p - slippage_points
        stop_dist = abs(actual_entry - stop)
        if stop_dist < 1.0:
            continue

        # TP distance from raw IRL
        tp_distance_raw = abs(tp_raw - actual_entry)

        # --- Compute TPs for each strategy ---
        # Strategy A: Scalp — TP1 = IRL × 2.0, 100% exit
        tp_a = actual_entry + tp_distance_raw * 2.0 if direction == 1 else actual_entry - tp_distance_raw * 2.0

        # Strategy B: Standard — TP1 = IRL × 2.0, 50% trim, trail 2nd swing, BE
        tp_b = tp_a  # same TP level

        # Strategy C: Runner — TP1 = IRL × 3.0, 25% trim, trail 3rd swing, BE
        tp_c = actual_entry + tp_distance_raw * 3.0 if direction == 1 else actual_entry - tp_distance_raw * 3.0

        # Strategy D: Multi-TP with liquidity ladder
        # TP1 uses ny_tp_mult = 3.0 (Config E)
        tp_d_base = actual_entry + tp_distance_raw * 3.0 if direction == 1 else actual_entry - tp_distance_raw * 3.0
        # Apply MSS long TP mult if applicable
        is_mss_signal = str(sig_type[i]) == "mss"
        if mss_mgmt_enabled and is_mss_signal and direction == 1:
            tp_d_base = actual_entry + tp_distance_raw * 2.5 if direction == 1 else actual_entry - tp_distance_raw * 2.5

        if direction == 1:
            tp_d1, tp_d2 = build_liquidity_ladder_long(
                actual_entry, stop, tp_d_base, i, d_extra)
        else:
            tp_d1, tp_d2 = build_liquidity_ladder_short(
                actual_entry, stop, tp_d_base, i, d_extra)

        # --- Forward bar arrays for simulation ---
        start_bar = i + 1
        end_bar = min(i + 1 + MAX_HOLD, n)
        if start_bar >= end_bar:
            continue
        fwd_h = h[start_bar:end_bar]
        fwd_l = l[start_bar:end_bar]
        fwd_c = c_arr[start_bar:end_bar]
        fwd_o = o[start_bar:end_bar]

        sim_kwargs = dict(
            h_arr=fwd_h, l_arr=fwd_l, c_arr=fwd_c, o_arr=fwd_o,
            swing_high_mask=swing_high_mask[start_bar:end_bar],
            swing_low_mask=swing_low_mask[start_bar:end_bar],
            high_prices=full_high[start_bar:end_bar],
            low_prices=full_low[start_bar:end_bar],
            atr_arr=atr_arr[start_bar:end_bar],
            abs_entry_idx=i,
            full_swing_high_mask=swing_high_mask,
            full_swing_low_mask=swing_low_mask,
            full_high_arr=full_high,
            full_low_arr=full_low,
            slippage_points=slippage_points,
            point_value=point_value,
            commission_per_side=commission_per_side,
        )

        # Strategy A: Scalp — 100% trim at TP1, no runner
        r_a = _simulate_single_trade(
            direction=direction, entry_price=actual_entry, stop_price=stop,
            tp1_price=tp_a, tp2_price=tp_a,
            trim_pct=1.0, be_after_trim=True,
            trail_nth_swing=2, max_hold=MAX_HOLD,
            is_multi_tp=False, **sim_kwargs,
        )

        # Strategy B: Standard — 50% trim, trail 2nd swing, BE
        r_b = _simulate_single_trade(
            direction=direction, entry_price=actual_entry, stop_price=stop,
            tp1_price=tp_b, tp2_price=tp_b,
            trim_pct=0.50, be_after_trim=True,
            trail_nth_swing=2, max_hold=MAX_HOLD,
            is_multi_tp=False, **sim_kwargs,
        )

        # Strategy C: Runner — 25% trim at higher TP, trail 3rd swing, BE
        r_c = _simulate_single_trade(
            direction=direction, entry_price=actual_entry, stop_price=stop,
            tp1_price=tp_c, tp2_price=tp_c,
            trim_pct=0.25, be_after_trim=True,
            trail_nth_swing=3, max_hold=MAX_HOLD,
            is_multi_tp=False, **sim_kwargs,
        )

        # Strategy D: Multi-TP (Config E)
        r_d = _simulate_single_trade(
            direction=direction, entry_price=actual_entry, stop_price=stop,
            tp1_price=tp_d1, tp2_price=tp_d2,
            trim_pct=0.25, be_after_trim=False,
            trail_nth_swing=3, max_hold=MAX_HOLD,
            is_multi_tp=True,
            tp1_trim_pct=0.25, tp2_trim_pct=0.25, be_after_tp2=True,
            **sim_kwargs,
        )

        # --- Features ---
        atr_val = atr_arr[i] if not np.isnan(atr_arr[i]) else 10.0
        flu_val = fluency_arr[i] if not np.isnan(fluency_arr[i]) else 0.5
        sq_val = signal_quality[i] if not np.isnan(signal_quality[i]) else 0.5
        bias_aligned = 1 if (direction == np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0) else 0
        bias_conf = bias_conf_arr[i] if not np.isnan(bias_conf_arr[i]) else 0.0
        regime_val = regime_arr[i] if not np.isnan(regime_arr[i]) else 0.0
        et_hour = int(et_frac)
        dow_val = dow_arr[i]
        stop_atr_val = stop_atr_ratio[i] if not np.isnan(stop_atr_ratio[i]) else 0.0
        trr = target_rr_arr[i] if not np.isnan(target_rr_arr[i]) else 0.0

        grade_num = 2 if grade == "A+" else (1 if grade == "B+" else 0)

        # ATR percentile (vs rolling 100-bar mean)
        atr_window = atr_arr[max(0, i - 100):i + 1]
        atr_mean = np.nanmean(atr_window) if len(atr_window) > 0 else atr_val
        atr_pctile = atr_val / atr_mean if atr_mean > 0 else 1.0

        # Entry vs overnight range
        on_h = sess_overnight_high[i]
        on_l = sess_overnight_low[i]
        if not np.isnan(on_h) and not np.isnan(on_l) and on_h > on_l:
            entry_vs_on = (actual_entry - on_l) / (on_h - on_l)
        else:
            entry_vs_on = 0.5

        # FVG size / ATR approximation (stop distance is related to FVG zone)
        fvg_size_atr = stop_dist / atr_val if atr_val > 0 else 0.0

        records.append({
            "idx": i,
            "entry_time": nq.index[i],
            "direction": direction,
            "signal_type": 0 if str(sig_type[i]) == "trend" else 1,
            "grade": grade_num,
            "fluency": flu_val,
            "atr_14": atr_val,
            "atr_percentile": atr_pctile,
            "signal_quality": sq_val,
            "bias_aligned": bias_aligned,
            "bias_confidence": bias_conf,
            "regime": regime_val,
            "hour_et": et_hour,
            "day_of_week": dow_val,
            "stop_dist_atr": stop_atr_val,
            "target_rr": trr,
            "fvg_size_atr": fvg_size_atr,
            "entry_vs_overnight_pct": entry_vs_on,
            "r_A": r_a,
            "r_B": r_b,
            "r_C": r_c,
            "r_D": r_d,
        })

    df = pd.DataFrame(records)
    elapsed = _time.perf_counter() - t0
    print(f"[STEP 1] Simulated {len(df)} signals × 4 strategies in {elapsed:.1f}s")
    return df


# ======================================================================
# Step 3: Label and analyze
# ======================================================================

def analyze_strategy_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Label each signal with best strategy, print analysis."""
    strategies = ["r_A", "r_B", "r_C", "r_D"]
    labels = ["Scalp", "Standard", "Runner", "Multi-TP"]

    r_matrix = df[strategies].values
    best_idx = np.argmax(r_matrix, axis=1)
    df["best_strategy"] = best_idx
    df["best_r"] = np.max(r_matrix, axis=1)
    df["worst_r"] = np.min(r_matrix, axis=1)
    df["r_advantage"] = df["best_r"] - df["worst_r"]

    print("\n" + "=" * 80)
    print("STEP 3: STRATEGY OUTCOME ANALYSIS")
    print("=" * 80)

    # How often each wins
    counts = Counter(best_idx)
    print("\n  Strategy wins:")
    for k in range(4):
        pct = counts.get(k, 0) / len(df) * 100
        avg_r = df.loc[df["best_strategy"] == k, "best_r"].mean() if counts.get(k, 0) > 0 else 0
        print(f"    {labels[k]:12s}: {counts.get(k, 0):5d} ({pct:5.1f}%) | avg best R = {avg_r:+.3f}")

    # Average R per strategy
    print("\n  Average R per strategy (all signals):")
    for j, (col, lbl) in enumerate(zip(strategies, labels)):
        print(f"    {lbl:12s}: {df[col].mean():+.4f} | sum = {df[col].sum():+.1f}R")

    # R advantage
    print(f"\n  Mean R advantage (best - worst): {df['r_advantage'].mean():.4f}")
    print(f"  Median R advantage: {df['r_advantage'].median():.4f}")
    pct_same = (df["r_advantage"] < 0.01).mean() * 100
    print(f"  Signals where all strategies ~same (<0.01R diff): {pct_same:.1f}%")

    # Is there clear separation?
    corr = df[strategies].corr()
    print("\n  Strategy R correlation matrix:")
    print(corr.to_string())

    return df


# ======================================================================
# Step 4: Train XGBoost classifier
# ======================================================================

def train_exit_classifier(df: pd.DataFrame) -> tuple:
    """Walk-forward XGBoost: train 2016-2021, test 2022-2025."""
    try:
        import xgboost as xgb
        from sklearn.metrics import classification_report, accuracy_score
    except ImportError:
        print("[STEP 4] ERROR: xgboost or sklearn not installed. Skipping ML.")
        return None, None, None

    feature_cols = [
        "signal_type", "direction", "grade", "fluency", "atr_14", "atr_percentile",
        "signal_quality", "bias_aligned", "bias_confidence", "regime",
        "hour_et", "day_of_week", "stop_dist_atr", "target_rr",
        "fvg_size_atr", "entry_vs_overnight_pct",
    ]

    df["year"] = pd.to_datetime(df["entry_time"]).dt.year

    train_mask = df["year"] <= 2021
    test_mask = df["year"] >= 2022

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, "best_strategy"].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, "best_strategy"].values

    print("\n" + "=" * 80)
    print("STEP 4: XGBoost EXIT STRATEGY CLASSIFIER")
    print("=" * 80)
    print(f"  Train: {train_mask.sum()} signals (<=2021)")
    print(f"  Test:  {test_mask.sum()} signals (>=2022)")

    if len(X_train) < 50 or len(X_test) < 20:
        print("  Not enough data for ML. Skipping.")
        return None, None, None

    # Train distribution
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)
    labels = ["Scalp", "Standard", "Runner", "Multi-TP"]
    print("\n  Train label distribution:")
    for k in range(4):
        print(f"    {labels[k]:12s}: {train_counts.get(k, 0):5d}")
    print("  Test label distribution:")
    for k in range(4):
        print(f"    {labels[k]:12s}: {test_counts.get(k, 0):5d}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softmax",
        num_class=4,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\n  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy:  {test_acc:.3f}")

    print("\n  Classification report (TEST set):")
    target_names = [f"{k}={labels[k]}" for k in range(4)]
    report = classification_report(y_test, y_pred_test, target_names=target_names,
                                   zero_division=0)
    print(report)

    # Feature importance
    imp = model.feature_importances_
    imp_sorted = sorted(zip(feature_cols, imp), key=lambda x: -x[1])
    print("  Feature importance (top 10):")
    for fname, fval in imp_sorted[:10]:
        print(f"    {fname:25s}: {fval:.4f}")

    return model, feature_cols, df


# ======================================================================
# Step 5: Simulate adaptive strategy
# ======================================================================

def simulate_adaptive_backtest(
    d: dict,
    d_extra: dict,
    df: pd.DataFrame,
    model,
    feature_cols: list,
) -> None:
    """Run full path-dependent backtest with ML-selected exit strategy per signal.
    Also compare vs always-X strategies and oracle."""
    print("\n" + "=" * 80)
    print("STEP 5: ADAPTIVE vs FIXED STRATEGY COMPARISON")
    print("=" * 80)

    strategies = ["r_A", "r_B", "r_C", "r_D"]
    labels_strat = ["Scalp", "Standard", "Runner", "Multi-TP"]

    df["year"] = pd.to_datetime(df["entry_time"]).dt.year

    # ---- Path-independent comparison (simpler, more informative) ----
    # Sum R for each fixed strategy and for adaptive/oracle
    print("\n  === PATH-INDEPENDENT COMPARISON (sum of per-signal R) ===")
    print("  (No 0-for-2 or daily limits — pure per-signal R)\n")

    test_mask = df["year"] >= 2022

    if model is not None:
        X_test = df.loc[test_mask, feature_cols].values
        ml_pred = model.predict(X_test)
    else:
        ml_pred = None

    for period_label, mask in [("FULL PERIOD", pd.Series(True, index=df.index)),
                               ("TRAIN (<=2021)", df["year"] <= 2021),
                               ("TEST (>=2022)", test_mask)]:
        sub = df[mask].copy()
        if len(sub) == 0:
            continue

        results = {}
        for j, (col, lbl) in enumerate(zip(strategies, labels_strat)):
            total_r = sub[col].sum()
            n_trades = len(sub)
            # Compute PPDD and PF for this strategy
            r_arr = sub[col].values
            cumr = np.cumsum(r_arr)
            peak = np.maximum.accumulate(cumr)
            dd = (peak - cumr).max()
            ppdd = total_r / dd if dd > 0 else 999.0
            wins = r_arr[r_arr > 0].sum()
            losses = abs(r_arr[r_arr < 0].sum())
            pf = wins / losses if losses > 0 else 999.0
            wr = (r_arr > 0).mean() * 100
            results[lbl] = {"R": total_r, "PPDD": ppdd, "PF": pf, "WR": wr,
                            "MaxDD": dd, "trades": n_trades}

        # Oracle
        oracle_r = sub["best_r"].values
        o_total = oracle_r.sum()
        o_cumr = np.cumsum(oracle_r)
        o_peak = np.maximum.accumulate(o_cumr)
        o_dd = (o_peak - o_cumr).max()
        o_ppdd = o_total / o_dd if o_dd > 0 else 999.0
        o_wins = oracle_r[oracle_r > 0].sum()
        o_losses = abs(oracle_r[oracle_r < 0].sum())
        o_pf = o_wins / o_losses if o_losses > 0 else 999.0
        o_wr = (oracle_r > 0).mean() * 100
        results["Oracle"] = {"R": o_total, "PPDD": o_ppdd, "PF": o_pf, "WR": o_wr,
                             "MaxDD": o_dd, "trades": len(sub)}

        # ML adaptive (only for test period if model exists)
        if model is not None and period_label == "TEST (>=2022)":
            r_matrix = sub[strategies].values
            ml_r = np.array([r_matrix[row, ml_pred[row]] for row in range(len(sub))])
            ml_total = ml_r.sum()
            ml_cumr = np.cumsum(ml_r)
            ml_peak = np.maximum.accumulate(ml_cumr)
            ml_dd = (ml_peak - ml_cumr).max()
            ml_ppdd = ml_total / ml_dd if ml_dd > 0 else 999.0
            ml_wins = ml_r[ml_r > 0].sum()
            ml_losses = abs(ml_r[ml_r < 0].sum())
            ml_pf = ml_wins / ml_losses if ml_losses > 0 else 999.0
            ml_wr = (ml_r > 0).mean() * 100
            results["ML Adaptive"] = {"R": ml_total, "PPDD": ml_ppdd, "PF": ml_pf,
                                       "WR": ml_wr, "MaxDD": ml_dd, "trades": len(sub)}

        print(f"  --- {period_label} ({len(sub)} signals) ---")
        for lbl, m in results.items():
            print(f"    {lbl:15s} | {m['trades']:4d}t | R={m['R']:+8.1f} | "
                  f"PPDD={m['PPDD']:6.2f} | PF={m['PF']:5.2f} | "
                  f"WR={m['WR']:5.1f}% | MaxDD={m['MaxDD']:5.1f}R")
        print()

    # ---- Path-dependent full backtest for top strategies ----
    print("\n  === PATH-DEPENDENT BACKTEST (with 0-for-2, daily limits) ===")
    print("  Running full backtest for each fixed strategy + ML adaptive...\n")

    _run_path_dependent_comparison(d, d_extra, df, model, feature_cols)


def _run_path_dependent_comparison(
    d: dict, d_extra: dict, df: pd.DataFrame, model, feature_cols: list,
) -> None:
    """Run full path-dependent backtest with strategy selection per signal."""
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
    bias_conf_arr = d["bias_conf_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    signal_quality = d["signal_quality"]
    stop_atr_ratio = d["stop_atr_ratio"]
    fluency_arr = d["fluency_arr"]
    dates = d["dates"]
    dow_arr = d["dow_arr"]
    et_frac_arr = d["et_frac_arr"]
    pa_alt_arr = d["pa_alt_arr"]
    swing_high_mask = d["swing_high_mask"]
    swing_low_mask = d["swing_low_mask"]
    nq = d["nq"]
    target_rr_arr = d["target_rr_arr"]

    smt_cfg = params.get("smt", {})
    bias_relax = params.get("bias_relaxation", {})
    pa_threshold = params.get("pa_quality", {}).get("alt_dir_threshold", 1.0)
    session_rules = params.get("session_rules", {})
    session_regime = params.get("session_regime", {})
    dir_mgmt = params.get("direction_mgmt", {})
    mss_mgmt = params.get("mss_management", {})
    mss_mgmt_enabled = mss_mgmt.get("enabled", False)
    dual_mode = params.get("dual_mode", {})
    grading_params = params["grading"]
    c_skip = grading_params["c_skip"]
    bt_params = params["backtest"]
    slippage_ticks = bt_params["slippage_normal_ticks"]
    slippage_points = slippage_ticks * 0.25
    point_value = params["position"]["point_value"]
    commission_per_side = bt_params["commission_per_side_micro"]
    pos_params = params["position"]
    risk_params = params["risk"]
    normal_r = pos_params["normal_r"]
    reduced_r = pos_params["reduced_r"]
    daily_max_loss_r = risk_params["daily_max_loss_r"]
    max_consec_losses = risk_params["max_consecutive_losses"]
    a_plus_mult = grading_params["a_plus_size_mult"]
    b_plus_mult = grading_params["b_plus_size_mult"]

    full_high = nq["high"].values
    full_low = nq["low"].values

    sess_overnight_high = d_extra["sess_overnight_high"]
    sess_overnight_low = d_extra["sess_overnight_low"]

    # Build signal-to-bar lookup from df
    signal_idx_set = set(df["idx"].values)

    # Precompute ML predictions for all signals (if model available)
    ml_pred_map = {}
    if model is not None:
        X_all = df[feature_cols].values
        preds = model.predict(X_all)
        for row_i, sig_idx in enumerate(df["idx"].values):
            ml_pred_map[sig_idx] = int(preds[row_i])

    # Strategy configs: (label, strategy_id_per_signal_func)
    # strategy_id: 0=Scalp, 1=Standard, 2=Runner, 3=Multi-TP
    strategy_configs = [
        ("Always Scalp", lambda i: 0),
        ("Always Standard", lambda i: 1),
        ("Always Runner", lambda i: 2),
        ("Always Multi-TP", lambda i: 3),
    ]
    if model is not None:
        strategy_configs.append(
            ("ML Adaptive", lambda i: ml_pred_map.get(i, 3))
        )
    strategy_configs.append(
        ("Oracle", lambda i: int(df.loc[df["idx"] == i, "best_strategy"].values[0])
         if i in signal_idx_set else 3)
    )

    # Precompute oracle lookup for speed
    oracle_map = {}
    for _, row in df.iterrows():
        oracle_map[int(row["idx"])] = int(row["best_strategy"])

    # Fix oracle lambda to use map
    strategy_configs_fixed = []
    for label, func in strategy_configs:
        if label == "Oracle":
            strategy_configs_fixed.append(
                ("Oracle", lambda i, m=oracle_map: m.get(i, 3)))
        else:
            strategy_configs_fixed.append((label, func))
    strategy_configs = strategy_configs_fixed

    all_results = {}

    for strat_label, strat_func in strategy_configs:
        trades = _run_single_strategy_backtest(
            d=d, d_extra=d_extra,
            n=n, o=o, h=h, l=l, c_arr=c_arr, atr_arr=atr_arr,
            sig_mask=sig_mask, sig_dir=sig_dir, sig_type=sig_type,
            has_smt_arr=has_smt_arr,
            entry_price_arr=entry_price_arr, model_stop_arr=model_stop_arr,
            irl_target_arr=irl_target_arr, bias_dir_arr=bias_dir_arr,
            bias_conf_arr=bias_conf_arr, regime_arr=regime_arr,
            news_blackout_arr=news_blackout_arr,
            signal_quality=signal_quality, stop_atr_ratio=stop_atr_ratio,
            fluency_arr=fluency_arr,
            dates=dates, dow_arr=dow_arr, et_frac_arr=et_frac_arr,
            pa_alt_arr=pa_alt_arr,
            swing_high_mask=swing_high_mask, swing_low_mask=swing_low_mask,
            nq=nq, target_rr_arr=target_rr_arr,
            params=params, point_value=point_value,
            slippage_points=slippage_points, commission_per_side=commission_per_side,
            signal_idx_set=signal_idx_set,
            strat_func=strat_func,
            full_high=full_high, full_low=full_low,
            sess_overnight_high=sess_overnight_high,
            sess_overnight_low=sess_overnight_low,
        )
        m = compute_metrics(trades)
        all_results[strat_label] = (trades, m)

    # Print comparison
    for label, (trades, m) in all_results.items():
        print_metrics(label, m)

    # Walk-forward for best strategies
    for label in ["Always Multi-TP", "ML Adaptive", "Oracle"]:
        if label not in all_results:
            continue
        trades, _ = all_results[label]
        if trades:
            print(f"\n  Walk-forward for {label}:")
            wf = walk_forward_metrics(trades)
            for row in wf:
                print(f"    {row['year']} | {row['n']:3d}t | R={row['R']:+6.1f} | "
                      f"WR={row['WR']:5.1f}% | PF={row['PF']:5.2f} | PPDD={row['PPDD']:5.2f}")


def _run_single_strategy_backtest(
    d, d_extra, n, o, h, l, c_arr, atr_arr,
    sig_mask, sig_dir, sig_type, has_smt_arr,
    entry_price_arr, model_stop_arr, irl_target_arr,
    bias_dir_arr, bias_conf_arr, regime_arr, news_blackout_arr,
    signal_quality, stop_atr_ratio, fluency_arr,
    dates, dow_arr, et_frac_arr, pa_alt_arr,
    swing_high_mask, swing_low_mask, nq, target_rr_arr,
    params, point_value, slippage_points, commission_per_side,
    signal_idx_set, strat_func,
    full_high, full_low,
    sess_overnight_high, sess_overnight_low,
) -> list[dict]:
    """Full path-dependent backtest with per-signal strategy selection."""
    smt_cfg = params.get("smt", {})
    bias_relax = params.get("bias_relaxation", {})
    pa_threshold = params.get("pa_quality", {}).get("alt_dir_threshold", 1.0)
    session_rules = params.get("session_rules", {})
    session_regime = params.get("session_regime", {})
    dir_mgmt = params.get("direction_mgmt", {})
    mss_mgmt = params.get("mss_management", {})
    mss_mgmt_enabled = mss_mgmt.get("enabled", False)
    dual_mode = params.get("dual_mode", {})
    grading_params = params["grading"]
    c_skip = grading_params["c_skip"]
    pos_params = params["position"]
    risk_params = params["risk"]
    normal_r = pos_params["normal_r"]
    reduced_r = pos_params["reduced_r"]
    daily_max_loss_r = risk_params["daily_max_loss_r"]
    max_consec_losses = risk_params["max_consecutive_losses"]
    a_plus_mult = grading_params["a_plus_size_mult"]
    b_plus_mult = grading_params["b_plus_size_mult"]
    trim_params = params["trim"]
    be_after_trim_default = trim_params["be_after_trim"]

    sq_long = 0.68
    sq_short = 0.80
    min_stop_atr = 1.7
    block_pm_shorts = True

    MAX_HOLD = 100

    trades = []
    current_date = None
    daily_pnl_r = 0.0
    consecutive_losses = 0
    day_stopped = False
    in_position = False

    # Position state
    pos_direction = pos_entry_idx = 0
    pos_entry_price = pos_stop = pos_tp1 = pos_tp2 = 0.0
    pos_contracts = pos_remaining_contracts = 0
    pos_trim_stage = 0
    pos_be_stop = pos_trail_stop = 0.0
    pos_signal_type = pos_grade = ""
    pos_strategy_id = 0
    pos_trim_pct = 0.5
    pos_be_after_trim = True
    pos_trail_nth = 2
    pos_is_multi_tp = False
    pos_trim1_c = pos_trim2_c = 0
    pos_tp1_pnl = pos_tp2_pnl = 0.0
    pos_max_hold_bar = 0

    for i in range(n):
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

            # Max hold
            if bars_in_trade >= MAX_HOLD:
                exit_price = c_arr[i]
                exit_reason = "max_hold"
                exited = True

            # Early cut PA (pre-trim only)
            if not exited and pos_trim_stage == 0 and 2 <= bars_in_trade <= 4:
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
                    exit_price = o[i + 1] if i + 1 < n else c_arr[i]
                    exit_reason = "early_cut_pa"
                    exited = True

            if not exited:
                if pos_is_multi_tp:
                    # Multi-TP exit
                    if pos_direction == 1:
                        if pos_trim_stage >= 2:
                            eff_stop = pos_trail_stop if pos_trail_stop > 0 else pos_stop
                            if pos_be_stop > 0:
                                eff_stop = max(eff_stop, pos_be_stop)
                        elif pos_trim_stage == 1:
                            eff_stop = pos_stop
                        else:
                            eff_stop = pos_stop

                        if l[i] <= eff_stop:
                            exit_price = eff_stop - slippage_points
                            exit_reason = "be_sweep" if (pos_trim_stage > 0 and eff_stop >= pos_entry_price) else "stop"
                            exited = True
                        elif pos_trim_stage == 0 and h[i] >= pos_tp1:
                            tc1 = max(1, int(pos_contracts * 0.25))
                            pos_trim1_c = tc1
                            pos_remaining_contracts = pos_contracts - tc1
                            pos_tp1_pnl = pos_tp1 - pos_entry_price
                            pos_trim_stage = 1
                            if pos_remaining_contracts <= 0:
                                exit_price = pos_tp1
                                exit_reason = "tp1"
                                exit_contracts = pos_contracts
                                exited = True
                        if not exited and pos_trim_stage == 1 and h[i] >= pos_tp2:
                            tc2 = max(1, int(pos_contracts * 0.25))
                            tc2 = min(tc2, pos_remaining_contracts)
                            pos_trim2_c = tc2
                            pos_remaining_contracts -= tc2
                            pos_tp2_pnl = pos_tp2 - pos_entry_price
                            pos_trim_stage = 2
                            pos_be_stop = pos_entry_price
                            if pos_remaining_contracts > 0:
                                pos_trail_stop = _find_nth_swing(
                                    swing_low_mask, full_low, i, pos_trail_nth, 1)
                                if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                                    pos_trail_stop = pos_entry_price
                            if pos_remaining_contracts <= 0:
                                exit_price = pos_tp2
                                exit_reason = "tp2"
                                exit_contracts = pos_contracts
                                exited = True
                        if not exited and pos_trim_stage >= 2:
                            nt = _find_nth_swing(swing_low_mask, full_low, i, pos_trail_nth, 1)
                            if not np.isnan(nt) and nt > pos_trail_stop:
                                pos_trail_stop = nt

                    else:  # SHORT multi-TP
                        if pos_trim_stage >= 2:
                            eff_stop = pos_trail_stop if pos_trail_stop > 0 else pos_stop
                            if pos_be_stop > 0:
                                eff_stop = min(eff_stop, pos_be_stop)
                        elif pos_trim_stage == 1:
                            eff_stop = pos_stop
                        else:
                            eff_stop = pos_stop

                        if h[i] >= eff_stop:
                            exit_price = eff_stop + slippage_points
                            exit_reason = "be_sweep" if (pos_trim_stage > 0 and eff_stop <= pos_entry_price) else "stop"
                            exited = True
                        elif pos_trim_stage == 0 and l[i] <= pos_tp1:
                            tc1 = max(1, int(pos_contracts * 0.25))
                            pos_trim1_c = tc1
                            pos_remaining_contracts = pos_contracts - tc1
                            pos_tp1_pnl = pos_entry_price - pos_tp1
                            pos_trim_stage = 1
                            if pos_remaining_contracts <= 0:
                                exit_price = pos_tp1
                                exit_reason = "tp1"
                                exit_contracts = pos_contracts
                                exited = True
                        if not exited and pos_trim_stage == 1 and l[i] <= pos_tp2:
                            tc2 = max(1, int(pos_contracts * 0.25))
                            tc2 = min(tc2, pos_remaining_contracts)
                            pos_trim2_c = tc2
                            pos_remaining_contracts -= tc2
                            pos_tp2_pnl = pos_entry_price - pos_tp2
                            pos_trim_stage = 2
                            pos_be_stop = pos_entry_price
                            if pos_remaining_contracts > 0:
                                pos_trail_stop = _find_nth_swing(
                                    swing_high_mask, full_high, i, pos_trail_nth, -1)
                                if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                                    pos_trail_stop = pos_entry_price
                            if pos_remaining_contracts <= 0:
                                exit_price = pos_tp2
                                exit_reason = "tp2"
                                exit_contracts = pos_contracts
                                exited = True
                        if not exited and pos_trim_stage >= 2:
                            nt = _find_nth_swing(swing_high_mask, full_high, i, pos_trail_nth, -1)
                            if not np.isnan(nt) and nt < pos_trail_stop:
                                pos_trail_stop = nt

                else:
                    # Single-TP exit
                    if pos_direction == 1:
                        eff_stop = pos_trail_stop if pos_trim_stage > 0 and pos_trail_stop > 0 else pos_stop
                        if pos_trim_stage > 0 and pos_be_after_trim and pos_be_stop > 0:
                            eff_stop = max(eff_stop, pos_be_stop)
                        if l[i] <= eff_stop:
                            exit_price = eff_stop - slippage_points
                            exit_reason = "be_sweep" if (pos_trim_stage > 0 and eff_stop >= pos_entry_price) else "stop"
                            exited = True
                        elif pos_trim_stage == 0 and h[i] >= pos_tp1:
                            tc = max(1, int(pos_contracts * pos_trim_pct))
                            pos_remaining_contracts = pos_contracts - tc
                            pos_trim_stage = 1
                            pos_be_stop = pos_entry_price
                            if pos_remaining_contracts <= 0:
                                exit_price = pos_tp1
                                exit_reason = "tp1"
                                exit_contracts = pos_contracts
                                exited = True
                            elif pos_remaining_contracts > 0:
                                pos_trail_stop = _find_nth_swing(
                                    swing_low_mask, full_low, i, pos_trail_nth, 1)
                                if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                                    pos_trail_stop = pos_be_stop
                        if pos_trim_stage > 0 and not exited:
                            nt = _find_nth_swing(swing_low_mask, full_low, i, pos_trail_nth, 1)
                            if not np.isnan(nt) and nt > pos_trail_stop:
                                pos_trail_stop = nt

                    else:  # SHORT single-TP
                        eff_stop = pos_trail_stop if pos_trim_stage > 0 and pos_trail_stop > 0 else pos_stop
                        if pos_trim_stage > 0 and pos_be_after_trim and pos_be_stop > 0:
                            eff_stop = min(eff_stop, pos_be_stop)
                        if h[i] >= eff_stop:
                            exit_price = eff_stop + slippage_points
                            exit_reason = "be_sweep" if (pos_trim_stage > 0 and eff_stop <= pos_entry_price) else "stop"
                            exited = True
                        elif pos_trim_stage == 0 and l[i] <= pos_tp1:
                            tc = max(1, int(pos_contracts * pos_trim_pct))
                            pos_remaining_contracts = pos_contracts - tc
                            pos_trim_stage = 1
                            pos_be_stop = pos_entry_price
                            if pos_remaining_contracts <= 0:
                                exit_price = pos_tp1
                                exit_reason = "tp1"
                                exit_contracts = pos_contracts
                                exited = True
                            elif pos_remaining_contracts > 0:
                                pos_trail_stop = _find_nth_swing(
                                    swing_high_mask, full_high, i, pos_trail_nth, -1)
                                if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                                    pos_trail_stop = pos_be_stop
                        if pos_trim_stage > 0 and not exited:
                            nt = _find_nth_swing(swing_high_mask, full_high, i, pos_trail_nth, -1)
                            if not np.isnan(nt) and nt < pos_trail_stop:
                                pos_trail_stop = nt

            if exited:
                pnl_pts = (exit_price - pos_entry_price) if pos_direction == 1 else (pos_entry_price - exit_price)

                if pos_is_multi_tp:
                    total_pnl = 0.0
                    if pos_trim_stage >= 1:
                        total_pnl += pos_tp1_pnl * point_value * pos_trim1_c
                    if pos_trim_stage >= 2:
                        total_pnl += pos_tp2_pnl * point_value * pos_trim2_c
                    total_pnl += pnl_pts * point_value * exit_contracts
                    total_pnl -= commission_per_side * 2 * pos_contracts
                elif pos_trim_stage > 0 and exit_reason != "tp1":
                    trim_c = pos_contracts - exit_contracts
                    tp_pnl = (pos_tp1 - pos_entry_price) if pos_direction == 1 else (pos_entry_price - pos_tp1)
                    total_pnl = tp_pnl * point_value * trim_c + pnl_pts * point_value * exit_contracts
                    total_pnl -= commission_per_side * 2 * pos_contracts
                else:
                    total_pnl = pnl_pts * point_value * exit_contracts
                    total_pnl -= commission_per_side * 2 * exit_contracts

                stop_dist_exit = abs(pos_entry_price - pos_stop)
                total_risk = stop_dist_exit * point_value * pos_contracts
                r_mult = total_pnl / total_risk if total_risk > 0 else 0.0

                any_trimmed = pos_trim_stage > 0

                trades.append({
                    "entry_time": nq.index[pos_entry_idx],
                    "exit_time": nq.index[min(i, n - 1)],
                    "r": r_mult, "reason": exit_reason, "dir": pos_direction,
                    "type": pos_signal_type, "trimmed": any_trimmed,
                    "grade": pos_grade,
                    "entry_price": pos_entry_price, "exit_price": exit_price,
                    "stop_price": pos_stop, "tp1_price": pos_tp1,
                    "pnl_dollars": total_pnl,
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

        # Only trade signals in our analyzed set
        if i not in signal_idx_set:
            continue

        sig_f = params.get("signal_filter", {})
        if not sig_f.get("allow_mss", True) and str(sig_type[i]) == "mss":
            continue
        if not sig_f.get("allow_trend", True) and str(sig_type[i]) == "trend":
            continue

        entry_p = entry_price_arr[i]
        stop = model_stop_arr[i]
        tp_raw = irl_target_arr[i]
        if np.isnan(entry_p) or np.isnan(stop) or np.isnan(tp_raw):
            continue
        if direction == 1 and (stop >= entry_p or tp_raw <= entry_p):
            continue
        if direction == -1 and (stop <= entry_p or tp_raw >= entry_p):
            continue

        if block_pm_shorts and et_frac >= 14.0 and direction == -1:
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
        if not mss_bypass:
            if not is_ny:
                continue

        if not np.isnan(stop_atr_ratio[i]):
            if stop_atr_ratio[i] < min_stop_atr:
                continue

        if not np.isnan(signal_quality[i]):
            eff_sq = sq_long if direction == 1 else sq_short
            if signal_quality[i] < eff_sq:
                continue

        if i + 1 >= n:
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

        # --- Select strategy for this signal ---
        strategy_id = strat_func(i)
        tp_distance_raw = abs(tp_raw - actual_entry)
        is_mss_signal = str(sig_type[i]) == "mss"

        if strategy_id == 0:
            # Scalp: 100% at TP1 (IRL × 2.0)
            tp1 = actual_entry + tp_distance_raw * 2.0 if direction == 1 else actual_entry - tp_distance_raw * 2.0
            tp2 = tp1
            use_multi = False
            trim_pct_use = 1.0
            be_after_use = True
            trail_nth_use = 2
        elif strategy_id == 1:
            # Standard: 50% at TP1 (IRL × 2.0), trail 2nd swing, BE
            tp1 = actual_entry + tp_distance_raw * 2.0 if direction == 1 else actual_entry - tp_distance_raw * 2.0
            tp2 = tp1
            use_multi = False
            trim_pct_use = 0.50
            be_after_use = True
            trail_nth_use = 2
        elif strategy_id == 2:
            # Runner: 25% at TP1 (IRL × 3.0), trail 3rd swing, BE
            tp1 = actual_entry + tp_distance_raw * 3.0 if direction == 1 else actual_entry - tp_distance_raw * 3.0
            tp2 = tp1
            use_multi = False
            trim_pct_use = 0.25
            be_after_use = True
            trail_nth_use = 3
        else:
            # Multi-TP: 25/25/50 at liquidity ladder, late BE, trail 3rd swing
            tp_mult = 3.0
            if mss_mgmt_enabled and is_mss_signal and direction == 1:
                tp_mult = 2.5
            tp_base = actual_entry + tp_distance_raw * tp_mult if direction == 1 else actual_entry - tp_distance_raw * tp_mult
            if direction == 1:
                tp1, tp2 = build_liquidity_ladder_long(actual_entry, stop, tp_base, i, d_extra)
            else:
                tp1, tp2 = build_liquidity_ladder_short(actual_entry, stop, tp_base, i, d_extra)
            use_multi = True
            trim_pct_use = 0.25
            be_after_use = False  # late BE
            trail_nth_use = 3

        # Enter position
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
        pos_grade = grade
        pos_strategy_id = strategy_id
        pos_trim_pct = trim_pct_use
        pos_be_after_trim = be_after_use
        pos_trail_nth = trail_nth_use
        pos_is_multi_tp = use_multi
        pos_trim1_c = 0
        pos_trim2_c = 0
        pos_tp1_pnl = 0.0
        pos_tp2_pnl = 0.0

    return trades


# ======================================================================
# Step 6: Simple rule extraction
# ======================================================================

def extract_simple_rules(df: pd.DataFrame) -> None:
    """Analyze features to find simple decision rules for strategy selection."""
    print("\n" + "=" * 80)
    print("STEP 6: SIMPLE RULE EXTRACTION")
    print("=" * 80)

    strategies = ["r_A", "r_B", "r_C", "r_D"]
    labels = ["Scalp", "Standard", "Runner", "Multi-TP"]

    df["year"] = pd.to_datetime(df["entry_time"]).dt.year

    # ---- Rule 1: By signal type ----
    print("\n  --- By signal_type ---")
    for st_val, st_name in [(0, "Trend"), (1, "MSS")]:
        sub = df[df["signal_type"] == st_val]
        if len(sub) == 0:
            continue
        print(f"  {st_name} ({len(sub)} signals):")
        for j, (col, lbl) in enumerate(zip(strategies, labels)):
            avg_r = sub[col].mean()
            total_r = sub[col].sum()
            print(f"    {lbl:12s}: avgR={avg_r:+.4f} | sumR={total_r:+.1f}")
        best = labels[np.argmax([sub[c].sum() for c in strategies])]
        print(f"    => Best: {best}")

    # ---- Rule 2: By grade ----
    print("\n  --- By grade ---")
    for g_val, g_name in [(2, "A+"), (1, "B+"), (0, "C")]:
        sub = df[df["grade"] == g_val]
        if len(sub) == 0:
            continue
        print(f"  {g_name} ({len(sub)} signals):")
        for j, (col, lbl) in enumerate(zip(strategies, labels)):
            avg_r = sub[col].mean()
            total_r = sub[col].sum()
            print(f"    {lbl:12s}: avgR={avg_r:+.4f} | sumR={total_r:+.1f}")
        best = labels[np.argmax([sub[c].sum() for c in strategies])]
        print(f"    => Best: {best}")

    # ---- Rule 3: By hour ----
    print("\n  --- By hour (ET) ---")
    for hour in range(10, 16):
        sub = df[df["hour_et"] == hour]
        if len(sub) == 0:
            continue
        totals = [sub[c].sum() for c in strategies]
        best = labels[np.argmax(totals)]
        print(f"  {hour:2d}:00 ({len(sub):3d} signals): "
              f"Scalp={totals[0]:+.1f} Std={totals[1]:+.1f} "
              f"Runner={totals[2]:+.1f} Multi={totals[3]:+.1f} => {best}")

    # ---- Rule 4: By direction ----
    print("\n  --- By direction ---")
    for d_val, d_name in [(1, "Long"), (-1, "Short")]:
        sub = df[df["direction"] == d_val]
        if len(sub) == 0:
            continue
        totals = [sub[c].sum() for c in strategies]
        best = labels[np.argmax(totals)]
        print(f"  {d_name} ({len(sub)} signals): "
              f"Scalp={totals[0]:+.1f} Std={totals[1]:+.1f} "
              f"Runner={totals[2]:+.1f} Multi={totals[3]:+.1f} => {best}")

    # ---- Rule 5: By ATR percentile (high vol vs low vol) ----
    print("\n  --- By ATR percentile ---")
    for label, lo, hi in [("Low vol (<0.8)", 0.0, 0.8), ("Normal (0.8-1.2)", 0.8, 1.2),
                           ("High vol (>1.2)", 1.2, 99.0)]:
        sub = df[(df["atr_percentile"] >= lo) & (df["atr_percentile"] < hi)]
        if len(sub) == 0:
            continue
        totals = [sub[c].sum() for c in strategies]
        best = labels[np.argmax(totals)]
        print(f"  {label:20s} ({len(sub):3d}): "
              f"Scalp={totals[0]:+.1f} Std={totals[1]:+.1f} "
              f"Runner={totals[2]:+.1f} Multi={totals[3]:+.1f} => {best}")

    # ---- Rule 6: By target RR ----
    print("\n  --- By target RR ---")
    for label, lo, hi in [("Low RR (<1.5)", 0.0, 1.5), ("Medium RR (1.5-3.0)", 1.5, 3.0),
                           ("High RR (>3.0)", 3.0, 999.0)]:
        sub = df[(df["target_rr"] >= lo) & (df["target_rr"] < hi)]
        if len(sub) == 0:
            continue
        totals = [sub[c].sum() for c in strategies]
        best = labels[np.argmax(totals)]
        print(f"  {label:22s} ({len(sub):3d}): "
              f"Scalp={totals[0]:+.1f} Std={totals[1]:+.1f} "
              f"Runner={totals[2]:+.1f} Multi={totals[3]:+.1f} => {best}")

    # ---- Composite simple rules ----
    print("\n  --- Simple Rule Candidates ---")

    # Test a few simple rules on the test period
    test_df = df[df["year"] >= 2022].copy()
    if len(test_df) == 0:
        print("  No test data.")
        return

    def apply_rule(row, rule_func):
        return rule_func(row)

    # Rule A: "MSS->Scalp, everything else->Multi-TP"
    def rule_mss_scalp(row):
        return "r_A" if row["signal_type"] == 1 else "r_D"

    # Rule B: "Shorts->Scalp, Longs->Multi-TP"
    def rule_dir_based(row):
        return "r_A" if row["direction"] == -1 else "r_D"

    # Rule C: "A+ grade AM -> Runner, else Multi-TP"
    def rule_grade_time(row):
        if row["grade"] == 2 and row["hour_et"] < 12:
            return "r_C"
        return "r_D"

    # Rule D: "High vol -> Scalp, else Multi-TP"
    def rule_vol_based(row):
        return "r_A" if row["atr_percentile"] > 1.2 else "r_D"

    # Rule E: "MSS->Scalp, Short->Scalp, AM A+->Runner, else Multi-TP"
    def rule_composite(row):
        if row["signal_type"] == 1:
            return "r_A"
        if row["direction"] == -1:
            return "r_A"
        if row["grade"] == 2 and row["hour_et"] < 12:
            return "r_C"
        return "r_D"

    rules = [
        ("MSS->Scalp, else Multi-TP", rule_mss_scalp),
        ("Shorts->Scalp, Longs->Multi-TP", rule_dir_based),
        ("A+ AM->Runner, else Multi-TP", rule_grade_time),
        ("High vol->Scalp, else Multi-TP", rule_vol_based),
        ("MSS/Short->Scalp, A+AM->Runner, else Multi-TP", rule_composite),
    ]

    print(f"\n  TEST PERIOD ({len(test_df)} signals, >=2022):")
    # Baseline: always Multi-TP
    base_r = test_df["r_D"].sum()
    print(f"    {'Always Multi-TP (baseline)':50s}: R = {base_r:+.1f}")

    for rule_name, rule_func in rules:
        r_col = test_df.apply(rule_func, axis=1)
        r_values = np.array([test_df.iloc[j][r_col.iloc[j]] for j in range(len(test_df))])
        total_r = r_values.sum()
        delta = total_r - base_r
        print(f"    {rule_name:50s}: R = {total_r:+.1f} (delta = {delta:+.1f})")

    oracle_r = test_df["best_r"].sum()
    print(f"    {'Oracle (upper bound)':50s}: R = {oracle_r:+.1f}")


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 100)
    print("A2A EXIT STRATEGY CLASSIFIER — Adaptive Exit Selection")
    print("=" * 100)

    # ---- Load data ----
    print("\n[PHASE 1] Loading data...")
    d = load_all()
    d_extra = prepare_liquidity_data(d)

    # ---- Step 1: Generate per-signal outcomes ----
    print("\n[PHASE 2] Generating per-signal outcomes for 4 strategies...")
    df = generate_per_signal_outcomes(d, d_extra)

    if len(df) == 0:
        print("ERROR: No signals found. Check filters.")
        return

    # ---- Step 2: Feature matrix (built inside Step 1) ----
    print(f"\n[STEP 2] Feature matrix: {len(df)} signals × {df.shape[1]} columns")
    print(f"  Date range: {df['entry_time'].min()} to {df['entry_time'].max()}")
    print(f"  Longs: {(df['direction'] == 1).sum()}, Shorts: {(df['direction'] == -1).sum()}")
    print(f"  Trend: {(df['signal_type'] == 0).sum()}, MSS: {(df['signal_type'] == 1).sum()}")

    # ---- Step 3: Label and analyze ----
    df = analyze_strategy_outcomes(df)

    # ---- Step 4: Train XGBoost classifier ----
    model, feature_cols, df = train_exit_classifier(df)

    # ---- Step 5: Simulate adaptive strategy ----
    simulate_adaptive_backtest(d, d_extra, df, model, feature_cols)

    # ---- Step 6: Simple rule extraction ----
    extract_simple_rules(df)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
