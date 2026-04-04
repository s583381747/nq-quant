"""
backtest/engine_jit.py -- Numba JIT-optimized backtest engine.

Drop-in replacement for engine.run_backtest() with identical logic but
compiled inner loop for 10-50x speedup on large datasets.

The core loop is compiled with @numba.njit. All pandas/string operations
are handled in the Python wrapper; the JIT function receives only numpy
arrays and scalar parameters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "params.yaml"


def _load_params(config_path: Path = _CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Pre-compute nth swing lookup table
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _precompute_nth_swing_table(
    swing_low_mask: numba.boolean[:],
    swing_high_mask: numba.boolean[:],
    low_arr: numba.float64[:],
    high_arr: numba.float64[:],
    max_nth: numba.int64,
) -> tuple:
    """Pre-compute nth swing low and nth swing high for every bar.

    Returns
    -------
    nth_swing_low : float64[n, max_nth]
        nth_swing_low[i, k] = price of the (k+1)th most recent swing low at bar i
    nth_swing_high : float64[n, max_nth]
        nth_swing_high[i, k] = price of the (k+1)th most recent swing high at bar i
    """
    n = len(swing_low_mask)
    nth_swing_low = np.full((n, max_nth), np.nan, dtype=np.float64)
    nth_swing_high = np.full((n, max_nth), np.nan, dtype=np.float64)

    # For swing lows: maintain a rolling buffer of the last max_nth swing low prices
    buf_low = np.full(max_nth, np.nan, dtype=np.float64)
    buf_low_count = 0

    buf_high = np.full(max_nth, np.nan, dtype=np.float64)
    buf_high_count = 0

    for i in range(n):
        # Write table FIRST (before processing bar i's swing).
        # This ensures nth_swing_*[i] reflects swings at bars 0..i-1 only,
        # matching the original _find_nth_swing which scans from i-1 backward.
        for k in range(buf_low_count):
            nth_swing_low[i, k] = buf_low[k]
        for k in range(buf_high_count):
            nth_swing_high[i, k] = buf_high[k]

        # Then update buffers with bar i's swing (for future bars)
        if swing_low_mask[i]:
            if buf_low_count < max_nth:
                buf_low_count += 1
            for k in range(buf_low_count - 1, 0, -1):
                buf_low[k] = buf_low[k - 1]
            buf_low[0] = low_arr[i]

        if swing_high_mask[i]:
            if buf_high_count < max_nth:
                buf_high_count += 1
            for k in range(buf_high_count - 1, 0, -1):
                buf_high[k] = buf_high[k - 1]
            buf_high[0] = high_arr[i]

    return nth_swing_low, nth_swing_high


# ---------------------------------------------------------------------------
# Core JIT backtest loop
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _backtest_core(
    # OHLC arrays
    open_arr,       # float64[n]
    high_arr,       # float64[n]
    low_arr,        # float64[n]
    close_arr,      # float64[n]
    # Signal arrays
    sig_mask,       # bool[n]
    sig_dir,        # float64[n]
    sig_type_int,   # int64[n] -- 0=trend, 1=mss
    entry_price_arr,  # float64[n]
    model_stop_arr,   # float64[n]
    irl_target_arr,   # float64[n]
    has_smt_arr,      # bool[n] -- SMT divergence confirmation
    # Bias/regime
    bias_dir_arr,   # float64[n]
    bias_conf_arr,  # float64[n]
    regime_arr,     # float64[n]
    # ATR & PA quality
    atr_arr,        # float64[n]
    pa_alt_arr,     # float64[n]
    # Signal quality scores (pre-computed, NaN where not applicable)
    signal_quality_arr,  # float64[n]
    # Pre-computed swing tables (from _precompute_nth_swing_table)
    nth_swing_low_table,   # float64[n, max_nth]
    nth_swing_high_table,  # float64[n, max_nth]
    # Model predictions
    model_probs,    # float64[n]
    # Pre-computed datetime arrays
    session_date_int,  # int64[n] -- ordinal day number for session date
    et_frac_arr,       # float64[n] -- hour + min/60 in ET
    et_dow_arr,        # int64[n] -- day of week (0=Mon, 4=Fri)
    # News blackout
    news_blackout_arr,  # bool[n] (or length-0 if no news filter)
    has_news_filter,    # bool scalar
    # Scalar params
    normal_r,              # float64
    reduced_r,             # float64
    point_value,           # float64
    daily_max_loss_r,      # float64
    max_consec_losses,     # int64
    commission_per_side,   # float64
    slippage_points,       # float64
    a_plus_mult,           # float64
    b_plus_mult,           # float64
    c_skip,                # bool
    trim_pct,              # float64
    be_after_trim,         # bool
    nth_swing,             # int64 (1-indexed: 2 means "2nd most recent")
    pa_threshold,          # float64
    min_stop_atr_mult,     # float64
    ny_tp_mult,            # float64
    ny_tp_enabled,         # bool
    threshold,             # float64
    sf_enabled,            # bool -- session filter enabled
    sf_ny_direction,       # int64 -- 0=both, 1=long, -1=short
    sf_london_direction,   # int64 -- 0=both, 1=long, -1=short
    sf_skip_asia,          # bool -- skip Asia session
    sf_skip_london,        # bool -- skip London session entirely
    # Sprint 4-5 params (extracted as scalars in wrapper)
    smt_enabled,           # bool -- SMT feature enabled
    smt_bypass_session,    # bool -- MSS+SMT can bypass session filter
    bias_relax_enabled,    # bool -- opposing shorts allowed
    sq_enabled,            # bool -- signal quality filter enabled
    sq_threshold,          # float64 -- base SQ threshold
    dual_mode_enabled,     # bool -- dual-mode SQ + short management
    dual_mode_short_sq,    # float64 -- SQ threshold for shorts in dual mode
    dual_mode_short_rr,    # float64 -- short RR target in dual mode
    dual_mode_short_trim,  # float64 -- short trim pct in dual mode
    dir_mgmt_enabled,      # bool -- direction management enabled
    dir_mgmt_long_tp_mult, # float64
    dir_mgmt_short_tp_mult,# float64
    dir_mgmt_long_trim,    # float64
    dir_mgmt_short_trim,   # float64
    mss_mgmt_enabled,      # bool -- MSS-specific management
    mss_long_tp_mult,      # float64
    mss_short_rr,          # float64
    mss_short_trim,        # float64
    mss_long_trim,         # float64
    sr_enabled,            # bool -- session regime enabled
    sr_am_end,             # float64
    sr_lunch_start,        # float64
    sr_lunch_end,          # float64
    sr_pm_start,           # float64
    sr_am_mult,            # float64
    sr_lunch_mult,         # float64
    sr_pm_mult,            # float64
    allow_mss,             # bool -- signal filter: allow MSS signals
    allow_trend,           # bool -- signal filter: allow trend signals
):
    """Core backtest loop compiled with Numba.

    Returns tuple of arrays for trade fields.
    Max trades = n (way more than needed, we'll trim).
    """
    n = len(open_arr)
    max_trades = min(n, 100000)  # generous upper bound

    # Output arrays (pre-allocated, will trim later)
    t_entry_idx = np.empty(max_trades, dtype=np.int64)
    t_exit_idx = np.empty(max_trades, dtype=np.int64)
    t_direction = np.empty(max_trades, dtype=np.int64)
    t_entry_price = np.empty(max_trades, dtype=np.float64)
    t_exit_price = np.empty(max_trades, dtype=np.float64)
    t_stop_price = np.empty(max_trades, dtype=np.float64)
    t_tp1_price = np.empty(max_trades, dtype=np.float64)
    t_contracts = np.empty(max_trades, dtype=np.int64)
    t_pnl_points = np.empty(max_trades, dtype=np.float64)
    t_pnl_dollars = np.empty(max_trades, dtype=np.float64)
    t_r_multiple = np.empty(max_trades, dtype=np.float64)
    t_exit_reason = np.empty(max_trades, dtype=np.int64)
    # exit_reason codes: 0=stop, 1=be_sweep, 2=tp1, 3=early_cut_pa, 4=eod_close
    t_signal_type = np.empty(max_trades, dtype=np.int64)
    t_bias_dir = np.empty(max_trades, dtype=np.float64)
    t_regime = np.empty(max_trades, dtype=np.float64)
    t_model_prob = np.empty(max_trades, dtype=np.float64)
    t_grade = np.empty(max_trades, dtype=np.int64)
    # grade codes: 0=C, 1=B+, 2=A+
    t_trimmed = np.empty(max_trades, dtype=numba.boolean)
    trade_count = 0

    # ---- Backtest state ----
    current_date = np.int64(-1)
    daily_pnl_r = 0.0
    consecutive_losses = 0
    day_stopped = False

    in_position = False
    pos_direction = 0
    pos_entry_idx = 0
    pos_entry_price = 0.0
    pos_stop = 0.0
    pos_tp1 = 0.0
    pos_contracts = 0
    pos_remaining_contracts = 0
    pos_trimmed = False
    pos_be_stop = 0.0
    pos_trail_stop = 0.0
    pos_signal_type_int = 0
    pos_bias_dir = 0.0
    pos_regime = 0.0
    pos_model_prob = 0.0
    pos_grade_int = 0
    pos_trim_pct = trim_pct  # direction-aware trim percentage

    # nth_swing is 1-indexed (2 = 2nd most recent), table is 0-indexed
    swing_table_idx = nth_swing - 1
    if swing_table_idx < 0:
        swing_table_idx = 0

    for i in range(n):
        bar_date = session_date_int[i]

        # ---- New day reset ----
        if bar_date != current_date:
            current_date = bar_date
            daily_pnl_r = 0.0
            consecutive_losses = 0
            day_stopped = False

        # ---- Check exit conditions for open position ----
        if in_position:
            exited = False
            exit_reason_code = 0
            exit_price = 0.0
            exit_contracts = pos_remaining_contracts

            # --- Post-entry PA quality check (early exit) ---
            bars_in_trade = i - pos_entry_idx
            if not pos_trimmed and 2 <= bars_in_trade <= 4:
                pa_start = pos_entry_idx
                if pa_start < 0:
                    pa_start = 0
                pa_end = i + 1

                # Compute PA metrics for the window
                sum_wick_ratio = 0.0
                sum_favorable = 0.0
                count_bars = 0
                for j in range(pa_start, pa_end):
                    bar_range = high_arr[j] - low_arr[j]
                    bar_body = abs(close_arr[j] - open_arr[j])
                    safe_range = bar_range if bar_range > 0.0 else 1.0
                    wick_ratio = 1.0 - (bar_body / safe_range)
                    sum_wick_ratio += wick_ratio

                    bar_dir = 0.0
                    if close_arr[j] > open_arr[j]:
                        bar_dir = 1.0
                    elif close_arr[j] < open_arr[j]:
                        bar_dir = -1.0

                    if bar_dir == float(pos_direction):
                        sum_favorable += 1.0
                    count_bars += 1

                avg_wick = sum_wick_ratio / count_bars if count_bars > 0 else 0.0
                favorable = sum_favorable / count_bars if count_bars > 0 else 0.0

                if pos_direction == 1:
                    displacement_from_entry = close_arr[i] - pos_entry_price
                else:
                    displacement_from_entry = pos_entry_price - close_arr[i]

                cur_atr_val = atr_arr[i]
                if np.isnan(cur_atr_val):
                    cur_atr_val = 30.0
                no_progress = displacement_from_entry < cur_atr_val * 0.3
                bad_pa = avg_wick > 0.65 and favorable < 0.5

                if bad_pa and no_progress and bars_in_trade >= 3:
                    if i + 1 < n:
                        exit_price = open_arr[i + 1]
                    else:
                        exit_price = close_arr[i]
                    exit_reason_code = 3  # early_cut_pa
                    exited = True

            # ---- LONG exit logic ----
            if not exited and pos_direction == 1:
                effective_stop = pos_stop
                if pos_trimmed and pos_trail_stop > 0.0:
                    effective_stop = pos_trail_stop
                if pos_trimmed and be_after_trim and pos_be_stop > 0.0:
                    if pos_be_stop > effective_stop:
                        effective_stop = pos_be_stop

                if low_arr[i] <= effective_stop:
                    exit_price = effective_stop - slippage_points
                    if pos_trimmed and effective_stop >= pos_entry_price:
                        exit_reason_code = 1  # be_sweep
                    else:
                        exit_reason_code = 0  # stop
                    exited = True

                elif not pos_trimmed and high_arr[i] >= pos_tp1:
                    trim_contracts = max(1, int(pos_contracts * pos_trim_pct))
                    pos_remaining_contracts = pos_contracts - trim_contracts

                    pos_trimmed = True
                    pos_be_stop = pos_entry_price

                    # Update trail stop with nth swing
                    if pos_remaining_contracts > 0:
                        if swing_table_idx < nth_swing_low_table.shape[1]:
                            trail_val = nth_swing_low_table[i, swing_table_idx]
                        else:
                            trail_val = np.nan
                        if np.isnan(trail_val) or trail_val <= 0.0:
                            pos_trail_stop = pos_be_stop
                        else:
                            pos_trail_stop = trail_val

                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason_code = 2  # tp1
                        exit_contracts = pos_contracts
                        exited = True

                # Update trailing stop
                if pos_trimmed and not exited:
                    if swing_table_idx < nth_swing_low_table.shape[1]:
                        new_trail = nth_swing_low_table[i, swing_table_idx]
                    else:
                        new_trail = np.nan
                    if not np.isnan(new_trail) and new_trail > pos_trail_stop:
                        pos_trail_stop = new_trail

            # ---- SHORT exit logic ----
            elif not exited and pos_direction == -1:
                effective_stop = pos_stop
                if pos_trimmed and pos_trail_stop > 0.0:
                    effective_stop = pos_trail_stop
                if pos_trimmed and be_after_trim and pos_be_stop > 0.0:
                    if pos_be_stop < effective_stop:
                        effective_stop = pos_be_stop

                if high_arr[i] >= effective_stop:
                    exit_price = effective_stop + slippage_points
                    if pos_trimmed and effective_stop <= pos_entry_price:
                        exit_reason_code = 1  # be_sweep
                    else:
                        exit_reason_code = 0  # stop
                    exited = True

                elif not pos_trimmed and low_arr[i] <= pos_tp1:
                    trim_contracts = max(1, int(pos_contracts * pos_trim_pct))
                    pos_remaining_contracts = pos_contracts - trim_contracts

                    pos_trimmed = True
                    pos_be_stop = pos_entry_price

                    if pos_remaining_contracts > 0:
                        if swing_table_idx < nth_swing_high_table.shape[1]:
                            trail_val = nth_swing_high_table[i, swing_table_idx]
                        else:
                            trail_val = np.nan
                        if np.isnan(trail_val) or trail_val <= 0.0:
                            pos_trail_stop = pos_be_stop
                        else:
                            pos_trail_stop = trail_val

                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason_code = 2  # tp1
                        exit_contracts = pos_contracts
                        exited = True

                if pos_trimmed and not exited:
                    if swing_table_idx < nth_swing_high_table.shape[1]:
                        new_trail = nth_swing_high_table[i, swing_table_idx]
                    else:
                        new_trail = np.nan
                    if not np.isnan(new_trail) and new_trail < pos_trail_stop:
                        pos_trail_stop = new_trail

            # ---- Record completed trade ----
            if exited:
                if pos_direction == 1:
                    pnl_pts = exit_price - pos_entry_price
                else:
                    pnl_pts = pos_entry_price - exit_price

                if pos_trimmed and exit_reason_code != 2:
                    # Partial: trim already happened at TP1
                    remaining_pnl_pts = pnl_pts
                    trim_pnl_pts_total = (pos_tp1 - pos_entry_price) if pos_direction == 1 else (pos_entry_price - pos_tp1)
                    trim_c = pos_contracts - exit_contracts
                    total_pnl_dollars = (
                        trim_pnl_pts_total * point_value * trim_c
                        + remaining_pnl_pts * point_value * exit_contracts
                    )
                    total_commission = commission_per_side * 2.0 * pos_contracts
                    total_pnl_dollars -= total_commission
                    effective_pnl_pts = total_pnl_dollars / (point_value * pos_contracts) if pos_contracts > 0 else 0.0
                else:
                    total_pnl_dollars = pnl_pts * point_value * exit_contracts
                    total_commission = commission_per_side * 2.0 * exit_contracts
                    total_pnl_dollars -= total_commission
                    effective_pnl_pts = pnl_pts

                stop_dist = abs(pos_entry_price - pos_stop)
                r_per_contract = stop_dist * point_value
                total_risk = r_per_contract * pos_contracts
                r_multiple = total_pnl_dollars / total_risk if total_risk > 0.0 else 0.0

                if trade_count < max_trades:
                    t_entry_idx[trade_count] = pos_entry_idx
                    t_exit_idx[trade_count] = i
                    t_direction[trade_count] = pos_direction
                    t_entry_price[trade_count] = pos_entry_price
                    t_exit_price[trade_count] = exit_price
                    t_stop_price[trade_count] = pos_stop
                    t_tp1_price[trade_count] = pos_tp1
                    t_contracts[trade_count] = pos_contracts
                    t_pnl_points[trade_count] = effective_pnl_pts
                    t_pnl_dollars[trade_count] = total_pnl_dollars
                    t_r_multiple[trade_count] = r_multiple
                    t_exit_reason[trade_count] = exit_reason_code
                    t_signal_type[trade_count] = pos_signal_type_int
                    t_bias_dir[trade_count] = pos_bias_dir
                    t_regime[trade_count] = pos_regime
                    t_model_prob[trade_count] = pos_model_prob
                    t_grade[trade_count] = pos_grade_int
                    t_trimmed[trade_count] = pos_trimmed
                    trade_count += 1

                # Update daily state
                daily_pnl_r += r_multiple

                # Loss tracking
                if exit_reason_code == 1 and pos_trimmed:
                    # BE sweep after trim = profitable, NOT a loss
                    pass
                elif r_multiple < 0.0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                if consecutive_losses >= max_consec_losses:
                    day_stopped = True
                if daily_pnl_r <= -daily_max_loss_r:
                    day_stopped = True

                in_position = False

        # ---- Check for new entry ----
        # News filter
        if not in_position and has_news_filter and news_blackout_arr[i]:
            continue

        # ORM no-trade window (9:30-10:00 ET) — observation only
        if not in_position and not day_stopped:
            et_frac = et_frac_arr[i]
            if 9.5 <= et_frac < 10.0 + 1.0 / 60.0:
                continue

        if not in_position and not day_stopped and sig_mask[i]:
            prob = model_probs[i]
            if np.isnan(prob) or prob < threshold:
                continue

            direction = np.int64(sig_dir[i])
            if direction == 0:
                continue

            # Signal type filter
            if not allow_mss and sig_type_int[i] == 1:
                continue
            if not allow_trend and sig_type_int[i] == 0:
                continue

            entry_p = entry_price_arr[i]
            stop = model_stop_arr[i]
            tp1 = irl_target_arr[i]

            if np.isnan(entry_p) or np.isnan(stop) or np.isnan(tp1):
                continue

            # Validate stop/target
            if direction == 1:
                if stop >= entry_p or tp1 <= entry_p:
                    continue
            else:
                if stop <= entry_p or tp1 >= entry_p:
                    continue

            # Detect signal characteristics for filter exemptions
            is_mss_signal = sig_type_int[i] == 1
            is_smt_confirmed = smt_enabled and has_smt_arr[i]
            is_mss_smt = is_mss_signal and is_smt_confirmed

            # FILTER 1: Bias -- block opposing, allow aligned + neutral
            # Exception: MSS+SMT = confirmed reversal -> bypass bias filter
            # Also: opposing shorts allowed when bias_relaxation enabled
            bias_d = bias_dir_arr[i]
            dir_sign = 1.0 if direction == 1 else -1.0
            bias_sign = 1.0 if bias_d > 0 else (-1.0 if bias_d < 0 else 0.0)
            bias_opposing = (bias_d != 0.0 and dir_sign == -bias_sign)
            if bias_opposing:
                if is_mss_smt:
                    pass  # SMT-confirmed MSS reversal: allow through
                elif bias_relax_enabled and direction == -1:
                    pass  # opposing shorts relaxation
                else:
                    continue

            # FILTER 1b: PA quality
            if pa_alt_arr[i] >= pa_threshold:
                continue

            # FILTER 1c: Session filter (configurable, default OFF)
            # MSS+SMT can bypass session filter — reversals at session boundaries
            et_frac = et_frac_arr[i]
            mss_bypass_sf = is_mss_smt and smt_bypass_session
            if sf_enabled and not mss_bypass_sf:
                if 9.5 <= et_frac < 16.0:  # NY
                    if sf_ny_direction != 0 and direction != sf_ny_direction:
                        continue
                elif 3.0 <= et_frac < 9.5:  # London
                    if sf_skip_london:
                        continue
                    if sf_london_direction != 0 and direction != sf_london_direction:
                        continue
                else:  # Asia
                    if sf_skip_asia:
                        continue
            elif not sf_enabled and not mss_bypass_sf:
                # Even with filter off, still skip Asia (low volume)
                if not (3.0 <= et_frac < 16.0):
                    continue

            # FILTER 2: Minimum stop distance
            stop_dist = abs(entry_p - stop)
            atr_val = atr_arr[i]
            if np.isnan(atr_val):
                min_stop = 10.0
            else:
                min_stop = min_stop_atr_mult * atr_val
            if stop_dist < min_stop:
                continue

            # FILTER 4: Signal quality score (Sprint 4 Attack 1)
            # Dual-mode: shorts require higher SQ threshold
            if sq_enabled and not np.isnan(signal_quality_arr[i]):
                effective_sq_threshold = sq_threshold
                if dual_mode_enabled and direction == -1:
                    effective_sq_threshold = dual_mode_short_sq
                if signal_quality_arr[i] < effective_sq_threshold:
                    continue

            # Skip if last bar
            if i + 1 >= n:
                continue

            # Grade the setup
            ba = 0.0
            if bias_d != 0.0:
                dir_sign2 = 1.0 if direction == 1 else -1.0
                bias_sign2 = 1.0 if bias_d > 0 else -1.0
                if dir_sign2 == bias_sign2:
                    ba = 1.0

            grade_int = _compute_grade_jit(ba, bias_conf_arr[i], regime_arr[i])

            if grade_int == 0 and c_skip:
                continue

            # Position sizing
            dow = et_dow_arr[i]
            is_reduced = (dow == 0 or dow == 4) or (regime_arr[i] < 1.0)
            base_r = reduced_r if is_reduced else normal_r

            if grade_int == 2:  # A+
                r_amount = base_r * a_plus_mult
            elif grade_int == 1:  # B+
                r_amount = base_r * b_plus_mult
            else:
                r_amount = base_r * 0.5

            # Session regime sizing (lunch skip, AM/PM multipliers)
            if sr_enabled:
                if et_frac < sr_am_end:
                    sr_mult = sr_am_mult
                elif sr_lunch_start <= et_frac < sr_lunch_end:
                    sr_mult = sr_lunch_mult
                elif et_frac >= sr_pm_start:
                    sr_mult = sr_pm_mult
                else:
                    sr_mult = 1.0
                r_amount *= sr_mult
                if r_amount <= 0.0:
                    continue  # session regime skips this trade (mult=0)

            # Apply slippage to entry BEFORE position sizing
            if direction == 1:
                actual_entry = entry_p + slippage_points
            else:
                actual_entry = entry_p - slippage_points

            # AUDIT FIX: compute stop_dist from actual_entry (post-slippage)
            # This ensures position sizing, TP, and R-multiple all use the same anchor.
            stop_dist = abs(actual_entry - stop)
            if stop_dist < 1.0:
                continue

            risk_per_contract = stop_dist * point_value
            contracts = max(1, int(r_amount / risk_per_contract))
            if contracts <= 0:
                continue

            # --- TP multiplier (direction-aware when enabled) ---
            if ny_tp_enabled:
                actual_tp_mult = ny_tp_mult
                if dir_mgmt_enabled:
                    if direction == 1:
                        actual_tp_mult = dir_mgmt_long_tp_mult
                    else:
                        actual_tp_mult = dir_mgmt_short_tp_mult
                # MSS Long override: use mss_management.long_tp_mult
                if mss_mgmt_enabled and is_mss_signal and direction == 1:
                    actual_tp_mult = mss_long_tp_mult
                if 9.5 <= et_frac < 16.0:
                    # AUDIT FIX: anchor TP from actual_entry, not entry_p
                    if direction == 1:
                        tp_distance = tp1 - actual_entry
                        tp1 = actual_entry + tp_distance * actual_tp_mult
                    else:
                        tp_distance = actual_entry - tp1
                        tp1 = actual_entry - tp_distance * actual_tp_mult

            # --- Dual-mode short TP override ---
            # Shorts use a fixed R:R scalp target instead of swing-based TP.
            if dual_mode_enabled and direction == -1:
                short_rr = dual_mode_short_rr
                # MSS Short override
                if mss_mgmt_enabled and is_mss_signal:
                    short_rr = mss_short_rr
                # AUDIT FIX: anchor from actual_entry
                tp1 = actual_entry - stop_dist * short_rr

            # --- Direction-aware trim percentage ---
            # MSS override applied on top if mss_management is enabled
            cur_trim_pct = trim_pct
            if mss_mgmt_enabled and is_mss_signal:
                if direction == -1:
                    cur_trim_pct = mss_short_trim
                else:
                    cur_trim_pct = mss_long_trim
            elif dual_mode_enabled and direction == -1:
                cur_trim_pct = dual_mode_short_trim
            elif dir_mgmt_enabled:
                if direction == 1:
                    cur_trim_pct = dir_mgmt_long_trim
                else:
                    cur_trim_pct = dir_mgmt_short_trim

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
            pos_signal_type_int = sig_type_int[i]
            pos_bias_dir = bias_dir_arr[i]
            pos_regime = regime_arr[i]
            pos_model_prob = prob
            pos_grade_int = grade_int
            pos_trim_pct = cur_trim_pct

    # ---- Force-close any open position at end ----
    if in_position:
        exit_price = close_arr[n - 1]
        if pos_direction == 1:
            pnl_pts = exit_price - pos_entry_price
        else:
            pnl_pts = pos_entry_price - exit_price

        total_pnl = pnl_pts * point_value * pos_remaining_contracts
        total_pnl -= commission_per_side * 2.0 * pos_remaining_contracts
        stop_dist = abs(pos_entry_price - pos_stop)
        total_risk = stop_dist * point_value * pos_contracts
        r_mult = total_pnl / total_risk if total_risk > 0.0 else 0.0

        if trade_count < max_trades:
            t_entry_idx[trade_count] = pos_entry_idx
            t_exit_idx[trade_count] = n - 1
            t_direction[trade_count] = pos_direction
            t_entry_price[trade_count] = pos_entry_price
            t_exit_price[trade_count] = exit_price
            t_stop_price[trade_count] = pos_stop
            t_tp1_price[trade_count] = pos_tp1
            t_contracts[trade_count] = pos_contracts
            t_pnl_points[trade_count] = pnl_pts
            t_pnl_dollars[trade_count] = total_pnl
            t_r_multiple[trade_count] = r_mult
            t_exit_reason[trade_count] = 4  # eod_close
            t_signal_type[trade_count] = pos_signal_type_int
            t_bias_dir[trade_count] = pos_bias_dir
            t_regime[trade_count] = pos_regime
            t_model_prob[trade_count] = pos_model_prob
            t_grade[trade_count] = pos_grade_int
            t_trimmed[trade_count] = pos_trimmed
            trade_count += 1

    return (
        t_entry_idx[:trade_count],
        t_exit_idx[:trade_count],
        t_direction[:trade_count],
        t_entry_price[:trade_count],
        t_exit_price[:trade_count],
        t_stop_price[:trade_count],
        t_tp1_price[:trade_count],
        t_contracts[:trade_count],
        t_pnl_points[:trade_count],
        t_pnl_dollars[:trade_count],
        t_r_multiple[:trade_count],
        t_exit_reason[:trade_count],
        t_signal_type[:trade_count],
        t_bias_dir[:trade_count],
        t_regime[:trade_count],
        t_model_prob[:trade_count],
        t_grade[:trade_count],
        t_trimmed[:trade_count],
    )


@numba.njit(cache=True)
def _compute_grade_jit(
    bias_aligned: numba.float64,
    bias_confidence: numba.float64,
    regime: numba.float64,
) -> numba.int64:
    """Grade: 0=C, 1=B+, 2=A+

    Recalibrated grading (removed bias_confidence — anti-predictive):
      A+ = aligned + full regime (1.0)
      B+ = aligned + partial regime, OR neutral + full regime
      C  = neutral + partial regime
    """
    if np.isnan(bias_aligned) or np.isnan(regime):
        return 0

    if regime == 0.0:
        return 0

    aligned = bias_aligned > 0.5
    full_regime = regime >= 1.0

    if aligned and full_regime:
        return 2  # A+
    elif aligned or full_regime:
        return 1  # B+
    else:
        return 0  # C


# ---------------------------------------------------------------------------
# Mapping tables for string <-> int conversions
# ---------------------------------------------------------------------------

_EXIT_REASON_MAP = {0: "stop", 1: "be_sweep", 2: "tp1", 3: "early_cut_pa", 4: "eod_close"}
_GRADE_MAP = {0: "C", 1: "B+", 2: "A+"}
_SIGNAL_TYPE_MAP = {0: "trend", 1: "mss"}
_SIGNAL_TYPE_TO_INT = {"trend": 0, "mss": 1}


# ---------------------------------------------------------------------------
# Public wrapper: run_backtest_jit
# ---------------------------------------------------------------------------

def precompute_backtest_arrays(
    df_5m: pd.DataFrame,
    signals_df: pd.DataFrame,
    params: dict,
) -> dict:
    """Pre-compute expensive arrays that can be reused across backtest runs.

    Call this once and pass the result to run_backtest_jit(precomputed=...)
    to avoid re-computing news filter, ATR, PA quality, swings, and datetime
    arrays on every call. Useful for parameter sweeps.

    Returns
    -------
    dict
        Pre-computed arrays to pass as `precomputed` kwarg.
    """
    n = len(df_5m)
    result: dict[str, Any] = {}

    # News filter
    news_params = params.get("news", {})
    calendar_path = Path(__file__).resolve().parent.parent / "config" / "news_calendar.csv"
    if calendar_path.exists() and news_params.get("blackout_minutes_before", 0) > 0:
        from features.news_filter import build_news_blackout_mask
        news_blackout = build_news_blackout_mask(
            df_5m.index,
            str(calendar_path),
            news_params.get("blackout_minutes_before", 60),
            news_params.get("cooldown_minutes_after", 5),
        )
        result["news_blackout_arr"] = news_blackout.values.astype(np.bool_)
        result["has_news_filter"] = True
    else:
        result["news_blackout_arr"] = np.empty(0, dtype=np.bool_)
        result["has_news_filter"] = False

    # ATR
    from features.displacement import compute_atr
    result["atr_arr"] = compute_atr(df_5m, period=14).values.astype(np.float64)

    # PA quality
    from features.pa_quality import compute_alternating_dir_ratio
    result["pa_alt_arr"] = compute_alternating_dir_ratio(df_5m, window=6).values.astype(np.float64)

    # Swing levels
    from features.swing import compute_swing_levels
    swing_params_dict = {
        "left_bars": params["swing"]["left_bars"],
        "right_bars": params["swing"]["right_bars"],
    }
    swings = compute_swing_levels(df_5m, swing_params_dict)
    swings["swing_high_price"] = swings["swing_high_price"].shift(1).ffill()
    swings["swing_low_price"] = swings["swing_low_price"].shift(1).ffill()
    swing_high_mask_raw = swings["swing_high"].shift(1, fill_value=False).values.astype(np.bool_)
    swing_low_mask_raw = swings["swing_low"].shift(1, fill_value=False).values.astype(np.bool_)

    open_arr = df_5m["open"].values.astype(np.float64)
    high_arr = df_5m["high"].values.astype(np.float64)
    low_arr = df_5m["low"].values.astype(np.float64)
    close_arr = df_5m["close"].values.astype(np.float64)

    nth_swing = int(params["trail"]["use_nth_swing"])
    max_nth = max(nth_swing, 3)
    nth_swing_low_table, nth_swing_high_table = _precompute_nth_swing_table(
        swing_low_mask_raw, swing_high_mask_raw,
        low_arr, high_arr,
        np.int64(max_nth),
    )
    result["nth_swing_low_table"] = nth_swing_low_table
    result["nth_swing_high_table"] = nth_swing_high_table

    # Datetime arrays
    et_index = df_5m.index.tz_convert("US/Eastern")
    _et_local = et_index._data._local_timestamps()
    _unit = et_index._data.unit
    _ups = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}.get(_unit, 10**9)
    _seconds_of_day = ((_et_local // _ups) % 86400).astype(np.int64)
    et_hour_np = (_seconds_of_day // 3600).astype(np.int64)
    et_minute_np = ((_seconds_of_day % 3600) // 60).astype(np.int64)
    result["et_frac_arr"] = et_hour_np.astype(np.float64) + et_minute_np.astype(np.float64) / 60.0

    _et_days = (_et_local // (_ups * 86400)).astype(np.int64)
    _epoch_ordinal = np.int64(719163)
    ordinals = _et_days + _epoch_ordinal
    is_evening = et_hour_np >= 18
    session_date_int = ordinals.copy()
    session_date_int[is_evening] += 1
    result["session_date_int"] = session_date_int
    result["et_dow_arr"] = ((ordinals - 1) % 7).astype(np.int64)

    # Signal type int conversion
    sig_type_raw = signals_df["signal_type"].values
    result["sig_type_int"] = np.array(
        [_SIGNAL_TYPE_TO_INT.get(str(s), 0) for s in sig_type_raw],
        dtype=np.int64,
    )

    # OHLC arrays
    result["open_arr"] = open_arr
    result["high_arr"] = high_arr
    result["low_arr"] = low_arr
    result["close_arr"] = close_arr

    return result


def run_backtest_jit(
    df_5m: pd.DataFrame,
    signals_df: pd.DataFrame,
    bias_data: pd.DataFrame,
    regime: pd.Series,
    model: xgb.Booster,
    features_X: pd.DataFrame,
    params: dict | None = None,
    threshold: float = 0.50,
    precomputed: dict | None = None,
) -> pd.DataFrame:
    """JIT-optimized backtest — drop-in replacement for run_backtest().

    Same signature, same return format. All logic is identical to the
    original engine.py, but the inner loop is Numba-compiled.

    Parameters
    ----------
    precomputed : dict, optional
        Output of precompute_backtest_arrays(). When provided, skips
        expensive recomputation of news filter, ATR, swings, datetime
        arrays. Useful for parameter sweeps.
    """
    if params is None:
        params = _load_params()

    # Extract params
    pos_params = params["position"]
    risk_params = params["risk"]
    bt_params = params["backtest"]
    grading_params = params["grading"]
    trim_params = params["trim"]
    trail_params = params["trail"]

    normal_r = float(pos_params["normal_r"])
    reduced_r = float(pos_params["reduced_r"])
    point_value = float(pos_params["point_value"])
    daily_max_loss_r = float(risk_params["daily_max_loss_r"])
    max_consec_losses = int(risk_params["max_consecutive_losses"])

    commission_per_side = float(bt_params["commission_per_side_micro"])
    slippage_ticks = float(bt_params["slippage_normal_ticks"])
    slippage_points = slippage_ticks * 0.25

    a_plus_mult = float(grading_params["a_plus_size_mult"])
    b_plus_mult = float(grading_params["b_plus_size_mult"])
    c_skip = bool(grading_params["c_skip"])

    trim_pct = float(trim_params["pct"])
    be_after_trim = bool(trim_params["be_after_trim"])
    nth_swing = int(trail_params["use_nth_swing"])

    # ---- Use precomputed arrays if available ----
    pc = precomputed or {}
    n = len(df_5m)

    if pc:
        # Fast path: use cached arrays
        news_blackout_arr = pc["news_blackout_arr"]
        has_news_filter = pc["has_news_filter"]
        open_arr = pc["open_arr"]
        high_arr = pc["high_arr"]
        low_arr = pc["low_arr"]
        close_arr = pc["close_arr"]
        atr_arr = pc["atr_arr"]
        pa_alt_arr = pc["pa_alt_arr"]
        nth_swing_low_table = pc["nth_swing_low_table"]
        nth_swing_high_table = pc["nth_swing_high_table"]
        session_date_int = pc["session_date_int"]
        et_frac_arr = pc["et_frac_arr"]
        et_dow_arr = pc["et_dow_arr"]
        sig_type_int = pc["sig_type_int"]
    else:
        # Slow path: compute everything from scratch
        # News filter
        news_params = params.get("news", {})
        calendar_path = Path(__file__).resolve().parent.parent / "config" / "news_calendar.csv"
        has_news_filter = False
        if calendar_path.exists() and news_params.get("blackout_minutes_before", 0) > 0:
            from features.news_filter import build_news_blackout_mask
            news_blackout = build_news_blackout_mask(
                df_5m.index,
                str(calendar_path),
                news_params.get("blackout_minutes_before", 60),
                news_params.get("cooldown_minutes_after", 5),
            )
            news_blackout_arr = news_blackout.values.astype(np.bool_)
            has_news_filter = True
        else:
            news_blackout_arr = np.empty(0, dtype=np.bool_)

        # OHLC
        open_arr = df_5m["open"].values.astype(np.float64)
        high_arr = df_5m["high"].values.astype(np.float64)
        low_arr = df_5m["low"].values.astype(np.float64)
        close_arr = df_5m["close"].values.astype(np.float64)

        # ATR & PA quality
        from features.displacement import compute_atr
        atr_arr = compute_atr(df_5m, period=14).values.astype(np.float64)

        from features.pa_quality import compute_alternating_dir_ratio
        pa_alt_arr = compute_alternating_dir_ratio(df_5m, window=6).values.astype(np.float64)

        # Swing levels
        from features.swing import compute_swing_levels
        swing_params_dict = {
            "left_bars": params["swing"]["left_bars"],
            "right_bars": params["swing"]["right_bars"],
        }
        swings = compute_swing_levels(df_5m, swing_params_dict)
        swings["swing_high_price"] = swings["swing_high_price"].shift(1).ffill()
        swings["swing_low_price"] = swings["swing_low_price"].shift(1).ffill()
        swing_high_mask_raw = swings["swing_high"].shift(1, fill_value=False).values.astype(np.bool_)
        swing_low_mask_raw = swings["swing_low"].shift(1, fill_value=False).values.astype(np.bool_)

        max_nth = max(nth_swing, 3)
        nth_swing_low_table, nth_swing_high_table = _precompute_nth_swing_table(
            swing_low_mask_raw, swing_high_mask_raw,
            low_arr, high_arr,
            np.int64(max_nth),
        )

        # Signal type
        sig_type_raw = signals_df["signal_type"].values
        sig_type_int = np.array(
            [_SIGNAL_TYPE_TO_INT.get(str(s), 0) for s in sig_type_raw],
            dtype=np.int64,
        )

        # Datetime arrays
        et_index = df_5m.index.tz_convert("US/Eastern")
        _et_local = et_index._data._local_timestamps()
        _unit = et_index._data.unit
        _ups = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}.get(_unit, 10**9)
        _seconds_of_day = ((_et_local // _ups) % 86400).astype(np.int64)
        et_hour_np = (_seconds_of_day // 3600).astype(np.int64)
        et_minute_np = ((_seconds_of_day % 3600) // 60).astype(np.int64)
        et_frac_arr = et_hour_np.astype(np.float64) + et_minute_np.astype(np.float64) / 60.0

        _et_days = (_et_local // (_ups * 86400)).astype(np.int64)
        _epoch_ordinal = np.int64(719163)
        ordinals = _et_days + _epoch_ordinal
        is_evening = et_hour_np >= 18
        session_date_int = ordinals.copy()
        session_date_int[is_evening] += 1
        et_dow_arr = ((ordinals - 1) % 7).astype(np.int64)

    # ---- Signal arrays (always from signals_df, not cacheable) ----
    sig_mask = signals_df["signal"].values.astype(np.bool_)
    sig_dir = signals_df["signal_dir"].values.astype(np.float64)
    entry_price_arr = signals_df["entry_price"].values.astype(np.float64)
    model_stop_arr = signals_df["model_stop"].values.astype(np.float64)
    irl_target_arr = signals_df["irl_target"].values.astype(np.float64)

    # has_smt extraction (with warning when missing)
    if "has_smt" in signals_df.columns:
        has_smt_arr = signals_df["has_smt"].values.astype(np.bool_)
    else:
        has_smt_arr = np.zeros(n, dtype=np.bool_)
        if params.get("smt", {}).get("enabled", False):
            logger.warning("SMT enabled but 'has_smt' column missing — all SMT checks will be False")

    # ---- Bias/regime arrays ----
    bias_dir_arr = bias_data["bias_direction"].values.astype(np.float64)
    bias_conf_arr = bias_data["bias_confidence"].values.astype(np.float64)
    regime_arr = regime.values.astype(np.float64)

    # ---- Model predictions ----
    signal_indices = np.where(sig_mask)[0]
    model_probs = np.full(n, np.nan, dtype=np.float64)
    if len(signal_indices) > 0:
        signal_times = df_5m.index[signal_indices]
        common_mask = signal_times.isin(features_X.index)
        valid_signal_idx = signal_indices[common_mask]
        valid_signal_times = signal_times[common_mask]

        if len(valid_signal_times) > 0:
            X_sig = features_X.loc[valid_signal_times]
            dmat = xgb.DMatrix(X_sig, feature_names=list(features_X.columns))
            preds = model.predict(dmat)
            model_probs[valid_signal_idx] = preds.astype(np.float64)

    # ---- Signal quality score (Sprint 4 Attack 1) ----
    # Pre-compute composite quality at each signal bar (same logic as engine.py).
    sq_params = params.get("signal_quality", {})
    sq_enabled = bool(sq_params.get("enabled", False))
    sq_threshold = float(sq_params.get("threshold", 0.66))
    sq_w_size = sq_params.get("w_size", 0.30)
    sq_w_disp = sq_params.get("w_disp", 0.30)
    sq_w_flu = sq_params.get("w_flu", 0.20)
    sq_w_pa = sq_params.get("w_pa", 0.20)

    signal_quality = np.full(n, np.nan, dtype=np.float64)
    if sq_enabled and len(signal_indices) > 0:
        from features.displacement import compute_fluency
        fluency_arr = compute_fluency(df_5m, params).values

        for idx_val in signal_indices:
            a = atr_arr[idx_val] if not np.isnan(atr_arr[idx_val]) else 10.0

            # 1. Size: entry-stop distance / (ATR * 1.5), capped at 1.0
            gap = abs(entry_price_arr[idx_val] - model_stop_arr[idx_val])
            size_sc = min(1.0, gap / (a * 1.5)) if a > 0 else 0.5

            # 2. Displacement: signal candle body/range
            body = abs(close_arr[idx_val] - open_arr[idx_val])
            rng = high_arr[idx_val] - low_arr[idx_val]
            disp_sc = body / rng if rng > 0 else 0.0

            # 3. Fluency (from pre-computed fluency array)
            flu_val = fluency_arr[idx_val]
            flu_sc = min(1.0, max(0.0, flu_val)) if not np.isnan(flu_val) else 0.5

            # 4. PA cleanliness: 1 - alternating direction ratio (6-bar window)
            window = 6
            if idx_val >= window:
                dirs = np.sign(close_arr[idx_val - window:idx_val] - open_arr[idx_val - window:idx_val])
                alt = np.sum(dirs[1:] != dirs[:-1]) / (window - 1)
                pa_sc = 1.0 - alt
            else:
                pa_sc = 0.5

            signal_quality[idx_val] = (
                sq_w_size * size_sc
                + sq_w_disp * disp_sc
                + sq_w_flu * flu_sc
                + sq_w_pa * pa_sc
            )

        sq_valid = signal_quality[~np.isnan(signal_quality)]
        logger.info(
            "Signal quality: enabled, threshold=%.2f, scored %d signals, "
            "mean=%.3f, p25=%.3f, p75=%.3f",
            sq_threshold, len(sq_valid),
            sq_valid.mean() if len(sq_valid) > 0 else 0,
            np.percentile(sq_valid, 25) if len(sq_valid) > 0 else 0,
            np.percentile(sq_valid, 75) if len(sq_valid) > 0 else 0,
        )

    # ---- Extract remaining scalar params ----
    pa_threshold = float(params.get("pa_quality", {}).get("alt_dir_threshold", 0.334))
    min_stop_atr_mult = float(params.get("regime", {}).get("min_stop_atr_mult", 0.5))

    session_rules = params.get("session_rules", {})
    ny_tp_enabled = bool(session_rules.get("enabled", False))
    ny_tp_mult = float(session_rules.get("ny_tp_multiplier", 1.0))

    # Session filter params
    session_filter = params.get("session_filter", {})
    sf_enabled = bool(session_filter.get("enabled", False))
    sf_ny_direction = np.int64(session_filter.get("ny_direction", 0))
    sf_london_direction = np.int64(session_filter.get("london_direction", 0))
    sf_skip_asia = bool(session_filter.get("skip_asia", True))
    sf_skip_london = bool(session_filter.get("skip_london", False))

    # ---- Sprint 4-5 scalar params (extracted before JIT) ----
    smt_cfg = params.get("smt", {})
    smt_enabled = bool(smt_cfg.get("enabled", False))
    smt_bypass_session = bool(smt_cfg.get("bypass_session_filter", False))

    bias_relax = params.get("bias_relaxation", {})
    bias_relax_enabled = bool(bias_relax.get("enabled", False))

    dual_mode = params.get("dual_mode", {})
    dual_mode_enabled = bool(dual_mode.get("enabled", False))
    dual_mode_short_sq = float(dual_mode.get("short_sq_threshold", 0.80))
    dual_mode_short_rr = float(dual_mode.get("short_rr", 0.625))
    dual_mode_short_trim = float(dual_mode.get("short_trim_pct", 1.0))

    dir_mgmt = params.get("direction_mgmt", {})
    dir_mgmt_enabled = bool(dir_mgmt.get("enabled", False))
    dir_mgmt_long_tp_mult = float(dir_mgmt.get("long_tp_mult", ny_tp_mult))
    dir_mgmt_short_tp_mult = float(dir_mgmt.get("short_tp_mult", 1.25))
    dir_mgmt_long_trim = float(dir_mgmt.get("long_trim_pct", trim_pct))
    dir_mgmt_short_trim = float(dir_mgmt.get("short_trim_pct", 1.0))

    mss_mgmt = params.get("mss_management", {})
    mss_mgmt_enabled = bool(mss_mgmt.get("enabled", False))
    mss_long_tp_mult = float(mss_mgmt.get("long_tp_mult", ny_tp_mult))
    mss_short_rr = float(mss_mgmt.get("short_rr", dual_mode_short_rr))
    mss_short_trim = float(mss_mgmt.get("short_trim_pct", 1.0))
    mss_long_trim = float(mss_mgmt.get("long_trim_pct", trim_pct))

    session_regime = params.get("session_regime", {})
    sr_enabled = bool(session_regime.get("enabled", False))
    sr_am_end = float(session_regime.get("am_end", 12.0))
    sr_lunch_start = float(session_regime.get("lunch_start", 12.0))
    sr_lunch_end = float(session_regime.get("lunch_end", 13.5))
    sr_pm_start = float(session_regime.get("pm_start", 13.5))
    sr_am_mult = float(session_regime.get("am_mult", 1.0))
    sr_lunch_mult = float(session_regime.get("lunch_mult", 0.5))
    sr_pm_mult = float(session_regime.get("pm_mult", 0.75))

    sig_filter = params.get("signal_filter", {})
    allow_mss = bool(sig_filter.get("allow_mss", True))
    allow_trend = bool(sig_filter.get("allow_trend", True))

    # ---- Call JIT core ----
    result = _backtest_core(
        open_arr, high_arr, low_arr, close_arr,
        sig_mask, sig_dir, sig_type_int, entry_price_arr, model_stop_arr, irl_target_arr,
        has_smt_arr,
        bias_dir_arr, bias_conf_arr, regime_arr,
        atr_arr, pa_alt_arr,
        signal_quality,
        nth_swing_low_table, nth_swing_high_table,
        model_probs,
        session_date_int, et_frac_arr, et_dow_arr,
        news_blackout_arr, has_news_filter,
        normal_r, reduced_r, point_value,
        daily_max_loss_r, np.int64(max_consec_losses),
        commission_per_side, slippage_points,
        a_plus_mult, b_plus_mult, c_skip,
        trim_pct, be_after_trim, np.int64(nth_swing),
        pa_threshold, min_stop_atr_mult,
        ny_tp_mult, ny_tp_enabled,
        threshold,
        sf_enabled, sf_ny_direction, sf_london_direction, sf_skip_asia, sf_skip_london,
        # Sprint 4-5 params
        smt_enabled, smt_bypass_session,
        bias_relax_enabled,
        sq_enabled, sq_threshold,
        dual_mode_enabled, dual_mode_short_sq,
        dual_mode_short_rr, dual_mode_short_trim,
        dir_mgmt_enabled, dir_mgmt_long_tp_mult, dir_mgmt_short_tp_mult,
        dir_mgmt_long_trim, dir_mgmt_short_trim,
        mss_mgmt_enabled, mss_long_tp_mult, mss_short_rr, mss_short_trim, mss_long_trim,
        sr_enabled, sr_am_end, sr_lunch_start, sr_lunch_end, sr_pm_start,
        sr_am_mult, sr_lunch_mult, sr_pm_mult,
        allow_mss, allow_trend,
    )

    (
        t_entry_idx, t_exit_idx, t_direction,
        t_entry_price, t_exit_price, t_stop_price, t_tp1_price,
        t_contracts, t_pnl_points, t_pnl_dollars, t_r_multiple,
        t_exit_reason, t_signal_type, t_bias_dir, t_regime,
        t_model_prob, t_grade, t_trimmed,
    ) = result

    if len(t_entry_idx) == 0:
        logger.warning("Backtest produced 0 trades (threshold=%.2f)", threshold)
        return pd.DataFrame()

    # ---- Build DataFrame from arrays ----
    idx = df_5m.index
    trade_df = pd.DataFrame({
        "entry_time": pd.to_datetime([idx[j] for j in t_entry_idx]),
        "exit_time": pd.to_datetime([idx[j] for j in t_exit_idx]),
        "direction": t_direction,
        "entry_price": t_entry_price,
        "exit_price": t_exit_price,
        "stop_price": t_stop_price,
        "tp1_price": t_tp1_price,
        "contracts": t_contracts,
        "pnl_points": t_pnl_points,
        "pnl_dollars": t_pnl_dollars,
        "r_multiple": t_r_multiple,
        "exit_reason": [_EXIT_REASON_MAP.get(int(r), "unknown") for r in t_exit_reason],
        "signal_type": [_SIGNAL_TYPE_MAP.get(int(s), "unknown") for s in t_signal_type],
        "bias_direction": t_bias_dir,
        "regime": t_regime,
        "model_prob": t_model_prob,
        "grade": [_GRADE_MAP.get(int(g), "C") for g in t_grade],
        "trimmed": t_trimmed,
    })

    logger.info(
        "JIT Backtest complete: %d trades, threshold=%.2f, "
        "win_rate=%.1f%%, total_pnl=$%.0f, avg_R=%.2f",
        len(trade_df),
        threshold,
        100.0 * (trade_df["pnl_dollars"] > 0).mean(),
        trade_df["pnl_dollars"].sum(),
        trade_df["r_multiple"].mean(),
    )

    return trade_df
