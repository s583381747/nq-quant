"""
experiments/multi_level_tp.py — Multi-level take-profit system using liquidity levels.

Instead of single TP with 50% trim, uses MULTIPLE TP levels at successive liquidity points:
  - TP1: nearest swing target (IRL target), trim 25%
  - TP2: next liquidity level (session high OR 2nd/HTF swing), trim 25%
  - TP3: run remaining 50% with trailing stop (nth_swing=3)

Combines "lower trim is better" finding with actual liquidity targets for exits.

Variants tested:
  V1: Multi-TP, late BE (BE only after TP2)
  V2: Multi-TP, early BE (BE after TP1)
  V3: Multi-TP, aggressive split (33/33/34%)
  V4: Single TP with lower trim (control — trim=0.25, nth_swing=3)
  V5: Baseline Config D (reference)

Config D base: sq_short=0.80, block_pm_shorts=True

Usage: python experiments/multi_level_tp.py
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
from features.swing import compute_swing_levels


# ======================================================================
# Data preparation: load session levels + HTF swings
# ======================================================================
def prepare_liquidity_data(d: dict) -> dict:
    """Augment the data dict with session levels and HTF swing levels."""
    t0 = _time.perf_counter()
    nq = d["nq"]

    # --- Session levels ---
    sl_path = PROJECT / "data" / "cache_session_levels_10yr_v2.parquet"
    session_levels = pd.read_parquet(sl_path)
    # These are already forward-filled / computed per-bar

    sess_asia_high = session_levels["asia_high"].values
    sess_asia_low = session_levels["asia_low"].values
    sess_london_high = session_levels["london_high"].values
    sess_london_low = session_levels["london_low"].values
    sess_overnight_high = session_levels["overnight_high"].values
    sess_overnight_low = session_levels["overnight_low"].values

    # --- Regular swings (already in d, but we need the price arrays) ---
    swing_high_prices = d["nq"]["high"].values.copy()
    swing_low_prices = d["nq"]["low"].values.copy()

    # Recompute swing levels to get the forward-filled price series
    swing_p = {"left_bars": d["params"]["swing"]["left_bars"],
               "right_bars": d["params"]["swing"]["right_bars"]}
    swings_reg = compute_swing_levels(nq, swing_p)
    swings_reg["swing_high_price"] = swings_reg["swing_high_price"].shift(1).ffill()
    swings_reg["swing_low_price"] = swings_reg["swing_low_price"].shift(1).ffill()
    reg_swing_high_price = swings_reg["swing_high_price"].values
    reg_swing_low_price = swings_reg["swing_low_price"].values

    # --- HTF swings (wider parameters for bigger structural levels) ---
    htf_swing_p = {"left_bars": 10, "right_bars": 3}
    swings_htf = compute_swing_levels(nq, htf_swing_p)
    swings_htf["swing_high_price"] = swings_htf["swing_high_price"].shift(1).ffill()
    swings_htf["swing_low_price"] = swings_htf["swing_low_price"].shift(1).ffill()
    htf_swing_high_price = swings_htf["swing_high_price"].values
    htf_swing_low_price = swings_htf["swing_low_price"].values
    htf_swing_high_mask = swings_htf["swing_high"].shift(1, fill_value=False).values
    htf_swing_low_mask = swings_htf["swing_low"].shift(1, fill_value=False).values

    # --- Build nth-swing lookup arrays for regular swings ---
    # We need to find 2nd and 3rd swing above/below for each bar
    # Precompute: for each bar, the last N swing high/low prices
    n = d["n"]

    # For finding multiple swing levels, we'll use the mask + price approach
    # but collect top N distinct levels
    reg_sh_mask = d["swing_high_mask"]
    reg_sl_mask = d["swing_low_mask"]
    reg_high_arr = nq["high"].values
    reg_low_arr = nq["low"].values

    elapsed = _time.perf_counter() - t0
    print(f"[LIQUIDITY] Session levels + HTF swings loaded in {elapsed:.1f}s")

    d_extra = {
        "sess_asia_high": sess_asia_high,
        "sess_asia_low": sess_asia_low,
        "sess_london_high": sess_london_high,
        "sess_london_low": sess_london_low,
        "sess_overnight_high": sess_overnight_high,
        "sess_overnight_low": sess_overnight_low,
        "reg_swing_high_price": reg_swing_high_price,
        "reg_swing_low_price": reg_swing_low_price,
        "htf_swing_high_price": htf_swing_high_price,
        "htf_swing_low_price": htf_swing_low_price,
        "htf_swing_high_mask": htf_swing_high_mask,
        "htf_swing_low_mask": htf_swing_low_mask,
        "reg_sh_mask": reg_sh_mask,
        "reg_sl_mask": reg_sl_mask,
        "reg_high_arr": reg_high_arr,
        "reg_low_arr": reg_low_arr,
    }
    return d_extra


# ======================================================================
# Liquidity ladder builder
# ======================================================================
def _find_nth_swing_price(mask, prices, idx, n_val):
    """Find the nth swing point price looking backward from idx."""
    count = 0
    for j in range(idx - 1, -1, -1):
        if mask[j]:
            count += 1
            if count == n_val:
                return prices[j]
    return np.nan


def build_liquidity_ladder_long(
    entry_price: float,
    stop_price: float,
    irl_target: float,
    idx: int,
    d_extra: dict,
    min_r_dist: float = 0.5,
) -> tuple[float, float]:
    """Build TP1 and TP2 for a LONG trade.

    TP1 = IRL target (nearest swing above entry)
    TP2 = next distinct liquidity level above TP1
          Candidates: session highs, HTF swing high, 2nd regular swing high

    Returns (tp1, tp2).
    """
    stop_dist = abs(entry_price - stop_price)
    min_dist = stop_dist * min_r_dist  # TP1 must be at least 0.5R away

    # TP1 = IRL target
    tp1 = irl_target
    if tp1 - entry_price < min_dist:
        tp1 = entry_price + min_dist

    # Gather candidate levels above TP1 for TP2
    candidates = []

    # Session highs
    for arr in [d_extra["sess_asia_high"], d_extra["sess_london_high"],
                d_extra["sess_overnight_high"]]:
        val = arr[idx]
        if not np.isnan(val) and val > tp1 + 1.0:
            candidates.append(val)

    # HTF swing high
    htf_sh = d_extra["htf_swing_high_price"][idx]
    if not np.isnan(htf_sh) and htf_sh > tp1 + 1.0:
        candidates.append(htf_sh)

    # 2nd regular swing high (looking back)
    sh2 = _find_nth_swing_price(d_extra["reg_sh_mask"], d_extra["reg_high_arr"], idx, 2)
    if not np.isnan(sh2) and sh2 > tp1 + 1.0:
        candidates.append(sh2)

    # 3rd regular swing high
    sh3 = _find_nth_swing_price(d_extra["reg_sh_mask"], d_extra["reg_high_arr"], idx, 3)
    if not np.isnan(sh3) and sh3 > tp1 + 1.0:
        candidates.append(sh3)

    if candidates:
        # Pick the closest level above TP1
        tp2 = min(candidates)
    else:
        # Fallback: TP2 = TP1 * 1.5 distance from entry
        tp2 = entry_price + (tp1 - entry_price) * 1.5

    return tp1, tp2


def build_liquidity_ladder_short(
    entry_price: float,
    stop_price: float,
    irl_target: float,
    idx: int,
    d_extra: dict,
    min_r_dist: float = 0.5,
) -> tuple[float, float]:
    """Build TP1 and TP2 for a SHORT trade.

    TP1 = IRL target (nearest swing below entry)
    TP2 = next distinct liquidity level below TP1
    """
    stop_dist = abs(entry_price - stop_price)
    min_dist = stop_dist * min_r_dist

    tp1 = irl_target
    if entry_price - tp1 < min_dist:
        tp1 = entry_price - min_dist

    # Gather candidates below TP1
    candidates = []

    for arr in [d_extra["sess_asia_low"], d_extra["sess_london_low"],
                d_extra["sess_overnight_low"]]:
        val = arr[idx]
        if not np.isnan(val) and val < tp1 - 1.0:
            candidates.append(val)

    htf_sl = d_extra["htf_swing_low_price"][idx]
    if not np.isnan(htf_sl) and htf_sl < tp1 - 1.0:
        candidates.append(htf_sl)

    sl2 = _find_nth_swing_price(d_extra["reg_sl_mask"], d_extra["reg_low_arr"], idx, 2)
    if not np.isnan(sl2) and sl2 < tp1 - 1.0:
        candidates.append(sl2)

    sl3 = _find_nth_swing_price(d_extra["reg_sl_mask"], d_extra["reg_low_arr"], idx, 3)
    if not np.isnan(sl3) and sl3 < tp1 - 1.0:
        candidates.append(sl3)

    if candidates:
        tp2 = max(candidates)  # Closest below TP1
    else:
        tp2 = entry_price - (entry_price - tp1) * 1.5

    return tp1, tp2


# ======================================================================
# Multi-level TP backtest engine
# ======================================================================
def run_backtest_multi_tp(
    d: dict,
    d_extra: dict,
    # Config D base
    sq_long: float = 0.68,
    sq_short: float = 0.80,
    min_stop_atr: float = 1.7,
    block_pm_shorts: bool = True,
    # === MULTI-TP PARAMETERS ===
    tp1_trim_pct: float = 0.25,     # Fraction trimmed at TP1
    tp2_trim_pct: float = 0.25,     # Fraction trimmed at TP2
    be_after_tp1: bool = False,     # Move to BE after TP1 (True=early, False=late)
    be_after_tp2: bool = True,      # Move to BE after TP2
    trail_nth_swing: int = 3,       # Trail stop using nth swing
    use_multi_tp_for_shorts: bool = False,  # Apply multi-TP to shorts too?
    # NY TP multiplier (applied to IRL target for TP1 base)
    ny_tp_mult: float = 2.0,
    mss_long_tp_mult: float = 2.5,
    # Date range
    start_date=None, end_date=None,
) -> list[dict]:
    """Backtest with multi-level take-profit for longs."""

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
    pos_trim_stage = 0  # 0=untrimmed, 1=TP1 trimmed, 2=TP2 trimmed
    pos_be_stop = pos_trail_stop = 0.0
    pos_signal_type = pos_grade = ""
    pos_bias_dir = pos_regime = 0.0
    pos_is_multi_tp = False  # Whether this position uses multi-TP
    pos_trim1_contracts = 0  # Contracts trimmed at TP1
    pos_trim2_contracts = 0  # Contracts trimmed at TP2
    pos_tp1_pnl_pts = 0.0   # PnL points locked in at TP1
    pos_tp2_pnl_pts = 0.0   # PnL points locked in at TP2
    # For single-TP (shorts / fallback)
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

            bars_in_trade = i - pos_entry_idx

            # --- MANDATORY EOD CLOSE at 15:55 ET ---
            if et_frac_arr[i] >= 15.917:  # 15:55 ET
                exit_price = c_arr[i] - slippage_points if pos_direction == 1 else c_arr[i] + slippage_points
                exit_reason = "eod_close"
                exited = True

            # --- Early cut on bad PA (same as baseline, only pre-trim) ---
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
                    tc1 = max(1, (pos_contracts + 1) // 2)  # ceil div: TP1 gets extra on odd counts
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
                    # FIX: take ALL remaining contracts at TP2 to prevent phantom runners
                    tc2 = pos_remaining_contracts
                    pos_trim2_contracts = tc2
                    pos_remaining_contracts -= tc2
                    pos_tp2_pnl_pts = pos_tp2 - pos_entry_price
                    pos_trim_stage = 2
                    if be_after_tp2:
                        pos_be_stop = pos_entry_price
                    # Set trailing stop
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

                # Stage 2: Trail runner
                if not exited and pos_trim_stage == 2:
                    nt = _find_nth_swing(swing_low_mask, low_arr, i, trail_nth_swing, 1)
                    if not np.isnan(nt) and nt > pos_trail_stop:
                        pos_trail_stop = nt

            # ==================================================================
            # MULTI-TP EXIT MANAGEMENT (SHORTS) — only if use_multi_tp_for_shorts
            # ==================================================================
            elif not exited and pos_is_multi_tp and pos_direction == -1:
                # Determine effective stop
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
                    tc1 = max(1, (pos_contracts + 1) // 2)  # ceil div: TP1 gets extra on odd counts
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
                    # FIX: take ALL remaining contracts at TP2 to prevent phantom runners
                    tc2 = pos_remaining_contracts
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
                    # Multi-TP PnL accounting
                    total_pnl = 0.0
                    if pos_trim_stage >= 1:
                        # TP1 trim PnL
                        total_pnl += pos_tp1_pnl_pts * point_value * pos_trim1_contracts
                    if pos_trim_stage >= 2:
                        # TP2 trim PnL
                        total_pnl += pos_tp2_pnl_pts * point_value * pos_trim2_contracts
                    # Runner PnL (remaining contracts) — FIX: use actual remaining, not exit_contracts
                    total_pnl += pnl_pts_runner * point_value * pos_remaining_contracts
                    # Commission on all contracts
                    total_comm = commission_per_side * 2 * pos_contracts
                    total_pnl -= total_comm
                elif pos_single_trimmed and exit_reason != "tp1":
                    # Single-TP trimmed exit
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
                if exit_reason == "be_sweep" and any_trimmed:
                    pass
                elif exit_reason == "eod_close":
                    pass  # EOD close is neutral, don't count for 0-for-2
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
        # TP COMPUTATION
        # ==================================================================
        is_mss_signal = str(sig_type[i]) == "mss"

        # Determine if this trade gets multi-TP
        apply_multi = (direction == 1) or (direction == -1 and use_multi_tp_for_shorts)

        if apply_multi:
            # Apply NY TP multiplier to raw IRL target to get TP1 base
            tp1_base = tp1_raw
            if session_rules.get("enabled", False) and 9.5 <= et_frac < 16.0:
                actual_tp_mult = ny_tp_mult
                if mss_mgmt_enabled and is_mss_signal and direction == 1:
                    actual_tp_mult = mss_long_tp_mult
                tp_distance = (tp1_raw - actual_entry) if direction == 1 else (actual_entry - tp1_raw)
                tp1_base = (actual_entry + tp_distance * actual_tp_mult) if direction == 1 else (actual_entry - tp_distance * actual_tp_mult)

            # Build liquidity ladder
            if direction == 1:
                tp1, tp2 = build_liquidity_ladder_long(
                    actual_entry, stop, tp1_base, i, d_extra)
            else:
                tp1, tp2 = build_liquidity_ladder_short(
                    actual_entry, stop, tp1_base, i, d_extra)

            # Enter position with multi-TP
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
            # SHORT with single-TP (current system behavior)
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
        })

    return trades


# ======================================================================
# Single-TP control backtest (V4)
# ======================================================================
def run_backtest_single_tp_control(
    d: dict,
    sq_long: float = 0.68,
    sq_short: float = 0.80,
    min_stop_atr: float = 1.7,
    block_pm_shorts: bool = True,
    long_trim_pct: float = 0.25,
    nth_swing: int = 3,
    start_date=None, end_date=None,
) -> list[dict]:
    """Single TP with lower trim + wider trail (control for comparison)."""
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

    low_arr = nq["low"].values
    high_arr = nq["high"].values

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
    pos_trim_pct = 0.50

    for i in range(start_idx, end_idx):
        bar_date = dates[i]
        if bar_date != current_date:
            current_date = bar_date
            daily_pnl_r = 0.0
            consecutive_losses = 0
            day_stopped = False

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
                        pos_trail_stop = _find_nth_swing(swing_low_mask, low_arr, i, nth_swing, 1)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True
                if pos_trimmed and not exited:
                    nt = _find_nth_swing(swing_low_mask, low_arr, i, nth_swing, 1)
                    if not np.isnan(nt) and nt > pos_trail_stop:
                        pos_trail_stop = nt

            elif not exited:
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
                        pos_trail_stop = _find_nth_swing(swing_high_mask, high_arr, i, nth_swing, -1)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True
                if pos_trimmed and not exited:
                    nt = _find_nth_swing(swing_high_mask, high_arr, i, nth_swing, -1)
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

        # TP computation — same as baseline but with configurable params
        if session_rules.get("enabled", False):
            if dir_mgmt.get("enabled", False):
                actual_tp_mult = dir_mgmt.get("long_tp_mult", 2.0) if direction == 1 else dir_mgmt.get("short_tp_mult", 1.25)
            else:
                actual_tp_mult = 2.0  # ny_tp_mult
            if mss_mgmt_enabled and is_mss_signal and direction == 1:
                actual_tp_mult = mss_mgmt.get("long_tp_mult", 2.5)
            if 9.5 <= et_frac < 16.0:
                tp_distance = (tp1 - actual_entry) if direction == 1 else (actual_entry - tp1)
                tp1 = (actual_entry + tp_distance * actual_tp_mult) if direction == 1 else (actual_entry - tp_distance * actual_tp_mult)

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
        pos_contracts = contracts
        pos_remaining_contracts = contracts
        pos_trimmed = False
        pos_be_stop = 0.0
        pos_trail_stop = 0.0
        pos_signal_type = str(sig_type[i])
        pos_grade = grade

        # Trim pct — use long_trim_pct override for longs
        if direction == 1:
            pos_trim_pct = long_trim_pct
        elif mss_mgmt_enabled and is_mss_signal:
            pos_trim_pct = mss_mgmt.get("short_trim_pct", 1.0)
        elif dual_mode_enabled:
            pos_trim_pct = dual_mode.get("short_trim_pct", 1.0)
        else:
            pos_trim_pct = trim_params["pct"]

    # Force close
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
# Diagnostics
# ======================================================================
def print_trim_stage_breakdown(trades_list, label):
    """Print how many trades reached each trim stage (multi-TP only)."""
    if not trades_list:
        return
    stages = [t.get("trim_stage", -1) for t in trades_list]
    longs = [t for t in trades_list if t["dir"] == 1]
    shorts = [t for t in trades_list if t["dir"] == -1]

    long_stages = [t.get("trim_stage", -1) for t in longs]
    n_s0 = sum(1 for s in long_stages if s == 0)
    n_s1 = sum(1 for s in long_stages if s == 1)
    n_s2 = sum(1 for s in long_stages if s >= 2)

    print(f"  {label} — Long trim stages: S0(stop)={n_s0}, S1(TP1 only)={n_s1}, S2+(TP2+runner)={n_s2} / {len(longs)} longs")

    if longs:
        long_r = np.array([t["r"] for t in longs])
        print(f"    Longs:  R={long_r.sum():+.1f}, avgR={long_r.mean():+.4f}, WR={100*(long_r>0).mean():.1f}%")
    if shorts:
        short_r = np.array([t["r"] for t in shorts])
        print(f"    Shorts: R={short_r.sum():+.1f}, avgR={short_r.mean():+.4f}, WR={100*(short_r>0).mean():.1f}%")

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
    print("=" * 110)
    print("MULTI-LEVEL TAKE-PROFIT EXPERIMENT")
    print("Base: Config D (sq_short=0.80, block_pm_shorts=True)")
    print("=" * 110)

    d = load_all()
    d_extra = prepare_liquidity_data(d)

    # ==================================================================
    # V5: BASELINE Config D (reference — current system)
    # ==================================================================
    print("\n" + "=" * 110)
    print("V5: BASELINE Config D (current system: trim=0.50, nth_swing=2)")
    print("=" * 110)
    t0 = _time.perf_counter()
    from experiments.validate_improvements import run_backtest_improved
    v5_trades = run_backtest_improved(d, sq_short=0.80, block_pm_shorts=True)
    v5_m = compute_metrics(v5_trades)
    print_metrics("V5 Baseline Config D", v5_m)
    print(f"  (computed in {_time.perf_counter() - t0:.1f}s)")

    # ==================================================================
    # V4: CONTROL — Single TP, lower trim=0.25, nth_swing=3
    # ==================================================================
    print("\n" + "=" * 110)
    print("V4: CONTROL — Single TP, trim=0.25, nth_swing=3 (simple improvement)")
    print("=" * 110)
    t0 = _time.perf_counter()
    v4_trades = run_backtest_single_tp_control(d, long_trim_pct=0.25, nth_swing=3)
    v4_m = compute_metrics(v4_trades)
    print_metrics("V4 Single TP (trim=0.25, trail=3)", v4_m)
    delta_r = v4_m["R"] - v5_m["R"]
    delta_ppdd = v4_m["PPDD"] - v5_m["PPDD"]
    print(f"    vs Baseline: dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}")
    print(f"  (computed in {_time.perf_counter() - t0:.1f}s)")

    # ==================================================================
    # V1: Multi-TP, late BE (BE only after TP2)
    # ==================================================================
    print("\n" + "=" * 110)
    print("V1: Multi-TP, late BE (25/25/50, BE after TP2, trail=3rd swing)")
    print("=" * 110)
    t0 = _time.perf_counter()
    v1_trades = run_backtest_multi_tp(
        d, d_extra,
        tp1_trim_pct=0.25, tp2_trim_pct=0.25,
        be_after_tp1=False, be_after_tp2=True,
        trail_nth_swing=3,
    )
    v1_m = compute_metrics(v1_trades)
    print_metrics("V1 Multi-TP late BE", v1_m)
    delta_r = v1_m["R"] - v5_m["R"]
    delta_ppdd = v1_m["PPDD"] - v5_m["PPDD"]
    print(f"    vs Baseline: dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}")
    print_trim_stage_breakdown(v1_trades, "V1")
    print(f"  (computed in {_time.perf_counter() - t0:.1f}s)")

    # ==================================================================
    # V2: Multi-TP, early BE (BE after TP1)
    # ==================================================================
    print("\n" + "=" * 110)
    print("V2: Multi-TP, early BE (25/25/50, BE after TP1, trail=3rd swing)")
    print("=" * 110)
    t0 = _time.perf_counter()
    v2_trades = run_backtest_multi_tp(
        d, d_extra,
        tp1_trim_pct=0.25, tp2_trim_pct=0.25,
        be_after_tp1=True, be_after_tp2=True,
        trail_nth_swing=3,
    )
    v2_m = compute_metrics(v2_trades)
    print_metrics("V2 Multi-TP early BE", v2_m)
    delta_r = v2_m["R"] - v5_m["R"]
    delta_ppdd = v2_m["PPDD"] - v5_m["PPDD"]
    print(f"    vs Baseline: dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}")
    print_trim_stage_breakdown(v2_trades, "V2")
    print(f"  (computed in {_time.perf_counter() - t0:.1f}s)")

    # ==================================================================
    # V3: Multi-TP, aggressive split (33/33/34%)
    # ==================================================================
    print("\n" + "=" * 110)
    print("V3: Multi-TP, aggressive (33/33/34, BE after TP2, trail=3rd swing)")
    print("=" * 110)
    t0 = _time.perf_counter()
    v3_trades = run_backtest_multi_tp(
        d, d_extra,
        tp1_trim_pct=0.33, tp2_trim_pct=0.33,
        be_after_tp1=False, be_after_tp2=True,
        trail_nth_swing=3,
    )
    v3_m = compute_metrics(v3_trades)
    print_metrics("V3 Multi-TP aggressive 33/33/34", v3_m)
    delta_r = v3_m["R"] - v5_m["R"]
    delta_ppdd = v3_m["PPDD"] - v5_m["PPDD"]
    print(f"    vs Baseline: dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}")
    print_trim_stage_breakdown(v3_trades, "V3")
    print(f"  (computed in {_time.perf_counter() - t0:.1f}s)")

    # ==================================================================
    # SUMMARY TABLE
    # ==================================================================
    print("\n" + "=" * 110)
    print("SUMMARY TABLE")
    print("=" * 110)
    print(f"  {'Variant':45s} | {'Trades':>6} | {'R':>9} | {'PPDD':>7} | {'PF':>6} | {'WR':>6} | {'MaxDD':>6} | {'avgR':>8}")
    print(f"  {'-'*45}-+-{'-'*6}-+-{'-'*9}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")
    for label, m in [
        ("V5 Baseline Config D (trim=0.50, trail=2)", v5_m),
        ("V4 Single TP (trim=0.25, trail=3)", v4_m),
        ("V1 Multi-TP late BE (25/25/50)", v1_m),
        ("V2 Multi-TP early BE (25/25/50)", v2_m),
        ("V3 Multi-TP aggressive (33/33/34)", v3_m),
    ]:
        print(f"  {label:45s} | {m['trades']:6d} | {m['R']:+9.1f} | {m['PPDD']:7.2f} | {m['PF']:6.2f} | {m['WR']:5.1f}% | {m['MaxDD']:6.1f} | {m['avgR']:+.4f}")

    # ==================================================================
    # WALK-FORWARD for top variants
    # ==================================================================
    # Find top 2 by PPDD (excluding V5 baseline)
    variant_results = [
        ("V4 Single TP (trim=0.25, trail=3)", v4_m, v4_trades),
        ("V1 Multi-TP late BE", v1_m, v1_trades),
        ("V2 Multi-TP early BE", v2_m, v2_trades),
        ("V3 Multi-TP aggressive", v3_m, v3_trades),
    ]
    # Sort by PPDD descending
    variant_results.sort(key=lambda x: x[1]["PPDD"], reverse=True)
    top_variants = variant_results[:3]

    print("\n" + "=" * 110)
    print("WALK-FORWARD: TOP VARIANTS vs BASELINE")
    print("=" * 110)

    # Print header
    hdr_parts = [f"{'Year':>6}"]
    hdr_parts.append(f"{'--- V5 BASELINE ---':^32s}")
    for label, _, _ in top_variants:
        short_label = label[:25]
        hdr_parts.append(f"{'--- ' + short_label + ' ---':^32s}")
    print(" | ".join(hdr_parts))

    sub_parts = [f"{'':>6}"]
    for _ in range(1 + len(top_variants)):
        sub_parts.append(f"{'n':>4} {'R':>7} {'PF':>5} {'PPDD':>6}")
    print(" | ".join(sub_parts))
    print("-" * (8 + 34 * (1 + len(top_variants))))

    wf_baseline = walk_forward_metrics(v5_trades)
    wf_top = []
    for label, m, trades in top_variants:
        wf_top.append(walk_forward_metrics(trades))

    bl_dict = {w["year"]: w for w in wf_baseline}
    top_dicts = [{w["year"]: w for w in wf} for wf in wf_top]

    all_years = sorted(set(
        list(bl_dict.keys()) +
        [y for td in top_dicts for y in td.keys()]
    ))

    wins_count = [0] * len(top_variants)
    for y in all_years:
        bl = bl_dict.get(y, {"n": 0, "R": 0, "WR": 0, "PF": 0, "PPDD": 0})
        parts = [f"  {y:>4}"]
        parts.append(f"{bl['n']:4d} {bl['R']:+7.1f} {bl['PF']:5.2f} {bl['PPDD']:+6.2f}")
        for vi, td in enumerate(top_dicts):
            tv = td.get(y, {"n": 0, "R": 0, "WR": 0, "PF": 0, "PPDD": 0})
            marker = " *" if tv["R"] > bl["R"] else ""
            if tv["R"] > bl["R"]:
                wins_count[vi] += 1
            parts.append(f"{tv['n']:4d} {tv['R']:+7.1f} {tv['PF']:5.2f} {tv['PPDD']:+6.2f}{marker}")
        print(" | ".join(parts))

    print("-" * (8 + 34 * (1 + len(top_variants))))
    for vi, (label, _, _) in enumerate(top_variants):
        print(f"  {label}: wins {wins_count[vi]}/{len(all_years)} years vs baseline")

    # ==================================================================
    # BONUS: Sensitivity analysis on trail_nth_swing for best multi-TP
    # ==================================================================
    print("\n" + "=" * 110)
    print("SENSITIVITY: Trail nth_swing for Multi-TP variants")
    print("=" * 110)

    for nth_val in [1, 2, 3, 4]:
        t0 = _time.perf_counter()
        trades = run_backtest_multi_tp(
            d, d_extra,
            tp1_trim_pct=0.25, tp2_trim_pct=0.25,
            be_after_tp1=False, be_after_tp2=True,
            trail_nth_swing=nth_val,
        )
        m = compute_metrics(trades)
        marker = " <-- V1 default" if nth_val == 3 else ""
        print_metrics(f"V1 + nth_swing={nth_val}{marker}", m)
        delta_r = m["R"] - v5_m["R"]
        delta_ppdd = m["PPDD"] - v5_m["PPDD"]
        print(f"    vs Baseline: dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}  ({_time.perf_counter() - t0:.1f}s)")

    # ==================================================================
    # BONUS: TP1 multiplier sensitivity
    # ==================================================================
    print("\n" + "=" * 110)
    print("SENSITIVITY: NY TP mult for Multi-TP V1 (late BE)")
    print("=" * 110)

    for tp_mult in [1.0, 1.5, 2.0, 2.5, 3.0]:
        t0 = _time.perf_counter()
        trades = run_backtest_multi_tp(
            d, d_extra,
            tp1_trim_pct=0.25, tp2_trim_pct=0.25,
            be_after_tp1=False, be_after_tp2=True,
            trail_nth_swing=3,
            ny_tp_mult=tp_mult,
        )
        m = compute_metrics(trades)
        marker = " <-- current" if tp_mult == 2.0 else ""
        print_metrics(f"V1 + ny_tp_mult={tp_mult}{marker}", m)
        delta_r = m["R"] - v5_m["R"]
        delta_ppdd = m["PPDD"] - v5_m["PPDD"]
        print(f"    vs Baseline: dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}  ({_time.perf_counter() - t0:.1f}s)")

    print("\n" + "=" * 110)
    print("DONE")
    print("=" * 110)


if __name__ == "__main__":
    main()
