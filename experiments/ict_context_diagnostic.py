"""
experiments/ict_context_diagnostic.py -- ICT Contextual Edge Diagnostic
======================================================================

Sprint 8: Test whether the SEQUENCE (sweep -> displacement -> FVG -> approach -> rejection)
has quantifiable edge, beyond what isolated FVG detection provides.

Approach: Run U2-v2 backtest, capture extra context per trade, then split trades
by contextual features and compare PF/R/PPDD across groups.

Features tested:
  F1: Pre-FVG liquidity sweep (did price sweep a swing point before creating the FVG?)
  F2: Approach behavior (how does price return to the FVG zone?)
  F3: FVG test count (first touch vs Nth touch)
  F4: Composite ICT sequence score
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.u2_clean import load_all, compute_metrics, pr, walk_forward, _find_nth_swing, _compute_grade


# ======================================================================
# Extended FVG Zone -- tracks contextual features at creation time
# ======================================================================
@dataclass
class FVGZoneEx(object):
    type: str            # 'bull' or 'bear'
    top: float
    bottom: float
    size: float
    birth_bar: int
    birth_atr: float
    used: bool = False
    # --- Contextual features (computed at creation) ---
    pre_sweep: bool = False          # F1: was a swing point swept before this FVG formed?
    sweep_depth_pct: float = 0.0     # how far past the swing point did price go (as % of ATR)
    sweep_bar_offset: int = 0        # how many bars before FVG was the sweep?
    displacement_quality: float = 0.0  # body/range of the displacement candle
    displacement_atr_mult: float = 0.0 # body size as multiple of ATR
    # --- Runtime tracking ---
    touch_count: int = 0             # F3: how many times has price entered this zone?


# ======================================================================
# Feature 1: Pre-FVG Liquidity Sweep Detection
# ======================================================================
def detect_pre_sweep(
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    o: np.ndarray,
    swing_low_mask: np.ndarray,
    swing_high_mask: np.ndarray,
    swing_low_price_at_mask: np.ndarray,
    swing_high_price_at_mask: np.ndarray,
    irl_low_arr: np.ndarray,
    irl_high_arr: np.ndarray,
    birth_bar: int,
    fvg_type: str,
    atr: float,
    lookback: int = 30,
) -> dict:
    """Check if a swing point was swept before this FVG formed.

    For a BULL FVG: look for a swing low that was breached (price went below it)
    then price recovered above it -- "stop hunt below" -> bullish displacement -> bull FVG.

    For a BEAR FVG: look for a swing high that was breached (price went above it)
    then price recovered below it -- "stop hunt above" -> bearish displacement -> bear FVG.

    Returns dict with sweep features.
    """
    result = {
        "pre_sweep": False,
        "sweep_depth_pct": 0.0,
        "sweep_bar_offset": 0,
    }

    start = max(0, birth_bar - lookback)

    if fvg_type == "bull":
        # Look for sweep of a swing low: price went below swing_low_price, then closed back above
        # Walk backwards from birth_bar to find the most recent sweep
        for j in range(birth_bar - 1, start - 1, -1):
            # Get the most recent swing low price known at bar j
            sl_price = irl_low_arr[j]
            if np.isnan(sl_price) or sl_price <= 0:
                continue

            # Check if bar j's low breached the swing low
            if l[j] < sl_price:
                # Sweep happened -- did price close back above? (the recovery)
                if c[j] > sl_price:
                    # Classic sweep: wick below, close above = stop hunt
                    depth = sl_price - l[j]
                    result["pre_sweep"] = True
                    result["sweep_depth_pct"] = depth / atr if atr > 0 else 0
                    result["sweep_bar_offset"] = birth_bar - j
                    break
                else:
                    # Price closed below -- check if NEXT bar recovered
                    for k in range(j + 1, min(birth_bar, len(c))):
                        if c[k] > sl_price:
                            depth = sl_price - l[j]
                            result["pre_sweep"] = True
                            result["sweep_depth_pct"] = depth / atr if atr > 0 else 0
                            result["sweep_bar_offset"] = birth_bar - j
                            break
                    if result["pre_sweep"]:
                        break

    else:  # bear
        # Look for sweep of a swing high: price went above swing_high_price, then closed back below
        for j in range(birth_bar - 1, start - 1, -1):
            sh_price = irl_high_arr[j]
            if np.isnan(sh_price) or sh_price <= 0:
                continue

            if h[j] > sh_price:
                if c[j] < sh_price:
                    depth = h[j] - sh_price
                    result["pre_sweep"] = True
                    result["sweep_depth_pct"] = depth / atr if atr > 0 else 0
                    result["sweep_bar_offset"] = birth_bar - j
                    break
                else:
                    for k in range(j + 1, min(birth_bar, len(c))):
                        if c[k] < sh_price:
                            depth = h[j] - sh_price
                            result["pre_sweep"] = True
                            result["sweep_depth_pct"] = depth / atr if atr > 0 else 0
                            result["sweep_bar_offset"] = birth_bar - j
                            break
                    if result["pre_sweep"]:
                        break

    return result


# ======================================================================
# Feature 2: Approach Behavior
# ======================================================================
def compute_approach_features(
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    o: np.ndarray,
    atr_arr: np.ndarray,
    fill_bar: int,
    birth_bar: int,
    fvg_type: str,
    zone_top: float,
    zone_bottom: float,
) -> dict:
    """Measure how price approached the FVG zone before filling the limit order.

    Key metrics:
    - bars_to_fill: how many bars from FVG creation to limit fill
    - approach_deceleration: are bar sizes shrinking as price approaches? (exhaustion)
    - approach_bar_shrink: ratio of last 3 bars' avg range to first 3 bars' avg range
    - approach_direction_ratio: fraction of bars moving toward the zone
    """
    result = {
        "bars_to_fill": fill_bar - birth_bar,
        "approach_deceleration": 0.0,
        "approach_bar_shrink": 1.0,
        "approach_direction_ratio": 0.5,
        "approach_momentum_decay": 0.0,
    }

    # Need at least 4 bars of approach to measure
    n_approach = fill_bar - birth_bar
    if n_approach < 4:
        return result

    # Approach window: from birth_bar+1 to fill_bar (inclusive)
    start = birth_bar + 1
    end = fill_bar + 1  # exclusive

    ranges = h[start:end] - l[start:end]
    closes = c[start:end]
    opens = o[start:end]

    if len(ranges) < 4:
        return result

    # Bar shrinkage: compare first half vs second half average range
    mid = len(ranges) // 2
    first_half_avg = np.mean(ranges[:mid])
    second_half_avg = np.mean(ranges[mid:])

    if first_half_avg > 0:
        result["approach_bar_shrink"] = second_half_avg / first_half_avg

    # Direction ratio: fraction of bars moving toward the zone
    if fvg_type == "bull":
        # Bull FVG is below price -> price should be moving DOWN toward zone
        toward_zone = (closes < opens).sum()  # bearish bars = moving toward bull FVG
    else:
        # Bear FVG is above price -> price should be moving UP toward zone
        toward_zone = (closes > opens).sum()  # bullish bars = moving toward bear FVG

    result["approach_direction_ratio"] = toward_zone / len(closes)

    # Momentum decay: compare bar-by-bar movement in last 5 bars
    if len(ranges) >= 5:
        last5 = ranges[-5:]
        # Linear regression slope of bar sizes -> negative = decelerating
        x = np.arange(len(last5), dtype=float)
        if np.std(x) > 0 and np.std(last5) > 0:
            slope = np.corrcoef(x, last5)[0, 1]
            result["approach_momentum_decay"] = -slope  # positive = decelerating (good)

    # Deceleration: correlation of bar index with bar size (negative = shrinking)
    x = np.arange(len(ranges), dtype=float)
    if np.std(x) > 0 and np.std(ranges) > 0:
        corr = np.corrcoef(x, ranges)[0, 1]
        result["approach_deceleration"] = -corr  # positive = bars getting smaller (exhaustion)

    return result


# ======================================================================
# Feature: Displacement quality at FVG creation
# ======================================================================
def compute_displacement_at_creation(
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    o: np.ndarray,
    atr_arr: np.ndarray,
    birth_bar: int,
) -> dict:
    """Measure the quality of the displacement candle that created the FVG.

    The FVG signal at bar i is shifted by 1, so the actual candle-2 (displacement)
    is at birth_bar - 1. Candle-1 is at birth_bar - 2, candle-3 is at birth_bar.
    But due to the np.roll shift in detect_fvg, the actual displacement candle
    (candle 2) is at birth_bar itself or birth_bar-1.

    We measure the candle that created the big move.
    """
    result = {
        "disp_body_ratio": 0.0,
        "disp_atr_mult": 0.0,
        "disp_engulfs_prior": False,
    }

    # The FVG detection shifts by 1, so at birth_bar the signal was from
    # the 3-candle pattern ending at birth_bar (candle 3 = birth_bar,
    # candle 2 = birth_bar-1, candle 1 = birth_bar-2).
    # Actually: detect_fvg anchors at candle-2 then rolls by 1, so
    # fvg_bull[birth_bar] means candle-2 was at birth_bar-1.
    # The displacement candle is candle-2 = birth_bar - 1.

    disp_bar = birth_bar - 1
    if disp_bar < 1:
        return result

    bar_range = h[disp_bar] - l[disp_bar]
    bar_body = abs(c[disp_bar] - o[disp_bar])

    if bar_range > 0:
        result["disp_body_ratio"] = bar_body / bar_range

    atr = atr_arr[disp_bar]
    if not np.isnan(atr) and atr > 0:
        result["disp_atr_mult"] = bar_body / atr

    # Does it engulf the prior candle?
    prior_bar = disp_bar - 1
    if prior_bar >= 0:
        prior_range = h[prior_bar] - l[prior_bar]
        if bar_range > prior_range:
            result["disp_engulfs_prior"] = True

    return result


# ======================================================================
# Modified U2 loop -- captures contextual features per trade
# ======================================================================
def run_u2_with_context(
    d: dict,
    *,
    max_fvg_age: int = 200,
    fvg_size_mult: float = 0.3,
    stop_strategy: str = "A2",
    stop_buffer_pct: float = 0.15,
    min_stop_atr: float = 0.0,
    min_stop_pts: float = 5.0,
    block_pm_shorts: bool = True,
    pre_momentum: int = 0,
    trim_pct: float = 0.25,
    tp_mult: float = 0.35,
    nth_swing: int = 5,
    be_after_trim: bool = True,
    short_rr: float = 0.625,
    tighten_factor: float = 0.85,
    eod_close: bool = True,
    sweep_lookback: int = 30,
) -> list[dict]:
    """Run U2 backtest with contextual feature capture.

    Same logic as u2_clean.run_u2_backtest but:
    1. Uses FVGZoneEx with contextual features
    2. Tracks touch_count per zone
    3. Computes approach behavior at fill time
    4. Outputs extended trade records with all context features
    """

    params = d["params"]
    nq = d["nq"]
    n = d["n"]
    o, h, l, c_arr = d["o"], d["h"], d["l"], d["c"]
    atr_arr = d["atr_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    fvg_df = d["fvg_df"]
    dates = d["dates"]
    dow_arr = d["dow_arr"]
    et_frac_arr = d["et_frac_arr"]
    swing_high_mask = d["swing_high_mask"]
    swing_low_mask = d["swing_low_mask"]
    irl_high_arr = d["irl_high_arr"]
    irl_low_arr = d["irl_low_arr"]
    swing_low_price_at_mask = d.get("swing_low_price_at_mask")
    swing_high_price_at_mask = d.get("swing_high_price_at_mask", np.full(n, np.nan))

    pos_params = params["position"]
    risk_params = params["risk"]
    bt_params = params["backtest"]
    grading_params = params["grading"]
    session_regime = params.get("session_regime", {})

    normal_r = pos_params["normal_r"]
    reduced_r = pos_params["reduced_r"]
    point_value = pos_params["point_value"]
    daily_max_loss_r = risk_params["daily_max_loss_r"]
    max_consec_losses = risk_params["max_consecutive_losses"]
    commission_per_side = bt_params["commission_per_side_micro"]
    slippage_points = bt_params["slippage_normal_ticks"] * 0.25
    a_plus_mult = grading_params["a_plus_size_mult"]
    b_plus_mult = grading_params["b_plus_size_mult"]

    # FVG arrays
    bull_mask = fvg_df["fvg_bull"].values
    bear_mask = fvg_df["fvg_bear"].values
    fvg_bull_top = fvg_df["fvg_bull_top"].values
    fvg_bull_bottom = fvg_df["fvg_bull_bottom"].values
    fvg_bear_top = fvg_df["fvg_bear_top"].values
    fvg_bear_bottom = fvg_df["fvg_bear_bottom"].values

    # State
    active_zones: list[FVGZoneEx] = []
    trades: list[dict] = []
    in_position = False
    pos_direction = pos_entry_idx = 0
    pos_entry_price = pos_stop = pos_tp1 = 0.0
    pos_contracts = pos_remaining_contracts = 0
    pos_trimmed = False
    pos_be_stop = pos_trail_stop = 0.0
    pos_grade = ""
    pos_trim_pct = trim_pct
    pos_zone: FVGZoneEx | None = None  # track which zone produced the trade
    current_date = None
    daily_pnl_r = 0.0
    consecutive_losses = 0
    day_stopped = False

    for i in range(n):
        bar_date = dates[i]
        if bar_date != current_date:
            current_date = bar_date
            daily_pnl_r = 0.0
            consecutive_losses = 0
            day_stopped = False

        # === PHASE A: Register new FVG zones with context ===
        if bull_mask[i]:
            atr_val = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
            zone_size = fvg_bull_top[i] - fvg_bull_bottom[i]
            if zone_size >= fvg_size_mult * atr_val and zone_size > 0:
                # Compute contextual features at creation
                sweep_info = detect_pre_sweep(
                    h, l, c_arr, o,
                    swing_low_mask, swing_high_mask,
                    swing_low_price_at_mask, swing_high_price_at_mask,
                    irl_low_arr, irl_high_arr,
                    birth_bar=i, fvg_type="bull", atr=atr_val,
                    lookback=sweep_lookback,
                )
                disp_info = compute_displacement_at_creation(h, l, c_arr, o, atr_arr, i)

                zone = FVGZoneEx(
                    "bull", fvg_bull_top[i], fvg_bull_bottom[i],
                    zone_size, i, atr_val,
                    pre_sweep=sweep_info["pre_sweep"],
                    sweep_depth_pct=sweep_info["sweep_depth_pct"],
                    sweep_bar_offset=sweep_info["sweep_bar_offset"],
                    displacement_quality=disp_info["disp_body_ratio"],
                    displacement_atr_mult=disp_info["disp_atr_mult"],
                )
                active_zones.append(zone)

        if bear_mask[i]:
            atr_val = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
            zone_size = fvg_bear_top[i] - fvg_bear_bottom[i]
            if zone_size >= fvg_size_mult * atr_val and zone_size > 0:
                sweep_info = detect_pre_sweep(
                    h, l, c_arr, o,
                    swing_low_mask, swing_high_mask,
                    swing_low_price_at_mask, swing_high_price_at_mask,
                    irl_low_arr, irl_high_arr,
                    birth_bar=i, fvg_type="bear", atr=atr_val,
                    lookback=sweep_lookback,
                )
                disp_info = compute_displacement_at_creation(h, l, c_arr, o, atr_arr, i)

                zone = FVGZoneEx(
                    "bear", fvg_bear_top[i], fvg_bear_bottom[i],
                    zone_size, i, atr_val,
                    pre_sweep=sweep_info["pre_sweep"],
                    sweep_depth_pct=sweep_info["sweep_depth_pct"],
                    sweep_bar_offset=sweep_info["sweep_bar_offset"],
                    displacement_quality=disp_info["disp_body_ratio"],
                    displacement_atr_mult=disp_info["disp_atr_mult"],
                )
                active_zones.append(zone)

        # === PHASE B: Invalidate / expire + track touches ===
        surviving: list[FVGZoneEx] = []
        for zone in active_zones:
            if zone.used:
                continue
            if (i - zone.birth_bar) > max_fvg_age:
                continue
            if zone.type == "bull" and c_arr[i] < zone.bottom:
                continue
            if zone.type == "bear" and c_arr[i] > zone.top:
                continue

            # F3: Track touch count -- price entered the zone
            if zone.type == "bull":
                if l[i] <= zone.top and h[i] >= zone.bottom:
                    zone.touch_count += 1
            else:
                if h[i] >= zone.bottom and l[i] <= zone.top:
                    zone.touch_count += 1

            surviving.append(zone)
        if len(surviving) > 30:
            surviving = surviving[-30:]
        active_zones = surviving

        # === PHASE C: EXIT MANAGEMENT (identical to u2_clean) ===
        if in_position:
            exited = False
            exit_reason = ""
            exit_price = 0.0
            exit_contracts = pos_remaining_contracts

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
                        pos_trail_stop = _find_nth_swing(swing_low_mask, swing_low_price_at_mask, i, nth_swing)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True
                if pos_trimmed and not exited:
                    nt = _find_nth_swing(swing_low_mask, swing_low_price_at_mask, i, nth_swing)
                    if not np.isnan(nt) and nt > pos_trail_stop:
                        pos_trail_stop = nt

            elif not exited and pos_direction == -1:
                eff_stop = pos_stop
                if h[i] >= eff_stop:
                    exit_price = eff_stop + slippage_points
                    exit_reason = "stop"
                    exited = True
                elif not pos_trimmed and l[i] <= pos_tp1:
                    pos_remaining_contracts = 0
                    pos_trimmed = True
                    exit_price = pos_tp1
                    exit_reason = "tp1"
                    exit_contracts = pos_contracts
                    exited = True

            if not exited and eod_close and et_frac_arr[i] >= 15.917:
                slp = 0.25
                exit_price = c_arr[i] - slp if pos_direction == 1 else c_arr[i] + slp
                exit_reason = "eod_close"
                exited = True

            # PA early cut
            bars_in_trade = i - pos_entry_idx
            if not exited and not pos_trimmed and 2 <= bars_in_trade <= 4:
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
                if (avg_wick > 0.65 and favorable < 0.5 and disp < cur_atr * 0.3 and bars_in_trade >= 3):
                    exit_price = o[i+1] if i+1 < n else c_arr[i]
                    exit_reason = "early_cut_pa"
                    exited = True

            if exited:
                pnl_pts = (exit_price - pos_entry_price) if pos_direction == 1 else (pos_entry_price - exit_price)
                if pos_trimmed and exit_reason != "tp1":
                    trim_pnl = (pos_tp1 - pos_entry_price) if pos_direction == 1 else (pos_entry_price - pos_tp1)
                    trim_c = pos_contracts - exit_contracts
                    total_pnl = trim_pnl * point_value * trim_c + pnl_pts * point_value * exit_contracts
                    total_comm = commission_per_side * 2 * pos_contracts
                    total_pnl -= total_comm
                else:
                    total_pnl = pnl_pts * point_value * exit_contracts
                    total_comm = commission_per_side * 2 * exit_contracts
                    total_pnl -= total_comm
                stop_dist_exit = abs(pos_entry_price - pos_stop)
                total_risk = stop_dist_exit * point_value * pos_contracts
                r_mult = total_pnl / total_risk if total_risk > 0 else 0.0

                trade_rec = {
                    "entry_time": nq.index[pos_entry_idx],
                    "exit_time": nq.index[min(i + (1 if exit_reason == "early_cut_pa" else 0), n-1)],
                    "r": r_mult, "reason": exit_reason, "dir": pos_direction,
                    "type": "limit_fvg", "trimmed": pos_trimmed, "grade": pos_grade,
                    "entry_price": pos_entry_price, "exit_price": exit_price,
                    "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
                    "stop_dist_pts": stop_dist_exit,
                }

                # --- Attach zone context features ---
                if pos_zone is not None:
                    trade_rec["zone_birth_bar"] = pos_zone.birth_bar
                    trade_rec["pre_sweep"] = pos_zone.pre_sweep
                    trade_rec["sweep_depth_pct"] = pos_zone.sweep_depth_pct
                    trade_rec["sweep_bar_offset"] = pos_zone.sweep_bar_offset
                    trade_rec["disp_body_ratio"] = pos_zone.displacement_quality
                    trade_rec["disp_atr_mult"] = pos_zone.displacement_atr_mult
                    trade_rec["touch_count"] = pos_zone.touch_count
                    trade_rec["fvg_age_at_fill"] = pos_entry_idx - pos_zone.birth_bar
                    # Approach features (computed at entry time)
                    approach = compute_approach_features(
                        h, l, c_arr, o, atr_arr,
                        fill_bar=pos_entry_idx,
                        birth_bar=pos_zone.birth_bar,
                        fvg_type=pos_zone.type,
                        zone_top=pos_zone.top,
                        zone_bottom=pos_zone.bottom,
                    )
                    trade_rec.update(approach)

                trades.append(trade_rec)

                daily_pnl_r += r_mult
                if exit_reason == "be_sweep" and pos_trimmed:
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
                pos_zone = None

        # === PHASE D: ENTRY via limit order fill ===
        if in_position:
            continue
        if day_stopped:
            continue
        if news_blackout_arr is not None and news_blackout_arr[i]:
            continue

        et_frac = et_frac_arr[i]
        if 9.5 <= et_frac <= 10.0:
            continue
        if not (10.0 <= et_frac < 16.0):
            continue

        best_fill = None
        best_zone_info = None
        best_same_bar_stop = False

        for zone in active_zones:
            if zone.used or zone.birth_bar >= i:
                continue

            direction = 1 if zone.type == "bull" else -1

            if block_pm_shorts and direction == -1 and et_frac >= 14.0:
                continue

            if zone.type == "bull":
                entry_p = zone.top
                if stop_strategy == "A1":
                    stop_p = zone.bottom
                elif stop_strategy == "A2":
                    stop_p = zone.bottom - zone.size * stop_buffer_pct
                else:
                    stop_p = zone.bottom
            else:
                entry_p = zone.bottom
                if stop_strategy == "A1":
                    stop_p = zone.top
                elif stop_strategy == "A2":
                    stop_p = zone.top + zone.size * stop_buffer_pct
                else:
                    stop_p = zone.top

            stop_dist = abs(entry_p - stop_p)

            if tighten_factor < 1.0:
                if zone.type == "bull":
                    stop_p = entry_p - stop_dist * tighten_factor
                else:
                    stop_p = entry_p + stop_dist * tighten_factor
                stop_dist = abs(entry_p - stop_p)

            cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0

            if min_stop_atr > 0 and cur_atr > 0:
                if stop_dist / cur_atr < min_stop_atr:
                    continue
            if min_stop_pts > 0 and stop_dist < min_stop_pts:
                continue
            if stop_dist < 1.0:
                continue

            if pre_momentum > 0 and i >= pre_momentum:
                bullish_count = sum(1 for k in range(i - pre_momentum, i) if c_arr[k] > o[k])
                if bullish_count < pre_momentum:
                    continue

            is_same_bar_stop = False
            if zone.type == "bull":
                if l[i] > entry_p:
                    continue
                if l[i] <= stop_p:
                    is_same_bar_stop = True
            else:
                if h[i] < entry_p:
                    continue
                if h[i] >= stop_p:
                    is_same_bar_stop = True

            if direction == -np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0:
                continue

            fill_quality = -abs(c_arr[i] - entry_p)
            if best_fill is None or fill_quality > best_fill:
                best_fill = fill_quality
                best_zone_info = (zone, direction, entry_p, stop_p, stop_dist, cur_atr)
                best_same_bar_stop = is_same_bar_stop

        if best_zone_info is None:
            continue

        zone, direction, entry_p, stop_p, stop_dist, cur_atr = best_zone_info
        zone.used = True

        ba = 1.0 if (direction == np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0) else 0.0
        grade = _compute_grade(ba, regime_arr[i])
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
            if sr_lunch_s <= et_frac < sr_lunch_e:
                sr_mult = session_regime.get("lunch_mult", 0.0)
            elif et_frac < session_regime.get("am_end", 12.5):
                sr_mult = session_regime.get("am_mult", 1.0)
            elif et_frac >= session_regime.get("pm_start", 13.0):
                sr_mult = session_regime.get("pm_mult", 1.0)
            else:
                sr_mult = 1.0
            r_amount *= sr_mult
            if r_amount <= 0:
                zone.used = False
                continue

        contracts = max(1, int(r_amount / (stop_dist * point_value))) if stop_dist > 0 and point_value > 0 else 0
        if contracts <= 0:
            zone.used = False
            continue

        # Same-bar stop
        if best_same_bar_stop:
            exit_price_sbs = (stop_p - slippage_points) if direction == 1 else (stop_p + slippage_points)
            pnl_pts_sbs = (exit_price_sbs - entry_p) if direction == 1 else (entry_p - exit_price_sbs)
            total_pnl_sbs = pnl_pts_sbs * point_value * contracts - commission_per_side * 2 * contracts
            total_risk_sbs = stop_dist * point_value * contracts
            r_sbs = total_pnl_sbs / total_risk_sbs if total_risk_sbs > 0 else 0.0

            # Compute approach for same-bar-stop trades too
            approach = compute_approach_features(
                h, l, c_arr, o, atr_arr,
                fill_bar=i, birth_bar=zone.birth_bar,
                fvg_type=zone.type, zone_top=zone.top, zone_bottom=zone.bottom,
            )

            trades.append({
                "entry_time": nq.index[i], "exit_time": nq.index[i],
                "r": r_sbs, "reason": "same_bar_stop", "dir": direction,
                "type": "limit_fvg", "trimmed": False, "grade": grade,
                "entry_price": entry_p, "exit_price": exit_price_sbs,
                "stop_price": stop_p, "tp1_price": 0.0, "pnl_dollars": total_pnl_sbs,
                "stop_dist_pts": stop_dist,
                # Context
                "zone_birth_bar": zone.birth_bar,
                "pre_sweep": zone.pre_sweep,
                "sweep_depth_pct": zone.sweep_depth_pct,
                "sweep_bar_offset": zone.sweep_bar_offset,
                "disp_body_ratio": zone.displacement_quality,
                "disp_atr_mult": zone.displacement_atr_mult,
                "touch_count": zone.touch_count,
                "fvg_age_at_fill": i - zone.birth_bar,
                **approach,
            })

            daily_pnl_r += r_sbs
            consecutive_losses += 1
            if consecutive_losses >= max_consec_losses:
                day_stopped = True
            if daily_pnl_r <= -daily_max_loss_r:
                day_stopped = True
            continue

        # TP computation
        if direction == 1:
            irl_target = irl_high_arr[i]
            if np.isnan(irl_target) or irl_target <= entry_p:
                irl_target = entry_p + stop_dist * 2.0
            tp_distance = irl_target - entry_p
            tp1_price = entry_p + tp_distance * tp_mult
        else:
            tp1_price = entry_p - stop_dist * short_rr

        # Enter position
        in_position = True
        pos_direction = direction
        pos_entry_idx = i
        pos_entry_price = entry_p
        pos_stop = stop_p
        pos_tp1 = tp1_price
        pos_contracts = contracts
        pos_remaining_contracts = contracts
        pos_trimmed = False
        pos_be_stop = 0.0
        pos_trail_stop = 0.0
        pos_grade = grade
        pos_zone = zone  # track which zone produced this trade
        pos_trim_pct = 1.0 if direction == -1 else trim_pct

    # Force close at end
    if in_position:
        last_i = n - 1
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
            "entry_time": nq.index[pos_entry_idx], "exit_time": nq.index[last_i],
            "r": r_mult, "reason": "force_close", "dir": pos_direction,
            "type": "limit_fvg", "trimmed": pos_trimmed, "grade": pos_grade,
            "entry_price": pos_entry_price, "exit_price": exit_price,
            "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
            "stop_dist_pts": stop_dist_exit,
            "zone_birth_bar": pos_zone.birth_bar if pos_zone else -1,
            "pre_sweep": pos_zone.pre_sweep if pos_zone else False,
            "sweep_depth_pct": pos_zone.sweep_depth_pct if pos_zone else 0,
            "sweep_bar_offset": pos_zone.sweep_bar_offset if pos_zone else 0,
            "disp_body_ratio": pos_zone.displacement_quality if pos_zone else 0,
            "disp_atr_mult": pos_zone.displacement_atr_mult if pos_zone else 0,
            "touch_count": pos_zone.touch_count if pos_zone else 0,
            "fvg_age_at_fill": pos_entry_idx - pos_zone.birth_bar if pos_zone else 0,
            "bars_to_fill": pos_entry_idx - pos_zone.birth_bar if pos_zone else 0,
            "approach_deceleration": 0, "approach_bar_shrink": 1,
            "approach_direction_ratio": 0.5, "approach_momentum_decay": 0,
        })

    return trades


# ======================================================================
# Analysis functions
# ======================================================================
def split_analysis(df: pd.DataFrame, col: str, bins: list | None = None,
                   labels: list[str] | None = None, min_trades: int = 50) -> pd.DataFrame:
    """Split trades by a feature column, compute PF/R/PPDD for each group."""
    if bins is not None:
        df = df.copy()
        df["_grp"] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    else:
        df = df.copy()
        df["_grp"] = df[col]

    rows = []
    for grp_name, grp_df in df.groupby("_grp", observed=True):
        n_trades = len(grp_df)
        if n_trades < min_trades:
            continue
        r_arr = grp_df["r"].values
        total_r = r_arr.sum()
        wr = (r_arr > 0).mean() * 100
        wins = r_arr[r_arr > 0].sum()
        losses = abs(r_arr[r_arr < 0].sum())
        pf = wins / losses if losses > 0 else 999.0
        cumr = np.cumsum(r_arr)
        peak = np.maximum.accumulate(cumr)
        dd = (peak - cumr).max()
        ppdd = total_r / dd if dd > 0 else 999.0
        avg_r = total_r / n_trades
        rows.append({
            "group": grp_name, "trades": n_trades, "R": round(total_r, 1),
            "PF": round(pf, 2), "PPDD": round(ppdd, 2), "WR": round(wr, 1),
            "MaxDD": round(dd, 1), "avgR": round(avg_r, 4),
        })
    return pd.DataFrame(rows)


def print_split(title: str, result_df: pd.DataFrame):
    """Pretty-print split analysis results."""
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    if result_df.empty:
        print("  (no groups with enough trades)")
        return
    for _, row in result_df.iterrows():
        tpd = int(row["trades"]) / (252 * 10.5)
        print(f"  {str(row['group']):30s} | {int(row['trades']):5d}t | R={row['R']:+8.1f} | "
              f"PF={row['PF']:5.2f} | PPDD={row['PPDD']:6.2f} | WR={row['WR']:5.1f}% | "
              f"DD={row['MaxDD']:5.1f}R | avgR={row['avgR']:+.4f} | {tpd:.2f}/d")

    # Statistical significance: compare best vs worst group PF
    if len(result_df) >= 2:
        best = result_df.loc[result_df["PF"].idxmax()]
        worst = result_df.loc[result_df["PF"].idxmin()]
        delta_pf = best["PF"] - worst["PF"]
        print(f"\n  PF spread: {worst['group']} ({worst['PF']:.2f}) -> {best['group']} ({best['PF']:.2f})  "
              f"delta={delta_pf:+.2f}")
        if delta_pf >= 0.20:
            print(f"  >>> SIGNAL: PF spread >= 0.20 -- this feature separates winners from losers")
        elif delta_pf >= 0.10:
            print(f"  >>> MARGINAL: PF spread 0.10-0.20 -- weak signal, may not survive OOS")
        else:
            print(f"  >>> DEAD: PF spread < 0.10 -- no meaningful separation")


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 120)
    print("ICT CONTEXTUAL EDGE DIAGNOSTIC -- Sprint 8")
    print("Does the ICT sequence (sweep > displacement > FVG > approach > rejection) have quantifiable edge?")
    print("=" * 120)

    d = load_all()

    # Run U2-v2 with context capture (production params)
    print("\n[RUN] Running U2-v2 with context capture...")
    t0 = _time.perf_counter()
    trades = run_u2_with_context(d,
        stop_strategy="A2", fvg_size_mult=0.3, max_fvg_age=200,
        min_stop_pts=5.0, tighten_factor=0.85, tp_mult=0.35, nth_swing=5,
    )
    elapsed = _time.perf_counter() - t0
    print(f"[RUN] Done in {elapsed:.1f}s -- {len(trades)} trades")

    # Filter to LONGS ONLY (shorts are dead, PF=0.59 -- confirmed Sprint 7)
    trades_all = trades
    trades = [t for t in trades if t["dir"] == 1]
    print(f"[FILTER] Longs only: {len(trades)} of {len(trades_all)} trades")

    # Verify baseline matches U2-v2 corrected
    m = compute_metrics(trades)
    print(f"\n[BASELINE] Verify matches U2-v2 corrected (longs only):")
    pr("U2-v2 longs", m)
    print(f"  Expected: ~2589t, R~+791.5, PF~1.59, PPDD~28.35")

    df = pd.DataFrame(trades)

    # ================================================================
    # FEATURE 1: Pre-FVG Liquidity Sweep
    # ================================================================
    print("\n" + "#" * 120)
    print("# FEATURE 1: PRE-FVG LIQUIDITY SWEEP")
    print("# Core ICT claim: FVGs formed after a stop hunt are higher quality")
    print("#" * 120)

    if "pre_sweep" in df.columns:
        # Binary split: sweep vs no-sweep
        print_split("F1a: Pre-sweep (binary)", split_analysis(df, "pre_sweep", min_trades=30))

        # By sweep depth
        sweep_trades = df[df["pre_sweep"] == True]
        if len(sweep_trades) > 100:
            print_split("F1b: Sweep depth (% of ATR) -- sweep trades only",
                split_analysis(sweep_trades, "sweep_depth_pct",
                    bins=[0, 0.1, 0.3, 0.5, 1.0, 99],
                    labels=["tiny<10%", "small 10-30%", "medium 30-50%", "deep 50-100%", "very deep>100%"],
                    min_trades=20))

        # By sweep recency
        if len(sweep_trades) > 100:
            print_split("F1c: Sweep recency (bars before FVG) -- sweep trades only",
                split_analysis(sweep_trades, "sweep_bar_offset",
                    bins=[0, 3, 8, 15, 99],
                    labels=["immediate 1-3", "recent 4-8", "moderate 9-15", "distant 16+"],
                    min_trades=20))

    # ================================================================
    # FEATURE 2: Approach Behavior
    # ================================================================
    print("\n" + "#" * 120)
    print("# FEATURE 2: APPROACH BEHAVIOR")
    print("# How does price return to the FVG zone? Exhaustion (slow) vs aggressive (fast)?")
    print("#" * 120)

    if "bars_to_fill" in df.columns:
        print_split("F2a: Bars to fill (age at entry)",
            split_analysis(df, "bars_to_fill",
                bins=[0, 5, 15, 40, 80, 999],
                labels=["instant 1-5", "quick 6-15", "medium 16-40", "slow 41-80", "very slow 81+"],
                min_trades=30))

    if "approach_bar_shrink" in df.columns:
        # Bar shrink < 1 = bars getting smaller (exhaustion = good for reversal)
        valid = df[df["bars_to_fill"] >= 6]  # need enough approach bars
        if len(valid) > 100:
            print_split("F2b: Approach bar shrinkage (< 1 = exhaustion)",
                split_analysis(valid, "approach_bar_shrink",
                    bins=[0, 0.5, 0.8, 1.0, 1.5, 99],
                    labels=["strong exhaust <0.5", "exhaust 0.5-0.8", "neutral 0.8-1.0",
                            "accelerating 1.0-1.5", "strong accel >1.5"],
                    min_trades=20))

    if "approach_deceleration" in df.columns:
        valid = df[df["bars_to_fill"] >= 6]
        if len(valid) > 100:
            print_split("F2c: Approach deceleration (positive = slowing down)",
                split_analysis(valid, "approach_deceleration",
                    bins=[-99, -0.3, 0.0, 0.3, 0.6, 99],
                    labels=["accelerating <-0.3", "slight accel -0.3-0", "slight decel 0-0.3",
                            "decelerating 0.3-0.6", "strong decel >0.6"],
                    min_trades=20))

    if "approach_direction_ratio" in df.columns:
        valid = df[df["bars_to_fill"] >= 6]
        if len(valid) > 100:
            print_split("F2d: Approach direction ratio (high = more bars moving toward zone)",
                split_analysis(valid, "approach_direction_ratio",
                    bins=[0, 0.3, 0.5, 0.7, 1.01],
                    labels=["few toward <30%", "mixed 30-50%", "mostly toward 50-70%", "strong toward >70%"],
                    min_trades=20))

    # ================================================================
    # FEATURE 3: FVG Touch Count
    # ================================================================
    print("\n" + "#" * 120)
    print("# FEATURE 3: FVG TOUCH COUNT")
    print("# First touch = highest probability? Or does repeated testing matter?")
    print("#" * 120)

    if "touch_count" in df.columns:
        print_split("F3: Touch count at entry",
            split_analysis(df, "touch_count",
                bins=[-1, 1, 2, 3, 5, 999],
                labels=["1st touch", "2nd touch", "3rd touch", "4th-5th", "6th+"],
                min_trades=30))

    # ================================================================
    # FEATURE: Displacement Quality at FVG Creation
    # ================================================================
    print("\n" + "#" * 120)
    print("# DISPLACEMENT QUALITY AT FVG CREATION")
    print("# Does a strong displacement candle predict better FVG trades?")
    print("#" * 120)

    if "disp_body_ratio" in df.columns:
        print_split("Displacement body ratio",
            split_analysis(df, "disp_body_ratio",
                bins=[0, 0.3, 0.5, 0.7, 0.9, 1.01],
                labels=["weak <0.3", "low 0.3-0.5", "medium 0.5-0.7", "strong 0.7-0.9", "very strong >0.9"],
                min_trades=30))

    if "disp_atr_mult" in df.columns:
        print_split("Displacement ATR multiple",
            split_analysis(df, "disp_atr_mult",
                bins=[0, 0.3, 0.6, 1.0, 1.5, 99],
                labels=["tiny <0.3x", "small 0.3-0.6x", "medium 0.6-1.0x", "large 1.0-1.5x", "huge >1.5x"],
                min_trades=30))

    # ================================================================
    # COMPOSITE: Full ICT Sequence
    # ================================================================
    print("\n" + "#" * 120)
    print("# COMPOSITE: FULL ICT SEQUENCE SCORE")
    print("# sweep + strong displacement + exhaustion approach + fresh FVG")
    print("#" * 120)

    if all(c in df.columns for c in ["pre_sweep", "disp_body_ratio", "touch_count"]):
        df_c = df.copy()
        # Build composite score (0-4 points)
        df_c["ict_score"] = 0
        df_c.loc[df_c["pre_sweep"] == True, "ict_score"] += 1
        df_c.loc[df_c["disp_body_ratio"] >= 0.6, "ict_score"] += 1
        df_c.loc[df_c["touch_count"] <= 1, "ict_score"] += 1
        if "approach_bar_shrink" in df_c.columns:
            df_c.loc[df_c["approach_bar_shrink"] < 0.8, "ict_score"] += 1

        print_split("ICT composite score (0=nothing, 4=perfect sequence)",
            split_analysis(df_c, "ict_score", min_trades=30))

        # High-quality only (score >= 3)
        hq = df_c[df_c["ict_score"] >= 3]
        lq = df_c[df_c["ict_score"] <= 1]
        if len(hq) >= 30 and len(lq) >= 30:
            m_hq = compute_metrics(hq.to_dict("records"))
            m_lq = compute_metrics(lq.to_dict("records"))
            print(f"\n  HEAD-TO-HEAD: High quality (score>=3) vs Low quality (score<=1)")
            pr("HIGH quality (score>=3)", m_hq)
            pr("LOW  quality (score<=1)", m_lq)
            delta_pf = m_hq["PF"] - m_lq["PF"]
            print(f"\n  deltaPF = {delta_pf:+.2f}  |  deltaR/trade = {m_hq['avgR'] - m_lq['avgR']:+.4f}")
            if delta_pf >= 0.20:
                print(f"  >>> BREAKTHROUGH: ICT sequence has significant edge (deltaPF>=0.20)")
            elif delta_pf >= 0.10:
                print(f"  >>> MARGINAL: Weak signal, needs more investigation")
            else:
                print(f"  >>> ICT sequence does NOT separate winners from losers")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 120)
    print("SUMMARY -- ICT CONTEXTUAL EDGE DIAGNOSTIC")
    print("=" * 120)
    print(f"Total trades analyzed: {len(df)}")
    if "pre_sweep" in df.columns:
        n_sweep = df["pre_sweep"].sum()
        print(f"Trades with pre-sweep: {n_sweep} ({n_sweep/len(df)*100:.1f}%)")
    print(f"\nBaseline: R={m['R']:+.1f} | PF={m['PF']:.2f} | PPDD={m['PPDD']:.2f}")
    print("\nCheck each feature section above for SIGNAL / MARGINAL / DEAD verdict.")
    print("=" * 120)


if __name__ == "__main__":
    main()
