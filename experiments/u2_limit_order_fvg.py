"""
experiments/u2_limit_order_fvg.py — Limit-Order-at-FVG-Zone Entry System
=========================================================================

Architecture:
  1. Detect 5m FVGs (same pipeline)
  2. Each active FVG zone = potential limit order
  3. Bull FVG: limit BUY at fvg_bull_top (top of the gap)
  4. Bear FVG: limit SELL at fvg_bear_bottom (bottom of the gap)
  5. When price touches the level -> fill at limit price
  6. Stop = fvg far side (fvg_bull_bottom for longs, fvg_bear_top for shorts)
  7. Cancel if: FVG invalidated (close through) / too old / EOD

NOT lookahead: FVG zone is known from the past. Limit order placed in advance.
Price touching the level is a real-time event.

Sweep parameters:
  A (stop): A1=far side, A2=far side*1.15, A3=far side-1*ATR
  B (min stop): B1=5pts, B2=0.5*ATR, B3=1.0*ATR, B4=1.7*ATR
  C (max age): C1=50, C2=100, C3=200, C4=500 bars
  D (size filter): D1=0.3*ATR, D2=0.5*ATR, D3=1.0*ATR

Usage: python experiments/u2_limit_order_fvg.py
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter
from itertools import product

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
    _find_nth_swing_price,
    build_liquidity_ladder_long,
    build_liquidity_ladder_short,
)
from experiments.pure_liquidity_tp import (
    run_backtest_pure_liquidity,
    _compute_tp_v1_raw_irl,
)
from experiments.a2c_stop_widening_engine import widen_stop_array
from features.fvg import detect_fvg

SEP = "=" * 120
THIN = "-" * 80


# ======================================================================
# FVG Zone Data Structure
# ======================================================================
@dataclass
class FVGZone:
    type: str            # 'bull' or 'bear'
    top: float           # fvg top price
    bottom: float        # fvg bottom price
    size: float          # top - bottom
    birth_bar: int       # 5m bar index when FVG was created (visible)
    birth_atr: float     # ATR at creation
    used: bool = False   # True after a fill (1 fill per zone max)


# ======================================================================
# Step 1: Build Active FVG Zone Tracker
# ======================================================================
def build_active_zones_per_bar(
    nq: pd.DataFrame,
    atr_arr: np.ndarray,
    min_fvg_atr_mult: float = 0.3,
    max_age: int = 200,
    max_zones: int = 30,
) -> list[list[FVGZone]]:
    """
    For each 5m bar, maintain a list of active FVG zones.

    Returns list of length n, where each element is a list of FVGZone
    objects that are active AT that bar (available for limit order placement).

    Zone lifecycle:
      - Created: when FVG detection fires (already shift(1) in detect_fvg)
      - Active: price has not closed through it, not expired, not used
      - Invalidated: price CLOSES through (close < bottom for bull, close > top for bear)
      - Expired: older than max_age bars
    """
    fvg_df = detect_fvg(nq)
    n = len(nq)
    close = nq["close"].values
    high = nq["high"].values
    low = nq["low"].values

    bull_mask = fvg_df["fvg_bull"].values
    bear_mask = fvg_df["fvg_bear"].values

    # We return per-bar active zone snapshots for the backtest engine
    # But for memory efficiency, we just track the active list and
    # the engine will call us bar-by-bar. Instead, let's return a
    # generator-friendly structure.

    # Actually for the bar-by-bar backtest, we'll integrate zone tracking
    # directly in the backtest loop. So this function returns the fvg_df
    # and we track zones inline.

    return fvg_df


# ======================================================================
# Step 2: Limit Order Fill Detection + Full Backtest
# ======================================================================
def run_limit_order_backtest(
    d: dict,
    d_extra: dict,
    *,
    # Stop placement strategy
    stop_strategy: str = "A1",   # A1/A2/A3
    # Min stop filter
    min_stop_mode: str = "B1",   # B1/B2/B3/B4
    # Max FVG age
    max_fvg_age: int = 200,      # 5m bars
    # FVG size filter
    fvg_size_mult: float = 0.3,  # min size as ATR multiple
    # Standard Config H++ filters
    block_pm_shorts: bool = True,
    # Tighten factor (for stop_strategy A2)
    stop_buffer_pct: float = 0.15,
    # TP
    tp_strategy: str = "v1",
    tp1_trim_pct: float = 0.50,
    be_after_tp1: bool = False,
    be_after_tp2: bool = False,
) -> tuple[list[dict], dict]:
    """
    Run limit-order-at-FVG-zone backtest on 5m data.

    Returns (trades, stats_dict).
    """
    params = d["params"]
    nq = d["nq"]
    n = d["n"]
    o, h, l, c_arr = d["o"], d["h"], d["l"], d["c"]
    atr_arr = d["atr_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    dates = d["dates"]
    dow_arr = d["dow_arr"]
    et_frac_arr = d["et_frac_arr"]
    swing_high_mask = d["swing_high_mask"]
    swing_low_mask = d["swing_low_mask"]
    irl_target_arr = d["irl_target_arr"]
    sig_type = d["sig_type"]
    has_smt_arr = d["has_smt_arr"]

    # Config from params.yaml
    pos_params = params["position"]
    risk_params = params["risk"]
    bt_params = params["backtest"]
    grading_params = params["grading"]
    smt_cfg = params.get("smt", {})
    session_regime = params.get("session_regime", {})

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

    low_arr = nq["low"].values
    high_arr = nq["high"].values

    # ================================================================
    # Detect FVGs on 5m data
    # ================================================================
    fvg_df = detect_fvg(nq)
    bull_mask = fvg_df["fvg_bull"].values
    bear_mask = fvg_df["fvg_bear"].values
    fvg_bull_top = fvg_df["fvg_bull_top"].values
    fvg_bull_bottom = fvg_df["fvg_bull_bottom"].values
    fvg_bear_top = fvg_df["fvg_bear_top"].values
    fvg_bear_bottom = fvg_df["fvg_bear_bottom"].values
    fvg_size_arr = fvg_df["fvg_size"].values

    # ================================================================
    # Precompute IRL targets for limit fills (nearest swing)
    # We need irl_target for each bar, using shift(1).ffill() swing prices
    # ================================================================
    from features.swing import compute_swing_levels
    swing_p = {"left_bars": params["swing"]["left_bars"],
               "right_bars": params["swing"]["right_bars"]}
    swings = compute_swing_levels(nq, swing_p)
    swings["swing_high_price"] = swings["swing_high_price"].shift(1).ffill()
    swings["swing_low_price"] = swings["swing_low_price"].shift(1).ffill()
    irl_high_arr = swings["swing_high_price"].values  # nearest swing high (for long TP1)
    irl_low_arr = swings["swing_low_price"].values     # nearest swing low (for short TP1)

    # ================================================================
    # BAR-BY-BAR LOOP
    # ================================================================
    active_zones: list[FVGZone] = []
    trades: list[dict] = []

    # Position state
    in_position = False
    pos_direction = 0
    pos_entry_idx = 0
    pos_entry_price = 0.0
    pos_stop = 0.0
    pos_tp1 = 0.0
    pos_tp2 = 0.0
    pos_contracts = 0
    pos_remaining_contracts = 0
    pos_trim_stage = 0
    pos_be_stop = 0.0
    pos_trail_stop = 0.0
    pos_grade = ""
    pos_bias_dir = 0.0
    pos_regime = 0.0
    pos_is_multi_tp = False
    pos_trim1_contracts = 0
    pos_trim2_contracts = 0
    pos_tp1_pnl_pts = 0.0
    pos_tp2_pnl_pts = 0.0
    pos_single_trimmed = False
    pos_single_trim_pct = 1.0
    pos_tp1_hit_bar = -1

    # Daily state
    current_date = None
    daily_pnl_r = 0.0
    consecutive_losses = 0
    day_stopped = False

    # Stats
    zones_created = 0
    zones_invalidated = 0
    zones_expired = 0
    zones_filled = 0
    fill_wait_bars = []  # bars from zone creation to fill

    for i in range(n):
        bar_date = dates[i]
        if bar_date != current_date:
            current_date = bar_date
            daily_pnl_r = 0.0
            consecutive_losses = 0
            day_stopped = False

        # ================================================================
        # Phase A: Register new FVG zones
        # ================================================================
        if bull_mask[i]:
            atr_val = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
            zone_size = fvg_bull_top[i] - fvg_bull_bottom[i]
            if zone_size >= fvg_size_mult * atr_val and zone_size > 0:
                zone = FVGZone(
                    type="bull",
                    top=fvg_bull_top[i],
                    bottom=fvg_bull_bottom[i],
                    size=zone_size,
                    birth_bar=i,
                    birth_atr=atr_val,
                )
                active_zones.append(zone)
                zones_created += 1

        if bear_mask[i]:
            atr_val = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
            zone_size = fvg_bear_top[i] - fvg_bear_bottom[i]
            if zone_size >= fvg_size_mult * atr_val and zone_size > 0:
                zone = FVGZone(
                    type="bear",
                    top=fvg_bear_top[i],
                    bottom=fvg_bear_bottom[i],
                    size=zone_size,
                    birth_bar=i,
                    birth_atr=atr_val,
                )
                active_zones.append(zone)
                zones_created += 1

        # ================================================================
        # Phase B: Invalidate / expire zones
        # ================================================================
        surviving: list[FVGZone] = []
        for zone in active_zones:
            age = i - zone.birth_bar
            if age > max_fvg_age:
                zones_expired += 1
                continue
            # Invalidation: price closes through
            if zone.type == "bull" and c_arr[i] < zone.bottom:
                zones_invalidated += 1
                continue
            if zone.type == "bear" and c_arr[i] > zone.top:
                zones_invalidated += 1
                continue
            if zone.used:
                continue
            surviving.append(zone)

        # Cap at max_zones (prune oldest)
        if len(surviving) > 30:
            surviving = surviving[-30:]
        active_zones = surviving

        # ================================================================
        # Phase C: EXIT MANAGEMENT (if in position)
        # ================================================================
        if in_position:
            exited = False
            exit_reason = ""
            exit_price = 0.0
            exit_contracts = pos_remaining_contracts

            # --- MANDATORY EOD CLOSE at 15:55 ET ---
            if et_frac_arr[i] >= 15.917:
                exit_price = c_arr[i] - slippage_points if pos_direction == 1 else c_arr[i] + slippage_points
                exit_reason = "eod_close"
                exited = True

            # --- Multi-TP EXIT (LONGS: 50/50/0) ---
            if not exited and pos_is_multi_tp and pos_direction == 1:
                # Effective stop
                if pos_trim_stage >= 1 and be_after_tp1:
                    eff_stop = max(pos_stop, pos_be_stop) if pos_be_stop > 0 else pos_stop
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

                # Stage 0: TP1
                elif pos_trim_stage == 0 and h[i] >= pos_tp1:
                    tc1 = max(1, int(np.ceil(pos_contracts * tp1_trim_pct)))
                    pos_trim1_contracts = tc1
                    pos_remaining_contracts = pos_contracts - tc1
                    pos_tp1_pnl_pts = pos_tp1 - pos_entry_price
                    pos_trim_stage = 1
                    pos_tp1_hit_bar = i  # BUG FIX: record TP1 bar
                    if be_after_tp1:
                        pos_be_stop = pos_entry_price
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True

                # Stage 1: TP2 - take ALL remaining (BUG FIX: only on NEXT bar after TP1)
                if not exited and pos_trim_stage == 1 and i > pos_tp1_hit_bar and h[i] >= pos_tp2:
                    tc2 = pos_remaining_contracts
                    pos_trim2_contracts = tc2
                    pos_remaining_contracts -= tc2
                    pos_tp2_pnl_pts = pos_tp2 - pos_entry_price
                    pos_trim_stage = 2
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp2
                        exit_reason = "tp2"
                        exit_contracts = pos_contracts
                        exited = True

            # --- Single-TP EXIT (SHORTS: scalp) ---
            elif not exited and not pos_is_multi_tp and pos_direction == -1:
                eff_stop = pos_stop
                if h[i] >= eff_stop:
                    exit_price = eff_stop + slippage_points
                    exit_reason = "stop"
                    exited = True
                elif not pos_single_trimmed and l[i] <= pos_tp1:
                    tc = max(1, int(pos_contracts * pos_single_trim_pct))
                    pos_remaining_contracts = pos_contracts - tc
                    pos_single_trimmed = True
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True

            # --- PNL CALCULATION ---
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
                    "exit_time": nq.index[i],
                    "r": r_mult, "reason": exit_reason, "dir": pos_direction,
                    "type": "limit_fvg", "trimmed": any_trimmed, "grade": pos_grade,
                    "entry_price": pos_entry_price, "exit_price": exit_price,
                    "stop_price": pos_stop, "tp1_price": pos_tp1, "tp2_price": pos_tp2,
                    "pnl_dollars": total_pnl, "trim_stage": pos_trim_stage,
                    "source": "limit",
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

        # ================================================================
        # Phase D: ENTRY via limit order fills (only if not in position)
        # ================================================================
        if in_position:
            continue
        if day_stopped:
            continue
        if news_blackout_arr is not None and news_blackout_arr[i]:
            continue

        et_frac = et_frac_arr[i]

        # Session filter: NY only (after 10:00 ET), skip observation period
        if 9.5 <= et_frac <= 10.0:
            continue
        is_ny = (10.0 <= et_frac < 16.0)
        is_london = (3.0 <= et_frac < 9.5)
        is_asia = (et_frac >= 18.0) or (et_frac < 3.0)

        # Allow MSS-bypass for London/Asia (same logic as Config H++)
        # For limit orders, we only trade NY (simplify)
        if not is_ny:
            continue

        # Check all active zones for fills
        best_fill = None
        best_fill_zone = None

        for zone in active_zones:
            if zone.used:
                continue
            # Anti-lookahead: zone must be created BEFORE this bar
            if zone.birth_bar >= i:
                continue

            direction = 1 if zone.type == "bull" else -1

            # PM shorts block
            if block_pm_shorts and direction == -1 and et_frac >= 14.0:
                continue

            # Determine entry price and stop price
            if zone.type == "bull":
                limit_price = zone.top  # buy at top of bull FVG
                if stop_strategy == "A1":
                    stop_price = zone.bottom
                elif stop_strategy == "A2":
                    stop_price = zone.bottom - zone.size * stop_buffer_pct
                elif stop_strategy == "A3":
                    stop_price = zone.bottom - zone.birth_atr
                else:
                    stop_price = zone.bottom

                # Fill condition: bar low touches or goes below the limit price
                if l[i] > limit_price:
                    continue  # price didn't reach our limit

                # Bar-within-bar check: if stop is ALSO hit in same bar, skip
                if l[i] <= stop_price:
                    continue  # conservative: assume stop hit first

            else:  # bear
                limit_price = zone.bottom  # sell at bottom of bear FVG
                if stop_strategy == "A1":
                    stop_price = zone.top
                elif stop_strategy == "A2":
                    stop_price = zone.top + zone.size * stop_buffer_pct
                elif stop_strategy == "A3":
                    stop_price = zone.top + zone.birth_atr
                else:
                    stop_price = zone.top

                # Fill condition: bar high reaches or exceeds the limit price
                if h[i] < limit_price:
                    continue  # price didn't reach our limit

                # Bar-within-bar check
                if h[i] >= stop_price:
                    continue  # conservative: assume stop hit first

            # Min stop filter
            stop_dist = abs(limit_price - stop_price)
            cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0

            if min_stop_mode == "B1":
                if stop_dist < 5.0:
                    continue
            elif min_stop_mode == "B2":
                if stop_dist < 0.5 * cur_atr:
                    continue
            elif min_stop_mode == "B3":
                if stop_dist < 1.0 * cur_atr:
                    continue
            elif min_stop_mode == "B4":
                if stop_dist < 1.7 * cur_atr:
                    continue

            # Also enforce absolute minimum
            if stop_dist < 1.0:
                continue

            # Bias alignment
            bias_opposing = (direction == -np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0)
            if bias_opposing:
                continue

            # Prefer the zone closest to current price (most relevant)
            fill_quality = -abs(c_arr[i] - limit_price)  # closer = better
            if best_fill is None or fill_quality > best_fill:
                best_fill = fill_quality
                best_fill_zone = (zone, direction, limit_price, stop_price, stop_dist, cur_atr)

        # Execute the best fill
        if best_fill_zone is not None:
            zone, direction, entry_p, stop, stop_dist, cur_atr = best_fill_zone
            zone.used = True
            zones_filled += 1
            fill_wait_bars.append(i - zone.birth_bar)

            # Grade
            ba = 1.0 if (direction == np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0) else 0.0
            grade = _compute_grade_fast(ba, regime_arr[i])

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
                    zones_filled -= 1
                    fill_wait_bars.pop()
                    continue

            # No slippage on limit orders (this is the whole point!)
            actual_entry = entry_p

            contracts = max(1, int(r_amount / (stop_dist * point_value))) if stop_dist > 0 and point_value > 0 else 0
            if contracts <= 0:
                zone.used = False
                zones_filled -= 1
                fill_wait_bars.pop()
                continue

            # TP computation
            if direction == 1:
                # Long: use raw IRL + liquidity ladder
                irl_target = irl_high_arr[i]
                if np.isnan(irl_target) or irl_target <= actual_entry:
                    irl_target = actual_entry + stop_dist * 2.0

                tp1, tp2 = _compute_tp_v1_raw_irl(
                    actual_entry, stop, irl_target, i, d_extra, cur_atr)

                in_position = True
                pos_direction = 1
                pos_entry_idx = i
                pos_entry_price = actual_entry
                pos_stop = stop
                pos_tp1 = tp1
                pos_tp2 = tp2
                pos_contracts = contracts
                pos_remaining_contracts = contracts
                pos_trim_stage = 0
                pos_be_stop = 0.0
                pos_trail_stop = 0.0
                pos_grade = grade
                pos_bias_dir = bias_dir_arr[i]
                pos_regime = regime_arr[i]
                pos_is_multi_tp = True
                pos_trim1_contracts = 0
                pos_trim2_contracts = 0
                pos_tp1_pnl_pts = 0.0
                pos_tp2_pnl_pts = 0.0
                pos_single_trimmed = False
                pos_tp1_hit_bar = -1

            else:
                # Short: dual mode scalp (100% exit at RR target)
                short_rr = params.get("dual_mode", {}).get("short_rr", 0.625)
                tp1 = actual_entry - stop_dist * short_rr

                in_position = True
                pos_direction = -1
                pos_entry_idx = i
                pos_entry_price = actual_entry
                pos_stop = stop
                pos_tp1 = tp1
                pos_tp2 = 0.0
                pos_contracts = contracts
                pos_remaining_contracts = contracts
                pos_trim_stage = 0
                pos_be_stop = 0.0
                pos_trail_stop = 0.0
                pos_grade = grade
                pos_bias_dir = bias_dir_arr[i]
                pos_regime = regime_arr[i]
                pos_is_multi_tp = False
                pos_trim1_contracts = 0
                pos_trim2_contracts = 0
                pos_tp1_pnl_pts = 0.0
                pos_tp2_pnl_pts = 0.0
                pos_single_trimmed = False
                pos_single_trim_pct = 1.0
                pos_tp1_hit_bar = -1

    # Force close if still in position
    if in_position and pos_entry_idx < n:
        last_i = n - 1
        exit_price = c_arr[last_i]
        pnl_pts = (exit_price - pos_entry_price) if pos_direction == 1 else (pos_entry_price - exit_price)
        total_pnl = pnl_pts * point_value * pos_remaining_contracts
        total_comm = commission_per_side * 2 * pos_remaining_contracts
        total_pnl -= total_comm
        stop_dist_exit = abs(pos_entry_price - pos_stop)
        total_risk = stop_dist_exit * point_value * pos_contracts
        r_mult = total_pnl / total_risk if total_risk > 0 else 0.0
        trades.append({
            "entry_time": nq.index[pos_entry_idx],
            "exit_time": nq.index[last_i],
            "r": r_mult, "reason": "force_close", "dir": pos_direction,
            "type": "limit_fvg", "trimmed": False, "grade": pos_grade,
            "entry_price": pos_entry_price, "exit_price": exit_price,
            "stop_price": pos_stop, "tp1_price": pos_tp1, "tp2_price": pos_tp2,
            "pnl_dollars": total_pnl, "trim_stage": 0,
            "source": "limit",
        })

    stats = {
        "zones_created": zones_created,
        "zones_invalidated": zones_invalidated,
        "zones_expired": zones_expired,
        "zones_filled": zones_filled,
        "fill_rate_pct": 100.0 * zones_filled / zones_created if zones_created > 0 else 0.0,
        "avg_fill_wait_bars": np.mean(fill_wait_bars) if fill_wait_bars else 0.0,
        "median_fill_wait_bars": np.median(fill_wait_bars) if fill_wait_bars else 0.0,
    }
    return trades, stats


# ======================================================================
# Combined backtest: limit + 5m signals
# ======================================================================
def run_combined_backtest(
    d: dict,
    d_extra: dict,
    trades_5m: list[dict],
    trades_limit: list[dict],
) -> list[dict]:
    """
    Combine limit order trades with 5m Config H++ trades.
    One position at a time, 5m has priority.
    """
    for t in trades_5m:
        if "source" not in t:
            t["source"] = "5m"

    all_trades = sorted(trades_5m + trades_limit, key=lambda t: t["entry_time"])

    combined = []
    last_exit = pd.Timestamp.min.tz_localize("UTC")

    for t in all_trades:
        entry_t = t["entry_time"]
        if hasattr(entry_t, "tz") and entry_t.tz is None:
            entry_t = entry_t.tz_localize("UTC")
        if entry_t <= last_exit:
            continue
        combined.append(t)
        exit_t = t["exit_time"]
        if hasattr(exit_t, "tz") and exit_t.tz is None:
            exit_t = exit_t.tz_localize("UTC")
        last_exit = exit_t

    return combined


# ======================================================================
# Main
# ======================================================================
def main():
    t_start = _time.perf_counter()

    print(SEP)
    print("U2: LIMIT ORDER AT FVG ZONE — Entry System Experiment")
    print("  Bull FVG: limit BUY at fvg_top, stop at fvg_bottom")
    print("  Bear FVG: limit SELL at fvg_bottom, stop at fvg_top")
    print("  Fill when price touches zone boundary, no confirmation needed")
    print(SEP)

    # ================================================================
    # LOAD DATA
    # ================================================================
    print("\n[STEP 1] Loading 5m data + caches...")
    d = load_all()
    d_extra = prepare_liquidity_data(d)
    nq = d["nq"]
    n = d["n"]
    print(f"  {n} bars, {nq.index[0]} to {nq.index[-1]}")

    # ================================================================
    # CONFIG H++ BASELINE (5m only, stop*0.85)
    # ================================================================
    print(f"\n{SEP}")
    print("[STEP 2] Config H++ Baseline (5m only)")
    print(SEP)

    original_stop = d["model_stop_arr"].copy()
    tightened_stop = widen_stop_array(
        original_stop, d["entry_price_arr"], d["sig_dir"],
        d["sig_mask"], 0.85)
    d["model_stop_arr"] = tightened_stop

    baseline_trades, baseline_diag = run_backtest_pure_liquidity(
        d, d_extra,
        sq_long=0.68, sq_short=0.80, min_stop_atr=1.7,
        block_pm_shorts=True, tp_strategy="v1",
        tp1_trim_pct=0.50, be_after_tp1=False, be_after_tp2=False,
    )
    d["model_stop_arr"] = original_stop

    baseline_m = compute_metrics(baseline_trades)
    print_metrics("Config H++ (5m baseline)", baseline_m)
    for t in baseline_trades:
        t["source"] = "5m"

    # ================================================================
    # STEP 3: FVG ZONE ANALYSIS
    # ================================================================
    print(f"\n{SEP}")
    print("[STEP 3] FVG Zone Analysis")
    print(SEP)

    fvg_df = detect_fvg(nq)
    n_bull = int(fvg_df["fvg_bull"].sum())
    n_bear = int(fvg_df["fvg_bear"].sum())
    print(f"  Total 5m FVGs: {n_bull} bullish, {n_bear} bearish = {n_bull + n_bear} total")
    valid_sizes = fvg_df["fvg_size"].dropna()
    print(f"  FVG size: mean={valid_sizes.mean():.1f}, median={valid_sizes.median():.1f}, "
          f"min={valid_sizes.min():.1f}, max={valid_sizes.max():.1f} pts")

    atr_mean = np.nanmean(d["atr_arr"])
    print(f"  5m ATR mean: {atr_mean:.1f} pts")
    print(f"  0.3*ATR = {0.3*atr_mean:.1f}, 0.5*ATR = {0.5*atr_mean:.1f}, 1.0*ATR = {atr_mean:.1f}")

    # ================================================================
    # STEP 4: PARAMETER SWEEP
    # ================================================================
    print(f"\n{SEP}")
    print("[STEP 4] Parameter Sweep — Standalone Limit Orders")
    print(SEP)

    stop_strategies = ["A1", "A2", "A3"]
    min_stop_modes = ["B1", "B2", "B3", "B4"]
    max_ages = [("C1", 50), ("C2", 100), ("C3", 200), ("C4", 500)]
    fvg_sizes = [("D1", 0.3), ("D2", 0.5), ("D3", 1.0)]

    # Phase 1: Coarse sweep — fix D1 and sweep A x B x C
    print("\n  Phase 1: Coarse sweep (D1=0.3*ATR fixed)")
    print(f"  {'Config':20s} | {'Trades':>6s} | {'R':>8s} | {'PPDD':>7s} | {'PF':>6s} | {'WR':>6s} | {'MaxDD':>6s} | {'avgR':>8s} | {'AvgStop':>7s} | {'Fills/yr':>8s}")
    print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}")

    all_results = {}
    for a_strat in stop_strategies:
        for b_mode in min_stop_modes:
            for c_label, c_val in max_ages:
                config_name = f"{a_strat}_{b_mode}_{c_label}_D1"
                trades_l, stats_l = run_limit_order_backtest(
                    d, d_extra,
                    stop_strategy=a_strat,
                    min_stop_mode=b_mode,
                    max_fvg_age=c_val,
                    fvg_size_mult=0.3,
                    block_pm_shorts=True,
                )
                m = compute_metrics(trades_l)
                avg_stop = np.mean([abs(t["entry_price"] - t["stop_price"]) for t in trades_l]) if trades_l else 0
                years = 10.0
                fills_yr = len(trades_l) / years if years > 0 else 0
                all_results[config_name] = {
                    "trades": trades_l, "metrics": m, "stats": stats_l,
                    "avg_stop": avg_stop, "fills_yr": fills_yr,
                }
                print(f"  {config_name:20s} | {m['trades']:>6d} | {m['R']:>+8.1f} | {m['PPDD']:>7.2f} | "
                      f"{m['PF']:>6.2f} | {m['WR']:>5.1f}% | {m['MaxDD']:>6.1f} | {m['avgR']:>+8.4f} | "
                      f"{avg_stop:>7.1f} | {fills_yr:>8.1f}")

    # Phase 2: Best A/B/C combo, now sweep D
    best_abc = max(all_results.keys(), key=lambda k: all_results[k]["metrics"]["PPDD"])
    best_abc_parts = best_abc.split("_")
    best_a = best_abc_parts[0]
    best_b = best_abc_parts[1]
    best_c_label = best_abc_parts[2]
    best_c_val = dict(max_ages)[best_c_label]

    print(f"\n  Best A/B/C combo: {best_a}_{best_b}_{best_c_label} "
          f"(R={all_results[best_abc]['metrics']['R']:+.1f}, PPDD={all_results[best_abc]['metrics']['PPDD']:.2f})")

    print(f"\n  Phase 2: FVG Size sweep (best A/B/C fixed = {best_a}_{best_b}_{best_c_label})")
    print(f"  {'Config':20s} | {'Trades':>6s} | {'R':>8s} | {'PPDD':>7s} | {'PF':>6s} | {'WR':>6s} | {'MaxDD':>6s} | {'avgR':>8s} | {'AvgStop':>7s}")
    print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}")

    for d_label, d_val in fvg_sizes:
        config_name = f"{best_a}_{best_b}_{best_c_label}_{d_label}"
        if config_name in all_results:
            m = all_results[config_name]["metrics"]
            avg_stop = all_results[config_name]["avg_stop"]
        else:
            trades_l, stats_l = run_limit_order_backtest(
                d, d_extra,
                stop_strategy=best_a,
                min_stop_mode=best_b,
                max_fvg_age=best_c_val,
                fvg_size_mult=d_val,
                block_pm_shorts=True,
            )
            m = compute_metrics(trades_l)
            avg_stop = np.mean([abs(t["entry_price"] - t["stop_price"]) for t in trades_l]) if trades_l else 0
            fills_yr = len(trades_l) / 10.0
            all_results[config_name] = {
                "trades": trades_l, "metrics": m, "stats": stats_l,
                "avg_stop": avg_stop, "fills_yr": fills_yr,
            }
        print(f"  {config_name:20s} | {m['trades']:>6d} | {m['R']:>+8.1f} | {m['PPDD']:>7.2f} | "
              f"{m['PF']:>6.2f} | {m['WR']:>5.1f}% | {m['MaxDD']:>6.1f} | {m['avgR']:>+8.4f} | "
              f"{avg_stop:>7.1f}")

    # ================================================================
    # STEP 5: TOP 5 STANDALONE CONFIGS
    # ================================================================
    print(f"\n{SEP}")
    print("[STEP 5] Top 5 Standalone Limit-Order Configs (by PPDD)")
    print(SEP)

    sorted_configs = sorted(all_results.keys(),
                            key=lambda k: all_results[k]["metrics"]["PPDD"], reverse=True)
    top5 = sorted_configs[:5]

    print(f"\n  {'Rank':>4s} | {'Config':20s} | {'Trades':>6s} | {'R':>8s} | {'PPDD':>7s} | {'PF':>6s} | {'WR':>6s} | {'MaxDD':>6s} | {'Fills/yr':>8s}")
    print(f"  {'-'*4}-+-{'-'*20}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")
    for rank, cfg in enumerate(top5, 1):
        m = all_results[cfg]["metrics"]
        fy = all_results[cfg]["fills_yr"]
        print(f"  {rank:>4d} | {cfg:20s} | {m['trades']:>6d} | {m['R']:>+8.1f} | {m['PPDD']:>7.2f} | "
              f"{m['PF']:>6.2f} | {m['WR']:>5.1f}% | {m['MaxDD']:>6.1f} | {fy:>8.1f}")

    best_standalone = top5[0]
    best_standalone_m = all_results[best_standalone]["metrics"]
    best_standalone_trades = all_results[best_standalone]["trades"]
    best_standalone_stats = all_results[best_standalone]["stats"]

    print(f"\n  BEST STANDALONE: {best_standalone}")
    print(f"    R={best_standalone_m['R']:+.1f} | PPDD={best_standalone_m['PPDD']:.2f} | PF={best_standalone_m['PF']:.2f} | MaxDD={best_standalone_m['MaxDD']:.1f}")
    print(f"    Zones created: {best_standalone_stats['zones_created']}")
    print(f"    Zones filled: {best_standalone_stats['zones_filled']} ({best_standalone_stats['fill_rate_pct']:.1f}%)")
    print(f"    Zones invalidated: {best_standalone_stats['zones_invalidated']}")
    print(f"    Zones expired: {best_standalone_stats['zones_expired']}")
    print(f"    Avg fill wait: {best_standalone_stats['avg_fill_wait_bars']:.1f} bars ({best_standalone_stats['avg_fill_wait_bars']*5:.0f} min)")

    # ================================================================
    # STEP 6: COMBINED (limit + 5m Config H++)
    # ================================================================
    print(f"\n{SEP}")
    print("[STEP 6] Combined: Limit Orders + 5m Config H++ (top 5 limit configs)")
    print(SEP)

    print(f"\n  {'Config':20s} | {'Total':>5s} | {'5m':>4s} | {'Lim':>4s} | {'R':>8s} | {'PPDD':>7s} | {'PF':>6s} | {'WR':>6s} | {'MaxDD':>6s} | {'dR':>6s} | {'dPPDD':>7s}")
    print(f"  {'-'*20}-+-{'-'*5}-+-{'-'*4}-+-{'-'*4}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")

    # Baseline row
    print(f"  {'H++ baseline':20s} | {baseline_m['trades']:>5d} | {baseline_m['trades']:>4d} | {'0':>4s} | "
          f"{baseline_m['R']:>+8.1f} | {baseline_m['PPDD']:>7.2f} | {baseline_m['PF']:>6.2f} | {baseline_m['WR']:>5.1f}% | "
          f"{baseline_m['MaxDD']:>6.1f} | {'--':>6s} | {'--':>7s}")

    combined_results = {}
    for cfg in top5:
        limit_trades = all_results[cfg]["trades"]
        combined = run_combined_backtest(d, d_extra,
            [t.copy() for t in baseline_trades],
            [t.copy() for t in limit_trades])
        m_c = compute_metrics(combined)
        n5 = sum(1 for t in combined if t.get("source") == "5m")
        nl = sum(1 for t in combined if t.get("source") == "limit")
        dR = m_c["R"] - baseline_m["R"]
        dPPDD = m_c["PPDD"] - baseline_m["PPDD"]
        combined_results[cfg] = {"trades": combined, "metrics": m_c, "n5": n5, "nl": nl}
        print(f"  {cfg:20s} | {m_c['trades']:>5d} | {n5:>4d} | {nl:>4d} | "
              f"{m_c['R']:>+8.1f} | {m_c['PPDD']:>7.2f} | {m_c['PF']:>6.2f} | {m_c['WR']:>5.1f}% | "
              f"{m_c['MaxDD']:>6.1f} | {dR:>+6.1f} | {dPPDD:>+7.2f}")

    # Best combined
    best_combined_cfg = max(combined_results.keys(), key=lambda k: combined_results[k]["metrics"]["PPDD"])
    best_c_m = combined_results[best_combined_cfg]["metrics"]
    best_c_trades = combined_results[best_combined_cfg]["trades"]

    print(f"\n  BEST COMBINED: {best_combined_cfg}")
    print(f"    R={best_c_m['R']:+.1f} | PPDD={best_c_m['PPDD']:.2f} | PF={best_c_m['PF']:.2f} | MaxDD={best_c_m['MaxDD']:.1f}")
    print(f"    vs baseline: dR={best_c_m['R']-baseline_m['R']:+.1f} | dPPDD={best_c_m['PPDD']-baseline_m['PPDD']:+.2f}")

    # ================================================================
    # STEP 7: WALK-FORWARD for top 3 configs
    # ================================================================
    print(f"\n{SEP}")
    print("[STEP 7] Walk-Forward Validation (top 3 standalone)")
    print(SEP)

    for rank, cfg in enumerate(top5[:3], 1):
        trades_l = all_results[cfg]["trades"]
        wf = walk_forward_metrics(trades_l)
        print(f"\n  #{rank} {cfg} (standalone)")
        if wf:
            neg_years = sum(1 for yr in wf if yr["R"] < 0)
            print(f"  {'Year':>6s} | {'N':>4s} | {'R':>8s} | {'WR':>6s} | {'PF':>6s} | {'PPDD':>7s}")
            print(f"  {'-'*6}-+-{'-'*4}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")
            for yr in wf:
                print(f"  {yr['year']:>6d} | {yr['n']:>4d} | {yr['R']:>+8.1f} | {yr['WR']:>5.1f}% | {yr['PF']:>6.2f} | {yr['PPDD']:>7.2f}")
            print(f"  Negative years: {neg_years}/{len(wf)}")
        else:
            print("  No trades.")

    # Walk-forward for best combined
    print(f"\n  Best combined: {best_combined_cfg}")
    wf_c = walk_forward_metrics(best_c_trades)
    if wf_c:
        neg_years = sum(1 for yr in wf_c if yr["R"] < 0)
        print(f"  {'Year':>6s} | {'N':>4s} | {'5m':>4s} | {'Lim':>4s} | {'R':>8s} | {'WR':>6s} | {'PF':>6s} | {'PPDD':>7s}")
        print(f"  {'-'*6}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")
        for yr in wf_c:
            y = yr["year"]
            n5 = sum(1 for t in best_c_trades if t.get("source") == "5m" and pd.Timestamp(t["entry_time"]).year == y)
            nl = sum(1 for t in best_c_trades if t.get("source") == "limit" and pd.Timestamp(t["entry_time"]).year == y)
            print(f"  {y:>6d} | {yr['n']:>4d} | {n5:>4d} | {nl:>4d} | {yr['R']:>+8.1f} | {yr['WR']:>5.1f}% | {yr['PF']:>6.2f} | {yr['PPDD']:>7.2f}")
        print(f"  Negative years: {neg_years}/{len(wf_c)}")

    # ================================================================
    # STEP 8: ANALYSIS
    # ================================================================
    print(f"\n{SEP}")
    print("[STEP 8] Deep Analysis")
    print(SEP)

    best_limit_trades = best_standalone_trades
    best_limit_stats = best_standalone_stats

    # 8.1: Fills per year
    if best_limit_trades:
        df_t = pd.DataFrame(best_limit_trades)
        df_t["year"] = pd.to_datetime(df_t["entry_time"]).dt.year
        yearly = df_t.groupby("year").size()
        print(f"\n  8.1 Fills per year (best standalone: {best_standalone}):")
        for y, cnt in yearly.items():
            print(f"    {y}: {cnt} fills")
        print(f"    Average: {yearly.mean():.1f} fills/year")
    else:
        print("\n  8.1 No fills.")

    # 8.2: Zone fill rate
    print(f"\n  8.2 Zone Statistics:")
    print(f"    Zones created:     {best_limit_stats['zones_created']}")
    print(f"    Zones filled:      {best_limit_stats['zones_filled']} ({best_limit_stats['fill_rate_pct']:.1f}%)")
    print(f"    Zones invalidated: {best_limit_stats['zones_invalidated']}")
    print(f"    Zones expired:     {best_limit_stats['zones_expired']}")

    # 8.3: Fill wait time
    print(f"\n  8.3 Time from zone creation to fill:")
    print(f"    Mean:   {best_limit_stats['avg_fill_wait_bars']:.1f} bars ({best_limit_stats['avg_fill_wait_bars']*5:.0f} min)")
    print(f"    Median: {best_limit_stats['median_fill_wait_bars']:.1f} bars ({best_limit_stats['median_fill_wait_bars']*5:.0f} min)")

    # 8.4: Fill quality comparison
    if best_limit_trades:
        r_limit = np.array([t["r"] for t in best_limit_trades])
        r_5m = np.array([t["r"] for t in baseline_trades])
        print(f"\n  8.4 Fill Quality Comparison:")
        print(f"    {'Metric':25s} | {'Limit Orders':>14s} | {'5m Signals':>14s}")
        print(f"    {'-'*25}-+-{'-'*14}-+-{'-'*14}")
        print(f"    {'Trades':25s} | {len(r_limit):>14d} | {len(r_5m):>14d}")
        print(f"    {'Total R':25s} | {r_limit.sum():>+14.1f} | {r_5m.sum():>+14.1f}")
        print(f"    {'Avg R':25s} | {r_limit.mean():>+14.4f} | {r_5m.mean():>+14.4f}")
        print(f"    {'Win Rate':25s} | {100*(r_limit>0).mean():>13.1f}% | {100*(r_5m>0).mean():>13.1f}%")
        print(f"    {'Median R':25s} | {np.median(r_limit):>+14.4f} | {np.median(r_5m):>+14.4f}")

        # Stop distance comparison
        stop_limit = [abs(t["entry_price"] - t["stop_price"]) for t in best_limit_trades]
        stop_5m = [abs(t["entry_price"] - t["stop_price"]) for t in baseline_trades]
        print(f"    {'Avg Stop (pts)':25s} | {np.mean(stop_limit):>14.1f} | {np.mean(stop_5m):>14.1f}")

        # Exit reason distribution
        reasons_l = Counter(t["reason"] for t in best_limit_trades)
        reasons_5 = Counter(t["reason"] for t in baseline_trades)
        all_reasons = sorted(set(list(reasons_l.keys()) + list(reasons_5.keys())))
        print(f"\n    Exit Reason Distribution:")
        print(f"    {'Reason':15s} | {'Limit #':>8s} {'%':>6s} {'R':>8s} | {'5m #':>8s} {'%':>6s} {'R':>8s}")
        print(f"    {'-'*15}-+-{'-'*24}-+-{'-'*24}")
        for reason in all_reasons:
            cl = reasons_l.get(reason, 0)
            c5 = reasons_5.get(reason, 0)
            rl = sum(t["r"] for t in best_limit_trades if t["reason"] == reason)
            r5 = sum(t["r"] for t in baseline_trades if t["reason"] == reason)
            pl = 100 * cl / len(best_limit_trades) if best_limit_trades else 0
            p5 = 100 * c5 / len(baseline_trades) if baseline_trades else 0
            print(f"    {reason:15s} | {cl:>8d} {pl:>5.1f}% {rl:>+8.1f} | {c5:>8d} {p5:>5.1f}% {r5:>+8.1f}")

    # 8.5: Direction breakdown
    if best_limit_trades:
        longs = [t for t in best_limit_trades if t["dir"] == 1]
        shorts = [t for t in best_limit_trades if t["dir"] == -1]
        print(f"\n  8.5 Direction Breakdown (limit orders):")
        if longs:
            lr = np.array([t["r"] for t in longs])
            print(f"    Longs:  {len(longs)}t, R={lr.sum():+.1f}, avgR={lr.mean():+.4f}, WR={100*(lr>0).mean():.1f}%")
        if shorts:
            sr = np.array([t["r"] for t in shorts])
            print(f"    Shorts: {len(shorts)}t, R={sr.sum():+.1f}, avgR={sr.mean():+.4f}, WR={100*(sr>0).mean():.1f}%")

    # 8.6: Time distribution
    if best_limit_trades:
        print(f"\n  8.6 Time Distribution (hour ET):")
        et_hours = []
        for t in best_limit_trades:
            et = t["entry_time"]
            if hasattr(et, "tz_convert"):
                et_local = et.tz_convert("US/Eastern")
            else:
                et_local = pd.Timestamp(et).tz_localize("UTC").tz_convert("US/Eastern")
            et_hours.append(et_local.hour)
        hour_counts = Counter(et_hours)
        for hr in sorted(hour_counts.keys()):
            r_h = sum(t["r"] for t, h_val in zip(best_limit_trades, et_hours) if h_val == hr)
            print(f"    {hr:02d}:00 ET  {hour_counts[hr]:4d} trades  R={r_h:+.1f}")

    # 8.6b: Anti-lookahead verification
    if best_limit_trades:
        print(f"\n  8.6b Anti-Lookahead Verification:")
        violations = 0
        for t in best_limit_trades:
            # Entry must be before exit
            if t["entry_time"] >= t["exit_time"]:
                violations += 1
            # Long: stop < entry < tp1
            if t["dir"] == 1:
                if t["stop_price"] >= t["entry_price"]:
                    violations += 1
                if t["tp1_price"] <= t["entry_price"]:
                    violations += 1
            # Short: stop > entry > tp1
            if t["dir"] == -1:
                if t["stop_price"] <= t["entry_price"]:
                    violations += 1
                if t["tp1_price"] >= t["entry_price"]:
                    violations += 1
        print(f"    {len(best_limit_trades)} trades checked, {violations} violations -> {'PASS' if violations == 0 else 'FAIL'}")

        # Check R distribution sanity
        r_vals = np.array([t["r"] for t in best_limit_trades])
        print(f"    R distribution: min={r_vals.min():.2f}, p5={np.percentile(r_vals,5):.2f}, "
              f"median={np.median(r_vals):.2f}, p95={np.percentile(r_vals,95):.2f}, max={r_vals.max():.2f}")
        print(f"    R > 5: {(r_vals > 5).sum()} trades ({100*(r_vals > 5).mean():.1f}%)")
        print(f"    R > 10: {(r_vals > 10).sum()} trades ({100*(r_vals > 10).mean():.1f}%)")
        print(f"    R < -1: {(r_vals < -1).sum()} trades ({100*(r_vals < -1).mean():.1f}%)")

    # 8.7: Overlap analysis (do limit trades overlap with 5m trades?)
    if best_limit_trades and baseline_trades:
        print(f"\n  8.7 Overlap Analysis (limit vs 5m):")
        limit_dates = set()
        for t in best_limit_trades:
            et = t["entry_time"]
            if hasattr(et, "date"):
                limit_dates.add(et.date() if hasattr(et, "date") else pd.Timestamp(et).date())
            else:
                limit_dates.add(pd.Timestamp(et).date())
        fivem_dates = set()
        for t in baseline_trades:
            et = t["entry_time"]
            if hasattr(et, "date"):
                fivem_dates.add(et.date() if hasattr(et, "date") else pd.Timestamp(et).date())
            else:
                fivem_dates.add(pd.Timestamp(et).date())

        overlap_dates = limit_dates & fivem_dates
        unique_limit_dates = limit_dates - fivem_dates
        print(f"    Limit trade dates: {len(limit_dates)}")
        print(f"    5m trade dates:    {len(fivem_dates)}")
        print(f"    Overlapping dates: {len(overlap_dates)}")
        print(f"    Unique limit dates (no 5m trade): {len(unique_limit_dates)} ({100*len(unique_limit_dates)/max(1,len(limit_dates)):.1f}%)")

    # ================================================================
    # GRAND SUMMARY
    # ================================================================
    print(f"\n{SEP}")
    print("GRAND SUMMARY")
    print(SEP)

    print(f"\n  {'Config':50s} | {'Trades':>6s} | {'R':>9s} | {'PPDD':>7s} | {'PF':>6s} | {'MaxDD':>6s}")
    print(f"  {'-'*50}-+-{'-'*6}-+-{'-'*9}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}")

    # Baseline
    print(f"  {'Config H++ (5m only) [BASELINE]':50s} | {baseline_m['trades']:>6d} | "
          f"{baseline_m['R']:>+9.1f} | {baseline_m['PPDD']:>7.2f} | {baseline_m['PF']:>6.2f} | {baseline_m['MaxDD']:>6.1f}")

    # Best standalone
    print(f"  {'Best Standalone Limit: ' + best_standalone:50s} | {best_standalone_m['trades']:>6d} | "
          f"{best_standalone_m['R']:>+9.1f} | {best_standalone_m['PPDD']:>7.2f} | {best_standalone_m['PF']:>6.2f} | {best_standalone_m['MaxDD']:>6.1f}")

    # Best combined
    print(f"  {'Best Combined: ' + best_combined_cfg:50s} | {best_c_m['trades']:>6d} | "
          f"{best_c_m['R']:>+9.1f} | {best_c_m['PPDD']:>7.2f} | {best_c_m['PF']:>6.2f} | {best_c_m['MaxDD']:>6.1f}")

    dR = best_c_m["R"] - baseline_m["R"]
    dPPDD = best_c_m["PPDD"] - baseline_m["PPDD"]
    dMaxDD = best_c_m["MaxDD"] - baseline_m["MaxDD"]
    print(f"\n  Combined vs baseline: dR={dR:+.1f} | dPPDD={dPPDD:+.2f} | dMaxDD={dMaxDD:+.1f}")

    elapsed = _time.perf_counter() - t_start
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"\n{SEP}")
    print("DONE")
    print(SEP)


if __name__ == "__main__":
    main()
