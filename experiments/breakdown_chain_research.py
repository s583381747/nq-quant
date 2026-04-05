"""
experiments/breakdown_chain_research.py -- Natural ICT Chain Entry Research
==========================================================================

Instead of "U2 + filter", build entry from validated ICT chain:
  1. Breakdown at significant level (PDL best)
  2. FVG forms after breakdown (moderate displacement)
  3. Price returns to FVG zone -> limit entry
  4. Trade management (trim/BE/trail)

Start from 10,000+ breakdown events, follow the chain naturally,
see how many become trades and what the quality is.
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.u2_clean import load_all, compute_metrics, pr, _find_nth_swing, _compute_grade
from experiments.sweep_research import compute_pdhl, compute_htf_swings, compute_htf_swings_4h
from features.fvg import detect_fvg


# ======================================================================
# Step 1: Detect all breakdown events at significant levels
# ======================================================================

def detect_breakdowns(
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    level: np.ndarray,
    level_type: str,
    min_depth_pts: float = 1.0,
) -> list[dict]:
    """Detect bars where price breaks through a significant level.

    For "low" levels: bar closes below the level (breakdown of support).
    For "high" levels: bar closes above the level (breakdown of resistance).

    Simple, clean definition. No reaction window needed -- just the fact
    that price CLOSED through the level.
    """
    n = len(h)
    events = []
    last_event_bar = -10  # avoid duplicate events within 3 bars

    for i in range(1, n):
        lev = level[i]
        if np.isnan(lev) or lev <= 0:
            continue

        if i - last_event_bar < 3:
            continue

        if level_type == "low":
            # Close below level = breakdown of support (stop hunt)
            if c[i] < lev - min_depth_pts and l[i] < lev:
                # Confirm: previous bar's close was ABOVE level (this is a FRESH breakdown)
                if i > 0 and c[i-1] >= lev:
                    events.append({
                        "bar_idx": i,
                        "level_value": lev,
                        "depth_pts": lev - c[i],
                        "wick_below": lev - l[i],
                    })
                    last_event_bar = i
        else:
            if c[i] > lev + min_depth_pts and h[i] > lev:
                if i > 0 and c[i-1] <= lev:
                    events.append({
                        "bar_idx": i,
                        "level_value": lev,
                        "depth_pts": c[i] - lev,
                        "wick_above": h[i] - lev,
                    })
                    last_event_bar = i

    return events


# ======================================================================
# Step 2: After breakdown, find qualifying FVG
# ======================================================================

def find_fvg_after_breakdown(
    fvg_bull_mask: np.ndarray,
    fvg_bull_top: np.ndarray,
    fvg_bull_bottom: np.ndarray,
    fvg_bear_mask: np.ndarray,
    fvg_bear_top: np.ndarray,
    fvg_bear_bottom: np.ndarray,
    atr_arr: np.ndarray,
    breakdown_bar: int,
    breakdown_level_type: str,  # "low" -> look for bull FVG, "high" -> look for bear FVG
    max_wait_bars: int = 30,
    min_disp_atr: float = 0.3,
    max_disp_atr: float = 1.5,
    min_fvg_size_atr: float = 0.3,
    h: np.ndarray = None,
    l: np.ndarray = None,
    c: np.ndarray = None,
    o: np.ndarray = None,
) -> dict | None:
    """After a breakdown, look for a qualifying FVG in the reversal direction.

    Breakdown of LOW level (stop hunt below) -> look for BULL FVG (reversal up)
    Breakdown of HIGH level (stop hunt above) -> look for BEAR FVG (reversal down)
    """
    n = len(fvg_bull_mask)

    if breakdown_level_type == "low":
        mask, tops, bottoms, direction = fvg_bull_mask, fvg_bull_top, fvg_bull_bottom, 1
    else:
        mask, tops, bottoms, direction = fvg_bear_mask, fvg_bear_top, fvg_bear_bottom, -1

    for i in range(breakdown_bar + 1, min(breakdown_bar + max_wait_bars + 1, n)):
        if not mask[i]:
            continue

        top = tops[i]
        bottom = bottoms[i]
        size = top - bottom
        atr_val = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0

        if atr_val > 0 and size < min_fvg_size_atr * atr_val:
            continue

        disp_bar = i - 1
        if disp_bar < 0 or h is None:
            continue

        disp_body = abs(c[disp_bar] - o[disp_bar])
        disp_atr = disp_body / atr_val if atr_val > 0 else 0

        if disp_atr < min_disp_atr or disp_atr > max_disp_atr:
            continue

        return {
            "fvg_bar": i,
            "fvg_top": top,
            "fvg_bottom": bottom,
            "fvg_size": size,
            "disp_atr": disp_atr,
            "bars_after_breakdown": i - breakdown_bar,
            "direction": direction,
        }

    return None


# ======================================================================
# Step 3: Full chain backtest
# ======================================================================

def run_chain_backtest(
    d: dict,
    breakdowns: list[dict],
    *,
    # FVG search params
    max_wait_bars: int = 30,
    min_disp_atr: float = 0.3,
    max_disp_atr: float = 1.5,
    min_fvg_size_atr: float = 0.3,
    # Entry params
    max_fvg_age: int = 200,
    min_stop_pts: float = 5.0,
    stop_buffer_pct: float = 0.15,
    tighten_factor: float = 0.85,
    # Exit params
    trim_pct: float = 0.25,
    tp_mult: float = 0.35,
    nth_swing: int = 5,
    be_after_trim: bool = True,
    eod_close: bool = True,
) -> list[dict]:
    """Run backtest following the natural ICT chain.

    Flow per breakdown event:
      1. Breakdown at level -> start watching for FVG
      2. FVG found -> create zone, wait for limit fill
      3. Limit fills -> trade management (same as U2)
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
    swing_low_mask = d["swing_low_mask"]
    swing_low_price_at_mask = d.get("swing_low_price_at_mask")

    fvg_bull_mask = fvg_df["fvg_bull"].values
    fvg_bull_top = fvg_df["fvg_bull_top"].values
    fvg_bull_bottom = fvg_df["fvg_bull_bottom"].values

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

    # Build zones from breakdown -> FVG chain
    fvg_bear_mask = fvg_df["fvg_bear"].values
    fvg_bear_top = fvg_df["fvg_bear_top"].values
    fvg_bear_bottom = fvg_df["fvg_bear_bottom"].values

    @dataclass
    class ChainZone:
        direction: int  # 1=long, -1=short
        top: float
        bottom: float
        size: float
        birth_bar: int
        birth_atr: float
        breakdown_bar: int
        disp_atr: float
        bars_after_bd: int
        used: bool = False
        touch_count: int = 0

    # Pre-compute zones from all breakdowns
    zones_by_bar: dict[int, list[ChainZone]] = {}
    total_fvgs_found = 0

    for bd in breakdowns:
        bd_level_type = bd.get("level_type", "low")
        fvg = find_fvg_after_breakdown(
            fvg_bull_mask, fvg_bull_top, fvg_bull_bottom,
            fvg_bear_mask, fvg_bear_top, fvg_bear_bottom,
            atr_arr,
            bd["bar_idx"], bd_level_type, max_wait_bars,
            min_disp_atr, max_disp_atr, min_fvg_size_atr,
            h, l, c_arr, o,
        )
        if fvg is None:
            continue
        total_fvgs_found += 1
        bar = fvg["fvg_bar"]
        zone = ChainZone(
            direction=fvg["direction"],
            top=fvg["fvg_top"], bottom=fvg["fvg_bottom"], size=fvg["fvg_size"],
            birth_bar=bar, birth_atr=atr_arr[bar] if not np.isnan(atr_arr[bar]) else 30.0,
            breakdown_bar=bd["bar_idx"], disp_atr=fvg["disp_atr"],
            bars_after_bd=fvg["bars_after_breakdown"],
        )
        if bar not in zones_by_bar:
            zones_by_bar[bar] = []
        zones_by_bar[bar].append(zone)

    # Now run the standard bar-by-bar engine with these zones
    active_zones: list[ChainZone] = []
    trades: list[dict] = []
    in_position = False
    pos_direction = pos_entry_idx = 0
    pos_entry_price = pos_stop = pos_tp1 = 0.0
    pos_contracts = pos_remaining_contracts = 0
    pos_trimmed = False
    pos_be_stop = pos_trail_stop = 0.0
    pos_grade = ""
    pos_trim_pct = trim_pct
    pos_zone: ChainZone | None = None
    current_date = None
    daily_pnl_r = 0.0
    consecutive_losses = 0
    day_stopped = False
    irl_high_arr = d["irl_high_arr"]

    for i in range(n):
        bar_date = dates[i]
        if bar_date != current_date:
            current_date = bar_date
            daily_pnl_r = 0.0
            consecutive_losses = 0
            day_stopped = False

        # Register new zones born on this bar
        if i in zones_by_bar:
            for z in zones_by_bar[i]:
                active_zones.append(z)

        # Invalidate / expire zones
        surviving = []
        for zone in active_zones:
            if zone.used:
                continue
            if (i - zone.birth_bar) > max_fvg_age:
                continue
            # Invalidation: price closes through the zone
            if zone.direction == 1 and c_arr[i] < zone.bottom:
                continue
            if zone.direction == -1 and c_arr[i] > zone.top:
                continue
            # Track touches
            if l[i] <= zone.top and h[i] >= zone.bottom:
                zone.touch_count += 1
            surviving.append(zone)
        if len(surviving) > 30:
            surviving = surviving[-30:]
        active_zones = surviving

        # EXIT MANAGEMENT
        if in_position:
            exited = False
            exit_reason = ""
            exit_price = 0.0
            exit_contracts = pos_remaining_contracts

            if pos_direction == 1:
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

            else:  # Short exit
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

            # EOD close
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
                disp = (c_arr[i] - pos_entry_price) * pos_direction
                cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                if avg_wick > 0.65 and favorable < 0.5 and disp < cur_atr * 0.3 and bars_in_trade >= 3:
                    exit_price = o[i+1] if i+1 < n else c_arr[i]
                    exit_reason = "early_cut_pa"
                    exited = True

            if exited:
                pnl_pts = (exit_price - pos_entry_price) * pos_direction
                if pos_trimmed and exit_reason != "tp1":
                    trim_c = pos_contracts - exit_contracts
                    trim_pnl = (pos_tp1 - pos_entry_price) * point_value * trim_c
                    total_pnl = trim_pnl + pnl_pts * point_value * exit_contracts
                    total_pnl -= commission_per_side * 2 * pos_contracts
                else:
                    total_pnl = pnl_pts * point_value * exit_contracts
                    total_pnl -= commission_per_side * 2 * exit_contracts
                stop_dist_exit = abs(pos_entry_price - pos_stop)
                total_risk = stop_dist_exit * point_value * pos_contracts
                r_mult = total_pnl / total_risk if total_risk > 0 else 0.0

                trades.append({
                    "entry_time": nq.index[pos_entry_idx],
                    "exit_time": nq.index[min(i + (1 if exit_reason == "early_cut_pa" else 0), n-1)],
                    "r": r_mult, "reason": exit_reason, "dir": pos_direction,
                    "type": "chain_entry", "trimmed": pos_trimmed, "grade": pos_grade,
                    "entry_price": pos_entry_price, "exit_price": exit_price,
                    "stop_price": pos_stop, "tp1_price": pos_tp1,
                    "pnl_dollars": total_pnl, "stop_dist_pts": stop_dist_exit,
                    "disp_atr": pos_zone.disp_atr if pos_zone else 0,
                    "bars_after_bd": pos_zone.bars_after_bd if pos_zone else 0,
                    "touch_count": pos_zone.touch_count if pos_zone else 0,
                })

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

        # ENTRY
        if in_position or day_stopped:
            continue
        if news_blackout_arr is not None and news_blackout_arr[i]:
            continue

        et_frac = et_frac_arr[i]
        if 9.5 <= et_frac <= 10.0:
            continue
        if not (10.0 <= et_frac < 16.0):
            continue

        # Check active zones for fills (both directions)
        best_fill = None
        best_zone = None
        best_same_bar_stop = False

        for zone in active_zones:
            if zone.used or zone.birth_bar >= i:
                continue

            if zone.direction == 1:  # Long
                entry_p = zone.top
                stop_p = zone.bottom - zone.size * stop_buffer_pct
            else:  # Short
                entry_p = zone.bottom
                stop_p = zone.top + zone.size * stop_buffer_pct

            stop_dist = abs(entry_p - stop_p)

            if tighten_factor < 1.0:
                if zone.direction == 1:
                    stop_p = entry_p - stop_dist * tighten_factor
                else:
                    stop_p = entry_p + stop_dist * tighten_factor
                stop_dist = abs(entry_p - stop_p)

            if stop_dist < min_stop_pts or stop_dist < 1.0:
                continue

            # Fill check
            is_same_bar_stop = False
            if zone.direction == 1:
                if l[i] > entry_p:
                    continue
                if l[i] <= stop_p:
                    is_same_bar_stop = True
            else:
                if h[i] < entry_p:
                    continue
                if h[i] >= stop_p:
                    is_same_bar_stop = True

            # Bias alignment: skip if bias opposes direction
            if zone.direction == 1 and bias_dir_arr[i] < 0:
                continue
            if zone.direction == -1 and bias_dir_arr[i] > 0:
                continue

            fill_quality = -abs(c_arr[i] - entry_p)
            if best_fill is None or fill_quality > best_fill:
                best_fill = fill_quality
                best_zone = (zone, entry_p, stop_p, stop_dist)
                best_same_bar_stop = is_same_bar_stop

        if best_zone is None:
            continue

        zone, entry_p, stop_p, stop_dist = best_zone
        zone.used = True
        cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0

        # Grade + sizing
        ba = 1.0 if bias_dir_arr[i] > 0 else 0.0
        grade = _compute_grade(ba, regime_arr[i])
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
                zone.used = False
                continue

        contracts = max(1, int(r_amount / (stop_dist * point_value))) if stop_dist > 0 else 0
        if contracts <= 0:
            zone.used = False
            continue

        direction = zone.direction

        # Same-bar stop
        if best_same_bar_stop:
            exit_price_sbs = (stop_p - slippage_points) if direction == 1 else (stop_p + slippage_points)
            pnl_sbs = (exit_price_sbs - entry_p) if direction == 1 else (entry_p - exit_price_sbs)
            pnl_sbs = pnl_sbs * point_value * contracts - commission_per_side * 2 * contracts
            total_risk_sbs = stop_dist * point_value * contracts
            r_sbs = pnl_sbs / total_risk_sbs if total_risk_sbs > 0 else 0.0
            trades.append({
                "entry_time": nq.index[i], "exit_time": nq.index[i],
                "r": r_sbs, "reason": "same_bar_stop", "dir": direction,
                "type": "chain_entry", "trimmed": False, "grade": grade,
                "entry_price": entry_p, "exit_price": exit_price_sbs,
                "stop_price": stop_p, "tp1_price": 0.0,
                "pnl_dollars": pnl_sbs, "stop_dist_pts": stop_dist,
                "disp_atr": zone.disp_atr, "bars_after_bd": zone.bars_after_bd,
                "touch_count": zone.touch_count,
            })
            daily_pnl_r += r_sbs
            consecutive_losses += 1
            if consecutive_losses >= max_consec_losses:
                day_stopped = True
            if daily_pnl_r <= -daily_max_loss_r:
                day_stopped = True
            continue

        # TP
        if direction == 1:
            irl_target = irl_high_arr[i]
            if np.isnan(irl_target) or irl_target <= entry_p:
                irl_target = entry_p + stop_dist * 2.0
            tp1_price = entry_p + (irl_target - entry_p) * tp_mult
        else:
            # Short: use swing low as target, or fixed RR
            irl_target = d["irl_low_arr"][i]
            if np.isnan(irl_target) or irl_target >= entry_p:
                irl_target = entry_p - stop_dist * 2.0
            tp1_price = entry_p - abs(entry_p - irl_target) * tp_mult

        # Enter
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
        pos_zone = zone
        pos_trim_pct = trim_pct if direction == 1 else 1.0  # shorts: full exit at TP

    return trades, {"total_breakdowns": len(breakdowns), "fvgs_found": total_fvgs_found,
                    "zones_created": sum(len(v) for v in zones_by_bar.values())}


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 120)
    print("NATURAL ICT CHAIN ENTRY -- Breakdown > FVG > Limit Fill")
    print("=" * 120)

    d = load_all()
    nq = d["nq"]
    n = d["n"]
    h, l, c, o = d["h"], d["l"], d["c"], d["o"]
    atr = d["atr_arr"]
    et_frac_arr = d["et_frac_arr"]

    # Compute levels
    print("\n[COMPUTE] Levels...")
    session_cache = pd.read_parquet(PROJECT / "data" / "cache_session_levels_10yr_v2.parquet")
    pdhl = compute_pdhl(nq)
    htf_1h = compute_htf_swings(nq, left=12, right=3)

    # Detect breakdowns at each level
    print("[DETECT] Breakdowns...")
    level_sources = {
        # Low levels -> bull FVG after breakdown
        "pdl": (pdhl["pdl"].values, "low"),
        "overnight_low": (session_cache["overnight_low"].values, "low"),
        "asia_low": (session_cache["asia_low"].values, "low"),
        "london_low": (session_cache["london_low"].values, "low"),
        "htf_1h_low": (htf_1h["htf_swing_low_price"].values, "low"),
        # High levels -> bear FVG after breakdown
        "pdh": (pdhl["pdh"].values, "high"),
        "overnight_high": (session_cache["overnight_high"].values, "high"),
        "asia_high": (session_cache["asia_high"].values, "high"),
        "london_high": (session_cache["london_high"].values, "high"),
        "htf_1h_high": (htf_1h["htf_swing_high_price"].values, "high"),
    }

    all_breakdowns = {}
    for lname, (arr, ltype) in level_sources.items():
        bds = detect_breakdowns(h, l, c, arr, ltype, min_depth_pts=1.0)
        for bd in bds:
            bd["level_type"] = ltype
        all_breakdowns[lname] = bds
        print(f"  {lname:20s}: {len(bds):6d} breakdowns")

    # Combine all breakdowns (deduplicate by bar proximity)
    combined = []
    for lname, bds in all_breakdowns.items():
        for bd in bds:
            bd["level_name"] = lname
            combined.append(bd)
    combined.sort(key=lambda x: x["bar_idx"])

    # Deduplicate: if multiple breakdowns within 3 bars, keep first
    deduped = []
    last_bar = -10
    for bd in combined:
        if bd["bar_idx"] - last_bar >= 3:
            deduped.append(bd)
            last_bar = bd["bar_idx"]
    print(f"\n  Combined (deduped): {len(deduped)} breakdown events")

    # ================================================================
    # Run chain backtest with different configs
    # ================================================================
    print(f"\n{'='*120}")
    print("CHAIN BACKTEST RESULTS")
    print(f"{'='*120}")

    # Separate low-type and high-type breakdowns
    low_levels = {k: v for k, v in all_breakdowns.items() if k.endswith("_low") or k == "pdl"}
    high_levels = {k: v for k, v in all_breakdowns.items() if k.endswith("_high") or k == "pdh"}

    all_low = []
    for bds in low_levels.values():
        all_low.extend(bds)
    all_low.sort(key=lambda x: x["bar_idx"])

    all_high = []
    for bds in high_levels.values():
        all_high.extend(bds)
    all_high.sort(key=lambda x: x["bar_idx"])

    all_combined = all_low + all_high
    all_combined.sort(key=lambda x: x["bar_idx"])

    configs = [
        # (label, breakdowns, max_wait, min_disp, max_disp)
        # --- Longs (low-level breakdown -> bull FVG) ---
        ("PDL only (longs)", all_breakdowns["pdl"], 30, 0.3, 1.5),
        ("PDL (any disp)", all_breakdowns["pdl"], 30, 0.0, 99.0),
        ("PDL (wide wait=60)", all_breakdowns["pdl"], 60, 0.3, 1.5),
        ("Overnight low (longs)", all_breakdowns["overnight_low"], 30, 0.3, 1.5),
        ("Asia low (longs)", all_breakdowns["asia_low"], 30, 0.3, 1.5),
        ("London low (longs)", all_breakdowns["london_low"], 30, 0.3, 1.5),
        ("HTF 1H low (longs)", all_breakdowns["htf_1h_low"], 30, 0.3, 1.5),
        ("ALL lows combined", all_low, 30, 0.3, 1.5),
        # --- Shorts (high-level breakdown -> bear FVG) ---
        ("PDH only (shorts)", all_breakdowns["pdh"], 30, 0.3, 1.5),
        ("PDH (any disp)", all_breakdowns["pdh"], 30, 0.0, 99.0),
        ("Overnight high (shorts)", all_breakdowns["overnight_high"], 30, 0.3, 1.5),
        ("Asia high (shorts)", all_breakdowns["asia_high"], 30, 0.3, 1.5),
        ("ALL highs combined", all_high, 30, 0.3, 1.5),
        # --- Both directions ---
        ("ALL levels both dirs", all_combined, 30, 0.3, 1.5),
        ("PDL + PDH", all_breakdowns["pdl"] + all_breakdowns["pdh"], 30, 0.3, 1.5),
    ]

    best_trades = None
    best_pf = 0

    for label, bds, max_wait, min_disp, max_disp in configs:
        trades, stats = run_chain_backtest(d, bds,
            max_wait_bars=max_wait, min_disp_atr=min_disp, max_disp_atr=max_disp)
        if len(trades) < 10:
            print(f"  {label:40s} | {len(trades):5d}t | too few | bd={stats['total_breakdowns']} fvg={stats['fvgs_found']}")
            continue
        m = compute_metrics(trades)
        tpd = m["trades"] / (252 * 10.5)
        print(f"  {label:40s} | {m['trades']:5d}t | R={m['R']:+8.1f} | PF={m['PF']:5.2f} | "
              f"PPDD={m['PPDD']:6.2f} | WR={m['WR']:5.1f}% | DD={m['MaxDD']:5.1f}R | {tpd:.2f}/d | "
              f"bd={stats['total_breakdowns']} fvg={stats['fvgs_found']}")
        if m["PF"] > best_pf and m["trades"] >= 100:
            best_pf = m["PF"]
            best_trades = trades
            best_label = label

    # ================================================================
    # Walk-forward for best config
    # ================================================================
    if best_trades and len(best_trades) >= 50:
        print(f"\n{'='*120}")
        print(f"WALK-FORWARD: {best_label}")
        print(f"{'='*120}")

        bt_df = pd.DataFrame(best_trades)
        bt_df["year"] = pd.to_datetime(bt_df["entry_time"]).dt.year
        neg = 0
        for year, grp in bt_df.groupby("year"):
            r = grp["r"].values
            total_r = r.sum()
            wins = r[r > 0].sum()
            losses = abs(r[r < 0].sum())
            pf = wins / losses if losses > 0 else 999
            flag = " NEG" if total_r < 0 else ""
            if total_r < 0:
                neg += 1
            print(f"  {year}: {len(grp):3d}t  R={total_r:+7.1f}  PF={pf:5.2f}{flag}")
        print(f"  Negative years: {neg}")

        # Exit reasons
        print(f"\n  Exit reasons:")
        for reason, cnt in bt_df["reason"].value_counts().items():
            sub_r = bt_df[bt_df["reason"] == reason]["r"].sum()
            print(f"    {reason:20s}: {cnt:4d}t  R={sub_r:+.1f}")

    # ================================================================
    # Long/Short split for best config
    # ================================================================
    if best_trades:
        bt_df = pd.DataFrame(best_trades)
        longs_only = bt_df[bt_df["dir"] == 1].to_dict("records")
        shorts_only = bt_df[bt_df["dir"] == -1].to_dict("records")
        if len(longs_only) > 0:
            m_l = compute_metrics(longs_only)
            pr(f"  {best_label} LONGS", m_l)
        if len(shorts_only) > 0:
            m_s = compute_metrics(shorts_only)
            pr(f"  {best_label} SHORTS", m_s)

    # ================================================================
    # Compare to U2-v2 baseline
    # ================================================================
    print(f"\n{'='*120}")
    print("COMPARISON TO U2-v2 BASELINE")
    print(f"{'='*120}")
    from experiments.u2_clean import run_u2_backtest
    u2_trades, _ = run_u2_backtest(d,
        stop_strategy="A2", fvg_size_mult=0.3, max_fvg_age=200,
        min_stop_pts=5.0, tighten_factor=0.85, tp_mult=0.35, nth_swing=5,
    )
    u2_longs = [t for t in u2_trades if t["dir"] == 1]
    m_u2 = compute_metrics(u2_longs)
    m_best = compute_metrics(best_trades) if best_trades else {"trades": 0}

    print(f"  {'':40s} | {'Trades':>7s} | {'R':>8s} | {'PF':>6s} | {'PPDD':>7s} | {'MaxDD':>6s} | {'t/day':>6s}")
    pr("U2-v2 baseline (longs only)", m_u2)
    if best_trades:
        pr(f"Chain: {best_label} (all dirs)", m_best)


if __name__ == "__main__":
    main()
