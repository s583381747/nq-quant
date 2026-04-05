"""
experiments/unified_engine_1m.py -- Hybrid Engine: 5m zones, 1m execution
=========================================================================

Zone detection (FVG, breakdown, NOT-MSS, trend) stays on 5m bars.
Entry/exit execution runs on 1m bars for proper intrabar resolution.

This eliminates the trim-bar BE artifact found in the 5m engine where
58% of runners survived only because 5m bars couldn't resolve the
TP1-then-BE sequence within a single bar.
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
CONFIG = PROJECT / "config" / "params.yaml"
sys.path.insert(0, str(PROJECT))

from experiments.chain_engine import (
    load_all, detect_breakdowns, find_fvg_not_mss,
    _find_nth_swing, _compute_grade, compute_metrics, pr,
)
from experiments.unified_engine import UnifiedZone


@dataclass
class Position:
    tier: int
    direction: int
    entry_bar_1m: int
    entry_bar_5m: int
    entry_price: float
    stop_price: float
    tp1_price: float
    contracts: int
    remaining: int
    trimmed: bool = False
    be_stop: float = 0.0
    trail_stop: float = 0.0
    grade: str = ""
    sweep_atr: float = 0.0
    daily_weight: float = 1.0


def build_5m_to_1m_map(nq5: pd.DataFrame, nq1: pd.DataFrame) -> np.ndarray:
    """For each 1m bar, find the corresponding 5m bar index.

    Returns array of length n_1m where map[i_1m] = i_5m.
    Uses searchsorted: find the 5m bar whose timestamp is <= the 1m timestamp.
    """
    ts5 = nq5.index.values  # numpy datetime64
    ts1 = nq1.index.values
    # For each 1m timestamp, find the index of the 5m bar it belongs to
    # searchsorted('right') - 1 gives the last 5m bar <= the 1m timestamp
    idx = np.searchsorted(ts5, ts1, side='right') - 1
    idx = np.clip(idx, 0, len(ts5) - 1)
    return idx


def reindex_to_1m(arr_5m: np.ndarray, map_1m_to_5m: np.ndarray) -> np.ndarray:
    """Forward-fill a 5m array to 1m using the index mapping."""
    return arr_5m[map_1m_to_5m]


def run_hybrid_1m(
    d5: dict,
    nq1: pd.DataFrame,
    *,
    breakdown_sources=None,
    max_wait_bars: int = 30,
    min_fvg_size_atr: float = 0.3,
    max_fvg_age: int = 200,
    stop_buffer_pct: float = 0.15,
    tighten_factor: float = 0.85,
    min_stop_atr_mult: float = 0.15,
    trim_pct: float = 0.25,
    fixed_tp_r: float = 1.0,
    nth_swing: int = 5,
    eod_close: bool = True,
    big_sweep_threshold: float = 1.3,
    big_sweep_mult: float = 1.5,
    am_short_mult: float = 0.5,
    trend_r_mult: float = 0.5,
    max_positions: int = 2,
    # Exit variant flags
    use_be: bool = True,
    be_offset_r: float = 0.0,  # 0 = BE at entry, 0.5 = entry - 0.5R
) -> tuple[list[dict], dict]:

    # ---- 5m data for zone detection ----
    params = d5["params"]
    nq5 = d5["nq"]
    n5 = d5["n"]
    o5, h5, l5, c5 = d5["o"], d5["h"], d5["l"], d5["c"]
    atr5 = d5["atr_arr"]
    bias5 = d5["bias_dir_arr"]
    regime5 = d5["regime_arr"]
    news5 = d5["news_blackout_arr"]
    fvg_df = d5["fvg_df"]
    on_hi5, on_lo5 = d5["on_hi"], d5["on_lo"]
    dates5, dow5, et_frac5 = d5["dates"], d5["dow_arr"], d5["et_frac_arr"]
    swing_high_mask_5m = d5["swing_high_mask"]
    swing_low_mask_5m = d5["swing_low_mask"]
    swing_high_price_5m = d5["swing_high_price_at_mask"]
    swing_low_price_5m = d5["swing_low_price_at_mask"]

    pos_params = params["position"]
    risk_params = params["risk"]
    bt_params = params["backtest"]
    grading_params = params["grading"]
    session_regime = params.get("session_regime", {})

    normal_r = pos_params["normal_r"]
    reduced_r = pos_params["reduced_r"]
    point_value = pos_params["point_value"]
    daily_max_loss_r = risk_params["daily_max_loss_r"]
    max_consec = risk_params["max_consecutive_losses"]
    comm = bt_params["commission_per_side_micro"]
    slip = bt_params["slippage_normal_ticks"] * 0.25
    a_mult = grading_params["a_plus_size_mult"]
    b_mult = grading_params["b_plus_size_mult"]

    # ---- 1m data for execution ----
    n1 = len(nq1)
    o1 = nq1["open"].values
    h1 = nq1["high"].values
    l1 = nq1["low"].values
    c1 = nq1["close"].values

    # Build 1m-to-5m mapping
    map_1m_5m = build_5m_to_1m_map(nq5, nq1)

    # Reindex 5m arrays to 1m
    bias1 = reindex_to_1m(bias5, map_1m_5m)
    regime1 = reindex_to_1m(regime5, map_1m_5m)
    on_hi1 = reindex_to_1m(on_hi5, map_1m_5m)
    on_lo1 = reindex_to_1m(on_lo5, map_1m_5m)
    atr1 = reindex_to_1m(atr5, map_1m_5m)  # 5m ATR, ffilled to 1m
    dow1 = reindex_to_1m(dow5, map_1m_5m)
    news1 = reindex_to_1m(news5, map_1m_5m) if news5 is not None else None

    # Compute 1m time arrays directly
    et1 = nq1.index.tz_convert("US/Eastern")
    et_frac1 = np.array([et1[j].hour + et1[j].minute / 60.0 for j in range(n1)])
    dates1 = np.array([
        (et1[j] + pd.Timedelta(days=1)).date() if et1[j].hour >= 18 else et1[j].date()
        for j in range(n1)
    ])

    # ================================================================
    # PHASE 1: Zone detection on 5m (IDENTICAL to unified_engine_v2)
    # ================================================================
    fvg_bm = fvg_df["fvg_bull"].values
    fvg_bt = fvg_df["fvg_bull_top"].values
    fvg_bb = fvg_df["fvg_bull_bottom"].values
    fvg_em = fvg_df["fvg_bear"].values
    fvg_et = fvg_df["fvg_bear_top"].values
    fvg_eb = fvg_df["fvg_bear_bottom"].values

    if breakdown_sources is None:
        breakdown_sources = [(on_lo5, "low"), (on_hi5, "high")]

    all_bds_raw = []
    for level_arr, level_type in breakdown_sources:
        bds = detect_breakdowns(h5, l5, c5, level_arr, level_type,
                                min_depth_pts=params.get("chain", {}).get("min_depth_pts", 1.0))
        all_bds_raw.extend(bds)
    all_bds_raw.sort(key=lambda x: x["bar_idx"])
    all_bds = []
    last_bar = -10
    for bd in all_bds_raw:
        if bd["bar_idx"] - last_bar >= 3:
            all_bds.append(bd)
            last_bar = bd["bar_idx"]

    chain_zones_by_5m = {}
    breakdown_territory = set()
    n_chain = 0
    for bd in all_bds:
        bd_idx = bd["bar_idx"]
        for b in range(bd_idx, min(bd_idx + max_wait_bars + 1, n5)):
            breakdown_territory.add(b)
        zone = find_fvg_not_mss(
            fvg_bm, fvg_bt, fvg_bb, fvg_em, fvg_et, fvg_eb,
            h5, l5, c5, o5, atr5, swing_high_mask_5m, swing_low_mask_5m,
            swing_high_price_5m, swing_low_price_5m,
            bd_idx, bd["level_type"], max_wait_bars, min_fvg_size_atr)
        if zone is None:
            continue
        bd_atr = atr5[bd_idx] if not np.isnan(atr5[bd_idx]) else 30.0
        sweep_range = h5[bd_idx] - l5[bd_idx]
        uz = UnifiedZone(zone.direction, zone.top, zone.bottom, zone.size,
                          zone.birth_bar, zone.birth_atr, 1,
                          sweep_range / bd_atr if bd_atr > 0 else 0)
        chain_zones_by_5m.setdefault(uz.birth_bar, []).append(uz)
        n_chain += 1

    trend_zones_by_5m = {}
    n_trend = 0
    for i in range(n5):
        for is_bull in [True, False]:
            mask = fvg_bm if is_bull else fvg_em
            if not mask[i] or i in breakdown_territory:
                continue
            direction = 1 if is_bull else -1
            top = fvg_bt[i] if is_bull else fvg_et[i]
            bottom = fvg_bb[i] if is_bull else fvg_eb[i]
            size = top - bottom
            atr_val = atr5[i] if not np.isnan(atr5[i]) else 30.0
            if atr_val > 0 and size < min_fvg_size_atr * atr_val:
                continue
            on_h, on_l = on_hi5[i], on_lo5[i]
            on_range = on_h - on_l if not (np.isnan(on_h) or np.isnan(on_l)) else 0
            if on_range <= 0:
                continue
            mid = (top + bottom) / 2
            on_pos = (mid - on_l) / on_range
            if direction == 1 and on_pos >= 0.5:
                continue
            if direction == -1 and on_pos < 0.5:
                continue
            bias = bias5[i]
            if direction == 1 and bias < 0:
                continue
            if direction == -1 and bias > 0:
                continue
            uz = UnifiedZone(direction, top, bottom, size, i, atr_val, 2)
            trend_zones_by_5m.setdefault(i, []).append(uz)
            n_trend += 1

    # ================================================================
    # PHASE 2: Execution loop on 1m
    # ================================================================
    active_zones: list[UnifiedZone] = []
    positions: list[Position] = []
    trades: list[dict] = []
    cur_date = None
    day_pnl = 0.0
    consec_loss = 0
    day_stopped = False
    last_activated_5m = -1  # Track which 5m bars we've activated zones from

    def exit_position(pos: Position, i_1m: int, ex_price: float, ex_reason: str, ex_contracts: int):
        nonlocal day_pnl, consec_loss, day_stopped

        pnl_pts = (ex_price - pos.entry_price) * pos.direction
        if pos.trimmed and ex_reason != "tp1":
            trim_c = pos.contracts - ex_contracts
            trim_pnl = (pos.tp1_price - pos.entry_price) * pos.direction * point_value * trim_c
            total_pnl = trim_pnl + pnl_pts * point_value * ex_contracts
            total_pnl -= comm * 2 * pos.contracts
        else:
            total_pnl = pnl_pts * point_value * ex_contracts
            total_pnl -= comm * 2 * ex_contracts
        sd = abs(pos.entry_price - pos.stop_price)
        r = total_pnl / normal_r if normal_r > 0 else 0.0

        trades.append({
            "entry_time": nq1.index[pos.entry_bar_1m],
            "exit_time": nq1.index[min(i_1m, n1 - 1)],
            "r": r, "reason": ex_reason, "dir": pos.direction,
            "tier": pos.tier, "trimmed": pos.trimmed, "grade": pos.grade,
            "entry_price": pos.entry_price, "exit_price": ex_price,
            "stop_price": pos.stop_price, "tp1_price": pos.tp1_price,
            "pnl_dollars": total_pnl, "stop_dist_pts": sd,
            "contracts": pos.contracts,
        })

        day_pnl += r
        if ex_reason == "be_sweep" and pos.trimmed:
            pass
        elif ex_reason == "eod_close":
            pass
        elif r < 0:
            consec_loss += 1
        else:
            consec_loss = 0
        if consec_loss >= max_consec:
            day_stopped = True
        if day_pnl <= -daily_max_loss_r:
            day_stopped = True

    for i_1m in range(n1):
        i_5m = map_1m_5m[i_1m]

        # ---- Day boundary ----
        if dates1[i_1m] != cur_date:
            for pos in positions:
                prev = max(0, i_1m - 1)
                ex_price = c1[prev] - 0.25 if pos.direction == 1 else c1[prev] + 0.25
                exit_position(pos, prev, ex_price, "eod_close", pos.remaining)
            positions.clear()
            cur_date = dates1[i_1m]
            day_pnl = 0.0
            consec_loss = 0
            day_stopped = False

        # ---- Activate zones from newly reached 5m bars ----
        if i_5m > last_activated_5m:
            for bar_5m in range(last_activated_5m + 1, i_5m + 1):
                if bar_5m in chain_zones_by_5m:
                    active_zones.extend(chain_zones_by_5m[bar_5m])
                if bar_5m in trend_zones_by_5m:
                    active_zones.extend(trend_zones_by_5m[bar_5m])
            last_activated_5m = i_5m

        # ---- Zone invalidation ----
        surviving = []
        for z in active_zones:
            if z.used or (i_5m - z.birth_bar) > max_fvg_age:
                continue
            if z.direction == 1 and c1[i_1m] < z.bottom:
                continue
            if z.direction == -1 and c1[i_1m] > z.top:
                continue
            surviving.append(z)
        active_zones = surviving[-50:] if len(surviving) > 50 else surviving

        # ---- EXIT all active positions ----
        exit_events = []
        to_remove = []
        for pi, pos in enumerate(positions):
            exited = False
            ex_reason = ""
            ex_price = 0.0
            ex_contracts = pos.remaining

            if pos.direction == 1:
                eff = pos.trail_stop if pos.trimmed and pos.trail_stop > 0 else pos.stop_price
                if pos.trimmed and pos.be_stop > 0:
                    eff = max(eff, pos.be_stop)
                if l1[i_1m] <= eff:
                    ex_price = eff - slip
                    ex_reason = "be_sweep" if (pos.trimmed and eff >= pos.be_stop) else "stop"
                    exited = True
                elif not pos.trimmed and h1[i_1m] >= pos.tp1_price:
                    tc = max(1, int(pos.contracts * trim_pct))
                    pos.remaining = pos.contracts - tc
                    pos.trimmed = True
                    if use_be:
                        pos.be_stop = pos.entry_price - be_offset_r * abs(pos.entry_price - pos.stop_price)
                    if pos.remaining > 0:
                        pos.trail_stop = _find_nth_swing(
                            swing_low_mask_5m, swing_low_price_5m, i_5m, nth_swing)
                        if np.isnan(pos.trail_stop) or pos.trail_stop <= 0:
                            pos.trail_stop = pos.be_stop if use_be else pos.stop_price
                    if pos.remaining <= 0:
                        ex_price = pos.tp1_price
                        ex_reason = "tp1"
                        ex_contracts = pos.contracts
                        exited = True
                if pos.trimmed and not exited:
                    nt = _find_nth_swing(swing_low_mask_5m, swing_low_price_5m, i_5m, nth_swing)
                    if not np.isnan(nt) and nt > pos.trail_stop:
                        pos.trail_stop = nt
            else:
                eff = pos.trail_stop if pos.trimmed and pos.trail_stop > 0 else pos.stop_price
                if pos.trimmed and pos.be_stop > 0:
                    if eff > pos.be_stop:
                        eff = pos.be_stop
                if h1[i_1m] >= eff:
                    ex_price = eff + slip
                    ex_reason = "be_sweep" if (pos.trimmed and eff <= pos.be_stop) else "stop"
                    exited = True
                elif not pos.trimmed and l1[i_1m] <= pos.tp1_price:
                    tc = max(1, int(pos.contracts * trim_pct))
                    pos.remaining = pos.contracts - tc
                    pos.trimmed = True
                    if use_be:
                        pos.be_stop = pos.entry_price + be_offset_r * abs(pos.entry_price - pos.stop_price)
                    if pos.remaining > 0:
                        nt_init = _find_nth_swing(
                            swing_high_mask_5m, swing_high_price_5m, i_5m, nth_swing)
                        if np.isnan(nt_init) or nt_init <= 0 or nt_init > pos.entry_price:
                            pos.trail_stop = pos.be_stop if use_be else pos.stop_price
                        else:
                            pos.trail_stop = nt_init
                    if pos.remaining <= 0:
                        ex_price = pos.tp1_price
                        ex_reason = "tp1"
                        ex_contracts = pos.contracts
                        exited = True
                if pos.trimmed and not exited:
                    nt = _find_nth_swing(swing_high_mask_5m, swing_high_price_5m, i_5m, nth_swing)
                    if not np.isnan(nt) and nt > 0 and nt < pos.trail_stop:
                        pos.trail_stop = nt

            if not exited and eod_close and et_frac1[i_1m] >= 15.917:
                ex_price = c1[i_1m] - 0.25 if pos.direction == 1 else c1[i_1m] + 0.25
                ex_reason = "eod_close"
                exited = True

            # PA early cut (adjusted: 10-20 1m bars = 2-4 5m bars)
            bars_in_1m = i_1m - pos.entry_bar_1m
            if not exited and not pos.trimmed and 10 <= bars_in_1m <= 20:
                pa_s = max(pos.entry_bar_1m, 0)
                pa_e = i_1m + 1
                pa_range = h1[pa_s:pa_e] - l1[pa_s:pa_e]
                pa_body = np.abs(c1[pa_s:pa_e] - o1[pa_s:pa_e])
                safe = np.where(pa_range > 0, pa_range, 1.0)
                avg_wick = float(np.mean(1.0 - pa_body / safe))
                pa_dirs = np.sign(c1[pa_s:pa_e] - o1[pa_s:pa_e])
                favorable = (pa_dirs == pos.direction).mean()
                disp = (c1[i_1m] - pos.entry_price) * pos.direction
                cur_atr = atr1[i_1m] if not np.isnan(atr1[i_1m]) else 30.0
                if avg_wick > 0.65 and favorable < 0.5 and disp < cur_atr * 0.3 and bars_in_1m >= 15:
                    ex_price = o1[i_1m + 1] if i_1m + 1 < n1 else c1[i_1m]
                    ex_reason = "early_cut_pa"
                    exited = True

            if exited:
                is_loss = ex_reason in ("stop", "same_bar_stop", "early_cut_pa")
                exit_events.append((pi, pos, ex_price, ex_reason, ex_contracts, is_loss))

        # Process losses first (Axiom 5)
        exit_events.sort(key=lambda x: (0 if x[5] else 1))
        for pi, pos, ex_price, ex_reason, ex_contracts, _ in exit_events:
            exit_position(pos, i_1m, ex_price, ex_reason, ex_contracts)
            to_remove.append(pi)

        for pi in reversed(sorted(set(to_remove))):
            positions.pop(pi)

        # ---- ENTRY ----
        if day_stopped:
            continue
        if news1 is not None and news1[i_1m]:
            continue
        ef = et_frac1[i_1m]
        if 9.5 <= ef <= 10.0 or not (10.0 <= ef < 16.0):
            continue

        occupied_tiers = {pos.tier for pos in positions}
        if len(positions) >= max_positions:
            continue

        for target_tier in [1, 2]:
            if day_stopped:
                break
            if target_tier in occupied_tiers:
                continue
            if len(positions) >= max_positions:
                break

            best_fill = None
            best_info = None
            best_sbs = False

            for z in active_zones:
                if z.used or z.birth_bar >= i_5m or z.tier != target_tier:
                    continue
                if z.direction == 1:
                    ep = z.top
                    sp = z.bottom - z.size * stop_buffer_pct
                else:
                    ep = z.bottom
                    sp = z.top + z.size * stop_buffer_pct
                sd = abs(ep - sp)
                if tighten_factor < 1.0:
                    sp = ep - sd * tighten_factor if z.direction == 1 else ep + sd * tighten_factor
                    sd = abs(ep - sp)
                min_stop = min_stop_atr_mult * (atr1[i_1m] if not np.isnan(atr1[i_1m]) else 30.0)
                if sd < min_stop or sd < 0.5:
                    continue

                sbs = False
                if z.direction == 1:
                    if l1[i_1m] > ep:
                        continue
                    if l1[i_1m] <= sp:
                        sbs = True
                else:
                    if h1[i_1m] < ep:
                        continue
                    if h1[i_1m] >= sp:
                        sbs = True

                if z.direction == 1 and bias1[i_1m] < 0:
                    continue
                if z.direction == -1 and bias1[i_1m] > 0:
                    continue

                if z.tier == 2:
                    on_h_fill = on_hi1[i_1m]
                    on_l_fill = on_lo1[i_1m]
                    on_rng = on_h_fill - on_l_fill if not (np.isnan(on_h_fill) or np.isnan(on_l_fill)) else 0
                    if on_rng > 0:
                        fill_pos = (ep - on_l_fill) / on_rng
                        if z.direction == 1 and fill_pos >= 0.5:
                            continue
                        if z.direction == -1 and fill_pos < 0.5:
                            continue

                fq = -abs(c1[i_1m] - ep)
                if best_fill is None or fq > best_fill:
                    best_fill = fq
                    best_info = (z, ep, sp, sd)
                    best_sbs = sbs

            if best_info is None:
                continue

            z, ep, sp, sd = best_info
            z.used = True
            direction = z.direction
            tier = z.tier

            ba = 1.0 if (direction == np.sign(bias1[i_1m]) and bias1[i_1m] != 0) else 0.0
            grade = _compute_grade(ba, regime1[i_1m])
            is_reduced = (dow1[i_1m] in (0, 4)) or (regime1[i_1m] < 1.0)
            base_r = reduced_r if is_reduced else normal_r
            if grade == "A+":
                r_amount = base_r * a_mult
            elif grade == "B+":
                r_amount = base_r * b_mult
            else:
                r_amount = base_r * 0.5

            if tier == 1:
                if z.sweep_range_atr >= big_sweep_threshold:
                    r_amount *= big_sweep_mult
                if direction == -1 and 10.0 <= ef < 12.0:
                    r_amount *= am_short_mult
            else:
                r_amount *= trend_r_mult

            contracts = max(1, int(r_amount / (sd * point_value))) if sd > 0 else 0
            if contracts <= 0:
                z.used = False
                continue

            if best_sbs:
                exp = (sp - slip) if direction == 1 else (sp + slip)
                pp = ((exp - ep) if direction == 1 else (ep - exp)) * point_value * contracts
                pp -= comm * 2 * contracts
                rr = pp / normal_r if normal_r > 0 else 0
                trades.append({
                    "entry_time": nq1.index[i_1m], "exit_time": nq1.index[i_1m],
                    "r": rr, "reason": "same_bar_stop", "dir": direction,
                    "tier": tier, "trimmed": False, "grade": grade,
                    "entry_price": ep, "exit_price": exp,
                    "stop_price": sp, "tp1_price": 0.0,
                    "pnl_dollars": pp, "stop_dist_pts": sd,
                    "contracts": contracts,
                })
                day_pnl += rr
                consec_loss += 1
                if consec_loss >= max_consec:
                    day_stopped = True
                if day_pnl <= -daily_max_loss_r:
                    day_stopped = True
                continue

            tp1 = ep + sd * fixed_tp_r if direction == 1 else ep - sd * fixed_tp_r

            positions.append(Position(
                tier=tier, direction=direction,
                entry_bar_1m=i_1m, entry_bar_5m=i_5m,
                entry_price=ep, stop_price=sp, tp1_price=tp1,
                contracts=contracts, remaining=contracts,
                grade=grade, sweep_atr=z.sweep_range_atr,
            ))

    # Force close remaining
    for pos in positions:
        last = n1 - 1
        ex_price = c1[last] - 0.25 if pos.direction == 1 else c1[last] + 0.25
        exit_position(pos, last, ex_price, "force_close", pos.remaining)

    stats = {"chain_zones": n_chain, "trend_zones": n_trend, "breakdowns": len(all_bds)}
    return trades, stats


def main():
    print("=" * 120)
    print("HYBRID ENGINE: 5m zones + 1m execution")
    print("=" * 120)

    t0 = _time.perf_counter()
    d5 = load_all()
    print(f"Loading 1m data...")
    nq1 = pd.read_parquet(DATA / "NQ_1min_10yr.parquet")
    print(f"1m loaded: {len(nq1):,} bars ({_time.perf_counter() - t0:.1f}s)")

    # --- Baseline: 5m engine ---
    from experiments.unified_engine_v2 import run_unified_v2
    trades_5m, _ = run_unified_v2(d5, trend_r_mult=0.5)

    # --- Ground truth: 1m hybrid ---
    print("\nRunning 1m hybrid engine...")
    t1 = _time.perf_counter()
    trades_1m, stats = run_hybrid_1m(d5, nq1, trend_r_mult=0.5)
    elapsed = _time.perf_counter() - t1
    print(f"1m engine done in {elapsed:.1f}s")

    df_1m = pd.DataFrame(trades_1m)
    t1_trades = [t for t in trades_1m if t["tier"] == 1]
    t2_trades = [t for t in trades_1m if t["tier"] == 2]

    print(f"\n{'=' * 120}")
    print("5m ENGINE (reference)")
    print(f"{'=' * 120}")
    pr("ALL 5m", compute_metrics(trades_5m))

    print(f"\n{'=' * 120}")
    print("1m HYBRID ENGINE (ground truth)")
    print(f"{'=' * 120}")
    pr("ALL 1m", compute_metrics(trades_1m))
    pr("  Chain", compute_metrics(t1_trades))
    pr("  Trend", compute_metrics(t2_trades))

    # Runner analysis
    runners = df_1m[df_1m["trimmed"] == True]
    non_runners = df_1m[df_1m["trimmed"] == False]
    print(f"\n  Runners: {len(runners)}t ({len(runners)/len(df_1m)*100:.1f}%)  R={runners['r'].sum():+.1f}")
    print(f"  Non-runners: {len(non_runners)}t  R={non_runners['r'].sum():+.1f}")

    # Exit reasons
    print(f"\n  Exit reasons:")
    for reason, grp in df_1m.groupby("reason"):
        print(f"    {reason:20s}: {len(grp):4d}t  R={grp['r'].sum():+.1f}  avgR={grp['r'].mean():+.2f}")

    # Sharpe
    df_1m["date"] = pd.to_datetime(df_1m["entry_time"]).dt.date
    daily = df_1m.groupby("date")["r"].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252)
    print(f"\n  Sharpe: {sharpe:.2f}")

    # Walk-forward
    print(f"\n  Walk-forward:")
    df_1m["year"] = pd.to_datetime(df_1m["entry_time"]).dt.year
    neg = 0
    for yr, grp in df_1m.groupby("year"):
        r = grp["r"].values
        total_r = r.sum()
        w, lo = r[r > 0].sum(), abs(r[r < 0].sum())
        pf = w / lo if lo > 0 else 999
        flag = " NEG" if total_r < 0 else ""
        if total_r < 0:
            neg += 1
        print(f"    {yr}: {len(grp):3d}t  R={total_r:+7.1f}  PF={pf:.2f}{flag}")
    print(f"    Negative years: {neg}")

    # --- Exit variants (if 1m baseline is bad) ---
    m = compute_metrics(trades_1m)
    if m["PF"] < 1.3:
        print(f"\n{'=' * 120}")
        print("1m PF < 1.3 — RUNNING EXIT VARIANTS")
        print(f"{'=' * 120}")

        variants = [
            ("No BE", dict(use_be=False)),
            ("BE offset 0.5R", dict(use_be=True, be_offset_r=0.5)),
            ("TP 2R", dict(fixed_tp_r=2.0)),
            ("TP 3R", dict(fixed_tp_r=3.0)),
            ("No trim (100% trail)", dict(trim_pct=0.0)),
            ("50% trim", dict(trim_pct=0.50)),
            ("No BE + TP 2R", dict(use_be=False, fixed_tp_r=2.0)),
            ("No BE + TP 3R", dict(use_be=False, fixed_tp_r=3.0)),
        ]

        for name, kwargs in variants:
            t_var, _ = run_hybrid_1m(d5, nq1, trend_r_mult=0.5, **kwargs)
            pr(f"  {name}", compute_metrics(t_var))


if __name__ == "__main__":
    main()
