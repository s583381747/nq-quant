"""
experiments/unified_engine_v2.py -- Unified Engine with Independent Tier Positions
==================================================================================

Fix: Allow one position per tier (max 2 simultaneous).
Chain and trend don't block each other.
Shared daily P&L for risk management only.
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

from experiments.chain_engine import (
    load_all, detect_breakdowns, find_fvg_not_mss,
    _find_nth_swing, _compute_grade, compute_metrics, pr,
)
from experiments.unified_engine import UnifiedZone


# ======================================================================
# Position state (one per active position)
# ======================================================================
@dataclass
class Position:
    tier: int
    direction: int
    entry_bar: int
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


# ======================================================================
# Engine
# ======================================================================
def run_unified_v2(
    d: dict,
    *,
    breakdown_sources=None,
    max_wait_bars: int = 30,
    min_fvg_size_atr: float = 0.3,
    max_fvg_age: int = 200,
    stop_buffer_pct: float = 0.15,
    tighten_factor: float = 0.85,
    min_stop_pts: float = 5.0,
    trim_pct: float = 0.25,
    fixed_tp_r: float = 1.0,
    nth_swing: int = 5,
    eod_close: bool = True,
    big_sweep_threshold: float = 1.3,
    big_sweep_mult: float = 1.5,
    am_short_mult: float = 0.5,
    trend_r_mult: float = 0.5,
    max_positions: int = 2,  # max simultaneous (1 per tier)
) -> tuple[list[dict], dict]:

    params = d["params"]
    nq, n = d["nq"], d["n"]
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    atr_arr = d["atr_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    fvg_df = d["fvg_df"]
    on_hi, on_lo = d["on_hi"], d["on_lo"]
    dates, dow_arr, et_frac_arr = d["dates"], d["dow_arr"], d["et_frac_arr"]
    swing_high_mask = d["swing_high_mask"]
    swing_low_mask = d["swing_low_mask"]
    swing_high_price_at_mask = d["swing_high_price_at_mask"]
    swing_low_price_at_mask = d["swing_low_price_at_mask"]

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

    fvg_bm = fvg_df["fvg_bull"].values
    fvg_bt = fvg_df["fvg_bull_top"].values
    fvg_bb = fvg_df["fvg_bull_bottom"].values
    fvg_em = fvg_df["fvg_bear"].values
    fvg_et = fvg_df["fvg_bear_top"].values
    fvg_eb = fvg_df["fvg_bear_bottom"].values
    sh_prices = swing_high_price_at_mask
    sl_prices = swing_low_price_at_mask

    # Pre-compute zones (same as unified_engine.py)
    if breakdown_sources is None:
        breakdown_sources = [(on_lo, "low"), (on_hi, "high")]

    all_bds_raw = []
    for level_arr, level_type in breakdown_sources:
        bds = detect_breakdowns(h, l, c, level_arr, level_type,
                                min_depth_pts=params.get("chain", {}).get("min_depth_pts", 1.0))
        all_bds_raw.extend(bds)
    all_bds_raw.sort(key=lambda x: x["bar_idx"])
    all_bds = []
    last_bar = -10
    for bd in all_bds_raw:
        if bd["bar_idx"] - last_bar >= 3:
            all_bds.append(bd)
            last_bar = bd["bar_idx"]

    chain_zones_by_bar = {}
    breakdown_territory = set()
    n_chain = 0
    for bd in all_bds:
        bd_idx = bd["bar_idx"]
        for b in range(bd_idx, min(bd_idx + max_wait_bars + 1, n)):
            breakdown_territory.add(b)
        zone = find_fvg_not_mss(
            fvg_bm, fvg_bt, fvg_bb, fvg_em, fvg_et, fvg_eb,
            h, l, c, o, atr_arr, swing_high_mask, swing_low_mask, sh_prices, sl_prices,
            bd_idx, bd["level_type"], max_wait_bars, min_fvg_size_atr)
        if zone is None:
            continue
        bd_atr = atr_arr[bd_idx] if not np.isnan(atr_arr[bd_idx]) else 30.0
        sweep_range = h[bd_idx] - l[bd_idx]
        uz = UnifiedZone(zone.direction, zone.top, zone.bottom, zone.size,
                          zone.birth_bar, zone.birth_atr, 1,
                          sweep_range / bd_atr if bd_atr > 0 else 0)
        chain_zones_by_bar.setdefault(uz.birth_bar, []).append(uz)
        n_chain += 1

    trend_zones_by_bar = {}
    n_trend = 0
    for i in range(n):
        for is_bull in [True, False]:
            mask = fvg_bm if is_bull else fvg_em
            if not mask[i] or i in breakdown_territory:
                continue
            direction = 1 if is_bull else -1
            top = fvg_bt[i] if is_bull else fvg_et[i]
            bottom = fvg_bb[i] if is_bull else fvg_eb[i]
            size = top - bottom
            atr_val = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
            if atr_val > 0 and size < min_fvg_size_atr * atr_val:
                continue
            on_h, on_l = on_hi[i], on_lo[i]
            on_range = on_h - on_l if not (np.isnan(on_h) or np.isnan(on_l)) else 0
            if on_range <= 0:
                continue
            mid = (top + bottom) / 2
            on_pos = (mid - on_l) / on_range
            if direction == 1 and on_pos >= 0.5:
                continue
            if direction == -1 and on_pos < 0.5:
                continue
            bias = bias_dir_arr[i]
            if direction == 1 and bias < 0:
                continue
            if direction == -1 and bias > 0:
                continue
            uz = UnifiedZone(direction, top, bottom, size, i, atr_val, 2)
            trend_zones_by_bar.setdefault(i, []).append(uz)
            n_trend += 1

    # ---- BAR-BY-BAR with multi-position ----
    active_zones: list[UnifiedZone] = []
    positions: list[Position] = []  # active positions (max 2)
    trades: list[dict] = []
    cur_date = None
    day_pnl = 0.0
    consec_loss = 0
    day_stopped = False

    def exit_position(pos: Position, i: int, ex_price: float, ex_reason: str, ex_contracts: int):
        """Record trade exit and update daily state."""
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
        risk = sd * point_value * pos.contracts
        r = total_pnl / risk if risk > 0 else 0.0

        trades.append({
            "entry_time": nq.index[pos.entry_bar],
            "exit_time": nq.index[min(i + (1 if ex_reason == "early_cut_pa" else 0), n-1)],
            "r": r, "reason": ex_reason, "dir": pos.direction,
            "tier": pos.tier, "trimmed": pos.trimmed, "grade": pos.grade,
            "entry_price": pos.entry_price, "exit_price": ex_price,
            "stop_price": pos.stop_price, "tp1_price": pos.tp1_price,
            "pnl_dollars": total_pnl, "stop_dist_pts": sd,
            "contracts": pos.contracts,
        })

        day_pnl += r * pos.daily_weight
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

    for i in range(n):
        if dates[i] != cur_date:
            # AUDIT FIX #3: Force-close any positions from previous day
            for pos in positions:
                prev_bar = max(0, i - 1)
                ex_price = c[prev_bar] - 0.25 if pos.direction == 1 else c[prev_bar] + 0.25
                exit_position(pos, prev_bar, ex_price, "eod_close", pos.remaining)
            positions.clear()
            cur_date = dates[i]
            day_pnl = 0.0
            consec_loss = 0
            day_stopped = False

        if i in chain_zones_by_bar:
            active_zones.extend(chain_zones_by_bar[i])
        if i in trend_zones_by_bar:
            active_zones.extend(trend_zones_by_bar[i])

        surviving = []
        for z in active_zones:
            if z.used or (i - z.birth_bar) > max_fvg_age:
                continue
            if z.direction == 1 and c[i] < z.bottom:
                continue
            if z.direction == -1 and c[i] > z.top:
                continue
            surviving.append(z)
        active_zones = surviving[-50:] if len(surviving) > 50 else surviving

        # ---- EXIT all active positions ----
        # AUDIT FIX #2: Process exits in worst-case order (losses first)
        # Collect all exit events, then sort so losses process before wins
        exit_events = []  # (pi, ex_price, ex_reason, ex_contracts, is_loss_estimate)
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
                if l[i] <= eff:
                    ex_price = eff - slip
                    ex_reason = "be_sweep" if (pos.trimmed and eff >= pos.entry_price) else "stop"
                    exited = True
                elif not pos.trimmed and h[i] >= pos.tp1_price:
                    tc = max(1, int(pos.contracts * trim_pct))
                    pos.remaining = pos.contracts - tc
                    pos.trimmed = True
                    pos.be_stop = pos.entry_price
                    if pos.remaining > 0:
                        pos.trail_stop = _find_nth_swing(swing_low_mask, swing_low_price_at_mask, i, nth_swing)
                        if np.isnan(pos.trail_stop) or pos.trail_stop <= 0:
                            pos.trail_stop = pos.be_stop
                    if pos.remaining <= 0:
                        ex_price = pos.tp1_price
                        ex_reason = "tp1"
                        ex_contracts = pos.contracts
                        exited = True
                if pos.trimmed and not exited:
                    nt = _find_nth_swing(swing_low_mask, swing_low_price_at_mask, i, nth_swing)
                    if not np.isnan(nt) and nt > pos.trail_stop:
                        pos.trail_stop = nt
            else:
                eff = pos.trail_stop if pos.trimmed and pos.trail_stop > 0 else pos.stop_price
                if pos.trimmed and pos.be_stop > 0:
                    if eff > pos.be_stop:
                        eff = pos.be_stop
                if h[i] >= eff:
                    ex_price = eff + slip
                    ex_reason = "be_sweep" if (pos.trimmed and eff <= pos.entry_price) else "stop"
                    exited = True
                elif not pos.trimmed and l[i] <= pos.tp1_price:
                    tc = max(1, int(pos.contracts * trim_pct))
                    pos.remaining = pos.contracts - tc
                    pos.trimmed = True
                    pos.be_stop = pos.entry_price
                    if pos.remaining > 0:
                        nt_init = _find_nth_swing(swing_high_mask, swing_high_price_at_mask, i, nth_swing)
                        if np.isnan(nt_init) or nt_init <= 0 or nt_init > pos.entry_price:
                            pos.trail_stop = pos.be_stop
                        else:
                            pos.trail_stop = nt_init
                    if pos.remaining <= 0:
                        ex_price = pos.tp1_price
                        ex_reason = "tp1"
                        ex_contracts = pos.contracts
                        exited = True
                if pos.trimmed and not exited:
                    nt = _find_nth_swing(swing_high_mask, swing_high_price_at_mask, i, nth_swing)
                    if not np.isnan(nt) and nt > 0 and nt < pos.trail_stop:
                        pos.trail_stop = nt

            if not exited and eod_close and et_frac_arr[i] >= 15.917:
                ex_price = c[i] - 0.25 if pos.direction == 1 else c[i] + 0.25
                ex_reason = "eod_close"
                exited = True

            # PA early cut
            bars_in = i - pos.entry_bar
            if not exited and not pos.trimmed and 2 <= bars_in <= 4:
                pa_s, pa_e = max(pos.entry_bar, 0), i + 1
                pa_range = h[pa_s:pa_e] - l[pa_s:pa_e]
                pa_body = np.abs(c[pa_s:pa_e] - o[pa_s:pa_e])
                safe = np.where(pa_range > 0, pa_range, 1.0)
                avg_wick = float(np.mean(1.0 - pa_body / safe))
                pa_dirs = np.sign(c[pa_s:pa_e] - o[pa_s:pa_e])
                favorable = (pa_dirs == pos.direction).mean()
                disp = (c[i] - pos.entry_price) * pos.direction
                cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                if avg_wick > 0.65 and favorable < 0.5 and disp < cur_atr * 0.3 and bars_in >= 3:
                    ex_price = o[i+1] if i+1 < n else c[i]
                    ex_reason = "early_cut_pa"
                    exited = True

            if exited:
                # Estimate if this is a loss for ordering
                is_loss = ex_reason in ("stop", "same_bar_stop", "early_cut_pa")
                exit_events.append((pi, pos, ex_price, ex_reason, ex_contracts, is_loss))

        # AUDIT FIX #2: Process losses first (worst case, Axiom 5)
        exit_events.sort(key=lambda x: (0 if x[5] else 1))
        for pi, pos, ex_price, ex_reason, ex_contracts, _ in exit_events:
            exit_position(pos, i, ex_price, ex_reason, ex_contracts)
            to_remove.append(pi)

        for pi in reversed(sorted(set(to_remove))):
            positions.pop(pi)

        # ---- ENTRY (one per tier, if slot available) ----
        if day_stopped:
            continue
        if news_blackout_arr is not None and news_blackout_arr[i]:
            continue
        ef = et_frac_arr[i]
        if 9.5 <= ef <= 10.0 or not (10.0 <= ef < 16.0):
            continue

        # Which tiers have open slots?
        occupied_tiers = {pos.tier for pos in positions}
        if len(positions) >= max_positions:
            continue

        # Find best fill for each available tier
        for target_tier in [1, 2]:
            # AUDIT FIX #1: Re-check day_stopped between tier entries
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
                if z.used or z.birth_bar >= i or z.tier != target_tier:
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
                if sd < min_stop_pts or sd < 1.0:
                    continue

                sbs = False
                if z.direction == 1:
                    if l[i] > ep: continue
                    if l[i] <= sp: sbs = True
                else:
                    if h[i] < ep: continue
                    if h[i] >= sp: sbs = True

                if z.direction == 1 and bias_dir_arr[i] < 0: continue
                if z.direction == -1 and bias_dir_arr[i] > 0: continue

                # Tier 2: re-check PD alignment at FILL time (not just zone creation)
                if z.tier == 2:
                    on_h_fill = on_hi[i]
                    on_l_fill = on_lo[i]
                    on_rng = on_h_fill - on_l_fill if not (np.isnan(on_h_fill) or np.isnan(on_l_fill)) else 0
                    if on_rng > 0:
                        fill_pos = (ep - on_l_fill) / on_rng
                        if z.direction == 1 and fill_pos >= 0.5:
                            continue  # long entry no longer in discount
                        if z.direction == -1 and fill_pos < 0.5:
                            continue  # short entry no longer in premium

                fq = -abs(c[i] - ep)
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

            ba = 1.0 if (direction == np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0) else 0.0
            grade = _compute_grade(ba, regime_arr[i])
            is_reduced = (dow_arr[i] in (0, 4)) or (regime_arr[i] < 1.0)
            base_r = reduced_r if is_reduced else normal_r
            if grade == "A+": r_amount = base_r * a_mult
            elif grade == "B+": r_amount = base_r * b_mult
            else: r_amount = base_r * 0.5

            if tier == 1:
                if z.sweep_range_atr >= big_sweep_threshold:
                    r_amount *= big_sweep_mult
                if direction == -1 and 10.0 <= ef < 12.0:
                    r_amount *= am_short_mult
                daily_weight = 1.0
            else:
                r_amount *= trend_r_mult
                daily_weight = trend_r_mult

            if session_regime.get("enabled", False):
                ls = session_regime.get("lunch_start", 12.0)
                le = session_regime.get("lunch_end", 13.0)
                if ls <= ef < le:
                    z.used = False
                    continue

            contracts = max(1, int(r_amount / (sd * point_value))) if sd > 0 else 0
            if contracts <= 0:
                z.used = False
                continue

            if best_sbs:
                exp = (sp - slip) if direction == 1 else (sp + slip)
                pp = ((exp - ep) if direction == 1 else (ep - exp)) * point_value * contracts
                pp -= comm * 2 * contracts
                rr = pp / (sd * point_value * contracts) if sd > 0 else 0
                trades.append({
                    "entry_time": nq.index[i], "exit_time": nq.index[i],
                    "r": rr, "reason": "same_bar_stop", "dir": direction,
                    "tier": tier, "trimmed": False, "grade": grade,
                    "entry_price": ep, "exit_price": exp,
                    "stop_price": sp, "tp1_price": 0.0,
                    "pnl_dollars": pp, "stop_dist_pts": sd,
                    "contracts": contracts,
                })
                day_pnl += rr * daily_weight
                consec_loss += 1
                if consec_loss >= max_consec: day_stopped = True
                if day_pnl <= -daily_max_loss_r: day_stopped = True
                continue

            tp1 = ep + sd * fixed_tp_r if direction == 1 else ep - sd * fixed_tp_r

            positions.append(Position(
                tier=tier, direction=direction, entry_bar=i,
                entry_price=ep, stop_price=sp, tp1_price=tp1,
                contracts=contracts, remaining=contracts,
                grade=grade, sweep_atr=z.sweep_range_atr,
                daily_weight=daily_weight,
            ))

    # Force close all remaining
    for pos in positions:
        last_i = n - 1
        ex_price = c[last_i] - 0.25 if pos.direction == 1 else c[last_i] + 0.25
        exit_position(pos, last_i, ex_price, "force_close", pos.remaining)

    stats = {"chain_zones": n_chain, "trend_zones": n_trend, "breakdowns": len(all_bds)}
    return trades, stats


def main():
    print("=" * 120)
    print("UNIFIED ENGINE V2 -- Independent Tier Positions")
    print("=" * 120)

    d = load_all()

    # V1 (shared position)
    from experiments.unified_engine import run_unified
    trades_v1, _ = run_unified(d)

    # V2 (independent positions)
    trades_v2, stats = run_unified_v2(d)

    df2 = pd.DataFrame(trades_v2)
    t1 = [t for t in trades_v2 if t["tier"] == 1]
    t2 = [t for t in trades_v2 if t["tier"] == 2]

    print(f"\n[STATS] Chain: {stats['chain_zones']}, Trend: {stats['trend_zones']}")

    print(f"\n{'='*120}")
    print("V2 RESULTS (independent positions)")
    print(f"{'='*120}")
    pr("ALL", compute_metrics(trades_v2))
    pr("  Tier 1 (chain)", compute_metrics(t1))
    pr("  Tier 2 (trend)", compute_metrics(t2))

    # Walk-forward
    print(f"\n  Walk-forward:")
    df2["year"] = pd.to_datetime(df2["entry_time"]).dt.year
    neg = 0
    for yr, grp in df2.groupby("year"):
        r = grp["r"].values
        total_r = r.sum()
        w, lo = r[r > 0].sum(), abs(r[r < 0].sum())
        pf = w / lo if lo > 0 else 999
        nc = (grp["tier"] == 1).sum()
        nt = (grp["tier"] == 2).sum()
        flag = " NEG" if total_r < 0 else ""
        if total_r < 0: neg += 1
        print(f"    {yr}: {len(grp):3d}t (C:{nc:3d} T:{nt:3d})  R={total_r:+7.1f}  PF={pf:5.2f}{flag}")
    print(f"    Negative years: {neg}")

    # Comparison
    print(f"\n{'='*120}")
    print("COMPARISON")
    print(f"{'='*120}")
    m1 = compute_metrics(trades_v1)
    m2 = compute_metrics(trades_v2)
    pr("V1 (shared position)", m1)
    pr("V2 (independent positions)", m2)
    print(f"  {'Correct merge (target)':55s} |   873t | R=         | PPDD=       | PF= 2.78 | DD= 14.1R |")
    print(f"  {'U2 baseline':55s} |  2589t | R=  +791.5 | PPDD= 28.35 | PF= 1.59 | DD= 27.9R | 0.98/d")


if __name__ == "__main__":
    main()
