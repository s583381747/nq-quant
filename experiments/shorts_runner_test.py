"""
experiments/shorts_runner_test.py -- Test shorts with proper runner management
=============================================================================

Hypothesis: Shorts fail because they get 100% trim at TP1 (no runner).
If we give shorts the same 25% trim + trail + BE management as longs,
shorts should become profitable.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.breakdown_chain_research import detect_breakdowns, find_fvg_after_breakdown
from experiments.sweep_research import compute_pdhl
from experiments.u2_clean import load_all, compute_metrics, pr, _find_nth_swing, _compute_grade
from features.swing import compute_swing_levels
from dataclasses import dataclass


@dataclass
class Zone:
    direction: int
    top: float
    bottom: float
    size: float
    birth_bar: int
    birth_atr: float
    used: bool = False


def run_with_symmetric_management(d, breakdowns, trim_pct=0.25, tp_mult=0.35,
                                   nth_swing=5, tighten_factor=0.85,
                                   min_stop_pts=5.0, stop_buffer_pct=0.15,
                                   max_fvg_age=200, max_wait_bars=30,
                                   min_disp_atr=0.0, max_disp_atr=99.0):
    params = d["params"]
    nq, n = d["nq"], d["n"]
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    atr_arr = d["atr_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    fvg_df = d["fvg_df"]
    dates, dow_arr, et_frac_arr = d["dates"], d["dow_arr"], d["et_frac_arr"]
    swing_low_mask = d["swing_low_mask"]
    swing_low_price_at_mask = d.get("swing_low_price_at_mask")
    swing_high_mask = d["swing_high_mask"]

    # Build swing_high_price_at_mask for short trailing
    rb = params["swing"]["right_bars"]
    raw_sh = compute_swing_levels(nq, {"left_bars": params["swing"]["left_bars"],
                                        "right_bars": rb})["swing_high"].values
    swing_high_price_at_mask = np.full(n, np.nan)
    for j in range(n):
        if raw_sh[j] and j + rb < n:
            swing_high_price_at_mask[j + rb] = h[j]

    fvg_bull_mask = fvg_df["fvg_bull"].values
    fvg_bull_top = fvg_df["fvg_bull_top"].values
    fvg_bull_bottom = fvg_df["fvg_bull_bottom"].values
    fvg_bear_mask = fvg_df["fvg_bear"].values
    fvg_bear_top = fvg_df["fvg_bear_top"].values
    fvg_bear_bottom = fvg_df["fvg_bear_bottom"].values

    pos_params = params["position"]
    risk_params = params["risk"]
    bt_params = params["backtest"]
    grading_params = params["grading"]
    session_regime = params.get("session_regime", {})

    normal_r, reduced_r = pos_params["normal_r"], pos_params["reduced_r"]
    point_value = pos_params["point_value"]
    daily_max_loss_r = risk_params["daily_max_loss_r"]
    max_consec = risk_params["max_consecutive_losses"]
    comm = bt_params["commission_per_side_micro"]
    slip = bt_params["slippage_normal_ticks"] * 0.25
    a_mult = grading_params["a_plus_size_mult"]
    b_mult = grading_params["b_plus_size_mult"]

    # Pre-compute zones
    zones_by_bar = {}
    for bd in breakdowns:
        fvg = find_fvg_after_breakdown(
            fvg_bull_mask, fvg_bull_top, fvg_bull_bottom,
            fvg_bear_mask, fvg_bear_top, fvg_bear_bottom,
            atr_arr, bd["bar_idx"], bd["level_type"], max_wait_bars,
            min_disp_atr, max_disp_atr, 0.3, h, l, c, o)
        if fvg is None:
            continue
        bar = fvg["fvg_bar"]
        zone = Zone(fvg["direction"], fvg["fvg_top"], fvg["fvg_bottom"],
                     fvg["fvg_size"], bar, atr_arr[bar] if not np.isnan(atr_arr[bar]) else 30.0)
        zones_by_bar.setdefault(bar, []).append(zone)

    active_zones = []
    trades = []
    in_pos = False
    p_dir = p_idx = 0
    p_entry = p_stop = p_tp1 = 0.0
    p_contracts = p_remaining = 0
    p_trimmed = False
    p_be = p_trail = 0.0
    p_grade = ""
    cur_date = None
    day_pnl = 0.0
    consec_loss = 0
    day_stopped = False

    for i in range(n):
        if dates[i] != cur_date:
            cur_date = dates[i]
            day_pnl = 0.0
            consec_loss = 0
            day_stopped = False

        if i in zones_by_bar:
            active_zones.extend(zones_by_bar[i])

        surviving = []
        for z in active_zones:
            if z.used or (i - z.birth_bar) > max_fvg_age:
                continue
            if z.direction == 1 and c[i] < z.bottom:
                continue
            if z.direction == -1 and c[i] > z.top:
                continue
            surviving.append(z)
        active_zones = surviving[-30:] if len(surviving) > 30 else surviving

        # EXIT
        if in_pos:
            exited = False
            ex_reason = ""
            ex_price = 0.0
            ex_contracts = p_remaining

            if p_dir == 1:  # Long
                eff = p_trail if p_trimmed and p_trail > 0 else p_stop
                if p_trimmed and p_be > 0:
                    eff = max(eff, p_be)
                if l[i] <= eff:
                    ex_price = eff - slip
                    ex_reason = "be_sweep" if (p_trimmed and eff >= p_entry) else "stop"
                    exited = True
                elif not p_trimmed and h[i] >= p_tp1:
                    tc = max(1, int(p_contracts * trim_pct))
                    p_remaining = p_contracts - tc
                    p_trimmed = True
                    p_be = p_entry
                    if p_remaining > 0:
                        p_trail = _find_nth_swing(swing_low_mask, swing_low_price_at_mask, i, nth_swing)
                        if np.isnan(p_trail) or p_trail <= 0:
                            p_trail = p_be
                    if p_remaining <= 0:
                        ex_price = p_tp1
                        ex_reason = "tp1"
                        ex_contracts = p_contracts
                        exited = True
                if p_trimmed and not exited:
                    nt = _find_nth_swing(swing_low_mask, swing_low_price_at_mask, i, nth_swing)
                    if not np.isnan(nt) and nt > p_trail:
                        p_trail = nt

            else:  # Short — SAME management pattern, mirrored
                eff = p_trail if p_trimmed and p_trail > 0 else p_stop
                if p_trimmed and p_be > 0:
                    eff = min(eff, p_be)  # for shorts, BE is above entry
                if h[i] >= eff:
                    ex_price = eff + slip
                    ex_reason = "be_sweep" if (p_trimmed and eff <= p_entry) else "stop"
                    exited = True
                elif not p_trimmed and l[i] <= p_tp1:
                    tc = max(1, int(p_contracts * trim_pct))
                    p_remaining = p_contracts - tc
                    p_trimmed = True
                    p_be = p_entry
                    if p_remaining > 0:
                        p_trail = _find_nth_swing(swing_high_mask, swing_high_price_at_mask, i, nth_swing)
                        if np.isnan(p_trail) or p_trail <= 0:
                            p_trail = p_be
                    if p_remaining <= 0:
                        ex_price = p_tp1
                        ex_reason = "tp1"
                        ex_contracts = p_contracts
                        exited = True
                if p_trimmed and not exited:
                    nt = _find_nth_swing(swing_high_mask, swing_high_price_at_mask, i, nth_swing)
                    if not np.isnan(nt) and nt > 0 and nt < p_trail:
                        p_trail = nt

            # EOD
            if not exited and et_frac_arr[i] >= 15.917:
                ex_price = c[i] - 0.25 if p_dir == 1 else c[i] + 0.25
                ex_reason = "eod_close"
                exited = True

            if exited:
                pnl_pts = (ex_price - p_entry) * p_dir
                if p_trimmed and ex_reason != "tp1":
                    trim_c = p_contracts - ex_contracts
                    trim_pnl = (p_tp1 - p_entry) * p_dir * point_value * trim_c
                    total_pnl = trim_pnl + pnl_pts * point_value * ex_contracts
                    total_pnl -= comm * 2 * p_contracts
                else:
                    total_pnl = pnl_pts * point_value * ex_contracts - comm * 2 * ex_contracts
                sd = abs(p_entry - p_stop)
                risk = sd * point_value * p_contracts
                r = total_pnl / risk if risk > 0 else 0

                trades.append({"entry_time": nq.index[p_idx], "exit_time": nq.index[i],
                    "r": r, "reason": ex_reason, "dir": p_dir, "trimmed": p_trimmed,
                    "grade": p_grade, "entry_price": p_entry, "exit_price": ex_price,
                    "stop_price": p_stop, "tp1_price": p_tp1, "pnl_dollars": total_pnl,
                    "stop_dist_pts": sd})

                day_pnl += r
                if ex_reason == "be_sweep" and p_trimmed:
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
                in_pos = False

        # ENTRY
        if in_pos or day_stopped:
            continue
        if news_blackout_arr is not None and news_blackout_arr[i]:
            continue
        ef = et_frac_arr[i]
        if 9.5 <= ef <= 10.0 or not (10.0 <= ef < 16.0):
            continue

        best = None
        best_z = None
        best_sbs = False

        for z in active_zones:
            if z.used or z.birth_bar >= i:
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

            fq = -abs(c[i] - ep)
            if best is None or fq > best:
                best = fq
                best_z = (z, ep, sp, sd)
                best_sbs = sbs

        if best_z is None:
            continue
        z, ep, sp, sd = best_z
        z.used = True
        d_dir = z.direction

        ba = 1.0 if (d_dir == np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0) else 0.0
        grade = _compute_grade(ba, regime_arr[i])
        is_red = (dow_arr[i] in (0, 4)) or (regime_arr[i] < 1.0)
        base = reduced_r if is_red else normal_r
        if grade == "A+": ra = base * a_mult
        elif grade == "B+": ra = base * b_mult
        else: ra = base * 0.5

        if session_regime.get("enabled", False):
            ls = session_regime.get("lunch_start", 12.0)
            le = session_regime.get("lunch_end", 13.0)
            if ls <= ef < le:
                z.used = False
                continue

        cts = max(1, int(ra / (sd * point_value))) if sd > 0 else 0
        if cts <= 0:
            z.used = False
            continue

        if best_sbs:
            exp = (sp - slip) if d_dir == 1 else (sp + slip)
            pp = ((exp - ep) if d_dir == 1 else (ep - exp)) * point_value * cts - comm * 2 * cts
            rr = pp / (sd * point_value * cts) if sd > 0 else 0
            trades.append({"entry_time": nq.index[i], "exit_time": nq.index[i],
                "r": rr, "reason": "same_bar_stop", "dir": d_dir, "trimmed": False,
                "grade": grade, "entry_price": ep, "exit_price": exp,
                "stop_price": sp, "tp1_price": 0, "pnl_dollars": pp, "stop_dist_pts": sd})
            day_pnl += rr
            consec_loss += 1
            if consec_loss >= max_consec: day_stopped = True
            if day_pnl <= -daily_max_loss_r: day_stopped = True
            continue

        if d_dir == 1:
            irl = d["irl_high_arr"][i]
            if np.isnan(irl) or irl <= ep: irl = ep + sd * 2
            tp = ep + (irl - ep) * tp_mult
        else:
            irl = d["irl_low_arr"][i]
            if np.isnan(irl) or irl >= ep: irl = ep - sd * 2
            tp = ep - abs(ep - irl) * tp_mult

        in_pos = True
        p_dir = d_dir
        p_idx = i
        p_entry = ep
        p_stop = sp
        p_tp1 = tp
        p_contracts = cts
        p_remaining = cts
        p_trimmed = False
        p_be = 0.0
        p_trail = 0.0
        p_grade = grade

    return trades


def main():
    d = load_all()
    h, l, c, o = d["h"], d["l"], d["c"], d["o"]
    session_cache = pd.read_parquet(PROJECT / "data" / "cache_session_levels_10yr_v2.parquet")
    pdhl = compute_pdhl(d["nq"])

    pdl_bds = detect_breakdowns(h, l, c, pdhl["pdl"].values, "low", min_depth_pts=1.0)
    for bd in pdl_bds: bd["level_type"] = "low"

    pdh_bds = detect_breakdowns(h, l, c, pdhl["pdh"].values, "high", min_depth_pts=1.0)
    for bd in pdh_bds: bd["level_type"] = "high"

    on_lo_bds = detect_breakdowns(h, l, c, session_cache["overnight_low"].values, "low", min_depth_pts=1.0)
    for bd in on_lo_bds: bd["level_type"] = "low"

    on_hi_bds = detect_breakdowns(h, l, c, session_cache["overnight_high"].values, "high", min_depth_pts=1.0)
    for bd in on_hi_bds: bd["level_type"] = "high"

    print("=" * 100)
    print("SHORTS WITH SYMMETRIC RUNNER MANAGEMENT")
    print("=" * 100)

    configs = [
        ("PDL longs only", pdl_bds),
        ("PDH shorts only", pdh_bds),
        ("PDL + PDH", pdl_bds + pdh_bds),
        ("ON_low + ON_high", on_lo_bds + on_hi_bds),
        ("PDL+PDH+ON", pdl_bds + pdh_bds + on_lo_bds + on_hi_bds),
    ]

    for label, bds in configs:
        trades = run_with_symmetric_management(d, bds)
        longs = [t for t in trades if t["dir"] == 1]
        shorts = [t for t in trades if t["dir"] == -1]
        all_m = compute_metrics(trades)

        print(f"\n  --- {label} ---")
        if longs:
            m = compute_metrics(longs)
            tpd = m["trades"] / (252 * 10.5)
            print(f"    LONGS:  {m['trades']:4d}t R={m['R']:+8.1f} PF={m['PF']:5.2f} PPDD={m['PPDD']:6.2f} DD={m['MaxDD']:5.1f}R {tpd:.2f}/d")
        if shorts:
            m = compute_metrics(shorts)
            tpd = m["trades"] / (252 * 10.5)
            print(f"    SHORTS: {m['trades']:4d}t R={m['R']:+8.1f} PF={m['PF']:5.2f} PPDD={m['PPDD']:6.2f} DD={m['MaxDD']:5.1f}R {tpd:.2f}/d")
        tpd = all_m["trades"] / (252 * 10.5)
        print(f"    ALL:    {all_m['trades']:4d}t R={all_m['R']:+8.1f} PF={all_m['PF']:5.2f} PPDD={all_m['PPDD']:6.2f} DD={all_m['MaxDD']:5.1f}R {tpd:.2f}/d")

        # Exit reasons for shorts
        if shorts:
            df_s = pd.DataFrame(shorts)
            print(f"    Short exit reasons:")
            for reason, grp in df_s.groupby("reason"):
                print(f"      {reason:20s}: {len(grp):4d}  R={grp['r'].sum():+8.1f}  avgR={grp['r'].mean():+.3f}")


if __name__ == "__main__":
    main()
