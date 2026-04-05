"""
experiments/tp_targeting_research.py -- TP Targeting Research for Chain Strategy
===============================================================================

Current TP = IRL_swing × 0.35 (arbitrary from U2).
Research: what's the optimal TP for breakdown→FVG chain entries?

Approach:
  1. Run chain strategy with track_mfe=True (max favorable excursion)
  2. For each trade, record how far price went in our favor
  3. Test different TP targets: fixed RR, IRL-based, session levels, ATR-based
  4. Find optimal trim point and runner target
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


def run_mfe_capture(d, breakdowns, max_wait_bars=30, min_stop_pts=5.0,
                     stop_buffer_pct=0.15, tighten_factor=0.85, max_fvg_age=200):
    """Run chain entry but DON'T exit at TP. Track MFE and MAE for each trade.
    Exit only at stop, EOD, or max holding period.
    This tells us how far price ACTUALLY goes after our entry."""

    params = d["params"]
    nq, n = d["nq"], d["n"]
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    atr_arr = d["atr_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    fvg_df = d["fvg_df"]
    dates, dow_arr, et_frac_arr = d["dates"], d["dow_arr"], d["et_frac_arr"]

    session_cache = pd.read_parquet(PROJECT / "data" / "cache_session_levels_10yr_v2.parquet")
    on_hi = session_cache["overnight_high"].values
    on_lo = session_cache["overnight_low"].values

    fvg_bull_mask = fvg_df["fvg_bull"].values
    fvg_bull_top = fvg_df["fvg_bull_top"].values
    fvg_bull_bottom = fvg_df["fvg_bull_bottom"].values
    fvg_bear_mask = fvg_df["fvg_bear"].values
    fvg_bear_top = fvg_df["fvg_bear_top"].values
    fvg_bear_bottom = fvg_df["fvg_bear_bottom"].values

    slippage = params["backtest"]["slippage_normal_ticks"] * 0.25

    # Pre-compute zones
    zones_by_bar = {}
    for bd in breakdowns:
        fvg = find_fvg_after_breakdown(
            fvg_bull_mask, fvg_bull_top, fvg_bull_bottom,
            fvg_bear_mask, fvg_bear_top, fvg_bear_bottom,
            atr_arr, bd["bar_idx"], bd["level_type"], max_wait_bars,
            0.0, 99.0, 0.3, h, l, c, o)
        if fvg is None:
            continue
        bar = fvg["fvg_bar"]
        zone = Zone(fvg["direction"], fvg["fvg_top"], fvg["fvg_bottom"],
                     fvg["fvg_size"], bar,
                     atr_arr[bar] if not np.isnan(atr_arr[bar]) else 30.0)
        zones_by_bar.setdefault(bar, []).append(zone)

    active_zones = []
    results = []
    in_pos = False
    p_dir = p_idx = 0
    p_entry = p_stop = 0.0
    p_stop_dist = 0.0
    p_mfe = 0.0  # max favorable excursion (points)
    p_mae = 0.0  # max adverse excursion (points)
    p_atr_at_entry = 30.0
    p_on_hi = p_on_lo = 0.0
    p_irl_hi = p_irl_lo = 0.0
    cur_date = None
    day_pnl = 0.0
    consec_loss = 0
    day_stopped = False

    irl_high_arr = d["irl_high_arr"]
    irl_low_arr = d["irl_low_arr"]

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

        # EXIT: stop or EOD only (no TP — we want to see full MFE)
        if in_pos:
            # Track MFE/MAE
            if p_dir == 1:
                fe = h[i] - p_entry
                ae = p_entry - l[i]
            else:
                fe = p_entry - l[i]
                ae = h[i] - p_entry
            if fe > p_mfe:
                p_mfe = fe
            if ae > p_mae:
                p_mae = ae

            exited = False
            ex_reason = ""
            ex_price = 0.0

            # Stop
            if p_dir == 1 and l[i] <= p_stop:
                ex_price = p_stop - slippage
                ex_reason = "stop"
                exited = True
            elif p_dir == -1 and h[i] >= p_stop:
                ex_price = p_stop + slippage
                ex_reason = "stop"
                exited = True

            # EOD
            if not exited and et_frac_arr[i] >= 15.917:
                ex_price = c[i] - 0.25 if p_dir == 1 else c[i] + 0.25
                ex_reason = "eod_close"
                exited = True

            if exited:
                pnl_pts = (ex_price - p_entry) * p_dir
                results.append({
                    "entry_time": nq.index[p_idx],
                    "dir": p_dir,
                    "entry_price": p_entry,
                    "stop_price": p_stop,
                    "stop_dist": p_stop_dist,
                    "exit_price": ex_price,
                    "exit_reason": ex_reason,
                    "pnl_pts": pnl_pts,
                    "mfe_pts": p_mfe,
                    "mae_pts": p_mae,
                    "mfe_r": p_mfe / p_stop_dist if p_stop_dist > 0 else 0,
                    "mae_r": p_mae / p_stop_dist if p_stop_dist > 0 else 0,
                    "mfe_atr": p_mfe / p_atr_at_entry if p_atr_at_entry > 0 else 0,
                    "atr_at_entry": p_atr_at_entry,
                    "on_hi": p_on_hi,
                    "on_lo": p_on_lo,
                    "irl_hi": p_irl_hi,
                    "irl_lo": p_irl_lo,
                    "bars_held": i - p_idx,
                })
                # Simple daily tracking
                r = pnl_pts / p_stop_dist if p_stop_dist > 0 else 0
                day_pnl += r
                if r < -0.5:
                    consec_loss += 1
                else:
                    consec_loss = 0
                if consec_loss >= 2:
                    day_stopped = True
                if day_pnl <= -2.0:
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

            # Bias alignment
            if z.direction == 1 and bias_dir_arr[i] < 0:
                continue
            if z.direction == -1 and bias_dir_arr[i] > 0:
                continue

            fq = -abs(c[i] - ep)
            if best is None or fq > best:
                best = fq
                best_z = (z, ep, sp, sd)
                best_sbs = sbs

        if best_z is None:
            continue
        z, ep, sp, sd = best_z
        z.used = True

        if best_sbs:
            # Record same-bar stop as MFE=0
            results.append({
                "entry_time": nq.index[i], "dir": z.direction,
                "entry_price": ep, "stop_price": sp, "stop_dist": sd,
                "exit_price": sp - slippage if z.direction == 1 else sp + slippage,
                "exit_reason": "same_bar_stop",
                "pnl_pts": -sd - slippage, "mfe_pts": 0, "mae_pts": sd + slippage,
                "mfe_r": 0, "mae_r": 1.0 + slippage/sd if sd > 0 else 1,
                "mfe_atr": 0, "atr_at_entry": atr_arr[i] if not np.isnan(atr_arr[i]) else 30,
                "on_hi": on_hi[i] if not np.isnan(on_hi[i]) else 0,
                "on_lo": on_lo[i] if not np.isnan(on_lo[i]) else 0,
                "irl_hi": irl_high_arr[i] if not np.isnan(irl_high_arr[i]) else 0,
                "irl_lo": irl_low_arr[i] if not np.isnan(irl_low_arr[i]) else 0,
                "bars_held": 0,
            })
            day_pnl -= 1.0
            consec_loss += 1
            if consec_loss >= 2: day_stopped = True
            if day_pnl <= -2.0: day_stopped = True
            continue

        in_pos = True
        p_dir = z.direction
        p_idx = i
        p_entry = ep
        p_stop = sp
        p_stop_dist = sd
        p_mfe = 0.0
        p_mae = 0.0
        p_atr_at_entry = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
        p_on_hi = on_hi[i] if not np.isnan(on_hi[i]) else 0
        p_on_lo = on_lo[i] if not np.isnan(on_lo[i]) else 0
        p_irl_hi = irl_high_arr[i] if not np.isnan(irl_high_arr[i]) else 0
        p_irl_lo = irl_low_arr[i] if not np.isnan(irl_low_arr[i]) else 0

    return results


def simulate_tp(results, tp_r_mult):
    """Simulate a fixed TP at tp_r_mult × stop_distance.
    If MFE >= TP, trade wins tp_r_mult R. Else, use actual exit."""
    trades = []
    for r in results:
        sd = r["stop_dist"]
        if sd <= 0:
            continue
        tp_dist = sd * tp_r_mult

        if r["exit_reason"] == "same_bar_stop":
            trades.append({"r": -1.0, "dir": r["dir"]})
            continue

        if r["mfe_pts"] >= tp_dist:
            # TP hit
            trades.append({"r": tp_r_mult, "dir": r["dir"]})
        else:
            # TP not hit, use actual exit
            actual_r = r["pnl_pts"] / sd if sd > 0 else 0
            trades.append({"r": actual_r, "dir": r["dir"]})
    return trades


def simulate_trim_runner(results, trim_r, trim_pct=0.25):
    """Simulate trim at trim_r × stop_dist, runner to actual exit."""
    trades = []
    for r in results:
        sd = r["stop_dist"]
        if sd <= 0:
            continue
        tp_dist = sd * trim_r

        if r["exit_reason"] == "same_bar_stop":
            trades.append({"r": -1.0, "dir": r["dir"]})
            continue

        if r["mfe_pts"] >= tp_dist:
            # Trim hit: trim_pct at TP, rest at actual exit
            trim_r_val = trim_r * trim_pct
            actual_r = r["pnl_pts"] / sd if sd > 0 else 0
            runner_r_val = actual_r * (1 - trim_pct)
            # But runner can't be worse than -stop (BE after trim)
            if runner_r_val < 0:
                runner_r_val = 0  # BE stop
            total_r = trim_r_val + runner_r_val
            trades.append({"r": total_r, "dir": r["dir"]})
        else:
            actual_r = r["pnl_pts"] / sd if sd > 0 else 0
            trades.append({"r": actual_r, "dir": r["dir"]})
    return trades


def main():
    print("=" * 120)
    print("TP TARGETING RESEARCH -- Chain Strategy")
    print("=" * 120)

    d = load_all()
    h, l, c, o = d["h"], d["l"], d["c"], d["o"]
    session_cache = pd.read_parquet(PROJECT / "data" / "cache_session_levels_10yr_v2.parquet")
    pdhl = compute_pdhl(d["nq"])

    # Breakdowns: ON_low + ON_high (best validated config)
    on_lo_bds = detect_breakdowns(h, l, c, session_cache["overnight_low"].values, "low", min_depth_pts=1.0)
    for bd in on_lo_bds: bd["level_type"] = "low"

    on_hi_bds = detect_breakdowns(h, l, c, session_cache["overnight_high"].values, "high", min_depth_pts=1.0)
    for bd in on_hi_bds: bd["level_type"] = "high"

    all_bds = on_lo_bds + on_hi_bds

    # ================================================================
    # STEP 1: Capture MFE for all trades
    # ================================================================
    print("\n[RUN] Capturing MFE (no TP, stop + EOD only)...")
    results = run_mfe_capture(d, all_bds)
    print(f"[RUN] {len(results)} trades captured")

    df = pd.DataFrame(results)
    longs = df[df["dir"] == 1]
    shorts = df[df["dir"] == -1]
    non_sbs = df[df["exit_reason"] != "same_bar_stop"]

    # ================================================================
    # STEP 2: MFE distribution
    # ================================================================
    print(f"\n{'='*100}")
    print("MFE DISTRIBUTION (how far does price go in our favor?)")
    print(f"{'='*100}")

    for tag, sub in [("ALL", non_sbs), ("LONGS", longs[longs["exit_reason"] != "same_bar_stop"]),
                      ("SHORTS", shorts[shorts["exit_reason"] != "same_bar_stop"])]:
        if len(sub) < 30:
            continue
        print(f"\n  --- {tag} ({len(sub)} trades) ---")
        mfe_r = sub["mfe_r"]
        print(f"  MFE (R multiples):  mean={mfe_r.mean():.2f}  median={mfe_r.median():.2f}  "
              f"p75={mfe_r.quantile(0.75):.2f}  p90={mfe_r.quantile(0.9):.2f}  max={mfe_r.max():.1f}")

        # What % of trades reach various R levels?
        for rr in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]:
            pct = (mfe_r >= rr).mean() * 100
            print(f"    MFE >= {rr:4.1f}R: {pct:5.1f}%")

    # ================================================================
    # STEP 3: Simulate different TP levels (full exit)
    # ================================================================
    print(f"\n{'='*100}")
    print("FULL EXIT TP SWEEP (100% exit at TP)")
    print(f"{'='*100}")

    print(f"\n  {'TP (R)':>8s} | {'Trades':>7s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>8s} | {'PPDD':>7s}")
    print(f"  {'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}")

    for tp_r in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0]:
        trades = simulate_tp(results, tp_r)
        if len(trades) < 30:
            continue
        m = compute_metrics(trades)
        print(f"  {tp_r:8.2f} | {m['trades']:7d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['WR']:5.1f} | {m['avgR']:+7.4f} | {m['PPDD']:6.2f}")

    # ================================================================
    # STEP 4: Simulate trim + runner at different trim points
    # ================================================================
    print(f"\n{'='*100}")
    print("TRIM + RUNNER SWEEP (25% trim at TP, 75% runner to actual exit, BE after trim)")
    print(f"{'='*100}")

    print(f"\n  {'Trim R':>8s} | {'Trades':>7s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>8s} | {'PPDD':>7s}")
    print(f"  {'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}")

    for trim_r in [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]:
        trades = simulate_trim_runner(results, trim_r, trim_pct=0.25)
        if len(trades) < 30:
            continue
        m = compute_metrics(trades)
        print(f"  {trim_r:8.2f} | {m['trades']:7d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['WR']:5.1f} | {m['avgR']:+7.4f} | {m['PPDD']:6.2f}")

    # ================================================================
    # STEP 5: Long vs Short optimal TP
    # ================================================================
    print(f"\n{'='*100}")
    print("LONG vs SHORT OPTIMAL TP (trim + runner)")
    print(f"{'='*100}")

    for tag, dir_filter in [("LONGS", 1), ("SHORTS", -1)]:
        sub_results = [r for r in results if r["dir"] == dir_filter]
        if len(sub_results) < 30:
            continue
        print(f"\n  --- {tag} ---")
        print(f"  {'Trim R':>8s} | {'Trades':>7s} | {'R':>8s} | {'PF':>6s} | {'WR':>6s} | {'avgR':>8s} | {'PPDD':>7s}")
        print(f"  {'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}")
        for trim_r in [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
            trades = simulate_trim_runner(sub_results, trim_r, trim_pct=0.25)
            m = compute_metrics(trades)
            print(f"  {trim_r:8.2f} | {m['trades']:7d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['WR']:5.1f} | {m['avgR']:+7.4f} | {m['PPDD']:6.2f}")

    # ================================================================
    # STEP 6: IRL-based TP vs fixed RR
    # ================================================================
    print(f"\n{'='*100}")
    print("IRL-BASED TP (dynamic target based on swing levels)")
    print(f"{'='*100}")

    for irl_mult in [0.2, 0.35, 0.5, 0.75, 1.0, 1.5]:
        trades = []
        for r in results:
            sd = r["stop_dist"]
            if sd <= 0:
                continue
            if r["exit_reason"] == "same_bar_stop":
                trades.append({"r": -1.0, "dir": r["dir"]})
                continue

            if r["dir"] == 1:
                irl_target = r["irl_hi"]
                if irl_target <= r["entry_price"] or irl_target == 0:
                    irl_target = r["entry_price"] + sd * 2
                tp_dist = (irl_target - r["entry_price"]) * irl_mult
            else:
                irl_target = r["irl_lo"]
                if irl_target >= r["entry_price"] or irl_target == 0:
                    irl_target = r["entry_price"] - sd * 2
                tp_dist = (r["entry_price"] - irl_target) * irl_mult

            if tp_dist <= 0:
                tp_dist = sd * 1.0

            # Trim + runner
            if r["mfe_pts"] >= tp_dist:
                trim_r_val = (tp_dist / sd) * 0.25
                actual_r = r["pnl_pts"] / sd if sd > 0 else 0
                runner_r_val = max(0, actual_r) * 0.75
                trades.append({"r": trim_r_val + runner_r_val, "dir": r["dir"]})
            else:
                actual_r = r["pnl_pts"] / sd if sd > 0 else 0
                trades.append({"r": actual_r, "dir": r["dir"]})

        if len(trades) >= 30:
            m = compute_metrics(trades)
            tpd = m["trades"] / (252 * 10.5)
            print(f"  IRL x {irl_mult:4.2f} | {m['trades']:5d}t | R={m['R']:+8.1f} | PF={m['PF']:5.2f} | "
                  f"PPDD={m['PPDD']:6.2f} | WR={m['WR']:5.1f}% | {tpd:.2f}/d")

    # ================================================================
    # STEP 7: Session level as TP target
    # ================================================================
    print(f"\n{'='*100}")
    print("SESSION LEVEL TP (trim at overnight high/low)")
    print(f"{'='*100}")

    trades_session = []
    for r in results:
        sd = r["stop_dist"]
        if sd <= 0:
            continue
        if r["exit_reason"] == "same_bar_stop":
            trades_session.append({"r": -1.0, "dir": r["dir"]})
            continue

        if r["dir"] == 1:
            # Long: target = overnight high
            target = r["on_hi"]
            if target <= r["entry_price"] or target == 0:
                target = r["entry_price"] + sd * 2
            tp_dist = target - r["entry_price"]
        else:
            target = r["on_lo"]
            if target >= r["entry_price"] or target == 0:
                target = r["entry_price"] - sd * 2
            tp_dist = r["entry_price"] - target

        if tp_dist <= 0:
            tp_dist = sd * 1.0

        if r["mfe_pts"] >= tp_dist:
            trim_r_val = (tp_dist / sd) * 0.25
            actual_r = r["pnl_pts"] / sd if sd > 0 else 0
            runner_r_val = max(0, actual_r) * 0.75
            trades_session.append({"r": trim_r_val + runner_r_val, "dir": r["dir"]})
        else:
            actual_r = r["pnl_pts"] / sd if sd > 0 else 0
            trades_session.append({"r": actual_r, "dir": r["dir"]})

    if len(trades_session) >= 30:
        m = compute_metrics(trades_session)
        print(f"  Session level TP | {m['trades']:5d}t | R={m['R']:+8.1f} | PF={m['PF']:5.2f} | PPDD={m['PPDD']:6.2f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*100}")
    print("SUMMARY: Compare current TP (IRL x 0.35) vs best alternatives")
    print(f"{'='*100}")

    # Current: IRL x 0.35
    current_trades = simulate_trim_runner(results, 0.75, 0.25)  # ~equivalent to IRL*0.35 which is roughly 0.75R
    m_current = compute_metrics(current_trades)
    print(f"  Current approx (trim 0.75R): {m_current['trades']:5d}t R={m_current['R']:+8.1f} PF={m_current['PF']:5.2f} PPDD={m_current['PPDD']:6.2f}")
    print(f"  Reference: ON chain actual:  1685t R=+955.2 PF=2.04 PPDD=42.99")


if __name__ == "__main__":
    main()
