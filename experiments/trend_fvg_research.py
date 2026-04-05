"""
experiments/trend_fvg_research.py -- Trend FVG Entry Research
=============================================================

Chain engine uses 294 trades (breakdown->FVG). But there are 143,642 FVGs
in 10 years. The ~2000 FVGs that U2 traded without breakdown context had
PF=1.59. Can we identify which non-breakdown FVGs are worth trading?

Key question: WITHOUT a breakdown event, what makes an FVG tradeable?

Approach:
  1. Take ALL FVG limit fills during NY session (like U2 does)
  2. Exclude ones that are already captured by chain engine (breakdown FVGs)
  3. Analyze: what distinguishes good trend FVGs from bad ones?
  4. Find filters/sizing that make trend FVGs additive to the chain strategy
"""
from __future__ import annotations
import sys, copy
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.chain_engine import (
    load_all, compute_metrics, pr, detect_breakdowns,
    find_fvg_not_mss, ChainZone, _find_nth_swing, _compute_grade
)


def run_trend_fvg_backtest(
    d: dict,
    *,
    # Which FVGs to include
    include_breakdown_fvgs: bool = False,  # False = trend only (exclude chain entries)
    require_not_mss: bool = False,  # Apply NOT-MSS filter to trend FVGs too?
    # FVG params
    min_fvg_size_atr: float = 0.3,
    max_fvg_age: int = 200,
    # Stop
    stop_buffer_pct: float = 0.15,
    tighten_factor: float = 0.85,
    min_stop_pts: float = 5.0,
    # Exit
    trim_pct: float = 0.25,
    fixed_tp_r: float = 1.0,
    nth_swing: int = 5,
    eod_close: bool = True,
    # Direction filter
    longs_only: bool = False,
    shorts_only: bool = False,
) -> list[dict]:
    """Run FVG limit order backtest WITHOUT requiring breakdown.

    This is essentially the U2 engine but with:
    - Symmetric management (longs + shorts get runners)
    - Fixed 1R TP
    - Optional NOT-MSS filter
    - Optional exclusion of breakdown-associated FVGs
    """
    params = d["params"]
    nq, n = d["nq"], d["n"]
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    atr_arr = d["atr_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    fvg_df = d["fvg_df"]
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

    bull_mask = fvg_df["fvg_bull"].values
    bear_mask = fvg_df["fvg_bear"].values
    fvg_bt = fvg_df["fvg_bull_top"].values
    fvg_bb = fvg_df["fvg_bull_bottom"].values
    fvg_et = fvg_df["fvg_bear_top"].values
    fvg_eb = fvg_df["fvg_bear_bottom"].values

    sh_prices = swing_high_price_at_mask
    sl_prices = swing_low_price_at_mask

    # Pre-compute breakdown bars to EXCLUDE if needed
    exclude_bars = set()
    if not include_breakdown_fvgs:
        on_lo, on_hi = d["on_lo"], d["on_hi"]
        for level_arr, ltype in [(on_lo, "low"), (on_hi, "high")]:
            bds = detect_breakdowns(h, l, c, level_arr, ltype, min_depth_pts=1.0)
            for bd in bds:
                # Mark bars within 30 of breakdown as "chain territory"
                for b in range(bd["bar_idx"], min(bd["bar_idx"] + 31, n)):
                    exclude_bars.add(b)

    # Build FVG zones bar by bar
    from dataclasses import dataclass as _dc

    @_dc
    class TrendZone:
        direction: int
        top: float
        bottom: float
        size: float
        birth_bar: int
        birth_atr: float
        used: bool = False

    active_zones: list[TrendZone] = []
    trades: list[dict] = []
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

        # Register new FVG zones
        for is_bull in [True, False]:
            mask = bull_mask if is_bull else bear_mask
            if not mask[i]:
                continue
            if not include_breakdown_fvgs and i in exclude_bars:
                continue

            direction = 1 if is_bull else -1
            if longs_only and direction == -1:
                continue
            if shorts_only and direction == 1:
                continue

            top = fvg_bt[i] if is_bull else fvg_et[i]
            bottom = fvg_bb[i] if is_bull else fvg_eb[i]
            size = top - bottom
            atr_val = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0

            if atr_val > 0 and size < min_fvg_size_atr * atr_val:
                continue

            # Optional NOT-MSS filter
            if require_not_mss:
                disp_bar = i - 1
                if disp_bar < 0:
                    continue
                breaks_swing = False
                if direction == 1:
                    for j in range(disp_bar, max(0, disp_bar - 100), -1):
                        if swing_high_mask[j]:
                            p = sh_prices[j]
                            if not np.isnan(p) and p > 0 and h[disp_bar] > p:
                                breaks_swing = True
                                break
                else:
                    for j in range(disp_bar, max(0, disp_bar - 100), -1):
                        if swing_low_mask[j]:
                            p = sl_prices[j]
                            if not np.isnan(p) and p > 0 and l[disp_bar] < p:
                                breaks_swing = True
                                break
                if breaks_swing:
                    continue

            active_zones.append(TrendZone(direction, top, bottom, size, i, atr_val))

        # Invalidate/expire
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

        # EXIT (symmetric, same as chain engine)
        if in_pos:
            exited = False
            ex_reason = ""
            ex_price = 0.0
            ex_contracts = p_remaining

            if p_dir == 1:
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
            else:
                eff = p_trail if p_trimmed and p_trail > 0 else p_stop
                if p_trimmed and p_be > 0:
                    if eff > p_be:
                        eff = p_be
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
                        nt_init = _find_nth_swing(swing_high_mask, swing_high_price_at_mask, i, nth_swing)
                        if np.isnan(nt_init) or nt_init <= 0:
                            p_trail = p_be
                        elif nt_init > p_entry:
                            p_trail = p_be
                        else:
                            p_trail = nt_init
                    if p_remaining <= 0:
                        ex_price = p_tp1
                        ex_reason = "tp1"
                        ex_contracts = p_contracts
                        exited = True
                if p_trimmed and not exited:
                    nt = _find_nth_swing(swing_high_mask, swing_high_price_at_mask, i, nth_swing)
                    if not np.isnan(nt) and nt > 0 and nt < p_trail:
                        p_trail = nt

            if not exited and eod_close and et_frac_arr[i] >= 15.917:
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
                    total_pnl = pnl_pts * point_value * ex_contracts
                    total_pnl -= comm * 2 * ex_contracts
                sd = abs(p_entry - p_stop)
                risk = sd * point_value * p_contracts
                r = total_pnl / risk if risk > 0 else 0.0

                trades.append({
                    "entry_time": nq.index[p_idx],
                    "exit_time": nq.index[i],
                    "r": r, "reason": ex_reason, "dir": p_dir,
                    "trimmed": p_trimmed, "grade": p_grade,
                    "entry_price": p_entry, "exit_price": ex_price,
                    "stop_price": p_stop, "tp1_price": p_tp1,
                    "pnl_dollars": total_pnl, "stop_dist_pts": sd,
                })
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

        best_fill = None
        best_zone_info = None
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

            if z.direction == 1 and bias_dir_arr[i] < 0: continue
            if z.direction == -1 and bias_dir_arr[i] > 0: continue

            fq = -abs(c[i] - ep)
            if best_fill is None or fq > best_fill:
                best_fill = fq
                best_zone_info = (z, ep, sp, sd)
                best_sbs = sbs

        if best_zone_info is None:
            continue

        z, ep, sp, sd = best_zone_info
        z.used = True
        direction = z.direction

        ba = 1.0 if (direction == np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0) else 0.0
        grade = _compute_grade(ba, regime_arr[i])
        is_reduced = (dow_arr[i] in (0, 4)) or (regime_arr[i] < 1.0)
        base_r = reduced_r if is_reduced else normal_r
        if grade == "A+": r_amount = base_r * a_mult
        elif grade == "B+": r_amount = base_r * b_mult
        else: r_amount = base_r * 0.5

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
                "trimmed": False, "grade": grade,
                "entry_price": ep, "exit_price": exp,
                "stop_price": sp, "tp1_price": 0.0,
                "pnl_dollars": pp, "stop_dist_pts": sd,
            })
            day_pnl += rr
            consec_loss += 1
            if consec_loss >= max_consec: day_stopped = True
            if day_pnl <= -daily_max_loss_r: day_stopped = True
            continue

        tp1 = ep + sd * fixed_tp_r if direction == 1 else ep - sd * fixed_tp_r

        in_pos = True
        p_dir = direction
        p_idx = i
        p_entry = ep
        p_stop = sp
        p_tp1 = tp1
        p_contracts = contracts
        p_remaining = contracts
        p_trimmed = False
        p_be = 0.0
        p_trail = 0.0
        p_grade = grade

    return trades


def main():
    d = load_all()

    print("=" * 120)
    print("TREND FVG RESEARCH -- Non-breakdown FVG entries")
    print("=" * 120)

    # Baseline: chain engine (breakdown only)
    from experiments.chain_engine import run_chain_backtest
    chain_trades, _ = run_chain_backtest(d)
    m_chain = compute_metrics(chain_trades)
    print(f"\n  Chain baseline:")
    pr("Chain (breakdown only)", m_chain)

    # Config sweep
    configs = [
        ("ALL FVGs (longs+shorts)", {"include_breakdown_fvgs": True}),
        ("ALL FVGs longs only", {"include_breakdown_fvgs": True, "longs_only": True}),
        ("ALL FVGs + NOT-MSS", {"include_breakdown_fvgs": True, "require_not_mss": True}),
        ("ALL FVGs + NOT-MSS longs", {"include_breakdown_fvgs": True, "require_not_mss": True, "longs_only": True}),
        ("TREND only (excl breakdown)", {"include_breakdown_fvgs": False}),
        ("TREND only longs", {"include_breakdown_fvgs": False, "longs_only": True}),
        ("TREND only + NOT-MSS", {"include_breakdown_fvgs": False, "require_not_mss": True}),
        ("TREND only + NOT-MSS longs", {"include_breakdown_fvgs": False, "require_not_mss": True, "longs_only": True}),
    ]

    print(f"\n{'='*120}")
    print("TREND FVG CONFIGS")
    print(f"{'='*120}")
    print(f"\n  {'Config':40s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'PPDD':>7s} | {'DD':>6s} | {'/day':>5s} | {'L:PF':>6s} | {'S:PF':>6s}")
    print(f"  {'-'*40}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}")

    for label, kwargs in configs:
        trades = run_trend_fvg_backtest(d, **kwargs)
        if len(trades) < 20:
            print(f"  {label:40s} | {len(trades):5d} | too few")
            continue
        longs = [t for t in trades if t["dir"] == 1]
        shorts = [t for t in trades if t["dir"] == -1]
        m = compute_metrics(trades)
        ml = compute_metrics(longs) if longs else {"PF": 0}
        ms = compute_metrics(shorts) if shorts else {"PF": 0}
        tpd = m["trades"] / (252 * 10.5)
        print(f"  {label:40s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['PPDD']:6.2f} | {m['MaxDD']:5.1f}R | {tpd:.2f} | {ml['PF']:5.2f} | {ms['PF']:5.2f}")

    # Walk-forward for best trend config
    print(f"\n{'='*120}")
    print("WALK-FORWARD: TREND only + NOT-MSS")
    print(f"{'='*120}")
    trades_best = run_trend_fvg_backtest(d, include_breakdown_fvgs=False, require_not_mss=True)
    df = pd.DataFrame(trades_best)
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year
    neg = 0
    for year, grp in df.groupby("year"):
        r = grp["r"].values
        total_r = r.sum()
        w = r[r > 0].sum()
        lo = abs(r[r < 0].sum())
        pf = w / lo if lo > 0 else 999
        flag = " NEG" if total_r < 0 else ""
        if total_r < 0: neg += 1
        nl = len(grp[grp["dir"] == 1])
        ns = len(grp[grp["dir"] == -1])
        print(f"  {year}: {len(grp):3d}t (L:{nl:3d} S:{ns:3d})  R={total_r:+7.1f}  PF={pf:5.2f}{flag}")
    print(f"  Negative years: {neg}/{len(df['year'].unique())}")

    # Combined: chain + trend (simulated)
    print(f"\n{'='*120}")
    print("COMBINED: Chain (1.0x R) + Trend (0.5x R)")
    print(f"{'='*120}")
    chain_r = np.array([t["r"] for t in chain_trades])
    trend_r = np.array([t["r"] for t in trades_best]) * 0.5  # half size
    combined_r = np.concatenate([chain_r, trend_r])
    combined_r_sorted = []
    # Need to merge by time for proper equity curve
    all_trades = [(t["entry_time"], t["r"], "chain") for t in chain_trades]
    all_trades += [(t["entry_time"], t["r"] * 0.5, "trend") for t in trades_best]
    all_trades.sort(key=lambda x: x[0])
    combined_r_arr = np.array([t[1] for t in all_trades])
    total_r = combined_r_arr.sum()
    wins = combined_r_arr[combined_r_arr > 0].sum()
    losses = abs(combined_r_arr[combined_r_arr < 0].sum())
    pf = wins / losses if losses > 0 else 999
    cumr = np.cumsum(combined_r_arr)
    peak = np.maximum.accumulate(cumr)
    dd = (peak - cumr).max()
    ppdd = total_r / dd if dd > 0 else 999
    n_trades = len(all_trades)
    tpd = n_trades / (252 * 10.5)
    n_chain = sum(1 for t in all_trades if t[2] == "chain")
    n_trend = sum(1 for t in all_trades if t[2] == "trend")

    print(f"  Chain trades:    {n_chain}")
    print(f"  Trend trades:    {n_trend} (at 0.5x R)")
    print(f"  Total:           {n_trades}  ({tpd:.2f}/day)")
    print(f"  Combined R:      {total_r:+.1f}")
    print(f"  Combined PF:     {pf:.2f}")
    print(f"  Combined PPDD:   {ppdd:.2f}")
    print(f"  Combined MaxDD:  {dd:.1f}R")
    print(f"\n  vs Chain alone:  {m_chain['trades']}t  R={m_chain['R']:+.1f}  PF={m_chain['PF']:.2f}  PPDD={m_chain['PPDD']:.2f}")


if __name__ == "__main__":
    main()
