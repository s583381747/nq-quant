"""
experiments/trim_runner_research.py -- Trim 1R + Runner to Swing Targets
========================================================================

User's idea: trim 50% at 1R, run the rest to swing hi/lo targets.
Research: how often does the runner reach swing_1, swing_2, etc.?
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
from experiments.u2_clean import load_all, compute_metrics, pr
from features.swing import compute_swing_levels
from dataclasses import dataclass


def main():
    d = load_all()
    nq, n = d["nq"], d["n"]
    h, l, c, o = d["h"], d["l"], d["c"], d["o"]
    atr_arr = d["atr_arr"]
    params = d["params"]
    session_cache = pd.read_parquet(PROJECT / "data" / "cache_session_levels_10yr_v2.parquet")
    pdhl = compute_pdhl(nq)

    rb = params["swing"]["right_bars"]
    swings = compute_swing_levels(nq, {"left_bars": params["swing"]["left_bars"], "right_bars": rb})
    swing_hi_shifted = swings["swing_high"].shift(rb, fill_value=False).values
    swing_lo_shifted = swings["swing_low"].shift(rb, fill_value=False).values
    raw_sh, raw_sl = swings["swing_high"].values, swings["swing_low"].values
    sh_prices = np.full(n, np.nan)
    sl_prices = np.full(n, np.nan)
    for j in range(n):
        if raw_sh[j] and j + rb < n: sh_prices[j + rb] = h[j]
        if raw_sl[j] and j + rb < n: sl_prices[j + rb] = l[j]

    on_lo_bds = detect_breakdowns(h, l, c, session_cache["overnight_low"].values, "low", min_depth_pts=1.0)
    for bd in on_lo_bds: bd["level_type"] = "low"
    on_hi_bds = detect_breakdowns(h, l, c, session_cache["overnight_high"].values, "high", min_depth_pts=1.0)
    for bd in on_hi_bds: bd["level_type"] = "high"
    all_bds = on_lo_bds + on_hi_bds

    fvg_df = d["fvg_df"]
    fvg_bm, fvg_bt, fvg_bb = fvg_df["fvg_bull"].values, fvg_df["fvg_bull_top"].values, fvg_df["fvg_bull_bottom"].values
    fvg_em, fvg_et, fvg_eb = fvg_df["fvg_bear"].values, fvg_df["fvg_bear_top"].values, fvg_df["fvg_bear_bottom"].values

    @dataclass
    class Z:
        direction: int
        top: float
        bottom: float
        size: float
        birth_bar: int
        used: bool = False

    zones_by_bar = {}
    for bd in all_bds:
        fvg = find_fvg_after_breakdown(fvg_bm, fvg_bt, fvg_bb, fvg_em, fvg_et, fvg_eb,
            atr_arr, bd["bar_idx"], bd["level_type"], 30, 0.0, 99.0, 0.3, h, l, c, o)
        if fvg is None: continue
        bar = fvg["fvg_bar"]
        zones_by_bar.setdefault(bar, []).append(
            Z(fvg["direction"], fvg["fvg_top"], fvg["fvg_bottom"], fvg["fvg_size"], bar))

    dates = d["dates"]
    et_frac_arr = d["et_frac_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    bias_dir_arr = d["bias_dir_arr"]

    active_zones = []
    results = []
    cur_date = None
    day_stopped = False

    for i in range(n):
        if dates[i] != cur_date:
            cur_date = dates[i]
            day_stopped = False
        if i in zones_by_bar:
            active_zones.extend(zones_by_bar[i])

        surviving = []
        for z in active_zones:
            if z.used or (i - z.birth_bar) > 200: continue
            if z.direction == 1 and c[i] < z.bottom: continue
            if z.direction == -1 and c[i] > z.top: continue
            surviving.append(z)
        active_zones = surviving[-30:] if len(surviving) > 30 else surviving

        if day_stopped: continue
        if news_blackout_arr is not None and news_blackout_arr[i]: continue
        ef = et_frac_arr[i]
        if 9.5 <= ef <= 10.0 or not (10.0 <= ef < 16.0): continue

        best = None
        best_z = None
        for z in active_zones:
            if z.used or z.birth_bar >= i: continue
            ep = z.top if z.direction == 1 else z.bottom
            sp_raw = (z.bottom - z.size * 0.15) if z.direction == 1 else (z.top + z.size * 0.15)
            sd = abs(ep - sp_raw)
            sp = ep - sd * 0.85 if z.direction == 1 else ep + sd * 0.85
            sd = abs(ep - sp)
            if sd < 5.0: continue
            if z.direction == 1 and l[i] > ep: continue
            if z.direction == -1 and h[i] < ep: continue
            if z.direction == 1 and l[i] <= sp: continue  # skip SBS
            if z.direction == -1 and h[i] >= sp: continue
            if z.direction == 1 and bias_dir_arr[i] < 0: continue
            if z.direction == -1 and bias_dir_arr[i] > 0: continue
            fq = -abs(c[i] - ep)
            if best is None or fq > best:
                best = fq
                best_z = (z, ep, sp, sd)
        if best_z is None: continue
        z, ep, sp, sd = best_z
        z.used = True
        cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0

        # Find swing targets
        targets = []
        if z.direction == 1:
            cnt = 0
            for j in range(i - 1, max(0, i - 500), -1):
                if swing_hi_shifted[j]:
                    p = sh_prices[j]
                    if np.isnan(p) or p <= ep + 1: continue
                    cnt += 1
                    targets.append(p)
                    if cnt >= 5: break
        else:
            cnt = 0
            for j in range(i - 1, max(0, i - 500), -1):
                if swing_lo_shifted[j]:
                    p = sl_prices[j]
                    if np.isnan(p) or p >= ep - 1: continue
                    cnt += 1
                    targets.append(p)
                    if cnt >= 5: break

        # Forward scan
        trim_price = ep + sd if z.direction == 1 else ep - sd
        trimmed = False
        mfe = 0.0
        runner_mfe = 0.0
        targets_reached = [False] * min(len(targets), 5)
        exit_reason = "max_bars"
        eod_price = 0.0

        for j in range(i + 1, min(n, i + 500)):
            if dates[j] != dates[i] and et_frac_arr[j] < 10.0:
                exit_reason = "next_day"
                break
            if et_frac_arr[j] >= 15.917:
                exit_reason = "eod"
                fe = (h[j] - ep) if z.direction == 1 else (ep - l[j])
                mfe = max(mfe, fe)
                eod_price = c[j]
                if trimmed:
                    runner_mfe = max(runner_mfe, fe)
                for ti in range(len(targets_reached)):
                    if z.direction == 1 and h[j] >= targets[ti]:
                        targets_reached[ti] = True
                    elif z.direction == -1 and l[j] <= targets[ti]:
                        targets_reached[ti] = True
                break

            # Stop
            if not trimmed:
                if z.direction == 1 and l[j] <= sp:
                    exit_reason = "stop"
                    break
                if z.direction == -1 and h[j] >= sp:
                    exit_reason = "stop"
                    break

            fe = (h[j] - ep) if z.direction == 1 else (ep - l[j])
            mfe = max(mfe, fe)

            if not trimmed:
                if z.direction == 1 and h[j] >= trim_price:
                    trimmed = True
                elif z.direction == -1 and l[j] <= trim_price:
                    trimmed = True

            if trimmed:
                runner_mfe = max(runner_mfe, fe)
                # BE stop for runner
                if z.direction == 1 and l[j] <= ep:
                    exit_reason = "be_sweep"
                    eod_price = ep
                    break
                if z.direction == -1 and h[j] >= ep:
                    exit_reason = "be_sweep"
                    eod_price = ep
                    break

            for ti in range(len(targets_reached)):
                if z.direction == 1 and h[j] >= targets[ti]:
                    targets_reached[ti] = True
                elif z.direction == -1 and l[j] <= targets[ti]:
                    targets_reached[ti] = True

        rec = {
            "dir": z.direction, "entry": ep, "stop": sp, "sd": sd, "atr": cur_atr,
            "mfe_r": mfe / sd if sd > 0 else 0,
            "trimmed": trimmed,
            "runner_mfe_r": runner_mfe / sd if sd > 0 else 0,
            "exit_reason": exit_reason,
        }
        for ti in range(5):
            rec[f"tgt_{ti+1}_reached"] = targets_reached[ti] if ti < len(targets_reached) else False
            rec[f"tgt_{ti+1}_dist_r"] = (abs(targets[ti] - ep) / sd) if ti < len(targets) and sd > 0 else 0
        results.append(rec)

    df = pd.DataFrame(results)

    print("=" * 100)
    print("TRIM 1R + RUNNER TO SWING TARGETS")
    print("=" * 100)

    print(f"\nTotal entries: {len(df)}")
    print(f"  Longs:  {(df['dir']==1).sum()}")
    print(f"  Shorts: {(df['dir']==-1).sum()}")

    # Trim rate
    print(f"\nTrim at 1R hit rate:")
    for tag, sub in [("All", df), ("Longs", df[df["dir"]==1]), ("Shorts", df[df["dir"]==-1])]:
        if len(sub) < 10: continue
        print(f"  {tag:8s}: {sub['trimmed'].mean()*100:.1f}% ({sub['trimmed'].sum()}/{len(sub)})")

    # Runner MFE after trim
    trimmed = df[df["trimmed"]]
    print(f"\nRunner MFE after 1R trim ({len(trimmed)} trimmed trades):")
    for tag, sub in [("All", trimmed), ("Longs", trimmed[trimmed["dir"]==1]), ("Shorts", trimmed[trimmed["dir"]==-1])]:
        if len(sub) < 10: continue
        r = sub["runner_mfe_r"]
        print(f"  {tag:8s}: mean={r.mean():.2f}R  med={r.median():.2f}R  p75={r.quantile(0.75):.2f}R  p90={r.quantile(0.9):.2f}R")

    # Swing targets reached
    print(f"\nSwing target reach rates:")
    for tag, sub in [("All", df), ("Longs", df[df["dir"]==1]), ("Shorts", df[df["dir"]==-1])]:
        if len(sub) < 10: continue
        parts = []
        for ti in range(1, 4):
            reached_pct = sub[f"tgt_{ti}_reached"].mean() * 100
            has_target = (sub[f"tgt_{ti}_dist_r"] > 0)
            avg_dist = sub.loc[has_target, f"tgt_{ti}_dist_r"].median() if has_target.sum() > 0 else 0
            parts.append(f"sw{ti}={reached_pct:.0f}% ({avg_dist:.1f}R)")
        print(f"  {tag:8s}: {' | '.join(parts)}")

    # After trim: where does runner end up?
    print(f"\nRunner exit reasons (after trim):")
    for reason, grp in trimmed.groupby("exit_reason"):
        avg_r = grp["runner_mfe_r"].mean()
        print(f"  {reason:15s}: {len(grp):4d} ({len(grp)/len(trimmed)*100:.0f}%)  runner MFE={avg_r:.2f}R")

    # Simulated strategies
    print(f"\n{'='*100}")
    print("SIMULATED R for different trim/runner combos")
    print(f"{'='*100}")

    def sim(res, trim_r, trim_pct, runner_target_idx):
        """Simulate trim+runner. runner_target_idx: 0=BE, 1=swing_1, 2=swing_2, -1=trail(approx MFE*0.5)"""
        trades = []
        for r in res:
            sd = r["sd"]
            if sd <= 0: continue

            if r["exit_reason"] == "stop" and not r["trimmed"]:
                trades.append({"r": -1.0, "dir": r["dir"]})
                continue

            if not r["trimmed"]:
                # Didn't reach trim -> exited at stop or poor EOD
                if r["exit_reason"] in ("stop", "next_day"):
                    trades.append({"r": -1.0, "dir": r["dir"]})
                else:
                    trades.append({"r": r["mfe_r"] * 0.3, "dir": r["dir"]})
                continue

            # Trimmed
            trim_profit = trim_r * trim_pct

            if runner_target_idx == 0:
                runner_profit = 0  # BE
            elif runner_target_idx == -1:
                # Trail approximation: capture ~50% of runner MFE
                runner_profit = r["runner_mfe_r"] * (1 - trim_pct) * 0.5
            else:
                tgt_key = f"tgt_{runner_target_idx}_reached"
                dist_key = f"tgt_{runner_target_idx}_dist_r"
                if r.get(tgt_key, False) and r.get(dist_key, 0) > 0:
                    runner_profit = r[dist_key] * (1 - trim_pct)
                else:
                    # Target not reached -> BE or trail
                    if r["exit_reason"] == "be_sweep":
                        runner_profit = 0
                    else:
                        runner_profit = r["runner_mfe_r"] * (1 - trim_pct) * 0.3

            trades.append({"r": trim_profit + runner_profit, "dir": r["dir"]})
        return trades

    combos = [
        ("Trim 1R 50%, BE runner", 1.0, 0.50, 0),
        ("Trim 1R 50%, swing_1 runner", 1.0, 0.50, 1),
        ("Trim 1R 50%, swing_2 runner", 1.0, 0.50, 2),
        ("Trim 1R 50%, trail runner", 1.0, 0.50, -1),
        ("Trim 1R 25%, swing_1 runner", 1.0, 0.25, 1),
        ("Trim 1R 25%, trail runner", 1.0, 0.25, -1),
        ("Trim 0.5R 50%, swing_1 runner", 0.5, 0.50, 1),
        ("Trim 1.5R 50%, swing_1 runner", 1.5, 0.50, 1),
    ]

    print(f"\n  {'Strategy':45s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'PPDD':>7s} | {'avgR':>8s}")
    print(f"  {'-'*45}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*8}")
    for label, tr, tp, ri in combos:
        trades = sim(results, tr, tp, ri)
        m = compute_metrics(trades)
        print(f"  {label:45s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['PPDD']:6.2f} | {m['avgR']:+7.4f}")

    # Long/Short split for best combo
    print(f"\n  --- Best combo split ---")
    for tag, dir_val in [("LONGS", 1), ("SHORTS", -1)]:
        sub_res = [r for r in results if r["dir"] == dir_val]
        trades = sim(sub_res, 1.0, 0.50, 1)
        m = compute_metrics(trades)
        print(f"  Trim 1R 50% + swing_1  {tag:8s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['PPDD']:6.2f}")

        trades2 = sim(sub_res, 1.0, 0.50, -1)
        m2 = compute_metrics(trades2)
        print(f"  Trim 1R 50% + trail    {tag:8s} | {m2['trades']:5d} | {m2['R']:+8.1f} | {m2['PF']:5.2f} | {m2['PPDD']:6.2f}")


if __name__ == "__main__":
    main()
