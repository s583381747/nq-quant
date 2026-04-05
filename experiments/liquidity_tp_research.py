"""
experiments/liquidity_tp_research.py -- Liquidity-Based TP Research
===================================================================

ICT: Price moves from one liquidity pool to the next.
Identify the NEXT liquidity target after entry, set TP there.

Liquidity targets (for longs):
  - Nearest swing high (internal liquidity)
  - 2nd/3rd swing high (stacked liquidity)
  - Overnight high, session highs
  - PDH (previous day high)
  - Equal highs (clustered swing points = stop concentration)

For each chain trade, identify all targets, see which ones price reaches,
find optimal target for maximizing R.
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
from features.swing import detect_swing_highs, detect_swing_lows
from dataclasses import dataclass


def find_liquidity_targets_long(
    i: int, entry_price: float, stop_dist: float,
    h: np.ndarray, l: np.ndarray,
    swing_high_prices: np.ndarray,  # array of all swing high prices (NaN where no swing)
    swing_high_shifted: np.ndarray,  # boolean mask (shifted, no lookahead)
    pdh: np.ndarray,
    on_hi: np.ndarray,
    asia_hi: np.ndarray,
    london_hi: np.ndarray,
    atr: float,
) -> list[dict]:
    """Find all liquidity targets ABOVE entry for a long trade.

    Returns list of targets sorted by distance from entry.
    """
    targets = []

    # Swing highs above entry (walk back to find recent ones)
    seen_prices = set()
    count = 0
    for j in range(i - 1, max(0, i - 500), -1):
        if swing_high_shifted[j]:
            price = swing_high_prices[j]
            if np.isnan(price) or price <= 0:
                continue
            if price > entry_price + 1.0:  # at least 1pt above entry
                rounded = round(price, 0)  # group nearby
                if rounded not in seen_prices:
                    seen_prices.add(rounded)
                    count += 1
                    targets.append({
                        "type": f"swing_hi_{count}",
                        "price": price,
                        "dist_pts": price - entry_price,
                        "dist_r": (price - entry_price) / stop_dist if stop_dist > 0 else 0,
                    })
                    if count >= 5:
                        break

    # Session levels
    for name, arr in [("pdh", pdh), ("on_hi", on_hi), ("asia_hi", asia_hi), ("london_hi", london_hi)]:
        val = arr[i]
        if not np.isnan(val) and val > entry_price + 1.0:
            targets.append({
                "type": name,
                "price": val,
                "dist_pts": val - entry_price,
                "dist_r": (val - entry_price) / stop_dist if stop_dist > 0 else 0,
            })

    # Equal highs: find 2+ swing highs within 0.1*ATR of each other
    sh_above = [t for t in targets if t["type"].startswith("swing_hi")]
    for ti in range(len(sh_above)):
        for tj in range(ti + 1, len(sh_above)):
            if abs(sh_above[ti]["price"] - sh_above[tj]["price"]) < atr * 0.15:
                eq_price = max(sh_above[ti]["price"], sh_above[tj]["price"])
                targets.append({
                    "type": "equal_hi",
                    "price": eq_price,
                    "dist_pts": eq_price - entry_price,
                    "dist_r": (eq_price - entry_price) / stop_dist if stop_dist > 0 else 0,
                })
                break

    # ATR-based targets
    for mult in [0.5, 1.0, 1.5, 2.0, 3.0]:
        targets.append({
            "type": f"atr_{mult}",
            "price": entry_price + atr * mult,
            "dist_pts": atr * mult,
            "dist_r": (atr * mult) / stop_dist if stop_dist > 0 else 0,
        })

    # Fixed R targets
    for rr in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        targets.append({
            "type": f"fixed_{rr}R",
            "price": entry_price + stop_dist * rr,
            "dist_pts": stop_dist * rr,
            "dist_r": rr,
        })

    targets.sort(key=lambda x: x["dist_pts"])
    return targets


def find_liquidity_targets_short(
    i: int, entry_price: float, stop_dist: float,
    h: np.ndarray, l: np.ndarray,
    swing_low_prices: np.ndarray,
    swing_low_shifted: np.ndarray,
    pdl: np.ndarray,
    on_lo: np.ndarray,
    asia_lo: np.ndarray,
    london_lo: np.ndarray,
    atr: float,
) -> list[dict]:
    """Find all liquidity targets BELOW entry for a short trade."""
    targets = []

    seen_prices = set()
    count = 0
    for j in range(i - 1, max(0, i - 500), -1):
        if swing_low_shifted[j]:
            price = swing_low_prices[j]
            if np.isnan(price) or price <= 0:
                continue
            if price < entry_price - 1.0:
                rounded = round(price, 0)
                if rounded not in seen_prices:
                    seen_prices.add(rounded)
                    count += 1
                    targets.append({
                        "type": f"swing_lo_{count}",
                        "price": price,
                        "dist_pts": entry_price - price,
                        "dist_r": (entry_price - price) / stop_dist if stop_dist > 0 else 0,
                    })
                    if count >= 5:
                        break

    for name, arr in [("pdl", pdl), ("on_lo", on_lo), ("asia_lo", asia_lo), ("london_lo", london_lo)]:
        val = arr[i]
        if not np.isnan(val) and val < entry_price - 1.0:
            targets.append({
                "type": name,
                "price": val,
                "dist_pts": entry_price - val,
                "dist_r": (entry_price - val) / stop_dist if stop_dist > 0 else 0,
            })

    sh_below = [t for t in targets if t["type"].startswith("swing_lo")]
    for ti in range(len(sh_below)):
        for tj in range(ti + 1, len(sh_below)):
            if abs(sh_below[ti]["price"] - sh_below[tj]["price"]) < atr * 0.15:
                eq_price = min(sh_below[ti]["price"], sh_below[tj]["price"])
                targets.append({
                    "type": "equal_lo",
                    "price": eq_price,
                    "dist_pts": entry_price - eq_price,
                    "dist_r": (entry_price - eq_price) / stop_dist if stop_dist > 0 else 0,
                })
                break

    for mult in [0.5, 1.0, 1.5, 2.0, 3.0]:
        targets.append({
            "type": f"atr_{mult}",
            "price": entry_price - atr * mult,
            "dist_pts": atr * mult,
            "dist_r": (atr * mult) / stop_dist if stop_dist > 0 else 0,
        })

    for rr in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        targets.append({
            "type": f"fixed_{rr}R",
            "price": entry_price - stop_dist * rr,
            "dist_pts": stop_dist * rr,
            "dist_r": rr,
        })

    targets.sort(key=lambda x: x["dist_pts"])
    return targets


def main():
    print("=" * 120)
    print("LIQUIDITY-BASED TP RESEARCH -- Chain Strategy")
    print("=" * 120)

    d = load_all()
    nq, n = d["nq"], d["n"]
    h, l, c, o = d["h"], d["l"], d["c"], d["o"]
    atr_arr = d["atr_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    et_frac_arr = d["et_frac_arr"]
    dates = d["dates"]
    news_blackout_arr = d["news_blackout_arr"]
    fvg_df = d["fvg_df"]
    params = d["params"]

    session_cache = pd.read_parquet(PROJECT / "data" / "cache_session_levels_10yr_v2.parquet")
    pdhl = compute_pdhl(nq)

    # Swing arrays (with proper shift)
    rb = params["swing"]["right_bars"]
    from features.swing import compute_swing_levels
    swings = compute_swing_levels(nq, {"left_bars": params["swing"]["left_bars"], "right_bars": rb})

    swing_hi_shifted = swings["swing_high"].shift(rb, fill_value=False).values
    swing_lo_shifted = swings["swing_low"].shift(rb, fill_value=False).values

    # Build price arrays at shifted positions
    raw_sh = swings["swing_high"].values
    raw_sl = swings["swing_low"].values
    sh_prices = np.full(n, np.nan)
    sl_prices = np.full(n, np.nan)
    for j in range(n):
        if raw_sh[j] and j + rb < n:
            sh_prices[j + rb] = h[j]
        if raw_sl[j] and j + rb < n:
            sl_prices[j + rb] = l[j]

    # Breakdowns
    on_lo_bds = detect_breakdowns(h, l, c, session_cache["overnight_low"].values, "low", min_depth_pts=1.0)
    for bd in on_lo_bds: bd["level_type"] = "low"
    on_hi_bds = detect_breakdowns(h, l, c, session_cache["overnight_high"].values, "high", min_depth_pts=1.0)
    for bd in on_hi_bds: bd["level_type"] = "high"
    all_bds = on_lo_bds + on_hi_bds

    # Pre-compute zones
    fvg_bull_mask = fvg_df["fvg_bull"].values
    fvg_bull_top = fvg_df["fvg_bull_top"].values
    fvg_bull_bottom = fvg_df["fvg_bull_bottom"].values
    fvg_bear_mask = fvg_df["fvg_bear"].values
    fvg_bear_top = fvg_df["fvg_bear_top"].values
    fvg_bear_bottom = fvg_df["fvg_bear_bottom"].values
    slippage = params["backtest"]["slippage_normal_ticks"] * 0.25
    stop_buffer_pct = 0.15
    tighten_factor = 0.85
    min_stop_pts = 5.0
    max_fvg_age = 200

    @dataclass
    class Zone:
        direction: int
        top: float
        bottom: float
        size: float
        birth_bar: int
        used: bool = False

    zones_by_bar = {}
    for bd in all_bds:
        fvg = find_fvg_after_breakdown(
            fvg_bull_mask, fvg_bull_top, fvg_bull_bottom,
            fvg_bear_mask, fvg_bear_top, fvg_bear_bottom,
            atr_arr, bd["bar_idx"], bd["level_type"], 30,
            0.0, 99.0, 0.3, h, l, c, o)
        if fvg is None:
            continue
        bar = fvg["fvg_bar"]
        zones_by_bar.setdefault(bar, []).append(
            Zone(fvg["direction"], fvg["fvg_top"], fvg["fvg_bottom"], fvg["fvg_size"], bar))

    # ================================================================
    # Run entry logic + capture MFE with target analysis
    # ================================================================
    print("\n[RUN] Analyzing liquidity targets for each entry...")

    active_zones = []
    target_results = []  # per-target: did price reach it?
    trade_results = []  # per-trade: MFE + all targets
    in_pos = False
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

        # Skip exit management for simplicity - we track MFE forward from entry
        if in_pos:
            continue

        if day_stopped:
            continue
        if news_blackout_arr is not None and news_blackout_arr[i]:
            continue
        ef = et_frac_arr[i]
        if 9.5 <= ef <= 10.0 or not (10.0 <= ef < 16.0):
            continue

        best = None
        best_z = None

        for z in active_zones:
            if z.used or z.birth_bar >= i:
                continue
            ep = z.top if z.direction == 1 else z.bottom
            sp_raw = (z.bottom - z.size * stop_buffer_pct) if z.direction == 1 else (z.top + z.size * stop_buffer_pct)
            sd = abs(ep - sp_raw)
            if tighten_factor < 1.0:
                sp = ep - sd * tighten_factor if z.direction == 1 else ep + sd * tighten_factor
                sd = abs(ep - sp)
            else:
                sp = sp_raw
            if sd < min_stop_pts or sd < 1.0:
                continue

            # Fill check
            if z.direction == 1 and l[i] > ep:
                continue
            if z.direction == -1 and h[i] < ep:
                continue

            # Same-bar stop check
            if z.direction == 1 and l[i] <= sp:
                continue  # skip SBS for this analysis
            if z.direction == -1 and h[i] >= sp:
                continue

            if z.direction == 1 and bias_dir_arr[i] < 0:
                continue
            if z.direction == -1 and bias_dir_arr[i] > 0:
                continue

            fq = -abs(c[i] - ep)
            if best is None or fq > best:
                best = fq
                best_z = (z, ep, sp, sd)

        if best_z is None:
            continue

        z, ep, sp, sd = best_z
        z.used = True
        cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0

        # Find all liquidity targets
        if z.direction == 1:
            targets = find_liquidity_targets_long(
                i, ep, sd, h, l, sh_prices, swing_hi_shifted,
                pdhl["pdh"].values, session_cache["overnight_high"].values,
                session_cache["asia_high"].values, session_cache["london_high"].values, cur_atr)
        else:
            targets = find_liquidity_targets_short(
                i, ep, sd, h, l, sl_prices, swing_lo_shifted,
                pdhl["pdl"].values, session_cache["overnight_low"].values,
                session_cache["asia_low"].values, session_cache["london_low"].values, cur_atr)

        # Forward-scan: did price reach each target? (up to EOD)
        mfe = 0.0
        for j in range(i + 1, min(n, i + 500)):
            if dates[j] != dates[i] and et_frac_arr[j] < 10.0:
                break  # stop at next day
            if et_frac_arr[j] >= 15.917:
                # EOD: check one last time
                if z.direction == 1:
                    fe = h[j] - ep
                else:
                    fe = ep - l[j]
                mfe = max(mfe, fe)
                break

            # Stop hit = stop scanning
            if z.direction == 1 and l[j] <= sp:
                break
            if z.direction == -1 and h[j] >= sp:
                break

            if z.direction == 1:
                fe = h[j] - ep
            else:
                fe = ep - l[j]
            mfe = max(mfe, fe)

        # Record which targets were reached
        for t in targets:
            reached = mfe >= t["dist_pts"]
            target_results.append({
                "dir": z.direction,
                "target_type": t["type"],
                "target_price": t["price"],
                "dist_pts": t["dist_pts"],
                "dist_r": t["dist_r"],
                "reached": reached,
                "mfe_pts": mfe,
                "stop_dist": sd,
                "atr": cur_atr,
            })

        trade_results.append({
            "dir": z.direction,
            "entry_price": ep,
            "stop_dist": sd,
            "mfe_pts": mfe,
            "mfe_r": mfe / sd if sd > 0 else 0,
            "atr": cur_atr,
            "n_targets": len(targets),
        })

        in_pos = True  # block next entry until simple reset
        # Simple: just set in_pos=False after this bar (one trade per scan)
        in_pos = False

    print(f"[RUN] {len(trade_results)} entries, {len(target_results)} target evaluations")

    df_targets = pd.DataFrame(target_results)
    df_trades = pd.DataFrame(trade_results)

    # ================================================================
    # ANALYSIS: Target reach rates
    # ================================================================
    print(f"\n{'='*120}")
    print("TARGET REACH RATES -- What % of trades reach each target?")
    print(f"{'='*120}")

    for dir_tag, dir_val in [("ALL", None), ("LONGS", 1), ("SHORTS", -1)]:
        sub = df_targets if dir_val is None else df_targets[df_targets["dir"] == dir_val]
        if len(sub) == 0:
            continue

        print(f"\n  --- {dir_tag} ---")
        print(f"  {'Target':20s} | {'Reach%':>7s} | {'avgDist_R':>10s} | {'medDist_R':>10s} | {'Count':>6s}")
        print(f"  {'-'*20}-+-{'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")

        for ttype in sorted(sub["target_type"].unique()):
            t_sub = sub[sub["target_type"] == ttype]
            reach_pct = t_sub["reached"].mean() * 100
            avg_dist_r = t_sub["dist_r"].mean()
            med_dist_r = t_sub["dist_r"].median()
            count = len(t_sub)
            print(f"  {ttype:20s} | {reach_pct:6.1f}% | {avg_dist_r:10.2f} | {med_dist_r:10.2f} | {count:6d}")

    # ================================================================
    # ANALYSIS: Optimal trim point (which target to trim at)
    # ================================================================
    print(f"\n{'='*120}")
    print("OPTIMAL TRIM: If we trim 25% at target, run rest to stop/EOD")
    print("Simulating R for each target type as trim point")
    print(f"{'='*120}")

    # Group by target type, compute simulated R
    liquidity_types = ["swing_hi_1", "swing_hi_2", "swing_hi_3",
                       "pdh", "on_hi", "asia_hi", "london_hi", "equal_hi",
                       "swing_lo_1", "swing_lo_2", "swing_lo_3",
                       "pdl", "on_lo", "asia_lo", "london_lo", "equal_lo"]
    fixed_types = [f"fixed_{r}R" for r in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]]
    atr_types = [f"atr_{m}" for m in [0.5, 1.0, 1.5, 2.0, 3.0]]

    all_types = liquidity_types + fixed_types + atr_types

    print(f"\n  {'Target':20s} | {'Reach%':>7s} | {'If trim 25%':>12s} | {'R if reached':>12s} | {'N':>5s}")
    print(f"  {'-'*20}-+-{'-'*7}-+-{'-'*12}-+-{'-'*12}-+-{'-'*5}")

    for ttype in all_types:
        sub = df_targets[df_targets["target_type"] == ttype]
        if len(sub) < 30:
            continue

        reach_pct = sub["reached"].mean() * 100
        avg_dist_r = sub["dist_r"].mean()

        # Simulate: trim 25% at this target (if reached), runner gets BE or actual MFE
        sim_r_vals = []
        for _, row in sub.iterrows():
            sd = row["stop_dist"]
            if sd <= 0:
                continue
            mfe_r = row["mfe_pts"] / sd

            if row["reached"]:
                # Trim 25% at target
                trim_r = row["dist_r"] * 0.25
                # Runner: min(mfe, actual exit which we approximate as mfe-based)
                # With BE after trim, runner can't lose more than 0
                runner_r = max(0, mfe_r) * 0.75
                sim_r_vals.append(trim_r + runner_r)
            else:
                # Target not reached: full position exits (stopped or EOD)
                # Approximate: if MFE < target, we hit stop → -1R
                # (simplified: actual exit is complex)
                if mfe_r < 0.3:
                    sim_r_vals.append(-1.0)
                else:
                    sim_r_vals.append(mfe_r * 0.5)  # rough approximation

        if len(sim_r_vals) < 30:
            continue
        sim_r = np.array(sim_r_vals)
        total_r = sim_r.sum()
        avg_r = sim_r.mean()

        # R if reached only
        reached_r = sub[sub["reached"]]["dist_r"].mean() if sub["reached"].sum() > 0 else 0

        print(f"  {ttype:20s} | {reach_pct:6.1f}% | {total_r:+11.1f}R | {reached_r:11.2f}R | {len(sub):5d}")

    # ================================================================
    # BEST LIQUIDITY TARGET per trade
    # ================================================================
    print(f"\n{'='*120}")
    print("NEAREST REACHABLE LIQUIDITY TARGET (not fixed R)")
    print("For each trade: what's the nearest real liquidity level that price reached?")
    print(f"{'='*120}")

    liq_only = df_targets[df_targets["target_type"].isin(liquidity_types)]
    reached_liq = liq_only[liq_only["reached"] == True]

    if len(reached_liq) > 0:
        print(f"\n  Reached liquidity targets: {len(reached_liq)} / {len(liq_only)} ({len(reached_liq)/len(liq_only)*100:.1f}%)")
        print(f"\n  Distance to reached targets (R multiples):")
        print(f"    Mean:   {reached_liq['dist_r'].mean():.2f}R")
        print(f"    Median: {reached_liq['dist_r'].median():.2f}R")
        print(f"    P25:    {reached_liq['dist_r'].quantile(0.25):.2f}R")
        print(f"    P75:    {reached_liq['dist_r'].quantile(0.75):.2f}R")

        print(f"\n  By target type (reached only):")
        for ttype in liquidity_types:
            sub = reached_liq[reached_liq["target_type"] == ttype]
            if len(sub) < 10:
                continue
            print(f"    {ttype:20s}: {len(sub):4d} reached | avg dist = {sub['dist_r'].mean():.2f}R | median = {sub['dist_r'].median():.2f}R")

    print(f"\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")
    print("Compare fixed R targets vs liquidity-based targets above.")
    print("The target type with highest total simulated R = optimal TP.")


if __name__ == "__main__":
    main()
