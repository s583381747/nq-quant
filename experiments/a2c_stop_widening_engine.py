"""
experiments/a2c_stop_widening_engine.py — Stop Widening/Tightening via Full Engine
==================================================================================

Tests whether widening or tightening the model stop improves performance when run
through the FULL multi-TP backtest engine (Config H: raw IRL TP, 50/50/0, EOD close).

Previous signal-DB analysis (a2c_dynamic_stop.py) showed stop widening by 10-20%
could improve R by 27-41%, but that was a simplified calculation without the
trim/trail/EOD close mechanics. This experiment validates that finding.

Methodology:
  - For each stop_factor in [0.80, 0.90, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.50]:
    1. Modify model_stop_arr: push stop further away by factor
       - LONG: new_stop = entry - (entry - stop) * factor  (further below)
       - SHORT: new_stop = entry + (stop - entry) * factor  (further above)
    2. Run full backtest with Config H settings
    3. Wider stop → fewer contracts (same R budget / wider distance)
    4. Wider stop → fewer stop-outs but smaller position per trade

  - Walk-forward validation for top 2-3 factors
  - Compare with signal-DB predictions

Usage: python experiments/a2c_stop_widening_engine.py
"""
from __future__ import annotations

import copy
import sys
import time as _time
from pathlib import Path

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
    compute_metrics,
    walk_forward_metrics,
    print_metrics,
)
from experiments.multi_level_tp import prepare_liquidity_data
from experiments.pure_liquidity_tp import (
    run_backtest_pure_liquidity,
    print_diagnostics,
    print_long_short_breakdown,
)


# ======================================================================
# Stop widening utility
# ======================================================================
def widen_stop_array(
    model_stop_arr: np.ndarray,
    entry_price_arr: np.ndarray,
    sig_dir: np.ndarray,
    sig_mask: np.ndarray,
    factor: float,
) -> np.ndarray:
    """
    Create a modified stop array where each signal's stop is widened by `factor`.

    factor > 1.0 = wider stop (further from entry)
    factor < 1.0 = tighter stop (closer to entry)
    factor = 1.0 = no change

    For LONG: new_stop = entry - (entry - old_stop) * factor
    For SHORT: new_stop = entry + (old_stop - entry) * factor
    """
    new_stop = model_stop_arr.copy()
    signal_indices = np.where(sig_mask)[0]

    for idx in signal_indices:
        entry = entry_price_arr[idx]
        stop = model_stop_arr[idx]
        direction = sig_dir[idx]

        if np.isnan(entry) or np.isnan(stop) or direction == 0:
            continue

        stop_dist = abs(entry - stop)

        if direction == 1:  # LONG: stop is below entry
            new_stop[idx] = entry - stop_dist * factor
        elif direction == -1:  # SHORT: stop is above entry
            new_stop[idx] = entry + stop_dist * factor

    return new_stop


def count_saved_trades(
    baseline_trades: list[dict],
    test_trades: list[dict],
) -> dict:
    """
    Compare baseline vs test trades to count:
    - saved: trades that were stopped out in baseline but survived in test
    - killed: trades that survived in baseline but stopped out in test
    """
    # Build lookup by entry_time
    bl_by_time = {}
    for t in baseline_trades:
        key = str(t["entry_time"])
        bl_by_time[key] = t

    test_by_time = {}
    for t in test_trades:
        key = str(t["entry_time"])
        test_by_time[key] = t

    saved = 0
    killed = 0
    saved_net_r = 0.0
    killed_net_r = 0.0

    # Trades that exist in both
    common_keys = set(bl_by_time.keys()) & set(test_by_time.keys())

    for key in common_keys:
        bl = bl_by_time[key]
        tt = test_by_time[key]

        bl_stopped = bl["reason"] == "stop"
        tt_stopped = tt["reason"] == "stop"

        if bl_stopped and not tt_stopped:
            saved += 1
            # Net benefit: test R - baseline R (baseline was -1R-ish)
            saved_net_r += tt["r"] - bl["r"]
        elif not bl_stopped and tt_stopped:
            killed += 1
            killed_net_r += tt["r"] - bl["r"]

    # Trades only in baseline (wider stop may have different filter outcomes)
    only_baseline = len(bl_by_time) - len(common_keys)
    only_test = len(test_by_time) - len(common_keys)

    return {
        "common_trades": len(common_keys),
        "saved": saved,
        "killed": killed,
        "saved_net_r": round(saved_net_r, 2),
        "killed_net_r": round(killed_net_r, 2),
        "only_baseline": only_baseline,
        "only_test": only_test,
    }


def exit_reason_breakdown(trades: list[dict], direction: int = None) -> dict:
    """Count trades by exit reason."""
    reasons = {}
    for t in trades:
        if direction is not None and t["dir"] != direction:
            continue
        r = t["reason"]
        if r not in reasons:
            reasons[r] = {"count": 0, "r_sum": 0.0}
        reasons[r]["count"] += 1
        reasons[r]["r_sum"] += t["r"]
    return reasons


# ======================================================================
# Main experiment
# ======================================================================
def main():
    print("=" * 120)
    print("A2C STOP WIDENING/TIGHTENING — FULL ENGINE TEST")
    print("Config H: Raw IRL TP (V1), 50/50/0, EOD close 15:55 ET, sq_short=0.80")
    print("=" * 120)

    # ------------------------------------------------------------------
    # Step 0: Load data
    # ------------------------------------------------------------------
    d = load_all()
    d_extra = prepare_liquidity_data(d)

    # Store original stop array
    original_model_stop = d["model_stop_arr"].copy()

    # ------------------------------------------------------------------
    # Step 1: Baseline (factor=1.0)
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("BASELINE: Config H (stop_factor=1.00)")
    print("=" * 120)
    t0 = _time.perf_counter()
    baseline_trades, baseline_diag = run_backtest_pure_liquidity(
        d, d_extra,
        tp_strategy="v1",
        sq_short=0.80,
        block_pm_shorts=True,
        tp1_trim_pct=0.50,
        be_after_tp1=True,
        be_after_tp2=True,
    )
    baseline_m = compute_metrics(baseline_trades)
    print_metrics("BASELINE (factor=1.00)", baseline_m)
    print_diagnostics("Baseline", baseline_diag)
    print_long_short_breakdown(baseline_trades, "Baseline")
    elapsed = _time.perf_counter() - t0
    print(f"  ({elapsed:.1f}s)")

    # Compute baseline stop distance stats
    bl_longs = [t for t in baseline_trades if t["dir"] == 1]
    bl_shorts = [t for t in baseline_trades if t["dir"] == -1]
    bl_long_stops = [abs(t["entry_price"] - t["stop_price"]) for t in bl_longs]
    bl_short_stops = [abs(t["entry_price"] - t["stop_price"]) for t in bl_shorts]
    if bl_long_stops:
        print(f"\n  Baseline stop distances (LONGS): "
              f"mean={np.mean(bl_long_stops):.1f}pts, "
              f"median={np.median(bl_long_stops):.1f}pts, "
              f"P25={np.percentile(bl_long_stops, 25):.1f}pts, "
              f"P75={np.percentile(bl_long_stops, 75):.1f}pts")
    if bl_short_stops:
        print(f"  Baseline stop distances (SHORTS): "
              f"mean={np.mean(bl_short_stops):.1f}pts, "
              f"median={np.median(bl_short_stops):.1f}pts")

    # Count baseline stops
    bl_reasons = exit_reason_breakdown(baseline_trades, direction=1)
    bl_stop_count = bl_reasons.get("stop", {}).get("count", 0)
    bl_total_longs = len(bl_longs)
    print(f"  Baseline long stops: {bl_stop_count}/{bl_total_longs} "
          f"({100*bl_stop_count/bl_total_longs:.1f}%)" if bl_total_longs > 0 else "")

    # ------------------------------------------------------------------
    # Step 2: Sweep stop factors
    # ------------------------------------------------------------------
    factors = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.50]
    all_results = {}

    print("\n" + "=" * 120)
    print("STOP FACTOR SWEEP")
    print("=" * 120)
    print(f"\n  {'Factor':>7} | {'Trades':>6} | {'R':>9} | {'PPDD':>7} | {'PF':>6} | "
          f"{'WR':>6} | {'MaxDD':>6} | {'avgR':>8} | {'dR':>7} | {'dPPDD':>7} | "
          f"{'Saved':>5} | {'Killed':>6} | {'LongStops':>10}")
    print("  " + "-" * 115)

    for factor in factors:
        t0 = _time.perf_counter()

        # Modify stop array
        widened_stop = widen_stop_array(
            original_model_stop,
            d["entry_price_arr"],
            d["sig_dir"],
            d["sig_mask"],
            factor,
        )
        d["model_stop_arr"] = widened_stop

        # Run backtest
        trades, diag = run_backtest_pure_liquidity(
            d, d_extra,
            tp_strategy="v1",
            sq_short=0.80,
            block_pm_shorts=True,
            tp1_trim_pct=0.50,
            be_after_tp1=True,
            be_after_tp2=True,
        )
        m = compute_metrics(trades)
        elapsed = _time.perf_counter() - t0

        # Delta from baseline
        delta_r = m["R"] - baseline_m["R"]
        delta_ppdd = m["PPDD"] - baseline_m["PPDD"]

        # Count saves/kills vs baseline
        comparison = count_saved_trades(baseline_trades, trades)

        # Count long stops
        reasons = exit_reason_breakdown(trades, direction=1)
        stop_count = reasons.get("stop", {}).get("count", 0)
        n_longs = len([t for t in trades if t["dir"] == 1])
        stop_pct = 100 * stop_count / n_longs if n_longs > 0 else 0

        marker = " <-- BASELINE" if factor == 1.0 else ""
        print(f"  {factor:>7.2f} | {m['trades']:6d} | {m['R']:+9.1f} | {m['PPDD']:7.2f} | "
              f"{m['PF']:6.2f} | {m['WR']:5.1f}% | {m['MaxDD']:6.1f} | {m['avgR']:+.4f} | "
              f"{delta_r:+7.1f} | {delta_ppdd:+7.2f} | "
              f"{comparison['saved']:5d} | {comparison['killed']:6d} | "
              f"{stop_count:4d}/{n_longs:4d}({stop_pct:4.1f}%)"
              f"{marker}")

        all_results[factor] = {
            "trades": trades,
            "diag": diag,
            "metrics": m,
            "comparison": comparison,
            "elapsed": elapsed,
        }

    # Restore original stop array
    d["model_stop_arr"] = original_model_stop

    # ------------------------------------------------------------------
    # Step 3: Detailed analysis for each factor
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("DETAILED EXIT REASON ANALYSIS (LONGS ONLY)")
    print("=" * 120)

    print(f"\n  {'Factor':>7} | {'stop':>12} | {'tp1':>12} | {'tp2':>12} | "
          f"{'be_sweep':>12} | {'eod_close':>12} | {'early_cut':>12}")
    print("  " + "-" * 90)

    for factor in factors:
        trades = all_results[factor]["trades"]
        reasons = exit_reason_breakdown(trades, direction=1)

        def fmt_reason(name):
            d_r = reasons.get(name, {"count": 0, "r_sum": 0.0})
            return f"{d_r['count']:4d} ({d_r['r_sum']:+.1f}R)"

        print(f"  {factor:>7.2f} | {fmt_reason('stop'):>12} | {fmt_reason('tp1'):>12} | "
              f"{fmt_reason('tp2'):>12} | {fmt_reason('be_sweep'):>12} | "
              f"{fmt_reason('eod_close'):>12} | {fmt_reason('early_cut_pa'):>12}")

    # ------------------------------------------------------------------
    # Step 4: Walk-forward for top 3 by PPDD (excluding baseline)
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("WALK-FORWARD VALIDATION: TOP 3 FACTORS vs BASELINE")
    print("=" * 120)

    # Find top 5 by PPDD (excluding 1.0)
    ranked = sorted(
        [(f, r) for f, r in all_results.items() if f != 1.00],
        key=lambda x: x[1]["metrics"]["PPDD"],
        reverse=True,
    )
    top3 = ranked[:5]

    # Also add the best by R
    best_by_r = max(
        [(f, r) for f, r in all_results.items() if f != 1.00],
        key=lambda x: x[1]["metrics"]["R"],
    )
    # Add if not already in top3
    top3_factors = [f for f, _ in top3]
    if best_by_r[0] not in top3_factors:
        top3.append(best_by_r)

    print(f"\n  Top candidates: {[f'{f:.2f}' for f, _ in top3]}")

    # Walk-forward for baseline
    wf_baseline = walk_forward_metrics(baseline_trades)
    bl_dict = {w["year"]: w for w in wf_baseline}

    # Walk-forward for each top factor
    wf_top = {}
    for factor, result in top3:
        wf = walk_forward_metrics(result["trades"])
        wf_top[factor] = {w["year"]: w for w in wf}

    all_years = sorted(set(
        list(bl_dict.keys()) +
        [y for td in wf_top.values() for y in td.keys()]
    ))

    # Print header
    hdr = f"  {'Year':>6} | {'--- Baseline (1.00) ---':^26s}"
    for factor, _ in top3:
        hdr += f" | {'--- Factor ' + f'{factor:.2f}' + ' ---':^26s}"
    print(hdr)

    sub = f"  {'':>6} | {'n':>4} {'R':>7} {'PF':>5} {'PPDD':>6}"
    for _ in top3:
        sub += f" | {'n':>4} {'R':>7} {'PF':>5} {'PPDD':>6}"
    print(sub)
    print("  " + "-" * (8 + 28 * (1 + len(top3))))

    wins_count = {f: 0 for f, _ in top3}
    ppdd_wins = {f: 0 for f, _ in top3}

    for y in all_years:
        bl = bl_dict.get(y, {"n": 0, "R": 0, "WR": 0, "PF": 0, "PPDD": 0})
        parts = f"  {y:>4}   | {bl['n']:4d} {bl['R']:+7.1f} {bl['PF']:5.2f} {bl['PPDD']:+6.2f}"

        for factor, _ in top3:
            td = wf_top[factor]
            tv = td.get(y, {"n": 0, "R": 0, "WR": 0, "PF": 0, "PPDD": 0})
            r_marker = " *" if tv["R"] > bl["R"] else "  "
            if tv["R"] > bl["R"]:
                wins_count[factor] += 1
            if tv["PPDD"] > bl["PPDD"]:
                ppdd_wins[factor] += 1
            parts += f" | {tv['n']:4d} {tv['R']:+7.1f} {tv['PF']:5.2f} {tv['PPDD']:+6.2f}{r_marker}"
        print(parts)

    print("  " + "-" * (8 + 28 * (1 + len(top3))))
    for factor, _ in top3:
        m = all_results[factor]["metrics"]
        print(f"  Factor {factor:.2f}: R-wins={wins_count[factor]}/{len(all_years)} years, "
              f"PPDD-wins={ppdd_wins[factor]}/{len(all_years)} years, "
              f"Total R={m['R']:+.1f}, PPDD={m['PPDD']:.2f}")

    # ------------------------------------------------------------------
    # Step 5: Trade-off analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("TRADE-OFF ANALYSIS: STOP WIDTH vs PERFORMANCE")
    print("=" * 120)

    print(f"\n  {'Factor':>7} | {'Avg Stop':>8} | {'Avg Contracts':>13} | {'Stop Rate':>9} | "
          f"{'R':>9} | {'PPDD':>7} | {'PF':>6} | {'Net Impact':>10}")
    print("  " + "-" * 90)

    for factor in factors:
        trades = all_results[factor]["trades"]
        longs = [t for t in trades if t["dir"] == 1]
        stop_dists = [abs(t["entry_price"] - t["stop_price"]) for t in longs]
        avg_stop = np.mean(stop_dists) if stop_dists else 0
        # Approximate avg contracts (we don't store this, but can infer from stop distance)
        # Wider stop → fewer contracts for same R
        avg_contracts_ratio = 1.0 / factor if factor > 0 else 0
        reasons = exit_reason_breakdown(trades, direction=1)
        stop_count = reasons.get("stop", {}).get("count", 0)
        n_longs = len(longs)
        stop_rate = 100 * stop_count / n_longs if n_longs > 0 else 0
        m = all_results[factor]["metrics"]
        delta_r = m["R"] - baseline_m["R"]

        print(f"  {factor:>7.2f} | {avg_stop:8.1f} | {avg_contracts_ratio:10.2f}x    | "
              f"{stop_rate:7.1f}%  | {m['R']:+9.1f} | {m['PPDD']:7.2f} | {m['PF']:6.2f} | "
              f"{delta_r:+10.1f}R")

    # ------------------------------------------------------------------
    # Step 6: Signal-DB prediction validation
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("SIGNAL-DB PREDICTION VALIDATION")
    print("=" * 120)
    print("\n  Signal-DB analysis predicted +27-41% R improvement at factors 1.10-1.20")
    print("  (using simplified per-signal calculation without trim/trail/EOD)")
    print()

    for factor in [1.10, 1.15, 1.20]:
        if factor in all_results:
            m = all_results[factor]["metrics"]
            delta_r = m["R"] - baseline_m["R"]
            pct_change = 100 * delta_r / baseline_m["R"] if baseline_m["R"] != 0 else 0
            print(f"  Factor {factor:.2f}: R={m['R']:+.1f} (delta={delta_r:+.1f}R, {pct_change:+.1f}%), "
                  f"PPDD={m['PPDD']:.2f}, PF={m['PF']:.2f}")
    print()
    delta_110 = all_results[1.10]["metrics"]["R"] - baseline_m["R"] if 1.10 in all_results else 0
    delta_120 = all_results[1.20]["metrics"]["R"] - baseline_m["R"] if 1.20 in all_results else 0
    pct_110 = 100 * delta_110 / baseline_m["R"] if baseline_m["R"] != 0 else 0
    pct_120 = 100 * delta_120 / baseline_m["R"] if baseline_m["R"] != 0 else 0
    print(f"  Actual change at 1.10: {pct_110:+.1f}% (signal-DB predicted: +27%)")
    print(f"  Actual change at 1.20: {pct_120:+.1f}% (signal-DB predicted: +41%)")

    if abs(pct_110) < 5 and abs(pct_120) < 5:
        print("  --> Signal-DB predictions did NOT replicate. Full engine mechanics dominate.")
    elif pct_110 > 10 or pct_120 > 10:
        print("  --> Signal-DB predictions partially replicated. Stop widening shows benefit.")
    else:
        print("  --> Results are mixed. The full engine's trim/trail/EOD mechanics change dynamics.")

    # ------------------------------------------------------------------
    # Step 7: Comprehensive summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 120)

    # Best by PPDD
    best_ppdd_factor = max(all_results.items(), key=lambda x: x[1]["metrics"]["PPDD"])
    best_r_factor = max(all_results.items(), key=lambda x: x[1]["metrics"]["R"])

    print(f"\n  BEST BY PPDD: factor={best_ppdd_factor[0]:.2f}")
    m = best_ppdd_factor[1]["metrics"]
    print(f"    R={m['R']:+.1f} | PPDD={m['PPDD']:.2f} | PF={m['PF']:.2f} | WR={m['WR']:.1f}%")

    print(f"\n  BEST BY R: factor={best_r_factor[0]:.2f}")
    m = best_r_factor[1]["metrics"]
    print(f"    R={m['R']:+.1f} | PPDD={m['PPDD']:.2f} | PF={m['PF']:.2f} | WR={m['WR']:.1f}%")

    print(f"\n  BASELINE (factor=1.00):")
    print(f"    R={baseline_m['R']:+.1f} | PPDD={baseline_m['PPDD']:.2f} | PF={baseline_m['PF']:.2f} | WR={baseline_m['WR']:.1f}%")

    # Direction of optimal
    if best_ppdd_factor[0] > 1.0 and best_r_factor[0] > 1.0:
        print("\n  CONCLUSION: Both PPDD and R optimized with WIDER stops.")
        print(f"    Recommended factor range: {min(best_ppdd_factor[0], best_r_factor[0]):.2f} - {max(best_ppdd_factor[0], best_r_factor[0]):.2f}")
    elif best_ppdd_factor[0] < 1.0 and best_r_factor[0] < 1.0:
        print("\n  CONCLUSION: Both PPDD and R optimized with TIGHTER stops.")
    elif best_ppdd_factor[0] == 1.0 and best_r_factor[0] == 1.0:
        print("\n  CONCLUSION: Current stop placement is already optimal.")
    else:
        print(f"\n  CONCLUSION: PPDD and R disagree on direction.")
        print(f"    PPDD prefers factor={best_ppdd_factor[0]:.2f}, R prefers factor={best_r_factor[0]:.2f}")
        print(f"    If risk-adjusted return matters more, use factor={best_ppdd_factor[0]:.2f}")

    # Marginal improvement check
    for factor in [1.10, 1.15, 1.20]:
        if factor in all_results:
            m_test = all_results[factor]["metrics"]
            r_diff = m_test["R"] - baseline_m["R"]
            ppdd_diff = m_test["PPDD"] - baseline_m["PPDD"]
            if r_diff > 5 and ppdd_diff > 0.5:
                print(f"\n  Factor {factor:.2f} is a CLEAR improvement: +{r_diff:.1f}R, +{ppdd_diff:.2f} PPDD")
            elif r_diff > 0 and ppdd_diff > 0:
                print(f"\n  Factor {factor:.2f} is a MARGINAL improvement: +{r_diff:.1f}R, +{ppdd_diff:.2f} PPDD")
            elif r_diff > 0:
                print(f"\n  Factor {factor:.2f}: +R but -PPDD (more R but also more risk)")
            elif ppdd_diff > 0:
                print(f"\n  Factor {factor:.2f}: -R but +PPDD (less R but better risk-adjusted)")

    print("\n" + "=" * 120)
    print("DONE")
    print("=" * 120)


if __name__ == "__main__":
    main()
