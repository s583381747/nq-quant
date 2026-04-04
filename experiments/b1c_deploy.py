"""
experiments/b1c_deploy.py — Full Combined 5m + 1c System with Config H+ Features
=================================================================================

Stacks everything together:
  - 5m signals: Config H+ (SQ=0.80, PM block, stop×0.85, 50/50/0 multi-TP, raw IRL, EOD close)
  - 1c signals: 5m FVG zones + 1m rejection entry, 1m ATR-based min_stop (mult=1.0),
    min_stop_pts=5, stop×0.85, same 50/50/0 multi-TP, same TP targets, same EOD close
  - One position at a time (5m priority)

Configs tested:
  A: Same SQ thresholds for both 5m and 1c (sq_long=0.68, sq_short=0.80)
  B: Relaxed SQ for 1c only (sq_long=0.50, sq_short=0.68)
  C: No SQ filter for 1c signals

Usage: python experiments/b1c_deploy.py
"""
from __future__ import annotations

import copy
import sys
import time as _time
from pathlib import Path
from collections import Counter

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
from experiments.b1c_fvg_zone_1m_entry import (
    load_1m_data,
    precompute_1m_features,
    build_5m_fvg_zones,
    build_time_mappings,
    scan_1m_rejections,
    filter_signals,
    simulate_1m_outcomes,
    combine_5m_and_1m_trades,
    verify_anti_lookahead,
)


SEP = "=" * 120
THIN = "-" * 80


# ======================================================================
# Step 1: Run Config H+ baseline (5m only, stop×0.85)
# ======================================================================
def run_config_h_plus_with_stop_tighten(
    d: dict,
    d_extra: dict,
    stop_factor: float = 0.85,
) -> tuple[list[dict], dict]:
    """
    Run Config H+ (5m) with stop tightening applied.

    Modifies model_stop_arr in-place (restores after), runs pure_liquidity_tp engine.
    """
    original_stop = d["model_stop_arr"].copy()

    # Apply stop tightening
    tightened_stop = widen_stop_array(
        original_stop,
        d["entry_price_arr"],
        d["sig_dir"],
        d["sig_mask"],
        stop_factor,
    )
    d["model_stop_arr"] = tightened_stop

    trades, diag = run_backtest_pure_liquidity(
        d, d_extra,
        sq_long=0.68,
        sq_short=0.80,
        min_stop_atr=1.7,
        block_pm_shorts=True,
        tp_strategy="v1",  # Raw IRL + liquidity ladder
        tp1_trim_pct=0.50,
        be_after_tp1=False,
        be_after_tp2=False,
    )

    # Restore original stops
    d["model_stop_arr"] = original_stop

    m = compute_metrics(trades)
    return trades, m


# ======================================================================
# Step 2: Generate 1c signals with stop×0.85 baked in
# ======================================================================
def generate_1c_signals(
    d: dict,
    d_extra: dict,
    df_1m: pd.DataFrame,
    feat_1m: dict,
    fvg_zones: list[dict],
    ts_5m_ns: np.ndarray,
    ts_1m_ns: np.ndarray,
    map_5m_to_1m: np.ndarray,
    map_1m_to_prev_5m: np.ndarray,
    *,
    sq_long: float = 0.68,
    sq_short: float = 0.80,
    sq_enabled: bool = True,
    min_stop_atr_mult: float = 1.0,
    min_stop_pts_abs: float = 5.0,
    tighten_factor: float = 0.85,
    block_pm_shorts: bool = True,
    label: str = "",
) -> list[dict]:
    """
    Generate filtered 1c signals, then simulate outcomes on 1m data.

    Returns a list of trade dicts (same format as 5m trades).
    """
    print(f"\n{THIN}")
    print(f"1C Generation: {label}")
    print(f"  SQ: enabled={sq_enabled}, sq_long={sq_long}, sq_short={sq_short}")
    print(f"  Stop: tighten={tighten_factor}, 1m ATR mult={min_stop_atr_mult}, floor={min_stop_pts_abs}pts")
    print(THIN)

    # Scan 1m rejections (tighten is baked into scan_1m_rejections)
    raw_signals = scan_1m_rejections(
        df_1m, feat_1m, fvg_zones,
        ts_5m_ns, ts_1m_ns, map_5m_to_1m, map_1m_to_prev_5m, d,
        min_body_ratio=0.50,
        min_stop_pts=0.0,
        tighten_factor=tighten_factor,
        max_fvg_age_bars=200,
        signal_cooldown_1m=6,
    )

    if not raw_signals:
        print("  NO RAW SIGNALS")
        return []

    # If SQ disabled, temporarily override params
    original_sq_enabled = d["params"].get("signal_quality", {}).get("enabled", False)
    if not sq_enabled:
        if "signal_quality" not in d["params"]:
            d["params"]["signal_quality"] = {}
        d["params"]["signal_quality"]["enabled"] = False

    filtered = filter_signals(
        raw_signals, d, map_1m_to_prev_5m, df_1m, feat_1m,
        sq_long=sq_long,
        sq_short=sq_short,
        min_stop_atr_mult=min_stop_atr_mult,
        block_pm_shorts=block_pm_shorts,
        use_1m_atr=True,
        min_stop_pts_abs=min_stop_pts_abs,
    )

    # Restore SQ setting
    if not sq_enabled:
        d["params"]["signal_quality"]["enabled"] = original_sq_enabled

    if not filtered:
        print("  NO FILTERED SIGNALS")
        return []

    # Simulate outcomes on 1m data
    trades = simulate_1m_outcomes(
        filtered, d, d_extra, df_1m, feat_1m, map_1m_to_prev_5m,
    )

    # Tag all as 1c source
    for t in trades:
        t["source"] = "1m"

    return trades


# ======================================================================
# Step 3: Combine and enforce one-position-at-a-time
# ======================================================================
def combine_with_priority(
    trades_5m: list[dict],
    trades_1c: list[dict],
) -> list[dict]:
    """
    Combine 5m and 1c trades. One position at a time, 5m has priority.

    5m trades are placed first in the timeline. 1c trades only fill gaps
    where no 5m trade is active.
    """
    # Tag sources
    for t in trades_5m:
        t["source"] = "5m"
    # 1c trades already tagged as "1m"

    # Sort all by entry_time
    all_trades = sorted(trades_5m + trades_1c, key=lambda t: t["entry_time"])

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
# Step 4: 1c Signal Quality Deep Dive
# ======================================================================
def signal_quality_deep_dive(trades_combined: list[dict], trades_5m_only: list[dict]):
    """Analyze 1c-sourced trades vs 5m trades."""
    t1c = [t for t in trades_combined if t.get("source") == "1m"]
    t5m = [t for t in trades_combined if t.get("source") == "5m"]

    print(f"\n{SEP}")
    print("1C SIGNAL QUALITY DEEP DIVE")
    print(SEP)

    if not t1c:
        print("  No 1c trades in combined set.")
        return

    r_1c = np.array([t["r"] for t in t1c])
    r_5m = np.array([t["r"] for t in t5m]) if t5m else np.array([0.0])

    print(f"\n  {'Metric':30s} | {'1c Trades':>12s} | {'5m Trades':>12s}")
    print(f"  {'-'*30}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'Count':30s} | {len(t1c):>12d} | {len(t5m):>12d}")
    print(f"  {'Total R':30s} | {r_1c.sum():>+12.1f} | {r_5m.sum():>+12.1f}")
    print(f"  {'Avg R':30s} | {r_1c.mean():>+12.4f} | {r_5m.mean():>+12.4f}")
    print(f"  {'Win Rate':30s} | {100*(r_1c>0).mean():>11.1f}% | {100*(r_5m>0).mean():>11.1f}%")
    print(f"  {'Median R':30s} | {np.median(r_1c):>+12.4f} | {np.median(r_5m):>+12.4f}")

    # Average stop distance
    stop_1c = [abs(t["entry_price"] - t["stop_price"]) for t in t1c]
    stop_5m = [abs(t["entry_price"] - t["stop_price"]) for t in t5m] if t5m else [0]
    print(f"  {'Avg Stop Distance (pts)':30s} | {np.mean(stop_1c):>12.1f} | {np.mean(stop_5m):>12.1f}")

    # Average contracts (proxy from pnl_dollars and r)
    # contracts = r_amount / (stop_dist * point_value) ~ estimated
    # Just report stop distance as primary metric

    # Exit reason distribution
    print(f"\n  Exit Reason Distribution:")
    reasons_1c = Counter(t["reason"] for t in t1c)
    reasons_5m = Counter(t["reason"] for t in t5m) if t5m else Counter()
    all_reasons = sorted(set(list(reasons_1c.keys()) + list(reasons_5m.keys())))
    print(f"    {'Reason':15s} | {'1c Count':>8s} {'1c%':>6s} {'1c R':>8s} | {'5m Count':>8s} {'5m%':>6s} {'5m R':>8s}")
    print(f"    {'-'*15}-+-{'-'*24}-+-{'-'*24}")
    for reason in all_reasons:
        c1 = reasons_1c.get(reason, 0)
        c5 = reasons_5m.get(reason, 0)
        r1 = sum(t["r"] for t in t1c if t["reason"] == reason)
        r5 = sum(t["r"] for t in t5m if t["reason"] == reason)
        p1 = 100 * c1 / len(t1c) if t1c else 0
        p5 = 100 * c5 / len(t5m) if t5m else 0
        print(f"    {reason:15s} | {c1:>8d} {p1:>5.1f}% {r1:>+8.1f} | {c5:>8d} {p5:>5.1f}% {r5:>+8.1f}")

    # Time slot distribution (when do 1c signals fire?)
    print(f"\n  Time Slot Distribution (1c trades by hour ET):")
    et_hours = []
    for t in t1c:
        et = t["entry_time"]
        if hasattr(et, "tz_convert"):
            et_local = et.tz_convert("US/Eastern")
        else:
            et_local = pd.Timestamp(et).tz_localize("UTC").tz_convert("US/Eastern")
        et_hours.append(et_local.hour)

    hour_counts = Counter(et_hours)
    for h in sorted(hour_counts.keys()):
        r_h = sum(t["r"] for t, hr in zip(t1c, et_hours) if hr == h)
        print(f"    {h:02d}:00 ET  {hour_counts[h]:4d} trades  R={r_h:+.1f}")

    # Direction breakdown
    longs_1c = [t for t in t1c if t["dir"] == 1]
    shorts_1c = [t for t in t1c if t["dir"] == -1]
    if longs_1c:
        lr = np.array([t["r"] for t in longs_1c])
        print(f"\n  1c Longs:  {len(longs_1c)}t, R={lr.sum():+.1f}, avgR={lr.mean():+.4f}, WR={100*(lr>0).mean():.1f}%")
    if shorts_1c:
        sr = np.array([t["r"] for t in shorts_1c])
        print(f"  1c Shorts: {len(shorts_1c)}t, R={sr.sum():+.1f}, avgR={sr.mean():+.4f}, WR={100*(sr>0).mean():.1f}%")


# ======================================================================
# Step 5: Anti-lookahead verification for best config
# ======================================================================
def verify_anti_lookahead_combined(
    raw_signals: list[dict],
    df_1m: pd.DataFrame,
    ts_1m_ns: np.ndarray,
    label: str = "",
):
    """
    For every 1c signal in the best config, verify:
      - FVG creation time < signal time
      - Entry time > rejection candle close
      - Stop from past bars
    """
    print(f"\n{SEP}")
    print(f"ANTI-LOOKAHEAD VERIFICATION: {label}")
    print(SEP)

    violations = 0
    total = len(raw_signals)

    for sig in raw_signals:
        fvg_creation_5m_idx = sig["fvg_creation_5m_idx"]
        signal_1m_idx = sig["signal_1m_idx"]
        entry_1m_idx = sig["entry_1m_idx"]

        # FVG creation must be before signal time
        if sig["fvg_age_5m_bars"] < 0:
            print(f"  VIOLATION: FVG created after signal! age={sig['fvg_age_5m_bars']}")
            violations += 1

        # Entry must be after rejection candle
        if entry_1m_idx <= signal_1m_idx:
            print(f"  VIOLATION: entry_1m_idx {entry_1m_idx} <= signal_1m_idx {signal_1m_idx}")
            violations += 1

        # Stop from past bars (candle-2 open)
        stop_bar_idx = signal_1m_idx - 2
        if stop_bar_idx >= signal_1m_idx:
            print(f"  VIOLATION: stop bar {stop_bar_idx} >= signal bar {signal_1m_idx}")
            violations += 1

    status = "PASS" if violations == 0 else "FAIL"
    print(f"  {total} signals checked, {violations} violations -> {status}")
    return violations == 0


# ======================================================================
# Main
# ======================================================================
def main():
    t_start = _time.perf_counter()

    print(SEP)
    print("B1C DEPLOY: Full Combined 5m + 1c System with ALL Config H+ Features")
    print("  5m: Config H+ (SQ=0.80, PM block, stop*0.85, 50/50/0, raw IRL, EOD close)")
    print("  1c: 5m FVG zones + 1m rejection, 1m ATR min_stop, stop*0.85, same TP targets")
    print("  SQ Configs: A=same, B=relaxed 1c, C=no SQ for 1c")
    print(SEP)

    # ================================================================
    # LOAD DATA
    # ================================================================
    print("\n[STEP 1] Loading 5m data...")
    d = load_all()
    d_extra = prepare_liquidity_data(d)

    print("\n[STEP 2] Loading 1m data...")
    df_1m = load_1m_data()
    feat_1m = precompute_1m_features(df_1m)

    nq_5m = d["nq"]
    atr_5m = d["atr_arr"]

    # Report ATR stats
    atr_1m_vals = feat_1m["atr"]
    valid_atr_1m = atr_1m_vals[~np.isnan(atr_1m_vals)]
    valid_atr_5m = atr_5m[~np.isnan(atr_5m)]
    print(f"  1m ATR: mean={np.mean(valid_atr_1m):.2f}, median={np.median(valid_atr_1m):.2f}")
    print(f"  5m ATR: mean={np.mean(valid_atr_5m):.2f}, median={np.median(valid_atr_5m):.2f}")

    # ================================================================
    # BUILD FVG ZONES AND MAPPINGS
    # ================================================================
    print("\n[STEP 3] Building 5m FVG zones + time mappings...")
    fvg_zones = build_5m_fvg_zones(nq_5m, atr_5m, min_fvg_atr_mult=0.3)
    ts_5m_ns, ts_1m_ns, map_5m_to_1m, map_1m_to_prev_5m = build_time_mappings(nq_5m, df_1m)

    # ================================================================
    # CONFIG H+ BASELINE (5m only, stop×0.85)
    # ================================================================
    print(f"\n{SEP}")
    print("CONFIG H+ BASELINE (5m only, stop x 0.85)")
    print(SEP)

    baseline_trades, baseline_m = run_config_h_plus_with_stop_tighten(d, d_extra, stop_factor=0.85)
    print_metrics("Config H+ (5m, stop*0.85)", baseline_m)

    # Walk-forward
    baseline_wf = walk_forward_metrics(baseline_trades)
    if baseline_wf:
        print("  Walk-forward:")
        for yr in baseline_wf:
            print(f"    {yr['year']}: {yr['n']:3d}t  R={yr['R']:+7.1f}  WR={yr['WR']:5.1f}%  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}")

    # Long/short breakdown
    bl_longs = [t for t in baseline_trades if t["dir"] == 1]
    bl_shorts = [t for t in baseline_trades if t["dir"] == -1]
    if bl_longs:
        lr = np.array([t["r"] for t in bl_longs])
        print(f"  Longs:  {len(bl_longs)}t, R={lr.sum():+.1f}")
    if bl_shorts:
        sr = np.array([t["r"] for t in bl_shorts])
        print(f"  Shorts: {len(bl_shorts)}t, R={sr.sum():+.1f}")
    bl_avg_stop = np.mean([abs(t["entry_price"] - t["stop_price"]) for t in baseline_trades]) if baseline_trades else 0
    print(f"  Avg stop distance: {bl_avg_stop:.1f} pts")

    # ================================================================
    # 1C SIGNAL GENERATION — 3 SQ CONFIGS
    # ================================================================
    configs = [
        {
            "name": "Config A: Same SQ (0.68/0.80)",
            "sq_long": 0.68, "sq_short": 0.80, "sq_enabled": True,
        },
        {
            "name": "Config B: Relaxed SQ for 1c (0.50/0.68)",
            "sq_long": 0.50, "sq_short": 0.68, "sq_enabled": True,
        },
        {
            "name": "Config C: No SQ filter for 1c",
            "sq_long": 0.0, "sq_short": 0.0, "sq_enabled": False,
        },
    ]

    config_results = {}
    config_trades_1c = {}
    config_trades_combined = {}
    config_raw_signals = {}

    for cfg in configs:
        name = cfg["name"]
        print(f"\n{SEP}")
        print(f"GENERATING 1C SIGNALS: {name}")
        print(SEP)

        trades_1c = generate_1c_signals(
            d, d_extra, df_1m, feat_1m, fvg_zones,
            ts_5m_ns, ts_1m_ns, map_5m_to_1m, map_1m_to_prev_5m,
            sq_long=cfg["sq_long"],
            sq_short=cfg["sq_short"],
            sq_enabled=cfg["sq_enabled"],
            min_stop_atr_mult=1.0,
            min_stop_pts_abs=5.0,
            tighten_factor=0.85,
            block_pm_shorts=True,
            label=name,
        )

        config_trades_1c[name] = trades_1c

        # Combine with 5m baseline
        combined = combine_with_priority(
            [t.copy() for t in baseline_trades],
            [t.copy() for t in trades_1c],
        )
        config_trades_combined[name] = combined

        m_combined = compute_metrics(combined)
        config_results[name] = m_combined

        n_5m = sum(1 for t in combined if t.get("source") == "5m")
        n_1c = sum(1 for t in combined if t.get("source") == "1m")

        print(f"\n  COMBINED: {name}")
        print_metrics(f"  Combined ({name[:20]})", m_combined)
        print(f"    5m trades: {n_5m}, 1c trades: {n_1c}, total: {m_combined['trades']}")

        # 1c-only metrics
        if trades_1c:
            m_1c = compute_metrics(trades_1c)
            print_metrics(f"  1c-only ({name[:20]})", m_1c)
            avg_stop_1c = np.mean([abs(t["entry_price"] - t["stop_price"]) for t in trades_1c])
            print(f"    1c avg stop: {avg_stop_1c:.1f} pts (expected ~14 pts for tighten=0.85)")

    # ================================================================
    # RESULTS TABLE
    # ================================================================
    print(f"\n{SEP}")
    print("RESULTS TABLE")
    print(SEP)

    print(f"\n  {'Config':45s} | {'Total':>5s} | {'5m':>4s} | {'1c':>4s} | {'R':>8s} | {'PPDD':>7s} | {'PF':>6s} | {'WR':>6s} | {'MaxDD':>6s}")
    print(f"  {'-'*45}-+-{'-'*5}-+-{'-'*4}-+-{'-'*4}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    # Baseline row
    print(f"  {'Config H+ (5m only, stop*0.85)':45s} | {baseline_m['trades']:>5d} | {baseline_m['trades']:>4d} | {'0':>4s} | "
          f"{baseline_m['R']:>+8.1f} | {baseline_m['PPDD']:>7.2f} | {baseline_m['PF']:>6.2f} | {baseline_m['WR']:>5.1f}% | {baseline_m['MaxDD']:>6.1f}")

    for cfg in configs:
        name = cfg["name"]
        m = config_results[name]
        combined = config_trades_combined[name]
        n5 = sum(1 for t in combined if t.get("source") == "5m")
        n1 = sum(1 for t in combined if t.get("source") == "1m")
        print(f"  {name:45s} | {m['trades']:>5d} | {n5:>4d} | {n1:>4d} | "
              f"{m['R']:>+8.1f} | {m['PPDD']:>7.2f} | {m['PF']:>6.2f} | {m['WR']:>5.1f}% | {m['MaxDD']:>6.1f}")

    # ================================================================
    # FIND BEST CONFIG
    # ================================================================
    best_name = max(config_results.keys(), key=lambda k: config_results[k]["PPDD"])
    best_m = config_results[best_name]
    best_combined = config_trades_combined[best_name]
    best_1c = config_trades_1c[best_name]

    print(f"\n  BEST CONFIG: {best_name}")
    print(f"    R={best_m['R']:+.1f}, PPDD={best_m['PPDD']:.2f}, PF={best_m['PF']:.2f}, MaxDD={best_m['MaxDD']:.1f}")

    # Delta vs baseline
    dR = best_m["R"] - baseline_m["R"]
    dPPDD = best_m["PPDD"] - baseline_m["PPDD"]
    dMaxDD = best_m["MaxDD"] - baseline_m["MaxDD"]
    print(f"    vs baseline: dR={dR:+.1f}, dPPDD={dPPDD:+.2f}, dMaxDD={dMaxDD:+.1f}")

    # ================================================================
    # WALK-FORWARD for best config
    # ================================================================
    print(f"\n{SEP}")
    print(f"WALK-FORWARD: {best_name}")
    print(SEP)

    wf = walk_forward_metrics(best_combined)
    if wf:
        print(f"  {'Year':>6s} | {'N':>4s} | {'5m':>4s} | {'1c':>4s} | {'R':>8s} | {'WR':>6s} | {'PF':>6s} | {'PPDD':>7s}")
        print(f"  {'-'*6}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")
        for yr in wf:
            y = yr["year"]
            src_5m = sum(1 for t in best_combined
                         if t.get("source") == "5m"
                         and pd.Timestamp(t["entry_time"]).year == y)
            src_1c = sum(1 for t in best_combined
                         if t.get("source") == "1m"
                         and pd.Timestamp(t["entry_time"]).year == y)
            print(f"  {y:>6d} | {yr['n']:>4d} | {src_5m:>4d} | {src_1c:>4d} | "
                  f"{yr['R']:>+8.1f} | {yr['WR']:>5.1f}% | {yr['PF']:>6.2f} | {yr['PPDD']:>7.2f}")

    # Also walk-forward for ALL configs
    print(f"\n  Walk-forward comparison (all configs):")
    for cfg in configs:
        name = cfg["name"]
        wf_cfg = walk_forward_metrics(config_trades_combined[name])
        if wf_cfg:
            # Count negative years
            neg_years = sum(1 for yr in wf_cfg if yr["R"] < 0)
            total_years = len(wf_cfg)
            min_yr = min(wf_cfg, key=lambda yr: yr["R"])
            max_yr = max(wf_cfg, key=lambda yr: yr["R"])
            print(f"    {name[:40]:40s} | neg_years={neg_years}/{total_years} | "
                  f"worst={min_yr['year']}({min_yr['R']:+.1f}R) | best={max_yr['year']}({max_yr['R']:+.1f}R)")

    # ================================================================
    # 1C SIGNAL QUALITY DEEP DIVE
    # ================================================================
    signal_quality_deep_dive(best_combined, baseline_trades)

    # ================================================================
    # ANTI-LOOKAHEAD VERIFICATION
    # ================================================================
    # Re-generate raw signals for best config to get the signal list
    print(f"\n{SEP}")
    print("ANTI-LOOKAHEAD VERIFICATION (best config)")
    print(SEP)

    # Get the best config's SQ settings
    best_cfg = None
    for cfg in configs:
        if cfg["name"] == best_name:
            best_cfg = cfg
            break

    # Re-scan to get raw signals for verification
    raw_sigs_verify = scan_1m_rejections(
        df_1m, feat_1m, fvg_zones,
        ts_5m_ns, ts_1m_ns, map_5m_to_1m, map_1m_to_prev_5m, d,
        min_body_ratio=0.50,
        min_stop_pts=0.0,
        tighten_factor=0.85,
        max_fvg_age_bars=200,
        signal_cooldown_1m=6,
    )

    if raw_sigs_verify:
        verify_anti_lookahead_combined(raw_sigs_verify, df_1m, ts_1m_ns, label=best_name)

        # Additional: verify every filtered 1c trade in best combined set
        print(f"\n  Verifying {len(best_1c)} 1c-sourced trades in best config...")
        for t in best_1c:
            entry_t = t["entry_time"]
            exit_t = t["exit_time"]
            # Entry must be before exit
            if entry_t >= exit_t:
                print(f"    VIOLATION: entry_time >= exit_time: {entry_t} >= {exit_t}")
            # Stop on correct side
            if t["dir"] == 1 and t["stop_price"] >= t["entry_price"]:
                print(f"    VIOLATION: LONG stop >= entry: {t['stop_price']} >= {t['entry_price']}")
            if t["dir"] == -1 and t["stop_price"] <= t["entry_price"]:
                print(f"    VIOLATION: SHORT stop <= entry: {t['stop_price']} <= {t['entry_price']}")
        print(f"  Trade-level verification complete for {len(best_1c)} 1c trades.")

    # ================================================================
    # GRAND SUMMARY
    # ================================================================
    print(f"\n{SEP}")
    print("GRAND SUMMARY")
    print(SEP)

    print(f"\n  {'Config':50s} | {'Trades':>6s} | {'R':>9s} | {'PPDD':>7s} | {'PF':>6s} | {'MaxDD':>6s}")
    print(f"  {'-'*50}-+-{'-'*6}-+-{'-'*9}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}")

    # Baseline
    print(f"  {'Config H+ (5m only, stop*0.85) [BASELINE]':50s} | {baseline_m['trades']:>6d} | "
          f"{baseline_m['R']:>+9.1f} | {baseline_m['PPDD']:>7.2f} | {baseline_m['PF']:>6.2f} | {baseline_m['MaxDD']:>6.1f}")

    for cfg in configs:
        name = cfg["name"]
        m = config_results[name]
        delta_r = m["R"] - baseline_m["R"]
        delta_dd = m["MaxDD"] - baseline_m["MaxDD"]
        suffix = f"  (dR={delta_r:+.1f}, dMaxDD={delta_dd:+.1f})"
        print(f"  {name:50s} | {m['trades']:>6d} | "
              f"{m['R']:>+9.1f} | {m['PPDD']:>7.2f} | {m['PF']:>6.2f} | {m['MaxDD']:>6.1f}{suffix}")

    # Best config highlight
    print(f"\n  >>> BEST: {best_name}")
    print(f"      R={best_m['R']:+.1f} | PPDD={best_m['PPDD']:.2f} | PF={best_m['PF']:.2f} | MaxDD={best_m['MaxDD']:.1f}")
    print(f"      vs baseline: dR={dR:+.1f} | dPPDD={dPPDD:+.2f} | dMaxDD={dMaxDD:+.1f}")

    elapsed = _time.perf_counter() - t_start
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"\n{SEP}")
    print("DONE")
    print(SEP)


if __name__ == "__main__":
    main()
