"""
experiments/sweep_research.py -- ICT Sweep Definition Research
==============================================================

Problem: Current sweep detection uses 5m fractal swing points (left=3, right=1).
These are too frequent -> 91.6% of trades show "pre-sweep" -> no discriminative power.

ICT real sweep = price takes out a SIGNIFICANT liquidity level:
  1. Previous Day Low/High (PDL/PDH)
  2. Session lows (Asia, London, Overnight)
  3. HTF swing points (1H/4H significance)

This script:
  - Computes proper significant levels
  - Tests different sweep definitions
  - Finds which level types actually predict FVG quality
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.u2_clean import load_all, compute_metrics, pr, _find_nth_swing, _compute_grade
from features.swing import detect_swing_highs, detect_swing_lows


# ======================================================================
# Compute significant liquidity levels
# ======================================================================

def compute_pdhl(nq: pd.DataFrame) -> pd.DataFrame:
    """Compute Previous Day High/Low (PDH/PDL).

    Uses trading day boundaries (18:00 ET to 18:00 ET next day).
    PDH/PDL are from the COMPLETED prior day, forward-filled during current day.
    """
    et_idx = nq.index.tz_convert("US/Eastern")

    # Trading day: 18:00 ET to next 18:00 ET
    # If hour >= 18, belongs to next trading day
    dates = []
    for i in range(len(nq)):
        ts = et_idx[i]
        if ts.hour >= 18:
            dates.append((ts + pd.Timedelta(days=1)).date())
        else:
            dates.append(ts.date())

    df = pd.DataFrame({
        "high": nq["high"].values,
        "low": nq["low"].values,
        "tday": dates,
    }, index=nq.index)

    # Day high/low
    day_hl = df.groupby("tday").agg(day_high=("high", "max"), day_low=("low", "min"))

    # Shift by 1 day to get PREVIOUS day
    day_hl["pdh"] = day_hl["day_high"].shift(1)
    day_hl["pdl"] = day_hl["day_low"].shift(1)

    # Merge back to bar level
    df = df.join(day_hl[["pdh", "pdl"]], on="tday")

    return pd.DataFrame({
        "pdh": df["pdh"].values,
        "pdl": df["pdl"].values,
    }, index=nq.index)


def compute_htf_swings(nq: pd.DataFrame, left: int = 12, right: int = 3) -> pd.DataFrame:
    """Compute HTF-significance swing points on 5m data.

    left=12, right=3 on 5m = requires 1 hour of dominance on each side.
    This approximates 1H-level swing points without resampling.
    """
    sh = detect_swing_highs(nq["high"], left, right)
    sl = detect_swing_lows(nq["low"], left, right)

    # Shift by right_bars to avoid lookahead
    sh_shifted = sh.shift(right, fill_value=False)
    sl_shifted = sl.shift(right, fill_value=False)

    # Forward-fill prices
    sh_price = np.where(sh, nq["high"].values, np.nan)
    sl_price = np.where(sl, nq["low"].values, np.nan)

    sh_price_shifted = pd.Series(sh_price).shift(right).ffill().values
    sl_price_shifted = pd.Series(sl_price).shift(right).ffill().values

    return pd.DataFrame({
        "htf_swing_high": sh_shifted.values,
        "htf_swing_low": sl_shifted.values,
        "htf_swing_high_price": sh_price_shifted,
        "htf_swing_low_price": sl_price_shifted,
    }, index=nq.index)


def compute_htf_swings_4h(nq: pd.DataFrame, left: int = 48, right: int = 12) -> pd.DataFrame:
    """4H-level significance: left=48 (4H), right=12 (1H) on 5m bars."""
    sh = detect_swing_highs(nq["high"], left, right)
    sl = detect_swing_lows(nq["low"], left, right)

    sh_shifted = sh.shift(right, fill_value=False)
    sl_shifted = sl.shift(right, fill_value=False)

    sh_price = np.where(sh, nq["high"].values, np.nan)
    sl_price = np.where(sl, nq["low"].values, np.nan)

    sh_price_shifted = pd.Series(sh_price).shift(right).ffill().values
    sl_price_shifted = pd.Series(sl_price).shift(right).ffill().values

    return pd.DataFrame({
        "htf4h_swing_high_price": sh_price_shifted,
        "htf4h_swing_low_price": sl_price_shifted,
    }, index=nq.index)


# ======================================================================
# Proper ICT sweep detection
# ======================================================================

def detect_sweep_of_level(
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    level: np.ndarray,
    birth_bar: int,
    fvg_type: str,
    lookback: int = 30,
    min_depth_pts: float = 1.0,
) -> dict:
    """Detect if price swept a specific level before FVG formation.

    For bull FVG: check if price went BELOW a low-type level (PDL, session low, etc.)
                  and then recovered above it.
    For bear FVG: check if price went ABOVE a high-type level and then recovered below it.

    Returns:
        dict with sweep_found, sweep_depth_pts, sweep_bar_offset
    """
    result = {"sweep_found": False, "sweep_depth_pts": 0.0, "sweep_bar_offset": 0}
    start = max(0, birth_bar - lookback)

    if fvg_type == "bull":
        # Looking for sweep of a LOW level
        for j in range(birth_bar - 1, start - 1, -1):
            lev = level[j]
            if np.isnan(lev) or lev <= 0:
                continue

            # Did price breach below the level?
            if l[j] < lev - min_depth_pts:
                depth = lev - l[j]
                # Did price close back above? (sweep = wick below, close above)
                if c[j] >= lev:
                    result = {"sweep_found": True, "sweep_depth_pts": depth,
                              "sweep_bar_offset": birth_bar - j}
                    break
                # Or did a subsequent bar close above? (delayed recovery)
                for k in range(j + 1, min(birth_bar, len(c))):
                    if c[k] >= lev:
                        result = {"sweep_found": True, "sweep_depth_pts": depth,
                                  "sweep_bar_offset": birth_bar - j}
                        break
                if result["sweep_found"]:
                    break
    else:  # bear FVG
        # Looking for sweep of a HIGH level
        for j in range(birth_bar - 1, start - 1, -1):
            lev = level[j]
            if np.isnan(lev) or lev <= 0:
                continue

            if h[j] > lev + min_depth_pts:
                depth = h[j] - lev
                if c[j] <= lev:
                    result = {"sweep_found": True, "sweep_depth_pts": depth,
                              "sweep_bar_offset": birth_bar - j}
                    break
                for k in range(j + 1, min(birth_bar, len(c))):
                    if c[k] <= lev:
                        result = {"sweep_found": True, "sweep_depth_pts": depth,
                                  "sweep_bar_offset": birth_bar - j}
                        break
                if result["sweep_found"]:
                    break

    return result


# ======================================================================
# Run U2 with proper sweep tagging
# ======================================================================
def tag_trades_with_sweeps(d: dict, trades: list[dict], levels: dict) -> list[dict]:
    """Tag existing U2 trades with proper sweep features.

    levels: dict of level_name -> {
        'low': np.ndarray (for bull sweep checking),
        'high': np.ndarray (for bear sweep checking),
    }
    """
    h = d["h"]
    l = d["l"]
    c = d["c"]
    nq = d["nq"]

    for t in trades:
        birth_bar = t.get("zone_birth_bar", -1)
        if birth_bar < 0:
            continue

        direction = t["dir"]
        fvg_type = "bull" if direction == 1 else "bear"

        # Check each level type
        any_sweep = False
        sweep_types = []

        for level_name, level_data in levels.items():
            if fvg_type == "bull":
                arr = level_data.get("low")
            else:
                arr = level_data.get("high")

            if arr is None:
                continue

            result = detect_sweep_of_level(
                h, l, c, arr,
                birth_bar=birth_bar,
                fvg_type=fvg_type,
                lookback=60,   # 5 hours on 5m
                min_depth_pts=1.0,
            )

            key = f"sweep_{level_name}"
            t[key] = result["sweep_found"]
            t[f"{key}_depth"] = result["sweep_depth_pts"]

            if result["sweep_found"]:
                any_sweep = True
                sweep_types.append(level_name)

        t["ict_sweep"] = any_sweep
        t["sweep_types"] = ",".join(sweep_types) if sweep_types else "none"
        t["n_sweep_types"] = len(sweep_types)

    return trades


# ======================================================================
# Analysis
# ======================================================================
def analyze_sweep(df: pd.DataFrame, col: str, label: str, min_trades: int = 50):
    """Analyze a single sweep feature."""
    for val in [True, False]:
        sub = df[df[col] == val]
        if len(sub) < min_trades:
            continue
        m = compute_metrics(sub.to_dict("records"))
        tag = "YES" if val else "NO "
        tpd = m["trades"] / (252 * 10.5)
        print(f"  {label} {tag}: {m['trades']:5d}t | R={m['R']:+8.1f} | PF={m['PF']:5.2f} | "
              f"PPDD={m['PPDD']:6.2f} | WR={m['WR']:5.1f}% | DD={m['MaxDD']:5.1f}R | {tpd:.2f}/d")


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 120)
    print("ICT SWEEP DEFINITION RESEARCH -- Sprint 8")
    print("=" * 120)

    d = load_all()
    nq = d["nq"]
    n = d["n"]

    # Compute significant levels
    print("\n[COMPUTE] Building significant liquidity levels...")
    t0 = _time.perf_counter()

    # 1. Session levels (from cache)
    session_cache = pd.read_parquet(PROJECT / "data" / "cache_session_levels_10yr_v2.parquet")
    assert len(session_cache) == n, f"Session cache len {len(session_cache)} != {n}"

    # 2. Previous Day High/Low
    pdhl = compute_pdhl(nq)

    # 3. HTF swings (1H-level: left=12, right=3 on 5m)
    htf_1h = compute_htf_swings(nq, left=12, right=3)

    # 4. HTF swings (4H-level: left=48, right=12 on 5m)
    htf_4h = compute_htf_swings_4h(nq, left=48, right=12)

    elapsed = _time.perf_counter() - t0
    print(f"[COMPUTE] Done in {elapsed:.1f}s")

    # Count swing points at each level
    print(f"\n[DENSITY] Swing point frequency:")
    for name, mask_col in [
        ("5m fractal (current)", d["swing_low_mask"]),
    ]:
        count = mask_col.sum()
        print(f"  {name:30s}: {count:6d} ({count/n*100:.2f}%)")

    htf1h_sl_count = htf_1h["htf_swing_low"].sum()
    htf4h_sl_data = detect_swing_lows(nq["low"], 48, 12)
    htf4h_sl_count = htf4h_sl_data.sum()
    print(f"  {'1H-level (L=12 R=3)':30s}: {htf1h_sl_count:6d} ({htf1h_sl_count/n*100:.2f}%)")
    print(f"  {'4H-level (L=48 R=12)':30s}: {htf4h_sl_count:6d} ({htf4h_sl_count/n*100:.2f}%)")

    # Run U2 with context to get birth_bar per trade
    print("\n[RUN] Running U2 with context capture...")
    from experiments.ict_context_diagnostic import run_u2_with_context
    trades = run_u2_with_context(d,
        stop_strategy="A2", fvg_size_mult=0.3, max_fvg_age=200,
        min_stop_pts=5.0, tighten_factor=0.85, tp_mult=0.35, nth_swing=5,
    )
    longs = [t for t in trades if t["dir"] == 1]
    print(f"[RUN] {len(longs)} long trades")

    # Build levels dict
    levels = {
        "pdl": {
            "low": pdhl["pdl"].values,
            "high": pdhl["pdh"].values,
        },
        "overnight": {
            "low": session_cache["overnight_low"].values,
            "high": session_cache["overnight_high"].values,
        },
        "asia": {
            "low": session_cache["asia_low"].values,
            "high": session_cache["asia_high"].values,
        },
        "london": {
            "low": session_cache["london_low"].values,
            "high": session_cache["london_high"].values,
        },
        "htf_1h": {
            "low": htf_1h["htf_swing_low_price"].values,
            "high": htf_1h["htf_swing_high_price"].values,
        },
        "htf_4h": {
            "low": htf_4h["htf4h_swing_low_price"].values,
            "high": htf_4h["htf4h_swing_high_price"].values,
        },
    }

    # Tag trades
    print("[TAG] Tagging trades with proper sweep features...")
    longs = tag_trades_with_sweeps(d, longs, levels)
    df = pd.DataFrame(longs)

    # Baseline
    m_base = compute_metrics(longs)
    print(f"\n{'='*120}")
    print(f"BASELINE:")
    pr("U2-v2 longs", m_base)

    # ================================================================
    # Analyze each level type independently
    # ================================================================
    print(f"\n{'='*120}")
    print(f"SWEEP BY LEVEL TYPE (each tested independently)")
    print(f"{'='*120}")

    for level_name in ["pdl", "overnight", "asia", "london", "htf_1h", "htf_4h"]:
        col = f"sweep_{level_name}"
        if col not in df.columns:
            continue
        n_swept = df[col].sum()
        pct = n_swept / len(df) * 100
        print(f"\n  --- {level_name.upper()} (swept: {n_swept}/{len(df)} = {pct:.1f}%) ---")
        analyze_sweep(df, col, level_name)

    # ================================================================
    # Composite: ANY significant sweep
    # ================================================================
    print(f"\n{'='*120}")
    print(f"COMPOSITE SWEEP (any significant level)")
    print(f"{'='*120}")

    n_any = df["ict_sweep"].sum()
    pct_any = n_any / len(df) * 100
    print(f"  Any ICT sweep: {n_any}/{len(df)} = {pct_any:.1f}%")
    analyze_sweep(df, "ict_sweep", "any_ict_sweep")

    # ================================================================
    # Number of level types swept
    # ================================================================
    print(f"\n{'='*120}")
    print(f"NUMBER OF LEVEL TYPES SWEPT (confluence)")
    print(f"{'='*120}")

    for n_types in sorted(df["n_sweep_types"].unique()):
        sub = df[df["n_sweep_types"] == n_types]
        if len(sub) < 30:
            continue
        m = compute_metrics(sub.to_dict("records"))
        tpd = m["trades"] / (252 * 10.5)
        print(f"  {int(n_types)} level types: {m['trades']:5d}t | R={m['R']:+8.1f} | PF={m['PF']:5.2f} | "
              f"PPDD={m['PPDD']:6.2f} | WR={m['WR']:5.1f}% | DD={m['MaxDD']:5.1f}R | {tpd:.2f}/d")

    # ================================================================
    # Best individual levels: sweep depth analysis
    # ================================================================
    print(f"\n{'='*120}")
    print(f"SWEEP DEPTH ANALYSIS (for levels with signal)")
    print(f"{'='*120}")

    for level_name in ["pdl", "overnight", "htf_1h", "htf_4h"]:
        col_depth = f"sweep_{level_name}_depth"
        col_found = f"sweep_{level_name}"
        if col_depth not in df.columns:
            continue
        swept = df[df[col_found] == True]
        if len(swept) < 50:
            continue
        print(f"\n  --- {level_name.upper()} depth distribution ---")
        print(f"  {swept[col_depth].describe().to_string()}")

    # ================================================================
    # Combine best sweep + displacement filter
    # ================================================================
    print(f"\n{'='*120}")
    print(f"COMBINED: Best sweep + displacement filter")
    print(f"{'='*120}")

    # Test various combinations
    has_disp_filter = (df["disp_atr_mult"] >= 0.3) & (df["disp_atr_mult"] <= 1.5)

    combos = [
        ("disp 0.3-1.5 only", has_disp_filter),
        ("disp + PDL sweep", has_disp_filter & (df["sweep_pdl"] == True)),
        ("disp + overnight sweep", has_disp_filter & (df["sweep_overnight"] == True)),
        ("disp + HTF 1H sweep", has_disp_filter & (df["sweep_htf_1h"] == True)),
        ("disp + HTF 4H sweep", has_disp_filter & (df["sweep_htf_4h"] == True)),
        ("disp + any ICT sweep", has_disp_filter & (df["ict_sweep"] == True)),
        ("disp + NO sweep", has_disp_filter & (df["ict_sweep"] == False)),
        ("disp + PDL or overnight", has_disp_filter & ((df["sweep_pdl"] == True) | (df["sweep_overnight"] == True))),
        ("disp + 2+ level types", has_disp_filter & (df["n_sweep_types"] >= 2)),
        ("disp + 3+ level types", has_disp_filter & (df["n_sweep_types"] >= 3)),
    ]

    for label, mask in combos:
        sub = df[mask]
        if len(sub) < 30:
            print(f"  {label:40s} | too few trades ({len(sub)})")
            continue
        m = compute_metrics(sub.to_dict("records"))
        tpd = m["trades"] / (252 * 10.5)
        print(f"  {label:40s} | {m['trades']:5d}t | R={m['R']:+8.1f} | PF={m['PF']:5.2f} | "
              f"PPDD={m['PPDD']:6.2f} | WR={m['WR']:5.1f}% | DD={m['MaxDD']:5.1f}R | {tpd:.2f}/d")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*120}")
    print(f"SUMMARY")
    print(f"{'='*120}")
    print(f"Old sweep (5m fractal): 91.6% of trades -> useless")
    print(f"New sweep frequency by level type:")
    for level_name in ["pdl", "overnight", "asia", "london", "htf_1h", "htf_4h"]:
        col = f"sweep_{level_name}"
        if col in df.columns:
            n_swept = df[col].sum()
            pct = n_swept / len(df) * 100
            print(f"  {level_name:15s}: {pct:5.1f}%")
    print(f"  {'any ICT':15s}: {df['ict_sweep'].sum()/len(df)*100:5.1f}%")


if __name__ == "__main__":
    main()
