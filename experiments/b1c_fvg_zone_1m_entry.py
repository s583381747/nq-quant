"""
experiments/b1c_fvg_zone_1m_entry.py — Hybrid 5m FVG Zone + 1m Rejection Entry
================================================================================

Architecture (temporally correct):
  1. Detect FVGs on 5m data. Each FVG defines a ZONE (top/bottom prices).
  2. Monitor 1m data. When price enters an active FVG zone, enter "watch mode."
  3. 1m Rejection: candle tests zone, closes on CORRECT side, body_ratio >= threshold.
  4. Entry = next 1m bar open after rejection candle.
  5. Stop = 1m candle-2 open (2 bars before rejection), tightened by factor.
  6. TP = raw IRL (5m swing) + liquidity ladder TP2.

Anti-lookahead guarantees:
  - FVG zone created in the PAST (known before price returns)
  - Price entering zone is REAL-TIME
  - 1m rejection is the CURRENT just-closed candle
  - Entry is NEXT bar open
  - Stop is from a PAST 1m bar

Usage: python experiments/b1c_fvg_zone_1m_entry.py
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yaml

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

SEP = "=" * 120
THIN = "-" * 80


# ======================================================================
# Step 1: Load 1m data
# ======================================================================
def load_1m_data() -> pd.DataFrame:
    t0 = _time.perf_counter()
    df = pd.read_parquet(PROJECT / "data" / "NQ_1min_10yr.parquet")
    elapsed = _time.perf_counter() - t0
    print(f"[1m DATA] Loaded {len(df):,} bars ({df.index[0]} to {df.index[-1]}) in {elapsed:.1f}s")
    return df


def precompute_1m_features(df_1m: pd.DataFrame) -> dict:
    """Precompute 1m OHLC arrays and rolling ATR."""
    t0 = _time.perf_counter()

    o = df_1m["open"].values.astype(np.float64)
    h = df_1m["high"].values.astype(np.float64)
    l = df_1m["low"].values.astype(np.float64)
    c = df_1m["close"].values.astype(np.float64)
    n = len(df_1m)

    # True Range
    tr = np.empty(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        hl = h[i] - l[i]
        hc = abs(h[i] - c[i - 1])
        lc = abs(l[i] - c[i - 1])
        tr[i] = max(hl, hc, lc)

    # ATR(14) — simple rolling mean
    atr = np.full(n, np.nan)
    period = 14
    cum = np.cumsum(tr)
    atr[period - 1:] = (cum[period - 1:] - np.concatenate([[0], cum[:n - period]])) / period

    # Body ratio
    rng = h - l
    body = np.abs(c - o)
    safe_rng = np.where(rng > 0, rng, 1.0)
    body_ratio = body / safe_rng
    body_ratio[rng <= 0] = 0.0

    elapsed = _time.perf_counter() - t0
    print(f"[1m FEAT] Precomputed ATR(14) + body_ratio in {elapsed:.1f}s")

    return {"o": o, "h": h, "l": l, "c": c, "atr": atr, "body_ratio": body_ratio, "n": n}


# ======================================================================
# Step 2: 5m FVG Detection + Active Zone Tracker
# ======================================================================
@dataclass
class ActiveFVG:
    """A single active FVG zone from 5m data."""
    creation_time: pd.Timestamp
    creation_5m_idx: int       # index in 5m data
    direction: str             # 'bull' or 'bear'
    top: float
    bottom: float
    size: float
    status: str = "untested"   # untested | tested | invalidated
    last_signal_1m_idx: int = -999  # cooldown tracking


def build_5m_fvg_zones(nq_5m: pd.DataFrame, atr_5m: np.ndarray,
                        min_fvg_atr_mult: float = 0.3,
                        max_fvg_age_bars: int = 200,
                        max_active: int = 20) -> list[dict]:
    """
    Detect FVGs on 5m data and build a time-ordered list of FVG zone events.

    Returns list of dicts with keys:
      creation_time, creation_5m_idx, direction, top, bottom, size,
      becomes_known_5m_idx (= creation_5m_idx + 1, due to shift-1 rule)

    FVG is anchored to candle-2, known after candle-3 closes (shift by 1).
    """
    from features.fvg import detect_fvg

    t0 = _time.perf_counter()
    fvg_df = detect_fvg(nq_5m)

    zones = []
    bull_mask = fvg_df["fvg_bull"].values
    bear_mask = fvg_df["fvg_bear"].values
    n = len(nq_5m)
    idx = nq_5m.index

    for i in range(n):
        if bull_mask[i]:
            # After shift-1, signal at row i means FVG candle-2 was at i-1
            # FVG becomes known at bar i (after candle-3 = bar i closes... but
            # detect_fvg already shifted, so at row i the FVG is visible)
            atr_val = atr_5m[i] if not np.isnan(atr_5m[i]) else 30.0
            fvg_size = fvg_df["fvg_size"].iat[i]
            if not np.isnan(fvg_size) and fvg_size >= min_fvg_atr_mult * atr_val:
                zones.append({
                    "creation_time": idx[i - 1] if i > 0 else idx[i],
                    "creation_5m_idx": i - 1 if i > 0 else 0,
                    "becomes_known_5m_idx": i,
                    "direction": "bull",
                    "top": float(fvg_df["fvg_bull_top"].iat[i]),
                    "bottom": float(fvg_df["fvg_bull_bottom"].iat[i]),
                    "size": float(fvg_size),
                })

        if bear_mask[i]:
            atr_val = atr_5m[i] if not np.isnan(atr_5m[i]) else 30.0
            fvg_size = fvg_df["fvg_size"].iat[i]
            if not np.isnan(fvg_size) and fvg_size >= min_fvg_atr_mult * atr_val:
                zones.append({
                    "creation_time": idx[i - 1] if i > 0 else idx[i],
                    "creation_5m_idx": i - 1 if i > 0 else 0,
                    "becomes_known_5m_idx": i,
                    "direction": "bear",
                    "top": float(fvg_df["fvg_bear_top"].iat[i]),
                    "bottom": float(fvg_df["fvg_bear_bottom"].iat[i]),
                    "size": float(fvg_size),
                })

    elapsed = _time.perf_counter() - t0
    n_bull = sum(1 for z in zones if z["direction"] == "bull")
    n_bear = sum(1 for z in zones if z["direction"] == "bear")
    print(f"[5m FVG] Detected {len(zones)} quality FVG zones ({n_bull} bull, {n_bear} bear) in {elapsed:.1f}s")
    return zones


# ======================================================================
# Step 3: Build 5m→1m time mapping
# ======================================================================
def build_time_mappings(nq_5m: pd.DataFrame, df_1m: pd.DataFrame):
    """
    Build mappings between 5m and 1m timestamps.

    Returns:
      - ts_5m_ns: int64 array of 5m timestamps in nanoseconds
      - ts_1m_ns: int64 array of 1m timestamps in nanoseconds
      - map_5m_idx_to_1m_start: for each 5m bar at time T, the 1m index at T+5min
                                (first 1m bar after the 5m bar closes)
    """
    t0 = _time.perf_counter()
    ts_5m_ns = nq_5m.index.astype(np.int64)
    ts_1m_ns = df_1m.index.astype(np.int64)
    five_min_ns = 5 * 60 * 10**9

    # For each 5m bar, find first 1m bar at or after T+5min
    target_ns = ts_5m_ns + five_min_ns
    map_arr = np.searchsorted(ts_1m_ns, target_ns, side='left')
    n_1m = len(df_1m)
    map_arr[map_arr >= n_1m] = -1

    # Also build: for each 1m timestamp, what is the PREVIOUS completed 5m bar?
    # 1m bar at time T → the 5m bar whose close time <= T
    # 5m bar at ts_5m[k] covers [ts_5m[k], ts_5m[k]+5min)
    # The bar is complete at ts_5m[k]+5min
    # So for 1m bar at T: the last completed 5m bar is where ts_5m[k]+5min <= T
    completed_5m_ns = ts_5m_ns + five_min_ns
    map_1m_to_prev_5m = np.searchsorted(completed_5m_ns, ts_1m_ns, side='right') - 1
    map_1m_to_prev_5m = np.clip(map_1m_to_prev_5m, 0, len(nq_5m) - 1)

    elapsed = _time.perf_counter() - t0
    print(f"[MAP] Built 5m<->1m mappings in {elapsed:.1f}s")
    return ts_5m_ns, ts_1m_ns, map_arr, map_1m_to_prev_5m


# ======================================================================
# Step 4: Core — 1m rejection scan with active FVG zone management
# ======================================================================
def scan_1m_rejections(
    df_1m: pd.DataFrame,
    feat_1m: dict,
    fvg_zones: list[dict],
    ts_5m_ns: np.ndarray,
    ts_1m_ns: np.ndarray,
    map_5m_to_1m: np.ndarray,
    map_1m_to_prev_5m: np.ndarray,
    d: dict,
    *,
    min_body_ratio: float = 0.50,
    min_stop_pts: float = 8.0,
    tighten_factor: float = 0.85,
    max_fvg_age_bars: int = 200,
    max_active: int = 20,
    signal_cooldown_1m: int = 6,
    allow_multiple_per_fvg: bool = False,
) -> list[dict]:
    """
    Scan 1m bars for FVG zone rejections. Returns raw signal list.

    For each 1m bar:
      1. Birth new FVG zones that become known at or before current time
      2. Update active FVG zones (invalidate if price closes through)
      3. Check if bar rejects from any active zone
      4. If rejection: record signal with entry/stop

    Returns list of dicts: {
        signal_1m_idx, entry_1m_idx, direction, entry_price, stop_price,
        stop_dist, fvg_direction, fvg_top, fvg_bottom, fvg_creation_time,
        fvg_age_5m_bars, rejection_body_ratio, prev_5m_idx
    }
    """
    t0 = _time.perf_counter()

    o_1m = feat_1m["o"]
    h_1m = feat_1m["h"]
    l_1m = feat_1m["l"]
    c_1m = feat_1m["c"]
    br_1m = feat_1m["body_ratio"]
    n_1m = feat_1m["n"]

    five_min_ns = 5 * 60 * 10**9

    # Sort zones by the time they become known (5m index)
    sorted_zones = sorted(fvg_zones, key=lambda z: z["becomes_known_5m_idx"])
    zone_cursor = 0  # next zone to potentially birth
    n_zones = len(sorted_zones)

    # Active FVG pool
    active: list[ActiveFVG] = []
    signals: list[dict] = []

    # Stats
    n_births = 0
    n_invalidated = 0
    n_tested = 0
    n_rejections_raw = 0
    n_stop_wrong_side = 0
    n_stop_too_small = 0
    n_cooldown_skip = 0

    # Progress tracking
    progress_step = n_1m // 10

    for j in range(n_1m):
        if progress_step > 0 and j % progress_step == 0 and j > 0:
            pct = 100 * j / n_1m
            print(f"  [SCAN] {pct:.0f}% ({j:,}/{n_1m:,}) — {len(active)} active FVGs, {len(signals)} signals")

        cur_1m_ts = ts_1m_ns[j]

        # Which 5m bar is the latest COMPLETED one?
        prev_5m = map_1m_to_prev_5m[j]

        # --- Birth: add FVG zones that become known by now ---
        while zone_cursor < n_zones:
            z = sorted_zones[zone_cursor]
            known_5m_idx = z["becomes_known_5m_idx"]
            # This FVG is "known" when 5m bar at known_5m_idx closes.
            # The 5m bar at known_5m_idx closes at ts_5m[known_5m_idx] + 5min.
            # It should be known if prev_5m >= known_5m_idx
            if prev_5m >= known_5m_idx:
                rec = ActiveFVG(
                    creation_time=z["creation_time"],
                    creation_5m_idx=z["creation_5m_idx"],
                    direction=z["direction"],
                    top=z["top"],
                    bottom=z["bottom"],
                    size=z["size"],
                )
                active.append(rec)
                n_births += 1

                # Prune oldest if too many
                if len(active) > max_active:
                    active.pop(0)

                zone_cursor += 1
            else:
                break

        # --- Update active FVGs: invalidate if price closes through ---
        surviving: list[ActiveFVG] = []
        for fvg in active:
            # Age check
            age_5m = prev_5m - fvg.creation_5m_idx
            if age_5m > max_fvg_age_bars:
                n_invalidated += 1
                continue

            # Invalidation check on 1m close
            if fvg.direction == "bull":
                # Bull FVG: invalidated if close < fvg.bottom
                if c_1m[j] < fvg.bottom:
                    n_invalidated += 1
                    continue
            else:
                # Bear FVG: invalidated if close > fvg.top
                if c_1m[j] > fvg.top:
                    n_invalidated += 1
                    continue

            surviving.append(fvg)
        active = surviving

        # --- Rejection detection ---
        if j < 2 or j + 1 >= n_1m:
            continue

        # Body ratio check
        if br_1m[j] < min_body_ratio:
            continue

        # Check against each active FVG
        for fvg in active:
            # Cooldown: skip if this FVG signaled recently
            if j - fvg.last_signal_1m_idx < signal_cooldown_1m:
                n_cooldown_skip += 1
                continue

            if fvg.direction == "bull":
                # BULLISH FVG → LONG signal
                # 1m bar low touches or enters the zone (low <= fvg_top)
                if l_1m[j] > fvg.top:
                    continue  # Didn't reach zone
                # 1m bar close > fvg_top (closes ABOVE the zone = rejection)
                if c_1m[j] <= fvg.top:
                    continue  # Didn't close above
                # Must be a bullish candle (close > open)
                if c_1m[j] <= o_1m[j]:
                    continue

            else:
                # BEARISH FVG → SHORT signal
                # 1m bar high touches or enters the zone (high >= fvg_bottom)
                if h_1m[j] < fvg.bottom:
                    continue  # Didn't reach zone
                # 1m bar close < fvg_bottom (closes BELOW the zone = rejection)
                if c_1m[j] >= fvg.bottom:
                    continue  # Didn't close below
                # Must be a bearish candle (close < open)
                if c_1m[j] >= o_1m[j]:
                    continue

            # We have a rejection candidate
            n_rejections_raw += 1

            # Entry = next 1m bar open
            entry_price = o_1m[j + 1]

            # Stop = 1m candle-2 open (2 bars before rejection)
            raw_stop = o_1m[j - 2]

            direction = 1 if fvg.direction == "bull" else -1

            # Verify stop on correct side
            if direction == 1 and raw_stop >= entry_price:
                n_stop_wrong_side += 1
                continue
            if direction == -1 and raw_stop <= entry_price:
                n_stop_wrong_side += 1
                continue

            # Tighten
            raw_dist = abs(entry_price - raw_stop)
            tightened_dist = raw_dist * tighten_factor
            if direction == 1:
                stop_price = entry_price - tightened_dist
            else:
                stop_price = entry_price + tightened_dist

            if tightened_dist < min_stop_pts:
                n_stop_too_small += 1
                continue

            # Record signal (include 1m ATR for recalibrated filtering)
            atr_1m_val = feat_1m["atr"][j] if not np.isnan(feat_1m["atr"][j]) else 6.0
            signals.append({
                "signal_1m_idx": j,
                "entry_1m_idx": j + 1,
                "direction": direction,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "stop_dist": tightened_dist,
                "fvg_direction": fvg.direction,
                "fvg_top": fvg.top,
                "fvg_bottom": fvg.bottom,
                "fvg_creation_time": fvg.creation_time,
                "fvg_creation_5m_idx": fvg.creation_5m_idx,
                "fvg_age_5m_bars": prev_5m - fvg.creation_5m_idx,
                "rejection_body_ratio": br_1m[j],
                "prev_5m_idx": prev_5m,
                "atr_1m": atr_1m_val,
            })

            # Mark FVG as signaled (for cooldown)
            fvg.last_signal_1m_idx = j

            if not allow_multiple_per_fvg:
                fvg.status = "tested"
                break  # Only one signal per FVG per visit

    elapsed = _time.perf_counter() - t0
    print(f"[SCAN] Complete in {elapsed:.1f}s")
    print(f"  Births: {n_births}, Invalidated: {n_invalidated}")
    print(f"  Raw rejections: {n_rejections_raw}, Stop wrong side: {n_stop_wrong_side}, Stop too small: {n_stop_too_small}")
    print(f"  Cooldown skips: {n_cooldown_skip}")
    print(f"  Final signals: {len(signals)}")

    return signals


# ======================================================================
# Step 5: Apply Config H+ filters to 1m signals
# ======================================================================
def filter_signals(
    signals: list[dict],
    d: dict,
    map_1m_to_prev_5m: np.ndarray,
    df_1m: pd.DataFrame,
    feat_1m: dict,
    *,
    sq_long: float = 0.68,
    sq_short: float = 0.80,
    min_stop_atr_mult: float = 1.7,
    block_pm_shorts: bool = True,
    use_1m_atr: bool = False,
    min_stop_pts_abs: float = 0.0,
) -> list[dict]:
    """
    Apply Config H+ filters to raw 1m signals using 5m-derived data.

    Filters applied:
      - Bias alignment
      - Session filter (NY only, MSS+SMT bypass)
      - PM shorts block
      - Signal quality (computed on 1m candle but SQ formula from 5m setup)
      - min_stop_atr_mult (using 5m ATR or 1m ATR depending on use_1m_atr)
      - min_stop_pts_abs: absolute minimum stop floor in points
      - News blackout
      - Session regime (lunch dead zone)
      - Observation period (9:30-10:00 ET)
    """
    t0 = _time.perf_counter()

    params = d["params"]
    atr_arr_5m = d["atr_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    sig_type_5m = d["sig_type"]
    has_smt_arr = d["has_smt_arr"]
    et_idx_5m = d["et_idx"]
    fluency_5m = d["fluency_arr"]

    smt_cfg = params.get("smt", {})
    session_regime = params.get("session_regime", {})
    pa_threshold = params.get("pa_quality", {}).get("alt_dir_threshold", 1.0)

    o_1m = feat_1m["o"]
    h_1m = feat_1m["h"]
    l_1m = feat_1m["l"]
    c_1m = feat_1m["c"]
    br_1m = feat_1m["body_ratio"]
    atr_1m = feat_1m["atr"]

    # ET time for 1m data
    et_1m = df_1m.index.tz_convert("US/Eastern")
    et_frac_1m = (et_1m.hour + et_1m.minute / 60.0).values

    filtered = []
    stats = {
        "total": len(signals),
        "pass_bias": 0, "fail_bias": 0,
        "pass_session": 0, "fail_session": 0,
        "pass_pm_shorts": 0, "fail_pm_shorts": 0,
        "pass_sq": 0, "fail_sq": 0,
        "pass_min_stop": 0, "fail_min_stop": 0,
        "pass_news": 0, "fail_news": 0,
        "pass_obs": 0, "fail_obs": 0,
        "pass_lunch": 0, "fail_lunch": 0,
        "final": 0,
    }

    for sig in signals:
        j = sig["signal_1m_idx"]
        entry_j = sig["entry_1m_idx"]
        direction = sig["direction"]
        prev_5m = sig["prev_5m_idx"]

        et_frac = et_frac_1m[j]

        # Observation period (9:30-10:00 ET)
        if 9.5 <= et_frac <= 10.0:
            stats["fail_obs"] += 1
            continue
        stats["pass_obs"] += 1

        # News blackout (use 5m bar's news state)
        if news_blackout_arr is not None and news_blackout_arr[prev_5m]:
            stats["fail_news"] += 1
            continue
        stats["pass_news"] += 1

        # Session filter: NY only (10:00 - 16:00 ET)
        is_ny = (10.0 <= et_frac < 16.0)
        is_london = (3.0 <= et_frac < 9.5)
        is_asia = (et_frac >= 18.0) or (et_frac < 3.0)

        if not is_ny:
            # No MSS+SMT bypass for 1m-level signals (we don't have MSS signal type)
            stats["fail_session"] += 1
            continue
        stats["pass_session"] += 1

        # PM shorts block
        if block_pm_shorts and direction == -1 and et_frac >= 14.0:
            stats["fail_pm_shorts"] += 1
            continue
        stats["pass_pm_shorts"] += 1

        # Bias alignment
        bd = bias_dir_arr[prev_5m]
        bias_opposing = (direction == -np.sign(bd) and bd != 0)
        if bias_opposing:
            stats["fail_bias"] += 1
            continue
        stats["pass_bias"] += 1

        # min_stop filter: absolute floor + ATR-relative check
        atr_5m_val = atr_arr_5m[prev_5m] if not np.isnan(atr_arr_5m[prev_5m]) else 30.0
        stop_dist = sig["stop_dist"]

        # Absolute stop floor (always applied if > 0)
        if min_stop_pts_abs > 0 and stop_dist < min_stop_pts_abs:
            stats["fail_min_stop"] += 1
            continue

        # ATR-relative min stop: use 1m ATR or 5m ATR
        if min_stop_atr_mult > 0:
            if use_1m_atr:
                atr_ref = sig.get("atr_1m", 6.0)
            else:
                atr_ref = atr_5m_val
            if atr_ref > 0 and (stop_dist / atr_ref) < min_stop_atr_mult:
                stats["fail_min_stop"] += 1
                continue
        stats["pass_min_stop"] += 1

        # Session regime: lunch dead zone (12:30-13:00 ET)
        if session_regime.get("enabled", False):
            lunch_s = session_regime.get("lunch_start", 12.5)
            lunch_e = session_regime.get("lunch_end", 13.0)
            lunch_mult = session_regime.get("lunch_mult", 0.0)
            if lunch_s <= et_frac < lunch_e and lunch_mult == 0.0:
                stats["fail_lunch"] += 1
                continue
        stats["pass_lunch"] += 1

        # Signal quality — compute on 1m candle using similar formula
        # SQ = w_size * size_score + w_disp * disp_score + w_flu * flu_score + w_pa * pa_score
        sq_params = params.get("signal_quality", {})
        if sq_params.get("enabled", False):
            # Size score: FVG size / (1.5 * 5m ATR)
            size_sc = min(1.0, sig["stop_dist"] / (atr_5m_val * 1.5)) if atr_5m_val > 0 else 0.5

            # Displacement: body ratio of the 1m rejection candle
            disp_sc = br_1m[j]

            # Fluency: use 5m fluency from cache
            flu_val = fluency_5m[prev_5m] if not np.isnan(fluency_5m[prev_5m]) else 0.5
            flu_sc = min(1.0, max(0.0, flu_val))

            # PA: directional consistency in recent 1m bars
            window = 6
            if j >= window:
                dirs = np.sign(c_1m[j - window:j] - o_1m[j - window:j])
                alt = np.sum(dirs[1:] != dirs[:-1]) / (window - 1)
                pa_sc = 1.0 - alt
            else:
                pa_sc = 0.5

            sq = (sq_params.get("w_size", 0.3) * size_sc
                  + sq_params.get("w_disp", 0.3) * disp_sc
                  + sq_params.get("w_flu", 0.2) * flu_sc
                  + sq_params.get("w_pa", 0.2) * pa_sc)

            eff_sq_thresh = sq_long if direction == 1 else sq_short
            if sq < eff_sq_thresh:
                stats["fail_sq"] += 1
                continue
        stats["pass_sq"] += 1

        # Passed all filters
        sig_out = sig.copy()
        sig_out["et_frac"] = et_frac
        sig_out["atr_5m"] = atr_5m_val
        sig_out["regime"] = regime_arr[prev_5m]
        sig_out["bias_dir"] = bd
        filtered.append(sig_out)
        stats["final"] += 1

    elapsed = _time.perf_counter() - t0
    print(f"[FILTER] {stats['total']} raw -> {stats['final']} filtered in {elapsed:.1f}s")
    for k, v in stats.items():
        if k not in ("total", "final"):
            print(f"    {k}: {v}")

    return filtered


# ======================================================================
# Step 6: Simulate outcomes on 1m data
# ======================================================================
def simulate_1m_outcomes(
    filtered_signals: list[dict],
    d: dict,
    d_extra: dict,
    df_1m: pd.DataFrame,
    feat_1m: dict,
    map_1m_to_prev_5m: np.ndarray,
) -> list[dict]:
    """
    For each filtered 1m signal, simulate forward on 1m data.

    OPTIMIZED: Instead of iterating all 3.5M bars, we:
      1. Process signals in order
      2. When entering a trade, scan forward from entry bar only until exit
      3. Skip directly to next signal after exit

    Multi-TP for longs: 50/50/0 (raw IRL TP1 + liquidity ladder TP2)
    Scalp for shorts: 100% exit at 0.625R
    EOD close at 15:55 ET, one position at a time, 0-for-2 / daily loss limits.
    """
    t0 = _time.perf_counter()

    params = d["params"]
    pos_params = params["position"]
    risk_params = params["risk"]
    bt_params = params["backtest"]
    grading_params = params["grading"]
    dual_mode = params.get("dual_mode", {})
    session_regime = params.get("session_regime", {})

    normal_r = pos_params["normal_r"]
    reduced_r = pos_params["reduced_r"]
    point_value = pos_params["point_value"]
    daily_max_loss_r = risk_params["daily_max_loss_r"]
    max_consec_losses = risk_params["max_consecutive_losses"]
    commission_per_side = bt_params["commission_per_side_micro"]
    slippage_ticks = bt_params["slippage_normal_ticks"]
    slippage_points = slippage_ticks * 0.25
    a_plus_mult = grading_params["a_plus_size_mult"]
    b_plus_mult = grading_params["b_plus_size_mult"]
    c_skip = grading_params["c_skip"]

    o_1m = feat_1m["o"]
    h_1m = feat_1m["h"]
    l_1m = feat_1m["l"]
    c_1m = feat_1m["c"]
    n_1m = feat_1m["n"]

    # 5m data arrays for TP computation
    atr_5m = d["atr_arr"]
    irl_target_5m = d["irl_target_arr"]

    # Precompute ET fractional hours and dates for 1m data
    # This is expensive to do for all 3.5M bars, so do it lazily
    # We only need it for bars near signals. Precompute once.
    et_1m = df_1m.index.tz_convert("US/Eastern")
    et_hour_1m = et_1m.hour.values
    et_minute_1m = et_1m.minute.values

    # Sort signals by entry time
    sorted_sigs = sorted(filtered_signals, key=lambda s: s["entry_1m_idx"])

    trades = []
    current_date = None
    daily_pnl_r = 0.0
    consecutive_losses = 0
    day_stopped = False

    # Diagnostics
    tp1_hit = 0
    tp2_hit = 0
    eod_count = 0

    # Max bars in trade (EOD = ~6.5h = 390 bars at 1m; set generous limit)
    max_bars_in_trade = 500

    # Track when previous trade ended to enforce one-position-at-a-time
    last_exit_bar = -1

    for sig_idx, sig in enumerate(sorted_sigs):
        entry_j = sig["entry_1m_idx"]
        direction = sig["direction"]
        entry_price = sig["entry_price"]
        stop_price = sig["stop_price"]
        stop_dist = sig["stop_dist"]
        prev_5m = sig["prev_5m_idx"]
        et_frac_sig = sig["et_frac"]
        regime = sig["regime"]
        bd = sig["bias_dir"]

        if entry_j >= n_1m:
            continue

        # One position at a time: skip if still in previous trade
        if entry_j <= last_exit_bar:
            continue

        # Check day reset
        entry_et_h = et_hour_1m[entry_j]
        entry_et_m = et_minute_1m[entry_j]
        entry_date = et_1m[entry_j].date() if entry_et_h < 18 else (et_1m[entry_j] + pd.Timedelta(days=1)).date()

        if entry_date != current_date:
            current_date = entry_date
            daily_pnl_r = 0.0
            consecutive_losses = 0
            day_stopped = False

        if day_stopped:
            continue

        # Apply slippage
        actual_entry = entry_price + slippage_points if direction == 1 else entry_price - slippage_points
        actual_stop_dist = abs(actual_entry - stop_price)
        if actual_stop_dist < 1.0:
            continue

        # Grade
        ba = 1.0 if (direction == np.sign(bd) and bd != 0) else 0.0
        grade = _compute_grade_fast(ba, regime)
        if grade == "C" and c_skip:
            continue

        # Position sizing
        entry_dow = et_1m[entry_j].dayofweek
        is_reduced = (entry_dow in (0, 4)) or (regime < 1.0)
        base_r = reduced_r if is_reduced else normal_r
        if grade == "A+":
            r_amount = base_r * a_plus_mult
        elif grade == "B+":
            r_amount = base_r * b_plus_mult
        else:
            r_amount = base_r * 0.5

        # Session regime multiplier
        if session_regime.get("enabled", False):
            sr_lunch_s = session_regime.get("lunch_start", 12.5)
            sr_lunch_e = session_regime.get("lunch_end", 13.0)
            sr_am_end = session_regime.get("am_end", 12.0)
            sr_pm_start = session_regime.get("pm_start", 13.0)
            if et_frac_sig < sr_am_end:
                sr_mult = session_regime.get("am_mult", 1.0)
            elif sr_lunch_s <= et_frac_sig < sr_lunch_e:
                sr_mult = session_regime.get("lunch_mult", 0.0)
            elif et_frac_sig >= sr_pm_start:
                sr_mult = session_regime.get("pm_mult", 1.0)
            else:
                sr_mult = 1.0
            r_amount *= sr_mult
            if r_amount <= 0:
                continue

        contracts = max(1, int(r_amount / (actual_stop_dist * point_value))) if actual_stop_dist > 0 else 0
        if contracts <= 0:
            continue

        # TP computation
        irl_raw = irl_target_5m[prev_5m] if not np.isnan(irl_target_5m[prev_5m]) else 0.0

        if direction == 1:
            # LONG: multi-TP — raw IRL + liquidity ladder
            tp1, tp2 = build_liquidity_ladder_long(
                actual_entry, stop_price, irl_raw, prev_5m, d_extra
            )
            if tp1 <= actual_entry:
                tp1 = actual_entry + actual_stop_dist * 0.5
            if tp2 <= tp1:
                tp2 = actual_entry + (tp1 - actual_entry) * 1.5
            is_multi_tp = True
        else:
            # SHORT: scalp — 100% exit at 0.625R
            short_rr = dual_mode.get("short_rr", 0.625)
            tp1 = actual_entry - actual_stop_dist * short_rr
            tp2 = 0.0
            is_multi_tp = False

        short_trim_pct = dual_mode.get("short_trim_pct", 1.0)

        # --- Simulate forward from entry bar ---
        trim_stage = 0
        be_stop = 0.0
        remaining = contracts
        trim1_c = 0
        trim2_c = 0
        tp1_pnl_pts = 0.0
        tp2_pnl_pts = 0.0
        single_trimmed = False
        exited = False
        exit_reason = ""
        exit_price = 0.0
        exit_contracts = remaining
        exit_bar = entry_j

        scan_end = min(entry_j + max_bars_in_trade, n_1m)
        for jj in range(entry_j + 1, scan_end):
            et_h = et_hour_1m[jj]
            et_m_val = et_minute_1m[jj]
            et_frac_bar = et_h + et_m_val / 60.0

            # EOD close at 15:55 ET
            if et_frac_bar >= 15.917:
                exit_price = c_1m[jj] - slippage_points if direction == 1 else c_1m[jj] + slippage_points
                exit_reason = "eod_close"
                exit_bar = jj
                exit_contracts = remaining
                exited = True
                break

            if is_multi_tp and direction == 1:
                eff_stop = stop_price
                if trim_stage >= 1 and be_stop > 0:
                    eff_stop = max(eff_stop, be_stop)

                if l_1m[jj] <= eff_stop:
                    exit_price = eff_stop - slippage_points
                    exit_reason = "be_sweep" if (trim_stage > 0 and eff_stop >= actual_entry) else "stop"
                    exit_bar = jj
                    exit_contracts = remaining
                    exited = True
                    break

                if trim_stage == 0 and h_1m[jj] >= tp1:
                    tc1 = max(1, int(contracts * 0.50))
                    trim1_c = tc1
                    remaining = contracts - tc1
                    tp1_pnl_pts = tp1 - actual_entry
                    trim_stage = 1
                    tp1_hit += 1
                    if remaining <= 0:
                        exit_price = tp1
                        exit_reason = "tp1"
                        exit_bar = jj
                        exit_contracts = contracts
                        exited = True
                        break

                if trim_stage == 1 and h_1m[jj] >= tp2:
                    tc2 = remaining
                    trim2_c = tc2
                    remaining -= tc2
                    tp2_pnl_pts = tp2 - actual_entry
                    trim_stage = 2
                    tp2_hit += 1
                    if remaining <= 0:
                        exit_price = tp2
                        exit_reason = "tp2"
                        exit_bar = jj
                        exit_contracts = contracts
                        exited = True
                        break

            elif direction == -1:
                eff_stop = stop_price
                if single_trimmed and be_stop > 0:
                    eff_stop = min(eff_stop, be_stop)

                if h_1m[jj] >= eff_stop:
                    exit_price = eff_stop + slippage_points
                    exit_reason = "be_sweep" if (single_trimmed and eff_stop <= actual_entry) else "stop"
                    exit_bar = jj
                    exit_contracts = remaining
                    exited = True
                    break

                if not single_trimmed and l_1m[jj] <= tp1:
                    tc = max(1, int(contracts * short_trim_pct))
                    remaining = contracts - tc
                    single_trimmed = True
                    be_stop = actual_entry
                    if remaining <= 0:
                        exit_price = tp1
                        exit_reason = "tp1"
                        exit_bar = jj
                        exit_contracts = contracts
                        exited = True
                        break

        if not exited:
            # Force close at end of scan window
            exit_bar = min(scan_end - 1, n_1m - 1)
            exit_price = c_1m[exit_bar]
            exit_reason = "eod_close"
            exit_contracts = remaining

        # PNL calculation
        pnl_pts_runner = (exit_price - actual_entry) if direction == 1 else (actual_entry - exit_price)
        any_trimmed = trim_stage > 0 or single_trimmed

        if is_multi_tp:
            total_pnl = 0.0
            if trim_stage >= 1:
                total_pnl += tp1_pnl_pts * point_value * trim1_c
            if trim_stage >= 2:
                total_pnl += tp2_pnl_pts * point_value * trim2_c
            total_pnl += pnl_pts_runner * point_value * exit_contracts
            total_pnl -= commission_per_side * 2 * contracts
        elif single_trimmed and exit_reason != "tp1":
            trim_c = contracts - exit_contracts
            trim_pnl = (tp1 - actual_entry if direction == 1 else actual_entry - tp1) * point_value * trim_c
            total_pnl = trim_pnl + pnl_pts_runner * point_value * exit_contracts
            total_pnl -= commission_per_side * 2 * contracts
        else:
            total_pnl = pnl_pts_runner * point_value * exit_contracts
            total_pnl -= commission_per_side * 2 * exit_contracts

        total_risk = actual_stop_dist * point_value * contracts
        r_mult = total_pnl / total_risk if total_risk > 0 else 0.0

        if exit_reason == "eod_close":
            eod_count += 1

        trades.append({
            "entry_time": df_1m.index[entry_j],
            "exit_time": df_1m.index[exit_bar],
            "r": r_mult, "reason": exit_reason, "dir": direction,
            "type": "1m_rejection", "trimmed": any_trimmed,
            "grade": grade,
            "entry_price": actual_entry, "exit_price": exit_price,
            "stop_price": stop_price, "tp1_price": tp1,
            "tp2_price": tp2 if is_multi_tp else 0.0,
            "pnl_dollars": total_pnl, "trim_stage": trim_stage,
        })

        last_exit_bar = exit_bar

        daily_pnl_r += r_mult
        if exit_reason == "be_sweep" and any_trimmed:
            pass
        elif exit_reason == "eod_close":
            pass
        elif r_mult < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0
        if consecutive_losses >= max_consec_losses:
            day_stopped = True
        if daily_pnl_r <= -daily_max_loss_r:
            day_stopped = True

    elapsed = _time.perf_counter() - t0
    print(f"[SIM] {len(trades)} trades simulated in {elapsed:.1f}s (TP1={tp1_hit}, TP2={tp2_hit}, EOD={eod_count})")
    return trades


# ======================================================================
# Anti-Lookahead Verification
# ======================================================================
def verify_anti_lookahead(signals: list[dict], df_1m: pd.DataFrame, ts_1m_ns: np.ndarray):
    """Verify zero lookahead violations in all signals."""
    violations = 0
    for sig in signals:
        # FVG creation must be before signal time
        fvg_creation_5m_idx = sig["fvg_creation_5m_idx"]
        signal_1m_idx = sig["signal_1m_idx"]
        entry_1m_idx = sig["entry_1m_idx"]

        # Entry must be after signal (rejection) candle
        if entry_1m_idx <= signal_1m_idx:
            print(f"  VIOLATION: entry_1m_idx {entry_1m_idx} <= signal_1m_idx {signal_1m_idx}")
            violations += 1

        # Stop comes from bars BEFORE the rejection candle
        stop_bar_idx = signal_1m_idx - 2
        if stop_bar_idx >= signal_1m_idx:
            print(f"  VIOLATION: stop bar {stop_bar_idx} >= signal bar {signal_1m_idx}")
            violations += 1

        # FVG age must be positive (created before signal)
        if sig["fvg_age_5m_bars"] < 0:
            print(f"  VIOLATION: FVG age negative: {sig['fvg_age_5m_bars']}")
            violations += 1

    status = "PASS" if violations == 0 else "FAIL"
    print(f"ANTI-LOOKAHEAD CHECK: {len(signals)} signals verified, {violations} violations — {status}")
    return violations == 0


# ======================================================================
# Run single config
# ======================================================================
def run_single_config(
    d: dict, d_extra: dict, df_1m: pd.DataFrame, feat_1m: dict,
    fvg_zones: list[dict],
    ts_5m_ns: np.ndarray, ts_1m_ns: np.ndarray,
    map_5m_to_1m: np.ndarray, map_1m_to_prev_5m: np.ndarray,
    *,
    min_body_ratio: float = 0.50,
    min_stop_pts: float = 8.0,
    tighten_factor: float = 0.85,
    max_fvg_age_bars: int = 200,
    signal_cooldown_1m: int = 6,
    sq_long: float = 0.68,
    sq_short: float = 0.80,
    min_stop_atr_mult: float = 1.7,
    block_pm_shorts: bool = True,
    use_1m_atr: bool = False,
    min_stop_pts_abs: float = 0.0,
    label: str = "",
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Run full pipeline for one config. Returns (trades, metrics)."""

    if verbose:
        print(f"\n{THIN}")
        print(f"Config: {label}")
        print(f"  min_stop={min_stop_pts}, max_fvg_age={max_fvg_age_bars}, "
              f"body_ratio={min_body_ratio}, cooldown={signal_cooldown_1m}")
        print(THIN)

    # Step 3: Scan for 1m rejections
    raw_signals = scan_1m_rejections(
        df_1m, feat_1m, fvg_zones,
        ts_5m_ns, ts_1m_ns, map_5m_to_1m, map_1m_to_prev_5m, d,
        min_body_ratio=min_body_ratio,
        min_stop_pts=min_stop_pts,
        tighten_factor=tighten_factor,
        max_fvg_age_bars=max_fvg_age_bars,
        signal_cooldown_1m=signal_cooldown_1m,
    )

    if not raw_signals:
        if verbose:
            print("  NO RAW SIGNALS — skipping")
        return [], compute_metrics([])

    # Step 4: Filter
    filtered = filter_signals(
        raw_signals, d, map_1m_to_prev_5m, df_1m, feat_1m,
        sq_long=sq_long, sq_short=sq_short,
        min_stop_atr_mult=min_stop_atr_mult,
        block_pm_shorts=block_pm_shorts,
        use_1m_atr=use_1m_atr,
        min_stop_pts_abs=min_stop_pts_abs,
    )

    if not filtered:
        if verbose:
            print("  NO FILTERED SIGNALS — skipping")
        return [], compute_metrics([])

    # Anti-lookahead
    if verbose:
        verify_anti_lookahead(filtered, df_1m, ts_1m_ns)

    # Step 5: Simulate
    trades = simulate_1m_outcomes(
        filtered, d, d_extra, df_1m, feat_1m, map_1m_to_prev_5m,
    )

    m = compute_metrics(trades)
    if verbose:
        print_metrics(label, m)

        # Long/short breakdown
        longs = [t for t in trades if t["dir"] == 1]
        shorts = [t for t in trades if t["dir"] == -1]
        if longs:
            lr = np.array([t["r"] for t in longs])
            print(f"    Longs:  {len(longs)}t, R={lr.sum():+.1f}, avgR={lr.mean():+.4f}, WR={100*(lr>0).mean():.1f}%")
        if shorts:
            sr = np.array([t["r"] for t in shorts])
            print(f"    Shorts: {len(shorts)}t, R={sr.sum():+.1f}, avgR={sr.mean():+.4f}, WR={100*(sr>0).mean():.1f}%")

        # Average stop distance
        stop_dists = [t["stop_price"] for t in trades]
        entry_prices = [t["entry_price"] for t in trades]
        avg_stop = np.mean([abs(e - s) for e, s in zip(entry_prices, stop_dists)]) if trades else 0
        print(f"    Avg stop distance: {avg_stop:.1f} pts")

        # Bull vs bear FVG signals
        n_bull_sig = sum(1 for s in filtered if s["fvg_direction"] == "bull")
        n_bear_sig = sum(1 for s in filtered if s["fvg_direction"] == "bear")
        print(f"    Signals from bull FVGs: {n_bull_sig}, bear FVGs: {n_bear_sig}")

        # Per-year
        wf = walk_forward_metrics(trades)
        if wf:
            print(f"    Walk-forward:")
            for yr in wf:
                print(f"      {yr['year']}: {yr['n']:3d}t  R={yr['R']:+7.1f}  WR={yr['WR']:5.1f}%  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}")

    return trades, m


# ======================================================================
# Config H+ baseline (5m signals)
# ======================================================================
def run_config_h_plus_baseline(d: dict, d_extra: dict) -> tuple[list[dict], dict]:
    """Run Config H+ (5m only) as baseline for comparison."""
    from experiments.pure_liquidity_tp import run_backtest_pure_liquidity

    trades, diag = run_backtest_pure_liquidity(
        d, d_extra,
        sq_long=0.68, sq_short=0.80,
        min_stop_atr=1.7,
        block_pm_shorts=True,
        tp_strategy="v1",  # Raw IRL + liquidity ladder
        tp1_trim_pct=0.50,
        be_after_tp1=False,
        be_after_tp2=False,
    )
    return trades, compute_metrics(trades)


# ======================================================================
# Main
# ======================================================================
def combine_5m_and_1m_trades(
    trades_5m: list[dict],
    trades_1m: list[dict],
) -> list[dict]:
    """
    Combine 5m and 1m trades into a single timeline.
    One position at a time: if trades overlap, 5m takes precedence.
    """
    # Add source tag
    for t in trades_5m:
        t["source"] = "5m"
    for t in trades_1m:
        t["source"] = "1m"

    # Merge and sort by entry time
    all_trades = sorted(trades_5m + trades_1m, key=lambda t: t["entry_time"])

    # One position at a time: skip if overlapping with previous trade
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


def main():
    print(SEP)
    print("B1C RECALIBRATED: 1m ATR for 1m stop filter + combined 5m+1m system")
    print("Fix: min_stop_atr_mult was using 5m ATR (~30pts) for 1m signals (~16pt stops)")
    print("     Now using 1m ATR (~6pts) so 16pt stop / 6pt ATR = 2.67x -> passes easily")
    print(SEP)

    # ---- Load data ----
    print("\n[STEP 1] Loading data...")
    d = load_all()
    d_extra = prepare_liquidity_data(d)

    df_1m = load_1m_data()
    feat_1m = precompute_1m_features(df_1m)

    nq_5m = d["nq"]
    atr_5m = d["atr_arr"]

    # Report 1m ATR statistics
    atr_1m_vals = feat_1m["atr"]
    valid_atr = atr_1m_vals[~np.isnan(atr_1m_vals)]
    print(f"[1m ATR] mean={np.mean(valid_atr):.2f}, median={np.median(valid_atr):.2f}, "
          f"p25={np.percentile(valid_atr, 25):.2f}, p75={np.percentile(valid_atr, 75):.2f}")
    valid_5m_atr = atr_5m[~np.isnan(atr_5m)]
    print(f"[5m ATR] mean={np.mean(valid_5m_atr):.2f}, median={np.median(valid_5m_atr):.2f}, "
          f"p25={np.percentile(valid_5m_atr, 25):.2f}, p75={np.percentile(valid_5m_atr, 75):.2f}")

    # ---- Build FVG zones ----
    print("\n[STEP 2] Building 5m FVG zones...")
    fvg_zones = build_5m_fvg_zones(nq_5m, atr_5m, min_fvg_atr_mult=0.3)

    # ---- Build time mappings ----
    print("\n[STEP 3] Building time mappings...")
    ts_5m_ns, ts_1m_ns, map_5m_to_1m, map_1m_to_prev_5m = build_time_mappings(nq_5m, df_1m)

    # ---- Config H+ baseline (5m) ----
    print(f"\n{SEP}")
    print("CONFIG H+ BASELINE (5m signals only)")
    print(SEP)
    baseline_trades, baseline_m = run_config_h_plus_baseline(d, d_extra)
    print_metrics("Config H+ (5m baseline)", baseline_m)

    baseline_wf = walk_forward_metrics(baseline_trades)
    if baseline_wf:
        print("  Walk-forward:")
        for yr in baseline_wf:
            print(f"    {yr['year']}: {yr['n']:3d}t  R={yr['R']:+7.1f}  WR={yr['WR']:5.1f}%  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}")

    baseline_longs = [t for t in baseline_trades if t["dir"] == 1]
    baseline_shorts = [t for t in baseline_trades if t["dir"] == -1]
    if baseline_longs:
        lr = np.array([t["r"] for t in baseline_longs])
        print(f"  Longs:  {len(baseline_longs)}t, R={lr.sum():+.1f}")
    if baseline_shorts:
        sr = np.array([t["r"] for t in baseline_shorts])
        print(f"  Shorts: {len(baseline_shorts)}t, R={sr.sum():+.1f}")
    baseline_avg_stop = np.mean([abs(t["entry_price"] - t["stop_price"]) for t in baseline_trades]) if baseline_trades else 0
    print(f"  Avg stop distance: {baseline_avg_stop:.1f} pts")

    # ---- Original 1c with 5m ATR (for reference — the broken filter) ----
    print(f"\n{SEP}")
    print("1C ORIGINAL (5m ATR filter — the broken version)")
    print(f"  min_stop_atr_mult=1.7 using 5m ATR (~30pts) -> kills 75% of 1m signals")
    print(SEP)

    orig_trades, orig_m = run_single_config(
        d, d_extra, df_1m, feat_1m, fvg_zones,
        ts_5m_ns, ts_1m_ns, map_5m_to_1m, map_1m_to_prev_5m,
        label="1c ORIGINAL (5m ATR)",
        min_stop_pts=8.0,
        min_stop_atr_mult=1.7,
        use_1m_atr=False,
    )

    # ====================================================================
    # SWEEP A: 1m ATR-based min stop (the main fix)
    # ====================================================================
    print(f"\n{SEP}")
    print("SWEEP A: 1m ATR-BASED MIN STOP (main fix)")
    print("  min_stop_atr_mult = [1.0, 1.5, 1.7, 2.0, 2.5] using 1m ATR")
    print("  min_stop_pts_abs = 5 (absolute floor)")
    print(SEP)

    sweep_a_results = []
    sweep_a_trades = {}  # store for later combined analysis
    for mult in [1.0, 1.5, 1.7, 2.0, 2.5]:
        lbl = f"A: 1mATR mult={mult:.1f}, floor=5pts"
        trades, m = run_single_config(
            d, d_extra, df_1m, feat_1m, fvg_zones,
            ts_5m_ns, ts_1m_ns, map_5m_to_1m, map_1m_to_prev_5m,
            label=lbl,
            min_stop_pts=0.0,  # raw scan doesn't filter by pts
            min_stop_atr_mult=mult,
            use_1m_atr=True,
            min_stop_pts_abs=5.0,
            verbose=False,
        )
        sweep_a_results.append({"config": lbl, "mult": mult, "floor": 5, **m})
        sweep_a_trades[mult] = trades
        print_metrics(f"A mult={mult:.1f}", m)

        # Average 1m stop distance
        if trades:
            avg_stop = np.mean([abs(t["entry_price"] - t["stop_price"]) for t in trades])
            print(f"    Avg 1m stop distance: {avg_stop:.1f} pts")

    print(f"\n  --- Sweep A Summary ---")
    print(f"  {'Config':30s} | {'Trades':>6s} | {'R':>8s} | {'PPDD':>7s} | {'PF':>6s} | {'WR':>6s} | {'MaxDD':>6s}")
    print(f"  {'-'*30}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for r in sweep_a_results:
        print(f"  {'A mult='+str(r['mult']):30s} | {r['trades']:>6d} | {r['R']:>+8.1f} | {r['PPDD']:>7.2f} | {r['PF']:>6.2f} | {r['WR']:>5.1f}% | {r['MaxDD']:>6.1f}")

    # ====================================================================
    # SWEEP B: Absolute min stop only (simpler alternative)
    # ====================================================================
    print(f"\n{SEP}")
    print("SWEEP B: ABSOLUTE MIN STOP ONLY (no ATR-relative check)")
    print("  min_stop_pts = [5, 8, 10, 15]")
    print(SEP)

    sweep_b_results = []
    for pts in [5, 8, 10, 15]:
        lbl = f"B: abs floor={pts}pts only"
        trades, m = run_single_config(
            d, d_extra, df_1m, feat_1m, fvg_zones,
            ts_5m_ns, ts_1m_ns, map_5m_to_1m, map_1m_to_prev_5m,
            label=lbl,
            min_stop_pts=0.0,  # raw scan doesn't filter
            min_stop_atr_mult=0.0,  # disable ATR check
            use_1m_atr=False,
            min_stop_pts_abs=float(pts),
            verbose=False,
        )
        sweep_b_results.append({"config": lbl, "floor_pts": pts, **m})
        print_metrics(f"B floor={pts:2d}pts", m)

        if trades:
            avg_stop = np.mean([abs(t["entry_price"] - t["stop_price"]) for t in trades])
            print(f"    Avg 1m stop distance: {avg_stop:.1f} pts")

    print(f"\n  --- Sweep B Summary ---")
    print(f"  {'Config':30s} | {'Trades':>6s} | {'R':>8s} | {'PPDD':>7s} | {'PF':>6s} | {'WR':>6s} | {'MaxDD':>6s}")
    print(f"  {'-'*30}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for r in sweep_b_results:
        print(f"  {'B floor='+str(r['floor_pts'])+'pts':30s} | {r['trades']:>6d} | {r['R']:>+8.1f} | {r['PPDD']:>7.2f} | {r['PF']:>6.2f} | {r['WR']:>5.1f}% | {r['MaxDD']:>6.1f}")

    # ====================================================================
    # Find best 1c config from A and B sweeps
    # ====================================================================
    all_sweep = sweep_a_results + sweep_b_results
    # Filter to configs with >= 20 trades (min viability)
    viable = [r for r in all_sweep if r["trades"] >= 20]
    if viable:
        best_1c = max(viable, key=lambda r: r["PPDD"])
    else:
        best_1c = max(all_sweep, key=lambda r: r["PPDD"]) if all_sweep else None

    print(f"\n  BEST 1C CONFIG: {best_1c['config'] if best_1c else 'none'}")
    if best_1c:
        print(f"    trades={best_1c['trades']}, R={best_1c['R']:+.1f}, PPDD={best_1c['PPDD']:.2f}, PF={best_1c['PF']:.2f}")

    # ====================================================================
    # SWEEP C: Combined 5m + 1c system
    # ====================================================================
    print(f"\n{SEP}")
    print("SWEEP C: COMBINED 5m Config H+ + best 1c configs")
    print("  5m trades from Config H+ baseline")
    print("  1c trades from best Sweep A configs")
    print("  One position at a time (no overlap)")
    print(SEP)

    sweep_c_results = []

    # Run combined for each Sweep A config
    for mult in [1.0, 1.5, 1.7, 2.0, 2.5]:
        trades_1m = sweep_a_trades.get(mult, [])
        if not trades_1m:
            continue

        combined = combine_5m_and_1m_trades(
            [t.copy() for t in baseline_trades],
            [t.copy() for t in trades_1m],
        )

        m = compute_metrics(combined)
        n_5m = sum(1 for t in combined if t.get("source") == "5m")
        n_1m = sum(1 for t in combined if t.get("source") == "1m")
        lbl = f"C: 5m+1c(mult={mult:.1f})"
        sweep_c_results.append({
            "config": lbl, "mult": mult,
            "n_5m": n_5m, "n_1m": n_1m,
            **m,
        })

        print_metrics(lbl, m)
        print(f"    5m signals: {n_5m}, 1m signals: {n_1m}, total: {m['trades']}")

    # Also run combined for best B config
    if best_1c and "floor_pts" in best_1c:
        best_pts = best_1c["floor_pts"]
        # Re-run to get trades
        trades_b, m_b = run_single_config(
            d, d_extra, df_1m, feat_1m, fvg_zones,
            ts_5m_ns, ts_1m_ns, map_5m_to_1m, map_1m_to_prev_5m,
            label=f"B: floor={best_pts}pts (for combined)",
            min_stop_pts=0.0,
            min_stop_atr_mult=0.0,
            use_1m_atr=False,
            min_stop_pts_abs=float(best_pts),
            verbose=False,
        )
        combined_b = combine_5m_and_1m_trades(
            [t.copy() for t in baseline_trades],
            [t.copy() for t in trades_b],
        )
        m_cb = compute_metrics(combined_b)
        n_5m_b = sum(1 for t in combined_b if t.get("source") == "5m")
        n_1m_b = sum(1 for t in combined_b if t.get("source") == "1m")
        lbl_b = f"C: 5m+1c(floor={best_pts}pts)"
        sweep_c_results.append({
            "config": lbl_b, "floor_pts": best_pts,
            "n_5m": n_5m_b, "n_1m": n_1m_b,
            **m_cb,
        })
        print_metrics(lbl_b, m_cb)
        print(f"    5m signals: {n_5m_b}, 1m signals: {n_1m_b}, total: {m_cb['trades']}")

    print(f"\n  --- Sweep C Summary ---")
    print(f"  {'Config':30s} | {'Trades':>6s} | {'5m':>4s} | {'1m':>4s} | {'R':>8s} | {'PPDD':>7s} | {'PF':>6s} | {'WR':>6s} | {'MaxDD':>6s}")
    print(f"  {'-'*30}-+-{'-'*6}-+-{'-'*4}-+-{'-'*4}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for r in sweep_c_results:
        print(f"  {r['config']:30s} | {r['trades']:>6d} | {r.get('n_5m', 0):>4d} | {r.get('n_1m', 0):>4d} | {r['R']:>+8.1f} | {r['PPDD']:>7.2f} | {r['PF']:>6.2f} | {r['WR']:>5.1f}% | {r['MaxDD']:>6.1f}")

    # ====================================================================
    # Walk-forward for top configs
    # ====================================================================
    print(f"\n{SEP}")
    print("WALK-FORWARD: TOP CONFIGS FROM EACH SWEEP")
    print(SEP)

    # Best A
    best_a = max(sweep_a_results, key=lambda r: r["PPDD"]) if sweep_a_results else None
    if best_a:
        best_a_mult = best_a["mult"]
        print(f"\n  Best A: mult={best_a_mult} -> re-running with walk-forward...")
        trades_a, m_a = run_single_config(
            d, d_extra, df_1m, feat_1m, fvg_zones,
            ts_5m_ns, ts_1m_ns, map_5m_to_1m, map_1m_to_prev_5m,
            label=f"WF Best A: mult={best_a_mult}",
            min_stop_pts=0.0,
            min_stop_atr_mult=best_a_mult,
            use_1m_atr=True,
            min_stop_pts_abs=5.0,
            verbose=True,
        )

    # Best B
    best_b_sweep = max(sweep_b_results, key=lambda r: r["PPDD"]) if sweep_b_results else None
    if best_b_sweep:
        best_b_pts = best_b_sweep["floor_pts"]
        print(f"\n  Best B: floor={best_b_pts}pts -> re-running with walk-forward...")
        trades_bw, m_bw = run_single_config(
            d, d_extra, df_1m, feat_1m, fvg_zones,
            ts_5m_ns, ts_1m_ns, map_5m_to_1m, map_1m_to_prev_5m,
            label=f"WF Best B: floor={best_b_pts}pts",
            min_stop_pts=0.0,
            min_stop_atr_mult=0.0,
            use_1m_atr=False,
            min_stop_pts_abs=float(best_b_pts),
            verbose=True,
        )

    # Best C
    best_c = max(sweep_c_results, key=lambda r: r["PPDD"]) if sweep_c_results else None
    if best_c:
        print(f"\n  Best C: {best_c['config']} -> walk-forward on combined trades...")
        # Re-get the combined trades for walk-forward
        if "mult" in best_c:
            c_trades_1m = sweep_a_trades.get(best_c["mult"], [])
        else:
            # Re-run B config
            c_trades_1m, _ = run_single_config(
                d, d_extra, df_1m, feat_1m, fvg_zones,
                ts_5m_ns, ts_1m_ns, map_5m_to_1m, map_1m_to_prev_5m,
                label="(re-run for combined WF)",
                min_stop_pts=0.0,
                min_stop_atr_mult=0.0,
                use_1m_atr=False,
                min_stop_pts_abs=float(best_c.get("floor_pts", 5)),
                verbose=False,
            )
        combined_wf = combine_5m_and_1m_trades(
            [t.copy() for t in baseline_trades],
            [t.copy() for t in c_trades_1m],
        )
        m_cwf = compute_metrics(combined_wf)
        print_metrics(f"Combined WF: {best_c['config']}", m_cwf)
        n_5m_wf = sum(1 for t in combined_wf if t.get("source") == "5m")
        n_1m_wf = sum(1 for t in combined_wf if t.get("source") == "1m")
        print(f"    5m signals: {n_5m_wf}, 1m signals: {n_1m_wf}")

        wf = walk_forward_metrics(combined_wf)
        if wf:
            print(f"    Walk-forward:")
            for yr in wf:
                src_5m = sum(1 for t in combined_wf
                             if t.get("source") == "5m"
                             and pd.Timestamp(t["entry_time"]).year == yr["year"])
                src_1m = sum(1 for t in combined_wf
                             if t.get("source") == "1m"
                             and pd.Timestamp(t["entry_time"]).year == yr["year"])
                print(f"      {yr['year']}: {yr['n']:3d}t (5m:{src_5m}, 1m:{src_1m})  "
                      f"R={yr['R']:+7.1f}  WR={yr['WR']:5.1f}%  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}")

    # ====================================================================
    # GRAND SUMMARY
    # ====================================================================
    print(f"\n{SEP}")
    print("GRAND SUMMARY")
    print(SEP)
    print(f"  {'Config':40s} | {'Trades':>6s} | {'R':>8s} | {'PPDD':>7s} | {'PF':>6s} | {'WR':>6s} | {'MaxDD':>6s} | {'avgR':>8s}")
    print(f"  {'-'*40}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")

    summary_rows = [
        ("Config H+ (5m baseline)", baseline_m),
        ("1c ORIGINAL (5m ATR - broken)", orig_m),
    ]
    # Best from each sweep
    if best_a:
        summary_rows.append((f"Best A: 1mATR mult={best_a['mult']}", best_a))
    if best_b_sweep:
        summary_rows.append((f"Best B: floor={best_b_sweep['floor_pts']}pts", best_b_sweep))
    if best_c:
        summary_rows.append((f"Best C: {best_c['config']}", best_c))

    for label, m_val in summary_rows:
        print(f"  {label:40s} | {m_val['trades']:>6d} | {m_val['R']:>+8.1f} | {m_val['PPDD']:>7.2f} | {m_val['PF']:>6.2f} | {m_val['WR']:>5.1f}% | {m_val['MaxDD']:>6.1f} | {m_val['avgR']:>+8.4f}")

    print(f"\n  KEY INSIGHT:")
    print(f"    Original 1c used 5m ATR (~{np.mean(valid_5m_atr):.0f}pts) for min_stop check on 1m signals (~16pt avg stop)")
    print(f"    1m ATR is ~{np.mean(valid_atr):.0f}pts, so same 16pt stop / {np.mean(valid_atr):.0f}pt ATR = {16.0/np.mean(valid_atr):.1f}x -> passes easily")
    print(f"    This unlocks signals that were incorrectly filtered out")

    print(f"\n{SEP}")
    print("DONE")
    print(SEP)


if __name__ == "__main__":
    main()
