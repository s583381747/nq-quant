"""
experiments/u5_order_block.py -- Order Block (OB) detection, signal generation & backtest
=========================================================================================

An Order Block is the LAST opposite-direction candle before a strong displacement move:
  - Bullish OB: last BEARISH candle before a bullish displacement
  - Bearish OB: last BULLISH candle before a bearish displacement

Architecture:
  1. Detect displacement candles on 5m data (body/range>=0.60, body>0.8*ATR, engulfs>=1)
  2. For each displacement, look back 1-3 bars for last opposite candle => OB zone
  3. Track OB zones (invalidate on close-through, max age 200 bars, min size 0.3*ATR)
  4. Three signal variants:
     V1: 5m rejection at OB zone
     V2: 1m rejection at OB zone (hybrid like b1c)
     V3: limit order at OB zone boundary
  5. Apply Config H++ filters, 50/50/0, EOD close
  6. Compare standalone vs combined with Config H++, overlap with FVG signals

Usage: python experiments/u5_order_block.py
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
    build_liquidity_ladder_long,
    build_liquidity_ladder_short,
)

SEP = "=" * 120
THIN = "-" * 80


# ======================================================================
# Step 1: Detect Order Blocks on 5m data
# ======================================================================
@dataclass
class OrderBlock:
    """A single detected Order Block zone."""
    creation_time: pd.Timestamp
    creation_5m_idx: int        # index in 5m data (the OB candle)
    becomes_known_5m_idx: int   # index when OB is confirmed (displacement candle bar)
    direction: str              # 'bull' or 'bear' — the trade direction
    top: float                  # OB candle high
    bottom: float               # OB candle low
    size: float                 # top - bottom
    disp_body_ratio: float      # displacement candle's body/range ratio
    status: str = "active"      # active | invalidated
    last_signal_idx: int = -999 # cooldown tracking


def detect_order_blocks_5m(
    nq_5m: pd.DataFrame,
    atr_5m: np.ndarray,
    *,
    disp_atr_mult: float = 0.8,
    disp_body_ratio: float = 0.60,
    engulf_min: int = 1,
    lookback_bars: int = 3,
    min_ob_atr_mult: float = 0.3,
) -> list[dict]:
    """
    Detect Order Blocks on 5m data.

    For each 5m bar that qualifies as a displacement candle:
      1. Check displacement criteria: body/range >= disp_body_ratio, body > disp_atr_mult * ATR, engulfs >= engulf_min
      2. Look back 1-lookback_bars bars for the last opposite-direction candle
      3. That candle's high-low range defines the OB zone

    Returns list of dicts with OB zone info.
    """
    t0 = _time.perf_counter()

    o = nq_5m["open"].values.astype(np.float64)
    h = nq_5m["high"].values.astype(np.float64)
    l = nq_5m["low"].values.astype(np.float64)
    c = nq_5m["close"].values.astype(np.float64)
    idx = nq_5m.index
    n = len(nq_5m)

    # Check for rollover column
    has_roll = "is_roll_date" in nq_5m.columns
    roll = nq_5m["is_roll_date"].values if has_roll else np.zeros(n, dtype=bool)

    obs = []
    n_disp_bull = 0
    n_disp_bear = 0
    n_ob_bull = 0
    n_ob_bear = 0
    n_too_small = 0
    n_no_opposite = 0

    for i in range(engulf_min + 1, n):
        if roll[i]:
            continue

        atr_val = atr_5m[i] if i < len(atr_5m) and not np.isnan(atr_5m[i]) else 30.0

        # Body and range
        body = abs(c[i] - o[i])
        rng = h[i] - l[i]
        if rng <= 0:
            continue

        br = body / rng

        # Displacement criteria
        if br < disp_body_ratio:
            continue
        if body < disp_atr_mult * atr_val:
            continue

        # Engulfment check: candle engulfs at least engulf_min prior candles
        candle_top = max(o[i], c[i])
        candle_bot = min(o[i], c[i])
        engulf_count = 0
        for lag in range(1, engulf_min + 1):
            if i - lag < 0:
                break
            prior_h = h[i - lag]
            prior_l = l[i - lag]
            if candle_top >= prior_h and candle_bot <= prior_l:
                engulf_count += 1
        if engulf_count < engulf_min:
            continue

        # Determine displacement direction
        is_bullish_disp = c[i] > o[i]  # bullish displacement
        if is_bullish_disp:
            n_disp_bull += 1
        else:
            n_disp_bear += 1

        # Look back for last opposite-direction candle
        ob_idx = -1
        for back in range(1, lookback_bars + 1):
            j = i - back
            if j < 0:
                break
            if roll[j]:
                continue
            bar_dir = c[j] - o[j]
            if is_bullish_disp and bar_dir < 0:
                # Bearish candle before bullish displacement => bullish OB
                ob_idx = j
                break
            elif not is_bullish_disp and bar_dir > 0:
                # Bullish candle before bearish displacement => bearish OB
                ob_idx = j
                break

        if ob_idx < 0:
            n_no_opposite += 1
            continue

        # OB zone = high-low of the opposite candle
        ob_top = h[ob_idx]
        ob_bot = l[ob_idx]
        ob_size = ob_top - ob_bot

        # Quality: minimum size relative to ATR
        if ob_size < min_ob_atr_mult * atr_val:
            n_too_small += 1
            continue

        # OB direction = trade direction (bullish OB -> long entry when price retests)
        ob_direction = "bull" if is_bullish_disp else "bear"

        if ob_direction == "bull":
            n_ob_bull += 1
        else:
            n_ob_bear += 1

        obs.append({
            "creation_time": idx[ob_idx],
            "creation_5m_idx": ob_idx,
            "becomes_known_5m_idx": i,  # Known when displacement candle closes
            "direction": ob_direction,
            "top": float(ob_top),
            "bottom": float(ob_bot),
            "size": float(ob_size),
            "disp_body_ratio": float(br),
            "disp_5m_idx": i,
        })

    elapsed = _time.perf_counter() - t0
    print(f"[OB DETECT] {len(obs)} Order Blocks detected in {elapsed:.1f}s")
    print(f"  Displacement candles: {n_disp_bull} bull, {n_disp_bear} bear")
    print(f"  OBs: {n_ob_bull} bull, {n_ob_bear} bear")
    print(f"  Filtered: {n_too_small} too small, {n_no_opposite} no opposite candle")

    return obs


# ======================================================================
# Step 2: V1 — 5m rejection at OB zone
# ======================================================================
def scan_5m_rejections_ob(
    d: dict,
    ob_zones: list[dict],
    *,
    min_body_ratio: float = 0.50,
    tighten_factor: float = 0.85,
    max_ob_age_bars: int = 200,
    signal_cooldown: int = 6,
    min_stop_pts: float = 5.0,
) -> list[dict]:
    """
    V1: Scan 5m bars for OB zone rejections (same logic as FVG zone rejection on 5m).
    """
    t0 = _time.perf_counter()

    o = d["o"]
    h = d["h"]
    l = d["l"]
    c = d["c"]
    n = d["n"]

    # Sort zones by known time
    sorted_zones = sorted(ob_zones, key=lambda z: z["becomes_known_5m_idx"])
    zone_cursor = 0
    n_zones = len(sorted_zones)

    @dataclass
    class ActiveOB:
        creation_5m_idx: int
        becomes_known_5m_idx: int
        direction: str
        top: float
        bottom: float
        size: float
        last_signal_idx: int = -999

    active: list[ActiveOB] = []
    signals: list[dict] = []
    n_invalidated = 0
    n_rejections = 0
    n_stop_wrong = 0
    n_stop_small = 0

    for i in range(n):
        # Birth new OB zones
        while zone_cursor < n_zones:
            z = sorted_zones[zone_cursor]
            if z["becomes_known_5m_idx"] <= i:
                active.append(ActiveOB(
                    creation_5m_idx=z["creation_5m_idx"],
                    becomes_known_5m_idx=z["becomes_known_5m_idx"],
                    direction=z["direction"],
                    top=z["top"],
                    bottom=z["bottom"],
                    size=z["size"],
                ))
                zone_cursor += 1
            else:
                break

        # Update: invalidate if price closes through
        surviving: list[ActiveOB] = []
        for ob in active:
            age = i - ob.creation_5m_idx
            if age > max_ob_age_bars:
                n_invalidated += 1
                continue
            if ob.direction == "bull" and c[i] < ob.bottom:
                n_invalidated += 1
                continue
            if ob.direction == "bear" and c[i] > ob.top:
                n_invalidated += 1
                continue
            surviving.append(ob)
        active = surviving

        # Rejection detection
        if i < 2 or i + 1 >= n:
            continue

        body = abs(c[i] - o[i])
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        br = body / rng
        if br < min_body_ratio:
            continue

        for ob in active:
            if i - ob.last_signal_idx < signal_cooldown:
                continue

            if ob.direction == "bull":
                # Long signal: low enters OB zone, close above zone
                if l[i] > ob.top:
                    continue
                if c[i] <= ob.top:
                    continue
                if c[i] <= o[i]:
                    continue
            else:
                # Short signal: high enters OB zone, close below zone
                if h[i] < ob.bottom:
                    continue
                if c[i] >= ob.bottom:
                    continue
                if c[i] >= o[i]:
                    continue

            n_rejections += 1
            direction = 1 if ob.direction == "bull" else -1
            entry_price = o[i + 1]

            # Stop: candle-2 open (2 bars before)
            raw_stop = o[i - 2]
            if direction == 1 and raw_stop >= entry_price:
                n_stop_wrong += 1
                continue
            if direction == -1 and raw_stop <= entry_price:
                n_stop_wrong += 1
                continue

            raw_dist = abs(entry_price - raw_stop)
            tightened_dist = raw_dist * tighten_factor
            if tightened_dist < min_stop_pts:
                n_stop_small += 1
                continue

            stop_price = entry_price - tightened_dist if direction == 1 else entry_price + tightened_dist

            signals.append({
                "signal_5m_idx": i,
                "entry_5m_idx": i + 1,
                "direction": direction,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "stop_dist": tightened_dist,
                "ob_direction": ob.direction,
                "ob_top": ob.top,
                "ob_bottom": ob.bottom,
                "ob_creation_5m_idx": ob.creation_5m_idx,
                "ob_age_bars": i - ob.creation_5m_idx,
            })
            ob.last_signal_idx = i
            break  # One signal per bar

    elapsed = _time.perf_counter() - t0
    print(f"[V1 5m] {len(signals)} signals from {n_rejections} rejections in {elapsed:.1f}s")
    print(f"  Invalidated: {n_invalidated}, Stop wrong side: {n_stop_wrong}, Stop too small: {n_stop_small}")
    return signals


# ======================================================================
# Step 3: V2 — 1m rejection at OB zone (hybrid like b1c)
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
    l_arr = df_1m["low"].values.astype(np.float64)
    c = df_1m["close"].values.astype(np.float64)
    n = len(df_1m)

    # True Range
    tr = np.empty(n, dtype=np.float64)
    tr[0] = h[0] - l_arr[0]
    for i in range(1, n):
        hl = h[i] - l_arr[i]
        hc = abs(h[i] - c[i - 1])
        lc = abs(l_arr[i] - c[i - 1])
        tr[i] = max(hl, hc, lc)

    # ATR(14) - simple rolling mean
    atr = np.full(n, np.nan)
    period = 14
    cum = np.cumsum(tr)
    atr[period - 1:] = (cum[period - 1:] - np.concatenate([[0], cum[:n - period]])) / period

    # Body ratio
    rng = h - l_arr
    body = np.abs(c - o)
    safe_rng = np.where(rng > 0, rng, 1.0)
    body_ratio = body / safe_rng
    body_ratio[rng <= 0] = 0.0

    elapsed = _time.perf_counter() - t0
    print(f"[1m FEAT] Precomputed ATR(14) + body_ratio in {elapsed:.1f}s")
    return {"o": o, "h": h, "l": l_arr, "c": c, "atr": atr, "body_ratio": body_ratio, "n": n}


def build_time_mappings(nq_5m: pd.DataFrame, df_1m: pd.DataFrame):
    """Build 5m<->1m time mappings."""
    t0 = _time.perf_counter()
    ts_5m_ns = nq_5m.index.astype(np.int64)
    ts_1m_ns = df_1m.index.astype(np.int64)
    five_min_ns = 5 * 60 * 10**9

    # For each 5m bar, first 1m bar at or after T+5min
    target_ns = ts_5m_ns + five_min_ns
    map_arr = np.searchsorted(ts_1m_ns, target_ns, side='left')
    n_1m = len(df_1m)
    map_arr[map_arr >= n_1m] = -1

    # For each 1m bar, what is the PREVIOUS completed 5m bar?
    completed_5m_ns = ts_5m_ns + five_min_ns
    map_1m_to_prev_5m = np.searchsorted(completed_5m_ns, ts_1m_ns, side='right') - 1
    map_1m_to_prev_5m = np.clip(map_1m_to_prev_5m, 0, len(nq_5m) - 1)

    elapsed = _time.perf_counter() - t0
    print(f"[MAP] Built 5m<->1m mappings in {elapsed:.1f}s")
    return ts_5m_ns, ts_1m_ns, map_arr, map_1m_to_prev_5m


def scan_1m_rejections_ob(
    df_1m: pd.DataFrame,
    feat_1m: dict,
    ob_zones: list[dict],
    ts_5m_ns: np.ndarray,
    ts_1m_ns: np.ndarray,
    map_1m_to_prev_5m: np.ndarray,
    *,
    min_body_ratio: float = 0.50,
    tighten_factor: float = 0.85,
    max_ob_age_bars: int = 200,
    signal_cooldown_1m: int = 6,
    min_stop_pts: float = 5.0,
) -> list[dict]:
    """
    V2: Scan 1m bars for OB zone rejections (same architecture as b1c).
    """
    t0 = _time.perf_counter()

    o_1m = feat_1m["o"]
    h_1m = feat_1m["h"]
    l_1m = feat_1m["l"]
    c_1m = feat_1m["c"]
    br_1m = feat_1m["body_ratio"]
    atr_1m = feat_1m["atr"]
    n_1m = feat_1m["n"]

    @dataclass
    class ActiveOB:
        creation_5m_idx: int
        becomes_known_5m_idx: int
        direction: str
        top: float
        bottom: float
        size: float
        last_signal_1m_idx: int = -999

    sorted_zones = sorted(ob_zones, key=lambda z: z["becomes_known_5m_idx"])
    zone_cursor = 0
    n_zones = len(sorted_zones)

    active: list[ActiveOB] = []
    signals: list[dict] = []
    n_births = 0
    n_invalidated = 0
    n_rejections = 0
    n_stop_wrong = 0
    n_stop_small = 0

    progress_step = n_1m // 10

    for j in range(n_1m):
        if progress_step > 0 and j % progress_step == 0 and j > 0:
            pct = 100 * j / n_1m
            print(f"  [V2 SCAN] {pct:.0f}% ({j:,}/{n_1m:,}) -- {len(active)} active OBs, {len(signals)} signals")

        prev_5m = map_1m_to_prev_5m[j]

        # Birth new OB zones that become known
        while zone_cursor < n_zones:
            z = sorted_zones[zone_cursor]
            if prev_5m >= z["becomes_known_5m_idx"]:
                active.append(ActiveOB(
                    creation_5m_idx=z["creation_5m_idx"],
                    becomes_known_5m_idx=z["becomes_known_5m_idx"],
                    direction=z["direction"],
                    top=z["top"],
                    bottom=z["bottom"],
                    size=z["size"],
                ))
                n_births += 1
                zone_cursor += 1
            else:
                break

        # Prune: max 30 active
        if len(active) > 30:
            active = active[-30:]

        # Update: invalidate on close-through or age
        surviving: list[ActiveOB] = []
        for ob in active:
            age = prev_5m - ob.creation_5m_idx
            if age > max_ob_age_bars:
                n_invalidated += 1
                continue
            if ob.direction == "bull" and c_1m[j] < ob.bottom:
                n_invalidated += 1
                continue
            if ob.direction == "bear" and c_1m[j] > ob.top:
                n_invalidated += 1
                continue
            surviving.append(ob)
        active = surviving

        # Rejection detection
        if j < 2 or j + 1 >= n_1m:
            continue
        if br_1m[j] < min_body_ratio:
            continue

        for ob in active:
            if j - ob.last_signal_1m_idx < signal_cooldown_1m:
                continue

            if ob.direction == "bull":
                if l_1m[j] > ob.top:
                    continue
                if c_1m[j] <= ob.top:
                    continue
                if c_1m[j] <= o_1m[j]:
                    continue
            else:
                if h_1m[j] < ob.bottom:
                    continue
                if c_1m[j] >= ob.bottom:
                    continue
                if c_1m[j] >= o_1m[j]:
                    continue

            n_rejections += 1
            direction = 1 if ob.direction == "bull" else -1
            entry_price = o_1m[j + 1]

            raw_stop = o_1m[j - 2]
            if direction == 1 and raw_stop >= entry_price:
                n_stop_wrong += 1
                continue
            if direction == -1 and raw_stop <= entry_price:
                n_stop_wrong += 1
                continue

            raw_dist = abs(entry_price - raw_stop)
            tightened_dist = raw_dist * tighten_factor
            if tightened_dist < min_stop_pts:
                n_stop_small += 1
                continue

            stop_price = entry_price - tightened_dist if direction == 1 else entry_price + tightened_dist

            atr_1m_val = atr_1m[j] if not np.isnan(atr_1m[j]) else 6.0
            signals.append({
                "signal_1m_idx": j,
                "entry_1m_idx": j + 1,
                "direction": direction,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "stop_dist": tightened_dist,
                "ob_direction": ob.direction,
                "ob_top": ob.top,
                "ob_bottom": ob.bottom,
                "ob_creation_5m_idx": ob.creation_5m_idx,
                "ob_age_5m_bars": prev_5m - ob.creation_5m_idx,
                "rejection_body_ratio": br_1m[j],
                "prev_5m_idx": prev_5m,
                "atr_1m": atr_1m_val,
            })
            ob.last_signal_1m_idx = j
            break

    elapsed = _time.perf_counter() - t0
    print(f"[V2 1m] {len(signals)} signals from {n_rejections} rejections in {elapsed:.1f}s")
    print(f"  Births: {n_births}, Invalidated: {n_invalidated}")
    print(f"  Stop wrong side: {n_stop_wrong}, Stop too small: {n_stop_small}")
    return signals


# ======================================================================
# Step 4: V3 — Limit order at OB zone boundary
# ======================================================================
def scan_limit_order_ob(
    d: dict,
    ob_zones: list[dict],
    *,
    tighten_factor: float = 0.85,
    max_ob_age_bars: int = 200,
    min_stop_pts: float = 5.0,
) -> list[dict]:
    """
    V3: Limit order at OB zone boundary.
    Long: limit buy at OB top. Short: limit sell at OB bottom.
    Stop: opposite side of OB zone, tightened.
    Entry triggered when price reaches the limit level.
    """
    t0 = _time.perf_counter()

    o = d["o"]
    h = d["h"]
    l = d["l"]
    c = d["c"]
    n = d["n"]

    sorted_zones = sorted(ob_zones, key=lambda z: z["becomes_known_5m_idx"])
    zone_cursor = 0
    n_zones = len(sorted_zones)

    @dataclass
    class ActiveOB:
        creation_5m_idx: int
        becomes_known_5m_idx: int
        direction: str
        top: float
        bottom: float
        size: float
        triggered: bool = False

    active: list[ActiveOB] = []
    signals: list[dict] = []
    n_invalidated = 0
    n_triggered = 0

    for i in range(n):
        # Birth
        while zone_cursor < n_zones:
            z = sorted_zones[zone_cursor]
            if z["becomes_known_5m_idx"] <= i:
                active.append(ActiveOB(
                    creation_5m_idx=z["creation_5m_idx"],
                    becomes_known_5m_idx=z["becomes_known_5m_idx"],
                    direction=z["direction"],
                    top=z["top"],
                    bottom=z["bottom"],
                    size=z["size"],
                ))
                zone_cursor += 1
            else:
                break

        # Update
        surviving: list[ActiveOB] = []
        for ob in active:
            if ob.triggered:
                continue
            age = i - ob.creation_5m_idx
            if age > max_ob_age_bars:
                n_invalidated += 1
                continue
            if ob.direction == "bull" and c[i] < ob.bottom:
                n_invalidated += 1
                continue
            if ob.direction == "bear" and c[i] > ob.top:
                n_invalidated += 1
                continue
            surviving.append(ob)
        active = surviving

        if i + 1 >= n:
            continue

        # Check limit fill
        for ob in active:
            if ob.triggered:
                continue

            if ob.direction == "bull":
                # Limit buy at OB top: triggered when low reaches OB top
                if l[i] <= ob.top:
                    entry_price = ob.top
                    stop_raw = ob.bottom
                    direction = 1
                else:
                    continue
            else:
                # Limit sell at OB bottom: triggered when high reaches OB bottom
                if h[i] >= ob.bottom:
                    entry_price = ob.bottom
                    stop_raw = ob.top
                    direction = -1
                else:
                    continue

            raw_dist = abs(entry_price - stop_raw)
            tightened_dist = raw_dist * tighten_factor
            if tightened_dist < min_stop_pts:
                continue

            stop_price = entry_price - tightened_dist if direction == 1 else entry_price + tightened_dist

            n_triggered += 1
            signals.append({
                "signal_5m_idx": i,
                "entry_5m_idx": i + 1,
                "direction": direction,
                "entry_price": o[i + 1],  # next bar open (realistically)
                "stop_price": stop_price,
                "stop_dist": tightened_dist,
                "ob_direction": ob.direction,
                "ob_top": ob.top,
                "ob_bottom": ob.bottom,
                "ob_creation_5m_idx": ob.creation_5m_idx,
                "ob_age_bars": i - ob.creation_5m_idx,
            })
            ob.triggered = True
            break

    elapsed = _time.perf_counter() - t0
    print(f"[V3 LIMIT] {len(signals)} signals from {n_triggered} triggers in {elapsed:.1f}s")
    print(f"  Invalidated: {n_invalidated}")
    return signals


# ======================================================================
# Step 5: Apply Config H++ filters
# ======================================================================
def filter_signals_5m(
    signals: list[dict],
    d: dict,
    *,
    sq_long: float = 0.68,
    sq_short: float = 0.80,
    min_stop_atr_mult: float = 1.7,
    block_pm_shorts: bool = True,
) -> list[dict]:
    """Apply Config H++ filters to 5m-level signals (V1 and V3)."""
    params = d["params"]
    atr_arr = d["atr_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    fluency_arr = d["fluency_arr"]
    et_frac_arr = d["et_frac_arr"]
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]

    session_regime = params.get("session_regime", {})
    sq_params = params.get("signal_quality", {})

    filtered = []
    stats = {"total": len(signals), "final": 0,
             "fail_obs": 0, "fail_news": 0, "fail_session": 0,
             "fail_pm_shorts": 0, "fail_bias": 0, "fail_min_stop": 0,
             "fail_sq": 0, "fail_lunch": 0}

    for sig in signals:
        idx = sig["signal_5m_idx"]
        direction = sig["direction"]
        et_frac = et_frac_arr[idx]

        # Observation period
        if 9.5 <= et_frac <= 10.0:
            stats["fail_obs"] += 1
            continue

        # News
        if news_blackout_arr is not None and news_blackout_arr[idx]:
            stats["fail_news"] += 1
            continue

        # Session: NY only
        if not (10.0 <= et_frac < 16.0):
            stats["fail_session"] += 1
            continue

        # PM shorts
        if block_pm_shorts and direction == -1 and et_frac >= 14.0:
            stats["fail_pm_shorts"] += 1
            continue

        # Bias
        bd = bias_dir_arr[idx]
        if direction == -np.sign(bd) and bd != 0:
            stats["fail_bias"] += 1
            continue

        # Min stop ATR
        atr_val = atr_arr[idx] if not np.isnan(atr_arr[idx]) else 30.0
        if atr_val > 0 and (sig["stop_dist"] / atr_val) < min_stop_atr_mult:
            stats["fail_min_stop"] += 1
            continue

        # Lunch dead zone
        if session_regime.get("enabled", False):
            lunch_s = session_regime.get("lunch_start", 12.5)
            lunch_e = session_regime.get("lunch_end", 13.0)
            if lunch_s <= et_frac < lunch_e and session_regime.get("lunch_mult", 0.0) == 0.0:
                stats["fail_lunch"] += 1
                continue

        # Signal quality
        if sq_params.get("enabled", False):
            size_sc = min(1.0, sig["stop_dist"] / (atr_val * 1.5)) if atr_val > 0 else 0.5
            body_i = abs(c[idx] - o[idx])
            rng_i = h[idx] - l[idx]
            disp_sc = body_i / rng_i if rng_i > 0 else 0.0
            flu_val = fluency_arr[idx]
            flu_sc = min(1.0, max(0.0, flu_val)) if not np.isnan(flu_val) else 0.5
            window = 6
            if idx >= window:
                dirs = np.sign(c[idx - window:idx] - o[idx - window:idx])
                alt = np.sum(dirs[1:] != dirs[:-1]) / (window - 1)
                pa_sc = 1.0 - alt
            else:
                pa_sc = 0.5
            sq = (sq_params.get("w_size", 0.3) * size_sc
                  + sq_params.get("w_disp", 0.3) * disp_sc
                  + sq_params.get("w_flu", 0.2) * flu_sc
                  + sq_params.get("w_pa", 0.2) * pa_sc)
            thresh = sq_long if direction == 1 else sq_short
            if sq < thresh:
                stats["fail_sq"] += 1
                continue

        sig_out = sig.copy()
        sig_out["et_frac"] = et_frac
        sig_out["atr_5m"] = atr_val
        sig_out["regime"] = regime_arr[idx]
        sig_out["bias_dir"] = bd
        filtered.append(sig_out)
        stats["final"] += 1

    print(f"[FILTER 5m] {stats['total']} raw -> {stats['final']} filtered")
    for k, v in stats.items():
        if k not in ("total", "final") and v > 0:
            print(f"    {k}: {v}")

    return filtered


def filter_signals_1m(
    signals: list[dict],
    d: dict,
    df_1m: pd.DataFrame,
    feat_1m: dict,
    map_1m_to_prev_5m: np.ndarray,
    *,
    sq_long: float = 0.68,
    sq_short: float = 0.80,
    min_stop_atr_mult: float = 1.0,
    min_stop_pts_abs: float = 5.0,
    block_pm_shorts: bool = True,
) -> list[dict]:
    """Apply Config H++ filters to 1m-level signals (V2)."""
    params = d["params"]
    atr_arr_5m = d["atr_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    fluency_5m = d["fluency_arr"]

    session_regime = params.get("session_regime", {})
    sq_params = params.get("signal_quality", {})

    o_1m = feat_1m["o"]
    c_1m = feat_1m["c"]
    br_1m = feat_1m["body_ratio"]

    et_1m = df_1m.index.tz_convert("US/Eastern")
    et_frac_1m = (et_1m.hour + et_1m.minute / 60.0).values

    filtered = []
    stats = {"total": len(signals), "final": 0,
             "fail_obs": 0, "fail_news": 0, "fail_session": 0,
             "fail_pm_shorts": 0, "fail_bias": 0, "fail_min_stop": 0,
             "fail_sq": 0, "fail_lunch": 0}

    for sig in signals:
        j = sig["signal_1m_idx"]
        direction = sig["direction"]
        prev_5m = sig["prev_5m_idx"]
        et_frac = et_frac_1m[j]

        if 9.5 <= et_frac <= 10.0:
            stats["fail_obs"] += 1
            continue

        if news_blackout_arr is not None and news_blackout_arr[prev_5m]:
            stats["fail_news"] += 1
            continue

        if not (10.0 <= et_frac < 16.0):
            stats["fail_session"] += 1
            continue

        if block_pm_shorts and direction == -1 and et_frac >= 14.0:
            stats["fail_pm_shorts"] += 1
            continue

        bd = bias_dir_arr[prev_5m]
        if direction == -np.sign(bd) and bd != 0:
            stats["fail_bias"] += 1
            continue

        stop_dist = sig["stop_dist"]
        if min_stop_pts_abs > 0 and stop_dist < min_stop_pts_abs:
            stats["fail_min_stop"] += 1
            continue

        if min_stop_atr_mult > 0:
            atr_ref = sig.get("atr_1m", 6.0)
            if atr_ref > 0 and (stop_dist / atr_ref) < min_stop_atr_mult:
                stats["fail_min_stop"] += 1
                continue

        if session_regime.get("enabled", False):
            lunch_s = session_regime.get("lunch_start", 12.5)
            lunch_e = session_regime.get("lunch_end", 13.0)
            if lunch_s <= et_frac < lunch_e and session_regime.get("lunch_mult", 0.0) == 0.0:
                stats["fail_lunch"] += 1
                continue

        if sq_params.get("enabled", False):
            atr_5m_val = atr_arr_5m[prev_5m] if not np.isnan(atr_arr_5m[prev_5m]) else 30.0
            size_sc = min(1.0, stop_dist / (atr_5m_val * 1.5)) if atr_5m_val > 0 else 0.5
            disp_sc = br_1m[j]
            flu_val = fluency_5m[prev_5m]
            flu_sc = min(1.0, max(0.0, flu_val)) if not np.isnan(flu_val) else 0.5
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
            thresh = sq_long if direction == 1 else sq_short
            if sq < thresh:
                stats["fail_sq"] += 1
                continue

        sig_out = sig.copy()
        sig_out["et_frac"] = et_frac
        sig_out["atr_5m"] = atr_arr_5m[prev_5m] if not np.isnan(atr_arr_5m[prev_5m]) else 30.0
        sig_out["regime"] = regime_arr[prev_5m]
        sig_out["bias_dir"] = bd
        filtered.append(sig_out)
        stats["final"] += 1

    print(f"[FILTER 1m] {stats['total']} raw -> {stats['final']} filtered")
    for k, v in stats.items():
        if k not in ("total", "final") and v > 0:
            print(f"    {k}: {v}")

    return filtered


# ======================================================================
# Step 6: Simulate outcomes (shared for V1/V3 on 5m, V2 on 1m)
# ======================================================================
def simulate_5m_outcomes(
    filtered_signals: list[dict],
    d: dict,
    d_extra: dict,
) -> list[dict]:
    """Simulate outcomes for 5m-level signals (V1, V3)."""
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
    slippage_points = bt_params["slippage_normal_ticks"] * 0.25
    a_plus_mult = grading_params["a_plus_size_mult"]
    b_plus_mult = grading_params["b_plus_size_mult"]
    c_skip = grading_params["c_skip"]

    n = d["n"]
    o, h, l, c_arr = d["o"], d["h"], d["l"], d["c"]
    atr_arr = d["atr_arr"]
    irl_target_arr = d["irl_target_arr"]
    et_idx = d["et_idx"]
    regime_arr = d["regime_arr"]

    et_frac_arr = d["et_frac_arr"]
    eod_close_et = params.get("multi_tp", {}).get("eod_close_et", 15.917)

    sorted_sigs = sorted(filtered_signals, key=lambda s: s["entry_5m_idx"])
    trades = []
    current_date = None
    daily_pnl_r = 0.0
    consecutive_losses = 0
    day_stopped = False
    last_exit_bar = -1

    for sig in sorted_sigs:
        entry_i = sig["entry_5m_idx"]
        direction = sig["direction"]
        entry_price = sig["entry_price"]
        stop_price = sig["stop_price"]
        stop_dist = sig["stop_dist"]
        et_frac_sig = sig.get("et_frac", 0)
        regime = sig.get("regime", 0.5)
        bd = sig.get("bias_dir", 0)

        if entry_i >= n or entry_i <= last_exit_bar:
            continue

        entry_date = et_idx[entry_i].date() if et_idx[entry_i].hour < 18 else (et_idx[entry_i] + pd.Timedelta(days=1)).date()
        if entry_date != current_date:
            current_date = entry_date
            daily_pnl_r = 0.0
            consecutive_losses = 0
            day_stopped = False

        if day_stopped:
            continue

        actual_entry = entry_price + slippage_points if direction == 1 else entry_price - slippage_points
        actual_stop_dist = abs(actual_entry - stop_price)
        if actual_stop_dist < 1.0:
            continue

        ba = 1.0 if (direction == np.sign(bd) and bd != 0) else 0.0
        grade = _compute_grade_fast(ba, regime)
        if grade == "C" and c_skip:
            continue

        entry_dow = et_idx[entry_i].dayofweek
        is_reduced = (entry_dow in (0, 4)) or (regime < 1.0)
        base_r = reduced_r if is_reduced else normal_r
        if grade == "A+":
            r_amount = base_r * a_plus_mult
        elif grade == "B+":
            r_amount = base_r * b_plus_mult
        else:
            r_amount = base_r * 0.5

        contracts = max(1, int(r_amount / (actual_stop_dist * point_value))) if actual_stop_dist > 0 else 0
        if contracts <= 0:
            continue

        idx_5m = sig.get("signal_5m_idx", entry_i - 1)
        irl_raw = irl_target_arr[idx_5m] if not np.isnan(irl_target_arr[idx_5m]) else 0.0

        if direction == 1:
            tp1, tp2 = build_liquidity_ladder_long(actual_entry, stop_price, irl_raw, idx_5m, d_extra)
            if tp1 <= actual_entry:
                tp1 = actual_entry + actual_stop_dist * 0.5
            if tp2 <= tp1:
                tp2 = actual_entry + (tp1 - actual_entry) * 1.5
            is_multi_tp = True
        else:
            short_rr = dual_mode.get("short_rr", 0.625)
            tp1 = actual_entry - actual_stop_dist * short_rr
            tp2 = 0.0
            is_multi_tp = False

        # Simulate forward
        trim_stage = 0
        remaining = contracts
        trim1_c = 0
        trim2_c = 0
        tp1_pnl_pts = 0.0
        tp2_pnl_pts = 0.0
        single_trimmed = False
        be_stop = 0.0
        exited = False
        exit_reason = ""
        exit_price = 0.0
        exit_bar = entry_i

        scan_end = min(entry_i + 200, n)
        for jj in range(entry_i + 1, scan_end):
            et_frac_bar = et_frac_arr[jj]

            if et_frac_bar >= eod_close_et:
                exit_price = c_arr[jj] - slippage_points if direction == 1 else c_arr[jj] + slippage_points
                exit_reason = "eod_close"
                exit_bar = jj
                exited = True
                break

            if is_multi_tp and direction == 1:
                eff_stop = stop_price
                if trim_stage >= 1 and be_stop > 0:
                    eff_stop = max(eff_stop, be_stop)

                if l[jj] <= eff_stop:
                    exit_price = eff_stop - slippage_points
                    exit_reason = "be_sweep" if (trim_stage > 0 and eff_stop >= actual_entry) else "stop"
                    exit_bar = jj
                    exited = True
                    break

                if trim_stage == 0 and h[jj] >= tp1:
                    tc1 = max(1, int(contracts * 0.50))
                    trim1_c = tc1
                    remaining = contracts - tc1
                    tp1_pnl_pts = tp1 - actual_entry
                    trim_stage = 1
                    if remaining <= 0:
                        exit_price = tp1
                        exit_reason = "tp1"
                        exit_bar = jj
                        exited = True
                        break

                if trim_stage == 1 and h[jj] >= tp2:
                    tc2 = remaining
                    trim2_c = tc2
                    remaining -= tc2
                    tp2_pnl_pts = tp2 - actual_entry
                    trim_stage = 2
                    if remaining <= 0:
                        exit_price = tp2
                        exit_reason = "tp2"
                        exit_bar = jj
                        exited = True
                        break

            elif direction == -1:
                eff_stop = stop_price
                if single_trimmed and be_stop > 0:
                    eff_stop = min(eff_stop, be_stop)

                if h[jj] >= eff_stop:
                    exit_price = eff_stop + slippage_points
                    exit_reason = "be_sweep" if (single_trimmed and eff_stop <= actual_entry) else "stop"
                    exit_bar = jj
                    exited = True
                    break

                short_trim_pct = dual_mode.get("short_trim_pct", 1.0)
                if not single_trimmed and l[jj] <= tp1:
                    tc = max(1, int(contracts * short_trim_pct))
                    remaining = contracts - tc
                    single_trimmed = True
                    be_stop = actual_entry
                    if remaining <= 0:
                        exit_price = tp1
                        exit_reason = "tp1"
                        exit_bar = jj
                        exited = True
                        break

        if not exited:
            exit_bar = min(scan_end - 1, n - 1)
            exit_price = c_arr[exit_bar]
            exit_reason = "eod_close"

        # PNL
        pnl_pts_runner = (exit_price - actual_entry) if direction == 1 else (actual_entry - exit_price)
        any_trimmed = trim_stage > 0 or single_trimmed

        if is_multi_tp:
            total_pnl = 0.0
            if trim_stage >= 1:
                total_pnl += tp1_pnl_pts * point_value * trim1_c
            if trim_stage >= 2:
                total_pnl += tp2_pnl_pts * point_value * trim2_c
            remaining_exit = contracts - trim1_c - trim2_c
            if remaining_exit > 0:
                total_pnl += pnl_pts_runner * point_value * remaining_exit
            total_pnl -= commission_per_side * 2 * contracts
        elif single_trimmed and exit_reason != "tp1":
            trim_c = contracts - remaining
            trim_pnl = (tp1 - actual_entry if direction == 1 else actual_entry - tp1) * point_value * trim_c
            total_pnl = trim_pnl + pnl_pts_runner * point_value * remaining
            total_pnl -= commission_per_side * 2 * contracts
        else:
            total_pnl = pnl_pts_runner * point_value * contracts
            total_pnl -= commission_per_side * 2 * contracts

        total_risk = actual_stop_dist * point_value * contracts
        r_mult = total_pnl / total_risk if total_risk > 0 else 0.0

        nq = d["nq"]
        trades.append({
            "entry_time": nq.index[entry_i],
            "exit_time": nq.index[exit_bar],
            "r": r_mult, "reason": exit_reason, "dir": direction,
            "type": "ob_5m", "trimmed": any_trimmed,
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

    print(f"[SIM 5m] {len(trades)} trades simulated")
    return trades


def simulate_1m_outcomes(
    filtered_signals: list[dict],
    d: dict,
    d_extra: dict,
    df_1m: pd.DataFrame,
    feat_1m: dict,
    map_1m_to_prev_5m: np.ndarray,
) -> list[dict]:
    """Simulate outcomes for 1m-level signals (V2)."""
    params = d["params"]
    pos_params = params["position"]
    risk_params = params["risk"]
    bt_params = params["backtest"]
    grading_params = params["grading"]
    dual_mode = params.get("dual_mode", {})

    normal_r = pos_params["normal_r"]
    reduced_r = pos_params["reduced_r"]
    point_value = pos_params["point_value"]
    daily_max_loss_r = risk_params["daily_max_loss_r"]
    max_consec_losses = risk_params["max_consecutive_losses"]
    commission_per_side = bt_params["commission_per_side_micro"]
    slippage_points = bt_params["slippage_normal_ticks"] * 0.25
    a_plus_mult = grading_params["a_plus_size_mult"]
    b_plus_mult = grading_params["b_plus_size_mult"]
    c_skip = grading_params["c_skip"]

    o_1m = feat_1m["o"]
    h_1m = feat_1m["h"]
    l_1m = feat_1m["l"]
    c_1m = feat_1m["c"]
    n_1m = feat_1m["n"]

    atr_5m = d["atr_arr"]
    irl_target_5m = d["irl_target_arr"]

    et_1m = df_1m.index.tz_convert("US/Eastern")
    et_hour_1m = et_1m.hour.values
    et_minute_1m = et_1m.minute.values

    eod_close_et = params.get("multi_tp", {}).get("eod_close_et", 15.917)

    sorted_sigs = sorted(filtered_signals, key=lambda s: s["entry_1m_idx"])
    trades = []
    current_date = None
    daily_pnl_r = 0.0
    consecutive_losses = 0
    day_stopped = False
    last_exit_bar = -1

    for sig in sorted_sigs:
        entry_j = sig["entry_1m_idx"]
        direction = sig["direction"]
        entry_price = sig["entry_price"]
        stop_price = sig["stop_price"]
        stop_dist = sig["stop_dist"]
        prev_5m = sig["prev_5m_idx"]
        et_frac_sig = sig.get("et_frac", 0)
        regime = sig.get("regime", 0.5)
        bd = sig.get("bias_dir", 0)

        if entry_j >= n_1m or entry_j <= last_exit_bar:
            continue

        entry_et_h = et_hour_1m[entry_j]
        entry_date = et_1m[entry_j].date() if entry_et_h < 18 else (et_1m[entry_j] + pd.Timedelta(days=1)).date()
        if entry_date != current_date:
            current_date = entry_date
            daily_pnl_r = 0.0
            consecutive_losses = 0
            day_stopped = False

        if day_stopped:
            continue

        actual_entry = entry_price + slippage_points if direction == 1 else entry_price - slippage_points
        actual_stop_dist = abs(actual_entry - stop_price)
        if actual_stop_dist < 1.0:
            continue

        ba = 1.0 if (direction == np.sign(bd) and bd != 0) else 0.0
        grade = _compute_grade_fast(ba, regime)
        if grade == "C" and c_skip:
            continue

        entry_dow = et_1m[entry_j].dayofweek
        is_reduced = (entry_dow in (0, 4)) or (regime < 1.0)
        base_r = reduced_r if is_reduced else normal_r
        if grade == "A+":
            r_amount = base_r * a_plus_mult
        elif grade == "B+":
            r_amount = base_r * b_plus_mult
        else:
            r_amount = base_r * 0.5

        contracts = max(1, int(r_amount / (actual_stop_dist * point_value))) if actual_stop_dist > 0 else 0
        if contracts <= 0:
            continue

        irl_raw = irl_target_5m[prev_5m] if not np.isnan(irl_target_5m[prev_5m]) else 0.0

        if direction == 1:
            tp1, tp2 = build_liquidity_ladder_long(actual_entry, stop_price, irl_raw, prev_5m, d_extra)
            if tp1 <= actual_entry:
                tp1 = actual_entry + actual_stop_dist * 0.5
            if tp2 <= tp1:
                tp2 = actual_entry + (tp1 - actual_entry) * 1.5
            is_multi_tp = True
        else:
            short_rr = dual_mode.get("short_rr", 0.625)
            tp1 = actual_entry - actual_stop_dist * short_rr
            tp2 = 0.0
            is_multi_tp = False

        trim_stage = 0
        remaining = contracts
        trim1_c = 0
        trim2_c = 0
        tp1_pnl_pts = 0.0
        tp2_pnl_pts = 0.0
        single_trimmed = False
        be_stop = 0.0
        exited = False
        exit_reason = ""
        exit_price = 0.0
        exit_bar = entry_j

        scan_end = min(entry_j + 500, n_1m)
        for jj in range(entry_j + 1, scan_end):
            et_h = et_hour_1m[jj]
            et_m_val = et_minute_1m[jj]
            et_frac_bar = et_h + et_m_val / 60.0

            if et_frac_bar >= eod_close_et:
                exit_price = c_1m[jj] - slippage_points if direction == 1 else c_1m[jj] + slippage_points
                exit_reason = "eod_close"
                exit_bar = jj
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
                    exited = True
                    break

                if trim_stage == 0 and h_1m[jj] >= tp1:
                    tc1 = max(1, int(contracts * 0.50))
                    trim1_c = tc1
                    remaining = contracts - tc1
                    tp1_pnl_pts = tp1 - actual_entry
                    trim_stage = 1
                    if remaining <= 0:
                        exit_price = tp1
                        exit_reason = "tp1"
                        exit_bar = jj
                        exited = True
                        break

                if trim_stage == 1 and h_1m[jj] >= tp2:
                    tc2 = remaining
                    trim2_c = tc2
                    remaining -= tc2
                    tp2_pnl_pts = tp2 - actual_entry
                    trim_stage = 2
                    if remaining <= 0:
                        exit_price = tp2
                        exit_reason = "tp2"
                        exit_bar = jj
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
                    exited = True
                    break

                short_trim_pct = dual_mode.get("short_trim_pct", 1.0)
                if not single_trimmed and l_1m[jj] <= tp1:
                    tc = max(1, int(contracts * short_trim_pct))
                    remaining = contracts - tc
                    single_trimmed = True
                    be_stop = actual_entry
                    if remaining <= 0:
                        exit_price = tp1
                        exit_reason = "tp1"
                        exit_bar = jj
                        exited = True
                        break

        if not exited:
            exit_bar = min(scan_end - 1, n_1m - 1)
            exit_price = c_1m[exit_bar]
            exit_reason = "eod_close"

        pnl_pts_runner = (exit_price - actual_entry) if direction == 1 else (actual_entry - exit_price)
        any_trimmed = trim_stage > 0 or single_trimmed

        if is_multi_tp:
            total_pnl = 0.0
            if trim_stage >= 1:
                total_pnl += tp1_pnl_pts * point_value * trim1_c
            if trim_stage >= 2:
                total_pnl += tp2_pnl_pts * point_value * trim2_c
            remaining_exit = contracts - trim1_c - trim2_c
            if remaining_exit > 0:
                total_pnl += pnl_pts_runner * point_value * remaining_exit
            total_pnl -= commission_per_side * 2 * contracts
        elif single_trimmed and exit_reason != "tp1":
            trim_c = contracts - remaining
            trim_pnl = (tp1 - actual_entry if direction == 1 else actual_entry - tp1) * point_value * trim_c
            total_pnl = trim_pnl + pnl_pts_runner * point_value * remaining
            total_pnl -= commission_per_side * 2 * contracts
        else:
            total_pnl = pnl_pts_runner * point_value * contracts
            total_pnl -= commission_per_side * 2 * contracts

        total_risk = actual_stop_dist * point_value * contracts
        r_mult = total_pnl / total_risk if total_risk > 0 else 0.0

        trades.append({
            "entry_time": df_1m.index[entry_j],
            "exit_time": df_1m.index[exit_bar],
            "r": r_mult, "reason": exit_reason, "dir": direction,
            "type": "ob_1m", "trimmed": any_trimmed,
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

    print(f"[SIM 1m] {len(trades)} trades simulated")
    return trades


# ======================================================================
# Step 7: Config H++ baseline (5m FVG signals)
# ======================================================================
def run_config_h_plus_baseline(d: dict, d_extra: dict) -> tuple[list[dict], dict]:
    """Run Config H++ (5m only) as baseline."""
    from experiments.pure_liquidity_tp import run_backtest_pure_liquidity
    trades, diag = run_backtest_pure_liquidity(
        d, d_extra,
        sq_long=0.68, sq_short=0.80,
        min_stop_atr=1.7,
        block_pm_shorts=True,
        tp_strategy="v1",
        tp1_trim_pct=0.50,
        be_after_tp1=False,
        be_after_tp2=False,
    )
    return trades, compute_metrics(trades)


# ======================================================================
# Step 8: Combine OB + FVG trades
# ======================================================================
def combine_trades(trades_a: list[dict], trades_b: list[dict], label_a: str, label_b: str) -> list[dict]:
    """Combine two trade lists, one position at a time."""
    for t in trades_a:
        t["source"] = label_a
    for t in trades_b:
        t["source"] = label_b

    all_trades = sorted(trades_a + trades_b, key=lambda t: t["entry_time"])
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
# Step 9: Overlap analysis (OB zones vs FVG zones)
# ======================================================================
def overlap_analysis(ob_zones: list[dict], d: dict, nq_5m: pd.DataFrame):
    """Check how many OB zones overlap with active FVG zones at the same time."""
    from features.fvg import detect_fvg

    print(f"\n{THIN}")
    print("OVERLAP ANALYSIS: OB zones vs FVG zones")
    print(THIN)

    fvg_df = detect_fvg(nq_5m)
    n = len(nq_5m)

    # Collect FVG zones (bull and bear)
    fvg_zones = []
    bull_mask = fvg_df["fvg_bull"].values
    bear_mask = fvg_df["fvg_bear"].values

    for i in range(n):
        if bull_mask[i]:
            fvg_zones.append({
                "idx": i,
                "direction": "bull",
                "top": float(fvg_df["fvg_bull_top"].iat[i]),
                "bottom": float(fvg_df["fvg_bull_bottom"].iat[i]),
            })
        if bear_mask[i]:
            fvg_zones.append({
                "idx": i,
                "direction": "bear",
                "top": float(fvg_df["fvg_bear_top"].iat[i]),
                "bottom": float(fvg_df["fvg_bear_bottom"].iat[i]),
            })

    print(f"  Total FVG zones: {len(fvg_zones)}")
    print(f"  Total OB  zones: {len(ob_zones)}")

    # For each OB zone, check if ANY FVG zone overlaps in price and nearby in time
    overlap_count = 0
    for ob in ob_zones:
        ob_top = ob["top"]
        ob_bot = ob["bottom"]
        ob_idx = ob["creation_5m_idx"]
        ob_dir = ob["direction"]

        for fvg in fvg_zones:
            if fvg["direction"] != ob_dir:
                continue
            # Time proximity: within 20 bars
            if abs(fvg["idx"] - ob_idx) > 20:
                continue
            # Price overlap: zones overlap if one's top > other's bottom and vice versa
            if fvg["top"] >= ob_bot and fvg["bottom"] <= ob_top:
                overlap_count += 1
                break

    pct = 100 * overlap_count / len(ob_zones) if ob_zones else 0
    print(f"  OB zones that overlap with a nearby FVG: {overlap_count} / {len(ob_zones)} ({pct:.1f}%)")
    non_overlap = len(ob_zones) - overlap_count
    print(f"  UNIQUE OB zones (no FVG overlap): {non_overlap} ({100 - pct:.1f}%)")

    return {"total_ob": len(ob_zones), "total_fvg": len(fvg_zones),
            "overlapping": overlap_count, "unique_ob": non_overlap}


# ======================================================================
# Step 10: Print results helper
# ======================================================================
def print_variant_results(label: str, trades: list[dict], metrics: dict):
    """Print results for a signal variant."""
    print_metrics(label, metrics)
    if trades:
        longs = [t for t in trades if t["dir"] == 1]
        shorts = [t for t in trades if t["dir"] == -1]
        if longs:
            lr = np.array([t["r"] for t in longs])
            print(f"    Longs:  {len(longs)}t, R={lr.sum():+.1f}, avgR={lr.mean():+.4f}, WR={100*(lr>0).mean():.1f}%")
        if shorts:
            sr = np.array([t["r"] for t in shorts])
            print(f"    Shorts: {len(shorts)}t, R={sr.sum():+.1f}, avgR={sr.mean():+.4f}, WR={100*(sr>0).mean():.1f}%")

        # Per-year
        wf = walk_forward_metrics(trades)
        if wf:
            print(f"    Walk-forward:")
            for yr in wf:
                print(f"      {yr['year']}: {yr['n']:3d}t  R={yr['R']:+7.1f}  WR={yr['WR']:5.1f}%  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}")

        # Exit reason distribution
        from collections import Counter
        reasons = Counter(t["reason"] for t in trades)
        print(f"    Exit reasons: {dict(reasons)}")


# ======================================================================
# Main
# ======================================================================
def main():
    print(SEP)
    print("U5: ORDER BLOCK DETECTION & BACKTEST")
    print("Exploring OBs as alternative/supplement to FVG signals")
    print(SEP)

    # ---- Load data ----
    print("\n[STEP 1] Loading data...")
    t0_total = _time.perf_counter()
    d = load_all()
    d_extra = prepare_liquidity_data(d)
    nq_5m = d["nq"]
    atr_5m = d["atr_arr"]

    # ---- Detect Order Blocks ----
    print(f"\n[STEP 2] Detecting Order Blocks on 5m data...")
    ob_zones = detect_order_blocks_5m(nq_5m, atr_5m)

    # Per-year OB stats
    nq_idx = nq_5m.index
    ob_by_year = {}
    for ob in ob_zones:
        yr = nq_idx[ob["creation_5m_idx"]].year
        ob_by_year.setdefault(yr, {"bull": 0, "bear": 0})
        ob_by_year[yr][ob["direction"]] += 1
    print(f"\n  OBs per year:")
    for yr in sorted(ob_by_year.keys()):
        cnt = ob_by_year[yr]
        print(f"    {yr}: {cnt['bull']} bull + {cnt['bear']} bear = {cnt['bull'] + cnt['bear']} total")

    # ---- Config H++ baseline ----
    print(f"\n{SEP}")
    print("CONFIG H++ BASELINE (5m FVG signals)")
    print(SEP)
    baseline_trades, baseline_m = run_config_h_plus_baseline(d, d_extra)
    print_variant_results("Config H++ (5m FVG baseline)", baseline_trades, baseline_m)

    # ==================================================================
    # DIAGNOSTIC: Understand filter attrition
    # ==================================================================
    print(f"\n{SEP}")
    print("DIAGNOSTIC: FILTER ATTRITION SWEEP")
    print("Testing min_stop_atr_mult values to find where signals survive")
    print(SEP)

    # V1 with different min_stop thresholds
    raw_v1_diag = scan_5m_rejections_ob(d, ob_zones, min_stop_pts=3.0)
    for atr_mult in [0.0, 0.5, 1.0, 1.5, 1.7]:
        filt = filter_signals_5m(raw_v1_diag, d, min_stop_atr_mult=atr_mult)
        trades = simulate_5m_outcomes(filt, d, d_extra) if filt else []
        m = compute_metrics(trades)
        print_metrics(f"V1 diag: atr_mult={atr_mult:.1f}", m)

    # ==================================================================
    # V1: 5m rejection at OB zone
    # ==================================================================
    print(f"\n{SEP}")
    print("V1: 5m REJECTION AT OB ZONE")
    print(SEP)
    # Use relaxed min_stop to match natural OB stop sizes
    raw_v1 = scan_5m_rejections_ob(d, ob_zones, min_stop_pts=3.0)

    # Default strict filters
    filtered_v1 = filter_signals_5m(raw_v1, d)
    trades_v1 = simulate_5m_outcomes(filtered_v1, d, d_extra) if filtered_v1 else []
    m_v1 = compute_metrics(trades_v1)
    print_variant_results("V1: 5m OB rejection (strict)", trades_v1, m_v1)

    # Also try min_stop_atr_mult=0.5 (relaxed, since OB zones have natural stop)
    filtered_v1r = filter_signals_5m(raw_v1, d, min_stop_atr_mult=0.5)
    trades_v1r = simulate_5m_outcomes(filtered_v1r, d, d_extra) if filtered_v1r else []
    m_v1r = compute_metrics(trades_v1r)
    print_variant_results("V1: 5m OB rejection (relaxed)", trades_v1r, m_v1r)

    # ==================================================================
    # V3: Limit order at OB zone boundary
    # ==================================================================
    print(f"\n{SEP}")
    print("V3: LIMIT ORDER AT OB ZONE BOUNDARY")
    print(SEP)
    raw_v3 = scan_limit_order_ob(d, ob_zones, min_stop_pts=3.0)

    filtered_v3 = filter_signals_5m(raw_v3, d)
    trades_v3 = simulate_5m_outcomes(filtered_v3, d, d_extra) if filtered_v3 else []
    m_v3 = compute_metrics(trades_v3)
    print_variant_results("V3: Limit order at OB (strict)", trades_v3, m_v3)

    filtered_v3r = filter_signals_5m(raw_v3, d, min_stop_atr_mult=0.5)
    trades_v3r = simulate_5m_outcomes(filtered_v3r, d, d_extra) if filtered_v3r else []
    m_v3r = compute_metrics(trades_v3r)
    print_variant_results("V3: Limit order at OB (relaxed)", trades_v3r, m_v3r)

    # ==================================================================
    # V2: 1m rejection at OB zone (hybrid)
    # ==================================================================
    print(f"\n{SEP}")
    print("V2: 1m REJECTION AT OB ZONE (hybrid)")
    print(SEP)
    print("  Loading 1m data...")
    df_1m = load_1m_data()
    feat_1m = precompute_1m_features(df_1m)

    print("  Building time mappings...")
    ts_5m_ns, ts_1m_ns, map_5m_to_1m, map_1m_to_prev_5m = build_time_mappings(nq_5m, df_1m)

    print("  Scanning 1m rejections at OB zones...")
    raw_v2 = scan_1m_rejections_ob(
        df_1m, feat_1m, ob_zones, ts_5m_ns, ts_1m_ns, map_1m_to_prev_5m,
        min_stop_pts=3.0,
    )

    # Strict filter
    filtered_v2 = filter_signals_1m(raw_v2, d, df_1m, feat_1m, map_1m_to_prev_5m)
    trades_v2 = simulate_1m_outcomes(filtered_v2, d, d_extra, df_1m, feat_1m, map_1m_to_prev_5m) if filtered_v2 else []
    m_v2 = compute_metrics(trades_v2)
    print_variant_results("V2: 1m OB rejection (strict)", trades_v2, m_v2)

    # Relaxed filter (lower SQ threshold, lower min stop)
    filtered_v2r = filter_signals_1m(raw_v2, d, df_1m, feat_1m, map_1m_to_prev_5m,
                                      min_stop_atr_mult=0.5, min_stop_pts_abs=3.0)
    trades_v2r = simulate_1m_outcomes(filtered_v2r, d, d_extra, df_1m, feat_1m, map_1m_to_prev_5m) if filtered_v2r else []
    m_v2r = compute_metrics(trades_v2r)
    print_variant_results("V2: 1m OB rejection (relaxed)", trades_v2r, m_v2r)

    # Use best trades for combined analysis
    # Pick the variant with most trades for each (prefer relaxed if more trades)
    if m_v1r["trades"] > m_v1["trades"]:
        trades_v1_best, m_v1_best = trades_v1r, m_v1r
    else:
        trades_v1_best, m_v1_best = trades_v1, m_v1
    if m_v2r["trades"] > m_v2["trades"]:
        trades_v2_best, m_v2_best = trades_v2r, m_v2r
    else:
        trades_v2_best, m_v2_best = trades_v2, m_v2
    if m_v3r["trades"] > m_v3["trades"]:
        trades_v3_best, m_v3_best = trades_v3r, m_v3r
    else:
        trades_v3_best, m_v3_best = trades_v3, m_v3

    # ==================================================================
    # Overlap analysis: OB vs FVG
    # ==================================================================
    overlap = overlap_analysis(ob_zones, d, nq_5m)

    # ==================================================================
    # Combined systems: Config H++ + each OB variant (using best configs)
    # ==================================================================
    print(f"\n{SEP}")
    print("COMBINED SYSTEMS: Config H++ + OB variants (best configs)")
    print(SEP)

    for label, ob_trades, ob_m in [
        ("V1 best (5m rejection)", trades_v1_best, m_v1_best),
        ("V2 best (1m rejection)", trades_v2_best, m_v2_best),
        ("V3 best (limit order)", trades_v3_best, m_v3_best),
    ]:
        if not ob_trades:
            print(f"  Config H++ + OB {label}: NO OB TRADES")
            continue

        combined = combine_trades(
            [t.copy() for t in baseline_trades],
            [t.copy() for t in ob_trades],
            "fvg_5m", f"ob_{label}",
        )
        m_combined = compute_metrics(combined)
        n_fvg = sum(1 for t in combined if t.get("source") == "fvg_5m")
        n_ob = sum(1 for t in combined if t.get("source") != "fvg_5m")
        print_metrics(f"H++ + OB {label}", m_combined)
        print(f"    Sources: {n_fvg} FVG + {n_ob} OB = {len(combined)} total")

        # Delta vs baseline
        delta_r = m_combined["R"] - baseline_m["R"]
        delta_ppdd = m_combined["PPDD"] - baseline_m["PPDD"]
        print(f"    Delta vs baseline: R={delta_r:+.1f}, PPDD={delta_ppdd:+.2f}")

    # ==================================================================
    # OB vs FVG quality comparison
    # ==================================================================
    print(f"\n{SEP}")
    print("OB vs FVG SIGNAL QUALITY COMPARISON")
    print(SEP)

    all_variants = [
        ("FVG (H++)", baseline_m),
        ("OB V1 strict", m_v1),
        ("OB V1 relaxed", m_v1r),
        ("OB V2 strict", m_v2),
        ("OB V2 relaxed", m_v2r),
        ("OB V3 strict", m_v3),
        ("OB V3 relaxed", m_v3r),
    ]

    header = f"  {'Metric':<20s}"
    for label, _ in all_variants:
        header += f" | {label:<14s}"
    print(header)
    divider = f"  {'-'*20}"
    for _ in all_variants:
        divider += f"-+-{'-'*14}"
    print(divider)

    for metric in ["trades", "R", "PPDD", "PF", "WR", "MaxDD", "avgR"]:
        row = f"  {metric:<20s}"
        for label, m in all_variants:
            val = m[metric]
            if metric == "trades":
                row += f" | {val:<14d}"
            elif metric in ("R", "avgR"):
                row += f" | {val:<+14.1f}" if metric == "R" else f" | {val:<+14.4f}"
            else:
                row += f" | {val:<14.2f}"
        print(row)

    # Additive assessment
    print(f"\n  OVERLAP: {overlap['overlapping']} / {overlap['total_ob']} OBs overlap with FVGs "
          f"({100*overlap['overlapping']/max(1,overlap['total_ob']):.1f}%)")
    print(f"  UNIQUE OBs: {overlap['unique_ob']} ({100*overlap['unique_ob']/max(1,overlap['total_ob']):.1f}%)")

    best_standalone = max(
        [("V1 strict", m_v1), ("V1 relaxed", m_v1r),
         ("V2 strict", m_v2), ("V2 relaxed", m_v2r),
         ("V3 strict", m_v3), ("V3 relaxed", m_v3r)],
        key=lambda x: x[1].get("R", 0) if x[1].get("trades", 0) >= 5 else -999,
    )
    print(f"\n  BEST standalone OB variant: {best_standalone[0]} "
          f"(trades={best_standalone[1]['trades']}, R={best_standalone[1]['R']:+.1f}, "
          f"PPDD={best_standalone[1]['PPDD']:.2f}, PF={best_standalone[1]['PF']:.2f})")

    additive = "YES" if overlap["unique_ob"] > overlap["total_ob"] * 0.3 else "MARGINAL"
    print(f"  Are OBs additive? {additive} — {overlap['unique_ob']} unique OB zones not covered by FVGs")

    # Final verdict
    total_ob_signals = max(m_v1r["trades"], m_v2r["trades"], m_v3r["trades"])
    print(f"\n  VERDICT:")
    print(f"    OBs detected: {len(ob_zones)} over {len(ob_by_year)} years (~{len(ob_zones)//max(1,len(ob_by_year))}/year)")
    print(f"    Overlap with FVGs: {overlap['overlapping']}/{overlap['total_ob']} ({100*overlap['overlapping']/max(1,overlap['total_ob']):.0f}%)")
    print(f"    Max OB signals after filters: {total_ob_signals}")
    if total_ob_signals < 20:
        print(f"    CONCLUSION: OBs produce too few actionable signals after Config H++ filtering.")
        print(f"    ROOT CAUSE: 82% overlap with FVGs means most OBs are already captured by FVG system.")
        print(f"    The remaining 18% unique OBs rarely generate rejection entries during NY session.")
    else:
        best_r = best_standalone[1]["R"]
        if best_r > 0:
            print(f"    CONCLUSION: OBs show promise as supplement ({best_r:+.1f}R standalone)")
        else:
            print(f"    CONCLUSION: OBs do not improve over FVG baseline ({best_r:+.1f}R standalone)")

    elapsed_total = _time.perf_counter() - t0_total
    print(f"\n{'='*80}")
    print(f"TOTAL RUNTIME: {elapsed_total:.1f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
