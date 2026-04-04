"""
experiments/u4_htf_fvg_1m_entry.py — HTF (1H/4H) FVG Zone + 1m Rejection Entry
================================================================================

Hypothesis:  Higher-timeframe FVGs are structurally stronger levels.
  - 1H FVGs are institutional-level gaps.
  - 4H FVGs are even stronger, but fewer and wider.
  - CLAUDE.md §3.1: "Higher TF FVGs carry more weight."

Architecture (temporally correct):
  1. Detect FVGs on 1H and 4H data. Each FVG defines a ZONE (top/bottom prices).
  2. Quality filter: zone size >= 0.5 × ATR of that timeframe.
  3. Zone becomes known only AFTER the 3rd HTF candle closes (shift-1 equivalent).
  4. Forward-fill zone into 1m timeframe. Price enters zone -> watch mode.
  5. 1m Rejection: candle tests zone, closes on correct side, body_ratio >= 0.50.
  6. Entry = next 1m bar open after rejection candle.
  7. Stop = far side of HTF FVG zone (wider than 5m, but stronger level).
  8. TP = raw IRL (5m swing) + liquidity ladder TP2.

Anti-lookahead guarantees:
  - HTF FVG zone created in the PAST (known after 3rd candle closes)
  - Zone availability shifted by 1 HTF bar into the future
  - Price entering zone is REAL-TIME (1m)
  - 1m rejection is the CURRENT just-closed candle
  - Entry is NEXT bar open
  - Stop is a PAST-known zone boundary

Usage: python experiments/u4_htf_fvg_1m_entry.py
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from dataclasses import dataclass

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
# Step 2: HTF FVG Detection (1H and 4H)
# ======================================================================
def load_htf_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load 1H and 4H parquet files."""
    t0 = _time.perf_counter()
    df_1h = pd.read_parquet(PROJECT / "data" / "NQ_1H_10yr.parquet")
    df_4h = pd.read_parquet(PROJECT / "data" / "NQ_4H_10yr.parquet")
    elapsed = _time.perf_counter() - t0
    print(f"[HTF DATA] 1H: {len(df_1h):,} bars, 4H: {len(df_4h):,} bars  ({elapsed:.1f}s)")
    return df_1h, df_4h


def compute_htf_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Compute ATR(period) on HTF OHLC data."""
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        hl = h[i] - l[i]
        hc = abs(h[i] - c[i - 1])
        lc = abs(l[i] - c[i - 1])
        tr[i] = max(hl, hc, lc)
    atr = np.full(n, np.nan)
    cum = np.cumsum(tr)
    atr[period - 1:] = (cum[period - 1:] - np.concatenate([[0], cum[:n - period]])) / period
    return atr


@dataclass
class HTFZone:
    """An active HTF FVG zone."""
    creation_time: pd.Timestamp
    creation_htf_idx: int
    direction: str            # 'bull' or 'bear'
    top: float
    bottom: float
    size: float
    timeframe: str            # '1H' or '4H'
    status: str = "untested"
    last_signal_1m_idx: int = -999


def build_htf_fvg_zones(
    df_htf: pd.DataFrame,
    atr_htf: np.ndarray,
    timeframe_label: str,
    min_fvg_atr_mult: float = 0.5,
    max_fvg_age_bars: int = 100,
) -> list[dict]:
    """
    Detect FVGs on HTF data and build zone list.

    Each zone becomes known after candle-3 closes. detect_fvg already shifts
    by 1, so at row i the FVG is visible. We additionally note that in HTF
    time the zone is available starting from the bar AFTER the detection bar.

    Returns list of dicts with zone metadata including HTF timestamps.
    """
    from features.fvg import detect_fvg

    t0 = _time.perf_counter()
    fvg_df = detect_fvg(df_htf)

    zones = []
    bull_mask = fvg_df["fvg_bull"].values
    bear_mask = fvg_df["fvg_bear"].values
    n = len(df_htf)
    idx = df_htf.index

    for i in range(n):
        if bull_mask[i]:
            atr_val = atr_htf[i] if not np.isnan(atr_htf[i]) else 50.0
            fvg_size = fvg_df["fvg_size"].iat[i]
            if not np.isnan(fvg_size) and fvg_size >= min_fvg_atr_mult * atr_val:
                zones.append({
                    "creation_time": idx[i - 1] if i > 0 else idx[i],
                    "creation_htf_idx": i - 1 if i > 0 else 0,
                    "becomes_known_htf_idx": i,
                    # The HTF bar at idx[i] has just closed, so the zone is
                    # available starting from idx[i]'s close time (= start of next bar)
                    "becomes_known_time": idx[i],
                    "direction": "bull",
                    "top": float(fvg_df["fvg_bull_top"].iat[i]),
                    "bottom": float(fvg_df["fvg_bull_bottom"].iat[i]),
                    "size": float(fvg_size),
                    "timeframe": timeframe_label,
                })

        if bear_mask[i]:
            atr_val = atr_htf[i] if not np.isnan(atr_htf[i]) else 50.0
            fvg_size = fvg_df["fvg_size"].iat[i]
            if not np.isnan(fvg_size) and fvg_size >= min_fvg_atr_mult * atr_val:
                zones.append({
                    "creation_time": idx[i - 1] if i > 0 else idx[i],
                    "creation_htf_idx": i - 1 if i > 0 else 0,
                    "becomes_known_htf_idx": i,
                    "becomes_known_time": idx[i],
                    "direction": "bear",
                    "top": float(fvg_df["fvg_bear_top"].iat[i]),
                    "bottom": float(fvg_df["fvg_bear_bottom"].iat[i]),
                    "size": float(fvg_size),
                    "timeframe": timeframe_label,
                })

    elapsed = _time.perf_counter() - t0
    n_bull = sum(1 for z in zones if z["direction"] == "bull")
    n_bear = sum(1 for z in zones if z["direction"] == "bear")
    avg_size = np.mean([z["size"] for z in zones]) if zones else 0.0
    print(f"[{timeframe_label} FVG] {len(zones)} quality zones ({n_bull} bull, {n_bear} bear), "
          f"avg size={avg_size:.1f} pts  ({elapsed:.1f}s)")
    return zones


# ======================================================================
# Step 3: HTF -> 1m time mapping
# ======================================================================
def build_htf_to_1m_mapping(
    df_htf: pd.DataFrame,
    df_1m: pd.DataFrame,
    bar_duration_minutes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build mapping from HTF to 1m timestamps.

    Returns:
      - ts_htf_ns: int64 array of HTF bar timestamps (nanoseconds)
      - ts_1m_ns: int64 array of 1m bar timestamps (nanoseconds)
      - map_1m_to_prev_htf: for each 1m bar, the index of the latest
        fully-completed HTF bar
    """
    t0 = _time.perf_counter()
    ts_htf_ns = df_htf.index.astype(np.int64)
    ts_1m_ns = df_1m.index.astype(np.int64)
    bar_ns = bar_duration_minutes * 60 * 10**9

    # For each 1m bar at time T: the last completed HTF bar
    # HTF bar at ts_htf[k] closes at ts_htf[k] + bar_ns
    completed_htf_ns = ts_htf_ns + bar_ns
    map_1m_to_prev_htf = np.searchsorted(completed_htf_ns, ts_1m_ns, side='right') - 1
    map_1m_to_prev_htf = np.clip(map_1m_to_prev_htf, 0, len(df_htf) - 1)

    elapsed = _time.perf_counter() - t0
    print(f"[MAP] HTF({bar_duration_minutes}m)->1m mapping built in {elapsed:.1f}s")
    return ts_htf_ns, ts_1m_ns, map_1m_to_prev_htf


# ======================================================================
# Step 4: 1m rejection scan with HTF FVG zones
# ======================================================================
def scan_1m_rejections_htf(
    df_1m: pd.DataFrame,
    feat_1m: dict,
    fvg_zones: list[dict],
    map_1m_to_prev_htf: np.ndarray,
    map_1m_to_prev_5m: np.ndarray,
    d: dict,
    *,
    min_body_ratio: float = 0.50,
    min_stop_pts: float = 5.0,
    tighten_factor: float = 1.0,       # No tightening for HTF zones (already structural)
    max_fvg_age_htf_bars: int = 100,    # 100 1H bars = ~4 days, 100 4H bars = ~16 days
    max_active: int = 30,
    signal_cooldown_1m: int = 30,       # Wider cooldown for HTF zones (less frequent signals)
    stop_mode: str = "htf_zone",        # "htf_zone" = far side, "1m_candle2" = old style
) -> list[dict]:
    """
    Scan 1m bars for rejections at HTF FVG zones.

    Stop modes:
      - "htf_zone": stop at far side of HTF FVG (structural stop)
      - "1m_candle2": stop at 1m candle-2 open (tighter, like b1c)

    Returns list of signal dicts.
    """
    t0 = _time.perf_counter()

    o_1m = feat_1m["o"]
    h_1m = feat_1m["h"]
    l_1m = feat_1m["l"]
    c_1m = feat_1m["c"]
    br_1m = feat_1m["body_ratio"]
    atr_1m = feat_1m["atr"]
    n_1m = feat_1m["n"]

    # Sort zones by the time they become known
    sorted_zones = sorted(fvg_zones, key=lambda z: z["becomes_known_htf_idx"])
    zone_cursor = 0
    n_zones = len(sorted_zones)

    active: list[HTFZone] = []
    signals: list[dict] = []

    # Stats
    n_births = 0
    n_invalidated = 0
    n_age_expired = 0
    n_rejections_raw = 0
    n_stop_wrong_side = 0
    n_stop_too_small = 0
    n_cooldown_skip = 0

    progress_step = n_1m // 10

    for j in range(n_1m):
        if progress_step > 0 and j % progress_step == 0 and j > 0:
            pct = 100 * j / n_1m
            print(f"  [SCAN] {pct:.0f}% ({j:,}/{n_1m:,}) — {len(active)} active HTF zones, {len(signals)} signals")

        prev_htf = map_1m_to_prev_htf[j]

        # --- Birth: add zones that become known by now ---
        while zone_cursor < n_zones:
            z = sorted_zones[zone_cursor]
            known_htf_idx = z["becomes_known_htf_idx"]
            if prev_htf >= known_htf_idx:
                rec = HTFZone(
                    creation_time=z["creation_time"],
                    creation_htf_idx=z["creation_htf_idx"],
                    direction=z["direction"],
                    top=z["top"],
                    bottom=z["bottom"],
                    size=z["size"],
                    timeframe=z["timeframe"],
                )
                active.append(rec)
                n_births += 1

                if len(active) > max_active:
                    active.pop(0)

                zone_cursor += 1
            else:
                break

        # --- Update: invalidate zones if price closes through ---
        surviving: list[HTFZone] = []
        for fvg in active:
            age_htf = prev_htf - fvg.creation_htf_idx
            if age_htf > max_fvg_age_htf_bars:
                n_age_expired += 1
                continue

            if fvg.direction == "bull":
                if c_1m[j] < fvg.bottom:
                    n_invalidated += 1
                    continue
            else:
                if c_1m[j] > fvg.top:
                    n_invalidated += 1
                    continue

            surviving.append(fvg)
        active = surviving

        # --- Rejection detection ---
        if j < 2 or j + 1 >= n_1m:
            continue

        if br_1m[j] < min_body_ratio:
            continue

        for fvg in active:
            # Cooldown
            if j - fvg.last_signal_1m_idx < signal_cooldown_1m:
                n_cooldown_skip += 1
                continue

            if fvg.direction == "bull":
                # BULLISH zone -> LONG signal
                # 1m bar must touch zone: low <= fvg.top AND close above it
                if l_1m[j] > fvg.top:
                    continue
                if c_1m[j] <= fvg.top:
                    continue
                if c_1m[j] <= o_1m[j]:
                    continue
            else:
                # BEARISH zone -> SHORT signal
                # 1m bar must touch zone: high >= fvg.bottom AND close below it
                if h_1m[j] < fvg.bottom:
                    continue
                if c_1m[j] >= fvg.bottom:
                    continue
                if c_1m[j] >= o_1m[j]:
                    continue

            n_rejections_raw += 1

            entry_price = o_1m[j + 1]

            # Stop determination
            if stop_mode == "htf_zone":
                # Far side of HTF FVG = structural stop
                if fvg.direction == "bull":
                    raw_stop = fvg.bottom   # Long: stop below zone bottom
                else:
                    raw_stop = fvg.top      # Short: stop above zone top
            else:
                # 1m candle-2 open (like b1c)
                raw_stop = o_1m[j - 2]

            direction = 1 if fvg.direction == "bull" else -1

            # Verify stop on correct side
            if direction == 1 and raw_stop >= entry_price:
                n_stop_wrong_side += 1
                continue
            if direction == -1 and raw_stop <= entry_price:
                n_stop_wrong_side += 1
                continue

            raw_dist = abs(entry_price - raw_stop)
            stop_dist = raw_dist * tighten_factor

            if direction == 1:
                stop_price = entry_price - stop_dist
            else:
                stop_price = entry_price + stop_dist

            if stop_dist < min_stop_pts:
                n_stop_too_small += 1
                continue

            atr_1m_val = atr_1m[j] if not np.isnan(atr_1m[j]) else 6.0
            prev_5m = map_1m_to_prev_5m[j]

            signals.append({
                "signal_1m_idx": j,
                "entry_1m_idx": j + 1,
                "direction": direction,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "stop_dist": stop_dist,
                "fvg_direction": fvg.direction,
                "fvg_top": fvg.top,
                "fvg_bottom": fvg.bottom,
                "fvg_size": fvg.size,
                "fvg_timeframe": fvg.timeframe,
                "fvg_creation_time": fvg.creation_time,
                "fvg_creation_htf_idx": fvg.creation_htf_idx,
                "fvg_age_htf_bars": prev_htf - fvg.creation_htf_idx,
                "rejection_body_ratio": br_1m[j],
                "prev_5m_idx": prev_5m,
                "atr_1m": atr_1m_val,
            })

            fvg.last_signal_1m_idx = j
            # Only one signal per zone per visit
            fvg.status = "tested"
            break

    elapsed = _time.perf_counter() - t0
    print(f"[SCAN] Complete in {elapsed:.1f}s")
    print(f"  Births: {n_births}, Invalidated: {n_invalidated}, Age-expired: {n_age_expired}")
    print(f"  Raw rejections: {n_rejections_raw}, Stop wrong side: {n_stop_wrong_side}, Stop too small: {n_stop_too_small}")
    print(f"  Cooldown skips: {n_cooldown_skip}")
    print(f"  Final signals: {len(signals)}")

    return signals


# ======================================================================
# Step 5: Apply Config H+ filters (same as b1c)
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
    min_stop_atr_mult: float = 0.0,    # Disabled for HTF zones (stops are already wide)
    block_pm_shorts: bool = True,
    use_1m_atr: bool = True,
    min_stop_pts_abs: float = 5.0,
) -> list[dict]:
    """Apply Config H+ filters to raw HTF zone signals."""
    t0 = _time.perf_counter()

    params = d["params"]
    atr_arr_5m = d["atr_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    fluency_5m = d["fluency_arr"]

    session_regime = params.get("session_regime", {})

    o_1m = feat_1m["o"]
    h_1m = feat_1m["h"]
    l_1m = feat_1m["l"]
    c_1m = feat_1m["c"]
    br_1m = feat_1m["body_ratio"]
    atr_1m = feat_1m["atr"]

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
        direction = sig["direction"]
        prev_5m = sig["prev_5m_idx"]
        et_frac = et_frac_1m[j]

        # Observation period (9:30-10:00 ET)
        if 9.5 <= et_frac <= 10.0:
            stats["fail_obs"] += 1
            continue
        stats["pass_obs"] += 1

        # News blackout
        if news_blackout_arr is not None and news_blackout_arr[prev_5m]:
            stats["fail_news"] += 1
            continue
        stats["pass_news"] += 1

        # Session filter: NY only (10:00 - 16:00 ET)
        is_ny = (10.0 <= et_frac < 16.0)
        if not is_ny:
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

        # Absolute stop floor
        stop_dist = sig["stop_dist"]
        if min_stop_pts_abs > 0 and stop_dist < min_stop_pts_abs:
            stats["fail_min_stop"] += 1
            continue

        # ATR-relative min stop (if enabled)
        if min_stop_atr_mult > 0:
            atr_ref = sig.get("atr_1m", 6.0) if use_1m_atr else (atr_arr_5m[prev_5m] if not np.isnan(atr_arr_5m[prev_5m]) else 30.0)
            if atr_ref > 0 and (stop_dist / atr_ref) < min_stop_atr_mult:
                stats["fail_min_stop"] += 1
                continue
        stats["pass_min_stop"] += 1

        # Session regime: lunch dead zone
        if session_regime.get("enabled", False):
            lunch_s = session_regime.get("lunch_start", 12.5)
            lunch_e = session_regime.get("lunch_end", 13.0)
            lunch_mult = session_regime.get("lunch_mult", 0.0)
            if lunch_s <= et_frac < lunch_e and lunch_mult == 0.0:
                stats["fail_lunch"] += 1
                continue
        stats["pass_lunch"] += 1

        # Signal quality
        atr_5m_val = atr_arr_5m[prev_5m] if not np.isnan(atr_arr_5m[prev_5m]) else 30.0
        sq_params = params.get("signal_quality", {})
        if sq_params.get("enabled", False):
            size_sc = min(1.0, sig["fvg_size"] / (atr_5m_val * 1.5)) if atr_5m_val > 0 else 0.5
            disp_sc = br_1m[j]
            flu_val = fluency_5m[prev_5m] if not np.isnan(fluency_5m[prev_5m]) else 0.5
            flu_sc = min(1.0, max(0.0, flu_val))
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
    For each filtered HTF zone signal, simulate forward on 1m data.

    Multi-TP for longs: 50/50/0 (raw IRL TP1 + liquidity ladder TP2)
    Scalp for shorts: 100% exit at short_rr * stop_dist
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

    atr_5m = d["atr_arr"]
    irl_target_5m = d["irl_target_arr"]

    et_1m = df_1m.index.tz_convert("US/Eastern")
    et_hour_1m = et_1m.hour.values
    et_minute_1m = et_1m.minute.values

    sorted_sigs = sorted(filtered_signals, key=lambda s: s["entry_1m_idx"])

    trades = []
    current_date = None
    daily_pnl_r = 0.0
    consecutive_losses = 0
    day_stopped = False

    tp1_hit = 0
    tp2_hit = 0
    eod_count = 0
    max_bars_in_trade = 500
    last_exit_bar = -1

    for sig in sorted_sigs:
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

        if entry_j <= last_exit_bar:
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
            tp1, tp2 = build_liquidity_ladder_long(
                actual_entry, stop_price, irl_raw, prev_5m, d_extra
            )
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

        short_trim_pct = dual_mode.get("short_trim_pct", 1.0)

        # --- Simulate forward ---
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
            "type": f"htf_{sig.get('fvg_timeframe', '??')}_rejection",
            "trimmed": any_trimmed,
            "grade": grade,
            "entry_price": actual_entry, "exit_price": exit_price,
            "stop_price": stop_price, "tp1_price": tp1,
            "tp2_price": tp2 if is_multi_tp else 0.0,
            "pnl_dollars": total_pnl, "trim_stage": trim_stage,
            "fvg_size": sig["fvg_size"],
            "fvg_timeframe": sig.get("fvg_timeframe", ""),
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
# Combine with 5m Config H++ baseline
# ======================================================================
def combine_trades(
    trades_5m: list[dict],
    trades_htf: list[dict],
) -> list[dict]:
    """Merge 5m and HTF trades, one position at a time (5m takes precedence)."""
    for t in trades_5m:
        t["source"] = "5m"
    for t in trades_htf:
        t["source"] = "htf"

    all_trades = sorted(trades_5m + trades_htf, key=lambda t: t["entry_time"])

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
# Run single HTF config
# ======================================================================
def run_htf_config(
    d: dict,
    d_extra: dict,
    df_1m: pd.DataFrame,
    feat_1m: dict,
    fvg_zones: list[dict],
    map_1m_to_prev_htf: np.ndarray,
    map_1m_to_prev_5m: np.ndarray,
    *,
    min_body_ratio: float = 0.50,
    min_stop_pts: float = 5.0,
    tighten_factor: float = 1.0,
    max_fvg_age_htf_bars: int = 100,
    signal_cooldown_1m: int = 30,
    stop_mode: str = "htf_zone",
    sq_long: float = 0.68,
    sq_short: float = 0.80,
    min_stop_atr_mult: float = 0.0,
    block_pm_shorts: bool = True,
    label: str = "",
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Run full HTF zone pipeline for one config."""
    if verbose:
        print(f"\n{THIN}")
        print(f"Config: {label}")
        print(THIN)

    raw_signals = scan_1m_rejections_htf(
        df_1m, feat_1m, fvg_zones,
        map_1m_to_prev_htf, map_1m_to_prev_5m, d,
        min_body_ratio=min_body_ratio,
        min_stop_pts=min_stop_pts,
        tighten_factor=tighten_factor,
        max_fvg_age_htf_bars=max_fvg_age_htf_bars,
        signal_cooldown_1m=signal_cooldown_1m,
        stop_mode=stop_mode,
    )

    if not raw_signals:
        if verbose:
            print("  NO RAW SIGNALS")
        return [], compute_metrics([])

    filtered = filter_signals(
        raw_signals, d, map_1m_to_prev_5m, df_1m, feat_1m,
        sq_long=sq_long, sq_short=sq_short,
        min_stop_atr_mult=min_stop_atr_mult,
        block_pm_shorts=block_pm_shorts,
    )

    if not filtered:
        if verbose:
            print("  NO FILTERED SIGNALS")
        return [], compute_metrics([])

    trades = simulate_1m_outcomes(
        filtered, d, d_extra, df_1m, feat_1m, map_1m_to_prev_5m,
    )

    m = compute_metrics(trades)
    if verbose:
        print_metrics(label, m)

        longs = [t for t in trades if t["dir"] == 1]
        shorts = [t for t in trades if t["dir"] == -1]
        if longs:
            lr = np.array([t["r"] for t in longs])
            print(f"    Longs:  {len(longs)}t, R={lr.sum():+.1f}, avgR={lr.mean():+.4f}, WR={100*(lr>0).mean():.1f}%")
        if shorts:
            sr = np.array([t["r"] for t in shorts])
            print(f"    Shorts: {len(shorts)}t, R={sr.sum():+.1f}, avgR={sr.mean():+.4f}, WR={100*(sr>0).mean():.1f}%")

        avg_stop = np.mean([abs(t["entry_price"] - t["stop_price"]) for t in trades]) if trades else 0
        avg_fvg_size = np.mean([t.get("fvg_size", 0) for t in trades]) if trades else 0
        print(f"    Avg stop distance: {avg_stop:.1f} pts, Avg FVG zone size: {avg_fvg_size:.1f} pts")

        wf = walk_forward_metrics(trades)
        if wf:
            print(f"    Walk-forward:")
            for yr in wf:
                print(f"      {yr['year']}: {yr['n']:3d}t  R={yr['R']:+7.1f}  WR={yr['WR']:5.1f}%  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}")

    return trades, m


# ======================================================================
# Zone statistics
# ======================================================================
def zone_statistics(fvg_zones: list[dict], trades: list[dict], label: str):
    """How many HTF zones get tested and generate signals per year?"""
    print(f"\n  --- Zone Statistics: {label} ---")

    if not fvg_zones:
        print("    No zones detected")
        return

    zone_df = pd.DataFrame(fvg_zones)
    zone_df["year"] = pd.to_datetime(zone_df["creation_time"]).dt.year

    print(f"    Total zones: {len(zone_df)}")
    print(f"    Per year:")
    for year, grp in zone_df.groupby("year"):
        bull = (grp["direction"] == "bull").sum()
        bear = (grp["direction"] == "bear").sum()
        avg_size = grp["size"].mean()
        print(f"      {year}: {len(grp):4d} zones ({bull} bull, {bear} bear), avg size={avg_size:.1f} pts")

    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df["year"] = pd.to_datetime(trade_df["entry_time"]).dt.year
        print(f"    Trades generated: {len(trades)}")
        for year, grp in trade_df.groupby("year"):
            r_arr = grp["r"].values
            print(f"      {year}: {len(grp):3d} trades, R={r_arr.sum():+.1f}")

    # Conversion rate
    n_years = zone_df["year"].nunique()
    zones_per_year = len(zone_df) / n_years if n_years > 0 else 0
    trades_per_year = len(trades) / n_years if n_years > 0 else 0
    conversion = len(trades) / len(zone_df) * 100 if len(zone_df) > 0 else 0
    print(f"    Zones/year: {zones_per_year:.0f}, Trades/year: {trades_per_year:.1f}, Conversion: {conversion:.1f}%")


# ======================================================================
# Zone quality comparison (5m vs 1H vs 4H)
# ======================================================================
def compare_zone_quality(trades_5m: list[dict], trades_1h: list[dict], trades_4h: list[dict]):
    """Compare win rates across zone timeframes."""
    print(f"\n{SEP}")
    print("ZONE QUALITY COMPARISON: 5m FVG vs 1H FVG vs 4H FVG")
    print(SEP)

    print(f"  {'Source':12s} | {'Trades':>6s} | {'R':>8s} | {'PPDD':>7s} | {'PF':>6s} | {'WR':>6s} | {'MaxDD':>6s} | {'avgR':>8s} | {'Avg Stop':>8s}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")

    for label, trades in [("5m FVG", trades_5m), ("1H FVG", trades_1h), ("4H FVG", trades_4h)]:
        if not trades:
            print(f"  {label:12s} | {0:>6d} | {'N/A':>8s} | {'N/A':>7s} | {'N/A':>6s} | {'N/A':>6s} | {'N/A':>6s} | {'N/A':>8s} | {'N/A':>8s}")
            continue
        m = compute_metrics(trades)
        avg_stop = np.mean([abs(t["entry_price"] - t["stop_price"]) for t in trades])
        print(f"  {label:12s} | {m['trades']:>6d} | {m['R']:>+8.1f} | {m['PPDD']:>7.2f} | {m['PF']:>6.2f} | {m['WR']:>5.1f}% | {m['MaxDD']:>6.1f} | {m['avgR']:>+8.4f} | {avg_stop:>8.1f}")

    # Long-only comparison
    print(f"\n  LONG-ONLY COMPARISON:")
    print(f"  {'Source':12s} | {'Trades':>6s} | {'R':>8s} | {'WR':>6s} | {'avgR':>8s}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}")
    for label, trades in [("5m FVG", trades_5m), ("1H FVG", trades_1h), ("4H FVG", trades_4h)]:
        longs = [t for t in trades if t["dir"] == 1]
        if not longs:
            print(f"  {label:12s} | {0:>6d} | {'N/A':>8s} | {'N/A':>6s} | {'N/A':>8s}")
            continue
        r_arr = np.array([t["r"] for t in longs])
        print(f"  {label:12s} | {len(longs):>6d} | {r_arr.sum():>+8.1f} | {100*(r_arr>0).mean():>5.1f}% | {r_arr.mean():>+8.4f}")

    # Short-only comparison
    print(f"\n  SHORT-ONLY COMPARISON:")
    print(f"  {'Source':12s} | {'Trades':>6s} | {'R':>8s} | {'WR':>6s} | {'avgR':>8s}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}")
    for label, trades in [("5m FVG", trades_5m), ("1H FVG", trades_1h), ("4H FVG", trades_4h)]:
        shorts = [t for t in trades if t["dir"] == -1]
        if not shorts:
            print(f"  {label:12s} | {0:>6d} | {'N/A':>8s} | {'N/A':>6s} | {'N/A':>8s}")
            continue
        r_arr = np.array([t["r"] for t in shorts])
        print(f"  {label:12s} | {len(shorts):>6d} | {r_arr.sum():>+8.1f} | {100*(r_arr>0).mean():>5.1f}% | {r_arr.mean():>+8.4f}")


# ======================================================================
# Main
# ======================================================================
def main():
    print(SEP)
    print("U4: HTF (1H/4H) FVG ZONE + 1m REJECTION ENTRY")
    print("Hypothesis: Larger HTF FVGs are structurally stronger levels")
    print("  -> higher WR, wider stops (fewer contracts), but better quality")
    print(SEP)

    # ---- Step 1: Load all data ----
    print("\n[STEP 1] Loading data...")
    d = load_all()
    d_extra = prepare_liquidity_data(d)

    df_1m = load_1m_data()
    feat_1m = precompute_1m_features(df_1m)

    df_1h, df_4h = load_htf_data()

    nq_5m = d["nq"]

    # ---- Step 2: Build 5m->1m mapping (for filter alignment) ----
    print("\n[STEP 2] Building time mappings...")
    ts_5m_ns = nq_5m.index.astype(np.int64)
    ts_1m_ns = df_1m.index.astype(np.int64)
    five_min_ns = 5 * 60 * 10**9
    completed_5m_ns = ts_5m_ns + five_min_ns
    map_1m_to_prev_5m = np.searchsorted(completed_5m_ns, ts_1m_ns, side='right') - 1
    map_1m_to_prev_5m = np.clip(map_1m_to_prev_5m, 0, len(nq_5m) - 1)

    # HTF -> 1m mappings
    _, _, map_1m_to_prev_1h = build_htf_to_1m_mapping(df_1h, df_1m, 60)
    _, _, map_1m_to_prev_4h = build_htf_to_1m_mapping(df_4h, df_1m, 240)

    # ---- Step 3: Detect HTF FVG zones ----
    print("\n[STEP 3] Detecting HTF FVG zones...")
    atr_1h = compute_htf_atr(df_1h)
    atr_4h = compute_htf_atr(df_4h)

    zones_1h = build_htf_fvg_zones(df_1h, atr_1h, "1H", min_fvg_atr_mult=0.5, max_fvg_age_bars=100)
    zones_4h = build_htf_fvg_zones(df_4h, atr_4h, "4H", min_fvg_atr_mult=0.5, max_fvg_age_bars=100)

    # Zone size statistics
    if zones_1h:
        sizes_1h = np.array([z["size"] for z in zones_1h])
        print(f"  1H zone sizes: mean={sizes_1h.mean():.1f}, median={np.median(sizes_1h):.1f}, "
              f"p25={np.percentile(sizes_1h, 25):.1f}, p75={np.percentile(sizes_1h, 75):.1f}")
    if zones_4h:
        sizes_4h = np.array([z["size"] for z in zones_4h])
        print(f"  4H zone sizes: mean={sizes_4h.mean():.1f}, median={np.median(sizes_4h):.1f}, "
              f"p25={np.percentile(sizes_4h, 25):.1f}, p75={np.percentile(sizes_4h, 75):.1f}")

    # ---- Step 4: Config H++ baseline (5m) ----
    print(f"\n{SEP}")
    print("CONFIG H++ BASELINE (5m signals only)")
    print(SEP)
    from experiments.pure_liquidity_tp import run_backtest_pure_liquidity
    baseline_trades, diag = run_backtest_pure_liquidity(
        d, d_extra,
        sq_long=0.68, sq_short=0.80,
        min_stop_atr=1.7,
        block_pm_shorts=True,
        tp_strategy="v1",
        tp1_trim_pct=0.50,
        be_after_tp1=False,
        be_after_tp2=False,
    )
    baseline_m = compute_metrics(baseline_trades)
    print_metrics("Config H++ (5m baseline)", baseline_m)
    baseline_wf = walk_forward_metrics(baseline_trades)
    if baseline_wf:
        print("  Walk-forward:")
        for yr in baseline_wf:
            print(f"    {yr['year']}: {yr['n']:3d}t  R={yr['R']:+7.1f}  WR={yr['WR']:5.1f}%  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}")

    # ---- Step 5: U4 — Standalone 1H zones ----
    print(f"\n{SEP}")
    print("U4-A: STANDALONE 1H FVG ZONES + 1m REJECTION")
    print(SEP)

    trades_1h, m_1h = run_htf_config(
        d, d_extra, df_1m, feat_1m, zones_1h,
        map_1m_to_prev_1h, map_1m_to_prev_5m,
        label="U4-A: 1H zones, htf_zone stop",
        stop_mode="htf_zone",
        max_fvg_age_htf_bars=100,
        signal_cooldown_1m=30,
    )
    zone_statistics(zones_1h, trades_1h, "1H zones")

    # Also try with 1m candle-2 stop (tighter, more contracts)
    print(f"\n{THIN}")
    print("U4-A2: 1H zones with 1m candle-2 stop (tighter)")
    print(THIN)
    trades_1h_tight, m_1h_tight = run_htf_config(
        d, d_extra, df_1m, feat_1m, zones_1h,
        map_1m_to_prev_1h, map_1m_to_prev_5m,
        label="U4-A2: 1H zones, 1m_candle2 stop",
        stop_mode="1m_candle2",
        max_fvg_age_htf_bars=100,
        signal_cooldown_1m=30,
    )

    # ---- Step 6: U4 — Standalone 4H zones ----
    print(f"\n{SEP}")
    print("U4-B: STANDALONE 4H FVG ZONES + 1m REJECTION")
    print(SEP)

    trades_4h, m_4h = run_htf_config(
        d, d_extra, df_1m, feat_1m, zones_4h,
        map_1m_to_prev_4h, map_1m_to_prev_5m,
        label="U4-B: 4H zones, htf_zone stop",
        stop_mode="htf_zone",
        max_fvg_age_htf_bars=100,
        signal_cooldown_1m=30,
    )
    zone_statistics(zones_4h, trades_4h, "4H zones")

    # Also try with 1m candle-2 stop
    print(f"\n{THIN}")
    print("U4-B2: 4H zones with 1m candle-2 stop (tighter)")
    print(THIN)
    trades_4h_tight, m_4h_tight = run_htf_config(
        d, d_extra, df_1m, feat_1m, zones_4h,
        map_1m_to_prev_4h, map_1m_to_prev_5m,
        label="U4-B2: 4H zones, 1m_candle2 stop",
        stop_mode="1m_candle2",
        max_fvg_age_htf_bars=100,
        signal_cooldown_1m=30,
    )

    # ---- Step 7: Combined 1H + 4H zones ----
    print(f"\n{SEP}")
    print("U4-C: COMBINED 1H + 4H ZONES (one position at a time)")
    print(SEP)

    # Merge 1H and 4H zones
    zones_combined = zones_1h + zones_4h
    # Use the finer mapping (1H) since we want to check both timeframes
    # For combined, we need to handle that zones have different mappings.
    # Solution: run both separately, then merge trades
    all_htf_trades = []
    for t in trades_1h:
        t_copy = t.copy()
        t_copy["source"] = "1H"
        all_htf_trades.append(t_copy)
    for t in trades_4h:
        t_copy = t.copy()
        t_copy["source"] = "4H"
        all_htf_trades.append(t_copy)

    all_htf_trades.sort(key=lambda t: t["entry_time"])

    # One position at a time
    combined_htf = []
    last_exit = pd.Timestamp.min.tz_localize("UTC")
    for t in all_htf_trades:
        entry_t = t["entry_time"]
        if hasattr(entry_t, "tz") and entry_t.tz is None:
            entry_t = entry_t.tz_localize("UTC")
        if entry_t <= last_exit:
            continue
        combined_htf.append(t)
        exit_t = t["exit_time"]
        if hasattr(exit_t, "tz") and exit_t.tz is None:
            exit_t = exit_t.tz_localize("UTC")
        last_exit = exit_t

    m_combined_htf = compute_metrics(combined_htf)
    print_metrics("U4-C: Combined 1H+4H", m_combined_htf)

    n_from_1h = sum(1 for t in combined_htf if t.get("source") == "1H")
    n_from_4h = sum(1 for t in combined_htf if t.get("source") == "4H")
    print(f"    From 1H: {n_from_1h}, From 4H: {n_from_4h}")

    wf_c = walk_forward_metrics(combined_htf)
    if wf_c:
        print(f"    Walk-forward:")
        for yr in wf_c:
            print(f"      {yr['year']}: {yr['n']:3d}t  R={yr['R']:+7.1f}  WR={yr['WR']:5.1f}%  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}")

    # ---- Step 8: Combined 5m H++ + U4 HTF ----
    print(f"\n{SEP}")
    print("U4-D: CONFIG H++ (5m) + U4 HTF (1H+4H) COMBINED")
    print(SEP)

    combined_full = combine_trades(
        [t.copy() for t in baseline_trades],
        [t.copy() for t in combined_htf],
    )
    m_full = compute_metrics(combined_full)
    print_metrics("U4-D: H++ + HTF combined", m_full)

    n_5m = sum(1 for t in combined_full if t.get("source") == "5m")
    n_htf = sum(1 for t in combined_full if t.get("source") == "htf")
    print(f"    From 5m: {n_5m}, From HTF: {n_htf}")

    wf_d = walk_forward_metrics(combined_full)
    if wf_d:
        print(f"    Walk-forward:")
        for yr in wf_d:
            print(f"      {yr['year']}: {yr['n']:3d}t  R={yr['R']:+7.1f}  WR={yr['WR']:5.1f}%  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}")

    # ---- Step 9: Also combine with b1c (5m FVG 1m entry) trades ----
    # Load b1c best config for comparison
    print(f"\n{SEP}")
    print("U4-E: CONFIG H++ (5m) + B1C (5m zone 1m entry) + U4 (HTF zone 1m entry)")
    print(SEP)

    # Run b1c inline to get its trades
    from experiments.b1c_fvg_zone_1m_entry import (
        build_5m_fvg_zones,
        build_time_mappings,
        scan_1m_rejections as scan_1m_rejections_5m,
        filter_signals as filter_signals_5m,
        simulate_1m_outcomes as simulate_1m_outcomes_5m,
    )

    fvg_zones_5m = build_5m_fvg_zones(nq_5m, d["atr_arr"], min_fvg_atr_mult=0.3)
    ts_5m_ns_b, ts_1m_ns_b, map_5m_to_1m_b, map_1m_to_prev_5m_b = build_time_mappings(nq_5m, df_1m)

    raw_5m_1m = scan_1m_rejections_5m(
        df_1m, feat_1m, fvg_zones_5m,
        ts_5m_ns_b, ts_1m_ns_b, map_5m_to_1m_b, map_1m_to_prev_5m_b, d,
        min_body_ratio=0.50,
        min_stop_pts=0.0,
        tighten_factor=0.85,
        max_fvg_age_bars=200,
        signal_cooldown_1m=6,
    )

    filtered_5m_1m = filter_signals_5m(
        raw_5m_1m, d, map_1m_to_prev_5m_b, df_1m, feat_1m,
        sq_long=0.68, sq_short=0.80,
        min_stop_atr_mult=1.0,
        use_1m_atr=True,
        min_stop_pts_abs=5.0,
        block_pm_shorts=True,
    )

    b1c_trades = simulate_1m_outcomes_5m(
        filtered_5m_1m, d, d_extra, df_1m, feat_1m, map_1m_to_prev_5m_b,
    )
    b1c_m = compute_metrics(b1c_trades)
    print_metrics("B1C standalone", b1c_m)

    # Triple-combine: H++ 5m + b1c + HTF
    all_sources = []
    for t in baseline_trades:
        tc = t.copy()
        tc["source"] = "5m_h++"
        all_sources.append(tc)
    for t in b1c_trades:
        tc = t.copy()
        tc["source"] = "b1c"
        all_sources.append(tc)
    for t in combined_htf:
        tc = t.copy()
        tc["source"] = "u4_htf"
        all_sources.append(tc)

    all_sources.sort(key=lambda t: t["entry_time"])

    triple_combined = []
    last_exit = pd.Timestamp.min.tz_localize("UTC")
    for t in all_sources:
        entry_t = t["entry_time"]
        if hasattr(entry_t, "tz") and entry_t.tz is None:
            entry_t = entry_t.tz_localize("UTC")
        if entry_t <= last_exit:
            continue
        triple_combined.append(t)
        exit_t = t["exit_time"]
        if hasattr(exit_t, "tz") and exit_t.tz is None:
            exit_t = exit_t.tz_localize("UTC")
        last_exit = exit_t

    m_triple = compute_metrics(triple_combined)
    print_metrics("U4-E: H++ + B1C + HTF", m_triple)

    n_5m_e = sum(1 for t in triple_combined if t.get("source") == "5m_h++")
    n_b1c_e = sum(1 for t in triple_combined if t.get("source") == "b1c")
    n_htf_e = sum(1 for t in triple_combined if t.get("source") == "u4_htf")
    print(f"    From 5m H++: {n_5m_e}, From B1C: {n_b1c_e}, From U4 HTF: {n_htf_e}")

    wf_e = walk_forward_metrics(triple_combined)
    if wf_e:
        print(f"    Walk-forward:")
        for yr in wf_e:
            print(f"      {yr['year']}: {yr['n']:3d}t  R={yr['R']:+7.1f}  WR={yr['WR']:5.1f}%  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}")

    # ---- Step 10: Zone quality comparison ----
    compare_zone_quality(b1c_trades, trades_1h, trades_4h)

    # ---- Final Summary ----
    print(f"\n{SEP}")
    print("FINAL SUMMARY")
    print(SEP)
    print(f"  {'Config':45s} | {'Trades':>6s} | {'R':>8s} | {'PPDD':>7s} | {'PF':>6s} | {'WR':>6s} | {'MaxDD':>6s}")
    print(f"  {'-'*45}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    summary = [
        ("Config H++ (5m baseline)", baseline_m),
        ("B1C standalone (5m zone, 1m entry)", b1c_m),
        ("U4-A: 1H zones, htf_zone stop", m_1h),
        ("U4-A2: 1H zones, 1m_candle2 stop", m_1h_tight),
        ("U4-B: 4H zones, htf_zone stop", m_4h),
        ("U4-B2: 4H zones, 1m_candle2 stop", m_4h_tight),
        ("U4-C: Combined 1H+4H", m_combined_htf),
        ("U4-D: H++ + HTF combined", m_full),
        ("U4-E: H++ + B1C + HTF triple", m_triple),
    ]

    for label, m in summary:
        if m["trades"] > 0:
            print(f"  {label:45s} | {m['trades']:>6d} | {m['R']:>+8.1f} | {m['PPDD']:>7.2f} | {m['PF']:>6.2f} | {m['WR']:>5.1f}% | {m['MaxDD']:>6.1f}")
        else:
            print(f"  {label:45s} | {0:>6d} | {'N/A':>8s} | {'N/A':>7s} | {'N/A':>6s} | {'N/A':>6s} | {'N/A':>6s}")

    print(f"\n  Delta (U4-D vs H++ baseline):")
    if m_full["trades"] > 0 and baseline_m["trades"] > 0:
        print(f"    Trades: {m_full['trades']} vs {baseline_m['trades']} ({m_full['trades'] - baseline_m['trades']:+d})")
        print(f"    R: {m_full['R']:+.1f} vs {baseline_m['R']:+.1f} ({m_full['R'] - baseline_m['R']:+.1f})")
        print(f"    PPDD: {m_full['PPDD']:.2f} vs {baseline_m['PPDD']:.2f} ({m_full['PPDD'] - baseline_m['PPDD']:+.2f})")
        print(f"    PF: {m_full['PF']:.2f} vs {baseline_m['PF']:.2f}")

    print(f"\n  Delta (U4-E triple vs H++ baseline):")
    if m_triple["trades"] > 0 and baseline_m["trades"] > 0:
        print(f"    Trades: {m_triple['trades']} vs {baseline_m['trades']} ({m_triple['trades'] - baseline_m['trades']:+d})")
        print(f"    R: {m_triple['R']:+.1f} vs {baseline_m['R']:+.1f} ({m_triple['R'] - baseline_m['R']:+.1f})")
        print(f"    PPDD: {m_triple['PPDD']:.2f} vs {baseline_m['PPDD']:.2f} ({m_triple['PPDD'] - baseline_m['PPDD']:+.2f})")
        print(f"    PF: {m_triple['PF']:.2f} vs {baseline_m['PF']:.2f}")

    print(f"\n{SEP}")
    print("END U4 EXPERIMENT")
    print(SEP)


if __name__ == "__main__":
    main()
