"""
experiments/u3_bias_1m_fvg.py — U3: 5m Direction + Independent 1m FVG Entry
============================================================================

Architecture:
  5m: provides DIRECTION (bias_direction from cache, shift(1).ffill())
  1m: independently detects NEW FVGs using features/fvg.py detect_fvg()
  Signal: 1m FVG rejection WHERE 1m direction matches 5m bias
  Entry: next 1m bar open after rejection candle
  Stop: 1m candle-2 open × 0.85 tightening
  TP: 5m swing targets (same as Config H++)

Different from B1c:
  - B1c: uses EXISTING 5m FVG zones → 1m rejection at THOSE specific zones
  - U3: uses 5m for DIRECTION only → finds completely NEW 1m FVGs in that direction
  - Many more potential signals (1m has ~6x more FVGs than 5m)

Usage: python experiments/u3_bias_1m_fvg.py
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
    build_liquidity_ladder_long,
    build_liquidity_ladder_short,
)

SEP = "=" * 120
THIN = "-" * 80

DATA = PROJECT / "data"


# ======================================================================
# Step 1: Load 1m data + precompute features
# ======================================================================
def load_1m_data() -> pd.DataFrame:
    t0 = _time.perf_counter()
    df = pd.read_parquet(DATA / "NQ_1min_10yr.parquet")
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
# Step 2: Detect 1m FVGs year-by-year
# ======================================================================
@dataclass
class FVG1m:
    """A detected 1m FVG with state tracking."""
    creation_bar: int       # bar index where FVG becomes known (after shift-1)
    direction: str          # 'bull' or 'bear'
    top: float
    bottom: float
    size: float
    status: str = "active"  # active | invalidated
    last_signal_bar: int = -999  # cooldown tracking


def detect_1m_fvgs_fast(
    df_1m: pd.DataFrame,
    feat_1m: dict,
    min_fvg_atr_mult: float = 0.3,
) -> list[dict]:
    """
    Detect FVGs on 1m data using vectorized logic (same as features/fvg.py detect_fvg).
    Process year-by-year for memory efficiency.
    Returns list of dicts with FVG info.
    """
    t0 = _time.perf_counter()
    from features.fvg import detect_fvg

    h = feat_1m["h"]
    l = feat_1m["l"]
    atr = feat_1m["atr"]
    n = feat_1m["n"]

    # Process year by year
    years = df_1m.index.year.unique()
    all_fvgs = []
    offset = 0  # global bar index offset

    # Build year boundaries
    year_starts = {}
    idx_years = df_1m.index.year.values
    for yr in sorted(years):
        mask = idx_years == yr
        indices = np.where(mask)[0]
        if len(indices) > 0:
            year_starts[yr] = (indices[0], indices[-1] + 1)

    for yr in sorted(years):
        start_i, end_i = year_starts[yr]
        # Take an overlap of 3 bars from end of previous year for continuity
        actual_start = max(0, start_i - 3)
        sub_df = df_1m.iloc[actual_start:end_i]

        if len(sub_df) < 3:
            continue

        # Need is_roll_date column
        if "is_roll_date" not in sub_df.columns:
            sub_df = sub_df.copy()
            sub_df["is_roll_date"] = False

        fvg_df = detect_fvg(sub_df)

        bull_mask = fvg_df["fvg_bull"].values
        bear_mask = fvg_df["fvg_bear"].values

        for local_i in range(len(sub_df)):
            global_i = actual_start + local_i

            # Skip the overlap bars (they belong to previous year)
            if global_i < start_i and actual_start < start_i:
                continue

            if bull_mask[local_i]:
                atr_val = atr[global_i] if not np.isnan(atr[global_i]) else 6.0
                fvg_size = float(fvg_df["fvg_size"].iat[local_i])
                if not np.isnan(fvg_size) and fvg_size >= min_fvg_atr_mult * atr_val:
                    all_fvgs.append({
                        "bar": global_i,
                        "direction": "bull",
                        "top": float(fvg_df["fvg_bull_top"].iat[local_i]),
                        "bottom": float(fvg_df["fvg_bull_bottom"].iat[local_i]),
                        "size": fvg_size,
                    })

            if bear_mask[local_i]:
                atr_val = atr[global_i] if not np.isnan(atr[global_i]) else 6.0
                fvg_size = float(fvg_df["fvg_size"].iat[local_i])
                if not np.isnan(fvg_size) and fvg_size >= min_fvg_atr_mult * atr_val:
                    all_fvgs.append({
                        "bar": global_i,
                        "direction": "bear",
                        "top": float(fvg_df["fvg_bear_top"].iat[local_i]),
                        "bottom": float(fvg_df["fvg_bear_bottom"].iat[local_i]),
                        "size": fvg_size,
                    })

        n_yr = sum(1 for f in all_fvgs if start_i <= f["bar"] < end_i)
        print(f"  Year {yr}: {n_yr} quality 1m FVGs detected")

    elapsed = _time.perf_counter() - t0
    n_bull = sum(1 for f in all_fvgs if f["direction"] == "bull")
    n_bear = sum(1 for f in all_fvgs if f["direction"] == "bear")
    print(f"[1m FVG] Total: {len(all_fvgs)} quality FVGs ({n_bull} bull, {n_bear} bear) in {elapsed:.1f}s")
    return all_fvgs


# ======================================================================
# Step 3: Build 5m→1m time mapping + bias lookup
# ======================================================================
def build_time_mappings(nq_5m: pd.DataFrame, df_1m: pd.DataFrame):
    """
    Build mapping from 1m timestamps to most recent completed 5m bar.
    Returns map_1m_to_prev_5m array.
    """
    t0 = _time.perf_counter()
    ts_5m_ns = nq_5m.index.astype(np.int64)
    ts_1m_ns = df_1m.index.astype(np.int64)
    five_min_ns = 5 * 60 * 10**9

    # For each 1m bar at time T: the last completed 5m bar
    # 5m bar at ts_5m[k] covers [ts_5m[k], ts_5m[k]+5min)
    # Complete at ts_5m[k]+5min
    # For 1m bar at T: last completed 5m bar is where ts_5m[k]+5min <= T
    completed_5m_ns = ts_5m_ns + five_min_ns
    map_1m_to_prev_5m = np.searchsorted(completed_5m_ns, ts_1m_ns, side='right') - 1
    map_1m_to_prev_5m = np.clip(map_1m_to_prev_5m, 0, len(nq_5m) - 1)

    elapsed = _time.perf_counter() - t0
    print(f"[MAP] Built 1m->5m mapping in {elapsed:.1f}s")
    return map_1m_to_prev_5m


# ======================================================================
# Step 4: U3 Signal Detection — 1m FVG rejection + 5m bias alignment
# ======================================================================
def scan_u3_signals(
    df_1m: pd.DataFrame,
    feat_1m: dict,
    fvg_list: list[dict],
    map_1m_to_prev_5m: np.ndarray,
    bias_dir_arr: np.ndarray,     # 5m index, already shift(1).ffill()
    *,
    min_body_ratio: float = 0.50,
    min_stop_pts: float = 5.0,
    min_stop_atr_mult: float = 1.0,  # relative to 1m ATR
    tighten_factor: float = 0.85,
    max_fvg_age_bars: int = 200,    # 1m bars
    max_active: int = 30,
    signal_cooldown_1m: int = 6,
) -> list[dict]:
    """
    Scan 1m bars for FVG rejections matching 5m bias direction.

    For each 1m bar:
      1. Birth new 1m FVGs that become known
      2. Update active FVG pool (invalidate if price closes through)
      3. Check if bar rejects from any active FVG
      4. Check if rejection direction matches 5m bias
      5. Record signal with entry/stop

    Returns list of signal dicts.
    """
    t0 = _time.perf_counter()

    o = feat_1m["o"]
    h = feat_1m["h"]
    l = feat_1m["l"]
    c = feat_1m["c"]
    br = feat_1m["body_ratio"]
    atr = feat_1m["atr"]
    n = feat_1m["n"]

    # Sort FVGs by creation bar
    sorted_fvgs = sorted(fvg_list, key=lambda f: f["bar"])
    fvg_cursor = 0
    n_fvgs = len(sorted_fvgs)

    # Active FVG pool
    active: list[FVG1m] = []
    signals: list[dict] = []

    # Stats
    stats = {
        "births": 0,
        "invalidated": 0,
        "rejections_raw": 0,
        "fail_bias": 0,
        "fail_stop_wrong_side": 0,
        "fail_stop_too_small": 0,
        "fail_stop_atr": 0,
        "cooldown_skip": 0,
        "signals": 0,
    }

    progress_step = n // 10

    for j in range(n):
        if progress_step > 0 and j % progress_step == 0 and j > 0:
            pct = 100 * j / n
            print(f"  [U3 SCAN] {pct:.0f}% ({j:,}/{n:,}) — {len(active)} active, {len(signals)} signals")

        # --- Birth: add FVGs that become known at bar j ---
        while fvg_cursor < n_fvgs:
            f = sorted_fvgs[fvg_cursor]
            if f["bar"] < j:  # BUG FIX: strict < to prevent creation-bar rejection
                rec = FVG1m(
                    creation_bar=f["bar"],
                    direction=f["direction"],
                    top=f["top"],
                    bottom=f["bottom"],
                    size=f["size"],
                )
                active.append(rec)
                stats["births"] += 1

                # Prune oldest if too many
                if len(active) > max_active:
                    active.pop(0)

                fvg_cursor += 1
            else:
                break

        # --- Update active FVGs: invalidate if price closes through ---
        surviving: list[FVG1m] = []
        for fvg in active:
            # Age check
            age = j - fvg.creation_bar
            if age > max_fvg_age_bars:
                stats["invalidated"] += 1
                continue

            if fvg.direction == "bull":
                if c[j] < fvg.bottom:
                    stats["invalidated"] += 1
                    continue
            else:
                if c[j] > fvg.top:
                    stats["invalidated"] += 1
                    continue

            surviving.append(fvg)
        active = surviving

        # --- Rejection detection ---
        if j < 2 or j + 1 >= n:
            continue

        # Body ratio check
        if br[j] < min_body_ratio:
            continue

        # Get 5m bias for this 1m bar
        prev_5m = map_1m_to_prev_5m[j]
        bias = bias_dir_arr[prev_5m]

        # Check against each active FVG
        for fvg in active:
            # Cooldown
            if j - fvg.last_signal_bar < signal_cooldown_1m:
                stats["cooldown_skip"] += 1
                continue

            if fvg.direction == "bull":
                # BULLISH FVG -> LONG signal
                # Bar low touches zone, close above zone top
                if l[j] > fvg.top:
                    continue
                if c[j] <= fvg.top:
                    continue
                if c[j] <= o[j]:
                    continue
                signal_dir = 1
            else:
                # BEARISH FVG -> SHORT signal
                # Bar high touches zone, close below zone bottom
                if h[j] < fvg.bottom:
                    continue
                if c[j] >= fvg.bottom:
                    continue
                if c[j] >= o[j]:
                    continue
                signal_dir = -1

            stats["rejections_raw"] += 1

            # Bias alignment check
            if bias == 0:
                stats["fail_bias"] += 1
                continue
            if signal_dir != int(np.sign(bias)):
                stats["fail_bias"] += 1
                continue

            # Entry = next 1m bar open
            entry_price = o[j + 1]

            # Stop = 1m candle-2 open, tightened
            raw_stop = o[j - 2]

            # Verify stop on correct side
            if signal_dir == 1 and raw_stop >= entry_price:
                stats["fail_stop_wrong_side"] += 1
                continue
            if signal_dir == -1 and raw_stop <= entry_price:
                stats["fail_stop_wrong_side"] += 1
                continue

            # Tighten
            raw_dist = abs(entry_price - raw_stop)
            tightened_dist = raw_dist * tighten_factor
            if signal_dir == 1:
                stop_price = entry_price - tightened_dist
            else:
                stop_price = entry_price + tightened_dist

            # Min stop: absolute floor
            if tightened_dist < min_stop_pts:
                stats["fail_stop_too_small"] += 1
                continue

            # Min stop: ATR-relative (using 1m ATR)
            atr_val = atr[j] if not np.isnan(atr[j]) else 6.0
            if atr_val > 0 and (tightened_dist / atr_val) < min_stop_atr_mult:
                stats["fail_stop_atr"] += 1
                continue

            signals.append({
                "signal_1m_idx": j,
                "entry_1m_idx": j + 1,
                "direction": signal_dir,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "stop_dist": tightened_dist,
                "fvg_direction": fvg.direction,
                "fvg_top": fvg.top,
                "fvg_bottom": fvg.bottom,
                "fvg_creation_bar": fvg.creation_bar,
                "fvg_age_bars": j - fvg.creation_bar,
                "rejection_body_ratio": br[j],
                "prev_5m_idx": prev_5m,
                "bias_at_signal": float(bias),
                "atr_1m": atr_val,
            })
            stats["signals"] += 1

            fvg.last_signal_bar = j
            break  # One signal per bar

    elapsed = _time.perf_counter() - t0
    print(f"[U3 SCAN] Complete in {elapsed:.1f}s")
    for k, v in stats.items():
        print(f"    {k}: {v}")
    return signals


# ======================================================================
# Step 5: Apply session/time filters
# ======================================================================
def filter_u3_signals(
    signals: list[dict],
    df_1m: pd.DataFrame,
    d: dict,
    map_1m_to_prev_5m: np.ndarray,
    *,
    block_pm_shorts: bool = True,
    pm_shorts_cutoff: float = 14.0,
) -> list[dict]:
    """
    Apply session and time filters to raw U3 signals.
    Filters: NY session (10:00-16:00), PM shorts block, news blackout.
    """
    t0 = _time.perf_counter()

    news_blackout_arr = d.get("news_blackout_arr", None)

    et_1m = df_1m.index.tz_convert("US/Eastern")
    et_frac_1m = (et_1m.hour + et_1m.minute / 60.0).values

    filtered = []
    filter_stats = {
        "total": len(signals),
        "fail_obs": 0,
        "fail_session": 0,
        "fail_pm_shorts": 0,
        "fail_news": 0,
        "passed": 0,
    }

    for sig in signals:
        j = sig["signal_1m_idx"]
        direction = sig["direction"]
        prev_5m = sig["prev_5m_idx"]
        et_frac = et_frac_1m[j]

        # Observation period (9:30-10:00 ET)
        if 9.5 <= et_frac <= 10.0:
            filter_stats["fail_obs"] += 1
            continue

        # NY session only (10:00-16:00 ET)
        if not (10.0 <= et_frac < 16.0):
            filter_stats["fail_session"] += 1
            continue

        # PM shorts block
        if block_pm_shorts and direction == -1 and et_frac >= pm_shorts_cutoff:
            filter_stats["fail_pm_shorts"] += 1
            continue

        # News blackout
        if news_blackout_arr is not None and news_blackout_arr[prev_5m]:
            filter_stats["fail_news"] += 1
            continue

        sig_out = sig.copy()
        sig_out["et_frac"] = et_frac
        filtered.append(sig_out)
        filter_stats["passed"] += 1

    elapsed = _time.perf_counter() - t0
    print(f"[U3 FILTER] {filter_stats['total']} raw -> {filter_stats['passed']} filtered in {elapsed:.1f}s")
    for k, v in filter_stats.items():
        if k not in ("total", "passed"):
            print(f"    {k}: {v}")
    return filtered


# ======================================================================
# Step 6: Simulate trades on 1m data with 50/50/0 TP
# ======================================================================
def simulate_u3_trades(
    filtered_signals: list[dict],
    d: dict,
    d_extra: dict,
    df_1m: pd.DataFrame,
    feat_1m: dict,
    map_1m_to_prev_5m: np.ndarray,
) -> list[dict]:
    """
    Forward simulate each U3 trade on 1m bars.
    Longs: 50/50 multi-TP (TP1=nearest 5m swing, TP2=next liquidity), no runner, no BE.
    Shorts: 100% exit at 0.625R.
    EOD close 15:55 ET, one position at a time, 0-for-2 / daily loss limits.
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

    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
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
    last_exit_bar = -1

    tp1_count = 0
    tp2_count = 0
    eod_count = 0
    stop_count = 0

    max_bars_in_trade = 500

    for sig in sorted_sigs:
        entry_j = sig["entry_1m_idx"]
        direction = sig["direction"]
        entry_price = sig["entry_price"]
        stop_price = sig["stop_price"]
        stop_dist = sig["stop_dist"]
        prev_5m = sig["prev_5m_idx"]
        et_frac_sig = sig.get("et_frac", 12.0)

        if entry_j >= n_1m:
            continue

        # One position at a time
        if entry_j <= last_exit_bar:
            continue

        # Day reset
        entry_et_h = et_hour_1m[entry_j]
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
        bd = bias_dir_arr[prev_5m]
        regime = regime_arr[prev_5m]
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

        # TP computation using 5m swing targets
        irl_raw = irl_target_5m[prev_5m] if not np.isnan(irl_target_5m[prev_5m]) else 0.0

        if direction == 1:
            # LONG: multi-TP 50/50/0
            tp1, tp2 = build_liquidity_ladder_long(
                actual_entry, stop_price, irl_raw, prev_5m, d_extra
            )
            if tp1 <= actual_entry:
                tp1 = actual_entry + actual_stop_dist * 0.5
            if tp2 <= tp1:
                tp2 = actual_entry + (tp1 - actual_entry) * 1.5
            is_multi_tp = True
        else:
            # SHORT: scalp 100% exit at 0.625R
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
        tp1_hit_bar_1m = -1  # BUG FIX: track which bar TP1 was hit

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
                    tp1_hit_bar_1m = jj  # BUG FIX: record TP1 bar
                    tp1_count += 1
                    if remaining <= 0:
                        exit_price = tp1
                        exit_reason = "tp1"
                        exit_bar = jj
                        exit_contracts = contracts
                        exited = True
                        break

                # BUG FIX: only check TP2 on NEXT bar after TP1
                if trim_stage == 1 and jj > tp1_hit_bar_1m and h_1m[jj] >= tp2:
                    tc2 = remaining
                    trim2_c = tc2
                    remaining -= tc2
                    tp2_pnl_pts = tp2 - actual_entry
                    trim_stage = 2
                    tp2_count += 1
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
        elif exit_reason == "stop":
            stop_count += 1

        trades.append({
            "entry_time": df_1m.index[entry_j],
            "exit_time": df_1m.index[exit_bar],
            "r": r_mult, "reason": exit_reason, "dir": direction,
            "type": "u3_1m_fvg", "trimmed": any_trimmed,
            "grade": grade,
            "entry_price": actual_entry, "exit_price": exit_price,
            "stop_price": stop_price, "tp1_price": tp1,
            "tp2_price": tp2 if is_multi_tp else 0.0,
            "pnl_dollars": total_pnl,
            "source": "u3",
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
    print(f"[U3 SIM] {len(trades)} trades in {elapsed:.1f}s (TP1={tp1_count}, TP2={tp2_count}, Stop={stop_count}, EOD={eod_count})")
    return trades


# ======================================================================
# Step 7: Combine with 5m Config H++ trades
# ======================================================================
def combine_5m_and_u3_trades(
    trades_5m: list[dict],
    trades_u3: list[dict],
) -> list[dict]:
    """
    Combine 5m and U3 trades into single timeline.
    One position at a time: 5m takes precedence (comes first in sort order tie).
    """
    for t in trades_5m:
        if "source" not in t:
            t["source"] = "5m"
    for t in trades_u3:
        if "source" not in t:
            t["source"] = "u3"

    all_trades = sorted(trades_5m + trades_u3, key=lambda t: t["entry_time"])

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
# Step 8: Signal overlap analysis
# ======================================================================
def analyze_overlap(trades_5m: list[dict], trades_u3: list[dict], nq_5m: pd.DataFrame):
    """
    Analyze overlap between 5m signals and U3 signals.
    Two signals overlap if their entry times are within 5 minutes of each other.
    """
    if not trades_5m or not trades_u3:
        print("[OVERLAP] No trades to compare")
        return

    five_min_ns = 5 * 60 * 10**9

    # Get entry timestamps in nanoseconds for 5m trades
    entries_5m_ns_list = []
    for t in trades_5m:
        et = t["entry_time"]
        if hasattr(et, "tz") and et.tz is None:
            et = et.tz_localize("UTC")
        entries_5m_ns_list.append(et.value)
    entries_5m_ns_arr = np.array(sorted(entries_5m_ns_list), dtype=np.int64)

    # For each U3 trade, check if any 5m trade entry is within 5 minutes
    overlap_count = 0
    unique_count = 0
    for t in trades_u3:
        et = t["entry_time"]
        if hasattr(et, "tz") and et.tz is None:
            et = et.tz_localize("UTC")
        ns = et.value
        # Find nearest 5m trade entry time
        idx = np.searchsorted(entries_5m_ns_arr, ns, side='left')
        is_overlap = False
        for check_idx in [idx - 1, idx]:
            if 0 <= check_idx < len(entries_5m_ns_arr):
                diff = abs(ns - entries_5m_ns_arr[check_idx])
                if diff <= five_min_ns:
                    is_overlap = True
                    break
        if is_overlap:
            overlap_count += 1
        else:
            unique_count += 1

    total = overlap_count + unique_count
    print(f"[OVERLAP] U3 trades: {total}")
    if total > 0:
        print(f"  Overlap with 5m (within 5 min): {overlap_count} ({100*overlap_count/total:.1f}%)")
        print(f"  Unique to U3: {unique_count} ({100*unique_count/total:.1f}%)")


# ======================================================================
# Main
# ======================================================================
def main():
    print(SEP)
    print("U3: 5m DIRECTION + INDEPENDENT 1m FVG ENTRY")
    print("5m provides bias_direction only -> 1m detects NEW FVGs in that direction")
    print(SEP)

    # ---- Step 1: Load all data ----
    print("\n[STEP 1] Loading data...")
    d = load_all()
    d_extra = prepare_liquidity_data(d)

    df_1m = load_1m_data()
    feat_1m = precompute_1m_features(df_1m)

    nq_5m = d["nq"]

    # Bias from cache: shift(1).ffill() to prevent lookahead
    bias_dir_arr = d["bias_dir_arr"].copy()  # Already loaded from cache_bias_10yr_v2

    # Report stats
    atr_1m_vals = feat_1m["atr"]
    valid_atr = atr_1m_vals[~np.isnan(atr_1m_vals)]
    print(f"[1m ATR] mean={np.mean(valid_atr):.2f}, median={np.median(valid_atr):.2f}")
    print(f"[5m BIAS] distribution: +1={np.sum(bias_dir_arr > 0)}, -1={np.sum(bias_dir_arr < 0)}, 0={np.sum(bias_dir_arr == 0)}")

    # ---- Step 2: Detect 1m FVGs ----
    print(f"\n[STEP 2] Detecting 1m FVGs (year by year)...")

    # Add is_roll_date if missing
    if "is_roll_date" not in df_1m.columns:
        df_1m["is_roll_date"] = False
        print("  Added is_roll_date=False column to 1m data")

    fvg_list = detect_1m_fvgs_fast(df_1m, feat_1m, min_fvg_atr_mult=0.3)

    # ---- Step 3: Build time mappings ----
    print(f"\n[STEP 3] Building time mappings...")
    map_1m_to_prev_5m = build_time_mappings(nq_5m, df_1m)

    # ---- Step 4: U3 Signal Detection ----
    print(f"\n[STEP 4] U3 signal detection...")
    raw_signals = scan_u3_signals(
        df_1m, feat_1m, fvg_list, map_1m_to_prev_5m, bias_dir_arr,
        min_body_ratio=0.50,
        min_stop_pts=5.0,
        min_stop_atr_mult=1.0,
        tighten_factor=0.85,
        max_fvg_age_bars=200,
        signal_cooldown_1m=6,
    )

    if not raw_signals:
        print("  NO RAW SIGNALS — aborting")
        return

    # ---- Step 5: Filter signals ----
    print(f"\n[STEP 5] Applying session/time filters...")
    filtered = filter_u3_signals(
        raw_signals, df_1m, d, map_1m_to_prev_5m,
        block_pm_shorts=True,
        pm_shorts_cutoff=14.0,
    )

    if not filtered:
        print("  NO FILTERED SIGNALS — aborting")
        return

    # ---- Step 6: Simulate trades ----
    print(f"\n[STEP 6] Simulating U3 trades on 1m data...")
    u3_trades = simulate_u3_trades(
        filtered, d, d_extra, df_1m, feat_1m, map_1m_to_prev_5m,
    )

    # ---- Step 7: U3 Standalone Results ----
    print(f"\n{SEP}")
    print("U3 STANDALONE RESULTS")
    print(SEP)

    u3_m = compute_metrics(u3_trades)
    print_metrics("U3 (5m bias + 1m FVG)", u3_m)

    # Long/short breakdown
    longs = [t for t in u3_trades if t["dir"] == 1]
    shorts = [t for t in u3_trades if t["dir"] == -1]
    if longs:
        lr = np.array([t["r"] for t in longs])
        print(f"  Longs:  {len(longs)}t, R={lr.sum():+.1f}, avgR={lr.mean():+.4f}, WR={100*(lr>0).mean():.1f}%")
    if shorts:
        sr = np.array([t["r"] for t in shorts])
        print(f"  Shorts: {len(shorts)}t, R={sr.sum():+.1f}, avgR={sr.mean():+.4f}, WR={100*(sr>0).mean():.1f}%")

    # Exit reasons
    reasons = {}
    for t in u3_trades:
        r = t["reason"]
        if r not in reasons:
            reasons[r] = {"count": 0, "r_sum": 0.0}
        reasons[r]["count"] += 1
        reasons[r]["r_sum"] += t["r"]
    print(f"\n  Exit reasons:")
    for reason, info in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
        print(f"    {reason:15s}: {info['count']:4d}t  R={info['r_sum']:+.1f}")

    # Per-year walk-forward
    wf = walk_forward_metrics(u3_trades)
    if wf:
        print(f"\n  Walk-forward by year:")
        for yr in wf:
            print(f"    {yr['year']}: {yr['n']:3d}t  R={yr['R']:+7.1f}  WR={yr['WR']:5.1f}%  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}")

    # ---- Config H++ Baseline ----
    print(f"\n{SEP}")
    print("CONFIG H++ BASELINE (5m signals only)")
    print(SEP)

    from experiments.pure_liquidity_tp import run_backtest_pure_liquidity

    baseline_trades_raw, baseline_diag = run_backtest_pure_liquidity(
        d, d_extra,
        sq_long=0.68, sq_short=0.80,
        min_stop_atr=1.7,
        block_pm_shorts=True,
        tp_strategy="v1",
        tp1_trim_pct=0.50,
        be_after_tp1=False,
        be_after_tp2=False,
    )
    # Handle return value (may be tuple with diag)
    if isinstance(baseline_trades_raw, tuple):
        baseline_trades = baseline_trades_raw[0]
    else:
        baseline_trades = baseline_trades_raw

    baseline_m = compute_metrics(baseline_trades)
    print_metrics("Config H++ (5m baseline)", baseline_m)

    baseline_wf = walk_forward_metrics(baseline_trades)
    if baseline_wf:
        print(f"  Walk-forward:")
        for yr in baseline_wf:
            print(f"    {yr['year']}: {yr['n']:3d}t  R={yr['R']:+7.1f}  WR={yr['WR']:5.1f}%  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}")

    # ---- Step 8: Combined (5m + U3) ----
    print(f"\n{SEP}")
    print("COMBINED: 5m Config H++ + U3 (one position at a time)")
    print(SEP)

    combined_trades = combine_5m_and_u3_trades(baseline_trades, u3_trades)
    combined_m = compute_metrics(combined_trades)
    print_metrics("Combined (5m + U3)", combined_m)

    # Breakdown by source
    n_from_5m = sum(1 for t in combined_trades if t.get("source") == "5m")
    n_from_u3 = sum(1 for t in combined_trades if t.get("source") == "u3")
    r_from_5m = sum(t["r"] for t in combined_trades if t.get("source") == "5m")
    r_from_u3 = sum(t["r"] for t in combined_trades if t.get("source") == "u3")
    print(f"  From 5m: {n_from_5m}t, R={r_from_5m:+.1f}")
    print(f"  From U3: {n_from_u3}t, R={r_from_u3:+.1f}")

    combined_wf = walk_forward_metrics(combined_trades)
    if combined_wf:
        print(f"  Walk-forward:")
        for yr in combined_wf:
            print(f"    {yr['year']}: {yr['n']:3d}t  R={yr['R']:+7.1f}  WR={yr['WR']:5.1f}%  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}")

    # ---- Step 9: Overlap Analysis ----
    print(f"\n{SEP}")
    print("OVERLAP ANALYSIS")
    print(SEP)
    analyze_overlap(baseline_trades, u3_trades, nq_5m)

    # ---- Summary ----
    print(f"\n{SEP}")
    print("SUMMARY COMPARISON")
    print(SEP)
    print(f"  {'System':35s} | {'Trades':>6s} | {'R':>10s} | {'PPDD':>7s} | {'PF':>6s} | {'WR':>6s} | {'MaxDD':>7s}")
    print(f"  {'-'*35} | {'-'*6} | {'-'*10} | {'-'*7} | {'-'*6} | {'-'*6} | {'-'*7}")
    for label, m in [
        ("Config H++ (5m baseline)", baseline_m),
        ("U3 standalone", u3_m),
        ("Combined (5m + U3)", combined_m),
    ]:
        print(f"  {label:35s} | {m['trades']:6d} | {m['R']:+10.1f} | {m['PPDD']:7.2f} | {m['PF']:6.2f} | {m['WR']:5.1f}% | {m['MaxDD']:7.1f}")

    delta_r = combined_m["R"] - baseline_m["R"]
    delta_ppdd = combined_m["PPDD"] - baseline_m["PPDD"]
    delta_trades = combined_m["trades"] - baseline_m["trades"]
    print(f"\n  Combined vs Baseline Delta:")
    print(f"    R:      {delta_r:+.1f}")
    print(f"    PPDD:   {delta_ppdd:+.2f}")
    print(f"    Trades: {delta_trades:+d}")
    print(f"    PF:     {combined_m['PF'] - baseline_m['PF']:+.2f}")


if __name__ == "__main__":
    main()
