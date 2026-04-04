"""
experiments/u2_clean.py — U2 Limit Order at FVG Zone (CLEAN reimplementation)
=============================================================================

Clean rewrite with CORRECT single-TP PnL engine (no multi-TP bug).
Entry: limit order at FVG zone edge.
Exit: same as validate_improvements.py (trim/trail/BE, single-TP).

Sweeps stop strategy, min stop, max age, FVG size, TP multiplier.
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
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", stream=sys.stdout)

DATA = PROJECT / "data"
CONFIG = PROJECT / "config" / "params.yaml"


# ======================================================================
# Data loading (from validate_improvements.py — trusted, audited)
# ======================================================================
def load_all():
    t0 = _time.perf_counter()
    print("[LOAD] Loading data...")

    nq = pd.read_parquet(DATA / "NQ_5m_10yr.parquet")
    with open(CONFIG, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    from features.displacement import compute_atr
    from features.swing import compute_swing_levels
    from features.news_filter import build_news_blackout_mask
    from features.fvg import detect_fvg

    et_idx = nq.index.tz_convert("US/Eastern")
    o = nq["open"].values
    h = nq["high"].values
    l = nq["low"].values
    c = nq["close"].values
    n = len(nq)

    atr_cache = pd.read_parquet(DATA / "cache_atr_flu_10yr_v2.parquet")
    atr_arr = atr_cache["atr"].values

    bias = pd.read_parquet(DATA / "cache_bias_10yr_v2.parquet")
    regime = pd.read_parquet(DATA / "cache_regime_10yr_v2.parquet")
    bias_dir_arr = bias["bias_direction"].values.astype(float)
    regime_arr = regime["regime"].values.astype(float)

    swing_p = {"left_bars": params["swing"]["left_bars"],
               "right_bars": params["swing"]["right_bars"]}
    swings = compute_swing_levels(nq, swing_p)
    swings["swing_high_price"] = swings["swing_high_price"].shift(1).ffill()
    swings["swing_low_price"] = swings["swing_low_price"].shift(1).ffill()
    swing_high_mask = swings["swing_high"].shift(1, fill_value=False).values
    swing_low_mask = swings["swing_low"].shift(1, fill_value=False).values
    irl_high_arr = swings["swing_high_price"].values
    irl_low_arr = swings["swing_low_price"].values

    news_path = PROJECT / "config" / "news_calendar.csv"
    news_blackout_arr = None
    if news_path.exists():
        news_bl = build_news_blackout_mask(nq.index, str(news_path),
            params["news"]["blackout_minutes_before"],
            params["news"]["cooldown_minutes_after"])
        news_blackout_arr = news_bl.values

    # FVG detection
    fvg_df = detect_fvg(nq)

    dates = np.array([
        (et_idx[j] + pd.Timedelta(days=1)).date() if et_idx[j].hour >= 18 else et_idx[j].date()
        for j in range(n)
    ])
    dow_arr = np.array([et_idx[j].dayofweek for j in range(n)])
    et_frac_arr = np.array([et_idx[j].hour + et_idx[j].minute / 60.0 for j in range(n)])

    elapsed = _time.perf_counter() - t0
    print(f"[LOAD] All data loaded in {elapsed:.1f}s (n={n} bars)")

    return {
        "nq": nq, "params": params, "n": n,
        "o": o, "h": h, "l": l, "c": c,
        "atr_arr": atr_arr,
        "bias_dir_arr": bias_dir_arr, "regime_arr": regime_arr,
        "swing_high_mask": swing_high_mask, "swing_low_mask": swing_low_mask,
        "irl_high_arr": irl_high_arr, "irl_low_arr": irl_low_arr,
        "news_blackout_arr": news_blackout_arr,
        "fvg_df": fvg_df,
        "dates": dates, "dow_arr": dow_arr, "et_frac_arr": et_frac_arr,
        "et_idx": et_idx,
    }


# ======================================================================
# FVG Zone dataclass
# ======================================================================
@dataclass
class FVGZone:
    type: str            # 'bull' or 'bear'
    top: float
    bottom: float
    size: float
    birth_bar: int
    birth_atr: float
    used: bool = False


# ======================================================================
# Swing finder (same as validate_improvements.py)
# ======================================================================
def _find_nth_swing(mask, prices, idx, n_val):
    count = 0
    for j in range(idx - 1, -1, -1):
        if mask[j]:
            count += 1
            if count == n_val:
                return prices[j]
    return np.nan


def _compute_grade(ba, regime):
    if np.isnan(ba) or np.isnan(regime) or regime == 0.0:
        return "C"
    if ba > 0.5 and regime >= 1.0:
        return "A+"
    if ba > 0.5 or regime >= 1.0:
        return "B+"
    return "C"


# ======================================================================
# U2 Backtest — single-TP engine (CORRECT PnL)
# ======================================================================
def run_u2_backtest(
    d: dict,
    *,
    # FVG zone params
    max_fvg_age: int = 200,
    fvg_size_mult: float = 0.3,
    # Stop
    stop_strategy: str = "A1",       # A1=far side, A2=far side+buffer
    stop_buffer_pct: float = 0.15,   # for A2
    # Min stop filter
    min_stop_atr: float = 0.0,       # 0=disabled, else min stop as ATR mult
    min_stop_pts: float = 5.0,       # AUDIT FIX: hard floor 5pt (was 0). Sub-5pt stops unrealistic.
    # Filters
    block_pm_shorts: bool = True,
    # Exit (single-TP, same as F3)
    trim_pct: float = 0.25,          # F3: 25% trim at TP1
    tp_mult: float = 2.0,            # TP = IRL target * mult
    nth_swing: int = 3,              # trail nth swing
    be_after_trim: bool = True,
    short_rr: float = 0.625,         # short scalp target
    tighten_factor: float = 0.85,    # F3 stop tightening
    # EOD
    eod_close: bool = True,
    # Track MFE
    track_mfe: bool = False,
) -> tuple[list[dict], dict]:
    """Run U2 limit order backtest with correct single-TP PnL."""

    params = d["params"]
    nq = d["nq"]
    n = d["n"]
    o, h, l, c_arr = d["o"], d["h"], d["l"], d["c"]
    atr_arr = d["atr_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    fvg_df = d["fvg_df"]
    dates = d["dates"]
    dow_arr = d["dow_arr"]
    et_frac_arr = d["et_frac_arr"]
    swing_high_mask = d["swing_high_mask"]
    swing_low_mask = d["swing_low_mask"]
    irl_high_arr = d["irl_high_arr"]
    irl_low_arr = d["irl_low_arr"]

    pos_params = params["position"]
    risk_params = params["risk"]
    bt_params = params["backtest"]
    grading_params = params["grading"]
    session_regime = params.get("session_regime", {})

    normal_r = pos_params["normal_r"]
    reduced_r = pos_params["reduced_r"]
    point_value = pos_params["point_value"]
    daily_max_loss_r = risk_params["daily_max_loss_r"]
    max_consec_losses = risk_params["max_consecutive_losses"]
    commission_per_side = bt_params["commission_per_side_micro"]
    slippage_points = bt_params["slippage_normal_ticks"] * 0.25  # AUDIT FIX #3: stop exits get slippage
    # No slippage on limit order ENTRIES (that's the advantage)
    a_plus_mult = grading_params["a_plus_size_mult"]
    b_plus_mult = grading_params["b_plus_size_mult"]

    # FVG arrays
    bull_mask = fvg_df["fvg_bull"].values
    bear_mask = fvg_df["fvg_bear"].values
    fvg_bull_top = fvg_df["fvg_bull_top"].values
    fvg_bull_bottom = fvg_df["fvg_bull_bottom"].values
    fvg_bear_top = fvg_df["fvg_bear_top"].values
    fvg_bear_bottom = fvg_df["fvg_bear_bottom"].values

    # State
    active_zones: list[FVGZone] = []
    trades: list[dict] = []
    in_position = False
    pos_direction = pos_entry_idx = 0
    pos_entry_price = pos_stop = pos_tp1 = 0.0
    pos_contracts = pos_remaining_contracts = 0
    pos_trimmed = False
    pos_be_stop = pos_trail_stop = 0.0
    pos_grade = ""
    pos_trim_pct = trim_pct
    pos_mfe = 0.0
    current_date = None
    daily_pnl_r = 0.0
    consecutive_losses = 0
    day_stopped = False

    # Stats
    zones_created = 0
    zones_filled = 0

    for i in range(n):
        bar_date = dates[i]
        if bar_date != current_date:
            current_date = bar_date
            daily_pnl_r = 0.0
            consecutive_losses = 0
            day_stopped = False

        # === PHASE A: Register new FVG zones ===
        if bull_mask[i]:
            atr_val = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
            zone_size = fvg_bull_top[i] - fvg_bull_bottom[i]
            if zone_size >= fvg_size_mult * atr_val and zone_size > 0:
                active_zones.append(FVGZone("bull", fvg_bull_top[i], fvg_bull_bottom[i],
                                            zone_size, i, atr_val))
                zones_created += 1

        if bear_mask[i]:
            atr_val = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
            zone_size = fvg_bear_top[i] - fvg_bear_bottom[i]
            if zone_size >= fvg_size_mult * atr_val and zone_size > 0:
                active_zones.append(FVGZone("bear", fvg_bear_top[i], fvg_bear_bottom[i],
                                            zone_size, i, atr_val))
                zones_created += 1

        # === PHASE B: Invalidate / expire zones ===
        surviving: list[FVGZone] = []
        for zone in active_zones:
            if zone.used:
                continue
            if (i - zone.birth_bar) > max_fvg_age:
                continue
            if zone.type == "bull" and c_arr[i] < zone.bottom:
                continue
            if zone.type == "bear" and c_arr[i] > zone.top:
                continue
            surviving.append(zone)
        if len(surviving) > 30:
            surviving = surviving[-30:]
        active_zones = surviving

        # === PHASE C: EXIT MANAGEMENT (single-TP, same as F3) ===
        if in_position:
            exited = False
            exit_reason = ""
            exit_price = 0.0
            exit_contracts = pos_remaining_contracts

            # MFE tracking
            if track_mfe:
                fe = (h[i] - pos_entry_price) if pos_direction == 1 else (pos_entry_price - l[i])
                if fe > pos_mfe:
                    pos_mfe = fe

            # AUDIT FIX #4: Stop/TP checked BEFORE EOD close.
            # Reason: stop could be hit intrabar before 15:55 close.
            # Long exit (stop, TP, trail)
            if not exited and pos_direction == 1:
                eff_stop = pos_trail_stop if pos_trimmed and pos_trail_stop > 0 else pos_stop
                if pos_trimmed and be_after_trim and pos_be_stop > 0:
                    eff_stop = max(eff_stop, pos_be_stop)
                if l[i] <= eff_stop:
                    exit_price = eff_stop - slippage_points  # AUDIT FIX #3: slippage on stop
                    exit_reason = "be_sweep" if (pos_trimmed and eff_stop >= pos_entry_price) else "stop"
                    exited = True
                elif not pos_trimmed and h[i] >= pos_tp1:
                    tc = max(1, int(pos_contracts * pos_trim_pct))
                    pos_remaining_contracts = pos_contracts - tc
                    pos_trimmed = True
                    pos_be_stop = pos_entry_price
                    if pos_remaining_contracts > 0:
                        pos_trail_stop = _find_nth_swing(swing_low_mask, nq["low"].values, i, nth_swing)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True
                if pos_trimmed and not exited:
                    nt = _find_nth_swing(swing_low_mask, nq["low"].values, i, nth_swing)
                    if not np.isnan(nt) and nt > pos_trail_stop:
                        pos_trail_stop = nt

            # Short exit
            elif not exited and pos_direction == -1:
                eff_stop = pos_stop
                if h[i] >= eff_stop:
                    exit_price = eff_stop + slippage_points  # AUDIT FIX #3
                    exit_reason = "stop"
                    exited = True
                elif not pos_trimmed and l[i] <= pos_tp1:
                    pos_remaining_contracts = 0
                    pos_trimmed = True
                    exit_price = pos_tp1
                    exit_reason = "tp1"
                    exit_contracts = pos_contracts
                    exited = True

            # EOD close (AFTER stop/TP — AUDIT FIX #4)
            if not exited and eod_close and et_frac_arr[i] >= 15.917:
                slp = 0.25
                exit_price = c_arr[i] - slp if pos_direction == 1 else c_arr[i] + slp
                exit_reason = "eod_close"
                exited = True

            # PA early cut (bars 2-4)
            bars_in_trade = i - pos_entry_idx
            if not exited and not pos_trimmed and 2 <= bars_in_trade <= 4:
                pa_start = max(pos_entry_idx, 0)
                pa_end = i + 1
                pa_range = h[pa_start:pa_end] - l[pa_start:pa_end]
                pa_body = np.abs(c_arr[pa_start:pa_end] - o[pa_start:pa_end])
                safe_pa = np.where(pa_range > 0, pa_range, 1.0)
                pa_wick = 1.0 - (pa_body / safe_pa)
                avg_wick = float(np.mean(pa_wick))
                pa_dirs = np.sign(c_arr[pa_start:pa_end] - o[pa_start:pa_end])
                favorable = (pa_dirs == pos_direction).mean()
                disp = (c_arr[i] - pos_entry_price) if pos_direction == 1 else (pos_entry_price - c_arr[i])
                cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                if (avg_wick > 0.65 and favorable < 0.5 and disp < cur_atr * 0.3 and bars_in_trade >= 3):
                    exit_price = o[i+1] if i+1 < n else c_arr[i]
                    exit_reason = "early_cut_pa"
                    exited = True

            # PnL (CORRECT single-TP)
            if exited:
                pnl_pts = (exit_price - pos_entry_price) if pos_direction == 1 else (pos_entry_price - exit_price)
                if pos_trimmed and exit_reason != "tp1":
                    trim_pnl = (pos_tp1 - pos_entry_price) if pos_direction == 1 else (pos_entry_price - pos_tp1)
                    trim_c = pos_contracts - exit_contracts
                    total_pnl = trim_pnl * point_value * trim_c + pnl_pts * point_value * exit_contracts
                    total_comm = commission_per_side * 2 * pos_contracts
                    total_pnl -= total_comm
                else:
                    total_pnl = pnl_pts * point_value * exit_contracts
                    total_comm = commission_per_side * 2 * exit_contracts
                    total_pnl -= total_comm
                stop_dist_exit = abs(pos_entry_price - pos_stop)
                total_risk = stop_dist_exit * point_value * pos_contracts
                r_mult = total_pnl / total_risk if total_risk > 0 else 0.0

                trade_rec = {
                    "entry_time": nq.index[pos_entry_idx],
                    "exit_time": nq.index[min(i + (1 if exit_reason == "early_cut_pa" else 0), n-1)],
                    "r": r_mult, "reason": exit_reason, "dir": pos_direction,
                    "type": "limit_fvg", "trimmed": pos_trimmed, "grade": pos_grade,
                    "entry_price": pos_entry_price, "exit_price": exit_price,
                    "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
                    "stop_dist_pts": stop_dist_exit,
                }
                if track_mfe:
                    trade_rec["mfe_pts"] = pos_mfe
                    trade_rec["mfe_r"] = pos_mfe / stop_dist_exit if stop_dist_exit > 0 else 0
                trades.append(trade_rec)

                daily_pnl_r += r_mult
                if exit_reason == "be_sweep" and pos_trimmed:
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
                in_position = False

        # === PHASE D: ENTRY via limit order fill ===
        if in_position:
            continue
        if day_stopped:
            continue
        if news_blackout_arr is not None and news_blackout_arr[i]:
            continue

        et_frac = et_frac_arr[i]
        if 9.5 <= et_frac <= 10.0:
            continue
        if not (10.0 <= et_frac < 16.0):
            continue  # NY only

        # Check all active zones for fills
        best_fill = None
        best_zone_info = None

        for zone in active_zones:
            if zone.used or zone.birth_bar >= i:
                continue

            direction = 1 if zone.type == "bull" else -1

            if block_pm_shorts and direction == -1 and et_frac >= 14.0:
                continue

            # Entry + stop price
            if zone.type == "bull":
                entry_p = zone.top
                if stop_strategy == "A1":
                    stop_p = zone.bottom
                elif stop_strategy == "A2":
                    stop_p = zone.bottom - zone.size * stop_buffer_pct
                else:
                    stop_p = zone.bottom
            else:
                entry_p = zone.bottom
                if stop_strategy == "A1":
                    stop_p = zone.top
                elif stop_strategy == "A2":
                    stop_p = zone.top + zone.size * stop_buffer_pct
                else:
                    stop_p = zone.top

            stop_dist = abs(entry_p - stop_p)

            # AUDIT FIX #2: Apply tightening BEFORE fill-bar check
            if tighten_factor < 1.0:
                if zone.type == "bull":
                    stop_p = entry_p - stop_dist * tighten_factor
                else:
                    stop_p = entry_p + stop_dist * tighten_factor
                stop_dist = abs(entry_p - stop_p)

            cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0

            # Min stop filters (applied after tightening)
            if min_stop_atr > 0 and cur_atr > 0:
                if stop_dist / cur_atr < min_stop_atr:
                    continue
            if min_stop_pts > 0 and stop_dist < min_stop_pts:
                continue
            if stop_dist < 1.0:
                continue

            # Fill check: price must reach limit but NOT breach tightened stop
            if zone.type == "bull":
                if l[i] > entry_p:
                    continue  # price didn't reach limit
                if l[i] <= stop_p:
                    continue  # tightened stop also hit, skip
            else:
                if h[i] < entry_p:
                    continue
                if h[i] >= stop_p:
                    continue

            # Bias alignment
            if direction == -np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0:
                continue

            # Quality: prefer closest to current price
            fill_quality = -abs(c_arr[i] - entry_p)
            if best_fill is None or fill_quality > best_fill:
                best_fill = fill_quality
                best_zone_info = (zone, direction, entry_p, stop_p, stop_dist, cur_atr)

        if best_zone_info is None:
            continue

        zone, direction, entry_p, stop_p, stop_dist, cur_atr = best_zone_info
        zone.used = True
        zones_filled += 1
        # Stop tightening already applied in zone loop (AUDIT FIX #2)

        # Grade + sizing
        ba = 1.0 if (direction == np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0) else 0.0
        grade = _compute_grade(ba, regime_arr[i])
        is_reduced = (dow_arr[i] in (0, 4)) or (regime_arr[i] < 1.0)
        base_r = reduced_r if is_reduced else normal_r
        if grade == "A+":
            r_amount = base_r * a_plus_mult
        elif grade == "B+":
            r_amount = base_r * b_plus_mult
        else:
            r_amount = base_r * 0.5

        # Session regime (lunch dead zone)
        if session_regime.get("enabled", False):
            sr_lunch_s = session_regime.get("lunch_start", 12.0)
            sr_lunch_e = session_regime.get("lunch_end", 13.0)
            if sr_lunch_s <= et_frac < sr_lunch_e:
                sr_mult = session_regime.get("lunch_mult", 0.0)
            elif et_frac < session_regime.get("am_end", 12.5):
                sr_mult = session_regime.get("am_mult", 1.0)
            elif et_frac >= session_regime.get("pm_start", 13.0):
                sr_mult = session_regime.get("pm_mult", 1.0)
            else:
                sr_mult = 1.0
            r_amount *= sr_mult
            if r_amount <= 0:
                zone.used = False
                zones_filled -= 1
                continue

        contracts = max(1, int(r_amount / (stop_dist * point_value))) if stop_dist > 0 and point_value > 0 else 0
        if contracts <= 0:
            zone.used = False
            zones_filled -= 1
            continue

        # TP computation
        if direction == 1:
            irl_target = irl_high_arr[i]
            if np.isnan(irl_target) or irl_target <= entry_p:
                irl_target = entry_p + stop_dist * 2.0
            tp_distance = irl_target - entry_p
            tp1_price = entry_p + tp_distance * tp_mult
        else:
            tp1_price = entry_p - stop_dist * short_rr

        # Enter position
        in_position = True
        pos_direction = direction
        pos_entry_idx = i
        pos_entry_price = entry_p
        pos_stop = stop_p
        pos_tp1 = tp1_price
        pos_contracts = contracts
        pos_remaining_contracts = contracts
        pos_trimmed = False
        pos_be_stop = 0.0
        pos_trail_stop = 0.0
        pos_grade = grade
        pos_mfe = 0.0
        # Shorts: 100% trim (scalp). Longs: configurable.
        pos_trim_pct = 1.0 if direction == -1 else trim_pct

    # Force close at end
    if in_position:
        last_i = n - 1
        exit_price = c_arr[last_i]
        pnl_pts = (exit_price - pos_entry_price) if pos_direction == 1 else (pos_entry_price - exit_price)
        total_pnl = pnl_pts * point_value * pos_remaining_contracts
        if pos_trimmed:
            trim_c = pos_contracts - pos_remaining_contracts
            trim_pnl = ((pos_tp1 - pos_entry_price) if pos_direction == 1 else (pos_entry_price - pos_tp1)) * point_value * trim_c
            total_pnl += trim_pnl
            total_pnl -= commission_per_side * 2 * pos_contracts
        else:
            total_pnl -= commission_per_side * 2 * pos_remaining_contracts
        stop_dist_exit = abs(pos_entry_price - pos_stop)
        total_risk = stop_dist_exit * point_value * pos_contracts
        r_mult = total_pnl / total_risk if total_risk > 0 else 0.0
        trades.append({
            "entry_time": nq.index[pos_entry_idx], "exit_time": nq.index[last_i],
            "r": r_mult, "reason": "force_close", "dir": pos_direction,
            "type": "limit_fvg", "trimmed": pos_trimmed, "grade": pos_grade,
            "entry_price": pos_entry_price, "exit_price": exit_price,
            "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
        })

    stats = {"zones_created": zones_created, "zones_filled": zones_filled,
             "fill_rate_pct": 100.0 * zones_filled / zones_created if zones_created > 0 else 0}
    return trades, stats


# ======================================================================
# Metrics (same as validate_improvements.py)
# ======================================================================
def compute_metrics(trades_list):
    if not trades_list:
        return {"trades": 0, "R": 0.0, "PPDD": 0.0, "PF": 0.0, "WR": 0.0, "MaxDD": 0.0, "avgR": 0.0}
    r = np.array([t["r"] for t in trades_list])
    total_r = r.sum()
    wr = (r > 0).mean() * 100
    wins = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = wins / losses if losses > 0 else 999.0
    cumr = np.cumsum(r)
    peak = np.maximum.accumulate(cumr)
    dd = peak - cumr
    max_dd = dd.max() if len(dd) > 0 else 0
    ppdd = total_r / max_dd if max_dd > 0 else 999.0
    return {
        "trades": len(trades_list), "R": round(total_r, 1), "PPDD": round(ppdd, 2),
        "PF": round(pf, 2), "WR": round(wr, 1), "MaxDD": round(max_dd, 1),
        "avgR": round(total_r / len(trades_list), 4),
    }


def pr(label, m, extra=""):
    tpd = m["trades"] / (252 * 10.5) if m["trades"] > 0 else 0
    print(f"  {label:55s} | {m['trades']:5d}t | R={m['R']:+8.1f} | PPDD={m['PPDD']:6.2f} | PF={m['PF']:5.2f} | WR={m['WR']:5.1f}% | DD={m['MaxDD']:5.1f}R | {tpd:.2f}/d{extra}")


def walk_forward(trades_list):
    if not trades_list:
        return []
    df = pd.DataFrame(trades_list)
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year
    results = []
    for year, grp in df.groupby("year"):
        r = grp["r"].values
        total_r = r.sum()
        wr = (r > 0).mean() * 100
        wins = r[r > 0].sum()
        losses = abs(r[r < 0].sum())
        pf = wins / losses if losses > 0 else 999.0
        cumr = np.cumsum(r)
        peak = np.maximum.accumulate(cumr)
        dd = (peak - cumr).max()
        ppdd = total_r / dd if dd > 0 else 999.0
        results.append({"year": year, "n": len(grp), "R": round(total_r, 1),
                        "WR": round(wr, 1), "PF": round(pf, 2), "PPDD": round(ppdd, 2)})
    return results


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 120)
    print("U2 CLEAN — Limit Order at FVG Zone (CORRECT single-TP PnL)")
    print("=" * 120)

    d = load_all()

    # ================================================================
    # PHASE 1: Parameter sweep (EOD ON)
    # ================================================================
    print("\n" + "=" * 120)
    print("PHASE 1: U2 PARAMETER SWEEP (EOD=ON)")
    print("=" * 120)

    all_results = []

    # Stop strategy sweep
    for stop_strat in ["A1", "A2"]:
        for fvg_sz in [0.3, 0.5]:
            for max_age in [100, 200, 500]:
                for min_atr in [0.0, 0.5, 1.0, 1.7]:
                    trades, stats = run_u2_backtest(d,
                        stop_strategy=stop_strat, fvg_size_mult=fvg_sz,
                        max_fvg_age=max_age, min_stop_atr=min_atr)
                    m = compute_metrics(trades)
                    if m["trades"] >= 30:
                        label = f"U2 {stop_strat} sz>{fvg_sz} age<{max_age} minATR>{min_atr}"
                        all_results.append((label, m, trades, stats))

    all_results.sort(key=lambda x: x[1]["PF"], reverse=True)

    print(f"\n  Top 20 configs by PF (trades >= 30):")
    for label, m, _, stats in all_results[:20]:
        fill_rate = stats["fill_rate_pct"]
        pr(label, m, f" | fill={fill_rate:.1f}%")

    # ================================================================
    # PHASE 2: Same sweep WITHOUT EOD
    # ================================================================
    print("\n" + "=" * 120)
    print("PHASE 2: U2 PARAMETER SWEEP (EOD=OFF)")
    print("=" * 120)

    noEod_results = []
    for stop_strat in ["A1", "A2"]:
        for fvg_sz in [0.3, 0.5]:
            for max_age in [100, 200, 500]:
                for min_atr in [0.0, 0.5, 1.0, 1.7]:
                    trades, stats = run_u2_backtest(d,
                        stop_strategy=stop_strat, fvg_size_mult=fvg_sz,
                        max_fvg_age=max_age, min_stop_atr=min_atr, eod_close=False)
                    m = compute_metrics(trades)
                    if m["trades"] >= 30:
                        label = f"U2 noEOD {stop_strat} sz>{fvg_sz} age<{max_age} minATR>{min_atr}"
                        noEod_results.append((label, m, trades, stats))

    noEod_results.sort(key=lambda x: x[1]["PF"], reverse=True)

    print(f"\n  Top 20 configs by PF (trades >= 30, no EOD):")
    for label, m, _, stats in noEod_results[:20]:
        fill_rate = stats["fill_rate_pct"]
        pr(label, m, f" | fill={fill_rate:.1f}%")

    # ================================================================
    # PHASE 3: Best configs — walk-forward + PnL verify
    # ================================================================
    print("\n" + "=" * 120)
    print("PHASE 3: WALK-FORWARD for top configs")
    print("=" * 120)

    # Top 3 EOD, top 3 no-EOD
    for tag, results in [("EOD", all_results), ("noEOD", noEod_results)]:
        for label, m, trades, stats in results[:3]:
            tpd = m["trades"] / (252 * 10.5)
            print(f"\n  [{tag}] {label}")
            print(f"  {m['trades']}t R={m['R']:+.1f} PF={m['PF']:.2f} PPDD={m['PPDD']:.2f} {tpd:.2f}/day")
            wf = walk_forward(trades)
            neg = 0
            for yr in wf:
                flag = " NEG" if yr["R"] < 0 else ""
                if yr["R"] < 0: neg += 1
                print(f"    {yr['year']}: {yr['n']:3d}t  R={yr['R']:+7.1f}  WR={yr['WR']:5.1f}%  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}{flag}")
            print(f"    Negative years: {neg}/{len(wf)}")

            # Manual PnL verify (first 3 trades)
            print(f"    PnL VERIFY (first 3 trades):")
            for t in trades[:3]:
                ep = t["entry_price"]
                xp = t["exit_price"]
                sp = t["stop_price"]
                tp = t["tp1_price"]
                d_dir = "LONG" if t["dir"] == 1 else "SHORT"
                print(f"      {d_dir} entry={ep:.2f} stop={sp:.2f} tp1={tp:.2f} exit={xp:.2f} reason={t['reason']} R={t['r']:+.3f} ${t['pnl_dollars']:+.1f}")

    # ================================================================
    # PHASE 4: Exit distribution for best config
    # ================================================================
    print("\n" + "=" * 120)
    print("PHASE 4: EXIT DISTRIBUTION (best EOD config)")
    print("=" * 120)

    if all_results:
        _, m, trades, _ = all_results[0]
        df = pd.DataFrame(trades)
        for reason, grp in df.groupby("reason"):
            n_t = len(grp)
            avg_r = grp["r"].mean()
            total_r = grp["r"].sum()
            print(f"  {reason:15s}: {n_t:4d}t  avgR={avg_r:+.3f}  totalR={total_r:+.1f}")

        # Direction breakdown
        for dname, dval in [("LONG", 1), ("SHORT", -1)]:
            sub = df[df["dir"] == dval]
            if len(sub) == 0: continue
            wr = (sub["r"] > 0).mean() * 100
            print(f"\n  {dname}: {len(sub)}t, R={sub['r'].sum():+.1f}, WR={wr:.1f}%")

    # ================================================================
    # PHASE 5: SUMMARY
    # ================================================================
    print("\n" + "=" * 120)
    print("PHASE 5: SUMMARY — U2 vs F3 COMPARISON")
    print("=" * 120)

    # Run F3 baseline for reference
    from experiments.pure_scalp_validation import load_all as load_f3, run_backtest as run_f3, compute_metrics as cm_f3
    d_f3 = load_f3()
    f3_eod = cm_f3(run_f3(d_f3))
    f3_noeod = cm_f3(run_f3(d_f3, eod_close=False))

    print(f"\n  {'Config':55s} | {'Trades':>6} | {'t/day':>5} | {'PF':>5} | {'R':>8} | {'PPDD':>6} | {'WR':>5}")
    print("  " + "-" * 105)
    pr("F3 BASELINE (EOD)", f3_eod)
    pr("F3 BASELINE (no EOD)", f3_noeod)
    if all_results:
        pr(f"U2 BEST (EOD): {all_results[0][0]}", all_results[0][1])
    if noEod_results:
        pr(f"U2 BEST (noEOD): {noEod_results[0][0]}", noEod_results[0][1])

    # PF >= 1.8 check
    pf18_eod = [(l,m) for l,m,_,_ in all_results if m["PF"] >= 1.8]
    pf18_noEod = [(l,m) for l,m,_,_ in noEod_results if m["PF"] >= 1.8]
    print(f"\n  Configs with PF >= 1.8:")
    print(f"    EOD: {len(pf18_eod)} configs")
    for l, m in pf18_eod[:5]:
        tpd = m["trades"] / (252 * 10.5)
        print(f"      {l}: PF={m['PF']:.2f}, {m['trades']}t, {tpd:.2f}/day")
    print(f"    noEOD: {len(pf18_noEod)} configs")
    for l, m in pf18_noEod[:5]:
        tpd = m["trades"] / (252 * 10.5)
        print(f"      {l}: PF={m['PF']:.2f}, {m['trades']}t, {tpd:.2f}/day")

    print("\n" + "=" * 120)
    print("DONE")


if __name__ == "__main__":
    main()
