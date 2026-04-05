"""
experiments/chain_engine.py — ICT Chain Strategy Production Engine
=================================================================

Entry: Overnight level breakdown → FVG (NOT MSS) → limit order at FVG zone
TP:    Fixed 1R, 25% trim, 75% runner trail nth swing
Exit:  Symmetric management for longs and shorts
Filter: Reject FVG where displacement breaks swing point

Sprint 8 validated findings:
  - Breakdown at overnight levels (ON_low for longs, ON_high for shorts)
  - FVG NOT MSS: PF=2.96 (post-hoc), R=+1069.9
  - Fixed 1R trim: monotonically better than further TP
  - Symmetric runner management: shorts PF=1.82→2.48
  - Bootstrap CI > 1.0, temporal split HOLDS, 1/11 neg years
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
# Data loading
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
    assert len(atr_cache) == n
    atr_arr = atr_cache["atr"].values

    bias = pd.read_parquet(DATA / "cache_bias_10yr_v2.parquet")
    regime = pd.read_parquet(DATA / "cache_regime_10yr_v2.parquet")
    assert len(bias) == n and len(regime) == n
    bias_dir_arr = bias["bias_direction"].values.astype(float)
    regime_arr = regime["regime"].values.astype(float)

    # Swing points (5m fractal)
    swing_p = {"left_bars": params["swing"]["left_bars"],
               "right_bars": params["swing"]["right_bars"]}
    rb = swing_p["right_bars"]
    swings = compute_swing_levels(nq, swing_p)

    # Shifted masks (no lookahead)
    swing_high_mask = swings["swing_high"].shift(rb, fill_value=False).values
    swing_low_mask = swings["swing_low"].shift(rb, fill_value=False).values

    # Price at shifted mask positions (for trailing)
    raw_sh = swings["swing_high"].values
    raw_sl = swings["swing_low"].values
    swing_high_price_at_mask = np.full(n, np.nan)
    swing_low_price_at_mask = np.full(n, np.nan)
    for j in range(n):
        if raw_sl[j] and j + rb < n:
            swing_low_price_at_mask[j + rb] = l[j]
        if raw_sh[j] and j + rb < n:
            swing_high_price_at_mask[j + rb] = h[j]

    # News blackout
    news_path = PROJECT / "config" / "news_calendar.csv"
    news_blackout_arr = None
    if news_path.exists():
        news_bl = build_news_blackout_mask(nq.index, str(news_path),
            params["news"]["blackout_minutes_before"],
            params["news"]["cooldown_minutes_after"])
        news_blackout_arr = news_bl.values

    # FVG detection
    fvg_df = detect_fvg(nq)

    # Session levels
    session_cache = pd.read_parquet(DATA / "cache_session_levels_10yr_v2.parquet")
    assert len(session_cache) == n
    on_hi = session_cache["overnight_high"].values
    on_lo = session_cache["overnight_low"].values
    asia_hi = session_cache["asia_high"].values
    asia_lo = session_cache["asia_low"].values
    london_hi = session_cache["london_high"].values
    london_lo = session_cache["london_low"].values

    # Previous day high/low
    from experiments.sweep_research import compute_pdhl
    pdhl = compute_pdhl(nq)
    pdh = pdhl["pdh"].values
    pdl = pdhl["pdl"].values

    # Time arrays
    dates = np.array([
        (et_idx[j] + pd.Timedelta(days=1)).date() if et_idx[j].hour >= 18 else et_idx[j].date()
        for j in range(n)
    ])
    dow_arr = np.array([et_idx[j].dayofweek for j in range(n)])
    et_frac_arr = np.array([et_idx[j].hour + et_idx[j].minute / 60.0 for j in range(n)])

    elapsed = _time.perf_counter() - t0
    print(f"[LOAD] Done in {elapsed:.1f}s (n={n})")

    return {
        "nq": nq, "params": params, "n": n,
        "o": o, "h": h, "l": l, "c": c,
        "atr_arr": atr_arr,
        "bias_dir_arr": bias_dir_arr, "regime_arr": regime_arr,
        "swing_high_mask": swing_high_mask, "swing_low_mask": swing_low_mask,
        "swing_high_price_at_mask": swing_high_price_at_mask,
        "swing_low_price_at_mask": swing_low_price_at_mask,
        "news_blackout_arr": news_blackout_arr,
        "fvg_df": fvg_df,
        "on_hi": on_hi, "on_lo": on_lo,
        "asia_hi": asia_hi, "asia_lo": asia_lo,
        "london_hi": london_hi, "london_lo": london_lo,
        "pdh": pdh, "pdl": pdl,
        "dates": dates, "dow_arr": dow_arr, "et_frac_arr": et_frac_arr,
        "et_idx": et_idx,
    }


# ======================================================================
# Breakdown detection
# ======================================================================
def detect_breakdowns(h, l, c, level, level_type, min_depth_pts=1.0):
    """Detect bars where price closes through a significant level.
    Low levels: close below (bearish breakdown → bull reversal expected).
    High levels: close above (bullish breakdown → bear reversal expected).
    """
    n = len(h)
    events = []
    last_bar = -10

    for i in range(1, n):
        lev = level[i]
        if np.isnan(lev) or lev <= 0 or i - last_bar < 3:
            continue

        if level_type == "low":
            if c[i] < lev - min_depth_pts and l[i] < lev and c[i-1] >= lev:
                events.append({"bar_idx": i, "level_type": "low"})
                last_bar = i
        else:
            if c[i] > lev + min_depth_pts and h[i] > lev and c[i-1] <= lev:
                events.append({"bar_idx": i, "level_type": "high"})
                last_bar = i

    return events


# ======================================================================
# FVG zone with NOT-MSS filter
# ======================================================================
@dataclass
class ChainZone:
    direction: int       # 1=long, -1=short
    top: float
    bottom: float
    size: float
    birth_bar: int
    birth_atr: float
    breakdown_bar: int = -1
    sweep_range_atr: float = 0.0  # breakdown bar range / ATR (sizing signal)
    used: bool = False


def find_fvg_not_mss(
    fvg_bull_mask, fvg_bull_top, fvg_bull_bottom,
    fvg_bear_mask, fvg_bear_top, fvg_bear_bottom,
    h, l, c, o, atr_arr,
    swing_hi_shifted, swing_lo_shifted,
    sh_prices, sl_prices,
    breakdown_bar: int,
    level_type: str,
    max_wait_bars: int = 30,
    min_fvg_size_atr: float = 0.3,
) -> ChainZone | None:
    """Find first qualifying FVG after breakdown that does NOT break swing (NOT MSS).

    For low breakdown → look for bull FVG whose displacement does NOT break swing high
    For high breakdown → look for bear FVG whose displacement does NOT break swing low
    """
    n = len(h)
    direction = 1 if level_type == "low" else -1

    if direction == 1:
        mask, tops, bottoms = fvg_bull_mask, fvg_bull_top, fvg_bull_bottom
    else:
        mask, tops, bottoms = fvg_bear_mask, fvg_bear_top, fvg_bear_bottom

    for i in range(breakdown_bar + 1, min(breakdown_bar + max_wait_bars + 1, n)):
        if not mask[i]:
            continue

        top = tops[i]
        bottom = bottoms[i]
        size = top - bottom
        atr_val = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0

        if atr_val > 0 and size < min_fvg_size_atr * atr_val:
            continue

        # NOT MSS check: displacement candle must NOT break a swing point
        disp_bar = i - 1
        if disp_bar < 0:
            continue

        # AUDIT FIX #4: Search from disp_bar (not breakdown_bar) to catch
        # swing points that formed BETWEEN breakdown and FVG displacement.
        breaks_swing = False
        if direction == 1:
            for j in range(disp_bar, max(0, disp_bar - 100), -1):
                if swing_hi_shifted[j]:
                    p = sh_prices[j]
                    if not np.isnan(p) and p > 0 and h[disp_bar] > p:
                        breaks_swing = True
                        break
        else:
            for j in range(disp_bar, max(0, disp_bar - 100), -1):
                if swing_lo_shifted[j]:
                    p = sl_prices[j]
                    if not np.isnan(p) and p > 0 and l[disp_bar] < p:
                        breaks_swing = True
                        break

        if breaks_swing:
            continue  # REJECT: FVG is MSS, skip it

        return ChainZone(
            direction=direction,
            top=top, bottom=bottom, size=size,
            birth_bar=i,
            birth_atr=atr_val,
            breakdown_bar=breakdown_bar,
        )

    return None


# ======================================================================
# Swing finder
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
# Main engine
# ======================================================================
def run_chain_backtest(
    d: dict,
    *,
    # Breakdown sources: list of (level_array, level_type) tuples
    # If None, uses overnight only
    breakdown_sources: list[tuple[np.ndarray, str]] | None = None,
    # FVG params
    max_wait_bars: int = 30,
    min_fvg_size_atr: float = 0.3,
    max_fvg_age: int = 200,
    # Stop params
    stop_buffer_pct: float = 0.15,
    tighten_factor: float = 0.85,
    min_stop_pts: float = 5.0,
    # Exit params
    trim_pct: float = 0.25,
    fixed_tp_r: float = 1.0,
    nth_swing: int = 5,
    be_after_trim: bool = True,
    eod_close: bool = True,
    # Sizing signals (validated Sprint 8)
    big_sweep_threshold: float = 1.3,  # ATR mult: sweep bar range >= this → 1.5x
    big_sweep_mult: float = 1.5,
    am_short_mult: float = 0.5,       # AM (10-12 ET) shorts → reduced size
) -> tuple[list[dict], dict]:
    """Run ICT chain backtest.

    Flow per bar:
      1. Detect breakdowns at overnight levels
      2. For each breakdown, find FVG (NOT MSS)
      3. Limit order at FVG zone
      4. Symmetric trim/trail/BE management
      5. Sizing: big sweep → 1.5x, AM shorts → 0.5x
    """

    params = d["params"]
    nq, n = d["nq"], d["n"]
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    atr_arr = d["atr_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    fvg_df = d["fvg_df"]
    on_hi, on_lo = d["on_hi"], d["on_lo"]
    dates = d["dates"]
    dow_arr = d["dow_arr"]
    et_frac_arr = d["et_frac_arr"]
    swing_high_mask = d["swing_high_mask"]
    swing_low_mask = d["swing_low_mask"]
    swing_high_price_at_mask = d["swing_high_price_at_mask"]
    swing_low_price_at_mask = d["swing_low_price_at_mask"]

    pos_params = params["position"]
    risk_params = params["risk"]
    bt_params = params["backtest"]
    grading_params = params["grading"]
    session_regime = params.get("session_regime", {})

    normal_r = pos_params["normal_r"]
    reduced_r = pos_params["reduced_r"]
    point_value = pos_params["point_value"]
    daily_max_loss_r = risk_params["daily_max_loss_r"]
    max_consec = risk_params["max_consecutive_losses"]
    comm = bt_params["commission_per_side_micro"]
    slip = bt_params["slippage_normal_ticks"] * 0.25
    a_mult = grading_params["a_plus_size_mult"]
    b_mult = grading_params["b_plus_size_mult"]

    # FVG arrays
    fvg_bm = fvg_df["fvg_bull"].values
    fvg_bt = fvg_df["fvg_bull_top"].values
    fvg_bb = fvg_df["fvg_bull_bottom"].values
    fvg_em = fvg_df["fvg_bear"].values
    fvg_et = fvg_df["fvg_bear_top"].values
    fvg_eb = fvg_df["fvg_bear_bottom"].values

    # Swing price arrays (for NOT-MSS check)
    sh_prices = swing_high_price_at_mask
    sl_prices = swing_low_price_at_mask

    # ---- Pre-compute: detect breakdowns + find FVG zones ----
    if breakdown_sources is None:
        breakdown_sources = [
            (d["on_lo"], "low"),
            (d["on_hi"], "high"),
        ]

    all_bds_raw = []
    for level_arr, level_type in breakdown_sources:
        bds = detect_breakdowns(h, l, c, level_arr, level_type,
                                min_depth_pts=params.get("chain", {}).get("min_depth_pts", 1.0))
        all_bds_raw.extend(bds)

    # AUDIT FIX #3: Deduplicate cross-level breakdowns within 3 bars
    all_bds_raw.sort(key=lambda x: x["bar_idx"])
    all_bds = []
    last_bar = -10
    for bd in all_bds_raw:
        if bd["bar_idx"] - last_bar >= 3:
            all_bds.append(bd)
            last_bar = bd["bar_idx"]

    zones_by_bar: dict[int, list[ChainZone]] = {}
    n_bds = 0
    n_fvgs = 0
    n_fvgs_mss_rejected = 0

    for bd in all_bds:
        # Count total FVGs before NOT-MSS filter
        direction = 1 if bd["level_type"] == "low" else -1
        fvg_mask = fvg_bm if direction == 1 else fvg_em
        fvg_tops = fvg_bt if direction == 1 else fvg_et
        fvg_bots = fvg_bb if direction == 1 else fvg_eb

        # Try to find any FVG first (for counting)
        found_any = False
        for ii in range(bd["bar_idx"] + 1, min(bd["bar_idx"] + max_wait_bars + 1, n)):
            if fvg_mask[ii]:
                sz = fvg_tops[ii] - fvg_bots[ii]
                av = atr_arr[ii] if not np.isnan(atr_arr[ii]) else 30.0
                if av > 0 and sz >= min_fvg_size_atr * av:
                    found_any = True
                    break

        zone = find_fvg_not_mss(
            fvg_bm, fvg_bt, fvg_bb, fvg_em, fvg_et, fvg_eb,
            h, l, c, o, atr_arr,
            swing_high_mask, swing_low_mask, sh_prices, sl_prices,
            bd["bar_idx"], bd["level_type"],
            max_wait_bars, min_fvg_size_atr,
        )

        n_bds += 1
        if found_any and zone is None:
            n_fvgs_mss_rejected += 1
        if zone is not None:
            # Compute sweep bar range for sizing
            bd_idx = bd["bar_idx"]
            bd_atr = atr_arr[bd_idx] if not np.isnan(atr_arr[bd_idx]) else 30.0
            sweep_range = h[bd_idx] - l[bd_idx]
            zone.sweep_range_atr = sweep_range / bd_atr if bd_atr > 0 else 0
            n_fvgs += 1
            zones_by_bar.setdefault(zone.birth_bar, []).append(zone)

    # ---- Bar-by-bar engine ----
    active_zones: list[ChainZone] = []
    trades: list[dict] = []
    in_pos = False
    p_dir = p_idx = 0
    p_entry = p_stop = p_tp1 = 0.0
    p_contracts = p_remaining = 0
    p_trimmed = False
    p_be = p_trail = 0.0
    p_grade = ""
    p_sweep_atr = 0.0
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

        # Register new zones
        if i in zones_by_bar:
            active_zones.extend(zones_by_bar[i])

        # Invalidate / expire
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

        # ---- EXIT MANAGEMENT ----
        if in_pos:
            exited = False
            ex_reason = ""
            ex_price = 0.0
            ex_contracts = p_remaining

            if p_dir == 1:
                eff = p_trail if p_trimmed and p_trail > 0 else p_stop
                if p_trimmed and p_be > 0:
                    eff = max(eff, p_be)
                if l[i] <= eff:
                    ex_price = eff - slip
                    ex_reason = "be_sweep" if (p_trimmed and eff >= p_entry) else "stop"
                    exited = True
                elif not p_trimmed and h[i] >= p_tp1:
                    tc = max(1, int(p_contracts * trim_pct))
                    p_remaining = p_contracts - tc
                    p_trimmed = True
                    p_be = p_entry
                    if p_remaining > 0:
                        p_trail = _find_nth_swing(swing_low_mask, swing_low_price_at_mask, i, nth_swing)
                        if np.isnan(p_trail) or p_trail <= 0:
                            p_trail = p_be
                    if p_remaining <= 0:
                        ex_price = p_tp1
                        ex_reason = "tp1"
                        ex_contracts = p_contracts
                        exited = True
                if p_trimmed and not exited:
                    nt = _find_nth_swing(swing_low_mask, swing_low_price_at_mask, i, nth_swing)
                    if not np.isnan(nt) and nt > p_trail:
                        p_trail = nt

            else:  # Short — symmetric
                # AUDIT FIX #1: For shorts, effective stop = min(trail, BE) since trail is ABOVE price
                eff = p_trail if p_trimmed and p_trail > 0 else p_stop
                if p_trimmed and p_be > 0:
                    if eff > p_be:
                        eff = p_be  # trail above entry → use BE instead
                    # else trail already below BE, use trail (tighter)
                if h[i] >= eff:
                    ex_price = eff + slip
                    ex_reason = "be_sweep" if (p_trimmed and eff <= p_entry) else "stop"
                    exited = True
                elif not p_trimmed and l[i] <= p_tp1:
                    tc = max(1, int(p_contracts * trim_pct))
                    p_remaining = p_contracts - tc
                    p_trimmed = True
                    p_be = p_entry
                    if p_remaining > 0:
                        # AUDIT FIX #1: Initialize short trail to nth swing high.
                        # If swing high is above entry, start at BE (entry) — trail will
                        # tighten as price drops and new lower swing highs form.
                        nt_init = _find_nth_swing(swing_high_mask, swing_high_price_at_mask, i, nth_swing)
                        if np.isnan(nt_init) or nt_init <= 0:
                            p_trail = p_be
                        elif nt_init > p_entry:
                            p_trail = p_be  # swing high above entry → start at BE
                        else:
                            p_trail = nt_init  # swing high below entry → use it (tighter)
                    if p_remaining <= 0:
                        ex_price = p_tp1
                        ex_reason = "tp1"
                        ex_contracts = p_contracts
                        exited = True
                if p_trimmed and not exited:
                    nt = _find_nth_swing(swing_high_mask, swing_high_price_at_mask, i, nth_swing)
                    if not np.isnan(nt) and nt > 0 and nt < p_trail:
                        p_trail = nt

            # EOD close
            if not exited and eod_close and et_frac_arr[i] >= 15.917:
                ex_price = c[i] - 0.25 if p_dir == 1 else c[i] + 0.25
                ex_reason = "eod_close"
                exited = True

            # PA early cut (bars 2-4)
            bars_in = i - p_idx
            if not exited and not p_trimmed and 2 <= bars_in <= 4:
                pa_s, pa_e = max(p_idx, 0), i + 1
                pa_range = h[pa_s:pa_e] - l[pa_s:pa_e]
                pa_body = np.abs(c[pa_s:pa_e] - o[pa_s:pa_e])
                safe = np.where(pa_range > 0, pa_range, 1.0)
                avg_wick = float(np.mean(1.0 - pa_body / safe))
                pa_dirs = np.sign(c[pa_s:pa_e] - o[pa_s:pa_e])
                favorable = (pa_dirs == p_dir).mean()
                disp = (c[i] - p_entry) * p_dir
                cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                if avg_wick > 0.65 and favorable < 0.5 and disp < cur_atr * 0.3 and bars_in >= 3:
                    ex_price = o[i+1] if i+1 < n else c[i]
                    ex_reason = "early_cut_pa"
                    exited = True

            if exited:
                pnl_pts = (ex_price - p_entry) * p_dir
                if p_trimmed and ex_reason != "tp1":
                    trim_c = p_contracts - ex_contracts
                    trim_pnl = (p_tp1 - p_entry) * p_dir * point_value * trim_c
                    total_pnl = trim_pnl + pnl_pts * point_value * ex_contracts
                    total_pnl -= comm * 2 * p_contracts
                else:
                    total_pnl = pnl_pts * point_value * ex_contracts
                    total_pnl -= comm * 2 * ex_contracts
                sd = abs(p_entry - p_stop)
                risk = sd * point_value * p_contracts
                r = total_pnl / risk if risk > 0 else 0.0

                trades.append({
                    "entry_time": nq.index[p_idx],
                    "exit_time": nq.index[min(i + (1 if ex_reason == "early_cut_pa" else 0), n-1)],
                    "r": r, "reason": ex_reason, "dir": p_dir,
                    "type": "chain", "trimmed": p_trimmed, "grade": p_grade,
                    "entry_price": p_entry, "exit_price": ex_price,
                    "stop_price": p_stop, "tp1_price": p_tp1,
                    "pnl_dollars": total_pnl, "stop_dist_pts": sd,
                    "sweep_range_atr": p_sweep_atr,
                    "contracts": p_contracts,
                })

                day_pnl += r
                if ex_reason == "be_sweep" and p_trimmed:
                    pass
                elif ex_reason == "eod_close":
                    pass
                elif r < 0:
                    consec_loss += 1
                else:
                    consec_loss = 0
                if consec_loss >= max_consec:
                    day_stopped = True
                if day_pnl <= -daily_max_loss_r:
                    day_stopped = True
                in_pos = False

        # ---- ENTRY ----
        if in_pos or day_stopped:
            continue
        if news_blackout_arr is not None and news_blackout_arr[i]:
            continue
        ef = et_frac_arr[i]
        if 9.5 <= ef <= 10.0 or not (10.0 <= ef < 16.0):
            continue

        best_fill = None
        best_zone_info = None
        best_sbs = False

        for z in active_zones:
            if z.used or z.birth_bar >= i:
                continue

            if z.direction == 1:
                ep = z.top
                sp = z.bottom - z.size * stop_buffer_pct
            else:
                ep = z.bottom
                sp = z.top + z.size * stop_buffer_pct

            sd = abs(ep - sp)
            if tighten_factor < 1.0:
                sp = ep - sd * tighten_factor if z.direction == 1 else ep + sd * tighten_factor
                sd = abs(ep - sp)

            if sd < min_stop_pts or sd < 1.0:
                continue

            # Fill check
            sbs = False
            if z.direction == 1:
                if l[i] > ep:
                    continue
                if l[i] <= sp:
                    sbs = True
            else:
                if h[i] < ep:
                    continue
                if h[i] >= sp:
                    sbs = True

            # Bias alignment
            if z.direction == 1 and bias_dir_arr[i] < 0:
                continue
            if z.direction == -1 and bias_dir_arr[i] > 0:
                continue

            fq = -abs(c[i] - ep)
            if best_fill is None or fq > best_fill:
                best_fill = fq
                best_zone_info = (z, ep, sp, sd)
                best_sbs = sbs

        if best_zone_info is None:
            continue

        z, ep, sp, sd = best_zone_info
        z.used = True
        direction = z.direction

        # Grade + sizing
        ba = 1.0 if (direction == np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0) else 0.0
        grade = _compute_grade(ba, regime_arr[i])
        is_reduced = (dow_arr[i] in (0, 4)) or (regime_arr[i] < 1.0)
        base_r = reduced_r if is_reduced else normal_r
        if grade == "A+":
            r_amount = base_r * a_mult
        elif grade == "B+":
            r_amount = base_r * b_mult
        else:
            r_amount = base_r * 0.5

        # Sizing signals (validated)
        # Signal 1: Big sweep bar → increase size
        if z.sweep_range_atr >= big_sweep_threshold:
            r_amount *= big_sweep_mult
        # Signal 2: AM shorts → reduce size
        if direction == -1 and 10.0 <= ef < 12.0:
            r_amount *= am_short_mult

        # Session regime (skip lunch)
        if session_regime.get("enabled", False):
            ls = session_regime.get("lunch_start", 12.0)
            le = session_regime.get("lunch_end", 13.0)
            if ls <= ef < le:
                z.used = False
                continue

        contracts = max(1, int(r_amount / (sd * point_value))) if sd > 0 else 0
        if contracts <= 0:
            z.used = False
            continue

        # Same-bar stop
        if best_sbs:
            exp = (sp - slip) if direction == 1 else (sp + slip)
            pp = ((exp - ep) if direction == 1 else (ep - exp)) * point_value * contracts
            pp -= comm * 2 * contracts
            rr = pp / (sd * point_value * contracts) if sd > 0 else 0
            trades.append({
                "entry_time": nq.index[i], "exit_time": nq.index[i],
                "r": rr, "reason": "same_bar_stop", "dir": direction,
                "type": "chain", "trimmed": False, "grade": grade,
                "entry_price": ep, "exit_price": exp,
                "stop_price": sp, "tp1_price": 0.0,
                "pnl_dollars": pp, "stop_dist_pts": sd,
                "sweep_range_atr": z.sweep_range_atr,
                "contracts": contracts,
            })
            day_pnl += rr
            consec_loss += 1
            if consec_loss >= max_consec:
                day_stopped = True
            if day_pnl <= -daily_max_loss_r:
                day_stopped = True
            continue

        # Fixed 1R TP
        tp1 = ep + sd * fixed_tp_r if direction == 1 else ep - sd * fixed_tp_r

        # Enter position
        in_pos = True
        p_dir = direction
        p_idx = i
        p_entry = ep
        p_stop = sp
        p_tp1 = tp1
        p_contracts = contracts
        p_remaining = contracts
        p_trimmed = False
        p_be = 0.0
        p_trail = 0.0
        p_grade = grade
        p_sweep_atr = z.sweep_range_atr

    # Force close
    if in_pos:
        last_i = n - 1
        ex_price = c[last_i] - 0.25 if p_dir == 1 else c[last_i] + 0.25
        pnl_pts = (ex_price - p_entry) * p_dir
        total_pnl = pnl_pts * point_value * p_remaining
        if p_trimmed:
            trim_c = p_contracts - p_remaining
            total_pnl += (p_tp1 - p_entry) * p_dir * point_value * trim_c
            total_pnl -= comm * 2 * p_contracts
        else:
            total_pnl -= comm * 2 * p_remaining
        sd = abs(p_entry - p_stop)
        risk = sd * point_value * p_contracts
        r = total_pnl / risk if risk > 0 else 0
        trades.append({
            "entry_time": nq.index[p_idx], "exit_time": nq.index[last_i],
            "r": r, "reason": "force_close", "dir": p_dir,
            "type": "chain", "trimmed": p_trimmed, "grade": p_grade,
            "entry_price": p_entry, "exit_price": ex_price,
            "stop_price": p_stop, "tp1_price": p_tp1,
            "pnl_dollars": total_pnl, "stop_dist_pts": sd,
            "sweep_range_atr": p_sweep_atr,
            "contracts": p_contracts,
        })

    stats = {
        "breakdowns": n_bds,
        "fvgs_found": n_fvgs,
        "fvgs_mss_rejected": n_fvgs_mss_rejected,
    }
    return trades, stats


# ======================================================================
# Metrics
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


def pr(label, m):
    tpd = m["trades"] / (252 * 10.5) if m["trades"] > 0 else 0
    print(f"  {label:55s} | {m['trades']:5d}t | R={m['R']:+8.1f} | PPDD={m['PPDD']:6.2f} | PF={m['PF']:5.2f} | WR={m['WR']:5.1f}% | DD={m['MaxDD']:5.1f}R | {tpd:.2f}/d")


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 120)
    print("ICT CHAIN ENGINE -- Breakdown Source Sweep")
    print("=" * 120)

    d = load_all()

    # Define source configs
    configs = [
        ("ON only",
         [(d["on_lo"], "low"), (d["on_hi"], "high")]),
        ("PDL+PDH only",
         [(d["pdl"], "low"), (d["pdh"], "high")]),
        ("ON + PDL/PDH",
         [(d["on_lo"], "low"), (d["on_hi"], "high"),
          (d["pdl"], "low"), (d["pdh"], "high")]),
        ("ON + Asia",
         [(d["on_lo"], "low"), (d["on_hi"], "high"),
          (d["asia_lo"], "low"), (d["asia_hi"], "high")]),
        ("ON + London",
         [(d["on_lo"], "low"), (d["on_hi"], "high"),
          (d["london_lo"], "low"), (d["london_hi"], "high")]),
        ("ON + PDL/PDH + Asia",
         [(d["on_lo"], "low"), (d["on_hi"], "high"),
          (d["pdl"], "low"), (d["pdh"], "high"),
          (d["asia_lo"], "low"), (d["asia_hi"], "high")]),
        ("ALL levels",
         [(d["on_lo"], "low"), (d["on_hi"], "high"),
          (d["pdl"], "low"), (d["pdh"], "high"),
          (d["asia_lo"], "low"), (d["asia_hi"], "high"),
          (d["london_lo"], "low"), (d["london_hi"], "high")]),
    ]

    print(f"\n{'='*120}")
    print("BREAKDOWN SOURCE SWEEP (Fixed 1R TP, NOT-MSS filter)")
    print(f"{'='*120}")
    print(f"\n  {'Config':30s} | {'N':>5s} | {'R':>8s} | {'PF':>6s} | {'PPDD':>7s} | {'DD':>6s} | {'/day':>5s} | {'L:PF':>6s} | {'S:PF':>6s} | {'BDs':>6s} | {'FVGs':>5s} | {'Rej':>5s}")
    print(f"  {'-'*30}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}")

    best_trades = None
    best_label = ""
    best_ppdd = 0

    for label, sources in configs:
        trades, stats = run_chain_backtest(d, breakdown_sources=sources)
        if len(trades) < 20:
            print(f"  {label:30s} | {len(trades):5d} | too few")
            continue
        longs = [t for t in trades if t["dir"] == 1]
        shorts = [t for t in trades if t["dir"] == -1]
        m = compute_metrics(trades)
        ml = compute_metrics(longs) if longs else {"PF": 0}
        ms = compute_metrics(shorts) if shorts else {"PF": 0}
        tpd = m["trades"] / (252 * 10.5)
        print(f"  {label:30s} | {m['trades']:5d} | {m['R']:+8.1f} | {m['PF']:5.2f} | {m['PPDD']:6.2f} | {m['MaxDD']:5.1f}R | {tpd:.2f} | {ml['PF']:5.2f} | {ms['PF']:5.2f} | {stats['breakdowns']:6d} | {stats['fvgs_found']:5d} | {stats['fvgs_mss_rejected']:5d}")

        if m["PPDD"] > best_ppdd and m["trades"] >= 50:
            best_ppdd = m["PPDD"]
            best_trades = trades
            best_label = label

    # Walk-forward for best config
    if best_trades:
        print(f"\n{'='*120}")
        print(f"WALK-FORWARD: {best_label}")
        print(f"{'='*120}")
        df = pd.DataFrame(best_trades)
        df["year"] = pd.to_datetime(df["entry_time"]).dt.year
        neg = 0
        for year, grp in df.groupby("year"):
            r_arr = grp["r"].values
            total_r = r_arr.sum()
            wins = r_arr[r_arr > 0].sum()
            losses = abs(r_arr[r_arr < 0].sum())
            pf = wins / losses if losses > 0 else 999
            flag = " NEG" if total_r < 0 else ""
            if total_r < 0: neg += 1
            nl = len(grp[grp["dir"] == 1])
            ns = len(grp[grp["dir"] == -1])
            print(f"  {year}: {len(grp):3d}t (L:{nl:3d} S:{ns:3d})  R={total_r:+7.1f}  PF={pf:5.2f}{flag}")
        print(f"  Negative years: {neg}/{len(df['year'].unique())}")

        # Exit reasons
        print(f"\n  Exit reasons:")
        for reason, grp in df.groupby("reason"):
            print(f"    {reason:20s}: {len(grp):4d}t  R={grp['r'].sum():+8.1f}  avgR={grp['r'].mean():+.3f}")

        # PnL verify (first 3)
        print(f"\n  PnL verify (first 3):")
        for t in best_trades[:3]:
            tag = "LONG" if t["dir"] == 1 else "SHORT"
            print(f"    {tag} {t['entry_time']} ep={t['entry_price']:.2f} sp={t['stop_price']:.2f} "
                  f"tp={t['tp1_price']:.2f} xp={t['exit_price']:.2f} {t['reason']} R={t['r']:+.3f}")

    print(f"\n{'='*120}")
    print("BASELINE COMPARISON")
    print(f"{'='*120}")
    print(f"  U2-v2 baseline (longs only): 2589t R=+791.5 PF=1.59 PPDD=28.35 DD=27.9R 0.98/d")
    print(f"  {'Post-hoc estimate (FVG NOT MSS + 1R)':55s} | ~987t | R=+1069.9 | PPDD= 71.43 | PF= 2.96 |")


if __name__ == "__main__":
    main()
