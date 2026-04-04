"""
features/entry_signals.py — Entry signal detection for Lanto's two ICT models.

Model 1 (Trend): FVG test + rejection with displacement → enter on closure
Model 2 (MSS):   Liquidity sweep → FVG invalidation (IFVG) → retest + respect → enter

All signals fire at the **close** of the signal bar (no future leakage).
FVG state is tracked bar-by-bar using a rolling approach to ensure
only *previously known* FVGs are considered at each bar.

Quality filters (all from params.yaml):
  - Rejection candle body/range >= displacement.body_ratio (not just "not doji")
  - FVG size >= entry.min_fvg_atr_mult * ATR  (filter tiny FVGs)
  - Fluency >= fluency.threshold over recent bars (clean PA leading in)
  - Per-FVG cooldown: one signal per FVG per entry.signal_cooldown_bars
  - ATR-based FVG size filter ensures only meaningful gaps trigger

References: CLAUDE.md sections 4, 5, 6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from features.displacement import compute_atr, compute_fluency, detect_displacement
from features.fvg import FVGRecord, _update_fvg_status, detect_fvg
from features.swing import compute_swing_levels

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "params.yaml"

# Default entry params — overridden by params.yaml if present
_DEFAULT_ENTRY_PARAMS = {
    "min_fvg_atr_mult": 0.3,        # FVG size must be >= this * ATR
    "rejection_body_ratio": 0.50,    # rejection candle body/range min
    "signal_cooldown_bars": 6,       # bars before same FVG can signal again
    "require_displacement": False,   # if True, only displaced candles signal
    "sweep_lookback": 20,            # bars to look back for liquidity sweep (MSS)
}


def _load_params(config_path: Path = _CONFIG_PATH) -> dict:
    """Load tunable parameters from params.yaml."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _get_entry_params(params: dict) -> dict:
    """Extract entry-specific params, falling back to defaults."""
    entry_cfg = params.get("entry", {})
    result = dict(_DEFAULT_ENTRY_PARAMS)
    result.update(entry_cfg)
    return result


# ---------------------------------------------------------------------------
# Internal: lightweight FVG tracker for per-bar signal detection
# ---------------------------------------------------------------------------

@dataclass
class _LiveFVG:
    """Mutable record for a single FVG being tracked during signal scan."""
    idx: int              # bar index where FVG became visible
    direction: str        # 'bull' or 'bear'
    top: float
    bottom: float
    size: float
    candle2_open: float = 0.0  # open of the displacement candle (model stop level)
    status: str = "untested"   # untested | tested_rejected | invalidated
    is_ifvg: bool = False
    ifvg_direction: str = ""   # 'bull' or 'bear' for IFVGs
    swept_liquidity: bool = False  # did displacement candle sweep a swing/session level?
    sweep_score: int = 0           # how many significant levels were swept (0-10+)
    invalidated_at_idx: int = -1
    last_signal_idx: int = -999  # bar index of last signal fired from this FVG


# ---------------------------------------------------------------------------
# Liquidity level computation — running H/L per sub-session
# ---------------------------------------------------------------------------

def _compute_liquidity_levels(
    df: pd.DataFrame,
    params: dict,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute real-time running H/L for each sub-session as liquidity targets.

    Sub-sessions (ET):
      Asia:     18:00 - 03:00
      London:   03:00 - 09:30
      NY AM:    09:30 - 11:00 (Lanto's prime window)
      NY Lunch: 11:00 - 13:00
      NY PM:    13:00 - 16:00
      Overnight: 16:00 - 18:00

    For each sub-session, maintains a running high and low that resets at
    session boundary. At any bar, the current sub-session's H/L so far and
    all prior completed sub-sessions' H/L are available as liquidity targets.

    Returns
    -------
    (list_of_high_arrays, list_of_low_arrays)
        Each list has one array per liquidity source. Arrays are same length as df.
        These represent the most recent known H/L for each sub-session.
    """
    n = len(df)
    et_index = df.index.tz_convert("US/Eastern")
    high_arr = df["high"].values
    low_arr = df["low"].values

    frac = et_index.hour + et_index.minute / 60.0

    # Define sub-session boundaries (ET fractional hours)
    sessions = [
        ("asia",     18.0, 3.0,  True),   # wraps midnight
        ("london",   3.0,  9.5,  False),
        ("ny_am",    9.5,  11.0, False),
        ("ny_lunch", 11.0, 13.0, False),
        ("ny_pm",    13.0, 16.0, False),
    ]

    all_highs = []
    all_lows = []

    for name, start, end, wraps in sessions:
        sess_high = np.full(n, np.nan)
        sess_low = np.full(n, np.nan)

        # Track completed (prior) session H/L — only exposed AFTER session ends
        completed_h = np.nan
        completed_l = np.nan
        running_h = np.nan
        running_l = np.nan
        in_session = False

        for i in range(n):
            f = frac[i]

            if wraps:
                now_in = f >= start or f < end
            else:
                now_in = start <= f < end

            if now_in:
                if not in_session:
                    # Session just started — reset running
                    running_h = high_arr[i]
                    running_l = low_arr[i]
                    in_session = True
                else:
                    running_h = max(running_h, high_arr[i])
                    running_l = min(running_l, low_arr[i])
            else:
                if in_session:
                    # Session just ended — freeze as completed
                    completed_h = running_h
                    completed_l = running_l
                    in_session = False

            # Expose ONLY completed (prior) session H/L — not current running
            # This is the correct liquidity: a finished session's H/L is a known level
            sess_high[i] = completed_h
            sess_low[i] = completed_l

        all_highs.append(sess_high)
        all_lows.append(sess_low)

    return all_highs, all_lows


# ---------------------------------------------------------------------------
# 1. detect_fvg_test_rejection  (Trend Model)
# ---------------------------------------------------------------------------

def detect_fvg_test_rejection(
    df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Detect Trend Model entries: FVG test + rejection with quality filters.

    Quality gates for a valid signal:
      1. Active FVG exists (untested or tested_rejected), not an IFVG
      2. FVG size >= min_fvg_atr_mult * ATR(14) at the signal bar
      3. Price enters the FVG zone
      4. Price closes BACK OUT on the correct side (rejection)
      5. Rejection candle body/range >= rejection_body_ratio
      6. Fluency score >= fluency.threshold (clean PA leading in)
      7. Per-FVG cooldown: no repeat signal within signal_cooldown_bars

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV with columns [open, high, low, close, volume, is_roll_date].
    params : dict
        Full params.yaml dict.

    Returns
    -------
    pd.DataFrame  (same index as df)
        signal_trend : bool
        signal_dir   : int  (+1 long, -1 short, 0 none)
        fvg_top      : float
        fvg_bottom   : float
        fvg_size     : float
    """
    n = len(df)
    entry_cfg = _get_entry_params(params)

    fvg_df = detect_fvg(df)
    displacement = detect_displacement(df, params)
    fluency = compute_fluency(df, params)
    atr = compute_atr(df, period=14)

    # Compute swing levels for signal detection (5m swing — for stop/target)
    swing_params = {"left_bars": params["swing"]["left_bars"],
                    "right_bars": params["swing"]["right_bars"]}
    swing_df = compute_swing_levels(df, swing_params)
    sw_high = swing_df["swing_high_price"].shift(1).ffill().values
    sw_low = swing_df["swing_low_price"].shift(1).ffill().values

    # Liquidity sweep check uses SIGNIFICANT levels only:
    # 1. Completed sub-session H/L (Asia, London, NY AM, NY Lunch, NY PM)
    # 2. HTF swing H/L (left=10, right=3 — much sparser, ~1 per day)
    # NOT the dense 5m swings (left=3, right=1 = too many false positives)
    liq_highs, liq_lows = _compute_liquidity_levels(df, params)

    # HTF-like swing on 5m data: use wider params for "significant" swings
    htf_swing_params = {"left_bars": 10, "right_bars": 3}
    htf_swing_df = compute_swing_levels(df, htf_swing_params)
    htf_sw_high = htf_swing_df["swing_high_price"].shift(3).ffill().values  # shift by right_bars
    htf_sw_low = htf_swing_df["swing_low_price"].shift(3).ffill().values

    open_arr = df["open"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    close_arr = df["close"].values
    atr_arr = atr.values
    fluency_arr = fluency.values
    disp_arr = displacement.values

    body = np.abs(close_arr - open_arr)
    bar_range = high_arr - low_arr
    safe_range = np.where(bar_range == 0, np.nan, bar_range)
    body_ratio = body / safe_range

    bull_mask = fvg_df["fvg_bull"].values
    bear_mask = fvg_df["fvg_bear"].values

    # Thresholds
    min_fvg_atr = entry_cfg["min_fvg_atr_mult"]
    min_body_ratio = entry_cfg["rejection_body_ratio"]
    cooldown = entry_cfg["signal_cooldown_bars"]
    require_disp = entry_cfg["require_displacement"]
    fluency_thresh = params["fluency"]["threshold"]

    # Output arrays
    sig_trend = np.zeros(n, dtype=bool)
    sig_dir = np.zeros(n, dtype=np.int8)
    sig_fvg_top = np.full(n, np.nan)
    sig_fvg_bot = np.full(n, np.nan)
    sig_fvg_size = np.full(n, np.nan)
    sig_fvg_c2_open = np.full(n, np.nan)  # displacement candle open = model stop
    sig_swept = np.zeros(n, dtype=bool)    # did the FVG's creation sweep liquidity?
    sig_sweep_score = np.zeros(n, dtype=np.int8)  # how many levels swept (quality)

    # Active FVG pool — maintained bar-by-bar
    active: list[_LiveFVG] = []

    for i in range(n):
        # --- Birth: register new FVGs that become visible at bar i ---
        # FVG visible at bar i due to shift(1); displacement candle (candle 2) = bar i-1
        c2_idx = i - 1 if i > 0 else 0
        c2_open_val = open_arr[c2_idx]

        # Check if displacement candle swept a SIGNIFICANT liquidity level
        # Lanto: "Two key components: displaced + took liquidity in creation"
        # Count how many levels swept + prioritize session levels over swing levels
        swept = False
        sweep_score = 0  # higher = more significant sweep
        if c2_idx > 0:
            c2_high = high_arr[c2_idx]
            c2_low = low_arr[c2_idx]

            # Session levels (high significance — key intraday levels)
            for lh in liq_highs:
                if not np.isnan(lh[c2_idx]) and c2_high > lh[c2_idx]:
                    sweep_score += 2
            for ll in liq_lows:
                if not np.isnan(ll[c2_idx]) and c2_low < ll[c2_idx]:
                    sweep_score += 2

            # HTF swing levels (significant structural levels, not micro swings)
            sh = htf_sw_high[c2_idx]
            sl = htf_sw_low[c2_idx]
            if not np.isnan(sh) and c2_high > sh:
                sweep_score += 2  # HTF swing = significant
            if not np.isnan(sl) and c2_low < sl:
                sweep_score += 2

            # Swept = at least score 2 (1 significant level)
            swept = sweep_score >= 2

        if bull_mask[i]:
            fvg_size_val = fvg_df["fvg_size"].iat[i]
            rec = _LiveFVG(
                idx=i,
                direction="bull",
                top=fvg_df["fvg_bull_top"].iat[i],
                bottom=fvg_df["fvg_bull_bottom"].iat[i],
                size=fvg_size_val if not np.isnan(fvg_size_val) else 0.0,
                candle2_open=c2_open_val,
                swept_liquidity=swept,
                sweep_score=sweep_score,
            )
            active.append(rec)

        if bear_mask[i]:
            fvg_size_val = fvg_df["fvg_size"].iat[i]
            rec = _LiveFVG(
                idx=i,
                direction="bear",
                top=fvg_df["fvg_bear_top"].iat[i],
                bottom=fvg_df["fvg_bear_bottom"].iat[i],
                size=fvg_size_val if not np.isnan(fvg_size_val) else 0.0,
                candle2_open=c2_open_val,
                swept_liquidity=swept,
                sweep_score=sweep_score,
            )
            active.append(rec)

        # --- Pre-check: common quality gates ---
        # Body ratio of current bar
        cur_br = body_ratio[i] if not np.isnan(body_ratio[i]) else 0.0
        if cur_br < min_body_ratio:
            # Rejection candle too weak — skip signal check entirely
            pass
        else:
            # Fluency gate: is recent PA clean?
            cur_fluency = fluency_arr[i] if not np.isnan(fluency_arr[i]) else 0.0
            fluency_ok = cur_fluency >= fluency_thresh

            # Displacement gate
            disp_ok = disp_arr[i] if not require_disp else disp_arr[i]

            if fluency_ok and (disp_ok or not require_disp):
                # ATR at this bar for FVG size filter
                cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 0.0

                # --- Check for Trend signals BEFORE updating FVG status ---
                best_signal_dir = 0
                best_fvg: _LiveFVG | None = None
                best_score = -1.0

                for rec in active:
                    if rec.status == "invalidated":
                        continue
                    # Skip IFVGs — trend model uses original FVGs only
                    if rec.is_ifvg:
                        continue
                    # FIX 2: No signal on FVG creation bar — must age at least 1 bar
                    if rec.idx >= i:
                        continue
                    # FVG size filter
                    if cur_atr > 0 and rec.size < min_fvg_atr * cur_atr:
                        continue
                    # Cooldown check
                    if (i - rec.last_signal_idx) < cooldown:
                        continue

                    if rec.direction == "bull":
                        # Price enters zone: low touches into or through the FVG
                        entered = low_arr[i] <= rec.top and high_arr[i] >= rec.bottom
                        # Rejection: close above FVG top
                        rejected = close_arr[i] > rec.top
                        if entered and rejected:
                            score = rec.size + (100.0 if disp_arr[i] else 0.0) + (200.0 if rec.swept_liquidity else 0.0)
                            if score > best_score:
                                best_score = score
                                best_signal_dir = 1
                                best_fvg = rec

                    elif rec.direction == "bear":
                        entered = high_arr[i] >= rec.bottom and low_arr[i] <= rec.top
                        rejected = close_arr[i] < rec.bottom
                        if entered and rejected:
                            score = rec.size + (100.0 if disp_arr[i] else 0.0) + (200.0 if rec.swept_liquidity else 0.0)
                            if score > best_score:
                                best_score = score
                                best_signal_dir = -1
                                best_fvg = rec

                if best_fvg is not None:
                    sig_trend[i] = True
                    sig_dir[i] = best_signal_dir
                    sig_fvg_top[i] = best_fvg.top
                    sig_fvg_bot[i] = best_fvg.bottom
                    sig_fvg_size[i] = best_fvg.size
                    sig_fvg_c2_open[i] = best_fvg.candle2_open
                    sig_swept[i] = best_fvg.swept_liquidity
                    sig_sweep_score[i] = best_fvg.sweep_score if hasattr(best_fvg, 'sweep_score') else (1 if best_fvg.swept_liquidity else 0)
                    best_fvg.last_signal_idx = i

        # --- Update FVG states and handle invalidations ---
        surviving: list[_LiveFVG] = []
        new_ifvgs: list[_LiveFVG] = []
        for rec in active:
            old_status = rec.status
            new_status = _update_fvg_status_live(rec, high_arr[i], low_arr[i], close_arr[i])

            if new_status == "invalidated" and old_status != "invalidated":
                rec.status = "invalidated"
                rec.invalidated_at_idx = i
                if not rec.is_ifvg:
                    ifvg_dir = "bear" if rec.direction == "bull" else "bull"
                    ifvg = _LiveFVG(
                        idx=i,
                        direction=ifvg_dir,
                        top=rec.top,
                        bottom=rec.bottom,
                        size=rec.size,
                        # FIX 5: IFVG model_stop = close at invalidation bar
                        candle2_open=close_arr[i],
                        status="untested",
                        is_ifvg=True,
                        ifvg_direction=ifvg_dir,
                    )
                    new_ifvgs.append(ifvg)
            else:
                rec.status = new_status
                surviving.append(rec)

        active = surviving + new_ifvgs

    result = pd.DataFrame(
        {
            "signal_trend": sig_trend,
            "signal_dir": sig_dir,
            "fvg_top": sig_fvg_top,
            "fvg_bottom": sig_fvg_bot,
            "fvg_size": sig_fvg_size,
            "fvg_c2_open": sig_fvg_c2_open,
            "swept_liquidity": sig_swept,
            "sweep_score": sig_sweep_score,
        },
        index=df.index,
    )

    n_long = int((sig_dir == 1).sum())
    n_short = int((sig_dir == -1).sum())
    n_swept = int(sig_swept.sum())
    avg_sweep = sig_sweep_score[sig_trend].mean() if sig_trend.any() else 0
    logger.info(
        "detect_fvg_test_rejection: %d trend signals (%d long, %d short, %d swept, avg_sweep=%.1f) on %d bars",
        int(sig_trend.sum()), n_long, n_short, n_swept, avg_sweep, n,
    )
    return result


# ---------------------------------------------------------------------------
# 2. detect_mss_ifvg_retest  (MSS / Reversal Model)
# ---------------------------------------------------------------------------

def detect_mss_ifvg_retest(
    df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Detect MSS Model entries: liquidity sweep -> IFVG -> retest + respect.

    Quality gates:
      1. A swing level was swept in the last sweep_lookback bars before IFVG formed
      2. IFVG size >= min_fvg_atr_mult * ATR
      3. Retest candle enters IFVG zone and closes on the correct side
      4. Retest candle body/range >= rejection_body_ratio
      5. Fluency >= threshold
      6. Per-IFVG cooldown

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    params : dict
        Full params.yaml dict.

    Returns
    -------
    pd.DataFrame  (same index as df)
        signal_mss  : bool
        signal_dir  : int  (+1, -1, 0)
        ifvg_top    : float
        ifvg_bottom : float
    """
    n = len(df)
    entry_cfg = _get_entry_params(params)

    swing_params = {"left_bars": params["swing"]["left_bars"],
                    "right_bars": params["swing"]["right_bars"]}
    swing_df = compute_swing_levels(df, swing_params)
    fvg_df = detect_fvg(df)
    displacement = detect_displacement(df, params)
    fluency = compute_fluency(df, params)
    atr = compute_atr(df, period=14)

    open_arr = df["open"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    close_arr = df["close"].values
    atr_arr = atr.values
    fluency_arr = fluency.values
    disp_arr = displacement.values

    body = np.abs(close_arr - open_arr)
    bar_range = high_arr - low_arr
    safe_range = np.where(bar_range == 0, np.nan, bar_range)
    body_ratio = body / safe_range

    # FIX 4: Shift swing levels by 1 bar to account for right_bars lookahead
    swing_high_price = swing_df["swing_high_price"].shift(1).ffill().values
    swing_low_price = swing_df["swing_low_price"].shift(1).ffill().values

    bull_mask = fvg_df["fvg_bull"].values
    bear_mask = fvg_df["fvg_bear"].values

    # Thresholds
    min_fvg_atr = entry_cfg["min_fvg_atr_mult"]
    min_body_ratio = entry_cfg["rejection_body_ratio"]
    cooldown = entry_cfg["signal_cooldown_bars"]
    sweep_lookback = entry_cfg["sweep_lookback"]
    fluency_thresh = params["fluency"]["threshold"]

    # Output arrays
    sig_mss = np.zeros(n, dtype=bool)
    sig_dir = np.zeros(n, dtype=np.int8)
    sig_ifvg_top = np.full(n, np.nan)
    sig_ifvg_bot = np.full(n, np.nan)
    sig_ifvg_c2_open = np.full(n, np.nan)

    # Track sweeps: for each bar, check if a swing level was recently swept
    swept_low = np.zeros(n, dtype=bool)
    swept_high = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if not np.isnan(swing_low_price[i - 1]):
            if low_arr[i] < swing_low_price[i - 1]:
                swept_low[i] = True
        if not np.isnan(swing_high_price[i - 1]):
            if high_arr[i] > swing_high_price[i - 1]:
                swept_high[i] = True

    # Active FVG pool for IFVG tracking
    active: list[_LiveFVG] = []
    active_ifvgs: list[_LiveFVG] = []

    for i in range(n):
        # --- Birth: register new FVGs ---
        c2_open_val = open_arr[i - 1] if i > 0 else open_arr[i]

        if bull_mask[i]:
            fvg_size_val = fvg_df["fvg_size"].iat[i]
            rec = _LiveFVG(
                idx=i,
                direction="bull",
                top=fvg_df["fvg_bull_top"].iat[i],
                bottom=fvg_df["fvg_bull_bottom"].iat[i],
                size=fvg_size_val if not np.isnan(fvg_size_val) else 0.0,
                candle2_open=c2_open_val,
            )
            active.append(rec)

        if bear_mask[i]:
            fvg_size_val = fvg_df["fvg_size"].iat[i]
            rec = _LiveFVG(
                idx=i,
                direction="bear",
                top=fvg_df["fvg_bear_top"].iat[i],
                bottom=fvg_df["fvg_bear_bottom"].iat[i],
                size=fvg_size_val if not np.isnan(fvg_size_val) else 0.0,
                candle2_open=c2_open_val,
            )
            active.append(rec)

        # --- Pre-check: common quality gates ---
        cur_br = body_ratio[i] if not np.isnan(body_ratio[i]) else 0.0
        cur_fluency = fluency_arr[i] if not np.isnan(fluency_arr[i]) else 0.0
        cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 0.0

        if cur_br >= min_body_ratio and cur_fluency >= fluency_thresh:
            # --- Check for MSS signals from active IFVGs ---
            best_signal_dir = 0
            best_ifvg: _LiveFVG | None = None
            best_score = -1.0

            for ifvg in active_ifvgs:
                if ifvg.status == "invalidated":
                    continue
                # FIX 2: No signal on IFVG creation bar — must age at least 1 bar
                if ifvg.idx >= i:
                    continue
                # Size filter
                if cur_atr > 0 and ifvg.size < min_fvg_atr * cur_atr:
                    continue
                # Cooldown
                if (i - ifvg.last_signal_idx) < cooldown:
                    continue

                if ifvg.ifvg_direction == "bull":
                    # Check: recent swing low sweep before IFVG formed
                    sweep_start = max(0, ifvg.idx - sweep_lookback)
                    had_sweep = np.any(swept_low[sweep_start:ifvg.idx + 1])
                    if not had_sweep:
                        continue

                    entered = low_arr[i] <= ifvg.top and high_arr[i] >= ifvg.bottom
                    respected = close_arr[i] > ifvg.top
                    if entered and respected:
                        score = ifvg.size + (100.0 if disp_arr[i] else 0.0)
                        if score > best_score:
                            best_score = score
                            best_signal_dir = 1
                            best_ifvg = ifvg

                elif ifvg.ifvg_direction == "bear":
                    sweep_start = max(0, ifvg.idx - sweep_lookback)
                    had_sweep = np.any(swept_high[sweep_start:ifvg.idx + 1])
                    if not had_sweep:
                        continue

                    entered = high_arr[i] >= ifvg.bottom and low_arr[i] <= ifvg.top
                    respected = close_arr[i] < ifvg.bottom
                    if entered and respected:
                        score = ifvg.size + (100.0 if disp_arr[i] else 0.0)
                        if score > best_score:
                            best_score = score
                            best_signal_dir = -1
                            best_ifvg = ifvg

            if best_ifvg is not None:
                sig_mss[i] = True
                sig_dir[i] = best_signal_dir
                sig_ifvg_top[i] = best_ifvg.top
                sig_ifvg_bot[i] = best_ifvg.bottom
                sig_ifvg_c2_open[i] = best_ifvg.candle2_open
                best_ifvg.last_signal_idx = i

        # --- Update FVG states + spawn IFVGs ---
        surviving: list[_LiveFVG] = []
        new_ifvgs: list[_LiveFVG] = []
        for rec in active:
            old_status = rec.status
            new_status = _update_fvg_status_live(rec, high_arr[i], low_arr[i], close_arr[i])

            if new_status == "invalidated" and old_status != "invalidated":
                rec.status = "invalidated"
                rec.invalidated_at_idx = i
                if not rec.is_ifvg:
                    ifvg_dir = "bear" if rec.direction == "bull" else "bull"
                    # FIX 5 (v2): IFVG stop = FVG boundary on the correct side.
                    # Bull IFVG (long signal) → stop = bottom of zone (below entry)
                    # Bear IFVG (short signal) → stop = top of zone (above entry)
                    # This guarantees stop is always on the correct side of entry,
                    # fixing the 46% inverted-stop bug.
                    ifvg_stop = rec.bottom if ifvg_dir == "bull" else rec.top
                    ifvg = _LiveFVG(
                        idx=i,
                        direction=ifvg_dir,
                        top=rec.top,
                        bottom=rec.bottom,
                        size=rec.size,
                        candle2_open=ifvg_stop,
                        status="untested",
                        is_ifvg=True,
                        ifvg_direction=ifvg_dir,
                    )
                    new_ifvgs.append(ifvg)
                    active_ifvgs.append(ifvg)
            else:
                rec.status = new_status
                surviving.append(rec)

        active = surviving + new_ifvgs

        # Update IFVG states
        surviving_ifvgs: list[_LiveFVG] = []
        for ifvg in active_ifvgs:
            if ifvg.idx == i:
                surviving_ifvgs.append(ifvg)
                continue  # just born this bar, don't test yet
            if ifvg.status == "invalidated":
                continue
            new_status = _update_fvg_status_live(ifvg, high_arr[i], low_arr[i], close_arr[i])
            ifvg.status = new_status
            if new_status != "invalidated":
                surviving_ifvgs.append(ifvg)
        active_ifvgs = surviving_ifvgs

    result = pd.DataFrame(
        {
            "signal_mss": sig_mss,
            "signal_dir": sig_dir,
            "ifvg_top": sig_ifvg_top,
            "ifvg_bottom": sig_ifvg_bot,
            "fvg_c2_open": sig_ifvg_c2_open,
        },
        index=df.index,
    )

    n_long = int((sig_dir == 1).sum())
    n_short = int((sig_dir == -1).sum())
    logger.info(
        "detect_mss_ifvg_retest: %d MSS signals (%d long, %d short) on %d bars",
        int(sig_mss.sum()), n_long, n_short, n,
    )
    return result


# ---------------------------------------------------------------------------
# 3. detect_all_signals  (orchestrator)
# ---------------------------------------------------------------------------

def detect_all_signals(
    df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """High-level orchestrator: detect both Trend and MSS entry signals.

    Computes all needed intermediate features, calls both signal detectors,
    and combines results with stop-loss and target estimates.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV with [open, high, low, close, volume, is_roll_date].
    params : dict
        Full params.yaml dict.

    Returns
    -------
    pd.DataFrame  (same index as df)
        signal       : bool    (any signal)
        signal_type  : str     ('trend', 'mss', '')
        signal_dir   : int     (+1, -1, 0)
        entry_price  : float   (close of signal bar)
        model_stop   : float   (candle low/high primary, swing fallback)
        irl_target   : float   (nearest swing H/L in trade direction)
    """
    logger.info("detect_all_signals: starting on %d bars", len(df))
    n = len(df)

    # --- Detect both signal types ---
    trend_df = detect_fvg_test_rejection(df, params)
    mss_df = detect_mss_ifvg_retest(df, params)

    # --- Compute swing levels for stop/target ---
    swing_params = {"left_bars": params["swing"]["left_bars"],
                    "right_bars": params["swing"]["right_bars"]}
    swing_df = compute_swing_levels(df, swing_params)
    atr = compute_atr(df, period=14)

    open_arr = df["open"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    close_arr = df["close"].values
    atr_arr = atr.values
    # FIX 4: Swing points have right_bars=1 lookahead. Apply shift(1) to
    # prevent using a swing level before it is confirmed.
    swing_high_price = swing_df["swing_high_price"].shift(1).ffill().values
    swing_low_price = swing_df["swing_low_price"].shift(1).ffill().values

    small_candle_mult = params["stop_loss"]["small_candle_atr_mult"]

    # --- Combine signals ---
    # Trend has priority when both fire on the same bar
    signal = np.zeros(n, dtype=bool)
    signal_type = np.full(n, "", dtype=object)
    signal_dir = np.zeros(n, dtype=np.int8)
    entry_price = np.full(n, np.nan)
    model_stop = np.full(n, np.nan)
    irl_target = np.full(n, np.nan)

    trend_sig = trend_df["signal_trend"].values
    trend_dir = trend_df["signal_dir"].values
    trend_swept = trend_df["swept_liquidity"].values if "swept_liquidity" in trend_df.columns else np.zeros(n, dtype=bool)
    trend_sweep_sc = trend_df["sweep_score"].values if "sweep_score" in trend_df.columns else np.zeros(n, dtype=np.int8)
    mss_sig = mss_df["signal_mss"].values
    mss_dir = mss_df["signal_dir"].values

    swept_arr = np.zeros(n, dtype=bool)
    sweep_score_arr = np.zeros(n, dtype=np.int8)

    for i in range(n):
        if trend_sig[i]:
            signal[i] = True
            signal_type[i] = "trend"
            signal_dir[i] = trend_dir[i]
            swept_arr[i] = trend_swept[i]
            sweep_score_arr[i] = trend_sweep_sc[i]
            # FIX 1: Entry at open of NEXT bar (first achievable price after signal bar closes)
            entry_price[i] = open_arr[i + 1] if i + 1 < n else close_arr[i]
        elif mss_sig[i]:
            signal[i] = True
            signal_type[i] = "mss"
            signal_dir[i] = mss_dir[i]
            swept_arr[i] = True  # MSS requires liquidity sweep by definition
            sweep_score_arr[i] = 4  # MSS inherently swept (by definition)
            # FIX 1: Entry at open of NEXT bar (first achievable price after signal bar closes)
            entry_price[i] = open_arr[i + 1] if i + 1 < n else close_arr[i]

        if signal[i]:
            direction = int(signal_dir[i])
            bar_range = high_arr[i] - low_arr[i]
            cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 0.0

            # --- Model stop ---
            # Primary: displacement candle (candle 2) open price
            # This is the FVG-creating candle's open — if price returns here,
            # the displacement is fully negated = thesis invalidated
            # Fallback: rejection candle extreme if c2_open not available
            c2_open = np.nan
            if trend_sig[i] and "fvg_c2_open" in trend_df.columns:
                c2_open = trend_df["fvg_c2_open"].iat[i]
            elif mss_sig[i] and "fvg_c2_open" in mss_df.columns:
                c2_open = mss_df["fvg_c2_open"].iat[i] if "fvg_c2_open" in mss_df.columns else np.nan

            if not np.isnan(c2_open) and c2_open > 0:
                primary_stop = c2_open
            else:
                # Fallback: rejection candle extreme
                if direction == 1:
                    primary_stop = min(open_arr[i], low_arr[i])
                else:
                    primary_stop = max(open_arr[i], high_arr[i])

            # Fallback: if candle range < small_candle_atr_mult * ATR,
            # use swing level instead
            if cur_atr > 0 and bar_range < small_candle_mult * cur_atr:
                if direction == 1:
                    fallback = swing_low_price[i] if not np.isnan(swing_low_price[i]) else primary_stop
                    stop = min(primary_stop, fallback)
                else:
                    fallback = swing_high_price[i] if not np.isnan(swing_high_price[i]) else primary_stop
                    stop = max(primary_stop, fallback)
            else:
                stop = primary_stop

            model_stop[i] = stop

            # --- IRL target (nearest swing in trade direction) ---
            risk = abs(close_arr[i] - stop)
            if direction == 1:
                target = swing_high_price[i] if not np.isnan(swing_high_price[i]) else np.nan
                # If most recent swing high is at or below entry, estimate 2R target
                if not np.isnan(target) and target <= close_arr[i]:
                    target = close_arr[i] + risk * 2.0 if risk > 0 else np.nan
            else:
                target = swing_low_price[i] if not np.isnan(swing_low_price[i]) else np.nan
                if not np.isnan(target) and target >= close_arr[i]:
                    target = close_arr[i] - risk * 2.0 if risk > 0 else np.nan

            irl_target[i] = target

    result = pd.DataFrame(
        {
            "signal": signal,
            "signal_type": signal_type,
            "signal_dir": signal_dir,
            "entry_price": entry_price,
            "model_stop": model_stop,
            "irl_target": irl_target,
            "swept_liquidity": swept_arr,
            "sweep_score": sweep_score_arr,
        },
        index=df.index,
    )

    n_trend = int(trend_sig.sum())
    n_mss = int(mss_sig.sum())
    n_combined = int(signal.sum())
    n_swept = int(swept_arr[signal].sum()) if signal.any() else 0
    avg_ss = float(sweep_score_arr[signal].mean()) if signal.any() else 0
    logger.info(
        "detect_all_signals: %d combined signals (%d trend, %d mss, %d swept, avg_sweep=%.1f, %d overlap suppressed)",
        n_combined, n_trend, n_mss, n_swept, avg_ss, n_trend + n_mss - n_combined,
    )
    return result


# ---------------------------------------------------------------------------
# Internal: FVG status update (same logic as fvg.py but works with _LiveFVG)
# ---------------------------------------------------------------------------

def _update_fvg_status_live(
    rec: _LiveFVG, bar_high: float, bar_low: float, bar_close: float,
) -> str:
    """Determine new status for a _LiveFVG given one bar's OHLC.

    Mirrors the logic of fvg._update_fvg_status but operates on _LiveFVG.
    """
    if rec.status == "invalidated":
        return "invalidated"

    top = rec.top
    bottom = rec.bottom

    direction = rec.ifvg_direction if rec.is_ifvg else rec.direction

    if direction == "bull":
        entered = bar_low <= top and bar_high >= bottom
        if entered:
            if bar_close < bottom:
                return "invalidated"
            if bar_close >= top:
                return "tested_rejected" if rec.status == "untested" else rec.status
            return "tested_rejected" if rec.status == "untested" else rec.status
        if bar_close < bottom:
            return "invalidated"
        return rec.status
    else:
        entered = bar_high >= bottom and bar_low <= top
        if entered:
            if bar_close > top:
                return "invalidated"
            if bar_close <= bottom:
                return "tested_rejected" if rec.status == "untested" else rec.status
            return "tested_rejected" if rec.status == "untested" else rec.status
        if bar_close > top:
            return "invalidated"
        return rec.status


# ---------------------------------------------------------------------------
# __main__  — test on 5m data for 2025-01 to 2025-03
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time as _time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    # 1. Load data and params
    logger.info("Loading 5m data...")
    df = pd.read_parquet(project_root / "data" / "NQ_5m.parquet")
    params = _load_params(project_root / "config" / "params.yaml")

    # Slice to 2025-01 through 2025-03
    mask = (df.index >= "2025-01-01") & (df.index < "2025-04-01")
    df_window = df[mask].copy()
    logger.info(
        "Window: %d bars, %s to %s",
        len(df_window), df_window.index.min(), df_window.index.max(),
    )

    # 2. Detect all signals
    t0 = _time.perf_counter()
    signals = detect_all_signals(df_window, params)
    elapsed = _time.perf_counter() - t0

    # 3. Summary
    total = int(signals["signal"].sum())
    n_trend = int((signals["signal_type"] == "trend").sum())
    n_mss = int((signals["signal_type"] == "mss").sum())

    # Count trading days (unique UTC dates)
    signal_bars = signals[signals["signal"]]
    if len(signal_bars) > 0:
        trading_days = df_window.index.normalize().nunique()
    else:
        trading_days = 1

    print()
    print("=" * 70)
    print("ENTRY SIGNAL DETECTION — 5m NQ — 2025-01 to 2025-03")
    print("=" * 70)
    print(f"Bars in window:       {len(df_window):>8,}")
    print(f"Trading days:         {trading_days:>8,}")
    print(f"Computation time:     {elapsed:>8.1f}s")
    print()
    print(f"Total signals:        {total:>8,}")
    print(f"  Trend (FVG reject): {n_trend:>8,}")
    print(f"  MSS (IFVG retest):  {n_mss:>8,}")
    print(f"Signals per day:      {total / trading_days:>8.2f}")
    print()

    # Direction breakdown
    if total > 0:
        n_long = int((signals["signal_dir"] == 1).sum())
        n_short = int((signals["signal_dir"] == -1).sum())
        print(f"Long signals:         {n_long:>8,}")
        print(f"Short signals:        {n_short:>8,}")
        print()

    # 4. Sample signals
    if len(signal_bars) > 0:
        print("-" * 70)
        print("Sample signals (first 20):")
        print("-" * 70)
        sample = signal_bars.head(20)
        for ts, row in sample.iterrows():
            rr = np.nan
            if not np.isnan(row["model_stop"]) and not np.isnan(row["irl_target"]):
                risk = abs(row["entry_price"] - row["model_stop"])
                reward = abs(row["irl_target"] - row["entry_price"])
                rr = reward / risk if risk > 0 else np.nan
            dir_str = "LONG" if row["signal_dir"] == 1 else "SHORT"
            print(
                f"  {ts}  {row['signal_type']:>5s}  "
                f"dir={dir_str:>5s}  "
                f"entry={row['entry_price']:>10.2f}  "
                f"stop={row['model_stop']:>10.2f}  "
                f"target={row['irl_target']:>10.2f}  "
                f"RR={rr:>5.1f}"
            )
        print()

        # 5. Daily signal distribution
        print("-" * 70)
        print("Daily signal distribution (first 30 days with signals):")
        print("-" * 70)
        signal_dates = signal_bars.index.normalize()
        daily_counts = signal_dates.value_counts().sort_index()
        for dt, cnt in daily_counts.head(30).items():
            print(f"  {dt.strftime('%Y-%m-%d')}  signals={cnt}")

        print()
        print(f"Days with signals:    {len(daily_counts):>8,}")
        print(f"Mean signals/day:     {daily_counts.mean():>8.2f}")
        print(f"Max signals/day:      {daily_counts.max():>8,}")
        print(f"Min signals/day:      {daily_counts.min():>8,}")
        print(f"Median signals/day:   {daily_counts.median():>8.1f}")

        # 6. Signal type breakdown per direction
        print()
        print("-" * 70)
        print("Signal type x direction breakdown:")
        print("-" * 70)
        for stype in ["trend", "mss"]:
            subset = signal_bars[signal_bars["signal_type"] == stype]
            if len(subset) > 0:
                n_l = int((subset["signal_dir"] == 1).sum())
                n_s = int((subset["signal_dir"] == -1).sum())
                avg_rr_vals = []
                for _, row in subset.iterrows():
                    if not np.isnan(row["model_stop"]) and not np.isnan(row["irl_target"]):
                        risk = abs(row["entry_price"] - row["model_stop"])
                        reward = abs(row["irl_target"] - row["entry_price"])
                        if risk > 0:
                            avg_rr_vals.append(reward / risk)
                avg_rr = np.mean(avg_rr_vals) if avg_rr_vals else np.nan
                print(
                    f"  {stype:>5s}: {len(subset):>5d} total  "
                    f"({n_l} long, {n_s} short)  "
                    f"avg RR={avg_rr:.1f}"
                )
    else:
        print("  NO SIGNALS DETECTED — parameters may need tuning.")

    print("=" * 70)
