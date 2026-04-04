"""
features/bias.py — 4-step bias determination engine (Lanto's process).

Step 1: HTF Analysis (4H / 1H) — mark active untested FVGs as draw targets
Step 2: Overnight Price (6PM-9:30AM ET) — position relative to overnight range
Step 3: ORM (first 30 min of NY) — confirms or denies overnight bias
Step 4: Composite — weighted combination of all steps

Also includes regime detection: risk multiplier based on HTF clarity and PA quality.

All bias values only become available AFTER the relevant data is complete:
    - HTF bias uses shift(1) — only after the 4H/1H candle closes
    - ORM bias only after ORM window ends
    - Forward-fill carries bias values into subsequent bars

References: CLAUDE.md sections 0, 3.1–3.6, regime params in config/params.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "params.yaml"


def _load_params(config_path: Path = _CONFIG_PATH) -> dict[str, Any]:
    """Load tunable parameters from params.yaml."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 1. compute_htf_bias — Step 1: HTF FVG draw direction
# ---------------------------------------------------------------------------

def compute_htf_bias(
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    params: dict[str, Any],
) -> pd.Series:
    """Determine HTF bias from active untested FVGs on 4H and 1H.

    Logic:
        - Untested bullish FVGs ABOVE current price -> bullish draw
          (price wants to go up to fill them).
        - Untested bearish FVGs BELOW current price -> bearish draw
          (price wants to go down to fill them).
        - Weight: 4H FVGs count more than 1H FVGs.
        - If 4H fluency is below threshold, reduce confidence (dampen toward 0).

    All computations use shift(1) to avoid lookahead: a 4H candle's features
    are only available after the next 4H candle opens.

    Parameters
    ----------
    df_4h : pd.DataFrame
        4H OHLCV with UTC DatetimeIndex, must include is_roll_date column.
    df_1h : pd.DataFrame
        1H OHLCV with UTC DatetimeIndex, must include is_roll_date column.
    params : dict
        Parsed params.yaml.

    Returns
    -------
    pd.Series
        Float values on the 4H index: +1 (bullish), -1 (bearish), 0 (neutral).
        Shifted by 1 bar — the value at 4H bar T reflects what was known when
        bar T opened (i.e. features from bar T-1 and earlier).
    """
    from features.fvg import detect_fvg, FVGRecord, _update_fvg_status
    from features.displacement import compute_fluency

    fluency_threshold = params["fluency"]["threshold"]
    w_4h = 0.6  # Weight for 4H FVGs
    w_1h = 0.4  # Weight for 1H FVGs

    # --- Compute fluency on 4H (for confidence dampening) ---
    fluency_4h = compute_fluency(df_4h, params)

    # --- Walk each timeframe and compute per-bar draw direction ---
    bias_4h, count_4h = _walk_fvg_bias(df_4h, label="4H")
    bias_1h, count_1h = _walk_fvg_bias(df_1h, label="1H")

    # --- Align 1H bias to 4H index (shift(1) + ffill) ---
    # 1H is higher granularity; align to 4H by reindex + ffill.
    # First shift 1H by 1 bar (1H candle known only after close).
    bias_1h_shifted = bias_1h.shift(1)
    bias_1h_on_4h = bias_1h_shifted.reindex(df_4h.index, method="ffill")

    # Also align active counts (for regime PDA detection)
    count_4h_shifted = count_4h.shift(1)
    count_1h_shifted = count_1h.shift(1)
    count_1h_on_4h = count_1h_shifted.reindex(df_4h.index, method="ffill")

    # --- Shift 4H bias by 1 bar ---
    bias_4h_shifted = bias_4h.shift(1)

    # --- Fluency dampening ---
    # If 4H fluency is below threshold, dampen toward 0.
    # Fluency is also shifted by 1 (only known after 4H candle closes).
    fluency_shifted = fluency_4h.shift(1)
    fluency_factor = np.where(
        fluency_shifted.values >= fluency_threshold,
        1.0,
        0.5,  # Reduce confidence when PA is not fluent
    )

    # --- Combine ---
    raw_composite = (
        w_4h * bias_4h_shifted.fillna(0).values
        + w_1h * bias_1h_on_4h.fillna(0).values
    )

    # Apply fluency dampening
    composite = raw_composite * fluency_factor

    # Clip to [-1, +1]
    composite = np.clip(composite, -1.0, 1.0)

    result = pd.Series(composite, index=df_4h.index, name="htf_bias", dtype="float64")

    # Total active FVG count (sum of 4H + 1H), shifted for no-lookahead
    total_pda_count = (
        count_4h_shifted.fillna(0).values
        + count_1h_on_4h.fillna(0).values
    )
    pda_count = pd.Series(
        total_pda_count, index=df_4h.index, name="htf_pda_count", dtype="float64",
    )

    logger.info(
        "compute_htf_bias: mean=%.3f, bullish=%d, bearish=%d, neutral=%d, mean_pdas=%.1f",
        result.mean(),
        (result > 0.2).sum(),
        (result < -0.2).sum(),
        ((result >= -0.2) & (result <= 0.2)).sum(),
        pda_count.mean(),
    )
    return result, pda_count


def _walk_fvg_bias(df: pd.DataFrame, label: str = "") -> tuple[pd.Series, pd.Series]:
    """Walk bar-by-bar, tracking active FVGs and computing draw direction.

    For each bar, look at all currently active (untested or tested_rejected) FVGs:
        - Count bullish FVGs above current close -> bullish draw
        - Count bearish FVGs below current close -> bearish draw
        - Net direction = sign(bull_above - bear_below)

    This is a RAW (unshifted) computation. The caller must shift(1).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV with is_roll_date column.
    label : str
        Timeframe label for logging.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (bias, active_count): bias values in {-1, 0, +1}, and total active FVG count.
    """
    from features.fvg import detect_fvg, FVGRecord, _update_fvg_status

    fvg_df = detect_fvg(df)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    bull_mask = fvg_df["fvg_bull"].values
    bear_mask = fvg_df["fvg_bear"].values

    bias_arr = np.zeros(n, dtype=np.float64)
    count_arr = np.zeros(n, dtype=np.int32)

    active_bull: list[FVGRecord] = []
    active_bear: list[FVGRecord] = []

    for i in range(n):
        # Birth: register new FVGs
        if bull_mask[i]:
            rec = FVGRecord(
                time=df.index[i - 1] if i > 0 else df.index[i],
                direction="bull",
                top=fvg_df["fvg_bull_top"].iat[i],
                bottom=fvg_df["fvg_bull_bottom"].iat[i],
                size=fvg_df["fvg_size"].iat[i],
            )
            active_bull.append(rec)

        if bear_mask[i]:
            rec = FVGRecord(
                time=df.index[i - 1] if i > 0 else df.index[i],
                direction="bear",
                top=fvg_df["fvg_bear_top"].iat[i],
                bottom=fvg_df["fvg_bear_bottom"].iat[i],
                size=fvg_df["fvg_size"].iat[i],
            )
            active_bear.append(rec)

        # Update and prune
        surviving_bull: list[FVGRecord] = []
        for rec in active_bull:
            new_status = _update_fvg_status(rec, high[i], low[i], close[i])
            if new_status == "invalidated":
                rec.status = "invalidated"
            else:
                rec.status = new_status
                surviving_bull.append(rec)
        active_bull = surviving_bull

        surviving_bear: list[FVGRecord] = []
        for rec in active_bear:
            new_status = _update_fvg_status(rec, high[i], low[i], close[i])
            if new_status == "invalidated":
                rec.status = "invalidated"
            else:
                rec.status = new_status
                surviving_bear.append(rec)
        active_bear = surviving_bear

        # Compute draw direction at this bar
        c = close[i]

        # Total active FVGs (for regime PDA check)
        count_arr[i] = len(active_bull) + len(active_bear)

        # Bullish FVGs ABOVE current price = bullish draw (untested targets above)
        bull_above = sum(1 for r in active_bull if (r.top + r.bottom) / 2 > c)
        # Bearish FVGs BELOW current price = bearish draw (untested targets below)
        bear_below = sum(1 for r in active_bear if (r.top + r.bottom) / 2 < c)

        if bull_above > 0 and bear_below == 0:
            bias_arr[i] = 1.0
        elif bear_below > 0 and bull_above == 0:
            bias_arr[i] = -1.0
        elif bull_above > 0 and bear_below > 0:
            # Both directions have draws — compare counts
            net = bull_above - bear_below
            if net > 0:
                bias_arr[i] = 0.5
            elif net < 0:
                bias_arr[i] = -0.5
            else:
                bias_arr[i] = 0.0
        else:
            bias_arr[i] = 0.0

    result = pd.Series(bias_arr, index=df.index, dtype="float64")
    active_count = pd.Series(count_arr, index=df.index, dtype="int32")
    logger.info(
        "_walk_fvg_bias [%s]: bullish=%.1f%%, bearish=%.1f%%, neutral=%.1f%%, mean_active=%.1f",
        label,
        100.0 * (result > 0).mean(),
        100.0 * (result < 0).mean(),
        100.0 * (result == 0).mean(),
        active_count.mean(),
    )
    return result, active_count


# ---------------------------------------------------------------------------
# 2. compute_overnight_bias — Step 2: position relative to overnight range
# ---------------------------------------------------------------------------

def compute_overnight_bias(
    df: pd.DataFrame,
    session_levels: pd.DataFrame,
    params: dict[str, Any],
) -> pd.Series:
    """Determine bias from overnight price position.

    Uses the NY open price (first bar at 09:30 ET) compared to the overnight
    range. This locks the overnight bias at session open and prevents it from
    changing as intraday price moves.

        - NY open above 60% of overnight range -> bullish (+1)
        - NY open below 40% of overnight range -> bearish (-1)
        - NY open in the middle band -> neutral (0)

    Session levels (overnight_high, overnight_low, ny_open) are no-lookahead:
    they only become available after the overnight session ends (at 09:30 ET),
    computed by sessions.compute_session_levels().

    Parameters
    ----------
    df : pd.DataFrame
        1m or 5m OHLCV with UTC DatetimeIndex.
    session_levels : pd.DataFrame
        Output of sessions.compute_session_levels(), aligned to df.index.
        Must contain overnight_high, overnight_low, ny_open columns.
    params : dict
        Parsed params.yaml.

    Returns
    -------
    pd.Series
        Float values aligned to df.index: +1 (bullish), -1 (bearish), 0 (neutral).
        Locked once per day (uses ny_open, not real-time price).
    """
    overnight_high = session_levels["overnight_high"]
    overnight_low = session_levels["overnight_low"]
    overnight_range = overnight_high - overnight_low

    # Use NY open price (locked at session start) instead of live close.
    # This ensures overnight bias doesn't change as price moves intraday.
    ny_open = session_levels["ny_open"]

    # Position within overnight range: 0 = at low, 1 = at high
    safe_range = overnight_range.replace(0, np.nan)
    position = (ny_open - overnight_low) / safe_range

    # Bias based on position:
    # Above 0.6 -> bullish, below 0.4 -> bearish, in between -> neutral
    bias = np.where(
        position.values > 0.6,
        1.0,
        np.where(
            position.values < 0.4,
            -1.0,
            0.0,
        ),
    )

    # Where session levels are NaN (no completed overnight yet), bias is NaN
    mask_no_data = overnight_high.isna() | overnight_low.isna()
    bias = np.where(mask_no_data.values, np.nan, bias)

    result = pd.Series(bias, index=df.index, name="overnight_bias", dtype="float64")

    valid = result.dropna()
    if len(valid) > 0:
        logger.info(
            "compute_overnight_bias: bullish=%.1f%%, bearish=%.1f%%, neutral=%.1f%%",
            100.0 * (valid > 0).mean(),
            100.0 * (valid < 0).mean(),
            100.0 * (valid == 0).mean(),
        )
    return result


# ---------------------------------------------------------------------------
# 3. compute_orm_bias — Step 3: opening range move confirms or flips bias
# ---------------------------------------------------------------------------

def compute_orm_bias(
    df: pd.DataFrame,
    orm_data: pd.DataFrame,
    session_levels: pd.DataFrame,
    params: dict[str, Any],
) -> pd.Series:
    """Determine bias from the Opening Range Move (first 30 min of NY).

    Logic:
        - If ORM high broke above overnight high -> bullish confirmation (+1)
        - If ORM low broke below overnight low -> bearish confirmation (-1)
        - If ORM stayed within overnight range -> inconclusive (0)
        - If both broke -> net is inconclusive (0)

    ORM data (orm_high, orm_low) only becomes available after the ORM window
    closes, as computed by sessions.compute_orm().

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV with UTC DatetimeIndex.
    orm_data : pd.DataFrame
        Output of sessions.compute_orm(), aligned to df.index.
        Must contain orm_high, orm_low, is_orm_period.
    session_levels : pd.DataFrame
        Output of sessions.compute_session_levels().
        Must contain overnight_high, overnight_low.
    params : dict
        Parsed params.yaml.

    Returns
    -------
    pd.Series
        Float values aligned to df.index: +1, -1, 0, or NaN (before ORM ends).
    """
    orm_high = orm_data["orm_high"]
    orm_low = orm_data["orm_low"]
    overnight_high = session_levels["overnight_high"]
    overnight_low = session_levels["overnight_low"]

    broke_above = orm_high > overnight_high
    broke_below = orm_low < overnight_low

    bias = np.where(
        broke_above.values & ~broke_below.values,
        1.0,
        np.where(
            broke_below.values & ~broke_above.values,
            -1.0,
            0.0,  # Both or neither -> inconclusive
        ),
    )

    # Mark NaN where ORM data is not yet available
    mask_no_data = orm_high.isna() | orm_low.isna() | overnight_high.isna() | overnight_low.isna()
    bias = np.where(mask_no_data.values, np.nan, bias)

    result = pd.Series(bias, index=df.index, name="orm_bias", dtype="float64")

    valid = result.dropna()
    if len(valid) > 0:
        logger.info(
            "compute_orm_bias: bullish=%.1f%%, bearish=%.1f%%, inconclusive=%.1f%%",
            100.0 * (valid > 0).mean(),
            100.0 * (valid < 0).mean(),
            100.0 * (valid == 0).mean(),
        )
    return result


# ---------------------------------------------------------------------------
# 4. compute_daily_bias — Orchestrator: combine all 3 steps
# ---------------------------------------------------------------------------

def compute_daily_bias(
    df: pd.DataFrame,
    session_levels: pd.DataFrame,
    orm_data: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Orchestrate the 4-step bias process and produce composite bias.

    Combines:
        - HTF bias (weight 0.4) — from 4H/1H FVG draw direction
        - Overnight bias (weight 0.3) — position in overnight range
        - ORM bias (weight 0.3) — opening range move vs overnight levels

    Parameters
    ----------
    df : pd.DataFrame
        Base OHLCV (e.g. 5m or 1m) with UTC DatetimeIndex.
    session_levels : pd.DataFrame
        From sessions.compute_session_levels(), aligned to df.index.
    orm_data : pd.DataFrame
        From sessions.compute_orm(), aligned to df.index.
    df_4h : pd.DataFrame
        4H OHLCV.
    df_1h : pd.DataFrame
        1H OHLCV.
    params : dict
        Parsed params.yaml.

    Returns
    -------
    pd.DataFrame
        Aligned to df.index with columns:
        - htf_bias: from step 1
        - overnight_bias: from step 2
        - orm_bias: from step 3
        - composite_bias: weighted combination, clipped [-1, +1]
        - bias_confidence: abs(composite_bias), 0-1 scale
        - bias_direction: sign of composite: +1, -1, or 0 (if abs < 0.2)
    """
    # Weights from Lanto's methodology
    w_htf = 0.4
    w_overnight = 0.3
    w_orm = 0.3

    # --- Step 1: HTF bias (on 4H index, then align to df) ---
    htf_raw, pda_count_raw = compute_htf_bias(df_4h, df_1h, params)
    # htf_raw is on 4H index, already shift(1)-ed inside compute_htf_bias.
    # Align to df index with ffill.
    htf_on_df = htf_raw.reindex(df.index, method="ffill")
    pda_count_on_df = pda_count_raw.reindex(df.index, method="ffill")

    # --- Step 2: Overnight bias ---
    overnight = compute_overnight_bias(df, session_levels, params)

    # --- Step 3: ORM bias ---
    orm = compute_orm_bias(df, orm_data, session_levels, params)

    # --- Composite ---
    # Fill NaN with 0 for weighting (missing component contributes nothing).
    htf_filled = htf_on_df.fillna(0.0)
    overnight_filled = overnight.fillna(0.0)
    orm_filled = orm.fillna(0.0)

    composite = w_htf * htf_filled + w_overnight * overnight_filled + w_orm * orm_filled
    composite = composite.clip(-1.0, 1.0)

    # Confidence: absolute value of composite
    confidence = composite.abs()

    # Direction: discretize — only call a direction if |composite| > 0.2
    direction = np.where(
        composite.values > 0.2,
        1.0,
        np.where(
            composite.values < -0.2,
            -1.0,
            0.0,
        ),
    )

    result = pd.DataFrame(
        {
            "htf_bias": htf_on_df,
            "overnight_bias": overnight,
            "orm_bias": orm,
            "composite_bias": composite,
            "bias_confidence": confidence,
            "bias_direction": pd.Series(direction, index=df.index, dtype="float64"),
            "htf_pda_count": pda_count_on_df,
        },
        index=df.index,
    )

    logger.info(
        "compute_daily_bias: composite mean=%.3f, "
        "direction dist: bull=%.1f%% bear=%.1f%% neutral=%.1f%%",
        composite.mean(),
        100.0 * (direction == 1).mean(),
        100.0 * (direction == -1).mean(),
        100.0 * (direction == 0).mean(),
    )
    return result


# ---------------------------------------------------------------------------
# 5. compute_regime — risk multiplier based on HTF clarity + PA quality
# ---------------------------------------------------------------------------

def compute_regime(
    df: pd.DataFrame,
    df_4h: pd.DataFrame,
    bias_data: pd.DataFrame,
    params: dict[str, Any],
) -> pd.Series:
    """Compute regime risk multiplier.

    Not a binary on/off — outputs a risk multiplier:
        - 1.0 = full risk: HTF has untested PDAs, fluency OK, not choppy
        - 0.5 = reduced risk: HTF unclear or PA choppy, but some PDAs exist
        - 0.0 = skip day: zero PDAs on any timeframe

    Parameters
    ----------
    df : pd.DataFrame
        Base OHLCV (5m or 1m) with UTC DatetimeIndex.
    df_4h : pd.DataFrame
        4H OHLCV.
    bias_data : pd.DataFrame
        Output of compute_daily_bias(), aligned to df.index.
        Must contain htf_bias, bias_confidence, htf_pda_count.
    params : dict
        Parsed params.yaml.

    Returns
    -------
    pd.Series
        Float values on df.index: 1.0, 0.5, or 0.0.
    """
    from features.displacement import compute_fluency

    regime_params = params["regime"]
    fluency_threshold = params["fluency"]["threshold"]
    chop_range_pts = regime_params["chop_range_points"]
    chop_window = regime_params["chop_range_window_bars"]
    choppy_mult = regime_params["choppy_risk_mult"]

    # --- HTF has untested PDAs? ---
    # Use htf_pda_count: total active FVGs on 4H + 1H.
    # Even if bias is neutral (balanced draws), PDAs still exist if count > 0.
    has_pdas = bias_data["htf_pda_count"].fillna(0) > 0

    # --- HTF fluency check ---
    # Compute 4H fluency, shift(1), align to df
    fluency_4h = compute_fluency(df_4h, params)
    fluency_4h_shifted = fluency_4h.shift(1)
    fluency_on_df = fluency_4h_shifted.reindex(df.index, method="ffill")
    htf_fluent = fluency_on_df >= fluency_threshold

    # --- Chop detection on LTF ---
    # If the range over the last chop_window bars is below chop_range_pts,
    # the market is in chop territory.
    # For 5m data, scale the window: chop_window is in 1m bars
    # Detect the actual bar frequency
    if len(df) >= 2:
        bar_freq_minutes = (df.index[1] - df.index[0]).total_seconds() / 60.0
    else:
        bar_freq_minutes = 1.0

    scaled_window = max(2, int(chop_window / bar_freq_minutes))

    rolling_high = df["high"].rolling(window=scaled_window, min_periods=scaled_window).max()
    rolling_low = df["low"].rolling(window=scaled_window, min_periods=scaled_window).min()
    rolling_range = rolling_high - rolling_low

    not_choppy = rolling_range >= chop_range_pts

    # --- Combine into risk multiplier ---
    # Full risk: has PDAs AND fluent AND not choppy
    # Reduced risk: has PDAs but (not fluent OR choppy)
    # Skip: no PDAs at all
    regime = np.where(
        ~has_pdas.values,
        0.0,  # No PDAs -> skip
        np.where(
            htf_fluent.fillna(False).values & not_choppy.fillna(False).values,
            1.0,  # Full risk
            choppy_mult,  # Reduced risk
        ),
    )

    result = pd.Series(regime, index=df.index, name="regime", dtype="float64")

    valid = result.dropna()
    if len(valid) > 0:
        logger.info(
            "compute_regime: full=%.1f%%, reduced=%.1f%%, skip=%.1f%%",
            100.0 * (valid == 1.0).mean(),
            100.0 * (valid == 0.5).mean(),
            100.0 * (valid == 0.0).mean(),
        )
    return result


# ---------------------------------------------------------------------------
# __main__ — test on March 2025 5m data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time as _time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    params = _load_params(project_root / "config" / "params.yaml")

    # ---- Load data ----
    logger.info("Loading data...")
    t0 = _time.perf_counter()

    df_1m = pd.read_parquet(project_root / "data" / "NQ_1min.parquet")
    df_5m = pd.read_parquet(project_root / "data" / "NQ_5m.parquet")
    df_1h = pd.read_parquet(project_root / "data" / "NQ_1H.parquet")
    df_4h = pd.read_parquet(project_root / "data" / "NQ_4H.parquet")

    logger.info("Data loaded in %.1fs", _time.perf_counter() - t0)

    # ---- Slice to March 2025 ----
    march_start = "2025-03-01"
    march_end = "2025-03-31"

    df_5m_march = df_5m.loc[march_start:march_end]

    # Need extra lead-in for HTF computations (ATR warmup, FVG history)
    htf_lead = pd.Timedelta(days=60)
    htf_start = pd.Timestamp(march_start, tz="UTC") - htf_lead
    df_4h_slice = df_4h.loc[df_4h.index >= htf_start]
    df_1h_slice = df_1h.loc[df_1h.index >= htf_start]

    # 1m for session levels (need the march month)
    df_1m_march = df_1m.loc[march_start:march_end]

    logger.info("March 2025 5m: %d bars", len(df_5m_march))
    logger.info("4H slice: %d bars (from %s)", len(df_4h_slice), df_4h_slice.index[0])
    logger.info("1H slice: %d bars (from %s)", len(df_1h_slice), df_1h_slice.index[0])

    # ---- Compute session levels and ORM on 1m, then align to 5m ----
    from features.sessions import compute_session_levels, compute_orm

    logger.info("Computing session levels on 1m data...")
    t0 = _time.perf_counter()
    session_levels_1m = compute_session_levels(df_1m_march, params)
    orm_1m = compute_orm(df_1m_march, params)
    logger.info("Session levels + ORM computed in %.1fs", _time.perf_counter() - t0)

    # Align 1m session data to 5m by reindex + ffill
    session_levels_5m = session_levels_1m.reindex(df_5m_march.index, method="ffill")
    orm_5m = orm_1m.reindex(df_5m_march.index, method="ffill")

    # ---- Compute daily bias ----
    logger.info("Computing daily bias...")
    t0 = _time.perf_counter()
    bias = compute_daily_bias(
        df_5m_march, session_levels_5m, orm_5m,
        df_4h_slice, df_1h_slice, params,
    )
    elapsed_bias = _time.perf_counter() - t0
    logger.info("Daily bias computed in %.1fs", elapsed_bias)

    # ---- Compute regime ----
    logger.info("Computing regime...")
    t0 = _time.perf_counter()
    regime = compute_regime(df_5m_march, df_4h_slice, bias, params)
    logger.info("Regime computed in %.1fs", _time.perf_counter() - t0)

    # ====================================================================
    # Print results
    # ====================================================================
    print("\n" + "=" * 70)
    print("BIAS DETERMINATION — MARCH 2025 RESULTS")
    print("=" * 70)

    # 1. Distribution of bias directions
    direction = bias["bias_direction"]
    print("\n--- Bias Direction Distribution ---")
    print(f"  Bullish (+1): {(direction == 1).sum():>5d}  ({100.0*(direction==1).mean():.1f}%)")
    print(f"  Bearish (-1): {(direction == -1).sum():>5d}  ({100.0*(direction==-1).mean():.1f}%)")
    print(f"  Neutral  (0): {(direction == 0).sum():>5d}  ({100.0*(direction==0).mean():.1f}%)")

    # 2. Mean confidence
    print(f"\n--- Bias Confidence ---")
    print(f"  Mean:   {bias['bias_confidence'].mean():.4f}")
    print(f"  Median: {bias['bias_confidence'].median():.4f}")
    print(f"  Max:    {bias['bias_confidence'].max():.4f}")

    # 3. Regime distribution
    print(f"\n--- Regime Distribution ---")
    print(f"  Full risk (1.0):    {(regime == 1.0).sum():>5d}  ({100.0*(regime==1.0).mean():.1f}%)")
    print(f"  Reduced (0.5):      {(regime == 0.5).sum():>5d}  ({100.0*(regime==0.5).mean():.1f}%)")
    print(f"  Skip (0.0):         {(regime == 0.0).sum():>5d}  ({100.0*(regime==0.0).mean():.1f}%)")

    # 4. Sample day: 2025-03-19
    print("\n--- Sample Day: 2025-03-19 ---")

    # Convert to ET for filtering
    et_index = df_5m_march.index.tz_convert("US/Eastern")
    day_mask = (et_index.date == pd.Timestamp("2025-03-19").date())

    if day_mask.any():
        day_bias = bias.loc[day_mask]
        day_regime = regime.loc[day_mask]

        # Show key times: around NY open, around ORM end, noon
        key_times_et = ["09:30", "10:00", "10:30", "11:00", "12:00", "14:00"]
        et_frac = et_index[day_mask].hour + et_index[day_mask].minute / 60.0

        print(f"  {'Time (ET)':<16} {'HTF':>6} {'ON':>6} {'ORM':>6} {'Comp':>7} {'Conf':>6} {'Dir':>4} {'Rgm':>4}")
        print(f"  {'-'*14:<16} {'-'*6:>6} {'-'*6:>6} {'-'*6:>6} {'-'*7:>7} {'-'*6:>6} {'-'*4:>4} {'-'*4:>4}")

        for t_str in key_times_et:
            h, m = map(int, t_str.split(":"))
            target_frac = h + m / 60.0
            # Find closest bar at or after target time
            candidates = np.where(day_mask)[0]
            frac_all = et_index.hour + et_index.minute / 60.0
            close_mask = (frac_all[candidates] >= target_frac) & (frac_all[candidates] < target_frac + 5 / 60.0)
            if close_mask.any():
                idx = candidates[close_mask][0]
                ts = df_5m_march.index[idx]
                b = bias.loc[ts]
                r = regime.loc[ts]
                et_str = et_index[idx].strftime("%H:%M")
                print(
                    f"  {et_str:<16} {b['htf_bias']:>6.2f} {b['overnight_bias']:>6.2f} "
                    f"{b['orm_bias']:>6.2f} {b['composite_bias']:>7.3f} "
                    f"{b['bias_confidence']:>6.3f} {b['bias_direction']:>4.0f} {r:>4.1f}"
                )
    else:
        print("  No data for 2025-03-19")

    # 5. Verify: bias doesn't change mid-session
    print("\n--- No Mid-Session Change Verification ---")
    # For each trading day, check that bias_direction doesn't change
    # between 10:30 and 16:00 ET (after ORM settles, bias should hold)
    from features.sessions import _to_et, _et_frac

    et_full = _to_et(df_5m_march.index)
    frac_full = _et_frac(et_full)
    date_keys = (et_full.year * 10000 + et_full.month * 100 + et_full.day)
    date_keys = date_keys.to_numpy() if hasattr(date_keys, 'to_numpy') else np.array(date_keys)

    # Filter to post-ORM NY session (10:30 - 16:00)
    post_orm_mask = (frac_full >= 10.5) & (frac_full < 16.0)
    post_orm_dates = np.unique(date_keys[post_orm_mask])

    stable_days = 0
    total_days = 0
    unstable_examples = []

    for dk in post_orm_dates:
        day_post_orm = post_orm_mask & (date_keys == dk)
        if day_post_orm.sum() < 2:
            continue

        total_days += 1
        day_dirs = bias["bias_direction"].values[day_post_orm]
        unique_dirs = np.unique(day_dirs[~np.isnan(day_dirs)])

        if len(unique_dirs) <= 1:
            stable_days += 1
        else:
            if len(unstable_examples) < 3:
                unstable_examples.append((dk, unique_dirs))

    print(f"  Days checked: {total_days}")
    print(f"  Stable (no change 10:30-16:00): {stable_days} ({100.0*stable_days/max(1,total_days):.1f}%)")
    print(f"  Changed mid-session: {total_days - stable_days}")

    if unstable_examples:
        print(f"  Examples of change:")
        for dk, dirs in unstable_examples:
            print(f"    {dk}: directions = {dirs}")

    # Note: some instability is expected because:
    # - HTF bias ffills from 4H boundaries (can update when new 4H candle closes)
    # - Overnight bias is position-based and moves with price on the base TF
    # This is by design: bias is a real-time assessment, not a static daily label.
    # The ORM component locks after 10:00 ET, but HTF and overnight adapt.
    print("\n  NOTE: HTF bias updates at 4H candle boundaries (by design).")
    print("  Overnight bias is locked at NY open (uses ny_open vs ON range).")
    print("  ORM bias locks after ORM window closes (10:00 ET).")
    print("  Remaining instability comes from HTF bias shifting at 4H close.")

    print("\n" + "=" * 70)
    print("Done.")
