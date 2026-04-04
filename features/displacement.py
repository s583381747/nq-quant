"""
Displacement, fluency, and bad-candle detection for NQ 1-minute data.

Implements:
    - ATR computation
    - Displacement candle detection (large body, high body/range ratio, engulfs prior candles)
    - Fluency scoring (directional consistency, body quality, bar size)
    - Bad candle flagging (dojis and long-wick candles)

All thresholds are read from config/params.yaml. No hardcoded values.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "params.yaml"


def _load_params(config_path: Path = _CONFIG_PATH) -> dict:
    """Load tunable parameters from params.yaml."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR) over *period* bars.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|).
    Uses exponential moving average (Wilder smoothing) consistent with
    standard ATR definition.

    Parameters
    ----------
    df : DataFrame with columns ``high``, ``low``, ``close``.
    period : lookback window (default 14).

    Returns
    -------
    pd.Series of ATR values, same index as *df*.
    First *period* bars will contain NaN (insufficient history).
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    true_range.name = "true_range"

    # Wilder smoothing (equivalent to EMA with alpha = 1/period)
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    atr.name = "atr"

    logger.debug("ATR(%d) computed: mean=%.2f, median=%.2f", period, atr.mean(), atr.median())
    return atr


# ---------------------------------------------------------------------------
# Displacement detection
# ---------------------------------------------------------------------------


def detect_displacement(df: pd.DataFrame, params: dict) -> pd.Series:
    """Detect displacement candles per CLAUDE.md section 6.1.

    A candle is a displacement candle when ALL of the following hold:
        1. body > ``displacement.atr_mult`` * ATR(14)
        2. body / range > ``displacement.body_ratio``
        3. The candle engulfs at least ``displacement.engulf_min_candles``
           prior candles (its body spans from one end to the other of at
           least N preceding candles' ranges).

    Parameters
    ----------
    df : 1-minute OHLCV DataFrame.
    params : parsed params.yaml dict.

    Returns
    -------
    pd.Series[bool] — True where candle is a displacement candle.
    """
    cfg = params["displacement"]
    atr_mult: float = cfg["atr_mult"]
    body_ratio_min: float = cfg["body_ratio"]
    engulf_n: int = cfg["engulf_min_candles"]

    atr = compute_atr(df, period=14)

    body = (df["close"] - df["open"]).abs()
    bar_range = df["high"] - df["low"]

    # Guard against zero-range bars (exact dojis where high == low)
    safe_range = bar_range.replace(0, np.nan)

    # Criterion 1: body > atr_mult * ATR
    crit_body_size = body > (atr_mult * atr)

    # Criterion 2: body/range ratio
    body_ratio = body / safe_range
    crit_body_ratio = body_ratio > body_ratio_min

    # Criterion 3: engulfs at least N prior candles
    # A candle "engulfs" a prior candle if the current candle's body
    # covers the prior candle's full range (high to low).
    candle_top = df[["open", "close"]].max(axis=1)
    candle_bot = df[["open", "close"]].min(axis=1)

    # For each bar, count how many of the preceding N candles are engulfed
    # (current body top >= prior high AND current body bottom <= prior low).
    engulf_count = pd.Series(0, index=df.index, dtype="int64")
    for lag in range(1, engulf_n + 1):
        prior_high = df["high"].shift(lag)
        prior_low = df["low"].shift(lag)
        engulfed = (candle_top >= prior_high) & (candle_bot <= prior_low)
        engulf_count += engulfed.astype(int)

    crit_engulf = engulf_count >= engulf_n

    # Combine: all three must hold; NaN → False
    is_displacement = crit_body_size & crit_body_ratio & crit_engulf
    is_displacement = is_displacement.fillna(False).astype(bool)
    is_displacement.name = "is_displacement"

    n_disp = is_displacement.sum()
    logger.info(
        "Displacement candles: %s / %s (%.4f%%)",
        f"{n_disp:,}",
        f"{len(df):,}",
        100.0 * n_disp / len(df) if len(df) > 0 else 0,
    )
    return is_displacement


# ---------------------------------------------------------------------------
# Fluency scoring
# ---------------------------------------------------------------------------


def compute_fluency(df: pd.DataFrame, params: dict) -> pd.Series:
    """Compute fluency score per CLAUDE.md section 6.2.

    Fluency measures how directional / clean recent price action is.

    Components (all computed over a rolling window of N candles):
        1. ``directional_ratio`` — fraction of candles in the dominant direction
        2. ``avg_body_ratio``    — mean(body / range)
        3. ``avg_bar_size_vs_atr`` — mean(range / ATR), capped at 1.0

    Composite:
        fluency = w1 * directional_ratio + w2 * avg_body_ratio + w3 * bar_size_component

    Parameters
    ----------
    df : 1-minute OHLCV DataFrame.
    params : parsed params.yaml dict.

    Returns
    -------
    pd.Series[float] in [0, 1] — fluency score per bar.
    """
    cfg = params["fluency"]
    window: int = cfg["window"]
    w1: float = cfg["w_directional"]
    w2: float = cfg["w_body_ratio"]
    w3: float = cfg["w_bar_size"]

    atr = compute_atr(df, period=14)

    # Direction: +1 for bullish (close > open), -1 for bearish, 0 for doji
    direction = np.sign(df["close"] - df["open"])

    # directional_ratio: fraction of candles in the majority direction within
    # the rolling window.  E.g. 4 bullish + 1 bearish + 1 doji in window=6
    # → max(4, 1) / 6 = 0.667.
    #
    # NOTE: This replaces the old "net direction" formula (rolling sum / window)
    # which understated directional_ratio when bull and bear counts partially
    # cancelled.  The new formula produces HIGHER values on average, so the
    # existing fluency threshold (params) may now be more permissive.
    bull_count = (direction == 1).astype(float).rolling(window=window, min_periods=window).sum()
    bear_count = (direction == -1).astype(float).rolling(window=window, min_periods=window).sum()
    directional_ratio = np.maximum(bull_count, bear_count) / window

    # avg_body_ratio: mean(body / range) over window
    body = (df["close"] - df["open"]).abs()
    bar_range = df["high"] - df["low"]
    safe_range = bar_range.replace(0, np.nan)
    single_body_ratio = (body / safe_range).fillna(0.0)
    avg_body_ratio = single_body_ratio.rolling(window=window, min_periods=window).mean()

    # avg_bar_size_vs_atr: mean(range / ATR) over window, capped at 1.0 per bar
    safe_atr = atr.replace(0, np.nan)
    bar_size_ratio = (bar_range / safe_atr).clip(upper=2.0).fillna(0.0)
    avg_bar_size = bar_size_ratio.rolling(window=window, min_periods=window).mean()
    # Normalize: cap at 1.0 so it stays in [0,1] for weighting
    avg_bar_size_norm = avg_bar_size.clip(upper=1.0)

    # Composite
    fluency = w1 * directional_ratio + w2 * avg_body_ratio + w3 * avg_bar_size_norm
    # Clamp to [0, 1]
    fluency = fluency.clip(lower=0.0, upper=1.0)
    fluency.name = "fluency"

    logger.info(
        "Fluency: mean=%.4f, median=%.4f, >=threshold(%.2f): %.2f%%",
        fluency.mean(),
        fluency.median(),
        cfg["threshold"],
        100.0 * (fluency >= cfg["threshold"]).mean(),
    )
    return fluency


# ---------------------------------------------------------------------------
# Bad candle detection
# ---------------------------------------------------------------------------


def detect_bad_candles(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Flag doji and long-wick candles per CLAUDE.md section 6.3.

    A "bad candle" is one to avoid trading around:
        - **Doji**: body/range < ``bad_candle.doji_body_ratio``
        - **Long wick**: max(upper_wick, lower_wick) / range > ``bad_candle.long_wick_ratio``

    Zero-range bars (high == low) are classified as dojis and NOT as long-wick
    (to avoid division-by-zero artifacts).

    Parameters
    ----------
    df : 1-minute OHLCV DataFrame.
    params : parsed params.yaml dict.

    Returns
    -------
    pd.DataFrame with columns ``is_doji`` (bool), ``is_long_wick`` (bool),
    same index as *df*.
    """
    cfg = params["bad_candle"]
    doji_threshold: float = cfg["doji_body_ratio"]
    wick_threshold: float = cfg["long_wick_ratio"]

    body = (df["close"] - df["open"]).abs()
    bar_range = df["high"] - df["low"]

    # Zero-range bars → always doji, never long_wick
    is_zero_range = bar_range == 0

    safe_range = bar_range.replace(0, np.nan)

    # Doji detection
    body_ratio = body / safe_range
    is_doji = (body_ratio < doji_threshold) | is_zero_range
    is_doji = is_doji.fillna(True).astype(bool)  # NaN from zero range → doji
    is_doji.name = "is_doji"

    # Wick detection
    candle_top = df[["open", "close"]].max(axis=1)
    candle_bot = df[["open", "close"]].min(axis=1)
    upper_wick = df["high"] - candle_top
    lower_wick = candle_bot - df["low"]
    max_wick = pd.concat([upper_wick, lower_wick], axis=1).max(axis=1)

    wick_ratio = max_wick / safe_range
    is_long_wick = wick_ratio > wick_threshold
    # Zero-range bars: no meaningful wick ratio → False
    is_long_wick = is_long_wick.fillna(False).astype(bool)
    is_long_wick.name = "is_long_wick"

    n_doji = is_doji.sum()
    n_wick = is_long_wick.sum()
    logger.info(
        "Bad candles: %s doji (%.2f%%), %s long_wick (%.2f%%)",
        f"{n_doji:,}",
        100.0 * n_doji / len(df) if len(df) > 0 else 0,
        f"{n_wick:,}",
        100.0 * n_wick / len(df) if len(df) > 0 else 0,
    )

    return pd.DataFrame({"is_doji": is_doji, "is_long_wick": is_long_wick}, index=df.index)


# ---------------------------------------------------------------------------
# Main: summary stats & verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # 1. Load data
    data_path = Path(__file__).resolve().parent.parent / "data" / "NQ_1min.parquet"
    logger.info("Loading data from %s", data_path)
    df = pd.read_parquet(data_path, engine="pyarrow")
    logger.info("Loaded %s bars, %s to %s", f"{len(df):,}", df.index.min(), df.index.max())

    # 2. Load params
    params = _load_params()

    # 3. Compute all features
    logger.info("--- Computing ATR ---")
    atr = compute_atr(df, period=14)

    logger.info("--- Computing displacement ---")
    is_disp = detect_displacement(df, params)

    logger.info("--- Computing fluency ---")
    fluency = compute_fluency(df, params)

    logger.info("--- Detecting bad candles ---")
    bad = detect_bad_candles(df, params)

    # 4. Summary stats
    print("\n" + "=" * 60)
    print("DISPLACEMENT MODULE — SUMMARY STATS")
    print("=" * 60)
    print(f"Total bars:             {len(df):>12,}")
    print(f"Date range:             {df.index.min()} → {df.index.max()}")
    print()
    print(f"ATR(14) mean:           {atr.mean():>12.2f}")
    print(f"ATR(14) median:         {atr.median():>12.2f}")
    print(f"ATR(14) NaN count:      {atr.isna().sum():>12,}")
    print()
    print(f"Displacement candles:   {is_disp.sum():>12,}")
    print(f"Displacement %:         {100.0 * is_disp.mean():>12.4f}%")
    print(f"Displacement NaN:       {is_disp.isna().sum():>12,}")
    print()
    print(f"Fluency mean:           {fluency.mean():>12.4f}")
    print(f"Fluency median:         {fluency.median():>12.4f}")
    print(f"Fluency >= threshold:   {100.0 * (fluency >= params['fluency']['threshold']).mean():>12.2f}%")
    print(f"Fluency NaN count:      {fluency.isna().sum():>12,}")
    print()
    print(f"Doji candles:           {bad['is_doji'].sum():>12,}")
    print(f"Doji %:                 {100.0 * bad['is_doji'].mean():>12.2f}%")
    print(f"Long wick candles:      {bad['is_long_wick'].sum():>12,}")
    print(f"Long wick %:            {100.0 * bad['is_long_wick'].mean():>12.2f}%")
    print()

    # 5. Verify no NaN in boolean outputs
    nan_disp = is_disp.isna().sum()
    nan_fluency_non_warmup = fluency.iloc[params["fluency"]["window"] + 14:].isna().sum()
    nan_doji = bad["is_doji"].isna().sum()
    nan_wick = bad["is_long_wick"].isna().sum()

    print("--- NaN VERIFICATION ---")
    print(f"is_displacement NaN:    {nan_disp:>12,}  {'PASS' if nan_disp == 0 else 'FAIL'}")
    print(f"fluency NaN (post-warmup): {nan_fluency_non_warmup:>8,}  {'PASS' if nan_fluency_non_warmup == 0 else 'FAIL'}")
    print(f"is_doji NaN:            {nan_doji:>12,}  {'PASS' if nan_doji == 0 else 'FAIL'}")
    print(f"is_long_wick NaN:       {nan_wick:>12,}  {'PASS' if nan_wick == 0 else 'FAIL'}")
    print("=" * 60)
