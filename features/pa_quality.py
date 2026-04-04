"""
features/pa_quality.py — Price Action quality features (Phase A).

Quantizes Lanto's discretionary PA assessment into 7 computable features.
These capture the 30% of Lanto's edge that comes from "reading PA quality
and knowing when NOT to trade."

Features:
  1. eagerness      — how quickly price clears the FVG after testing it
  2. engulfing_count — count of engulfing patterns in recent N bars
  3. avg_wick_ratio  — average wick-to-range ratio in recent N bars (lower = cleaner)
  4. alternating_dir_ratio — fraction of direction changes (high = chop/50-50)
  5. new_fvg_in_bias_dir — count of new FVGs formed in bias direction recently
  6. price_acceleration — 2nd derivative of price (momentum speeding up or slowing)
  7. bars_inside_fvg — how long price has been sitting inside the nearest FVG

All features are purely backward-looking. No future data.
References: CLAUDE.md §6, .dev/HANDOFF.md Phase A
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "params.yaml"


def _load_params(config_path: Path = _CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 1. Eagerness — bars to clear FVG after test
# ---------------------------------------------------------------------------

def compute_eagerness(
    df: pd.DataFrame,
    fvg_top: pd.Series,
    fvg_bottom: pd.Series,
    fvg_direction: pd.Series,
    window: int = 10,
) -> pd.Series:
    """Measure how quickly price moves away from an FVG after testing it.

    For each bar, if price is near/inside an active FVG, count how many bars
    it takes for price to clear the FVG zone with conviction. Fewer bars =
    more eagerness = better setup.

    Lanto: "If price sits in a gap for 10-15 minutes without displacing, abandon."

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    fvg_top, fvg_bottom : pd.Series
        Top and bottom of the nearest active FVG (from entry_signals or cache).
    fvg_direction : pd.Series
        +1 for bullish FVG, -1 for bearish.
    window : int
        Lookback window to check for FVG clearance.

    Returns
    -------
    pd.Series[float]
        Eagerness score: 1.0 = instant clearance, 0.0 = stalling (>= window bars).
        NaN where no FVG context.
    """
    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    top = fvg_top.values
    bot = fvg_bottom.values
    direction = fvg_direction.values

    eagerness = np.full(n, np.nan)

    for i in range(1, n):
        if np.isnan(top[i]) or np.isnan(bot[i]) or np.isnan(direction[i]):
            continue

        # Check: is price currently inside or just tested the FVG?
        inside = low[i] <= top[i] and high[i] >= bot[i]
        if not inside:
            # Check if we recently cleared — look back
            bars_since_inside = 0
            for k in range(i - 1, max(i - window, 0) - 1, -1):
                if np.isnan(top[k]) or np.isnan(bot[k]):
                    break
                was_inside = low[k] <= top[k] and high[k] >= bot[k]
                if was_inside:
                    bars_since_inside = i - k
                    break

            if bars_since_inside > 0:
                # Cleared the FVG in bars_since_inside bars
                eagerness[i] = max(0.0, 1.0 - (bars_since_inside - 1) / window)
            # else: not near an FVG, leave NaN
        else:
            # Currently inside — count consecutive bars inside
            consec_inside = 0
            for k in range(i, max(i - window, 0) - 1, -1):
                if np.isnan(top[k]) or np.isnan(bot[k]):
                    break
                still_in = low[k] <= top[k] and high[k] >= bot[k]
                if still_in:
                    consec_inside += 1
                else:
                    break
            # More bars inside = less eager
            eagerness[i] = max(0.0, 1.0 - consec_inside / window)

    result = pd.Series(eagerness, index=df.index, name="eagerness")
    valid = result.notna()
    logger.info("eagerness: mean=%.3f, computed=%d/%d",
                result[valid].mean() if valid.any() else 0, valid.sum(), n)
    return result


# ---------------------------------------------------------------------------
# 2. Engulfing count — engulfing patterns in recent N bars
# ---------------------------------------------------------------------------

def compute_engulfing_count(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Count engulfing candle patterns in the last N bars.

    A candle engulfs the prior candle when its body covers the prior candle's
    entire range (high to low). More engulfing = more displacement/conviction.

    Lanto: "engulfing with no wicks, fluidity, clear PDA creation"

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    window : int
        Rolling lookback window.

    Returns
    -------
    pd.Series[int]
        Count of engulfing patterns in last `window` bars.
    """
    open_ = df["open"].values
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    body_top = np.maximum(open_, close)
    body_bot = np.minimum(open_, close)

    # A bar engulfs the prior bar if its body covers prior bar's full range
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values

    engulf = (body_top >= prev_high) & (body_bot <= prev_low)
    engulf = pd.Series(engulf.astype(float), index=df.index)

    # Rolling count
    result = engulf.rolling(window=window, min_periods=1).sum().astype(int)
    result.name = "engulfing_count"

    logger.info("engulfing_count: mean=%.2f, max=%d", result.mean(), result.max())
    return result


# ---------------------------------------------------------------------------
# 3. Avg wick ratio — candle quality in recent N bars
# ---------------------------------------------------------------------------

def compute_avg_wick_ratio(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Average wick-to-range ratio over last N bars.

    Lower = cleaner candles (more body, less wick). Higher = mass wicks = confusion.

    Lanto: "mass wicks give confusion through HTF narrative"

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    window : int
        Rolling lookback window.

    Returns
    -------
    pd.Series[float]
        Average wick ratio in [0, 1]. Lower is better.
    """
    open_ = df["open"].values
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    body_top = np.maximum(open_, close)
    body_bot = np.minimum(open_, close)

    upper_wick = high - body_top
    lower_wick = body_bot - low
    total_wick = upper_wick + lower_wick
    bar_range = high - low

    # Avoid div by zero
    safe_range = np.where(bar_range > 0, bar_range, np.nan)
    wick_ratio = total_wick / safe_range

    wick_series = pd.Series(wick_ratio, index=df.index)
    result = wick_series.rolling(window=window, min_periods=1).mean()
    result.name = "avg_wick_ratio"

    valid = result.notna()
    logger.info("avg_wick_ratio: mean=%.3f", result[valid].mean() if valid.any() else 0)
    return result


# ---------------------------------------------------------------------------
# 4. Alternating direction ratio — 50/50 tug-of-war detection
# ---------------------------------------------------------------------------

def compute_alternating_dir_ratio(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Fraction of direction changes in last N bars.

    High ratio = alternating bull/bear = 50/50 tug-of-war = chop.
    Low ratio = consistent direction = trending = good PA.

    Lanto: "50/50 buyers vs sellers, don't trade heavy leverage"

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    window : int
        Rolling lookback window.

    Returns
    -------
    pd.Series[float]
        Alternating ratio in [0, 1]. Higher = more chop.
    """
    direction = np.sign(df["close"].values - df["open"].values)
    dir_series = pd.Series(direction, index=df.index)

    # Direction change = current direction != previous direction (and neither is 0)
    prev_dir = dir_series.shift(1)
    changed = ((dir_series != prev_dir) & (dir_series != 0) & (prev_dir != 0)).astype(float)
    # Also count doji bars (direction=0) as a type of indecision
    is_doji = (dir_series == 0).astype(float)
    chop_signal = changed + is_doji * 0.5  # dojis count half

    result = chop_signal.rolling(window=window, min_periods=1).mean()
    result = result.clip(0.0, 1.0)
    result.name = "alternating_dir_ratio"

    logger.info("alternating_dir_ratio: mean=%.3f", result.mean())
    return result


# ---------------------------------------------------------------------------
# 5. New FVG in bias direction — PDA creation confirmation
# ---------------------------------------------------------------------------

def compute_new_fvg_in_bias_dir(
    fvg_bull: pd.Series,
    fvg_bear: pd.Series,
    bias_direction: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Count new FVGs formed in the bias direction over last N bars.

    More FVGs in bias direction = market creating new draws in our favor = good.

    Lanto: "clear PDA creation, momentum+speed, gap displaced + took liquidity"

    Parameters
    ----------
    fvg_bull : pd.Series[bool]
        True where a bullish FVG was detected.
    fvg_bear : pd.Series[bool]
        True where a bearish FVG was detected.
    bias_direction : pd.Series[float]
        +1 bullish, -1 bearish, 0 neutral.
    window : int
        Rolling lookback window.

    Returns
    -------
    pd.Series[int]
        Count of new FVGs aligned with bias in last N bars.
    """
    bull = fvg_bull.astype(float).values
    bear = fvg_bear.astype(float).values
    bias = bias_direction.values

    # FVG aligned with bias: bull FVG when bias is bullish, bear FVG when bearish
    aligned = np.where(bias > 0, bull, np.where(bias < 0, bear, 0.0))
    aligned_series = pd.Series(aligned, index=fvg_bull.index)

    result = aligned_series.rolling(window=window, min_periods=1).sum().astype(int)
    result.name = "new_fvg_in_bias_dir"

    logger.info("new_fvg_in_bias_dir: mean=%.2f, max=%d", result.mean(), result.max())
    return result


# ---------------------------------------------------------------------------
# 6. Price acceleration — 2nd derivative of price
# ---------------------------------------------------------------------------

def compute_price_acceleration(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Compute price acceleration (2nd derivative of close price).

    Positive acceleration = price speeding up in current direction = momentum.
    Negative acceleration = price slowing down = possible reversal.

    Lanto: "momentum+speed" as a TAKE criterion.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    window : int
        Smoothing window for velocity computation.

    Returns
    -------
    pd.Series[float]
        Acceleration normalized by ATR. Positive = speeding up, negative = slowing.
    """
    close = df["close"]

    # Velocity: change in price over window
    velocity = close.diff(window)
    # Acceleration: change in velocity
    acceleration = velocity.diff(window)

    # Normalize by ATR for scale-invariance across price eras
    from features.displacement import compute_atr
    atr = compute_atr(df, period=14)
    safe_atr = atr.replace(0, np.nan)

    # Normalize: acceleration / ATR
    result = acceleration / safe_atr
    result = result.clip(-5.0, 5.0)  # cap extreme values
    result.name = "price_acceleration"

    valid = result.notna()
    logger.info("price_acceleration: mean=%.3f, std=%.3f",
                result[valid].mean() if valid.any() else 0,
                result[valid].std() if valid.any() else 0)
    return result


# ---------------------------------------------------------------------------
# 7. Bars inside FVG — stalling detection
# ---------------------------------------------------------------------------

def compute_bars_inside_fvg(
    df: pd.DataFrame,
    fvg_top: pd.Series,
    fvg_bottom: pd.Series,
    max_count: int = 20,
) -> pd.Series:
    """Count consecutive bars where price is inside the nearest FVG.

    More bars inside = stalling = price can't decide = bad PA.

    Lanto: "1m entry sits in an FVG for > 10-15 minutes without displacing = abandon"

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    fvg_top, fvg_bottom : pd.Series
        Top and bottom of nearest active FVG.
    max_count : int
        Cap the count at this value.

    Returns
    -------
    pd.Series[int]
        Consecutive bars inside FVG. 0 = not inside. Higher = more stalling.
    """
    n = len(df)
    high = df["high"].values
    low = df["low"].values
    top = fvg_top.values
    bot = fvg_bottom.values

    result = np.zeros(n, dtype=int)
    streak = 0

    for i in range(n):
        if np.isnan(top[i]) or np.isnan(bot[i]):
            streak = 0
            result[i] = 0
            continue

        inside = low[i] <= top[i] and high[i] >= bot[i]
        if inside:
            streak += 1
            result[i] = min(streak, max_count)
        else:
            streak = 0
            result[i] = 0

    out = pd.Series(result, index=df.index, name="bars_inside_fvg")
    logger.info("bars_inside_fvg: mean=%.2f, max=%d, pct_inside=%.1f%%",
                out.mean(), out.max(), 100.0 * (out > 0).mean())
    return out


# ---------------------------------------------------------------------------
# Composite: compute all PA quality features at once
# ---------------------------------------------------------------------------

def compute_all_pa_features(
    df: pd.DataFrame,
    signals_df: pd.DataFrame | None = None,
    bias_data: pd.DataFrame | None = None,
    params: dict | None = None,
    pa_window: int = 10,
) -> pd.DataFrame:
    """Compute all 7 PA quality features and return as a single DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        5m OHLCV data.
    signals_df : pd.DataFrame
        Output from detect_all_signals (for FVG zone context).
    bias_data : pd.DataFrame
        Output from compute_daily_bias (for bias direction).
    params : dict
        From params.yaml.
    pa_window : int
        Default rolling window for PA features.

    Returns
    -------
    pd.DataFrame with 7 columns, same index as df.
    """
    if params is None:
        params = _load_params()

    features = pd.DataFrame(index=df.index)

    # Features that don't need FVG context
    features["engulfing_count"] = compute_engulfing_count(df, window=pa_window)
    features["avg_wick_ratio"] = compute_avg_wick_ratio(df, window=pa_window)
    features["alternating_dir_ratio"] = compute_alternating_dir_ratio(df, window=pa_window)
    features["price_acceleration"] = compute_price_acceleration(df, window=5)

    # Features that need FVG context — use detect_fvg for FVG zone boundaries
    from features.fvg import detect_fvg
    fvg_df = detect_fvg(df)

    # Build nearest FVG zone by forward-filling the most recent FVG boundaries
    # Bull FVG: fvg_bull_top / fvg_bull_bottom
    # Bear FVG: fvg_bear_top / fvg_bear_bottom
    bull_top = fvg_df["fvg_bull_top"].ffill()
    bull_bot = fvg_df["fvg_bull_bottom"].ffill()
    bear_top = fvg_df["fvg_bear_top"].ffill()
    bear_bot = fvg_df["fvg_bear_bottom"].ffill()

    # Pick nearest FVG to current price (by midpoint distance)
    close = df["close"]
    bull_mid = (bull_top + bull_bot) / 2
    bear_mid = (bear_top + bear_bot) / 2
    bull_dist = (close - bull_mid).abs()
    bear_dist = (close - bear_mid).abs()

    use_bull = bull_dist.fillna(np.inf) <= bear_dist.fillna(np.inf)
    fvg_top_nearest = pd.Series(
        np.where(use_bull, bull_top.values, bear_top.values),
        index=df.index
    )
    fvg_bot_nearest = pd.Series(
        np.where(use_bull, bull_bot.values, bear_bot.values),
        index=df.index
    )

    # Direction from bias or signal context
    if signals_df is not None:
        fvg_dir = signals_df.get("signal_dir", pd.Series(0.0, index=df.index))
        fvg_dir_ff = fvg_dir.replace(0, np.nan).ffill().fillna(0)
    elif bias_data is not None:
        fvg_dir_ff = bias_data["bias_direction"].reindex(df.index).ffill().fillna(0)
    else:
        fvg_dir_ff = pd.Series(0.0, index=df.index)

    features["eagerness"] = compute_eagerness(
        df, fvg_top_nearest, fvg_bot_nearest, fvg_dir_ff, window=pa_window
    )
    features["bars_inside_fvg"] = compute_bars_inside_fvg(
        df, fvg_top_nearest, fvg_bot_nearest
    )

    # New FVG in bias direction
    if bias_data is not None:
        from features.fvg import detect_fvg
        fvg_df = detect_fvg(df)
        bias_dir = bias_data["bias_direction"].reindex(df.index).ffill().fillna(0)
        features["new_fvg_in_bias_dir"] = compute_new_fvg_in_bias_dir(
            fvg_df["fvg_bull"], fvg_df["fvg_bear"], bias_dir, window=20
        )
    else:
        features["new_fvg_in_bias_dir"] = 0

    # --- 8. Pre-signal consolidation ratio (break & rotation proxy) ---
    # Tight range before signal = consolidation before breakout = better
    # Lanto: "I prefer break and rotation"
    from features.displacement import compute_atr as _compute_atr
    atr_for_consol = _compute_atr(df, period=14)
    lookback = pa_window
    rolling_high = df["high"].rolling(lookback, min_periods=1).max()
    rolling_low = df["low"].rolling(lookback, min_periods=1).min()
    rolling_range = rolling_high - rolling_low
    safe_atr = atr_for_consol.replace(0, np.nan)
    features["pre_signal_consol_ratio"] = rolling_range / safe_atr

    logger.info("PA quality features computed: %s columns, %d rows",
                list(features.columns), len(features))
    return features


# ---------------------------------------------------------------------------
# Standalone test
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

    params = _load_params()

    # Load 5m data (recent slice for speed)
    logger.info("Loading 5m data...")
    df = pd.read_parquet(project_root / "data" / "NQ_5m.parquet")
    df = df.loc["2025-01-01":"2025-03-31"].copy()
    logger.info("Loaded %d bars (%s to %s)", len(df), df.index[0], df.index[-1])

    # Compute PA features (without FVG/bias context for quick test)
    t0 = _time.perf_counter()
    pa = compute_all_pa_features(df, params=params)
    elapsed = _time.perf_counter() - t0

    print(f"\n{'='*60}")
    print("PA QUALITY FEATURES — Quick Test (2025 Q1)")
    print(f"{'='*60}")
    print(f"Computed in {elapsed:.1f}s on {len(df)} bars\n")

    for col in pa.columns:
        s = pa[col].dropna()
        if len(s) > 0:
            print(f"  {col:>25s}: mean={s.mean():.3f}  std={s.std():.3f}  "
                  f"min={s.min():.3f}  max={s.max():.3f}  NaN={pa[col].isna().sum()}")
        else:
            print(f"  {col:>25s}: ALL NaN")

    print(f"\n{'='*60}")
