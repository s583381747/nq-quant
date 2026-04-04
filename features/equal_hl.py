"""
features/equal_hl.py — Equal Highs/Lows detection.

Equal Highs/Lows are clusters of swing points at approximately the same price.
They represent concentrated liquidity (many stops/orders at similar levels).

Lanto ranks these #2 in liquidity importance (after HTF swing H/L):
  "Equal highs/lows are liquidity pools where multiple traders have likely
   placed orders at the same price."

Detection: scan recent swing points, find clusters within tolerance (default 3 pts).
Output: per-bar count of nearby equal H/L clusters + nearest equal H/L level.

References: .dev/lantogpt_deep_dive.md Q1
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_equal_highs(
    swing_high_flags: pd.Series,
    high_prices: pd.Series,
    tolerance: float = 3.0,
    lookback: int = 100,
    min_cluster: int = 2,
) -> pd.DataFrame:
    """Detect equal highs — clusters of swing highs at similar prices.

    Parameters
    ----------
    swing_high_flags : pd.Series[bool]
        True at confirmed swing high bars (already shifted for no-lookahead).
    high_prices : pd.Series[float]
        High prices at each bar.
    tolerance : float
        Max price difference to consider two swing highs "equal" (in points).
    lookback : int
        Number of bars to look back for recent swing highs.
    min_cluster : int
        Minimum number of swing highs at similar price to count as "equal highs".

    Returns
    -------
    pd.DataFrame with columns:
        equal_high_count : int — number of swing highs clustered at nearest equal high level
        equal_high_level : float — price of nearest equal high cluster (NaN if none)
    """
    n = len(swing_high_flags)
    flags = swing_high_flags.values.astype(bool)
    prices = high_prices.values

    eq_count = np.zeros(n, dtype=int)
    eq_level = np.full(n, np.nan)

    # Collect swing high prices as they appear
    recent_sh_prices: list[float] = []
    recent_sh_indices: list[int] = []

    for i in range(n):
        # Add new swing high if flagged
        if flags[i] and not np.isnan(prices[i]):
            recent_sh_prices.append(prices[i])
            recent_sh_indices.append(i)

        # Prune old entries beyond lookback
        while recent_sh_indices and (i - recent_sh_indices[0]) > lookback:
            recent_sh_indices.pop(0)
            recent_sh_prices.pop(0)

        if len(recent_sh_prices) < min_cluster:
            continue

        # Find clusters: for each unique-ish level, count how many swing highs are within tolerance
        best_cluster_size = 0
        best_cluster_level = np.nan

        # Simple O(n^2) within the lookback window (typically < 20 swings)
        arr = np.array(recent_sh_prices)
        for j in range(len(arr)):
            cluster = np.sum(np.abs(arr - arr[j]) <= tolerance)
            if cluster >= min_cluster and cluster > best_cluster_size:
                best_cluster_size = cluster
                best_cluster_level = arr[j]

        eq_count[i] = best_cluster_size
        eq_level[i] = best_cluster_level

    return pd.DataFrame({
        'equal_high_count': eq_count,
        'equal_high_level': eq_level,
    }, index=swing_high_flags.index)


def detect_equal_lows(
    swing_low_flags: pd.Series,
    low_prices: pd.Series,
    tolerance: float = 3.0,
    lookback: int = 100,
    min_cluster: int = 2,
) -> pd.DataFrame:
    """Detect equal lows — clusters of swing lows at similar prices.

    Same logic as detect_equal_highs but for lows.
    """
    n = len(swing_low_flags)
    flags = swing_low_flags.values.astype(bool)
    prices = low_prices.values

    eq_count = np.zeros(n, dtype=int)
    eq_level = np.full(n, np.nan)

    recent_sl_prices: list[float] = []
    recent_sl_indices: list[int] = []

    for i in range(n):
        if flags[i] and not np.isnan(prices[i]):
            recent_sl_prices.append(prices[i])
            recent_sl_indices.append(i)

        while recent_sl_indices and (i - recent_sl_indices[0]) > lookback:
            recent_sl_indices.pop(0)
            recent_sl_prices.pop(0)

        if len(recent_sl_prices) < min_cluster:
            continue

        arr = np.array(recent_sl_prices)
        best_cluster_size = 0
        best_cluster_level = np.nan

        for j in range(len(arr)):
            cluster = np.sum(np.abs(arr - arr[j]) <= tolerance)
            if cluster >= min_cluster and cluster > best_cluster_size:
                best_cluster_size = cluster
                best_cluster_level = arr[j]

        eq_count[i] = best_cluster_size
        eq_level[i] = best_cluster_level

    return pd.DataFrame({
        'equal_low_count': eq_count,
        'equal_low_level': eq_level,
    }, index=swing_low_flags.index)


def compute_equal_hl(
    df: pd.DataFrame,
    swing_df: pd.DataFrame,
    tolerance: float = 3.0,
    lookback: int = 100,
) -> pd.DataFrame:
    """Compute all equal H/L features.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    swing_df : pd.DataFrame
        Output from compute_swing_levels (must have swing_high, swing_low,
        swing_high_price, swing_low_price columns). Should already be shifted
        for no-lookahead.
    tolerance : float
        Price tolerance for clustering (points).
    lookback : int
        Bars to look back for recent swings.

    Returns
    -------
    pd.DataFrame with columns:
        equal_high_count, equal_high_level,
        equal_low_count, equal_low_level,
        total_equal_hl_count
    """
    eh = detect_equal_highs(
        swing_df['swing_high'], df['high'],
        tolerance=tolerance, lookback=lookback,
    )
    el = detect_equal_lows(
        swing_df['swing_low'], df['low'],
        tolerance=tolerance, lookback=lookback,
    )

    result = pd.DataFrame({
        'equal_high_count': eh['equal_high_count'],
        'equal_high_level': eh['equal_high_level'],
        'equal_low_count': el['equal_low_count'],
        'equal_low_level': el['equal_low_level'],
        'total_equal_hl_count': eh['equal_high_count'] + el['equal_low_count'],
    }, index=df.index)

    logger.info(
        "equal_hl: mean_eh=%.1f, mean_el=%.1f, bars_with_eq=%.1f%%",
        result['equal_high_count'].mean(),
        result['equal_low_count'].mean(),
        100 * (result['total_equal_hl_count'] > 0).mean(),
    )
    return result


if __name__ == '__main__':
    import sys, time
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from features.swing import compute_swing_levels

    df = pd.read_parquet(Path(__file__).resolve().parent.parent / 'data/NQ_5m.parquet')
    df = df.loc['2025-01-01':'2025-03-31']
    print(f'Data: {len(df)} bars')

    swings = compute_swing_levels(df, {'left_bars': 3, 'right_bars': 1})
    # Shift for no-lookahead
    swings['swing_high'] = swings['swing_high'].shift(1, fill_value=False)
    swings['swing_low'] = swings['swing_low'].shift(1, fill_value=False)

    t0 = time.time()
    eq = compute_equal_hl(df, swings, tolerance=3.0, lookback=100)
    print(f'Computed in {time.time()-t0:.1f}s')
    print(eq.describe())
