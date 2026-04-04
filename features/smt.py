"""
SMT (Smart Money Technique) Divergence Detection — NQ vs ES.

Lanto's SMT rule for MSS confirmation:
  Bull SMT: NQ sweeps a swing LOW but ES does NOT → NQ fake breakdown → long signal
  Bear SMT: NQ sweeps a swing HIGH but ES does NOT → NQ fake breakout → short signal

SMT is about liquidity sweep divergence between correlated instruments,
NOT about price direction divergence.

Usage:
    smt_df = compute_smt(nq_df, es_df, params)
    # Returns columns: smt_bull, smt_bear, nq_swept_high, nq_swept_low, etc.
"""

import logging
import numpy as np
import pandas as pd
from features.swing import detect_swing_highs, detect_swing_lows

logger = logging.getLogger(__name__)


def _detect_sweeps(
    high: np.ndarray,
    low: np.ndarray,
    swing_high_price: np.ndarray,
    swing_low_price: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect if price swept a swing level within the last `lookback` bars.

    A sweep occurs when:
      - high[i] > swing_high_price[i] (price broke above the most recent swing high)
      - low[i] < swing_low_price[i]   (price broke below the most recent swing low)

    We track whether a sweep happened in ANY of the last `lookback` bars.

    Parameters
    ----------
    high, low : np.ndarray
        Price arrays.
    swing_high_price, swing_low_price : np.ndarray
        Forward-filled most recent swing level prices.
    lookback : int
        Number of bars to look back for sweep detection.

    Returns
    -------
    swept_high, swept_low : np.ndarray (bool)
        True if a sweep of the respective level occurred within lookback bars.
    """
    n = len(high)
    swept_high = np.zeros(n, dtype=np.bool_)
    swept_low = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        if np.isnan(swing_high_price[i]) or np.isnan(swing_low_price[i]):
            continue

        # Check if any bar in [i-lookback+1, i] swept the swing level
        start = max(0, i - lookback + 1)
        for j in range(start, i + 1):
            if high[j] > swing_high_price[i]:
                swept_high[i] = True
                break

        for j in range(start, i + 1):
            if low[j] < swing_low_price[i]:
                swept_low[i] = True
                break

    return swept_high, swept_low


def _detect_sweeps_vectorized(
    high: np.ndarray,
    low: np.ndarray,
    swing_high_price: np.ndarray,
    swing_low_price: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized sweep detection using rolling max/min.

    A sweep of swing high = rolling max of high over lookback > swing_high_price.
    A sweep of swing low = rolling min of low over lookback < swing_low_price.
    """
    n = len(high)

    # Rolling max of high over lookback window
    high_s = pd.Series(high)
    rolling_high = high_s.rolling(window=lookback, min_periods=1).max().values

    # Rolling min of low over lookback window
    low_s = pd.Series(low)
    rolling_low = low_s.rolling(window=lookback, min_periods=1).min().values

    swept_high = (rolling_high > swing_high_price) & ~np.isnan(swing_high_price)
    swept_low = (rolling_low < swing_low_price) & ~np.isnan(swing_low_price)

    return swept_high.astype(np.bool_), swept_low.astype(np.bool_)


def compute_smt(
    nq: pd.DataFrame,
    es: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Compute SMT divergence between NQ and ES.

    Parameters
    ----------
    nq : pd.DataFrame
        NQ 5m OHLCV with columns: high, low, close (UTC index).
    es : pd.DataFrame
        ES 5m OHLCV with columns: high, low, close (UTC index).
    params : dict
        Must contain 'smt' section with:
          sweep_lookback: int (bars to look back for sweep, default 20)
          time_tolerance: int (bars tolerance for cross-instrument, default 2)
        And 'swing' section with left_bars, right_bars.

    Returns
    -------
    pd.DataFrame
        Columns:
          smt_bull       — bool: NQ swept low but ES didn't (long confirmation)
          smt_bear       — bool: NQ swept high but ES didn't (short confirmation)
          nq_swept_high  — bool: NQ swept its most recent swing high
          nq_swept_low   — bool: NQ swept its most recent swing low
          es_swept_high  — bool: ES swept its most recent swing high
          es_swept_low   — bool: ES swept its most recent swing low
    """
    smt_params = params.get("smt", {})
    sweep_lookback = smt_params.get("sweep_lookback", 20)
    time_tolerance = smt_params.get("time_tolerance", 2)
    swing_params = params.get("swing", {"left_bars": 3, "right_bars": 1})
    left = swing_params["left_bars"]
    right = swing_params["right_bars"]

    # Align indices — only use shared timestamps
    shared_idx = nq.index.intersection(es.index)
    logger.info("SMT: NQ %d bars, ES %d bars, shared %d", len(nq), len(es), len(shared_idx))

    nq_aligned = nq.loc[shared_idx].copy()
    es_aligned = es.loc[shared_idx].copy()

    # Step 1: Detect swing levels on both instruments
    logger.info("SMT: detecting swing levels on NQ...")
    nq_sh = detect_swing_highs(nq_aligned["high"], left, right)
    nq_sl = detect_swing_lows(nq_aligned["low"], left, right)

    # AUDIT FIX: shift(right) after ffill — swing at bar i is only confirmed
    # at bar i+right. Without this, sweep detection uses unconfirmed levels.
    nq_sh_price = np.where(nq_sh.values, nq_aligned["high"].values, np.nan)
    nq_sh_price = pd.Series(nq_sh_price).ffill().shift(right).values

    nq_sl_price = np.where(nq_sl.values, nq_aligned["low"].values, np.nan)
    nq_sl_price = pd.Series(nq_sl_price).ffill().shift(right).values

    logger.info("SMT: detecting swing levels on ES...")
    es_sh = detect_swing_highs(es_aligned["high"], left, right)
    es_sl = detect_swing_lows(es_aligned["low"], left, right)

    es_sh_price = np.where(es_sh.values, es_aligned["high"].values, np.nan)
    es_sh_price = pd.Series(es_sh_price).ffill().shift(right).values

    es_sl_price = np.where(es_sl.values, es_aligned["low"].values, np.nan)
    es_sl_price = pd.Series(es_sl_price).ffill().shift(right).values

    # Step 2: Detect sweeps on both instruments (vectorized)
    logger.info("SMT: detecting sweeps (lookback=%d)...", sweep_lookback)

    nq_swept_high, nq_swept_low = _detect_sweeps_vectorized(
        nq_aligned["high"].values, nq_aligned["low"].values,
        nq_sh_price, nq_sl_price, sweep_lookback
    )

    es_swept_high, es_swept_low = _detect_sweeps_vectorized(
        es_aligned["high"].values, es_aligned["low"].values,
        es_sh_price, es_sl_price, sweep_lookback
    )

    # Step 3: Apply time tolerance for ES non-sweep
    # If time_tolerance > 0, ES "not swept" means ES didn't sweep in
    # [i - time_tolerance, i + time_tolerance] range
    if time_tolerance > 0:
        # Expand ES sweep windows: if ES swept at bar j, mark j-tol..j+tol as swept
        es_swept_high_expanded = np.zeros(len(shared_idx), dtype=np.bool_)
        es_swept_low_expanded = np.zeros(len(shared_idx), dtype=np.bool_)

        # AUDIT FIX: only expand backward (non-negative offsets).
        # Negative offsets = future data (ES swept at bar i+1) = lookahead.
        for offset in range(0, time_tolerance + 1):
            shifted_high = np.roll(es_swept_high, offset)
            shifted_low = np.roll(es_swept_low, offset)
            # Fix edges
            if offset > 0:
                shifted_high[:offset] = False
                shifted_low[:offset] = False
            elif offset < 0:
                shifted_high[offset:] = False
                shifted_low[offset:] = False
            es_swept_high_expanded |= shifted_high
            es_swept_low_expanded |= shifted_low

        es_swept_high_for_smt = es_swept_high_expanded
        es_swept_low_for_smt = es_swept_low_expanded
    else:
        es_swept_high_for_smt = es_swept_high
        es_swept_low_for_smt = es_swept_low

    # Step 4: SMT divergence
    # Bull SMT: NQ swept low (fake breakdown) BUT ES did NOT sweep low
    smt_bull = nq_swept_low & ~es_swept_low_for_smt

    # Bear SMT: NQ swept high (fake breakout) BUT ES did NOT sweep high
    smt_bear = nq_swept_high & ~es_swept_high_for_smt

    # Step 5: shift(1) to prevent lookahead — SMT at bar i is only
    # knowable after bar i closes, so signal is available at bar i+1
    smt_bull = np.roll(smt_bull, 1)
    smt_bull[0] = False
    smt_bear = np.roll(smt_bear, 1)
    smt_bear[0] = False

    # Build output DataFrame — shift auxiliary columns by 1 to prevent lookahead
    def _shift1(arr):
        s = np.roll(arr, 1)
        s[0] = False
        return s

    out = pd.DataFrame(index=shared_idx)
    out["smt_bull"] = smt_bull
    out["smt_bear"] = smt_bear
    out["nq_swept_high"] = _shift1(nq_swept_high)
    out["nq_swept_low"] = _shift1(nq_swept_low)
    out["es_swept_high"] = _shift1(es_swept_high)
    out["es_swept_low"] = _shift1(es_swept_low)

    n_bull = int(smt_bull.sum())
    n_bear = int(smt_bear.sum())
    logger.info("SMT complete: %d bull divergences, %d bear divergences (%.1f%% / %.1f%% of bars)",
                n_bull, n_bear, n_bull / len(shared_idx) * 100, n_bear / len(shared_idx) * 100)

    return out
