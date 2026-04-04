"""
features/mtf_fvg.py -- Multi-timeframe FVG detection and alignment.

Detects FVGs on 1H and 4H, tracks state machine per timeframe,
projects active FVG features down to 5m bars for the engine.

CLAUDE.md references:
  - S3.1: HTF analysis (Daily -> 4H), FVGs as magnets/targets
  - S5: FVG state machine (untested -> tested_rejected -> invalidated -> IFVG)
  - S5.2: Two key PDA components: (a) displaced, (b) took liquidity in creation
  - S13: shift(1) all HTF features, ffill, no lookahead

Architecture:
  4H FVGs  --(detect + state track)--> per-4H-bar features --(shift(1)+ffill)--> 5m
  1H FVGs  --(detect + state track)--> per-1H-bar features --(shift(1)+ffill)--> 5m
  Combined --> htf_fvg_bullish_active, htf_fvg_bearish_active, distances, htf_bias
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from features.fvg import detect_fvg, FVGRecord, _update_fvg_status

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Per-bar HTF FVG features (computed on the HTF timeframe itself)
# ---------------------------------------------------------------------------

# Default: prune FVGs whose midpoint is > MAX_DIST_ATR_MULT * ATR(14) from
# current price.  On 1H (ATR~40-100), 5x = ~200-500 pts; on 4H (ATR~80-150),
# 5x = ~400-750 pts.  This keeps only FVGs within a realistic trading range
# (1-2 day reach) while pruning ancient FVGs from thousands of points away.
MAX_DIST_ATR_MULT: float = 5.0
ATR_PERIOD: int = 14

# DECISION-007: Quality filter defaults (overridden by params.yaml mtf_fvg section)
DEFAULT_MIN_FVG_SIZE_ATR: float = 0.8
DEFAULT_MIN_DISPLACEMENT_BODY_RATIO: float = 0.5
DEFAULT_MAX_FVG_AGE_BARS: int = 100

# Quality scoring defaults (FVG birth quality + time decay)
DEFAULT_SCORE_SIZE_WEIGHT: float = 0.40
DEFAULT_SCORE_DISP_WEIGHT: float = 0.35
DEFAULT_SCORE_LIQ_WEIGHT: float = 0.25
DEFAULT_SCORE_SIZE_ATR_CAP: float = 1.5  # gap_size / (atr * cap) capped at 1.0
DEFAULT_SCORE_HALF_LIFE_1H: int = 50    # bars on 1H
DEFAULT_SCORE_HALF_LIFE_4H: int = 30    # bars on 4H
DEFAULT_LIQ_LOOKBACK: int = 20          # bars to search for recent swing H/L


def _detect_swings_inline(
    highs: np.ndarray,
    lows: np.ndarray,
    left: int = 3,
    right: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Inline swing detection to avoid importing features.swing (circular deps).

    Returns two boolean arrays: (swing_high_mask, swing_low_mask).
    A swing high at bar i means high[i] > all highs in [i-left, i-1]
    AND high[i] > all highs in [i+1, i+right].  Analogous for swing lows.
    """
    n = len(highs)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        # Swing high
        is_sh = True
        for j in range(i - left, i):
            if highs[j] >= highs[i]:
                is_sh = False
                break
        if is_sh:
            for j in range(i + 1, i + 1 + right):
                if highs[j] >= highs[i]:
                    is_sh = False
                    break
        sh[i] = is_sh

        # Swing low
        is_sl = True
        for j in range(i - left, i):
            if lows[j] <= lows[i]:
                is_sl = False
                break
        if is_sl:
            for j in range(i + 1, i + 1 + right):
                if lows[j] <= lows[i]:
                    is_sl = False
                    break
        sl[i] = is_sl

    return sh, sl


def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> np.ndarray:
    """Compute ATR(period) as a numpy array, same length as df."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Simple moving average for ATR
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = atr[i - 1] + (tr[i] - tr[i - period]) / period

    # Fill leading NaN with the first valid ATR value
    first_valid = period - 1 if n >= period else n - 1
    if not np.isnan(atr[first_valid]):
        atr[:first_valid] = atr[first_valid]

    return atr


def _compute_htf_fvg_per_bar(
    df: pd.DataFrame,
    max_dist_atr_mult: float = MAX_DIST_ATR_MULT,
    min_fvg_size_atr: float = DEFAULT_MIN_FVG_SIZE_ATR,
    min_displacement_body_ratio: float = DEFAULT_MIN_DISPLACEMENT_BODY_RATIO,
    max_fvg_age_bars: int = DEFAULT_MAX_FVG_AGE_BARS,
    score_half_life: int | None = None,
    score_size_atr_cap: float = DEFAULT_SCORE_SIZE_ATR_CAP,
    score_weights: tuple[float, float, float] = (
        DEFAULT_SCORE_SIZE_WEIGHT,
        DEFAULT_SCORE_DISP_WEIGHT,
        DEFAULT_SCORE_LIQ_WEIGHT,
    ),
    liq_lookback: int = DEFAULT_LIQ_LOOKBACK,
) -> pd.DataFrame:
    """Compute per-bar FVG features on a single HTF DataFrame.

    Walks forward bar-by-bar, maintaining a list of active (untested or
    tested_rejected) FVGs.  For each bar, computes:

      - htf_bull_active (bool): any active bullish FVG exists
      - htf_bear_active (bool): any active bearish FVG exists
      - htf_nearest_bull_dist (float): signed distance from close to nearest
        active bullish FVG midpoint (positive = FVG below price)
      - htf_nearest_bear_dist (float): signed distance from close to nearest
        active bearish FVG midpoint (positive = FVG above price)
      - htf_num_bull (int): count of active bullish FVGs
      - htf_num_bear (int): count of active bearish FVGs
      - htf_bias (int): +1 if net bullish, -1 if net bearish, 0 if equal
      - htf_best_bull_score (float): max live_score among active bullish FVGs
      - htf_best_bear_score (float): max live_score among active bearish FVGs
      - htf_score_sum_bull (float): sum of live_scores of active bullish FVGs
      - htf_score_sum_bear (float): sum of live_scores of active bearish FVGs

    Quality scoring (per FVG at birth):
      birth_quality = w_size * size_score + w_disp * displacement_score + w_liq * liquidity_score
      live_score = birth_quality * exp(-age_bars / half_life)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame for a single HTF (e.g. 1H or 4H).
        Must have columns: open, high, low, close, volume, is_roll_date.
        Sorted ascending by DatetimeIndex (UTC).
    max_dist_atr_mult : float
        FVGs whose midpoint is further than this * ATR(14) from current
        close are pruned.
    min_fvg_size_atr : float
        DECISION-007 quality filter: minimum FVG gap size as multiple of ATR(14).
    min_displacement_body_ratio : float
        DECISION-007 quality filter: candle-2 body/range minimum.
    max_fvg_age_bars : int
        DECISION-007 quality filter: FVGs older than this auto-expire.
    score_half_life : int | None
        Half-life for exponential time decay of FVG quality score (bars).
        If None, uses DEFAULT_SCORE_HALF_LIFE_1H (50).
    score_size_atr_cap : float
        Cap for size_score normalization: size_score = min(1.0, gap / (atr * cap)).
    score_weights : tuple[float, float, float]
        (w_size, w_disp, w_liq) weights for composite birth_quality.
    liq_lookback : int
        Bars to look back for swing H/L when computing liquidity_score.

    Returns
    -------
    pd.DataFrame
        Same index as *df*, one row per HTF bar.
        Columns: htf_bull_active, htf_bear_active, htf_nearest_bull_dist,
                 htf_nearest_bear_dist, htf_num_bull, htf_num_bear, htf_bias,
                 htf_best_bull_score, htf_best_bear_score,
                 htf_score_sum_bull, htf_score_sum_bear.

    Notes
    -----
    * detect_fvg already applies shift(1) internally (FVG at candle-2 is
      visible starting at candle-3).  So the FVG birth at row i in fvg_df
      is already delayed by one bar.
    * The per-bar features here are computed on the HTF timeframe.  A second
      shift(1) + ffill is applied when projecting to 5m (in
      ``_align_htf_to_5m``), ensuring the 5m bar only sees HTF features
      from the *previous* completed HTF bar.
    * Distance pruning: FVGs whose midpoint exceeds max_dist_atr_mult * ATR(14)
      from current close are dropped each bar.
    * Quality filter (DECISION-007): At FVG birth, filter out FVGs where
      gap_size < min_fvg_size_atr * ATR(14) OR candle-2 body/range <
      min_displacement_body_ratio.  This removes ~84% of noise FVGs on 1H.
    * Quality scoring: each admitted FVG gets a birth_quality score in [0, 1]
      based on size, displacement, and liquidity sweep.  The live_score
      decays exponentially with age.
    """
    fvg_df = detect_fvg(df)
    atr = _compute_atr(df)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    n = len(df)
    index = df.index

    bull_mask = fvg_df["fvg_bull"].values
    bear_mask = fvg_df["fvg_bear"].values

    # Pre-compute candle body/range for quality filter on candle-2.
    # detect_fvg shifts by 1, so the FVG visible at bar i was created at
    # candle-2 = bar i-1.  We compute body/range for every bar upfront.
    body = np.abs(close - open_)
    bar_range = high - low
    safe_range = np.where(bar_range > 0, bar_range, np.nan)
    body_ratio_arr = body / safe_range  # NaN where bar_range == 0

    # --- Quality scoring: pre-compute swing points for liquidity_score ---
    swing_high_mask, swing_low_mask = _detect_swings_inline(high, low, left=3, right=1)
    # Pre-compute forward-filled swing high/low prices for lookback window
    swing_high_prices = np.where(swing_high_mask, high, np.nan)
    swing_low_prices = np.where(swing_low_mask, low, np.nan)

    half_life = score_half_life if score_half_life is not None else DEFAULT_SCORE_HALF_LIFE_1H
    # Decay constant: exp(-age / half_life) = 0.5 when age = half_life
    decay_lambda = np.log(2.0) / half_life if half_life > 0 else 0.0
    w_size, w_disp, w_liq = score_weights

    # Track quality filter stats for logging
    n_fvg_born = 0
    n_fvg_quality_rejected = 0

    # Output arrays
    out_bull_active = np.zeros(n, dtype=bool)
    out_bear_active = np.zeros(n, dtype=bool)
    out_nearest_bull_dist = np.full(n, np.nan)
    out_nearest_bear_dist = np.full(n, np.nan)
    out_num_bull = np.zeros(n, dtype=np.int32)
    out_num_bear = np.zeros(n, dtype=np.int32)
    out_bias = np.zeros(n, dtype=np.int8)
    out_best_bull_score = np.full(n, np.nan)
    out_best_bear_score = np.full(n, np.nan)
    out_score_sum_bull = np.full(n, np.nan)
    out_score_sum_bear = np.full(n, np.nan)

    # Each active FVG: (record, birth_bar_index, birth_quality)
    active_bull: list[tuple[FVGRecord, int, float]] = []
    active_bear: list[tuple[FVGRecord, int, float]] = []

    def _compute_birth_quality(
        candle2_idx: int,
        gap_size: float,
        atr_val: float,
        c2_body_ratio: float,
        direction: str,
    ) -> float:
        """Compute birth quality score for a new FVG.

        Components:
          size_score: min(1.0, gap_size / (atr * score_size_atr_cap))
          displacement_score: candle-2 body/range ratio (already computed)
          liquidity_score: 1.0 if candle-2 swept a recent swing H/L, else 0.0
        """
        # Size score
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(gap_size):
            s_score = 0.5  # fallback during warmup
        else:
            s_score = min(1.0, gap_size / (atr_val * score_size_atr_cap))

        # Displacement score (candle-2 body/range)
        d_score = float(c2_body_ratio) if not np.isnan(c2_body_ratio) else 0.5

        # Liquidity score: did candle-2 sweep a recent swing H/L?
        l_score = 0.0
        lookback_start = max(0, candle2_idx - liq_lookback)
        if direction == "bull":
            # Bullish FVG: candle-2 is a large up candle.
            # "Took liquidity" = candle2 low swept below a recent swing low
            recent_sl = swing_low_prices[lookback_start:candle2_idx]
            valid_sl = recent_sl[~np.isnan(recent_sl)]
            if len(valid_sl) > 0 and low[candle2_idx] < valid_sl.min():
                l_score = 1.0
        else:
            # Bearish FVG: candle-2 is a large down candle.
            # "Took liquidity" = candle2 high swept above a recent swing high
            recent_sh = swing_high_prices[lookback_start:candle2_idx]
            valid_sh = recent_sh[~np.isnan(recent_sh)]
            if len(valid_sh) > 0 and high[candle2_idx] > valid_sh.max():
                l_score = 1.0

        return w_size * s_score + w_disp * d_score + w_liq * l_score

    for i in range(n):
        c = close[i]
        max_dist = max_dist_atr_mult * atr[i] if not np.isnan(atr[i]) else np.inf

        # --- Birth: new FVGs becoming visible at bar i ---
        # Quality filter (DECISION-007): check gap size vs ATR and candle-2
        # body/range ratio BEFORE admitting the FVG into the active pool.
        # The FVG visible at bar i was created at candle-2 = bar i-1.
        if bull_mask[i]:
            n_fvg_born += 1
            candle2_idx = i - 1 if i > 0 else i
            candle2_time = index[candle2_idx]
            gap_size = fvg_df["fvg_size"].iat[i]
            atr_at_birth = atr[candle2_idx]
            c2_body_ratio = body_ratio_arr[candle2_idx]

            # Quality gate: size >= min_fvg_size_atr * ATR AND
            #               candle-2 body/range >= min_displacement_body_ratio
            size_ok = (
                np.isnan(atr_at_birth)
                or np.isnan(gap_size)
                or gap_size >= min_fvg_size_atr * atr_at_birth
            )
            disp_ok = (
                np.isnan(c2_body_ratio)
                or c2_body_ratio >= min_displacement_body_ratio
            )
            if size_ok and disp_ok:
                rec = FVGRecord(
                    time=candle2_time,
                    direction="bull",
                    top=fvg_df["fvg_bull_top"].iat[i],
                    bottom=fvg_df["fvg_bull_bottom"].iat[i],
                    size=gap_size,
                )
                bq = _compute_birth_quality(
                    candle2_idx, gap_size, atr_at_birth, c2_body_ratio, "bull"
                )
                active_bull.append((rec, i, bq))
            else:
                n_fvg_quality_rejected += 1

        if bear_mask[i]:
            n_fvg_born += 1
            candle2_idx = i - 1 if i > 0 else i
            candle2_time = index[candle2_idx]
            gap_size = fvg_df["fvg_size"].iat[i]
            atr_at_birth = atr[candle2_idx]
            c2_body_ratio = body_ratio_arr[candle2_idx]

            size_ok = (
                np.isnan(atr_at_birth)
                or np.isnan(gap_size)
                or gap_size >= min_fvg_size_atr * atr_at_birth
            )
            disp_ok = (
                np.isnan(c2_body_ratio)
                or c2_body_ratio >= min_displacement_body_ratio
            )
            if size_ok and disp_ok:
                rec = FVGRecord(
                    time=candle2_time,
                    direction="bear",
                    top=fvg_df["fvg_bear_top"].iat[i],
                    bottom=fvg_df["fvg_bear_bottom"].iat[i],
                    size=gap_size,
                )
                bq = _compute_birth_quality(
                    candle2_idx, gap_size, atr_at_birth, c2_body_ratio, "bear"
                )
                active_bear.append((rec, i, bq))
            else:
                n_fvg_quality_rejected += 1

        # --- Update active FVGs against this bar ---
        # Also apply age-based expiry (DECISION-007)
        surviving_bull: list[tuple[FVGRecord, int, float]] = []
        for rec, birth_i, bq in active_bull:
            # Age pruning: drop FVGs older than max_fvg_age_bars
            if (i - birth_i) > max_fvg_age_bars:
                continue

            # Distance pruning: drop FVGs too far from current price
            mid = (rec.top + rec.bottom) / 2.0
            if abs(c - mid) > max_dist:
                continue

            new_status = _update_fvg_status(rec, high[i], low[i], close[i])
            if new_status == "invalidated":
                rec.status = "invalidated"
                # Spawn IFVG: when a bullish FVG is invalidated, it becomes
                # bearish IFVG (former support -> resistance), and vice versa.
                # IFVGs inherit the parent's birth_quality.
                if not rec.is_ifvg:
                    ifvg = FVGRecord(
                        time=index[i],
                        direction="bear",  # inverted
                        top=rec.top,
                        bottom=rec.bottom,
                        size=rec.size,
                        status="untested",
                        is_ifvg=True,
                        ifvg_direction="bear",
                    )
                    active_bear.append((ifvg, i, bq))  # inherit birth_quality
            else:
                rec.status = new_status
                surviving_bull.append((rec, birth_i, bq))
        active_bull = surviving_bull

        surviving_bear: list[tuple[FVGRecord, int, float]] = []
        for rec, birth_i, bq in active_bear:
            if (i - birth_i) > max_fvg_age_bars:
                continue

            mid = (rec.top + rec.bottom) / 2.0
            if abs(c - mid) > max_dist:
                continue

            new_status = _update_fvg_status(rec, high[i], low[i], close[i])
            if new_status == "invalidated":
                rec.status = "invalidated"
                if not rec.is_ifvg:
                    ifvg = FVGRecord(
                        time=index[i],
                        direction="bull",  # inverted
                        top=rec.top,
                        bottom=rec.bottom,
                        size=rec.size,
                        status="untested",
                        is_ifvg=True,
                        ifvg_direction="bull",
                    )
                    active_bull.append((ifvg, i, bq))  # inherit birth_quality
            else:
                rec.status = new_status
                surviving_bear.append((rec, birth_i, bq))
        active_bear = surviving_bear

        # --- Compute features for this bar ---
        n_bull = len(active_bull)
        n_bear = len(active_bear)

        out_bull_active[i] = n_bull > 0
        out_bear_active[i] = n_bear > 0
        out_num_bull[i] = n_bull
        out_num_bear[i] = n_bear

        # Nearest bullish FVG distance (positive = FVG below current price)
        best_bull_dist = np.inf
        for rec, _, _ in active_bull:
            mid = (rec.top + rec.bottom) / 2.0
            dist = c - mid  # positive means price is above the FVG
            if abs(dist) < abs(best_bull_dist):
                best_bull_dist = dist
        if best_bull_dist != np.inf:
            out_nearest_bull_dist[i] = best_bull_dist

        # Nearest bearish FVG distance (positive = FVG above current price)
        best_bear_dist = np.inf
        for rec, _, _ in active_bear:
            mid = (rec.top + rec.bottom) / 2.0
            dist = mid - c  # positive means FVG is above price
            if abs(dist) < abs(best_bear_dist):
                best_bear_dist = dist
        if best_bear_dist != np.inf:
            out_nearest_bear_dist[i] = best_bear_dist

        # Bias: compare bull vs bear active FVGs
        if n_bull > n_bear:
            out_bias[i] = 1
        elif n_bear > n_bull:
            out_bias[i] = -1
        else:
            out_bias[i] = 0

        # --- Quality scores: live_score = birth_quality * exp(-age / half_life) ---
        # Bullish FVGs
        if n_bull > 0:
            best_score = 0.0
            sum_score = 0.0
            for _, birth_i, bq in active_bull:
                age = i - birth_i
                live = bq * np.exp(-decay_lambda * age)
                sum_score += live
                if live > best_score:
                    best_score = live
            out_best_bull_score[i] = best_score
            out_score_sum_bull[i] = sum_score

        # Bearish FVGs
        if n_bear > 0:
            best_score = 0.0
            sum_score = 0.0
            for _, birth_i, bq in active_bear:
                age = i - birth_i
                live = bq * np.exp(-decay_lambda * age)
                sum_score += live
                if live > best_score:
                    best_score = live
            out_best_bear_score[i] = best_score
            out_score_sum_bear[i] = sum_score

    result = pd.DataFrame(
        {
            "htf_bull_active": out_bull_active,
            "htf_bear_active": out_bear_active,
            "htf_nearest_bull_dist": out_nearest_bull_dist,
            "htf_nearest_bear_dist": out_nearest_bear_dist,
            "htf_num_bull": out_num_bull,
            "htf_num_bear": out_num_bear,
            "htf_bias": out_bias,
            "htf_best_bull_score": out_best_bull_score,
            "htf_best_bear_score": out_best_bear_score,
            "htf_score_sum_bull": out_score_sum_bull,
            "htf_score_sum_bear": out_score_sum_bear,
        },
        index=df.index,
    )

    reject_pct = (100.0 * n_fvg_quality_rejected / n_fvg_born) if n_fvg_born > 0 else 0.0
    # Score stats for logging
    valid_bull_scores = out_best_bull_score[~np.isnan(out_best_bull_score)]
    valid_bear_scores = out_best_bear_score[~np.isnan(out_best_bear_score)]
    logger.info(
        "_compute_htf_fvg_per_bar: %d bars, bull_active=%.1f%%, "
        "bear_active=%.1f%%, mean_bias=%.2f, "
        "quality_filter: %d/%d FVGs rejected (%.1f%%), "
        "bull_score: mean=%.3f (n=%d), bear_score: mean=%.3f (n=%d)",
        n,
        out_bull_active.mean() * 100,
        out_bear_active.mean() * 100,
        out_bias.astype(float).mean(),
        n_fvg_quality_rejected, n_fvg_born, reject_pct,
        valid_bull_scores.mean() if len(valid_bull_scores) > 0 else 0.0,
        len(valid_bull_scores),
        valid_bear_scores.mean() if len(valid_bear_scores) > 0 else 0.0,
        len(valid_bear_scores),
    )

    return result


# ---------------------------------------------------------------------------
# 2. Align HTF per-bar features down to 5m bars (shift(1) + ffill)
# ---------------------------------------------------------------------------

def _align_htf_to_5m(
    htf_features: pd.DataFrame,
    df_5m: pd.DataFrame,
    tf_prefix: str,
) -> pd.DataFrame:
    """Project HTF per-bar features onto the 5m index.

    Applies:
      1. shift(1) -- a completed HTF bar's features become available only
         at the NEXT HTF bar's timestamp.  This means the 5m bars within
         HTF bar T see the features from HTF bar T-1 (the last fully closed bar).
      2. reindex with ffill -- forward-fill the shifted values into every
         5m bar until the next HTF bar closes.

    Parameters
    ----------
    htf_features : pd.DataFrame
        Output of ``_compute_htf_fvg_per_bar`` for one timeframe.
    df_5m : pd.DataFrame
        The 5m base DataFrame (used only for its index).
    tf_prefix : str
        Prefix for output columns, e.g. "4H" or "1H".

    Returns
    -------
    pd.DataFrame
        Aligned to df_5m.index with columns prefixed by tf_prefix.
    """
    # Step 1: shift(1) -- delay by one HTF bar
    shifted = htf_features.shift(1)

    # Step 2: reindex to 5m with ffill
    aligned = shifted.reindex(df_5m.index, method="ffill")

    # Prefix column names
    rename_map = {c: f"{tf_prefix}_{c}" for c in aligned.columns}
    aligned = aligned.rename(columns=rename_map)

    n_nan = aligned.isna().all(axis=1).sum()
    logger.info(
        "_align_htf_to_5m [%s]: %d HTF bars -> %d 5m bars "
        "(%d leading NaN rows)",
        tf_prefix, len(htf_features), len(df_5m), n_nan,
    )

    return aligned


# ---------------------------------------------------------------------------
# 3. compute_htf_fvg_features -- main public API
# ---------------------------------------------------------------------------

def compute_htf_fvg_features(
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    params: dict | None = None,
) -> pd.DataFrame:
    """Build multi-timeframe FVG features aligned to the 5m index.

    This is the main entry point for TASK-018.  It detects FVGs on 1H and
    4H, tracks their state machines, and projects features down to 5m bars.

    Parameters
    ----------
    df_5m : pd.DataFrame
        5m OHLCV data (target index).
    df_1h : pd.DataFrame
        1H OHLCV data.
    df_4h : pd.DataFrame
        4H OHLCV data.
    params : dict | None
        Full params dict (from config/params.yaml).  If None, uses module
        defaults.  Quality filter params are read from params["mtf_fvg"].

    Returns
    -------
    pd.DataFrame
        Aligned to df_5m.index with columns:

        Per-timeframe (prefixed "1H_" and "4H_"):
          - {tf}_htf_bull_active (bool)
          - {tf}_htf_bear_active (bool)
          - {tf}_htf_nearest_bull_dist (float)
          - {tf}_htf_nearest_bear_dist (float)
          - {tf}_htf_num_bull (int)
          - {tf}_htf_num_bear (int)
          - {tf}_htf_bias (int)
          - {tf}_htf_best_bull_score (float)
          - {tf}_htf_best_bear_score (float)
          - {tf}_htf_score_sum_bull (float)
          - {tf}_htf_score_sum_bear (float)

        Combined:
          - htf_fvg_bullish_active (bool): any bullish FVG active on 1H OR 4H
          - htf_fvg_bearish_active (bool): any bearish FVG active on 1H OR 4H
          - htf_fvg_nearest_bull_dist (float): min distance to nearest bull FVG
            across 1H and 4H
          - htf_fvg_nearest_bear_dist (float): min distance to nearest bear FVG
            across 1H and 4H
          - htf_bias (int): combined bias from both timeframes
            (+1=bullish, -1=bearish, 0=neutral)
          - htf_best_bull_score (float): max live_score across 1H+4H bull FVGs
          - htf_best_bear_score (float): max live_score across 1H+4H bear FVGs
          - htf_score_sum_bull (float): sum of live_scores across 1H+4H bull FVGs
          - htf_score_sum_bear (float): sum of live_scores across 1H+4H bear FVGs
    """
    logger.info(
        "compute_htf_fvg_features: 5m=%d bars, 1H=%d bars, 4H=%d bars",
        len(df_5m), len(df_1h), len(df_4h),
    )

    # Read quality filter params from config (DECISION-007)
    mtf_cfg = (params or {}).get("mtf_fvg", {})
    min_fvg_size_atr = mtf_cfg.get("min_fvg_size_atr", DEFAULT_MIN_FVG_SIZE_ATR)
    min_displacement_body_ratio = mtf_cfg.get(
        "min_displacement_body_ratio", DEFAULT_MIN_DISPLACEMENT_BODY_RATIO
    )
    max_fvg_age_bars = mtf_cfg.get("max_fvg_age_bars", DEFAULT_MAX_FVG_AGE_BARS)

    # Read quality scoring params
    score_cfg = mtf_cfg.get("score", {})
    score_size_atr_cap = score_cfg.get("size_atr_cap", DEFAULT_SCORE_SIZE_ATR_CAP)
    score_weights = (
        score_cfg.get("w_size", DEFAULT_SCORE_SIZE_WEIGHT),
        score_cfg.get("w_disp", DEFAULT_SCORE_DISP_WEIGHT),
        score_cfg.get("w_liq", DEFAULT_SCORE_LIQ_WEIGHT),
    )
    half_life_1h = score_cfg.get("half_life_1h", DEFAULT_SCORE_HALF_LIFE_1H)
    half_life_4h = score_cfg.get("half_life_4h", DEFAULT_SCORE_HALF_LIFE_4H)
    liq_lookback = score_cfg.get("liq_lookback", DEFAULT_LIQ_LOOKBACK)

    logger.info(
        "Quality filter params: min_fvg_size_atr=%.2f, "
        "min_displacement_body_ratio=%.2f, max_fvg_age_bars=%d",
        min_fvg_size_atr, min_displacement_body_ratio, max_fvg_age_bars,
    )
    logger.info(
        "Quality scoring params: size_atr_cap=%.2f, weights=(%.2f,%.2f,%.2f), "
        "half_life_1h=%d, half_life_4h=%d, liq_lookback=%d",
        score_size_atr_cap, *score_weights, half_life_1h, half_life_4h, liq_lookback,
    )

    # --- Compute per-bar features on each HTF ---
    logger.info("Computing 1H FVG features...")
    feats_1h = _compute_htf_fvg_per_bar(
        df_1h,
        min_fvg_size_atr=min_fvg_size_atr,
        min_displacement_body_ratio=min_displacement_body_ratio,
        max_fvg_age_bars=max_fvg_age_bars,
        score_half_life=half_life_1h,
        score_size_atr_cap=score_size_atr_cap,
        score_weights=score_weights,
        liq_lookback=liq_lookback,
    )

    logger.info("Computing 4H FVG features...")
    feats_4h = _compute_htf_fvg_per_bar(
        df_4h,
        min_fvg_size_atr=min_fvg_size_atr,
        min_displacement_body_ratio=min_displacement_body_ratio,
        max_fvg_age_bars=max_fvg_age_bars,
        score_half_life=half_life_4h,
        score_size_atr_cap=score_size_atr_cap,
        score_weights=score_weights,
        liq_lookback=liq_lookback,
    )

    # --- Align to 5m with shift(1) + ffill ---
    aligned_1h = _align_htf_to_5m(feats_1h, df_5m, "1H")
    aligned_4h = _align_htf_to_5m(feats_4h, df_5m, "4H")

    # --- Combine into a single DataFrame ---
    result = pd.concat([aligned_1h, aligned_4h], axis=1)

    # --- Compute combined features ---
    # Boolean active: any HTF has an active FVG
    bull_1h = result["1H_htf_bull_active"].fillna(False).astype(bool)
    bull_4h = result["4H_htf_bull_active"].fillna(False).astype(bool)
    bear_1h = result["1H_htf_bear_active"].fillna(False).astype(bool)
    bear_4h = result["4H_htf_bear_active"].fillna(False).astype(bool)

    result["htf_fvg_bullish_active"] = bull_1h | bull_4h
    result["htf_fvg_bearish_active"] = bear_1h | bear_4h

    # Nearest distance: take the closer of 1H and 4H
    # For bull dist, smaller absolute value = closer magnet
    bull_dist_1h = result["1H_htf_nearest_bull_dist"]
    bull_dist_4h = result["4H_htf_nearest_bull_dist"]
    bear_dist_1h = result["1H_htf_nearest_bear_dist"]
    bear_dist_4h = result["4H_htf_nearest_bear_dist"]

    # Use abs for comparison, but keep the signed value
    result["htf_fvg_nearest_bull_dist"] = _pick_nearest(bull_dist_1h, bull_dist_4h)
    result["htf_fvg_nearest_bear_dist"] = _pick_nearest(bear_dist_1h, bear_dist_4h)

    # Combined bias: weight 4H more heavily (4H = 2x weight of 1H)
    # Per CLAUDE.md: "Higher TF FVGs are inherently higher quality (4H > 1H)"
    bias_1h = result["1H_htf_bias"].fillna(0).astype(float)
    bias_4h = result["4H_htf_bias"].fillna(0).astype(float)
    raw_bias = bias_1h + 2.0 * bias_4h  # 4H counts double
    result["htf_bias"] = np.sign(raw_bias).astype(np.int8)

    # --- Combined quality scores: max across 1H and 4H ---
    # best_bull_score: take the max of 1H and 4H best scores
    bull_score_1h = result["1H_htf_best_bull_score"]
    bull_score_4h = result["4H_htf_best_bull_score"]
    bear_score_1h = result["1H_htf_best_bear_score"]
    bear_score_4h = result["4H_htf_best_bear_score"]

    # For best: take max (both NaN -> NaN, one NaN -> use other)
    result["htf_best_bull_score"] = np.fmax(
        bull_score_1h.values, bull_score_4h.values
    )
    result["htf_best_bear_score"] = np.fmax(
        bear_score_1h.values, bear_score_4h.values
    )

    # For sum: add 1H + 4H (NaN + x -> x via nansum approach)
    sum_bull_1h = result["1H_htf_score_sum_bull"].fillna(0.0)
    sum_bull_4h = result["4H_htf_score_sum_bull"].fillna(0.0)
    sum_bear_1h = result["1H_htf_score_sum_bear"].fillna(0.0)
    sum_bear_4h = result["4H_htf_score_sum_bear"].fillna(0.0)

    # Only produce NaN if BOTH are NaN (no active FVGs on either TF)
    both_bull_nan = result["1H_htf_score_sum_bull"].isna() & result["4H_htf_score_sum_bull"].isna()
    both_bear_nan = result["1H_htf_score_sum_bear"].isna() & result["4H_htf_score_sum_bear"].isna()

    result["htf_score_sum_bull"] = sum_bull_1h + sum_bull_4h
    result.loc[both_bull_nan, "htf_score_sum_bull"] = np.nan
    result["htf_score_sum_bear"] = sum_bear_1h + sum_bear_4h
    result.loc[both_bear_nan, "htf_score_sum_bear"] = np.nan

    # Log summary
    _log_summary(result)

    return result


def _pick_nearest(
    dist_a: pd.Series,
    dist_b: pd.Series,
) -> pd.Series:
    """Pick the distance with smaller absolute value from two Series.

    Where both are NaN, result is NaN.
    Where one is NaN, use the other.
    Where both valid, pick the one with smaller abs value.
    """
    abs_a = dist_a.abs()
    abs_b = dist_b.abs()

    # Start with a, override with b where b is closer or a is NaN
    result = dist_a.copy()
    use_b = (abs_b < abs_a) | dist_a.isna()
    result[use_b] = dist_b[use_b]

    return result


def _log_summary(result: pd.DataFrame) -> None:
    """Log a summary of the combined features."""
    n = len(result)
    # Skip leading NaN rows for percentages
    valid_mask = result["htf_fvg_bullish_active"].notna()
    n_valid = valid_mask.sum()
    if n_valid == 0:
        logger.warning("compute_htf_fvg_features: all rows are NaN")
        return

    bull_pct = result.loc[valid_mask, "htf_fvg_bullish_active"].mean() * 100
    bear_pct = result.loc[valid_mask, "htf_fvg_bearish_active"].mean() * 100
    bias_mean = result.loc[valid_mask, "htf_bias"].mean()

    logger.info(
        "compute_htf_fvg_features: DONE. %d total bars, %d valid. "
        "bull_active=%.1f%%, bear_active=%.1f%%, mean_bias=%.2f",
        n, n_valid, bull_pct, bear_pct, bias_mean,
    )


# ---------------------------------------------------------------------------
# __main__ -- quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        stream=sys.stdout,
    )

    logger.info("=" * 70)
    logger.info("MTF FVG Features -- Smoke Test")
    logger.info("=" * 70)

    df_5m = pd.read_parquet("data/NQ_5m_10yr.parquet")
    df_1h = pd.read_parquet("data/NQ_1H_10yr.parquet")
    df_4h = pd.read_parquet("data/NQ_4H_10yr.parquet")

    # Use last ~2 months for a quick test
    LAST_N_5M = 20_000  # ~69 trading days
    df_5m_slice = df_5m.iloc[-LAST_N_5M:]
    start_dt = df_5m_slice.index[0] - pd.Timedelta(days=60)
    df_1h_slice = df_1h.loc[df_1h.index >= start_dt]
    df_4h_slice = df_4h.loc[df_4h.index >= start_dt]

    logger.info(
        "5m: %d bars [%s -> %s]",
        len(df_5m_slice), df_5m_slice.index[0], df_5m_slice.index[-1],
    )
    logger.info(
        "1H: %d bars [%s -> %s]",
        len(df_1h_slice), df_1h_slice.index[0], df_1h_slice.index[-1],
    )
    logger.info(
        "4H: %d bars [%s -> %s]",
        len(df_4h_slice), df_4h_slice.index[0], df_4h_slice.index[-1],
    )

    result = compute_htf_fvg_features(df_5m_slice, df_1h_slice, df_4h_slice)

    print(f"\nResult shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")
    print(f"\nHTF bullish active: {result['htf_fvg_bullish_active'].sum()} bars "
          f"({result['htf_fvg_bullish_active'].mean()*100:.1f}%)")
    print(f"HTF bearish active: {result['htf_fvg_bearish_active'].sum()} bars "
          f"({result['htf_fvg_bearish_active'].mean()*100:.1f}%)")
    print(f"HTF bias +1: {(result['htf_bias'] == 1).sum()}, "
          f"-1: {(result['htf_bias'] == -1).sum()}, "
          f"0: {(result['htf_bias'] == 0).sum()}")
    print(f"\nSample (last 10 rows):")
    print(result.tail(10).to_string())
