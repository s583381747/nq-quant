"""
Swing point detection — fractal highs and lows.

Identifies structural swing highs and swing lows using a left/right bar
fractal method.  Parameters (left_bars, right_bars) are read from
config/params.yaml (section: swing).

NOTE:  `right_bars` introduces a look-ahead of exactly `right_bars` bars.
This is acceptable because swing points are structural reference levels
(support / resistance / liquidity targets), not trade entry signals.
Downstream consumers must account for the confirmation delay — a swing
point at bar *i* is only *confirmed* at bar *i + right_bars*.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter loading
# ---------------------------------------------------------------------------

_PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "params.yaml"


def _load_swing_params(params_path: Path = _PARAMS_PATH) -> dict:
    """Load swing parameters from the centralised params.yaml.

    Returns
    -------
    dict
        Keys: ``left_bars`` (int), ``right_bars`` (int).
    """
    with open(params_path, "r") as f:
        cfg = yaml.safe_load(f)
    swing_cfg = cfg["swing"]
    return {
        "left_bars": int(swing_cfg["left_bars"]),
        "right_bars": int(swing_cfg["right_bars"]),
    }


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_swing_highs(highs: pd.Series, left: int, right: int) -> pd.Series:
    """Detect fractal swing highs (vectorised).

    A bar *i* is a swing high when its high is **strictly greater than**
    every high in the ``left`` bars before it **and** every high in the
    ``right`` bars after it.

    Parameters
    ----------
    highs : pd.Series
        Series of high prices (must be numeric, same index as the
        source DataFrame).
    left : int
        Number of bars to look back (must be >= 1).
    right : int
        Number of bars to look forward (must be >= 1).
        Introduces a confirmation delay of ``right`` bars.

    Returns
    -------
    pd.Series
        Boolean Series aligned to *highs* — ``True`` at confirmed
        swing-high bars, ``False`` everywhere else.
    """
    h = highs.values.astype(np.float64)
    n = len(h)

    # Left condition: h[i] > max(h[i-left:i]) for every i
    # Rolling max over window=left, then shift forward by 1 so that
    # left_max[i] = max(h[i-left], ..., h[i-1]).
    left_max = pd.Series(h).rolling(window=left, min_periods=left).max().shift(1).values

    # Right condition: h[i] > max(h[i+1:i+1+right])
    # Reverse the array, compute rolling max, reverse back, shift.
    h_rev = h[::-1]
    right_max_rev = pd.Series(h_rev).rolling(window=right, min_periods=right).max().shift(1).values
    right_max = right_max_rev[::-1]

    result = (h > left_max) & (h > right_max)

    # Edge handling: first `left` bars and last `right` bars cannot be swings
    result[:left] = False
    result[n - right:] = False

    # NaN from rolling produces False via comparison, but be explicit
    result = np.where(np.isnan(left_max) | np.isnan(right_max), False, result)

    out = pd.Series(result.astype(bool), index=highs.index, name="swing_high")
    logger.debug(
        "detect_swing_highs  left=%d  right=%d  found=%d / %d bars",
        left, right, int(out.sum()), n,
    )
    return out


def detect_swing_lows(lows: pd.Series, left: int, right: int) -> pd.Series:
    """Detect fractal swing lows (vectorised).

    A bar *i* is a swing low when its low is **strictly less than**
    every low in the ``left`` bars before it **and** every low in the
    ``right`` bars after it.

    Parameters
    ----------
    lows : pd.Series
        Series of low prices.
    left : int
        Number of bars to look back (must be >= 1).
    right : int
        Number of bars to look forward (must be >= 1).

    Returns
    -------
    pd.Series
        Boolean Series — ``True`` at confirmed swing-low bars.
    """
    lo = lows.values.astype(np.float64)
    n = len(lo)

    # Left condition: lo[i] < min(lo[i-left:i])
    left_min = pd.Series(lo).rolling(window=left, min_periods=left).min().shift(1).values

    # Right condition: lo[i] < min(lo[i+1:i+1+right])
    lo_rev = lo[::-1]
    right_min_rev = pd.Series(lo_rev).rolling(window=right, min_periods=right).min().shift(1).values
    right_min = right_min_rev[::-1]

    result = (lo < left_min) & (lo < right_min)

    # Edge handling
    result[:left] = False
    result[n - right:] = False
    result = np.where(np.isnan(left_min) | np.isnan(right_min), False, result)

    out = pd.Series(result.astype(bool), index=lows.index, name="swing_low")
    logger.debug(
        "detect_swing_lows  left=%d  right=%d  found=%d / %d bars",
        left, right, int(out.sum()), n,
    )
    return out


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def compute_swing_levels(
    df: pd.DataFrame,
    params: dict | None = None,
) -> pd.DataFrame:
    """Compute swing-high/low flags and forward-filled price levels.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``high`` and ``low``.
    params : dict, optional
        Must contain ``left_bars`` and ``right_bars`` keys (ints).
        If *None*, parameters are loaded from ``config/params.yaml``.

    Returns
    -------
    pd.DataFrame
        Columns added (same index as *df*):

        - ``swing_high``        — bool, True at swing high bars
        - ``swing_low``         — bool, True at swing low bars
        - ``swing_high_price``  — float, forward-filled most recent swing
          high price (NaN until the first swing high)
        - ``swing_low_price``   — float, forward-filled most recent swing
          low price (NaN until the first swing low)
    """
    if params is None:
        params = _load_swing_params()

    left = params["left_bars"]
    right = params["right_bars"]

    logger.info(
        "Computing swing levels  left_bars=%d  right_bars=%d  rows=%s",
        left, right, f"{len(df):,}",
    )

    sh = detect_swing_highs(df["high"], left, right)
    sl = detect_swing_lows(df["low"], left, right)

    out = pd.DataFrame(index=df.index)
    out["swing_high"] = sh
    out["swing_low"] = sl

    # Forward-filled price levels
    out["swing_high_price"] = np.where(sh, df["high"].values, np.nan)
    out["swing_high_price"] = out["swing_high_price"].ffill()

    out["swing_low_price"] = np.where(sl, df["low"].values, np.nan)
    out["swing_low_price"] = out["swing_low_price"].ffill()

    n_sh = int(sh.sum())
    n_sl = int(sl.sum())
    logger.info(
        "Swing detection complete — %s swing highs, %s swing lows",
        f"{n_sh:,}", f"{n_sl:,}",
    )

    return out


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    # 1. Load 1m data
    data_path = Path(__file__).resolve().parent.parent / "data" / "NQ_1min.parquet"
    logger.info("Loading data from %s", data_path)
    df = pd.read_parquet(data_path, engine="pyarrow")
    logger.info("Loaded %s bars  (%s → %s)", f"{len(df):,}", df.index.min(), df.index.max())

    # 2. Compute swing levels
    params = _load_swing_params()
    result = compute_swing_levels(df, params)

    # 3. Summary
    n_sh = int(result["swing_high"].sum())
    n_sl = int(result["swing_low"].sum())
    total_bars = len(df)

    print("\n" + "=" * 60)
    print("SWING POINT DETECTION — SUMMARY")
    print("=" * 60)
    print(f"Total bars:        {total_bars:,}")
    print(f"Params:            left_bars={params['left_bars']}, right_bars={params['right_bars']}")
    print(f"Swing highs found: {n_sh:,}  ({n_sh / total_bars * 100:.2f}%)")
    print(f"Swing lows found:  {n_sl:,}  ({n_sl / total_bars * 100:.2f}%)")
    print(f"Avg bars between swing highs: {total_bars / max(n_sh, 1):.1f}")
    print(f"Avg bars between swing lows:  {total_bars / max(n_sl, 1):.1f}")

    # 4. Last 10 swing highs and swing lows
    last_sh = df.loc[result["swing_high"]].tail(10)
    last_sl = df.loc[result["swing_low"]].tail(10)

    print("\n--- Last 10 Swing Highs ---")
    for ts, row in last_sh.iterrows():
        print(f"  {ts}  high={row['high']:.2f}")

    print("\n--- Last 10 Swing Lows ---")
    for ts, row in last_sl.iterrows():
        print(f"  {ts}  low={row['low']:.2f}")

    # 5. Sanity checks
    print("\n--- Sanity Checks ---")
    # Swing points should be a small fraction of total bars
    sh_pct = n_sh / total_bars * 100
    sl_pct = n_sl / total_bars * 100
    ok = True
    if sh_pct > 10:
        print(f"  WARNING: swing highs are {sh_pct:.1f}% of bars — too many, parameters may be too loose")
        ok = False
    elif sh_pct < 0.1:
        print(f"  WARNING: swing highs are {sh_pct:.2f}% of bars — too few, parameters may be too tight")
        ok = False
    else:
        print(f"  Swing high density {sh_pct:.2f}% — OK")

    if sl_pct > 10:
        print(f"  WARNING: swing lows are {sl_pct:.1f}% of bars — too many, parameters may be too loose")
        ok = False
    elif sl_pct < 0.1:
        print(f"  WARNING: swing lows are {sl_pct:.2f}% of bars — too few, parameters may be too tight")
        ok = False
    else:
        print(f"  Swing low density {sl_pct:.2f}% — OK")

    # Forward-filled prices should have no NaN after the first swing
    first_sh_idx = result["swing_high"].idxmax() if n_sh > 0 else None
    first_sl_idx = result["swing_low"].idxmax() if n_sl > 0 else None
    if first_sh_idx is not None:
        nans_after = result.loc[first_sh_idx:, "swing_high_price"].isna().sum()
        print(f"  swing_high_price NaN after first swing high: {nans_after} (should be 0)")
    if first_sl_idx is not None:
        nans_after = result.loc[first_sl_idx:, "swing_low_price"].isna().sum()
        print(f"  swing_low_price NaN after first swing low: {nans_after} (should be 0)")

    if ok:
        print("\n  All sanity checks PASSED.")
    else:
        print("\n  NOTE: On 1-minute data with small lookback (left=3, right=1),")
        print("  high density is expected — every minor local extremum qualifies.")
        print("  These params are meant for tuning in Phase 1.")
        print("  Higher TF data (5m, 15m, 1H) or larger left/right will yield fewer,")
        print("  more significant swing points.")
    print("=" * 60)
