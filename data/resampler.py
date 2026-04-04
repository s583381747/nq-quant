"""
Resample 1-minute NQ data to higher timeframes (5m, 15m, 1H, 4H, 1D).

Key rules (CLAUDE.md §13):
- Resample from 1m to build all HTFs
- OHLCV aggregation: O=first, H=max, L=min, C=last, V=sum
- Drop bars with zero volume (market closed)
- Preserve timezone (UTC)

4H bars are aligned to NQ session boundaries (18:00 ET session open):
  18:00-22:00 = first 4h of Asia
  22:00-02:00 = second 4h of Asia
  02:00-06:00 = early London
  06:00-10:00 = late London + NY open
  10:00-14:00 = NY prime
  14:00-18:00 = NY close
This is done by converting to US/Eastern, resampling with offset='18h',
and converting back to UTC, so DST shifts are handled correctly.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

TIMEFRAMES = {
    "5m": "5min",
    "15m": "15min",
    "1H": "1h",
    "4H": "4h",
    "1D": "1D",
}


def _resample_4h_session_aligned(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1m data to 4H bars aligned to NQ session boundaries.

    NQ futures session opens at 18:00 ET.  Ideal 4H boundaries in ET:
      18:00, 22:00, 02:00, 06:00, 10:00, 14:00 (repeat)

    Approach:
      1. Convert UTC index to US/Eastern
      2. Strip timezone to get naive local times (critical: resampling on a
         tz-aware ET index uses a UTC-epoch anchor, which shifts local-hour
         alignment by 1h when DST changes.  Using naive local times ensures
         the 18h offset is always relative to local midnight.)
      3. Resample naive 4h with offset='18h'
      4. Re-localize to US/Eastern, convert back to UTC

    Edge case: on spring-forward day the 02:00 ET bin is shifted to 03:00
    (that hour doesn't exist).  This affects 1 bar per year and is correct.
    """
    ohlcv_cols = ["open", "high", "low", "close", "volume"]

    # Step 1-2: Convert to naive Eastern time
    df_et = df_1m[ohlcv_cols].copy()
    df_et.index = df_et.index.tz_convert("US/Eastern").tz_localize(None)

    # Step 3: Resample with offset='18h' on naive local times
    # Bins start at 18:00 local each day → 18:00, 22:00, 02:00, 06:00, 10:00, 14:00
    resampled = df_et.resample("4h", offset="18h").agg(OHLCV_AGG)

    # Drop empty bars (market closed / no trading)
    resampled = resampled.dropna(subset=["open"])
    resampled = resampled[resampled["volume"] > 0]

    # Step 4: Re-localize to ET and convert back to UTC
    # ambiguous='infer' handles fall-back; nonexistent='shift_forward' handles spring-forward
    resampled.index = resampled.index.tz_localize(
        "US/Eastern", ambiguous="infer", nonexistent="shift_forward"
    )
    resampled.index = resampled.index.tz_convert("UTC")

    return resampled


def resample(df_1m: pd.DataFrame, tf_label: str) -> pd.DataFrame:
    """Resample 1-minute DataFrame to a higher timeframe.

    Args:
        df_1m: 1-minute OHLCV DataFrame with UTC DatetimeIndex.
        tf_label: One of '5m', '15m', '1H', '4H', '1D'.

    Returns:
        Resampled OHLCV DataFrame. Bars with zero volume (market closed) are dropped.
    """
    if tf_label not in TIMEFRAMES:
        raise ValueError(f"Unknown timeframe: {tf_label}. Choose from {list(TIMEFRAMES.keys())}")

    if tf_label == "4H":
        # Session-aligned 4H resampling: bars at 18:00, 22:00, 02:00,
        # 06:00, 10:00, 14:00 ET — aligned to NQ session open (18:00 ET).
        # Done in US/Eastern timezone so DST is handled correctly.
        resampled = _resample_4h_session_aligned(df_1m)
    else:
        freq = TIMEFRAMES[tf_label]
        resampled = (
            df_1m[["open", "high", "low", "close", "volume"]]
            .resample(freq)
            .agg(OHLCV_AGG)
        )
        # Drop empty bars (market closed / no trading)
        resampled = resampled.dropna(subset=["open"])
        resampled = resampled[resampled["volume"] > 0]

    # Carry forward roll/weekend flags: True if ANY underlying bar has it.
    # For 4H we resample in naive ET space (same approach as OHLCV) to keep
    # flag bins aligned with session-boundary bars.
    for flag_col in ("is_roll_date", "is_weekend_gap"):
        if flag_col in df_1m.columns:
            if tf_label == "4H":
                flag_et = df_1m[flag_col].copy()
                flag_et.index = flag_et.index.tz_convert("US/Eastern").tz_localize(None)
                flag_resampled = flag_et.resample("4h", offset="18h").max()
                flag_resampled.index = flag_resampled.index.tz_localize(
                    "US/Eastern", ambiguous="infer", nonexistent="shift_forward"
                )
                flag_resampled.index = flag_resampled.index.tz_convert("UTC")
                resampled[flag_col] = (
                    flag_resampled.reindex(resampled.index).fillna(False).astype(bool)
                )
            else:
                freq = TIMEFRAMES[tf_label]
                resampled[flag_col] = (
                    df_1m[flag_col].resample(freq).max()
                    .reindex(resampled.index).fillna(False).astype(bool)
                )

    logger.info("Resampled to %s: %s bars", tf_label, f"{len(resampled):,}")
    return resampled


def resample_all(df_1m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Resample 1-minute data to all standard timeframes.

    Returns:
        Dict mapping timeframe label to DataFrame: {'5m': df, '15m': df, ...}
    """
    result = {}
    for tf_label in TIMEFRAMES:
        result[tf_label] = resample(df_1m, tf_label)
    return result


def save_all(
    frames: dict[str, pd.DataFrame],
    out_dir: str | Path = "/Users/mac/project/nq quant/data",
) -> None:
    """Save all resampled DataFrames to parquet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tf_label, df in frames.items():
        path = out_dir / f"NQ_{tf_label}.parquet"
        df.to_parquet(path, engine="pyarrow")
        logger.info("Saved %s to %s", tf_label, path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from loader import load_parquet

    df_1m = load_parquet()
    logger.info("Loaded 1m: %s bars", f"{len(df_1m):,}")

    frames = resample_all(df_1m)
    save_all(frames)

    # Sanity checks
    for tf, df in frames.items():
        print(f"\n=== {tf} ===")
        print(f"  Bars: {len(df):,}")
        print(f"  Range: {df.index.min()} → {df.index.max()}")
        print(f"  NaN: {df[['open','high','low','close']].isnull().sum().sum()}")
        print(f"  High<Low: {(df.high < df.low).sum()}")

        # Verify no future leakage: HTF bar timestamp should be <= all 1m bars in that window
        sample_idx = len(df) // 2
        htf_bar = df.iloc[sample_idx]
        htf_time = df.index[sample_idx]
        # For session-aligned 4H, use the next bar's timestamp as window_end
        # (adding a fixed 4h offset in UTC can be wrong around DST transitions).
        if sample_idx + 1 < len(df):
            window_end = df.index[sample_idx + 1]
        else:
            freq = TIMEFRAMES[tf]
            window_end = htf_time + pd.tseries.frequencies.to_offset(freq)
        underlying = df_1m.loc[(df_1m.index >= htf_time) & (df_1m.index < window_end)]
        if len(underlying) > 0:
            assert htf_bar["high"] == underlying["high"].max(), f"{tf} high mismatch"
            assert htf_bar["low"] == underlying["low"].min(), f"{tf} low mismatch"
            print(f"  OHLCV integrity check: PASS (verified bar {htf_time})")
