"""
Data cleaner for NQ 1-minute data.

Handles:
- Verify no NaN in OHLCV
- Verify OHLC consistency (H >= O,C >= L)
- Verify timestamps are UTC
- Verify no duplicates
- Flag and optionally remove zero-volume bars
- Report data quality summary
"""

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def validate(df: pd.DataFrame) -> dict:
    """Run all data quality checks. Returns a summary dict.

    Raises ValueError if critical issues are found (NaN in OHLC, wrong timezone).
    """
    issues = {}

    # Timezone
    if df.index.tz is None:
        raise ValueError("DataFrame index must be tz-aware (expected UTC)")
    if str(df.index.tz) != "UTC":
        raise ValueError(f"Expected UTC timezone, got {df.index.tz}")
    issues["timezone"] = "UTC ✓"

    # NaN
    nan_counts = df[["open", "high", "low", "close", "volume"]].isnull().sum()
    total_nan = nan_counts.sum()
    if total_nan > 0:
        raise ValueError(f"Found {total_nan} NaN values in OHLCV:\n{nan_counts}")
    issues["nan"] = 0

    # Duplicates
    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        logger.warning("Found %d duplicate timestamps", dup_count)
    issues["duplicates"] = dup_count

    # OHLC consistency
    h_lt_l = (df["high"] < df["low"]).sum()
    h_lt_oc = ((df["high"] < df["open"]) | (df["high"] < df["close"])).sum()
    l_gt_oc = ((df["low"] > df["open"]) | (df["low"] > df["close"])).sum()
    issues["high_lt_low"] = h_lt_l
    issues["high_lt_open_close"] = h_lt_oc
    issues["low_gt_open_close"] = l_gt_oc

    if h_lt_l > 0:
        raise ValueError(f"Found {h_lt_l} bars where High < Low")

    # Zero volume
    zero_vol = (df["volume"] == 0).sum()
    issues["zero_volume_bars"] = zero_vol

    # Negative prices
    neg = (df[["open", "high", "low", "close"]] <= 0).sum().sum()
    issues["negative_prices"] = neg

    # Summary
    issues["total_bars"] = len(df)
    issues["date_range"] = f"{df.index.min()} → {df.index.max()}"

    return issues


def print_report(issues: dict) -> None:
    """Print a human-readable data quality report."""
    print("\n" + "=" * 50)
    print("DATA QUALITY REPORT")
    print("=" * 50)
    for key, val in issues.items():
        print(f"  {key}: {val}")
    print("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from loader import load_parquet

    df = load_parquet()
    issues = validate(df)
    print_report(issues)
