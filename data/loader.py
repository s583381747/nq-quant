"""
NQ 1-minute data loader.

Reads the pre-built continuous NQ CSV (Panama Canal back-adjusted),
adds unadjusted price column and rollover flags, converts to UTC,
and exports to parquet.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Rollover table: (roll_date, contract_transition, gap_points)
# From NQ_rollover_report.txt — Panama Canal back-adjustment
ROLLOVER_TABLE = [
    ("2022-03-17", "NQH22->NQM22", 215.25),
    ("2022-06-16", "NQM22->NQU22", 148.00),
    ("2022-09-15", "NQU22->NQZ22", -1.00),
    ("2022-12-15", "NQZ22->NQH23", 128.75),
    ("2023-03-16", "NQH23->NQM23", 64.50),
    ("2023-06-15", "NQM23->NQU23", 18.00),
    ("2023-09-14", "NQU23->NQZ23", 20.75),
    ("2023-12-14", "NQZ23->NQH24", 215.25),
    ("2024-03-14", "NQH24->NQM24", 330.75),
    ("2024-06-20", "NQM24->NQU24", 224.00),
    ("2024-09-19", "NQU24->NQZ24", 204.75),
    ("2024-12-19", "NQZ24->NQH25", 315.25),
    ("2025-03-20", "NQH25->NQM25", 400.00),
    ("2025-06-19", "NQM25->NQU25", 12.25),
    ("2025-09-18", "NQU25->NQZ25", 256.25),
    ("2025-12-18", "NQZ25->NQH26", 154.25),
    ("2026-03-19", "NQH26->NQM26", 283.50),
]


def _build_offset_series(index: pd.DatetimeIndex) -> pd.Series:
    """Build a Series of cumulative back-adjustment offsets aligned to the data index.

    Each bar gets the offset needed to recover the unadjusted price:
        unadjusted = adjusted + offset
    Newest contract period has offset=0; older periods accumulate.
    """
    # Build breakpoints: (roll_date, cumulative_offset)
    # Process newest-to-oldest to accumulate correctly
    breaks = []
    cumulative = 0.0
    for date_str, _, gap in reversed(ROLLOVER_TABLE):
        cumulative += gap
        breaks.append((pd.Timestamp(date_str), cumulative))
    breaks.reverse()  # now oldest-first: [(2022-03-17, 2990.50), ..., (2026-03-19, 283.50)]

    # Make tz-aware
    tz = index.tz
    if tz:
        breaks = [(ts.tz_localize(tz), val) for ts, val in breaks]

    # Assign offsets: newest-first so earlier (broader) assignments don't overwrite
    result = pd.Series(0.0, index=index, dtype="float64")
    for roll_ts, offset_val in reversed(breaks):
        result.loc[result.index < roll_ts] = offset_val

    return result


def _build_roll_flag(index: pd.DatetimeIndex) -> pd.Series:
    """Flag bars that fall on rollover dates (the roll date itself)."""
    roll_dates = {pd.Timestamp(d).date() for d, _, _ in ROLLOVER_TABLE}
    return pd.Series(
        [t.date() in roll_dates for t in index],
        index=index,
        dtype=bool,
    )


def _build_weekend_gap_flag(df: pd.DataFrame) -> pd.Series:
    """Flag the first bar after a weekend/holiday gap (>= 24h gap from prior bar).

    These bars may have large open gaps that are NOT standard FVGs.
    Per Lanto: weekend gaps CAN be FVGs but need extra confirmation.
    """
    time_diff = df.index.to_series().diff()
    return time_diff >= pd.Timedelta(hours=24)


def load_raw_csv(
    csv_path: str | Path = "/Users/mac/project/qqq/tradingbot/data/barchart_nq/NQ_1min_continuous_full.csv",
) -> pd.DataFrame:
    """Load raw NQ 1-min continuous CSV into a DataFrame.

    Returns DataFrame with DatetimeIndex (UTC) and columns:
        open, high, low, close, volume,
        unadjusted_close, is_roll_date, is_weekend_gap
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"NQ data not found at {csv_path}")

    logger.info("Loading NQ 1-min data from %s", csv_path)

    df = pd.read_csv(
        csv_path,
        parse_dates=["Time"],
        dtype={
            "Open": "float64",
            "High": "float64",
            "Low": "float64",
            "Close": "float64",
            "Volume": "int64",
        },
    )

    # Rename to lowercase
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"time": "datetime"})

    # Sort and deduplicate
    df = df.sort_values("datetime").drop_duplicates(subset="datetime", keep="first")

    # The source CSV timestamps are in US/Central (Barchart/CME convention).
    # Verified: session opens at 17:00 raw = 5 PM CT = 6 PM ET = NQ open.
    # Contract file headers also state "CDT".
    df["datetime"] = df["datetime"].dt.tz_localize("US/Central", ambiguous="NaT", nonexistent="shift_forward")
    df = df.dropna(subset=["datetime"])  # drop any ambiguous DST bars
    df["datetime"] = df["datetime"].dt.tz_convert("UTC")

    # Set index
    df = df.set_index("datetime")

    # Add unadjusted close
    offset = _build_offset_series(df.index)
    df["unadjusted_close"] = df["close"] + offset

    # Add rollover flag
    df["is_roll_date"] = _build_roll_flag(df.index)

    # Add weekend gap flag
    df["is_weekend_gap"] = _build_weekend_gap_flag(df)

    logger.info(
        "Loaded %s bars, %s to %s",
        f"{len(df):,}",
        df.index.min(),
        df.index.max(),
    )

    return df


def save_parquet(
    df: pd.DataFrame,
    out_path: str | Path = "/Users/mac/project/nq quant/data/NQ_1min.parquet",
) -> Path:
    """Save DataFrame to parquet."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow")
    logger.info("Saved %s bars to %s", f"{len(df):,}", out_path)
    return out_path


def load_parquet(
    path: str | Path = "/Users/mac/project/nq quant/data/NQ_1min.parquet",
) -> pd.DataFrame:
    """Load NQ 1-min parquet."""
    return pd.read_parquet(path, engine="pyarrow")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_raw_csv()
    save_parquet(df)

    # Quick sanity check
    print(f"\nRows: {len(df):,}")
    print(f"Range: {df.index.min()} → {df.index.max()}")
    print(f"NaN: {df.isnull().sum().sum()}")
    print(f"Roll date bars: {df.is_roll_date.sum()}")
    print(f"Weekend gap bars: {df.is_weekend_gap.sum()}")
    print(f"\nSample (recent):")
    print(df.tail(3))
    print(f"\nSample (oldest):")
    print(df.head(3))
    print(f"\nUnadjusted vs adjusted (oldest):")
    row = df.iloc[0]
    print(f"  adjusted close: {row['close']:.2f}")
    print(f"  unadjusted close: {row['unadjusted_close']:.2f}")
    print(f"  offset: {row['unadjusted_close'] - row['close']:.2f}")
