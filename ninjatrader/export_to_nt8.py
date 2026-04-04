"""
Convert parquet OHLCV data to NinjaTrader 8 CSV import format.

NinjaTrader 8 CSV format:
  yyyyMMdd HHmmss;Open;High;Low;Close;Volume
  Timezone: UTC (NT8 converts to platform timezone on import)
  Timestamp: bar OPEN time (beginning of bar)
  Separator: semicolon (;)
  No header row, no trailing newline

Usage:
  python ninjatrader/export_to_nt8.py

Output:
  ninjatrader/NQ_5m_NT8.txt   (NQ 5-minute, ~711K bars, UTC)
  ninjatrader/ES_5m_NT8.txt   (ES 5-minute, ~710K bars, UTC)

NT8 Import Steps:
  PREREQUISITE: NT8 > Tools > Options > General > Time zone = Eastern Time
  1. Historical Data Manager > Import
  2. Format: "NinjaTrader (timestamps represent start of bar time)"
  3. Data Type: Last
  4. Time Zone: UTC
  5. Select file (NQ_5m_NT8.txt or ES_5m_NT8.txt)
  6. Instrument: create as Index type (not Future), e.g. NQDATA / ESDATA
  7. Trading Hours: leave EMPTY (do not set ETH/RTH/any template)
  8. Point value: 20 (NQ mini) or 2 (MNQ micro)
  9. Tick size: 0.25

Data flow:
  Parquet (UTC, bar open) → this script → .txt (UTC, bar open)
  → NT8 import (UTC→ET auto) → Time[0] = bar close ET
  → C# code: barOpenET = Time[0] - 5min → matches Python index exactly
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parquet_to_nt8(
    parquet_path: str | Path,
    output_path: str | Path,
    instrument: str = "NQ",
) -> Path:
    """Convert parquet OHLCV to NinjaTrader 8 CSV format.

    Parameters
    ----------
    parquet_path : Path
        Path to .parquet file with UTC DatetimeIndex and OHLCV columns.
    output_path : Path
        Output .txt file path.
    instrument : str
        Instrument name for logging.

    Returns
    -------
    Path to output file.
    """
    parquet_path = Path(parquet_path)
    output_path = Path(output_path)

    print(f"Loading {instrument} from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df):,} bars, {df.index[0]} to {df.index[-1]}")

    # Verify UTC
    assert df.index.tz is not None, "Index must be timezone-aware"

    # Keep as UTC — NinjaTrader import dialog has "Time Zone" = UTC
    ct_index = df.index.tz_convert("UTC")

    # Build NT8 columns (vectorized, not iterrows)
    # NT8 format: "yyyyMMdd HHmmss;Open;High;Low;Close;Volume" (6 fields)
    # Date+Time combined in one field, separated by space
    datetimes = ct_index.strftime("%Y%m%d %H%M%S")

    # Format prices — NQ prices have 0.25 tick, 2 decimal places
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values.astype(np.int64)

    print(f"  Writing {len(df):,} bars to {output_path}...")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        for i in range(len(df)):
            if i > 0:
                f.write("\r\n")
            f.write(f"{datetimes[i]};{opens[i]:.2f};{highs[i]:.2f};{lows[i]:.2f};{closes[i]:.2f};{volumes[i]}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Done: {size_mb:.1f} MB")

    # Print sample
    print(f"\n  Sample (first 3 bars):")
    for i in range(min(3, len(df))):
        print(f"    {datetimes[i]};{opens[i]:.2f};{highs[i]:.2f};{lows[i]:.2f};{closes[i]:.2f};{volumes[i]}")

    print(f"  Sample (last 3 bars):")
    for i in range(max(0, len(df) - 3), len(df)):
        print(f"    {datetimes[i]};{opens[i]:.2f};{highs[i]:.2f};{lows[i]:.2f};{closes[i]:.2f};{volumes[i]}")

    return output_path


def main():
    data_dir = PROJECT_ROOT / "data"
    out_dir = PROJECT_ROOT / "ninjatrader"

    nq_parquet = data_dir / "NQ_5m_10yr.parquet"
    es_parquet = data_dir / "ES_5m_10yr.parquet"

    # Export NQ
    if nq_parquet.exists():
        parquet_to_nt8(nq_parquet, out_dir / "NQ_5m_NT8.txt", "NQ")
    else:
        print(f"WARNING: {nq_parquet} not found, skipping NQ")

    print()

    # Export ES
    if es_parquet.exists():
        parquet_to_nt8(es_parquet, out_dir / "ES_5m_NT8.txt", "ES")
    else:
        print(f"WARNING: {es_parquet} not found, skipping ES")

    print("\n=== IMPORT INSTRUCTIONS ===")
    print("PREREQUISITE: NT8 > Tools > Options > General > Time zone = Eastern Time")
    print()
    print("1. Historical Data Manager > Import")
    print("2. Format: NinjaTrader (timestamps represent start of bar time)")
    print("3. Data Type: Last")
    print("4. Time Zone: UTC")
    print("5. Select NQ_5m_NT8.txt → Instrument: NQDATA (Index type)")
    print("6. Select ES_5m_NT8.txt → Instrument: ESDATA (Index type)")
    print("7. Trading Hours: leave EMPTY")
    print("8. Point value: 20 (NQ) or 2 (MNQ), Tick size: 0.25")
    print()
    print("Time alignment: Time[0] = bar close ET, barOpenET = Time[0] - 5min = Python index")


if __name__ == "__main__":
    main()
