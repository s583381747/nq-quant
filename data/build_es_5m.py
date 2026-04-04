"""
Build ES 5-minute parquet from the continuous 1-minute CSV.

Steps:
1. Load ES_1min_continuous_full.csv
2. Convert timezone: CDT (US/Central) -> UTC
3. Resample 1m -> 5m (OHLCV aggregation)
4. Align index with NQ_5m_10yr.parquet (same timestamps)
5. Handle NaN: ffill for small gaps, mark large gaps
6. Save as ES_5m_10yr.parquet
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
ES_1M_CSV = Path(r"C:\projects\lanto quant\barchart dl\data\barchart_es\ES_1min_continuous_full.csv")
NQ_5M_PARQUET = Path(r"C:\projects\lanto quant\nq quant\data\NQ_5m_10yr.parquet")
ES_5M_OUTPUT = Path(r"C:\projects\lanto quant\nq quant\data\ES_5m_10yr.parquet")


def main():
    print("=" * 60)
    print("ES 5-MINUTE PARQUET BUILDER")
    print("=" * 60)

    # Step 1: Load ES 1-minute data
    print("\n--- Loading ES 1-minute continuous CSV ---")
    es_1m = pd.read_csv(ES_1M_CSV, index_col="Time", parse_dates=True)
    print(f"  Loaded: {len(es_1m):,} bars")
    print(f"  Range: {es_1m.index[0]} -> {es_1m.index[-1]}")
    print(f"  Columns: {es_1m.columns.tolist()}")

    # Step 2: Convert timezone CDT -> UTC
    # Barchart CSV timestamps are in US/Central (CDT/CST depending on DST)
    print("\n--- Converting timezone: US/Central -> UTC ---")
    es_1m.index = es_1m.index.tz_localize("US/Central", ambiguous="NaT", nonexistent="shift_forward")
    # Drop any NaT rows from ambiguous DST transitions
    nat_count = es_1m.index.isna().sum()
    if nat_count > 0:
        print(f"  Dropped {nat_count} NaT rows from ambiguous DST transitions")
        es_1m = es_1m[es_1m.index.notna()]
    es_1m.index = es_1m.index.tz_convert("UTC")
    print(f"  UTC range: {es_1m.index[0]} -> {es_1m.index[-1]}")

    # Step 3: Resample 1m -> 5m
    print("\n--- Resampling 1m -> 5m ---")
    es_5m = es_1m.resample("5min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })
    # Drop rows where all OHLC are NaN (no trading in that 5m window)
    es_5m = es_5m.dropna(subset=["Open", "High", "Low", "Close"], how="all")
    print(f"  ES 5m bars: {len(es_5m):,}")
    print(f"  Range: {es_5m.index[0]} -> {es_5m.index[-1]}")

    # Step 4: Load NQ 5m for alignment reference
    print("\n--- Loading NQ 5m parquet for alignment ---")
    nq_5m = pd.read_parquet(NQ_5M_PARQUET)
    print(f"  NQ 5m bars: {len(nq_5m):,}")
    print(f"  NQ range: {nq_5m.index[0]} -> {nq_5m.index[-1]}")

    # Align: reindex ES to the NQ timestamp grid
    # First, find overlapping range
    overlap_start = max(es_5m.index[0], nq_5m.index[0])
    overlap_end = min(es_5m.index[-1], nq_5m.index[-1])
    print(f"  Overlap: {overlap_start} -> {overlap_end}")

    # Get NQ timestamps within the overlap range
    nq_idx_overlap = nq_5m.index[(nq_5m.index >= overlap_start) & (nq_5m.index <= overlap_end)]

    # Reindex ES to NQ's timestamp grid within overlap
    es_aligned = es_5m.reindex(nq_idx_overlap)
    n_missing_before = es_aligned["Close"].isna().sum()
    print(f"  Bars after reindex to NQ grid: {len(es_aligned):,}")
    print(f"  Missing bars (NaN): {n_missing_before:,} ({n_missing_before/len(es_aligned)*100:.2f}%)")

    # Also include ES bars outside the NQ overlap range (ES may start earlier or end later)
    es_before = es_5m[es_5m.index < overlap_start]
    es_after = es_5m[es_5m.index > overlap_end]
    print(f"  ES bars before NQ start: {len(es_before):,}")
    print(f"  ES bars after NQ end: {len(es_after):,}")

    # Combine: ES-only bars + aligned bars + ES-only after bars
    es_final = pd.concat([es_before, es_aligned, es_after])
    es_final = es_final.sort_index()
    es_final = es_final[~es_final.index.duplicated(keep="first")]

    # Step 5: Handle NaN with ffill for small gaps
    print("\n--- Handling NaN ---")
    nan_before = es_final["Close"].isna().sum()
    print(f"  NaN bars before ffill: {nan_before:,}")

    # Forward-fill gaps up to 10 bars (50 minutes) — small session gaps
    es_final = es_final.ffill(limit=10)
    nan_after = es_final["Close"].isna().sum()
    print(f"  NaN bars after ffill(limit=10): {nan_after:,}")

    # Drop remaining NaN rows (large gaps, weekends, holidays)
    es_final = es_final.dropna(subset=["Open", "High", "Low", "Close"])
    print(f"  Bars after dropping remaining NaN: {len(es_final):,}")

    # Rename columns to lowercase to match NQ format
    es_final.columns = [c.lower() for c in es_final.columns]
    es_final.index.name = "Time"

    # Add metadata columns matching NQ format
    es_final["is_roll_date"] = False
    es_final["is_weekend_gap"] = False

    # Mark weekend gaps (time gap > 2 days between consecutive bars)
    time_diff = es_final.index.to_series().diff()
    weekend_mask = time_diff > pd.Timedelta(days=2)
    es_final.loc[weekend_mask, "is_weekend_gap"] = True
    print(f"  Weekend gaps marked: {weekend_mask.sum()}")

    # Ensure volume is integer
    es_final["volume"] = es_final["volume"].fillna(0).astype(int)

    # Step 6: Save parquet
    print("\n--- Saving parquet ---")
    es_final.to_parquet(ES_5M_OUTPUT, engine="pyarrow")
    file_size = ES_5M_OUTPUT.stat().st_size / 1024 / 1024
    print(f"  Saved: {ES_5M_OUTPUT} ({file_size:.1f} MB)")

    # Step 7: Validation
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")

    # Reload and verify
    es_check = pd.read_parquet(ES_5M_OUTPUT)
    nq_check = pd.read_parquet(NQ_5M_PARQUET)

    print(f"\n  {'Metric':<30} {'ES':>15} {'NQ':>15}")
    print(f"  {'-'*60}")
    print(f"  {'Total bars':<30} {len(es_check):>15,} {len(nq_check):>15,}")
    print(f"  {'Trading days':<30} {es_check.index.normalize().nunique():>15,} {nq_check.index.normalize().nunique():>15,}")
    print(f"  {'First timestamp':<30} {str(es_check.index[0])[:25]:>15} {str(nq_check.index[0])[:25]:>15}")
    print(f"  {'Last timestamp':<30} {str(es_check.index[-1])[:25]:>15} {str(nq_check.index[-1])[:25]:>15}")
    print(f"  {'Price min':<30} {es_check['close'].min():>15.2f} {nq_check['close'].min():>15.2f}")
    print(f"  {'Price max':<30} {es_check['close'].max():>15.2f} {nq_check['close'].max():>15.2f}")
    print(f"  {'Avg volume/bar':<30} {es_check['volume'].mean():>15,.0f} {nq_check['volume'].mean():>15,.0f}")
    print(f"  {'NaN in close':<30} {es_check['close'].isna().sum():>15} {nq_check['close'].isna().sum():>15}")
    print(f"  {'NaN in volume':<30} {es_check['volume'].isna().sum():>15} {nq_check['volume'].isna().sum():>15}")
    print(f"  {'Index timezone':<30} {str(es_check.index.tz):>15} {str(nq_check.index.tz):>15}")

    # Check timestamp overlap
    overlap_idx = es_check.index.intersection(nq_check.index)
    print(f"\n  Shared timestamps: {len(overlap_idx):,}")
    print(f"  ES-only timestamps: {len(es_check.index.difference(nq_check.index)):,}")
    print(f"  NQ-only timestamps: {len(nq_check.index.difference(es_check.index)):,}")

    # Price sanity check for ES (should be ~1800-6500+ range)
    price_ok = 1000 < es_check["close"].min() < 2500 and 5000 < es_check["close"].max() < 8000
    print(f"\n  ES price range sanity: {'PASS' if price_ok else 'FAIL'}")
    print(f"  ES close range: {es_check['close'].min():.2f} - {es_check['close'].max():.2f}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
