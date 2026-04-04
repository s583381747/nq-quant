"""
Merge NQ contract chunks into a 10-year continuous 1-minute series.

Extends the original merge_nq.py methodology to cover NQH16 through NQM26
(approximately Dec 2015 to Mar 2026).

Strategy (identical to merge_nq.py):
1. Load all chunks per contract, merge & deduplicate
2. For rollover: roll 1 business day before expiry (Thursday before Friday expiry)
3. Apply Panama Canal back-adjustment (add price diff at roll to all prior data)
4. Save full Globex session version

Uses multiprocessing for parallel chunk loading.

Output:
  - NQ_1min_10yr_full.csv (Full Globex session, ~10 years)
  - NQ_rollover_10yr_report.txt (audit of each roll)
  - NQ_1min_10yr.parquet (UTC, with unadjusted_close, is_roll_date, is_weekend_gap)
  - NQ_5m_10yr.parquet, NQ_1H_10yr.parquet, NQ_4H_10yr.parquet
"""
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("/Users/mac/project/qqq/tradingbot/data/barchart_nq")
OUT_DIR = Path("/Users/mac/project/nq quant/data")

# ── Full contract list: NQH16 through NQM26 ──────────────────────────────
# Expiry = 3rd Friday of expiry month.  Roll = 1 business day before expiry.
# H=Mar, M=Jun, U=Sep, Z=Dec
CONTRACTS = [
    # 2016
    ("NQH16", "2016-03-18"),
    ("NQM16", "2016-06-17"),
    ("NQU16", "2016-09-16"),
    ("NQZ16", "2016-12-16"),
    # 2017
    ("NQH17", "2017-03-17"),
    ("NQM17", "2017-06-16"),
    ("NQU17", "2017-09-15"),
    ("NQZ17", "2017-12-15"),
    # 2018
    ("NQH18", "2018-03-16"),
    ("NQM18", "2018-06-15"),
    ("NQU18", "2018-09-21"),
    ("NQZ18", "2018-12-21"),
    # 2019
    ("NQH19", "2019-03-15"),
    ("NQM19", "2019-06-21"),
    ("NQU19", "2019-09-20"),
    ("NQZ19", "2019-12-20"),
    # 2020
    ("NQH20", "2020-03-20"),
    ("NQM20", "2020-06-19"),
    ("NQU20", "2020-09-18"),
    ("NQZ20", "2020-12-18"),
    # 2021
    ("NQH21", "2021-03-19"),
    ("NQM21", "2021-06-18"),
    ("NQU21", "2021-09-17"),
    ("NQZ21", "2021-12-17"),
    # 2022 (same as original merge_nq.py)
    ("NQH22", "2022-03-18"),
    ("NQM22", "2022-06-17"),
    ("NQU22", "2022-09-16"),
    ("NQZ22", "2022-12-16"),
    # 2023
    ("NQH23", "2023-03-17"),
    ("NQM23", "2023-06-16"),
    ("NQU23", "2023-09-15"),
    ("NQZ23", "2023-12-15"),
    # 2024
    ("NQH24", "2024-03-15"),
    ("NQM24", "2024-06-21"),
    ("NQU24", "2024-09-20"),
    ("NQZ24", "2024-12-20"),
    # 2025
    ("NQH25", "2025-03-21"),
    ("NQM25", "2025-06-20"),
    ("NQU25", "2025-09-19"),
    ("NQZ25", "2025-12-19"),
    # 2026
    ("NQH26", "2026-03-20"),
    ("NQM26", "2026-06-19"),
]


# ── Chunk loading (parallelizable) ───────────────────────────────────────

def _load_single_chunk(path: str) -> pd.DataFrame:
    """Load a single CSV chunk file. Handles both 'Latest' and 'Close' columns."""
    df = pd.read_csv(path, skipfooter=1, engine="python")
    # Remove Barchart footer rows
    df = df[~df["Time"].astype(str).str.contains("Downloaded", na=False)]
    return df


def load_contract(symbol: str) -> pd.DataFrame:
    """Load all chunks for a contract and merge (parallel chunk reads)."""
    chunks = sorted(DATA_DIR.glob(f"{symbol}_chunk*.csv"))
    if not chunks:
        single = DATA_DIR / f"{symbol}_1min.csv"
        if single.exists():
            chunks = [single]
        else:
            print(f"  WARNING: No files found for {symbol}")
            return pd.DataFrame()

    # Parallel load of chunks using process pool
    chunk_paths = [str(c) for c in chunks]
    with mp.Pool(processes=min(len(chunk_paths), mp.cpu_count())) as pool:
        dfs = pool.map(_load_single_chunk, chunk_paths)

    df = pd.concat(dfs, ignore_index=True)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").drop_duplicates(subset=["Time"], keep="first")
    df = df.set_index("Time")

    # Handle column name variants: 'Latest' -> 'Close'
    if "Latest" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Latest": "Close"})
    elif "Latest" in df.columns and "Close" in df.columns:
        # If both exist, prefer 'Close', drop 'Latest'
        df = df.drop(columns=["Latest"])

    # Keep only OHLCV
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"Column {col} missing from {symbol} data. Columns: {list(df.columns)}")

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})

    print(f"  {symbol}: {len(df):>8,} bars, {df.index[0]} -> {df.index[-1]}")
    return df


def _load_contract_wrapper(args: tuple) -> tuple:
    """Wrapper for parallel contract loading."""
    symbol, _ = args
    df = load_contract(symbol)
    return (symbol, df)


def audit_contract(df: pd.DataFrame, symbol: str) -> dict:
    """Quick quality check."""
    if df.empty:
        return {}
    return {
        "symbol": symbol,
        "bars": len(df),
        "first": str(df.index[0]),
        "last": str(df.index[-1]),
        "trading_days": df.index.normalize().nunique(),
        "zero_range": int((df["High"] == df["Low"]).sum()),
        "zero_range_pct": round((df["High"] == df["Low"]).mean() * 100, 2),
        "avg_volume": round(df["Volume"].mean(), 1),
        "price_range": f"{df['Close'].min():.2f} - {df['Close'].max():.2f}",
    }


# ── Main build pipeline ──────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("NQ 10-YEAR CONTINUOUS SERIES BUILDER")
    print(f"Contracts: {CONTRACTS[0][0]} ({CONTRACTS[0][1]}) -> {CONTRACTS[-1][0]} ({CONTRACTS[-1][1]})")
    print(f"CPU cores: {mp.cpu_count()}")
    print("=" * 70)

    # ── Step 1: Load all contracts in parallel ────────────────────────────
    print("\n--- Loading contracts (parallel) ---")
    # Use a process pool to load contracts in parallel
    # But we need ordered results, so we load sequentially with parallel chunk reads
    contracts = {}
    all_stats = []
    for symbol, expiry in CONTRACTS:
        df = load_contract(symbol)
        if not df.empty:
            contracts[symbol] = df
            stats = audit_contract(df, symbol)
            all_stats.append(stats)

    elapsed_load = time.time() - t0
    print(f"\n  Loaded {len(contracts)} contracts in {elapsed_load:.1f}s")

    # ── Step 2: Audit report ──────────────────────────────────────────────
    print(f"\n--- Contract Audit ({len(contracts)} contracts) ---")
    hdr = f"{'Symbol':<10} {'Bars':>8} {'Days':>5} {'ZR%':>6} {'AvgVol':>10} {'Price Range':<25} {'First':<20} {'Last':<20}"
    print(hdr)
    for s in all_stats:
        print(f"{s['symbol']:<10} {s['bars']:>8,} {s['trading_days']:>5} {s['zero_range_pct']:>5.1f}% "
              f"{s['avg_volume']:>10,.0f} {s['price_range']:<25} {s['first'][:19]:<20} {s['last'][:19]:<20}")

    # ── Step 3: Rollover analysis ─────────────────────────────────────────
    print("\n--- Rollover Analysis ---")
    print("Roll strategy: 1 business day before expiry (Thursday before expiry Friday)")
    print(f"{'Roll':<20} {'Expiring':>12} {'New':>12} {'Gap':>8} {'Gap%':>8} {'Overlap bars':>12}")

    roll_report = []
    contract_list = list(contracts.items())

    for i in range(len(contract_list) - 1):
        old_sym, old_df = contract_list[i]
        new_sym, new_df = contract_list[i + 1]

        # Find expiry date
        expiry_str = [e for s, e in CONTRACTS if s == old_sym][0]
        expiry = pd.Timestamp(expiry_str)

        # Roll date = 1 business day before expiry
        roll_date = expiry - pd.tseries.offsets.BDay(1)

        # Find overlap
        overlap_idx = old_df.index.intersection(new_df.index)

        # Get prices at roll point (closest bar to roll date 15:00 CT = market close)
        roll_ts = pd.Timestamp(f"{roll_date.date()} 15:00")

        old_near_roll = old_df.index[old_df.index <= roll_ts]
        new_near_roll = new_df.index[new_df.index <= roll_ts]

        if len(old_near_roll) > 0 and len(new_near_roll) > 0:
            old_price = old_df.loc[old_near_roll[-1], "Close"]
            new_price = new_df.loc[new_near_roll[-1], "Close"]
            gap = new_price - old_price
            gap_pct = gap / old_price * 100
        else:
            # Fallback: use last/first available
            old_price = old_df["Close"].iloc[-1]
            new_price = new_df["Close"].iloc[0]
            gap = new_price - old_price
            gap_pct = gap / old_price * 100

        roll_info = {
            "transition": f"{old_sym}->{new_sym}",
            "roll_date": str(roll_date.date()),
            "old_price": old_price,
            "new_price": new_price,
            "gap": gap,
            "gap_pct": gap_pct,
            "overlap_bars": len(overlap_idx),
        }
        roll_report.append(roll_info)
        print(f"{roll_info['transition']:<20} {old_price:>12.2f} {new_price:>12.2f} "
              f"{gap:>+8.2f} {gap_pct:>+7.2f}% {len(overlap_idx):>12,}")

    # ── Step 4: Build continuous series with Panama Canal back-adjustment ─
    print("\n--- Building continuous series (Panama Canal back-adjustment) ---")

    cum_adjustment = 0.0
    adjusted_dfs = []

    for i in range(len(contract_list) - 1, -1, -1):
        sym, df = contract_list[i]
        expiry_str = [e for s, e in CONTRACTS if s == sym][0]
        expiry = pd.Timestamp(expiry_str)

        if i < len(contract_list) - 1:
            # Determine roll boundary: use data up to roll date
            roll_date = expiry - pd.tseries.offsets.BDay(1)
            roll_end = pd.Timestamp(f"{roll_date.date()} 23:59:59")
            df_slice = df[df.index <= roll_end].copy()
        else:
            # Last contract: use all data
            df_slice = df.copy()

        if i > 0:
            # Not the first contract: trim start to after previous contract's roll
            prev_sym = contract_list[i - 1][0]
            prev_expiry_str = [e for s, e in CONTRACTS if s == prev_sym][0]
            prev_expiry = pd.Timestamp(prev_expiry_str)
            prev_roll = prev_expiry - pd.tseries.offsets.BDay(1)
            roll_start = pd.Timestamp(f"{prev_roll.date()} 15:01")
            df_slice = df_slice[df_slice.index > roll_start]

        # Apply back-adjustment
        df_slice = df_slice.copy()
        df_slice[["Open", "High", "Low", "Close"]] += cum_adjustment

        print(f"  {sym}: {len(df_slice):>8,} bars, adj={cum_adjustment:+.2f}")
        adjusted_dfs.append(df_slice)

        # Accumulate adjustment for older contracts
        if i > 0:
            gap = roll_report[i - 1]["gap"]
            cum_adjustment -= gap

    # Reverse (oldest first) and concat
    adjusted_dfs.reverse()
    continuous = pd.concat(adjusted_dfs)
    continuous = continuous.sort_index()
    continuous = continuous[~continuous.index.duplicated(keep="first")]

    print(f"\n  Total continuous: {len(continuous):,} bars")
    print(f"  Range: {continuous.index[0]} -> {continuous.index[-1]}")
    print(f"  Trading days: {continuous.index.normalize().nunique()}")

    # ── Step 5: Save full session CSV ─────────────────────────────────────
    csv_path = OUT_DIR / "NQ_1min_10yr_full.csv"
    continuous.to_csv(csv_path)
    csv_mb = csv_path.stat().st_size / 1024 / 1024
    print(f"\n  Saved full session: {csv_path} ({csv_mb:.1f} MB)")

    # ── Step 6: Save rollover report ──────────────────────────────────────
    report_path = OUT_DIR / "NQ_rollover_10yr_report.txt"
    with open(report_path, "w") as f:
        f.write("NQ 10-Year Continuous Series - Rollover Report\n")
        f.write("=" * 70 + "\n\n")
        f.write("Strategy: Panama Canal back-adjustment\n")
        f.write("Roll: 1 business day before expiry (Thursday before Friday expiry)\n\n")
        for r in roll_report:
            f.write(f"{r['transition']}: roll={r['roll_date']}, "
                    f"old={r['old_price']:.2f}, new={r['new_price']:.2f}, "
                    f"gap={r['gap']:+.2f} ({r['gap_pct']:+.2f}%), "
                    f"overlap={r['overlap_bars']} bars\n")
        f.write(f"\nTotal back-adjustment (oldest): {cum_adjustment:+.2f}\n")
        f.write(f"Continuous bars: {len(continuous):,}\n")
    print(f"  Saved report: {report_path}")

    # ── Step 7: Convert to parquet (UTC, with enrichment) ─────────────────
    print("\n--- Converting to parquet (UTC, enriched) ---")
    parquet_df = _csv_to_parquet(continuous, roll_report)
    parquet_path = OUT_DIR / "NQ_1min_10yr.parquet"
    parquet_df.to_parquet(parquet_path, engine="pyarrow")
    print(f"  Saved parquet: {parquet_path} ({parquet_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # ── Step 8: Resample to 5m, 1H, 4H ───────────────────────────────────
    print("\n--- Resampling to higher timeframes ---")
    _resample_and_save(parquet_df)

    # ── Step 9: Verification ──────────────────────────────────────────────
    print("\n--- Verification ---")
    _verify(continuous, parquet_df, roll_report)

    elapsed_total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed_total:.1f}s")
    print(f"{'='*70}")


# ── Parquet conversion ────────────────────────────────────────────────────

def _csv_to_parquet(continuous: pd.DataFrame, roll_report: list) -> pd.DataFrame:
    """Convert the continuous CSV DataFrame to enriched parquet format.

    - Localize as US/Central, convert to UTC
    - Add unadjusted_close, is_roll_date, is_weekend_gap
    """
    df = continuous.copy()
    df.columns = [c.lower() for c in df.columns]

    # The source data timestamps are in US/Central (Barchart convention)
    df.index = df.index.tz_localize("US/Central", ambiguous="NaT", nonexistent="shift_forward")
    # Drop NaT (ambiguous DST)
    df = df[df.index.notna()]
    df.index = df.index.tz_convert("UTC")

    # Build rollover table from roll_report
    rollover_table = []
    for r in roll_report:
        rollover_table.append((r["roll_date"], r["transition"], r["gap"]))

    # Build unadjusted_close offset series
    # Process newest-to-oldest: cumulative offset
    cumulative = 0.0
    breaks = []
    for date_str, _, gap in reversed(rollover_table):
        cumulative += gap
        breaks.append((pd.Timestamp(date_str), cumulative))
    breaks.reverse()  # oldest-first

    # Make tz-aware
    tz = df.index.tz
    if tz:
        breaks = [(ts.tz_localize(tz), val) for ts, val in breaks]

    offset = pd.Series(0.0, index=df.index, dtype="float64")
    for roll_ts, offset_val in reversed(breaks):
        offset.loc[offset.index < roll_ts] = offset_val

    df["unadjusted_close"] = df["close"] + offset

    # is_roll_date
    roll_dates = {pd.Timestamp(d).date() for d, _, _ in rollover_table}
    # Convert UTC index to date for comparison
    df["is_roll_date"] = pd.Series(
        [t.date() in roll_dates for t in df.index],
        index=df.index,
        dtype=bool,
    )

    # is_weekend_gap (>= 24h gap from prior bar)
    time_diff = df.index.to_series().diff()
    df["is_weekend_gap"] = time_diff >= pd.Timedelta(hours=24)

    print(f"  Parquet rows: {len(df):,}")
    print(f"  Range: {df.index.min()} -> {df.index.max()}")
    print(f"  NaN count: {df[['open','high','low','close']].isnull().sum().sum()}")

    return df


# ── Resampling ────────────────────────────────────────────────────────────

OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
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

    # Convert to naive Eastern time
    df_et = df_1m[ohlcv_cols].copy()
    df_et.index = df_et.index.tz_convert("US/Eastern").tz_localize(None)

    # Resample with offset='18h' on naive local times
    resampled = df_et.resample("4h", offset="18h").agg(OHLCV_AGG)
    resampled = resampled.dropna(subset=["open"])
    resampled = resampled[resampled["volume"] > 0]

    # Re-localize to ET and convert back to UTC
    resampled.index = resampled.index.tz_localize(
        "US/Eastern", ambiguous="infer", nonexistent="shift_forward"
    )
    resampled.index = resampled.index.tz_convert("UTC")

    # Carry forward flags (same naive-ET approach)
    for flag_col in ("is_roll_date", "is_weekend_gap"):
        if flag_col in df_1m.columns:
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

    return resampled


def _resample_and_save(df_1m: pd.DataFrame) -> None:
    """Resample 1m to 5m, 1H, 4H and save as parquet.

    4H bars are session-aligned to NQ open (18:00 ET) so boundaries fall at
    18:00, 22:00, 02:00, 06:00, 10:00, 14:00 ET. 5m and 1H use standard
    UTC-based resampling (no session alignment needed).
    """
    # Standard resampling for 5m and 1H
    for tf_label, freq in [("5m", "5min"), ("1H", "1h")]:
        resampled = (
            df_1m[["open", "high", "low", "close", "volume"]]
            .resample(freq)
            .agg(OHLCV_AGG)
        )
        resampled = resampled.dropna(subset=["open"])
        resampled = resampled[resampled["volume"] > 0]

        # Carry forward flags
        for flag_col in ("is_roll_date", "is_weekend_gap"):
            if flag_col in df_1m.columns:
                resampled[flag_col] = (
                    df_1m[flag_col].resample(freq).max()
                    .reindex(resampled.index).fillna(False).astype(bool)
                )

        out_path = OUT_DIR / f"NQ_{tf_label}_10yr.parquet"
        resampled.to_parquet(out_path, engine="pyarrow")
        mb = out_path.stat().st_size / 1024 / 1024
        print(f"  {tf_label}: {len(resampled):,} bars -> {out_path.name} ({mb:.1f} MB)")

    # Session-aligned 4H resampling
    resampled_4h = _resample_4h_session_aligned(df_1m)
    out_path = OUT_DIR / f"NQ_4H_10yr.parquet"
    resampled_4h.to_parquet(out_path, engine="pyarrow")
    mb = out_path.stat().st_size / 1024 / 1024
    print(f"  4H: {len(resampled_4h):,} bars -> {out_path.name} ({mb:.1f} MB)")


# ── Verification ──────────────────────────────────────────────────────────

def _verify(continuous: pd.DataFrame, parquet_df: pd.DataFrame, roll_report: list) -> None:
    """Run comprehensive verification checks."""
    print(f"\n  1. Date range: {continuous.index[0]} -> {continuous.index[-1]}")
    print(f"     Row count: {len(continuous):,}")
    print(f"     Trading days: {continuous.index.normalize().nunique()}")

    # Check for NaN
    nan_count = continuous[["Open", "High", "Low", "Close"]].isnull().sum().sum()
    print(f"\n  2. NaN in OHLC: {nan_count}")

    # Check for duplicates
    dup_count = continuous.index.duplicated().sum()
    print(f"     Duplicate timestamps: {dup_count}")

    # Verify rollover gaps are smooth
    print(f"\n  3. Rollover gap smoothness:")
    for r in roll_report:
        gap_pct = abs(r["gap_pct"])
        status = "OK" if gap_pct < 0.1 else ("WARN" if gap_pct < 0.5 else "HIGH")
        print(f"     {r['transition']}: gap={r['gap']:+.2f} ({r['gap_pct']:+.3f}%) [{status}]")

    # Verify price ranges by era (rough sanity: NQ ~4000 in 2016, ~16000 in 2021)
    print(f"\n  4. Price sanity by era (unadjusted close from parquet):")
    eras = [
        ("2016", 3500, 5500),
        ("2017", 4500, 6500),
        ("2018", 5500, 8000),
        ("2019", 6000, 9000),
        ("2020", 6500, 14000),
        ("2021", 12000, 17000),
        ("2022", 10500, 17000),
        ("2023", 10500, 17500),
        ("2024", 15000, 22500),
        ("2025", 18000, 25000),
        ("2026", 18000, 26000),
    ]
    for year, lo_exp, hi_exp in eras:
        mask = parquet_df.index.year == int(year)
        if mask.sum() == 0:
            print(f"     {year}: no data")
            continue
        sub = parquet_df.loc[mask, "unadjusted_close"]
        lo, hi = sub.min(), sub.max()
        ok = lo >= lo_exp * 0.8 and hi <= hi_exp * 1.2  # 20% tolerance
        tag = "OK" if ok else "CHECK"
        print(f"     {year}: unadj close {lo:.0f} - {hi:.0f}  (expect {lo_exp}-{hi_exp}) [{tag}]")

    # Compare overlap period with existing 4-year data
    print(f"\n  5. Overlap comparison with existing NQ_1min.parquet:")
    existing_path = Path("/Users/mac/project/nq quant/data/NQ_1min.parquet")
    if existing_path.exists():
        existing = pd.read_parquet(existing_path, engine="pyarrow")
        # Find overlap range
        overlap_start = max(parquet_df.index.min(), existing.index.min())
        overlap_end = min(parquet_df.index.max(), existing.index.max())
        print(f"     Overlap range: {overlap_start} -> {overlap_end}")

        new_overlap = parquet_df.loc[overlap_start:overlap_end]
        old_overlap = existing.loc[overlap_start:overlap_end]

        # Find common timestamps
        common_ts = new_overlap.index.intersection(old_overlap.index)
        print(f"     Common bars: {len(common_ts):,}")

        if len(common_ts) > 0:
            new_common = new_overlap.loc[common_ts]
            old_common = old_overlap.loc[common_ts]

            # Check close price differences
            close_diff = (new_common["close"] - old_common["close"]).abs()
            max_diff = close_diff.max()
            mean_diff = close_diff.mean()
            pct_exact = (close_diff < 0.01).mean() * 100

            print(f"     Close price diff: max={max_diff:.4f}, mean={mean_diff:.4f}")
            print(f"     Exact match (< 0.01): {pct_exact:.1f}%")

            if pct_exact > 99.0:
                print(f"     MATCH: Overlap data is consistent with existing dataset")
            else:
                print(f"     WARNING: Significant differences found in overlap period")
    else:
        print(f"     Existing file not found, skipping overlap comparison")


if __name__ == "__main__":
    # Required for multiprocessing on macOS
    mp.set_start_method("fork", force=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
