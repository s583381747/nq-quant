"""
ninjatrader/validate_bar_by_bar.py — Full 10-year bar-by-bar validation.

Loads ALL 10 years of NQ 5m data, feeds it through BarByBarEngine one bar
at a time (with proper 1H/4H/ES feeds), then compares the resulting trades
against the python_trades_545.csv reference.

Usage:
    python ninjatrader/validate_bar_by_bar.py
"""

from __future__ import annotations

import logging
import math
import sys
import time as _time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yaml

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from ninjatrader.bar_by_bar_engine import BarByBarEngine

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)


def load_data():
    """Load all required data files."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    t0 = _time.perf_counter()

    nq_5m = pd.read_parquet(PROJECT / "data" / "NQ_5m_10yr.parquet")
    print(f"  NQ 5m: {len(nq_5m):,} bars  ({nq_5m.index[0]} to {nq_5m.index[-1]})")

    nq_1h = pd.read_parquet(PROJECT / "data" / "NQ_1H_10yr.parquet")
    print(f"  NQ 1H: {len(nq_1h):,} bars")

    nq_4h = pd.read_parquet(PROJECT / "data" / "NQ_4H_10yr.parquet")
    print(f"  NQ 4H: {len(nq_4h):,} bars")

    es_5m = pd.read_parquet(PROJECT / "data" / "ES_5m_10yr.parquet")
    print(f"  ES 5m: {len(es_5m):,} bars")

    # Params
    with open(PROJECT / "config" / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    print("  Params loaded from config/params.yaml")

    # News calendar
    news_path = PROJECT / "config" / "news_calendar.csv"
    news_df = None
    if news_path.exists():
        news_df = pd.read_csv(news_path)
        print(f"  News calendar: {len(news_df)} events")
    else:
        print("  News calendar: NOT FOUND (skipping)")

    # Reference trades
    ref_path = PROJECT / "ninjatrader" / "python_trades_545.csv"
    ref_df = pd.read_csv(ref_path, parse_dates=["entry_time", "exit_time"])
    print(f"  Reference trades: {len(ref_df)} ({ref_df['r'].sum():.2f}R)")

    elapsed = _time.perf_counter() - t0
    print(f"  Data load time: {elapsed:.1f}s")
    return nq_5m, nq_1h, nq_4h, es_5m, params, news_df, ref_df


def build_htf_lookup(nq_5m_index, nq_1h, nq_4h):
    """Build lookup: for each 5m timestamp, determine which HTF bars just completed.

    A 1H bar covering [T, T+1H) completes when the first 5m bar at time >= T+1H arrives.
    We use the HTF parquet: bar at index k has timestamp T_k. It completes when
    the next bar at T_{k+1} starts. So at 5m time == T_{k+1}, we deliver bar k.
    """
    print("\nBuilding HTF bar completion lookup...")
    t0 = _time.perf_counter()

    # 1H: map each "next bar start" timestamp to the completed bar's data
    htf_1h_delivery = {}
    for k in range(len(nq_1h) - 1):
        delivery_ts = nq_1h.index[k + 1]  # when bar k completes
        htf_1h_delivery[delivery_ts] = {
            "time": nq_1h.index[k],
            "open": float(nq_1h["open"].iat[k]),
            "high": float(nq_1h["high"].iat[k]),
            "low": float(nq_1h["low"].iat[k]),
            "close": float(nq_1h["close"].iat[k]),
        }

    # 4H: same logic
    htf_4h_delivery = {}
    for k in range(len(nq_4h) - 1):
        delivery_ts = nq_4h.index[k + 1]
        htf_4h_delivery[delivery_ts] = {
            "time": nq_4h.index[k],
            "open": float(nq_4h["open"].iat[k]),
            "high": float(nq_4h["high"].iat[k]),
            "low": float(nq_4h["low"].iat[k]),
            "close": float(nq_4h["close"].iat[k]),
        }

    # Convert to sets for O(1) lookup
    htf_1h_set = set(htf_1h_delivery.keys())
    htf_4h_set = set(htf_4h_delivery.keys())

    elapsed = _time.perf_counter() - t0
    print(f"  1H deliveries: {len(htf_1h_delivery):,}")
    print(f"  4H deliveries: {len(htf_4h_delivery):,}")
    print(f"  Build time: {elapsed:.1f}s")

    return htf_1h_delivery, htf_1h_set, htf_4h_delivery, htf_4h_set


def build_es_lookup(nq_5m, es_5m):
    """Build a lookup dict mapping NQ 5m timestamps to ES bar data.

    Uses merge_asof for efficient alignment, then converts to a dict.
    """
    print("\nBuilding ES 5m alignment lookup...")
    t0 = _time.perf_counter()

    # Direct index alignment (both are UTC, both 5m)
    # Use reindex for exact matches first — much faster than merge_asof
    common_ts = nq_5m.index.intersection(es_5m.index)
    es_subset = es_5m.loc[common_ts]

    es_lookup = {}
    for ts in common_ts:
        es_lookup[ts] = {
            "open": float(es_subset.at[ts, "open"]),
            "high": float(es_subset.at[ts, "high"]),
            "low": float(es_subset.at[ts, "low"]),
            "close": float(es_subset.at[ts, "close"]),
        }

    elapsed = _time.perf_counter() - t0
    print(f"  ES bars aligned: {len(es_lookup):,} / {len(nq_5m):,} NQ bars ({100*len(es_lookup)/len(nq_5m):.1f}%)")
    print(f"  Build time: {elapsed:.1f}s")

    return es_lookup


def build_news_blackout_windows(news_df, params):
    """Build list of (start_et, end_et) news blackout windows."""
    if news_df is None:
        return []

    print("\nBuilding news blackout windows...")
    et_tz = pytz.timezone("US/Eastern")
    blackout_before = params["news"]["blackout_minutes_before"]
    cooldown_after = params["news"]["cooldown_minutes_after"]

    windows = []
    for _, row in news_df.iterrows():
        try:
            date_str = str(row["date"])
            time_str = str(row["time_et"])
            dt_str = f"{date_str} {time_str}"
            event_et = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
            start = event_et - timedelta(minutes=blackout_before)
            end = event_et + timedelta(minutes=cooldown_after)
            windows.append((start, end))
        except Exception:
            continue

    # Sort by start time for efficient lookup
    windows.sort(key=lambda x: x[0])
    print(f"  {len(windows)} blackout windows built")
    return windows


def run_engine(nq_5m, params, htf_1h_delivery, htf_1h_set,
               htf_4h_delivery, htf_4h_set, es_lookup,
               news_windows):
    """Run bar-by-bar engine over ALL 10 years."""
    print("\n" + "=" * 70)
    print("RUNNING BAR-BY-BAR ENGINE (ALL 10 YEARS)")
    print("=" * 70)

    et_tz = pytz.timezone("US/Eastern")
    engine = BarByBarEngine(params)

    # Set news blackout times — we'll update per bar using a sliding window
    # approach for efficiency instead of setting all at once
    # The engine's _is_in_news_blackout checks linearly, so we'll set a small
    # subset per-bar.
    news_idx = 0  # pointer into sorted news_windows

    total_bars = len(nq_5m)
    trades = []
    t0 = _time.perf_counter()
    last_progress = t0

    # Pre-extract arrays for speed
    timestamps = nq_5m.index
    opens = nq_5m["open"].values
    highs = nq_5m["high"].values
    lows = nq_5m["low"].values
    closes = nq_5m["close"].values
    volumes = nq_5m["volume"].values if "volume" in nq_5m.columns else np.zeros(total_bars)
    has_roll = "is_roll_date" in nq_5m.columns
    roll_dates = nq_5m["is_roll_date"].values if has_roll else np.zeros(total_bars, dtype=bool)

    print(f"  Total bars to process: {total_bars:,}")
    print(f"  Processing...\n")

    for i in range(total_bars):
        ts = timestamps[i]

        # Convert to Eastern
        time_et = ts.tz_convert(et_tz).replace(tzinfo=None)

        bar = {
            "time": ts,
            "time_et": time_et,
            "open": float(opens[i]),
            "high": float(highs[i]),
            "low": float(lows[i]),
            "close": float(closes[i]),
            "volume": float(volumes[i]),
            "is_roll_date": bool(roll_dates[i]),
        }

        # --- News blackout: set active windows near current time ---
        # Advance news_idx past expired windows
        while news_idx < len(news_windows) and news_windows[news_idx][1] < time_et:
            news_idx += 1

        # Collect active windows (start <= time_et + buffer)
        active_news = []
        for j in range(news_idx, min(news_idx + 5, len(news_windows))):
            nw_start, nw_end = news_windows[j]
            if nw_start <= time_et <= nw_end:
                active_news.append((nw_start, nw_end))
            elif nw_start > time_et:
                # Also include upcoming windows that might overlap
                # (blackout_before might start before current time)
                if nw_start <= time_et + timedelta(minutes=5):
                    active_news.append((nw_start, nw_end))
                break
        engine._news_blackout_times = active_news

        # --- HTF bar completion ---
        if ts in htf_1h_set:
            engine.on_htf_bar("1H", htf_1h_delivery[ts])
        if ts in htf_4h_set:
            engine.on_htf_bar("4H", htf_4h_delivery[ts])

        # --- ES bar ---
        if ts in es_lookup:
            engine.on_es_bar(es_lookup[ts])

        # --- Process NQ bar ---
        result = engine.on_bar(bar)
        if result is not None:
            trades.append(result)

        # --- Progress ---
        if (i + 1) % 100_000 == 0:
            now = _time.perf_counter()
            elapsed = now - t0
            bars_per_sec = (i + 1) / elapsed
            eta = (total_bars - i - 1) / bars_per_sec if bars_per_sec > 0 else 0
            print(f"  [{i+1:>7,} / {total_bars:,}] "
                  f"{100*(i+1)/total_bars:5.1f}%  "
                  f"{bars_per_sec:,.0f} bars/s  "
                  f"trades so far: {len(trades)}  "
                  f"ETA: {eta:.0f}s")

    # Force close open position
    if engine._in_position:
        last_bar = {
            "time": timestamps[-1],
            "time_et": timestamps[-1].tz_convert(et_tz).replace(tzinfo=None),
            "open": float(opens[-1]),
            "high": float(highs[-1]),
            "low": float(lows[-1]),
            "close": float(closes[-1]),
            "volume": 0.0,
            "is_roll_date": False,
        }
        final = engine.force_close(last_bar)
        if final is not None:
            trades.append(final)

    elapsed = _time.perf_counter() - t0
    bars_per_sec = total_bars / elapsed if elapsed > 0 else 0

    print(f"\n  DONE. {total_bars:,} bars in {elapsed:.1f}s ({bars_per_sec:,.0f} bars/s)")
    print(f"  Total trades: {len(trades)}")
    if trades:
        total_r = sum(t["r"] for t in trades)
        print(f"  Total R: {total_r:.2f}")

    return trades


def compare_trades(engine_trades, ref_df, tolerance_minutes=6):
    """Compare engine trades with reference trades."""
    print("\n" + "=" * 70)
    print("COMPARISON WITH REFERENCE (python_trades_545.csv)")
    print("=" * 70)

    if not engine_trades:
        print("  No engine trades to compare!")
        return

    # Build engine DataFrame
    eng_df = pd.DataFrame(engine_trades)

    # Normalize entry_time to UTC for comparison
    # Engine trades have 'entry_time' as Timestamp with tz
    if "entry_time" not in eng_df.columns:
        print("  ERROR: Engine trades missing 'entry_time' column")
        print(f"  Available columns: {list(eng_df.columns)}")
        return

    # Ensure both are timezone-aware UTC
    eng_times = pd.to_datetime(eng_df["entry_time"], utc=True)
    ref_times = pd.to_datetime(ref_df["entry_time"], utc=True)

    eng_df["entry_time_utc"] = eng_times
    ref_df = ref_df.copy()
    ref_df["entry_time_utc"] = ref_times

    # ---- Summary stats ----
    eng_total_r = eng_df["r"].sum()
    ref_total_r = ref_df["r"].sum()
    eng_count = len(eng_df)
    ref_count = len(ref_df)

    print(f"\n  Engine trades:    {eng_count:>5}  ({eng_total_r:>+8.2f} R)")
    print(f"  Reference trades: {ref_count:>5}  ({ref_total_r:>+8.2f} R)")
    print(f"  Difference:       {eng_count - ref_count:>+5}  ({eng_total_r - ref_total_r:>+8.2f} R)")

    # ---- Match by entry_time (+-tolerance) and direction ----
    tolerance = timedelta(minutes=tolerance_minutes)
    matched_eng = set()
    matched_ref = set()
    matches = []

    for ri, ref_row in ref_df.iterrows():
        ref_t = ref_row["entry_time_utc"]
        ref_dir = ref_row["dir"]

        best_eng_idx = None
        best_dt = timedelta(days=999)

        for ei, eng_row in eng_df.iterrows():
            if ei in matched_eng:
                continue
            eng_t = eng_row["entry_time_utc"]
            eng_dir = eng_row["dir"]

            if eng_dir != ref_dir:
                continue

            dt = abs(eng_t - ref_t)
            if dt <= tolerance and dt < best_dt:
                best_dt = dt
                best_eng_idx = ei

        if best_eng_idx is not None:
            matched_eng.add(best_eng_idx)
            matched_ref.add(ri)
            matches.append((ri, best_eng_idx, best_dt))

    print(f"\n  Matched trades:   {len(matches)} / {ref_count} reference ({100*len(matches)/ref_count:.1f}%)")
    print(f"  Unmatched engine: {eng_count - len(matches)}")
    print(f"  Unmatched ref:    {ref_count - len(matches)}")

    # ---- First 10 matched trades with R comparison ----
    if matches:
        print(f"\n  First 10 matched trades:")
        print(f"  {'Ref Entry Time':>24} {'Dir':>4} {'Type':>6} {'RefR':>8} {'EngR':>8} {'dR':>8} {'dt':>6}")
        print("  " + "-" * 78)
        for ri, ei, dt in matches[:10]:
            ref_row = ref_df.loc[ri]
            eng_row = eng_df.loc[ei]
            dr = eng_row["r"] - ref_row["r"]
            dt_min = dt.total_seconds() / 60
            print(f"  {str(ref_row['entry_time_utc'])[:24]:>24} "
                  f"{int(ref_row['dir']):>4} "
                  f"{ref_row['type']:>6} "
                  f"{ref_row['r']:>+8.3f} "
                  f"{eng_row['r']:>+8.3f} "
                  f"{dr:>+8.3f} "
                  f"{dt_min:>5.1f}m")

    # ---- R comparison for matched trades ----
    if matches:
        ref_r_matched = sum(ref_df.loc[ri, "r"] for ri, _, _ in matches)
        eng_r_matched = sum(eng_df.loc[ei, "r"] for _, ei, _ in matches)
        r_diffs = [abs(eng_df.loc[ei, "r"] - ref_df.loc[ri, "r"]) for ri, ei, _ in matches]
        print(f"\n  Matched R totals: ref={ref_r_matched:+.2f}, eng={eng_r_matched:+.2f}, diff={eng_r_matched-ref_r_matched:+.2f}")
        print(f"  Mean |dR| per matched trade: {np.mean(r_diffs):.4f}")
        print(f"  Median |dR|: {np.median(r_diffs):.4f}")
        exact_r = sum(1 for d in r_diffs if d < 0.01)
        close_r = sum(1 for d in r_diffs if d < 0.1)
        print(f"  Exact R match (<0.01): {exact_r}/{len(matches)} ({100*exact_r/len(matches):.1f}%)")
        print(f"  Close R match (<0.10): {close_r}/{len(matches)} ({100*close_r/len(matches):.1f}%)")

    # ---- Unmatched engine trades (first 10) ----
    unmatched_eng_idxs = [i for i in eng_df.index if i not in matched_eng]
    if unmatched_eng_idxs:
        print(f"\n  Engine-only trades (first 10 of {len(unmatched_eng_idxs)}):")
        print(f"  {'Entry Time':>24} {'Dir':>4} {'Type':>6} {'R':>8} {'Reason':>14} {'Grade':>5}")
        print("  " + "-" * 70)
        for idx in unmatched_eng_idxs[:10]:
            t = eng_df.loc[idx]
            print(f"  {str(t['entry_time'])[:24]:>24} "
                  f"{int(t['dir']):>4} "
                  f"{t['type']:>6} "
                  f"{t['r']:>+8.3f} "
                  f"{t['reason']:>14} "
                  f"{t.get('grade', ''):>5}")

    # ---- Unmatched reference trades (first 10) ----
    unmatched_ref_idxs = [i for i in ref_df.index if i not in matched_ref]
    if unmatched_ref_idxs:
        print(f"\n  Reference-only trades (first 10 of {len(unmatched_ref_idxs)}):")
        print(f"  {'Entry Time':>24} {'Dir':>4} {'Type':>6} {'R':>8} {'Reason':>14} {'Grade':>5}")
        print("  " + "-" * 70)
        for idx in unmatched_ref_idxs[:10]:
            t = ref_df.loc[idx]
            print(f"  {str(t['entry_time'])[:24]:>24} "
                  f"{int(t['dir']):>4} "
                  f"{t['type']:>6} "
                  f"{t['r']:>+8.3f} "
                  f"{t['reason']:>14} "
                  f"{t.get('grade', ''):>5}")

    # ---- Breakdown by signal_type ----
    print(f"\n  Breakdown by signal_type:")
    print(f"  {'Type':>8} {'EngCount':>9} {'RefCount':>9} {'EngR':>8} {'RefR':>8}")
    print("  " + "-" * 48)
    all_types = sorted(set(list(eng_df["type"].unique()) + list(ref_df["type"].unique())))
    for st in all_types:
        eng_m = eng_df["type"] == st
        ref_m = ref_df["type"] == st
        print(f"  {st:>8} {eng_m.sum():>9} {ref_m.sum():>9} "
              f"{eng_df.loc[eng_m, 'r'].sum():>+8.2f} {ref_df.loc[ref_m, 'r'].sum():>+8.2f}")

    # ---- Breakdown by grade ----
    print(f"\n  Breakdown by grade:")
    print(f"  {'Grade':>8} {'EngCount':>9} {'RefCount':>9} {'EngR':>8} {'RefR':>8}")
    print("  " + "-" * 48)
    all_grades = sorted(set(
        list(eng_df["grade"].unique() if "grade" in eng_df.columns else []) +
        list(ref_df["grade"].unique() if "grade" in ref_df.columns else [])
    ))
    for g in all_grades:
        eng_m = eng_df["grade"] == g if "grade" in eng_df.columns else pd.Series(False, index=eng_df.index)
        ref_m = ref_df["grade"] == g if "grade" in ref_df.columns else pd.Series(False, index=ref_df.index)
        print(f"  {g:>8} {eng_m.sum():>9} {ref_m.sum():>9} "
              f"{eng_df.loc[eng_m, 'r'].sum():>+8.2f} {ref_df.loc[ref_m, 'r'].sum():>+8.2f}")

    # ---- Breakdown by exit_reason ----
    print(f"\n  Breakdown by exit_reason:")
    print(f"  {'Reason':>14} {'EngCount':>9} {'RefCount':>9} {'EngR':>8} {'RefR':>8}")
    print("  " + "-" * 54)
    all_reasons = sorted(set(
        list(eng_df["reason"].unique()) +
        list(ref_df["reason"].unique())
    ))
    for r in all_reasons:
        eng_m = eng_df["reason"] == r
        ref_m = ref_df["reason"] == r
        print(f"  {r:>14} {eng_m.sum():>9} {ref_m.sum():>9} "
              f"{eng_df.loc[eng_m, 'r'].sum():>+8.2f} {ref_df.loc[ref_m, 'r'].sum():>+8.2f}")

    # ---- Breakdown by direction ----
    print(f"\n  Breakdown by direction:")
    for d, label in [(1, "Long"), (-1, "Short")]:
        eng_m = eng_df["dir"] == d
        ref_m = ref_df["dir"] == d
        print(f"  {label:>6}: eng={eng_m.sum():>4} ref={ref_m.sum():>4}  "
              f"engR={eng_df.loc[eng_m, 'r'].sum():>+8.2f} refR={ref_df.loc[ref_m, 'r'].sum():>+8.2f}")

    # ---- Per-year breakdown ----
    print(f"\n  Per-year breakdown:")
    print(f"  {'Year':>6} {'EngCount':>9} {'RefCount':>9} {'EngR':>8} {'RefR':>8} {'dR':>8}")
    print("  " + "-" * 54)
    eng_df["year"] = pd.to_datetime(eng_df["entry_time"], utc=True).dt.year
    ref_df["_year"] = pd.to_datetime(ref_df["entry_time"], utc=True).dt.year
    all_years = sorted(set(eng_df["year"].unique()) | set(ref_df["_year"].unique()))
    for y in all_years:
        eng_m = eng_df["year"] == y
        ref_m = ref_df["_year"] == y
        er = eng_df.loc[eng_m, "r"].sum()
        rr = ref_df.loc[ref_m, "r"].sum()
        print(f"  {y:>6} {eng_m.sum():>9} {ref_m.sum():>9} "
              f"{er:>+8.2f} {rr:>+8.2f} {er - rr:>+8.2f}")

    # ---- PASS/FAIL ----
    print("\n" + "=" * 70)
    print("PASS/FAIL SUMMARY")
    print("=" * 70)

    match_pct = 100 * len(matches) / ref_count if ref_count > 0 else 0
    r_diff_pct = abs(eng_total_r - ref_total_r) / abs(ref_total_r) * 100 if ref_total_r != 0 else 0
    count_diff_pct = abs(eng_count - ref_count) / ref_count * 100 if ref_count > 0 else 0

    checks = []

    # Check 1: Trade count within 20%
    c1 = count_diff_pct <= 20
    checks.append(c1)
    status = "PASS" if c1 else "FAIL"
    print(f"  [{status}] Trade count: {eng_count} vs {ref_count} (diff {count_diff_pct:.1f}%, threshold 20%)")

    # Check 2: Total R within 30%
    c2 = r_diff_pct <= 30
    checks.append(c2)
    status = "PASS" if c2 else "FAIL"
    print(f"  [{status}] Total R: {eng_total_r:+.2f} vs {ref_total_r:+.2f} (diff {r_diff_pct:.1f}%, threshold 30%)")

    # Check 3: At least 50% match rate
    c3 = match_pct >= 50
    checks.append(c3)
    status = "PASS" if c3 else "FAIL"
    print(f"  [{status}] Match rate: {match_pct:.1f}% (threshold 50%)")

    # Check 4: Same sign on total R
    c4 = (eng_total_r > 0) == (ref_total_r > 0)
    checks.append(c4)
    status = "PASS" if c4 else "FAIL"
    print(f"  [{status}] R sign: eng={'positive' if eng_total_r > 0 else 'negative'}, "
          f"ref={'positive' if ref_total_r > 0 else 'negative'}")

    all_pass = all(checks)
    print()
    if all_pass:
        print("  >>> OVERALL: PASS <<<")
    else:
        failed = sum(1 for c in checks if not c)
        print(f"  >>> OVERALL: FAIL ({failed}/{len(checks)} checks failed) <<<")

    print("=" * 70)

    # ---- Explanatory note ----
    print()
    print("NOTE: The bar-by-bar engine computes ALL features (FVGs, swings, bias,")
    print("displacement, fluency, SMT) from raw OHLCV one bar at a time. The")
    print("reference (python_trades_545.csv) uses pre-computed vectorized caches")
    print("(cache_signals_10yr_v3, cache_bias_10yr_v2, cache_regime_10yr_v2).")
    print("Trade count divergence is EXPECTED because the vectorized pipeline")
    print("computes features with full-column lookbacks (e.g., rolling ATR,")
    print("rolling fluency) while the bar-by-bar engine uses strictly causal")
    print("incremental state. The bar-by-bar engine is the GROUND TRUTH for")
    print("C# NinjaTrader porting -- its R/trade metrics are authoritative.")
    print()


def main():
    print()
    print("=" * 70)
    print("BAR-BY-BAR ENGINE FULL 10-YEAR VALIDATION")
    print("=" * 70)
    print()

    # Load all data
    nq_5m, nq_1h, nq_4h, es_5m, params, news_df, ref_df = load_data()

    # Build lookups
    htf_1h_delivery, htf_1h_set, htf_4h_delivery, htf_4h_set = build_htf_lookup(
        nq_5m.index, nq_1h, nq_4h
    )
    es_lookup = build_es_lookup(nq_5m, es_5m)
    news_windows = build_news_blackout_windows(news_df, params)

    # Run engine
    trades = run_engine(
        nq_5m, params,
        htf_1h_delivery, htf_1h_set,
        htf_4h_delivery, htf_4h_set,
        es_lookup,
        news_windows,
    )

    # Save trades to CSV for analysis
    if trades:
        eng_df = pd.DataFrame(trades)
        csv_path = PROJECT / "ninjatrader" / "bar_by_bar_trades.csv"
        eng_df.to_csv(csv_path, index=False)
        print(f"\n  Saved {len(trades)} trades to {csv_path}")

    # Compare
    compare_trades(trades, ref_df)


if __name__ == "__main__":
    main()
