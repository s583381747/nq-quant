"""
features/news_filter.py -- News event filter for high-impact economic events.

Blocks new entries during blackout windows around high-impact events
(CPI, FOMC, NFP, GDP, PCE, PPI, Jobless Claims).

Per CLAUDE.md section 11.5:
  - 60-min blackout BEFORE high-impact events -> no new entries
  - 5-min cooldown AFTER event -> trading resumes
  - Existing positions NOT affected (only blocks new entries)

All parameters come from config/params.yaml under the 'news' key.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_CALENDAR_PATH = str(
    Path(__file__).resolve().parent.parent / "config" / "news_calendar.csv"
)


def load_news_calendar(path: str = _DEFAULT_CALENDAR_PATH) -> pd.DataFrame:
    """Load and parse news calendar CSV.

    Expects columns: date, time_et, event, impact.
    Combines date + time_et into timezone-aware UTC datetimes.

    Parameters
    ----------
    path : str
        Path to news_calendar.csv.

    Returns
    -------
    pd.DataFrame
        Calendar with added columns: datetime_et (US/Eastern), datetime_utc (UTC).
    """
    cal = pd.read_csv(path)

    # Validate required columns
    required = {"date", "time_et", "event", "impact"}
    missing = required - set(cal.columns)
    if missing:
        raise ValueError(f"news_calendar.csv missing columns: {missing}")

    # Parse date + time_et into ET-aware datetime, then convert to UTC
    dt_str = cal["date"].astype(str) + " " + cal["time_et"].astype(str)
    cal["datetime_et"] = pd.to_datetime(dt_str).dt.tz_localize(
        "US/Eastern", ambiguous="NaT", nonexistent="shift_forward"
    )
    cal["datetime_utc"] = cal["datetime_et"].dt.tz_convert("UTC")

    # Drop rows where datetime parsing failed (NaT from ambiguous DST)
    n_before = len(cal)
    cal = cal.dropna(subset=["datetime_utc"]).reset_index(drop=True)
    n_dropped = n_before - len(cal)
    if n_dropped > 0:
        logger.warning("Dropped %d events with ambiguous DST times", n_dropped)

    logger.info("Loaded %d news events from %s", len(cal), path)
    return cal


def build_news_blackout_mask(
    index: pd.DatetimeIndex,
    calendar_path: str = _DEFAULT_CALENDAR_PATH,
    blackout_minutes_before: int = 60,
    cooldown_minutes_after: int = 5,
) -> pd.Series:
    """Build a boolean Series aligned to `index` where True = no-trade zone.

    Uses vectorized numpy operations for efficiency with large datasets
    (270 events x 711K bars).

    Parameters
    ----------
    index : pd.DatetimeIndex
        DatetimeIndex of the price data (must be UTC-aware).
    calendar_path : str
        Path to news_calendar.csv.
    blackout_minutes_before : int
        Minutes before event to start blackout (default 60).
    cooldown_minutes_after : int
        Minutes after event before trading resumes (default 5).

    Returns
    -------
    pd.Series[bool]
        True = bar is in a news blackout window, no entries allowed.
    """
    if blackout_minutes_before <= 0:
        return pd.Series(False, index=index, dtype=bool)

    cal = load_news_calendar(calendar_path)

    if len(cal) == 0:
        logger.warning("Empty news calendar -- no blackout applied")
        return pd.Series(False, index=index, dtype=bool)

    # Convert event times and bar index to int64 in a common resolution.
    # pandas 2.0+ uses datetime64[us] (microseconds) by default, but older
    # versions use datetime64[ns].  Detect resolution from the bar index and
    # convert everything consistently.
    bar_dt64 = index.values  # numpy datetime64 array
    bar_int = bar_dt64.view("int64")  # int64 in native resolution

    # Detect resolution: check the dtype string, e.g. "datetime64[us]" or "datetime64[ns]"
    reso_str = str(bar_dt64.dtype)  # e.g. "datetime64[us, UTC]" or "datetime64[ns, UTC]"
    if "us" in reso_str:
        _UNIT_PER_MINUTE = np.int64(60) * 1_000_000       # microseconds per minute
    else:
        _UNIT_PER_MINUTE = np.int64(60) * 1_000_000_000   # nanoseconds per minute

    # Convert event timestamps to the same resolution as bar index
    event_utc_values = cal["datetime_utc"].values.astype(bar_dt64.dtype)
    event_int = event_utc_values.view("int64")  # shape (E,)

    blackout_before_units = np.int64(blackout_minutes_before) * _UNIT_PER_MINUTE
    cooldown_after_units = np.int64(cooldown_minutes_after) * _UNIT_PER_MINUTE

    # Compute blackout start/end for each event
    blackout_starts = event_int - blackout_before_units  # shape (E,)
    blackout_ends = event_int + cooldown_after_units      # shape (E,)

    # Narrow the search: only check events whose blackout window overlaps
    # with the data range, to avoid unnecessary comparisons
    data_start = bar_int[0]
    data_end = bar_int[-1]

    # Keep events whose blackout window overlaps with data range
    overlap_mask = (blackout_ends >= data_start) & (blackout_starts <= data_end)
    blackout_starts = blackout_starts[overlap_mask]
    blackout_ends = blackout_ends[overlap_mask]

    n_events = len(blackout_starts)
    if n_events == 0:
        return pd.Series(False, index=index, dtype=bool)

    logger.info(
        "Checking %d events against %d bars (blackout=%dmin before, %dmin after)",
        n_events, len(bar_int), blackout_minutes_before, cooldown_minutes_after,
    )

    # Vectorized approach: process events in chunks to control memory
    # For 270 events x 711K bars, full broadcasting would use ~1.5GB.
    # Chunk events to keep peak memory reasonable.
    CHUNK_SIZE = 50  # events per chunk
    result = np.zeros(len(bar_int), dtype=bool)

    for chunk_start in range(0, n_events, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_events)
        starts_chunk = blackout_starts[chunk_start:chunk_end]  # (C,)
        ends_chunk = blackout_ends[chunk_start:chunk_end]        # (C,)

        # Broadcasting: bar_int (N,1) vs starts/ends (C,)
        # Result: (N, C) boolean matrix -> reduce along event axis
        in_window = (
            (bar_int[:, np.newaxis] >= starts_chunk[np.newaxis, :])
            & (bar_int[:, np.newaxis] <= ends_chunk[np.newaxis, :])
        )
        result |= in_window.any(axis=1)

    return pd.Series(result, index=index, dtype=bool)
