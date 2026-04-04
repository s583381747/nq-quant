"""
Session labeling, session-level computation, and opening range observation (ORM).

Identifies Asia / London / NY sessions based on US/Eastern time,
computes completed session highs/lows (no forward-looking), overnight range,
NY open price, and the opening-range observation window.

All internal timestamps are UTC.  ET is used only for classification.

References: CLAUDE.md sections 2, 3.3, 11.7
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_params(path: str | Path = "config/params.yaml") -> dict[str, Any]:
    """Load the central params.yaml file."""
    with open(path) as f:
        return yaml.safe_load(f)


def _to_et(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert a UTC DatetimeIndex to US/Eastern (handles DST automatically)."""
    return idx.tz_convert("US/Eastern")


def _parse_hhmm(time_str: str) -> tuple[int, int]:
    """Parse 'HH:MM' string into (hour, minute)."""
    parts = time_str.split(":")
    return int(parts[0]), int(parts[1])


def _frac(h: int, m: int) -> float:
    """Convert hours and minutes to fractional hours."""
    return h + m / 60.0


def _et_frac(et_index: pd.DatetimeIndex) -> np.ndarray:
    """Return fractional-hour array from an ET DatetimeIndex."""
    return et_index.hour + et_index.minute / 60.0


def _session_date_key(et_index: pd.DatetimeIndex, wraps_midnight: bool) -> np.ndarray:
    """Build a session-date grouping key (as int YYYYMMDD for fast groupby).

    For sessions that cross midnight (Asia 18:00-03:00, overnight 18:00-09:30),
    bars before noon ET are attributed to the session that *started* the
    previous evening (date - 1 day).

    Returns an int64 array of YYYYMMDD values.
    """
    et_dates = et_index.date
    # Convert dates to integer YYYYMMDD for fast groupby
    year = et_index.year
    month = et_index.month
    day = et_index.day
    key = year * 10000 + month * 100 + day  # YYYYMMDD

    if wraps_midnight:
        before_noon = et_index.hour < 12
        if before_noon.any():
            key_arr = key.to_numpy(copy=True) if hasattr(key, 'to_numpy') else np.array(key, copy=True)
            # Vectorized date-1: normalize to midnight then subtract 1 day
            affected_ts = et_index[before_noon].normalize() - pd.Timedelta(days=1)
            key_arr[before_noon] = (
                affected_ts.year * 10000
                + affected_ts.month * 100
                + affected_ts.day
            )
            return key_arr

    return key.to_numpy() if hasattr(key, 'to_numpy') else np.array(key)


# ---------------------------------------------------------------------------
# 1. label_sessions
# ---------------------------------------------------------------------------

def label_sessions(df: pd.DataFrame, params: dict) -> pd.Series:
    """Assign a session label to every 1-min bar.

    Labels: 'asia', 'london', 'ny', 'overnight_other'

    Session boundaries (all US/Eastern, from params.yaml):
        Asia   : asia_start (18:00) - asia_end (03:00)
        London : london_start (03:00) - london_end (09:30)
        NY     : ny_prime_start (09:30) - ny_end (16:00)
        Other  : 16:00 - 18:00 (gap between NY close and Asia open)

    Parameters
    ----------
    df : pd.DataFrame
        1-min OHLCV with UTC DatetimeIndex.
    params : dict
        Full params dict (must contain 'sessions' key).

    Returns
    -------
    pd.Series
        String labels aligned to df.index.
    """
    sp = params["sessions"]
    asia_start_f = _frac(*_parse_hhmm(sp["asia_start"]))    # 18.0
    asia_end_f = _frac(*_parse_hhmm(sp["asia_end"]))         # 3.0
    london_start_f = _frac(*_parse_hhmm(sp["london_start"])) # 3.0
    london_end_f = _frac(*_parse_hhmm(sp["london_end"]))     # 9.5
    ny_start_f = _frac(*_parse_hhmm(sp["ny_prime_start"]))   # 9.5
    ny_end_f = _frac(*_parse_hhmm(sp["ny_end"]))             # 16.0

    et_index = _to_et(df.index)
    frac = _et_frac(et_index)

    # Build labels array (default: overnight_other = 3)
    # 0=asia, 1=london, 2=ny, 3=overnight_other
    codes = np.full(len(df), 3, dtype=np.int8)

    # Asia wraps midnight: 18:00-23:59 OR 00:00-02:59
    codes[(frac >= asia_start_f) | (frac < asia_end_f)] = 0

    # London: 03:00 - 09:30 (overrides asia at boundary)
    codes[(frac >= london_start_f) & (frac < london_end_f)] = 1

    # NY: 09:30 - 16:00 (overrides london at boundary)
    codes[(frac >= ny_start_f) & (frac < ny_end_f)] = 2

    mapping = {0: "asia", 1: "london", 2: "ny", 3: "overnight_other"}
    labels = pd.Series(
        pd.Categorical([mapping[c] for c in codes],
                        categories=["asia", "london", "ny", "overnight_other"]),
        index=df.index,
    )

    logger.info("Session labels: %s", labels.value_counts().to_dict())
    return labels


# ---------------------------------------------------------------------------
# 2. compute_session_levels  (fully vectorized)
# ---------------------------------------------------------------------------

def _vectorized_session_hl(
    df: pd.DataFrame,
    et_index: pd.DatetimeIndex,
    frac: np.ndarray,
    is_session_mask: np.ndarray,
    session_end_frac: float,
    wraps_midnight: bool,
) -> tuple[pd.Series, pd.Series]:
    """Compute session H/L, stamped at the first bar AFTER session ends.

    Fully vectorized: no per-date Python loops.

    Strategy:
    1. Mask session bars, group by session-date key, compute H/L.
    2. Identify the first bar at or after session_end_frac for each
       (end-)date. For wraps-midnight sessions the end-date is date+1.
    3. Map session H/L to those stamp positions via a join.
    4. Forward-fill.
    """
    n = len(df)
    high_arr = df["high"].values
    low_arr = df["low"].values

    # Session-date key (attributes midnight-crossing bars to start-date)
    sess_key = _session_date_key(et_index, wraps_midnight)

    # --- Compute H/L per session instance ---
    # Only look at session bars
    session_mask = np.asarray(is_session_mask)
    sess_high_vals = np.where(session_mask, high_arr, np.nan)
    sess_low_vals = np.where(session_mask, low_arr, np.nan)

    sess_df = pd.DataFrame({
        "sess_key": sess_key,
        "high": sess_high_vals,
        "low": sess_low_vals,
    })
    sess_df = sess_df.dropna(subset=["high"])

    if sess_df.empty:
        return (
            pd.Series(np.nan, index=df.index, dtype="float64"),
            pd.Series(np.nan, index=df.index, dtype="float64"),
        )

    hl_per_session = sess_df.groupby("sess_key").agg(
        s_high=("high", "max"),
        s_low=("low", "min"),
    )

    # --- Find stamp positions: first bar at/after session_end_frac on the end-date ---
    # For wraps-midnight: end-date = session start-date + 1 day
    # The "end-date key" in YYYYMMDD form:
    if wraps_midnight:
        # Convert sess_key YYYYMMDD ints -> dates -> +1 day -> YYYYMMDD int
        sk_arr = hl_per_session.index.values  # int64 YYYYMMDD
        y = sk_arr // 10000
        m = (sk_arr % 10000) // 100
        d = sk_arr % 100
        end_dates_idx = pd.DatetimeIndex(
            pd.to_datetime(pd.DataFrame({"year": y, "month": m, "day": d}))
        ) + pd.Timedelta(days=1)
        end_key = (
            end_dates_idx.year * 10000
            + end_dates_idx.month * 100
            + end_dates_idx.day
        ).values
    else:
        end_key = hl_per_session.index.values

    # Build a lookup: end_key -> (s_high, s_low)
    hl_lookup = {}
    for i, sk in enumerate(hl_per_session.index.values):
        ek = end_key[i]
        hl_lookup[ek] = (
            hl_per_session.iloc[i]["s_high"],
            hl_per_session.iloc[i]["s_low"],
        )

    # For every bar, compute its date key (no midnight adjustment -- plain calendar date)
    bar_date_key = (
        et_index.year * 10000 + et_index.month * 100 + et_index.day
    )
    bar_date_key = bar_date_key.to_numpy() if hasattr(bar_date_key, 'to_numpy') else np.array(bar_date_key)

    # Find the first bar at/after session_end_frac for each calendar date.
    # A bar qualifies if: frac >= session_end_frac AND its bar_date_key is in end_key set.
    # Among qualifying bars for a given date-key, pick the earliest (lowest index).
    frac_arr = np.asarray(frac)
    qualifies = frac_arr >= session_end_frac

    # Build result arrays
    high_result = np.full(n, np.nan, dtype=np.float64)
    low_result = np.full(n, np.nan, dtype=np.float64)

    # We need: for each end_key value, find the first qualifying bar index.
    # Vectorized approach: create a DataFrame of qualifying bars, groupby date, take first index.
    qual_indices = np.where(qualifies)[0]
    if len(qual_indices) > 0:
        qual_date_keys = bar_date_key[qual_indices]
        qual_df = pd.DataFrame({
            "pos": qual_indices,
            "date_key": qual_date_keys,
        })
        first_per_date = qual_df.groupby("date_key")["pos"].first()

        # Stamp H/L at the first qualifying bar for each end_key
        end_key_set = set(end_key)
        for dk, pos in first_per_date.items():
            if dk in hl_lookup:
                high_result[pos] = hl_lookup[dk][0]
                low_result[pos] = hl_lookup[dk][1]

    # Forward-fill
    high_series = pd.Series(high_result, index=df.index, dtype="float64").ffill()
    low_series = pd.Series(low_result, index=df.index, dtype="float64").ffill()

    return high_series, low_series


def compute_session_levels(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Compute completed session highs/lows and overnight range.

    For each completed session, forward-fill the H/L into subsequent bars.
    Session levels are NOT available during the session itself (no lookahead).

    Columns returned:
        asia_high, asia_low,
        london_high, london_low,
        overnight_high, overnight_low,
        ny_open

    Parameters
    ----------
    df : pd.DataFrame
        1-min OHLCV with UTC DatetimeIndex.
    params : dict
        Full params dict.

    Returns
    -------
    pd.DataFrame
        Session-level columns, same index as df.
    """
    sp = params["sessions"]
    et_index = _to_et(df.index)
    frac = _et_frac(et_index)

    asia_start_f = _frac(*_parse_hhmm(sp["asia_start"]))
    asia_end_f = _frac(*_parse_hhmm(sp["asia_end"]))
    london_start_f = _frac(*_parse_hhmm(sp["london_start"]))
    london_end_f = _frac(*_parse_hhmm(sp["london_end"]))
    overnight_start_f = _frac(*_parse_hhmm(sp["overnight_start"]))
    overnight_end_f = _frac(*_parse_hhmm(sp["overnight_end"]))
    ny_start_f = _frac(*_parse_hhmm(sp["ny_prime_start"]))

    result = pd.DataFrame(index=df.index)

    # --- Asia H/L (18:00 - 03:00 ET) ---
    is_asia = (frac >= asia_start_f) | (frac < asia_end_f)
    result["asia_high"], result["asia_low"] = _vectorized_session_hl(
        df, et_index, frac, is_asia, session_end_frac=asia_end_f,
        wraps_midnight=True,
    )

    # --- London H/L (03:00 - 09:30 ET) ---
    is_london = (frac >= london_start_f) & (frac < london_end_f)
    result["london_high"], result["london_low"] = _vectorized_session_hl(
        df, et_index, frac, is_london, session_end_frac=london_end_f,
        wraps_midnight=False,
    )

    # --- Overnight H/L (18:00 - 09:30 ET) ---
    is_overnight = (frac >= overnight_start_f) | (frac < overnight_end_f)
    result["overnight_high"], result["overnight_low"] = _vectorized_session_hl(
        df, et_index, frac, is_overnight, session_end_frac=overnight_end_f,
        wraps_midnight=True,
    )

    # --- NY open price ---
    result["ny_open"] = _vectorized_ny_open(df, et_index, frac, ny_start_f)

    logger.info("Session levels computed. Non-null counts:\n%s", result.count())
    return result


def _vectorized_ny_open(
    df: pd.DataFrame,
    et_index: pd.DatetimeIndex,
    frac: np.ndarray,
    ny_start_frac: float,
) -> pd.Series:
    """Find the open price of the first 1-min bar of each NY session, forward-filled.

    Fully vectorized: groups by ET calendar date, finds first bar >= ny_start_frac.
    """
    n = len(df)
    frac_arr = np.asarray(frac)
    bar_date_key = (
        et_index.year * 10000 + et_index.month * 100 + et_index.day
    )
    bar_date_key = bar_date_key.to_numpy() if hasattr(bar_date_key, 'to_numpy') else np.array(bar_date_key)

    qualifies = frac_arr >= ny_start_frac
    qual_indices = np.where(qualifies)[0]

    ny_open_arr = np.full(n, np.nan, dtype=np.float64)

    if len(qual_indices) > 0:
        qual_df = pd.DataFrame({
            "pos": qual_indices,
            "date_key": bar_date_key[qual_indices],
        })
        first_per_date = qual_df.groupby("date_key")["pos"].first()
        open_vals = df["open"].values
        for dk, pos in first_per_date.items():
            ny_open_arr[pos] = open_vals[pos]

    return pd.Series(ny_open_arr, index=df.index, dtype="float64").ffill()


# ---------------------------------------------------------------------------
# 3. compute_orm
# ---------------------------------------------------------------------------

def compute_orm(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Compute the Opening Range observation window (ORM).

    The opening range is the first N minutes of the NY session
    (orm_window_minutes from params).

    Columns returned:
        orm_high  : high of the completed opening range, ffilled
        orm_low   : low of the completed opening range, ffilled
        is_orm_period : bool, True during the observation window

    Parameters
    ----------
    df : pd.DataFrame
        1-min OHLCV with UTC DatetimeIndex.
    params : dict
        Full params dict.

    Returns
    -------
    pd.DataFrame
        ORM columns, same index as df.
    """
    sp = params["sessions"]
    orm_minutes = sp["orm_window_minutes"]
    ny_start_h, ny_start_m = _parse_hhmm(sp["ny_prime_start"])
    ny_start_f = ny_start_h + ny_start_m / 60.0

    orm_end_total_min = ny_start_h * 60 + ny_start_m + orm_minutes
    orm_end_h = orm_end_total_min // 60
    orm_end_m = orm_end_total_min % 60
    orm_end_f = orm_end_h + orm_end_m / 60.0

    et_index = _to_et(df.index)
    frac = _et_frac(et_index)
    frac_arr = np.asarray(frac)
    n = len(df)

    # is_orm_period: bars during [ny_start, ny_start + orm_minutes)
    is_orm = (frac_arr >= ny_start_f) & (frac_arr < orm_end_f)

    result = pd.DataFrame(index=df.index)
    result["is_orm_period"] = is_orm

    # Session H/L per day for ORM bars
    bar_date_key = (
        et_index.year * 10000 + et_index.month * 100 + et_index.day
    )
    bar_date_key = bar_date_key.to_numpy() if hasattr(bar_date_key, 'to_numpy') else np.array(bar_date_key)

    high_arr = df["high"].values
    low_arr = df["low"].values

    orm_indices = np.where(is_orm)[0]
    if len(orm_indices) == 0:
        result["orm_high"] = np.nan
        result["orm_low"] = np.nan
        return result

    orm_df = pd.DataFrame({
        "date_key": bar_date_key[orm_indices],
        "high": high_arr[orm_indices],
        "low": low_arr[orm_indices],
    })
    orm_hl = orm_df.groupby("date_key").agg(
        o_high=("high", "max"),
        o_low=("low", "min"),
    )

    # Stamp at first bar at/after orm_end_f per date
    qualifies = frac_arr >= orm_end_f
    qual_indices = np.where(qualifies)[0]

    orm_high_arr = np.full(n, np.nan, dtype=np.float64)
    orm_low_arr = np.full(n, np.nan, dtype=np.float64)

    if len(qual_indices) > 0:
        qual_df = pd.DataFrame({
            "pos": qual_indices,
            "date_key": bar_date_key[qual_indices],
        })
        first_per_date = qual_df.groupby("date_key")["pos"].first()
        hl_dict = orm_hl.to_dict("index")
        for dk, pos in first_per_date.items():
            if dk in hl_dict:
                orm_high_arr[pos] = hl_dict[dk]["o_high"]
                orm_low_arr[pos] = hl_dict[dk]["o_low"]

    result["orm_high"] = pd.Series(orm_high_arr, index=df.index, dtype="float64").ffill()
    result["orm_low"] = pd.Series(orm_low_arr, index=df.index, dtype="float64").ffill()

    logger.info(
        "ORM computed: %d-min window, %d days with ORM data",
        orm_minutes,
        len(orm_hl),
    )
    return result


# ---------------------------------------------------------------------------
# Main: load data, compute everything, verify no lookahead
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time as _time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    params = _load_params(project_root / "config" / "params.yaml")
    df = pd.read_parquet(project_root / "data" / "NQ_1min.parquet")

    print(f"Loaded {len(df):,} bars, {df.index.min()} to {df.index.max()}")
    print()

    # --- 1. Session labels ---
    t0 = _time.perf_counter()
    labels = label_sessions(df, params)
    print(f"=== Session Label Distribution === ({_time.perf_counter()-t0:.1f}s)")
    print(labels.value_counts().to_string())
    print()

    # --- 2. Session levels ---
    t0 = _time.perf_counter()
    levels = compute_session_levels(df, params)
    print(f"=== Session Levels ({_time.perf_counter()-t0:.1f}s) ===")

    # Show a sample around NY open on the most recent full trading day
    et_idx = _to_et(df.index)
    frac_all = _et_frac(et_idx)
    date_keys_all = (
        et_idx.year * 10000 + et_idx.month * 100 + et_idx.day
    ).to_numpy()

    # Find the last date that has bars at 09:30 ET
    last_ny_dates = np.unique(date_keys_all[frac_all >= 9.5])
    if len(last_ny_dates) > 0:
        target_dk = last_ny_dates[-1]
        mask_day = date_keys_all == target_dk
        mask_window = mask_day & (frac_all >= 9.0) & (frac_all < 10.5)
        sample = levels.loc[mask_window]
        # Show every 15th row to keep output readable
        step = max(1, len(sample) // 6)
        print(f"Sample day: {target_dk} (around NY open)")
        print(sample.iloc[::step].to_string())
    print()

    # --- 3. ORM ---
    t0 = _time.perf_counter()
    orm = compute_orm(df, params)
    print(f"=== ORM ({_time.perf_counter()-t0:.1f}s) ===")
    if len(last_ny_dates) > 0:
        mask_window_orm = mask_day & (frac_all >= 9.0) & (frac_all < 10.5)
        orm_sample = orm.loc[mask_window_orm]
        step = max(1, len(orm_sample) // 6)
        print(orm_sample.iloc[::step].to_string())
    print()

    # --- 4. No-lookahead verification ---
    print("=== No-Lookahead Verification ===")
    # Asia ends at 03:00 ET.  asia_high at 03:01 should equal asia_high at 09:30
    # (same completed session, forward-filled).
    test_passed = True
    check_count = 0

    # Check the last 15 unique ET dates
    unique_dks = np.unique(date_keys_all)
    for dk in unique_dks[-15:]:
        mask_dk = date_keys_all == dk
        # 03:01 ET
        m1 = mask_dk & (frac_all >= 3.0 + 1 / 60) & (frac_all < 3.0 + 2 / 60)
        # 09:30 ET
        m2 = mask_dk & (frac_all >= 9.5) & (frac_all < 9.5 + 1 / 60)

        idx1 = df.index[m1]
        idx2 = df.index[m2]

        if len(idx1) > 0 and len(idx2) > 0:
            ah1 = levels.at[idx1[0], "asia_high"]
            ah2 = levels.at[idx2[0], "asia_high"]
            if pd.notna(ah1) and pd.notna(ah2):
                if ah1 != ah2:
                    print(f"  FAIL on {dk}: asia_high@03:01={ah1} != asia_high@09:30={ah2}")
                    test_passed = False
                check_count += 1

    if test_passed and check_count > 0:
        print(f"  PASS: asia_high at 03:01 ET == asia_high at 09:30 ET ({check_count} days checked)")
    elif check_count == 0:
        print("  SKIP: no suitable days found for verification")
    print()
    print("Done.")
