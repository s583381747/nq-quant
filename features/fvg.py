"""
features/fvg.py — Fair Value Gap detection, state tracking, and active-FVG features.

FVG = 3 consecutive candles where candle-1's high/low and candle-3's high/low
do not overlap, creating a gap at candle-2 (the displacement candle).

  Bullish FVG: candle_1.high < candle_3.low   (gap above candle-1)
  Bearish FVG: candle_1.low  > candle_3.high   (gap below candle-1)

The FVG is anchored to candle-2's timestamp.  It becomes "known" only after
candle-3 closes, so all features are shifted to avoid future leakage.

State machine per FVG:
  untested  ->  tested_rejected  (price entered gap, closed back out)
  untested  ->  invalidated      (price *closed* through the gap)
  tested_rejected -> invalidated
  invalidated -> spawns an IFVG (inversion FVG) entry if displacement present

Rollover filter: any FVG whose 3-candle window includes an is_roll_date bar
is discarded (price levels are unreliable across contract rolls).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. detect_fvg  — raw FVG detection on a single-timeframe OHLC DataFrame
# ---------------------------------------------------------------------------

def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Fair Value Gaps on every bar of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame with columns [open, high, low, close, volume, is_roll_date].
        Must be sorted by datetime index in ascending order.

    Returns
    -------
    pd.DataFrame
        Same index as *df*, with the following columns added:

        - fvg_bull       (bool)  — bullish FVG detected at this bar (candle-2)
        - fvg_bear       (bool)  — bearish FVG detected at this bar (candle-2)
        - fvg_bull_top   (float) — top of bullish gap  (candle_3.low)
        - fvg_bull_bottom(float) — bottom of bullish gap (candle_1.high)
        - fvg_bear_top   (float) — top of bearish gap  (candle_1.low)
        - fvg_bear_bottom(float) — bottom of bearish gap (candle_3.high)
        - fvg_size        (float) — absolute gap size in points

    Notes
    -----
    * The FVG is anchored to **candle-2** (the middle / displacement candle).
    * The gap becomes known only after candle-3 closes, so the signal columns
      are shifted forward by 1 bar to prevent future leakage.  That is, the
      row labelled *candle-2's timestamp* will show the FVG **after** candle-3
      has closed (i.e. at candle-3's close, looking back).
    * FVGs that span a rollover date (any of the 3 candles has is_roll_date=True)
      are discarded.
    """
    n = len(df)
    if n < 3:
        logger.warning("detect_fvg: DataFrame has fewer than 3 rows, returning empty.")
        return _empty_fvg_frame(df)

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values  # noqa: F841 — kept for clarity
    roll = df["is_roll_date"].values

    # Pre-allocate output arrays (anchored to candle-2 index, i.e. i)
    fvg_bull = np.zeros(n, dtype=bool)
    fvg_bear = np.zeros(n, dtype=bool)
    fvg_bull_top = np.full(n, np.nan)
    fvg_bull_bottom = np.full(n, np.nan)
    fvg_bear_top = np.full(n, np.nan)
    fvg_bear_bottom = np.full(n, np.nan)
    fvg_size = np.full(n, np.nan)

    for i in range(1, n - 1):
        # candle indices: c1=i-1, c2=i, c3=i+1
        c1, c2, c3 = i - 1, i, i + 1

        # Skip if any candle is on a rollover date
        if roll[c1] or roll[c2] or roll[c3]:
            continue

        # Bullish FVG: candle_1.high < candle_3.low  (gap upward)
        if high[c1] < low[c3]:
            gap_top = low[c3]
            gap_bot = high[c1]
            fvg_bull[c2] = True
            fvg_bull_top[c2] = gap_top
            fvg_bull_bottom[c2] = gap_bot
            fvg_size[c2] = gap_top - gap_bot

        # Bearish FVG: candle_1.low > candle_3.high  (gap downward)
        if low[c1] > high[c3]:
            gap_top = low[c1]
            gap_bot = high[c3]
            fvg_bear[c2] = True
            fvg_bear_top[c2] = gap_top
            fvg_bear_bottom[c2] = gap_bot
            # If both bull and bear on same bar (extremely rare), keep the
            # larger gap.  fvg_size stores the bigger one.
            size_bear = gap_top - gap_bot
            if np.isnan(fvg_size[c2]) or size_bear > fvg_size[c2]:
                fvg_size[c2] = size_bear

    # ---- Shift forward by 1 bar (no future leakage) ----
    # The FVG at candle-2 is only knowable once candle-3 closes.
    # We shift the detection arrays so that the signal appears at candle-3's
    # row, meaning at candle-2's index we can only see FVGs detected *before*
    # it.  However, the spec says "anchored to candle-2" — so we keep the
    # candle-2 timestamp but delay visibility by 1 bar.
    #
    # Implementation: we set the *raw* arrays at c2 above, then shift by 1.
    fvg_bull = np.roll(fvg_bull, 1); fvg_bull[0] = False
    fvg_bear = np.roll(fvg_bear, 1); fvg_bear[0] = False
    fvg_bull_top = np.roll(fvg_bull_top, 1); fvg_bull_top[0] = np.nan
    fvg_bull_bottom = np.roll(fvg_bull_bottom, 1); fvg_bull_bottom[0] = np.nan
    fvg_bear_top = np.roll(fvg_bear_top, 1); fvg_bear_top[0] = np.nan
    fvg_bear_bottom = np.roll(fvg_bear_bottom, 1); fvg_bear_bottom[0] = np.nan
    fvg_size = np.roll(fvg_size, 1); fvg_size[0] = np.nan

    result = pd.DataFrame(
        {
            "fvg_bull": fvg_bull,
            "fvg_bear": fvg_bear,
            "fvg_bull_top": fvg_bull_top,
            "fvg_bull_bottom": fvg_bull_bottom,
            "fvg_bear_top": fvg_bear_top,
            "fvg_bear_bottom": fvg_bear_bottom,
            "fvg_size": fvg_size,
        },
        index=df.index,
    )

    n_bull = int(fvg_bull.sum())
    n_bear = int(fvg_bear.sum())
    logger.info("detect_fvg: %d bullish, %d bearish FVGs detected on %d bars", n_bull, n_bear, n)
    return result


def _empty_fvg_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return an empty FVG DataFrame aligned to *df*'s index."""
    return pd.DataFrame(
        {
            "fvg_bull": False,
            "fvg_bear": False,
            "fvg_bull_top": np.nan,
            "fvg_bull_bottom": np.nan,
            "fvg_bear_top": np.nan,
            "fvg_bear_bottom": np.nan,
            "fvg_size": np.nan,
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# 2. track_fvg_state  — walk forward through price to update each FVG status
# ---------------------------------------------------------------------------

@dataclass
class FVGRecord:
    """Mutable record for a single FVG as it gets tested / invalidated."""

    time: pd.Timestamp
    direction: str            # 'bull' or 'bear'
    top: float
    bottom: float
    size: float
    status: str = "untested"  # untested | tested_rejected | invalidated
    invalidated_at: pd.Timestamp | None = None
    is_ifvg: bool = False     # True if this record is an Inversion FVG
    ifvg_direction: str = ""  # 'bull' or 'bear' (reversed from original)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "direction": self.direction,
            "top": self.top,
            "bottom": self.bottom,
            "size": self.size,
            "status": self.status,
            "invalidated_at": self.invalidated_at,
            "is_ifvg": self.is_ifvg,
            "ifvg_direction": self.ifvg_direction,
        }


def track_fvg_state(
    df: pd.DataFrame,
    fvg_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Walk price bar-by-bar and update each FVG's status.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC data (same index as used in detect_fvg).
    fvg_df : pd.DataFrame
        Output of ``detect_fvg(df)`` — columns fvg_bull, fvg_bear, etc.

    Returns
    -------
    list[dict]
        Each dict represents one FVG (or IFVG) with keys:
        time, direction, top, bottom, size, status, invalidated_at,
        is_ifvg, ifvg_direction.

    Notes
    -----
    State transitions (per bar close):

    - **untested**: price high < fvg bottom (bull) or price low > fvg top (bear)
    - **tested_rejected**: price entered the gap (high >= bottom AND low <= top)
      but the close is back on the "correct" side:
        bull FVG: close >= fvg_top  (bounced up out of gap)
        bear FVG: close <= fvg_bottom  (bounced down out of gap)
    - **invalidated**: price *closed* through the gap:
        bull FVG: close < fvg_bottom  (closed below the gap)
        bear FVG: close > fvg_top     (closed above the gap)

    When a FVG is invalidated, a new IFVG record is spawned with reversed
    direction (former support -> resistance, vice versa).
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    index = df.index

    # Collect FVG birth events (from detect_fvg output, already shifted by 1)
    bull_mask = fvg_df["fvg_bull"].values
    bear_mask = fvg_df["fvg_bear"].values

    # We need the *original* candle-2 time for each FVG.
    # Since detect_fvg shifted by 1, the signal at row i corresponds to the
    # FVG whose candle-2 was at row i-1.
    records: list[FVGRecord] = []
    active: list[FVGRecord] = []  # currently untested or tested_rejected

    for i in range(len(df)):
        ts = index[i]

        # --- Birth: register new FVGs that become visible at bar i ---
        if bull_mask[i]:
            candle2_time = index[i - 1] if i > 0 else ts
            rec = FVGRecord(
                time=candle2_time,
                direction="bull",
                top=fvg_df["fvg_bull_top"].iat[i],
                bottom=fvg_df["fvg_bull_bottom"].iat[i],
                size=fvg_df["fvg_size"].iat[i],
            )
            records.append(rec)
            active.append(rec)

        if bear_mask[i]:
            candle2_time = index[i - 1] if i > 0 else ts
            rec = FVGRecord(
                time=candle2_time,
                direction="bear",
                top=fvg_df["fvg_bear_top"].iat[i],
                bottom=fvg_df["fvg_bear_bottom"].iat[i],
                size=fvg_df["fvg_size"].iat[i],
            )
            records.append(rec)
            active.append(rec)

        # --- Update active FVGs against this bar's price action ---
        still_active: list[FVGRecord] = []
        for rec in active:
            new_status = _update_fvg_status(rec, high[i], low[i], close[i])

            if new_status == "invalidated" and rec.status != "invalidated":
                rec.status = "invalidated"
                rec.invalidated_at = ts

                # Spawn Inversion FVG only from original FVGs (not from
                # other IFVGs — an invalidated IFVG is simply dead, no
                # further cascade).
                if not rec.is_ifvg:
                    ifvg_dir = "bear" if rec.direction == "bull" else "bull"
                    ifvg = FVGRecord(
                        time=ts,
                        direction=ifvg_dir,
                        top=rec.top,
                        bottom=rec.bottom,
                        size=rec.size,
                        status="untested",
                        is_ifvg=True,
                        ifvg_direction=ifvg_dir,
                    )
                    records.append(ifvg)
                    still_active.append(ifvg)
                # rec itself is no longer active
            elif new_status != rec.status:
                rec.status = new_status
                still_active.append(rec)
            else:
                still_active.append(rec)

        active = still_active

    n_untested = sum(1 for r in records if r.status == "untested")
    n_tested = sum(1 for r in records if r.status == "tested_rejected")
    n_inval = sum(1 for r in records if r.status == "invalidated")
    n_ifvg = sum(1 for r in records if r.is_ifvg)
    logger.info(
        "track_fvg_state: %d total records (untested=%d, tested_rejected=%d, "
        "invalidated=%d, ifvg=%d)",
        len(records), n_untested, n_tested, n_inval, n_ifvg,
    )

    return [r.to_dict() for r in records]


def _update_fvg_status(rec: FVGRecord, bar_high: float, bar_low: float, bar_close: float) -> str:
    """Determine new status for a single FVG given one bar's OHLC.

    Only transitions *forward* in the state machine:
        untested -> tested_rejected -> invalidated
        untested -> invalidated  (direct)
    """
    if rec.status == "invalidated":
        return "invalidated"

    top = rec.top
    bottom = rec.bottom

    if rec.direction == "bull" or (rec.is_ifvg and rec.ifvg_direction == "bull"):
        # Bull FVG sits below price as support.
        # Price entering gap: low <= top (dips into gap from above)
        entered = bar_low <= top and bar_high >= bottom

        if entered:
            # Invalidated: close below the gap bottom
            if bar_close < bottom:
                return "invalidated"
            # Tested & rejected: entered gap but closed above gap top
            if bar_close >= top:
                return "tested_rejected" if rec.status == "untested" else rec.status
            # Inside the gap at close — still "testing", keep current status
            # but mark as tested_rejected since price has interacted
            return "tested_rejected" if rec.status == "untested" else rec.status

        # Not entered at all
        # If price closes below gap bottom without entering (gap-down through)
        if bar_close < bottom:
            return "invalidated"

        return rec.status

    else:
        # Bear FVG sits above price as resistance.
        # Price entering gap: high >= bottom (rises into gap from below)
        entered = bar_high >= bottom and bar_low <= top

        if entered:
            # Invalidated: close above the gap top
            if bar_close > top:
                return "invalidated"
            # Tested & rejected: entered gap but closed below gap bottom
            if bar_close <= bottom:
                return "tested_rejected" if rec.status == "untested" else rec.status
            return "tested_rejected" if rec.status == "untested" else rec.status

        # Not entered — if price closes above top (gap-up through)
        if bar_close > top:
            return "invalidated"

        return rec.status


# ---------------------------------------------------------------------------
# 3. compute_active_fvgs  — per-bar features: distance, count of active FVGs
# ---------------------------------------------------------------------------

def compute_active_fvgs(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Build per-bar features describing active FVGs relative to current price.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC data (same as passed to detect_fvg).
    params : dict
        Full params dict (read from config/params.yaml).  Currently unused
        but reserved for future quality filters.

    Returns
    -------
    pd.DataFrame
        Aligned to *df*.index with columns:

        - nearest_bull_fvg_dist  (float) — distance from close to nearest active
          bullish FVG midpoint; positive = FVG is below price (support)
        - nearest_bear_fvg_dist  (float) — distance from close to nearest active
          bearish FVG midpoint; positive = FVG is above price (resistance)
        - num_active_bull_fvgs   (int)   — count of active (untested + tested_rejected)
          bullish FVGs
        - num_active_bear_fvgs   (int)   — count of active bearish FVGs
        - nearest_fvg_size       (float) — point-size of the single nearest FVG
          (bull or bear, whichever is closer)
    """
    logger.info("compute_active_fvgs: starting FVG detection + state tracking")

    fvg_df = detect_fvg(df)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    index = df.index
    n = len(df)

    # Output arrays
    nearest_bull_dist = np.full(n, np.nan)
    nearest_bear_dist = np.full(n, np.nan)
    num_bull = np.zeros(n, dtype=np.int32)
    num_bear = np.zeros(n, dtype=np.int32)
    nearest_fvg_size = np.full(n, np.nan)

    bull_mask = fvg_df["fvg_bull"].values
    bear_mask = fvg_df["fvg_bear"].values

    # Active FVG pools — walk forward, maintaining live lists
    active_bull: list[FVGRecord] = []
    active_bear: list[FVGRecord] = []

    for i in range(n):
        # --- Birth ---
        if bull_mask[i]:
            rec = FVGRecord(
                time=index[i - 1] if i > 0 else index[i],
                direction="bull",
                top=fvg_df["fvg_bull_top"].iat[i],
                bottom=fvg_df["fvg_bull_bottom"].iat[i],
                size=fvg_df["fvg_size"].iat[i],
            )
            active_bull.append(rec)

        if bear_mask[i]:
            rec = FVGRecord(
                time=index[i - 1] if i > 0 else index[i],
                direction="bear",
                top=fvg_df["fvg_bear_top"].iat[i],
                bottom=fvg_df["fvg_bear_bottom"].iat[i],
                size=fvg_df["fvg_size"].iat[i],
            )
            active_bear.append(rec)

        # --- Update & prune active bull FVGs ---
        surviving_bull: list[FVGRecord] = []
        for rec in active_bull:
            new_status = _update_fvg_status(rec, high[i], low[i], close[i])
            if new_status == "invalidated":
                rec.status = "invalidated"
                # Don't track IFVGs in compute_active_fvgs for simplicity;
                # they are captured by track_fvg_state for detailed analysis.
            else:
                rec.status = new_status
                surviving_bull.append(rec)
        active_bull = surviving_bull

        # --- Update & prune active bear FVGs ---
        surviving_bear: list[FVGRecord] = []
        for rec in active_bear:
            new_status = _update_fvg_status(rec, high[i], low[i], close[i])
            if new_status == "invalidated":
                rec.status = "invalidated"
            else:
                rec.status = new_status
                surviving_bear.append(rec)
        active_bear = surviving_bear

        # --- Compute features for this bar ---
        c = close[i]
        num_bull[i] = len(active_bull)
        num_bear[i] = len(active_bear)

        best_bull_dist = np.inf
        best_bull_size = np.nan
        for rec in active_bull:
            mid = (rec.top + rec.bottom) / 2.0
            dist = c - mid  # positive = price above FVG (FVG is support below)
            if abs(dist) < abs(best_bull_dist):
                best_bull_dist = dist
                best_bull_size = rec.size

        best_bear_dist = np.inf
        best_bear_size = np.nan
        for rec in active_bear:
            mid = (rec.top + rec.bottom) / 2.0
            dist = mid - c  # positive = FVG is above price (resistance above)
            if abs(dist) < abs(best_bear_dist):
                best_bear_dist = dist
                best_bear_size = rec.size

        if best_bull_dist != np.inf:
            nearest_bull_dist[i] = best_bull_dist
        if best_bear_dist != np.inf:
            nearest_bear_dist[i] = best_bear_dist

        # Nearest overall
        if abs(best_bull_dist) <= abs(best_bear_dist) and best_bull_dist != np.inf:
            nearest_fvg_size[i] = best_bull_size
        elif best_bear_dist != np.inf:
            nearest_fvg_size[i] = best_bear_size

    result = pd.DataFrame(
        {
            "nearest_bull_fvg_dist": nearest_bull_dist,
            "nearest_bear_fvg_dist": nearest_bear_dist,
            "num_active_bull_fvgs": num_bull,
            "num_active_bear_fvgs": num_bear,
            "nearest_fvg_size": nearest_fvg_size,
        },
        index=df.index,
    )

    logger.info(
        "compute_active_fvgs: done — mean active bull=%.1f, bear=%.1f",
        num_bull.mean(), num_bear.mean(),
    )
    return result


# ---------------------------------------------------------------------------
# Helper: load params
# ---------------------------------------------------------------------------

def _load_params(path: str = "config/params.yaml") -> dict:
    """Load YAML params file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# __main__  — quick smoke test on 5m data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        stream=sys.stdout,
    )

    logger.info("Loading 5m data...")
    df = pd.read_parquet("data/NQ_5m.parquet")
    logger.info("Loaded %d bars from %s to %s", len(df), df.index[0], df.index[-1])

    # 1. Detect FVGs
    fvg_df = detect_fvg(df)

    n_bull = int(fvg_df["fvg_bull"].sum())
    n_bear = int(fvg_df["fvg_bear"].sum())
    valid_sizes = fvg_df["fvg_size"].dropna()

    logger.info("=" * 60)
    logger.info("FVG Detection Summary (5m NQ)")
    logger.info("=" * 60)
    logger.info("Total bullish FVGs : %d", n_bull)
    logger.info("Total bearish FVGs : %d", n_bear)
    logger.info("Mean FVG size      : %.2f points", valid_sizes.mean() if len(valid_sizes) else 0)
    logger.info("Median FVG size    : %.2f points", valid_sizes.median() if len(valid_sizes) else 0)
    logger.info("Max FVG size       : %.2f points", valid_sizes.max() if len(valid_sizes) else 0)
    logger.info("Min FVG size       : %.2f points", valid_sizes.min() if len(valid_sizes) else 0)

    # Sample of detected FVGs
    sample_bull = fvg_df[fvg_df["fvg_bull"]].head(5)
    sample_bear = fvg_df[fvg_df["fvg_bear"]].head(5)

    logger.info("-" * 60)
    logger.info("Sample bullish FVGs (first 5):")
    for ts, row in sample_bull.iterrows():
        logger.info(
            "  %s  top=%.2f  bot=%.2f  size=%.2f",
            ts, row["fvg_bull_top"], row["fvg_bull_bottom"], row["fvg_size"],
        )

    logger.info("Sample bearish FVGs (first 5):")
    for ts, row in sample_bear.iterrows():
        logger.info(
            "  %s  top=%.2f  bot=%.2f  size=%.2f",
            ts, row["fvg_bear_top"], row["fvg_bear_bottom"], row["fvg_size"],
        )

    # 2. Track FVG state (on a subset for speed)
    logger.info("-" * 60)
    logger.info("Tracking FVG state on last 5000 bars...")
    subset = df.iloc[-5000:]
    fvg_sub = detect_fvg(subset)
    records = track_fvg_state(subset, fvg_sub)

    from collections import Counter
    status_counts = Counter(r["status"] for r in records)
    ifvg_count = sum(1 for r in records if r["is_ifvg"])

    logger.info("State tracking results (last 5000 bars):")
    logger.info("  Total FVG records : %d", len(records))
    for status, cnt in status_counts.items():
        logger.info("  %-20s: %d", status, cnt)
    logger.info("  IFVG spawned      : %d", ifvg_count)

    # Show a few active (untested/tested_rejected)
    active_records = [r for r in records if r["status"] in ("untested", "tested_rejected")]
    logger.info("Active FVGs at end of data (%d):", len(active_records))
    for r in active_records[-5:]:
        logger.info(
            "  %s %s  top=%.2f bot=%.2f  size=%.2f  status=%s  ifvg=%s",
            r["time"], r["direction"], r["top"], r["bottom"],
            r["size"], r["status"], r["is_ifvg"],
        )
