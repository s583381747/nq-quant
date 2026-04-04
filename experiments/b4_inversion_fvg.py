"""
experiments/b4_inversion_fvg.py — Inversion FVG (IFVG) signal exploration.

Assesses whether IFVGs (invalidated FVGs flipped to opposing direction)
are a viable signal source for the NQ trading system.

Lifecycle:
  1. Bullish FVG created  →  price closes below bottom  →  INVALIDATED
  2. Now it's a *bearish IFVG* (former support → resistance)
  3. Price returns up to the IFVG zone → rejection candle → SHORT signal
  (Vice versa for bearish FVGs → bullish IFVGs)

Usage: python experiments/b4_inversion_fvg.py
"""

import sys
import time as _time
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("b4_ifvg")

from features.fvg import detect_fvg
from features.displacement import compute_atr
from features.swing import compute_swing_levels


# ============================================================
# Config
# ============================================================
with open(PROJECT / "config" / "params.yaml", encoding="utf-8") as f:
    PARAMS = yaml.safe_load(f)

# IFVG-specific parameters (exploration — these would go into params.yaml if promoted)
IFVG_PARAMS = {
    "min_fvg_size_atr": 0.3,         # minimum FVG size as multiple of ATR (same as entry.min_fvg_atr_mult)
    "rejection_body_ratio": 0.50,     # rejection candle body/range minimum
    "max_bars_alive": 200,            # max bars an IFVG stays active before expiry
    "max_bars_to_retest": 150,        # max bars from invalidation to retest
    "forward_window": 100,            # bars to simulate forward for outcome
    "data_years": "2018-2025",        # date range
}


# ============================================================
# Data classes
# ============================================================
@dataclass
class IFVGRecord:
    """Tracks a single Inversion FVG from birth to signal/expiry."""
    original_fvg_time: pd.Timestamp    # when the original FVG was created
    invalidation_time: pd.Timestamp    # when the FVG was invalidated → IFVG born
    invalidation_idx: int              # bar index of invalidation
    direction: str                     # 'bull' or 'bear' (IFVG direction, opposite of original)
    original_direction: str            # original FVG direction
    top: float
    bottom: float
    size: float
    status: str = "active"             # active | signal_fired | expired | invalidated
    signal_time: pd.Timestamp | None = None
    signal_idx: int = -1
    signal_bar_open: float = np.nan
    signal_bar_close: float = np.nan
    signal_body_ratio: float = np.nan


@dataclass
class IFVGSignalOutcome:
    """Outcome of a simulated IFVG trade."""
    ifvg: IFVGRecord
    entry_price: float
    stop_price: float
    tp_price: float
    direction: int                     # +1 long, -1 short
    stop_dist: float
    tp_dist: float
    rr_target: float
    outcome: str = ""                  # "tp", "stop", "timeout"
    outcome_r: float = 0.0
    bars_held: int = 0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    session: str = ""
    hour_et: float = 0.0
    year: int = 0


# ============================================================
# Part 1: Detect and track FVG lifecycle → find IFVGs
# ============================================================
def detect_ifvgs(df: pd.DataFrame, atr: np.ndarray) -> list[IFVGRecord]:
    """Walk through price data, detect FVGs, track lifecycle, spawn IFVGs.

    This is a self-contained forward walk (no future leakage).
    """
    logger.info("Part 1: Detecting FVGs and tracking lifecycle...")
    t0 = _time.perf_counter()

    fvg_df = detect_fvg(df)

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    opn = df["open"].values
    index = df.index
    n = len(df)

    bull_mask = fvg_df["fvg_bull"].values
    bear_mask = fvg_df["fvg_bear"].values

    min_size_atr = IFVG_PARAMS["min_fvg_size_atr"]
    max_alive = IFVG_PARAMS["max_bars_alive"]

    # Track original FVGs
    @dataclass
    class OrigFVG:
        time: pd.Timestamp
        birth_idx: int
        direction: str       # 'bull' or 'bear'
        top: float
        bottom: float
        size: float
        status: str = "untested"  # untested | tested_rejected | invalidated

    active_fvgs: list[OrigFVG] = []
    ifvgs: list[IFVGRecord] = []
    stats = Counter()

    for i in range(n):
        # --- Birth: new FVGs ---
        if bull_mask[i]:
            candle2_time = index[i - 1] if i > 0 else index[i]
            rec = OrigFVG(
                time=candle2_time,
                birth_idx=i,
                direction="bull",
                top=fvg_df["fvg_bull_top"].iat[i],
                bottom=fvg_df["fvg_bull_bottom"].iat[i],
                size=fvg_df["fvg_size"].iat[i],
            )
            active_fvgs.append(rec)
            stats["fvg_bull_born"] += 1

        if bear_mask[i]:
            candle2_time = index[i - 1] if i > 0 else index[i]
            rec = OrigFVG(
                time=candle2_time,
                birth_idx=i,
                direction="bear",
                top=fvg_df["fvg_bear_top"].iat[i],
                bottom=fvg_df["fvg_bear_bottom"].iat[i],
                size=fvg_df["fvg_size"].iat[i],
            )
            active_fvgs.append(rec)
            stats["fvg_bear_born"] += 1

        # --- Update active FVGs ---
        surviving: list[OrigFVG] = []
        for rec in active_fvgs:
            # Age-out check
            bars_alive = i - rec.birth_idx
            if bars_alive > max_alive * 2:  # generous expiry for originals
                stats["fvg_expired"] += 1
                continue

            top = rec.top
            bottom = rec.bottom

            if rec.direction == "bull":
                entered = low[i] <= top and high[i] >= bottom
                if entered and close[i] < bottom:
                    # INVALIDATED → spawn bearish IFVG
                    rec.status = "invalidated"
                    stats["fvg_bull_invalidated"] += 1

                    # Size filter: only spawn IFVG if original FVG was meaningful
                    atr_val = atr[i] if not np.isnan(atr[i]) else 10.0
                    if rec.size >= min_size_atr * atr_val:
                        ifvg = IFVGRecord(
                            original_fvg_time=rec.time,
                            invalidation_time=index[i],
                            invalidation_idx=i,
                            direction="bear",  # flipped
                            original_direction="bull",
                            top=rec.top,
                            bottom=rec.bottom,
                            size=rec.size,
                        )
                        ifvgs.append(ifvg)
                        stats["ifvg_bear_spawned"] += 1
                    continue  # remove from active

                elif not entered and close[i] < bottom:
                    # Gap-down through → invalidated
                    rec.status = "invalidated"
                    stats["fvg_bull_invalidated"] += 1
                    atr_val = atr[i] if not np.isnan(atr[i]) else 10.0
                    if rec.size >= min_size_atr * atr_val:
                        ifvg = IFVGRecord(
                            original_fvg_time=rec.time,
                            invalidation_time=index[i],
                            invalidation_idx=i,
                            direction="bear",
                            original_direction="bull",
                            top=rec.top,
                            bottom=rec.bottom,
                            size=rec.size,
                        )
                        ifvgs.append(ifvg)
                        stats["ifvg_bear_spawned"] += 1
                    continue

                elif entered and close[i] >= top:
                    rec.status = "tested_rejected"
                    stats["fvg_bull_tested"] += 1

                surviving.append(rec)

            else:  # bear
                entered = high[i] >= bottom and low[i] <= top
                if entered and close[i] > top:
                    # INVALIDATED → spawn bullish IFVG
                    rec.status = "invalidated"
                    stats["fvg_bear_invalidated"] += 1

                    atr_val = atr[i] if not np.isnan(atr[i]) else 10.0
                    if rec.size >= min_size_atr * atr_val:
                        ifvg = IFVGRecord(
                            original_fvg_time=rec.time,
                            invalidation_time=index[i],
                            invalidation_idx=i,
                            direction="bull",  # flipped
                            original_direction="bear",
                            top=rec.top,
                            bottom=rec.bottom,
                            size=rec.size,
                        )
                        ifvgs.append(ifvg)
                        stats["ifvg_bull_spawned"] += 1
                    continue

                elif not entered and close[i] > top:
                    # Gap-up through → invalidated
                    rec.status = "invalidated"
                    stats["fvg_bear_invalidated"] += 1
                    atr_val = atr[i] if not np.isnan(atr[i]) else 10.0
                    if rec.size >= min_size_atr * atr_val:
                        ifvg = IFVGRecord(
                            original_fvg_time=rec.time,
                            invalidation_time=index[i],
                            invalidation_idx=i,
                            direction="bull",
                            original_direction="bear",
                            top=rec.top,
                            bottom=rec.bottom,
                            size=rec.size,
                        )
                        ifvgs.append(ifvg)
                        stats["ifvg_bull_spawned"] += 1
                    continue

                elif entered and close[i] <= bottom:
                    rec.status = "tested_rejected"
                    stats["fvg_bear_tested"] += 1

                surviving.append(rec)

        active_fvgs = surviving

    elapsed = _time.perf_counter() - t0
    logger.info("Part 1 complete in %.1fs", elapsed)

    print("\n" + "=" * 70)
    print("PART 1: FVG LIFECYCLE STATISTICS")
    print("=" * 70)
    for k, v in sorted(stats.items()):
        print(f"  {k:35s}: {v:>8,}")
    print(f"  {'total_ifvgs_spawned':35s}: {len(ifvgs):>8,}")

    # Per-year breakdown
    if ifvgs:
        years = defaultdict(int)
        for ifvg in ifvgs:
            y = ifvg.invalidation_time.year
            years[y] += 1
        print("\n  IFVGs per year:")
        for y in sorted(years):
            print(f"    {y}: {years[y]:>6,}")

    return ifvgs


# ============================================================
# Part 2: IFVG Signal Detection (retest + rejection)
# ============================================================
def detect_ifvg_signals(
    df: pd.DataFrame,
    ifvgs: list[IFVGRecord],
    atr: np.ndarray,
) -> list[IFVGRecord]:
    """For each IFVG, scan forward for price to return and produce a rejection signal.

    Signal criteria:
    - Price enters the IFVG zone from the NEW direction
    - Rejection candle: closes on "correct" side with body_ratio > threshold
    - For bearish IFVG (resistance): price comes UP, rejection candle closes BELOW bottom
    - For bullish IFVG (support): price comes DOWN, rejection candle closes ABOVE top
    """
    logger.info("Part 2: Detecting IFVG signals...")
    t0 = _time.perf_counter()

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    opn = df["open"].values
    n = len(df)

    max_retest = IFVG_PARAMS["max_bars_to_retest"]
    max_alive = IFVG_PARAMS["max_bars_alive"]
    body_ratio_min = IFVG_PARAMS["rejection_body_ratio"]

    signals: list[IFVGRecord] = []
    stats = Counter()

    for ifvg in ifvgs:
        start = ifvg.invalidation_idx + 1  # start scanning from bar after invalidation
        end = min(start + max_retest, n)

        if start >= n:
            stats["no_data_after_invalidation"] += 1
            continue

        found_signal = False

        for i in range(start, end):
            bars_since = i - ifvg.invalidation_idx

            # Check if IFVG has been invalidated (price closed through in opposite direction)
            if ifvg.direction == "bear":
                # Bearish IFVG = resistance zone
                # Invalidated if price closes above the top (broke through resistance)
                # Actually, for a bearish IFVG, if price has already moved far above,
                # it's no longer relevant. Check bar by bar.
                if close[i] > ifvg.top + (ifvg.top - ifvg.bottom):
                    # Price went well past → IFVG broken, stop looking
                    ifvg.status = "invalidated"
                    stats["ifvg_broken_through"] += 1
                    break

                # Signal check: price enters zone from below
                entered = high[i] >= ifvg.bottom and low[i] <= ifvg.top
                if entered:
                    # Rejection: close below bottom (bounced down)
                    body = abs(close[i] - opn[i])
                    rng = high[i] - low[i]
                    br = body / rng if rng > 0 else 0.0

                    if close[i] < ifvg.bottom and br >= body_ratio_min:
                        # Valid rejection signal
                        ifvg.status = "signal_fired"
                        ifvg.signal_time = df.index[i]
                        ifvg.signal_idx = i
                        ifvg.signal_bar_open = opn[i]
                        ifvg.signal_bar_close = close[i]
                        ifvg.signal_body_ratio = br
                        signals.append(ifvg)
                        stats["signal_bear"] += 1
                        found_signal = True
                        break
                    elif close[i] > ifvg.top:
                        # Price closed above the zone → IFVG invalidated
                        ifvg.status = "invalidated"
                        stats["ifvg_invalidated_on_test"] += 1
                        break

            else:
                # Bullish IFVG = support zone
                # Invalidated if price closes far below
                if close[i] < ifvg.bottom - (ifvg.top - ifvg.bottom):
                    ifvg.status = "invalidated"
                    stats["ifvg_broken_through"] += 1
                    break

                entered = low[i] <= ifvg.top and high[i] >= ifvg.bottom
                if entered:
                    body = abs(close[i] - opn[i])
                    rng = high[i] - low[i]
                    br = body / rng if rng > 0 else 0.0

                    if close[i] > ifvg.top and br >= body_ratio_min:
                        # Valid rejection signal
                        ifvg.status = "signal_fired"
                        ifvg.signal_time = df.index[i]
                        ifvg.signal_idx = i
                        ifvg.signal_bar_open = opn[i]
                        ifvg.signal_bar_close = close[i]
                        ifvg.signal_body_ratio = br
                        signals.append(ifvg)
                        stats["signal_bull"] += 1
                        found_signal = True
                        break
                    elif close[i] < ifvg.bottom:
                        ifvg.status = "invalidated"
                        stats["ifvg_invalidated_on_test"] += 1
                        break

        if not found_signal and ifvg.status == "active":
            ifvg.status = "expired"
            stats["ifvg_expired"] += 1

    elapsed = _time.perf_counter() - t0
    logger.info("Part 2 complete in %.1fs — %d signals found", elapsed, len(signals))

    print("\n" + "=" * 70)
    print("PART 2: IFVG SIGNAL DETECTION")
    print("=" * 70)
    for k, v in sorted(stats.items()):
        print(f"  {k:35s}: {v:>8,}")
    print(f"  {'total_signals':35s}: {len(signals):>8,}")
    print(f"  {'signal_rate':35s}: {len(signals)/max(len(ifvgs),1)*100:>7.1f}%")

    if signals:
        years = defaultdict(int)
        for s in signals:
            y = s.signal_time.year
            years[y] += 1
        print("\n  IFVG signals per year:")
        for y in sorted(years):
            print(f"    {y}: {years[y]:>6,}")

    return signals


# ============================================================
# Part 3: Simulate IFVG signal outcomes
# ============================================================
def simulate_outcomes(
    df: pd.DataFrame,
    signals: list[IFVGRecord],
    atr: np.ndarray,
    swing_df: pd.DataFrame,
) -> list[IFVGSignalOutcome]:
    """Forward-simulate each IFVG signal for outcome_r.

    Entry: next bar open after rejection
    Stop: far side of the IFVG OR candle-2 open (whichever gives tighter stop)
    TP: nearest swing in trade direction
    """
    logger.info("Part 3: Simulating outcomes for %d signals...", len(signals))
    t0 = _time.perf_counter()

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    opn = df["open"].values
    n = len(df)

    fw = IFVG_PARAMS["forward_window"]

    swing_high_price = swing_df["swing_high_price"].values
    swing_low_price = swing_df["swing_low_price"].values

    et_idx = df.index.tz_convert("US/Eastern")

    outcomes: list[IFVGSignalOutcome] = []

    for sig in signals:
        entry_idx = sig.signal_idx + 1
        if entry_idx >= n:
            continue

        entry_price = opn[entry_idx]

        if sig.direction == "bear":
            direction = -1  # short
            # Stop: above the IFVG top (resistance zone — if price goes above, we're wrong)
            stop_price = sig.top + 0.25  # 1 tick above top
            # TP: nearest swing low below entry
            tp_price = swing_low_price[sig.signal_idx]
            if np.isnan(tp_price) or tp_price >= entry_price:
                # Fallback: use 2x stop distance below entry
                stop_dist = abs(stop_price - entry_price)
                tp_price = entry_price - 2.0 * stop_dist
        else:
            direction = 1  # long
            stop_price = sig.bottom - 0.25  # 1 tick below bottom
            tp_price = swing_high_price[sig.signal_idx]
            if np.isnan(tp_price) or tp_price <= entry_price:
                stop_dist = abs(entry_price - stop_price)
                tp_price = entry_price + 2.0 * stop_dist

        stop_dist = abs(entry_price - stop_price)
        if stop_dist < 0.25:
            continue  # skip pathological cases

        tp_dist = abs(tp_price - entry_price)
        rr_target = tp_dist / stop_dist if stop_dist > 0 else 0.0

        # Simulate forward
        end_idx = min(entry_idx + fw, n)
        outcome = "timeout"
        outcome_r = 0.0
        bars_held = 0
        max_fav = 0.0
        max_adv = 0.0

        for j in range(entry_idx, end_idx):
            bars_held = j - entry_idx + 1

            if direction == 1:  # long
                excursion_fav = high[j] - entry_price
                excursion_adv = entry_price - low[j]
                max_fav = max(max_fav, excursion_fav)
                max_adv = max(max_adv, excursion_adv)

                if low[j] <= stop_price:
                    outcome = "stop"
                    outcome_r = -1.0
                    break
                if high[j] >= tp_price:
                    outcome = "tp"
                    outcome_r = rr_target
                    break
            else:  # short
                excursion_fav = entry_price - low[j]
                excursion_adv = high[j] - entry_price
                max_fav = max(max_fav, excursion_fav)
                max_adv = max(max_adv, excursion_adv)

                if high[j] >= stop_price:
                    outcome = "stop"
                    outcome_r = -1.0
                    break
                if low[j] <= tp_price:
                    outcome = "tp"
                    outcome_r = rr_target
                    break

        if outcome == "timeout":
            # Mark-to-market at last bar
            if direction == 1:
                outcome_r = (close[min(end_idx - 1, n - 1)] - entry_price) / stop_dist
            else:
                outcome_r = (entry_price - close[min(end_idx - 1, n - 1)]) / stop_dist

        # Session info
        et_ts = et_idx[sig.signal_idx]
        h_et = et_ts.hour + et_ts.minute / 60.0
        if h_et >= 18.0 or h_et < 3.0:
            session = "asia"
        elif 3.0 <= h_et < 9.5:
            session = "london"
        elif 9.5 <= h_et < 16.0:
            session = "ny"
        else:
            session = "other"

        oc = IFVGSignalOutcome(
            ifvg=sig,
            entry_price=entry_price,
            stop_price=stop_price,
            tp_price=tp_price,
            direction=direction,
            stop_dist=stop_dist,
            tp_dist=tp_dist,
            rr_target=rr_target,
            outcome=outcome,
            outcome_r=outcome_r,
            bars_held=bars_held,
            max_favorable=max_fav,
            max_adverse=max_adv,
            session=session,
            hour_et=h_et,
            year=sig.signal_time.year,
        )
        outcomes.append(oc)

    elapsed = _time.perf_counter() - t0
    logger.info("Part 3 complete in %.1fs — %d outcomes", elapsed, len(outcomes))

    return outcomes


# ============================================================
# Part 4: Statistics & Analysis
# ============================================================
def analyze_outcomes(outcomes: list[IFVGSignalOutcome], ifvgs: list[IFVGRecord]):
    """Comprehensive statistics on IFVG signal performance."""

    if not outcomes:
        print("\n  NO OUTCOMES TO ANALYZE")
        return

    # Convert to arrays for easy stats
    rs = np.array([o.outcome_r for o in outcomes])
    dirs = np.array([o.direction for o in outcomes])
    sessions = np.array([o.session for o in outcomes])
    ocs = np.array([o.outcome for o in outcomes])
    years = np.array([o.year for o in outcomes])
    rr_targets = np.array([o.rr_target for o in outcomes])
    stops = np.array([o.stop_dist for o in outcomes])
    ifvg_sizes = np.array([o.ifvg.size for o in outcomes])
    body_ratios = np.array([o.ifvg.signal_body_ratio for o in outcomes])
    bars_held = np.array([o.bars_held for o in outcomes])

    n_total = len(outcomes)
    n_tp = (ocs == "tp").sum()
    n_stop = (ocs == "stop").sum()
    n_timeout = (ocs == "timeout").sum()
    wr = n_tp / n_total * 100
    total_r = rs.sum()
    avg_r = rs.mean()
    pf = abs(rs[rs > 0].sum() / rs[rs < 0].sum()) if rs[rs < 0].sum() != 0 else np.inf

    # PPDD calculation
    cumr = np.cumsum(rs)
    peak = np.maximum.accumulate(cumr)
    dd = peak - cumr
    max_dd = dd.max()
    ppdd = total_r / max_dd if max_dd > 0 else np.inf

    print("\n" + "=" * 70)
    print("PART 4: IFVG SIGNAL PERFORMANCE")
    print("=" * 70)

    print(f"\n  --- Overall ---")
    print(f"  Total signals:     {n_total:>8,}")
    print(f"  TP hit:            {n_tp:>8,}  ({n_tp/n_total*100:.1f}%)")
    print(f"  Stop hit:          {n_stop:>8,}  ({n_stop/n_total*100:.1f}%)")
    print(f"  Timeout:           {n_timeout:>8,}  ({n_timeout/n_total*100:.1f}%)")
    print(f"  Win rate (TP):     {wr:>8.1f}%")
    print(f"  Total R:           {total_r:>+8.1f}")
    print(f"  Average R:         {avg_r:>+8.3f}")
    print(f"  Profit Factor:     {pf:>8.2f}")
    print(f"  Max Drawdown:      {max_dd:>8.1f} R")
    print(f"  PPDD:              {ppdd:>8.2f}")
    print(f"  Avg RR target:     {rr_targets.mean():>8.2f}")
    print(f"  Avg stop dist:     {stops.mean():>8.1f} pts")
    print(f"  Avg bars held:     {bars_held.mean():>8.1f}")

    # --- By direction ---
    print(f"\n  --- By Direction ---")
    for d, label in [(1, "Long (bull IFVG)"), (-1, "Short (bear IFVG)")]:
        mask = dirs == d
        if mask.sum() == 0:
            continue
        r_d = rs[mask]
        n_d = mask.sum()
        wr_d = (ocs[mask] == "tp").sum() / n_d * 100
        avg_d = r_d.mean()
        tot_d = r_d.sum()
        pf_d = abs(r_d[r_d > 0].sum() / r_d[r_d < 0].sum()) if r_d[r_d < 0].sum() != 0 else np.inf
        print(f"  {label:25s}: n={n_d:>5,}  WR={wr_d:.1f}%  avgR={avg_d:+.3f}  totR={tot_d:+.1f}  PF={pf_d:.2f}")

    # --- By session ---
    print(f"\n  --- By Session ---")
    for sess in ["ny", "london", "asia", "other"]:
        mask = sessions == sess
        if mask.sum() == 0:
            continue
        r_s = rs[mask]
        n_s = mask.sum()
        wr_s = (ocs[mask] == "tp").sum() / n_s * 100 if n_s > 0 else 0
        avg_s = r_s.mean()
        tot_s = r_s.sum()
        print(f"  {sess:10s}: n={n_s:>5,}  WR={wr_s:.1f}%  avgR={avg_s:+.3f}  totR={tot_s:+.1f}")

    # --- By year ---
    print(f"\n  --- By Year ---")
    for y in sorted(set(years)):
        mask = years == y
        r_y = rs[mask]
        n_y = mask.sum()
        wr_y = (ocs[mask] == "tp").sum() / n_y * 100 if n_y > 0 else 0
        avg_y = r_y.mean()
        tot_y = r_y.sum()
        pf_y = abs(r_y[r_y > 0].sum() / r_y[r_y < 0].sum()) if r_y[r_y < 0].sum() != 0 else np.inf
        print(f"  {y}: n={n_y:>5,}  WR={wr_y:.1f}%  avgR={avg_y:+.3f}  totR={tot_y:+.1f}  PF={pf_y:.2f}")

    # --- By outcome type ---
    print(f"\n  --- Outcome R Distribution ---")
    for oc_type in ["tp", "stop", "timeout"]:
        mask = ocs == oc_type
        if mask.sum() == 0:
            continue
        r_oc = rs[mask]
        print(f"  {oc_type:10s}: mean_R={r_oc.mean():+.3f}  med_R={np.median(r_oc):+.3f}  "
              f"min={r_oc.min():+.2f}  max={r_oc.max():+.2f}")

    # --- FVG size analysis ---
    print(f"\n  --- By IFVG Size (ATR-relative) ---")
    # Bin by size: small (<0.5 ATR), medium (0.5-1.0), large (>1.0)
    # We already have ifvg_sizes in points; need ATR context
    p25, p50, p75 = np.percentile(ifvg_sizes, [25, 50, 75])
    for lo, hi, label in [(0, p25, f"small (<p25={p25:.1f}pt)"),
                          (p25, p50, f"medium (p25-p50)"),
                          (p50, p75, f"large (p50-p75)"),
                          (p75, 9999, f"huge (>p75={p75:.1f}pt)")]:
        mask = (ifvg_sizes >= lo) & (ifvg_sizes < hi)
        if mask.sum() == 0:
            continue
        r_b = rs[mask]
        n_b = mask.sum()
        wr_b = (ocs[mask] == "tp").sum() / n_b * 100 if n_b > 0 else 0
        avg_b = r_b.mean()
        print(f"  {label:30s}: n={n_b:>5,}  WR={wr_b:.1f}%  avgR={avg_b:+.3f}")

    # --- Body ratio of rejection candle ---
    print(f"\n  --- By Rejection Candle Body Ratio ---")
    for lo, hi, label in [(0.50, 0.60, "0.50-0.60"),
                          (0.60, 0.70, "0.60-0.70"),
                          (0.70, 0.80, "0.70-0.80"),
                          (0.80, 1.01, "0.80+")]:
        mask = (body_ratios >= lo) & (body_ratios < hi)
        if mask.sum() == 0:
            continue
        r_b = rs[mask]
        n_b = mask.sum()
        wr_b = (ocs[mask] == "tp").sum() / n_b * 100 if n_b > 0 else 0
        avg_b = r_b.mean()
        print(f"  body_ratio {label:10s}: n={n_b:>5,}  WR={wr_b:.1f}%  avgR={avg_b:+.3f}")

    # --- Time-of-day distribution ---
    print(f"\n  --- Time of Day (ET hour) ---")
    hours_et = np.array([o.hour_et for o in outcomes])
    for h in range(0, 24, 2):
        mask = (hours_et >= h) & (hours_et < h + 2)
        if mask.sum() == 0:
            continue
        r_h = rs[mask]
        n_h = mask.sum()
        wr_h = (ocs[mask] == "tp").sum() / n_h * 100 if n_h > 0 else 0
        print(f"  {h:02d}:00-{h+2:02d}:00 ET: n={n_h:>5,}  WR={wr_h:.1f}%  avgR={r_h.mean():+.3f}")

    # --- NY-only performance (the relevant filter for deployment) ---
    print(f"\n  --- NY Session Only (10:00-16:00 ET) ---")
    ny_mask = (hours_et >= 10.0) & (hours_et < 16.0)
    if ny_mask.sum() > 0:
        r_ny = rs[ny_mask]
        n_ny = ny_mask.sum()
        wr_ny = (ocs[ny_mask] == "tp").sum() / n_ny * 100
        tot_ny = r_ny.sum()
        pf_ny = abs(r_ny[r_ny > 0].sum() / r_ny[r_ny < 0].sum()) if r_ny[r_ny < 0].sum() != 0 else np.inf
        cumr_ny = np.cumsum(r_ny)
        peak_ny = np.maximum.accumulate(cumr_ny)
        dd_ny = (peak_ny - cumr_ny).max()
        ppdd_ny = tot_ny / dd_ny if dd_ny > 0 else np.inf

        print(f"  Total signals:     {n_ny:>8,}")
        print(f"  Win rate:          {wr_ny:>8.1f}%")
        print(f"  Total R:           {tot_ny:>+8.1f}")
        print(f"  Average R:         {r_ny.mean():>+8.3f}")
        print(f"  Profit Factor:     {pf_ny:>8.2f}")
        print(f"  PPDD:              {ppdd_ny:>8.2f}")
        print(f"  Max DD:            {dd_ny:>8.1f} R")

        # NY by direction
        for d, label in [(1, "Long"), (-1, "Short")]:
            m2 = ny_mask & (dirs == d)
            if m2.sum() == 0:
                continue
            r_d = rs[m2]
            n_d = m2.sum()
            wr_d = (ocs[m2] == "tp").sum() / n_d * 100
            print(f"  NY {label:6s}: n={n_d:>5,}  WR={wr_d:.1f}%  avgR={r_d.mean():+.3f}  totR={r_d.sum():+.1f}")


# ============================================================
# Part 5: Feasibility Assessment
# ============================================================
def assess_feasibility(outcomes: list[IFVGSignalOutcome], ifvgs: list[IFVGRecord]):
    """Assessment of whether IFVGs should be added to the pipeline."""

    if not outcomes:
        print("\n  CANNOT ASSESS — no outcomes")
        return

    rs = np.array([o.outcome_r for o in outcomes])
    years = np.array([o.year for o in outcomes])
    hours = np.array([o.hour_et for o in outcomes])
    ocs = np.array([o.outcome for o in outcomes])

    # NY session filter (10:00-16:00 ET)
    ny_mask = (hours >= 10.0) & (hours < 16.0)
    # Lunch block (12:30-13:00)
    lunch_mask = (hours >= 12.5) & (hours < 13.0)
    # PM shorts block (after 14:00)
    dirs = np.array([o.direction for o in outcomes])
    pm_shorts_mask = (hours >= 14.0) & (dirs == -1)
    # Asia/London skip
    not_off_hours = ny_mask

    # After all existing session filters
    filtered_mask = ny_mask & ~lunch_mask & ~pm_shorts_mask

    print("\n" + "=" * 70)
    print("PART 5: FEASIBILITY ASSESSMENT")
    print("=" * 70)

    print("\n  --- Filter Chain Impact ---")
    print(f"  Total IFVG signals (all sessions): {len(outcomes):>8,}")
    print(f"  After NY filter (10-16 ET):        {ny_mask.sum():>8,}")
    print(f"  After lunch block (12:30-13):      {(ny_mask & ~lunch_mask).sum():>8,}")
    print(f"  After PM shorts block:             {filtered_mask.sum():>8,}")

    if filtered_mask.sum() > 0:
        r_f = rs[filtered_mask]
        n_f = filtered_mask.sum()
        wr_f = (ocs[filtered_mask] == "tp").sum() / n_f * 100
        tot_f = r_f.sum()
        avg_f = r_f.mean()
        pf_f = abs(r_f[r_f > 0].sum() / r_f[r_f < 0].sum()) if r_f[r_f < 0].sum() != 0 else np.inf

        cumr_f = np.cumsum(r_f)
        peak_f = np.maximum.accumulate(cumr_f)
        dd_f = (peak_f - cumr_f).max()
        ppdd_f = tot_f / dd_f if dd_f > 0 else np.inf

        # Per year after filters
        unique_years = sorted(set(years[filtered_mask]))
        n_years = len(unique_years)
        trades_per_year = n_f / max(n_years, 1)

        print(f"\n  --- After All Filters ---")
        print(f"  Total signals:     {n_f:>8,}")
        print(f"  Trades/year:       {trades_per_year:>8.1f}")
        print(f"  Win rate:          {wr_f:>8.1f}%")
        print(f"  Total R:           {tot_f:>+8.1f}")
        print(f"  Average R:         {avg_f:>+8.3f}")
        print(f"  Profit Factor:     {pf_f:>8.2f}")
        print(f"  Max DD:            {dd_f:>8.1f} R")
        print(f"  PPDD:              {ppdd_f:>8.2f}")

        # Comparison reference: current system = ~1104 trades, 92/year, +319R, PPDD=20.69
        print(f"\n  --- Comparison to Current System ---")
        print(f"  Current system:  1104 trades, 92/year, +319R, PPDD=20.69")
        print(f"  IFVG standalone: {n_f} trades, {trades_per_year:.0f}/year, {tot_f:+.1f}R, PPDD={ppdd_f:.2f}")
        if trades_per_year > 0:
            r_per_year = tot_f / n_years
            print(f"  IFVG R/year:     {r_per_year:+.1f}")
            print(f"  Current R/year:  +{319/10:.1f}")  # rough ~10yr

        # Year-by-year filtered
        print(f"\n  --- Year-by-Year (After Filters) ---")
        for y in unique_years:
            ym = filtered_mask & (years == y)
            r_y = rs[ym]
            n_y = ym.sum()
            wr_y = (ocs[ym] == "tp").sum() / n_y * 100 if n_y > 0 else 0
            print(f"    {y}: n={n_y:>4,}  WR={wr_y:.1f}%  totR={r_y.sum():+.1f}  avgR={r_y.mean():+.3f}")

    print(f"\n  --- Implementation Complexity ---")
    print(f"  1. IFVG detection: already in fvg.py (track_fvg_state spawns IFVGs)")
    print(f"  2. Signal cache: needs new signal_type='ifvg' in cache_signals")
    print(f"  3. Entry/stop logic: different from trend/MSS (stop = IFVG boundary)")
    print(f"  4. Filter chain: same as existing (session, SQ, bias, news)")
    print(f"  5. Trade management: same as trend (trim, trail, BE)")
    print(f"  Estimated effort: 2-3 sessions to integrate into main pipeline")

    # Final verdict
    print(f"\n  ========================================")
    print(f"  VERDICT (raw IFVG):")
    if filtered_mask.sum() > 0:
        if avg_f > 0 and ppdd_f > 1.0:
            print(f"  PROMISING — positive expectancy ({avg_f:+.3f} R/trade)")
            print(f"  Recommend: integrate as signal_type='ifvg' with SQ filter")
        elif avg_f > 0:
            print(f"  MARGINAL — barely positive ({avg_f:+.3f} R/trade), low PPDD")
            print(f"  Recommend: park for now, revisit with quality filters")
        else:
            print(f"  NOT VIABLE — negative expectancy ({avg_f:+.3f} R/trade)")
            print(f"  Recommend: do not add to pipeline")
    else:
        print(f"  INSUFFICIENT DATA after filters")
    print(f"  ========================================")


# ============================================================
# Part 6: Quality-Filtered Subset Analysis
# ============================================================
def quality_filtered_analysis(outcomes: list[IFVGSignalOutcome], atr_arr: np.ndarray):
    """Test multiple quality filter combinations to find viable subsets."""

    if not outcomes:
        return

    print("\n" + "=" * 70)
    print("PART 6: QUALITY-FILTERED SUBSET ANALYSIS")
    print("=" * 70)

    rs = np.array([o.outcome_r for o in outcomes])
    dirs = np.array([o.direction for o in outcomes])
    hours = np.array([o.hour_et for o in outcomes])
    ocs = np.array([o.outcome for o in outcomes])
    years = np.array([o.year for o in outcomes])
    body_ratios = np.array([o.ifvg.signal_body_ratio for o in outcomes])
    ifvg_sizes = np.array([o.ifvg.size for o in outcomes])
    stop_dists = np.array([o.stop_dist for o in outcomes])
    rr_targets = np.array([o.rr_target for o in outcomes])
    bars_to_signal = np.array([
        o.ifvg.signal_idx - o.ifvg.invalidation_idx for o in outcomes
    ])
    # ATR at signal time
    sig_atr = np.array([
        atr_arr[o.ifvg.signal_idx] if not np.isnan(atr_arr[o.ifvg.signal_idx]) else 10.0
        for o in outcomes
    ])
    ifvg_size_atr = ifvg_sizes / sig_atr

    # Base: NY 10-16 ET
    ny_mask = (hours >= 10.0) & (hours < 16.0)

    def print_stats(label, mask):
        if mask.sum() == 0:
            print(f"  {label:45s}: n=0")
            return
        r = rs[mask]
        n = mask.sum()
        wr = (ocs[mask] == "tp").sum() / n * 100
        tot = r.sum()
        avg = r.mean()
        pf = abs(r[r > 0].sum() / r[r < 0].sum()) if r[r < 0].sum() != 0 else np.inf
        cumr = np.cumsum(r)
        peak = np.maximum.accumulate(cumr)
        dd = (peak - cumr).max()
        ppdd = tot / dd if dd > 0 else np.inf
        yrs = sorted(set(years[mask]))
        tpy = n / max(len(yrs), 1)
        print(f"  {label:45s}: n={n:>5,}  {tpy:>5.0f}/yr  WR={wr:.1f}%  avgR={avg:+.3f}  totR={tot:>+7.1f}  PF={pf:.2f}  PPDD={ppdd:.2f}")

    # --- A: Body ratio filters ---
    print("\n  === A: Body Ratio Filters (NY only) ===")
    for br_min in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80]:
        m = ny_mask & (body_ratios >= br_min)
        print_stats(f"body_ratio >= {br_min:.2f}", m)

    # --- B: IFVG size filters (ATR-relative) ---
    print("\n  === B: IFVG Size Filters (NY only) ===")
    for sz_min in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        m = ny_mask & (ifvg_size_atr >= sz_min)
        print_stats(f"ifvg_size >= {sz_min:.1f}x ATR", m)

    # --- C: Bars-to-signal (recency) ---
    print("\n  === C: Recency Filters (NY only) ===")
    for max_bars in [10, 20, 30, 50, 100]:
        m = ny_mask & (bars_to_signal <= max_bars)
        print_stats(f"bars_to_signal <= {max_bars}", m)

    # --- D: Stop distance filters ---
    print("\n  === D: Stop Distance Filters (NY only) ===")
    for stop_min in [5, 10, 15, 20, 30]:
        m = ny_mask & (stop_dists >= stop_min)
        print_stats(f"stop_dist >= {stop_min} pts", m)

    # --- E: RR target filters ---
    print("\n  === E: RR Target Filters (NY only) ===")
    for rr_min in [1.0, 1.5, 2.0, 2.5, 3.0]:
        m = ny_mask & (rr_targets >= rr_min)
        print_stats(f"rr_target >= {rr_min:.1f}", m)

    # --- F: Combined best filters ---
    print("\n  === F: Combined Quality Filters (NY only) ===")
    combos = [
        ("large+high_br", ny_mask & (ifvg_size_atr >= 1.0) & (body_ratios >= 0.65)),
        ("large+high_br+recent", ny_mask & (ifvg_size_atr >= 1.0) & (body_ratios >= 0.65) & (bars_to_signal <= 50)),
        ("huge+any_br", ny_mask & (ifvg_size_atr >= 1.5) & (body_ratios >= 0.50)),
        ("huge+high_br", ny_mask & (ifvg_size_atr >= 1.5) & (body_ratios >= 0.65)),
        ("huge+high_br+recent20", ny_mask & (ifvg_size_atr >= 1.5) & (body_ratios >= 0.65) & (bars_to_signal <= 20)),
        ("size1+br70+stop15", ny_mask & (ifvg_size_atr >= 1.0) & (body_ratios >= 0.70) & (stop_dists >= 15)),
        ("size1.5+br70+stop15", ny_mask & (ifvg_size_atr >= 1.5) & (body_ratios >= 0.70) & (stop_dists >= 15)),
    ]
    for label, m in combos:
        print_stats(label, m)

    # --- G: Direction-specific combined ---
    print("\n  === G: Direction-Specific (NY only) ===")
    for d, dname in [(1, "Long"), (-1, "Short")]:
        dm = ny_mask & (dirs == d)
        print_stats(f"{dname} (all)", dm)
        for label_suffix, extra in [
            ("large+high_br", (ifvg_size_atr >= 1.0) & (body_ratios >= 0.65)),
            ("huge+high_br", (ifvg_size_atr >= 1.5) & (body_ratios >= 0.65)),
        ]:
            m = dm & extra
            print_stats(f"{dname} {label_suffix}", m)

    # --- H: Best subset year-by-year ---
    print("\n  === H: Best Subset Year-by-Year ===")
    # Pick the best combo from above (let's test a few)
    best_combos = {
        "huge+high_br": ny_mask & (ifvg_size_atr >= 1.5) & (body_ratios >= 0.65),
        "large+high_br": ny_mask & (ifvg_size_atr >= 1.0) & (body_ratios >= 0.65),
        "Long large+high_br": ny_mask & (dirs == 1) & (ifvg_size_atr >= 1.0) & (body_ratios >= 0.65),
    }
    for combo_name, combo_mask in best_combos.items():
        if combo_mask.sum() == 0:
            continue
        print(f"\n  [{combo_name}] year-by-year:")
        for y in sorted(set(years)):
            ym = combo_mask & (years == y)
            if ym.sum() == 0:
                continue
            r_y = rs[ym]
            n_y = ym.sum()
            wr_y = (ocs[ym] == "tp").sum() / n_y * 100
            print(f"    {y}: n={n_y:>4,}  WR={wr_y:.1f}%  totR={r_y.sum():>+7.1f}  avgR={r_y.mean():+.3f}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("B4: INVERSION FVG (IFVG) EXPLORATION")
    print("=" * 70)
    t_start = _time.perf_counter()

    # 1. Load data
    logger.info("Loading 5m data...")
    nq = pd.read_parquet(PROJECT / "data" / "NQ_5m_10yr.parquet")
    logger.info("Loaded %d bars from %s to %s", len(nq), nq.index[0], nq.index[-1])

    # Filter to date range for speed
    date_range = IFVG_PARAMS["data_years"]
    start_year, end_year = date_range.split("-")
    nq = nq.loc[f"{start_year}":f"{end_year}"]
    logger.info("Filtered to %s: %d bars (%s to %s)",
                date_range, len(nq), nq.index[0], nq.index[-1])

    # 2. Compute ATR
    logger.info("Computing ATR(14)...")
    atr_series = compute_atr(nq, period=14)
    atr = atr_series.values

    # 3. Compute swing levels (for TP targets)
    logger.info("Computing swing levels...")
    swing_params = {"left_bars": PARAMS["swing"]["left_bars"],
                    "right_bars": PARAMS["swing"]["right_bars"]}
    swing_df = compute_swing_levels(nq, swing_params)
    # Shift by 1 to avoid lookahead
    swing_df["swing_high_price"] = swing_df["swing_high_price"].shift(1).ffill()
    swing_df["swing_low_price"] = swing_df["swing_low_price"].shift(1).ffill()

    # 4. Part 1: Detect IFVGs
    ifvgs = detect_ifvgs(nq, atr)

    # 5. Part 2: Detect IFVG signals
    signals = detect_ifvg_signals(nq, ifvgs, atr)

    # 6. Part 3: Simulate outcomes
    outcomes = simulate_outcomes(nq, signals, atr, swing_df)

    # 7. Part 4: Statistics
    analyze_outcomes(outcomes, ifvgs)

    # 8. Part 5: Feasibility
    assess_feasibility(outcomes, ifvgs)

    # 9. Part 6: Quality-filtered subset analysis
    quality_filtered_analysis(outcomes, atr)

    elapsed = _time.perf_counter() - t_start
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
