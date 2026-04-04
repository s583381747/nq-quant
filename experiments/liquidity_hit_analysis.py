"""
Liquidity Hit Analysis — NQ Quantitative Trading System

For each traded signal (after SMT gate), identifies all available liquidity
levels in the trade direction and simulates forward to measure hit rates,
time-to-hit, MFE, and MAE.

Produces:
  Part 1: Per-signal forward simulation
  Part 2: Statistical summary by level type
  Part 3: Distance-binned hit rates (ATR multiples)
  Part 4: Best TP candidates
  Part 5: Multi-level probability cascade
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.swing import detect_swing_highs, detect_swing_lows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_FORWARD_BARS = 100       # 500 minutes on 5m data
SAMPLE_STRIDE = 3            # use every Nth signal for speed (3 => ~18K signals)
SWING_HISTORY_WINDOW = 200   # bars to look back for swing collection

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_all() -> tuple:
    """Load and merge all required caches."""
    t0 = time.time()

    logger.info("Loading NQ 5m data...")
    nq = pd.read_parquet(ROOT / "data" / "NQ_5m_10yr.parquet")

    logger.info("Loading signal cache v4...")
    sig = pd.read_parquet(ROOT / "data" / "cache_signals_10yr_v4.parquet")

    logger.info("Loading session levels cache v2...")
    sess = pd.read_parquet(ROOT / "data" / "cache_session_levels_10yr_v2.parquet")

    logger.info("Loading ATR/fluency cache v2...")
    af = pd.read_parquet(ROOT / "data" / "cache_atr_flu_10yr_v2.parquet")

    logger.info("Loading SMT cache...")
    smt = pd.read_parquet(ROOT / "data" / "cache_smt_10yr.parquet")

    logger.info("Loading ORM cache v2...")
    orm = pd.read_parquet(ROOT / "data" / "cache_orm_10yr_v2.parquet")

    logger.info("All data loaded in %.1f seconds", time.time() - t0)
    return nq, sig, sess, af, smt, orm


# ---------------------------------------------------------------------------
# Signal Filtering (SMT gate)
# ---------------------------------------------------------------------------

def apply_smt_gate(sig: pd.DataFrame, smt: pd.DataFrame) -> pd.DataFrame:
    """Filter signals: trend passes through, MSS requires SMT divergence."""
    merged = sig.join(smt, how="left")
    signals = merged[merged["signal"] == True].copy()

    # Trend: all pass
    trend_mask = signals["signal_type"] == "trend"

    # MSS: long needs smt_bull, short needs smt_bear
    mss_mask = signals["signal_type"] == "mss"
    mss_long_ok = mss_mask & (signals["signal_dir"] == 1) & (signals["smt_bull"] == True)
    mss_short_ok = mss_mask & (signals["signal_dir"] == -1) & (signals["smt_bear"] == True)

    keep = trend_mask | mss_long_ok | mss_short_ok
    result = signals[keep].copy()
    logger.info("SMT gate: %d -> %d signals", len(signals), len(result))
    return result


# ---------------------------------------------------------------------------
# Swing Level Collection
# ---------------------------------------------------------------------------

def build_swing_arrays(nq: pd.DataFrame) -> tuple:
    """Detect all swing highs/lows on 5m data.

    Returns shifted+ffilled boolean arrays and the raw high/low prices
    at swing points (for collecting N-th swings).
    """
    t0 = time.time()
    logger.info("Computing swing highs/lows on %d bars...", len(nq))

    sh_bool = detect_swing_highs(nq["high"], left=3, right=1)
    sl_bool = detect_swing_lows(nq["low"], left=3, right=1)

    # Shift by 1 to prevent look-ahead (swing at bar i confirmed at bar i+1)
    sh_bool_shifted = sh_bool.shift(1).fillna(False).astype(bool)
    sl_bool_shifted = sl_bool.shift(1).fillna(False).astype(bool)

    # Store swing prices (NaN where not a swing)
    sh_prices = np.where(sh_bool_shifted, nq["high"].shift(1).values, np.nan)
    sl_prices = np.where(sl_bool_shifted, nq["low"].shift(1).values, np.nan)

    logger.info("Swing detection done in %.1f seconds — %d highs, %d lows",
                time.time() - t0, sh_bool_shifted.sum(), sl_bool_shifted.sum())
    return sh_prices, sl_prices


def get_nth_swings_above_below(
    idx: int,
    entry_price: float,
    direction: int,
    sh_prices: np.ndarray,
    sl_prices: np.ndarray,
    n_levels: int = 3,
    lookback: int = SWING_HISTORY_WINDOW,
) -> dict:
    """Find the 1st, 2nd, 3rd swing levels above (long) or below (short) entry.

    For LONG: find swing highs above entry price, sorted ascending (closest first).
    For SHORT: find swing lows below entry price, sorted descending (closest first).

    Returns dict like {'swing_1': price, 'swing_2': price, ...}
    """
    start = max(0, idx - lookback)
    result = {}

    if direction == 1:  # Long — swing highs above entry
        window = sh_prices[start:idx + 1]
        candidates = window[~np.isnan(window)]
        above = candidates[candidates > entry_price]
        above_unique = np.unique(above)
        above_sorted = np.sort(above_unique)  # ascending: closest first
        for i in range(min(n_levels, len(above_sorted))):
            result[f"swing_{i+1}"] = above_sorted[i]
    else:  # Short — swing lows below entry
        window = sl_prices[start:idx + 1]
        candidates = window[~np.isnan(window)]
        below = candidates[candidates < entry_price]
        below_unique = np.unique(below)
        below_sorted = np.sort(below_unique)[::-1]  # descending: closest first
        for i in range(min(n_levels, len(below_sorted))):
            result[f"swing_{i+1}"] = below_sorted[i]

    return result


# ---------------------------------------------------------------------------
# Forward Simulation
# ---------------------------------------------------------------------------

def simulate_forward(
    ohlc_high: np.ndarray,
    ohlc_low: np.ndarray,
    start_idx: int,
    entry_price: float,
    direction: int,
    targets: dict,
    max_bars: int = MAX_FORWARD_BARS,
) -> dict:
    """Simulate forward from entry bar to check if price reaches each target.

    For each target level, records:
      - hit: bool
      - bars_to_hit: int or NaN
      - mae_before_hit: max adverse excursion before hitting target (points)
      - mfe_total: max favorable excursion during entire window (points)

    Parameters
    ----------
    ohlc_high, ohlc_low : np.ndarray
        Full arrays of high/low prices.
    start_idx : int
        Index of entry bar (simulation starts at start_idx + 1).
    entry_price : float
        Entry price.
    direction : int
        1 for long, -1 for short.
    targets : dict
        {level_name: target_price, ...}
    max_bars : int
        Maximum bars to simulate forward.
    """
    n = len(ohlc_high)
    end_idx = min(start_idx + 1 + max_bars, n)

    results = {}
    mfe_total = 0.0

    # Precompute cumulative MFE and MAE for the window
    # For LONG: favorable = high - entry, adverse = entry - low
    # For SHORT: favorable = entry - low, adverse = high - entry
    window_high = ohlc_high[start_idx + 1: end_idx]
    window_low = ohlc_low[start_idx + 1: end_idx]

    if len(window_high) == 0:
        for name in targets:
            results[name] = {"hit": False, "bars_to_hit": np.nan,
                             "mae_before_hit": np.nan, "mfe_total": 0.0}
        return results

    if direction == 1:
        cum_mfe = np.maximum.accumulate(window_high) - entry_price
        cum_mae = entry_price - np.minimum.accumulate(window_low)
        mfe_total = float(cum_mfe[-1])
    else:
        cum_mfe = entry_price - np.minimum.accumulate(window_low)
        cum_mae = np.maximum.accumulate(window_high) - entry_price
        mfe_total = float(cum_mfe[-1])

    for name, target_price in targets.items():
        if np.isnan(target_price):
            results[name] = {"hit": False, "bars_to_hit": np.nan,
                             "mae_before_hit": np.nan, "mfe_total": mfe_total}
            continue

        # Check when target is hit
        if direction == 1:
            # Long: target is above entry, hit when high >= target
            hit_mask = window_high >= target_price
        else:
            # Short: target is below entry, hit when low <= target
            hit_mask = window_low <= target_price

        if hit_mask.any():
            bar_idx = np.argmax(hit_mask)  # first True
            bars_to_hit = bar_idx + 1  # 1-indexed

            # MAE before hit: worst adverse excursion from entry to bar of hit
            if direction == 1:
                mae_slice = entry_price - np.min(window_low[:bar_idx + 1])
            else:
                mae_slice = np.max(window_high[:bar_idx + 1]) - entry_price
            mae_before_hit = float(max(0, mae_slice))

            results[name] = {
                "hit": True,
                "bars_to_hit": int(bars_to_hit),
                "mae_before_hit": mae_before_hit,
                "mfe_total": mfe_total,
            }
        else:
            results[name] = {
                "hit": False,
                "bars_to_hit": np.nan,
                "mae_before_hit": np.nan,
                "mfe_total": mfe_total,
            }

    return results


# ---------------------------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    # ---- Load data ----
    nq, sig, sess, af, smt, orm = load_all()

    # ---- Apply SMT gate ----
    signals = apply_smt_gate(sig, smt)

    # ---- Merge all into a single frame for signal info ----
    # We need: entry_price, model_stop, irl_target, signal_dir, signal_type
    #          + session levels + ATR + ORM

    # ---- Build swing arrays (shifted to prevent lookahead) ----
    sh_prices, sl_prices = build_swing_arrays(nq)

    # ---- Precompute numpy arrays for fast forward simulation ----
    ohlc_high = nq["high"].values.astype(np.float64)
    ohlc_low = nq["low"].values.astype(np.float64)

    # ---- Session levels, ATR, ORM as numpy for fast access ----
    sess_vals = {c: sess[c].values for c in sess.columns}
    atr_vals = af["atr"].values
    orm_high_vals = orm["orm_high"].values
    orm_low_vals = orm["orm_low"].values

    # ---- Map signal timestamps to integer indices ----
    idx_map = pd.Series(np.arange(len(nq)), index=nq.index)

    # Sample signals for speed
    signal_indices = signals.index
    sampled = signal_indices[::SAMPLE_STRIDE]
    logger.info("Sampled %d signals (stride=%d) from %d total",
                len(sampled), SAMPLE_STRIDE, len(signal_indices))

    # ---- Process each signal ----
    all_records = []
    t_loop = time.time()

    for count, ts in enumerate(sampled):
        if count % 5000 == 0 and count > 0:
            elapsed = time.time() - t_loop
            rate = count / elapsed
            eta = (len(sampled) - count) / rate
            logger.info("Processed %d/%d signals (%.0f/sec, ETA %.0fs)",
                        count, len(sampled), rate, eta)

        # Get integer index in nq
        if ts not in idx_map.index:
            continue
        idx = idx_map[ts]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
        idx = int(idx)

        row = signals.loc[ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        direction = int(row["signal_dir"])
        entry = float(row["entry_price"])
        stop = float(row["model_stop"])
        irl = float(row["irl_target"])
        sig_type = str(row["signal_type"])

        # ATR at signal bar
        atr = float(atr_vals[idx]) if idx < len(atr_vals) and not np.isnan(atr_vals[idx]) else np.nan
        if np.isnan(atr) or atr < 0.1:
            continue

        stop_dist = abs(entry - stop)

        # ---- Collect all liquidity levels in trade direction ----
        targets = {}

        # 1. Swing targets (1st, 2nd, 3rd)
        swing_levels = get_nth_swings_above_below(
            idx, entry, direction, sh_prices, sl_prices, n_levels=3
        )
        targets.update(swing_levels)

        # 2. IRL target from signal cache
        if not np.isnan(irl):
            targets["irl_target"] = irl

        # 3. Session levels in trade direction
        if direction == 1:  # Long — levels above entry
            for name in ["asia_high", "london_high", "overnight_high"]:
                val = sess_vals[name][idx]
                if not np.isnan(val) and val > entry:
                    targets[name] = val
            # ORM high
            val = orm_high_vals[idx]
            if not np.isnan(val) and val > entry:
                targets["orm_high"] = val
            # NY open as reference
            val = sess_vals["ny_open"][idx]
            if not np.isnan(val) and val > entry:
                targets["ny_open"] = val
        else:  # Short — levels below entry
            for name in ["asia_low", "london_low", "overnight_low"]:
                val = sess_vals[name][idx]
                if not np.isnan(val) and val < entry:
                    targets[name] = val
            # ORM low
            val = orm_low_vals[idx]
            if not np.isnan(val) and val < entry:
                targets["orm_low"] = val
            # NY open as reference
            val = sess_vals["ny_open"][idx]
            if not np.isnan(val) and val < entry:
                targets["ny_open"] = val

        # 4. Fixed RR targets (for benchmarking)
        if stop_dist > 0:
            targets["1R_target"] = entry + direction * stop_dist
            targets["2R_target"] = entry + direction * 2 * stop_dist
            targets["3R_target"] = entry + direction * 3 * stop_dist

        if len(targets) == 0:
            continue

        # ---- Forward simulate ----
        sim = simulate_forward(ohlc_high, ohlc_low, idx, entry, direction,
                               targets, max_bars=MAX_FORWARD_BARS)

        # ---- Record results ----
        for level_name, level_price in targets.items():
            res = sim[level_name]
            dist_pts = abs(level_price - entry)
            dist_atr = dist_pts / atr if atr > 0 else np.nan

            all_records.append({
                "timestamp": ts,
                "direction": direction,
                "signal_type": sig_type,
                "entry_price": entry,
                "stop_price": stop,
                "stop_dist_pts": stop_dist,
                "atr": atr,
                "level_type": level_name,
                "level_price": level_price,
                "dist_pts": dist_pts,
                "dist_atr": dist_atr,
                "hit": res["hit"],
                "bars_to_hit": res["bars_to_hit"],
                "mae_before_hit": res["mae_before_hit"],
                "mfe_total": res["mfe_total"],
            })

    df = pd.DataFrame(all_records)
    logger.info("Forward simulation complete: %d records from %d signals in %.1f sec",
                len(df), len(sampled), time.time() - t_start)

    # ================================================================
    # PART 2: Statistical Summary by Level Type
    # ================================================================
    print("\n" + "=" * 100)
    print("PART 2: STATISTICAL SUMMARY BY LEVEL TYPE")
    print("=" * 100)

    level_types = sorted(df["level_type"].unique())

    header = f"{'Level Type':<20} {'Count':>7} {'Hit%':>7} {'MedBars':>8} {'MedDist':>9} {'MedDistATR':>11} {'AvgMAE':>8} {'MedMAE':>8}"
    print(header)
    print("-" * 100)

    summary_rows = []
    for lt in level_types:
        sub = df[df["level_type"] == lt]
        n = len(sub)
        hit_rate = sub["hit"].mean() * 100
        med_bars = sub.loc[sub["hit"], "bars_to_hit"].median() if sub["hit"].any() else np.nan
        med_dist = sub["dist_pts"].median()
        med_dist_atr = sub["dist_atr"].median()
        avg_mae = sub.loc[sub["hit"], "mae_before_hit"].mean() if sub["hit"].any() else np.nan
        med_mae = sub.loc[sub["hit"], "mae_before_hit"].median() if sub["hit"].any() else np.nan

        summary_rows.append({
            "level_type": lt, "count": n, "hit_rate": hit_rate,
            "med_bars": med_bars, "med_dist": med_dist, "med_dist_atr": med_dist_atr,
            "avg_mae": avg_mae, "med_mae": med_mae,
        })

        print(f"{lt:<20} {n:>7,} {hit_rate:>6.1f}% {med_bars:>8.1f} {med_dist:>9.1f} {med_dist_atr:>11.2f} {avg_mae:>8.1f} {med_mae:>8.1f}")

    # ---- By direction ----
    for dir_label, dir_val in [("LONG", 1), ("SHORT", -1)]:
        print(f"\n--- {dir_label} only ---")
        print(header)
        print("-" * 100)
        for lt in level_types:
            sub = df[(df["level_type"] == lt) & (df["direction"] == dir_val)]
            n = len(sub)
            if n == 0:
                continue
            hit_rate = sub["hit"].mean() * 100
            med_bars = sub.loc[sub["hit"], "bars_to_hit"].median() if sub["hit"].any() else np.nan
            med_dist = sub["dist_pts"].median()
            med_dist_atr = sub["dist_atr"].median()
            avg_mae = sub.loc[sub["hit"], "mae_before_hit"].mean() if sub["hit"].any() else np.nan
            med_mae = sub.loc[sub["hit"], "mae_before_hit"].median() if sub["hit"].any() else np.nan
            print(f"{lt:<20} {n:>7,} {hit_rate:>6.1f}% {med_bars:>8.1f} {med_dist:>9.1f} {med_dist_atr:>11.2f} {avg_mae:>8.1f} {med_mae:>8.1f}")

    # ================================================================
    # PART 3: Distance-Binned Hit Rates
    # ================================================================
    print("\n" + "=" * 100)
    print("PART 3: DISTANCE-BINNED HIT RATES (ATR multiples)")
    print("=" * 100)

    bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 999]
    bin_labels = ["0-0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0", "3.0-5.0", "5.0+"]

    df["dist_bin"] = pd.cut(df["dist_atr"], bins=bins, labels=bin_labels, right=False)

    print(f"\n{'Bin':<12} {'Count':>8} {'Hit%':>7} {'MedBars':>8} {'AvgMAE':>8} {'MedDist':>9}")
    print("-" * 60)

    for label in bin_labels:
        sub = df[df["dist_bin"] == label]
        n = len(sub)
        if n == 0:
            continue
        hit_rate = sub["hit"].mean() * 100
        med_bars = sub.loc[sub["hit"], "bars_to_hit"].median() if sub["hit"].any() else np.nan
        avg_mae = sub.loc[sub["hit"], "mae_before_hit"].mean() if sub["hit"].any() else np.nan
        med_dist = sub["dist_pts"].median()
        print(f"{label:<12} {n:>8,} {hit_rate:>6.1f}% {med_bars:>8.1f} {avg_mae:>8.1f} {med_dist:>9.1f}")

    # ---- By level type within each bin ----
    print(f"\n{'Bin':<12} {'Level Type':<20} {'Count':>8} {'Hit%':>7} {'MedBars':>8}")
    print("-" * 70)
    for label in bin_labels:
        bin_df = df[df["dist_bin"] == label]
        if len(bin_df) == 0:
            continue
        for lt in level_types:
            sub = bin_df[bin_df["level_type"] == lt]
            n = len(sub)
            if n < 50:
                continue
            hit_rate = sub["hit"].mean() * 100
            med_bars = sub.loc[sub["hit"], "bars_to_hit"].median() if sub["hit"].any() else np.nan
            print(f"{label:<12} {lt:<20} {n:>8,} {hit_rate:>6.1f}% {med_bars:>8.1f}")

    # ================================================================
    # PART 4: Best TP Candidates
    # ================================================================
    print("\n" + "=" * 100)
    print("PART 4: BEST TP CANDIDATES")
    print("=" * 100)
    print("Criteria: hit_rate > 60%, dist_atr in [0.5, 4.0], count > 200, ranked by hit_rate / MAE")
    print()

    candidates = []
    for lt in level_types:
        sub = df[df["level_type"] == lt]
        n = len(sub)
        if n < 200:
            continue
        hit_rate = sub["hit"].mean() * 100
        med_dist_atr = sub["dist_atr"].median()
        if hit_rate < 40:  # relaxed for display
            continue
        if med_dist_atr < 0.3 or med_dist_atr > 10:
            continue
        med_bars = sub.loc[sub["hit"], "bars_to_hit"].median() if sub["hit"].any() else np.nan
        avg_mae = sub.loc[sub["hit"], "mae_before_hit"].mean() if sub["hit"].any() else np.nan
        med_mae = sub.loc[sub["hit"], "mae_before_hit"].median() if sub["hit"].any() else np.nan
        # Score: higher hit rate and lower MAE is better
        score = hit_rate / (med_mae + 1) if not np.isnan(med_mae) else 0

        candidates.append({
            "level_type": lt, "count": n, "hit_rate": hit_rate,
            "med_dist_atr": med_dist_atr, "med_bars": med_bars,
            "avg_mae": avg_mae, "med_mae": med_mae, "score": score,
        })

    candidates.sort(key=lambda x: -x["score"])

    print(f"{'Rank':<6} {'Level Type':<20} {'Count':>7} {'Hit%':>7} {'MedDistATR':>11} {'MedBars':>8} {'MedMAE':>8} {'Score':>8}")
    print("-" * 90)
    for i, c in enumerate(candidates):
        star = " ***" if c["hit_rate"] >= 60 and 0.5 <= c["med_dist_atr"] <= 4.0 else ""
        print(f"{i+1:<6} {c['level_type']:<20} {c['count']:>7,} {c['hit_rate']:>6.1f}% "
              f"{c['med_dist_atr']:>11.2f} {c['med_bars']:>8.1f} {c['med_mae']:>8.1f} {c['score']:>8.2f}{star}")

    # ---- TP candidates by direction ----
    for dir_label, dir_val in [("LONG", 1), ("SHORT", -1)]:
        print(f"\n--- {dir_label} TP Candidates ---")
        candidates_dir = []
        for lt in level_types:
            sub = df[(df["level_type"] == lt) & (df["direction"] == dir_val)]
            n = len(sub)
            if n < 100:
                continue
            hit_rate = sub["hit"].mean() * 100
            med_dist_atr = sub["dist_atr"].median()
            if hit_rate < 30:
                continue
            med_bars = sub.loc[sub["hit"], "bars_to_hit"].median() if sub["hit"].any() else np.nan
            med_mae = sub.loc[sub["hit"], "mae_before_hit"].median() if sub["hit"].any() else np.nan
            score = hit_rate / (med_mae + 1) if not np.isnan(med_mae) else 0
            candidates_dir.append({
                "level_type": lt, "count": n, "hit_rate": hit_rate,
                "med_dist_atr": med_dist_atr, "med_bars": med_bars,
                "med_mae": med_mae, "score": score,
            })
        candidates_dir.sort(key=lambda x: -x["score"])
        print(f"{'Rank':<6} {'Level Type':<20} {'Count':>7} {'Hit%':>7} {'MedDistATR':>11} {'MedBars':>8} {'MedMAE':>8} {'Score':>8}")
        print("-" * 90)
        for i, c in enumerate(candidates_dir):
            star = " ***" if c["hit_rate"] >= 60 and 0.5 <= c["med_dist_atr"] <= 4.0 else ""
            print(f"{i+1:<6} {c['level_type']:<20} {c['count']:>7,} {c['hit_rate']:>6.1f}% "
                  f"{c['med_dist_atr']:>11.2f} {c['med_bars']:>8.1f} {c['med_mae']:>8.1f} {c['score']:>8.2f}{star}")

    # ================================================================
    # PART 5: Multi-Level Probability Cascade
    # ================================================================
    print("\n" + "=" * 100)
    print("PART 5: MULTI-LEVEL PROBABILITY CASCADE")
    print("=" * 100)

    # Helper: extract timestamp set as proper pandas Timestamps (not numpy datetime64)
    def _ts_set(sub_df: pd.DataFrame) -> set:
        return set(sub_df["timestamp"].tolist())

    def _run_cascade(dir_signals: pd.DataFrame, level_names: list, label: str):
        print(f"\n{label}:")
        prev_hit_ts = None
        for i, sname in enumerate(level_names):
            sub = dir_signals[dir_signals["level_type"] == sname]
            n_total = len(sub)
            if n_total == 0:
                print(f"  {sname}: no data")
                continue

            if i == 0:
                n_hit = int(sub["hit"].sum())
                hit_rate = n_hit / n_total * 100
                prev_hit_ts = _ts_set(sub[sub["hit"]])
                print(f"  {sname}: {n_hit}/{n_total} = {hit_rate:.1f}% hit rate")
            else:
                if prev_hit_ts is None or len(prev_hit_ts) == 0:
                    print(f"  {sname} | {level_names[i-1]} hit: no prior hits")
                    continue
                conditional = sub[sub["timestamp"].isin(prev_hit_ts)]
                n_cond = len(conditional)
                if n_cond == 0:
                    print(f"  {sname} | {level_names[i-1]} hit: no data (0 conditional signals)")
                    continue
                n_hit = int(conditional["hit"].sum())
                hit_rate = n_hit / n_cond * 100
                prev_hit_ts = _ts_set(conditional[conditional["hit"]])
                print(f"  {sname} | {level_names[i-1]} hit: {n_hit}/{n_cond} = {hit_rate:.1f}% (conditional)")

    for dir_label, dir_val in [("LONG", 1), ("SHORT", -1)]:
        print(f"\n--- {dir_label} Cascade ---")
        dir_signals = df[df["direction"] == dir_val]

        _run_cascade(dir_signals, ["swing_1", "swing_2", "swing_3"], "Swing Level Cascade")

        if dir_val == 1:
            _run_cascade(dir_signals, ["orm_high", "overnight_high", "london_high", "asia_high"], "Session Level Cascade")
        else:
            _run_cascade(dir_signals, ["orm_low", "overnight_low", "london_low", "asia_low"], "Session Level Cascade")

        _run_cascade(dir_signals, ["1R_target", "2R_target", "3R_target"], "Fixed RR Cascade")

        # Cross-type cascade: swing_1 -> irl_target -> 2R_target
        _run_cascade(dir_signals, ["swing_1", "swing_2", "irl_target", "1R_target", "2R_target"], "Progressive Cascade (swing -> IRL -> RR)")

    # ================================================================
    # PART 6: MFE Distribution (bonus)
    # ================================================================
    print("\n" + "=" * 100)
    print("PART 6: MFE DISTRIBUTION (Max Favorable Excursion in 100 bars)")
    print("=" * 100)

    # Get one record per signal for MFE (use 1R_target record since all share mfe_total)
    mfe_df = df[df["level_type"] == "1R_target"].copy()
    if len(mfe_df) > 0:
        mfe_df["mfe_atr"] = mfe_df["mfe_total"] / mfe_df["atr"]

        for dir_label, dir_val in [("ALL", None), ("LONG", 1), ("SHORT", -1)]:
            sub = mfe_df if dir_val is None else mfe_df[mfe_df["direction"] == dir_val]
            if len(sub) == 0:
                continue
            print(f"\n{dir_label} (n={len(sub):,}):")
            pcts = [10, 25, 50, 75, 90, 95]
            print(f"  MFE (points): " + ", ".join(
                [f"P{p}={sub['mfe_total'].quantile(p/100):.1f}" for p in pcts]))
            print(f"  MFE (ATR):    " + ", ".join(
                [f"P{p}={sub['mfe_atr'].quantile(p/100):.2f}" for p in pcts]))
            print(f"  Mean MFE: {sub['mfe_total'].mean():.1f} pts ({sub['mfe_atr'].mean():.2f} ATR)")

    # ================================================================
    # PART 7: Time-to-Hit by Hour of Day
    # ================================================================
    print("\n" + "=" * 100)
    print("PART 7: HIT RATE BY SIGNAL HOUR (ET)")
    print("=" * 100)

    df_hits = df[df["level_type"] == "1R_target"].copy()
    if len(df_hits) > 0:
        # Convert timestamp to ET — use pd.DatetimeIndex for reliable tz conversion
        ts_as_dt = pd.DatetimeIndex(df_hits["timestamp"])
        df_hits["hour_et"] = ts_as_dt.tz_convert("US/Eastern").hour

        # Also add hour_et to the full df for direct lookups
        ts_all = pd.DatetimeIndex(df["timestamp"])
        df["hour_et"] = ts_all.tz_convert("US/Eastern").hour

        print(f"\n{'Hour(ET)':<10} {'Count':>7} {'1R Hit%':>8} {'2R Hit%':>8} {'3R Hit%':>8} {'MedMFE':>9}")
        print("-" * 60)

        for h in range(6, 22):
            mask_h = df_hits["hour_et"] == h
            n = mask_h.sum()
            if n < 50:
                continue

            # Direct filter on full df by hour
            full_mask = df["hour_et"] == h
            r1_sub = df[full_mask & (df["level_type"] == "1R_target")]
            r2_sub = df[full_mask & (df["level_type"] == "2R_target")]
            r3_sub = df[full_mask & (df["level_type"] == "3R_target")]

            r1_hit = r1_sub["hit"].mean() * 100 if len(r1_sub) > 0 else 0
            r2_hit = r2_sub["hit"].mean() * 100 if len(r2_sub) > 0 else 0
            r3_hit = r3_sub["hit"].mean() * 100 if len(r3_sub) > 0 else 0
            med_mfe = r1_sub["mfe_total"].median() if len(r1_sub) > 0 else 0

            print(f"{h:>2}:00     {n:>7,} {r1_hit:>7.1f}% {r2_hit:>7.1f}% {r3_hit:>7.1f}% {med_mfe:>9.1f}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 100)
    print("EXECUTION SUMMARY")
    print("=" * 100)
    print(f"Total signals analyzed: {len(sampled):,} (sampled from {len(signal_indices):,})")
    print(f"Total level-signal records: {len(df):,}")
    print(f"Forward simulation window: {MAX_FORWARD_BARS} bars ({MAX_FORWARD_BARS * 5} minutes)")
    print(f"Runtime: {time.time() - t_start:.1f} seconds")


if __name__ == "__main__":
    main()
