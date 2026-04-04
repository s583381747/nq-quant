"""
features/labeler.py -- Liquidity-based labeling for NQ quantitative system.

Instead of fixed TP/SL barriers, labels are based on Lanto's actual trade logic:
- SL = model stop (candle open or 2nd swing H/L — chart-based)
- TP = nearest internal liquidity target (next swing H/L in trade direction)
- Label 1 = price reaches IRL target before model stop
- Label 0 = model stop hit first, or timeout

This models: "Does the liquidity draw pull price to target before invalidation?"

References: CLAUDE.md §8 (stop loss), §10 (targets, IRL/ERL), §14 (labeling)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "params.yaml"


def _load_params(config_path: Path = _CONFIG_PATH) -> dict[str, Any]:
    """Load tunable parameters from params.yaml."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def label_liquidity_based(
    df: pd.DataFrame,
    entry_mask: pd.Series,
    swing_high_price: pd.Series,
    swing_low_price: pd.Series,
    atr: pd.Series,
    params: dict[str, Any],
    direction: pd.Series | None = None,
) -> pd.DataFrame:
    """Label entries based on whether price reaches IRL target or model stop first.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV with DatetimeIndex (sorted ascending).
    entry_mask : pd.Series[bool]
        True = potential entry bar.
    swing_high_price : pd.Series[float]
        Forward-filled most recent swing high price (from swing.py).
    swing_low_price : pd.Series[float]
        Forward-filled most recent swing low price (from swing.py).
    atr : pd.Series[float]
        ATR(14) at each bar (from displacement.py).
    params : dict
        Full params dict. Uses labeling.max_holding_bars, labeling.min_rr.
    direction : pd.Series[int] | None
        +1 = long, -1 = short. If None, inferred from close vs open.

    Returns
    -------
    pd.DataFrame with columns:
        label, holding_time, exit_type, entry_dir,
        entry_price, tp_price, sl_price, rr_ratio
    """
    cfg = params["labeling"]
    max_hold: int = cfg["max_holding_bars"]
    min_rr: float = cfg.get("min_rr", 1.0)

    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values

    sw_high = swing_high_price.reindex(df.index).values
    sw_low = swing_low_price.reindex(df.index).values
    atr_arr = atr.reindex(df.index).values

    # Direction
    if direction is not None:
        dir_arr = direction.reindex(df.index).values.astype(np.float64)
    else:
        dir_arr = np.where(close > open_, 1.0, -1.0)

    mask = entry_mask.reindex(df.index, fill_value=False).values.astype(bool)
    entry_indices = np.where(mask)[0]

    # Output arrays
    label_arr = np.full(n, np.nan)
    hold_arr = np.full(n, np.nan)
    exit_type_arr = np.empty(n, dtype=object)
    exit_type_arr[:] = ""
    entry_dir_arr = np.full(n, np.nan)
    entry_price_arr = np.full(n, np.nan)
    tp_price_arr = np.full(n, np.nan)
    sl_price_arr = np.full(n, np.nan)
    rr_arr = np.full(n, np.nan)

    skipped_no_levels = 0
    skipped_bad_rr = 0

    for idx in entry_indices:
        d = dir_arr[idx]
        ep = close[idx]
        current_atr = atr_arr[idx]

        if np.isnan(current_atr) or current_atr <= 0:
            skipped_no_levels += 1
            continue

        entry_dir_arr[idx] = d
        entry_price_arr[idx] = ep

        if d > 0:  # LONG
            # SL = candle open (primary), with minimum of 1 ATR below entry
            # Use the lower of: candle open, or entry - 2*ATR as fallback
            sl_level = open_[idx]
            if ep - sl_level < current_atr * 0.5:
                # Candle open too close — use 2nd swing low
                sl_level = sw_low[idx] if not np.isnan(sw_low[idx]) else ep - current_atr * 2
            sl_level = min(sl_level, ep - current_atr * 0.5)  # minimum distance

            # TP = nearest swing high above entry (IRL target)
            tp_level = sw_high[idx] if not np.isnan(sw_high[idx]) else ep + current_atr * 3
            if tp_level <= ep:
                # Swing high is below entry (stale) — use ATR-based target
                tp_level = ep + current_atr * 3

        else:  # SHORT
            sl_level = open_[idx]
            if sl_level - ep < current_atr * 0.5:
                sl_level = sw_high[idx] if not np.isnan(sw_high[idx]) else ep + current_atr * 2
            sl_level = max(sl_level, ep + current_atr * 0.5)

            tp_level = sw_low[idx] if not np.isnan(sw_low[idx]) else ep - current_atr * 3
            if tp_level >= ep:
                tp_level = ep - current_atr * 3

        # Compute R:R
        sl_dist = abs(ep - sl_level)
        tp_dist = abs(tp_level - ep)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0

        if rr < min_rr:
            skipped_bad_rr += 1
            label_arr[idx] = np.nan  # skip this entry
            continue

        tp_price_arr[idx] = tp_level
        sl_price_arr[idx] = sl_level
        rr_arr[idx] = rr

        # Scan forward
        end_idx = min(idx + max_hold, n - 1)
        hit_label = 0.0
        hit_type = "timeout"
        hit_bars = max_hold

        for j in range(idx + 1, end_idx + 1):
            bars_elapsed = j - idx

            if d > 0:  # LONG
                tp_hit = high[j] >= tp_level
                sl_hit = low[j] <= sl_level
            else:  # SHORT
                tp_hit = low[j] <= tp_level
                sl_hit = high[j] >= sl_level

            if tp_hit and sl_hit:
                # Disambiguate: conservative = SL wins
                if d > 0:
                    if open_[j] >= tp_level:
                        hit_label, hit_type = 1.0, "tp"
                    else:
                        hit_label, hit_type = 0.0, "sl"
                else:
                    if open_[j] <= tp_level:
                        hit_label, hit_type = 1.0, "tp"
                    else:
                        hit_label, hit_type = 0.0, "sl"
                hit_bars = bars_elapsed
                break
            elif tp_hit:
                hit_label, hit_type = 1.0, "tp"
                hit_bars = bars_elapsed
                break
            elif sl_hit:
                hit_label, hit_type = 0.0, "sl"
                hit_bars = bars_elapsed
                break

        label_arr[idx] = hit_label
        hold_arr[idx] = hit_bars
        exit_type_arr[idx] = hit_type

    result = pd.DataFrame({
        "label": label_arr,
        "holding_time": hold_arr,
        "exit_type": exit_type_arr,
        "entry_dir": entry_dir_arr,
        "entry_price": entry_price_arr,
        "tp_price": tp_price_arr,
        "sl_price": sl_price_arr,
        "rr_ratio": rr_arr,
    }, index=df.index)

    entries = result["label"].dropna()
    if len(entries) > 0:
        n_tp = (entries == 1.0).sum()
        n_sl = (entries == 0.0).sum()
        logger.info(
            "label_liquidity_based: %d entries — %d TP (%.1f%%), %d SL/timeout (%.1f%%), "
            "%d skipped (no levels: %d, bad RR: %d), mean RR: %.2f",
            len(entries), n_tp, 100 * n_tp / len(entries),
            n_sl, 100 * n_sl / len(entries),
            skipped_no_levels + skipped_bad_rr, skipped_no_levels, skipped_bad_rr,
            result["rr_ratio"].dropna().mean(),
        )

    return result


def compute_entry_candidates(
    df: pd.DataFrame,
    features_dict: dict[str, pd.Series | pd.DataFrame],
    params: dict[str, Any],
) -> pd.Series:
    """Compute boolean mask of bars eligible for labeling.

    Criteria:
    1. NY session
    2. Not ORM period
    3. Not weekend gap
    4. Fluency >= threshold
    5. Active FVG nearby
    """
    fluency_threshold: float = params["fluency"]["threshold"]
    n = len(df)
    mask = np.ones(n, dtype=bool)

    if "session_label" in features_dict:
        session = features_dict["session_label"].reindex(df.index)
        mask &= (session == "ny").values

    if "is_orm_period" in features_dict:
        is_orm = features_dict["is_orm_period"].reindex(df.index).fillna(False).values
        mask &= ~is_orm.astype(bool)

    if "is_weekend_gap" in df.columns:
        mask &= ~df["is_weekend_gap"].values.astype(bool)

    if "fluency" in features_dict:
        fluency = features_dict["fluency"].reindex(df.index).values
        mask &= (fluency >= fluency_threshold)

    if "num_active_bull_fvgs" in features_dict and "num_active_bear_fvgs" in features_dict:
        total_active = (
            features_dict["num_active_bull_fvgs"].reindex(df.index).fillna(0).values
            + features_dict["num_active_bear_fvgs"].reindex(df.index).fillna(0).values
        )
        mask &= total_active > 0

    result = pd.Series(mask, index=df.index, dtype=bool, name="entry_candidate")
    logger.info("compute_entry_candidates: %d / %d (%.1f%%)", mask.sum(), n, 100 * mask.sum() / n if n else 0)
    return result


if __name__ == "__main__":
    import sys
    import time as _time

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from features.displacement import compute_atr
    from features.swing import compute_swing_levels
    from features.sessions import label_sessions, compute_orm

    params = _load_params()
    df = pd.read_parquet(project_root / "data" / "NQ_5m.parquet")
    df = df.loc["2024-01-01":"2024-12-31"].copy()
    logger.info("2024 5m: %d bars", len(df))

    # Compute needed features
    atr = compute_atr(df)
    swing_params = params["swing"]
    swings = compute_swing_levels(df, swing_params)
    sessions = label_sessions(df, params)
    orm = compute_orm(df, params)

    # Entry candidates: NY session, bullish bars, not ORM
    is_ny = sessions == "ny"
    is_not_orm = ~orm["is_orm_period"]
    is_not_weekend = ~df["is_weekend_gap"]
    entry_mask = is_ny & is_not_orm & is_not_weekend

    # Infer direction from candle
    direction = pd.Series(np.where(df["close"] > df["open"], 1.0, -1.0), index=df.index)

    logger.info("Entry candidates: %d", entry_mask.sum())

    t0 = _time.perf_counter()
    result = label_liquidity_based(
        df, entry_mask,
        swings["swing_high_price"], swings["swing_low_price"],
        atr, params, direction,
    )
    elapsed = _time.perf_counter() - t0

    labeled = result[result["label"].notna()]
    total = len(labeled)

    print(f"\n{'='*60}")
    print("LIQUIDITY-BASED LABELING — 2024 NQ 5m")
    print(f"{'='*60}")
    print(f"Total labeled entries: {total:,} ({elapsed:.1f}s)")

    if total > 0:
        wr = labeled["label"].mean()
        print(f"Win rate (IRL reached): {100*wr:.1f}%")
        print(f"Mean R:R ratio: {labeled['rr_ratio'].mean():.2f}")
        print(f"Mean holding time: {labeled['holding_time'].mean():.1f} bars ({labeled['holding_time'].mean()*5:.0f} min)")

        for et in ["tp", "sl", "timeout"]:
            n_et = (labeled["exit_type"] == et).sum()
            print(f"  {et}: {n_et:,} ({100*n_et/total:.1f}%)")

        print(f"\nBy direction:")
        for d_label, d_val in [("LONG", 1.0), ("SHORT", -1.0)]:
            sub = labeled[labeled["entry_dir"] == d_val]
            if len(sub) > 0:
                print(f"  {d_label}: {len(sub):,} entries, WR={100*sub['label'].mean():.1f}%, avg RR={sub['rr_ratio'].mean():.2f}")

    print(f"{'='*60}")
