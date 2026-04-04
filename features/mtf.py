"""
features/mtf.py — Multi-Timeframe feature alignment.

Merges HTF (5m, 15m, 1H, 4H, 1D) features into the 1m base DataFrame
with strict anti-lookahead guarantees:

    1. shift(1) every HTF column before alignment — a 4H candle's
       features only become available after the candle closes, i.e.
       during the *next* 4H period.
    2. reindex to 1m with method="ffill" — carry the last known HTF
       value forward until the next HTF bar closes.

See CLAUDE.md §13 for the full specification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from features.fvg import detect_fvg, compute_active_fvgs
from features.displacement import (
    detect_displacement,
    compute_fluency,
    compute_atr,
    detect_bad_candles,
)

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "params.yaml"


def _load_params(config_path: Path = _CONFIG_PATH) -> dict:
    """Load tunable parameters from params.yaml."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 1. align_htf_to_ltf — generic HTF-to-LTF alignment with shift(1) + ffill
# ---------------------------------------------------------------------------

def align_htf_to_ltf(
    htf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    columns: list[str],
    tf_label: str = "",
) -> pd.DataFrame:
    """Align HTF features to an LTF (lower-timeframe) index.

    For each column in *columns*, the HTF data is:
        1. ``shift(1)``  — so the value at HTF bar T becomes available
           only at HTF bar T+1 (i.e. after bar T has closed).
        2. ``reindex`` to the LTF index with ``method="ffill"`` — forward-fill
           the last known value into every LTF bar until the next HTF close.

    Parameters
    ----------
    htf_df : pd.DataFrame
        Higher-timeframe DataFrame.  Must have a DatetimeIndex (UTC).
    ltf_df : pd.DataFrame
        Lower-timeframe (target) DataFrame.  The returned DataFrame will have
        this index.
    columns : list[str]
        Column names to transfer from *htf_df*.
    tf_label : str, optional
        Timeframe label (e.g. ``"4H"``, ``"1D"``).  If provided, output
        column names are prefixed as ``htf_{tf_label}_{col}``.

    Returns
    -------
    pd.DataFrame
        Aligned to *ltf_df*.index with one column per entry in *columns*.
        Rows before the first available HTF value will be NaN.
    """
    if not columns:
        logger.warning("align_htf_to_ltf: empty column list, returning empty DataFrame")
        return pd.DataFrame(index=ltf_df.index)

    missing = [c for c in columns if c not in htf_df.columns]
    if missing:
        raise ValueError(f"Columns not found in htf_df: {missing}")

    # Step 1: shift(1) — delay HTF features by one HTF bar
    shifted = htf_df[columns].shift(1)

    # Step 2: reindex to LTF index with forward-fill
    # Use the union of indices to ensure correct alignment, then select
    # only LTF timestamps.
    aligned = shifted.reindex(ltf_df.index, method="ffill")

    # Prefix column names with timeframe label
    if tf_label:
        rename_map = {c: f"htf_{tf_label}_{c}" for c in columns}
        aligned = aligned.rename(columns=rename_map)

    n_nan_rows = aligned.isna().all(axis=1).sum()
    logger.info(
        "align_htf_to_ltf [%s]: aligned %d HTF bars → %d LTF bars "
        "(%d columns, %d leading NaN rows)",
        tf_label or "?",
        len(htf_df),
        len(ltf_df),
        len(columns),
        n_nan_rows,
    )
    return aligned


# ---------------------------------------------------------------------------
# 2. _compute_htf_features — compute FVG + displacement features on one TF
# ---------------------------------------------------------------------------

def _compute_htf_features(
    df: pd.DataFrame,
    params: dict,
    tf_label: str,
) -> pd.DataFrame:
    """Compute FVG and displacement features on a single timeframe.

    Returns a DataFrame with the same index as *df* containing:
        - nearest_bull_fvg_dist, nearest_bear_fvg_dist
        - num_active_bull_fvgs, num_active_bear_fvgs
        - nearest_fvg_size
        - is_displacement (bool)
        - fluency (float)
        - atr (float)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data for one timeframe.
    params : dict
        Parsed params.yaml.
    tf_label : str
        Human-readable label (e.g. "4H") for logging.
    """
    logger.info("Computing HTF features for %s (%d bars)...", tf_label, len(df))

    # FVG features (distance to nearest, count of active FVGs)
    fvg_features = compute_active_fvgs(df, params)

    # Displacement and fluency
    is_disp = detect_displacement(df, params)
    fluency = compute_fluency(df, params)
    atr = compute_atr(df, period=14)

    # Combine into a single DataFrame
    result = fvg_features.copy()
    result["is_displacement"] = is_disp.astype(int)  # int for easier aggregation
    result["fluency_score"] = fluency
    result["atr"] = atr

    logger.info(
        "HTF features for %s: %d columns, %d bars",
        tf_label,
        len(result.columns),
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# 3. build_mtf_features — orchestrate multi-TF feature build + alignment
# ---------------------------------------------------------------------------

# Default HTF timeframes to align (in order of priority, lowest first)
_DEFAULT_HTF_LABELS = ["5m", "15m", "1H", "4H", "1D"]


def build_mtf_features(
    dfs: dict[str, pd.DataFrame],
    params: dict | None = None,
    htf_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Build a combined multi-timeframe feature DataFrame aligned to 1m.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Mapping from timeframe label to OHLCV DataFrame.
        Must include ``"1m"`` as the base timeframe.
        Other expected keys: ``"5m"``, ``"15m"``, ``"1H"``, ``"4H"``, ``"1D"``.
    params : dict, optional
        Parsed params.yaml.  If None, loaded from the default path.
    htf_labels : list[str], optional
        Which HTF timeframes to process.  Defaults to all available
        non-1m keys in *dfs* that are in ``_DEFAULT_HTF_LABELS``.

    Returns
    -------
    pd.DataFrame
        Aligned to the 1m index, with columns prefixed by timeframe:
        ``htf_4H_nearest_bull_fvg_dist``, ``htf_1H_fluency_score``, etc.
    """
    if "1m" not in dfs:
        raise ValueError("dfs must contain a '1m' key for the base timeframe")

    if params is None:
        params = _load_params()

    ltf_df = dfs["1m"]

    if htf_labels is None:
        htf_labels = [tf for tf in _DEFAULT_HTF_LABELS if tf in dfs]

    logger.info(
        "build_mtf_features: base=1m (%d bars), HTFs=%s",
        len(ltf_df),
        htf_labels,
    )

    aligned_frames: list[pd.DataFrame] = []

    for tf_label in htf_labels:
        if tf_label not in dfs:
            logger.warning("Timeframe '%s' not found in dfs, skipping", tf_label)
            continue

        htf_df = dfs[tf_label]

        # Compute features on the HTF data
        htf_feats = _compute_htf_features(htf_df, params, tf_label)

        # Align to 1m with shift(1) + ffill
        feature_cols = list(htf_feats.columns)
        aligned = align_htf_to_ltf(
            htf_df=htf_feats,
            ltf_df=ltf_df,
            columns=feature_cols,
            tf_label=tf_label,
        )
        aligned_frames.append(aligned)

    if not aligned_frames:
        logger.warning("build_mtf_features: no HTF features computed")
        return pd.DataFrame(index=ltf_df.index)

    result = pd.concat(aligned_frames, axis=1)

    logger.info(
        "build_mtf_features: final shape %s, columns=%s",
        result.shape,
        list(result.columns),
    )
    return result


# ---------------------------------------------------------------------------
# __main__ — verification test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        stream=sys.stdout,
    )

    logger.info("=" * 70)
    logger.info("MTF Alignment — Verification Test")
    logger.info("=" * 70)

    # ---- Load data ----
    data_dir = Path(__file__).resolve().parent.parent / "data"
    params = _load_params()

    logger.info("Loading 1m and 4H data...")
    df_1m = pd.read_parquet(data_dir / "NQ_1min.parquet")
    df_4h = pd.read_parquet(data_dir / "NQ_4H.parquet")

    # Use a small slice for speed (last ~2 weeks of 1m data)
    SLICE_BARS_1M = 20_000  # ~14 trading days of 1m data
    df_1m_slice = df_1m.iloc[-SLICE_BARS_1M:]

    # Trim 4H to cover the same date range (with some lead-in for ATR warmup)
    start_dt = df_1m_slice.index[0] - pd.Timedelta(days=30)
    df_4h_slice = df_4h.loc[df_4h.index >= start_dt]

    logger.info(
        "1m slice: %d bars [%s → %s]",
        len(df_1m_slice),
        df_1m_slice.index[0],
        df_1m_slice.index[-1],
    )
    logger.info(
        "4H slice: %d bars [%s → %s]",
        len(df_4h_slice),
        df_4h_slice.index[0],
        df_4h_slice.index[-1],
    )

    # ---- Compute 4H features and align to 1m ----
    logger.info("Computing 4H features...")
    htf_feats_4h = _compute_htf_features(df_4h_slice, params, "4H")

    logger.info("Aligning 4H → 1m with shift(1) + ffill...")
    aligned = align_htf_to_ltf(
        htf_df=htf_feats_4h,
        ltf_df=df_1m_slice,
        columns=list(htf_feats_4h.columns),
        tf_label="4H",
    )

    # ---- SHIFT(1) VERIFICATION ----
    # Find a 4H boundary where the feature changed, and verify that the
    # 1m bars in the NEXT 4H period see the PREVIOUS 4H's feature values.
    logger.info("")
    logger.info("=" * 70)
    logger.info("SHIFT(1) VERIFICATION")
    logger.info("=" * 70)

    # Get 4H bars that fall within our 1m slice range (need previous bar too)
    overlap_4h = htf_feats_4h.loc[
        (htf_feats_4h.index >= df_1m_slice.index[0])
        & (htf_feats_4h.index <= df_1m_slice.index[-1])
    ]

    test_col_raw = "fluency_score"
    test_col_aligned = "htf_4H_fluency_score"

    # Use 4H bars from the overlap that have a previous bar available
    # (so we can compare shift(1) behavior)
    test_indices = []
    for idx in overlap_4h.index:
        prev_mask = htf_feats_4h.index < idx
        if prev_mask.any():
            test_indices.append(idx)
        if len(test_indices) >= 5:
            break

    logger.info(
        "Found %d testable 4H boundaries in 1m range",
        len(test_indices),
    )

    all_pass = True

    for htf_boundary in test_indices[:3]:
        # The RAW (unshifted) 4H feature at this boundary
        raw_val = htf_feats_4h.loc[htf_boundary, test_col_raw]

        # The SHIFTED value: after shift(1), the value at htf_boundary
        # is actually what was at the PREVIOUS 4H bar.
        shifted_4h = htf_feats_4h[test_col_raw].shift(1)
        shifted_val = shifted_4h.loc[htf_boundary]

        # What the 1m bars see right at this 4H boundary
        if htf_boundary in aligned.index:
            aligned_val = aligned.loc[htf_boundary, test_col_aligned]
        else:
            # Find nearest 1m bar at or after boundary
            mask = aligned.index >= htf_boundary
            if mask.any():
                nearest = aligned.index[mask][0]
                aligned_val = aligned.loc[nearest, test_col_aligned]
            else:
                continue

        # Find the PREVIOUS 4H bar
        prev_4h_idx = htf_feats_4h.index[htf_feats_4h.index < htf_boundary]
        if len(prev_4h_idx) == 0:
            continue
        prev_4h_bar = prev_4h_idx[-1]
        prev_raw_val = htf_feats_4h.loc[prev_4h_bar, test_col_raw]

        # Check: the 1m bars at 4H boundary T should see the PREVIOUS
        # 4H bar's value (not the current one)
        correct = (
            (pd.isna(aligned_val) and pd.isna(prev_raw_val))
            or (not pd.isna(aligned_val) and np.isclose(aligned_val, prev_raw_val, atol=1e-10))
        )

        status = "PASS" if correct else "FAIL"
        if not correct:
            all_pass = False

        print(f"\n--- 4H boundary: {htf_boundary} ---")
        print(f"  Previous 4H bar:            {prev_4h_bar}")
        print(f"  Raw fluency at PREV 4H:     {prev_raw_val:.6f}")
        print(f"  Raw fluency at THIS 4H:     {raw_val:.6f}")
        print(f"  Aligned 1m value at boundary: {aligned_val:.6f}")
        print(f"  Expected (prev 4H value):   {prev_raw_val:.6f}")
        print(f"  Match: {status}")

        # Also show a few 1m bars just before and at the boundary
        window_start = htf_boundary - pd.Timedelta(minutes=5)
        window_end = htf_boundary + pd.Timedelta(minutes=5)
        window_mask = (aligned.index >= window_start) & (aligned.index <= window_end)
        window_slice = aligned.loc[window_mask, [test_col_aligned]]
        if not window_slice.empty:
            print(f"  1m bars around boundary (+-5min):")
            for ts, row in window_slice.iterrows():
                marker = " <<<" if ts == htf_boundary else ""
                print(f"    {ts}  {test_col_aligned}={row[test_col_aligned]:.6f}{marker}")

    print("\n" + "=" * 70)
    print(f"SHIFT(1) VERIFICATION RESULT: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 70)

    # ---- Show overall alignment sample ----
    print("\n--- Aligned MTF Features (sample of last 10 rows) ---")
    print(aligned.tail(10).to_string())

    # ---- Column summary ----
    print(f"\n--- Aligned columns ({len(aligned.columns)}) ---")
    for col in aligned.columns:
        non_null = aligned[col].notna().sum()
        print(f"  {col}: {non_null}/{len(aligned)} non-null")

    # ---- Quick build_mtf_features test with 4H only ----
    print("\n" + "=" * 70)
    print("build_mtf_features — Quick Test (4H only)")
    print("=" * 70)

    dfs = {"1m": df_1m_slice, "4H": df_4h_slice}
    mtf_result = build_mtf_features(dfs, params=params, htf_labels=["4H"])
    print(f"Result shape: {mtf_result.shape}")
    print(f"Columns: {list(mtf_result.columns)}")
    print(mtf_result.tail(5).to_string())
