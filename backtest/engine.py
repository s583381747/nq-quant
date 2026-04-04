"""
backtest/engine.py -- Full Lanto trade management backtest engine.

Processes signal bars where model probability > threshold, applies:
  - Position sizing with grading (A+/B+/C)
  - Model stop from entry_signals
  - TP1 at IRL target (trim 50%)
  - After TP1: move stop to BE, trail with 2nd swing
  - 0-for-2 rule, daily 2R loss limit
  - One position at a time
  - Commission + slippage

References: CLAUDE.md sections 8, 9, 10, 11, 12
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "params.yaml"


def _load_params(config_path: Path = _CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Position sizing + grading
# ---------------------------------------------------------------------------

def _compute_grade(
    bias_aligned: float,
    bias_confidence: float,
    regime: float,
) -> str:
    """Grade the setup: A+ / B+ / C.

    Recalibrated grading (removed bias_confidence — anti-predictive):
      A+ = aligned + full regime (1.0)
      B+ = aligned + partial regime, OR neutral + full regime
      C  = neutral + partial regime

    Previous grading used bias_confidence >= 0.4 to promote trades to B+.
    Analysis showed confident+partial-regime trades (old B+) had avgR=0.168,
    while low-confidence+aligned trades (old C) had avgR=0.539.
    Confidence was anti-correlated with returns — removed from grading.
    """
    if np.isnan(bias_aligned) or np.isnan(regime):
        return "C"

    if regime == 0.0:
        return "C"

    aligned = bias_aligned > 0.5
    full_regime = regime >= 1.0

    if aligned and full_regime:
        return "A+"
    elif aligned or full_regime:
        return "B+"
    else:
        return "C"


def _compute_contracts(
    r_amount: float,
    stop_distance_points: float,
    point_value: float,
) -> int:
    """Compute number of contracts from dollar risk and stop distance."""
    if stop_distance_points <= 0 or point_value <= 0:
        return 0
    risk_per_contract = stop_distance_points * point_value
    return max(1, int(r_amount / risk_per_contract))


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

def run_backtest(
    df_5m: pd.DataFrame,
    signals_df: pd.DataFrame,
    bias_data: pd.DataFrame,
    regime: pd.Series,
    model: xgb.Booster,
    features_X: pd.DataFrame,
    params: dict | None = None,
    threshold: float = 0.50,
    htf_fvg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run full backtest with Lanto trade management.

    Parameters
    ----------
    df_5m : pd.DataFrame
        5m OHLCV data with DatetimeIndex.
    signals_df : pd.DataFrame
        Output of detect_all_signals, aligned to df_5m.
    bias_data : pd.DataFrame
        Output of compute_daily_bias, aligned to df_5m.
    regime : pd.Series
        Output of compute_regime, aligned to df_5m.
    model : xgb.Booster
        Trained XGBoost model.
    features_X : pd.DataFrame
        Feature matrix for ALL bars (used for model prediction).
    params : dict
        From params.yaml.
    threshold : float
        Minimum model probability to enter.
    htf_fvg : pd.DataFrame | None
        Output of compute_htf_fvg_features (TASK-018), aligned to df_5m.
        Must contain columns: htf_fvg_bullish_active, htf_fvg_bearish_active.
        When provided AND mtf_fvg.enabled is True in params, long entries
        require htf_fvg_bullish_active and short entries require
        htf_fvg_bearish_active (per Lanto: LTF entries align with HTF FVG).

    Returns
    -------
    pd.DataFrame
        Trade log with one row per completed trade.
    """
    if params is None:
        params = _load_params()

    # Extract params
    pos_params = params["position"]
    risk_params = params["risk"]
    bt_params = params["backtest"]
    grading_params = params["grading"]
    trim_params = params["trim"]
    trail_params = params["trail"]

    normal_r = pos_params["normal_r"]
    reduced_r = pos_params["reduced_r"]
    point_value = pos_params["point_value"]
    daily_max_loss_r = risk_params["daily_max_loss_r"]
    max_consec_losses = risk_params["max_consecutive_losses"]

    commission_per_side = bt_params["commission_per_side_micro"]  # MNQ micro contracts
    slippage_ticks = bt_params["slippage_normal_ticks"]
    slippage_points = slippage_ticks * 0.25  # NQ tick = 0.25 pts

    a_plus_mult = grading_params["a_plus_size_mult"]
    b_plus_mult = grading_params["b_plus_size_mult"]
    c_skip = grading_params["c_skip"]

    trim_pct = trim_params["pct"]
    be_after_trim = trim_params["be_after_trim"]
    nth_swing = trail_params["use_nth_swing"]

    # Direction-aware trade management (direction_mgmt section)
    dir_mgmt = params.get("direction_mgmt", {})

    # Dual-mode: direction-specific SQ thresholds + short scalp TP
    dual_mode = params.get("dual_mode", {})
    dual_mode_enabled = dual_mode.get("enabled", False)

    # MSS-specific trade management (separate TP/trim for reversal signals)
    mss_mgmt = params.get("mss_management", {})
    mss_mgmt_enabled = mss_mgmt.get("enabled", False)

    # ---- News filter (CLAUDE.md §11.5) ----
    news_params = params.get("news", {})
    news_blackout = None
    calendar_path = Path(__file__).resolve().parent.parent / "config" / "news_calendar.csv"
    if calendar_path.exists() and news_params.get("blackout_minutes_before", 0) > 0:
        from features.news_filter import build_news_blackout_mask
        news_blackout = build_news_blackout_mask(
            df_5m.index,
            str(calendar_path),
            news_params.get("blackout_minutes_before", 60),
            news_params.get("cooldown_minutes_after", 5),
        )
        news_blackout_arr = news_blackout.values  # bool numpy array for fast access
        logger.info(
            "News filter: %d / %d bars in blackout (%.1f%%)",
            news_blackout_arr.sum(), len(news_blackout_arr),
            100.0 * news_blackout_arr.sum() / len(news_blackout_arr),
        )
    else:
        news_blackout_arr = None

    # ---- MTF FVG alignment filter (TASK-018, CLAUDE.md §3.1) ----
    mtf_fvg_params = params.get("mtf_fvg", {})
    mtf_fvg_enabled = mtf_fvg_params.get("enabled", False) and htf_fvg is not None
    htf_fvg_bull_arr: np.ndarray | None = None
    htf_fvg_bear_arr: np.ndarray | None = None

    if mtf_fvg_enabled:
        require_alignment = mtf_fvg_params.get("require_alignment", True)
        if require_alignment and htf_fvg is not None:
            # Align htf_fvg to df_5m index (reindex in case of slight misalignment)
            htf_aligned = htf_fvg.reindex(df_5m.index, method="ffill")
            htf_fvg_bull_arr = htf_aligned["htf_fvg_bullish_active"].fillna(False).values.astype(bool)
            htf_fvg_bear_arr = htf_aligned["htf_fvg_bearish_active"].fillna(False).values.astype(bool)
            logger.info(
                "MTF FVG filter: ENABLED. bull_active=%.1f%%, bear_active=%.1f%%",
                100.0 * htf_fvg_bull_arr.mean(),
                100.0 * htf_fvg_bear_arr.mean(),
            )
        else:
            mtf_fvg_enabled = False
            logger.info("MTF FVG filter: disabled (require_alignment=False)")
    else:
        if htf_fvg is not None and not mtf_fvg_params.get("enabled", False):
            logger.info("MTF FVG filter: disabled by config (mtf_fvg.enabled=false)")
        elif htf_fvg is None:
            logger.debug("MTF FVG filter: no htf_fvg data provided")

    # ---- HTF FVG TP target (DECISION-006 Application A) ----
    # Use HTF FVG as TP when available — only EXTEND TP, never shrink
    tp_target_enabled = mtf_fvg_params.get("tp_target_enabled", False) and htf_fvg is not None
    htf_fvg_bear_dist_arr: np.ndarray | None = None
    htf_fvg_bull_dist_arr: np.ndarray | None = None
    tp_min_dist_atr = mtf_fvg_params.get("tp_min_dist_atr", 2.0)
    tp_max_dist_atr = mtf_fvg_params.get("tp_max_dist_atr", 8.0)

    if tp_target_enabled:
        htf_aligned_tp = htf_fvg.reindex(df_5m.index, method="ffill")
        # bear_dist: positive = bearish FVG above price (TP target for longs)
        htf_fvg_bear_dist_arr = htf_aligned_tp["htf_fvg_nearest_bear_dist"].values.astype(float)
        # bull_dist: positive = bullish FVG below price (TP target for shorts)
        htf_fvg_bull_dist_arr = htf_aligned_tp["htf_fvg_nearest_bull_dist"].values.astype(float)
        logger.info(
            "HTF FVG TP target: ENABLED. min_dist=%.1f×ATR, max_dist=%.1f×ATR",
            tp_min_dist_atr, tp_max_dist_atr,
        )

    # ---- Prepare arrays ----
    n = len(df_5m)
    open_arr = df_5m["open"].values
    high_arr = df_5m["high"].values
    low_arr = df_5m["low"].values
    close_arr = df_5m["close"].values

    sig_mask = signals_df["signal"].values.astype(bool)
    sig_dir = signals_df["signal_dir"].values.astype(float)
    sig_type = signals_df["signal_type"].values
    if "has_smt" in signals_df.columns:
        has_smt_arr = signals_df["has_smt"].values.astype(bool)
    else:
        has_smt_arr = np.zeros(len(signals_df), dtype=bool)
        if params.get("smt", {}).get("enabled", False):
            logger.warning("SMT enabled but 'has_smt' column missing — all SMT checks will be False")
    entry_price_arr = signals_df["entry_price"].values
    model_stop_arr = signals_df["model_stop"].values
    irl_target_arr = signals_df["irl_target"].values

    bias_dir_arr = bias_data["bias_direction"].values.astype(float)
    bias_conf_arr = bias_data["bias_confidence"].values.astype(float)
    regime_arr = regime.values.astype(float)

    # ATR for post-entry PA quality check
    from features.displacement import compute_atr, compute_fluency
    atr_series = compute_atr(df_5m, period=14)
    atr_arr = atr_series.values

    # Fluency for signal quality scoring
    fluency_series = compute_fluency(df_5m, params)
    fluency_arr = fluency_series.values

    # PA quality: alternating direction ratio (chop detection)
    from features.pa_quality import compute_alternating_dir_ratio
    pa_alt_arr = compute_alternating_dir_ratio(df_5m, window=6).values

    # Compute swing levels for trailing
    from features.swing import compute_swing_levels
    swing_params = {"left_bars": params["swing"]["left_bars"],
                    "right_bars": params["swing"]["right_bars"]}
    swings = compute_swing_levels(df_5m, swing_params)
    # FIX 4: Shift swing levels by 1 bar to account for right_bars lookahead
    swings["swing_high_price"] = swings["swing_high_price"].shift(1).ffill()
    swings["swing_low_price"] = swings["swing_low_price"].shift(1).ffill()
    swing_high_arr = swings["swing_high_price"].values
    swing_low_arr = swings["swing_low_price"].values

    # Track swing highs/lows for nth swing trailing
    # Shift boolean masks by 1 bar too — swing at bar i is only confirmed at bar i+1
    swing_high_mask = swings["swing_high"].shift(1, fill_value=False).values
    swing_low_mask = swings["swing_low"].shift(1, fill_value=False).values

    # ---- Model predictions on signal bars ----
    # We need to predict on signal bars that fall in our test period
    # Build DMatrix for signal bars
    signal_indices = np.where(sig_mask)[0]

    # Get predictions for signal bars
    model_probs = np.full(n, np.nan)
    if len(signal_indices) > 0:
        # Get feature rows for signal bars that exist in features_X
        valid_signal_times = []
        valid_signal_idx = []
        for si in signal_indices:
            ts = df_5m.index[si]
            if ts in features_X.index:
                valid_signal_times.append(ts)
                valid_signal_idx.append(si)

        if valid_signal_times:
            X_sig = features_X.loc[valid_signal_times]
            dmat = xgb.DMatrix(X_sig, feature_names=list(features_X.columns))
            preds = model.predict(dmat)
            for i, si in enumerate(valid_signal_idx):
                model_probs[si] = preds[i]

    # ---- Signal quality score (Sprint 4 Attack 1, NODE 10) ----
    # Pre-compute composite quality at each signal bar.
    # Components: FVG size, displacement body/range, fluency, PA cleanliness.
    # Momentum excluded (counterproductive — chasing signals score high but lose).
    sq_params = params.get("signal_quality", {})
    sq_enabled = sq_params.get("enabled", False)
    sq_threshold = sq_params.get("threshold", 0.66)
    sq_w_size = sq_params.get("w_size", 0.30)
    sq_w_disp = sq_params.get("w_disp", 0.30)
    sq_w_flu = sq_params.get("w_flu", 0.20)
    sq_w_pa = sq_params.get("w_pa", 0.20)

    signal_quality = np.full(n, np.nan)
    if sq_enabled and len(signal_indices) > 0:
        for idx in signal_indices:
            a = atr_arr[idx] if not np.isnan(atr_arr[idx]) else 10.0

            # 1. Size: entry-stop distance / (ATR * 1.5), capped at 1.0
            gap = abs(entry_price_arr[idx] - model_stop_arr[idx])
            size_sc = min(1.0, gap / (a * 1.5)) if a > 0 else 0.5

            # 2. Displacement: signal candle body/range
            body = abs(close_arr[idx] - open_arr[idx])
            rng = high_arr[idx] - low_arr[idx]
            disp_sc = body / rng if rng > 0 else 0.0

            # 3. Fluency (from pre-computed fluency array)
            flu_val = fluency_arr[idx]
            flu_sc = min(1.0, max(0.0, flu_val)) if not np.isnan(flu_val) else 0.5

            # 4. PA cleanliness: 1 - alternating direction ratio (6-bar window)
            window = 6
            if idx >= window:
                dirs = np.sign(close_arr[idx - window:idx] - open_arr[idx - window:idx])
                alt = np.sum(dirs[1:] != dirs[:-1]) / (window - 1)
                pa_sc = 1.0 - alt
            else:
                pa_sc = 0.5

            signal_quality[idx] = (
                sq_w_size * size_sc
                + sq_w_disp * disp_sc
                + sq_w_flu * flu_sc
                + sq_w_pa * pa_sc
            )

        sq_valid = signal_quality[~np.isnan(signal_quality)]
        logger.info(
            "Signal quality: enabled, threshold=%.2f, scored %d signals, "
            "mean=%.3f, p25=%.3f, p75=%.3f",
            sq_threshold, len(sq_valid),
            sq_valid.mean() if len(sq_valid) > 0 else 0,
            np.percentile(sq_valid, 25) if len(sq_valid) > 0 else 0,
            np.percentile(sq_valid, 75) if len(sq_valid) > 0 else 0,
        )

    # ---- Date tracking for daily risk rules ----
    et_index = df_5m.index.tz_convert("US/Eastern")
    # FIX 7: Session date — bars from 18:00-23:59 ET belong to next day's session
    # (futures session starts at 6PM ET the prior calendar day)
    dates = np.array([
        (et_index[j] + pd.Timedelta(days=1)).date() if et_index[j].hour >= 18
        else et_index[j].date()
        for j in range(n)
    ])

    # ---- Backtest state ----
    trades: list[dict] = []

    # Daily state
    current_date = None
    daily_pnl_r = 0.0
    consecutive_losses = 0
    day_stopped = False

    # Position state
    in_position = False
    pos_direction = 0
    pos_entry_idx = 0
    pos_entry_price = 0.0
    pos_stop = 0.0
    pos_tp1 = 0.0
    pos_contracts = 0
    pos_remaining_contracts = 0
    pos_trimmed = False
    pos_be_stop = 0.0
    pos_trail_stop = 0.0
    pos_signal_type = ""
    pos_bias_dir = 0.0
    pos_regime = 0.0
    pos_model_prob = 0.0
    pos_grade = ""
    pos_trim_pct = trim_pct  # direction-aware trim percentage

    for i in range(n):
        bar_date = dates[i]

        # ---- New day reset ----
        if bar_date != current_date:
            current_date = bar_date
            daily_pnl_r = 0.0
            consecutive_losses = 0
            day_stopped = False

        # ---- Check exit conditions for open position ----
        if in_position:
            exited = False
            exit_reason = ""
            exit_price = 0.0
            exit_contracts = pos_remaining_contracts

            # --- Post-entry PA quality check (early exit) ---
            # Lanto: "I do cut a trade before my model stop if I see consolidation
            # or wicks forming, or if I don't see propulsion in speed and momentum."
            bars_in_trade = i - pos_entry_idx
            if not pos_trimmed and 2 <= bars_in_trade <= 4:
                # Check PA quality of last 2-3 bars since entry
                # Wick ratio: high wicks = indecision
                pa_start = max(pos_entry_idx, 0)
                pa_end = i + 1
                pa_range = high_arr[pa_start:pa_end] - low_arr[pa_start:pa_end]
                pa_body = np.abs(close_arr[pa_start:pa_end] - open_arr[pa_start:pa_end])
                safe_pa_range = np.where(pa_range > 0, pa_range, 1.0)
                pa_wick_ratio = 1.0 - (pa_body / safe_pa_range)
                avg_wick = float(np.mean(pa_wick_ratio))

                # Direction consistency: are bars moving in our direction?
                pa_dirs = np.sign(close_arr[pa_start:pa_end] - open_arr[pa_start:pa_end])
                favorable = (pa_dirs == pos_direction).mean()

                # Price displacement: has price moved away from entry?
                if pos_direction == 1:
                    displacement_from_entry = close_arr[i] - pos_entry_price
                else:
                    displacement_from_entry = pos_entry_price - close_arr[i]

                # Early exit conditions:
                # 1. High wick ratio (>0.65) + unfavorable direction + no displacement
                # 2. Price hasn't moved 0.5 ATR in our direction after 3 bars
                cur_atr_val = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                no_progress = displacement_from_entry < cur_atr_val * 0.3
                bad_pa = avg_wick > 0.65 and favorable < 0.5

                if bad_pa and no_progress and bars_in_trade >= 3:
                    # FIX: exit at next bar open (can't exit at current bar close)
                    exit_price = open_arr[i + 1] if i + 1 < n else close_arr[i]
                    exit_reason = "early_cut_pa"
                    exited = True

            # FIX: guard stop/TP checks — if early_cut already fired, skip
            if not exited and pos_direction == 1:  # LONG
                # Check stop hit
                effective_stop = pos_trail_stop if pos_trimmed and pos_trail_stop > 0 else pos_stop
                if pos_trimmed and be_after_trim and pos_be_stop > 0:
                    effective_stop = max(effective_stop, pos_be_stop)

                if low_arr[i] <= effective_stop:
                    exit_price = effective_stop - slippage_points
                    if pos_trimmed and effective_stop >= pos_entry_price:
                        exit_reason = "be_sweep"
                    else:
                        exit_reason = "stop"
                    exited = True

                # Check TP1 (if not yet trimmed)
                elif not pos_trimmed and high_arr[i] >= pos_tp1:
                    # Trim at TP1 (direction-aware: pos_trim_pct)
                    trim_contracts = max(1, int(pos_contracts * pos_trim_pct))
                    pos_remaining_contracts = pos_contracts - trim_contracts

                    # Record the trim as partial exit
                    # TP1 is a limit order — no favorable slippage (fill at limit or not at all)
                    trim_pnl_pts = pos_tp1 - pos_entry_price
                    trim_pnl_dollars = trim_pnl_pts * point_value * trim_contracts
                    trim_commission = commission_per_side * 2 * trim_contracts

                    pos_trimmed = True
                    pos_be_stop = pos_entry_price  # move to BE

                    # Only compute trailing stop if there are remaining contracts
                    if pos_remaining_contracts > 0:
                        pos_trail_stop = _find_nth_swing(
                            swing_low_mask, df_5m["low"].values, i, nth_swing, direction=1
                        )
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop

                    if pos_remaining_contracts <= 0:
                        # All trimmed -- full exit at TP1 (limit order, no slippage)
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True

                # Update trailing stop for remaining position
                if pos_trimmed and not exited:
                    new_trail = _find_nth_swing(
                        swing_low_mask, df_5m["low"].values, i, nth_swing, direction=1
                    )
                    if not np.isnan(new_trail) and new_trail > pos_trail_stop:
                        pos_trail_stop = new_trail

            elif not exited:  # SHORT
                effective_stop = pos_trail_stop if pos_trimmed and pos_trail_stop > 0 else pos_stop
                if pos_trimmed and be_after_trim and pos_be_stop > 0:
                    effective_stop = min(effective_stop, pos_be_stop)

                if high_arr[i] >= effective_stop:
                    exit_price = effective_stop + slippage_points
                    if pos_trimmed and effective_stop <= pos_entry_price:
                        exit_reason = "be_sweep"
                    else:
                        exit_reason = "stop"
                    exited = True

                elif not pos_trimmed and low_arr[i] <= pos_tp1:
                    # Trim at TP1 (direction-aware: pos_trim_pct)
                    trim_contracts = max(1, int(pos_contracts * pos_trim_pct))
                    pos_remaining_contracts = pos_contracts - trim_contracts

                    pos_trimmed = True
                    pos_be_stop = pos_entry_price

                    # Only compute trailing stop if there are remaining contracts
                    if pos_remaining_contracts > 0:
                        pos_trail_stop = _find_nth_swing(
                            swing_high_mask, df_5m["high"].values, i, nth_swing, direction=-1
                        )
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop

                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1  # limit order, no slippage
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True

                if pos_trimmed and not exited:
                    new_trail = _find_nth_swing(
                        swing_high_mask, df_5m["high"].values, i, nth_swing, direction=-1
                    )
                    if not np.isnan(new_trail) and new_trail < pos_trail_stop:
                        pos_trail_stop = new_trail

            if exited:
                # Compute PnL
                if pos_direction == 1:
                    pnl_pts = exit_price - pos_entry_price
                else:
                    pnl_pts = pos_entry_price - exit_price

                if pos_trimmed and exit_reason != "tp1":
                    # Partial: trim already happened at TP1
                    # Remaining contracts exit here
                    remaining_pnl_pts = pnl_pts
                    remaining_pnl_dollars = remaining_pnl_pts * point_value * exit_contracts

                    # Total PnL includes trim + remaining
                    trim_pnl_pts_total = (pos_tp1 - pos_entry_price) if pos_direction == 1 else (pos_entry_price - pos_tp1)
                    trim_c = pos_contracts - exit_contracts
                    total_pnl_dollars = (
                        trim_pnl_pts_total * point_value * trim_c
                        + remaining_pnl_pts * point_value * exit_contracts
                    )
                    total_commission = commission_per_side * 2 * pos_contracts
                    total_pnl_dollars -= total_commission
                    effective_pnl_pts = total_pnl_dollars / (point_value * pos_contracts) if pos_contracts > 0 else 0
                else:
                    # Simple exit (stop before trim, or full TP1 exit)
                    total_pnl_dollars = pnl_pts * point_value * exit_contracts
                    total_commission = commission_per_side * 2 * exit_contracts
                    total_pnl_dollars -= total_commission
                    effective_pnl_pts = pnl_pts

                # Compute R-multiple
                stop_dist = abs(pos_entry_price - pos_stop)
                r_per_contract = stop_dist * point_value
                total_risk = r_per_contract * pos_contracts
                r_multiple = total_pnl_dollars / total_risk if total_risk > 0 else 0.0

                trades.append({
                    "entry_time": df_5m.index[pos_entry_idx],
                    # AUDIT FIX: early_cut exits at open[i+1], so exit_time should be i+1
                    "exit_time": df_5m.index[i + 1] if exit_reason == "early_cut_pa" and i + 1 < n else df_5m.index[i],
                    "direction": pos_direction,
                    "entry_price": pos_entry_price,
                    "exit_price": exit_price,
                    "stop_price": pos_stop,
                    "tp1_price": pos_tp1,
                    "contracts": pos_contracts,
                    "pnl_points": effective_pnl_pts,
                    "pnl_dollars": total_pnl_dollars,
                    "r_multiple": r_multiple,
                    "exit_reason": exit_reason,
                    "signal_type": pos_signal_type,
                    "bias_direction": pos_bias_dir,
                    "regime": pos_regime,
                    "model_prob": pos_model_prob,
                    "grade": pos_grade,
                    "trimmed": pos_trimmed,
                })

                # Update daily state
                daily_pnl_r += r_multiple

                # Loss tracking for 0-for-2
                if exit_reason == "be_sweep" and pos_trimmed:
                    # BE sweep after trim is profitable -- NOT a loss
                    pass
                elif r_multiple < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                # Check daily limits
                if consecutive_losses >= max_consec_losses:
                    day_stopped = True
                if daily_pnl_r <= -daily_max_loss_r:
                    day_stopped = True

                in_position = False

        # ---- Check for new entry ----
        # News filter: skip entry if bar is in blackout window (CLAUDE.md §11.5)
        # Only blocks NEW entries; existing position management above is unaffected.
        if not in_position and news_blackout_arr is not None and news_blackout_arr[i]:
            continue

        # FIX 3: ORM no-trade window (9:30-10:00 ET observation only)
        if not in_position and not day_stopped:
            et_hour = et_index[i].hour
            et_min = et_index[i].minute
            if (et_hour == 9 and et_min >= 30) or (et_hour == 10 and et_min == 0):
                continue  # ORM observation period — no entries

        if not in_position and not day_stopped and sig_mask[i]:
            prob = model_probs[i]
            if np.isnan(prob) or prob < threshold:
                continue

            direction = int(sig_dir[i])
            if direction == 0:
                continue

            # Signal type filter
            sig_filter = params.get("signal_filter", {})
            if not sig_filter.get("allow_mss", True) and str(sig_type[i]) == "mss":
                continue
            if not sig_filter.get("allow_trend", True) and str(sig_type[i]) == "trend":
                continue

            entry_p = entry_price_arr[i]
            stop = model_stop_arr[i]
            tp1 = irl_target_arr[i]

            if np.isnan(entry_p) or np.isnan(stop) or np.isnan(tp1):
                continue

            # Validate stop/target make sense
            if direction == 1:
                if stop >= entry_p or tp1 <= entry_p:
                    continue
            else:
                if stop <= entry_p or tp1 >= entry_p:
                    continue

            # --- FILTER 1: Bias — block opposing, allow aligned + neutral ---
            # Neutral bias (bias_dir=0) has positive expectancy; only block opposing.
            # Exception: MSS + SMT divergence = confirmed reversal → exempt from bias filter.
            # Also: opposing shorts validated +11.7R WR=72.2% (optional, via bias_relaxation).
            bias_opposing = (direction == -np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0)
            if bias_opposing:
                smt_cfg = params.get("smt", {})
                bias_relax = params.get("bias_relaxation", {})
                # MSS + SMT = confirmed reversal → bypass bias filter
                if smt_cfg.get("enabled", False) and has_smt_arr[i] and str(sig_type[i]) == "mss":
                    pass  # SMT-confirmed MSS reversal: allow through
                elif bias_relax.get("enabled", False) and direction == -1:
                    pass  # opposing shorts relaxation (validated separately)
                else:
                    continue

            # --- FILTER 1b: PA quality — alt_dir threshold from params ---
            pa_threshold = params.get("pa_quality", {}).get("alt_dir_threshold", 0.334)
            if pa_alt_arr[i] >= pa_threshold:
                continue

            # --- FILTER 1c: Session filter (configurable) ---
            # Previously: NY=long-only, London=short-only, Asia=skip (no statistical basis).
            # NY shorts WR=57.1% > NY longs WR=53.1% — old filter blocked better signals.
            # Now configurable via params; DEFAULT OFF.
            et_h = et_index[i].hour
            et_m = et_index[i].minute
            et_frac = et_h + et_m / 60.0

            session_filter = params.get("session_filter", {})
            sf_enabled = session_filter.get("enabled", False)  # DEFAULT OFF
            # MSS+SMT can bypass session filter — reversals happen at session boundaries
            is_mss_smt = (str(sig_type[i]) == "mss" and has_smt_arr[i]
                          and params.get("smt", {}).get("enabled", False))
            mss_bypass_session = is_mss_smt and params.get("smt", {}).get("bypass_session_filter", False)
            if sf_enabled and not mss_bypass_session:
                if 9.5 <= et_frac < 16.0:  # NY (after ORM)
                    allowed_dir = session_filter.get("ny_direction", 0)  # 0=both, 1=long, -1=short
                    if allowed_dir != 0 and direction != allowed_dir:
                        continue
                elif 3.0 <= et_frac < 9.5:  # London
                    if session_filter.get("skip_london", False):
                        continue
                    allowed_dir = session_filter.get("london_direction", 0)
                    if allowed_dir != 0 and direction != allowed_dir:
                        continue
                else:  # Asia / Other
                    if session_filter.get("skip_asia", True):
                        continue
            elif not sf_enabled and not mss_bypass_session:
                # Even with filter off, still skip Asia (low volume, no edge)
                if not (3.0 <= et_frac < 16.0):
                    continue

            # --- FILTER 2: Minimum stop distance (ATR-relative, TASK-014) ---
            stop_dist = abs(entry_p - stop)
            min_stop_atr = params.get("regime", {}).get("min_stop_atr_mult", 0.5)
            min_stop = min_stop_atr * atr_arr[i] if not np.isnan(atr_arr[i]) else 10.0
            if stop_dist < min_stop:
                continue  # candle too small = not real displacement

            # --- FILTER 3: MTF FVG alignment (TASK-018, CLAUDE.md §3.1) ---
            # Per Lanto: LTF entries must align with HTF FVG direction.
            # Long needs active bullish HTF FVG; short needs active bearish HTF FVG.
            if mtf_fvg_enabled and htf_fvg_bull_arr is not None:
                if direction == 1 and not htf_fvg_bull_arr[i]:
                    continue  # long but no HTF bullish FVG active
                if direction == -1 and not htf_fvg_bear_arr[i]:
                    continue  # short but no HTF bearish FVG active

            # --- FILTER 4: Signal quality score (Sprint 4 Attack 1) ---
            # Composite score of FVG size, displacement, fluency, PA cleanliness.
            # Removes low-quality entries (choppy PA, wicky candles).
            # Dual-mode: shorts require higher SQ threshold (0.80 vs 0.66 for longs).
            if sq_enabled and not np.isnan(signal_quality[i]):
                effective_sq_threshold = sq_threshold
                if dual_mode_enabled and direction == -1:
                    effective_sq_threshold = dual_mode.get("short_sq_threshold", 0.80)
                if signal_quality[i] < effective_sq_threshold:
                    continue

            # FIX 1: Entry price is open[i+1] (already set by entry_signals).
            # No 5m confirmation bar needed — use entry_price_arr[i] directly.
            # Skip if this is the last bar (no next bar to enter on).
            if i + 1 >= n:
                continue

            # Grade the setup
            ba = 1.0 if (direction == np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0) else 0.0
            grade = _compute_grade(ba, bias_conf_arr[i], regime_arr[i])

            if grade == "C" and c_skip:
                continue

            # Position sizing
            is_reduced = (et_index[i].dayofweek in (0, 4)) or (regime_arr[i] < 1.0)
            base_r = reduced_r if is_reduced else normal_r

            if grade == "A+":
                r_amount = base_r * a_plus_mult
            elif grade == "B+":
                r_amount = base_r * b_plus_mult
            else:
                r_amount = base_r * 0.5

            # --- Session regime sizing (NODE 6: intraday time-of-day risk) ---
            # Lanto's preferred window is "open to ~11:00 AM." Lunch is dead zone.
            # Adjusts r_amount based on empirical trade quality by session window.
            session_regime = params.get("session_regime", {})
            if session_regime.get("enabled", False):
                sr_am_end = session_regime.get("am_end", 12.0)
                sr_lunch_start = session_regime.get("lunch_start", 12.0)
                sr_lunch_end = session_regime.get("lunch_end", 13.5)
                sr_pm_start = session_regime.get("pm_start", 13.5)
                if et_frac < sr_am_end:
                    sr_mult = session_regime.get("am_mult", 1.0)
                elif sr_lunch_start <= et_frac < sr_lunch_end:
                    sr_mult = session_regime.get("lunch_mult", 0.5)
                elif et_frac >= sr_pm_start:
                    sr_mult = session_regime.get("pm_mult", 0.75)
                else:
                    sr_mult = 1.0
                r_amount *= sr_mult
                if r_amount <= 0:
                    continue  # session regime skips this trade (mult=0)

            # Apply slippage to entry BEFORE position sizing
            if direction == 1:
                actual_entry = entry_p + slippage_points
            else:
                actual_entry = entry_p - slippage_points

            # AUDIT FIX: compute stop_dist from actual_entry (post-slippage).
            # This ensures position sizing, TP, and R-multiple all use the same anchor.
            stop_dist = abs(actual_entry - stop)
            if stop_dist < 1.0:
                continue  # safety check

            contracts = _compute_contracts(r_amount, stop_dist, point_value)
            if contracts <= 0:
                continue

            # --- TP multiplier (direction-aware when enabled) ---
            session_rules = params.get("session_rules", {})
            is_mss_signal = str(sig_type[i]) == "mss"
            if session_rules.get("enabled", False):
                ny_tp_mult = session_rules.get("ny_tp_multiplier", 1.0)
                if dir_mgmt.get("enabled", False):
                    if direction == 1:
                        actual_tp_mult = dir_mgmt.get("long_tp_mult", ny_tp_mult)
                    else:
                        actual_tp_mult = dir_mgmt.get("short_tp_mult", 1.25)
                else:
                    actual_tp_mult = ny_tp_mult

                # MSS Long override: use mss_management.long_tp_mult instead
                if mss_mgmt_enabled and is_mss_signal and direction == 1:
                    actual_tp_mult = mss_mgmt.get("long_tp_mult", actual_tp_mult)

                if 9.5 <= et_frac < 16.0:
                    # AUDIT FIX: anchor TP from actual_entry, not entry_p
                    tp_distance = tp1 - actual_entry if direction == 1 else actual_entry - tp1
                    tp1 = actual_entry + tp_distance * actual_tp_mult if direction == 1 else actual_entry - tp_distance * actual_tp_mult

            # --- HTF FVG TP target (DECISION-006 Application A) ---
            # Only EXTEND TP when HTF FVG provides a structurally better target
            # Lanto: HTF FVGs are magnets/draw on liquidity — use as TP targets
            if tp_target_enabled and htf_fvg_bear_dist_arr is not None:
                cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 8.0
                min_fvg_dist = tp_min_dist_atr * cur_atr
                max_fvg_dist = tp_max_dist_atr * cur_atr

                if direction == 1:  # LONG — target is bearish FVG above price
                    fvg_dist = htf_fvg_bear_dist_arr[i]
                    if not np.isnan(fvg_dist) and fvg_dist > 0 and min_fvg_dist < fvg_dist < max_fvg_dist:
                        fvg_tp = entry_p + fvg_dist
                        if fvg_tp > tp1:  # only extend, never shrink
                            tp1 = fvg_tp
                else:  # SHORT — target is bullish FVG below price
                    fvg_dist = htf_fvg_bull_dist_arr[i]
                    if not np.isnan(fvg_dist) and fvg_dist > 0 and min_fvg_dist < fvg_dist < max_fvg_dist:
                        fvg_tp = entry_p - fvg_dist
                        if fvg_tp < tp1:  # only extend, never shrink
                            tp1 = fvg_tp

            # --- Dual-mode short TP override ---
            # Shorts use a fixed R:R scalp target instead of swing-based TP.
            # Applied AFTER session_rules TP mult and HTF FVG TP, overriding both.
            if dual_mode_enabled and direction == -1:
                short_rr = dual_mode.get("short_rr", 0.625)
                # MSS Short override: use mss_management.short_rr
                if mss_mgmt_enabled and is_mss_signal:
                    short_rr = mss_mgmt.get("short_rr", short_rr)
                # AUDIT FIX: anchor from actual_entry, not entry_p
                tp1 = actual_entry - stop_dist * short_rr

            # Enter position (FIX 1: entry happens on bar i+1, not bar i)
            in_position = True
            pos_direction = direction
            pos_entry_idx = i + 1
            pos_entry_price = actual_entry
            pos_stop = stop
            pos_tp1 = tp1
            pos_contracts = contracts
            pos_remaining_contracts = contracts
            pos_trimmed = False
            pos_be_stop = 0.0
            pos_trail_stop = 0.0
            pos_signal_type = str(sig_type[i])
            pos_bias_dir = bias_dir_arr[i]
            pos_regime = regime_arr[i]
            pos_model_prob = prob
            pos_grade = grade

            # Direction-aware trim percentage (dual_mode or direction_mgmt)
            # MSS override applied on top if mss_management is enabled
            if mss_mgmt_enabled and is_mss_signal:
                if direction == -1:
                    pos_trim_pct = mss_mgmt.get("short_trim_pct", 1.0)
                else:
                    pos_trim_pct = mss_mgmt.get("long_trim_pct", trim_pct)
            elif dual_mode_enabled and direction == -1:
                pos_trim_pct = dual_mode.get("short_trim_pct", 1.0)
            elif dir_mgmt.get("enabled", False):
                if direction == 1:
                    pos_trim_pct = dir_mgmt.get("long_trim_pct", trim_pct)
                else:
                    pos_trim_pct = dir_mgmt.get("short_trim_pct", 1.0)
            else:
                pos_trim_pct = trim_pct

    # ---- Force-close any open position at end ----
    if in_position:
        exit_price = close_arr[-1]
        if pos_direction == 1:
            pnl_pts = exit_price - pos_entry_price
        else:
            pnl_pts = pos_entry_price - exit_price

        # AUDIT FIX: include trim PnL if position was partially trimmed
        total_pnl = pnl_pts * point_value * pos_remaining_contracts
        if pos_trimmed:
            trim_c = pos_contracts - pos_remaining_contracts
            if pos_direction == 1:
                trim_pnl = (pos_tp1 - pos_entry_price) * point_value * trim_c
            else:
                trim_pnl = (pos_entry_price - pos_tp1) * point_value * trim_c
            total_pnl += trim_pnl
            total_pnl -= commission_per_side * 2 * pos_contracts  # all contracts
        else:
            total_pnl -= commission_per_side * 2 * pos_remaining_contracts
        stop_dist = abs(pos_entry_price - pos_stop)
        total_risk = stop_dist * point_value * pos_contracts
        r_mult = total_pnl / total_risk if total_risk > 0 else 0.0

        trades.append({
            "entry_time": df_5m.index[pos_entry_idx],
            "exit_time": df_5m.index[-1],
            "direction": pos_direction,
            "entry_price": pos_entry_price,
            "exit_price": exit_price,
            "stop_price": pos_stop,
            "tp1_price": pos_tp1,
            "contracts": pos_contracts,
            "pnl_points": pnl_pts,
            "pnl_dollars": total_pnl,
            "r_multiple": r_mult,
            "exit_reason": "eod_close",
            "signal_type": pos_signal_type,
            "bias_direction": pos_bias_dir,
            "regime": pos_regime,
            "model_prob": pos_model_prob,
            "grade": pos_grade,
            "trimmed": pos_trimmed,
        })

    if len(trades) == 0:
        logger.warning("Backtest produced 0 trades (threshold=%.2f)", threshold)
        return pd.DataFrame()

    trade_df = pd.DataFrame(trades)
    trade_df["entry_time"] = pd.to_datetime(trade_df["entry_time"])
    trade_df["exit_time"] = pd.to_datetime(trade_df["exit_time"])

    logger.info(
        "Backtest complete: %d trades, threshold=%.2f, "
        "win_rate=%.1f%%, total_pnl=$%.0f, avg_R=%.2f",
        len(trade_df),
        threshold,
        100.0 * (trade_df["pnl_dollars"] > 0).mean(),
        trade_df["pnl_dollars"].sum(),
        trade_df["r_multiple"].mean(),
    )

    return trade_df


def _find_nth_swing(
    swing_mask: np.ndarray,
    price_arr: np.ndarray,
    current_idx: int,
    n: int,
    direction: int,
) -> float:
    """Find the nth most recent swing low (long) or swing high (short).

    Parameters
    ----------
    swing_mask : np.ndarray[bool]
        True at swing point bars.
    price_arr : np.ndarray[float]
        Low prices (direction=1) or high prices (direction=-1).
    current_idx : int
        Current bar index.
    n : int
        Which swing to pick (2 = 2nd most recent).
    direction : int
        1 for long (swing lows), -1 for short (swing highs).

    Returns
    -------
    float
        Price of the nth swing, or NaN if not enough swings.
    """
    count = 0
    for j in range(current_idx - 1, -1, -1):
        if swing_mask[j]:
            count += 1
            if count == n:
                return price_arr[j]
    return np.nan


