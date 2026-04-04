"""
experiments/optimize_v4.py — Re-run key optimization experiments on v4 signal cache.

v4 cache: 77,643 signals (min_fvg_atr_mult=0.3) vs v3: 15,894 signals.
Phase 2 experiments were all on v3. The 5x larger signal pool may unlock
configurations that were impossible before.

Experiments:
  1. Session filter on v4 (NY only, London MSS+SMT, London all, Asia MSS+SMT)
  2. Dual-threshold by session (different min_stop_atr_mult per session)
  3. SQ threshold re-optimization on v4
  4. Combined optimal grid search (90 combos) + walk-forward top 3

References:
  - v3 baseline: 534 trades, +156.6R, PPDD=10.92
  - v4 + mult=1.7: ~1,156 trades, +179R, PPDD=9.34

Usage: python experiments/optimize_v4.py
"""
from __future__ import annotations

import itertools
import logging
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

DATA = PROJECT / "data"
CONFIG = PROJECT / "config" / "params.yaml"


# ======================================================================
# Data loading — load everything once, pre-compute arrays
# ======================================================================

def load_all():
    """Load all data and pre-compute feature arrays (once)."""
    t0 = _time.perf_counter()
    print("[LOAD] Loading data for v4 optimization...")

    nq = pd.read_parquet(DATA / "NQ_5m_10yr.parquet")
    es = pd.read_parquet(DATA / "ES_5m_10yr.parquet")
    sig_v4 = pd.read_parquet(DATA / "cache_signals_10yr_v4.parquet")
    bias = pd.read_parquet(DATA / "cache_bias_10yr_v2.parquet")
    regime = pd.read_parquet(DATA / "cache_regime_10yr_v2.parquet")
    atr_cache = pd.read_parquet(DATA / "cache_atr_flu_10yr_v2.parquet")
    with open(CONFIG, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # SMT divergence
    print("[LOAD] Computing SMT divergence...")
    from features.smt import compute_smt
    smt = compute_smt(nq, es, {'swing': {'left_bars': 3, 'right_bars': 1},
                                'smt': {'sweep_lookback': 15, 'time_tolerance': 1}})

    # Apply SMT gate for MSS + kill MSS overnight
    def make_sig(sig_raw, smt_df):
        s = sig_raw.copy()
        mm = s['signal'].astype(bool) & (s['signal_type'] == 'mss')
        mi = s.index[mm]
        s.loc[mi, 'signal'] = False
        s.loc[mi, 'signal_dir'] = 0
        s['has_smt'] = False
        c_idx = mi.intersection(smt_df.index)
        if len(c_idx) == 0:
            return s
        md = sig_raw.loc[c_idx, 'signal_dir'].values
        ok = ((md == 1) & smt_df.loc[c_idx, 'smt_bull'].values.astype(bool)) | \
             ((md == -1) & smt_df.loc[c_idx, 'smt_bear'].values.astype(bool))
        g = c_idx[ok]
        s.loc[g, 'signal'] = sig_raw.loc[g, 'signal']
        s.loc[g, 'signal_dir'] = sig_raw.loc[g, 'signal_dir']
        s.loc[g, 'has_smt'] = True
        # Kill MSS in overnight (16:00-03:00 ET)
        rem = s['signal'].astype(bool) & (s['signal_type'] == 'mss')
        mi2 = s.index[rem]
        if len(mi2) > 0:
            et = mi2.tz_convert('US/Eastern')
            ef = et.hour + et.minute / 60.0
            kill = (ef >= 16.0) | (ef < 3.0)
            if kill.any():
                s.loc[mi2[kill], ['signal', 'signal_dir']] = [False, 0]
        return s

    ss = make_sig(sig_v4, smt)
    n_raw = int(sig_v4['signal'].sum())
    n_gated = int(ss['signal'].sum())
    print(f"[LOAD] v4 raw signals: {n_raw}, after SMT gate: {n_gated}")

    # Pre-compute all feature arrays (same approach as phase2_combined.py)
    from features.displacement import compute_atr, compute_fluency
    from features.swing import compute_swing_levels
    from features.pa_quality import compute_alternating_dir_ratio
    from features.news_filter import build_news_blackout_mask

    et_idx = nq.index.tz_convert("US/Eastern")
    o = nq["open"].values
    h = nq["high"].values
    l = nq["low"].values
    c = nq["close"].values
    n = len(nq)

    atr_arr = atr_cache["atr"].values
    fluency_arr = atr_cache["fluency"].values
    pa_alt_arr = compute_alternating_dir_ratio(nq, window=6).values

    swing_p = {"left_bars": params["swing"]["left_bars"],
               "right_bars": params["swing"]["right_bars"]}
    swings = compute_swing_levels(nq, swing_p)
    swings["swing_high_price"] = swings["swing_high_price"].shift(1).ffill()
    swings["swing_low_price"] = swings["swing_low_price"].shift(1).ffill()
    swing_high_mask = swings["swing_high"].shift(1, fill_value=False).values
    swing_low_mask = swings["swing_low"].shift(1, fill_value=False).values

    sig_mask = ss["signal"].values.astype(bool)
    sig_dir = ss["signal_dir"].values.astype(float)
    sig_type = ss["signal_type"].values
    has_smt_arr = ss["has_smt"].values.astype(bool)
    entry_price_arr = ss["entry_price"].values
    model_stop_arr = ss["model_stop"].values
    irl_target_arr = ss["irl_target"].values
    bias_dir_arr = bias["bias_direction"].values.astype(float)
    bias_conf_arr = bias["bias_confidence"].values.astype(float)
    regime_arr = regime["regime"].values.astype(float)

    # News filter
    news_path = PROJECT / "config" / "news_calendar.csv"
    news_blackout_arr = None
    if news_path.exists():
        news_bl = build_news_blackout_mask(nq.index, str(news_path),
            params["news"]["blackout_minutes_before"],
            params["news"]["cooldown_minutes_after"])
        news_blackout_arr = news_bl.values

    # Signal quality (pre-compute at every signal bar)
    sq_params = params.get("signal_quality", {})
    signal_quality = np.full(n, np.nan)
    signal_indices = np.where(sig_mask)[0]
    if len(signal_indices) > 0:
        for idx in signal_indices:
            a = atr_arr[idx] if not np.isnan(atr_arr[idx]) else 10.0
            gap = abs(entry_price_arr[idx] - model_stop_arr[idx])
            size_sc = min(1.0, gap / (a * 1.5)) if a > 0 else 0.5
            body = abs(c[idx] - o[idx])
            rng = h[idx] - l[idx]
            disp_sc = body / rng if rng > 0 else 0.0
            flu_val = fluency_arr[idx]
            flu_sc = min(1.0, max(0.0, flu_val)) if not np.isnan(flu_val) else 0.5
            window = 6
            if idx >= window:
                dirs = np.sign(c[idx-window:idx] - o[idx-window:idx])
                alt = np.sum(dirs[1:] != dirs[:-1]) / (window - 1)
                pa_sc = 1.0 - alt
            else:
                pa_sc = 0.5
            signal_quality[idx] = (
                sq_params.get("w_size", 0.3) * size_sc
                + sq_params.get("w_disp", 0.3) * disp_sc
                + sq_params.get("w_flu", 0.2) * flu_sc
                + sq_params.get("w_pa", 0.2) * pa_sc
            )

    # Stop distance / ATR ratio for each signal (for min_stop_atr_mult filtering)
    stop_atr_ratio = np.full(n, np.nan)
    if len(signal_indices) > 0:
        for idx in signal_indices:
            a = atr_arr[idx] if not np.isnan(atr_arr[idx]) else 10.0
            if a > 0:
                stop_atr_ratio[idx] = abs(entry_price_arr[idx] - model_stop_arr[idx]) / a

    # Date tracking (session dates)
    dates = np.array([
        (et_idx[j] + pd.Timedelta(days=1)).date() if et_idx[j].hour >= 18 else et_idx[j].date()
        for j in range(n)
    ])
    dow_arr = np.array([et_idx[j].dayofweek for j in range(n)])
    et_frac_arr = np.array([et_idx[j].hour + et_idx[j].minute / 60.0 for j in range(n)])

    elapsed = _time.perf_counter() - t0
    print(f"[LOAD] All data loaded in {elapsed:.1f}s (n={n} bars)")

    return {
        "nq": nq, "params": params, "n": n,
        "o": o, "h": h, "l": l, "c": c,
        "atr_arr": atr_arr, "fluency_arr": fluency_arr, "pa_alt_arr": pa_alt_arr,
        "swing_high_mask": swing_high_mask, "swing_low_mask": swing_low_mask,
        "sig_mask": sig_mask, "sig_dir": sig_dir, "sig_type": sig_type,
        "has_smt_arr": has_smt_arr,
        "entry_price_arr": entry_price_arr, "model_stop_arr": model_stop_arr,
        "irl_target_arr": irl_target_arr,
        "bias_dir_arr": bias_dir_arr, "bias_conf_arr": bias_conf_arr,
        "regime_arr": regime_arr,
        "news_blackout_arr": news_blackout_arr,
        "signal_quality": signal_quality, "stop_atr_ratio": stop_atr_ratio,
        "dates": dates, "dow_arr": dow_arr, "et_frac_arr": et_frac_arr,
        "et_idx": et_idx,
    }


# ======================================================================
# Fast inlined backtest engine (adapted from phase2_combined.py)
# ======================================================================

def _find_nth_swing(mask, prices, idx, n_val, direction):
    """Find the nth swing high/low looking backwards."""
    count = 0
    for j in range(idx - 1, -1, -1):
        if mask[j]:
            count += 1
            if count == n_val:
                return prices[j]
    return np.nan


def _compute_grade_fast(ba: float, regime: float) -> str:
    if np.isnan(ba) or np.isnan(regime) or regime == 0.0:
        return "C"
    aligned = ba > 0.5
    full = regime >= 1.0
    if aligned and full:
        return "A+"
    if aligned or full:
        return "B+"
    return "C"


def run_backtest_fast(
    d: dict,
    # --- Grid params ---
    session_mode: str = "NY_only",
    sq_long: float = 0.68,
    sq_short: float = 0.82,
    min_stop_atr: float = 1.7,
    # --- Optional: per-session min_stop_atr override ---
    # When provided, overrides min_stop_atr based on session
    session_stop_atr: dict | None = None,
    # --- Session-specific signal type filter ---
    # "london_mss_smt_only": London only allows MSS+SMT
    # "london_all": London allows everything
    # "asia_mss_smt_only": Asia only allows MSS+SMT
    london_mode: str = "closed",      # "closed", "mss_smt_only", "all"
    asia_mode: str = "closed",        # "closed", "mss_smt_only", "all"
    # --- Date range for walk-forward ---
    start_date=None, end_date=None,
) -> list[dict]:
    """Run backtest with specific filter configuration. Returns list of trade dicts."""

    params = d["params"]
    n = d["n"]
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    atr_arr = d["atr_arr"]
    sig_mask = d["sig_mask"]
    sig_dir = d["sig_dir"]
    sig_type = d["sig_type"]
    has_smt_arr = d["has_smt_arr"]
    entry_price_arr = d["entry_price_arr"]
    model_stop_arr = d["model_stop_arr"]
    irl_target_arr = d["irl_target_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    bias_conf_arr = d["bias_conf_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    signal_quality = d["signal_quality"]
    stop_atr_ratio = d["stop_atr_ratio"]
    dates = d["dates"]
    dow_arr = d["dow_arr"]
    et_frac_arr = d["et_frac_arr"]
    pa_alt_arr = d["pa_alt_arr"]
    swing_high_mask = d["swing_high_mask"]
    swing_low_mask = d["swing_low_mask"]
    nq = d["nq"]

    # Fixed params from config
    pos_params = params["position"]
    risk_params = params["risk"]
    bt_params = params["backtest"]
    grading_params = params["grading"]
    trim_params = params["trim"]
    trail_params = params["trail"]
    smt_cfg = params.get("smt", {})
    mss_mgmt = params.get("mss_management", {})
    mss_mgmt_enabled = mss_mgmt.get("enabled", False)
    dual_mode = params.get("dual_mode", {})
    session_rules = params.get("session_rules", {})
    session_regime = params.get("session_regime", {})
    dir_mgmt = params.get("direction_mgmt", {})
    bias_relax = params.get("bias_relaxation", {})
    pa_threshold = params.get("pa_quality", {}).get("alt_dir_threshold", 1.0)

    normal_r = pos_params["normal_r"]
    reduced_r = pos_params["reduced_r"]
    point_value = pos_params["point_value"]
    daily_max_loss_r = risk_params["daily_max_loss_r"]
    max_consec_losses = risk_params["max_consecutive_losses"]
    commission_per_side = bt_params["commission_per_side_micro"]
    slippage_ticks = bt_params["slippage_normal_ticks"]
    slippage_points = slippage_ticks * 0.25
    a_plus_mult = grading_params["a_plus_size_mult"]
    b_plus_mult = grading_params["b_plus_size_mult"]
    c_skip = grading_params["c_skip"]
    trim_pct = trim_params["pct"]
    be_after_trim = trim_params["be_after_trim"]
    nth_swing = trail_params["use_nth_swing"]

    # Date range filtering
    start_idx = 0
    end_idx = n
    if start_date is not None:
        start_d = pd.Timestamp(start_date).date()
        for j in range(n):
            if dates[j] >= start_d:
                start_idx = j
                break
    if end_date is not None:
        end_d = pd.Timestamp(end_date).date()
        for j in range(n - 1, -1, -1):
            if dates[j] <= end_d:
                end_idx = j + 1
                break

    # ---- Bar-by-bar backtest ----
    trades = []
    current_date = None
    daily_pnl_r = 0.0
    consecutive_losses = 0
    day_stopped = False
    in_position = False
    pos_direction = pos_entry_idx = 0
    pos_entry_price = pos_stop = pos_tp1 = 0.0
    pos_contracts = pos_remaining_contracts = 0
    pos_trimmed = False
    pos_be_stop = pos_trail_stop = 0.0
    pos_signal_type = pos_grade = ""
    pos_bias_dir = pos_regime = 0.0
    pos_trim_pct = trim_pct

    for i in range(start_idx, end_idx):
        bar_date = dates[i]
        if bar_date != current_date:
            current_date = bar_date
            daily_pnl_r = 0.0
            consecutive_losses = 0
            day_stopped = False

        # ---- EXIT MANAGEMENT ----
        if in_position:
            exited = False
            exit_reason = ""
            exit_price = 0.0
            exit_contracts = pos_remaining_contracts

            bars_in_trade = i - pos_entry_idx
            if not pos_trimmed and 2 <= bars_in_trade <= 4:
                pa_start = max(pos_entry_idx, 0)
                pa_end = i + 1
                pa_range = h[pa_start:pa_end] - l[pa_start:pa_end]
                pa_body = np.abs(c[pa_start:pa_end] - o[pa_start:pa_end])
                safe_pa = np.where(pa_range > 0, pa_range, 1.0)
                pa_wick = 1.0 - (pa_body / safe_pa)
                avg_wick = float(np.mean(pa_wick))
                pa_dirs = np.sign(c[pa_start:pa_end] - o[pa_start:pa_end])
                favorable = (pa_dirs == pos_direction).mean()
                disp = (c[i] - pos_entry_price) if pos_direction == 1 else (pos_entry_price - c[i])
                cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                no_progress = disp < cur_atr * 0.3
                bad_pa = avg_wick > 0.65 and favorable < 0.5
                if bad_pa and no_progress and bars_in_trade >= 3:
                    exit_price = o[i+1] if i+1 < end_idx else c[i]
                    exit_reason = "early_cut_pa"
                    exited = True

            if not exited and pos_direction == 1:
                eff_stop = pos_trail_stop if pos_trimmed and pos_trail_stop > 0 else pos_stop
                if pos_trimmed and be_after_trim and pos_be_stop > 0:
                    eff_stop = max(eff_stop, pos_be_stop)
                if l[i] <= eff_stop:
                    exit_price = eff_stop - slippage_points
                    exit_reason = "be_sweep" if (pos_trimmed and eff_stop >= pos_entry_price) else "stop"
                    exited = True
                elif not pos_trimmed and h[i] >= pos_tp1:
                    tc = max(1, int(pos_contracts * pos_trim_pct))
                    pos_remaining_contracts = pos_contracts - tc
                    pos_trimmed = True
                    pos_be_stop = pos_entry_price
                    if pos_remaining_contracts > 0:
                        pos_trail_stop = _find_nth_swing(swing_low_mask, nq["low"].values, i, nth_swing, 1)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True
                if pos_trimmed and not exited:
                    nt = _find_nth_swing(swing_low_mask, nq["low"].values, i, nth_swing, 1)
                    if not np.isnan(nt) and nt > pos_trail_stop:
                        pos_trail_stop = nt

            elif not exited:  # SHORT
                eff_stop = pos_trail_stop if pos_trimmed and pos_trail_stop > 0 else pos_stop
                if pos_trimmed and be_after_trim and pos_be_stop > 0:
                    eff_stop = min(eff_stop, pos_be_stop)
                if h[i] >= eff_stop:
                    exit_price = eff_stop + slippage_points
                    exit_reason = "be_sweep" if (pos_trimmed and eff_stop <= pos_entry_price) else "stop"
                    exited = True
                elif not pos_trimmed and l[i] <= pos_tp1:
                    tc = max(1, int(pos_contracts * pos_trim_pct))
                    pos_remaining_contracts = pos_contracts - tc
                    pos_trimmed = True
                    pos_be_stop = pos_entry_price
                    if pos_remaining_contracts > 0:
                        pos_trail_stop = _find_nth_swing(swing_high_mask, nq["high"].values, i, nth_swing, -1)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True
                if pos_trimmed and not exited:
                    nt = _find_nth_swing(swing_high_mask, nq["high"].values, i, nth_swing, -1)
                    if not np.isnan(nt) and nt < pos_trail_stop:
                        pos_trail_stop = nt

            if exited:
                pnl_pts = (exit_price - pos_entry_price) if pos_direction == 1 else (pos_entry_price - exit_price)
                if pos_trimmed and exit_reason != "tp1":
                    trim_pnl_total = (pos_tp1 - pos_entry_price) if pos_direction == 1 else (pos_entry_price - pos_tp1)
                    trim_c = pos_contracts - exit_contracts
                    total_pnl = trim_pnl_total * point_value * trim_c + pnl_pts * point_value * exit_contracts
                    total_comm = commission_per_side * 2 * pos_contracts
                    total_pnl -= total_comm
                else:
                    total_pnl = pnl_pts * point_value * exit_contracts
                    total_comm = commission_per_side * 2 * exit_contracts
                    total_pnl -= total_comm
                stop_dist_exit = abs(pos_entry_price - pos_stop)
                total_risk = stop_dist_exit * point_value * pos_contracts
                r_mult = total_pnl / total_risk if total_risk > 0 else 0.0

                trades.append({
                    "entry_time": nq.index[pos_entry_idx],
                    "exit_time": nq.index[i + 1] if exit_reason == "early_cut_pa" and i + 1 < end_idx else nq.index[i],
                    "r": r_mult, "reason": exit_reason, "dir": pos_direction,
                    "type": pos_signal_type, "trimmed": pos_trimmed, "grade": pos_grade,
                    "entry_price": pos_entry_price, "exit_price": exit_price,
                    "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
                })
                daily_pnl_r += r_mult
                if exit_reason == "be_sweep" and pos_trimmed:
                    pass
                elif r_mult < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
                if consecutive_losses >= max_consec_losses:
                    day_stopped = True
                if daily_pnl_r <= -daily_max_loss_r:
                    day_stopped = True
                in_position = False

        if in_position:
            continue

        if news_blackout_arr is not None and news_blackout_arr[i]:
            continue

        # ORM no-trade window (9:30-10:00 ET)
        et_frac = et_frac_arr[i]
        if 9.5 <= et_frac <= 10.0:
            continue

        if day_stopped:
            continue

        if not sig_mask[i]:
            continue

        direction = int(sig_dir[i])
        if direction == 0:
            continue

        # Signal type filter
        sig_f = params.get("signal_filter", {})
        if not sig_f.get("allow_mss", True) and str(sig_type[i]) == "mss":
            continue
        if not sig_f.get("allow_trend", True) and str(sig_type[i]) == "trend":
            continue

        entry_p = entry_price_arr[i]
        stop = model_stop_arr[i]
        tp1 = irl_target_arr[i]
        if np.isnan(entry_p) or np.isnan(stop) or np.isnan(tp1):
            continue
        if direction == 1 and (stop >= entry_p or tp1 <= entry_p):
            continue
        if direction == -1 and (stop <= entry_p or tp1 >= entry_p):
            continue

        # Bias filter (exempt MSS+SMT)
        bias_opposing = (direction == -np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0)
        if bias_opposing:
            is_mss_smt_sig = (smt_cfg.get("enabled", False) and has_smt_arr[i]
                              and str(sig_type[i]) == "mss")
            if is_mss_smt_sig:
                pass
            elif bias_relax.get("enabled", False) and direction == -1:
                pass
            else:
                continue

        # PA quality
        if pa_alt_arr[i] >= pa_threshold:
            continue

        # ---- SESSION FILTER (grid params: session_mode + london_mode + asia_mode) ----
        is_mss_smt = (str(sig_type[i]) == "mss" and has_smt_arr[i]
                      and smt_cfg.get("enabled", False))
        mss_bypass = is_mss_smt and smt_cfg.get("bypass_session_filter", False)

        # Determine which session this bar is in
        is_ny = (10.0 <= et_frac < 16.0)  # after ORM
        is_london = (3.0 <= et_frac < 9.5)
        is_asia = (et_frac >= 18.0) or (et_frac < 3.0)

        if not mss_bypass:
            if is_ny:
                pass  # NY always allowed
            elif is_london:
                if london_mode == "closed":
                    continue
                elif london_mode == "mss_smt_only":
                    if not is_mss_smt:
                        continue
                # else london_mode == "all": allow everything
            elif is_asia:
                if asia_mode == "closed":
                    continue
                elif asia_mode == "mss_smt_only":
                    if not is_mss_smt:
                        continue
                # else asia_mode == "all": allow everything
            else:
                # 9:30-10:00 already blocked by ORM check above
                # 16:00-18:00 post-market, block
                continue

        # ---- MIN STOP ATR FILTER ----
        # Use per-session threshold if provided, otherwise global
        if session_stop_atr is not None:
            if is_ny:
                eff_min_stop_atr = session_stop_atr.get("ny", min_stop_atr)
            elif is_london:
                eff_min_stop_atr = session_stop_atr.get("london", min_stop_atr)
            elif is_asia:
                eff_min_stop_atr = session_stop_atr.get("asia", min_stop_atr)
            else:
                eff_min_stop_atr = min_stop_atr
        else:
            eff_min_stop_atr = min_stop_atr

        if not np.isnan(stop_atr_ratio[i]):
            if stop_atr_ratio[i] < eff_min_stop_atr:
                continue
        else:
            stop_dist_check = abs(entry_p - stop)
            a_check = atr_arr[i] if not np.isnan(atr_arr[i]) else 10.0
            if a_check > 0 and (stop_dist_check / a_check) < eff_min_stop_atr:
                continue

        # ---- SQ FILTER ----
        if not np.isnan(signal_quality[i]):
            eff_sq = sq_long if direction == 1 else sq_short
            if signal_quality[i] < eff_sq:
                continue

        if i + 1 >= end_idx:
            continue

        # Grade
        ba = 1.0 if (direction == np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0) else 0.0
        grade = _compute_grade_fast(ba, regime_arr[i])
        if grade == "C" and c_skip:
            continue

        # Sizing
        is_reduced = (dow_arr[i] in (0, 4)) or (regime_arr[i] < 1.0)
        base_r = reduced_r if is_reduced else normal_r
        if grade == "A+":
            r_amount = base_r * a_plus_mult
        elif grade == "B+":
            r_amount = base_r * b_plus_mult
        else:
            r_amount = base_r * 0.5

        # Session regime
        if session_regime.get("enabled", False):
            sr_lunch_s = session_regime.get("lunch_start", 12.0)
            sr_lunch_e = session_regime.get("lunch_end", 13.0)
            sr_am_end = session_regime.get("am_end", 12.0)
            sr_pm_start = session_regime.get("pm_start", 13.0)
            if et_frac < sr_am_end:
                sr_mult = session_regime.get("am_mult", 1.0)
            elif sr_lunch_s <= et_frac < sr_lunch_e:
                sr_mult = session_regime.get("lunch_mult", 0.5)
            elif et_frac >= sr_pm_start:
                sr_mult = session_regime.get("pm_mult", 0.75)
            else:
                sr_mult = 1.0
            r_amount *= sr_mult
            if r_amount <= 0:
                continue

        # Slippage
        actual_entry = entry_p + slippage_points if direction == 1 else entry_p - slippage_points
        stop_dist = abs(actual_entry - stop)
        if stop_dist < 1.0:
            continue

        contracts = max(1, int(r_amount / (stop_dist * point_value))) if stop_dist > 0 and point_value > 0 else 0
        if contracts <= 0:
            continue

        # TP
        is_mss_signal = str(sig_type[i]) == "mss"
        if session_rules.get("enabled", False):
            ny_tp_mult = session_rules.get("ny_tp_multiplier", 1.0)
            if dir_mgmt.get("enabled", False):
                actual_tp_mult = dir_mgmt.get("long_tp_mult", ny_tp_mult) if direction == 1 else dir_mgmt.get("short_tp_mult", 1.25)
            else:
                actual_tp_mult = ny_tp_mult
            if mss_mgmt_enabled and is_mss_signal and direction == 1:
                actual_tp_mult = mss_mgmt.get("long_tp_mult", actual_tp_mult)
            if 9.5 <= et_frac < 16.0:
                tp_distance = (tp1 - actual_entry) if direction == 1 else (actual_entry - tp1)
                tp1 = (actual_entry + tp_distance * actual_tp_mult) if direction == 1 else (actual_entry - tp_distance * actual_tp_mult)

        dual_mode_enabled = dual_mode.get("enabled", False)
        if dual_mode_enabled and direction == -1:
            short_rr = dual_mode.get("short_rr", 0.625)
            if mss_mgmt_enabled and is_mss_signal:
                short_rr = mss_mgmt.get("short_rr", short_rr)
            tp1 = actual_entry - stop_dist * short_rr

        # Enter
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
        pos_grade = grade

        if mss_mgmt_enabled and is_mss_signal:
            pos_trim_pct = mss_mgmt.get("short_trim_pct", 1.0) if direction == -1 else mss_mgmt.get("long_trim_pct", trim_pct)
        elif dual_mode_enabled and direction == -1:
            pos_trim_pct = dual_mode.get("short_trim_pct", 1.0)
        elif dir_mgmt.get("enabled", False):
            pos_trim_pct = dir_mgmt.get("long_trim_pct", trim_pct) if direction == 1 else dir_mgmt.get("short_trim_pct", 1.0)
        else:
            pos_trim_pct = trim_pct

    # Force close open position at end
    if in_position and pos_entry_idx < end_idx:
        last_i = end_idx - 1
        exit_price = c[last_i]
        pnl_pts = (exit_price - pos_entry_price) if pos_direction == 1 else (pos_entry_price - exit_price)
        total_pnl = pnl_pts * point_value * pos_remaining_contracts
        if pos_trimmed:
            trim_c = pos_contracts - pos_remaining_contracts
            trim_pnl = ((pos_tp1 - pos_entry_price) if pos_direction == 1 else (pos_entry_price - pos_tp1)) * point_value * trim_c
            total_pnl += trim_pnl
            total_pnl -= commission_per_side * 2 * pos_contracts
        else:
            total_pnl -= commission_per_side * 2 * pos_remaining_contracts
        stop_dist_exit = abs(pos_entry_price - pos_stop)
        total_risk = stop_dist_exit * point_value * pos_contracts
        r_mult = total_pnl / total_risk if total_risk > 0 else 0.0
        trades.append({
            "entry_time": nq.index[pos_entry_idx], "exit_time": nq.index[last_i],
            "r": r_mult, "reason": "eod_close", "dir": pos_direction,
            "type": pos_signal_type, "trimmed": pos_trimmed, "grade": pos_grade,
            "entry_price": pos_entry_price, "exit_price": exit_price,
            "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
        })

    return trades


# ======================================================================
# Metrics
# ======================================================================

def compute_metrics(trades_list: list[dict]) -> dict:
    """Compute performance metrics from trade list."""
    if not trades_list:
        return {"trades": 0, "R": 0.0, "PPDD": 0.0, "MaxDD_R": 0.0,
                "WR": 0.0, "PF": 0.0, "avgR": 0.0}

    r_arr = np.array([t["r"] for t in trades_list])
    n = len(r_arr)
    total_r = float(r_arr.sum())
    wr = float((r_arr > 0).mean()) * 100

    cum_r = np.cumsum(r_arr)
    peak = np.maximum.accumulate(cum_r)
    dd_r = cum_r - peak
    max_dd_r = float(abs(dd_r.min())) if len(dd_r) > 0 else 0.0
    ppdd = total_r / max_dd_r if max_dd_r > 0.01 else total_r

    gross_win = float(r_arr[r_arr > 0].sum()) if (r_arr > 0).any() else 0.0
    gross_loss = float(abs(r_arr[r_arr <= 0].sum())) if (r_arr <= 0).any() else 0.001
    pf = gross_win / gross_loss if gross_loss > 0.001 else 999.0

    return {
        "trades": n,
        "R": round(total_r, 2),
        "PPDD": round(ppdd, 2),
        "MaxDD_R": round(max_dd_r, 2),
        "WR": round(wr, 1),
        "PF": round(pf, 2),
        "avgR": round(float(r_arr.mean()), 4),
    }


def compute_yearly(trades_list: list[dict]) -> dict:
    """Per-year R values."""
    if not trades_list:
        return {}
    df = pd.DataFrame(trades_list)
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year
    return df.groupby("year")["r"].sum().to_dict()


# ======================================================================
# Walk-forward
# ======================================================================

WF_WINDOWS = [
    ("2016-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2016-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2016-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2016-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2016-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2016-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]


def walk_forward_validate(d: dict, **bt_kwargs) -> dict:
    """Run expanding-window walk-forward. Returns OOS aggregates."""
    oos_trades_all = []
    window_results = []

    for train_s, train_e, test_s, test_e in WF_WINDOWS:
        oos_trades = run_backtest_fast(d, start_date=test_s, end_date=test_e, **bt_kwargs)
        m = compute_metrics(oos_trades)
        m["window"] = test_s[:4]
        window_results.append(m)
        oos_trades_all.extend(oos_trades)

    oos_agg = compute_metrics(oos_trades_all)
    oos_agg["n_windows"] = len(WF_WINDOWS)
    oos_agg["win_windows"] = sum(1 for w in window_results if w["R"] > 0)
    oos_agg["per_window"] = window_results
    return oos_agg


# ======================================================================
# Pretty printing
# ======================================================================

def print_table(title: str, rows: list[dict], columns: list[str], label_key: str = "label"):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")

    # Header
    header_parts = [f"{'Config':<40}"]
    for col in columns:
        header_parts.append(f"{col:>8}")
    print("  " + " ".join(header_parts))
    print("  " + "-" * (40 + 9 * len(columns)))

    for row in rows:
        parts = [f"{row.get(label_key, '?'):<40}"]
        for col in columns:
            val = row.get(col, 0)
            if isinstance(val, float):
                parts.append(f"{val:>8.2f}")
            else:
                parts.append(f"{val:>8}")
        print("  " + " ".join(parts))
    print(f"{'=' * 100}")


# ======================================================================
# EXPERIMENT 1: Session filter on v4
# ======================================================================

def experiment_1_sessions(d):
    """Test session filters on v4 cache."""
    print("\n" + "#" * 100)
    print("  EXPERIMENT 1: SESSION FILTER ON v4")
    print("  With v3, opening London/Asia was destructive. v4 has 5x more signals.")
    print("#" * 100)

    configs = [
        {"label": "a) v4 NY only (baseline, mult=1.7)",
         "session_mode": "NY_only", "london_mode": "closed", "asia_mode": "closed",
         "min_stop_atr": 1.7, "sq_long": 0.68, "sq_short": 0.82},
        {"label": "b) v4 + London MSS+SMT only, mult=1.7",
         "session_mode": "NY_only", "london_mode": "mss_smt_only", "asia_mode": "closed",
         "min_stop_atr": 1.7, "sq_long": 0.68, "sq_short": 0.82},
        {"label": "c) v4 + London all, mult=2.0",
         "session_mode": "NY_only", "london_mode": "all", "asia_mode": "closed",
         "min_stop_atr": 2.0, "sq_long": 0.68, "sq_short": 0.82},
        {"label": "d) v4 + Asia MSS+SMT only, mult=2.0",
         "session_mode": "NY_only", "london_mode": "closed", "asia_mode": "mss_smt_only",
         "min_stop_atr": 2.0, "sq_long": 0.68, "sq_short": 0.82},
        {"label": "e) v4 + London+Asia MSS+SMT, mult=2.0",
         "session_mode": "NY_only", "london_mode": "mss_smt_only", "asia_mode": "mss_smt_only",
         "min_stop_atr": 2.0, "sq_long": 0.68, "sq_short": 0.82},
    ]

    results = []
    for cfg in configs:
        label = cfg["label"]
        t0 = _time.perf_counter()
        trades = run_backtest_fast(
            d,
            session_mode=cfg["session_mode"],
            london_mode=cfg["london_mode"],
            asia_mode=cfg["asia_mode"],
            min_stop_atr=cfg["min_stop_atr"],
            sq_long=cfg["sq_long"],
            sq_short=cfg["sq_short"],
        )
        m = compute_metrics(trades)
        m["label"] = label
        results.append(m)
        elapsed = _time.perf_counter() - t0
        print(f"  {label}: {m['trades']}t, R={m['R']:.1f}, PPDD={m['PPDD']:.2f}, "
              f"PF={m['PF']:.2f}, MaxDD={m['MaxDD_R']:.1f}R ({elapsed:.1f}s)")

    print_table("EXPERIMENT 1: SESSION FILTER RESULTS",
                results, ["trades", "R", "PPDD", "PF", "WR", "MaxDD_R"])
    return results


# ======================================================================
# EXPERIMENT 2: Dual-threshold by session
# ======================================================================

def experiment_2_dual_threshold(d):
    """Test different min_stop_atr_mult by session."""
    print("\n" + "#" * 100)
    print("  EXPERIMENT 2: DUAL-THRESHOLD BY SESSION")
    print("  Different min_stop_atr_mult for NY, London, Asia")
    print("#" * 100)

    configs = [
        {"label": "a) Uniform mult=1.7 (baseline)",
         "london_mode": "all", "asia_mode": "all",
         "session_stop_atr": None, "min_stop_atr": 1.7},
        {"label": "b) NY=1.5, London=2.5, Asia=3.0",
         "london_mode": "all", "asia_mode": "all",
         "session_stop_atr": {"ny": 1.5, "london": 2.5, "asia": 3.0}, "min_stop_atr": 1.7},
        {"label": "c) NY=1.5, London=2.0, Asia=2.5",
         "london_mode": "all", "asia_mode": "all",
         "session_stop_atr": {"ny": 1.5, "london": 2.0, "asia": 2.5}, "min_stop_atr": 1.7},
        {"label": "d) NY=1.7, London=2.0, Asia=2.5",
         "london_mode": "all", "asia_mode": "all",
         "session_stop_atr": {"ny": 1.7, "london": 2.0, "asia": 2.5}, "min_stop_atr": 1.7},
        {"label": "e) NY=1.3, London=2.5, Asia=3.0",
         "london_mode": "all", "asia_mode": "all",
         "session_stop_atr": {"ny": 1.3, "london": 2.5, "asia": 3.0}, "min_stop_atr": 1.7},
        {"label": "f) NY=1.5, London MSS+SMT=2.0, Asia=closed",
         "london_mode": "mss_smt_only", "asia_mode": "closed",
         "session_stop_atr": {"ny": 1.5, "london": 2.0, "asia": 99.0}, "min_stop_atr": 1.7},
        {"label": "g) NY=1.5, London MSS+SMT=2.5, Asia MSS+SMT=3.0",
         "london_mode": "mss_smt_only", "asia_mode": "mss_smt_only",
         "session_stop_atr": {"ny": 1.5, "london": 2.5, "asia": 3.0}, "min_stop_atr": 1.7},
    ]

    results = []
    for cfg in configs:
        label = cfg["label"]
        t0 = _time.perf_counter()
        trades = run_backtest_fast(
            d,
            london_mode=cfg["london_mode"],
            asia_mode=cfg["asia_mode"],
            session_stop_atr=cfg["session_stop_atr"],
            min_stop_atr=cfg["min_stop_atr"],
            sq_long=0.68, sq_short=0.82,
        )
        m = compute_metrics(trades)
        m["label"] = label
        results.append(m)
        elapsed = _time.perf_counter() - t0
        print(f"  {label}: {m['trades']}t, R={m['R']:.1f}, PPDD={m['PPDD']:.2f}, "
              f"PF={m['PF']:.2f}, MaxDD={m['MaxDD_R']:.1f}R ({elapsed:.1f}s)")

    print_table("EXPERIMENT 2: DUAL-THRESHOLD RESULTS",
                results, ["trades", "R", "PPDD", "PF", "WR", "MaxDD_R"])
    return results


# ======================================================================
# EXPERIMENT 3: SQ threshold re-optimization on v4
# ======================================================================

def experiment_3_sq_sweep(d):
    """Sweep SQ thresholds on v4 with mult=1.7."""
    print("\n" + "#" * 100)
    print("  EXPERIMENT 3: SQ THRESHOLD RE-OPTIMIZATION ON v4")
    print("  v3 optimal: SQ long=0.68, short=0.80. Does v4 shift the optimal?")
    print("#" * 100)

    long_grid = [0.60, 0.65, 0.68, 0.70, 0.75]
    short_grid = [0.75, 0.78, 0.80, 0.82, 0.85]

    results = []
    total = len(long_grid) * len(short_grid)
    idx = 0
    for sq_l in long_grid:
        for sq_s in short_grid:
            idx += 1
            t0 = _time.perf_counter()
            trades = run_backtest_fast(
                d, min_stop_atr=1.7, sq_long=sq_l, sq_short=sq_s,
                london_mode="closed", asia_mode="closed",
            )
            m = compute_metrics(trades)
            m["label"] = f"L={sq_l:.2f} S={sq_s:.2f}"
            m["sq_long"] = sq_l
            m["sq_short"] = sq_s
            m["is_current"] = (sq_l == 0.68 and sq_s == 0.82)
            results.append(m)
            elapsed = _time.perf_counter() - t0
            marker = " <<<" if m["is_current"] else ""
            print(f"  [{idx:>2}/{total}] L={sq_l:.2f} S={sq_s:.2f} | "
                  f"n={m['trades']:>5} R={m['R']:>7.1f} PPDD={m['PPDD']:>6.2f} "
                  f"PF={m['PF']:>5.2f} WR={m['WR']:>5.1f}% MaxDD={m['MaxDD_R']:>5.1f}R "
                  f"({elapsed:.1f}s){marker}")

    # Sort by PPDD
    results.sort(key=lambda x: x["PPDD"], reverse=True)
    print_table("EXPERIMENT 3: SQ SWEEP RESULTS (sorted by PPDD)",
                results, ["trades", "R", "PPDD", "PF", "WR", "MaxDD_R"])

    # Heatmap: PPDD by long_sq x short_sq
    print(f"\n  {'SQ HEATMAP: PPDD (Long SQ rows x Short SQ cols)'}")
    print(f"  {'':>8}", end="")
    for sq_s in short_grid:
        print(f"  S={sq_s:.2f}", end="")
    print()
    for sq_l in long_grid:
        print(f"  L={sq_l:.2f}", end="")
        for sq_s in short_grid:
            row = next((r for r in results if r.get("sq_long") == sq_l and r.get("sq_short") == sq_s), None)
            if row:
                print(f"  {row['PPDD']:>6.2f}", end="")
            else:
                print(f"  {'?':>6}", end="")
        print()

    # Heatmap: R
    print(f"\n  {'SQ HEATMAP: R (Long SQ rows x Short SQ cols)'}")
    print(f"  {'':>8}", end="")
    for sq_s in short_grid:
        print(f"  S={sq_s:.2f}", end="")
    print()
    for sq_l in long_grid:
        print(f"  L={sq_l:.2f}", end="")
        for sq_s in short_grid:
            row = next((r for r in results if r.get("sq_long") == sq_l and r.get("sq_short") == sq_s), None)
            if row:
                print(f"  {row['R']:>6.1f}", end="")
            else:
                print(f"  {'?':>6}", end="")
        print()

    # Heatmap: Trades
    print(f"\n  {'SQ HEATMAP: Trades (Long SQ rows x Short SQ cols)'}")
    print(f"  {'':>8}", end="")
    for sq_s in short_grid:
        print(f"  S={sq_s:.2f}", end="")
    print()
    for sq_l in long_grid:
        print(f"  L={sq_l:.2f}", end="")
        for sq_s in short_grid:
            row = next((r for r in results if r.get("sq_long") == sq_l and r.get("sq_short") == sq_s), None)
            if row:
                print(f"  {row['trades']:>6}", end="")
            else:
                print(f"  {'?':>6}", end="")
        print()

    return results


# ======================================================================
# EXPERIMENT 4: Combined optimal grid search
# ======================================================================

def experiment_4_combined(d):
    """Grid search: mult x SQ_long x SQ_short x Session. 90 combos.
    Find top 5 by PPDD with trades > 800. Walk-forward top 3."""
    print("\n" + "#" * 100)
    print("  EXPERIMENT 4: COMBINED OPTIMAL GRID SEARCH")
    print("  5 mult x 3 SQ_long x 3 SQ_short x 2 Session = 90 combos")
    print("#" * 100)

    mult_grid = [1.5, 1.6, 1.7, 1.8, 2.0]
    sq_long_grid = [0.60, 0.68, 0.75]
    sq_short_grid = [0.75, 0.80, 0.85]
    session_grid = [
        ("NY_only", "closed", "closed"),
        ("NY+London_MSS_SMT", "mss_smt_only", "closed"),
    ]

    total = len(mult_grid) * len(sq_long_grid) * len(sq_short_grid) * len(session_grid)
    print(f"  Total combos: {total}")

    results = []
    idx = 0
    t_grid_start = _time.perf_counter()

    for mult in mult_grid:
        for sq_l in sq_long_grid:
            for sq_s in sq_short_grid:
                for sess_label, london_m, asia_m in session_grid:
                    idx += 1
                    t0 = _time.perf_counter()
                    trades = run_backtest_fast(
                        d,
                        min_stop_atr=mult,
                        sq_long=sq_l, sq_short=sq_s,
                        london_mode=london_m, asia_mode=asia_m,
                    )
                    m = compute_metrics(trades)
                    m["label"] = f"m={mult:.1f} L={sq_l:.2f} S={sq_s:.2f} {sess_label}"
                    m["mult"] = mult
                    m["sq_long"] = sq_l
                    m["sq_short"] = sq_s
                    m["session"] = sess_label
                    m["london_mode"] = london_m
                    m["asia_mode"] = asia_m
                    results.append(m)

                    elapsed = _time.perf_counter() - t0
                    if idx % 10 == 0 or idx == total:
                        elapsed_total = _time.perf_counter() - t_grid_start
                        eta = elapsed_total / idx * (total - idx)
                        print(f"  [{idx:>3}/{total}] elapsed={elapsed_total:.0f}s ETA={eta:.0f}s | "
                              f"last: {m['trades']}t R={m['R']:.1f} PPDD={m['PPDD']:.2f}")

    # Sort by PPDD
    results.sort(key=lambda x: x["PPDD"], reverse=True)

    # Print full results
    print(f"\n  FULL RESULTS (sorted by PPDD):")
    print(f"  {'Config':<50} {'Trades':>7} {'R':>8} {'PPDD':>7} {'PF':>6} {'WR%':>6} {'MaxDD':>7}")
    print(f"  {'-'*50} {'-'*7} {'-'*8} {'-'*7} {'-'*6} {'-'*6} {'-'*7}")
    for r in results:
        print(f"  {r['label']:<50} {r['trades']:>7} {r['R']:>8.1f} {r['PPDD']:>7.2f} "
              f"{r['PF']:>6.2f} {r['WR']:>5.1f}% {r['MaxDD_R']:>7.1f}")

    # ---- Top 5 by PPDD with trades > 800 ----
    viable = [r for r in results if r["trades"] > 800]
    if len(viable) < 5:
        print(f"\n  Only {len(viable)} configs with trades > 800. "
              f"Relaxing to trades > 600...")
        viable = [r for r in results if r["trades"] > 600]
    if len(viable) < 5:
        print(f"  Only {len(viable)} configs with trades > 600. Using top 5 overall.")
        viable = results[:5]
    else:
        viable = viable[:5]

    print(f"\n  TOP 5 BY PPDD (trades > threshold):")
    print_table("TOP 5 VIABLE CONFIGS", viable,
                ["trades", "R", "PPDD", "PF", "WR", "MaxDD_R"])

    # ---- Walk-forward top 3 ----
    print(f"\n  WALK-FORWARD VALIDATION (top 3 by PPDD)")
    print(f"  {'=' * 80}")

    wf_results = []
    for r in viable[:3]:
        t0 = _time.perf_counter()
        wf = walk_forward_validate(
            d,
            min_stop_atr=r["mult"],
            sq_long=r["sq_long"],
            sq_short=r["sq_short"],
            london_mode=r["london_mode"],
            asia_mode=r["asia_mode"],
        )
        elapsed = _time.perf_counter() - t0

        print(f"\n  Config: {r['label']}")
        print(f"  Full-sample: {r['trades']}t, R={r['R']:.1f}, PPDD={r['PPDD']:.2f}, PF={r['PF']:.2f}")
        print(f"  WF-OOS:      {wf['trades']}t, R={wf['R']:.1f}, PPDD={wf['PPDD']:.2f}, "
              f"Wins={wf['win_windows']}/{wf['n_windows']} ({elapsed:.1f}s)")

        for pw in wf["per_window"]:
            flag = "+" if pw["R"] > 0 else "-"
            print(f"    {pw['window']}: {flag} t={pw['trades']:>3} R={pw['R']:>6.1f} "
                  f"PPDD={pw['PPDD']:>5.2f} WR={pw['WR']:>5.1f}%")

        wf_results.append({
            "config": r["label"],
            "full_trades": r["trades"], "full_R": r["R"], "full_PPDD": r["PPDD"], "full_PF": r["PF"],
            "oos_trades": wf["trades"], "oos_R": wf["R"], "oos_PPDD": wf["PPDD"],
            "oos_wins": wf["win_windows"], "oos_total": wf["n_windows"],
        })

    # WF Summary table
    print(f"\n  WALK-FORWARD SUMMARY:")
    print(f"  {'Config':<50} {'FullT':>6} {'FullR':>7} {'FullPPDD':>9} {'OOST':>5} "
          f"{'OOSR':>7} {'OOSPPDD':>8} {'Wins':>6}")
    print(f"  {'-'*50} {'-'*6} {'-'*7} {'-'*9} {'-'*5} {'-'*7} {'-'*8} {'-'*6}")
    for w in wf_results:
        print(f"  {w['config']:<50} {w['full_trades']:>6} {w['full_R']:>7.1f} "
              f"{w['full_PPDD']:>9.2f} {w['oos_trades']:>5} {w['oos_R']:>7.1f} "
              f"{w['oos_PPDD']:>8.2f} {w['oos_wins']:>2}/{w['oos_total']}")

    return results, wf_results


# ======================================================================
# FINAL COMPARISON
# ======================================================================

def final_comparison(d, exp1_results, exp2_results, exp3_results, exp4_results, exp4_wf):
    """Compare best from each experiment against v3 baseline."""
    print("\n" + "#" * 100)
    print("  FINAL COMPARISON: BEST v4 CONFIGS vs v3 BASELINE")
    print("#" * 100)

    # v3 baseline reference
    v3_ref = {"label": "v3 BASELINE (reference)", "trades": 534, "R": 156.6,
              "PPDD": 10.92, "PF": 1.59, "WR": 45.9, "MaxDD_R": 14.3}

    # v4 current (NY only, mult=1.7, SQ 0.68/0.82) — from Exp 1 result a
    v4_current = exp1_results[0].copy()
    v4_current["label"] = "v4 CURRENT (NY only, mult=1.7)"

    # Best from Exp 1 (by PPDD)
    best_exp1 = max(exp1_results, key=lambda x: x["PPDD"])
    best_exp1_label = best_exp1["label"]
    best_exp1["label"] = f"Best Exp1: {best_exp1_label[:30]}"

    # Best from Exp 2 (by PPDD)
    best_exp2 = max(exp2_results, key=lambda x: x["PPDD"])
    best_exp2_label = best_exp2["label"]
    best_exp2["label"] = f"Best Exp2: {best_exp2_label[:30]}"

    # Best from Exp 3 (by PPDD)
    best_exp3 = max(exp3_results, key=lambda x: x["PPDD"])
    best_exp3_label = best_exp3["label"]
    best_exp3["label"] = f"Best Exp3: {best_exp3_label[:30]}"

    # Best from Exp 4 full-sample (by PPDD, trades > 800)
    exp4_viable = [r for r in exp4_results if r["trades"] > 800]
    if not exp4_viable:
        exp4_viable = [r for r in exp4_results if r["trades"] > 600]
    if not exp4_viable:
        exp4_viable = exp4_results[:3]
    best_exp4 = max(exp4_viable, key=lambda x: x["PPDD"])
    best_exp4_copy = best_exp4.copy()
    best_exp4_copy["label"] = f"Best Exp4: {best_exp4['label'][:30]}"

    # Best from Exp 4 WF (by OOS R)
    if exp4_wf:
        best_wf = max(exp4_wf, key=lambda x: x["oos_R"])
        best_wf_row = {"label": f"Best WF: {best_wf['config'][:30]}",
                       "trades": best_wf["full_trades"], "R": best_wf["full_R"],
                       "PPDD": best_wf["full_PPDD"], "PF": best_wf.get("full_PF", 0),
                       "WR": 0, "MaxDD_R": 0,
                       "oos_R": best_wf["oos_R"], "oos_PPDD": best_wf["oos_PPDD"],
                       "oos_wins": best_wf["oos_wins"]}
    else:
        best_wf_row = None

    summary = [v3_ref, v4_current, best_exp1, best_exp2, best_exp3, best_exp4_copy]

    print(f"\n  {'Config':<45} {'Trades':>7} {'R':>8} {'PPDD':>7} {'PF':>6} {'WR%':>6} {'MaxDD':>7}")
    print(f"  {'-'*45} {'-'*7} {'-'*8} {'-'*7} {'-'*6} {'-'*6} {'-'*7}")
    for s in summary:
        print(f"  {s['label']:<45} {s['trades']:>7} {s.get('R', 0):>8.1f} "
              f"{s.get('PPDD', 0):>7.2f} {s.get('PF', 0):>6.2f} {s.get('WR', 0):>5.1f}% "
              f"{s.get('MaxDD_R', 0):>7.1f}")

    if best_wf_row:
        print(f"\n  Best WF config: {best_wf_row['label']}")
        print(f"    Full-sample: {best_wf_row['trades']}t, R={best_wf_row['R']:.1f}, PPDD={best_wf_row['PPDD']:.2f}")
        print(f"    OOS: R={best_wf_row['oos_R']:.1f}, PPDD={best_wf_row['oos_PPDD']:.2f}, "
              f"Wins={best_wf_row['oos_wins']}")

    # Check goal: beats v3 on ALL three metrics
    print(f"\n  GOAL CHECK: Find v4 config that beats v3 (534t/+157R/PPDD=10.92) on ALL metrics")
    print(f"  {'-'*70}")
    for s in summary[1:]:  # skip v3 ref itself
        beats_trades = s["trades"] > 534
        beats_r = s.get("R", 0) > 156.6
        beats_ppdd = s.get("PPDD", 0) >= 10.0  # "comparable" PPDD = within ~10%
        all_three = beats_trades and beats_r and beats_ppdd
        status = "YES!" if all_three else "no"
        print(f"  {s['label']:<45} trades={'Y' if beats_trades else 'N'} "
              f"R={'Y' if beats_r else 'N'} PPDD={'Y' if beats_ppdd else 'N'} "
              f" => {status}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    t_start = _time.perf_counter()

    # Load everything once
    d = load_all()

    # Run all 4 experiments
    exp1_results = experiment_1_sessions(d)
    exp2_results = experiment_2_dual_threshold(d)
    exp3_results = experiment_3_sq_sweep(d)
    exp4_results, exp4_wf = experiment_4_combined(d)

    # Final comparison
    final_comparison(d, exp1_results, exp2_results, exp3_results, exp4_results, exp4_wf)

    # Save results
    all_exp4 = pd.DataFrame(exp4_results)
    out_path = PROJECT / "experiments" / "optimize_v4_results.csv"
    all_exp4.to_csv(out_path, index=False)
    print(f"\n  Exp4 grid results saved to: {out_path}")

    total_elapsed = _time.perf_counter() - t_start
    print(f"\n  Total runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print("  Done.")


if __name__ == "__main__":
    main()
