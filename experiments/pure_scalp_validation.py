"""
experiments/pure_scalp_validation.py — Pure Scalp mode validation

HYPOTHESIS: Setting trim_pct=1.0 for ALL trades (not just shorts) eliminates
the runner that gets killed by EOD close, potentially recovering PF from 1.21 -> 1.8+.

Sweeps:
  Phase 1: Pure scalp (100% trim) vs current (25% trim) at different TP multipliers
  Phase 2: Point-based stop filter sweep (>35pt, >40pt, >45pt, >50pt, >55pt)
  Phase 3: Combined best scalp TP + point filter + walk-forward
  Phase 4: MFE analysis — what % of trades reach 1R, 1.5R, 2R etc.

Based on validate_improvements.py engine (correct single-TP, EOD close enforced).
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", stream=sys.stdout)

DATA = PROJECT / "data"
CONFIG = PROJECT / "config" / "params.yaml"


# ======================================================================
# Data loading (from validate_improvements.py)
# ======================================================================
def load_all():
    t0 = _time.perf_counter()
    print("[LOAD] Loading data...")

    nq = pd.read_parquet(DATA / "NQ_5m_10yr.parquet")
    es = pd.read_parquet(DATA / "ES_5m_10yr.parquet")
    sig_v4 = pd.read_parquet(DATA / "cache_signals_10yr_v4.parquet")
    bias = pd.read_parquet(DATA / "cache_bias_10yr_v2.parquet")
    regime = pd.read_parquet(DATA / "cache_regime_10yr_v2.parquet")
    atr_cache = pd.read_parquet(DATA / "cache_atr_flu_10yr_v2.parquet")
    with open(CONFIG, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    from features.smt import compute_smt
    smt = compute_smt(nq, es, {'swing': {'left_bars': 3, 'right_bars': 1},
                                'smt': {'sweep_lookback': 15, 'time_tolerance': 1}})

    ss = sig_v4.copy()
    mm = ss['signal'].astype(bool) & (ss['signal_type'] == 'mss')
    mi = ss.index[mm]
    ss.loc[mi, 'signal'] = False
    ss.loc[mi, 'signal_dir'] = 0
    ss['has_smt'] = False
    c_idx = mi.intersection(smt.index)
    if len(c_idx) > 0:
        md = sig_v4.loc[c_idx, 'signal_dir'].values
        ok = ((md == 1) & smt.loc[c_idx, 'smt_bull'].values.astype(bool)) | \
             ((md == -1) & smt.loc[c_idx, 'smt_bear'].values.astype(bool))
        g = c_idx[ok]
        ss.loc[g, 'signal'] = sig_v4.loc[g, 'signal']
        ss.loc[g, 'signal_dir'] = sig_v4.loc[g, 'signal_dir']
        ss.loc[g, 'has_smt'] = True

    rem = ss['signal'].astype(bool) & (ss['signal_type'] == 'mss')
    mi2 = ss.index[rem]
    if len(mi2) > 0:
        et = mi2.tz_convert('US/Eastern')
        ef = et.hour + et.minute / 60.0
        kill = (ef >= 16.0) | (ef < 3.0)
        if kill.any():
            ss.loc[mi2[kill], ['signal', 'signal_dir']] = [False, 0]

    n_gated = int(ss['signal'].sum())
    print(f"[LOAD] v4 signals after SMT gate: {n_gated}")

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

    news_path = PROJECT / "config" / "news_calendar.csv"
    news_blackout_arr = None
    if news_path.exists():
        news_bl = build_news_blackout_mask(nq.index, str(news_path),
            params["news"]["blackout_minutes_before"],
            params["news"]["cooldown_minutes_after"])
        news_blackout_arr = news_bl.values

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

    stop_atr_ratio = np.full(n, np.nan)
    stop_points_arr = np.full(n, np.nan)
    if len(signal_indices) > 0:
        for idx in signal_indices:
            a = atr_arr[idx] if not np.isnan(atr_arr[idx]) else 10.0
            sd = abs(entry_price_arr[idx] - model_stop_arr[idx])
            stop_points_arr[idx] = sd
            if a > 0:
                stop_atr_ratio[idx] = sd / a

    target_rr_arr = np.full(n, np.nan)
    if len(signal_indices) > 0:
        for idx in signal_indices:
            ep = entry_price_arr[idx]
            sp = model_stop_arr[idx]
            tp = irl_target_arr[idx]
            stop_d = abs(ep - sp)
            targ_d = abs(tp - ep)
            if stop_d > 0:
                target_rr_arr[idx] = targ_d / stop_d

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
        "stop_points_arr": stop_points_arr,
        "target_rr_arr": target_rr_arr,
        "dates": dates, "dow_arr": dow_arr, "et_frac_arr": et_frac_arr,
        "et_idx": et_idx,
    }


# ======================================================================
# Backtest engine — pure scalp variant
# ======================================================================
def _find_nth_swing(mask, prices, idx, n_val, direction):
    count = 0
    for j in range(idx - 1, -1, -1):
        if mask[j]:
            count += 1
            if count == n_val:
                return prices[j]
    return np.nan


def _compute_grade_fast(ba, regime):
    if np.isnan(ba) or np.isnan(regime) or regime == 0.0:
        return "C"
    aligned = ba > 0.5
    full = regime >= 1.0
    if aligned and full:
        return "A+"
    if aligned or full:
        return "B+"
    return "C"


def run_backtest(
    d: dict,
    # Standard params
    sq_long: float = 0.68,
    sq_short: float = 0.82,
    min_stop_atr: float = 1.7,
    # === PURE SCALP PARAMS ===
    force_trim_pct: float | None = None,  # Override trim pct for ALL trades (1.0 = pure scalp)
    tp_mult_override: float | None = None,  # Override TP multiplier for longs
    min_stop_points: float = 0.0,  # Minimum stop distance in points (0=disabled)
    eod_close: bool = True,  # Enable/disable EOD close
    block_pm_shorts: bool = True,  # F3 default
    # MFE tracking
    track_mfe: bool = False,
) -> list[dict]:
    """Backtest with pure scalp support and point-based stop filter."""

    params = d["params"]
    n = d["n"]
    o, h, l, c_arr = d["o"], d["h"], d["l"], d["c"]
    atr_arr = d["atr_arr"]
    sig_mask = d["sig_mask"]
    sig_dir = d["sig_dir"]
    sig_type = d["sig_type"]
    has_smt_arr = d["has_smt_arr"]
    entry_price_arr = d["entry_price_arr"]
    model_stop_arr = d["model_stop_arr"]
    irl_target_arr = d["irl_target_arr"]
    bias_dir_arr = d["bias_dir_arr"]
    regime_arr = d["regime_arr"]
    news_blackout_arr = d["news_blackout_arr"]
    signal_quality = d["signal_quality"]
    stop_atr_ratio = d["stop_atr_ratio"]
    stop_points_arr = d["stop_points_arr"]
    dates = d["dates"]
    dow_arr = d["dow_arr"]
    et_frac_arr = d["et_frac_arr"]
    pa_alt_arr = d["pa_alt_arr"]
    swing_high_mask = d["swing_high_mask"]
    swing_low_mask = d["swing_low_mask"]
    nq = d["nq"]

    # Config
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
    tighten_factor = params.get("stop_loss", {}).get("tighten_factor", 1.0)

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
    pos_mfe = 0.0  # track MFE

    for i in range(n):
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

            # MFE tracking
            if track_mfe:
                if pos_direction == 1:
                    fe = h[i] - pos_entry_price
                else:
                    fe = pos_entry_price - l[i]
                if fe > pos_mfe:
                    pos_mfe = fe

            # EOD close at 15:55 ET
            if eod_close and et_frac_arr[i] >= 15.917:
                exit_price = c_arr[i] - slippage_points if pos_direction == 1 else c_arr[i] + slippage_points
                exit_reason = "eod_close"
                exited = True

            # PA quality early cut
            bars_in_trade = i - pos_entry_idx
            if not exited and not pos_trimmed and 2 <= bars_in_trade <= 4:
                pa_start = max(pos_entry_idx, 0)
                pa_end = i + 1
                pa_range = h[pa_start:pa_end] - l[pa_start:pa_end]
                pa_body = np.abs(c_arr[pa_start:pa_end] - o[pa_start:pa_end])
                safe_pa = np.where(pa_range > 0, pa_range, 1.0)
                pa_wick = 1.0 - (pa_body / safe_pa)
                avg_wick = float(np.mean(pa_wick))
                pa_dirs = np.sign(c_arr[pa_start:pa_end] - o[pa_start:pa_end])
                favorable = (pa_dirs == pos_direction).mean()
                disp = (c_arr[i] - pos_entry_price) if pos_direction == 1 else (pos_entry_price - c_arr[i])
                cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 30.0
                no_progress = disp < cur_atr * 0.3
                bad_pa = avg_wick > 0.65 and favorable < 0.5
                if bad_pa and no_progress and bars_in_trade >= 3:
                    exit_price = o[i+1] if i+1 < n else c_arr[i]
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

                trade_rec = {
                    "entry_time": nq.index[pos_entry_idx],
                    "exit_time": nq.index[i + 1] if exit_reason == "early_cut_pa" and i + 1 < n else nq.index[i],
                    "r": r_mult, "reason": exit_reason, "dir": pos_direction,
                    "type": pos_signal_type, "trimmed": pos_trimmed, "grade": pos_grade,
                    "entry_price": pos_entry_price, "exit_price": exit_price,
                    "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
                    "stop_dist_pts": stop_dist_exit,
                }
                if track_mfe:
                    trade_rec["mfe_pts"] = pos_mfe
                    trade_rec["mfe_r"] = (pos_mfe / stop_dist_exit) if stop_dist_exit > 0 else 0.0
                trades.append(trade_rec)

                daily_pnl_r += r_mult
                if exit_reason == "be_sweep" and pos_trimmed:
                    pass
                elif exit_reason == "eod_close":
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

        # Block PM shorts (F3 default)
        if block_pm_shorts and et_frac >= 14.0 and direction == -1:
            continue

        # Bias filter
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

        # Session filter
        is_mss_smt = (str(sig_type[i]) == "mss" and has_smt_arr[i]
                      and smt_cfg.get("enabled", False))
        mss_bypass = is_mss_smt and smt_cfg.get("bypass_session_filter", False)

        is_ny = (10.0 <= et_frac < 16.0)
        is_london = (3.0 <= et_frac < 9.5)
        is_asia = (et_frac >= 18.0) or (et_frac < 3.0)

        if not mss_bypass:
            if is_ny:
                pass
            elif is_london:
                continue
            elif is_asia:
                continue
            else:
                continue

        # Min stop ATR
        if not np.isnan(stop_atr_ratio[i]):
            if stop_atr_ratio[i] < min_stop_atr:
                continue
        else:
            stop_dist_check = abs(entry_p - stop)
            a_check = atr_arr[i] if not np.isnan(atr_arr[i]) else 10.0
            if a_check > 0 and (stop_dist_check / a_check) < min_stop_atr:
                continue

        # Min stop points filter
        if min_stop_points > 0:
            sd = stop_points_arr[i]
            if np.isnan(sd) or sd < min_stop_points:
                continue

        # SQ filter
        if not np.isnan(signal_quality[i]):
            eff_sq = sq_long if direction == 1 else sq_short
            if signal_quality[i] < eff_sq:
                continue

        if i + 1 >= n:
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

        # Session regime (lunch dead zone)
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

        # Slippage + entry
        actual_entry = entry_p + slippage_points if direction == 1 else entry_p - slippage_points
        stop_dist = abs(actual_entry - stop)
        if stop_dist < 1.0:
            continue

        # Apply stop tightening
        if tighten_factor < 1.0:
            if direction == 1:
                stop = actual_entry - stop_dist * tighten_factor
            else:
                stop = actual_entry + stop_dist * tighten_factor
            stop_dist = abs(actual_entry - stop)

        contracts = max(1, int(r_amount / (stop_dist * point_value))) if stop_dist > 0 and point_value > 0 else 0
        if contracts <= 0:
            continue

        # TP computation
        is_mss_signal = str(sig_type[i]) == "mss"
        if session_rules.get("enabled", False):
            ny_tp_mult = session_rules.get("ny_tp_multiplier", 1.0)
            actual_tp_mult = ny_tp_mult
            if mss_mgmt_enabled and is_mss_signal and direction == 1:
                actual_tp_mult = mss_mgmt.get("long_tp_mult", actual_tp_mult)
            # Override TP mult if specified
            if tp_mult_override is not None and direction == 1:
                actual_tp_mult = tp_mult_override
            if 9.5 <= et_frac < 16.0:
                tp_distance = (tp1 - actual_entry) if direction == 1 else (actual_entry - tp1)
                tp1 = (actual_entry + tp_distance * actual_tp_mult) if direction == 1 else (actual_entry - tp_distance * actual_tp_mult)

        # Short scalp mode (always 100% for shorts)
        dual_mode_enabled = dual_mode.get("enabled", False)
        if dual_mode_enabled and direction == -1:
            short_rr = dual_mode.get("short_rr", 0.625)
            if mss_mgmt_enabled and is_mss_signal:
                short_rr = mss_mgmt.get("short_rr", short_rr)
            tp1 = actual_entry - stop_dist * short_rr

        # Enter position
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
        pos_mfe = 0.0

        # Determine trim pct
        if force_trim_pct is not None:
            pos_trim_pct = force_trim_pct
        elif mss_mgmt_enabled and is_mss_signal:
            pos_trim_pct = mss_mgmt.get("short_trim_pct", 1.0) if direction == -1 else mss_mgmt.get("long_trim_pct", trim_pct)
        elif dual_mode_enabled and direction == -1:
            pos_trim_pct = dual_mode.get("short_trim_pct", 1.0)
        else:
            pos_trim_pct = trim_pct

    # Force close at end
    if in_position and pos_entry_idx < n:
        last_i = n - 1
        exit_price = c_arr[last_i]
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
            "entry_time": nq.index[pos_entry_idx],
            "exit_time": nq.index[last_i],
            "r": r_mult, "reason": "eod_close", "dir": pos_direction,
            "type": pos_signal_type, "trimmed": pos_trimmed, "grade": pos_grade,
            "entry_price": pos_entry_price, "exit_price": exit_price,
            "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
            "stop_dist_pts": stop_dist_exit,
        })

    return trades


# ======================================================================
# Metrics
# ======================================================================
def compute_metrics(trades_list):
    if not trades_list:
        return {"trades": 0, "R": 0.0, "PPDD": 0.0, "PF": 0.0, "WR": 0.0, "MaxDD": 0.0, "avgR": 0.0}
    r = np.array([t["r"] for t in trades_list])
    total_r = r.sum()
    wr = (r > 0).mean() * 100
    wins = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = wins / losses if losses > 0 else 999.0
    cumr = np.cumsum(r)
    peak = np.maximum.accumulate(cumr)
    dd = peak - cumr
    max_dd = dd.max() if len(dd) > 0 else 0
    ppdd = total_r / max_dd if max_dd > 0 else 999.0
    return {
        "trades": len(trades_list),
        "R": round(total_r, 1),
        "PPDD": round(ppdd, 2),
        "PF": round(pf, 2),
        "WR": round(wr, 1),
        "MaxDD": round(max_dd, 1),
        "avgR": round(total_r / len(trades_list), 4),
    }


def print_metrics(label, m, extra=""):
    # Calculate trades/day (approx 252 trading days/year, ~10.5 years of data)
    tpd = m["trades"] / (252 * 10.5) if m["trades"] > 0 else 0
    print(f"  {label:50s} | {m['trades']:4d}t | R={m['R']:+8.1f} | PPDD={m['PPDD']:6.2f} | PF={m['PF']:5.2f} | WR={m['WR']:5.1f}% | MaxDD={m['MaxDD']:5.1f}R | {tpd:.2f}/day{extra}")


def walk_forward(trades_list):
    if not trades_list:
        return []
    df = pd.DataFrame(trades_list)
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year
    results = []
    for year, grp in df.groupby("year"):
        r = grp["r"].values
        total_r = r.sum()
        wr = (r > 0).mean() * 100
        wins = r[r > 0].sum()
        losses = abs(r[r < 0].sum())
        pf = wins / losses if losses > 0 else 999.0
        cumr = np.cumsum(r)
        peak = np.maximum.accumulate(cumr)
        dd = (peak - cumr).max()
        ppdd = total_r / dd if dd > 0 else 999.0
        results.append({"year": year, "n": len(grp), "R": round(total_r, 1),
                        "WR": round(wr, 1), "PF": round(pf, 2), "PPDD": round(ppdd, 2)})
    return results


def exit_distribution(trades_list):
    """Print exit reason distribution."""
    if not trades_list:
        return
    df = pd.DataFrame(trades_list)
    print("\n  Exit Distribution:")
    for reason, grp in df.groupby("reason"):
        n = len(grp)
        avg_r = grp["r"].mean()
        total_r = grp["r"].sum()
        print(f"    {reason:15s}: {n:4d}t  avgR={avg_r:+.3f}  totalR={total_r:+.1f}")


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 120)
    print("PURE SCALP VALIDATION — Does 100% trim at TP1 recover PF killed by EOD close?")
    print("=" * 120)

    d = load_all()

    # ================================================================
    # PHASE 1: Current (F3) vs Pure Scalp at different TP multipliers
    # ================================================================
    print("\n" + "=" * 120)
    print("PHASE 1: F3 BASELINE vs PURE SCALP (trim=100%) — TP multiplier sweep")
    print("  F3 baseline: trim=25% longs, 100% shorts, TP=2.0x, EOD close ON")
    print("=" * 120)

    # F3 baseline
    t0 = _time.perf_counter()
    bl_trades = run_backtest(d)
    bl = compute_metrics(bl_trades)
    print_metrics("F3 BASELINE (trim=25%, TP=2.0x, EOD=ON)", bl)
    exit_distribution(bl_trades)
    print(f"  ({_time.perf_counter() - t0:.1f}s)")

    # F3 without EOD (for comparison)
    t0 = _time.perf_counter()
    no_eod_trades = run_backtest(d, eod_close=False)
    no_eod = compute_metrics(no_eod_trades)
    print_metrics("F3 NO EOD (trim=25%, TP=2.0x, EOD=OFF)", no_eod)
    exit_distribution(no_eod_trades)
    print(f"  ({_time.perf_counter() - t0:.1f}s)")

    # Pure scalp sweep
    tp_mults = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    print(f"\n  Pure Scalp (trim=100%) — TP mult sweep:")
    scalp_results = {}
    for tp in tp_mults:
        t0 = _time.perf_counter()
        trades = run_backtest(d, force_trim_pct=1.0, tp_mult_override=tp)
        m = compute_metrics(trades)
        label = f"SCALP trim=100% TP={tp:.2f}x EOD=ON"
        delta_pf = m["PF"] - bl["PF"]
        print_metrics(label, m, f" | ΔPF={delta_pf:+.2f}")
        scalp_results[tp] = (m, trades)

    # Find best scalp TP by PF
    best_tp = max(scalp_results, key=lambda k: scalp_results[k][0]["PF"])
    best_scalp = scalp_results[best_tp]
    print(f"\n  >>> Best scalp TP by PF: {best_tp:.2f}x -> PF={best_scalp[0]['PF']:.2f}")
    exit_distribution(best_scalp[1])

    # ================================================================
    # PHASE 2: Point-based stop filter sweep (with best scalp TP)
    # ================================================================
    print("\n" + "=" * 120)
    print(f"PHASE 2: POINT-BASED STOP FILTER SWEEP (scalp TP={best_tp:.2f}x)")
    print("  Higher min_stop_points -> fewer but higher quality trades")
    print("=" * 120)

    stop_pts = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    pt_results = {}
    for sp in stop_pts:
        trades = run_backtest(d, force_trim_pct=1.0, tp_mult_override=best_tp, min_stop_points=sp)
        m = compute_metrics(trades)
        label = f"SCALP TP={best_tp:.1f}x stop>{sp}pt"
        print_metrics(label, m)
        pt_results[sp] = (m, trades)

    # Also test without scalp (current runner mode) + point filter
    print(f"\n  Comparison: RUNNER mode (trim=25%) + point filter:")
    for sp in [0, 35, 40, 45, 50]:
        trades = run_backtest(d, min_stop_points=sp)
        m = compute_metrics(trades)
        label = f"RUNNER trim=25% stop>{sp}pt"
        print_metrics(label, m)

    # ================================================================
    # PHASE 3: Combined best — walk forward
    # ================================================================
    print("\n" + "=" * 120)
    print("PHASE 3: TOP CONFIGS — Walk-Forward Breakdown")
    print("=" * 120)

    # Top scalp configs by PF (filter PF >= 1.5 and trades >= 50)
    top_configs = []
    for sp, (m, trades) in pt_results.items():
        if m["PF"] >= 1.3 and m["trades"] >= 50:
            top_configs.append((sp, m, trades))
    top_configs.sort(key=lambda x: x[1]["PF"], reverse=True)

    for sp, m, trades in top_configs[:5]:
        label = f"SCALP TP={best_tp:.1f}x stop>{sp}pt"
        print(f"\n  {label}: {m['trades']}t R={m['R']:+.1f} PF={m['PF']:.2f} PPDD={m['PPDD']:.2f}")
        wf = walk_forward(trades)
        neg_years = 0
        for yr in wf:
            flag = " NEG" if yr["R"] < 0 else ""
            if yr["R"] < 0:
                neg_years += 1
            print(f"    {yr['year']}: {yr['n']:3d}t  R={yr['R']:+7.1f}  WR={yr['WR']:5.1f}%  PF={yr['PF']:5.2f}  PPDD={yr['PPDD']:6.2f}{flag}")
        print(f"    Negative years: {neg_years}/{len(wf)}")

    # ================================================================
    # PHASE 4: MFE Analysis — what % of trades reach each R level
    # ================================================================
    print("\n" + "=" * 120)
    print("PHASE 4: MFE ANALYSIS — What % of trades reach each R-multiple?")
    print("=" * 120)

    mfe_trades = run_backtest(d, track_mfe=True)
    mfe_df = pd.DataFrame(mfe_trades)
    if "mfe_r" in mfe_df.columns:
        mfe_r = mfe_df["mfe_r"].values
        print(f"\n  Total trades with MFE data: {len(mfe_r)}")
        print(f"  MFE distribution (R-multiples):")
        print(f"    {'Level':>8}  {'Hit%':>6}  {'Cumul Trades':>12}  {'Implied PF @scalp':>18}")
        for level in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]:
            hit = (mfe_r >= level).mean()
            miss = 1.0 - hit
            # Implied PF if we exit at this level: winners get level*R, losers get ~-1R
            # PF = (hit * level) / (miss * 1.0)
            implied_pf = (hit * level) / (miss * 1.0) if miss > 0 else 999
            n_hit = int((mfe_r >= level).sum())
            print(f"    {level:>6.1f}R  {hit*100:>5.1f}%  {n_hit:>12d}        PF~{implied_pf:.2f}")

        # By direction
        for dir_name, dir_val in [("LONG", 1), ("SHORT", -1)]:
            sub = mfe_df[mfe_df["dir"] == dir_val]
            if len(sub) == 0:
                continue
            mfe_sub = sub["mfe_r"].values
            print(f"\n  {dir_name} MFE ({len(sub)} trades):")
            for level in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                hit = (mfe_sub >= level).mean()
                miss = 1.0 - hit
                implied_pf = (hit * level) / (miss * 1.0) if miss > 0 else 999
                print(f"    {level:>6.1f}R  {hit*100:>5.1f}%  PF~{implied_pf:.2f}")

        # By signal type
        for stype in ["trend", "mss"]:
            sub = mfe_df[mfe_df["type"] == stype]
            if len(sub) == 0:
                continue
            mfe_sub = sub["mfe_r"].values
            print(f"\n  {stype.upper()} MFE ({len(sub)} trades):")
            for level in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                hit = (mfe_sub >= level).mean()
                miss = 1.0 - hit
                implied_pf = (hit * level) / (miss * 1.0) if miss > 0 else 999
                print(f"    {level:>6.1f}R  {hit*100:>5.1f}%  PF~{implied_pf:.2f}")

    # ================================================================
    # PHASE 5: Best achievable PF + frequency summary
    # ================================================================
    print("\n" + "=" * 120)
    print("PHASE 5: SUMMARY — PF vs FREQUENCY FRONTIER")
    print("=" * 120)
    print(f"\n  {'Config':50s} | {'Trades':>6} | {'t/day':>5} | {'PF':>5} | {'R':>8} | {'PPDD':>6} | {'WR':>5}")
    print("  " + "-" * 100)

    # Collect all tested configs
    all_configs = []
    all_configs.append(("F3 BASELINE (runner, EOD)", bl))
    all_configs.append(("F3 NO EOD (runner, no EOD)", no_eod))
    for tp, (m, _) in scalp_results.items():
        all_configs.append((f"SCALP TP={tp:.1f}x (no pt filter)", m))
    for sp, (m, _) in pt_results.items():
        if sp > 0:
            all_configs.append((f"SCALP TP={best_tp:.1f}x stop>{sp}pt", m))

    all_configs.sort(key=lambda x: x[1]["PF"], reverse=True)
    for label, m in all_configs:
        tpd = m["trades"] / (252 * 10.5) if m["trades"] > 0 else 0
        pf_flag = " OK" if m["PF"] >= 1.8 else ""
        freq_flag = " OK" if tpd >= 0.5 else ""
        print(f"  {label:50s} | {m['trades']:6d} | {tpd:5.2f}{freq_flag} | {m['PF']:5.2f}{pf_flag} | {m['R']:+8.1f} | {m['PPDD']:6.2f} | {m['WR']:5.1f}%")

    print("\n  TARGET: PF≥1.80 OK AND freq≥0.50/day OK")
    target_met = [(l, m) for l, m in all_configs
                  if m["PF"] >= 1.8 and (m["trades"] / (252 * 10.5)) >= 0.5]
    if target_met:
        print(f"  >>> CONFIGS MEETING BOTH TARGETS: {len(target_met)}")
        for l, m in target_met:
            print(f"      {l}")
    else:
        print("  >>> NO CONFIG MEETS BOTH TARGETS SIMULTANEOUSLY")
        # Show closest
        best_pf = max(all_configs, key=lambda x: x[1]["PF"])
        best_freq_with_pf = max(
            [(l, m) for l, m in all_configs if m["PF"] >= 1.5],
            key=lambda x: x[1]["trades"],
            default=None
        )
        print(f"  Highest PF: {best_pf[0]} -> PF={best_pf[1]['PF']:.2f}, {best_pf[1]['trades']/(252*10.5):.2f}/day")
        if best_freq_with_pf:
            tpd = best_freq_with_pf[1]["trades"] / (252 * 10.5)
            print(f"  Most trades with PF≥1.5: {best_freq_with_pf[0]} -> PF={best_freq_with_pf[1]['PF']:.2f}, {tpd:.2f}/day")

    print("\n" + "=" * 120)
    print("DONE")
    print("=" * 120)


if __name__ == "__main__":
    main()
