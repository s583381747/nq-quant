"""
experiments/sweep_trim_ratio.py — Sweep the long trim ratio parameter.

The trim ratio is the fraction of position exited at TP1.
  - Higher trim → lock more profit at TP1, smaller runner, less upside
  - Lower trim → lock less at TP1, bigger runner, more trailing potential

For shorts (both dual_mode and MSS): always 1.0 (full exit). Not changed.

Uses Config D: sq_short=0.80, block_pm_shorts=True

Usage: python experiments/sweep_trim_ratio.py
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

# Import load_all and helpers from validate_improvements
from experiments.validate_improvements import (
    load_all,
    _find_nth_swing,
    _compute_grade_fast,
    compute_metrics,
    walk_forward_metrics,
    print_metrics,
)


# ======================================================================
# Modified backtest with long_trim_pct override
# ======================================================================
def run_backtest_trim_sweep(
    d: dict,
    long_trim_pct: float = 0.50,
    # Config D params
    sq_short: float = 0.80,
    block_pm_shorts: bool = True,
    # Date range
    start_date=None, end_date=None,
) -> list[dict]:
    """
    Backtest with overridable long_trim_pct.

    - For long trend signals: uses long_trim_pct
    - For MSS longs: uses long_trim_pct
    - For short signals (dual_mode + MSS): always 1.0
    """

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
    target_rr_arr = d["target_rr_arr"]
    dates = d["dates"]
    dow_arr = d["dow_arr"]
    et_frac_arr = d["et_frac_arr"]
    pa_alt_arr = d["pa_alt_arr"]
    swing_high_mask = d["swing_high_mask"]
    swing_low_mask = d["swing_low_mask"]
    nq = d["nq"]
    ts_minutes = d["ts_minutes"]

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
    # Default trim_pct from params (overridden for longs below)
    default_trim_pct = trim_params["pct"]
    be_after_trim = trim_params["be_after_trim"]
    nth_swing = trail_params["use_nth_swing"]

    # SQ thresholds — Config D uses sq_short override
    sq_long_val = 0.68  # standard
    sq_short_val = sq_short  # from parameter (0.80 for Config D)
    min_stop_atr = 1.7  # standard

    # Date range
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

    # State
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
    pos_trim_pct = default_trim_pct

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
                    exit_price = o[i+1] if i+1 < end_idx else c_arr[i]
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

                # Track trim vs runner R contribution
                trim_r_contrib = 0.0
                runner_r_contrib = 0.0
                if pos_trimmed and exit_reason != "tp1":
                    trim_pnl_only = (pos_tp1 - pos_entry_price) if pos_direction == 1 else (pos_entry_price - pos_tp1)
                    trim_c_only = pos_contracts - exit_contracts
                    trim_r_contrib = (trim_pnl_only * point_value * trim_c_only) / total_risk if total_risk > 0 else 0.0
                    runner_pnl = pnl_pts * point_value * exit_contracts
                    runner_r_contrib = runner_pnl / total_risk if total_risk > 0 else 0.0
                elif exit_reason == "tp1":
                    # Full exit at TP1 — all is "trim" contribution
                    trim_r_contrib = r_mult
                    runner_r_contrib = 0.0
                else:
                    # Stopped out before trim — all is "runner" (untrimmed) contribution
                    trim_r_contrib = 0.0
                    runner_r_contrib = r_mult

                trades.append({
                    "entry_time": nq.index[pos_entry_idx],
                    "exit_time": nq.index[i + 1] if exit_reason == "early_cut_pa" and i + 1 < end_idx else nq.index[i],
                    "r": r_mult, "reason": exit_reason, "dir": pos_direction,
                    "type": pos_signal_type, "trimmed": pos_trimmed, "grade": pos_grade,
                    "entry_price": pos_entry_price, "exit_price": exit_price,
                    "stop_price": pos_stop, "tp1_price": pos_tp1, "pnl_dollars": total_pnl,
                    "trim_r": trim_r_contrib, "runner_r": runner_r_contrib,
                    "actual_trim_pct": pos_trim_pct,
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

        # Config D: block PM shorts
        if block_pm_shorts:
            if et_frac >= 14.0 and direction == -1:
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

        # SQ filter — Config D uses sq_short=0.80
        if not np.isnan(signal_quality[i]):
            eff_sq = sq_long_val if direction == 1 else sq_short_val
            if signal_quality[i] < eff_sq:
                continue

        if i + 1 >= end_idx:
            continue

        # Grade
        ba = 1.0 if (direction == np.sign(bias_dir_arr[i]) and bias_dir_arr[i] != 0) else 0.0
        grade = _compute_grade_fast(ba, regime_arr[i])
        if grade == "C" and c_skip:
            continue

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

        # Slippage + entry
        actual_entry = entry_p + slippage_points if direction == 1 else entry_p - slippage_points
        stop_dist = abs(actual_entry - stop)
        if stop_dist < 1.0:
            continue

        contracts = max(1, int(r_amount / (stop_dist * point_value))) if stop_dist > 0 and point_value > 0 else 0
        if contracts <= 0:
            continue

        # TP computation
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

        # === TRIM PCT ASSIGNMENT — THE KEY OVERRIDE ===
        # Shorts always 1.0 (full exit). Longs use the sweep value.
        if mss_mgmt_enabled and is_mss_signal:
            if direction == -1:
                pos_trim_pct = mss_mgmt.get("short_trim_pct", 1.0)
            else:
                pos_trim_pct = long_trim_pct  # OVERRIDE
        elif dual_mode_enabled and direction == -1:
            pos_trim_pct = dual_mode.get("short_trim_pct", 1.0)
        elif direction == -1:
            # Any other short: keep 1.0
            pos_trim_pct = 1.0
        else:
            # LONG — use sweep value
            pos_trim_pct = long_trim_pct

    # Force close at end
    if in_position and pos_entry_idx < end_idx:
        last_i = end_idx - 1
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
            "trim_r": 0.0, "runner_r": r_mult,
            "actual_trim_pct": pos_trim_pct,
        })

    return trades


# ======================================================================
# Extended metrics with trim/runner breakdown
# ======================================================================
def compute_extended_metrics(trades_list):
    """Standard metrics + trim/runner breakdown."""
    base = compute_metrics(trades_list)
    if not trades_list:
        base.update({"trim_R": 0.0, "runner_R": 0.0, "trimmed_trades": 0, "untrimmed_trades": 0})
        return base

    df = pd.DataFrame(trades_list)
    trim_r_total = df["trim_r"].sum()
    runner_r_total = df["runner_r"].sum()
    trimmed_n = int(df["trimmed"].sum())
    untrimmed_n = len(df) - trimmed_n

    # Break down by direction
    longs = df[df["dir"] == 1]
    shorts = df[df["dir"] == -1]

    base.update({
        "trim_R": round(trim_r_total, 1),
        "runner_R": round(runner_r_total, 1),
        "trimmed_trades": trimmed_n,
        "untrimmed_trades": untrimmed_n,
        "long_n": len(longs),
        "long_R": round(longs["r"].sum(), 1) if len(longs) > 0 else 0.0,
        "short_n": len(shorts),
        "short_R": round(shorts["r"].sum(), 1) if len(shorts) > 0 else 0.0,
    })
    return base


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 120)
    print("TRIM RATIO SWEEP — Config D (sq_short=0.80, block_pm_shorts=True)")
    print("Long trim_pct sweep. Shorts always 1.0 (full exit).")
    print("=" * 120)

    d = load_all()

    # Sweep values
    sweep_values = [0.00, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.00]

    all_results = {}
    all_trades = {}

    print("\n" + "=" * 120)
    print("SWEEP RESULTS")
    print("=" * 120)
    header = (f"{'trim%':>6} | {'trades':>6} | {'R':>8} | {'PPDD':>7} | {'PF':>5} | "
              f"{'WR':>6} | {'MaxDD':>6} | {'avgR':>8} | "
              f"{'trimR':>7} | {'runR':>7} | {'trim_n':>6} | {'untrim':>6} | "
              f"{'longN':>5} {'longR':>7} | {'shrtN':>5} {'shrtR':>7}")
    print(header)
    print("-" * 150)

    for trim_val in sweep_values:
        t0 = _time.perf_counter()
        trades = run_backtest_trim_sweep(d, long_trim_pct=trim_val)
        elapsed = _time.perf_counter() - t0
        m = compute_extended_metrics(trades)
        all_results[trim_val] = m
        all_trades[trim_val] = trades

        marker = " <-- CURRENT" if trim_val == 0.50 else ""
        print(f"  {trim_val:4.2f} | {m['trades']:6d} | {m['R']:+8.1f} | {m['PPDD']:7.2f} | "
              f"{m['PF']:5.2f} | {m['WR']:5.1f}% | {m['MaxDD']:6.1f} | {m['avgR']:+8.4f} | "
              f"{m['trim_R']:+7.1f} | {m['runner_R']:+7.1f} | {m['trimmed_trades']:6d} | {m['untrimmed_trades']:6d} | "
              f"{m['long_n']:5d} {m['long_R']:+7.1f} | {m['short_n']:5d} {m['short_R']:+7.1f}"
              f"  ({elapsed:.1f}s){marker}")

    # ---- Find top 3 by PPDD ----
    sorted_by_ppdd = sorted(all_results.items(), key=lambda x: x[1]["PPDD"], reverse=True)
    top3_ppdd = [x[0] for x in sorted_by_ppdd[:3]]

    # Also find top 3 by R
    sorted_by_r = sorted(all_results.items(), key=lambda x: x[1]["R"], reverse=True)
    top3_r = [x[0] for x in sorted_by_r[:3]]

    print("\n" + "=" * 120)
    print(f"TOP 3 by PPDD: {[f'{v:.2f}' for v in top3_ppdd]}")
    print(f"TOP 3 by R:    {[f'{v:.2f}' for v in top3_r]}")
    print("=" * 120)

    # ---- Walk-forward for top 3 by PPDD ----
    print("\n" + "=" * 120)
    print("WALK-FORWARD PER-YEAR BREAKDOWN (top 3 by PPDD)")
    print("=" * 120)

    wf_data = {}
    for trim_val in top3_ppdd:
        wf = walk_forward_metrics(all_trades[trim_val])
        wf_data[trim_val] = {w["year"]: w for w in wf}

    # Also include current 0.50 if not in top3
    if 0.50 not in top3_ppdd:
        wf = walk_forward_metrics(all_trades[0.50])
        wf_data[0.50] = {w["year"]: w for w in wf}

    # Collect all years
    all_years = sorted(set(y for wd in wf_data.values() for y in wd.keys()))

    # Print walk-forward table
    trim_vals_to_show = sorted(set(top3_ppdd + [0.50]))
    header_parts = [f"{'Year':>6}"]
    for tv in trim_vals_to_show:
        label = f"trim={tv:.2f}"
        if tv == 0.50:
            label += "*"
        header_parts.append(f"{'n':>4} {'R':>7} {'WR':>5} {'PF':>5} {'PPDD':>7}")
    print("  " + " | ".join(header_parts))
    print("  " + "-" * (8 + len(trim_vals_to_show) * 35))

    # Print headers with trim labels
    label_row = f"{'':>6}"
    for tv in trim_vals_to_show:
        lbl = f"trim={tv:.2f}" + ("*" if tv == 0.50 else "")
        label_row += f" | {lbl:^32s}"
    print("  " + label_row)
    print("  " + "-" * (8 + len(trim_vals_to_show) * 35))

    for year in all_years:
        row = f"  {year:>4}"
        for tv in trim_vals_to_show:
            yw = wf_data.get(tv, {}).get(year, {"n": 0, "R": 0, "WR": 0, "PF": 0, "PPDD": 0})
            row += f" | {yw['n']:4d} {yw['R']:+7.1f} {yw['WR']:5.1f} {yw['PF']:5.2f} {yw['PPDD']:+7.2f}"
        print(row)

    # ---- Per-year PPDD stability analysis ----
    print("\n" + "=" * 120)
    print("STABILITY ANALYSIS: Which trim ratio is best per year?")
    print("=" * 120)

    # For all sweep values, compute per-year metrics
    print(f"\n  {'Year':>6}", end="")
    for tv in sweep_values:
        print(f" | {tv:5.2f}", end="")
    print(" | best")
    print("  " + "-" * (8 + len(sweep_values) * 8 + 10))

    year_best = {}
    for year in all_years:
        row = f"  {year:>4}"
        best_ppdd_year = -999
        best_tv_year = 0.50
        for tv in sweep_values:
            wf = walk_forward_metrics(all_trades[tv])
            wf_dict = {w["year"]: w for w in wf}
            yw = wf_dict.get(year, {"PPDD": 0})
            ppdd_val = yw["PPDD"]
            row += f" | {ppdd_val:5.2f}"
            if ppdd_val > best_ppdd_year:
                best_ppdd_year = ppdd_val
                best_tv_year = tv
        row += f" | {best_tv_year:.2f}"
        year_best[year] = best_tv_year
        print(row)

    # Summary
    print(f"\n  Per-year optimal trim ratios: {year_best}")
    best_vals = list(year_best.values())
    print(f"  Mean optimal: {np.mean(best_vals):.2f}, Median: {np.median(best_vals):.2f}, Std: {np.std(best_vals):.2f}")

    # ---- Per-year R for top 3 (detect stability) ----
    print("\n" + "=" * 120)
    print("R BY YEAR for top 3 PPDD trim values")
    print("=" * 120)

    print(f"\n  {'Year':>6}", end="")
    for tv in top3_ppdd:
        print(f" | trim={tv:.2f} R", end="")
    print()
    print("  " + "-" * (8 + len(top3_ppdd) * 18))

    for year in all_years:
        row = f"  {year:>4}"
        for tv in top3_ppdd:
            wf = walk_forward_metrics(all_trades[tv])
            wf_dict = {w["year"]: w for w in wf}
            yw = wf_dict.get(year, {"R": 0})
            row += f" | {yw['R']:+13.1f}"
        print(row)

    # ---- Final recommendation ----
    print("\n" + "=" * 120)
    print("ANALYSIS & RECOMMENDATION")
    print("=" * 120)

    best_ppdd_val = sorted_by_ppdd[0]
    best_r_val = sorted_by_r[0]
    current_val = all_results[0.50]

    print(f"\n  Current (0.50): R={current_val['R']:+.1f}, PPDD={current_val['PPDD']:.2f}, PF={current_val['PF']:.2f}")
    print(f"  Best by PPDD ({best_ppdd_val[0]:.2f}): R={best_ppdd_val[1]['R']:+.1f}, PPDD={best_ppdd_val[1]['PPDD']:.2f}, PF={best_ppdd_val[1]['PF']:.2f}")
    print(f"  Best by R ({best_r_val[0]:.2f}): R={best_r_val[1]['R']:+.1f}, PPDD={best_r_val[1]['PPDD']:.2f}, PF={best_r_val[1]['PF']:.2f}")

    delta_r_ppdd = best_ppdd_val[1]["R"] - current_val["R"]
    delta_ppdd_ppdd = best_ppdd_val[1]["PPDD"] - current_val["PPDD"]
    delta_r_r = best_r_val[1]["R"] - current_val["R"]
    delta_ppdd_r = best_r_val[1]["PPDD"] - current_val["PPDD"]

    print(f"\n  vs current 0.50:")
    print(f"    Best PPDD ({best_ppdd_val[0]:.2f}): delta_R={delta_r_ppdd:+.1f}, delta_PPDD={delta_ppdd_ppdd:+.2f}")
    print(f"    Best R ({best_r_val[0]:.2f}):    delta_R={delta_r_r:+.1f}, delta_PPDD={delta_ppdd_r:+.2f}")

    # Plateau detection
    print("\n  Plateau analysis (PPDD within 5% of best):")
    best_ppdd_score = best_ppdd_val[1]["PPDD"]
    threshold = best_ppdd_score * 0.95
    plateau = [tv for tv, m in all_results.items() if m["PPDD"] >= threshold]
    print(f"    Trim values within 5% of best PPDD: {[f'{v:.2f}' for v in sorted(plateau)]}")
    if len(plateau) >= 3:
        print(f"    --> PLATEAU detected: {len(plateau)} values in the range [{min(plateau):.2f}, {max(plateau):.2f}]")
    else:
        print(f"    --> SHARP OPTIMUM: only {len(plateau)} value(s) near the top")

    print("\n" + "=" * 120)
    print("DONE")
    print("=" * 120)


if __name__ == "__main__":
    main()
