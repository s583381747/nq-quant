"""
experiments/sweep_trail_tp.py — Joint sweep of trail and TP parameters.

Sweeps:
  - trail nth_swing: 1, 2, 3
  - ny_tp_mult: 1.0, 1.5, 2.0, 2.5, 3.0
  - short_rr: 0.375, 0.50, 0.625, 0.75, 1.0
  - mss_long_tp_mult: 1.5, 2.0, 2.5, 3.0, 3.5

Base config: Config D (sq_short=0.80, block_pm_shorts=True)

Usage: python experiments/sweep_trail_tp.py
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", stream=sys.stdout)

# Reuse load_all from validate_improvements
from experiments.validate_improvements import (
    load_all,
    _find_nth_swing,
    _compute_grade_fast,
    compute_metrics,
    walk_forward_metrics,
    print_metrics,
)


# ======================================================================
# Backtest with parameterized trail + TP
# ======================================================================
def run_backtest_trail_tp(
    d: dict,
    # Config D base
    sq_long: float = 0.68,
    sq_short: float = 0.80,
    min_stop_atr: float = 1.7,
    block_pm_shorts: bool = True,
    # === SWEEP PARAMETERS ===
    nth_swing: int = 2,             # trail: which swing to use
    ny_tp_mult_param: float = 2.0,  # NY TP multiplier for longs
    short_rr_param: float = 0.625,  # short scalp TP ratio
    mss_long_tp_mult_param: float = 2.5,  # MSS long TP multiplier
    mss_short_rr_param: float = 0.50,     # MSS short TP ratio
    # Date range
    start_date=None, end_date=None,
) -> list[dict]:
    """Backtest with sweep-able trail and TP parameters."""

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
    dates = d["dates"]
    dow_arr = d["dow_arr"]
    et_frac_arr = d["et_frac_arr"]
    pa_alt_arr = d["pa_alt_arr"]
    swing_high_mask = d["swing_high_mask"]
    swing_low_mask = d["swing_low_mask"]
    nq = d["nq"]

    # Config from params.yaml
    pos_params = params["position"]
    risk_params = params["risk"]
    bt_params = params["backtest"]
    grading_params = params["grading"]
    trim_params = params["trim"]
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

    # Pre-extract price arrays for _find_nth_swing
    low_arr = nq["low"].values
    high_arr = nq["high"].values

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
                        # SWEEP: use nth_swing parameter
                        pos_trail_stop = _find_nth_swing(swing_low_mask, low_arr, i, nth_swing, 1)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True
                if pos_trimmed and not exited:
                    # SWEEP: use nth_swing parameter for ongoing trail
                    nt = _find_nth_swing(swing_low_mask, low_arr, i, nth_swing, 1)
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
                        # SWEEP: use nth_swing parameter
                        pos_trail_stop = _find_nth_swing(swing_high_mask, high_arr, i, nth_swing, -1)
                        if np.isnan(pos_trail_stop) or pos_trail_stop <= 0:
                            pos_trail_stop = pos_be_stop
                    if pos_remaining_contracts <= 0:
                        exit_price = pos_tp1
                        exit_reason = "tp1"
                        exit_contracts = pos_contracts
                        exited = True
                if pos_trimmed and not exited:
                    # SWEEP: use nth_swing parameter for ongoing trail
                    nt = _find_nth_swing(swing_high_mask, high_arr, i, nth_swing, -1)
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

        # Block PM shorts (Config D)
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

        # SQ filter
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

        # TP computation — SWEEP PARAMETERS APPLIED HERE
        is_mss_signal = str(sig_type[i]) == "mss"

        if session_rules.get("enabled", False):
            # SWEEP: use ny_tp_mult_param instead of config value
            if dir_mgmt.get("enabled", False):
                actual_tp_mult = dir_mgmt.get("long_tp_mult", ny_tp_mult_param) if direction == 1 else dir_mgmt.get("short_tp_mult", 1.25)
            else:
                actual_tp_mult = ny_tp_mult_param

            # SWEEP: use mss_long_tp_mult_param
            if mss_mgmt_enabled and is_mss_signal and direction == 1:
                actual_tp_mult = mss_long_tp_mult_param

            if 9.5 <= et_frac < 16.0:
                tp_distance = (tp1 - actual_entry) if direction == 1 else (actual_entry - tp1)
                tp1 = (actual_entry + tp_distance * actual_tp_mult) if direction == 1 else (actual_entry - tp_distance * actual_tp_mult)

        dual_mode_enabled = dual_mode.get("enabled", False)
        if dual_mode_enabled and direction == -1:
            # SWEEP: use short_rr_param
            eff_short_rr = short_rr_param
            # SWEEP: use mss_short_rr_param for MSS shorts
            if mss_mgmt_enabled and is_mss_signal:
                eff_short_rr = mss_short_rr_param
            tp1 = actual_entry - stop_dist * eff_short_rr

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

        if mss_mgmt_enabled and is_mss_signal:
            pos_trim_pct = mss_mgmt.get("short_trim_pct", 1.0) if direction == -1 else mss_mgmt.get("long_trim_pct", trim_pct)
        elif dual_mode_enabled and direction == -1:
            pos_trim_pct = dual_mode.get("short_trim_pct", 1.0)
        elif dir_mgmt.get("enabled", False):
            pos_trim_pct = dir_mgmt.get("long_trim_pct", trim_pct) if direction == 1 else dir_mgmt.get("short_trim_pct", 1.0)
        else:
            pos_trim_pct = trim_pct

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
        })

    return trades


# ======================================================================
# Helper: run one config and return metrics
# ======================================================================
def run_config(d, label, **kwargs):
    t0 = _time.perf_counter()
    trades = run_backtest_trail_tp(d, **kwargs)
    m = compute_metrics(trades)
    elapsed = _time.perf_counter() - t0
    return trades, m, elapsed


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 110)
    print("TRAIL + TP PARAMETER SWEEP")
    print("Base: Config D (sq_short=0.80, block_pm_shorts=True)")
    print("=" * 110)

    d = load_all()

    # Current defaults (Config D baseline)
    DEFAULTS = {
        "nth_swing": 2,
        "ny_tp_mult_param": 2.0,
        "short_rr_param": 0.625,
        "mss_long_tp_mult_param": 2.5,
        "mss_short_rr_param": 0.50,
    }

    # ---- BASELINE ----
    print("\n" + "=" * 110)
    print("BASELINE (Config D defaults)")
    print("=" * 110)
    bl_trades, bl_m, bl_t = run_config(d, "BASELINE", **DEFAULTS)
    print_metrics("BASELINE (Config D)", bl_m)
    print(f"  (computed in {bl_t:.1f}s)")

    # ======================================================================
    # PHASE 1: Individual sweeps
    # ======================================================================
    print("\n" + "=" * 110)
    print("PHASE 1: INDIVIDUAL PARAMETER SWEEPS (others fixed at Config D)")
    print("=" * 110)

    all_results = {}  # (param_name, value) -> (trades, metrics)

    # --- 1A: nth_swing ---
    print(f"\n{'─'*80}")
    print("1A. Trail nth_swing sweep")
    print(f"{'─'*80}")
    nth_values = [1, 2, 3]
    best_nth = {"PPDD": -999, "val": DEFAULTS["nth_swing"]}
    for val in nth_values:
        kw = {**DEFAULTS, "nth_swing": val}
        trades, m, elapsed = run_config(d, f"nth_swing={val}", **kw)
        delta_r = m["R"] - bl_m["R"]
        delta_ppdd = m["PPDD"] - bl_m["PPDD"]
        marker = " <-- CURRENT" if val == DEFAULTS["nth_swing"] else ""
        print_metrics(f"nth_swing={val}{marker}", m)
        print(f"    dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}  ({elapsed:.1f}s)")
        all_results[("nth_swing", val)] = (trades, m)
        if m["PPDD"] > best_nth["PPDD"]:
            best_nth = {"PPDD": m["PPDD"], "R": m["R"], "val": val}
    print(f"\n  >>> Best nth_swing: {best_nth['val']} (PPDD={best_nth['PPDD']:.2f}, R={best_nth['R']:.1f})")

    # --- 1B: ny_tp_mult ---
    print(f"\n{'─'*80}")
    print("1B. NY TP multiplier sweep (longs)")
    print(f"{'─'*80}")
    ny_tp_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    best_nytp = {"PPDD": -999, "val": DEFAULTS["ny_tp_mult_param"]}
    for val in ny_tp_values:
        kw = {**DEFAULTS, "ny_tp_mult_param": val}
        trades, m, elapsed = run_config(d, f"ny_tp_mult={val}", **kw)
        delta_r = m["R"] - bl_m["R"]
        delta_ppdd = m["PPDD"] - bl_m["PPDD"]
        marker = " <-- CURRENT" if val == DEFAULTS["ny_tp_mult_param"] else ""
        print_metrics(f"ny_tp_mult={val}{marker}", m)
        print(f"    dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}  ({elapsed:.1f}s)")
        all_results[("ny_tp_mult", val)] = (trades, m)
        if m["PPDD"] > best_nytp["PPDD"]:
            best_nytp = {"PPDD": m["PPDD"], "R": m["R"], "val": val}
    print(f"\n  >>> Best ny_tp_mult: {best_nytp['val']} (PPDD={best_nytp['PPDD']:.2f}, R={best_nytp['R']:.1f})")

    # --- 1C: short_rr ---
    print(f"\n{'─'*80}")
    print("1C. Short RR sweep (trend shorts)")
    print(f"{'─'*80}")
    short_rr_values = [0.375, 0.50, 0.625, 0.75, 1.0]
    best_srr = {"PPDD": -999, "val": DEFAULTS["short_rr_param"]}
    for val in short_rr_values:
        kw = {**DEFAULTS, "short_rr_param": val}
        trades, m, elapsed = run_config(d, f"short_rr={val}", **kw)
        delta_r = m["R"] - bl_m["R"]
        delta_ppdd = m["PPDD"] - bl_m["PPDD"]
        marker = " <-- CURRENT" if val == DEFAULTS["short_rr_param"] else ""
        print_metrics(f"short_rr={val}{marker}", m)
        print(f"    dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}  ({elapsed:.1f}s)")
        all_results[("short_rr", val)] = (trades, m)
        if m["PPDD"] > best_srr["PPDD"]:
            best_srr = {"PPDD": m["PPDD"], "R": m["R"], "val": val}
    print(f"\n  >>> Best short_rr: {best_srr['val']} (PPDD={best_srr['PPDD']:.2f}, R={best_srr['R']:.1f})")

    # --- 1D: mss_long_tp_mult ---
    print(f"\n{'─'*80}")
    print("1D. MSS Long TP multiplier sweep")
    print(f"{'─'*80}")
    mss_lt_values = [1.5, 2.0, 2.5, 3.0, 3.5]
    best_mlt = {"PPDD": -999, "val": DEFAULTS["mss_long_tp_mult_param"]}
    for val in mss_lt_values:
        kw = {**DEFAULTS, "mss_long_tp_mult_param": val}
        trades, m, elapsed = run_config(d, f"mss_long_tp={val}", **kw)
        delta_r = m["R"] - bl_m["R"]
        delta_ppdd = m["PPDD"] - bl_m["PPDD"]
        marker = " <-- CURRENT" if val == DEFAULTS["mss_long_tp_mult_param"] else ""
        print_metrics(f"mss_long_tp_mult={val}{marker}", m)
        print(f"    dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}  ({elapsed:.1f}s)")
        all_results[("mss_long_tp", val)] = (trades, m)
        if m["PPDD"] > best_mlt["PPDD"]:
            best_mlt = {"PPDD": m["PPDD"], "R": m["R"], "val": val}
    print(f"\n  >>> Best mss_long_tp_mult: {best_mlt['val']} (PPDD={best_mlt['PPDD']:.2f}, R={best_mlt['R']:.1f})")

    # ======================================================================
    # PHASE 1 SUMMARY
    # ======================================================================
    print("\n" + "=" * 110)
    print("PHASE 1 SUMMARY — Best individual values")
    print("=" * 110)
    print(f"  nth_swing:       {best_nth['val']} (current={DEFAULTS['nth_swing']})       PPDD={best_nth['PPDD']:.2f}")
    print(f"  ny_tp_mult:      {best_nytp['val']} (current={DEFAULTS['ny_tp_mult_param']})    PPDD={best_nytp['PPDD']:.2f}")
    print(f"  short_rr:        {best_srr['val']} (current={DEFAULTS['short_rr_param']})  PPDD={best_srr['PPDD']:.2f}")
    print(f"  mss_long_tp_mult:{best_mlt['val']} (current={DEFAULTS['mss_long_tp_mult_param']})    PPDD={best_mlt['PPDD']:.2f}")

    # ======================================================================
    # PHASE 2: Top combinations
    # ======================================================================
    print("\n" + "=" * 110)
    print("PHASE 2: TOP COMBINATIONS")
    print("=" * 110)

    combos = [
        ("BASELINE (Config D)",
         DEFAULTS),
        (f"ALL BEST (nth={best_nth['val']}, nytp={best_nytp['val']}, srr={best_srr['val']}, mlt={best_mlt['val']})",
         {"nth_swing": best_nth["val"], "ny_tp_mult_param": best_nytp["val"],
          "short_rr_param": best_srr["val"], "mss_long_tp_mult_param": best_mlt["val"],
          "mss_short_rr_param": DEFAULTS["mss_short_rr_param"]}),
        (f"BEST trail+TP (nth={best_nth['val']}, nytp={best_nytp['val']})",
         {**DEFAULTS, "nth_swing": best_nth["val"], "ny_tp_mult_param": best_nytp["val"]}),
        (f"BEST shorts (srr={best_srr['val']}, mlt={best_mlt['val']})",
         {**DEFAULTS, "short_rr_param": best_srr["val"], "mss_long_tp_mult_param": best_mlt["val"]}),
    ]

    # Also add a combo: best nth + best short_rr (trail tighter + TP adjusted)
    combos.append(
        (f"TRAIL+SHORT (nth={best_nth['val']}, srr={best_srr['val']})",
         {**DEFAULTS, "nth_swing": best_nth["val"], "short_rr_param": best_srr["val"]})
    )

    combo_results = {}
    for label, kw in combos:
        trades, m, elapsed = run_config(d, label, **kw)
        delta_r = m["R"] - bl_m["R"]
        delta_ppdd = m["PPDD"] - bl_m["PPDD"]
        print_metrics(label, m)
        print(f"    dR={delta_r:+.1f}, dPPDD={delta_ppdd:+.2f}  ({elapsed:.1f}s)")
        combo_results[label] = (trades, m, kw)

    # ======================================================================
    # WALK-FORWARD for top 3 configs
    # ======================================================================
    print("\n" + "=" * 110)
    print("WALK-FORWARD: PER-YEAR R BREAKDOWN")
    print("=" * 110)

    # Rank combos by PPDD (skip baseline)
    ranked = sorted(
        [(lab, tr, m, kw) for lab, (tr, m, kw) in combo_results.items()],
        key=lambda x: x[2]["PPDD"], reverse=True
    )

    # Take top 3 (including baseline for comparison)
    wf_configs = ranked[:3]
    # Always include baseline if not already there
    bl_in = any("BASELINE" in c[0] for c in wf_configs)
    if not bl_in:
        wf_configs.append(("BASELINE (Config D)", bl_trades, bl_m, DEFAULTS))

    print(f"\nConfigs for walk-forward:")
    for lab, _, m, _ in wf_configs:
        print(f"  {lab}: R={m['R']:+.1f}, PPDD={m['PPDD']:.2f}")

    # Print walk-forward table
    for lab, trades, m, kw in wf_configs:
        wf = walk_forward_metrics(trades)
        print(f"\n--- {lab} ---")
        print(f"  {'Year':>6} | {'n':>4} {'R':>7} {'WR':>6} {'PF':>6} {'PPDD':>7}")
        print(f"  {'-'*50}")
        for w in wf:
            print(f"  {w['year']:>4}   | {w['n']:4d} {w['R']:+7.1f} {w['WR']:5.1f}% {w['PF']:5.2f} {w['PPDD']:+7.2f}")
        # Count positive years
        pos_years = sum(1 for w in wf if w["R"] > 0)
        print(f"  Positive years: {pos_years}/{len(wf)}")

    # ======================================================================
    # ANALYSIS
    # ======================================================================
    print("\n" + "=" * 110)
    print("ANALYSIS")
    print("=" * 110)

    # 1. Which parameter has most impact on R?
    print("\n--- Parameter Impact on R ---")
    param_ranges = {}
    for param_name, values in [("nth_swing", nth_values), ("ny_tp_mult", ny_tp_values),
                                ("short_rr", short_rr_values), ("mss_long_tp", mss_lt_values)]:
        rs = [all_results[(param_name, v)][1]["R"] for v in values]
        r_range = max(rs) - min(rs)
        param_ranges[param_name] = r_range
        print(f"  {param_name:20s}: R range = {r_range:+.1f} (min={min(rs):+.1f}, max={max(rs):+.1f})")
    most_impact_r = max(param_ranges, key=param_ranges.get)
    print(f"  >>> Most impact on R: {most_impact_r} (range={param_ranges[most_impact_r]:+.1f})")

    # 2. Which parameter has most impact on PPDD?
    print("\n--- Parameter Impact on PPDD ---")
    ppdd_ranges = {}
    for param_name, values in [("nth_swing", nth_values), ("ny_tp_mult", ny_tp_values),
                                ("short_rr", short_rr_values), ("mss_long_tp", mss_lt_values)]:
        ppdds = [all_results[(param_name, v)][1]["PPDD"] for v in values]
        ppdd_range = max(ppdds) - min(ppdds)
        ppdd_ranges[param_name] = ppdd_range
        print(f"  {param_name:20s}: PPDD range = {ppdd_range:.2f} (min={min(ppdds):.2f}, max={max(ppdds):.2f})")
    most_impact_ppdd = max(ppdd_ranges, key=ppdd_ranges.get)
    print(f"  >>> Most impact on PPDD: {most_impact_ppdd} (range={ppdd_ranges[most_impact_ppdd]:.2f})")

    # 3. Interaction: trail vs TP
    print("\n--- Interaction: Trail x TP ---")
    # Compare: does combining best trail + best TP beat sum of individual improvements?
    bl_r = bl_m["R"]
    bl_ppdd = bl_m["PPDD"]
    nth_best_r = all_results[("nth_swing", best_nth["val"])][1]["R"]
    nytp_best_r = all_results[("ny_tp_mult", best_nytp["val"])][1]["R"]
    # Find the combo that has both
    combo_label = [l for l, (_, _, kw) in combo_results.items()
                   if kw.get("nth_swing") == best_nth["val"] and kw.get("ny_tp_mult_param") == best_nytp["val"]
                   and kw.get("short_rr_param") == DEFAULTS["short_rr_param"]]
    if combo_label:
        combo_r = combo_results[combo_label[0]][1]["R"]
        individual_sum = (nth_best_r - bl_r) + (nytp_best_r - bl_r)
        actual_combo_gain = combo_r - bl_r
        print(f"  Individual gains: trail dR={nth_best_r - bl_r:+.1f}, TP dR={nytp_best_r - bl_r:+.1f}, sum={individual_sum:+.1f}")
        print(f"  Combined gain:    dR={actual_combo_gain:+.1f}")
        if abs(individual_sum) > 0:
            interaction = actual_combo_gain - individual_sum
            print(f"  Interaction effect: {interaction:+.1f}R ({'synergistic' if interaction > 0 else 'antagonistic' if interaction < -1 else 'additive'})")
    else:
        print("  (could not isolate trail+TP combo)")

    # 4. Short management assessment
    print("\n--- Short Management Assessment ---")
    for val in short_rr_values:
        m = all_results[("short_rr", val)][1]
        trades_list = all_results[("short_rr", val)][0]
        short_trades = [t for t in trades_list if t["dir"] == -1]
        short_r = sum(t["r"] for t in short_trades)
        short_n = len(short_trades)
        short_wr = (sum(1 for t in short_trades if t["r"] > 0) / short_n * 100) if short_n > 0 else 0
        print(f"  short_rr={val:.3f}: {short_n} shorts, shortR={short_r:+.1f}, shortWR={short_wr:.1f}%, totalR={m['R']:+.1f}, PPDD={m['PPDD']:.2f}")

    # ======================================================================
    # FINAL RECOMMENDATION
    # ======================================================================
    print("\n" + "=" * 110)
    print("RECOMMENDATION")
    print("=" * 110)
    best_combo = ranked[0]
    print(f"  Best config by PPDD: {best_combo[0]}")
    print(f"  Metrics: {best_combo[2]['trades']}t, R={best_combo[2]['R']:+.1f}, PPDD={best_combo[2]['PPDD']:.2f}, PF={best_combo[2]['PF']:.2f}, MaxDD={best_combo[2]['MaxDD']:.1f}R")
    print(f"  vs Baseline: dR={best_combo[2]['R'] - bl_m['R']:+.1f}, dPPDD={best_combo[2]['PPDD'] - bl_m['PPDD']:+.2f}")

    print("\n" + "=" * 110)
    print("DONE")
    print("=" * 110)


if __name__ == "__main__":
    main()
