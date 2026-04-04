"""
experiments/build_signal_database.py — Build comprehensive signal feature database.

Creates a single parquet file where EVERY row is a raw signal (not just traded ones),
with full features + outcome labels. Foundation for all subsequent feature importance analysis.

Usage: python experiments/build_signal_database.py
"""

import sys
import time as _time
from pathlib import Path

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
logger = logging.getLogger("build_signal_db")


# ============================================================
# 1. LOAD ALL DATA (same pipeline as validate_nt_logic.py)
# ============================================================
def load_all_data():
    """Load all data sources needed for feature computation."""
    logger.info("Loading data...")
    t0 = _time.perf_counter()

    nq = pd.read_parquet(PROJECT / "data" / "NQ_5m_10yr.parquet")
    es = pd.read_parquet(PROJECT / "data" / "ES_5m_10yr.parquet")
    sig3 = pd.read_parquet(PROJECT / "data" / "cache_signals_10yr_v3.parquet")
    bias = pd.read_parquet(PROJECT / "data" / "cache_bias_10yr_v2.parquet")
    regime = pd.read_parquet(PROJECT / "data" / "cache_regime_10yr_v2.parquet")
    atr_flu = pd.read_parquet(PROJECT / "data" / "cache_atr_flu_10yr_v2.parquet")
    orm = pd.read_parquet(PROJECT / "data" / "cache_orm_10yr_v2.parquet")

    with open(PROJECT / "config" / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    logger.info("Loaded in %.1fs — NQ: %d bars, signals: %d",
                _time.perf_counter() - t0, len(nq), sig3["signal"].sum())
    return nq, es, sig3, bias, regime, atr_flu, orm, params


# ============================================================
# 2. COMPUTE SMT + MERGE SIGNALS (same as validate_nt_logic.py)
# ============================================================
def compute_smt_and_merge(nq, es, sig3, params):
    """Compute SMT divergence and gate MSS signals."""
    logger.info("Computing SMT divergence...")
    from features.smt import compute_smt
    smt = compute_smt(nq, es, {
        "swing": {"left_bars": 3, "right_bars": 1},
        "smt": {"sweep_lookback": 15, "time_tolerance": 1},
    })

    # Merge: gate MSS by SMT
    ss = sig3.copy()
    mm = ss["signal"].astype(bool) & (ss["signal_type"] == "mss")
    mi = ss.index[mm]
    ss.loc[mi, "signal"] = False
    ss.loc[mi, "signal_dir"] = 0
    ss["has_smt"] = False

    c = mi.intersection(smt.index)
    if len(c) > 0:
        md = sig3.loc[c, "signal_dir"].values
        ok = ((md == 1) & smt.loc[c, "smt_bull"].values.astype(bool)) | \
             ((md == -1) & smt.loc[c, "smt_bear"].values.astype(bool))
        g = c[ok]
        ss.loc[g, "signal"] = sig3.loc[g, "signal"]
        ss.loc[g, "signal_dir"] = sig3.loc[g, "signal_dir"]
        ss.loc[g, "has_smt"] = True

    # Kill MSS in overnight (16:00-03:00 ET)
    rem = ss["signal"].astype(bool) & (ss["signal_type"] == "mss")
    mi2 = ss.index[rem]
    if len(mi2) > 0:
        et = mi2.tz_convert("US/Eastern")
        ef = et.hour + et.minute / 60.0
        kill = (ef >= 16.0) | (ef < 3.0)
        if kill.any():
            ss.loc[mi2[kill], ["signal", "signal_dir"]] = [False, 0]

    # For the database, we want ALL signals from v3 cache (before SMT gating)
    # We'll track has_smt separately
    raw_sig_mask = sig3["signal"].astype(bool)
    logger.info("Raw signals: %d, After SMT gate: %d",
                raw_sig_mask.sum(), ss["signal"].sum())

    return ss, smt


# ============================================================
# 3. PRE-COMPUTE ARRAYS (same as engine.py / validate_nt_logic)
# ============================================================
def precompute_arrays(nq, ss, sig3, bias, regime, atr_flu, orm, params):
    """Pre-compute all arrays needed for feature extraction and filtering."""
    logger.info("Pre-computing arrays...")
    from features.displacement import compute_atr, compute_fluency
    from features.swing import compute_swing_levels
    from features.pa_quality import compute_alternating_dir_ratio
    from features.news_filter import build_news_blackout_mask

    n = len(nq)
    et_idx = nq.index.tz_convert("US/Eastern")
    o = nq["open"].values
    h = nq["high"].values
    l = nq["low"].values
    c = nq["close"].values

    # ATR and fluency
    atr_arr = atr_flu["atr"].values if "atr" in atr_flu.columns else compute_atr(nq).values
    fluency_arr = atr_flu["fluency"].values if "fluency" in atr_flu.columns else compute_fluency(nq, params).values

    # PA quality
    pa_alt_arr = compute_alternating_dir_ratio(nq, window=6).values

    # Swings
    swing_p = {"left_bars": params["swing"]["left_bars"],
               "right_bars": params["swing"]["right_bars"]}
    swings = compute_swing_levels(nq, swing_p)
    swings["swing_high_price"] = swings["swing_high_price"].shift(1).ffill()
    swings["swing_low_price"] = swings["swing_low_price"].shift(1).ffill()
    swing_high_mask = swings["swing_high"].shift(1, fill_value=False).values
    swing_low_mask = swings["swing_low"].shift(1, fill_value=False).values

    # Signal arrays
    sig_mask = ss["signal"].values.astype(bool)
    sig_dir = ss["signal_dir"].values.astype(float)
    sig_type = ss["signal_type"].values
    has_smt_arr = ss["has_smt"].values.astype(bool)
    entry_price_arr = ss["entry_price"].values
    model_stop_arr = ss["model_stop"].values
    irl_target_arr = ss["irl_target"].values

    # Bias/regime
    bias_dir_arr = bias["bias_direction"].values.astype(float)
    bias_conf_arr = bias["bias_confidence"].values.astype(float)
    regime_arr = regime["regime"].values.astype(float)

    # News filter
    news_path = PROJECT / "config" / "news_calendar.csv"
    news_blackout_arr = None
    if news_path.exists():
        news_bl = build_news_blackout_mask(
            nq.index, str(news_path),
            params["news"]["blackout_minutes_before"],
            params["news"]["cooldown_minutes_after"],
        )
        news_blackout_arr = news_bl.values

    # Signal quality (replicate engine.py SQ computation)
    # Compute for ALL raw signals (sig3 mask), not just merged ones
    sq_params = params.get("signal_quality", {})
    sq_enabled = sq_params.get("enabled", False)
    signal_quality = np.full(n, np.nan)
    raw_signal_quality = np.full(n, np.nan)

    # Compute for merged signals first (for d["signal_quality"])
    signal_indices = np.where(sig_mask)[0]

    # Also compute for ALL raw signals
    raw_sig3_mask = sig3["signal"].astype(bool).values
    raw_signal_indices = np.where(raw_sig3_mask)[0]

    # Use sig3's entry_price/model_stop for raw signal quality
    raw_entry = sig3["entry_price"].values
    raw_stop = sig3["model_stop"].values

    if sq_enabled and len(raw_signal_indices) > 0:
        for idx in raw_signal_indices:
            a = atr_arr[idx] if not np.isnan(atr_arr[idx]) else 10.0
            gap = abs(raw_entry[idx] - raw_stop[idx]) if not (np.isnan(raw_entry[idx]) or np.isnan(raw_stop[idx])) else 0.0
            size_sc = min(1.0, gap / (a * 1.5)) if a > 0 else 0.5
            body = abs(c[idx] - o[idx])
            rng = h[idx] - l[idx]
            disp_sc = body / rng if rng > 0 else 0.0
            flu_val = fluency_arr[idx]
            flu_sc = min(1.0, max(0.0, flu_val)) if not np.isnan(flu_val) else 0.5
            window = 6
            if idx >= window:
                dirs = np.sign(c[idx - window:idx] - o[idx - window:idx])
                alt = np.sum(dirs[1:] != dirs[:-1]) / (window - 1)
                pa_sc = 1.0 - alt
            else:
                pa_sc = 0.5
            sq_val = (
                sq_params.get("w_size", 0.3) * size_sc
                + sq_params.get("w_disp", 0.3) * disp_sc
                + sq_params.get("w_flu", 0.2) * flu_sc
                + sq_params.get("w_pa", 0.2) * pa_sc
            )
            raw_signal_quality[idx] = sq_val
            # Also populate merged signal_quality for those that pass the gate
            if sig_mask[idx]:
                signal_quality[idx] = sq_val

    # ORM
    is_orm_arr = orm["is_orm_period"].values if "is_orm_period" in orm.columns else np.zeros(n, dtype=bool)

    # Session dates (for daily tracking)
    dates = np.array([
        (et_idx[j] + pd.Timedelta(days=1)).date() if et_idx[j].hour >= 18
        else et_idx[j].date()
        for j in range(n)
    ])

    return {
        "n": n, "et_idx": et_idx,
        "o": o, "h": h, "l": l, "c": c,
        "atr_arr": atr_arr, "fluency_arr": fluency_arr,
        "pa_alt_arr": pa_alt_arr,
        "swing_high_mask": swing_high_mask, "swing_low_mask": swing_low_mask,
        "swings": swings,
        "sig_mask": sig_mask, "sig_dir": sig_dir, "sig_type": sig_type,
        "has_smt_arr": has_smt_arr,
        "entry_price_arr": entry_price_arr, "model_stop_arr": model_stop_arr,
        "irl_target_arr": irl_target_arr,
        "bias_dir_arr": bias_dir_arr, "bias_conf_arr": bias_conf_arr,
        "regime_arr": regime_arr,
        "news_blackout_arr": news_blackout_arr,
        "signal_quality": signal_quality,
        "raw_signal_quality": raw_signal_quality,
        "is_orm_arr": is_orm_arr,
        "dates": dates,
    }


# ============================================================
# 4. COMPUTE SESSION/SUB-SESSION FOR EACH BAR
# ============================================================
def get_session_info(et_ts):
    """Return session and sub_session for a single ET timestamp."""
    h = et_ts.hour
    m = et_ts.minute
    frac = h + m / 60.0

    if frac >= 18.0 or frac < 3.0:
        return "asia", "asia"
    elif 3.0 <= frac < 9.5:
        return "london", "london"
    elif 9.5 <= frac < 12.0:
        return "ny", "ny_am"
    elif 12.0 <= frac < 13.5:
        return "ny", "ny_lunch"
    elif 13.5 <= frac < 16.0:
        return "ny", "ny_pm"
    else:
        return "other", "other"


# ============================================================
# 5. COMPUTE FILTER RESULTS (would this signal be blocked?)
# ============================================================
def compute_filter_results(idx, d, params, sig3, smt_bull_arr, smt_bear_arr):
    """Determine which filters would block a signal at bar index idx.

    Uses the EXACT same filter logic as engine.py / validate_nt_logic.py.
    Returns a dict of booleans: True = blocked by that filter.
    """
    direction = int(sig3["signal_dir"].values[idx])
    entry_p = sig3["entry_price"].values[idx]
    stop = sig3["model_stop"].values[idx]
    tp1 = sig3["irl_target"].values[idx]
    et_ts = d["et_idx"][idx]
    et_frac = et_ts.hour + et_ts.minute / 60.0
    sig_type_val = str(sig3["signal_type"].values[idx])

    result = {}

    # 0. MSS without SMT confirmation (gated before any other filter)
    is_mss = sig_type_val == "mss"
    smt_cfg = params.get("smt", {})
    has_smt_for_signal = False
    if is_mss:
        if direction == 1 and smt_bull_arr[idx]:
            has_smt_for_signal = True
        elif direction == -1 and smt_bear_arr[idx]:
            has_smt_for_signal = True
    # MSS signals without SMT are gated entirely
    blocked_smt_gate = is_mss and smt_cfg.get("require_for_mss", True) and not has_smt_for_signal
    result["blocked_by_smt_gate"] = blocked_smt_gate

    # MSS in overnight (16:00-03:00 ET) also killed
    blocked_mss_overnight = False
    if is_mss and has_smt_for_signal:
        if et_frac >= 16.0 or et_frac < 3.0:
            blocked_mss_overnight = True
    result["blocked_by_mss_overnight"] = blocked_mss_overnight

    # For subsequent filters, use the effective has_smt (after gating)
    effective_has_smt = has_smt_for_signal and not blocked_smt_gate and not blocked_mss_overnight

    # 1. ORM (9:30-10:00 ET observation period)
    result["blocked_by_orm"] = (
        (et_ts.hour == 9 and et_ts.minute >= 30) or
        (et_ts.hour == 10 and et_ts.minute == 0)
    )

    # 2. Session filter
    session_filter = params.get("session_filter", {})
    sf_enabled = session_filter.get("enabled", False)
    is_mss_smt = is_mss and effective_has_smt and smt_cfg.get("enabled", False)
    mss_bypass = is_mss_smt and smt_cfg.get("bypass_session_filter", False)

    blocked_session = False
    if not mss_bypass:
        if sf_enabled:
            if 9.5 <= et_frac < 16.0:
                ad = session_filter.get("ny_direction", 0)
                if ad != 0 and direction != ad:
                    blocked_session = True
            elif 3.0 <= et_frac < 9.5:
                if session_filter.get("skip_london", False):
                    blocked_session = True
                else:
                    ad = session_filter.get("london_direction", 0)
                    if ad != 0 and direction != ad:
                        blocked_session = True
            else:
                if session_filter.get("skip_asia", True):
                    blocked_session = True
        else:
            if not (3.0 <= et_frac < 16.0):
                blocked_session = True
    result["blocked_by_session"] = blocked_session

    # 3. Bias filter
    bias_opposing = (direction == -np.sign(d["bias_dir_arr"][idx]) and d["bias_dir_arr"][idx] != 0)
    blocked_bias = False
    if bias_opposing:
        bias_relax = params.get("bias_relaxation", {})
        if is_mss_smt:
            blocked_bias = False  # MSS+SMT exempt
        elif bias_relax.get("enabled", False) and direction == -1:
            blocked_bias = False  # opposing shorts relaxation
        else:
            blocked_bias = True
    result["blocked_by_bias"] = blocked_bias

    # 4. Signal quality (must recompute for raw signals since d["signal_quality"]
    #    is only populated for merged/gated signals)
    sq_params_cfg = params.get("signal_quality", {})
    sq_enabled = sq_params_cfg.get("enabled", False)
    dual_mode = params.get("dual_mode", {})
    dual_mode_enabled = dual_mode.get("enabled", False)
    blocked_sq = False
    # Use the raw signal quality score computed over ALL raw signals
    sq_val = d.get("raw_signal_quality", d["signal_quality"])[idx]
    if sq_enabled and not np.isnan(sq_val):
        eff_sq = sq_params_cfg.get("threshold", 0.68)
        if dual_mode_enabled and direction == -1:
            eff_sq = dual_mode.get("short_sq_threshold", 0.82)
        if sq_val < eff_sq:
            blocked_sq = True
    result["blocked_by_sq"] = blocked_sq

    # 5. Minimum stop distance
    stop_dist = abs(entry_p - stop) if not (np.isnan(entry_p) or np.isnan(stop)) else 0.0
    min_stop_atr = params.get("regime", {}).get("min_stop_atr_mult", 0.5)
    atr_val = d["atr_arr"][idx] if not np.isnan(d["atr_arr"][idx]) else 10.0
    min_stop = min_stop_atr * atr_val
    result["blocked_by_min_stop"] = stop_dist < min_stop

    # 6. News blackout
    if d["news_blackout_arr"] is not None:
        result["blocked_by_news"] = bool(d["news_blackout_arr"][idx])
    else:
        result["blocked_by_news"] = False

    # 7. Lunch dead zone (session_regime with lunch_mult=0.0)
    session_regime = params.get("session_regime", {})
    blocked_lunch = False
    if session_regime.get("enabled", False):
        lunch_start = session_regime.get("lunch_start", 12.5)
        lunch_end = session_regime.get("lunch_end", 13.0)
        lunch_mult = session_regime.get("lunch_mult", 0.5)
        if lunch_start <= et_frac < lunch_end and lunch_mult == 0.0:
            blocked_lunch = True
    result["blocked_by_lunch"] = blocked_lunch

    # 8. PA quality (alt_dir_threshold)
    pa_threshold = params.get("pa_quality", {}).get("alt_dir_threshold", 1.0)
    result["blocked_by_pa_quality"] = d["pa_alt_arr"][idx] >= pa_threshold

    # 9. Invalid geometry (stop/target wrong side)
    geom_bad = False
    if np.isnan(entry_p) or np.isnan(stop) or np.isnan(tp1):
        geom_bad = True
    elif direction == 1 and (stop >= entry_p or tp1 <= entry_p):
        geom_bad = True
    elif direction == -1 and (stop <= entry_p or tp1 >= entry_p):
        geom_bad = True
    result["blocked_by_geometry"] = geom_bad

    # Passes all = none blocked
    result["passes_all_filters"] = not any(result.values())

    return result


# ============================================================
# 6. TRIPLE BARRIER OUTCOME SIMULATION
# ============================================================
def simulate_outcome(idx, d, sig3, max_bars=60):
    """Simulate what WOULD happen if you entered this signal, regardless of filters.

    Uses actual entry_price (open[i+1]), model_stop, irl_target from signal cache.
    Scans forward from entry bar to find TP hit, SL hit, or timeout.

    Returns dict with outcome features.
    """
    n = d["n"]
    direction = int(sig3["signal_dir"].values[idx])
    entry_p = sig3["entry_price"].values[idx]
    stop = sig3["model_stop"].values[idx]
    target = sig3["irl_target"].values[idx]
    h = d["h"]
    l = d["l"]
    c = d["c"]

    result = {
        "hit_tp": False,
        "hit_sl": False,
        "outcome_r": 0.0,
        "max_favorable_excursion": 0.0,
        "max_adverse_excursion": 0.0,
        "bars_to_outcome": max_bars,
        "outcome_label": -1,  # -1 = timeout
    }

    if np.isnan(entry_p) or np.isnan(stop) or np.isnan(target):
        return result

    stop_dist = abs(entry_p - stop)
    if stop_dist <= 0:
        return result

    entry_bar = idx + 1  # entry on next bar open
    if entry_bar >= n:
        return result

    best_favorable = 0.0
    worst_adverse = 0.0

    for j in range(entry_bar, min(entry_bar + max_bars, n)):
        if direction == 1:
            favorable = h[j] - entry_p
            adverse = entry_p - l[j]

            best_favorable = max(best_favorable, favorable)
            worst_adverse = max(worst_adverse, adverse)

            # Check SL first (conservative: if both hit on same bar, SL wins)
            if l[j] <= stop:
                result["hit_sl"] = True
                result["outcome_r"] = -(stop_dist / stop_dist)  # -1R
                result["bars_to_outcome"] = j - entry_bar + 1
                result["outcome_label"] = 0
                break

            if h[j] >= target:
                result["hit_tp"] = True
                target_dist = target - entry_p
                result["outcome_r"] = target_dist / stop_dist
                result["bars_to_outcome"] = j - entry_bar + 1
                result["outcome_label"] = 1
                break

        else:  # short
            favorable = entry_p - l[j]
            adverse = h[j] - entry_p

            best_favorable = max(best_favorable, favorable)
            worst_adverse = max(worst_adverse, adverse)

            if h[j] >= stop:
                result["hit_sl"] = True
                result["outcome_r"] = -(stop_dist / stop_dist)
                result["bars_to_outcome"] = j - entry_bar + 1
                result["outcome_label"] = 0
                break

            if l[j] <= target:
                result["hit_tp"] = True
                target_dist = entry_p - target
                result["outcome_r"] = target_dist / stop_dist
                result["bars_to_outcome"] = j - entry_bar + 1
                result["outcome_label"] = 1
                break

    # If timed out, compute R from last close
    if not result["hit_tp"] and not result["hit_sl"]:
        last_bar = min(entry_bar + max_bars - 1, n - 1)
        if direction == 1:
            pnl = c[last_bar] - entry_p
        else:
            pnl = entry_p - c[last_bar]
        result["outcome_r"] = pnl / stop_dist if stop_dist > 0 else 0.0

    result["max_favorable_excursion"] = best_favorable / stop_dist if stop_dist > 0 else 0.0
    result["max_adverse_excursion"] = worst_adverse / stop_dist if stop_dist > 0 else 0.0

    return result


# ============================================================
# 7. BUILD THE DATABASE
# ============================================================
def build_signal_database(nq, es, sig3, bias, regime, atr_flu, orm, params):
    """Build the comprehensive signal feature database."""

    ss, smt = compute_smt_and_merge(nq, es, sig3, params)
    d = precompute_arrays(nq, ss, sig3, bias, regime, atr_flu, orm, params)

    # We use ALL signals from the v3 cache (before any SMT gating for MSS)
    # This gives us the full ~15,894 raw signal universe
    raw_sig_mask = sig3["signal"].astype(bool)
    signal_indices = np.where(raw_sig_mask)[0]
    n_signals = len(signal_indices)
    logger.info("Building database for %d raw signals...", n_signals)

    # Pre-compute SMT arrays aligned to NQ index
    smt_aligned = smt.reindex(nq.index).fillna(False)
    smt_bull_arr = smt_aligned["smt_bull"].values.astype(bool) if "smt_bull" in smt_aligned.columns else np.zeros(d["n"], dtype=bool)
    smt_bear_arr = smt_aligned["smt_bear"].values.astype(bool) if "smt_bear" in smt_aligned.columns else np.zeros(d["n"], dtype=bool)

    # ATR percentile (rolling 100-bar)
    atr_series = pd.Series(d["atr_arr"], index=nq.index)
    atr_pct = atr_series.rolling(100, min_periods=20).rank(pct=True)
    atr_pct_arr = atr_pct.values

    # Collect rows
    rows = []
    t0 = _time.perf_counter()

    for count, idx in enumerate(signal_indices):
        if count % 2000 == 0 and count > 0:
            elapsed = _time.perf_counter() - t0
            logger.info("  Processed %d/%d signals (%.1fs)", count, n_signals, elapsed)

        direction = int(sig3["signal_dir"].values[idx])
        if direction == 0:
            continue

        entry_p = sig3["entry_price"].values[idx]
        stop = sig3["model_stop"].values[idx]
        target = sig3["irl_target"].values[idx]
        et_ts = d["et_idx"][idx]

        # ---- Signal Identity ----
        row = {
            "bar_time_utc": nq.index[idx],
            "bar_time_et": et_ts.tz_localize(None) if hasattr(et_ts, 'tz_localize') else et_ts,
            "signal_dir": direction,
            "signal_type": str(sig3["signal_type"].values[idx]),
            "entry_price": entry_p,
            "model_stop": stop,
            "irl_target": target,
        }

        # ---- FVG Features ----
        sig_swept = sig3["swept_liquidity"].values[idx] if "swept_liquidity" in sig3.columns else False
        sig_sweep_score = sig3["sweep_score"].values[idx] if "sweep_score" in sig3.columns else 0

        stop_dist_pts = abs(entry_p - stop) if not (np.isnan(entry_p) or np.isnan(stop)) else np.nan

        # FVG size approximation: use the stop distance as proxy for FVG size
        # (in signal cache, FVG size isn't stored directly, but entry-stop encodes it)
        atr_val = d["atr_arr"][idx] if not np.isnan(d["atr_arr"][idx]) else np.nan
        row["fvg_size_pts"] = stop_dist_pts  # approximate: model stop ~ FVG boundary
        row["fvg_size_atr"] = stop_dist_pts / atr_val if atr_val and atr_val > 0 and not np.isnan(stop_dist_pts) else np.nan
        row["fvg_swept_liquidity"] = bool(sig_swept)
        row["fvg_sweep_score"] = int(sig_sweep_score)

        # ---- Signal Bar Features ----
        o_val = d["o"][idx]
        h_val = d["h"][idx]
        l_val = d["l"][idx]
        c_val = d["c"][idx]
        body = abs(c_val - o_val)
        bar_range = h_val - l_val

        row["bar_body_ratio"] = body / bar_range if bar_range > 0 else 0.0
        row["bar_body_atr"] = body / atr_val if atr_val and atr_val > 0 else np.nan
        row["bar_range_atr"] = bar_range / atr_val if atr_val and atr_val > 0 else np.nan

        # Displacement detection (body > atr_mult * ATR and body_ratio > threshold)
        disp_cfg = params["displacement"]
        row["is_displaced"] = (
            body > disp_cfg["atr_mult"] * atr_val if atr_val and atr_val > 0 else False
        ) and (row["bar_body_ratio"] > disp_cfg["body_ratio"])

        flu_val = d["fluency_arr"][idx]
        row["fluency_score"] = flu_val if not np.isnan(flu_val) else np.nan

        sq_val = d["raw_signal_quality"][idx]
        row["signal_quality"] = sq_val if not np.isnan(sq_val) else np.nan

        # ---- Market Context ----
        row["atr_14"] = atr_val
        row["atr_percentile"] = atr_pct_arr[idx] if not np.isnan(atr_pct_arr[idx]) else np.nan

        session, sub_session = get_session_info(et_ts)
        row["session"] = session
        row["sub_session"] = sub_session
        row["hour_et"] = et_ts.hour
        row["day_of_week"] = et_ts.dayofweek
        row["is_monday"] = et_ts.dayofweek == 0
        row["is_friday"] = et_ts.dayofweek == 4

        # ---- Bias/Regime ----
        row["bias_direction"] = d["bias_dir_arr"][idx]
        row["bias_confidence"] = d["bias_conf_arr"][idx]
        row["bias_aligned"] = (direction == np.sign(d["bias_dir_arr"][idx]) and d["bias_dir_arr"][idx] != 0)
        row["regime"] = d["regime_arr"][idx]

        # Grade computation (same as engine.py)
        ba = 1.0 if row["bias_aligned"] else 0.0
        reg = d["regime_arr"][idx]
        if np.isnan(ba) or np.isnan(reg) or reg == 0.0:
            grade = "C"
        else:
            aligned = ba > 0.5
            full_reg = reg >= 1.0
            if aligned and full_reg:
                grade = "A+"
            elif aligned or full_reg:
                grade = "B+"
            else:
                grade = "C"
        row["grade"] = grade

        # ---- SMT ----
        row["has_smt"] = bool(smt_bull_arr[idx] or smt_bear_arr[idx])
        row["smt_bull"] = bool(smt_bull_arr[idx])
        row["smt_bear"] = bool(smt_bear_arr[idx])

        # ---- Stop/Target Geometry ----
        row["stop_distance_pts"] = stop_dist_pts
        row["stop_distance_atr"] = stop_dist_pts / atr_val if atr_val and atr_val > 0 and not np.isnan(stop_dist_pts) else np.nan

        target_dist_pts = abs(entry_p - target) if not (np.isnan(entry_p) or np.isnan(target)) else np.nan
        row["target_distance_pts"] = target_dist_pts
        row["target_rr"] = target_dist_pts / stop_dist_pts if stop_dist_pts and stop_dist_pts > 0 and not np.isnan(target_dist_pts) else np.nan

        # ---- PA Quality features ----
        row["pa_alt_dir_ratio"] = d["pa_alt_arr"][idx] if not np.isnan(d["pa_alt_arr"][idx]) else np.nan

        # ---- ORM ----
        row["is_orm_period"] = bool(d["is_orm_arr"][idx])

        # ---- Outcome Labels (triple barrier simulation) ----
        outcome = simulate_outcome(idx, d, sig3, max_bars=60)
        row["hit_tp"] = outcome["hit_tp"]
        row["hit_sl"] = outcome["hit_sl"]
        row["outcome_r"] = outcome["outcome_r"]
        row["max_favorable_excursion"] = outcome["max_favorable_excursion"]
        row["max_adverse_excursion"] = outcome["max_adverse_excursion"]
        row["bars_to_outcome"] = outcome["bars_to_outcome"]
        row["outcome_label"] = outcome["outcome_label"]

        # ---- Filter Results ----
        filters = compute_filter_results(idx, d, params, sig3, smt_bull_arr, smt_bear_arr)
        for k, v in filters.items():
            row[k] = v

        rows.append(row)

    elapsed = _time.perf_counter() - t0
    logger.info("Built %d signal rows in %.1fs", len(rows), elapsed)

    df = pd.DataFrame(rows)
    return df


# ============================================================
# 8. PRINT SUMMARY
# ============================================================
def print_summary(df):
    """Print comprehensive summary of the signal database."""
    print("\n" + "=" * 80)
    print("SIGNAL FEATURE DATABASE — SUMMARY")
    print("=" * 80)

    print(f"\nTotal signals: {len(df):,}")
    print(f"Date range: {df['bar_time_utc'].min()} to {df['bar_time_utc'].max()}")

    # By session
    print(f"\nBy session:")
    for s in ["asia", "london", "ny", "other"]:
        n = (df["session"] == s).sum()
        print(f"  {s:>8}: {n:>6,} ({100*n/len(df):.1f}%)")

    # By sub-session
    print(f"\nBy sub-session:")
    for s in ["asia", "london", "ny_am", "ny_lunch", "ny_pm", "other"]:
        n = (df["sub_session"] == s).sum()
        print(f"  {s:>10}: {n:>6,} ({100*n/len(df):.1f}%)")

    # By type
    print(f"\nBy type:")
    for t in df["signal_type"].unique():
        sub = df[df["signal_type"] == t]
        print(f"  {t:>6}: {len(sub):>6,} ({100*len(sub)/len(df):.1f}%)")

    # By direction
    print(f"\nBy direction:")
    for d_val in [1, -1]:
        sub = df[df["signal_dir"] == d_val]
        label = "Long" if d_val == 1 else "Short"
        print(f"  {label:>6}: {len(sub):>6,} ({100*len(sub)/len(df):.1f}%)")

    # By outcome
    print(f"\nBy outcome:")
    tp_pct = (df["outcome_label"] == 1).mean() * 100
    sl_pct = (df["outcome_label"] == 0).mean() * 100
    to_pct = (df["outcome_label"] == -1).mean() * 100
    print(f"  TP hit:   {(df['outcome_label']==1).sum():>6,} ({tp_pct:.1f}%)")
    print(f"  SL hit:   {(df['outcome_label']==0).sum():>6,} ({sl_pct:.1f}%)")
    print(f"  Timeout:  {(df['outcome_label']==-1).sum():>6,} ({to_pct:.1f}%)")

    # Average R
    print(f"\nAverage R (all signals):      {df['outcome_r'].mean():.4f}")
    print(f"Average R (TP+SL only):       {df[df['outcome_label']>=0]['outcome_r'].mean():.4f}")

    # Filtered signals
    passes = df["passes_all_filters"].sum()
    print(f"\nSignals passing all filters:  {passes:>6,} (should be ~534)")

    if passes > 0:
        filtered = df[df["passes_all_filters"]]
        print(f"Average R (filtered):         {filtered['outcome_r'].mean():.4f}")
        print(f"Filtered TP rate:             {(filtered['outcome_label']==1).mean()*100:.1f}%")

    # Filter block counts
    print(f"\nFilter blocking counts:")
    filter_cols = [c for c in df.columns if c.startswith("blocked_by_")]
    for col in sorted(filter_cols):
        n_blocked = df[col].sum()
        print(f"  {col:>30}: {n_blocked:>6,} ({100*n_blocked/len(df):.1f}%)")

    # Feature value ranges
    print(f"\nKey feature value ranges:")
    feature_cols = [
        "fvg_size_pts", "fvg_size_atr", "bar_body_ratio", "bar_body_atr",
        "bar_range_atr", "fluency_score", "signal_quality", "atr_14",
        "atr_percentile", "stop_distance_pts", "stop_distance_atr",
        "target_distance_pts", "target_rr", "pa_alt_dir_ratio",
        "max_favorable_excursion", "max_adverse_excursion", "bars_to_outcome",
    ]
    for col in feature_cols:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                print(f"  {col:>28}: min={s.min():.3f}  p25={s.quantile(0.25):.3f}  "
                      f"med={s.median():.3f}  p75={s.quantile(0.75):.3f}  max={s.max():.3f}")

    print("=" * 80)


# ============================================================
# 9. QUICK FEATURE IMPORTANCE PREVIEW
# ============================================================
def feature_importance_preview(df):
    """For top features, split into 5 quantile bins and show performance per bin."""
    print("\n" + "=" * 80)
    print("QUICK FEATURE IMPORTANCE PREVIEW")
    print("Signals split into 5 quantile bins per feature — shows which features are predictive")
    print("=" * 80)

    features_to_check = [
        "signal_quality", "fluency_score", "bar_body_ratio", "bar_range_atr",
        "stop_distance_atr", "target_rr", "atr_percentile", "fvg_size_atr",
        "pa_alt_dir_ratio", "fvg_sweep_score",
    ]

    # Only use signals where outcome is TP or SL (exclude timeouts for cleaner analysis)
    valid = df[df["outcome_label"] >= 0].copy()
    print(f"\nUsing {len(valid):,} signals with TP/SL outcomes (excluding {(df['outcome_label']==-1).sum():,} timeouts)")

    for feat in features_to_check:
        if feat not in valid.columns:
            continue
        col = valid[feat].dropna()
        if len(col) < 50:
            continue

        print(f"\n  Feature: {feat}")
        print(f"  {'Bin':>12} | {'Count':>6} | {'Avg R':>8} | {'WR (TP%)':>8} | {'PF':>8}")
        print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

        try:
            # Create quintile bins
            mask = valid[feat].notna()
            subset = valid[mask].copy()
            if len(subset) < 50:
                print(f"  (too few non-NaN values: {len(subset)})")
                continue

            subset["bin"] = pd.qcut(subset[feat], q=5, duplicates="drop")
            for bin_label, grp in subset.groupby("bin", observed=True):
                count = len(grp)
                avg_r = grp["outcome_r"].mean()
                wr = (grp["outcome_label"] == 1).mean() * 100
                gross_win = grp.loc[grp["outcome_r"] > 0, "outcome_r"].sum()
                gross_loss = abs(grp.loc[grp["outcome_r"] < 0, "outcome_r"].sum())
                pf = gross_win / gross_loss if gross_loss > 0 else np.inf
                print(f"  {str(bin_label):>12} | {count:>6} | {avg_r:>+8.3f} | {wr:>7.1f}% | {pf:>8.2f}")
        except Exception as e:
            print(f"  (error: {e})")

    # Boolean features
    print(f"\n  Boolean Features:")
    bool_features = ["bias_aligned", "is_displaced", "fvg_swept_liquidity",
                     "has_smt", "is_monday", "is_friday"]
    print(f"  {'Feature':>25} | {'True N':>7} {'True WR':>8} {'True AvgR':>9} | {'False N':>8} {'False WR':>8} {'False AvgR':>9}")
    print(f"  {'-'*25}-+-{'-'*7}-{'-'*8}-{'-'*9}-+-{'-'*8}-{'-'*8}-{'-'*9}")
    for feat in bool_features:
        if feat not in valid.columns:
            continue
        grp_t = valid[valid[feat] == True]
        grp_f = valid[valid[feat] == False]
        if len(grp_t) == 0 or len(grp_f) == 0:
            continue
        t_wr = (grp_t["outcome_label"] == 1).mean() * 100
        f_wr = (grp_f["outcome_label"] == 1).mean() * 100
        t_r = grp_t["outcome_r"].mean()
        f_r = grp_f["outcome_r"].mean()
        print(f"  {feat:>25} | {len(grp_t):>7} {t_wr:>7.1f}% {t_r:>+8.3f} | {len(grp_f):>8} {f_wr:>7.1f}% {f_r:>+8.3f}")

    # By signal type
    print(f"\n  By Signal Type:")
    for st in ["trend", "mss"]:
        grp = valid[valid["signal_type"] == st]
        if len(grp) == 0:
            continue
        wr = (grp["outcome_label"] == 1).mean() * 100
        avg_r = grp["outcome_r"].mean()
        print(f"  {st:>8}: N={len(grp):>6,}, WR={wr:.1f}%, AvgR={avg_r:+.4f}")

    # By grade
    print(f"\n  By Grade:")
    for g in ["A+", "B+", "C"]:
        grp = valid[valid["grade"] == g]
        if len(grp) == 0:
            continue
        wr = (grp["outcome_label"] == 1).mean() * 100
        avg_r = grp["outcome_r"].mean()
        print(f"  {g:>4}: N={len(grp):>6,}, WR={wr:.1f}%, AvgR={avg_r:+.4f}")

    # By session
    print(f"\n  By Session:")
    for s in ["asia", "london", "ny"]:
        grp = valid[valid["session"] == s]
        if len(grp) == 0:
            continue
        wr = (grp["outcome_label"] == 1).mean() * 100
        avg_r = grp["outcome_r"].mean()
        print(f"  {s:>8}: N={len(grp):>6,}, WR={wr:.1f}%, AvgR={avg_r:+.4f}")

    print("\n" + "=" * 80)


# ============================================================
# MAIN
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("BUILD SIGNAL FEATURE DATABASE")
    logger.info("=" * 60)

    t_start = _time.perf_counter()

    # Load data
    nq, es, sig3, bias, regime, atr_flu, orm, params = load_all_data()

    # Build database
    df = build_signal_database(nq, es, sig3, bias, regime, atr_flu, orm, params)

    # Save
    out_path = PROJECT / "data" / "signal_feature_database.parquet"
    df.to_parquet(out_path, engine="pyarrow", index=False)
    logger.info("Saved signal database to %s (%d rows, %d columns)",
                out_path, len(df), len(df.columns))

    # Summary
    print_summary(df)

    # Feature importance preview
    feature_importance_preview(df)

    elapsed = _time.perf_counter() - t_start
    logger.info("Total time: %.1fs", elapsed)

    return df


if __name__ == "__main__":
    main()
