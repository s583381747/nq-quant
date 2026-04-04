"""
experiments/audit_indicator_reconstruction.py — Adversarial Audit of Indicator Reconstruction

Tests whether the "indicator reconstruction" findings are REAL or OVERFIT.

Audit covers:
1. Multiple testing correction (permutation tests)
2. In-sample vs out-of-sample validation (time splits)
3. Year-by-year stability
4. Fair fluency comparison (all from same pipeline)
5. Composite construction bias
6. Sample size / standard errors
7. Outcome distribution economics
8. "Every cutoff" claim re-evaluation on OOS data

Verdict: ROBUST / FRAGILE / OVERFIT for each indicator.
"""

import sys
import time as _time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("audit")

np.random.seed(42)  # Reproducibility

# ════════════════════════════════════════════════════════════════════════
#  HELPERS (replicated from original for self-contained audit)
# ════════════════════════════════════════════════════════════════════════

def compute_metrics(outcomes: np.ndarray) -> dict:
    n = len(outcomes)
    if n == 0:
        return dict(count=0, avg_r=np.nan, wr=np.nan, pf=np.nan,
                    total_r=np.nan, ppdd=np.nan, max_dd=np.nan)
    total_r = float(np.nansum(outcomes))
    avg_r = float(np.nanmean(outcomes))
    wr = float(np.mean(outcomes > 0)) * 100
    wins = outcomes[outcomes > 0]
    losses = outcomes[outcomes < 0]
    sum_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
    sum_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    pf = sum_wins / sum_losses if sum_losses > 0 else (np.inf if sum_wins > 0 else 0.0)
    cum = np.nancumsum(outcomes)
    running_max = np.maximum.accumulate(cum)
    dd = running_max - cum
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    ppdd = total_r / max_dd if max_dd > 0 else (np.inf if total_r > 0 else 0.0)
    return dict(count=n, avg_r=round(avg_r, 4), wr=round(wr, 1),
                pf=round(pf, 3), total_r=round(total_r, 2),
                ppdd=round(ppdd, 3), max_dd=round(max_dd, 2))


def quintile_analysis(outcomes: np.ndarray, feature_arr: np.ndarray,
                      n_bins: int = 5) -> pd.DataFrame:
    """Split signals into quintiles, return per-quintile metrics."""
    valid = ~np.isnan(feature_arr) & ~np.isnan(outcomes)
    vals = feature_arr[valid]
    outs = outcomes[valid]
    if len(vals) < n_bins * 10:
        return pd.DataFrame()
    try:
        bin_labels = pd.qcut(vals, n_bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    rows = []
    for b in sorted(set(bin_labels)):
        mask = bin_labels == b
        m = compute_metrics(outs[mask])
        m["bin_idx"] = b
        m["bin_mean"] = float(np.mean(vals[mask]))
        rows.append(m)
    return pd.DataFrame(rows)


def quintile_spread(qt: pd.DataFrame) -> float:
    if qt.empty or len(qt) < 2:
        return 0.0
    return float(qt["avg_r"].max() - qt["avg_r"].min())


def binary_spread(outcomes: np.ndarray, flag: np.ndarray) -> float:
    """For binary indicators: avg_r(True) - avg_r(False)."""
    flag_bool = flag.astype(bool)
    valid = ~np.isnan(outcomes)
    t_mask = flag_bool & valid
    f_mask = (~flag_bool) & valid
    if t_mask.sum() < 10 or f_mask.sum() < 10:
        return 0.0
    return float(np.mean(outcomes[t_mask]) - np.mean(outcomes[f_mask]))


def print_section(title: str):
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


def print_table(df_table: pd.DataFrame):
    if df_table.empty:
        print("  [No data]")
        return
    print(df_table.to_string(index=False))
    print()


# ════════════════════════════════════════════════════════════════════════
#  LOAD DATA (replicate original pipeline)
# ════════════════════════════════════════════════════════════════════════

print_section("LOADING DATA")
t0 = _time.perf_counter()

db = pd.read_parquet(PROJECT / "data" / "signal_feature_database.parquet")
raw = pd.read_parquet(PROJECT / "data" / "NQ_5m_10yr.parquet")

raw_idx_map = pd.Series(np.arange(len(raw)), index=raw.index)
signal_positions = raw_idx_map.reindex(db["bar_time_utc"]).values.astype(int)

raw_open = raw["open"].values
raw_high = raw["high"].values
raw_low = raw["low"].values
raw_close = raw["close"].values
raw_volume = raw["volume"].values

# ATR(14) Wilder
tr_hl = raw_high - raw_low
tr_hc = np.abs(raw_high - np.roll(raw_close, 1))
tr_lc = np.abs(raw_low - np.roll(raw_close, 1))
tr_hc[0] = tr_hl[0]
tr_lc[0] = tr_hl[0]
true_range = np.maximum(tr_hl, np.maximum(tr_hc, tr_lc))
atr_full = np.full(len(raw), np.nan)
alpha = 1.0 / 14
atr_full[13] = np.mean(true_range[:14])
for i in range(14, len(raw)):
    atr_full[i] = alpha * true_range[i] + (1 - alpha) * atr_full[i - 1]

n_signals = len(db)
outcomes = db["outcome_r"].values
signal_dirs = db["signal_dir"].values

# Year column for splitting
db["year"] = pd.to_datetime(db["bar_time_et"]).dt.year

print(f"Loaded: {n_signals} signals, {len(raw)} raw bars in {_time.perf_counter()-t0:.1f}s")
overall = compute_metrics(outcomes)
print(f"Overall: count={overall['count']}, R={overall['total_r']}, "
      f"PPDD={overall['ppdd']}, PF={overall['pf']}, WR={overall['wr']}%")


# ════════════════════════════════════════════════════════════════════════
#  RECOMPUTE ALL INDICATOR VARIANTS (from original script)
# ════════════════════════════════════════════════════════════════════════

print_section("RECOMPUTING ALL INDICATOR VARIANTS")
t_start = _time.perf_counter()

# ---------- FLUENCY (8 variants) ----------
def compute_fluency_variants(positions, signal_dirs):
    n = len(positions)
    variants = {
        "a_current_w6": np.full(n, np.nan),
        "b_net_direction": np.full(n, np.nan),
        "c_pure_directional": np.full(n, np.nan),
        "d_window_4": np.full(n, np.nan),
        "e_window_10": np.full(n, np.nan),
        "f_inverse": np.full(n, np.nan),
        "g_dir_specific": np.full(n, np.nan),
        "h_range_compress": np.full(n, np.nan),
    }
    for i in range(n):
        pos = positions[i]
        sig_dir = signal_dirs[i]
        if pos < 14:
            continue
        atr_val = atr_full[pos]
        if np.isnan(atr_val) or atr_val == 0:
            continue

        def _fluency_composite(window):
            if pos < window:
                return np.nan
            o_w = raw_open[pos - window + 1:pos + 1]
            h_w = raw_high[pos - window + 1:pos + 1]
            l_w = raw_low[pos - window + 1:pos + 1]
            c_w = raw_close[pos - window + 1:pos + 1]
            dirs = np.sign(c_w - o_w)
            bull = np.sum(dirs == 1)
            bear = np.sum(dirs == -1)
            directional_ratio = max(bull, bear) / window
            bodies = np.abs(c_w - o_w)
            ranges = h_w - l_w
            safe_ranges = np.where(ranges > 0, ranges, np.nan)
            with np.errstate(invalid="ignore"):
                body_ratios = bodies / safe_ranges
            avg_body_ratio = float(np.nanmean(body_ratios))
            with np.errstate(invalid="ignore"):
                bar_size_ratios = np.clip(ranges / atr_val, 0, 2.0)
            avg_bar_size = min(1.0, float(np.nanmean(bar_size_ratios)))
            return 0.4 * directional_ratio + 0.3 * avg_body_ratio + 0.3 * avg_bar_size

        variants["a_current_w6"][i] = _fluency_composite(6)
        if pos >= 6:
            o_w = raw_open[pos - 5:pos + 1]
            c_w = raw_close[pos - 5:pos + 1]
            dirs = np.sign(c_w - o_w)
            variants["b_net_direction"][i] = abs(float(np.sum(dirs))) / 6.0
        if pos >= 6:
            o_w = raw_open[pos - 5:pos + 1]
            c_w = raw_close[pos - 5:pos + 1]
            dirs = np.sign(c_w - o_w)
            bull = np.sum(dirs == 1)
            bear = np.sum(dirs == -1)
            variants["c_pure_directional"][i] = max(bull, bear) / 6.0
        variants["d_window_4"][i] = _fluency_composite(4)
        if pos >= 10:
            variants["e_window_10"][i] = _fluency_composite(10)
        v = _fluency_composite(6)
        if not np.isnan(v):
            variants["f_inverse"][i] = 1.0 - v
        if pos >= 6:
            o_w = raw_open[pos - 5:pos + 1]
            c_w = raw_close[pos - 5:pos + 1]
            dirs = np.sign(c_w - o_w)
            matching = np.sum(dirs == sig_dir)
            variants["g_dir_specific"][i] = matching / 6.0
        if pos >= 6:
            h_w = raw_high[pos - 5:pos + 1]
            l_w = raw_low[pos - 5:pos + 1]
            ranges = h_w - l_w
            max_r = np.max(ranges)
            min_recent = np.min(ranges[-3:]) if len(ranges) >= 3 else np.min(ranges)
            variants["h_range_compress"][i] = min_recent / max_r if max_r > 0 else np.nan
    return variants


# ---------- DISPLACEMENT (6 variants, binary) ----------
def compute_displacement_variants(positions, signal_dirs):
    n = len(positions)
    variants = {
        "a_current_3crit": np.zeros(n, dtype=bool),
        "b_body_only": np.zeros(n, dtype=bool),
        "c_range_only": np.zeros(n, dtype=bool),
        "d_no_engulf": np.zeros(n, dtype=bool),
        "e_directional": np.zeros(n, dtype=bool),
        "f_relative_body": np.zeros(n, dtype=bool),
    }
    atr_mult = 0.8
    body_ratio_min = 0.60
    for i in range(n):
        pos = positions[i]
        sig_dir = signal_dirs[i]
        if pos < 14:
            continue
        atr_val = atr_full[pos]
        if np.isnan(atr_val) or atr_val == 0:
            continue
        o_i = raw_open[pos]
        h_i = raw_high[pos]
        l_i = raw_low[pos]
        c_i = raw_close[pos]
        body = abs(c_i - o_i)
        rng = h_i - l_i
        body_ratio = body / rng if rng > 0 else 0.0
        engulfs = False
        if pos >= 1:
            candle_top = max(o_i, c_i)
            candle_bot = min(o_i, c_i)
            prior_h = raw_high[pos - 1]
            prior_l = raw_low[pos - 1]
            engulfs = (candle_top >= prior_h) and (candle_bot <= prior_l)
        crit_body = body > atr_mult * atr_val
        crit_ratio = body_ratio > body_ratio_min
        candle_dir = 1 if c_i > o_i else (-1 if c_i < o_i else 0)
        dir_match = (candle_dir == sig_dir)
        variants["a_current_3crit"][i] = crit_body and crit_ratio and engulfs
        variants["b_body_only"][i] = crit_body
        variants["c_range_only"][i] = crit_ratio
        variants["d_no_engulf"][i] = crit_body and crit_ratio
        variants["e_directional"][i] = crit_body and crit_ratio and engulfs and dir_match
        if pos >= 10:
            bodies_prior = np.abs(raw_close[pos-10:pos] - raw_open[pos-10:pos])
            avg_body = np.mean(bodies_prior)
            if avg_body > 0:
                variants["f_relative_body"][i] = body / avg_body > 2.0
    return variants


# ---------- FVG QUALITY (7 variants) ----------
def compute_fvg_quality_features(db, positions):
    n = len(db)
    features = {
        "a_size_atr": db["fvg_size_atr"].values.copy(),
        "b_size_pts": db["fvg_size_pts"].values.copy(),
        "c_size_percentile": np.full(n, np.nan),
        "d_fvg_age": np.full(n, np.nan),
        "e_zone_position": np.full(n, np.nan),
        "f_midpoint_dist": np.full(n, np.nan),
        "g_gap_fill_pct": np.full(n, np.nan),
    }
    size_atr = db["fvg_size_atr"].values
    for i in range(n):
        start = max(0, i - 50)
        window = size_atr[start:i+1]
        if len(window) > 1:
            features["c_size_percentile"][i] = np.sum(window <= size_atr[i]) / len(window)
        else:
            features["c_size_percentile"][i] = 0.5
    entry_prices = db["entry_price"].values
    model_stops = db["model_stop"].values
    fvg_sizes = db["fvg_size_pts"].values
    signal_dirs_local = db["signal_dir"].values
    for i in range(n):
        pos = positions[i]
        if pos < 20:
            continue
        atr_val = atr_full[pos]
        if np.isnan(atr_val) or atr_val == 0:
            continue
        entry = entry_prices[i]
        stop = model_stops[i]
        fvg_size = fvg_sizes[i]
        sig_dir = signal_dirs_local[i]
        age = np.nan
        search_limit = min(100, pos)
        for j in range(1, search_limit + 1):
            if abs(raw_open[pos - j] - stop) < 0.01:
                age = j
                break
        features["d_fvg_age"][i] = age
        if pos >= 20:
            ma20 = np.mean(raw_close[pos - 19:pos + 1])
            zone = 1 if entry > ma20 else -1
            features["e_zone_position"][i] = zone * sig_dir
        fvg_mid = (entry + stop) / 2.0
        dist = abs(raw_close[pos] - fvg_mid)
        features["f_midpoint_dist"][i] = dist / atr_val
        if not np.isnan(age) and age > 1 and fvg_size > 0:
            fvg_birth = pos - int(age)
            if sig_dir == 1:
                fvg_top = entry
                fvg_bot = entry - fvg_size
                lows_between = raw_low[fvg_birth:pos]
                if len(lows_between) > 0:
                    deepest = np.min(lows_between)
                    filled = max(0, fvg_top - deepest)
                    features["g_gap_fill_pct"][i] = min(1.0, filled / fvg_size)
            else:
                fvg_bot = entry
                fvg_top = entry + fvg_size
                highs_between = raw_high[fvg_birth:pos]
                if len(highs_between) > 0:
                    deepest = np.max(highs_between)
                    filled = max(0, deepest - fvg_bot)
                    features["g_gap_fill_pct"][i] = min(1.0, filled / fvg_size)
    return features


# ---------- STOP QUALITY (4 variants) ----------
def compute_stop_quality(db, positions):
    n = len(db)
    features = {
        "a_stop_dist_atr": db["stop_distance_atr"].values.copy(),
        "b_stop_tightness": np.full(n, np.nan),
        "c_stop_vs_recent": np.full(n, np.nan),
        "d_stop_atr_band": np.full(n, np.nan),
    }
    entry_prices = db["entry_price"].values
    model_stops = db["model_stop"].values
    fvg_sizes = db["fvg_size_pts"].values
    signal_dirs_local = db["signal_dir"].values
    stop_dist = db["stop_distance_pts"].values
    for i in range(n):
        pos = positions[i]
        if pos < 14:
            continue
        atr_val = atr_full[pos]
        if np.isnan(atr_val) or atr_val == 0:
            continue
        sd = stop_dist[i]
        fvg_size = fvg_sizes[i]
        sig_dir = signal_dirs_local[i]
        stop = model_stops[i]
        if fvg_size > 0:
            features["b_stop_tightness"][i] = sd / fvg_size
        if pos >= 10:
            if sig_dir == 1:
                recent_low = np.min(raw_low[pos - 9:pos + 1])
                features["c_stop_vs_recent"][i] = (recent_low - stop) / atr_val
            else:
                recent_high = np.max(raw_high[pos - 9:pos + 1])
                features["c_stop_vs_recent"][i] = (stop - recent_high) / atr_val
        features["d_stop_atr_band"][i] = sd / atr_val
    return features


# ---------- TARGET QUALITY (4 variants) ----------
def compute_target_quality(db, positions):
    n = len(db)
    features = {
        "a_target_rr": db["target_rr"].values.copy(),
        "b_target_dist_atr": np.full(n, np.nan),
        "c_target_reachability": np.full(n, np.nan),
        "d_target_vs_atr": np.full(n, np.nan),
    }
    target_dist = db["target_distance_pts"].values
    signal_dirs_local = db["signal_dir"].values
    for i in range(n):
        pos = positions[i]
        if pos < 20:
            continue
        atr_val = atr_full[pos]
        if np.isnan(atr_val) or atr_val == 0:
            continue
        td = target_dist[i]
        sig_dir = signal_dirs_local[i]
        features["b_target_dist_atr"][i] = td / atr_val
        features["d_target_vs_atr"][i] = td / atr_val
        if pos >= 20 and td > 0:
            lookback = 20
            start = max(0, pos - lookback)
            if sig_dir == 1:
                highs_after = np.array([
                    np.max(raw_high[j:min(j + 20, pos)]) - raw_close[j]
                    for j in range(start, pos)
                    if j + 1 < pos
                ])
                if len(highs_after) > 0:
                    features["c_target_reachability"][i] = np.mean(highs_after >= td)
            else:
                lows_after = np.array([
                    raw_close[j] - np.min(raw_low[j:min(j + 20, pos)])
                    for j in range(start, pos)
                    if j + 1 < pos
                ])
                if len(lows_after) > 0:
                    features["c_target_reachability"][i] = np.mean(lows_after >= td)
    return features


# ---------- CANDLE QUALITY (7 variants) ----------
def compute_candle_quality(db, positions):
    n = len(db)
    features = {
        "existing_body_ratio": db["bar_body_ratio"].values.copy(),
        "existing_body_atr": db["bar_body_atr"].values.copy(),
        "existing_range_atr": db["bar_range_atr"].values.copy(),
        "a_close_position": np.full(n, np.nan),
        "b_wick_ratio": np.full(n, np.nan),
        "c_body_vs_prior": np.full(n, np.nan),
        "d_volume_spike": np.full(n, np.nan),
    }
    signal_dirs_local = db["signal_dir"].values
    for i in range(n):
        pos = positions[i]
        if pos < 10:
            continue
        o_i = raw_open[pos]
        h_i = raw_high[pos]
        l_i = raw_low[pos]
        c_i = raw_close[pos]
        v_i = raw_volume[pos]
        rng = h_i - l_i
        body = abs(c_i - o_i)
        sig_dir = signal_dirs_local[i]
        if rng > 0:
            if sig_dir == 1:
                features["a_close_position"][i] = (c_i - l_i) / rng
            else:
                features["a_close_position"][i] = (h_i - c_i) / rng
            candle_top = max(o_i, c_i)
            candle_bot = min(o_i, c_i)
            upper_wick = h_i - candle_top
            lower_wick = candle_bot - l_i
            features["b_wick_ratio"][i] = max(upper_wick, lower_wick) / rng
        if pos >= 5:
            prior_bodies = np.abs(raw_close[pos-5:pos] - raw_open[pos-5:pos])
            avg_prior = np.mean(prior_bodies)
            if avg_prior > 0:
                features["c_body_vs_prior"][i] = body / avg_prior
        if pos >= 10:
            prior_vol = raw_volume[pos-10:pos]
            avg_vol = np.mean(prior_vol)
            if avg_vol > 0:
                features["d_volume_spike"][i] = v_i / avg_vol
    return features


# ---------- REGIME (5 variants) ----------
def compute_regime_features(db, positions):
    n = len(db)
    features = {
        "existing_atr_percentile": db["atr_percentile"].values.copy(),
        "a_vol_regime": np.full(n, np.nan),
        "b_trend_strength": np.full(n, np.nan),
        "c_gap_from_close": np.full(n, np.nan),
        "d_time_since_last": np.full(n, np.nan),
    }
    for i in range(n):
        if i == 0:
            features["d_time_since_last"][i] = np.nan
        else:
            features["d_time_since_last"][i] = positions[i] - positions[i - 1]
    for i in range(n):
        pos = positions[i]
        if pos < 100:
            continue
        atr_val = atr_full[pos]
        if np.isnan(atr_val) or atr_val == 0:
            continue
        ranges_20 = raw_high[pos-19:pos+1] - raw_low[pos-19:pos+1]
        ranges_100 = raw_high[pos-99:pos+1] - raw_low[pos-99:pos+1]
        atr_20 = np.mean(ranges_20)
        atr_100 = np.mean(ranges_100)
        if atr_100 > 0:
            features["a_vol_regime"][i] = atr_20 / atr_100
        ret_20 = raw_close[pos] - raw_close[pos - 20]
        range_20 = np.max(raw_high[pos-19:pos+1]) - np.min(raw_low[pos-19:pos+1])
        if range_20 > 0:
            features["b_trend_strength"][i] = ret_20 / range_20
        if pos >= 1:
            gap = abs(raw_open[pos] - raw_close[pos - 1])
            features["c_gap_from_close"][i] = gap / atr_val
    return features


logger.info("Computing all variants...")
t1 = _time.perf_counter()
fluency_variants = compute_fluency_variants(signal_positions, signal_dirs)
disp_variants = compute_displacement_variants(signal_positions, signal_dirs)
fvg_features = compute_fvg_quality_features(db, signal_positions)
stop_features = compute_stop_quality(db, signal_positions)
target_features = compute_target_quality(db, signal_positions)
candle_features = compute_candle_quality(db, signal_positions)
regime_features = compute_regime_features(db, signal_positions)
logger.info("All variants computed in %.1fs", _time.perf_counter() - t1)


# ════════════════════════════════════════════════════════════════════════
#  Identify the "best" variant from each category (as original did)
# ════════════════════════════════════════════════════════════════════════

# Fluency: best by quintile spread
fluency_spreads = {}
for key, arr in fluency_variants.items():
    qt = quintile_analysis(outcomes, arr)
    fluency_spreads[key] = quintile_spread(qt)
best_fluency_key = max(fluency_spreads, key=lambda k: abs(fluency_spreads[k]))

# Displacement: best by binary spread (True avg_r - False avg_r)
disp_spreads = {}
for key, arr in disp_variants.items():
    disp_spreads[key] = binary_spread(outcomes, arr)
best_disp_key = max(disp_spreads, key=lambda k: disp_spreads[k])

# FVG: best by quintile spread
fvg_spreads = {}
for key, arr in fvg_features.items():
    qt = quintile_analysis(outcomes, arr)
    fvg_spreads[key] = quintile_spread(qt)
best_fvg_key = max(fvg_spreads, key=lambda k: abs(fvg_spreads[k]))

# Stop: best by quintile spread
stop_spreads_dict = {}
for key, arr in stop_features.items():
    qt = quintile_analysis(outcomes, arr)
    stop_spreads_dict[key] = quintile_spread(qt)
best_stop_key = max(stop_spreads_dict, key=lambda k: abs(stop_spreads_dict[k]))

# Target: best by quintile spread
target_spreads = {}
for key, arr in target_features.items():
    qt = quintile_analysis(outcomes, arr)
    target_spreads[key] = quintile_spread(qt)
best_target_key = max(target_spreads, key=lambda k: abs(target_spreads[k]))

# Candle: best by quintile spread
candle_spreads = {}
for key, arr in candle_features.items():
    qt = quintile_analysis(outcomes, arr)
    candle_spreads[key] = quintile_spread(qt)
best_candle_key = max(candle_spreads, key=lambda k: abs(candle_spreads[k]))

# Regime: best by quintile spread
regime_spreads = {}
for key, arr in regime_features.items():
    qt = quintile_analysis(outcomes, arr)
    regime_spreads[key] = quintile_spread(qt)
best_regime_key = max(regime_spreads, key=lambda k: abs(regime_spreads[k]))

# Labels for display
category_labels = {
    "fluency": {"key": best_fluency_key, "spread": fluency_spreads[best_fluency_key]},
    "displacement": {"key": best_disp_key, "spread": disp_spreads[best_disp_key]},
    "fvg_quality": {"key": best_fvg_key, "spread": fvg_spreads[best_fvg_key]},
    "stop_quality": {"key": best_stop_key, "spread": stop_spreads_dict[best_stop_key]},
    "target_quality": {"key": best_target_key, "spread": target_spreads[best_target_key]},
    "candle_quality": {"key": best_candle_key, "spread": candle_spreads[best_candle_key]},
    "regime": {"key": best_regime_key, "spread": regime_spreads[best_regime_key]},
}

print("\nBest variants selected (same as original script):")
for cat, info in category_labels.items():
    print(f"  {cat:20s}: {info['key']:30s} spread={info['spread']:+.4f}")


# ════════════════════════════════════════════════════════════════════════
#  Build a lookup of ALL continuous variants and their category
# ════════════════════════════════════════════════════════════════════════

# Combine all continuous variant arrays for audit
all_continuous = {}
for key, arr in fluency_variants.items():
    all_continuous[f"fluency_{key}"] = arr
for key, arr in fvg_features.items():
    all_continuous[f"fvg_{key}"] = arr
for key, arr in stop_features.items():
    all_continuous[f"stop_{key}"] = arr
for key, arr in target_features.items():
    all_continuous[f"target_{key}"] = arr
for key, arr in candle_features.items():
    all_continuous[f"candle_{key}"] = arr
for key, arr in regime_features.items():
    all_continuous[f"regime_{key}"] = arr

# Best continuous features (one per category, excluding displacement which is binary)
best_continuous = {
    "fluency": fluency_variants[best_fluency_key],
    "fvg_quality": fvg_features[best_fvg_key],
    "stop_quality": stop_features[best_stop_key],
    "target_quality": target_features[best_target_key],
    "candle_quality": candle_features[best_candle_key],
    "regime": regime_features[best_regime_key],
}

# Also the best displacement (binary)
best_disp_arr = disp_variants[best_disp_key]


# ════════════════════════════════════════════════════════════════════════
#  AUDIT 1: MULTIPLE TESTING — Permutation Tests
# ════════════════════════════════════════════════════════════════════════

print_section("AUDIT 1: MULTIPLE TESTING — Permutation Tests (1000 shuffles)")

N_PERM = 1000


def permutation_test_quintile(feature_arr: np.ndarray, outcome_arr: np.ndarray,
                               n_perm: int = N_PERM) -> tuple:
    """Permutation test for quintile spread.
    Returns (observed_spread, p_value, null_distribution)."""
    observed = quintile_spread(quintile_analysis(outcome_arr, feature_arr))
    null_spreads = np.zeros(n_perm)
    for p in range(n_perm):
        shuffled = outcome_arr.copy()
        np.random.shuffle(shuffled)
        qt = quintile_analysis(shuffled, feature_arr)
        null_spreads[p] = quintile_spread(qt)
    p_value = np.mean(null_spreads >= observed)
    return observed, p_value, null_spreads


def permutation_test_binary(flag_arr: np.ndarray, outcome_arr: np.ndarray,
                             n_perm: int = N_PERM) -> tuple:
    """Permutation test for binary spread (True avg_r - False avg_r)."""
    observed = binary_spread(outcome_arr, flag_arr)
    null_spreads = np.zeros(n_perm)
    for p in range(n_perm):
        shuffled = outcome_arr.copy()
        np.random.shuffle(shuffled)
        null_spreads[p] = binary_spread(shuffled, flag_arr)
    p_value = np.mean(null_spreads >= observed)
    return observed, p_value, null_spreads


perm_results = {}

# Test each best continuous variant
for cat_name, arr in best_continuous.items():
    logger.info("Permutation test: %s (%s)...", cat_name, category_labels[cat_name]["key"])
    obs, pval, null = permutation_test_quintile(arr, outcomes)
    perm_results[cat_name] = {"observed": obs, "p_value": pval,
                               "null_mean": np.mean(null), "null_std": np.std(null),
                               "null_95": np.percentile(null, 95)}
    print(f"  {cat_name:20s}: observed spread={obs:.4f}, p={pval:.4f}, "
          f"null_mean={np.mean(null):.4f}, null_95pct={np.percentile(null, 95):.4f}")

# Displacement (binary)
logger.info("Permutation test: displacement (%s)...", best_disp_key)
obs_d, pval_d, null_d = permutation_test_binary(best_disp_arr, outcomes)
perm_results["displacement"] = {"observed": obs_d, "p_value": pval_d,
                                 "null_mean": np.mean(null_d), "null_std": np.std(null_d),
                                 "null_95": np.percentile(null_d, 95)}
print(f"  {'displacement':20s}: observed spread={obs_d:.4f}, p={pval_d:.4f}, "
      f"null_mean={np.mean(null_d):.4f}, null_95pct={np.percentile(null_d, 95):.4f}")

# Also test ALL 37 variants to show the multiple testing problem
print("\n--- ALL 37 variants permutation tests (showing how many are 'significant') ---")
n_significant_005 = 0
n_significant_010 = 0
n_total_tested = 0

# Test all continuous variants (31 continuous)
for full_key, arr in all_continuous.items():
    valid = ~np.isnan(arr)
    if valid.sum() < 100:
        continue
    obs = quintile_spread(quintile_analysis(outcomes, arr))
    if obs == 0:
        continue
    # Quick permutation (100 shuffles for speed)
    null = np.zeros(100)
    for p in range(100):
        shuffled = outcomes.copy()
        np.random.shuffle(shuffled)
        null[p] = quintile_spread(quintile_analysis(shuffled, arr))
    pval = np.mean(null >= obs)
    n_total_tested += 1
    if pval < 0.05:
        n_significant_005 += 1
    if pval < 0.10:
        n_significant_010 += 1

# Test all displacement variants (6 binary)
for key, arr in disp_variants.items():
    obs = binary_spread(outcomes, arr)
    null = np.zeros(100)
    for p in range(100):
        shuffled = outcomes.copy()
        np.random.shuffle(shuffled)
        null[p] = binary_spread(shuffled, arr)
    pval = np.mean(null >= obs)
    n_total_tested += 1
    if pval < 0.05:
        n_significant_005 += 1
    if pval < 0.10:
        n_significant_010 += 1

print(f"  Total variants tested: {n_total_tested}")
print(f"  Significant at p<0.05: {n_significant_005} (expected by chance: {n_total_tested * 0.05:.1f})")
print(f"  Significant at p<0.10: {n_significant_010} (expected by chance: {n_total_tested * 0.10:.1f})")

# Bonferroni correction
print("\n--- Bonferroni-corrected p-values for BEST variants ---")
n_tests = n_total_tested
for cat_name, res in perm_results.items():
    bonf_p = min(1.0, res["p_value"] * n_tests)
    sig = "SIGNIFICANT" if bonf_p < 0.05 else "NOT SIGNIFICANT"
    print(f"  {cat_name:20s}: raw p={res['p_value']:.4f}, Bonferroni p={bonf_p:.4f} => {sig}")


# ════════════════════════════════════════════════════════════════════════
#  AUDIT 2: IN-SAMPLE vs OUT-OF-SAMPLE
# ════════════════════════════════════════════════════════════════════════

print_section("AUDIT 2: IN-SAMPLE vs OUT-OF-SAMPLE")

years = db["year"].values


def compute_spread_on_subset(mask, feature_arr, is_binary=False, outcome_arr=None):
    """Compute quintile spread or binary spread on a subset."""
    if outcome_arr is None:
        outcome_arr = outcomes
    sub_out = outcome_arr[mask]
    sub_feat = feature_arr[mask]
    if is_binary:
        return binary_spread(sub_out, sub_feat)
    else:
        qt = quintile_analysis(sub_out, sub_feat)
        return quintile_spread(qt)


# Split 1: 2016-2022 train, 2023-2026 test
train_mask_1 = (years >= 2016) & (years <= 2022)
test_mask_1 = (years >= 2023)

# Split 2: odd years train, even years test
train_mask_2 = np.isin(years, [2015, 2017, 2019, 2021, 2023, 2025])
test_mask_2 = np.isin(years, [2016, 2018, 2020, 2022, 2024, 2026])

print(f"Split 1: TRAIN 2016-2022 ({train_mask_1.sum()} signals), TEST 2023-2026 ({test_mask_1.sum()} signals)")
print(f"Split 2: TRAIN odd years ({train_mask_2.sum()} signals), TEST even years ({test_mask_2.sum()} signals)")

oos_results = {}

for cat_name, arr in best_continuous.items():
    is_bin = False
    train_sp1 = compute_spread_on_subset(train_mask_1, arr, is_bin)
    test_sp1 = compute_spread_on_subset(test_mask_1, arr, is_bin)
    train_sp2 = compute_spread_on_subset(train_mask_2, arr, is_bin)
    test_sp2 = compute_spread_on_subset(test_mask_2, arr, is_bin)

    oos_results[cat_name] = {
        "train_sp1": train_sp1, "test_sp1": test_sp1,
        "train_sp2": train_sp2, "test_sp2": test_sp2,
    }

# Displacement
train_sp1_d = compute_spread_on_subset(train_mask_1, best_disp_arr, True)
test_sp1_d = compute_spread_on_subset(test_mask_1, best_disp_arr, True)
train_sp2_d = compute_spread_on_subset(train_mask_2, best_disp_arr, True)
test_sp2_d = compute_spread_on_subset(test_mask_2, best_disp_arr, True)
oos_results["displacement"] = {
    "train_sp1": train_sp1_d, "test_sp1": test_sp1_d,
    "train_sp2": train_sp2_d, "test_sp2": test_sp2_d,
}

print("\n--- Split 1: Train (2016-2022) vs Test (2023-2026) ---")
rows = []
for cat_name, res in oos_results.items():
    degradation = 0 if res["train_sp1"] == 0 else (1 - res["test_sp1"] / res["train_sp1"]) * 100
    survives = "YES" if res["test_sp1"] > 0 and np.sign(res["train_sp1"]) == np.sign(res["test_sp1"]) else "NO"
    rows.append({
        "category": cat_name,
        "train_spread": round(res["train_sp1"], 4),
        "test_spread": round(res["test_sp1"], 4),
        "degradation_%": round(degradation, 1),
        "survives_OOS": survives,
    })
print_table(pd.DataFrame(rows))

print("\n--- Split 2: Odd years vs Even years ---")
rows = []
for cat_name, res in oos_results.items():
    degradation = 0 if res["train_sp2"] == 0 else (1 - res["test_sp2"] / res["train_sp2"]) * 100
    survives = "YES" if res["test_sp2"] > 0 and np.sign(res["train_sp2"]) == np.sign(res["test_sp2"]) else "NO"
    rows.append({
        "category": cat_name,
        "train_spread": round(res["train_sp2"], 4),
        "test_spread": round(res["test_sp2"], 4),
        "degradation_%": round(degradation, 1),
        "survives_OOS": survives,
    })
print_table(pd.DataFrame(rows))


# ════════════════════════════════════════════════════════════════════════
#  AUDIT 2b: COMPOSITE OOS TEST
# ════════════════════════════════════════════════════════════════════════

print_section("AUDIT 2b: COMPOSITE SCORE — Train/Test Split")


def build_composite(feature_dict, outcome_arr, binary_disp=None):
    """Build composite score from feature dict using percentile rank normalization.
    Returns normalized composite array (same length as outcome_arr)."""
    from scipy.stats import rankdata, spearmanr

    n = len(outcome_arr)
    normalized = {}
    directions = {}

    for name, arr in feature_dict.items():
        valid = ~np.isnan(arr)
        if valid.sum() < 100:
            continue
        ranked = np.full(n, np.nan)
        vals = arr[valid]
        ranks = rankdata(vals, method="average") / len(vals)
        ranked[valid] = ranks
        normalized[name] = ranked

        corr, _ = spearmanr(arr[valid], outcome_arr[valid])
        directions[name] = 1 if corr >= 0 else -1

    composite = np.zeros(n)
    valid_count = np.zeros(n)
    for name, arr in normalized.items():
        direction = directions[name]
        valid = ~np.isnan(arr)
        composite[valid] += arr[valid] * direction
        valid_count[valid] += 1

    composite = np.where(valid_count > 0, composite / valid_count, np.nan)
    return composite


# Build composite on FULL sample (as original did)
composite_full = build_composite(best_continuous, outcomes)
current_sq = db["signal_quality"].values

# Build on TRAIN only, evaluate on TEST
# Split 1
train_continuous_1 = {k: v[train_mask_1] for k, v in best_continuous.items()}
composite_train_only_1 = build_composite(train_continuous_1, outcomes[train_mask_1])

# For test: normalize using TRAIN percentiles (proper OOS)
def normalize_oos(train_arr, test_arr, train_outcomes, test_outcomes):
    """Normalize test data using train statistics, build composite."""
    from scipy.stats import rankdata, spearmanr

    n_test = len(test_outcomes)
    composites = np.zeros(n_test)
    valid_counts = np.zeros(n_test)

    for name in train_arr:
        tr = train_arr[name]
        te = test_arr[name]

        valid_tr = ~np.isnan(tr)
        valid_te = ~np.isnan(te)

        if valid_tr.sum() < 100 or valid_te.sum() < 50:
            continue

        # Determine direction from train
        corr, _ = spearmanr(tr[valid_tr], train_outcomes[valid_tr])
        direction = 1 if corr >= 0 else -1

        # Percentile rank test values within train distribution
        tr_vals = tr[valid_tr]
        te_vals = te[valid_te]

        # For each test value, compute its percentile in train
        ranked_test = np.full(n_test, np.nan)
        for j in range(n_test):
            if np.isnan(te[j]):
                continue
            ranked_test[j] = np.mean(tr_vals <= te[j])

        valid = ~np.isnan(ranked_test)
        composites[valid] += ranked_test[valid] * direction
        valid_counts[valid] += 1

    return np.where(valid_counts > 0, composites / valid_counts, np.nan)


test_continuous_1 = {k: v[test_mask_1] for k, v in best_continuous.items()}
composite_test_oos_1 = normalize_oos(
    {k: v[train_mask_1] for k, v in best_continuous.items()},
    {k: v[test_mask_1] for k, v in best_continuous.items()},
    outcomes[train_mask_1],
    outcomes[test_mask_1],
)

print("--- Composite on Train (2016-2022) ---")
qt_train = quintile_analysis(outcomes[train_mask_1], composite_train_only_1)
print_table(qt_train)
print(f"  Train spread: {quintile_spread(qt_train):.4f}")

print("--- Composite on Test (2023-2026, normalized using train) ---")
qt_test = quintile_analysis(outcomes[test_mask_1], composite_test_oos_1)
print_table(qt_test)
print(f"  Test spread: {quintile_spread(qt_test):.4f}")

print("--- Current SQ on Train ---")
qt_sq_train = quintile_analysis(outcomes[train_mask_1], current_sq[train_mask_1])
print(f"  SQ Train spread: {quintile_spread(qt_sq_train):.4f}")

print("--- Current SQ on Test ---")
qt_sq_test = quintile_analysis(outcomes[test_mask_1], current_sq[test_mask_1])
print(f"  SQ Test spread: {quintile_spread(qt_sq_test):.4f}")


# ════════════════════════════════════════════════════════════════════════
#  AUDIT 3: YEAR-BY-YEAR STABILITY
# ════════════════════════════════════════════════════════════════════════

print_section("AUDIT 3: YEAR-BY-YEAR STABILITY")

unique_years = sorted([y for y in db["year"].unique() if y >= 2016 and y <= 2025])  # 2015 too small, 2026 partial
print(f"Testing stability across {len(unique_years)} years: {unique_years}")

stability_results = {}

for cat_name in list(best_continuous.keys()) + ["displacement"]:
    is_binary = (cat_name == "displacement")
    arr = best_disp_arr if is_binary else best_continuous[cat_name]

    year_data = []
    for yr in unique_years:
        yr_mask = years == yr
        n_yr = yr_mask.sum()
        if n_yr < 50:
            continue

        if is_binary:
            spread = binary_spread(outcomes[yr_mask], arr[yr_mask])
            # For binary: just record spread sign
            year_data.append({"year": yr, "n": n_yr, "spread": spread,
                              "best_group": "True" if spread > 0 else "False"})
        else:
            qt = quintile_analysis(outcomes[yr_mask], arr[yr_mask])
            if qt.empty:
                continue
            spread = quintile_spread(qt)
            best_q = qt.loc[qt["avg_r"].idxmax(), "bin_idx"] if "bin_idx" in qt.columns else -1
            year_data.append({"year": yr, "n": n_yr, "spread": spread, "best_q": int(best_q)})

    if not year_data:
        continue

    df_yr = pd.DataFrame(year_data)

    # Check consistency
    if is_binary:
        n_positive = sum(1 for d in year_data if d["spread"] > 0)
        n_total = len(year_data)
        flip_rate = 1 - (n_positive / n_total) if n_positive > n_total / 2 else n_positive / n_total
    else:
        # Check: what fraction of years have the SAME best quintile as overall?
        overall_qt = quintile_analysis(outcomes, arr)
        if not overall_qt.empty and "bin_idx" in overall_qt.columns:
            overall_best = int(overall_qt.loc[overall_qt["avg_r"].idxmax(), "bin_idx"])
        else:
            overall_best = -1

        matching_years = sum(1 for d in year_data if d.get("best_q") == overall_best)
        n_total = len(year_data)
        flip_rate = 1 - matching_years / n_total if n_total > 0 else 1.0

    stable = "STABLE" if flip_rate < 0.30 else "UNSTABLE"
    stability_results[cat_name] = {"flip_rate": flip_rate, "verdict": stable, "data": df_yr}

    print(f"\n--- {cat_name} (best variant: {category_labels[cat_name]['key']}) ---")
    print_table(df_yr)
    print(f"  Flip rate: {flip_rate:.1%} => {stable}")
    if not is_binary and "best_q" in df_yr.columns:
        print(f"  Best quintile distribution across years: {dict(df_yr['best_q'].value_counts())}")


# ════════════════════════════════════════════════════════════════════════
#  AUDIT 4: FLUENCY PARADOX RE-EXAMINATION
# ════════════════════════════════════════════════════════════════════════

print_section("AUDIT 4: FLUENCY PARADOX — Fair Comparison")

print("All 8 fluency variants were computed FRESH from raw OHLCV data in this audit.")
print("The database's 'fluency_score' column may come from a different formula/cache.")
print()
print("Checking: is db.fluency_score correlated with any recomputed variant?")

db_fluency = db["fluency_score"].values
for key, arr in fluency_variants.items():
    valid = ~np.isnan(arr) & ~np.isnan(db_fluency)
    if valid.sum() > 100:
        corr, pval = stats.spearmanr(arr[valid], db_fluency[valid])
        print(f"  {key:25s} vs db.fluency_score: rho={corr:+.4f}, p={pval:.2e}")

print()
print("--- Fair ranking of ALL fluency variants (all from same pipeline) ---")
print("(This addresses the concern that dir_specific used fresh data vs stale cache)")
print()
fair_fluency_rows = []
for key, arr in fluency_variants.items():
    qt = quintile_analysis(outcomes, arr)
    spread = quintile_spread(qt)
    fair_fluency_rows.append({"variant": key, "spread": round(spread, 4)})

fair_fluency_df = pd.DataFrame(fair_fluency_rows).sort_values("spread", ascending=False)
print_table(fair_fluency_df)

# Also test: does db.fluency_score predict as well?
qt_db = quintile_analysis(outcomes, db_fluency)
spread_db = quintile_spread(qt_db)
print(f"  Database fluency_score spread: {spread_db:.4f}")
print(f"  Best recomputed variant spread: {fair_fluency_df.iloc[0]['spread']:.4f}")
print(f"  -> The comparison is fair since all variants use the same raw OHLCV pipeline.")


# ════════════════════════════════════════════════════════════════════════
#  AUDIT 5: COMPOSITE CONSTRUCTION BIAS
# ════════════════════════════════════════════════════════════════════════

print_section("AUDIT 5: COMPOSITE CONSTRUCTION BIAS")

# The composite was built by picking the BEST variant per category, then combining.
# Test: what if we randomly pick 6 variants instead?

print("--- Testing: random composite vs best-pick composite ---")
print(f"  Best-pick composite spread (full sample): "
      f"{quintile_spread(quintile_analysis(outcomes, composite_full)):.4f}")

# Gather all variant arrays (continuous only)
all_variant_list = list(all_continuous.values())
all_variant_names = list(all_continuous.keys())

# Random composites: pick 6 random variants, build composite, measure spread
N_RANDOM = 200
random_spreads = np.zeros(N_RANDOM)

for r in range(N_RANDOM):
    idx = np.random.choice(len(all_variant_list), size=6, replace=False)
    random_features = {all_variant_names[i]: all_variant_list[i] for i in idx}
    comp = build_composite(random_features, outcomes)
    qt = quintile_analysis(outcomes, comp)
    random_spreads[r] = quintile_spread(qt)

observed_composite_spread = quintile_spread(quintile_analysis(outcomes, composite_full))
p_random = np.mean(random_spreads >= observed_composite_spread)

print(f"  Random composite spread: mean={np.mean(random_spreads):.4f}, "
      f"std={np.std(random_spreads):.4f}, "
      f"95pct={np.percentile(random_spreads, 95):.4f}")
print(f"  Best-pick composite spread: {observed_composite_spread:.4f}")
print(f"  P(random >= best-pick): {p_random:.4f}")
print(f"  -> {'BEST-PICK IS SPECIAL' if p_random < 0.05 else 'BEST-PICK IS NOT SPECIAL (selection bias!)'}")

# Also: composite built on train, evaluated on test (proper OOS)
print()
print("--- Composite OOS: built on 2016-2022, evaluated on 2023-2026 ---")

# Already computed above - just report
train_spread = quintile_spread(qt_train)
test_spread = quintile_spread(qt_test)
sq_train_spread = quintile_spread(qt_sq_train)
sq_test_spread = quintile_spread(qt_sq_test)

print(f"  New Composite: train={train_spread:.4f}, test={test_spread:.4f}, "
      f"degradation={100*(1 - test_spread/train_spread) if train_spread != 0 else 0:.1f}%")
print(f"  Current SQ:    train={sq_train_spread:.4f}, test={sq_test_spread:.4f}, "
      f"degradation={100*(1 - sq_test_spread/sq_train_spread) if sq_train_spread != 0 else 0:.1f}%")
print(f"  -> Composite beats SQ on test? {'YES' if test_spread > sq_test_spread else 'NO'}")


# ════════════════════════════════════════════════════════════════════════
#  AUDIT 6: SAMPLE SIZE / STANDARD ERRORS
# ════════════════════════════════════════════════════════════════════════

print_section("AUDIT 6: SAMPLE SIZE & STATISTICAL SIGNIFICANCE")

print("--- Standard errors per quintile for each best indicator ---")

for cat_name, arr in best_continuous.items():
    valid = ~np.isnan(arr) & ~np.isnan(outcomes)
    vals = arr[valid]
    outs = outcomes[valid]

    if len(vals) < 50:
        continue

    try:
        bin_labels = pd.qcut(vals, 5, labels=False, duplicates="drop")
    except ValueError:
        continue

    print(f"\n  {cat_name} ({category_labels[cat_name]['key']}):")
    q_means = []
    q_ses = []
    q_ns = []
    for b in sorted(set(bin_labels)):
        mask = bin_labels == b
        q_out = outs[mask]
        n_q = len(q_out)
        mean_q = np.mean(q_out)
        se_q = np.std(q_out, ddof=1) / np.sqrt(n_q)
        q_means.append(mean_q)
        q_ses.append(se_q)
        q_ns.append(n_q)
        print(f"    Q{b+1}: n={n_q}, avg_r={mean_q:+.4f}, SE={se_q:.4f}, "
              f"95%CI=[{mean_q - 1.96*se_q:+.4f}, {mean_q + 1.96*se_q:+.4f}]")

    # T-test: Q5 vs Q1 (or best vs worst)
    if len(q_means) >= 2:
        best_idx = np.argmax(q_means)
        worst_idx = np.argmin(q_means)

        # Extract actual quintile outcomes for t-test
        best_q_out = outs[bin_labels == sorted(set(bin_labels))[best_idx]]
        worst_q_out = outs[bin_labels == sorted(set(bin_labels))[worst_idx]]

        t_stat, t_pval = stats.ttest_ind(best_q_out, worst_q_out)
        print(f"    T-test (best Q{best_idx+1} vs worst Q{worst_idx+1}): "
              f"t={t_stat:.3f}, p={t_pval:.4f}, "
              f"{'SIGNIFICANT' if t_pval < 0.05 else 'NOT SIGNIFICANT'}")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(best_q_out, ddof=1) + np.var(worst_q_out, ddof=1)) / 2)
        cohens_d = (np.mean(best_q_out) - np.mean(worst_q_out)) / pooled_std if pooled_std > 0 else 0
        effect = "negligible" if abs(cohens_d) < 0.2 else (
            "small" if abs(cohens_d) < 0.5 else (
                "medium" if abs(cohens_d) < 0.8 else "large"))
        print(f"    Cohen's d={cohens_d:.4f} ({effect} effect)")


# ════════════════════════════════════════════════════════════════════════
#  AUDIT 7: OUTCOME DISTRIBUTION & ECONOMIC SIGNIFICANCE
# ════════════════════════════════════════════════════════════════════════

print_section("AUDIT 7: OUTCOME DISTRIBUTION & ECONOMIC SIGNIFICANCE")

print("--- Outcome distribution ---")
print(f"  Full stop (-1R): {(outcomes == -1).sum()} ({(outcomes == -1).mean()*100:.1f}%)")
print(f"  Winners (>0R):   {(outcomes > 0).sum()} ({(outcomes > 0).mean()*100:.1f}%)")
print(f"  Flat (0R):       {(outcomes == 0).sum()} ({(outcomes == 0).mean()*100:.1f}%)")
print(f"  Other losses:    {((outcomes < 0) & (outcomes != -1)).sum()}")
print(f"  Mean: {np.mean(outcomes):.4f}, Median: {np.median(outcomes):.4f}")
print(f"  Std:  {np.std(outcomes):.4f}")

print("\n--- Win rate sensitivity ---")
print("  A quintile with avg_r difference of 0.05 implies:")
wr_base = (outcomes > 0).mean()
mean_winner = np.mean(outcomes[outcomes > 0])
mean_loser = np.mean(outcomes[outcomes < 0])
print(f"  Base WR={wr_base:.3f}, mean winner={mean_winner:.3f}R, mean loser={mean_loser:.3f}R")
print(f"  avg_r = WR * mean_winner + (1-WR) * mean_loser")
print(f"  To shift avg_r by +0.05: need WR change of ~{0.05 / (mean_winner - mean_loser):.3f}")
print(f"  That's about {0.05 / (mean_winner - mean_loser) * 100:.1f} percentage points of WR")

print("\n--- Commission/slippage impact ---")
# From CLAUDE.md: commission = $2.05/side, slippage = 1 tick = $5
# Per trade round-trip: 2 * $2.05 + 2 * $5 = $14.10
# With 1R = $1000: commission+slippage = $14.10 / $1000 = 0.0141 R per trade
comm_slip_r = 14.10 / 1000.0
print(f"  Commission + slippage per round-trip: ~${14.10:.2f} = {comm_slip_r:.4f}R")
print(f"  For a quintile spread of 0.05R to be meaningful after costs:")
print(f"  The spread must exceed 2 * {comm_slip_r:.4f} = {2*comm_slip_r:.4f}R (to cover both sides)")
print(f"  Most quintile spreads we see: check if they exceed this threshold")

for cat_name in list(best_continuous.keys()) + ["displacement"]:
    spread_val = category_labels[cat_name]["spread"]
    meaningful = "ECONOMICALLY MEANINGFUL" if abs(spread_val) > 2 * comm_slip_r else "BELOW COST THRESHOLD"
    print(f"    {cat_name:20s}: spread={spread_val:+.4f} => {meaningful}")


# ════════════════════════════════════════════════════════════════════════
#  AUDIT 8: "EVERY CUTOFF" CLAIM — OOS Re-evaluation
# ════════════════════════════════════════════════════════════════════════

print_section('AUDIT 8: "EVERY CUTOFF" CLAIM — OOS (2023-2026) Re-evaluation')

# Recompute composite and SQ on test data only
# We need composite_test_oos_1 from Audit 2b
test_outcomes = outcomes[test_mask_1]
test_sq = current_sq[test_mask_1]

print("--- Comparing New Composite vs Current SQ at various cutoffs (2023-2026 ONLY) ---")

cutoff_rows = []
for pct_keep in [50, 40, 30, 20, 10]:
    # Composite
    valid_comp = ~np.isnan(composite_test_oos_1)
    if valid_comp.sum() > 0:
        threshold_comp = np.nanpercentile(composite_test_oos_1[valid_comp], 100 - pct_keep)
        mask_comp = (composite_test_oos_1 >= threshold_comp) & valid_comp
        m_comp = compute_metrics(test_outcomes[mask_comp])
    else:
        m_comp = compute_metrics(np.array([]))

    # SQ
    valid_sq = ~np.isnan(test_sq)
    if valid_sq.sum() > 0:
        threshold_sq = np.nanpercentile(test_sq[valid_sq], 100 - pct_keep)
        mask_sq = (test_sq >= threshold_sq) & valid_sq
        m_sq = compute_metrics(test_outcomes[mask_sq])
    else:
        m_sq = compute_metrics(np.array([]))

    comp_wins = "COMPOSITE" if m_comp["avg_r"] > m_sq["avg_r"] else "SQ"
    cutoff_rows.append({
        "cutoff": f"top {pct_keep}%",
        "comp_n": m_comp["count"],
        "comp_avg_r": m_comp["avg_r"],
        "comp_ppdd": m_comp["ppdd"],
        "sq_n": m_sq["count"],
        "sq_avg_r": m_sq["avg_r"],
        "sq_ppdd": m_sq["ppdd"],
        "winner": comp_wins,
    })

print_table(pd.DataFrame(cutoff_rows))

n_comp_wins = sum(1 for r in cutoff_rows if r["winner"] == "COMPOSITE")
print(f"  Composite wins at {n_comp_wins}/{len(cutoff_rows)} cutoffs on OOS data (2023-2026)")
print(f"  Original claim: 'beats at every cutoff' on full sample")
if n_comp_wins < len(cutoff_rows):
    print(f"  => CLAIM DOES NOT HOLD OOS. Composite fails at "
          f"{len(cutoff_rows) - n_comp_wins}/{len(cutoff_rows)} cutoffs.")
else:
    print(f"  => Claim holds OOS.")


# ════════════════════════════════════════════════════════════════════════
#  FINAL VERDICTS
# ════════════════════════════════════════════════════════════════════════

print_section("FINAL VERDICTS")

def verdict(cat_name):
    """Combine evidence to produce ROBUST / FRAGILE / OVERFIT verdict."""
    scores = {
        "permutation": 0,
        "oos": 0,
        "stability": 0,
        "effect_size": 0,
        "economic": 0,
    }

    # Permutation
    if cat_name in perm_results:
        p = perm_results[cat_name]["p_value"]
        bonf_p = min(1.0, p * n_total_tested)
        if bonf_p < 0.05:
            scores["permutation"] = 2  # Strong
        elif p < 0.05:
            scores["permutation"] = 1  # Marginal
        else:
            scores["permutation"] = 0  # Fail

    # OOS
    if cat_name in oos_results:
        res = oos_results[cat_name]
        # Both splits should show same sign AND >50% of in-sample spread
        split1_ok = (res["test_sp1"] > 0) and (res["train_sp1"] > 0) and (res["test_sp1"] > 0.3 * res["train_sp1"])
        split2_ok = (res["test_sp2"] > 0) and (res["train_sp2"] > 0) and (res["test_sp2"] > 0.3 * res["train_sp2"])
        if split1_ok and split2_ok:
            scores["oos"] = 2
        elif split1_ok or split2_ok:
            scores["oos"] = 1
        else:
            scores["oos"] = 0

    # Stability
    if cat_name in stability_results:
        fr = stability_results[cat_name]["flip_rate"]
        if fr < 0.20:
            scores["stability"] = 2
        elif fr < 0.40:
            scores["stability"] = 1
        else:
            scores["stability"] = 0

    # Economic significance
    spread_val = abs(category_labels[cat_name]["spread"])
    if spread_val > 0.10:
        scores["economic"] = 2
    elif spread_val > 2 * comm_slip_r:
        scores["economic"] = 1
    else:
        scores["economic"] = 0

    total = sum(scores.values())
    max_possible = 8  # 4 categories * 2

    if total >= 6:
        v = "ROBUST"
    elif total >= 3:
        v = "FRAGILE"
    else:
        v = "OVERFIT"

    return v, scores, total


verdict_rows = []
for cat_name in list(best_continuous.keys()) + ["displacement"]:
    v, scores, total = verdict(cat_name)
    verdict_rows.append({
        "category": cat_name,
        "variant": category_labels[cat_name]["key"],
        "spread": round(category_labels[cat_name]["spread"], 4),
        "perm_p": round(perm_results.get(cat_name, {}).get("p_value", 1.0), 4),
        "OOS_survives": oos_results.get(cat_name, {}).get("test_sp1", 0) > 0,
        "stability": stability_results.get(cat_name, {}).get("verdict", "?"),
        "score": f"{total}/8",
        "VERDICT": v,
    })

print_table(pd.DataFrame(verdict_rows))

# ---- Overall composite verdict ----
print("\n--- COMPOSITE SCORE VERDICT ---")
comp_train = quintile_spread(qt_train) if not qt_train.empty else 0
comp_test = quintile_spread(qt_test) if not qt_test.empty else 0
comp_full = quintile_spread(quintile_analysis(outcomes, composite_full))
sq_full = quintile_spread(quintile_analysis(outcomes, current_sq))

print(f"  Full-sample:  composite={comp_full:.4f}, SQ={sq_full:.4f}")
print(f"  Train (16-22): composite={comp_train:.4f}")
print(f"  Test (23-26):  composite={comp_test:.4f}")
print(f"  Degradation:   {100*(1 - comp_test/comp_train) if comp_train != 0 else 0:.1f}%")
print(f"  Random composite mean: {np.mean(random_spreads):.4f}")

if comp_test > sq_test_spread and comp_test > 0.5 * comp_train:
    print(f"  => COMPOSITE VERDICT: FRAGILE-TO-ROBUST (holds OOS but with degradation)")
elif comp_test > 0:
    print(f"  => COMPOSITE VERDICT: FRAGILE (some OOS signal but weak)")
else:
    print(f"  => COMPOSITE VERDICT: OVERFIT (no OOS signal)")

# ---- Summary of key claims ----
print_section("CLAIM-BY-CLAIM ASSESSMENT")

claims = [
    ("Direction-specific fluency is 2.3x better",
     "Compare g_dir_specific spread to a_current_w6 spread"),
    ("Simplest displacement wins (body-only)",
     "Check if b_body_only has best spread among displacement variants"),
    ("FVG size percentile beats absolute size",
     "Compare c_size_percentile to a_size_atr"),
    ("Trend strength is a new useful feature",
     "Check regime b_trend_strength significance"),
    ("New composite beats SQ at every cutoff",
     "Check OOS cutoff comparison"),
]

for claim, method in claims:
    print(f"\n  CLAIM: '{claim}'")
    print(f"  METHOD: {method}")

# Claim 1: direction-specific fluency
spread_dir = fluency_spreads.get("g_dir_specific", 0)
spread_curr = fluency_spreads.get("a_current_w6", 0)
ratio = spread_dir / spread_curr if spread_curr != 0 else float("inf")
# OOS check
train_dir = compute_spread_on_subset(train_mask_1, fluency_variants["g_dir_specific"])
test_dir = compute_spread_on_subset(test_mask_1, fluency_variants["g_dir_specific"])
train_curr = compute_spread_on_subset(train_mask_1, fluency_variants["a_current_w6"])
test_curr = compute_spread_on_subset(test_mask_1, fluency_variants["a_current_w6"])
print(f"    Full sample: dir_specific={spread_dir:.4f}, current={spread_curr:.4f}, ratio={ratio:.2f}x")
print(f"    OOS (23-26): dir_specific={test_dir:.4f}, current={test_curr:.4f}")
if test_dir > test_curr:
    print(f"    => HOLDS OOS")
else:
    print(f"    => DOES NOT HOLD OOS (dir_specific underperforms current on test)")

# Claim 2: simplest displacement
best_disp_spread = disp_spreads[best_disp_key]
body_only_spread = disp_spreads.get("b_body_only", 0)
print(f"\n    Best displacement: {best_disp_key} (spread={best_disp_spread:.4f})")
print(f"    Body-only: spread={body_only_spread:.4f}")
if best_disp_key == "b_body_only":
    # OOS
    train_body = compute_spread_on_subset(train_mask_1, disp_variants["b_body_only"], True)
    test_body = compute_spread_on_subset(test_mask_1, disp_variants["b_body_only"], True)
    print(f"    OOS: train={train_body:.4f}, test={test_body:.4f}")
    print(f"    => {'HOLDS OOS' if test_body > 0 else 'DOES NOT HOLD OOS'}")
else:
    print(f"    => Body-only is NOT the best (claim may be wrong)")

# Claim 3: FVG size percentile vs absolute
pctile_spread = fvg_spreads.get("c_size_percentile", 0)
atr_spread = fvg_spreads.get("a_size_atr", 0)
print(f"\n    FVG percentile: {pctile_spread:.4f}, FVG ATR: {atr_spread:.4f}")
train_pctile = compute_spread_on_subset(train_mask_1, fvg_features["c_size_percentile"])
test_pctile = compute_spread_on_subset(test_mask_1, fvg_features["c_size_percentile"])
train_atr_fvg = compute_spread_on_subset(train_mask_1, fvg_features["a_size_atr"])
test_atr_fvg = compute_spread_on_subset(test_mask_1, fvg_features["a_size_atr"])
print(f"    OOS: percentile test={test_pctile:.4f}, ATR test={test_atr_fvg:.4f}")
if test_pctile > test_atr_fvg:
    print(f"    => HOLDS OOS")
else:
    print(f"    => DOES NOT HOLD OOS")

# Claim 4: trend strength
trend_spread = regime_spreads.get("b_trend_strength", 0)
trend_pval = perm_results.get("regime", {}).get("p_value", 1.0)
train_trend = compute_spread_on_subset(train_mask_1, regime_features["b_trend_strength"])
test_trend = compute_spread_on_subset(test_mask_1, regime_features["b_trend_strength"])
print(f"\n    Trend strength: full spread={trend_spread:.4f}, perm p={trend_pval:.4f}")
print(f"    OOS: train={train_trend:.4f}, test={test_trend:.4f}")
if trend_pval < 0.05 and test_trend > 0:
    print(f"    => LIKELY REAL")
elif test_trend > 0:
    print(f"    => WEAK SIGNAL (not significant but positive OOS)")
else:
    print(f"    => LIKELY NOISE")

# Claim 5: every cutoff
print(f"\n    Already evaluated in Audit 8: {n_comp_wins}/{len(cutoff_rows)} cutoffs held OOS")
if n_comp_wins == len(cutoff_rows):
    print(f"    => HOLDS OOS")
else:
    print(f"    => DOES NOT HOLD OOS")


print("\n" + "=" * 100)
print("  AUDIT COMPLETE")
print("=" * 100)
