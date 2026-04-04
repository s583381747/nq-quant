"""
experiments/indicator_reconstruction.py — Indicator Reconstruction from First Principles

Tests MULTIPLE construction methods for every concept (fluency, displacement, FVG quality,
stop quality, target quality, candle quality, market regime) and finds which version
best predicts trade outcomes.

Database: data/signal_feature_database.parquet (15,894 signals x 58 features)
Raw data: data/NQ_5m_10yr.parquet (711,141 bars)

Every comparison table shows: R, PPDD, PF.
"""

import sys
import time as _time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("indicator_reconstruction")


# ════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════

def compute_metrics(outcomes: np.ndarray) -> dict:
    """Compute standard trading metrics from an array of R outcomes."""
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


def quintile_analysis(db: pd.DataFrame, feature_arr: np.ndarray,
                      feature_name: str, n_bins: int = 5) -> pd.DataFrame:
    """Split signals into quintiles by feature_arr, compute metrics per bin."""
    valid = ~np.isnan(feature_arr)
    vals = feature_arr[valid]
    outcomes = db["outcome_r"].values[valid]

    if len(vals) < n_bins * 10:
        return pd.DataFrame()

    try:
        bin_labels = pd.qcut(vals, n_bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    edges = pd.qcut(vals, n_bins, duplicates="drop", retbins=True)[1]

    rows = []
    for b in sorted(set(bin_labels)):
        mask = bin_labels == b
        m = compute_metrics(outcomes[mask])
        lo = edges[b] if b < len(edges) else np.nan
        hi = edges[b + 1] if b + 1 < len(edges) else np.nan
        m["bin"] = f"Q{b+1} [{lo:.3f},{hi:.3f})"
        m["bin_idx"] = b
        m["bin_mean"] = float(np.mean(vals[mask]))
        rows.append(m)

    df = pd.DataFrame(rows)
    cols = ["bin", "count", "avg_r", "wr", "pf", "total_r", "ppdd", "max_dd", "bin_mean"]
    return df[[c for c in cols if c in df.columns]]


def quintile_spread(qt: pd.DataFrame) -> float:
    """Compute spread: best quintile avg_r - worst quintile avg_r."""
    if qt.empty or len(qt) < 2:
        return 0.0
    return float(qt["avg_r"].max() - qt["avg_r"].min())


def binary_analysis(db: pd.DataFrame, flag_arr: np.ndarray,
                    feature_name: str) -> pd.DataFrame:
    """Split on True/False and compute metrics."""
    valid = ~np.isnan(flag_arr.astype(float))
    outcomes = db["outcome_r"].values

    rows = []
    for val, label in [(True, "True"), (False, "False")]:
        mask = flag_arr.astype(bool) == (val) & valid
        if mask.sum() > 0:
            m = compute_metrics(outcomes[mask])
            m["bin"] = f"{feature_name}={label}"
            m["pct"] = round(100.0 * mask.sum() / valid.sum(), 1)
            rows.append(m)

    df = pd.DataFrame(rows)
    cols = ["bin", "count", "pct", "avg_r", "wr", "pf", "total_r", "ppdd", "max_dd"]
    return df[[c for c in cols if c in df.columns]]


def print_section(title: str):
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


def print_table(df_table: pd.DataFrame):
    """Pretty-print a DataFrame as an aligned table."""
    if df_table.empty:
        print("  [No data]")
        return
    print(df_table.to_string(index=False))
    print()


def fmt_spread(spread: float) -> str:
    return f"{spread:+.4f}"


# ════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════════════

print_section("LOADING DATA")
t0 = _time.perf_counter()

db = pd.read_parquet(PROJECT / "data" / "signal_feature_database.parquet")
raw = pd.read_parquet(PROJECT / "data" / "NQ_5m_10yr.parquet")

# Build integer index mapping: signal bar_time_utc -> raw integer position
raw_idx_map = pd.Series(np.arange(len(raw)), index=raw.index)
signal_positions = raw_idx_map.reindex(db["bar_time_utc"]).values.astype(int)

# Pre-extract raw arrays for speed
raw_open = raw["open"].values
raw_high = raw["high"].values
raw_low = raw["low"].values
raw_close = raw["close"].values
raw_volume = raw["volume"].values

# Pre-compute ATR for full raw data (Wilder smoothing, period=14)
tr_hl = raw_high - raw_low
tr_hc = np.abs(raw_high - np.roll(raw_close, 1))
tr_lc = np.abs(raw_low - np.roll(raw_close, 1))
tr_hc[0] = tr_hl[0]
tr_lc[0] = tr_hl[0]
true_range = np.maximum(tr_hl, np.maximum(tr_hc, tr_lc))

# Wilder EMA for ATR
atr_full = np.full(len(raw), np.nan)
alpha = 1.0 / 14
atr_full[13] = np.mean(true_range[:14])
for i in range(14, len(raw)):
    atr_full[i] = alpha * true_range[i] + (1 - alpha) * atr_full[i - 1]

n_signals = len(db)
outcomes = db["outcome_r"].values
signal_dirs = db["signal_dir"].values

print(f"Loaded: {n_signals} signals, {len(raw)} raw bars in {_time.perf_counter()-t0:.1f}s")
overall = compute_metrics(outcomes)
print(f"Overall: count={overall['count']}, R={overall['total_r']}, "
      f"PPDD={overall['ppdd']}, PF={overall['pf']}, WR={overall['wr']}%")


# ════════════════════════════════════════════════════════════════════════
#  HELPER: extract lookback data for each signal
# ════════════════════════════════════════════════════════════════════════

def get_lookback(pos: int, n_bars: int) -> tuple:
    """Return (open, high, low, close, volume) arrays for n_bars BEFORE pos (inclusive of pos)."""
    start = max(0, pos - n_bars + 1)
    return (raw_open[start:pos+1], raw_high[start:pos+1],
            raw_low[start:pos+1], raw_close[start:pos+1],
            raw_volume[start:pos+1])


# ════════════════════════════════════════════════════════════════════════
#  1. FLUENCY — 8 ALTERNATIVE CONSTRUCTIONS
# ════════════════════════════════════════════════════════════════════════

print_section("1. FLUENCY — 8 Alternative Constructions")

def compute_fluency_variants(positions: np.ndarray, signal_dirs: np.ndarray) -> dict:
    """Compute all fluency variants for each signal position."""
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

        # Need at least 10 bars lookback for longest window
        if pos < 14:
            continue

        atr_val = atr_full[pos]
        if np.isnan(atr_val) or atr_val == 0:
            continue

        # ---- Helper: compute fluency for a given window ----
        def _fluency_composite(window: int):
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

        # (a) Current formula, window=6
        variants["a_current_w6"][i] = _fluency_composite(6)

        # (b) Net direction: abs(sum(signs))/window (old formula)
        if pos >= 6:
            o_w = raw_open[pos - 5:pos + 1]
            c_w = raw_close[pos - 5:pos + 1]
            dirs = np.sign(c_w - o_w)
            variants["b_net_direction"][i] = abs(float(np.sum(dirs))) / 6.0

        # (c) Pure directional: just max(bull,bear)/window
        if pos >= 6:
            o_w = raw_open[pos - 5:pos + 1]
            c_w = raw_close[pos - 5:pos + 1]
            dirs = np.sign(c_w - o_w)
            bull = np.sum(dirs == 1)
            bear = np.sum(dirs == -1)
            variants["c_pure_directional"][i] = max(bull, bear) / 6.0

        # (d) Current formula, window=4
        variants["d_window_4"][i] = _fluency_composite(4)

        # (e) Current formula, window=10
        if pos >= 10:
            variants["e_window_10"][i] = _fluency_composite(10)

        # (f) Inverse fluency
        v = _fluency_composite(6)
        if not np.isnan(v):
            variants["f_inverse"][i] = 1.0 - v

        # (g) Direction-specific: count only candles matching signal direction
        if pos >= 6:
            o_w = raw_open[pos - 5:pos + 1]
            c_w = raw_close[pos - 5:pos + 1]
            dirs = np.sign(c_w - o_w)
            matching = np.sum(dirs == sig_dir)
            variants["g_dir_specific"][i] = matching / 6.0

        # (h) Range compression: min(ranges[-3:])/max(ranges[-6:])
        if pos >= 6:
            h_w = raw_high[pos - 5:pos + 1]
            l_w = raw_low[pos - 5:pos + 1]
            ranges = h_w - l_w
            max_r = np.max(ranges)
            min_recent = np.min(ranges[-3:]) if len(ranges) >= 3 else np.min(ranges)
            variants["h_range_compress"][i] = min_recent / max_r if max_r > 0 else np.nan

    return variants


logger.info("Computing fluency variants for %d signals...", n_signals)
t1 = _time.perf_counter()
fluency_variants = compute_fluency_variants(signal_positions, signal_dirs)
logger.info("Fluency variants computed in %.1fs", _time.perf_counter() - t1)

fluency_labels = {
    "a_current_w6": "Current (w=6, composite)",
    "b_net_direction": "Net direction (old formula)",
    "c_pure_directional": "Pure directional only",
    "d_window_4": "Current formula (w=4)",
    "e_window_10": "Current formula (w=10)",
    "f_inverse": "Inverse fluency (1 - current)",
    "g_dir_specific": "Direction-specific count",
    "h_range_compress": "Range compression",
}

fluency_spreads = {}
for key, label in fluency_labels.items():
    arr = fluency_variants[key]
    valid_count = np.sum(~np.isnan(arr))
    print(f"\n--- {label} (valid={valid_count}) ---")
    qt = quintile_analysis(db, arr, key)
    print_table(qt)
    spread = quintile_spread(qt)
    fluency_spreads[key] = spread
    print(f"  Quintile spread: {fmt_spread(spread)}")

# Rank by spread
print("\n--- FLUENCY RANKING (by quintile spread) ---")
ranked = sorted(fluency_spreads.items(), key=lambda x: abs(x[1]), reverse=True)
rank_rows = []
for rank, (key, spread) in enumerate(ranked, 1):
    rank_rows.append({"rank": rank, "variant": fluency_labels[key], "spread": round(spread, 4)})
print_table(pd.DataFrame(rank_rows))

best_fluency_key = ranked[0][0]
print(f">>> BEST FLUENCY: {fluency_labels[best_fluency_key]} (spread={ranked[0][1]:.4f})")


# ════════════════════════════════════════════════════════════════════════
#  2. DISPLACEMENT — 6 ALTERNATIVE CONSTRUCTIONS
# ════════════════════════════════════════════════════════════════════════

print_section("2. DISPLACEMENT — 6 Alternative Constructions")

def compute_displacement_variants(positions: np.ndarray, signal_dirs: np.ndarray) -> dict:
    """Compute all displacement variants for each signal position."""
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

        # Engulf check: does body span prior candle's range?
        engulfs = False
        if pos >= 1:
            candle_top = max(o_i, c_i)
            candle_bot = min(o_i, c_i)
            prior_h = raw_high[pos - 1]
            prior_l = raw_low[pos - 1]
            engulfs = (candle_top >= prior_h) and (candle_bot <= prior_l)

        crit_body = body > atr_mult * atr_val
        crit_ratio = body_ratio > body_ratio_min
        crit_engulf = engulfs

        # Candle direction matches signal direction?
        candle_dir = 1 if c_i > o_i else (-1 if c_i < o_i else 0)
        dir_match = (candle_dir == sig_dir)

        # (a) Current: all 3 criteria
        variants["a_current_3crit"][i] = crit_body and crit_ratio and crit_engulf

        # (b) Body-only
        variants["b_body_only"][i] = crit_body

        # (c) Range-only (body/range check)
        variants["c_range_only"][i] = crit_ratio

        # (d) No engulf
        variants["d_no_engulf"][i] = crit_body and crit_ratio

        # (e) Directional: current + direction match
        variants["e_directional"][i] = crit_body and crit_ratio and crit_engulf and dir_match

        # (f) Relative body: body / avg(body[-10:]) > 2.0
        if pos >= 10:
            bodies_prior = np.abs(raw_close[pos-10:pos] - raw_open[pos-10:pos])
            avg_body = np.mean(bodies_prior)
            if avg_body > 0:
                variants["f_relative_body"][i] = body / avg_body > 2.0

    return variants


logger.info("Computing displacement variants for %d signals...", n_signals)
t1 = _time.perf_counter()
disp_variants = compute_displacement_variants(signal_positions, signal_dirs)
logger.info("Displacement variants computed in %.1fs", _time.perf_counter() - t1)

disp_labels = {
    "a_current_3crit": "Current (body+ratio+engulf)",
    "b_body_only": "Body > 0.8*ATR only",
    "c_range_only": "Body/range > 0.60 only",
    "d_no_engulf": "Body+ratio (no engulf)",
    "e_directional": "Current + direction match",
    "f_relative_body": "Relative body > 2x avg(10)",
}

disp_results = {}
for key, label in disp_labels.items():
    arr = disp_variants[key]
    pct_true = 100.0 * arr.sum() / n_signals
    outcomes_true = outcomes[arr]
    outcomes_false = outcomes[~arr]

    m_true = compute_metrics(outcomes_true)
    m_false = compute_metrics(outcomes_false)

    print(f"\n--- {label} ({pct_true:.1f}% True) ---")
    rows = [
        {"group": "displaced=True", "count": m_true["count"], "avg_r": m_true["avg_r"],
         "wr": m_true["wr"], "pf": m_true["pf"], "total_r": m_true["total_r"],
         "ppdd": m_true["ppdd"]},
        {"group": "displaced=False", "count": m_false["count"], "avg_r": m_false["avg_r"],
         "wr": m_false["wr"], "pf": m_false["pf"], "total_r": m_false["total_r"],
         "ppdd": m_false["ppdd"]},
    ]
    print_table(pd.DataFrame(rows))

    spread = m_true["avg_r"] - m_false["avg_r"] if not np.isnan(m_true["avg_r"]) and not np.isnan(m_false["avg_r"]) else 0.0
    disp_results[key] = {"spread": spread, "pct": pct_true, "true_avg_r": m_true["avg_r"],
                         "false_avg_r": m_false["avg_r"], "true_pf": m_true["pf"],
                         "false_pf": m_false["pf"]}
    print(f"  Spread (True avg_r - False avg_r): {fmt_spread(spread)}")

# Rank
print("\n--- DISPLACEMENT RANKING ---")
disp_ranked = sorted(disp_results.items(), key=lambda x: x[1]["spread"], reverse=True)
rank_rows = []
for rank, (key, res) in enumerate(disp_ranked, 1):
    rank_rows.append({"rank": rank, "variant": disp_labels[key],
                      "spread": round(res["spread"], 4), "pct_true": round(res["pct"], 1),
                      "true_pf": res["true_pf"], "false_pf": res["false_pf"]})
print_table(pd.DataFrame(rank_rows))

best_disp_key = disp_ranked[0][0]
print(f">>> BEST DISPLACEMENT: {disp_labels[best_disp_key]} (spread={disp_ranked[0][1]['spread']:.4f})")


# ════════════════════════════════════════════════════════════════════════
#  3. FVG QUALITY — 7 METRICS
# ════════════════════════════════════════════════════════════════════════

print_section("3. FVG QUALITY — 7 Metrics")

def compute_fvg_quality_features(db: pd.DataFrame, positions: np.ndarray) -> dict:
    """Compute FVG quality features."""
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

    # (c) Size percentile: relative to rolling 50-signal percentile
    # Use a simple approach: rank within nearby signals
    size_atr = db["fvg_size_atr"].values
    for i in range(n):
        # Look at last 50 signals (or all prior)
        start = max(0, i - 50)
        window = size_atr[start:i+1]
        if len(window) > 1:
            features["c_size_percentile"][i] = np.sum(window <= size_atr[i]) / len(window)
        else:
            features["c_size_percentile"][i] = 0.5

    # Per-signal features requiring raw data
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

        # (d) FVG age: approximate by looking backwards for the FVG formation bar
        # The FVG is formed when model_stop is the open of the displacement candle.
        # We search backwards from signal bar for the bar whose open == model_stop.
        # Limit search to 100 bars.
        age = np.nan
        search_limit = min(100, pos)
        for j in range(1, search_limit + 1):
            if abs(raw_open[pos - j] - stop) < 0.01:
                age = j
                break
        features["d_fvg_age"][i] = age

        # (e) Zone position: is FVG above or below 20-bar MA?
        if pos >= 20:
            ma20 = np.mean(raw_close[pos - 19:pos + 1])
            fvg_mid = entry  # entry is close to FVG zone
            # +1 if FVG above MA (bullish zone), -1 if below
            zone = 1 if fvg_mid > ma20 else -1
            # Multiply by signal direction: aligned = positive, opposed = negative
            features["e_zone_position"][i] = zone * sig_dir

        # (f) Midpoint distance: distance from price to FVG mid / ATR
        # FVG midpoint = midpoint between entry and stop (approximately)
        fvg_mid = (entry + stop) / 2.0
        dist = abs(raw_close[pos] - fvg_mid)
        features["f_midpoint_dist"][i] = dist / atr_val

        # (g) Gap fill %: how much of FVG has been filled by price after creation?
        # Look at bars between FVG birth and signal bar
        if not np.isnan(age) and age > 1 and fvg_size > 0:
            fvg_birth = pos - int(age)
            if sig_dir == 1:
                # Bullish FVG: gap between candle1 high and candle3 low
                fvg_top = entry
                fvg_bot = entry - fvg_size
                # How deep did price penetrate into the FVG?
                lows_between = raw_low[fvg_birth:pos]
                if len(lows_between) > 0:
                    deepest_penetration = np.min(lows_between)
                    filled = max(0, fvg_top - deepest_penetration)
                    features["g_gap_fill_pct"][i] = min(1.0, filled / fvg_size)
            else:
                # Bearish FVG: gap between candle3 high and candle1 low
                fvg_bot = entry
                fvg_top = entry + fvg_size
                highs_between = raw_high[fvg_birth:pos]
                if len(highs_between) > 0:
                    deepest_penetration = np.max(highs_between)
                    filled = max(0, deepest_penetration - fvg_bot)
                    features["g_gap_fill_pct"][i] = min(1.0, filled / fvg_size)

    return features


logger.info("Computing FVG quality features...")
t1 = _time.perf_counter()
fvg_features = compute_fvg_quality_features(db, signal_positions)
logger.info("FVG quality computed in %.1fs", _time.perf_counter() - t1)

fvg_labels = {
    "a_size_atr": "FVG size (ATR units)",
    "b_size_pts": "FVG size (points)",
    "c_size_percentile": "FVG size percentile (rolling 50)",
    "d_fvg_age": "FVG age (bars since birth)",
    "e_zone_position": "FVG zone (trend-relative)",
    "f_midpoint_dist": "Distance to FVG midpoint / ATR",
    "g_gap_fill_pct": "FVG gap fill %",
}

fvg_spreads = {}
for key, label in fvg_labels.items():
    arr = fvg_features[key]
    valid = np.sum(~np.isnan(arr))
    print(f"\n--- {label} (valid={valid}) ---")
    qt = quintile_analysis(db, arr, key)
    print_table(qt)
    spread = quintile_spread(qt)
    fvg_spreads[key] = spread
    print(f"  Quintile spread: {fmt_spread(spread)}")

print("\n--- FVG QUALITY RANKING ---")
fvg_ranked = sorted(fvg_spreads.items(), key=lambda x: abs(x[1]), reverse=True)
rank_rows = []
for rank, (key, spread) in enumerate(fvg_ranked, 1):
    rank_rows.append({"rank": rank, "variant": fvg_labels[key], "spread": round(spread, 4)})
print_table(pd.DataFrame(rank_rows))

best_fvg_key = fvg_ranked[0][0]
print(f">>> BEST FVG QUALITY: {fvg_labels[best_fvg_key]} (spread={fvg_ranked[0][1]:.4f})")


# ════════════════════════════════════════════════════════════════════════
#  4. STOP QUALITY — 4 METRICS
# ════════════════════════════════════════════════════════════════════════

print_section("4. STOP QUALITY — 4 Metrics")

def compute_stop_quality(db: pd.DataFrame, positions: np.ndarray) -> dict:
    """Compute stop quality features."""
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

        entry = entry_prices[i]
        stop = model_stops[i]
        fvg_size = fvg_sizes[i]
        sig_dir = signal_dirs_local[i]
        sd = stop_dist[i]

        # (b) Stop tightness: stop_distance / fvg_size
        if fvg_size > 0:
            features["b_stop_tightness"][i] = sd / fvg_size

        # (c) Stop vs recent low/high
        if pos >= 10:
            if sig_dir == 1:
                # Long: how far is stop below recent 10-bar low?
                recent_low = np.min(raw_low[pos - 9:pos + 1])
                features["c_stop_vs_recent"][i] = (recent_low - stop) / atr_val
                # Positive = stop is safely below recent low
                # Negative = stop is ABOVE recent low (danger!)
            else:
                # Short: how far is stop above recent 10-bar high?
                recent_high = np.max(raw_high[pos - 9:pos + 1])
                features["c_stop_vs_recent"][i] = (stop - recent_high) / atr_val
                # Positive = stop is safely above recent high
                # Negative = stop is BELOW recent high (danger!)

        # (d) Stop ATR band: stop_distance / ATR
        features["d_stop_atr_band"][i] = sd / atr_val

    return features


logger.info("Computing stop quality features...")
t1 = _time.perf_counter()
stop_features = compute_stop_quality(db, signal_positions)
logger.info("Stop quality computed in %.1fs", _time.perf_counter() - t1)

stop_labels = {
    "a_stop_dist_atr": "Stop distance (ATR units)",
    "b_stop_tightness": "Stop tightness (stop/FVG size)",
    "c_stop_vs_recent": "Stop vs recent extreme (ATR)",
    "d_stop_atr_band": "Stop ATR band (stop_dist/ATR)",
}

stop_spreads = {}
for key, label in stop_labels.items():
    arr = stop_features[key]
    valid = np.sum(~np.isnan(arr))
    print(f"\n--- {label} (valid={valid}) ---")
    qt = quintile_analysis(db, arr, key)
    print_table(qt)
    spread = quintile_spread(qt)
    stop_spreads[key] = spread
    print(f"  Quintile spread: {fmt_spread(spread)}")

print("\n--- STOP QUALITY RANKING ---")
stop_ranked = sorted(stop_spreads.items(), key=lambda x: abs(x[1]), reverse=True)
rank_rows = []
for rank, (key, spread) in enumerate(stop_ranked, 1):
    rank_rows.append({"rank": rank, "variant": stop_labels[key], "spread": round(spread, 4)})
print_table(pd.DataFrame(rank_rows))

best_stop_key = stop_ranked[0][0]
print(f">>> BEST STOP QUALITY: {stop_labels[best_stop_key]} (spread={stop_ranked[0][1]:.4f})")


# ════════════════════════════════════════════════════════════════════════
#  5. TARGET QUALITY — 4 METRICS
# ════════════════════════════════════════════════════════════════════════

print_section("5. TARGET QUALITY — 4 Metrics")

def compute_target_quality(db: pd.DataFrame, positions: np.ndarray) -> dict:
    """Compute target quality features."""
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

        # (b) Absolute target distance in ATR
        features["b_target_dist_atr"][i] = td / atr_val

        # (d) Target vs ATR (same as b, kept for naming clarity)
        features["d_target_vs_atr"][i] = td / atr_val

        # (c) Target reachability: in last 20 bars, what fraction reached target_dist?
        if pos >= 20 and td > 0:
            lookback = 20
            start = max(0, pos - lookback)
            if sig_dir == 1:
                # Long: how often did price go UP by target_dist from bar close?
                closes = raw_close[start:pos]
                highs_after = np.array([
                    np.max(raw_high[j:min(j + 20, pos)]) - raw_close[j]
                    for j in range(start, pos)
                    if j + 1 < pos
                ])
                if len(highs_after) > 0:
                    features["c_target_reachability"][i] = np.mean(highs_after >= td)
            else:
                # Short: how often did price go DOWN by target_dist?
                lows_after = np.array([
                    raw_close[j] - np.min(raw_low[j:min(j + 20, pos)])
                    for j in range(start, pos)
                    if j + 1 < pos
                ])
                if len(lows_after) > 0:
                    features["c_target_reachability"][i] = np.mean(lows_after >= td)

    return features


logger.info("Computing target quality features...")
t1 = _time.perf_counter()
target_features = compute_target_quality(db, signal_positions)
logger.info("Target quality computed in %.1fs", _time.perf_counter() - t1)

target_labels = {
    "a_target_rr": "Target RR (target/stop)",
    "b_target_dist_atr": "Target distance (ATR units)",
    "c_target_reachability": "Target reachability (20-bar hist)",
    "d_target_vs_atr": "Target vs ATR",
}

target_spreads = {}
for key, label in target_labels.items():
    arr = target_features[key]
    valid = np.sum(~np.isnan(arr))
    print(f"\n--- {label} (valid={valid}) ---")
    qt = quintile_analysis(db, arr, key)
    print_table(qt)
    spread = quintile_spread(qt)
    target_spreads[key] = spread
    print(f"  Quintile spread: {fmt_spread(spread)}")

print("\n--- TARGET QUALITY RANKING ---")
target_ranked = sorted(target_spreads.items(), key=lambda x: abs(x[1]), reverse=True)
rank_rows = []
for rank, (key, spread) in enumerate(target_ranked, 1):
    rank_rows.append({"rank": rank, "variant": target_labels[key], "spread": round(spread, 4)})
print_table(pd.DataFrame(rank_rows))

best_target_key = target_ranked[0][0]
print(f">>> BEST TARGET QUALITY: {target_labels[best_target_key]} (spread={target_ranked[0][1]:.4f})")


# ════════════════════════════════════════════════════════════════════════
#  6. CANDLE QUALITY — 4 METRICS
# ════════════════════════════════════════════════════════════════════════

print_section("6. CANDLE QUALITY — 4 Metrics (+ existing 3)")

def compute_candle_quality(db: pd.DataFrame, positions: np.ndarray) -> dict:
    """Compute candle quality features for signal bars."""
    n = len(db)

    # Existing features from DB
    features = {
        "existing_body_ratio": db["bar_body_ratio"].values.copy(),
        "existing_body_atr": db["bar_body_atr"].values.copy(),
        "existing_range_atr": db["bar_range_atr"].values.copy(),
    }

    # New features
    features["a_close_position"] = np.full(n, np.nan)
    features["b_wick_ratio"] = np.full(n, np.nan)
    features["c_body_vs_prior"] = np.full(n, np.nan)
    features["d_volume_spike"] = np.full(n, np.nan)

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
            # (a) Close position in range
            if sig_dir == 1:
                # Long: close near top = strong
                features["a_close_position"][i] = (c_i - l_i) / rng
            else:
                # Short: close near bottom = strong
                features["a_close_position"][i] = (h_i - c_i) / rng

            # (b) Wick ratio: max(upper_wick, lower_wick) / range
            candle_top = max(o_i, c_i)
            candle_bot = min(o_i, c_i)
            upper_wick = h_i - candle_top
            lower_wick = candle_bot - l_i
            features["b_wick_ratio"][i] = max(upper_wick, lower_wick) / rng

        # (c) Body vs prior body: body / avg(body[-5:])
        if pos >= 5:
            prior_bodies = np.abs(raw_close[pos-5:pos] - raw_open[pos-5:pos])
            avg_prior = np.mean(prior_bodies)
            if avg_prior > 0:
                features["c_body_vs_prior"][i] = body / avg_prior

        # (d) Volume spike: volume / avg(volume[-10:])
        if pos >= 10:
            prior_vol = raw_volume[pos-10:pos]
            avg_vol = np.mean(prior_vol)
            if avg_vol > 0:
                features["d_volume_spike"][i] = v_i / avg_vol

    return features


logger.info("Computing candle quality features...")
t1 = _time.perf_counter()
candle_features = compute_candle_quality(db, signal_positions)
logger.info("Candle quality computed in %.1fs", _time.perf_counter() - t1)

candle_labels = {
    "existing_body_ratio": "Body/range (existing)",
    "existing_body_atr": "Body/ATR (existing)",
    "existing_range_atr": "Range/ATR (existing)",
    "a_close_position": "Close position in range (dir-aware)",
    "b_wick_ratio": "Max wick / range",
    "c_body_vs_prior": "Body / avg(prior 5 bodies)",
    "d_volume_spike": "Volume / avg(prior 10 volume)",
}

candle_spreads = {}
for key, label in candle_labels.items():
    arr = candle_features[key]
    valid = np.sum(~np.isnan(arr))
    print(f"\n--- {label} (valid={valid}) ---")
    qt = quintile_analysis(db, arr, key)
    print_table(qt)
    spread = quintile_spread(qt)
    candle_spreads[key] = spread
    print(f"  Quintile spread: {fmt_spread(spread)}")

print("\n--- CANDLE QUALITY RANKING ---")
candle_ranked = sorted(candle_spreads.items(), key=lambda x: abs(x[1]), reverse=True)
rank_rows = []
for rank, (key, spread) in enumerate(candle_ranked, 1):
    rank_rows.append({"rank": rank, "variant": candle_labels[key], "spread": round(spread, 4)})
print_table(pd.DataFrame(rank_rows))

best_candle_key = candle_ranked[0][0]
print(f">>> BEST CANDLE QUALITY: {candle_labels[best_candle_key]} (spread={candle_ranked[0][1]:.4f})")


# ════════════════════════════════════════════════════════════════════════
#  7. MARKET REGIME — 4 METRICS (+ existing)
# ════════════════════════════════════════════════════════════════════════

print_section("7. MARKET REGIME — 4 New Metrics (+ existing)")

def compute_regime_features(db: pd.DataFrame, positions: np.ndarray) -> dict:
    """Compute market regime features for signal bars."""
    n = len(db)

    # Existing
    features = {
        "existing_atr_percentile": db["atr_percentile"].values.copy(),
    }

    # New
    features["a_vol_regime"] = np.full(n, np.nan)
    features["b_trend_strength"] = np.full(n, np.nan)
    features["c_gap_from_close"] = np.full(n, np.nan)
    features["d_time_since_last"] = np.full(n, np.nan)

    # (d) Time since last signal: bars between consecutive signals
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

        # (a) Volatility regime: ATR(20) / ATR(100)
        # Compute short-term and long-term ATR from raw ranges
        ranges_20 = raw_high[pos-19:pos+1] - raw_low[pos-19:pos+1]
        ranges_100 = raw_high[pos-99:pos+1] - raw_low[pos-99:pos+1]
        atr_20 = np.mean(ranges_20)
        atr_100 = np.mean(ranges_100)
        if atr_100 > 0:
            features["a_vol_regime"][i] = atr_20 / atr_100

        # (b) Trend strength: 20-bar return / (20-bar high - 20-bar low)
        ret_20 = raw_close[pos] - raw_close[pos - 20]
        range_20 = np.max(raw_high[pos-19:pos+1]) - np.min(raw_low[pos-19:pos+1])
        if range_20 > 0:
            features["b_trend_strength"][i] = ret_20 / range_20

        # (c) Gap from prior close
        if pos >= 1:
            gap = abs(raw_open[pos] - raw_close[pos - 1])
            features["c_gap_from_close"][i] = gap / atr_val

    return features


logger.info("Computing market regime features...")
t1 = _time.perf_counter()
regime_features = compute_regime_features(db, signal_positions)
logger.info("Market regime computed in %.1fs", _time.perf_counter() - t1)

regime_labels = {
    "existing_atr_percentile": "ATR percentile (existing)",
    "a_vol_regime": "Vol regime (ATR20/ATR100)",
    "b_trend_strength": "Trend strength (return/range)",
    "c_gap_from_close": "Gap from prior close / ATR",
    "d_time_since_last": "Bars since last signal",
}

regime_spreads = {}
for key, label in regime_labels.items():
    arr = regime_features[key]
    valid = np.sum(~np.isnan(arr))
    print(f"\n--- {label} (valid={valid}) ---")
    qt = quintile_analysis(db, arr, key)
    print_table(qt)
    spread = quintile_spread(qt)
    regime_spreads[key] = spread
    print(f"  Quintile spread: {fmt_spread(spread)}")

print("\n--- MARKET REGIME RANKING ---")
regime_ranked = sorted(regime_spreads.items(), key=lambda x: abs(x[1]), reverse=True)
rank_rows = []
for rank, (key, spread) in enumerate(regime_ranked, 1):
    rank_rows.append({"rank": rank, "variant": regime_labels[key], "spread": round(spread, 4)})
print_table(pd.DataFrame(rank_rows))

best_regime_key = regime_ranked[0][0]
print(f">>> BEST REGIME: {regime_labels[best_regime_key]} (spread={regime_ranked[0][1]:.4f})")


# ════════════════════════════════════════════════════════════════════════
#  8. BUILD OPTIMAL INDICATOR SET + COMPOSITE SCORE
# ════════════════════════════════════════════════════════════════════════

print_section("8. OPTIMAL INDICATOR SET — Composite Score Comparison")

print("\n--- BEST FEATURE FROM EACH CATEGORY ---")
best_features = {
    "fluency": (best_fluency_key, fluency_labels[best_fluency_key], fluency_spreads[best_fluency_key]),
    "displacement": (best_disp_key, disp_labels[best_disp_key], disp_results[best_disp_key]["spread"]),
    "fvg_quality": (best_fvg_key, fvg_labels[best_fvg_key], fvg_spreads[best_fvg_key]),
    "stop_quality": (best_stop_key, stop_labels[best_stop_key], stop_spreads[best_stop_key]),
    "target_quality": (best_target_key, target_labels[best_target_key], target_spreads[best_target_key]),
    "candle_quality": (best_candle_key, candle_labels[best_candle_key], candle_spreads[best_candle_key]),
    "regime": (best_regime_key, regime_labels[best_regime_key], regime_spreads[best_regime_key]),
}

for cat, (key, label, spread) in best_features.items():
    print(f"  {cat:20s}: {label:45s} spread={spread:+.4f}")

# Gather best feature arrays
print("\n--- Building composite score from best features ---")

feature_arrays = {}
feature_arrays["fluency"] = fluency_variants[best_fluency_key]
feature_arrays["fvg_quality"] = fvg_features[best_fvg_key]
feature_arrays["stop_quality"] = stop_features[best_stop_key]
feature_arrays["target_quality"] = target_features[best_target_key]
feature_arrays["candle_quality"] = candle_features[best_candle_key]
feature_arrays["regime"] = regime_features[best_regime_key]

# Displacement is binary — use it as a filter, not a continuous score
best_disp_arr = disp_variants[best_disp_key]

# Normalize each continuous feature to [0, 1] via percentile rank
normalized = {}
for name, arr in feature_arrays.items():
    valid = ~np.isnan(arr)
    if valid.sum() > 0:
        ranked = np.full(len(arr), np.nan)
        vals = arr[valid]
        # Percentile rank
        from scipy.stats import rankdata
        ranks = rankdata(vals, method="average") / len(vals)
        ranked[valid] = ranks
        normalized[name] = ranked
    else:
        normalized[name] = arr

# Check if higher values of each feature correlate with better outcomes
# If negative correlation, flip the feature
print("\n--- Feature-Outcome correlations (Spearman) ---")
from scipy.stats import spearmanr

feature_directions = {}
for name, arr in normalized.items():
    valid = ~np.isnan(arr)
    if valid.sum() > 100:
        corr, pval = spearmanr(arr[valid], outcomes[valid])
        direction = 1 if corr >= 0 else -1
        feature_directions[name] = direction
        print(f"  {name:20s}: rho={corr:+.4f}  p={pval:.4e}  direction={'keep' if direction == 1 else 'FLIP'}")
    else:
        feature_directions[name] = 1

# Build composite: equal-weight average of normalized (direction-corrected) features
composite = np.zeros(n_signals)
valid_count = np.zeros(n_signals)

for name, arr in normalized.items():
    direction = feature_directions[name]
    valid = ~np.isnan(arr)
    composite[valid] += arr[valid] * direction
    valid_count[valid] += 1

# Average
composite = np.where(valid_count > 0, composite / valid_count, np.nan)

# Also build the current SQ for comparison
current_sq = db["signal_quality"].values

print("\n--- Composite Score Quintile Analysis ---")
qt_composite = quintile_analysis(db, composite, "new_composite")
print_table(qt_composite)
spread_composite = quintile_spread(qt_composite)
print(f"  New composite quintile spread: {fmt_spread(spread_composite)}")

print("\n--- Current SQ Quintile Analysis ---")
qt_sq = quintile_analysis(db, current_sq, "current_sq")
print_table(qt_sq)
spread_sq = quintile_spread(qt_sq)
print(f"  Current SQ quintile spread: {fmt_spread(spread_sq)}")

# ---- Compare as filter: top N% ----
print_section("8b. FILTER COMPARISON — Current SQ vs New Composite")

for pct_keep in [100, 50, 40, 30, 20, 10]:
    print(f"\n{'='*60}")
    print(f"  TOP {pct_keep}% OF SIGNALS")
    print(f"{'='*60}")

    rows = []

    # Unfiltered
    if pct_keep == 100:
        m = compute_metrics(outcomes)
        m["filter"] = "ALL SIGNALS"
        rows.append(m)

    # Current SQ filter
    valid_sq = ~np.isnan(current_sq)
    if valid_sq.sum() > 0:
        threshold_sq = np.nanpercentile(current_sq, 100 - pct_keep)
        mask_sq = (current_sq >= threshold_sq) & valid_sq
        if mask_sq.sum() > 0:
            m = compute_metrics(outcomes[mask_sq])
            m["filter"] = f"Current SQ >= {threshold_sq:.3f}"
            rows.append(m)

    # New composite filter
    valid_comp = ~np.isnan(composite)
    if valid_comp.sum() > 0:
        threshold_comp = np.nanpercentile(composite, 100 - pct_keep)
        mask_comp = (composite >= threshold_comp) & valid_comp
        if mask_comp.sum() > 0:
            m = compute_metrics(outcomes[mask_comp])
            m["filter"] = f"New Composite >= {threshold_comp:.3f}"
            rows.append(m)

    # New composite + displacement filter
    if valid_comp.sum() > 0:
        mask_comp_disp = mask_comp & best_disp_arr
        if mask_comp_disp.sum() > 0:
            m = compute_metrics(outcomes[mask_comp_disp])
            m["filter"] = f"Composite + Disp"
            rows.append(m)

    if rows:
        df_cmp = pd.DataFrame(rows)
        cols = ["filter", "count", "avg_r", "wr", "pf", "total_r", "ppdd", "max_dd"]
        print_table(df_cmp[[c for c in cols if c in df_cmp.columns]])


# ════════════════════════════════════════════════════════════════════════
#  9. GRAND SUMMARY
# ════════════════════════════════════════════════════════════════════════

print_section("9. GRAND SUMMARY — All Category Winners")

summary_rows = []
all_categories = [
    ("FLUENCY", best_fluency_key, fluency_labels, fluency_spreads),
    ("FVG QUALITY", best_fvg_key, fvg_labels, fvg_spreads),
    ("STOP QUALITY", best_stop_key, stop_labels, stop_spreads),
    ("TARGET QUALITY", best_target_key, target_labels, target_spreads),
    ("CANDLE QUALITY", best_candle_key, candle_labels, candle_spreads),
    ("REGIME", best_regime_key, regime_labels, regime_spreads),
]

for cat_name, best_key, labels, spreads in all_categories:
    summary_rows.append({
        "category": cat_name,
        "best_variant": labels[best_key],
        "quintile_spread": round(spreads[best_key], 4),
    })

# Displacement special
summary_rows.append({
    "category": "DISPLACEMENT",
    "best_variant": disp_labels[best_disp_key],
    "quintile_spread": round(disp_results[best_disp_key]["spread"], 4),
})

print_table(pd.DataFrame(summary_rows))

print("\n--- COMPOSITE PERFORMANCE ---")
print(f"  Current SQ quintile spread:   {fmt_spread(spread_sq)}")
print(f"  New Composite quintile spread: {fmt_spread(spread_composite)}")
print(f"  Improvement: {fmt_spread(spread_composite - spread_sq)}")

# Rebuild final comparison at passes_all_filters level
print("\n--- COMPARISON ON FILTERED SIGNALS (passes_all_filters=True) ---")
filtered_mask = db["passes_all_filters"].values
if filtered_mask.sum() > 0:
    m_filtered = compute_metrics(outcomes[filtered_mask])
    print(f"  Current system (passes_all_filters): count={m_filtered['count']}, "
          f"R={m_filtered['total_r']}, PPDD={m_filtered['ppdd']}, PF={m_filtered['pf']}")

    # New composite top percentile matching same count
    target_count = int(filtered_mask.sum())
    valid_comp = ~np.isnan(composite)
    if valid_comp.sum() > 0:
        # Find threshold that gives ~same count
        sorted_comp = np.sort(composite[valid_comp])[::-1]
        if target_count < len(sorted_comp):
            threshold_match = sorted_comp[target_count - 1]
            mask_match = (composite >= threshold_match) & valid_comp
            m_new = compute_metrics(outcomes[mask_match])
            print(f"  New composite (top {target_count}):           count={m_new['count']}, "
                  f"R={m_new['total_r']}, PPDD={m_new['ppdd']}, PF={m_new['pf']}")

print("\n" + "=" * 100)
print("  INDICATOR RECONSTRUCTION COMPLETE")
print("=" * 100)
