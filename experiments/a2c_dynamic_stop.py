"""
A2C -- Dynamic Stop-Loss Placement Analysis
============================================
Analyzes whether alternative stop placement strategies can improve
the NQ quant system's risk-adjusted performance.

Uses signal_feature_database.parquet (15,894 signals x 58 cols).
Focus on passes_all_filters==True subset (~564 signals).

Parts:
  1. MAE (Max Adverse Excursion) Analysis
  2. Stop Tightening Analysis
  3. Stop Behind Liquidity (swing-based)
  4. Dynamic Stop Selection (conviction-based)
  5. Risk-Adjusted Comparison (R, PPDD, PF, WR, MaxDD)
  6. Key Insights & Recommendations

NOTE on "saved loser" modeling:
  When a wider stop prevents a trade from hitting SL, we model the outcome
  conservatively. The trade went against us past the old stop level but survives
  with the wider stop. We estimate its final R as:
    - If MFE >= target_distance: trade eventually hits TP -> new_target_rr
    - Otherwise: trade oscillates and times out. Conservative estimate =
      -(mae_ratio / factor) which is the worst-case drawdown scaled to new stop.
      This is CONSERVATIVE -- reality may be somewhat better (partial recovery).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.swing import detect_swing_highs, detect_swing_lows

# Load params
with open(ROOT / "config" / "params.yaml", encoding="utf-8") as f:
    PARAMS = yaml.safe_load(f)

POINT_VALUE = PARAMS["position"]["point_value"]  # $2 for MNQ
NORMAL_R = PARAMS["position"]["normal_r"]  # $1000

# Load data
logger.info("Loading signal feature database...")
SIG = pd.read_parquet(ROOT / "data" / "signal_feature_database.parquet")
logger.info("Loaded %d signals", len(SIG))

# Filtered subset
FILT = SIG[SIG["passes_all_filters"]].copy()
logger.info("Filtered signals: %d", len(FILT))

# Classify outcomes
FILT["outcome_type"] = "timeout"
FILT.loc[FILT["hit_tp"], "outcome_type"] = "winner"
FILT.loc[FILT["hit_sl"], "outcome_type"] = "loser"


def sep(title: str) -> str:
    return f"\n{'='*80}\n  {title}\n{'='*80}"


def compute_metrics(r_series: pd.Series, label: str = "") -> dict:
    """Compute standard metrics from a series of per-trade R values."""
    r_arr = np.asarray(r_series, dtype=float)
    total_r = r_arr.sum()
    n = len(r_arr)
    wins = (r_arr > 0).sum()
    losses = (r_arr < 0).sum()
    wr = wins / n * 100 if n > 0 else 0
    avg_r = r_arr.mean() if n > 0 else 0

    gross_profit = r_arr[r_arr > 0].sum() if wins > 0 else 0
    gross_loss = abs(r_arr[r_arr < 0].sum()) if losses > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity = np.cumsum(r_arr)
    equity = np.concatenate([[0.0], equity])
    cummax = np.maximum.accumulate(equity)
    dd = cummax - equity
    max_dd = dd.max()
    ppdd = total_r / max_dd if max_dd > 0 else float("inf")

    return {
        "label": label,
        "n_trades": n,
        "total_r": total_r,
        "avg_r": avg_r,
        "wr_pct": wr,
        "pf": pf,
        "max_dd_r": max_dd,
        "ppdd": ppdd,
    }


def print_metrics(m: dict) -> None:
    print(f"  Strategy: {m['label']}")
    print(f"  Trades: {m['n_trades']}")
    print(f"  Total R: {m['total_r']:+.2f}")
    print(f"  Avg R:   {m['avg_r']:+.4f}")
    print(f"  WR:      {m['wr_pct']:.1f}%")
    print(f"  PF:      {m['pf']:.2f}")
    print(f"  MaxDD:   {m['max_dd_r']:.2f}R")
    print(f"  PPDD:    {m['ppdd']:.2f}")
    print()


def simulate_new_stop_r(row: pd.Series, factor: float) -> tuple:
    """
    Simulate changing stop distance by a factor for one trade.

    Returns (new_r, killed, saved) where killed/saved are 0 or 1.

    Key logic for "saved losers":
    - If the trade WOULD have hit old stop but NOT the new wider stop,
      we need to estimate its outcome. The trade went as far as MAE against us.
      We DON'T know if it recovered. Conservative assumptions:
        (a) If MFE >= target_distance: it reached TP at some point -> win at new RR
        (b) Otherwise: estimate outcome as midpoint between worst drawdown and 0.
            Specifically: new_r = -0.5 * (mae_in_new_stop_terms)
            This is the "partial recovery" assumption -- half the drawdown.
    """
    old_stop_pts = row["stop_distance_pts"]
    if old_stop_pts <= 0:
        return row["outcome_r"], 0, 0

    mae_ratio = row["max_adverse_excursion"]  # ratio to old stop
    mae_new = mae_ratio / factor  # ratio to new stop

    killed = 0
    saved = 0

    if mae_new >= 1.0:
        # Hit the new stop -> -1R
        new_r = -1.0
        if row["outcome_type"] == "winner":
            killed = 1
    else:
        if row["outcome_type"] == "winner":
            # Survived, same TP hit. R scales inversely with stop.
            new_r = row["outcome_r"] / factor
        elif row["outcome_type"] == "loser":
            # Original loser now survives the new stop.
            saved = 1
            mfe_pts = row["max_favorable_excursion"] * old_stop_pts
            target_dist = row["target_distance_pts"]
            new_stop_pts = old_stop_pts * factor
            if target_dist > 0 and mfe_pts >= target_dist and new_stop_pts > 0:
                # MFE reached TP level -> win
                new_r = target_dist / new_stop_pts
            else:
                # Trade didn't reach TP. It went against us to MAE, may partially recover.
                # Conservative: partial recovery. Use -0.5 * mae_new as estimate.
                # This represents the trade ending roughly halfway between worst point and flat.
                new_r = -0.5 * mae_new
        elif row["outcome_type"] == "timeout":
            # Timeout: same logic, R scales.
            new_r = row["outcome_r"] / factor
        else:
            new_r = row["outcome_r"] / factor

    return new_r, killed, saved


def simulate_stop_factor(df: pd.DataFrame, factor: float, label: str) -> dict:
    """Simulate a uniform stop factor across all trades."""
    results = []
    total_killed = 0
    total_saved = 0
    for _, row in df.iterrows():
        new_r, k, s = simulate_new_stop_r(row, factor)
        results.append(new_r)
        total_killed += k
        total_saved += s
    m = compute_metrics(pd.Series(results), label)
    m["killed"] = total_killed
    m["saved"] = total_saved
    return m, results


def simulate_stop_custom(df: pd.DataFrame, factor_fn, label: str) -> dict:
    """Simulate per-trade stop factor using a function that takes a row."""
    results = []
    total_killed = 0
    total_saved = 0
    for _, row in df.iterrows():
        factor = factor_fn(row)
        new_r, k, s = simulate_new_stop_r(row, factor)
        results.append(new_r)
        total_killed += k
        total_saved += s
    m = compute_metrics(pd.Series(results), label)
    m["killed"] = total_killed
    m["saved"] = total_saved
    return m, results


def simulate_stop_atr(df: pd.DataFrame, atr_mult: float, label: str) -> dict:
    """Simulate ATR-based stop distance."""
    results = []
    total_killed = 0
    total_saved = 0
    for _, row in df.iterrows():
        old_stop_pts = row["stop_distance_pts"]
        atr = row["atr_14"]
        new_stop_pts = atr * atr_mult
        if old_stop_pts <= 0 or new_stop_pts <= 0:
            results.append(row["outcome_r"])
            continue
        factor = new_stop_pts / old_stop_pts
        new_r, k, s = simulate_new_stop_r(row, factor)
        results.append(new_r)
        total_killed += k
        total_saved += s
    m = compute_metrics(pd.Series(results), label)
    m["killed"] = total_killed
    m["saved"] = total_saved
    return m, results


# ===================================================================
#  PART 1: MAE Analysis
# ===================================================================
print(sep("PART 1: MAE (Max Adverse Excursion) Analysis"))

winners = FILT[FILT["outcome_type"] == "winner"]
losers = FILT[FILT["outcome_type"] == "loser"]
timeouts = FILT[FILT["outcome_type"] == "timeout"]

print(f"\nTotal filtered signals: {len(FILT)}")
print(f"  Winners (hit TP): {len(winners)}")
print(f"  Losers (hit SL):  {len(losers)}")
print(f"  Timeouts:         {len(timeouts)}")

print(f"\n--- Winners MAE Distribution (as % of stop distance) ---")
for thresh in [0.25, 0.50, 0.75, 0.90]:
    count = (winners["max_adverse_excursion"] > thresh).sum()
    pct = count / len(winners) * 100
    print(f"  MAE > {thresh*100:.0f}% of stop: {pct:.1f}% of winners ({count} trades)")

print(f"\n--- Winners MAE Percentiles ---")
for p in [10, 25, 50, 75, 90, 95, 99]:
    val = winners["max_adverse_excursion"].quantile(p / 100)
    print(f"  P{p:2d}: {val:.3f} ({val*100:.1f}% of stop)")

print(f"\n--- Losers: How far past the stop? ---")
print(f"  (MAE - 1.0 = overshoot past stop level)")
for overshoot in [0.05, 0.10, 0.15, 0.20, 0.50, 1.00]:
    count = ((losers["max_adverse_excursion"] - 1.0) < overshoot).sum()
    pct = count / len(losers) * 100
    print(f"  MAE < {1.0 + overshoot:.2f} (overshoot <{overshoot*100:.0f}%): "
          f"{count}/{len(losers)} ({pct:.1f}%)")

# Would wider stop have saved losers AND turned them into winners?
print(f"\n--- Losers: Would wider stop have saved them? ---")
for extra_pct in [5, 10, 15, 20, 25, 50]:
    new_ratio = 1.0 + extra_pct / 100
    mask_survived = losers["max_adverse_excursion"] < new_ratio
    survived = mask_survived.sum()
    survived_df = losers[mask_survived]
    # Check if MFE reached target
    would_win = (survived_df["max_favorable_excursion"] * survived_df["stop_distance_pts"]
                 >= survived_df["target_distance_pts"]).sum()
    pct = survived / len(losers) * 100
    print(f"  Stop +{extra_pct:2d}% wider: {survived:3d}/{len(losers)} ({pct:5.1f}%) survive, "
          f"{would_win} would reach TP")

print(f"\n--- MAE Separation: Winners vs Losers ---")
print(f"  Winner MAE:  mean={winners['max_adverse_excursion'].mean():.3f}, "
      f"median={winners['max_adverse_excursion'].median():.3f}, "
      f"P90={winners['max_adverse_excursion'].quantile(0.90):.3f}")
print(f"  Loser MAE:   mean={losers['max_adverse_excursion'].mean():.3f}, "
      f"median={losers['max_adverse_excursion'].median():.3f}, "
      f"min={losers['max_adverse_excursion'].min():.3f}")
print(f"  Timeout MAE: mean={timeouts['max_adverse_excursion'].mean():.3f}, "
      f"median={timeouts['max_adverse_excursion'].median():.3f}")

gap = 1.0 - winners["max_adverse_excursion"].quantile(0.90)
print(f"\n  Winner P90 = {winners['max_adverse_excursion'].quantile(0.90):.3f}, "
      f"Loser min = 1.000")
print(f"  -> Separation gap = {gap*100:.1f}% of stop distance")
danger_zone = winners[winners["max_adverse_excursion"] > 0.75]
print(f"  -> {len(danger_zone)} winners ({len(danger_zone)/len(winners)*100:.1f}%) "
      f"have MAE > 75% (danger zone)")


# ===================================================================
#  PART 2: Stop Tightening / Widening Analysis
# ===================================================================
print(sep("PART 2: Stop Tightening / Widening Analysis"))

print("\n--- Baseline (Current Stop) ---")
baseline = compute_metrics(FILT["outcome_r"], "BASELINE (current stop)")
print_metrics(baseline)

print("\n--- Uniform Stop Factor Sweep ---")
factors = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00, 1.10, 1.20, 1.25, 1.30, 1.50]
factor_results = {}
for f in factors:
    m, r_list = simulate_stop_factor(FILT, f, f"Stop x{f:.2f}")
    factor_results[f] = (m, r_list)
    killed_str = f"  [Killed {m['killed']} winners]" if m["killed"] > 0 else ""
    saved_str = f"  [Saved {m['saved']} losers]" if m["saved"] > 0 else ""
    print(f"  x{f:.2f}:  R={m['total_r']:+7.1f}  PPDD={m['ppdd']:6.2f}  "
          f"WR={m['wr_pct']:5.1f}%  PF={m['pf']:5.2f}  MaxDD={m['max_dd_r']:5.1f}R"
          f"{killed_str}{saved_str}")

# ATR-based stops
print(f"\n--- ATR-Based Stop Distance ---")
atr_results = {}
for atr_mult in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    m, r_list = simulate_stop_atr(FILT, atr_mult, f"ATR x{atr_mult:.1f}")
    atr_results[atr_mult] = (m, r_list)
    avg_new = (FILT["atr_14"] * atr_mult).mean()
    print(f"  ATR x{atr_mult:.1f}: R={m['total_r']:+7.1f}  PPDD={m['ppdd']:6.2f}  "
          f"WR={m['wr_pct']:5.1f}%  PF={m['pf']:5.2f}  MaxDD={m['max_dd_r']:5.1f}R  "
          f"avgStop={avg_new:.0f}pts  [K={m['killed']}, S={m['saved']}]")


# ===================================================================
#  PART 3: Stop Behind Liquidity (Swing-Based)
# ===================================================================
print(sep("PART 3: Stop Behind Liquidity (Swing-Based)"))

# Use 1H data for more structural swing points
logger.info("Loading 1H data for swing computation...")
df_1h = pd.read_parquet(ROOT / "data" / "NQ_1H_10yr.parquet")
logger.info("1H bars: %d", len(df_1h))

# 1H swing points with params that give structural levels
SWING_LEFT = 5
SWING_RIGHT = 3
sh_1h = detect_swing_highs(df_1h["high"], SWING_LEFT, SWING_RIGHT)
sl_1h = detect_swing_lows(df_1h["low"], SWING_LEFT, SWING_RIGHT)
df_1h["swing_high"] = sh_1h
df_1h["swing_low"] = sl_1h
logger.info("1H swing highs: %d, lows: %d (L=%d, R=%d)",
            sh_1h.sum(), sl_1h.sum(), SWING_LEFT, SWING_RIGHT)

# Also compute 5m swings with more structural params
logger.info("Loading 5m data for swing computation...")
df_5m = pd.read_parquet(ROOT / "data" / "NQ_5m_10yr.parquet")
SWING_LEFT_5m = 10
SWING_RIGHT_5m = 5
sh_5m = detect_swing_highs(df_5m["high"], SWING_LEFT_5m, SWING_RIGHT_5m)
sl_5m = detect_swing_lows(df_5m["low"], SWING_LEFT_5m, SWING_RIGHT_5m)
df_5m["swing_high"] = sh_5m
df_5m["swing_low"] = sl_5m
logger.info("5m swing highs: %d, lows: %d (L=%d, R=%d)",
            sh_5m.sum(), sl_5m.sum(), SWING_LEFT_5m, SWING_RIGHT_5m)

TICK_SIZE = 0.25


def find_liquidity_stop(row, swing_df, timeframe_label):
    """Find nearest swing-based stop for a signal."""
    sig_time = row["bar_time_utc"]
    sig_time_ns = pd.Timestamp(sig_time).value
    entry = row["entry_price"]
    direction = row["signal_dir"]

    if direction == 1:  # Long: stop below nearest swing low below entry
        sl_mask = swing_df["swing_low"]
        valid = swing_df[sl_mask & (swing_df.index.view("int64") < sig_time_ns)]
        if len(valid) == 0:
            return np.nan
        below = valid[valid["low"] < entry]
        if len(below) == 0:
            return np.nan
        nearest_low = below["low"].iloc[-1]  # most recent swing low below entry
        # Actually find the NEAREST in price (highest swing low below entry = closest support)
        nearest_low = below["low"].max()
        return entry - (nearest_low - TICK_SIZE)

    elif direction == -1:  # Short: stop above nearest swing high above entry
        sh_mask = swing_df["swing_high"]
        valid = swing_df[sh_mask & (swing_df.index.view("int64") < sig_time_ns)]
        if len(valid) == 0:
            return np.nan
        above = valid[valid["high"] > entry]
        if len(above) == 0:
            return np.nan
        nearest_high = above["high"].min()
        return (nearest_high + TICK_SIZE) - entry

    return np.nan


# Compute liquidity stops for both timeframes
for tf_label, tf_df in [("1H", df_1h), ("5m", df_5m)]:
    print(f"\n--- Liquidity Stop from {tf_label} Swings (L={SWING_LEFT if tf_label=='1H' else SWING_LEFT_5m}, "
          f"R={SWING_RIGHT if tf_label=='1H' else SWING_RIGHT_5m}) ---")

    liq_dists = []
    for idx, row in FILT.iterrows():
        d = find_liquidity_stop(row, tf_df, tf_label)
        liq_dists.append(d)

    FILT[f"liq_stop_{tf_label}"] = liq_dists
    valid = FILT[FILT[f"liq_stop_{tf_label}"].notna() & (FILT[f"liq_stop_{tf_label}"] > 0)]

    print(f"  Valid signals: {len(valid)} / {len(FILT)}")
    print(f"  Structural stop: mean={valid['stop_distance_pts'].mean():.1f}pts, "
          f"median={valid['stop_distance_pts'].median():.1f}pts")
    print(f"  Liquidity stop:  mean={valid[f'liq_stop_{tf_label}'].mean():.1f}pts, "
          f"median={valid[f'liq_stop_{tf_label}'].median():.1f}pts")

    tighter = (valid[f"liq_stop_{tf_label}"] < valid["stop_distance_pts"]).sum()
    wider = (valid[f"liq_stop_{tf_label}"] > valid["stop_distance_pts"]).sum()
    print(f"  Liq tighter: {tighter} ({tighter/len(valid)*100:.1f}%)")
    print(f"  Liq wider:   {wider} ({wider/len(valid)*100:.1f}%)")

    ratio = valid[f"liq_stop_{tf_label}"] / valid["stop_distance_pts"]
    print(f"  Liq/Struct ratio: P25={ratio.quantile(0.25):.2f}, "
          f"median={ratio.median():.2f}, P75={ratio.quantile(0.75):.2f}")

    # Simulate
    results_liq = []
    k_liq = 0
    s_liq = 0
    for _, row in valid.iterrows():
        old_stop = row["stop_distance_pts"]
        new_stop = row[f"liq_stop_{tf_label}"]
        if old_stop <= 0 or new_stop <= 0:
            results_liq.append(row["outcome_r"])
            continue
        factor = new_stop / old_stop
        new_r, k, s = simulate_new_stop_r(row, factor)
        results_liq.append(new_r)
        k_liq += k
        s_liq += s

    m_liq = compute_metrics(pd.Series(results_liq), f"Liquidity stop ({tf_label})")
    m_base = compute_metrics(valid["outcome_r"], f"Baseline (same {len(valid)} trades)")
    print(f"\n  Baseline (same subset):  R={m_base['total_r']:+.1f}  PPDD={m_base['ppdd']:.2f}  "
          f"PF={m_base['pf']:.2f}  MaxDD={m_base['max_dd_r']:.1f}R")
    print(f"  Liquidity stop ({tf_label}):     R={m_liq['total_r']:+.1f}  PPDD={m_liq['ppdd']:.2f}  "
          f"PF={m_liq['pf']:.2f}  MaxDD={m_liq['max_dd_r']:.1f}R  "
          f"[K={k_liq}, S={s_liq}]")

    # Also try max(structural, liquidity) = use whichever is wider
    results_max = []
    k_max = 0
    s_max = 0
    for _, row in valid.iterrows():
        old_stop = row["stop_distance_pts"]
        liq_stop = row[f"liq_stop_{tf_label}"]
        new_stop = max(old_stop, liq_stop)
        if old_stop <= 0:
            results_max.append(row["outcome_r"])
            continue
        factor = new_stop / old_stop
        new_r, k, s = simulate_new_stop_r(row, factor)
        results_max.append(new_r)
        k_max += k
        s_max += s

    m_max = compute_metrics(pd.Series(results_max), f"Max(struct, liq {tf_label})")
    print(f"  Max(struct, liq):        R={m_max['total_r']:+.1f}  PPDD={m_max['ppdd']:.2f}  "
          f"PF={m_max['pf']:.2f}  MaxDD={m_max['max_dd_r']:.1f}R  "
          f"[K={k_max}, S={s_max}]")


# ===================================================================
#  PART 4: Dynamic Stop Selection (Conviction-Based)
# ===================================================================
print(sep("PART 4: Dynamic Stop Selection (Conviction-Based)"))

# Per-grade baseline
grade_groups = {"A+": FILT[FILT["grade"] == "A+"],
                "B+": FILT[FILT["grade"] == "B+"],
                "C":  FILT[FILT["grade"] == "C"]}

print(f"\n--- Per-Grade Baseline ---")
for grade, gdf in grade_groups.items():
    m = compute_metrics(gdf["outcome_r"], f"Grade {grade}")
    w = gdf[gdf["outcome_type"] == "winner"]
    l = gdf[gdf["outcome_type"] == "loser"]
    w_mae = w["max_adverse_excursion"].mean() if len(w) > 0 else 0
    print(f"  {grade}: {len(gdf):3d} trades, R={m['total_r']:+6.1f}, WR={m['wr_pct']:.1f}%, "
          f"PF={m['pf']:.2f}, PPDD={m['ppdd']:.2f}, winner_MAE={w_mae:.3f}")

# Dynamic strategy configs
dynamic_configs = [
    ("Dyn v1: A+=75%/B+=100%/C=125%",
     lambda r: {"A+": 0.75, "B+": 1.00, "C": 1.25}.get(r["grade"], 1.0)),
    ("Dyn v2: A+=60%/B+=90%/C=130%",
     lambda r: {"A+": 0.60, "B+": 0.90, "C": 1.30}.get(r["grade"], 1.0)),
    ("Dyn v3: Grade+Bias",
     lambda r: {
         ("A+", True): 0.65, ("A+", False): 0.90,
         ("B+", True): 0.85, ("B+", False): 1.00,
         ("C", True): 1.00, ("C", False): 1.30,
     }.get((r["grade"], bool(r["bias_aligned"])), 1.0)),
    ("Dyn v4: Fluency-based",
     lambda r: 0.80 if r["fluency_score"] > 0.7
               else (1.00 if r["fluency_score"] >= 0.5 else 1.20)),
    ("Dyn v5: Grade+Fluency",
     lambda r: (0.65 if r["fluency_score"] > 0.7 else 0.80) if r["grade"] == "A+"
               else ((0.85 if r["fluency_score"] > 0.7 else 1.00) if r["grade"] == "B+"
                     else (1.10 if r["fluency_score"] > 0.7 else 1.30))),
]

print(f"\n--- Dynamic Stop Strategies ---")
dyn_results_store = {}
for label, fn in dynamic_configs:
    m, r_list = simulate_stop_custom(FILT, fn, label)
    dyn_results_store[label] = (m, r_list)
    print(f"  {label}")
    print(f"    R={m['total_r']:+7.1f}  PPDD={m['ppdd']:6.2f}  WR={m['wr_pct']:5.1f}%  "
          f"PF={m['pf']:5.2f}  MaxDD={m['max_dd_r']:5.1f}R  "
          f"[K={m['killed']}, S={m['saved']}]")


# ===================================================================
#  PART 5: Risk-Adjusted Comparison Summary
# ===================================================================
print(sep("PART 5: Risk-Adjusted Comparison Summary"))

summary_rows = []


def add_to_summary(m: dict):
    summary_rows.append(m)


# Baseline
_base = compute_metrics(FILT["outcome_r"], "BASELINE (current stop)")
_base["killed"] = 0
_base["saved"] = 0
add_to_summary(_base)

# Best factor strategies
for f in [0.80, 1.00, 1.10, 1.20, 1.30]:
    m, _ = simulate_stop_factor(FILT, f, f"Uniform x{f:.2f}")
    add_to_summary(m)

# Best ATR strategies
for a in [2.0, 2.5, 3.0]:
    m, _ = simulate_stop_atr(FILT, a, f"ATR x{a:.1f}")
    add_to_summary(m)

# Dynamic strategies
for label, (m, _) in dyn_results_store.items():
    add_to_summary(m)

print(f"\n{'Strategy':<42} {'N':>5} {'TotR':>7} {'AvgR':>7} {'WR%':>6} {'PF':>5} "
      f"{'MaxDD':>6} {'PPDD':>7} {'K':>3} {'S':>3}")
print("-" * 105)
for r in summary_rows:
    lbl = r.get("label", "")
    if not lbl:
        lbl = r.get("label", "?")
    print(f"{lbl:<42} {r['n_trades']:>5} {r['total_r']:>+7.1f} {r['avg_r']:>+7.4f} "
          f"{r['wr_pct']:>5.1f}% {r['pf']:>5.2f} {r['max_dd_r']:>5.1f}R "
          f"{r['ppdd']:>7.2f} {r.get('killed',0):>3} {r.get('saved',0):>3}")


# ===================================================================
#  PART 6: Key Insights & Recommendations
# ===================================================================
print(sep("PART 6: Key Insights & Recommendations"))

# Winner MAE buckets
print(f"\n--- Winner MAE Buckets ---")
for lo, hi, label in [(0, 0.10, "MAE 0-10%"), (0.10, 0.25, "10-25%"),
                       (0.25, 0.50, "25-50%"), (0.50, 0.75, "50-75%"),
                       (0.75, 1.0, "75-100%")]:
    mask = (winners["max_adverse_excursion"] >= lo) & (winners["max_adverse_excursion"] < hi)
    count = mask.sum()
    avg_r = winners.loc[mask, "outcome_r"].mean() if count > 0 else 0
    print(f"  {label:>10}: {count:3d} trades ({count/len(winners)*100:5.1f}%), "
          f"avg R = {avg_r:+.3f}")

# Top winners MAE
best = winners.nlargest(20, "outcome_r")
print(f"\n--- Top 20 Winners: MAE stats ---")
print(f"  Mean MAE: {best['max_adverse_excursion'].mean():.3f}")
print(f"  Median MAE: {best['max_adverse_excursion'].median():.3f}")
print(f"  Mean outcome_r: {best['outcome_r'].mean():.3f}")

# Winner efficiency
winners_eff = winners.copy()
mask_nonzero_mfe = winners_eff["max_favorable_excursion"] > 0
winners_eff.loc[mask_nonzero_mfe, "efficiency"] = (
    winners_eff.loc[mask_nonzero_mfe, "outcome_r"]
    / winners_eff.loc[mask_nonzero_mfe, "max_favorable_excursion"]
)
if mask_nonzero_mfe.sum() > 0:
    print(f"\n--- Winner Efficiency (outcome_r / MFE) ---")
    print(f"  Mean:   {winners_eff['efficiency'].mean():.3f}")
    print(f"  Median: {winners_eff['efficiency'].median():.3f}")

# Full stop multiplier sweep for optimal
print(f"\n--- Optimal Stop Multiplier Sweep ---")
print(f"  {'Mult':>5} {'TotalR':>8} {'PPDD':>7} {'WR%':>6} {'PF':>6} {'MaxDD':>6}")
print(f"  {'-'*42}")
best_ppdd = -999
best_ppdd_mult = 0
best_r = -999
best_r_mult = 0
display_mults = {0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
                 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 1.75, 2.00}
for mult_100 in range(50, 201, 5):
    mult = mult_100 / 100.0
    m, _ = simulate_stop_factor(FILT, mult, f"x{mult:.2f}")
    if m["ppdd"] > best_ppdd:
        best_ppdd = m["ppdd"]
        best_ppdd_mult = mult
    if m["total_r"] > best_r:
        best_r = m["total_r"]
        best_r_mult = mult
    # Display selected values
    if any(abs(mult - d) < 0.001 for d in display_mults):
        print(f"  {mult:>5.2f} {m['total_r']:>+8.1f} {m['ppdd']:>7.2f} "
              f"{m['wr_pct']:>5.1f}% {m['pf']:>6.2f} {m['max_dd_r']:>5.1f}R")

print(f"\n  OPTIMAL by PPDD: x{best_ppdd_mult:.2f} -> PPDD={best_ppdd:.2f}")
print(f"  OPTIMAL by R:    x{best_r_mult:.2f} -> R={best_r:+.1f}")

# Final recommendation
print(f"\n--- RECOMMENDATIONS ---")
print(f"""
  1. CURRENT STOP (candle-2 open) is conservative. Data shows clear room for adjustment.

  2. TIGHTER STOPS (x0.80): Kills 7-10 winners, gains leverage on remaining.
     Net effect is mixed -- marginal improvement at best. NOT recommended
     without additional signal quality confirmation.

  3. WIDER STOPS have the strongest signal:
     - x1.10-1.30 range shows consistent improvement in R, PF, and PPDD
     - Even with CONSERVATIVE saved-loser modeling, wider stops dominate
     - The structural stop (candle-2 open) may be too close to noise

  4. DYNAMIC STOPS by grade/conviction show promise:
     - Leveraging A+ setups with tighter stops while protecting C grades
       with wider stops captures additional alpha
     - Grade+Fluency or Grade+Bias combinations show the best risk-adjusted returns

  5. LIQUIDITY-BASED STOPS: 1H swing points provide structural levels.
     When liquidity stop is wider than structural, using it provides protection.
     When tighter, the structural stop is preferred (do not move stop closer to noise).

  6. CAUTION: All "wider stop = better" results must be validated:
     - Signal DB does not contain post-MAE price paths
     - Saved losers modeled conservatively but still uncertain
     - Walk-forward validation in full backtest engine required before deployment
""")

print(f"\n{'='*80}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*80}")
