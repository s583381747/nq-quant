"""
experiments/b2_reentry_analysis.py — Re-entry after Stop-Loss Analysis
======================================================================

Investigates whether re-entering on the same FVG after a stop-out is
profitable. CLAUDE.md section 11.6: "No strict rule against re-entering
same FVG. Requires fresh displacement and good candle closure confirming
direction. Treat as a new setup."

Context:
  - Config H: 1104 trades, +316.4R, 68% of trades stop out (Stage 0, -277R)
  - Some stopped-out FVGs may still be valid and produce profitable re-entries
  - Goal: find additional R from re-entry without degrading PPDD

Parts:
  1. Analyze stopped trades — FVG validity after stop
  2. Simulate re-entry with displacement requirement
  3. Impact assessment: additional R, trade count, PPDD
  4. Risk considerations: correlated losses, max re-entries, reduced sizing

Uses signal_feature_database.parquet for stopped trade identification
and NQ_5m_10yr.parquet for forward price simulation.

Every summary table shows R + PPDD + PF together.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT = Path(r"C:/projects/lanto quant/nq quant")
DB_PATH = PROJECT / "data" / "signal_feature_database.parquet"
DATA_5M_PATH = PROJECT / "data" / "NQ_5m_10yr.parquet"
CONFIG_PATH = PROJECT / "config" / "params.yaml"

# ── Load params ────────────────────────────────────────────────────────
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# ── Metrics helper ─────────────────────────────────────────────────────

def compute_metrics(outcomes: np.ndarray, label: str = "") -> dict:
    """Compute R, PPDD, PF, WR, count from an array of R-multiples."""
    n = len(outcomes)
    if n == 0:
        return {"Label": label, "N": 0, "Total_R": 0.0, "Avg_R": 0.0,
                "WR%": 0.0, "PF": 0.0, "PPDD": 0.0, "MaxDD": 0.0}

    total_r = float(np.sum(outcomes))
    avg_r = float(np.mean(outcomes))
    winners = outcomes[outcomes > 0]
    losers = outcomes[outcomes <= 0]
    wr = len(winners) / n * 100 if n > 0 else 0.0

    gross_profit = float(np.sum(winners)) if len(winners) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losers))) if len(losers) > 0 else 0.001
    pf = gross_profit / gross_loss

    cumr = np.cumsum(outcomes)
    running_max = np.maximum.accumulate(cumr)
    drawdown = cumr - running_max
    ppdd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    maxdd = abs(ppdd)

    return {
        "Label": label, "N": n, "Total_R": round(total_r, 2),
        "Avg_R": round(avg_r, 4), "WR%": round(wr, 1),
        "PF": round(pf, 3), "PPDD": round(ppdd, 2), "MaxDD": round(maxdd, 2),
    }


def print_separator(title: str):
    log.info("")
    log.info("=" * 90)
    log.info(f"  {title}")
    log.info("=" * 90)


def print_metrics_table(rows: list[dict]):
    """Print a list of metric dicts as an aligned table."""
    df = pd.DataFrame(rows)
    if "Label" in df.columns:
        df = df.set_index("Label")
    log.info(df.to_string(float_format=lambda x: f"{x:.3f}" if abs(x) < 100 else f"{x:.1f}"))


# ══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print_separator("LOADING DATA")

log.info("Loading signal feature database...")
db = pd.read_parquet(DB_PATH)
log.info(f"  Loaded {len(db)} signals x {len(db.columns)} features")

log.info("Loading 5m OHLCV data...")
df5 = pd.read_parquet(DATA_5M_PATH)
log.info(f"  Loaded {len(df5)} bars, {df5.index.min()} to {df5.index.max()}")

# Pre-compute arrays for fast access
open_arr = df5["open"].values
high_arr = df5["high"].values
low_arr = df5["low"].values
close_arr = df5["close"].values

# Build time-to-index lookup
time_to_idx = {t: i for i, t in enumerate(df5.index)}

# ── Extract stopped trades (passes_all_filters + hit_sl) ──────────────
passing = db[db["passes_all_filters"]].copy()
stopped = passing[passing["hit_sl"] == True].copy()

log.info(f"  Total passing trades: {len(passing)}")
log.info(f"  Stopped-out trades:   {len(stopped)} ({100*len(stopped)/len(passing):.1f}%)")

# ── Build FVG zone approximation ──────────────────────────────────────
# For longs:  rejection candle closed ABOVE FVG top
#   FVG top ~ entry_price (close is just above FVG top)
#   FVG bottom ~ entry_price - fvg_size_pts
#   model_stop is candle2_open (displacement candle open), below FVG bottom
# For shorts: rejection candle closed BELOW FVG bottom
#   FVG bottom ~ entry_price (close is just below FVG bottom)
#   FVG top ~ entry_price + fvg_size_pts
#   model_stop is candle2_open (displacement candle open), above FVG top

# Better approximation: the entry_price is close of rejection candle.
# The rejection candle entered the FVG and closed out the other side.
# So for longs: FVG_top is approximately at or just below entry_price.
# FVG_bottom = FVG_top - fvg_size_pts.
# For the validity check, we use: price closing below FVG_bottom invalidates bull FVG.

# We approximate:
#   Bull FVG: top = entry_price, bottom = entry_price - fvg_size_pts
#   Bear FVG: bottom = entry_price, top = entry_price + fvg_size_pts

stopped["fvg_top"] = np.where(
    stopped["signal_dir"] == 1,
    stopped["entry_price"],
    stopped["entry_price"] + stopped["fvg_size_pts"],
)
stopped["fvg_bottom"] = np.where(
    stopped["signal_dir"] == 1,
    stopped["entry_price"] - stopped["fvg_size_pts"],
    stopped["entry_price"],
)

# ══════════════════════════════════════════════════════════════════════════
#  PART 1: POST-STOP FVG VALIDITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print_separator("PART 1: POST-STOP FVG VALIDITY ANALYSIS")

# Configuration
SCAN_WINDOW = 20     # bars after stop to look for re-entry
MIN_BODY_RATIO = 0.50  # rejection candle quality
SLIPPAGE_PTS = 0.25    # 1 tick slippage

results = []

for row_idx, row in stopped.iterrows():
    sig_time = pd.Timestamp(row["bar_time_utc"])
    direction = int(row["signal_dir"])
    entry_price = float(row["entry_price"])
    model_stop = float(row["model_stop"])
    fvg_top = float(row["fvg_top"])
    fvg_bottom = float(row["fvg_bottom"])
    fvg_size = float(row["fvg_size_pts"])
    bars_to_stop = int(row["bars_to_outcome"])
    irl_target = float(row["irl_target"])

    if sig_time not in time_to_idx:
        continue

    sig_idx = time_to_idx[sig_time]
    stop_bar_idx = sig_idx + bars_to_stop  # approximate bar where stop was hit

    if stop_bar_idx >= len(df5) - SCAN_WINDOW - 1:
        continue

    # Check FVG validity: from stop bar, is the FVG still intact?
    # Bull FVG invalidated when price CLOSES below fvg_bottom
    # Bear FVG invalidated when price CLOSES above fvg_top
    fvg_valid_at_stop = True
    for j in range(sig_idx + 1, min(stop_bar_idx + 1, len(df5))):
        if direction == 1 and close_arr[j] < fvg_bottom:
            fvg_valid_at_stop = False
            break
        elif direction == -1 and close_arr[j] > fvg_top:
            fvg_valid_at_stop = False
            break

    # Scan forward from stop bar for re-entry candidate
    reentry_found = False
    reentry_bar_idx = -1
    reentry_price = np.nan
    reentry_displacement = False
    bars_to_reentry = -1

    if fvg_valid_at_stop:
        for j in range(stop_bar_idx + 1, min(stop_bar_idx + SCAN_WINDOW + 1, len(df5))):
            # Check FVG still valid at this bar
            if direction == 1 and close_arr[j] < fvg_bottom:
                break  # FVG invalidated
            if direction == -1 and close_arr[j] > fvg_top:
                break  # FVG invalidated

            # Check: price returns to FVG zone
            bar_h = high_arr[j]
            bar_l = low_arr[j]
            bar_o = open_arr[j]
            bar_c = close_arr[j]
            bar_range = bar_h - bar_l
            if bar_range <= 0:
                continue
            bar_body = abs(bar_c - bar_o)
            body_ratio = bar_body / bar_range

            if direction == 1:
                # Long: price enters FVG zone (low touches into [fvg_bottom, fvg_top])
                entered_zone = bar_l <= fvg_top and bar_h >= fvg_bottom
                # Closes above FVG top (rejection)
                rejected = bar_c > fvg_top
                # Displacement: body ratio check
                has_displacement = body_ratio >= MIN_BODY_RATIO and bar_c > bar_o  # bullish candle

            else:
                # Short: price enters FVG zone (high reaches into [fvg_bottom, fvg_top])
                entered_zone = bar_h >= fvg_bottom and bar_l <= fvg_top
                # Closes below FVG bottom (rejection)
                rejected = bar_c < fvg_bottom
                # Displacement: body ratio check
                has_displacement = body_ratio >= MIN_BODY_RATIO and bar_c < bar_o  # bearish candle

            if entered_zone and rejected and has_displacement:
                reentry_found = True
                reentry_bar_idx = j
                # Entry at open of NEXT bar (consistent with engine logic)
                if j + 1 < len(df5):
                    reentry_price = open_arr[j + 1]
                else:
                    reentry_price = bar_c
                reentry_displacement = True
                bars_to_reentry = j - stop_bar_idx
                break

    # Simulate re-entry outcome if found
    reentry_outcome_r = np.nan
    reentry_hit_tp = False
    reentry_hit_sl = False
    reentry_bars_held = 0

    if reentry_found and not np.isnan(reentry_price):
        # Stop: use the original FVG-based stop approach
        # For longs: stop at fvg_bottom (below the FVG)
        # For shorts: stop at fvg_top (above the FVG)
        if direction == 1:
            re_stop = fvg_bottom - SLIPPAGE_PTS
            re_entry = reentry_price + SLIPPAGE_PTS
        else:
            re_stop = fvg_top + SLIPPAGE_PTS
            re_entry = reentry_price - SLIPPAGE_PTS

        stop_dist = abs(re_entry - re_stop)
        if stop_dist < 1.0:
            # Too tight, skip
            reentry_found = False
        else:
            # TP: use original IRL target scaled to new entry
            # Or use same RR as original trade
            original_stop_dist = abs(entry_price - model_stop)
            original_tp_dist = abs(irl_target - entry_price)
            if original_stop_dist > 0:
                original_rr = original_tp_dist / original_stop_dist
            else:
                original_rr = 2.0

            # Set TP at same RR from new entry
            tp_dist = stop_dist * max(original_rr, 1.0)  # at least 1:1 RR
            if direction == 1:
                re_tp = re_entry + tp_dist
            else:
                re_tp = re_entry - tp_dist

            # Forward simulation: walk bar by bar from re-entry
            MAX_HOLD = 40  # bars
            entry_bar = reentry_bar_idx + 1  # next bar open
            if entry_bar < len(df5):
                for k in range(entry_bar, min(entry_bar + MAX_HOLD, len(df5))):
                    reentry_bars_held = k - entry_bar + 1
                    if direction == 1:
                        # Stop hit?
                        if low_arr[k] <= re_stop:
                            reentry_hit_sl = True
                            exit_price = re_stop - SLIPPAGE_PTS
                            reentry_outcome_r = (exit_price - re_entry) / stop_dist
                            break
                        # TP hit?
                        if high_arr[k] >= re_tp:
                            reentry_hit_tp = True
                            exit_price = re_tp
                            reentry_outcome_r = (exit_price - re_entry) / stop_dist
                            break
                    else:
                        # Stop hit?
                        if high_arr[k] >= re_stop:
                            reentry_hit_sl = True
                            exit_price = re_stop + SLIPPAGE_PTS
                            reentry_outcome_r = (re_entry - exit_price) / stop_dist
                            break
                        # TP hit?
                        if low_arr[k] <= re_tp:
                            reentry_hit_tp = True
                            exit_price = re_tp
                            reentry_outcome_r = (re_entry - exit_price) / stop_dist
                            break

                # If neither hit within MAX_HOLD, exit at last bar close
                if not reentry_hit_tp and not reentry_hit_sl:
                    last_bar = min(entry_bar + MAX_HOLD - 1, len(df5) - 1)
                    if direction == 1:
                        reentry_outcome_r = (close_arr[last_bar] - re_entry) / stop_dist
                    else:
                        reentry_outcome_r = (re_entry - close_arr[last_bar]) / stop_dist

    results.append({
        "sig_time": sig_time,
        "direction": direction,
        "signal_type": row["signal_type"],
        "entry_price": entry_price,
        "model_stop": model_stop,
        "fvg_top": fvg_top,
        "fvg_bottom": fvg_bottom,
        "fvg_size": fvg_size,
        "irl_target": irl_target,
        "bars_to_stop": bars_to_stop,
        "fvg_valid_at_stop": fvg_valid_at_stop,
        "reentry_found": reentry_found,
        "reentry_bar_idx": reentry_bar_idx,
        "reentry_price": reentry_price,
        "bars_to_reentry": bars_to_reentry,
        "reentry_displacement": reentry_displacement,
        "reentry_outcome_r": reentry_outcome_r,
        "reentry_hit_tp": reentry_hit_tp,
        "reentry_hit_sl": reentry_hit_sl,
        "reentry_bars_held": reentry_bars_held,
        "year": sig_time.year,
    })

results_df = pd.DataFrame(results)
log.info(f"  Analyzed {len(results_df)} stopped trades")

# ── Summary statistics ─────────────────────────────────────────────────
n_total = len(results_df)
n_valid = results_df["fvg_valid_at_stop"].sum()
n_invalid = n_total - n_valid
n_reentry = results_df["reentry_found"].sum()

log.info("")
log.info(f"  Stopped trades analyzed:        {n_total}")
log.info(f"  FVG still valid at stop:        {n_valid} ({100*n_valid/n_total:.1f}%)")
log.info(f"  FVG invalidated before/at stop: {n_invalid} ({100*n_invalid/n_total:.1f}%)")
log.info(f"  Re-entry candidates found:      {n_reentry} ({100*n_reentry/n_total:.1f}%)")
if n_valid > 0:
    log.info(f"  Re-entry rate (valid FVGs):     {n_reentry}/{n_valid} = {100*n_reentry/n_valid:.1f}%")

# ── FVG validity by direction ──────────────────────────────────────────
log.info("")
log.info("  FVG validity by direction:")
for d, label in [(1, "Long"), (-1, "Short")]:
    subset = results_df[results_df["direction"] == d]
    n_d = len(subset)
    v_d = subset["fvg_valid_at_stop"].sum()
    r_d = subset["reentry_found"].sum()
    log.info(f"    {label}: {n_d} stops | {v_d} valid ({100*v_d/n_d:.0f}%) | {r_d} re-entries ({100*r_d/n_d:.0f}%)")

# ── FVG validity by signal type ────────────────────────────────────────
log.info("")
log.info("  FVG validity by signal type:")
for st in ["trend", "mss"]:
    subset = results_df[results_df["signal_type"] == st]
    n_s = len(subset)
    if n_s == 0:
        continue
    v_s = subset["fvg_valid_at_stop"].sum()
    r_s = subset["reentry_found"].sum()
    log.info(f"    {st}: {n_s} stops | {v_s} valid ({100*v_s/n_s:.0f}%) | {r_s} re-entries ({100*r_s/n_s:.0f}%)")


# ══════════════════════════════════════════════════════════════════════════
#  PART 2: RE-ENTRY SIMULATION RESULTS
# ══════════════════════════════════════════════════════════════════════════
print_separator("PART 2: RE-ENTRY SIMULATION RESULTS")

reentries = results_df[results_df["reentry_found"]].copy()
if len(reentries) == 0:
    log.info("  No re-entries found. Stopping analysis.")
else:
    log.info(f"  Total re-entries: {len(reentries)}")
    log.info(f"  Hit TP: {reentries['reentry_hit_tp'].sum()} ({100*reentries['reentry_hit_tp'].mean():.1f}%)")
    log.info(f"  Hit SL: {reentries['reentry_hit_sl'].sum()} ({100*reentries['reentry_hit_sl'].mean():.1f}%)")
    timed_out = len(reentries) - reentries["reentry_hit_tp"].sum() - reentries["reentry_hit_sl"].sum()
    log.info(f"  Timed out: {timed_out} ({100*timed_out/len(reentries):.1f}%)")
    log.info("")

    # R-multiple distribution
    valid_r = reentries["reentry_outcome_r"].dropna().values
    metrics = compute_metrics(valid_r, "All Re-entries")
    print_metrics_table([metrics])

    log.info("")
    log.info(f"  Avg bars to re-entry: {reentries['bars_to_reentry'].mean():.1f}")
    log.info(f"  Median bars to re-entry: {reentries['bars_to_reentry'].median():.1f}")
    log.info(f"  Avg bars held in re-entry: {reentries['reentry_bars_held'].mean():.1f}")

    # ── By direction ───────────────────────────────────────────────────
    log.info("")
    log.info("  Re-entry metrics by direction:")
    rows = []
    for d, label in [(1, "Long"), (-1, "Short")]:
        subset = reentries[reentries["direction"] == d]
        if len(subset) > 0:
            r_vals = subset["reentry_outcome_r"].dropna().values
            rows.append(compute_metrics(r_vals, label))
    print_metrics_table(rows)

    # ── By signal type ─────────────────────────────────────────────────
    log.info("")
    log.info("  Re-entry metrics by signal type:")
    rows = []
    for st in ["trend", "mss"]:
        subset = reentries[reentries["signal_type"] == st]
        if len(subset) > 0:
            r_vals = subset["reentry_outcome_r"].dropna().values
            rows.append(compute_metrics(r_vals, st))
    print_metrics_table(rows)

    # ── By year ────────────────────────────────────────────────────────
    log.info("")
    log.info("  Re-entry metrics by year:")
    rows = []
    for yr in sorted(reentries["year"].unique()):
        subset = reentries[reentries["year"] == yr]
        r_vals = subset["reentry_outcome_r"].dropna().values
        rows.append(compute_metrics(r_vals, str(yr)))
    print_metrics_table(rows)


# ══════════════════════════════════════════════════════════════════════════
#  PART 3: COMBINED IMPACT ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════
print_separator("PART 3: COMBINED IMPACT ASSESSMENT")

# Original passing trades (baseline)
original_r = passing["outcome_r"].values
baseline = compute_metrics(original_r, "Baseline (Config H)")

# Re-entry only
if len(reentries) > 0:
    reentry_r = reentries["reentry_outcome_r"].dropna().values
    reentry_only = compute_metrics(reentry_r, "Re-entries only")

    # Combined: original + re-entry (interleaved by time for correct PPDD)
    # Build combined time-sorted R series
    orig_trades = passing[["bar_time_utc", "outcome_r"]].copy()
    orig_trades.columns = ["time", "outcome_r"]
    orig_trades["source"] = "original"

    re_trades = reentries[["sig_time", "reentry_outcome_r"]].copy()
    re_trades.columns = ["time", "outcome_r"]
    re_trades["source"] = "reentry"

    combined = pd.concat([orig_trades, re_trades], ignore_index=True)
    combined = combined.sort_values("time").reset_index(drop=True)
    combined_r = combined["outcome_r"].dropna().values
    combined_metrics = compute_metrics(combined_r, "Combined (orig + re-entry)")

    log.info("")
    print_metrics_table([baseline, reentry_only, combined_metrics])

    # ── Trade count impact ─────────────────────────────────────────────
    log.info("")
    n_years = (passing["bar_time_utc"].max() - passing["bar_time_utc"].min()).days / 365.25
    orig_per_year = len(passing) / n_years
    reentry_per_year = len(reentries) / n_years
    combined_per_year = orig_per_year + reentry_per_year

    log.info(f"  Trade count impact:")
    log.info(f"    Original trades/year:   {orig_per_year:.1f}")
    log.info(f"    Re-entry trades/year:   {reentry_per_year:.1f}")
    log.info(f"    Combined trades/year:   {combined_per_year:.1f}")
    log.info(f"    Increase:               +{reentry_per_year:.1f}/yr (+{100*reentry_per_year/orig_per_year:.1f}%)")

    # ── R per year impact ──────────────────────────────────────────────
    orig_r_per_year = float(np.sum(original_r)) / n_years
    reentry_r_per_year = float(np.sum(reentry_r)) / n_years
    combined_r_per_year = orig_r_per_year + reentry_r_per_year

    log.info("")
    log.info(f"  R/year impact:")
    log.info(f"    Original R/year:        {orig_r_per_year:.2f}")
    log.info(f"    Re-entry R/year:        {reentry_r_per_year:.2f}")
    log.info(f"    Combined R/year:        {combined_r_per_year:.2f}")
    log.info(f"    R increase:             +{reentry_r_per_year:.2f}/yr")

else:
    log.info("  No re-entries to assess combined impact.")


# ══════════════════════════════════════════════════════════════════════════
#  PART 4: RE-ENTRY CLUSTERING & DEPTH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print_separator("PART 4: RE-ENTRY CLUSTERING & DEPTH ANALYSIS")

if len(reentries) > 0:
    # How many stopped trades on same day get re-entries?
    reentries["date"] = pd.to_datetime(reentries["sig_time"]).dt.date
    daily_counts = reentries.groupby("date").size()
    log.info(f"  Days with re-entries:      {len(daily_counts)}")
    log.info(f"  Max re-entries per day:    {daily_counts.max()}")
    log.info(f"  Avg re-entries per day:    {daily_counts.mean():.2f}")
    log.info(f"  Days with 2+ re-entries:   {(daily_counts >= 2).sum()}")

    # Distribution of bars to re-entry
    log.info("")
    log.info("  Bars-to-reentry distribution:")
    bins = [0, 3, 6, 10, 15, 20]
    labels_b = ["1-3", "4-6", "7-10", "11-15", "16-20"]
    reentries["bar_bin"] = pd.cut(reentries["bars_to_reentry"], bins=bins, labels=labels_b, right=True)
    for lbl in labels_b:
        subset = reentries[reentries["bar_bin"] == lbl]
        if len(subset) > 0:
            r_vals = subset["reentry_outcome_r"].dropna().values
            m = compute_metrics(r_vals, f"Bars {lbl}")
            log.info(f"    {lbl}: N={m['N']}, Total_R={m['Total_R']:.2f}, WR={m['WR%']:.1f}%, Avg_R={m['Avg_R']:.3f}")

    # ── FVG size vs re-entry success ───────────────────────────────────
    log.info("")
    log.info("  Re-entry outcome by FVG size quartile:")
    reentries["fvg_q"] = pd.qcut(reentries["fvg_size"], q=4, labels=["Q1 (small)", "Q2", "Q3", "Q4 (large)"], duplicates="drop")
    for q in reentries["fvg_q"].cat.categories:
        subset = reentries[reentries["fvg_q"] == q]
        if len(subset) > 0:
            r_vals = subset["reentry_outcome_r"].dropna().values
            m = compute_metrics(r_vals, str(q))
            log.info(f"    {q}: N={m['N']}, Total_R={m['Total_R']:.2f}, WR={m['WR%']:.1f}%, Avg_R={m['Avg_R']:.3f}")


# ══════════════════════════════════════════════════════════════════════════
#  PART 5: RISK-ADJUSTED RE-ENTRY VARIANTS
# ══════════════════════════════════════════════════════════════════════════
print_separator("PART 5: RISK-ADJUSTED RE-ENTRY VARIANTS")

if len(reentries) > 0:
    reentry_r_full = reentries["reentry_outcome_r"].dropna().values

    # Variant A: Full risk (1R) re-entry
    variant_a = compute_metrics(reentry_r_full, "A: Full risk (1R)")

    # Variant B: Reduced risk (0.5R) re-entry — multiply outcomes by 0.5
    reentry_r_half = reentry_r_full * 0.5
    variant_b = compute_metrics(reentry_r_half, "B: Half risk (0.5R)")

    # Variant C: Longs-only re-entry (shorts are more risky after stop)
    long_reentries = reentries[reentries["direction"] == 1]["reentry_outcome_r"].dropna().values
    variant_c = compute_metrics(long_reentries, "C: Long-only re-entry")

    # Variant D: Only re-enter if bars_to_reentry <= 10 (fresh)
    fresh = reentries[reentries["bars_to_reentry"] <= 10]["reentry_outcome_r"].dropna().values
    variant_d = compute_metrics(fresh, "D: Fresh only (<=10 bars)")

    # Variant E: Only trend signals (not MSS) for re-entry
    trend_re = reentries[reentries["signal_type"] == "trend"]["reentry_outcome_r"].dropna().values
    variant_e = compute_metrics(trend_re, "E: Trend-only re-entry")

    log.info("")
    print_metrics_table([variant_a, variant_b, variant_c, variant_d, variant_e])

    # ── Combined baseline + each variant ───────────────────────────────
    log.info("")
    log.info("  Combined impact (Baseline + variant):")
    log.info("")

    rows_combined = [baseline]
    for label, r_vals in [
        ("A: Full risk", reentry_r_full),
        ("B: Half risk", reentry_r_half),
        ("C: Long-only", long_reentries),
        ("D: Fresh <=10", fresh),
        ("E: Trend-only", trend_re),
    ]:
        # Interleave with original for correct PPDD
        orig_trades2 = passing[["bar_time_utc", "outcome_r"]].copy()
        orig_trades2.columns = ["time", "outcome_r"]

        # Use the subset of reentries appropriate for this variant
        if "Long" in label:
            re_sub = reentries[reentries["direction"] == 1].copy()
        elif "Trend" in label:
            re_sub = reentries[reentries["signal_type"] == "trend"].copy()
        elif "Fresh" in label:
            re_sub = reentries[reentries["bars_to_reentry"] <= 10].copy()
        else:
            re_sub = reentries.copy()

        re_trades2 = re_sub[["sig_time", "reentry_outcome_r"]].copy()
        re_trades2.columns = ["time", "outcome_r"]

        if "Half" in label:
            re_trades2["outcome_r"] = re_trades2["outcome_r"] * 0.5

        comb2 = pd.concat([orig_trades2, re_trades2], ignore_index=True)
        comb2 = comb2.sort_values("time").reset_index(drop=True)
        comb_r = comb2["outcome_r"].dropna().values
        rows_combined.append(compute_metrics(comb_r, f"Baseline + {label}"))

    print_metrics_table(rows_combined)


# ══════════════════════════════════════════════════════════════════════════
#  PART 6: CORRELATED LOSS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print_separator("PART 6: CORRELATED LOSS ANALYSIS")

if len(reentries) > 0:
    # Key question: when the original trade stops out AND the re-entry also stops,
    # what is the combined loss? This is the worst case.
    both_stop = reentries[reentries["reentry_hit_sl"] == True]
    log.info(f"  Original stop + re-entry stop (double loss): {len(both_stop)} / {len(reentries)} "
             f"({100*len(both_stop)/len(reentries):.1f}%)")

    if len(both_stop) > 0:
        # Combined loss: original -1R + re-entry loss
        combined_loss = -1.0 + both_stop["reentry_outcome_r"].values
        log.info(f"  Avg combined loss when both stop: {combined_loss.mean():.3f}R")
        log.info(f"  Max combined loss:                {combined_loss.min():.3f}R")
        log.info(f"  This exceeds 2R daily limit in:   "
                 f"{(combined_loss < -2.0).sum()} cases ({100*(combined_loss < -2.0).mean():.1f}%)")

    # When original stops but re-entry wins
    re_wins = reentries[reentries["reentry_hit_tp"] == True]
    log.info(f"  Original stop + re-entry WIN:     {len(re_wins)} / {len(reentries)} "
             f"({100*len(re_wins)/len(reentries):.1f}%)")

    if len(re_wins) > 0:
        # Net after original -1R + re-entry win
        combined_win = -1.0 + re_wins["reentry_outcome_r"].values
        log.info(f"  Avg net R when re-entry wins:     {combined_win.mean():.3f}R")
        log.info(f"  Cases where re-entry recovers original loss: "
                 f"{(combined_win >= 0).sum()} ({100*(combined_win >= 0).mean():.1f}%)")

    # 0-for-2 interaction: if original stop counts as loss 1, and re-entry
    # stops as loss 2 -> triggers 0-for-2 rule. Is that protective or harmful?
    log.info("")
    log.info("  0-for-2 rule interaction:")
    log.info(f"  If re-entry stop triggers 0-for-2: {len(both_stop)} potential day-stops")
    log.info(f"  Days affected (unique): {both_stop['sig_time'].apply(lambda x: x.date()).nunique()}")


# ══════════════════════════════════════════════════════════════════════════
#  PART 7: FINAL RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════
print_separator("PART 7: RECOMMENDATION")

if len(reentries) > 0:
    re_r = reentries["reentry_outcome_r"].dropna()
    total_re_r = re_r.sum()
    avg_re_r = re_r.mean()
    re_wr = (re_r > 0).mean() * 100

    log.info(f"  Re-entry total R:          {total_re_r:.2f}")
    log.info(f"  Re-entry avg R:            {avg_re_r:.4f}")
    log.info(f"  Re-entry win rate:          {re_wr:.1f}%")
    log.info(f"  Re-entry count:            {len(re_r)} ({len(re_r)/n_years:.1f}/year)")
    log.info("")

    # Decision criteria
    is_positive_ev = avg_re_r > 0
    is_meaningful_count = len(re_r) / n_years >= 5  # at least 5/year
    is_acceptable_wr = re_wr >= 40
    ppdd_acceptable = True  # check combined PPDD

    # Get combined PPDD
    orig_trades_f = passing[["bar_time_utc", "outcome_r"]].copy()
    orig_trades_f.columns = ["time", "outcome_r"]
    re_trades_f = reentries[["sig_time", "reentry_outcome_r"]].copy()
    re_trades_f.columns = ["time", "outcome_r"]
    comb_f = pd.concat([orig_trades_f, re_trades_f], ignore_index=True)
    comb_f = comb_f.sort_values("time").reset_index(drop=True)
    comb_r_f = comb_f["outcome_r"].dropna().values
    comb_m = compute_metrics(comb_r_f, "Combined")

    baseline_ppdd = abs(baseline["PPDD"])
    combined_ppdd = abs(comb_m["PPDD"])
    ppdd_degradation = combined_ppdd - baseline_ppdd

    log.info(f"  Baseline PPDD:             {baseline['PPDD']:.2f}")
    log.info(f"  Combined PPDD:             {comb_m['PPDD']:.2f}")
    log.info(f"  PPDD degradation:          {ppdd_degradation:.2f}R")
    ppdd_acceptable = ppdd_degradation < 2.0  # less than 2R worse

    log.info("")
    log.info(f"  Decision criteria:")
    log.info(f"    Positive EV:             {'YES' if is_positive_ev else 'NO'} (avg_r={avg_re_r:.4f})")
    log.info(f"    Meaningful count (>=5/yr):{'YES' if is_meaningful_count else 'NO'} ({len(re_r)/n_years:.1f}/yr)")
    log.info(f"    Acceptable WR (>=40%):   {'YES' if is_acceptable_wr else 'NO'} ({re_wr:.1f}%)")
    log.info(f"    PPDD acceptable (<2R):   {'YES' if ppdd_acceptable else 'NO'} (delta={ppdd_degradation:.2f}R)")

    if is_positive_ev and is_meaningful_count and is_acceptable_wr and ppdd_acceptable:
        log.info("")
        log.info("  >>> RECOMMENDATION: PROCEED with re-entry implementation")
        log.info("  >>> Suggested config:")
        log.info("  >>>   max_reentries_per_fvg: 1")
        log.info("  >>>   reentry_scan_window: 20 bars")
        log.info("  >>>   reentry_min_body_ratio: 0.50")
        log.info("  >>>   reentry_risk_mult: 0.5 (half risk on re-entry)")
        log.info("  >>>   reentry_max_daily: 1 (prevent compounding losses)")
    else:
        log.info("")
        log.info("  >>> RECOMMENDATION: DO NOT implement re-entry")
        reasons = []
        if not is_positive_ev:
            reasons.append("negative EV")
        if not is_meaningful_count:
            reasons.append("too few trades")
        if not is_acceptable_wr:
            reasons.append("low win rate")
        if not ppdd_acceptable:
            reasons.append(f"PPDD degradation too high ({ppdd_degradation:.1f}R)")
        log.info(f"  >>> Reason: {', '.join(reasons)}")
        log.info("  >>> The stopped-out FVGs that produce re-entries do not have")
        log.info("  >>> sufficient edge to justify the additional risk.")

else:
    log.info("  No re-entries found. Re-entry logic would add no trades.")
    log.info("  >>> RECOMMENDATION: DO NOT implement re-entry")


# ══════════════════════════════════════════════════════════════════════════
#  PART 8: ALTERNATIVE — WIDER SCAN / RELAXED CRITERIA
# ══════════════════════════════════════════════════════════════════════════
print_separator("PART 8: SENSITIVITY — VARYING SCAN WINDOW & BODY RATIO")

log.info("  Testing different scan windows and body ratio thresholds...")
log.info("")

sensitivity_rows = []
for scan_window in [10, 20, 30, 40]:
    for min_br in [0.40, 0.50, 0.60]:
        re_count = 0
        re_outcomes = []

        for _, row in stopped.iterrows():
            sig_time_s = pd.Timestamp(row["bar_time_utc"])
            direction_s = int(row["signal_dir"])
            entry_price_s = float(row["entry_price"])
            fvg_top_s = float(row["fvg_top"])
            fvg_bottom_s = float(row["fvg_bottom"])
            fvg_size_s = float(row["fvg_size_pts"])
            bars_to_stop_s = int(row["bars_to_outcome"])
            irl_target_s = float(row["irl_target"])
            model_stop_s = float(row["model_stop"])

            if sig_time_s not in time_to_idx:
                continue
            sig_idx_s = time_to_idx[sig_time_s]
            stop_bar_s = sig_idx_s + bars_to_stop_s

            if stop_bar_s >= len(df5) - scan_window - 1:
                continue

            # Check FVG validity at stop
            fvg_ok = True
            for j in range(sig_idx_s + 1, min(stop_bar_s + 1, len(df5))):
                if direction_s == 1 and close_arr[j] < fvg_bottom_s:
                    fvg_ok = False
                    break
                if direction_s == -1 and close_arr[j] > fvg_top_s:
                    fvg_ok = False
                    break

            if not fvg_ok:
                continue

            # Scan for re-entry
            for j in range(stop_bar_s + 1, min(stop_bar_s + scan_window + 1, len(df5))):
                if direction_s == 1 and close_arr[j] < fvg_bottom_s:
                    break
                if direction_s == -1 and close_arr[j] > fvg_top_s:
                    break

                bh = high_arr[j]
                bl = low_arr[j]
                bo = open_arr[j]
                bc = close_arr[j]
                br = bh - bl
                if br <= 0:
                    continue
                bb = abs(bc - bo)
                b_ratio = bb / br

                if direction_s == 1:
                    entered = bl <= fvg_top_s and bh >= fvg_bottom_s
                    rejected = bc > fvg_top_s
                    disp = b_ratio >= min_br and bc > bo
                else:
                    entered = bh >= fvg_bottom_s and bl <= fvg_top_s
                    rejected = bc < fvg_bottom_s
                    disp = b_ratio >= min_br and bc < bo

                if entered and rejected and disp and j + 1 < len(df5):
                    re_price = open_arr[j + 1]
                    if direction_s == 1:
                        re_stop_s = fvg_bottom_s - SLIPPAGE_PTS
                        re_entry_s = re_price + SLIPPAGE_PTS
                    else:
                        re_stop_s = fvg_top_s + SLIPPAGE_PTS
                        re_entry_s = re_price - SLIPPAGE_PTS

                    sd = abs(re_entry_s - re_stop_s)
                    if sd < 1.0:
                        break

                    orig_sd = abs(entry_price_s - model_stop_s)
                    orig_td = abs(irl_target_s - entry_price_s)
                    orig_rr = orig_td / orig_sd if orig_sd > 0 else 2.0
                    td = sd * max(orig_rr, 1.0)
                    if direction_s == 1:
                        re_tp_s = re_entry_s + td
                    else:
                        re_tp_s = re_entry_s - td

                    entry_b = j + 1
                    outcome_val = 0.0
                    for k in range(entry_b, min(entry_b + 40, len(df5))):
                        if direction_s == 1:
                            if low_arr[k] <= re_stop_s:
                                outcome_val = (re_stop_s - SLIPPAGE_PTS - re_entry_s) / sd
                                break
                            if high_arr[k] >= re_tp_s:
                                outcome_val = (re_tp_s - re_entry_s) / sd
                                break
                        else:
                            if high_arr[k] >= re_stop_s:
                                outcome_val = (re_entry_s - re_stop_s - SLIPPAGE_PTS) / sd
                                break
                            if low_arr[k] <= re_tp_s:
                                outcome_val = (re_entry_s - re_tp_s) / sd
                                break
                    else:
                        last_k = min(entry_b + 39, len(df5) - 1)
                        if direction_s == 1:
                            outcome_val = (close_arr[last_k] - re_entry_s) / sd
                        else:
                            outcome_val = (re_entry_s - close_arr[last_k]) / sd

                    re_count += 1
                    re_outcomes.append(outcome_val)
                    break

        if re_count > 0:
            outcomes_arr = np.array(re_outcomes)
            m = compute_metrics(outcomes_arr, f"W={scan_window},BR={min_br}")
            sensitivity_rows.append(m)
        else:
            sensitivity_rows.append({
                "Label": f"W={scan_window},BR={min_br}", "N": 0,
                "Total_R": 0.0, "Avg_R": 0.0, "WR%": 0.0,
                "PF": 0.0, "PPDD": 0.0, "MaxDD": 0.0,
            })

print_metrics_table(sensitivity_rows)

log.info("")
log.info("  Done.")
