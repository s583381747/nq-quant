"""
experiments/rebuild_cache_v4.py — Rebuild signal cache v4 with current params and run backtest.

Context: entry.min_fvg_atr_mult was just added to params.yaml at 0.3.
The old v3 cache was built with stricter params (~1.0-1.5 mult).
This script rebuilds with current params, runs backtest, and compares.

Usage: python experiments/rebuild_cache_v4.py
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
    stream=sys.stdout,
)
logger = logging.getLogger("rebuild_cache_v4")

DATA = PROJECT / "data"


# ============================================================
# STEP 1: Rebuild signal cache
# ============================================================
def rebuild_cache():
    """Rebuild signal cache v4 with current params.yaml."""
    print("\n" + "=" * 70)
    print("STEP 1: REBUILD SIGNAL CACHE v4")
    print("=" * 70)

    from features.entry_signals import detect_all_signals

    t0 = _time.perf_counter()
    nq = pd.read_parquet(DATA / "NQ_5m_10yr.parquet")
    with open(PROJECT / "config" / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    print(f"Loaded NQ 5m 10yr: {len(nq)} bars")
    print(f"Current entry params: {params.get('entry', {})}")
    print(f"Detecting signals...")

    signals = detect_all_signals(nq, params)

    elapsed = _time.perf_counter() - t0
    n_total = int(signals["signal"].sum())
    n_trend = int((signals["signal"] & (signals["signal_type"] == "trend")).sum())
    n_mss = int((signals["signal"] & (signals["signal_type"] == "mss")).sum())

    print(f"\nCache v4 built in {elapsed:.1f}s:")
    print(f"  Total signals: {n_total}")
    print(f"  Trend: {n_trend}")
    print(f"  MSS: {n_mss}")

    # Save
    out_path = DATA / "cache_signals_10yr_v4.parquet"
    signals.to_parquet(out_path)
    print(f"  Saved to: {out_path}")

    # Compare with v3
    v3_path = DATA / "cache_signals_10yr_v3.parquet"
    if v3_path.exists():
        sig3 = pd.read_parquet(v3_path)
        n3_total = int(sig3["signal"].sum())
        n3_trend = int((sig3["signal"] & (sig3["signal_type"] == "trend")).sum())
        n3_mss = int((sig3["signal"] & (sig3["signal_type"] == "mss")).sum())
        print(f"\n  v3 comparison:")
        print(f"    v3: {n3_total} total ({n3_trend} trend, {n3_mss} mss)")
        print(f"    v4: {n_total} total ({n_trend} trend, {n_mss} mss)")
        print(f"    Delta: {n_total - n3_total:+d} total ({n_trend - n3_trend:+d} trend, {n_mss - n3_mss:+d} mss)")

    return signals


# ============================================================
# STEP 2: Run backtest (same approach as validate_nt_logic.py)
# ============================================================
def run_backtest_with_cache(sig_cache, label="v4"):
    """Run full backtest using given signal cache + SMT gate."""
    print(f"\n{'=' * 70}")
    print(f"BACKTEST WITH {label} CACHE")
    print(f"{'=' * 70}")

    t0 = _time.perf_counter()

    nq = pd.read_parquet(DATA / "NQ_5m_10yr.parquet")
    es = pd.read_parquet(DATA / "ES_5m_10yr.parquet")
    bias = pd.read_parquet(DATA / "cache_bias_10yr_v2.parquet")
    regime = pd.read_parquet(DATA / "cache_regime_10yr_v2.parquet")
    with open(PROJECT / "config" / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # Compute SMT
    from features.smt import compute_smt
    smt = compute_smt(nq, es, {
        "swing": {"left_bars": 3, "right_bars": 1},
        "smt": {"sweep_lookback": 15, "time_tolerance": 1},
    })

    # SMT gate for MSS
    ss = sig_cache.copy()
    mm = ss["signal"].astype(bool) & (ss["signal_type"] == "mss")
    mi = ss.index[mm]
    ss.loc[mi, "signal"] = False
    ss.loc[mi, "signal_dir"] = 0
    ss["has_smt"] = False

    c = mi.intersection(smt.index)
    if len(c) > 0:
        md = sig_cache.loc[c, "signal_dir"].values
        ok = ((md == 1) & smt.loc[c, "smt_bull"].values.astype(bool)) | \
             ((md == -1) & smt.loc[c, "smt_bear"].values.astype(bool))
        g = c[ok]
        ss.loc[g, "signal"] = sig_cache.loc[g, "signal"]
        ss.loc[g, "signal_dir"] = sig_cache.loc[g, "signal_dir"]
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

    n_after_smt = int(ss["signal"].sum())
    print(f"After SMT gate: {n_after_smt} signals")

    # Run engine with threshold=0.0 (rules-based)
    class Dummy:
        def predict(self, d):
            return np.ones(d.num_row(), dtype=np.float32)

    dummy_X = pd.DataFrame({"d": np.zeros(len(nq))}, index=nq.index)

    from backtest.engine import run_backtest as _run_bt
    trades = _run_bt(nq, ss, bias, regime["regime"], Dummy(), dummy_X, params, threshold=0.0)

    elapsed = _time.perf_counter() - t0
    print(f"Backtest done in {elapsed:.1f}s")

    return trades, params, ss


def compute_metrics(trades, label=""):
    """Compute standard metrics from a trades DataFrame."""
    if len(trades) == 0:
        return {"label": label, "trades": 0, "R": 0, "PPDD": 0, "PF": 0, "WR": 0, "MaxDD": 0}

    r_col = "r_multiple" if "r_multiple" in trades.columns else "r"
    r_vals = trades[r_col].values
    total_r = float(r_vals.sum())
    wr = float((r_vals > 0).mean() * 100)

    # Max drawdown (in R)
    cumr = np.cumsum(r_vals)
    peak = np.maximum.accumulate(cumr)
    dd = peak - cumr
    max_dd = float(dd.max())

    # Profit factor
    wins = r_vals[r_vals > 0].sum()
    losses = abs(r_vals[r_vals < 0].sum())
    pf = float(wins / losses) if losses > 0 else 999.0

    # PPDD (peak profit / peak drawdown)
    ppdd = float(total_r / max_dd) if max_dd > 0 else 999.0

    return {
        "label": label,
        "trades": len(trades),
        "R": round(total_r, 1),
        "PPDD": round(ppdd, 2),
        "PF": round(pf, 2),
        "WR": round(wr, 1),
        "MaxDD": round(max_dd, 1),
    }


def print_comparison_table(metrics_list):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"{'COMPARISON TABLE':^80}")
    print(f"{'='*80}")
    header = f"{'Config':<40} {'Trades':>7} {'R':>8} {'PPDD':>7} {'PF':>6} {'WR%':>6} {'MaxDD':>7}"
    print(header)
    print("-" * 80)
    for m in metrics_list:
        row = f"{m['label']:<40} {m['trades']:>7d} {m['R']:>8.1f} {m['PPDD']:>7.2f} {m['PF']:>6.2f} {m['WR']:>5.1f}% {m['MaxDD']:>7.1f}"
        print(row)
    print("=" * 80)


def yearly_breakdown(trades, label=""):
    """Print yearly breakdown."""
    if len(trades) == 0:
        print(f"  {label}: No trades")
        return

    r_col = "r_multiple" if "r_multiple" in trades.columns else "r"
    time_col = "entry_time" if "entry_time" in trades.columns else trades.index.name

    if "entry_time" in trades.columns:
        trades = trades.copy()
        if hasattr(trades["entry_time"].iloc[0], "year"):
            trades["_year"] = pd.to_datetime(trades["entry_time"]).dt.year
        else:
            trades["_year"] = pd.to_datetime(trades["entry_time"]).dt.year
    else:
        trades = trades.copy()
        trades["_year"] = 2020  # fallback

    print(f"\nYEARLY BREAKDOWN ({label}):")
    for year, grp in trades.groupby("_year"):
        r_vals = grp[r_col].values
        n = len(grp)
        wr = (r_vals > 0).mean() * 100
        total_r = r_vals.sum()
        cumr = np.cumsum(r_vals)
        peak = np.maximum.accumulate(cumr)
        dd = peak - cumr
        max_dd = dd.max()
        ppdd = total_r / max_dd if max_dd > 0 else 999.0
        print(f"  {year}: n={n:>4d}, WR={wr:>5.1f}%, R={total_r:>+7.1f}, "
              f"MaxDD={max_dd:>5.1f}R, PPDD={ppdd:>5.2f}")


# ============================================================
# STEP 3: Run alternative filter configs
# ============================================================
def run_with_config_override(sig_cache, overrides, label=""):
    """Run backtest with param overrides."""
    nq = pd.read_parquet(DATA / "NQ_5m_10yr.parquet")
    es = pd.read_parquet(DATA / "ES_5m_10yr.parquet")
    bias = pd.read_parquet(DATA / "cache_bias_10yr_v2.parquet")
    regime = pd.read_parquet(DATA / "cache_regime_10yr_v2.parquet")
    with open(PROJECT / "config" / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # Apply overrides (nested)
    for key, val in overrides.items():
        parts = key.split(".")
        d = params
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val

    # SMT gate
    from features.smt import compute_smt
    smt = compute_smt(nq, es, {
        "swing": {"left_bars": 3, "right_bars": 1},
        "smt": {"sweep_lookback": 15, "time_tolerance": 1},
    })

    ss = sig_cache.copy()
    mm = ss["signal"].astype(bool) & (ss["signal_type"] == "mss")
    mi = ss.index[mm]
    ss.loc[mi, "signal"] = False
    ss.loc[mi, "signal_dir"] = 0
    ss["has_smt"] = False

    c = mi.intersection(smt.index)
    if len(c) > 0:
        md = sig_cache.loc[c, "signal_dir"].values
        ok = ((md == 1) & smt.loc[c, "smt_bull"].values.astype(bool)) | \
             ((md == -1) & smt.loc[c, "smt_bear"].values.astype(bool))
        g = c[ok]
        ss.loc[g, "signal"] = sig_cache.loc[g, "signal"]
        ss.loc[g, "signal_dir"] = sig_cache.loc[g, "signal_dir"]
        ss.loc[g, "has_smt"] = True

    # Kill MSS overnight
    rem = ss["signal"].astype(bool) & (ss["signal_type"] == "mss")
    mi2 = ss.index[rem]
    if len(mi2) > 0:
        et = mi2.tz_convert("US/Eastern")
        ef = et.hour + et.minute / 60.0
        kill = (ef >= 16.0) | (ef < 3.0)
        if kill.any():
            ss.loc[mi2[kill], ["signal", "signal_dir"]] = [False, 0]

    class Dummy:
        def predict(self, d):
            return np.ones(d.num_row(), dtype=np.float32)

    dummy_X = pd.DataFrame({"d": np.zeros(len(nq))}, index=nq.index)

    from backtest.engine import run_backtest as _run_bt
    trades = _run_bt(nq, ss, bias, regime["regime"], Dummy(), dummy_X, params, threshold=0.0)

    return trades


# ============================================================
# MAIN
# ============================================================
def main():
    t_total = _time.perf_counter()

    # Step 1: Rebuild cache v4
    sig_v4 = rebuild_cache()

    # Step 2: Run backtest with v4 cache (current filters)
    trades_v4, params, ss_v4 = run_backtest_with_cache(sig_v4, label="v4")
    m_v4 = compute_metrics(trades_v4, "A) v4 current filters")

    # Also run v3 for reference
    v3_path = DATA / "cache_signals_10yr_v3.parquet"
    if v3_path.exists():
        sig_v3 = pd.read_parquet(v3_path)
        trades_v3, _, _ = run_backtest_with_cache(sig_v3, label="v3")
        m_v3 = compute_metrics(trades_v3, "BASELINE: v3 (old cache)")
    else:
        m_v3 = {"label": "BASELINE: v3 (no file)", "trades": 0, "R": 0, "PPDD": 0, "PF": 0, "WR": 0, "MaxDD": 0}

    results = [m_v3, m_v4]

    # Step 3: Print raw signal comparison
    print(f"\n{'='*70}")
    print("RAW SIGNAL COMPARISON")
    print(f"{'='*70}")
    if v3_path.exists():
        sig_v3 = pd.read_parquet(v3_path)
        print(f"  v3 raw signals: {sig_v3['signal'].sum()}")
    print(f"  v4 raw signals: {sig_v4['signal'].sum()}")
    print(f"  v4 after SMT gate: {ss_v4['signal'].sum()}")

    # Step 4: Alternative filter configs (only if v4 has more signals)
    n_v4_total = int(sig_v4["signal"].sum())
    if n_v4_total > 20000:  # significantly more signals
        print(f"\n{'='*70}")
        print("STEP 5: ALTERNATIVE FILTER CONFIGS")
        print(f"{'='*70}")

        # Config B: More permissive SQ thresholds
        print("\nRunning config B: SQ 0.60/0.75...")
        trades_b = run_with_config_override(sig_v4, {
            "signal_quality.threshold": 0.60,
            "dual_mode.long_sq_threshold": 0.60,
            "dual_mode.short_sq_threshold": 0.75,
        }, label="B) SQ 0.60/0.75")
        m_b = compute_metrics(trades_b, "B) v4 SQ 0.60/0.75")
        results.append(m_b)

        # Config C: Open London/Asia for MSS+SMT
        print("\nRunning config C: London/Asia open for MSS+SMT...")
        trades_c = run_with_config_override(sig_v4, {
            "session_filter.skip_london": False,
            "session_filter.skip_asia": False,
        }, label="C) London/Asia open")
        m_c = compute_metrics(trades_c, "C) v4 London/Asia open")
        results.append(m_c)

        # Config D: Both B + C combined
        print("\nRunning config D: SQ 0.60/0.75 + London/Asia open...")
        trades_d = run_with_config_override(sig_v4, {
            "signal_quality.threshold": 0.60,
            "dual_mode.long_sq_threshold": 0.60,
            "dual_mode.short_sq_threshold": 0.75,
            "session_filter.skip_london": False,
            "session_filter.skip_asia": False,
        }, label="D) Both B+C")
        m_d = compute_metrics(trades_d, "D) v4 SQ+London/Asia")
        results.append(m_d)

    # Print final comparison
    print_comparison_table(results)

    # Yearly breakdown for best config
    print("\n" + "=" * 70)
    print("YEARLY BREAKDOWN")
    print("=" * 70)
    yearly_breakdown(trades_v4, "v4 current filters")
    if v3_path.exists():
        yearly_breakdown(trades_v3, "v3 baseline")

    # By signal type
    r_col = "r_multiple" if "r_multiple" in trades_v4.columns else "r"
    if "signal_type" in trades_v4.columns:
        print(f"\nBY SIGNAL TYPE (v4):")
        for st, grp in trades_v4.groupby("signal_type"):
            rv = grp[r_col].values
            print(f"  {st}: n={len(grp)}, WR={100*(rv>0).mean():.1f}%, R={rv.sum():+.1f}")

    elapsed = _time.perf_counter() - t_total
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
