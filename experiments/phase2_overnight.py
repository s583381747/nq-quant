"""
experiments/phase2_overnight.py
Phase 2 experiment: Open Asia/London sessions for trading.

Context: Current config skips all Asia & London signals.
The bar-by-bar engine found overnight trades are +250R with PPDD=18.14,
but that was from an unvalidated signal set. Now we test on the VALIDATED
Python pipeline (cache_signals_10yr_v3 + SMT + engine.py filters).

Variants:
  a. Baseline: skip_asia=True, skip_london=True (current ~534 trades)
  b. Open Asia: skip_asia=False, skip_london=True
  c. Open both: skip_asia=False, skip_london=False
  d. Open both + MSS only overnight (trend=NY only, MSS=24hr)
  e. Open both + higher SQ overnight (SQ=0.80 for non-NY sessions)
  f. Open both + long only overnight

For best variants: yearly breakdown, session breakdown, walk-forward.

Usage: python experiments/phase2_overnight.py
"""

import sys
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# ============================================================
# 1. Load data (same as validate_nt_logic.py)
# ============================================================
print("=" * 80)
print("PHASE 2 OVERNIGHT EXPERIMENT")
print("=" * 80)
print("\nLoading data...")

nq = pd.read_parquet(PROJECT / "data" / "NQ_5m_10yr.parquet")
es = pd.read_parquet(PROJECT / "data" / "ES_5m_10yr.parquet")
sig3 = pd.read_parquet(PROJECT / "data" / "cache_signals_10yr_v3.parquet")
bias = pd.read_parquet(PROJECT / "data" / "cache_bias_10yr_v2.parquet")
regime = pd.read_parquet(PROJECT / "data" / "cache_regime_10yr_v2.parquet")

with open(PROJECT / "config" / "params.yaml", encoding="utf-8") as f:
    params_base = yaml.safe_load(f)

print(f"  NQ 5m: {len(nq):,} bars, {nq.index[0]} to {nq.index[-1]}")
print(f"  Signals v3: {sig3['signal'].sum():,} raw signals")

# ============================================================
# 2. Compute SMT divergence
# ============================================================
print("\nComputing SMT divergence...")
from features.smt import compute_smt
smt = compute_smt(nq, es, {'swing': {'left_bars': 3, 'right_bars': 1},
                             'smt': {'sweep_lookback': 15, 'time_tolerance': 1}})
print(f"  SMT bull: {smt['smt_bull'].sum():,}, bear: {smt['smt_bear'].sum():,}")


# ============================================================
# 3. CRITICAL CHECK: Signal distribution by session
# ============================================================
print("\n" + "=" * 80)
print("CRITICAL CHECK: Signal distribution in cache_signals_10yr_v3")
print("=" * 80)

sig_mask_raw = sig3['signal'].astype(bool)
et_all = sig3.index[sig_mask_raw].tz_convert('US/Eastern')
hours_all = et_all.hour + et_all.minute / 60.0

asia_raw = ((hours_all >= 18.0) | (hours_all < 3.0)).sum()
london_raw = ((hours_all >= 3.0) & (hours_all < 9.5)).sum()
ny_raw = ((hours_all >= 9.5) & (hours_all < 16.0)).sum()
other_raw = len(et_all) - asia_raw - london_raw - ny_raw

print(f"\nRaw signals in cache (before SMT/filters):")
print(f"  Total:   {sig_mask_raw.sum():,}")
print(f"  Asia:    {asia_raw:,} ({100*asia_raw/sig_mask_raw.sum():.1f}%)")
print(f"  London:  {london_raw:,} ({100*london_raw/sig_mask_raw.sum():.1f}%)")
print(f"  NY:      {ny_raw:,} ({100*ny_raw/sig_mask_raw.sum():.1f}%)")
print(f"  Other:   {other_raw:,}")

# By signal type
for stype in ['trend', 'mss']:
    m = sig_mask_raw & (sig3['signal_type'] == stype)
    if m.sum() == 0:
        continue
    et_s = sig3.index[m].tz_convert('US/Eastern')
    h_s = et_s.hour + et_s.minute / 60.0
    a_s = ((h_s >= 18.0) | (h_s < 3.0)).sum()
    l_s = ((h_s >= 3.0) & (h_s < 9.5)).sum()
    n_s = ((h_s >= 9.5) & (h_s < 16.0)).sum()
    print(f"\n  {stype.upper()} signals:")
    print(f"    Total={m.sum():,}, Asia={a_s:,}, London={l_s:,}, NY={n_s:,}")
    # Direction breakdown
    for d, dname in [(1, 'Long'), (-1, 'Short')]:
        md = m & (sig3['signal_dir'] == d)
        if md.sum() > 0:
            et_d = sig3.index[md].tz_convert('US/Eastern')
            h_d = et_d.hour + et_d.minute / 60.0
            a_d = ((h_d >= 18.0) | (h_d < 3.0)).sum()
            l_d = ((h_d >= 3.0) & (h_d < 9.5)).sum()
            n_d = ((h_d >= 9.5) & (h_d < 16.0)).sum()
            print(f"    {dname}: Total={md.sum():,}, Asia={a_d:,}, London={l_d:,}, NY={n_d:,}")


# ============================================================
# 4. Build merged signals (v3 cache + SMT gate for MSS)
#    Same logic as validate_nt_logic.py, but WITHOUT the overnight MSS kill
#    — we want to test what happens if we allow overnight MSS.
# ============================================================
def make_sig_full(sig3_s, smt_s, kill_overnight_mss=True):
    """Build signal df with SMT gating for MSS signals.

    kill_overnight_mss: if True, kill MSS between 16:00-03:00 ET (original behavior).
                        if False, allow MSS signals 24hr.
    """
    s = sig3_s.copy()
    # Start by removing ALL MSS signals, then re-add those with SMT confirmation
    mm = s['signal'].astype(bool) & (s['signal_type'] == 'mss')
    mi = s.index[mm]
    s.loc[mi, 'signal'] = False
    s.loc[mi, 'signal_dir'] = 0
    s['has_smt'] = False

    c = mi.intersection(smt_s.index)
    if len(c) == 0:
        return s

    md = sig3_s.loc[c, 'signal_dir'].values
    ok = ((md == 1) & smt_s.loc[c, 'smt_bull'].values.astype(bool)) | \
         ((md == -1) & smt_s.loc[c, 'smt_bear'].values.astype(bool))
    g = c[ok]
    s.loc[g, 'signal'] = sig3_s.loc[g, 'signal']
    s.loc[g, 'signal_dir'] = sig3_s.loc[g, 'signal_dir']
    s.loc[g, 'has_smt'] = True

    # Optionally kill MSS in overnight (16:00-03:00 ET)
    if kill_overnight_mss:
        rem = s['signal'].astype(bool) & (s['signal_type'] == 'mss')
        mi2 = s.index[rem]
        if len(mi2) > 0:
            et = mi2.tz_convert('US/Eastern')
            ef = et.hour + et.minute / 60.0
            kill = (ef >= 16.0) | (ef < 3.0)
            if kill.any():
                s.loc[mi2[kill], ['signal', 'signal_dir']] = [False, 0]

    return s


# Build two signal sets: one with overnight MSS killed (baseline), one without
ss_baseline = make_sig_full(sig3, smt, kill_overnight_mss=True)
ss_24hr_mss = make_sig_full(sig3, smt, kill_overnight_mss=False)

print(f"\nSignals after SMT gating:")
print(f"  Baseline (kill overnight MSS): {ss_baseline['signal'].sum():,}")
print(f"  24hr MSS: {ss_24hr_mss['signal'].sum():,}")


# ============================================================
# 5. Run engine with custom params
# ============================================================
from backtest.engine import run_backtest


class DummyModel:
    """Dummy model that always predicts 1.0 (all signals pass threshold)."""
    def predict(self, d):
        return np.ones(d.num_row(), dtype=np.float32)


dummy_model = DummyModel()
dummy_X = pd.DataFrame({"d": np.zeros(len(nq))}, index=nq.index)


def run_variant(name, params_override, signals_df=None, description=""):
    """Run a backtest variant with overridden params."""
    p = copy.deepcopy(params_base)

    # Apply overrides (nested dict merge)
    for key, val in params_override.items():
        if isinstance(val, dict) and key in p:
            p[key].update(val)
        else:
            p[key] = val

    sig = signals_df if signals_df is not None else ss_baseline

    trades = run_backtest(nq, sig, bias, regime["regime"], dummy_model, dummy_X, p, threshold=0.0)
    return trades


def compute_metrics(trades_df, name=""):
    """Compute standard metrics from trade DataFrame."""
    if len(trades_df) == 0:
        return {"name": name, "trades": 0, "R": 0, "PPDD": 0, "MaxDD": 0,
                "WR": 0, "PF": 0, "avgR": 0}

    r_col = 'r_multiple' if 'r_multiple' in trades_df.columns else 'r'
    r_arr = trades_df[r_col].values
    cum_r = np.cumsum(r_arr)
    running_max = np.maximum.accumulate(cum_r)
    drawdowns = running_max - cum_r
    max_dd = drawdowns.max() if len(drawdowns) > 0 else 0

    total_r = r_arr.sum()
    wins = r_arr[r_arr > 0]
    losses = r_arr[r_arr < 0]
    wr = len(wins) / len(r_arr) * 100 if len(r_arr) > 0 else 0
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    ppdd = total_r / max_dd if max_dd > 0 else float('inf')
    avg_r = r_arr.mean()

    return {
        "name": name,
        "trades": len(trades_df),
        "R": round(total_r, 2),
        "PPDD": round(ppdd, 2),
        "MaxDD": round(max_dd, 2),
        "WR": round(wr, 1),
        "PF": round(pf, 2),
        "avgR": round(avg_r, 4),
    }


def print_metrics_table(results):
    """Print a formatted comparison table."""
    header = f"{'Variant':<45} {'Trades':>7} {'R':>8} {'PPDD':>7} {'MaxDD':>7} {'WR%':>6} {'PF':>6} {'avgR':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        ppdd_str = f"{r['PPDD']:.2f}" if r['PPDD'] != float('inf') else "inf"
        pf_str = f"{r['PF']:.2f}" if r['PF'] != float('inf') else "inf"
        print(f"{r['name']:<45} {r['trades']:>7} {r['R']:>8.1f} {ppdd_str:>7} {r['MaxDD']:>7.1f} {r['WR']:>5.1f}% {pf_str:>6} {r['avgR']:>8.4f}")


# ============================================================
# 6. Define and run all variants
# ============================================================
print("\n" + "=" * 80)
print("RUNNING VARIANTS")
print("=" * 80)

variants = {}

# --- A. Baseline (current config) ---
print("\n[A] Baseline: skip_asia=True, skip_london=True")
trades_a = run_variant("A_Baseline", {})
variants["A"] = trades_a
print(f"    => {len(trades_a)} trades, R={trades_a['r_multiple'].sum():.1f}")

# --- B. Open Asia only ---
print("\n[B] Open Asia: skip_asia=False, skip_london=True")
trades_b = run_variant("B_OpenAsia", {
    "session_filter": {"skip_asia": False, "skip_london": True, "enabled": True}
})
variants["B"] = trades_b
print(f"    => {len(trades_b)} trades, R={trades_b['r_multiple'].sum():.1f}")

# --- C. Open both ---
print("\n[C] Open both: skip_asia=False, skip_london=False")
trades_c = run_variant("C_OpenBoth", {
    "session_filter": {"skip_asia": False, "skip_london": False, "enabled": True}
})
variants["C"] = trades_c
print(f"    => {len(trades_c)} trades, R={trades_c['r_multiple'].sum():.1f}")

# --- D. Open both + MSS 24hr (trend=NY only, MSS=24hr) ---
# For this we need to modify the session filter more carefully.
# We use the 24hr MSS signal set and keep skip_asia/london=True for trend signals.
# The trick: MSS+SMT already has bypass_session_filter=True in params.
# So with the 24hr MSS signals (not killed overnight) + skip_asia/london=True,
# MSS+SMT will bypass the session filter but trend signals won't.
print("\n[D] Open both + MSS-only overnight (trend=NY, MSS=24hr)")
trades_d = run_variant("D_MSS24hr", {
    "session_filter": {"skip_asia": True, "skip_london": True, "enabled": True},
    "smt": {"bypass_session_filter": True},
}, signals_df=ss_24hr_mss)
variants["D"] = trades_d
print(f"    => {len(trades_d)} trades, R={trades_d['r_multiple'].sum():.1f}")

# --- E. Open both + higher SQ overnight (SQ=0.80 for non-NY) ---
# We implement this by running with open sessions but manually raising the SQ threshold.
# Since engine.py doesn't have session-specific SQ thresholds, we need a custom approach.
# Strategy: Run open-both, then post-filter: keep all NY trades + overnight trades with SQ >= 0.80.
# Better: We use a custom param set where the base SQ threshold is raised to 0.80.
# But that would also affect NY trades. Instead, we'll do two runs and merge.
print("\n[E] Open both + higher SQ overnight (SQ>=0.80 non-NY)")
# Run with everything open but SQ=0.80 globally, then compare.
# Actually - the cleanest approach: run with open sessions at SQ=0.80 for longs too.
# Since dual_mode already sets short SQ=0.82, we just need long_sq_threshold=0.80 too for overnight.
# The engine doesn't support session-specific SQ. So we'll do a post-hoc analysis:
# Run variant C (open both), tag each trade by session, filter overnight trades by SQ.
# For now, run with a global SQ=0.80 for all directions.
trades_e = run_variant("E_HighSQ_Overnight", {
    "session_filter": {"skip_asia": False, "skip_london": False, "enabled": True},
    "signal_quality": {"threshold": 0.80},
    "dual_mode": {"long_sq_threshold": 0.80, "short_sq_threshold": 0.82},
})
variants["E"] = trades_e
print(f"    => {len(trades_e)} trades, R={trades_e['r_multiple'].sum():.1f}")

# --- F. Open both + long only overnight ---
# Asia and London only allow longs.
print("\n[F] Open both + long-only overnight")
trades_f = run_variant("F_LongOnlyOvernight", {
    "session_filter": {
        "skip_asia": False, "skip_london": False, "enabled": True,
        "london_direction": 1,  # long only in London
    },
    # For Asia direction, we need a param. Engine uses skip_asia only.
    # Let's check: when skip_asia=False and sf_enabled=True, there's no asia_direction param.
    # Engine line 730-731: else (Asia) → if skip_asia: continue. No direction filter.
    # So we need to add custom handling. For now, we'll modify differently.
})
variants["F"] = trades_f
print(f"    => {len(trades_f)} trades, R={trades_f['r_multiple'].sum():.1f}")


# ============================================================
# 7. Summary comparison table
# ============================================================
print("\n" + "=" * 80)
print("VARIANT COMPARISON")
print("=" * 80)
print()

results = []
for key in ["A", "B", "C", "D", "E", "F"]:
    names = {
        "A": "A. Baseline (skip Asia+London)",
        "B": "B. Open Asia only",
        "C": "C. Open both sessions",
        "D": "D. MSS-only overnight (Trend=NY)",
        "E": "E. Open both + SQ>=0.80 globally",
        "F": "F. Open both + London long-only",
    }
    results.append(compute_metrics(variants[key], names[key]))

print_metrics_table(results)


# ============================================================
# 8. Session breakdown for each variant
# ============================================================
print("\n" + "=" * 80)
print("SESSION BREAKDOWN (per variant)")
print("=" * 80)


def session_breakdown(trades_df, name=""):
    """Break down trades by session (entry time)."""
    if len(trades_df) == 0:
        return

    et = trades_df['entry_time'].dt.tz_convert('US/Eastern')
    h = et.dt.hour + et.dt.minute / 60.0

    trades_df = trades_df.copy()
    trades_df['session'] = 'other'
    trades_df.loc[(h >= 18.0) | (h < 3.0), 'session'] = 'Asia'
    trades_df.loc[(h >= 3.0) & (h < 9.5), 'session'] = 'London'
    trades_df.loc[(h >= 9.5) & (h < 16.0), 'session'] = 'NY'

    r_col = 'r_multiple' if 'r_multiple' in trades_df.columns else 'r'

    print(f"\n  {name}:")
    for sess in ['Asia', 'London', 'NY', 'other']:
        sub = trades_df[trades_df['session'] == sess]
        if len(sub) == 0:
            continue
        r = sub[r_col]
        wr = 100 * (r > 0).mean()
        print(f"    {sess:>8}: {len(sub):>5} trades, R={r.sum():>7.1f}, "
              f"avgR={r.mean():>7.4f}, WR={wr:>5.1f}%")

    return trades_df


for key, name in [("A", "Baseline"), ("B", "Open Asia"), ("C", "Open Both"),
                   ("D", "MSS-only overnight"), ("E", "High SQ global"), ("F", "Long-only overnight")]:
    session_breakdown(variants[key], name)


# ============================================================
# 9. Yearly breakdown for promising variants (PPDD > 5)
# ============================================================
print("\n" + "=" * 80)
print("YEARLY BREAKDOWN (variants with PPDD > 5)")
print("=" * 80)


def yearly_breakdown(trades_df, name=""):
    """Yearly metrics."""
    if len(trades_df) == 0:
        return

    r_col = 'r_multiple' if 'r_multiple' in trades_df.columns else 'r'
    trades_df = trades_df.copy()
    trades_df['year'] = trades_df['entry_time'].dt.year

    print(f"\n  {name}:")
    header = f"    {'Year':>6} {'Trades':>7} {'R':>8} {'WR%':>6} {'avgR':>8} {'MaxDD':>7}"
    print(header)
    print("    " + "-" * (len(header) - 4))

    for year in sorted(trades_df['year'].unique()):
        sub = trades_df[trades_df['year'] == year]
        r = sub[r_col].values
        wr = 100 * (r > 0).mean()
        cum_r = np.cumsum(r)
        running_max = np.maximum.accumulate(cum_r)
        dd = (running_max - cum_r).max() if len(r) > 0 else 0
        print(f"    {year:>6} {len(sub):>7} {r.sum():>8.1f} {wr:>5.1f}% {r.mean():>8.4f} {dd:>7.1f}")


# Check which variants have PPDD > 5
for key, name in [("A", "Baseline"), ("B", "Open Asia"), ("C", "Open Both"),
                   ("D", "MSS-only overnight"), ("E", "High SQ global"), ("F", "Long-only overnight")]:
    m = compute_metrics(variants[key])
    if m['PPDD'] > 5 or key == "A":  # Always show baseline
        yearly_breakdown(variants[key], f"{key}. {name} (PPDD={m['PPDD']:.2f})")


# ============================================================
# 10. Direction breakdown for overnight trades
# ============================================================
print("\n" + "=" * 80)
print("DIRECTION BREAKDOWN (Long vs Short) — Variant C (Open Both)")
print("=" * 80)

if len(variants["C"]) > 0:
    tc = variants["C"].copy()
    r_col = 'r_multiple' if 'r_multiple' in tc.columns else 'r'
    et = tc['entry_time'].dt.tz_convert('US/Eastern')
    h = et.dt.hour + et.dt.minute / 60.0
    tc['session'] = 'NY'
    tc.loc[(h >= 18.0) | (h < 3.0), 'session'] = 'Asia'
    tc.loc[(h >= 3.0) & (h < 9.5), 'session'] = 'London'

    dir_col = 'direction' if 'direction' in tc.columns else 'dir'
    type_col = 'signal_type' if 'signal_type' in tc.columns else 'type'

    for sess in ['Asia', 'London', 'NY']:
        sub = tc[tc['session'] == sess]
        if len(sub) == 0:
            continue
        print(f"\n  {sess}:")
        for d, dname in [(1, 'Long'), (-1, 'Short')]:
            ds = sub[sub[dir_col] == d]
            if len(ds) == 0:
                continue
            r = ds[r_col]
            wr = 100 * (r > 0).mean()
            print(f"    {dname:>6}: {len(ds):>5} trades, R={r.sum():>7.1f}, avgR={r.mean():>7.4f}, WR={wr:>5.1f}%")
            # By signal type
            for t in ds[type_col].unique():
                ts = ds[ds[type_col] == t]
                rt = ts[r_col]
                wrt = 100 * (rt > 0).mean()
                print(f"           {t}: {len(ts):>4} trades, R={rt.sum():>7.1f}, avgR={rt.mean():>7.4f}, WR={wrt:>5.1f}%")


# ============================================================
# 11. Walk-forward validation for best variants
# ============================================================
print("\n" + "=" * 80)
print("WALK-FORWARD VALIDATION (expanding window, 1-year OOS)")
print("=" * 80)


def walk_forward(trades_df, name="", min_is_years=3):
    """Expanding window walk-forward: train on [start..year-1], test on year."""
    if len(trades_df) == 0:
        return

    r_col = 'r_multiple' if 'r_multiple' in trades_df.columns else 'r'
    trades_df = trades_df.copy()
    trades_df['year'] = trades_df['entry_time'].dt.year
    years = sorted(trades_df['year'].unique())

    if len(years) < min_is_years + 1:
        print(f"  {name}: Not enough years for WF ({len(years)} years, need {min_is_years + 1})")
        return

    print(f"\n  {name}:")
    header = f"    {'OOS Year':>10} {'IS trades':>10} {'IS R':>8} {'IS avgR':>8} {'OOS trades':>11} {'OOS R':>8} {'OOS avgR':>9} {'OOS WR%':>8}"
    print(header)
    print("    " + "-" * (len(header) - 4))

    oos_rs = []
    oos_wins = 0
    oos_total = 0

    for i, year in enumerate(years):
        if i < min_is_years:
            continue  # Need at least min_is_years of IS data

        is_years = years[:i]
        is_trades = trades_df[trades_df['year'].isin(is_years)]
        oos_trades = trades_df[trades_df['year'] == year]

        if len(is_trades) == 0 or len(oos_trades) == 0:
            continue

        is_r = is_trades[r_col].values
        oos_r = oos_trades[r_col].values
        is_avg = is_r.mean()
        oos_avg = oos_r.mean()
        oos_wr = 100 * (oos_r > 0).mean()

        oos_rs.extend(oos_r)
        oos_wins += (oos_r > 0).sum()
        oos_total += len(oos_r)

        print(f"    {year:>10} {len(is_trades):>10} {is_r.sum():>8.1f} {is_avg:>8.4f} "
              f"{len(oos_trades):>11} {oos_r.sum():>8.1f} {oos_avg:>9.4f} {oos_wr:>7.1f}%")

    if oos_total > 0:
        oos_arr = np.array(oos_rs)
        cum_r = np.cumsum(oos_arr)
        max_dd = (np.maximum.accumulate(cum_r) - cum_r).max()
        total_r = oos_arr.sum()
        ppdd = total_r / max_dd if max_dd > 0 else float('inf')
        print(f"\n    WF Summary: {oos_total} OOS trades, R={total_r:.1f}, "
              f"avgR={oos_arr.mean():.4f}, WR={100*oos_wins/oos_total:.1f}%, "
              f"MaxDD={max_dd:.1f}, PPDD={ppdd:.2f}")


for key, name in [("A", "Baseline"), ("B", "Open Asia"), ("C", "Open Both"),
                   ("D", "MSS-only overnight"), ("E", "High SQ global"), ("F", "Long-only overnight")]:
    m = compute_metrics(variants[key])
    if m['PPDD'] > 5 or key == "A":
        walk_forward(variants[key], f"{key}. {name}")


# ============================================================
# 12. Overnight signal quality analysis
# ============================================================
print("\n" + "=" * 80)
print("OVERNIGHT SIGNAL QUALITY ANALYSIS")
print("=" * 80)

# For variant C (open both), tag trades by session and compute quality metrics
if len(variants["C"]) > 0:
    tc = variants["C"].copy()
    r_col = 'r_multiple' if 'r_multiple' in tc.columns else 'r'
    et = tc['entry_time'].dt.tz_convert('US/Eastern')
    h = et.dt.hour + et.dt.minute / 60.0
    tc['session'] = 'NY'
    tc.loc[(h >= 18.0) | (h < 3.0), 'session'] = 'Asia'
    tc.loc[(h >= 3.0) & (h < 9.5), 'session'] = 'London'

    print("\nExit reason breakdown by session (Variant C):")
    exit_col = 'exit_reason' if 'exit_reason' in tc.columns else 'reason'
    for sess in ['Asia', 'London', 'NY']:
        sub = tc[tc['session'] == sess]
        if len(sub) == 0:
            continue
        print(f"\n  {sess}:")
        for reason in sub[exit_col].unique():
            rs = sub[sub[exit_col] == reason]
            r = rs[r_col]
            print(f"    {reason:>15}: {len(rs):>5} trades, R={r.sum():>7.1f}, avgR={r.mean():>7.4f}")

    # Grade breakdown by session
    if 'grade' in tc.columns:
        print("\nGrade breakdown by session (Variant C):")
        for sess in ['Asia', 'London', 'NY']:
            sub = tc[tc['session'] == sess]
            if len(sub) == 0:
                continue
            print(f"\n  {sess}:")
            for grade in ['A+', 'B+', 'C']:
                gs = sub[sub['grade'] == grade]
                if len(gs) == 0:
                    continue
                r = gs[r_col]
                wr = 100 * (r > 0).mean()
                print(f"    {grade:>3}: {len(gs):>5} trades, R={r.sum():>7.1f}, avgR={r.mean():>7.4f}, WR={wr:>5.1f}%")


# ============================================================
# 13. Variant G: Hybrid — Best overnight config
# ============================================================
print("\n" + "=" * 80)
print("VARIANT G: Hybrid best-of experiments")
print("=" * 80)

# Based on session breakdowns above, try a hybrid:
# - MSS signals allowed 24hr (bypass session filter)
# - Trend signals: NY only (skip Asia/London)
# - This is basically variant D but with open sessions for MSS bypass
print("\n[G] MSS 24hr + Trend NY-only (same as D but explicit)")
trades_g = run_variant("G_Hybrid", {
    "session_filter": {"skip_asia": True, "skip_london": True, "enabled": True},
    "smt": {"bypass_session_filter": True},
}, signals_df=ss_24hr_mss)
variants["G"] = trades_g
m_g = compute_metrics(trades_g, "G. MSS 24hr + Trend NY-only")
print(f"    => {m_g['trades']} trades, R={m_g['R']}, PPDD={m_g['PPDD']}")

# Variant H: Open both, but raise SQ to 0.75 (compromise)
print("\n[H] Open both + SQ>=0.75 globally")
trades_h = run_variant("H_SQ75", {
    "session_filter": {"skip_asia": False, "skip_london": False, "enabled": True},
    "signal_quality": {"threshold": 0.75},
    "dual_mode": {"long_sq_threshold": 0.75, "short_sq_threshold": 0.82},
})
variants["H"] = trades_h
m_h = compute_metrics(trades_h, "H. Open both + SQ>=0.75")
print(f"    => {m_h['trades']} trades, R={m_h['R']}, PPDD={m_h['PPDD']}")

# Variant I: Open both, NY TP mult = 2.0, overnight TP mult = 1.0 (conservative)
# Engine applies NY TP mult only for 9.5 <= et_frac < 16.0, so overnight already gets 1.0x
print("\n[I] Open both + conservative overnight (no TP scaling outside NY)")
trades_i = run_variant("I_ConservativeON", {
    "session_filter": {"skip_asia": False, "skip_london": False, "enabled": True},
})
# This is same as C since TP mult already only applies to NY. Let's try without session_regime too.
variants["I"] = trades_i
m_i = compute_metrics(trades_i, "I. Same as C (TP scaling already NY-only)")
print(f"    => {m_i['trades']} trades, R={m_i['R']}, PPDD={m_i['PPDD']}")


# ============================================================
# 14. Final summary with all variants
# ============================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY — ALL VARIANTS")
print("=" * 80)
print()

all_results = []
all_names = {
    "A": "A. Baseline (skip Asia+London)",
    "B": "B. Open Asia only",
    "C": "C. Open both sessions",
    "D": "D. MSS-only overnight (Trend=NY)",
    "E": "E. Open both + SQ>=0.80 globally",
    "F": "F. Open both + London long-only",
    "G": "G. MSS 24hr + Trend NY-only",
    "H": "H. Open both + SQ>=0.75 globally",
}
for key in ["A", "B", "C", "D", "E", "F", "G", "H"]:
    all_results.append(compute_metrics(variants[key], all_names[key]))

print_metrics_table(all_results)

# Highlight best
best = max(all_results, key=lambda x: x['PPDD'] if x['PPDD'] != float('inf') else 0)
print(f"\nBest PPDD: {best['name']} (PPDD={best['PPDD']:.2f})")
best_r = max(all_results, key=lambda x: x['R'])
print(f"Most R:    {best_r['name']} (R={best_r['R']:.1f})")
most_trades = max(all_results, key=lambda x: x['trades'])
print(f"Most trades: {most_trades['name']} ({most_trades['trades']} trades)")


# ============================================================
# 15. Incremental overnight R contribution
# ============================================================
print("\n" + "=" * 80)
print("INCREMENTAL R FROM OVERNIGHT (vs Baseline)")
print("=" * 80)

baseline_r = compute_metrics(variants["A"])["R"]
baseline_t = compute_metrics(variants["A"])["trades"]

for key in ["B", "C", "D", "E", "F", "G", "H"]:
    m = compute_metrics(variants[key])
    delta_r = m["R"] - baseline_r
    delta_t = m["trades"] - baseline_t
    print(f"  {all_names[key]:>45}: +{delta_t:>4} trades, {'+' if delta_r>=0 else ''}{delta_r:>7.1f} R")


print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)
