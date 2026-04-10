"""
experiments/random_baseline_1m.py — Random Entry Baseline Test
==============================================================

Purpose: Verify FVG zone edge is real by comparing real engine performance
against random entries with identical trade management.

If random entries produce similar PF/R, the edge is from trade management,
not from FVG zone selection.

Method:
  1. Run real engine: run_hybrid_1m() to get baseline trades
  2. Run N random permutations:
     - Same number of trades as real engine
     - Random 1m bars during NY session (10:00-16:00 ET)
     - Same long/short ratio as real trades
     - Same stop distance distribution (sampled from real trades)
     - Same trade management: trim 25%, BE, trail (5th swing), worst-case
     - Actual bar-by-bar 1m execution (NOT theoretical)
  3. Compare real PF/R vs average random PF/R
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
sys.path.insert(0, str(PROJECT))

from experiments.chain_engine import (
    load_all, _find_nth_swing, compute_metrics, pr,
)
from experiments.unified_engine_1m import run_hybrid_1m


# ======================================================================
# Random trade simulator — bar-by-bar on 1m
# ======================================================================
def simulate_random_trade(
    i_entry: int,
    direction: int,
    stop_dist: float,
    contracts: int,
    o1: np.ndarray,
    h1: np.ndarray,
    l1: np.ndarray,
    c1: np.ndarray,
    n1: int,
    et_frac1: np.ndarray,
    dates1: np.ndarray,
    swing_high_mask_5m: np.ndarray,
    swing_low_mask_5m: np.ndarray,
    swing_high_price_5m: np.ndarray,
    swing_low_price_5m: np.ndarray,
    map_1m_5m: np.ndarray,
    *,
    trim_pct: float = 0.25,
    fixed_tp_r: float = 1.0,
    nth_swing: int = 5,
    slip: float = 0.25,
    comm: float = 0.62,
    point_value: float = 2.0,
    normal_r: float = 1000.0,
    worst_case_trim_be: bool = True,
) -> dict | None:
    """Simulate a single trade on 1m bars from entry bar forward.

    Uses the SAME exit logic as the real engine:
      - Stop at entry - stop_dist (long) / entry + stop_dist (short)
      - TP1 at entry + stop_dist * fixed_tp_r (long) / entry - stop_dist * fixed_tp_r (short)
      - Trim trim_pct at TP1, move to BE
      - Trail remaining on nth swing
      - EOD close at 15:55+ ET
      - Axiom 9: worst-case trim+BE on same bar

    Returns a trade dict or None if entry bar is invalid.
    """
    if i_entry >= n1 - 5:
        return None

    entry_price = c1[i_entry]  # enter at close of the bar
    entry_date = dates1[i_entry]

    if direction == 1:
        stop_price = entry_price - stop_dist
        tp1_price = entry_price + stop_dist * fixed_tp_r
    else:
        stop_price = entry_price + stop_dist
        tp1_price = entry_price - stop_dist * fixed_tp_r

    trimmed = False
    be_stop = 0.0
    trail_stop = 0.0
    remaining = contracts
    trim_contracts = max(1, int(contracts * trim_pct))

    # Bar-by-bar execution starting from next bar
    for i in range(i_entry + 1, n1):
        # EOD close
        if dates1[i] != entry_date or et_frac1[i] >= 15.917:
            ex_price = c1[i] - 0.25 if direction == 1 else c1[i] + 0.25
            return _build_trade(
                i_entry, i, entry_price, ex_price, stop_price, tp1_price,
                direction, contracts, remaining, trimmed, "eod_close",
                slip, comm, point_value, normal_r
            )

        i_5m = map_1m_5m[i]

        if direction == 1:
            # Effective stop
            eff = trail_stop if trimmed and trail_stop > 0 else stop_price
            if trimmed and be_stop > 0:
                eff = max(eff, be_stop)

            # Check stop hit
            if l1[i] <= eff:
                ex_price = eff - slip
                reason = "be_sweep" if (trimmed and eff >= be_stop and be_stop > 0) else "stop"
                return _build_trade(
                    i_entry, i, entry_price, ex_price, stop_price, tp1_price,
                    direction, contracts, remaining, trimmed, reason,
                    slip, comm, point_value, normal_r
                )

            # Check TP1
            if not trimmed and h1[i] >= tp1_price:
                remaining = contracts - trim_contracts
                trimmed = True
                be_stop = entry_price  # BE at entry

                if remaining <= 0:
                    return _build_trade(
                        i_entry, i, entry_price, tp1_price, stop_price, tp1_price,
                        direction, contracts, contracts, False, "tp1",
                        slip, comm, point_value, normal_r
                    )

                # Initialize trail
                trail_stop = _find_nth_swing(
                    swing_low_mask_5m, swing_low_price_5m, i_5m, nth_swing)
                if np.isnan(trail_stop) or trail_stop <= 0:
                    trail_stop = be_stop

                # Axiom 9: worst-case same-bar BE check
                if worst_case_trim_be and l1[i] <= be_stop:
                    ex_price = be_stop - slip
                    return _build_trade(
                        i_entry, i, entry_price, ex_price, stop_price, tp1_price,
                        direction, contracts, remaining, True, "be_sweep",
                        slip, comm, point_value, normal_r
                    )

            # Trail update
            if trimmed:
                nt = _find_nth_swing(swing_low_mask_5m, swing_low_price_5m, i_5m, nth_swing)
                if not np.isnan(nt) and nt > trail_stop:
                    trail_stop = nt

        else:  # direction == -1
            # Effective stop
            eff = trail_stop if trimmed and trail_stop > 0 else stop_price
            if trimmed and be_stop > 0:
                if eff > be_stop:
                    eff = be_stop

            # Check stop hit
            if h1[i] >= eff:
                ex_price = eff + slip
                reason = "be_sweep" if (trimmed and eff <= be_stop and be_stop > 0) else "stop"
                return _build_trade(
                    i_entry, i, entry_price, ex_price, stop_price, tp1_price,
                    direction, contracts, remaining, trimmed, reason,
                    slip, comm, point_value, normal_r
                )

            # Check TP1
            if not trimmed and l1[i] <= tp1_price:
                remaining = contracts - trim_contracts
                trimmed = True
                be_stop = entry_price  # BE at entry

                if remaining <= 0:
                    return _build_trade(
                        i_entry, i, entry_price, tp1_price, stop_price, tp1_price,
                        direction, contracts, contracts, False, "tp1",
                        slip, comm, point_value, normal_r
                    )

                # Initialize trail
                nt_init = _find_nth_swing(
                    swing_high_mask_5m, swing_high_price_5m, i_5m, nth_swing)
                if np.isnan(nt_init) or nt_init <= 0 or nt_init > entry_price:
                    trail_stop = be_stop
                else:
                    trail_stop = nt_init

                # Axiom 9: worst-case same-bar BE check
                if worst_case_trim_be and h1[i] >= be_stop:
                    ex_price = be_stop + slip
                    return _build_trade(
                        i_entry, i, entry_price, ex_price, stop_price, tp1_price,
                        direction, contracts, remaining, True, "be_sweep",
                        slip, comm, point_value, normal_r
                    )

            # Trail update
            if trimmed:
                nt = _find_nth_swing(swing_high_mask_5m, swing_high_price_5m, i_5m, nth_swing)
                if not np.isnan(nt) and nt > 0 and nt < trail_stop:
                    trail_stop = nt

    # Reached end of data without exit — force close
    ex_price = c1[n1 - 1] - 0.25 if direction == 1 else c1[n1 - 1] + 0.25
    return _build_trade(
        i_entry, n1 - 1, entry_price, ex_price, stop_price, tp1_price,
        direction, contracts, remaining, trimmed, "force_close",
        slip, comm, point_value, normal_r
    )


def _build_trade(
    i_entry: int, i_exit: int,
    entry_price: float, ex_price: float,
    stop_price: float, tp1_price: float,
    direction: int, contracts: int, remaining: int,
    trimmed: bool, reason: str,
    slip: float, comm: float, point_value: float, normal_r: float,
) -> dict:
    """Build trade dict with PnL calculation matching the real engine."""
    pnl_pts = (ex_price - entry_price) * direction
    ex_contracts = remaining

    if trimmed and reason != "tp1":
        trim_c = contracts - remaining
        trim_pnl = (tp1_price - entry_price) * direction * point_value * trim_c
        total_pnl = trim_pnl + pnl_pts * point_value * ex_contracts
        total_pnl -= comm * 2 * contracts
    else:
        total_pnl = pnl_pts * point_value * ex_contracts
        total_pnl -= comm * 2 * ex_contracts

    r = total_pnl / normal_r if normal_r > 0 else 0.0
    sd = abs(entry_price - stop_price)

    return {
        "i_entry": i_entry,
        "i_exit": i_exit,
        "entry_price": entry_price,
        "exit_price": ex_price,
        "stop_price": stop_price,
        "tp1_price": tp1_price,
        "r": r,
        "reason": reason,
        "dir": direction,
        "trimmed": trimmed,
        "pnl_dollars": total_pnl,
        "stop_dist_pts": sd,
        "contracts": contracts,
    }


# ======================================================================
# Run one random permutation
# ======================================================================
def run_random_permutation(
    n_trades: int,
    long_ratio: float,
    stop_dist_pool: np.ndarray,
    contracts_pool: np.ndarray,
    ny_bar_indices: np.ndarray,
    o1: np.ndarray,
    h1: np.ndarray,
    l1: np.ndarray,
    c1: np.ndarray,
    n1: int,
    et_frac1: np.ndarray,
    dates1: np.ndarray,
    swing_high_mask_5m: np.ndarray,
    swing_low_mask_5m: np.ndarray,
    swing_high_price_5m: np.ndarray,
    swing_low_price_5m: np.ndarray,
    map_1m_5m: np.ndarray,
    rng: np.random.Generator,
    **kwargs,
) -> list[dict]:
    """Run one random permutation of trades."""
    trades = []
    n_long = int(n_trades * long_ratio)
    directions = np.array([1] * n_long + [-1] * (n_trades - n_long))
    rng.shuffle(directions)

    # Sample random entry bars (with replacement to avoid running out)
    entry_bars = rng.choice(ny_bar_indices, size=n_trades, replace=True)

    # Sample stop distances and contracts from real distribution
    stop_dists = rng.choice(stop_dist_pool, size=n_trades, replace=True)
    contracts = rng.choice(contracts_pool, size=n_trades, replace=True)

    for j in range(n_trades):
        trade = simulate_random_trade(
            int(entry_bars[j]),
            int(directions[j]),
            float(stop_dists[j]),
            int(contracts[j]),
            o1, h1, l1, c1, n1,
            et_frac1, dates1,
            swing_high_mask_5m, swing_low_mask_5m,
            swing_high_price_5m, swing_low_price_5m,
            map_1m_5m,
            **kwargs,
        )
        if trade is not None:
            trades.append(trade)

    return trades


# ======================================================================
# Main
# ======================================================================
def main():
    N_PERMUTATIONS = 20

    print("=" * 120)
    print("RANDOM ENTRY BASELINE TEST")
    print("=" * 120)
    print(f"Purpose: Compare real FVG zone entries vs {N_PERMUTATIONS} random entry permutations")
    print(f"Trade management is IDENTICAL: trim 25%, BE, trail 5th swing, worst-case Axiom 9")
    print()

    # ---- Load data ----
    t0 = _time.perf_counter()
    d5 = load_all()
    print("Loading 1m data...")
    nq1 = pd.read_parquet(DATA / "NQ_1min_10yr.parquet")
    print(f"1m loaded: {len(nq1):,} bars ({_time.perf_counter() - t0:.1f}s)")

    # ---- Run real engine ----
    print("\nRunning REAL engine (run_hybrid_1m)...")
    t1 = _time.perf_counter()
    real_trades, stats = run_hybrid_1m(d5, nq1, trend_r_mult=0.5, worst_case_trim_be=True)
    print(f"Real engine: {len(real_trades)} trades in {_time.perf_counter() - t1:.1f}s")

    real_metrics = compute_metrics(real_trades)
    print()
    pr("REAL ENGINE", real_metrics)

    # ---- Extract distributions from real trades ----
    n_real = len(real_trades)
    long_count = sum(1 for t in real_trades if t["dir"] == 1)
    long_ratio = long_count / n_real if n_real > 0 else 0.5
    stop_dist_pool = np.array([t["stop_dist_pts"] for t in real_trades])
    contracts_pool = np.array([t["contracts"] for t in real_trades])

    print(f"\n  Real trade stats:")
    print(f"    Trades:       {n_real}")
    print(f"    Long ratio:   {long_ratio:.1%}")
    print(f"    Stop dist:    mean={stop_dist_pool.mean():.1f}  median={np.median(stop_dist_pool):.1f}  "
          f"std={stop_dist_pool.std():.1f}  min={stop_dist_pool.min():.1f}  max={stop_dist_pool.max():.1f}")
    print(f"    Contracts:    mean={contracts_pool.mean():.1f}  median={np.median(contracts_pool):.0f}  "
          f"min={contracts_pool.min()}  max={contracts_pool.max()}")

    # ---- Build NY session bar index ----
    # Compute 1m time arrays
    et1 = nq1.index.tz_convert("US/Eastern")
    et_frac1 = np.array([et1[j].hour + et1[j].minute / 60.0 for j in range(len(nq1))])
    dates1 = np.array([
        (et1[j] + pd.Timedelta(days=1)).date() if et1[j].hour >= 18 else et1[j].date()
        for j in range(len(nq1))
    ])

    # Only bars in NY session 10:00 - 15:50 ET (leave room for exits before EOD)
    ny_mask = (et_frac1 >= 10.0) & (et_frac1 < 15.833)
    ny_bar_indices = np.where(ny_mask)[0]
    print(f"    NY session bars (10:00-15:50): {len(ny_bar_indices):,}")

    # ---- Prepare 1m arrays ----
    n1 = len(nq1)
    o1 = nq1["open"].values
    h1 = nq1["high"].values
    l1 = nq1["low"].values
    c1 = nq1["close"].values

    # 5m-to-1m mapping and swing arrays from d5
    from experiments.unified_engine_1m import build_5m_to_1m_map
    map_1m_5m = build_5m_to_1m_map(d5["nq"], nq1)

    swing_high_mask_5m = d5["swing_high_mask"]
    swing_low_mask_5m = d5["swing_low_mask"]
    swing_high_price_5m = d5["swing_high_price_at_mask"]
    swing_low_price_5m = d5["swing_low_price_at_mask"]

    # Config from real engine
    params = d5["params"]
    comm = params["backtest"]["commission_per_side_micro"]
    slip = params["backtest"]["slippage_normal_ticks"] * 0.25
    point_value = params["position"]["point_value"]
    normal_r = params["position"]["normal_r"]

    sim_kwargs = dict(
        trim_pct=0.25,
        fixed_tp_r=1.0,
        nth_swing=5,
        slip=slip,
        comm=comm,
        point_value=point_value,
        normal_r=normal_r,
        worst_case_trim_be=True,
    )

    # ---- Run random permutations ----
    print(f"\n{'=' * 120}")
    print(f"Running {N_PERMUTATIONS} random permutations ({n_real} trades each)...")
    print(f"{'=' * 120}")

    random_results = []
    rng = np.random.default_rng(seed=42)

    for perm in range(N_PERMUTATIONS):
        t_perm = _time.perf_counter()
        rand_trades = run_random_permutation(
            n_trades=n_real,
            long_ratio=long_ratio,
            stop_dist_pool=stop_dist_pool,
            contracts_pool=contracts_pool,
            ny_bar_indices=ny_bar_indices,
            o1=o1, h1=h1, l1=l1, c1=c1, n1=n1,
            et_frac1=et_frac1, dates1=dates1,
            swing_high_mask_5m=swing_high_mask_5m,
            swing_low_mask_5m=swing_low_mask_5m,
            swing_high_price_5m=swing_high_price_5m,
            swing_low_price_5m=swing_low_price_5m,
            map_1m_5m=map_1m_5m,
            rng=rng,
            **sim_kwargs,
        )
        m = compute_metrics(rand_trades)
        random_results.append(m)
        elapsed = _time.perf_counter() - t_perm
        pr(f"  Perm {perm+1:2d}/{N_PERMUTATIONS}", m)

    # ---- Summary ----
    print(f"\n{'=' * 120}")
    print("SUMMARY: REAL ENGINE vs RANDOM BASELINE")
    print(f"{'=' * 120}")

    rand_rs = [m["R"] for m in random_results]
    rand_pfs = [m["PF"] for m in random_results]
    rand_ppdds = [m["PPDD"] for m in random_results]
    rand_wrs = [m["WR"] for m in random_results]
    rand_maxdds = [m["MaxDD"] for m in random_results]

    print(f"\n  {'Metric':12s} | {'REAL':>10s} | {'Random Mean':>11s} | {'Random Std':>10s} | {'Random Min':>10s} | {'Random Max':>10s} | {'Edge':>10s}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*11}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    def fmt(val, decimals=1):
        return f"{val:+.{decimals}f}" if decimals == 1 else f"{val:.{decimals}f}"

    real_r = real_metrics["R"]
    real_pf = real_metrics["PF"]
    real_ppdd = real_metrics["PPDD"]
    real_wr = real_metrics["WR"]
    real_maxdd = real_metrics["MaxDD"]

    mean_r = np.mean(rand_rs)
    mean_pf = np.mean(rand_pfs)
    mean_ppdd = np.mean(rand_ppdds)
    mean_wr = np.mean(rand_wrs)
    mean_maxdd = np.mean(rand_maxdds)

    print(f"  {'R':12s} | {real_r:+10.1f} | {mean_r:+11.1f} | {np.std(rand_rs):10.1f} | {min(rand_rs):+10.1f} | {max(rand_rs):+10.1f} | {real_r - mean_r:+10.1f}")
    print(f"  {'PF':12s} | {real_pf:10.2f} | {mean_pf:11.2f} | {np.std(rand_pfs):10.2f} | {min(rand_pfs):10.2f} | {max(rand_pfs):10.2f} | {real_pf - mean_pf:+10.2f}")
    print(f"  {'PPDD':12s} | {real_ppdd:10.2f} | {mean_ppdd:11.2f} | {np.std(rand_ppdds):10.2f} | {min(rand_ppdds):10.2f} | {max(rand_ppdds):10.2f} | {real_ppdd - mean_ppdd:+10.2f}")
    print(f"  {'WR%':12s} | {real_wr:10.1f} | {mean_wr:11.1f} | {np.std(rand_wrs):10.1f} | {min(rand_wrs):10.1f} | {max(rand_wrs):10.1f} | {real_wr - mean_wr:+10.1f}")
    print(f"  {'MaxDD':12s} | {real_maxdd:10.1f} | {mean_maxdd:11.1f} | {np.std(rand_maxdds):10.1f} | {min(rand_maxdds):10.1f} | {max(rand_maxdds):10.1f} | {real_maxdd - mean_maxdd:+10.1f}")

    # Statistical significance: how many random perms beat real?
    n_beat_r = sum(1 for r in rand_rs if r >= real_r)
    n_beat_pf = sum(1 for pf in rand_pfs if pf >= real_pf)

    print(f"\n  Statistical edge:")
    print(f"    Random perms that beat real R:  {n_beat_r}/{N_PERMUTATIONS} ({n_beat_r/N_PERMUTATIONS*100:.0f}%)")
    print(f"    Random perms that beat real PF: {n_beat_pf}/{N_PERMUTATIONS} ({n_beat_pf/N_PERMUTATIONS*100:.0f}%)")

    # Empirical p-value
    p_val_r = (n_beat_r + 1) / (N_PERMUTATIONS + 1)
    p_val_pf = (n_beat_pf + 1) / (N_PERMUTATIONS + 1)
    print(f"    Empirical p-value (R):  {p_val_r:.3f}")
    print(f"    Empirical p-value (PF): {p_val_pf:.3f}")

    # Verdict
    print(f"\n  {'=' * 80}")
    if n_beat_r == 0 and n_beat_pf == 0:
        print(f"  VERDICT: FVG ZONE EDGE IS REAL")
        print(f"  Real engine outperforms ALL {N_PERMUTATIONS} random permutations on both R and PF.")
        print(f"  The edge comes from FVG zone SELECTION, not just trade management.")
    elif n_beat_r <= 1 and n_beat_pf <= 1:
        print(f"  VERDICT: STRONG EVIDENCE of FVG zone edge")
        print(f"  Real engine outperforms {N_PERMUTATIONS - max(n_beat_r, n_beat_pf)}/{N_PERMUTATIONS} random permutations.")
    elif mean_pf > 1.0 and real_pf > mean_pf * 1.3:
        print(f"  VERDICT: FVG zone edge EXISTS but trade management also contributes")
        print(f"  Random entries are profitable (PF={mean_pf:.2f}) but real is {(real_pf/mean_pf - 1)*100:.0f}% better.")
    elif mean_pf > real_pf * 0.9:
        print(f"  VERDICT: WEAK/NO FVG zone edge — trade management drives most of the PnL")
        print(f"  Random entries achieve {mean_pf/real_pf*100:.0f}% of real PF.")
    else:
        print(f"  VERDICT: MIXED — further analysis needed")
    print(f"  {'=' * 80}")

    total_elapsed = _time.perf_counter() - t0
    print(f"\n  Total elapsed: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
