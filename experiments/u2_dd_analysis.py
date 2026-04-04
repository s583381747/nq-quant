"""U2 Drawdown analysis + reduction strategies."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from experiments.u2_clean import load_all, run_u2_backtest, compute_metrics


def main():
    d = load_all()
    trades, _ = run_u2_backtest(d, stop_strategy="A2", fvg_size_mult=0.3,
                                 max_fvg_age=500, min_stop_pts=5.0)
    longs = [t for t in trades if t["dir"] == 1]
    df = pd.DataFrame(longs)
    r = df["r"].values
    cumr = np.cumsum(r)
    peak = np.maximum.accumulate(cumr)
    dd = peak - cumr

    # === DD ANATOMY ===
    print("=" * 100)
    print("DRAWDOWN ANATOMY (base: 2172 longs, MaxDD=38.0R)")
    print("=" * 100)

    # Find DD episodes > 5R
    dd_episodes = []
    in_dd = False
    dd_start = 0
    for i in range(len(dd)):
        if dd[i] > 0 and not in_dd:
            in_dd = True; dd_start = i
        elif dd[i] == 0 and in_dd:
            mx = dd[dd_start:i].max()
            if mx >= 5:
                dd_episodes.append({"s": dd_start, "e": i, "n": i - dd_start,
                    "dd": mx, "t0": df.iloc[dd_start]["entry_time"],
                    "t1": df.iloc[i-1]["entry_time"]})
            in_dd = False
    if in_dd:
        mx = dd[dd_start:].max()
        if mx >= 5:
            dd_episodes.append({"s": dd_start, "e": len(dd)-1, "n": len(dd)-dd_start,
                "dd": mx, "t0": df.iloc[dd_start]["entry_time"],
                "t1": df.iloc[-1]["entry_time"]})

    dd_episodes.sort(key=lambda x: x["dd"], reverse=True)
    print(f"\nDD episodes > 5R: {len(dd_episodes)}")
    print(f"{'MaxDD':>6} {'Trades':>6} {'Start':>12} {'Duration':>15}")
    for ep in dd_episodes[:10]:
        dur = pd.to_datetime(ep["t1"]) - pd.to_datetime(ep["t0"])
        print(f"{ep['dd']:6.1f}R {ep['n']:6d}t {str(ep['t0'])[:10]:>12} {str(dur.days)}d")

    # Worst episode detail
    w = dd_episodes[0]
    sub = df.iloc[w["s"]:w["e"]]
    print(f"\nWorst DD: {w['dd']:.1f}R over {w['n']} trades")
    print(f"  WR in episode: {100*(sub['r']>0).mean():.0f}%")
    print(f"  Avg winner: {sub[sub['r']>0]['r'].mean():+.2f}R")
    print(f"  Avg loser: {sub[sub['r']<0]['r'].mean():+.2f}R")
    print(f"  Worst single: {sub['r'].min():+.2f}R")

    # Individual trade R distribution
    print(f"\nTrade R distribution:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p:2d}: {np.percentile(r, p):+.2f}R")
    print(f"  Trades < -1.5R: {(r < -1.5).sum()} ({100*(r < -1.5).mean():.1f}%)")

    # Daily P&L
    df["date"] = pd.to_datetime(df["entry_time"]).dt.date
    daily = df.groupby("date")["r"].sum()
    print(f"\nDaily R: mean={daily.mean():+.3f} worst={daily.min():+.1f} best={daily.max():+.1f}")
    print(f"  Days < -3R: {(daily < -3).sum()}")
    print(f"  Days < -2R: {(daily < -2).sum()}")

    # === DD REDUCTION STRATEGIES ===
    print("\n" + "=" * 100)
    print("DD REDUCTION STRATEGIES")
    print("=" * 100)

    # Helper
    def calc_metrics(arr):
        if len(arr) == 0:
            return 0, 0, 0, 0, 0
        total = arr.sum()
        w = arr[arr > 0].sum()
        lo = abs(arr[arr < 0].sum())
        pf = w / lo if lo > 0 else 999
        cr = np.cumsum(arr)
        pk = np.maximum.accumulate(cr)
        mdd = (pk - cr).max()
        ppdd = total / mdd if mdd > 0 else 999
        return total, mdd, ppdd, pf, len(arr)

    # A: DD-based risk scaling
    print("\n--- A: DD-BASED RISK SCALING ---")
    print(f"  {'Config':35s} | {'R':>8} | {'MaxDD':>6} | {'PPDD':>6} | {'PF':>5} | {'Trades':>6}")
    for dd_th, sc in [(5, 0.75), (5, 0.5), (10, 0.75), (10, 0.5), (15, 0.5), (20, 0.5)]:
        adj = np.zeros_like(r)
        ac, ap = 0.0, 0.0
        for i in range(len(r)):
            cd = ap - ac
            adj[i] = r[i] * (sc if cd >= dd_th else 1.0)
            ac += adj[i]
            ap = max(ap, ac)
        t, mdd, ppdd, pf, n = calc_metrics(adj)
        print(f"  DD>{dd_th:2d}R -> {int(sc*100):3d}% risk          | {t:+8.1f} | {mdd:6.1f} | {ppdd:6.2f} | {pf:5.2f} | {n:6d}")

    # Baseline for reference
    t0, mdd0, ppdd0, pf0, n0 = calc_metrics(r)
    print(f"  {'BASELINE (no scaling)':35s} | {t0:+8.1f} | {mdd0:6.1f} | {ppdd0:6.2f} | {pf0:5.2f} | {n0:6d}")

    # B: Daily loss limit
    print("\n--- B: DAILY LOSS LIMIT ---")
    print(f"  {'Config':35s} | {'R':>8} | {'MaxDD':>6} | {'PPDD':>6} | {'PF':>5} | {'Trades':>6}")
    for dlim in [1.0, 1.5, 2.0, 3.0, 999]:
        adj2 = []
        cd2 = None; dp = 0.0; ds = False
        for i in range(len(df)):
            td = df.iloc[i]["date"]
            if td != cd2:
                cd2 = td; dp = 0.0; ds = False
            if ds:
                continue
            adj2.append(r[i])
            dp += r[i]
            if dp <= -dlim:
                ds = True
        ar = np.array(adj2)
        t, mdd, ppdd, pf, n = calc_metrics(ar)
        lab = f"DailyLimit={dlim}R" if dlim < 999 else "DailyLimit=NONE"
        print(f"  {lab:35s} | {t:+8.1f} | {mdd:6.1f} | {ppdd:6.2f} | {pf:5.2f} | {n:6d}")

    # C: Weekly loss limit
    print("\n--- C: WEEKLY LOSS LIMIT ---")
    df["year_week"] = pd.to_datetime(df["entry_time"]).dt.strftime("%Y-W%U")
    print(f"  {'Config':35s} | {'R':>8} | {'MaxDD':>6} | {'PPDD':>6} | {'PF':>5} | {'Trades':>6}")
    for wlim in [3, 5, 8, 10, 999]:
        adj3 = []
        cw = None; wp = 0.0; ws = False
        for i in range(len(df)):
            yw = df.iloc[i]["year_week"]
            if yw != cw:
                cw = yw; wp = 0.0; ws = False
            if ws:
                continue
            adj3.append(r[i])
            wp += r[i]
            if wp <= -wlim:
                ws = True
        ar = np.array(adj3)
        t, mdd, ppdd, pf, n = calc_metrics(ar)
        lab = f"WeeklyLimit={wlim}R" if wlim < 999 else "WeeklyLimit=NONE"
        print(f"  {lab:35s} | {t:+8.1f} | {mdd:6.1f} | {ppdd:6.2f} | {pf:5.2f} | {n:6d}")

    # D: Combined strategies
    print("\n--- D: COMBINED STRATEGIES ---")
    print(f"  {'Config':50s} | {'R':>8} | {'MaxDD':>6} | {'PPDD':>6} | {'PF':>5} | {'t':>5}")
    combos = [
        ("DD>10->50% + Daily2R", 10, 0.5, 2.0, 999),
        ("DD>10->50% + Daily2R + Weekly5R", 10, 0.5, 2.0, 5),
        ("DD>5->50% + Daily2R + Weekly5R", 5, 0.5, 2.0, 5),
        ("DD>10->75% + Daily2R + Weekly5R", 10, 0.75, 2.0, 5),
        ("DD>10->50% + Daily1.5R + Weekly5R", 10, 0.5, 1.5, 5),
        ("DD>15->50% + Daily2R + Weekly8R", 15, 0.5, 2.0, 8),
        ("DD>10->50% + Weekly5R (no daily)", 10, 0.5, 999, 5),
    ]
    for label, dd_th, dd_sc, dlim, wlim in combos:
        adj4 = []
        cd4 = None; dp4 = 0.0; ds4 = False
        cw4 = None; wp4 = 0.0; ws4 = False
        ac4, ap4 = 0.0, 0.0
        for i in range(len(df)):
            td = df.iloc[i]["date"]
            yw = df.iloc[i]["year_week"]
            if td != cd4:
                cd4 = td; dp4 = 0.0; ds4 = False
            if yw != cw4:
                cw4 = yw; wp4 = 0.0; ws4 = False
            if ds4 or ws4:
                continue
            cur_dd = ap4 - ac4
            scale = dd_sc if cur_dd >= dd_th else 1.0
            ri = r[i] * scale
            adj4.append(ri)
            ac4 += ri
            ap4 = max(ap4, ac4)
            dp4 += ri
            wp4 += ri
            if dp4 <= -dlim:
                ds4 = True
            if wp4 <= -wlim:
                ws4 = True
        ar = np.array(adj4)
        t, mdd, ppdd, pf, n = calc_metrics(ar)
        print(f"  {label:50s} | {t:+8.1f} | {mdd:6.1f} | {ppdd:6.2f} | {pf:5.2f} | {n:5d}")

    # Baseline
    print(f"  {'BASELINE':50s} | {t0:+8.1f} | {mdd0:6.1f} | {ppdd0:6.2f} | {pf0:5.2f} | {n0:5d}")

    # === COMBINED BEST WITH PREV-DAY RANGE ===
    print("\n" + "=" * 100)
    print("BEST COMBO + PREV-DAY RANGE FILTER")
    print("=" * 100)

    # Build prev-day range
    nq_h, nq_l = d["h"], d["l"]
    et_frac, dates = d["et_frac_arr"], d["dates"]
    daily_range = {}
    for i in range(d["n"]):
        date = dates[i]
        ef = et_frac[i]
        if 10.0 <= ef < 16.0:
            if date not in daily_range:
                daily_range[date] = {"h": nq_h[i], "l": nq_l[i]}
            else:
                daily_range[date]["h"] = max(daily_range[date]["h"], nq_h[i])
                daily_range[date]["l"] = min(daily_range[date]["l"], nq_l[i])
    sorted_dates = sorted(daily_range.keys())
    prev_day_range = {}
    for idx, date in enumerate(sorted_dates):
        if idx > 0:
            prev = sorted_dates[idx - 1]
            prev_day_range[date] = daily_range[prev]["h"] - daily_range[prev]["l"]

    # Filter longs by prev-day range, then apply best DD combo
    for min_range in [0, 100]:
        if min_range > 0:
            filtered = [t for t in longs
                        if pd.to_datetime(t["entry_time"]).date() in prev_day_range
                        and prev_day_range[pd.to_datetime(t["entry_time"]).date()] >= min_range]
        else:
            filtered = longs

        df2 = pd.DataFrame(filtered)
        r2 = df2["r"].values
        df2["date"] = pd.to_datetime(df2["entry_time"]).dt.date
        df2["year_week"] = pd.to_datetime(df2["entry_time"]).dt.strftime("%Y-W%U")

        # Apply DD>10->50% + Daily2R + Weekly5R
        adj = []
        cd = None; dp = 0.0; ds = False
        cw = None; wp = 0.0; wss = False
        ac, ap = 0.0, 0.0
        for i in range(len(df2)):
            td = df2.iloc[i]["date"]
            yw = df2.iloc[i]["year_week"]
            if td != cd:
                cd = td; dp = 0.0; ds = False
            if yw != cw:
                cw = yw; wp = 0.0; wss = False
            if ds or wss:
                continue
            cur_dd = ap - ac
            scale = 0.5 if cur_dd >= 10 else 1.0
            ri = r2[i] * scale
            adj.append(ri)
            ac += ri; ap = max(ap, ac)
            dp += ri; wp += ri
            if dp <= -2.0: ds = True
            if wp <= -5.0: wss = True

        ar = np.array(adj)
        t, mdd, ppdd, pf, n = calc_metrics(ar)
        tpd = n / (252 * 10.5)
        rng_label = f"prevRange>={min_range}pt" if min_range > 0 else "no range filter"
        print(f"  {rng_label} + DD>10->50% + D2R + W5R: {n}t {tpd:.2f}/d R={t:+.1f} MaxDD={mdd:.1f}R PPDD={ppdd:.2f} PF={pf:.2f}")


if __name__ == "__main__":
    main()
