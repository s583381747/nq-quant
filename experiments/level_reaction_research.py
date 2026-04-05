"""
experiments/level_reaction_research.py -- Level Reaction Research
================================================================

Core question: When price touches a significant level, does it REACT?
And does the reaction quality predict subsequent FVG trade quality?

This is NOT "was there a sweep before FVG?"
This IS "did price show a behavior change at a significant level,
and does that behavior change create a better FVG?"

Research steps:
  1. For each significant level (PDL, PDH, session H/L, HTF swings),
     detect every time price TOUCHES the level
  2. Classify the reaction: strong bounce, weak bounce, wick-only, or breakdown
  3. Build a database of level-touch events with reaction quality
  4. Link each FVG to its nearest preceding level reaction
  5. Test: do FVGs born from strong reactions perform better?
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.u2_clean import load_all, compute_metrics, pr
from experiments.sweep_research import compute_pdhl, compute_htf_swings, compute_htf_swings_4h
from features.swing import detect_swing_lows, detect_swing_highs


# ======================================================================
# Level touch detection + reaction classification
# ======================================================================

def detect_level_touches(
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    o: np.ndarray,
    atr: np.ndarray,
    level: np.ndarray,
    level_type: str,  # "low" or "high"
    touch_threshold_pts: float = 2.0,  # within N points of level = "touch"
    reaction_window: int = 6,  # measure reaction over next N bars
) -> list[dict]:
    """Detect every time price touches a significant level and classify the reaction.

    For "low" type levels (PDL, session low, etc.):
      - Touch = bar low reaches within threshold of level, or breaches below
      - Reaction = what happens in next N bars (bounce up, or continue down?)

    For "high" type levels:
      - Touch = bar high reaches within threshold of level, or breaches above
      - Reaction = what happens next (reject down, or continue up?)

    Returns list of touch events with reaction features.
    """
    n = len(h)
    events = []

    i = 0
    while i < n - reaction_window:
        lev = level[i]
        if np.isnan(lev) or lev <= 0:
            i += 1
            continue

        cur_atr = atr[i] if not np.isnan(atr[i]) else 30.0
        if cur_atr <= 0:
            i += 1
            continue

        touched = False

        if level_type == "low":
            # Price touches from above: low[i] reaches down to level
            if l[i] <= lev + touch_threshold_pts:
                touched = True
                breach_depth = max(0, lev - l[i])  # how far below level
                touch_wick = lev - l[i]  # negative = didn't reach, positive = breached
        else:  # "high"
            if h[i] >= lev - touch_threshold_pts:
                touched = True
                breach_depth = max(0, h[i] - lev)
                touch_wick = h[i] - lev

        if not touched:
            i += 1
            continue

        # Classify the reaction over next reaction_window bars
        end = min(i + reaction_window + 1, n)
        future_h = h[i+1:end]
        future_l = l[i+1:end]
        future_c = c[i+1:end]
        future_o = o[i+1:end]

        if len(future_c) < 3:
            i += 1
            continue

        if level_type == "low":
            # Bounce = price moves UP from the level
            # Measure: max move up in reaction window
            max_bounce = np.max(future_h) - l[i]  # from the touch low to highest point
            close_at_end = future_c[-1]
            bounce_from_level = close_at_end - lev  # positive = bounced above level
            close_above = (future_c > lev).sum() / len(future_c)  # % of bars closing above level
            immediate_bounce = future_c[0] - l[i]  # first bar bounce size

            # Did price stay below level? (breakdown)
            breakdown = (future_c < lev).sum() / len(future_c) > 0.7

            # Wick ratio of touch bar (long lower wick = strong rejection)
            bar_range = h[i] - l[i]
            if bar_range > 0:
                lower_wick = min(o[i], c[i]) - l[i]
                wick_ratio = lower_wick / bar_range  # high = strong wick rejection
            else:
                wick_ratio = 0

            # Touch bar close relative to level
            touch_bar_closed_above = c[i] > lev

            # Displacement: largest body in reaction window
            bodies = np.abs(future_c - future_o)
            max_body = np.max(bodies)
            max_body_atr = max_body / cur_atr

            # Direction of reaction bars
            bullish_bars = (future_c > future_o).sum()
            reaction_direction = bullish_bars / len(future_c)

        else:  # "high" level
            max_bounce = h[i] - np.min(future_l)
            close_at_end = future_c[-1]
            bounce_from_level = lev - close_at_end
            close_above = (future_c < lev).sum() / len(future_c)
            immediate_bounce = h[i] - future_c[0]

            breakdown = (future_c > lev).sum() / len(future_c) > 0.7

            bar_range = h[i] - l[i]
            if bar_range > 0:
                upper_wick = h[i] - max(o[i], c[i])
                wick_ratio = upper_wick / bar_range
            else:
                wick_ratio = 0

            touch_bar_closed_above = c[i] < lev

            bodies = np.abs(future_c - future_o)
            max_body = np.max(bodies)
            max_body_atr = max_body / cur_atr

            bearish_bars = (future_c < future_o).sum()
            reaction_direction = bearish_bars / len(future_c)

        # Classify reaction
        if breakdown:
            reaction_class = "breakdown"
        elif bounce_from_level / cur_atr > 0.5 and close_above > 0.5:
            reaction_class = "strong_bounce"
        elif bounce_from_level > 0 and close_above > 0.3:
            reaction_class = "weak_bounce"
        elif wick_ratio > 0.4 and touch_bar_closed_above:
            reaction_class = "wick_reject"
        else:
            reaction_class = "neutral"

        events.append({
            "bar_idx": i,
            "level_value": lev,
            "breach_depth_pts": breach_depth,
            "breach_depth_atr": breach_depth / cur_atr,
            "wick_ratio": wick_ratio,
            "touch_bar_closed_correct": touch_bar_closed_above,
            "bounce_pts": max_bounce,
            "bounce_atr": max_bounce / cur_atr,
            "bounce_from_level_atr": bounce_from_level / cur_atr,
            "close_correct_pct": close_above,
            "immediate_bounce_atr": immediate_bounce / cur_atr,
            "max_body_atr": max_body_atr,
            "reaction_direction": reaction_direction,
            "reaction_class": reaction_class,
            "breakdown": breakdown,
            "atr": cur_atr,
        })

        # Skip ahead to avoid counting the same touch multiple times
        i += max(3, reaction_window // 2)
        continue

    return events


# ======================================================================
# Link FVG trades to preceding level reactions
# ======================================================================

def link_trades_to_reactions(
    trades: list[dict],
    reactions: dict[str, list[dict]],
    max_lookback_bars: int = 60,
) -> list[dict]:
    """For each trade, find the nearest preceding level reaction and attach features.

    reactions: dict of level_name -> list of touch events
    """
    # Build sorted arrays for fast lookup
    reaction_indices = {}
    for level_name, events in reactions.items():
        if events:
            idxs = np.array([e["bar_idx"] for e in events])
            reaction_indices[level_name] = (idxs, events)

    for t in trades:
        birth_bar = t.get("zone_birth_bar", -1)
        if birth_bar < 0:
            continue

        best_reaction = None
        best_reaction_name = "none"
        best_reaction_dist = 9999

        for level_name, (idxs, events) in reaction_indices.items():
            # Find most recent reaction before birth_bar
            candidates = idxs[idxs < birth_bar]
            if len(candidates) == 0:
                continue

            # Nearest preceding
            nearest_idx = candidates[-1]
            dist = birth_bar - nearest_idx

            if dist > max_lookback_bars:
                continue

            # Find the event
            event_pos = np.searchsorted(idxs, nearest_idx)
            event = events[event_pos]

            if dist < best_reaction_dist:
                best_reaction = event
                best_reaction_name = level_name
                best_reaction_dist = dist

        # Attach to trade
        if best_reaction is not None:
            t["reaction_level"] = best_reaction_name
            t["reaction_dist_bars"] = best_reaction_dist
            t["reaction_class"] = best_reaction["reaction_class"]
            t["reaction_wick_ratio"] = best_reaction["wick_ratio"]
            t["reaction_bounce_atr"] = best_reaction["bounce_atr"]
            t["reaction_breach_atr"] = best_reaction["breach_depth_atr"]
            t["reaction_direction"] = best_reaction["reaction_direction"]
            t["reaction_closed_correct"] = best_reaction["touch_bar_closed_correct"]
            t["reaction_max_body_atr"] = best_reaction["max_body_atr"]
            t["reaction_immediate_bounce_atr"] = best_reaction["immediate_bounce_atr"]
        else:
            t["reaction_level"] = "none"
            t["reaction_dist_bars"] = 9999
            t["reaction_class"] = "none"
            t["reaction_wick_ratio"] = 0
            t["reaction_bounce_atr"] = 0
            t["reaction_breach_atr"] = 0
            t["reaction_direction"] = 0.5
            t["reaction_closed_correct"] = False
            t["reaction_max_body_atr"] = 0
            t["reaction_immediate_bounce_atr"] = 0

        # Also store per-level best reaction
        for level_name, (idxs, events) in reaction_indices.items():
            candidates = idxs[(idxs < birth_bar) & (idxs >= birth_bar - max_lookback_bars)]
            if len(candidates) == 0:
                t[f"react_{level_name}"] = "none"
                t[f"react_{level_name}_class"] = "none"
                continue
            nearest_idx = candidates[-1]
            event_pos = np.searchsorted(idxs, nearest_idx)
            event = events[event_pos]
            t[f"react_{level_name}"] = event["reaction_class"]
            t[f"react_{level_name}_wick"] = event["wick_ratio"]
            t[f"react_{level_name}_bounce"] = event["bounce_atr"]

    return trades


# ======================================================================
# Analysis helpers
# ======================================================================

def group_analyze(df, col, label, min_trades=30):
    """Group by column and show PF/R for each group."""
    print(f"\n  --- {label} ---")
    for grp_name, grp_df in df.groupby(col, observed=True):
        if len(grp_df) < min_trades:
            continue
        m = compute_metrics(grp_df.to_dict("records"))
        tpd = m["trades"] / (252 * 10.5)
        print(f"  {str(grp_name):25s} | {m['trades']:5d}t | R={m['R']:+8.1f} | PF={m['PF']:5.2f} | "
              f"PPDD={m['PPDD']:6.2f} | WR={m['WR']:5.1f}% | DD={m['MaxDD']:5.1f}R | {tpd:.2f}/d")


def bin_analyze(df, col, bins, labels, label, min_trades=30):
    """Bin a continuous column and show PF/R for each bin."""
    df = df.copy()
    df["_bin"] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    group_analyze(df, "_bin", label, min_trades)


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 120)
    print("LEVEL REACTION RESEARCH -- Sprint 8")
    print("Core: at which levels does price REACT, and does reaction quality predict FVG trades?")
    print("=" * 120)

    d = load_all()
    nq = d["nq"]
    n = d["n"]
    h, l, c, o = d["h"], d["l"], d["c"], d["o"]
    atr = d["atr_arr"]

    # Compute levels
    print("\n[COMPUTE] Building levels...")
    session_cache = pd.read_parquet(PROJECT / "data" / "cache_session_levels_10yr_v2.parquet")
    pdhl = compute_pdhl(nq)
    htf_1h = compute_htf_swings(nq, left=12, right=3)
    htf_4h = compute_htf_swings_4h(nq, left=48, right=12)
    print("[COMPUTE] Done")

    # ================================================================
    # STEP 1: Detect all level touches + reactions
    # ================================================================
    print("\n[TOUCH] Detecting level touches and classifying reactions...")
    t0 = _time.perf_counter()

    level_defs = {
        "pdl": (pdhl["pdl"].values, "low"),
        "pdh": (pdhl["pdh"].values, "high"),
        "overnight_low": (session_cache["overnight_low"].values, "low"),
        "overnight_high": (session_cache["overnight_high"].values, "high"),
        "asia_low": (session_cache["asia_low"].values, "low"),
        "asia_high": (session_cache["asia_high"].values, "high"),
        "london_low": (session_cache["london_low"].values, "low"),
        "london_high": (session_cache["london_high"].values, "high"),
        "htf_1h_low": (htf_1h["htf_swing_low_price"].values, "low"),
        "htf_1h_high": (htf_1h["htf_swing_high_price"].values, "high"),
        "htf_4h_low": (htf_4h["htf4h_swing_low_price"].values, "low"),
        "htf_4h_high": (htf_4h["htf4h_swing_high_price"].values, "high"),
    }

    all_reactions = {}
    for level_name, (level_arr, level_type) in level_defs.items():
        events = detect_level_touches(h, l, c, o, atr, level_arr, level_type,
                                       touch_threshold_pts=2.0, reaction_window=6)
        all_reactions[level_name] = events

    elapsed = _time.perf_counter() - t0
    print(f"[TOUCH] Done in {elapsed:.1f}s")

    # ================================================================
    # STEP 2: Level reaction statistics
    # ================================================================
    print(f"\n{'='*120}")
    print("STEP 2: LEVEL REACTION STATISTICS")
    print("Which levels does price actually respect (bounce from)?")
    print(f"{'='*120}")

    print(f"\n  {'Level':25s} | {'Touches':>8s} | {'strong':>8s} | {'weak':>8s} | {'wick':>8s} | {'neutral':>8s} | {'break':>8s} | {'bounce%':>8s}")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for level_name in level_defs:
        events = all_reactions[level_name]
        if not events:
            continue
        total = len(events)
        classes = pd.Series([e["reaction_class"] for e in events]).value_counts()
        strong = classes.get("strong_bounce", 0)
        weak = classes.get("weak_bounce", 0)
        wick = classes.get("wick_reject", 0)
        neutral = classes.get("neutral", 0)
        breakdown = classes.get("breakdown", 0)
        bounce_pct = (strong + weak + wick) / total * 100
        print(f"  {level_name:25s} | {total:8d} | {strong:8d} | {weak:8d} | {wick:8d} | {neutral:8d} | {breakdown:8d} | {bounce_pct:7.1f}%")

    # ================================================================
    # STEP 3: Which level has the strongest reactions?
    # ================================================================
    print(f"\n{'='*120}")
    print("STEP 3: REACTION STRENGTH BY LEVEL")
    print("Average bounce size (ATR multiples) at each level")
    print(f"{'='*120}")

    print(f"\n  {'Level':25s} | {'avg bounce':>12s} | {'avg wick':>10s} | {'avg breach':>12s} | {'med bounce':>12s}")
    print(f"  {'-'*25}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")

    for level_name in level_defs:
        events = all_reactions[level_name]
        if not events:
            continue
        bounces = [e["bounce_atr"] for e in events]
        wicks = [e["wick_ratio"] for e in events]
        breaches = [e["breach_depth_atr"] for e in events]
        avg_b = np.mean(bounces)
        avg_w = np.mean(wicks)
        avg_br = np.mean(breaches)
        med_b = np.median(bounces)
        print(f"  {level_name:25s} | {avg_b:12.3f} | {avg_w:10.3f} | {avg_br:12.3f} | {med_b:12.3f}")

    # ================================================================
    # STEP 4: Link to FVG trades
    # ================================================================
    print(f"\n{'='*120}")
    print("STEP 4: LINK LEVEL REACTIONS TO FVG TRADES")
    print(f"{'='*120}")

    from experiments.ict_context_diagnostic import run_u2_with_context
    trades = run_u2_with_context(d,
        stop_strategy="A2", fvg_size_mult=0.3, max_fvg_age=200,
        min_stop_pts=5.0, tighten_factor=0.85, tp_mult=0.35, nth_swing=5,
    )
    longs = [t for t in trades if t["dir"] == 1]
    print(f"  {len(longs)} long trades")

    # Only use low-type levels for bull FVG analysis
    low_reactions = {k: v for k, v in all_reactions.items() if k.endswith("_low") or k == "pdl"}
    longs = link_trades_to_reactions(longs, low_reactions, max_lookback_bars=60)
    df = pd.DataFrame(longs)

    m_base = compute_metrics(longs)
    print(f"\n  BASELINE:")
    pr("  U2-v2 longs", m_base)

    # ================================================================
    # STEP 5: Reaction class -> trade quality
    # ================================================================
    print(f"\n{'='*120}")
    print("STEP 5: REACTION CLASS -> TRADE QUALITY")
    print("Does the type of reaction at the nearest level predict trade outcome?")
    print(f"{'='*120}")

    group_analyze(df, "reaction_class", "Nearest level reaction class")
    group_analyze(df, "reaction_level", "Which level reacted?")

    # ================================================================
    # STEP 6: Reaction quality features -> trade quality
    # ================================================================
    print(f"\n{'='*120}")
    print("STEP 6: REACTION QUALITY FEATURES")
    print(f"{'='*120}")

    # Wick ratio of the touch bar
    valid = df[df["reaction_class"] != "none"]
    if len(valid) > 100:
        bin_analyze(valid, "reaction_wick_ratio",
            bins=[0, 0.1, 0.25, 0.4, 0.6, 1.01],
            labels=["no wick <10%", "small 10-25%", "moderate 25-40%", "strong 40-60%", "huge >60%"],
            label="Touch bar wick ratio (higher = stronger rejection candle)")

    # Bounce size
    if len(valid) > 100:
        bin_analyze(valid, "reaction_bounce_atr",
            bins=[0, 0.3, 0.7, 1.2, 2.0, 99],
            labels=["tiny <0.3 ATR", "small 0.3-0.7", "medium 0.7-1.2", "large 1.2-2.0", "huge >2.0"],
            label="Post-touch bounce size (ATR)")

    # Breach depth
    if len(valid) > 100:
        bin_analyze(valid, "reaction_breach_atr",
            bins=[-0.01, 0.0, 0.1, 0.3, 0.5, 99],
            labels=["no breach", "tiny <0.1", "small 0.1-0.3", "medium 0.3-0.5", "deep >0.5"],
            label="How far past level (breach depth)")

    # Immediate bounce (first bar after touch)
    if len(valid) > 100:
        bin_analyze(valid, "reaction_immediate_bounce_atr",
            bins=[-99, 0, 0.2, 0.5, 1.0, 99],
            labels=["negative (continued)", "tiny 0-0.2", "small 0.2-0.5", "medium 0.5-1.0", "large >1.0"],
            label="Immediate bounce (1st bar after touch)")

    # Touch bar closed on correct side
    if len(valid) > 100:
        group_analyze(valid, "reaction_closed_correct",
            label="Touch bar closed on correct side of level?")

    # Post-touch displacement body
    if len(valid) > 100:
        bin_analyze(valid, "reaction_max_body_atr",
            bins=[0, 0.3, 0.6, 1.0, 1.5, 99],
            labels=["small <0.3", "moderate 0.3-0.6", "medium 0.6-1.0", "large 1.0-1.5", "huge >1.5"],
            label="Max displacement body after touch (ATR)")

    # Distance from reaction to FVG
    if len(valid) > 100:
        bin_analyze(valid, "reaction_dist_bars",
            bins=[0, 5, 15, 30, 60, 9999],
            labels=["immediate 1-5", "close 6-15", "moderate 16-30", "far 31-60", "very far 60+"],
            label="Bars from level reaction to FVG creation")

    # ================================================================
    # STEP 7: Per-level reaction class
    # ================================================================
    print(f"\n{'='*120}")
    print("STEP 7: REACTION AT SPECIFIC LEVELS")
    print(f"{'='*120}")

    for level_name in ["pdl", "overnight_low", "asia_low", "london_low", "htf_1h_low", "htf_4h_low"]:
        react_col = f"react_{level_name}"
        if react_col not in df.columns:
            continue
        sub = df[df[react_col] != "none"]
        if len(sub) < 50:
            print(f"\n  {level_name}: too few reactions ({len(sub)})")
            continue
        group_analyze(sub, react_col, f"Reaction at {level_name}")

    # ================================================================
    # STEP 8: Combine best reaction features + displacement
    # ================================================================
    print(f"\n{'='*120}")
    print("STEP 8: COMBINED FILTERS (reaction + displacement)")
    print(f"{'='*120}")

    has_disp = (df["disp_atr_mult"] >= 0.3) & (df["disp_atr_mult"] <= 1.5)

    combos = [
        ("baseline (all longs)", pd.Series(True, index=df.index)),
        ("disp 0.3-1.5 only", has_disp),
        ("disp + strong_bounce reaction", has_disp & (df["reaction_class"] == "strong_bounce")),
        ("disp + any bounce (strong+weak+wick)", has_disp & df["reaction_class"].isin(["strong_bounce", "weak_bounce", "wick_reject"])),
        ("disp + closed correct side", has_disp & (df["reaction_closed_correct"] == True) & (df["reaction_class"] != "none")),
        ("disp + wick ratio > 0.3", has_disp & (df["reaction_wick_ratio"] > 0.3)),
        ("disp + bounce > 0.7 ATR", has_disp & (df["reaction_bounce_atr"] > 0.7)),
        ("disp + PDL reaction (any)", has_disp & (df["react_pdl"] != "none")),
        ("disp + PDL strong bounce", has_disp & (df["react_pdl"] == "strong_bounce")),
        ("disp + NOT breakdown", has_disp & (df["reaction_class"] != "breakdown")),
        ("disp + no reaction (FVG born fresh)", has_disp & (df["reaction_class"] == "none")),
    ]

    for label, mask in combos:
        sub = df[mask]
        if len(sub) < 30:
            print(f"  {label:50s} | too few ({len(sub)})")
            continue
        m = compute_metrics(sub.to_dict("records"))
        tpd = m["trades"] / (252 * 10.5)
        print(f"  {label:50s} | {m['trades']:5d}t | R={m['R']:+8.1f} | PF={m['PF']:5.2f} | "
              f"PPDD={m['PPDD']:6.2f} | WR={m['WR']:5.1f}% | DD={m['MaxDD']:5.1f}R | {tpd:.2f}/d")


if __name__ == "__main__":
    main()
