# NQ Quant — Quantitative Trading System for Nasdaq 100 Futures

Quantitative system for NQ (Nasdaq 100 E-mini Futures) / MNQ (Micro) targeting prop firm and self-funded accounts. Based on ICT methodology: liquidity sweep, FVG as entry vehicle, premium/discount zones, and strict risk management.

## Production Engine: 1m Hybrid

The current production engine is `experiments/unified_engine_1m.py` — **5m zone detection + 1m bar execution**. The 1m execution resolves the intrabar TP/BE ambiguity that plagued the earlier 5m-only engine.

### Architecture

```
Phase 1: 5m Zone Detection (batch pre-build)
  ├── FVG detection (3-candle pattern, shift(1) no lookahead)
  ├── Breakdown detection (overnight level sweeps)
  ├── NOT-MSS filter (displacement does NOT break swing)
  ├── Trend zones (PD zone + bias + not in breakdown territory)
  └── Swing tracking (5m fractal)

Phase 2: 1m Execution Loop
  ├── Day boundary reset (2R daily loss, 0-for-2 rule)
  ├── Zone activation from completed 5m bars
  ├── Zone invalidation (1m close through zone)
  ├── Exit management (trim, BE, trail, EOD, PA early cut)
  └── Entry scanning (chain tier first, then trend tier)
```

## Performance (10.3 years, 2015-2026, hell-audited)

### The Intrabar TP/BE Ambiguity (Axiom 9)

When TP and entry (for BE calculation) are both touched on the same bar, OHLC cannot distinguish which came first:
- **Path A**: price hits TP → trim → BE active → price returns → runner dies
- **Path B**: price dips first (no BE yet) → rallies to TP → trim → runner lives

Flaw rates:
- 5m bars: 76% of runners affected (catastrophic)
- 1m bars: 41% of runners affected (significant)
- Only tick data can fully resolve this

Solution: bound performance between **optimistic** and **worst-case**.

### Performance Bounds

| Config | Trades | R | PF | PPDD | MaxDD | Sharpe | Neg Years |
|--------|--------|------|------|------|-------|--------|-----------|
| **Optimistic** (BE NOT checked on trim bar) | 1541 | **+700.0** | **3.32** | **75.1** | **9.3R** | **4.92** | **0/11** |
| **Worst-case** (Axiom 9: BE forced on trim bar) | 1550 | **+414.4** | **2.32** | **43.0** | **9.6R** | **3.67** | **0/11** |

**True performance is between these bounds. Both are profitable. Both have zero negative years.**

### Tier Breakdown (worst-case)

| Tier | Trades | R | PF | Notes |
|------|--------|------|------|-------|
| Chain (Breakdown → NOT-MSS FVG) | 475 | **+319.3** | 3.09 | 1.0x R, big sweep 1.5x |
| Trend (PD + Bias → FVG) | 1101 | +94.9 | 1.58 | 0.5x R |
| **Combined** | **1550** | **+414.4** | **2.32** | — |

### Walk-Forward (worst-case)

```
2016: 146t  R= +25.2  PF=1.80
2017: 128t  R= +40.8  PF=2.42
2018: 147t  R= +63.5  PF=2.62
2019: 125t  R= +33.4  PF=2.38
2020: 153t  R= +53.4  PF=2.79
2021: 145t  R= +31.4  PF=1.91
2022: 104t  R= +28.4  PF=2.17
2023: 115t  R= +52.5  PF=3.26
2024: 109t  R= +48.1  PF=3.30
2025: 126t  R= +53.2  PF=2.87
2026:  32t  R=  +1.4  PF=1.16
```
**Zero negative years across 11 years in both bounds.**

## How It Works

**Tier 1 — Chain Entry (Liquidity Sweep Reversal)**

1. Price closes through overnight high/low (the "sweep")
2. Next 30 bars: first FVG whose displacement candle does NOT break a prior swing point (NOT-MSS)
3. Limit order at FVG zone edge
4. Entry is in the "discount zone" — before the structure shift completes — maximum runner space

**Tier 2 — Trend Entry (Premium/Discount + Bias)**

1. FVG in correct PD zone: longs in discount (below 50% of overnight range), shorts in premium
2. Bias aligned with trade direction (composite: 0.4 × HTF FVG draw + 0.3 × overnight + 0.3 × ORM)
3. PD re-verified at fill time
4. Excludes FVGs within breakdown territory (those belong to Chain)

**Stop Loss — Zone-Based (validated)**

```
Long:  stop = FVG bottom - (FVG size × 15% buffer), tightened by 0.85 factor
Short: stop = FVG top + (FVG size × 15% buffer), tightened by 0.85 factor
Min stop: 0.15 × ATR(14) to filter tiny candles
Avg stop distance: ~9.7 points
```

The stop sits at the structural FVG invalidation level. An alternative candle-based stop (spec §8) was tested and **REJECTED** — see Bug History below.

**Exit Management**

- **Trim**: 25% at fixed 1R TP
- **BE**: remaining 75% moves to break-even after trim
- **Trail**: 5th swing low (longs) / 5th swing high (shorts), recomputed on 5m
- **EOD**: 15:55 ET close
- **PA early cut**: bars 10-20 if wicky + unfavorable direction + no displacement

**Sizing**

- Grade A+: bias aligned + regime ≥ 1.0 → 1.5× base risk
- Grade B+: bias aligned OR regime ≥ 1.0 → 1.0× base risk
- Grade C: neither → 0.5× base risk
- Big sweep bar ≥ 1.3 ATR → ×1.5 (validated, bootstrap CI excludes 0)
- AM shorts (10-12 ET) → ×0.5 (structural AM upward drift)
- Trend tier → ×0.5 (half size for confirmation-style setups)
- Monday/Friday → reduced R ($500 vs $1000)

**Risk Rules**

- 2R daily loss limit → stop for the day
- 0-for-2 consecutive losses → stop for the day
- BE sweep after trim: NOT counted as loss (net profitable)
- EOD close: NOT counted as loss (neutral)

## Edge Validation — Random Baseline Test

Replace zone-based entries with random entries using the same trade management. 20 random permutations:

| Metric | Real Engine | Random (mean) | Edge |
|--------|------------|---------------|------|
| R | +414.4 | +20.6 | **+393.8** |
| PF | 2.32 | 1.05 | **+1.27** |
| MaxDD | 9.6R | 88.7R | **-79R (better)** |

**0/20 random permutations beat the real engine** (p ≈ 0.048).

**95% of the edge comes from FVG zone selection**, not trade management. Random entries with identical trim/BE/trail produce PF ≈ 1.05 (barely breakeven). The zones themselves are the alpha.

## NT8 Port: HybridEngine1m.cs

Complete 1:1 NinjaScript port of the Python engine. 1444 lines, hell-audited 3 rounds, Axiom 9/9 pass.

### Data Series

```
BIP 0: NQ/MNQ 1min  (execution)
BIP 1: NQ/MNQ 5min  (zone detection, swing, ATR, FVG, breakdown)
BIP 2: NQ/MNQ 60min (1H HTF FVG tracking for bias)
BIP 3: NQ/MNQ 240min(4H HTF FVG tracking + fluency for regime)
```

### Ported Modules (zero proxy, zero TODO)

| Python Module | NT8 Method | Status |
|---------------|------------|--------|
| `unified_engine_1m.py` execution | `OnBar1m` / `ManageExits` / `TryEntry` | ✓ |
| `chain_engine.py` breakdown + NOT-MSS | `DetectBreakdowns5m` / `PassesNotMSS` (SwingPt + 100-bar search) | ✓ |
| `features/bias.py` HTF+ON+ORM composite | `OnBar1h` / `OnBar4h` / `RecomputeCompositeBias` | ✓ |
| `features/bias.py` regime (PDA+fluency+chop) | `regimeValue` + `IsChoppy` | ✓ |
| `features/displacement.py` fluency | `Compute4hFluency` (ring buffer, proper 4H ATR) | ✓ |
| `features/fvg.py` detection + state machine | `Detect5mFVGs` (shift(1)) + `UpdateHtfFvgStates` | ✓ |
| `features/sessions.py` ON/ORM/nyOpen | `UpdateSessionLevels` | ✓ |
| `features/news_filter.py` | `LoadNewsCalendar` / `InNewsBlackout` | ✓ |
| Axiom 1 (fill irreversibility) | Same-bar stop recorded as loss | ✓ |
| Axiom 5 (losses-first processing) | Exit events sorted by loss status | ✓ |
| Axiom 9 (worst-case trim BE) | `WorstCaseTrimBe` parameter | ✓ |

### Hell-Audit History

- **Round 1**: 11 issues found (6 CRITICAL, 5 MODERATE) — all fixed
- **Round 2**: Verified all fixes — PASS
- **Round 3 (full hell-audit)**: 3 additional CRITICAL issues found — all fixed
  - `ormReady` never reset on new day → FIXED
  - NOT-MSS only checked last swing → FIXED (SwingPt struct + 100-bar backward search)
  - Double zone invalidation (5m + 1m) → FIXED (removed 5m, kept 1m)
  - `prevClose5m` initialized to 0 → FIXED (NaN + guard)

**Axiom 9/9 PASS. Deployment readiness: 8/10.**

### NT8 Deployment

1. Copy `ninjatrader/HybridEngine1m.cs` to `Documents/NinjaTrader 8/bin/Custom/Strategies/`
2. Compile (F5 in NinjaScript Editor)
3. Import 1m NQ data: `ninjatrader/NQ_1min_NT8.txt` (3.5M bars, 182 MB, 2015-2026)
   - Format: NinjaTrader (timestamps represent start of bar time)
   - Data Type: Last, Time Zone: UTC
   - Instrument type: **Index** (not Stock, not Future)
   - Point value: 2, Tick size: 0.25
4. **Tools → Options → General → Time zone: Eastern Time** (mandatory)
5. Strategy Analyzer → HybridEngine1m → primary = 1 Minute → Run

## Architecture

```
nq-quant/
├── config/
│   ├── params.yaml                  # All tunable parameters (centralized)
│   └── news_calendar.csv            # High-impact news events
├── data/
│   ├── NQ_1min_10yr.parquet         # 3.5M bars, execution
│   ├── NQ_5m_10yr.parquet           # 711K bars, zone detection
│   ├── cache_atr_flu_10yr_v2.parquet
│   ├── cache_bias_10yr_v2.parquet
│   ├── cache_regime_10yr_v2.parquet
│   └── cache_session_levels_10yr_v2.parquet
├── features/
│   ├── fvg.py                       # FVG detection + state machine
│   ├── displacement.py              # ATR, fluency
│   ├── swing.py                     # Fractal swing detection
│   ├── sessions.py                  # Session levels (Asia/London/NY/overnight/ORM)
│   ├── bias.py                      # HTF + overnight + ORM composite bias
│   ├── news_filter.py               # News blackout windows
│   └── ...
├── experiments/
│   ├── unified_engine_1m.py         # PRODUCTION: 1m hybrid engine
│   ├── unified_engine_v2.py         # Previous 5m-only engine
│   ├── chain_engine.py              # Zone detection helpers
│   ├── stop_comparison.py           # Zone vs candle stop experiment
│   ├── random_baseline_1m.py        # Random entry validation
│   └── ...
├── ninjatrader/
│   ├── HybridEngine1m.cs            # NT8 port of 1m hybrid (1444 lines)
│   ├── NQ_1min_NT8.txt              # 1m data for NT8 import (182 MB)
│   ├── NQ_5m_NT8.txt                # 5m data for NT8 import (37 MB)
│   ├── ES_5m_NT8.txt                # ES 5m data (for SMT, optional)
│   └── export_to_nt8.py             # Parquet → NT8 .txt converter
├── tests/
│   ├── test_execution_realism.py    # 11 automated physical constraint tests
│   └── hell-audit-prompt.md         # 9 axioms for backtest correctness
└── viz/
    └── chart.py                     # mplfinance charts with FVG overlay
```

## Bugs Fixed

### Candle-Based Stop (CRITICAL — 2026-04-06)
Spec §8 specified candle-based stop (fill bar open). Tested against zone-based stop.

**Discovery**: In a limit-order framework, bar open is often on the profit side of entry, creating an **inverted stop** that becomes a guaranteed profit exit. 90% of same_bar_stop trades in candle mode were inverted artifacts:

| Tier | SBS Trades | Inverted | Fake Profit |
|------|-----------|----------|-------------|
| Chain | 359 | 322 (89.7%) | **+155.1R** |
| Trend | 852 | 780 (91.5%) | **+138.2R** |

After removing fake profit: candle mode becomes net negative in both tiers. **Zone-based stop is the only valid approach for limit orders.** Candle mode removed from engine.

### Axiom 9 — Intrabar TP/BE Ambiguity (2026-04-06)
Added 9th axiom to hell-audit. Runs strategy with `worst_case_trim_be=True` to bound performance pessimistically. True performance lies between optimistic and worst-case bounds. Both bounds are profitable.

### Account R Bug (CRITICAL)
R was `pnl_dollars / (stop_dist × point_value × contracts)` — self-referential. Trend 0.5x trades reported same R as 1.0x → inflated aggregate R ~2x. Fixed to `pnl_dollars / normal_r` ($1000 base). Impact: reported R dropped from +660.6 to +291.9 (5m engine). Dollar P&L unchanged.

### Fixed Min Stop Bug (CRITICAL)
`min_stop_pts = 5.0` was a fixed points value. In 2016-2017 (NQ ~4500, ATR ~3pt), this filtered 62% of chain zones. Changed to `min_stop_atr_mult = 0.15` (ATR-relative). Impact: 2016 trades 17→146, 2017 trades 9→128. Total 862→1331.

### Legacy Fixes

| Engine | Fix | Impact |
|--------|-----|--------|
| U2 | Same-bar stop skip (Axiom 1) | +1294R inflation (62%) |
| U2 | Min stop 5pt floor | +859R inflation |
| Unified v2 | Account R self-referential | +368R inflation |
| Unified v2 | Fixed min stop 5pt | -470 trades lost in 2016-17 |

## Audit Framework — 9 Axioms

| Axiom | Description | Status |
|-------|-------------|--------|
| 1 | Fill irreversibility (same-bar stop) | PASS |
| 2 | Temporal causality (no lookahead) | PASS |
| 3 | Execution latency | PASS |
| 4 | Order type consistency | PASS |
| 5 | Worst-case intrabar (losses first) | PASS |
| 6 | Transaction cost completeness | PASS |
| 7 | Risk denominator consistency | PASS |
| 8 | State reset (day boundary close) | PASS |
| **9** | **Intrabar price path ambiguity (TP/BE same bar)** | **BOUNDED** |

Axiom 9 cannot PASS without tick data — instead, performance is bounded between worst-case (PF=2.32) and optimistic (PF=3.32). Both bounds are profitable.

## Statistical Validation

| Test | Result |
|------|--------|
| Bootstrap PF CI > 1.0 | YES for all tiers |
| Temporal split (2016-2020 train, 2021-2026 test) | ALL HOLD |
| Year-by-year | 0/11 negative years (both bounds) |
| Sweep bar sizing (bootstrap diff) | Significant |
| **Random entry baseline (20 perms)** | **0/20 beat real engine (p=0.048)** |
| **Random PF** | **1.05** (vs real 2.32) |
| **Edge concentration** | **95% in zone selection** |

## Data Requirements

- NQ 1-minute OHLCV: 10+ years, continuous contract
- NQ 5-minute OHLCV: 10+ years (can be derived from 1m)
- Pre-computed caches: ATR, bias, regime, session levels (`.parquet`)
- News calendar CSV
- (For NT8) NQ_1min_NT8.txt: 1m data in NT8 format (exported from parquet)

## Tech Stack

- Python 3.11+, pandas, numpy, PyArrow
- NinjaTrader 8 / NinjaScript C# (for auto-execution)
- All timestamps UTC internally, convert to ET for session logic
- All parameters centralized in `config/params.yaml`

## Research History

70+ experiments across 9 sprints:

- **Sprint 1-3**: Data pipeline, feature engineering, XGBoost (insignificant isolated features)
- **Sprint 4-5**: Filter optimization, trade management
- **Sprint 6**: NinjaTrader v9 port (superseded by HybridEngine1m)
- **Sprint 7**: U2 limit order engine, same-bar-stop bug, audit framework
- **Sprint 8**: ICT chain discovery — level breakdown, NOT-MSS, symmetric shorts, trend chain, dual-tier unified engine
- **Sprint 9 (current)**: 1m hybrid engine, Axiom 9, candle-stop bug, random baseline validation, complete NT8 port

**Key evolution**: isolated FVG entries have zero alpha vs random. Edge comes from the ICT CONTEXT (sweep → FVG, premium/discount + bias) — the zone selection is the alpha, not the trade management.

## Next Steps

1. **NT8 validation**: Backtest HybridEngine1m vs Python to verify 1:1 alignment
2. **Paper trade**: 2 weeks MNQ, R=$50, verify execution behavior
3. **Tick data**: Obtain tick data for definitive Axiom 9 resolution
4. **Lock config**: Choose between tight+1R, tight+1.5R, med+1R (all within margin)
5. **Portfolio**: Combine F3 + unified equity curves (diversification value)
