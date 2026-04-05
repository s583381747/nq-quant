# NQ Quant -- Quantitative Trading System for Nasdaq 100 Futures

Quantitative system for NQ (Nasdaq 100 E-mini Futures) / MNQ (Micro) targeting prop firm and self-funded accounts. Based on ICT methodology: liquidity sweep, FVG as entry vehicle, premium/discount zones, and strict risk management.

## Production Strategy: Unified ICT Chain Engine

Two-tier entry system combining liquidity sweep reversals and trend continuation entries, with shared risk management and independent position management per tier.

### Performance (10.3 years, 2015-2026, hell-audited)

| Metric | Unified | U2 (legacy) |
|--------|---------|-------------|
| Total R | **+660.6** | +791.5 |
| Profit Factor | **2.36** | 1.59 |
| PPDD (R/MaxDD) | **41.92** | 28.35 |
| Max Drawdown | **15.8R** | 27.9R |
| Trades | 863 | 2,589 |
| Frequency | 0.33/day | 0.98/day |
| Win Rate | 47.3% | 46.3% |
| Negative Years | **0/11** | 2/11 |

### How It Works

**Tier 1 -- Chain Entry (Liquidity Sweep Reversal)**

1. Price breaks through a significant level (overnight high/low) -- this is the "liquidity sweep"
2. After the sweep, the first FVG that forms WITHOUT breaking a swing point (NOT-MSS) is the entry signal
3. Limit order placed at FVG zone edge
4. The entry is in the "discount zone" -- before the structure shift completes -- giving maximum runner space

```
292 trades / PF=2.71 / 1.0x risk
```

**Tier 2 -- Trend Entry (Premium/Discount + Bias)**

1. FVG forms in the correct zone: longs in discount (below 50% of overnight range), shorts in premium
2. Bias must be aligned with the trade direction
3. PD alignment verified at both zone creation AND fill time
4. Excludes FVGs near breakdown events (those belong to Tier 1)

```
571 trades / PF=2.17 / 0.5x risk
```

**Exit Management (symmetric for longs and shorts)**

- 25% trim at fixed 1R take profit
- Move to breakeven after trim
- 75% runner trails with 5th swing low (longs) / swing high (shorts)
- EOD close at 15:55 ET
- PA early cut on bars 2-4 if wicky + unfavorable

**Validated Sizing Signals**

- Sweep bar range >= 1.3 ATR: 1.5x risk (validated: bootstrap CI excludes 0)
- AM shorts (10-12 ET): 0.5x risk (structural: AM upward drift)

### Walk-Forward Results

```
2016:  17t  R= +21.9  PF=4.12
2017:   9t  R=  +6.7  PF=2.96
2018:  77t  R= +74.9  PF=2.95
2019:  50t  R= +24.2  PF=1.95
2020: 128t  R=+136.7  PF=2.77
2021: 119t  R= +51.6  PF=1.74
2022: 107t  R= +88.9  PF=2.36
2023: 100t  R= +79.9  PF=2.55
2024: 103t  R= +95.4  PF=2.81
2025: 121t  R= +75.9  PF=2.05
2026:  32t  R=  +4.4  PF=1.20
```

Every single year profitable. Worst year (2019) still PF=1.95.

## Architecture

```
nq-quant/
├── config/
│   ├── params.yaml              # All tunable parameters (centralized)
│   └── news_calendar.csv        # High-impact news events
├── data/
│   ├── loader.py                # Raw data loading
│   ├── cleaner.py               # NaN, timezone, DST handling
│   ├── resampler.py             # 1m -> 5m/15m/1H/4H
│   └── rollover.py              # Contract rollover handling
├── features/
│   ├── fvg.py                   # FVG detection + state tracking
│   ├── displacement.py          # ATR, fluency scoring
│   ├── swing.py                 # Fractal swing high/low detection
│   ├── sessions.py              # Session marking + levels (Asia/London/NY)
│   ├── bias.py                  # Multi-timeframe bias computation
│   ├── news_filter.py           # News blackout window logic
│   └── ...
├── experiments/
│   ├── unified_engine_v2.py     # PRODUCTION: dual-tier engine (hell-audited)
│   ├── chain_engine.py          # Chain-only engine (hell-audited)
│   ├── u2_clean.py              # Legacy U2 engine
│   └── (50+ research files)     # Sprint 8 research archive
├── backtest/
│   ├── engine.py                # Core backtest loop
│   └── ...
├── ninjatrader/
│   ├── U2LimitOrderStrategy.cs  # NT8 port (needs update for unified)
│   └── ...
├── tests/
│   ├── test_execution_realism.py # 11 physical constraint tests
│   └── AUDIT_AXIOMS.md          # 8 axioms for backtest correctness
└── viz/
    └── chart.py                 # mplfinance charts with FVG overlay
```

## Key Discoveries (Sprint 8)

### 1. Level Breakdown = True ICT Sweep
Price closing through a significant level (not bouncing) then forming a reversal FVG is the quantitative definition of an ICT liquidity sweep. "Breakdown" trades have PF=2.31 vs "bounce" trades PF=1.39.

### 2. NOT-MSS = Discount Entry
FVG whose displacement does NOT break a swing point = entry before structure shift = maximum runner space. NOT-MSS PF=2.96 vs IS-MSS PF=1.08. Counterintuitive: less confirmation = better entry.

### 3. Symmetric Management Unlocks Shorts
Original shorts had 100% TP1 exit (no runner) -- a code asymmetry, not a market constraint. With identical trim/runner management, NQ shorts become profitable (PF=1.98).

### 4. Premium/Discount = Trend Entry Signal
Buy in discount (below 50% of overnight range), sell in premium (above 50%). Combined with bias alignment: 579 trades at PF=2.84. Core ICT concept validated quantitatively.

### 5. Runner IS the Edge
40% of trades exit via runner (be_sweep) at avgR=+2.36. 5.5% via EOD close at avgR=+4.75. Without runners, PF drops to ~1.0. The trim at 1R just activates breakeven protection; the trail mechanism captures the real move.

## Audit Trail

### Unified Engine V2 -- 8/8 Axioms Pass

| Axiom | Description | Status |
|-------|-------------|--------|
| 1 | Fill irreversibility (same-bar stop) | PASS |
| 2 | Temporal causality (no lookahead) | PASS |
| 3 | Execution latency | PASS |
| 4 | Order type consistency | PASS |
| 5 | Worst-case intrabar (losses first) | PASS (FIX #2) |
| 6 | Transaction cost completeness | PASS |
| 7 | Risk denominator consistency | PASS |
| 8 | State reset (day boundary close) | PASS (FIX #3) |

3 issues found and fixed:
1. Day-stopped bypass between tier entries (CRITICAL)
2. Non-deterministic exit ordering on same bar (MODERATE)
3. Position leak across day boundary (MODERATE)

### Legacy U2 Engine -- 5 historical fixes

| Fix | Impact |
|-----|--------|
| Same-bar stop skip (MOST CRITICAL) | +1,294R inflation (62% of reported R) |
| Min stop 5pt floor | +859R inflation |
| Slippage on stop exits | ~70R overstatement |
| Stop tightening order | 197 phantom entries |
| EOD vs stop race condition | 10 trades misclassified |

## Statistical Validation

| Test | Result |
|------|--------|
| Bootstrap PF CI > 1.0 | YES for all tiers: [1.95, 2.81] |
| Temporal split (2016-2020 train, 2021-2026 test) | ALL HOLD |
| Year-by-year | 0/11 negative years |
| Sweep bar sizing (bootstrap diff) | Significant: CI [+1.16, +4.44] |
| Trend chain (PD+bias) validation | Train PF=3.46 -> Test PF=2.66 HOLDS |

## Data Requirements

- NQ 5-minute OHLCV: 10+ years, continuous contract
- Pre-computed caches: ATR(14), bias, regime, session levels (`.parquet`)
- News calendar CSV
- Data files not included (proprietary/licensed)

## Tech Stack

- Python 3.11+, pandas, numpy, PyArrow
- NinjaTrader 8 / NinjaScript C# (for NT8 port)
- All timestamps UTC internally, convert to ET for session logic
- All parameters centralized in `config/params.yaml`

## Research History

70+ experiments across 8 sprints:
- **Sprint 1-3**: Data pipeline, feature engineering, XGBoost (all insignificant)
- **Sprint 4-5**: Filter optimization, trade management
- **Sprint 6**: NinjaTrader port
- **Sprint 7**: U2 limit order engine, same-bar-stop bug discovery, audit framework
- **Sprint 8**: ICT chain discovery, level breakdown, NOT-MSS, symmetric shorts, trend chain, unified engine

Key evolution: isolated FVG entry has ZERO alpha vs random. Edge comes from the ICT CONTEXT (sweep -> FVG, premium/discount) + trade management (trim/runner/trail).
