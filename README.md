# NQ Quant — Quantitative Trading System for Nasdaq 100 Futures

Quantitative system for NQ (Nasdaq 100 E-mini Futures) / MNQ (Micro) targeting prop firm and self-funded accounts. Based on ICT-derivative methodology: FVG as support/resistance, multi-timeframe bias, displacement-based entries, and strict risk management.

## Strategies

### U2 — Limit Order at FVG Zone (Primary)

Places limit buy orders at bullish Fair Value Gap zone edges. When price retraces into an FVG zone, the limit order fills at a superior entry price with a tight stop on the opposite side of the zone.

| Metric | Value |
|--------|-------|
| Total R | +1,270 |
| Profit Factor | 1.87 |
| PPDD (R/MaxDD) | 48.4 |
| Max Drawdown | 26.3R |
| Trades | 2,012 |
| Frequency | 0.76/day |
| Win Rate | 31% |
| Negative Years | 0/12 |
| Backtest Period | 2015-2026 |

**Key characteristics:**
- Long-only (short NQ has negative expectancy with this entry method)
- Trend-dependent: profits come from intraday trending days (EOD close = 134% of total R)
- Low win rate / high reward ratio: ~70% of trades lose ~1R, ~30% win 3-10R+
- Edge confirmed via random comparison: random PF = 0.89-1.08, U2 PF = 1.87

### F3 — Market Order FVG Rejection (Secondary)

Traditional signal-based entry at 5m FVG test+rejection candle close. Supports both long and short via dual-mode management.

| Metric | Value |
|--------|-------|
| Total R | +86 |
| Profit Factor | 1.21 |
| PPDD | 4.78 |
| Max Drawdown | 18.0R |
| Trades | 1,127 |
| Win Rate | 51% |

More conservative, lower returns but higher win rate and shorter drawdown recovery.

## Architecture

```
nq-quant/
├── config/
│   ├── params.yaml              # All tunable parameters (centralized)
│   └── news_calendar.csv        # High-impact news events (FOMC, CPI, NFP...)
├── data/
│   ├── loader.py                # Raw data loading
│   ├── cleaner.py               # NaN, timezone, DST handling
│   ├── resampler.py             # 1m -> 5m/15m/1H/4H
│   └── rollover.py              # Contract rollover handling
├── features/
│   ├── fvg.py                   # FVG detection + state tracking
│   ├── displacement.py          # ATR, fluency scoring
│   ├── swing.py                 # Fractal swing high/low detection
│   ├── sessions.py              # Session marking (Asia/London/NY)
│   ├── news_filter.py           # News blackout window logic
│   ├── smt.py                   # SMT divergence (NQ vs ES)
│   ├── bias.py                  # Multi-timeframe bias
│   ├── pa_quality.py            # Price action quality scoring
│   └── mtf_fvg.py               # Multi-timeframe FVG alignment
├── experiments/
│   ├── u2_clean.py              # U2 Limit Order engine (PRODUCTION)
│   ├── validate_improvements.py # F3 engine (PRODUCTION)
│   └── (40+ experiment files)   # Research archive
├── backtest/
│   ├── engine.py                # Core backtest loop
│   ├── engine_jit.py            # JIT-compiled variant
│   ├── position_sizer.py        # Contract calculation
│   ├── trade_manager.py         # Trim/BE/trail logic
│   └── report.py                # HTML report generation
├── ninjatrader/
│   ├── U2LimitOrderStrategy.cs  # NT8 port of U2 (audited)
│   ├── LantoNQStrategy.cs       # NT8 port of F3
│   └── export_signals_for_nt8.py
├── tests/
│   ├── test_fvg.py
│   ├── test_displacement.py
│   ├── test_no_lookahead.py     # Future function leak test
│   └── test_sessions.py
└── viz/
    ├── chart.py                 # mplfinance charts with FVG overlay
    └── feature_check.py         # Visual feature verification
```

## How U2 Works

### Entry Mechanism

1. **FVG Detection**: Scan 5-minute NQ bars for Fair Value Gaps (3-candle pattern where candle-1 high < candle-3 low). Anti-lookahead: FVG only visible 1 bar after candle-3 closes (`np.roll(1)` shift).

2. **Zone Tracking**: Each FVG becomes an active zone with `top` (candle-3 low) and `bottom` (candle-1 high). Zones expire after 200 bars or when price closes through them (invalidation).

3. **Limit Order Fill**: When bar low touches zone top (for bullish FVG), a limit buy fills at zone top. No slippage on limit order entries. Conservative: if bar low also breaches the tightened stop, the fill is rejected.

4. **Stop Placement**: Stop at zone bottom minus 15% buffer (A2 strategy), then tightened to 80% of stop distance. Minimum 5 points hard floor.

### Exit Management

- **TP1**: Nearest swing high target multiplied by 2.0x. 25% of position trimmed at TP1.
- **Breakeven**: Stop moved to entry after trim.
- **Trail**: Remaining 75% trailed by 3rd swing low.
- **EOD Close**: All positions closed at 15:55 ET (prop firm compliance). Stop/TP checked BEFORE EOD to avoid race condition.
- **PA Early Cut**: Bad price action (wicky, no progress) exits at next bar open (bars 3-4).

### Filters

- NY session only (10:00-16:00 ET)
- Observation window: no trades 9:30-10:00 ET
- Lunch dead zone: skip 12:30-13:00 ET
- Bias alignment: long only when bias is not opposing
- News blackout: 60 min before / 5 min after high-impact events
- 0-for-2: stop after 2 consecutive losses per day
- 2R daily loss limit

## Audit Trail

The U2 strategy underwent **two full adversarial code audits** (hell-audits). Four code bugs were found and fixed:

| Fix | Bug | Impact |
|-----|-----|--------|
| #1 | Stops < 3pt generated unrealistic R-multiples | +859R inflation (35%) |
| #2 | Stop tightened AFTER fill-bar check | 197 phantom entries |
| #3 | No slippage on stop exits | ~70R overstatement |
| #4 | EOD close checked before stop (race condition) | 10 trades misclassified |

Post-fix verification:
- Fill-bar stop violations: **0/2012**
- Manual PnL trace: **5/5 match**
- Random entry comparison: random PF=0.89-1.08 vs U2 PF=1.87

## Walk-Forward Results (U2)

```
2015:    3t  R=   +0.6  PF=1.26
2016:   73t  R=  +19.5  PF=1.37
2017:   48t  R=  +13.4  PF=1.46
2018:  171t  R=  +11.5  PF=1.09
2019:  167t  R= +123.1  PF=2.12
2020:  264t  R= +206.2  PF=2.02
2021:  246t  R=  +90.2  PF=1.50
2022:  216t  R= +150.7  PF=1.96
2023:  269t  R= +191.7  PF=2.03
2024:  249t  R= +205.3  PF=2.13
2025:  247t  R= +227.1  PF=2.29
2026:   59t  R=  +30.8  PF=1.66
```

Zero negative years across 12 years. Worst year (2018) still profitable at PF=1.09.

## Key Parameters (U2)

```yaml
fvg_size_mult: 0.3        # Min FVG size as ATR multiple
max_fvg_age: 200           # Max bars before zone expires
stop_strategy: A2           # Zone bottom - 15% buffer
tighten_factor: 0.80        # Tighten stop to 80% of distance
min_stop_pts: 5.0           # Hard floor (audit fix)
trim_pct: 0.25              # 25% at TP1
tp_mult: 2.0                # IRL target * 2.0
nth_swing: 3                # Trail 3rd swing low
risk_dollars: 1000           # $1000 per trade (normal)
daily_max_loss_r: 2.0        # Stop after -2R/day
max_consecutive_losses: 2    # 0-for-2 rule
```

## Known Limitations

1. **Long-only in a bull market**: NQ rose 1,270% over the backtest period. Edge confirmed vs random, but requires bounces.
2. **Trend-dependent**: On narrow/choppy days (range < 100pt), PF drops to 1.20.
3. **Low win rate (31%)**: Max consecutive losses = 25. Average losing streak = 3.3.
4. **Position sizing**: Average 46 MNQ contracts per trade at R=$1000.
5. **Prop firm drawdown**: MaxDD=26.3R. Need R<=$66 for $2,500 DD limit (annual ~$8.3K).

## Data Requirements

- NQ 5-minute OHLCV: 10+ years, continuous contract with rollover handling
- Pre-computed caches: ATR(14), bias direction, regime, session levels
- News calendar CSV
- Data files `.parquet` format (not included due to size/licensing)

## NinjaTrader Port

`ninjatrader/U2LimitOrderStrategy.cs` — NT8 NinjaScript port for independent backtest validation and live auto-execution. All 4 audit fixes ported. 8 NT8-specific fixes applied during cross-audit.

## Tech Stack

- Python 3.11+, pandas, numpy, PyArrow
- NinjaTrader 8 / NinjaScript C#
- All timestamps UTC internally, convert to ET for session logic
- All parameters centralized in `config/params.yaml`

## Research History

60+ experiments across 7 sprints:
- **Sprint 1-3**: Data pipeline, feature engineering, XGBoost (37 indicators all insignificant)
- **Sprint 4**: Filter optimization (243 combos), signal quality tuning
- **Sprint 5**: Trade management, F3 config
- **Sprint 6**: NinjaTrader port
- **Sprint 7**: U2 Limit Order discovery, 4-fix audit cycle, DD optimization

Key finding: **edge is NOT in entry selection or ML prediction**. Edge comes from superior entry price via limit orders at FVG zones + trade management that captures trending moves.
