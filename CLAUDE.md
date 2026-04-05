# NQ Quantitative Trading System — CLAUDE.md

## Project Overview
Quantitative system for NQ (Nasdaq 100 E-mini Futures) targeting Prop Firm accounts.
Strategy is based on Lanto's ICT-derivative methodology: FVG as magnet/support/resistance,
multi-timeframe bias, displacement-based entries, and strict risk management.

---

## 1. INSTRUMENT & CONTRACT SPECS

- Instrument: NQ (Nasdaq 100 E-mini Futures)
- Tick size: 0.25 points
- Tick value: $5.00
- Point value: $20.00
- Commission: ~$2.05/side per contract (mini), ~$0.62/side (micro)
- Micro NQ (MNQ): 1/10th of NQ, point value = $2.00

---

## 2. SESSION DEFINITIONS (All times EST/ET)

| Session  | Start     | End       |
|----------|-----------|-----------|
| Asia     | 6:00 PM   | 3:00 AM   |
| London   | 3:00 AM   | 9:30 AM   |
| New York | 9:30 AM   | 4:00 PM   |

- **Opening Range Observation**: 9:30 AM – 10:00 AM EST → NO TRADES, observation only (cash market opens at 9:30)
- **Primary trading session**: NY (after 10:00 AM)
- Asia/London sessions: only trade "if up a lot", otherwise observe only
- **Daylight Saving Time**: Handle EST ↔ EDT transitions carefully in data pipeline

---

## 3. BIAS DETERMINATION

### 3.1 HTF FVG Analysis (Priority: Daily > 4H > 1H > 15m)
- Check for active HTF FVGs on D, 4H, 1H timeframes
- FVG = magnet/support/resistance; price is attracted toward unfilled gaps
- Higher TF FVGs carry more weight, UNLESS the HTF FVG is low quality (small, wicky)

### 3.2 HTF FVG State Machine
Each HTF FVG maintains a status:
- `untested` → price has not reached the FVG yet (acts as magnet/target)
- `tested_rejected` → price entered FVG but closed back out with displacement (continuation signal)
- `invalidated` → price **closed** through the FVG (reversal, seek next FVG/liquidity level)

### 3.3 Overnight/Session Level Bias
- Mark Asia session high/low as liquidity levels
- Mark London session high/low as liquidity levels
- Mark overnight high/low
- Opening range (9:00-10:00) high/low
- If price approaches overnight high but fails → short bias
- If price approaches overnight low but fails → long bias
- Observe price reaction at these levels to confirm or flip bias

---

## 4. ENTRY RULES

### 4.1 Core Setup: FVG Test + Rejection
1. Price approaches a valid FVG on 1m/5m timeframe
2. **Rejection candle**: candle tests the FVG, then closes on the correct side
   - Long: close ABOVE the FVG
   - Short: close BELOW the FVG
   - Candle must show displacement (not a doji, not wicky)
3. Entry on the close of the rejection candle

### 4.2 Reversal Entry
- Price sweeps Asia/London high or low
- Post-sweep: price shows displacement in opposite direction (long candles, no chop)
- A small FVG forms in the reversal direction → enter on test of this FVG

### 4.3 Continuation Entry
- Price tests a known FVG and gets rejected
- FVG status becomes `tested_rejected`
- Enter in the direction of the original trend

### 4.4 Uncertain/Retest Entry
- If reversal is NOT fluent, wait for price to retest the FVG
- Confirmation: retest + close on correct side → then enter

### 4.5 Entry Timeframes
- All entries on 1m or 5m charts
- HTF (1H, 4H, D) for bias only, never for entry timing

---

## 5. FVG DEFINITION & QUALITY

### 5.1 What is an FVG
Three consecutive candles where candle 1's high/low and candle 3's high/low do not overlap,
creating a gap at candle 2. Bullish FVG: candle 1 high < candle 3 low. Bearish FVG: candle 1 low > candle 3 high.

### 5.2 FVG Quality Criteria
- **Displacement**: the FVG-creating candle must be large relative to recent price action
- **Liquidity sweep**: FVG should form after taking out internal or external liquidity
- **Fluency**: recent candles leading into the FVG should be directional, not choppy
- **Wick ratio**: candles forming the FVG should have high body-to-range ratio (minimal wicks)
- Higher TF FVGs are inherently higher quality

### 5.3 FVG Invalidation
- Price **closes** through the FVG → FVG is invalidated
- An invalidated FVG can become an **inversion FVG** (former support becomes resistance, vice versa)

---

## 6. DISPLACEMENT & FLUENCY (Quantization)

### 6.1 Displacement — ⚠️ REQUIRES TUNING
A candle shows "displacement" when it demonstrates conviction and directional commitment.
**Starting quantitative definition (tune in Phase 1):**
- Candle body (abs(close - open)) > `DISPLACEMENT_ATR_MULT` × ATR(14)
- Candle body/range ratio > `DISPLACEMENT_BODY_RATIO`
- Candle engulfs at least 1 prior candle

| Parameter                | Starting Value | Tune Range  |
|--------------------------|---------------|-------------|
| DISPLACEMENT_ATR_MULT    | 0.8           | 0.5 – 1.5   |
| DISPLACEMENT_BODY_RATIO  | 0.60          | 0.50 – 0.80 |

### 6.2 Fluency Score — ⚠️ REQUIRES TUNING
Measures how directional/clean recent price action is (opposite of "chop").
Computed over a rolling window of N candles.

**Components:**
1. `directional_ratio`: fraction of candles moving in the same direction (e.g., 4/5 bullish = 0.8)
2. `avg_body_ratio`: mean(body/range) across window — high = clean candles, low = wicky
3. `avg_bar_size_vs_atr`: mean(range/ATR) — large bars = conviction

**Composite:** `fluency_score = w1 * directional_ratio + w2 * avg_body_ratio + w3 * avg_bar_size_vs_atr`

| Parameter              | Starting Value | Tune Range |
|------------------------|---------------|------------|
| FLUENCY_WINDOW         | 6 candles     | 4 – 10     |
| FLUENCY_THRESHOLD      | 0.60          | 0.40 – 0.80|
| w1, w2, w3 (weights)   | 0.4, 0.3, 0.3| —          |

### 6.3 Bad Candle Detection
Flag candles as "bad" (avoid trading around them):
- Doji: body/range < 0.15
- Long wick candle: max(upper_wick, lower_wick) / range > 0.60

---

## 7. SWING POINT DETECTION — ⚠️ REQUIRES TUNING

"Relevant highs and lows" = fractal swing points.

| Parameter         | Starting Value | Tune Range |
|-------------------|---------------|------------|
| SWING_LEFT_BARS   | 3             | 2 – 5      |
| SWING_RIGHT_BARS  | 1             | 1 – 3      |

- Swing High: high[i] > high[i-1..i-L] AND high[i] > high[i+1..i+R]
- Swing Low:  low[i] < low[i-1..i-L] AND low[i] < low[i+1..i+R]
- Higher TF swings (1H, 4H, D) serve as external liquidity targets

---

## 8. STOP LOSS (Model Stop)

1. **Primary**: open price of the candle that tested/rejected the FVG
2. **Fallback**: if that candle's range < `SMALL_CANDLE_ATR_MULT` × ATR(14),
   use the previous swing low (long) or swing high (short)

| Parameter              | Starting Value | Tune Range |
|------------------------|---------------|------------|
| SMALL_CANDLE_ATR_MULT  | 0.30          | 0.20 – 0.50|

---

## 9. POSITION SIZING

- Fixed dollar risk per trade, adjust number of contracts based on stop distance
- **Normal risk**: 1R = $1000
- **Reduced risk**: 0.5R = $500 (applied on Monday, Friday, or when fluency_score < FLUENCY_THRESHOLD)

### Calculation:
```
stop_distance_points = abs(entry_price - stop_price)
risk_per_contract = stop_distance_points × $20 (mini) or × $2 (micro)
num_contracts = floor(R_amount / risk_per_contract)
```

Example: 1R = $1000, stop = 25 points → $500/contract → 2 mini contracts

---

## 10. EXIT / TRADE MANAGEMENT

### 10.1 Targets
- **Internal liquidity target**: ~1R away (nearby swing high/low, or slightly higher TF FVG fill)
- **Ultimate target**: HTF FVG entry/fill, or HTF swing point (external liquidity)
- Target RR typically 2:1 – 3:1 or higher
- News candles are also valid liquidity targets

### 10.2 Trim & Trail
1. When price reaches internal liquidity (~1R): **trim 50%** of position
2. Move stop to **break-even** on remaining 50%
3. Trail stop on remaining position: place below/above the most promising 1m/5m FVG
   that acts as support/resistance
4. Let remaining position run to ultimate target OR get stopped at trailed FVG level

### 10.3 BE Sweep Accounting
- If BE is hit after trim, this trade is overall PROFITABLE (locked in gains from trim)
- Does NOT count as a loss for 0-for-2 rule
- Does NOT count toward 2R daily loss limit

---

## 11. RISK MANAGEMENT & FILTERS

### 11.1 Daily Loss Limit
- Maximum daily loss: 2R
- Once hit → stop trading for the day, no exceptions

### 11.2 Two-Strike Rule (0-for-2)
- 2 consecutive LOSING trades → stop trading for the day
- A BE-sweep after trim is NOT a losing trade
- This rule applies BEFORE the 2R limit (can stop you earlier)

### 11.3 Trade Frequency
- Target: 1 quality trade per day
- No hard maximum if winning, but quality over quantity always

### 11.4 One Position at a Time
- NEVER hold simultaneous positions
- Must be flat before entering a new trade

### 11.5 News Filter
- **Pre-news blackout**: 60 minutes before high-impact events → no new entries
- **Post-news cooldown**: 5 minutes after event → trading resumes
- **Existing positions**: not affected by news filter (stay open)
- **High-impact events**: CPI, FOMC, NFP, GDP, PCE, Jobless Claims, PPI
- News schedule: maintain as CSV, update weekly (source: Forex Factory or similar)

### 11.6 Re-Entry After Stop Loss
- No strict rule against re-entering same FVG
- Requires fresh displacement and good candle closure confirming direction
- Treat as a new setup — must meet all entry criteria again

### 11.7 Session Filter
- Primary: NY session only (after 10:00 AM EST)
- 9:00-10:00 AM: observation only, mark opening range high/low
- Asia/London: observe, mark session highs/lows, only trade if already profitable

---

## 12. PROP FIRM CONSTRAINTS (Hardcoded in Backtest)

- **Daily loss limit**: varies by firm (e.g., $1,000 – $2,200 depending on account size)
- **Trailing max drawdown**: varies by firm
- **Safety buffer**: resource curve max drawdown must stay 30%+ below firm's red line
- Commission: $2.05/side/contract (mini), $0.62/side/contract (micro)
- Slippage model: 1 tick normal, 3-5 ticks during high-impact news

---

## 13. MULTI-TIMEFRAME ALIGNMENT RULES

- Always use `shift(1)` when merging HTF features into LTF DataFrame
- Use `ffill()` for forward-filling HTF state into 1m bars
- NEVER allow future data leakage — HTF candle features only available after candle closes
- Resample from 1m data to build 5m, 15m, 1H, 4H if not available natively

---

## 14. LABELING (Triple Barrier Method) — ⚠️ REQUIRES TUNING

For supervised learning (Phase 2), label each potential entry:

| Parameter           | Starting Value | Tune Range    |
|---------------------|---------------|---------------|
| TP_POINTS           | 20            | 15 – 40       |
| SL_POINTS           | 10            | 8 – 20        |
| MAX_HOLDING_BARS    | 20            | 10 – 40       |

- Label = 1 if TP hit first
- Label = 0 if SL hit first
- Label = 0 if MAX_HOLDING_BARS exceeded without hitting either (trade timed out = bad setup)

---

## 15. PROJECT STRUCTURE

```
nq-quant/
├── CLAUDE.md              # THIS FILE — strategy spec & coding rules
├── config/
│   ├── params.yaml        # All tunable parameters (centralized)
│   └── news_calendar.csv  # High-impact event schedule
├── data/
│   ├── loader.py          # Load raw data from source
│   ├── cleaner.py         # Handle NaN, timezone, DST
│   ├── resampler.py       # 1m → 5m/15m/1H/4H
│   └── rollover.py        # Contract rollover handling
├── features/
│   ├── fvg.py             # FVG detection across all timeframes
│   ├── displacement.py    # Displacement & fluency scoring
│   ├── swing.py           # Swing high/low detection (fractal)
│   ├── sessions.py        # Session marking, Asia/London/NY highs-lows
│   ├── mtf.py             # Multi-timeframe feature alignment
│   ├── news_filter.py     # News blackout window logic
│   └── labeler.py         # Triple barrier labeling
├── models/
│   ├── train_xgb.py       # XGBoost training pipeline
│   ├── evaluate.py        # Precision, recall, feature importance
│   └── saved/             # Saved model files (.json)
├── backtest/
│   ├── engine.py          # Core backtest loop with prop firm rules
│   ├── position_sizer.py  # Contract calculation from $ risk + stop
│   ├── trade_manager.py   # Trim, BE, trail logic
│   └── report.py          # Generate HTML backtest report
├── live/                   # Phase 4 (future)
│   ├── gateway.py
│   └── risk_guard.py
├── viz/
│   ├── chart.py           # mplfinance K-line charts with FVG overlay
│   └── feature_check.py   # Visual verification of feature accuracy
├── tests/
│   ├── test_fvg.py
│   ├── test_displacement.py
│   ├── test_sessions.py
│   └── test_no_lookahead.py  # Future function leak test
└── notebooks/
    └── exploration.ipynb   # Ad-hoc analysis
```

---

## 16. CODING CONVENTIONS

- Python 3.11+
- Data format: `.parquet` (PyArrow backend)
- All timestamps: UTC internally, convert to EST only for session logic
- All tunable parameters live in `config/params.yaml` — NEVER hardcode thresholds
- Type hints on all functions
- Docstrings on all public functions
- Unit tests for every feature function
- No print() in production code — use `logging` module

---

## 17. PHASE ROADMAP

| Phase | Focus                        | Deliverable                                      |
|-------|------------------------------|--------------------------------------------------|
| 0     | Data pipeline                | 5-year NQ 1m parquet, clean, no NaN              |
| 1     | Feature engineering          | 50-col DataFrame (X) + label (Y), visually verified |
| 2     | XGBoost baseline             | Model file, precision > 55% on validation set    |
| 3     | Prop firm backtest           | HTML report, equity curve, safety buffer > 30%    |
| 4     | Live execution / copilot     | Paper trading stable 1 week                       |
| 5     | Transformer R&D (optional)   | EV improvement > 15% vs XGBoost                  |

---

## 18. ALL TUNABLE PARAMETERS SUMMARY

These parameters need iterative tuning via Phase 1 visual verification:

```yaml
# config/params.yaml
displacement:
  atr_mult: 0.8          # min body size as multiple of ATR(14)
  body_ratio: 0.60       # min body/range ratio for displacement candle
  
fluency:
  window: 6              # rolling window size (candles)
  threshold: 0.60        # min score to consider price "fluent"
  w_directional: 0.4
  w_body_ratio: 0.3
  w_bar_size: 0.3

swing:
  left_bars: 3
  right_bars: 1

stop_loss:
  small_candle_atr_mult: 0.30  # fallback to swing if candle < this × ATR

position:
  normal_r: 1000         # dollars
  reduced_r: 500         # dollars (Mon/Fri/low fluency)
  
risk:
  daily_max_loss_r: 2.0
  max_consecutive_losses: 2

labeling:
  tp_points: 20
  sl_points: 10
  max_holding_bars: 20

news:
  blackout_minutes_before: 60
  cooldown_minutes_after: 5

sessions:
  observation_start: "09:00"  # EST
  observation_end: "10:00"    # EST
  ny_start: "09:30"
  ny_end: "16:00"
  asia_start: "18:00"
  asia_end: "03:00"
  london_start: "03:00"
  london_end: "09:30"

slippage:
  normal_ticks: 1
  news_ticks: 4             # 3-5 range, use midpoint

bad_candle:
  doji_body_ratio: 0.15
  long_wick_ratio: 0.60
```

---

## 19. CRITICAL REMINDERS FOR CLAUDE CODE

1. **No future leakage** — every feature must use only data available at or before the current bar
2. **shift(1) everything** from HTF — a 1H candle's features are only available after it closes
3. **Centralize all thresholds** in params.yaml — if you hardcode a number, you're doing it wrong
4. **Visual verification** — every feature module must have a companion viz function that overlays signals on K-line charts
5. **One module per session** — don't try to build everything at once, context window will overflow
6. **Test for lookahead** — `tests/test_no_lookahead.py` should verify that shuffling future data doesn't change current features

---

## 20. CTO AUTONOMOUS EXPLORATION PROTOCOL

### Role Model
Claude operates as **CTO** — responsible for high-quality decisions, NOT writing code directly.
- **CTO**: decides what to explore, evaluates results, kills dead branches, picks next direction
- **Coder Agent**: writes code, runs experiments, produces numbers
- **Hell-Audit Agent**: audits coder output for bugs, lookahead, PnL errors
- **CTO verifies**: reviews audit + numbers, accepts or rejects, updates decision tree

### Decision Tree Protocol
Every exploration follows this structure:
```
HYPOTHESIS → DIAGNOSTIC → RESULT → DECISION
  ├── Result positive → IMPLEMENT + AUDIT + VERIFY
  ├── Result neutral → LOG + MOVE TO NEXT BRANCH
  └── Result negative → LOG + MOVE TO NEXT BRANCH
```

### Self-Feedback Loop
After each experiment:
1. Record result in decision tree (update recovery memory)
2. Compare against baseline (U2: PF=1.87, +1270R, PPDD=48.4)
3. If improvement: hell-audit → verify PnL → accept
4. If no improvement: log why, move up one branch
5. After exhausting a category: summarize findings, pick next category

### Exploration Queue (prioritized)
```
Category A: Bias/MTF Optimization
  A1. Diagnostic — decompose bias component value (which components help U2?)
  A2. Daily FVG inclusion (highest priority per spec, currently missing)
  A3. FVG quality weighting (size/age/distance)
  A4. Bias weight sweep (0.4/0.3/0.3 → optimal)
  A5. Asia/London sweep detection
  A6. Continuous fluency dampening

Category B: Trade Management Refinement
  B1. Dynamic TP based on HTF FVG distance (not fixed IRL×2.0)
  B2. Volatility-adaptive stop tightening
  B3. Time-based exit (fade runners after N bars)
  B4. Partial trim at multiple levels (25%/25%/25%/25%)

Category C: Signal Generation
  C1. Short-side U2 (bearish FVG zone limit sell)
  C2. Multi-timeframe FVG confluence (1H+4H same zone)
  C3. Inversion FVG (invalidated FVG as reverse S/R)

Category D: Portfolio/Execution
  D1. F3+U2 combined equity curve analysis
  D2. Correlation between F3 and U2 trades (diversification value)
  D3. Dynamic R-sizing based on rolling DD
```

### Kill Criteria
- PF drops > 0.05 from baseline → reject
- PPDD drops > 10% from baseline → reject
- MaxDD increases > 15% from baseline → reject
- Fewer than 500 trades → insufficient sample, widen parameters
- p-value > 0.10 on improvement → not significant, reject

### Current Baselines (NEVER forget these)
| Strategy | R | PF | PPDD | MaxDD | Trades |
|----------|------|------|------|-------|--------|
| U2 | +1270.2 | 1.87 | 48.38 | 26.3R | 2012 |
| F3 | +86.0 | 1.21 | 4.78 | 18.0R | 1127 |
