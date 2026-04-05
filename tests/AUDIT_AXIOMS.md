# Backtest Physical Constraint Axioms

**Every backtest engine must satisfy ALL axioms below. No exceptions.**

These axioms encode physical constraints of real trading.
If an engine violates an axiom, the backtest is WRONG — regardless
of whether the code is "well-designed" or "conservative."

The same-bar-skip bug passed 4 adversarial audits because auditors
checked code correctness, not physical correctness. These axioms
prevent that failure class permanently.

---

## AXIOM 1: FILL IRREVERSIBILITY
**Once an order fills, it cannot be un-filled.**

A limit buy at price P fills when the market price reaches P.
After the fill, the trader owns the position. The fact that price
continues through the stop on the same bar does NOT un-fill the order.

**Test**: Search for any `continue` or `skip` after a fill condition
is TRUE. If the engine decides NOT to enter after price has reached
the limit level, it violates this axiom.

**Pattern to catch**:
```python
# VIOLATION:
if low <= entry_price:        # fill condition TRUE
    if low <= stop_price:     # but stop also hit
        continue              # ← SKIP = un-filling the order
```

**Correct**:
```python
if low <= entry_price:        # fill condition TRUE
    if low <= stop_price:
        # Enter AND immediately stop out = -1R loss
        record_same_bar_stop()
```

---

## AXIOM 2: TEMPORAL CAUSALITY
**Actions at time T can only use information from times ≤ T.**

No feature, signal, or decision at bar T may use data from bar T+1
or later. This includes:
- OHLC of the current bar (only O is known at bar open; HLC are future)
- Any computed feature that uses forward-looking windows
- Exit decisions that use future bars' data

**Exception**: Bar T's CLOSE can be used for decisions executed at
bar T+1's OPEN (e.g., "if today's close > X, buy tomorrow's open").
This is NOT lookahead because the decision is made after bar T completes.

**Test**: Corrupt all data after bar T. Features and signals at bar T
must remain unchanged. (Implemented in test_no_feature_lookahead)

---

## AXIOM 3: EXECUTION LATENCY
**There is always a delay between decision and execution.**

- A signal generated at bar T's close → earliest execution at bar T+1's open
- A limit order placed during bar T → earliest fill at bar T (if price reaches level)
- A market order → fills at next available price (next bar's open in bar-based backtest)

**Test**: For every entry, verify that the decision information was
available BEFORE the execution time. For limit orders, the order must
have been placed before the fill bar.

---

## AXIOM 4: ORDER TYPE CONSISTENCY
**The entry model must match the claimed order type.**

If the strategy claims to use limit orders:
- Entry price = the limit level (not bar open, not close)
- No slippage on entry (limit orders fill at limit price or better)
- Fill requires price to REACH the limit level

If the strategy claims to use market orders:
- Entry price = next bar's open + slippage
- Slippage must be modeled

**Test**: Check that entry_price matches the claimed order type.

---

## AXIOM 5: WORST-CASE INTRABAR RESOLUTION
**When multiple levels are hit on the same bar, assume worst case.**

With bar-based (not tick) data, we cannot determine intrabar price path.
When both stop and TP are hit on the same bar:
- Assume STOP is hit first (pessimistic)
- This is conservative and acceptable

When both limit entry and stop are hit on the same bar:
- The entry DOES fill (Axiom 1)
- Then the stop is hit = immediate loss
- Do NOT skip the trade (violates Axiom 1)

**Test**: Count trades where entry and stop are on the same bar.
If zero, the engine may be skipping them.

---

## AXIOM 6: COMMISSION AND SLIPPAGE COMPLETENESS
**Every trade must include realistic transaction costs.**

- Commission: charged on ALL contracts, both open and close
- Slippage: applied to market orders and stop orders
- Limit orders: no slippage (fill at limit price)
- Stop orders: 1 tick slippage minimum (price gaps through stop)

**Test**: Verify commission > 0 for every trade. Verify slippage on stop exits.

---

## AXIOM 7: POSITION SIZING CONSISTENCY
**Risk denominator must match actual risk taken.**

If position size is calculated as `contracts = risk_dollars / (stop_dist × point_value)`:
- `stop_dist` in the sizing must be the SAME as `stop_dist` in the PnL calculation
- If the stop is tightened after sizing, the risk denominator is wrong

**Test**: For each trade, verify that `stop_dist` used in R-multiple
calculation matches the stop actually used in the trade.

---

## AXIOM 8: STATE RESET COMPLETENESS
**All per-trade and per-day state must reset correctly.**

- Daily loss counter: resets at day boundary
- Consecutive loss counter: resets at day boundary
- Position state: fully cleared after exit
- Pending orders: cleared at day boundary (no overnight limit orders)

**Test**: Verify no state leaks between days or between trades.

---

## HOW TO USE THESE AXIOMS

### For automated testing:
Each axiom has a corresponding test in `tests/test_execution_realism.py`.
Run after EVERY engine change.

### For human/AI audits:
Before approving any engine code, the auditor MUST answer these questions:

1. "Show me where a filled order could be un-filled." (Axiom 1)
2. "Show me where bar T uses data from bar T+1." (Axiom 2)
3. "Show me the delay between decision and execution." (Axiom 3)
4. "Show me the entry price matches the order type." (Axiom 4)
5. "Show me what happens when stop+TP or stop+entry hit on same bar." (Axiom 5)
6. "Show me the commission and slippage on each exit type." (Axiom 6)
7. "Is the R denominator the same stop used in the trade?" (Axiom 7)
8. "What state carries over between trades/days?" (Axiom 8)

If the auditor cannot answer any question with a specific line number,
the audit is INCOMPLETE.

### For new engine development:
Before writing ANY backtest engine, read these axioms. Design the
engine to satisfy all 8 from the start. It's 100x cheaper to build
correctly than to find bugs after optimization.
