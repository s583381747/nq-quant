# NQ Quant Team — System Architecture (v2, HANDOFF-informed)

## Design Philosophy
Simulate a professional quant team using Claude Code sessions as agents.
Each session = one role. Roles communicate through files, never shared sessions.

### Lessons from HANDOFF that shaped this architecture
1. ML signal selection does NOT beat pure rules → no ML Engineer, replaced by Experiment Runner
2. 78% of initial alpha was lookahead → Risk Officer is the most valuable role
3. Config drift (params vs runtime) caused silent corruption → Config Auditor in review pipeline
4. Cache (1s) vs recompute (800s) = iteration vs stagnation → cache is first-class infra
5. Walk-forward is the ONLY truth → every improvement claim requires WF validation
6. Fixed-point thresholds break across regimes → all thresholds ATR-relative

---

## Team Roster

| Role | Layer | Scope | Writes Code? |
|------|-------|-------|-------------|
| **External Auditor** | Meta | System architecture, roles, review pipeline | Review scripts only |
| **CTO** | Governance | Task assignment, decisions, final approval | Never |
| **Data Engineer** | Execution | data/, config/news_calendar.csv | Yes |
| **Quant Researcher** | Execution | features/, viz/, config/params.yaml | Yes |
| **Experiment Runner** | Execution | .lab/, models/ (analysis only), notebooks/ | Yes |
| **Backtest Engineer** | Execution | backtest/ | Yes |
| **Risk Officer** | Verification | tests/, scripts/hellaudit.py | Tests only |
| **Execution Engineer** | Execution | live/ | Yes (Phase 4+) |

### Activation Frequency
- **Every session**: Data Engineer, Quant Researcher, Experiment Runner, Backtest Engineer
- **Every commit**: Risk Officer (via code_review.py), CTO (review reports)
- **Phase gates only**: External Auditor
- **Phase 4+ only**: Execution Engineer

---

## Three-Layer Enforcement

### Layer 1: Prompt (soft)
Each .claude/profiles/*.md contains identity lock, hard constraints, boundary self-check, quality gates.

### Layer 2: Filesystem (hard)
- `scripts/boundary_check.py <role>` — validates file ownership against CODEOWNERS
- `scripts/pre-commit` — git hook blocks out-of-scope commits
- `.claude/current_role` — must be set before any commit

### Layer 3: Review Pipeline (adversarial)
1. Boundary check → file ownership
2. Code review (4 adversarial reviewers via Anthropic API):
   - Quant Auditor — lookahead, trading logic, numerical stability
   - Software Architect — modularity, error handling, performance
   - Risk Adversary — edge cases, regime attacks, prop firm killers
   - Config Auditor — params.yaml drift, undocumented overrides, ATR-relative
3. Hellaudit (12 dimensions) — deploy gate ≥20/24, zero FAILs
4. CTO review — strategy alignment

---

## Cache Architecture

Caches are first-class artifacts. Version-tagged, never deleted.

```
data/cache_signals_<ver>.parquet   ← features/entry_signals.py
data/cache_bias_<ver>.parquet      ← features/bias.py
data/cache_regime_<ver>.parquet    ← features/bias.py
data/cache_atr_flu_<ver>.parquet   ← features/displacement.py
```

Rules: version increments on source change, hellaudit checks freshness,
experiments must declare cache version, old caches kept for reproducibility.

---

## LantoGPT Integration
- API: `POST https://www.lantogpt.com/api/chat`
- Sole strategy authority — never use external ICT sources
- CTO consults for ambiguity, logs to tasks/strategy_decisions.md

---

## External Auditor — System Governance

Activates at: phase gates, recurring patterns (3+), strategic pivots, quarterly, post-mortems.

Can directly modify: role profiles, CODEOWNERS, boundary_check.py, code_review.py, hellaudit.py.

12-dimension audit: lookahead, NaN, rollover, sessions, config drift, fixed-points,
cache staleness, entry price, risk rules, test coverage, strategy source, walk-forward infra.

---

## Anti-Overfit Protocol
Every optimization must pass: WF 8+ windows, bootstrap CI above zero,
IS→OOS shrink <50%, smooth parameter sensitivity, domain justification.

---

## Golden Rules
1. CTO never writes code
2. One session = one role
3. All communication through files
4. No code without review
5. Risk Officer vetoes code, External Auditor vetoes architecture
6. Walk-forward or it didn't happen
7. LantoGPT is sole strategy oracle
8. Cache versions are sacred
9. All thresholds ATR-relative
10. The system itself must evolve
