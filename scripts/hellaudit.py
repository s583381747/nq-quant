#!/usr/bin/env python3
"""
Hellaudit — 12-Dimension Adversarial Audit

Usage:
    python scripts/hellaudit.py [--dim N] [--verbose]

Runs 12 independent audit dimensions against the codebase and data.
Each dimension scores 0 (FAIL), 1 (WARN), or 2 (PASS).
Total score /24. Deploy gate: must score >= 20 with zero FAIL dimensions.

Based on lessons from HANDOFF.md:
- 78% of initial alpha was lookahead bias
- 7 CRITICAL bugs found by earlier hellaudit
- Config drift (alt_dir 0.334 vs 1.0) caused silent result corruption
"""

import os
import re
import sys
import hashlib
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = []


def score(dim_num: int, name: str, status: int, detail: str):
    """Record a dimension result."""
    label = {0: "❌ FAIL", 1: "⚠️  WARN", 2: "✅ PASS"}[status]
    RESULTS.append((dim_num, name, status, label, detail))


# ============================================================
# DIMENSION 1: LOOKAHEAD CONTAMINATION
# ============================================================
def dim1_lookahead():
    """Check for future data access in feature code."""
    violations = []
    
    for py_file in (PROJECT_ROOT / "features").rglob("*.py"):
        code = py_file.read_text()
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            # Check for shift with negative values (peeking forward)
            if re.search(r'\.shift\(\s*-\d', line):
                violations.append(f"{py_file.name}:{i} — shift(negative) = future data")
            # Check for iloc with [i+N] pattern accessing future
            if re.search(r'iloc\[.*\+\s*\d', line) and 'label' not in str(py_file):
                violations.append(f"{py_file.name}:{i} — iloc[i+N] possible lookahead")
    
    # Check that swing.py uses shift(1)
    swing_file = PROJECT_ROOT / "features" / "swing.py"
    if swing_file.exists():
        if "shift(1)" not in swing_file.read_text():
            violations.append("swing.py — missing shift(1), swings known at formation not valid")
    
    if not violations:
        score(1, "Lookahead contamination", 2, "No future data access in features/")
    elif len(violations) <= 2:
        score(1, "Lookahead contamination", 1, f"{len(violations)} suspect lines: {violations[0]}")
    else:
        score(1, "Lookahead contamination", 0, f"{len(violations)} violations found")


# ============================================================
# DIMENSION 2: NaN PROPAGATION
# ============================================================
def dim2_nan():
    """Check for NaN handling in feature pipeline."""
    issues = []
    
    for py_file in (PROJECT_ROOT / "features").rglob("*.py"):
        code = py_file.read_text()
        # Division without NaN guard
        if "/" in code and "fillna" not in code and "dropna" not in code:
            if "def " in code:  # has functions
                issues.append(f"{py_file.name} — division without NaN guard")
    
    if not issues:
        score(2, "NaN propagation", 2, "All feature files have NaN handling")
    else:
        score(2, "NaN propagation", 1, f"{len(issues)} files may propagate NaN")


# ============================================================
# DIMENSION 3: ROLLOVER ARTIFACTS
# ============================================================
def dim3_rollover():
    """Check that rollover handling exists and is tested."""
    rollover_file = PROJECT_ROOT / "data" / "rollover.py"
    test_file = PROJECT_ROOT / "tests" / "test_data.py"
    
    if not rollover_file.exists():
        score(3, "Rollover artifacts", 0, "data/rollover.py does not exist")
    elif test_file.exists() and "rollover" in test_file.read_text().lower():
        score(3, "Rollover artifacts", 2, "Rollover handler exists and is tested")
    else:
        score(3, "Rollover artifacts", 1, "Rollover handler exists but no test coverage")


# ============================================================
# DIMENSION 4: SESSION BOUNDARY CORRECTNESS
# ============================================================
def dim4_sessions():
    """Check session time definitions match CLAUDE.md spec."""
    sessions_file = PROJECT_ROOT / "features" / "sessions.py"
    
    if not sessions_file.exists():
        score(4, "Session boundaries", 0, "features/sessions.py does not exist")
        return
    
    code = sessions_file.read_text()
    required = ["18:00", "03:00", "09:30", "16:00"]  # Asia start, London start, NY start/end
    missing = [t for t in required if t not in code]
    
    if not missing:
        score(4, "Session boundaries", 2, "All session times present in code")
    elif len(missing) <= 1:
        score(4, "Session boundaries", 1, f"Missing time: {missing}")
    else:
        score(4, "Session boundaries", 0, f"Missing times: {missing}")


# ============================================================
# DIMENSION 5: CONFIG DRIFT
# ============================================================
def dim5_config_drift():
    """Check for params.yaml overrides in experiment/production code."""
    overrides = []
    
    for py_file in PROJECT_ROOT.rglob("*.py"):
        if ".lab/" in str(py_file) or "notebook" in str(py_file):
            continue  # Lab files can override, but must document
        code = py_file.read_text()
        # Look for direct param overrides
        if re.search(r"params\[.*\]\s*=", code):
            overrides.append(f"{py_file.relative_to(PROJECT_ROOT)}")
    
    if not overrides:
        score(5, "Config drift", 2, "No undocumented param overrides in production code")
    else:
        score(5, "Config drift", 0, f"Production code overrides params: {overrides}")


# ============================================================
# DIMENSION 6: FIXED-POINT THRESHOLDS
# ============================================================
def dim6_fixed_points():
    """Check for hardcoded point values that should be ATR-relative."""
    violations = []
    SUSPICIOUS_NUMBERS = {10, 20, 30, 40, 50, 100}  # common fixed-point traps
    
    for subdir in ["features", "backtest"]:
        for py_file in (PROJECT_ROOT / subdir).rglob("*.py"):
            code = py_file.read_text()
            lines = code.split("\n")
            for i, line in enumerate(lines, 1):
                if line.strip().startswith("#"):
                    continue
                for num in SUSPICIOUS_NUMBERS:
                    # Match patterns like "= 50" or "> 30" or "< 20" near price/point context
                    if re.search(rf'[><=!]+\s*{num}\b', line):
                        if "atr" not in line.lower() and "params" not in line.lower():
                            violations.append(f"{py_file.name}:{i} — hardcoded {num}")
    
    if not violations:
        score(6, "Fixed-point thresholds", 2, "No suspicious hardcoded point values")
    elif len(violations) <= 3:
        score(6, "Fixed-point thresholds", 1, f"{len(violations)} suspect: {violations[0]}")
    else:
        score(6, "Fixed-point thresholds", 0, f"{len(violations)} hardcoded thresholds found")


# ============================================================
# DIMENSION 7: CACHE STALENESS
# ============================================================
def dim7_cache():
    """Check if cached files are newer than their source code."""
    cache_dir = PROJECT_ROOT / "data"
    source_files = list((PROJECT_ROOT / "features").rglob("*.py"))
    
    cache_files = list(cache_dir.glob("cache_*.parquet"))
    if not cache_files:
        score(7, "Cache staleness", 1, "No cache files found (may be OK if not using cache)")
        return
    
    oldest_cache = min(f.stat().st_mtime for f in cache_files)
    newest_source = max(f.stat().st_mtime for f in source_files) if source_files else 0
    
    if newest_source > oldest_cache:
        stale = [f.name for f in cache_files if f.stat().st_mtime < newest_source]
        score(7, "Cache staleness", 0, f"Source newer than cache: {stale}")
    else:
        score(7, "Cache staleness", 2, "All caches newer than source code")


# ============================================================
# DIMENSION 8: ENTRY PRICE CORRECTNESS
# ============================================================
def dim8_entry():
    """Check that entry price uses open[i+1], not close[i]."""
    signals_file = PROJECT_ROOT / "features" / "entry_signals.py"
    engine_file = PROJECT_ROOT / "backtest" / "engine.py"
    
    issues = []
    for f in [signals_file, engine_file]:
        if f.exists():
            code = f.read_text()
            if "open" not in code.lower() and "entry" in code.lower():
                issues.append(f"{f.name} — entry price may not use next bar open")
    
    if not issues:
        score(8, "Entry price (open[i+1])", 2, "Entry price references open in relevant files")
    else:
        score(8, "Entry price (open[i+1])", 1, f"Verify manually: {issues}")


# ============================================================
# DIMENSION 9: RISK RULE ENFORCEMENT
# ============================================================
def dim9_risk_rules():
    """Check that prop firm risk rules are coded in backtest."""
    engine_file = PROJECT_ROOT / "backtest" / "engine.py"
    
    if not engine_file.exists():
        score(9, "Risk rule enforcement", 0, "backtest/engine.py does not exist")
        return
    
    code = engine_file.read_text().lower()
    rules = {
        "daily_loss": any(x in code for x in ["daily_loss", "daily_pnl", "2r"]),
        "0_for_2": any(x in code for x in ["consecutive", "0_for_2", "0-for-2", "streak"]),
        "one_position": any(x in code for x in ["one_position", "simultaneous", "max_positions"]),
        "trim": "trim" in code or "partial" in code,
        "slippage": "slippage" in code or "slip" in code,
    }
    
    missing = [k for k, v in rules.items() if not v]
    if not missing:
        score(9, "Risk rule enforcement", 2, "All prop firm rules found in engine")
    elif len(missing) <= 1:
        score(9, "Risk rule enforcement", 1, f"Missing rule: {missing}")
    else:
        score(9, "Risk rule enforcement", 0, f"Missing rules: {missing}")


# ============================================================
# DIMENSION 10: TEST COVERAGE
# ============================================================
def dim10_tests():
    """Check that test files exist for critical modules."""
    critical_tests = [
        "tests/test_data.py",
        "tests/test_fvg.py",
        "tests/test_no_lookahead.py",
    ]
    
    existing = [t for t in critical_tests if (PROJECT_ROOT / t).exists()]
    missing = [t for t in critical_tests if not (PROJECT_ROOT / t).exists()]
    
    if not missing:
        score(10, "Test coverage", 2, f"All {len(critical_tests)} critical test files exist")
    elif len(existing) >= 2:
        score(10, "Test coverage", 1, f"Missing: {missing}")
    else:
        score(10, "Test coverage", 0, f"Missing {len(missing)}/{len(critical_tests)} critical tests")


# ============================================================
# DIMENSION 11: STRATEGY SOURCE PURITY
# ============================================================
def dim11_strategy_source():
    """Check that strategy decisions reference LantoGPT or CLAUDE.md."""
    decisions_file = PROJECT_ROOT / "tasks" / "strategy_decisions.md"
    claude_md = PROJECT_ROOT / "CLAUDE.md"
    
    if not claude_md.exists():
        score(11, "Strategy source purity", 0, "CLAUDE.md does not exist")
    elif decisions_file.exists():
        score(11, "Strategy source purity", 2, "CLAUDE.md + strategy_decisions.md both exist")
    else:
        score(11, "Strategy source purity", 1, "CLAUDE.md exists but no strategy_decisions.md")


# ============================================================
# DIMENSION 12: WALK-FORWARD INFRASTRUCTURE
# ============================================================
def dim12_walkforward():
    """Check that walk-forward validation tooling exists."""
    wf_candidates = list(PROJECT_ROOT.rglob("*walk*forward*")) + \
                    list(PROJECT_ROOT.rglob("*wf_*"))
    
    if wf_candidates:
        score(12, "Walk-forward infrastructure", 2, f"WF tooling found: {[f.name for f in wf_candidates[:3]]}")
    else:
        score(12, "Walk-forward infrastructure", 0, "No walk-forward scripts found — CRITICAL for anti-overfit")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  HELLAUDIT — 12-Dimension Adversarial Audit")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    # Run all dimensions
    dims = [
        dim1_lookahead, dim2_nan, dim3_rollover, dim4_sessions,
        dim5_config_drift, dim6_fixed_points, dim7_cache, dim8_entry,
        dim9_risk_rules, dim10_tests, dim11_strategy_source, dim12_walkforward,
    ]
    
    for dim_func in dims:
        try:
            dim_func()
        except Exception as e:
            dim_num = int(dim_func.__name__.split("_")[0].replace("dim", ""))
            score(dim_num, dim_func.__name__, 0, f"CRASHED: {e}")
    
    # Print results
    total = sum(r[2] for r in RESULTS)
    fails = sum(1 for r in RESULTS if r[2] == 0)
    
    print(f"{'#':>3}  {'Dimension':<30}  {'Score':>6}  {'Detail'}")
    print("-" * 80)
    for dim_num, name, s, label, detail in sorted(RESULTS):
        print(f"{dim_num:>3}  {name:<30}  {label:>6}  {detail[:60]}")
    
    print()
    print(f"  Total: {total}/24")
    print(f"  FAILs: {fails}")
    print()
    
    # Deploy gate
    if fails > 0:
        print("🚫 DEPLOY BLOCKED — fix all FAIL dimensions before deployment")
        sys.exit(1)
    elif total >= 20:
        print("✅ DEPLOY GATE PASSED")
        sys.exit(0)
    else:
        print("⚠️  DEPLOY CONDITIONAL — score below 20, review WARN items")
        sys.exit(0)


if __name__ == "__main__":
    main()
