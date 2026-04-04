#!/usr/bin/env python3
"""
Multi-Angle Adversarial Code Review — Calls Anthropic API.

Usage:
    python scripts/code_review.py <file_path> [--role <role_name>]

This script sends the code to Claude (Sonnet) with THREE different adversarial
review angles, then consolidates the results into a single review report.

The three reviewers:
1. Quant Auditor — checks trading logic correctness, future leakage, edge cases
2. Software Architect — checks code quality, modularity, error handling
3. Risk Adversary — actively tries to break the code, find exploit paths

All three reviews are independent (no context sharing between them).
Results saved to reviews/<filename>_review_<timestamp>.md
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("❌ anthropic package not installed. Run: pip install anthropic")
    sys.exit(1)


# Load CLAUDE.md for strategy context
PROJECT_ROOT = Path(__file__).parent.parent
CLAUDE_MD = PROJECT_ROOT / "CLAUDE.md"
PARAMS_YAML = PROJECT_ROOT / "config" / "params.yaml"
REVIEWS_DIR = PROJECT_ROOT / "reviews"
REVIEWS_DIR.mkdir(exist_ok=True)


def load_context() -> str:
    """Load strategy context for reviewers."""
    context_parts = []
    if CLAUDE_MD.exists():
        context_parts.append(f"=== STRATEGY SPEC (CLAUDE.md) ===\n{CLAUDE_MD.read_text()[:8000]}")
    if PARAMS_YAML.exists():
        context_parts.append(f"\n=== PARAMETERS ===\n{PARAMS_YAML.read_text()}")
    return "\n".join(context_parts)


REVIEWER_PROMPTS = {
    "quant_auditor": {
        "name": "Quant Auditor",
        "emoji": "📊",
        "system": """You are a senior quantitative auditor reviewing trading system code.
You are paranoid about ONE thing above all: FUTURE DATA LEAKAGE (lookahead bias).

Your review checklist:
1. FUTURE LEAKAGE — For every line that accesses price data, verify it only uses data
   available at or before the current bar. Flag ANY use of future data, even subtle ones like:
   - Using .shift(-1) or accessing index+1
   - Using rolling windows that include the current bar in the label
   - HTF candle features used before that candle closes
   - Any operation that could peek at future prices

2. TRADING LOGIC — Does the code correctly implement the strategy described in CLAUDE.md?
   - FVG detection: is the three-candle gap logic correct?
   - Displacement: does it match the ATR-based definition?
   - Session times: are they correct for EST/EDT?
   - Edge cases: market open, market close, rollover gaps

3. LABEL CORRECTNESS — If this is labeling code, verify:
   - Labels are computed from FUTURE prices (this is expected and correct for labels)
   - But features used alongside labels do NOT use future prices
   - Triple barrier logic handles all three barriers correctly

4. NUMERICAL STABILITY — Division by zero? NaN propagation? Integer overflow on large datasets?

Be ruthlessly specific. Quote the exact line numbers. Severity: CRITICAL / WARNING / SUGGESTION.
Do NOT praise the code. Only report problems.""",
    },
    "software_architect": {
        "name": "Software Architect",
        "emoji": "🏗️",
        "system": """You are a senior software architect reviewing production trading system code.
This code will handle real money. Bugs = financial loss.

Your review checklist:
1. MODULARITY — Is each function doing exactly one thing? Can it be unit tested in isolation?
2. ERROR HANDLING — What happens with empty DataFrames? Missing columns? Wrong dtypes?
3. HARDCODED VALUES — Are there magic numbers that should be in params.yaml?
4. TYPE SAFETY — Are type hints present and correct? Would mypy pass?
5. PERFORMANCE — Will this run fast enough on 5 years of 1-minute data (~1.3M rows)?
   Flag any O(n²) operations, unnecessary copies, or repeated full-DataFrame scans.
6. NAMING — Are variable names clear? Would a new team member understand the code?
7. DOCSTRINGS — Does every public function have a docstring explaining inputs, outputs, and edge cases?
8. TESTABILITY — Can the key logic be tested with a 10-row DataFrame? Or does it require
   the full dataset to even run?
9. CONFIG COUPLING — Does the code read from params.yaml correctly? Is there a clean
   separation between config and logic?
10. BOUNDARY CHECK — Does the code respect its directory scope? features/ code should NOT
    import from models/ or backtest/.

Be constructive but firm. Quote exact line numbers. Severity: CRITICAL / WARNING / SUGGESTION.""",
    },
    "risk_adversary": {
        "name": "Risk Adversary",
        "emoji": "🔴",
        "system": """You are a hostile adversarial tester trying to BREAK this trading system code.
Your goal is to find inputs that cause incorrect behavior, crashes, or financial loss.

Your attack vectors:
1. EDGE CASE INPUTS
   - What if the DataFrame has only 1 row? 0 rows? 
   - What if all candles are doji (open == close)?
   - What if there's a gap of 500+ points (flash crash)?
   - What if volume is 0 for an entire session?
   - What if the data has a timezone that's neither UTC nor EST?

2. MARKET REGIME ATTACKS
   - CPI day: 200-point candle in 1 second. Does the FVG detector go insane?
   - Dead market: ATR is near zero. Does displacement threshold produce division by zero?
   - Overnight gap: price gaps 300 points at Sunday open. What happens?

3. DATA CORRUPTION
   - What if a few rows have negative prices?
   - What if timestamps are not sorted?
   - What if there are duplicate timestamps?
   - What if a column is accidentally all NaN?

4. PROP FIRM KILLERS
   - Does the code correctly enforce daily loss limits?
   - Can a sequence of events bypass the 0-for-2 rule?
   - What if slippage exceeds the assumed 1 tick?

5. CONCURRENCY / STATE
   - If this code is called twice rapidly, does it produce consistent results?
   - Does it mutate the input DataFrame? (It shouldn't)

For each vulnerability found:
- Describe the exact attack input
- Explain what goes wrong
- Rate the severity: CRITICAL (financial loss) / HIGH (incorrect signals) / MEDIUM (code crash) / LOW (cosmetic)

You are not here to be helpful. You are here to break things.""",
    },
}


def run_review(client: anthropic.Anthropic, code: str, context: str, reviewer_key: str) -> str:
    """Run a single review angle."""
    reviewer = REVIEWER_PROMPTS[reviewer_key]
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=reviewer["system"],
        messages=[
            {
                "role": "user",
                "content": f"""Review the following code. The strategy specification and parameters are provided for context.

{context}

=== CODE TO REVIEW ===
```python
{code}
```

Provide your review now. Be specific, quote line numbers, and rate severity."""
            }
        ]
    )
    
    return message.content[0].text


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/code_review.py <file_path> [--role <role_name>]")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    
    # Parse optional role argument
    role = None
    if "--role" in sys.argv:
        role_idx = sys.argv.index("--role") + 1
        if role_idx < len(sys.argv):
            role = sys.argv[role_idx]
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    # Check boundary first if role is specified
    if role:
        print(f"🔍 Running boundary check for role: {role}...")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "boundary_check.py"), role, str(file_path)],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            print("❌ Boundary check failed. Fix violations before requesting review.")
            sys.exit(1)
    
    code = file_path.read_text()
    context = load_context()
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set. Export it or add to .env")
        sys.exit(1)
    
    client = anthropic.Anthropic(api_key=api_key)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"{file_path.stem}_review_{timestamp}.md"
    report_path = REVIEWS_DIR / report_name
    
    report_lines = [
        f"# Code Review Report: {file_path}",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Role**: {role or 'unspecified'}",
        f"**File**: {file_path}",
        f"**Lines**: {len(code.splitlines())}",
        "",
        "---",
        "",
    ]
    
    all_critical = []
    
    for key, reviewer in REVIEWER_PROMPTS.items():
        print(f"\n{reviewer['emoji']} Running {reviewer['name']} review...")
        
        try:
            review = run_review(client, code, context, key)
            report_lines.append(f"## {reviewer['emoji']} {reviewer['name']}")
            report_lines.append("")
            report_lines.append(review)
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
            
            # Track critical issues
            if "CRITICAL" in review.upper():
                all_critical.append(f"[{reviewer['name']}] has CRITICAL findings")
            
            print(f"  ✅ {reviewer['name']} complete")
            
        except Exception as e:
            report_lines.append(f"## {reviewer['emoji']} {reviewer['name']}")
            report_lines.append(f"\n❌ Review failed: {e}\n")
            print(f"  ❌ {reviewer['name']} failed: {e}")
    
    # Summary
    report_lines.insert(8, "## Summary")
    if all_critical:
        report_lines.insert(9, "")
        report_lines.insert(10, "### 🚨 CRITICAL ISSUES FOUND — DO NOT MERGE")
        for c in all_critical:
            report_lines.insert(11, f"- {c}")
        report_lines.insert(12, "")
    else:
        report_lines.insert(9, "")
        report_lines.insert(10, "### ✅ No critical issues found")
        report_lines.insert(11, "")
    
    report_path.write_text("\n".join(report_lines))
    
    print(f"\n{'='*60}")
    print(f"📋 Review report saved to: {report_path}")
    if all_critical:
        print(f"🚨 {len(all_critical)} reviewer(s) found CRITICAL issues!")
        print(f"   Fix all critical issues, then run this script again.")
        sys.exit(1)
    else:
        print(f"✅ No critical issues. Code is ready for CTO review.")
        sys.exit(0)


if __name__ == "__main__":
    main()
