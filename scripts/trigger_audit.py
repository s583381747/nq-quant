#!/usr/bin/env python3
"""
External Auditor Trigger Check — Runs at end of EVERY session.

Usage:
    python scripts/trigger_audit.py

This script is role-agnostic. ANY team member can (and should) run it.
It checks automated conditions and writes a trigger file if audit is needed.
Nobody controls when audit happens — the conditions do.

If triggered, next session MUST be External Auditor before any other work.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
TRIGGER_FILE = PROJECT_ROOT / ".claude" / "audit_triggered.json"
TASKS_DIR = PROJECT_ROOT / "tasks"
REVIEWS_DIR = PROJECT_ROOT / "reviews"
AUDITS_DIR = PROJECT_ROOT / "audits"

TRIGGERS_FIRED = []


def check_phase_gate() -> bool:
    """Trigger 1: A phase was just completed in tasks/completed.md."""
    completed = TASKS_DIR / "completed.md"
    if not completed.exists():
        return False
    
    content = completed.read_text()
    # Look for phase completion markers added in last entry
    lines = content.strip().split("\n")
    for line in lines[-10:]:  # check last 10 lines
        if re.search(r"phase\s*[0-5]\s*(complete|done|finished)", line, re.IGNORECASE):
            TRIGGERS_FIRED.append(f"PHASE_GATE: {line.strip()}")
            return True
    return False


def check_recurring_bugs() -> bool:
    """Trigger 2: Same bug category found 3+ times in reviews/."""
    if not REVIEWS_DIR.exists():
        return False
    
    critical_counts = {}
    for review_file in REVIEWS_DIR.glob("*.md"):
        content = review_file.read_text()
        # Count CRITICAL findings by category
        for match in re.finditer(r"CRITICAL.*?(?:lookahead|nan|config|boundary|cache|overfit)", 
                                  content, re.IGNORECASE):
            category = match.group(0).lower()
            for key in ["lookahead", "nan", "config", "boundary", "cache", "overfit"]:
                if key in category:
                    critical_counts[key] = critical_counts.get(key, 0) + 1
    
    repeats = {k: v for k, v in critical_counts.items() if v >= 3}
    if repeats:
        TRIGGERS_FIRED.append(f"RECURRING_BUGS: {repeats}")
        return True
    return False


def check_review_rubber_stamp() -> bool:
    """Trigger 3: 10+ consecutive reviews with zero CRITICAL findings = suspicious."""
    if not REVIEWS_DIR.exists():
        return False
    
    review_files = sorted(REVIEWS_DIR.glob("*.md"), key=lambda f: f.stat().st_mtime)
    if len(review_files) < 10:
        return False
    
    recent_10 = review_files[-10:]
    all_clean = all(
        "CRITICAL" not in f.read_text().upper() 
        for f in recent_10
    )
    
    if all_clean:
        TRIGGERS_FIRED.append("REVIEW_RUBBER_STAMP: 10 consecutive reviews with 0 CRITICALs")
        return True
    return False


def check_time_based() -> bool:
    """Trigger 4: No audit in 30+ days (quarterly review)."""
    if not AUDITS_DIR.exists():
        return True
    
    audit_files = list(AUDITS_DIR.glob("audit_*.md"))
    if not audit_files:
        TRIGGERS_FIRED.append("TIME_BASED: No audits exist yet — initial audit required")
        return True
    
    latest = max(f.stat().st_mtime for f in audit_files)
    days_since = (datetime.now().timestamp() - latest) / 86400
    
    if days_since > 30:
        TRIGGERS_FIRED.append(f"TIME_BASED: {int(days_since)} days since last audit (threshold: 30)")
        return True
    return False


def check_task_stagnation() -> bool:
    """Trigger 5: Tasks stuck in active.md for too long (CTO bottleneck)."""
    active = TASKS_DIR / "active.md"
    if not active.exists():
        return False
    
    content = active.read_text()
    # Count how many tasks are listed
    task_count = len(re.findall(r"## TASK-\d+", content))
    
    if task_count >= 5:
        TRIGGERS_FIRED.append(f"TASK_STAGNATION: {task_count} tasks in active.md (threshold: 5)")
        return True
    return False


def check_boundary_violations() -> bool:
    """Trigger 6: Any boundary violation detected in git log."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-20", "--grep=BOUNDARY"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        violations = [l for l in result.stdout.strip().split("\n") if l.strip()]
        if len(violations) >= 2:
            TRIGGERS_FIRED.append(f"BOUNDARY_VIOLATIONS: {len(violations)} in recent git log")
            return True
    except Exception:
        pass
    return False


def check_any_role_requests_audit() -> bool:
    """Trigger 7: Any role can force-request an audit via a marker file."""
    request_file = PROJECT_ROOT / ".claude" / "request_audit.md"
    if request_file.exists():
        content = request_file.read_text().strip()
        TRIGGERS_FIRED.append(f"ROLE_REQUEST: {content[:100]}")
        return True
    return False


def main():
    print("🔍 External Auditor trigger check...")
    print()
    
    triggered = False
    checks = [
        ("Phase gate completed", check_phase_gate),
        ("Recurring bug pattern (3+)", check_recurring_bugs),
        ("Review rubber-stamping (10 clean)", check_review_rubber_stamp),
        ("Time-based (30 days)", check_time_based),
        ("Task stagnation (5+ active)", check_task_stagnation),
        ("Boundary violations in git", check_boundary_violations),
        ("Manual audit request", check_any_role_requests_audit),
    ]
    
    for name, check_fn in checks:
        try:
            result = check_fn()
            status = "⚡ TRIGGERED" if result else "   quiet"
            print(f"  {status}  {name}")
            if result:
                triggered = True
        except Exception as e:
            print(f"  ❌ ERROR   {name}: {e}")
    
    print()
    
    if triggered:
        # Write trigger file
        trigger_data = {
            "triggered_at": datetime.now().isoformat(),
            "reasons": TRIGGERS_FIRED,
            "instruction": "Next session MUST use external-auditor role before any other work.",
        }
        TRIGGER_FILE.write_text(json.dumps(trigger_data, indent=2))
        
        print("=" * 60)
        print("⚡ EXTERNAL AUDIT TRIGGERED")
        print()
        for reason in TRIGGERS_FIRED:
            print(f"  → {reason}")
        print()
        print("  Next session MUST be: external-auditor")
        print("  Run: echo 'external-auditor' > .claude/current_role")
        print("=" * 60)
    else:
        # Clear trigger file if it exists
        if TRIGGER_FILE.exists():
            TRIGGER_FILE.unlink()
        print("✅ No audit needed. Continue normal work.")
    
    # Also check if a previous trigger is still unresolved
    if TRIGGER_FILE.exists() and not triggered:
        data = json.loads(TRIGGER_FILE.read_text())
        print()
        print(f"⚠️  Previous trigger from {data['triggered_at']} still unresolved!")
        print("  Run external-auditor session to clear it.")


if __name__ == "__main__":
    main()
