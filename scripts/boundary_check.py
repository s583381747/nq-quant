#!/usr/bin/env python3
"""
Boundary Validator — Enforces CODEOWNERS rules.

Usage:
    python scripts/boundary_check.py <role_name>
    
Checks git diff (staged files) against CODEOWNERS.
If any file is outside the role's scope, it BLOCKS the commit.

Run this BEFORE code_review.py — no point reviewing code that violates boundaries.
"""

import subprocess
import sys
from pathlib import Path

CODEOWNERS_PATH = Path(__file__).parent.parent / "CODEOWNERS"

# CTO is NEVER allowed to create or modify these file types
CTO_BANNED_EXTENSIONS = {".py", ".pyx", ".ipynb", ".js", ".ts", ".sh"}

# Role-specific hard bans (actions they must never perform)
ROLE_HARD_BANS = {
    "cto": {
        "ban_extensions": CTO_BANNED_EXTENSIONS,
        "ban_message": "CTO must NEVER write code files. Delegate to the appropriate engineer.",
    },
    "data-engineer": {
        "ban_dirs": {"features/", "models/", "backtest/", "live/"},
        "ban_message": "Data Engineer must only work in data/ and tests/test_data.py.",
    },
    "quant-researcher": {
        "ban_dirs": {"data/", "models/", "backtest/", "live/"},
        "ban_message": "Quant Researcher must only work in features/, viz/, and related tests.",
    },
    "experiment-runner": {
        "ban_dirs": {"data/", "features/", "backtest/", "live/"},
        "ban_message": "Experiment Runner works in .lab/, models/, notebooks/. Never production code.",
    },
    "backtest-engineer": {
        "ban_dirs": {"data/", "features/", "models/", "live/"},
        "ban_message": "Backtest Engineer must only work in backtest/ and related tests.",
    },
    "risk-officer": {
        "ban_dirs": {"data/", "features/", "models/", "backtest/", "live/"},
        "allow_only": {"tests/", "scripts/risk_audit.py", "scripts/hellaudit.py"},
        "ban_message": "Risk Officer can only write tests and audit scripts. Never production code.",
    },
    "execution-engineer": {
        "ban_dirs": {"data/", "features/", "models/", "backtest/"},
        "ban_message": "Execution Engineer must only work in live/ and related tests.",
    },
    "external-auditor": {
        "ban_dirs": {"data/", "features/", "models/", "backtest/", "live/"},
        "allow_only": {".claude/profiles/", "audits/", "scripts/code_review.py",
                       "scripts/boundary_check.py", "scripts/hellaudit.py",
                       "CODEOWNERS", "TEAM_ARCHITECTURE.md"},
        "ban_message": "External Auditor modifies only architecture, profiles, review scripts, and audits/.",
    },
}


def parse_codeowners() -> dict[str, str]:
    """Parse CODEOWNERS file into {pattern: role} mapping."""
    ownership = {}
    with open(CODEOWNERS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                ownership[parts[0]] = parts[1]
    return ownership


def get_changed_files() -> list[str]:
    """Get list of staged files from git diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, check=True
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        # If nothing staged, check unstaged changes
        if not files:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                capture_output=True, text=True, check=True
            )
            files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        # If still nothing, check untracked
        if not files:
            result = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                capture_output=True, text=True, check=True
            )
            files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        return files
    except subprocess.CalledProcessError:
        return []


def check_extension_ban(role: str, files: list[str]) -> list[str]:
    """Check if role is trying to create/modify banned file types."""
    violations = []
    bans = ROLE_HARD_BANS.get(role, {})
    banned_exts = bans.get("ban_extensions", set())
    
    for f in files:
        ext = Path(f).suffix
        if ext in banned_exts:
            violations.append(f"  ❌ BLOCKED: {f} — {role} cannot create/modify {ext} files")
    return violations


def check_directory_ban(role: str, files: list[str]) -> list[str]:
    """Check if role is touching directories outside their scope."""
    violations = []
    bans = ROLE_HARD_BANS.get(role, {})
    banned_dirs = bans.get("ban_dirs", set())
    allow_only = bans.get("allow_only", None)
    
    for f in files:
        # Check explicit directory bans
        for banned in banned_dirs:
            if f.startswith(banned):
                violations.append(f"  ❌ BLOCKED: {f} — {role} cannot touch {banned}")
                break
        
        # Check allow-only restriction (e.g., Risk Officer)
        if allow_only:
            allowed = any(f.startswith(a) for a in allow_only)
            if not allowed and not any(f.startswith(banned) for banned in banned_dirs):
                violations.append(f"  ❌ BLOCKED: {f} — {role} can only modify: {', '.join(allow_only)}")
    
    return violations


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/boundary_check.py <role_name>")
        print("Roles: cto, data-engineer, quant-researcher, ml-engineer, backtest-engineer, risk-officer, execution-engineer")
        sys.exit(1)
    
    role = sys.argv[1].lower()
    
    if role not in ROLE_HARD_BANS:
        print(f"❌ Unknown role: {role}")
        sys.exit(1)
    
    # Get files to check (from args or git)
    if len(sys.argv) > 2:
        files = sys.argv[2:]
    else:
        files = get_changed_files()
    
    if not files:
        print(f"✅ No files to check for role: {role}")
        sys.exit(0)
    
    print(f"🔍 Boundary check for role: {role}")
    print(f"   Files to check: {len(files)}")
    print()
    
    violations = []
    violations.extend(check_extension_ban(role, files))
    violations.extend(check_directory_ban(role, files))
    
    if violations:
        print("🚫 BOUNDARY VIOLATIONS DETECTED:")
        print()
        for v in violations:
            print(v)
        print()
        print(f"💡 {ROLE_HARD_BANS[role]['ban_message']}")
        print()
        print("Action: Revert unauthorized changes and delegate to the correct team member.")
        sys.exit(1)
    else:
        print(f"✅ All {len(files)} files are within {role}'s scope.")
        for f in files:
            print(f"  ✓ {f}")
        sys.exit(0)


if __name__ == "__main__":
    main()
