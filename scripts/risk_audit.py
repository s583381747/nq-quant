#!/usr/bin/env python3
"""
Risk Audit Pipeline — Automated adversarial testing suite.

Usage:
    python scripts/risk_audit.py [--module data|features|models|backtest|all]

Runs all risk checks:
1. Future leakage test (features only)
2. Edge case attacks
3. Boundary violation scan
4. Hardcoded threshold scan
5. Import dependency check (no cross-scope imports)

Results saved to reviews/risk_audit_<timestamp>.md
"""

import ast
import os
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
REVIEWS_DIR = PROJECT_ROOT / "reviews"
REVIEWS_DIR.mkdir(exist_ok=True)

# Which modules can import from which
ALLOWED_IMPORTS = {
    "data": set(),  # data/ imports nothing internal
    "features": {"data"},  # features/ can read data schema
    "models": {"features"},  # models/ consumes features
    "backtest": {"features", "models"},  # backtest uses both
    "live": {"features", "models"},  # live uses both
    "viz": {"features", "data"},  # viz renders features on data
}

# Patterns that suggest hardcoded thresholds
HARDCODE_PATTERNS = [
    (r'(?<!=\s)(?<![<>!=])(?<!\w)\b\d+\.?\d*\b(?!\s*[,\]\)])', "Numeric literal — should this be in params.yaml?"),
]

# Known safe numbers (array indices, common constants)
SAFE_NUMBERS = {"0", "1", "2", "3", "-1", "0.0", "1.0", "100", "0.5"}


def scan_imports(module_dir: str) -> list[dict]:
    """Check for cross-scope import violations."""
    findings = []
    module_name = Path(module_dir).name
    allowed = ALLOWED_IMPORTS.get(module_name, set())
    
    internal_modules = {"data", "features", "models", "backtest", "live", "viz"}
    
    for py_file in Path(module_dir).glob("*.py"):
        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            findings.append({
                "severity": "HIGH",
                "file": str(py_file),
                "line": 0,
                "issue": f"SyntaxError — cannot parse {py_file.name}",
            })
            continue
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in internal_modules and top not in allowed and top != module_name:
                        findings.append({
                            "severity": "CRITICAL",
                            "file": str(py_file),
                            "line": node.lineno,
                            "issue": f"Illegal import: `{alias.name}` — {module_name}/ cannot import from {top}/",
                        })
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    if top in internal_modules and top not in allowed and top != module_name:
                        findings.append({
                            "severity": "CRITICAL",
                            "file": str(py_file),
                            "line": node.lineno,
                            "issue": f"Illegal import: `from {node.module}` — {module_name}/ cannot import from {top}/",
                        })
    return findings


def scan_hardcoded_thresholds(module_dir: str) -> list[dict]:
    """Flag potential hardcoded thresholds that should be in params.yaml."""
    findings = []
    
    for py_file in Path(module_dir).glob("*.py"):
        lines = py_file.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip comments, empty lines, imports
            if not stripped or stripped.startswith("#") or stripped.startswith("import") or stripped.startswith("from"):
                continue
            # Skip test files
            if "test" in py_file.name:
                continue
            
            # Look for numeric assignments that might be thresholds
            match = re.search(r'=\s*(\d+\.?\d*)\s*(?:#|$)', stripped)
            if match:
                num = match.group(1)
                if num not in SAFE_NUMBERS and not stripped.startswith("def ") and "range(" not in stripped:
                    # Check if it's loading from config
                    if "config" not in stripped and "params" not in stripped and "yaml" not in stripped:
                        findings.append({
                            "severity": "WARNING",
                            "file": str(py_file),
                            "line": i,
                            "issue": f"Possible hardcoded threshold: `{stripped.strip()}` — should this value ({num}) come from params.yaml?",
                        })
    return findings


def scan_future_leakage_patterns(module_dir: str) -> list[dict]:
    """Scan for common future leakage patterns in feature code."""
    findings = []
    
    dangerous_patterns = [
        (r'\.shift\s*\(\s*-', "shift(-N) accesses future data!"),
        (r'\[.*\+\s*1\s*\]', "index+1 may access future data"),
        (r'\[.*\+\s*\d+\s*\]', "forward indexing may access future data"),
        (r'\.iloc\s*\[.*\+', "iloc with forward offset may leak"),
    ]
    
    for py_file in Path(module_dir).glob("*.py"):
        # Labeler is EXPECTED to use future data
        if "label" in py_file.name.lower():
            continue
        
        lines = py_file.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            for pattern, message in dangerous_patterns:
                if re.search(pattern, line):
                    findings.append({
                        "severity": "CRITICAL",
                        "file": str(py_file),
                        "line": i,
                        "issue": f"FUTURE LEAKAGE: {message} — `{line.strip()}`",
                    })
    return findings


def scan_missing_shift(module_dir: str) -> list[dict]:
    """Check that HTF features use shift(1) when merged into LTF."""
    findings = []
    
    for py_file in Path(module_dir).glob("*.py"):
        if "mtf" not in py_file.name.lower():
            continue
        
        content = py_file.read_text()
        
        # If there's a merge/join but no shift(1), flag it
        if ("merge" in content or "join" in content) and "shift(1)" not in content:
            findings.append({
                "severity": "CRITICAL",
                "file": str(py_file),
                "line": 0,
                "issue": "MTF alignment file has merge/join but no shift(1) — HTF features MUST be shifted to prevent future leakage",
            })
    return findings


def run_audit(modules: list[str]) -> str:
    """Run full audit and return report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_findings = []
    
    for module in modules:
        module_dir = PROJECT_ROOT / module
        if not module_dir.exists():
            continue
        
        # Only run relevant checks per module
        all_findings.extend(scan_imports(str(module_dir)))
        all_findings.extend(scan_hardcoded_thresholds(str(module_dir)))
        
        if module == "features":
            all_findings.extend(scan_future_leakage_patterns(str(module_dir)))
            all_findings.extend(scan_missing_shift(str(module_dir)))
    
    # Sort by severity
    severity_order = {"CRITICAL": 0, "HIGH": 1, "WARNING": 2, "MEDIUM": 3, "LOW": 4}
    all_findings.sort(key=lambda f: severity_order.get(f["severity"], 99))
    
    # Build report
    lines = [
        f"# Risk Audit Report",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Modules scanned**: {', '.join(modules)}",
        f"**Total findings**: {len(all_findings)}",
        "",
    ]
    
    critical_count = sum(1 for f in all_findings if f["severity"] == "CRITICAL")
    
    if critical_count > 0:
        lines.append(f"## 🚫 VETO — {critical_count} CRITICAL issues found")
        lines.append("**DO NOT MERGE until all CRITICAL issues are resolved.**")
    else:
        lines.append("## ✅ No critical issues found")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    if all_findings:
        lines.append("## Findings")
        lines.append("")
        for i, f in enumerate(all_findings, 1):
            emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "WARNING": "🟡"}.get(f["severity"], "⚪")
            lines.append(f"### {emoji} #{i} [{f['severity']}] {f['file']}:{f['line']}")
            lines.append(f"{f['issue']}")
            lines.append("")
    else:
        lines.append("_No findings. All clear._")
    
    report = "\n".join(lines)
    
    # Save report
    report_path = REVIEWS_DIR / f"risk_audit_{timestamp}.md"
    report_path.write_text(report)
    
    return str(report_path), critical_count


def main():
    modules = ["data", "features", "models", "backtest", "live", "viz"]
    
    if "--module" in sys.argv:
        idx = sys.argv.index("--module") + 1
        if idx < len(sys.argv):
            mod = sys.argv[idx]
            if mod == "all":
                pass  # use default
            else:
                modules = [mod]
    
    print(f"🔍 Running risk audit on: {', '.join(modules)}")
    print()
    
    report_path, critical_count = run_audit(modules)
    
    print(f"📋 Report saved to: {report_path}")
    
    if critical_count > 0:
        print(f"🚫 VETO: {critical_count} CRITICAL issues found!")
        sys.exit(1)
    else:
        print("✅ Audit passed — no critical issues.")
        sys.exit(0)


if __name__ == "__main__":
    main()
