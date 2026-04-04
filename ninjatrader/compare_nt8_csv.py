"""
Phase 5: Auto-compare NT8 CSV export vs Python 534 trades.
Reads the CSV from NT8 Strategy Analyzer and compares trade-by-trade.

Usage: python ninjatrader/compare_nt8_csv.py <path_to_nt8_csv>
       python ninjatrader/compare_nt8_csv.py  (auto-finds latest on Desktop)
"""
import sys, glob, os
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

# Load Python reference trades
py_path = PROJECT / "ninjatrader" / "python_trades_545.csv"
if not py_path.exists():
    print(f"ERROR: Python reference trades not found at {py_path}")
    print("Run: python ninjatrader/validate_nt_logic.py first")
    sys.exit(1)

py = pd.read_csv(py_path)
print(f"Python reference: {len(py)} trades, R={py['r'].sum():.2f}")

# Find NT8 CSV
if len(sys.argv) > 1:
    nt_path = sys.argv[1]
else:
    # Auto-find latest nt8_trades_*.csv on Desktop
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    csvs = sorted(glob.glob(os.path.join(desktop, "nt8_trades_*.csv")))
    if not csvs:
        print(f"ERROR: No nt8_trades_*.csv found on Desktop ({desktop})")
        print("Run NT8 Strategy Analyzer first — CSV auto-exports to Desktop")
        sys.exit(1)
    nt_path = csvs[-1]
    print(f"Auto-found: {nt_path}")

nt = pd.read_csv(nt_path)
print(f"NT8 results:     {len(nt)} trades, R={nt['r_multiple'].sum():.2f}")

# Summary comparison
print(f"\n{'='*70}")
print("COMPARISON SUMMARY")
print(f"{'='*70}")
print(f"Trade count:  Python={len(py)}, NT8={len(nt)}, diff={len(nt)-len(py)}")
print(f"Total R:      Python={py['r'].sum():.2f}, NT8={nt['r_multiple'].sum():.2f}")
print(f"Win rate:     Python={100*(py['r']>0).mean():.1f}%, NT8={100*(nt['r_multiple']>0).mean():.1f}%")

# By type
print(f"\nPython by type:")
print(py.groupby('type')['r'].agg(['count','sum']).to_string())
print(f"\nNT8 by type:")
if 'signal_type' in nt.columns:
    print(nt.groupby('signal_type')['r_multiple'].agg(['count','sum']).to_string())

# By exit reason
print(f"\nPython exits:")
print(py.groupby('reason')['r'].agg(['count','sum']).to_string())
print(f"\nNT8 exits:")
if 'exit_reason' in nt.columns:
    print(nt.groupby('exit_reason')['r_multiple'].agg(['count','sum']).to_string())

# Per-trade comparison (if same count)
if len(py) == len(nt):
    r_diff = nt['r_multiple'].values - py['r'].values
    print(f"\nPer-trade R diff: max={r_diff.max():.4f}, min={r_diff.min():.4f}, mean={r_diff.mean():.6f}")
    mismatches = np.abs(r_diff) > 0.1
    print(f"Trades with R diff > 0.1: {mismatches.sum()}")
    if mismatches.sum() > 0:
        print("\nTop mismatches:")
        idx = np.argsort(-np.abs(r_diff))[:10]
        for j in idx:
            if np.abs(r_diff[j]) > 0.05:
                print(f"  Trade {j}: Py R={py.iloc[j]['r']:.4f}, NT R={nt.iloc[j]['r_multiple']:.4f}, "
                      f"diff={r_diff[j]:.4f}, type={py.iloc[j]['type']}")
else:
    print(f"\n*** TRADE COUNT MISMATCH: NT8 {len(nt)} vs Python {len(py)} ***")
    print("Direction breakdown:")
    print(f"  Python: long={sum(py['dir']==1)}, short={sum(py['dir']==-1)}")
    if 'direction' in nt.columns:
        print(f"  NT8:    long={sum(nt['direction']==1)}, short={sum(nt['direction']==-1)}")

print(f"{'='*70}")
