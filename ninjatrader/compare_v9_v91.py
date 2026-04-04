"""
Compare v9.0 and v9.1 trade outputs to measure the impact of signal detection fixes.

Usage:
    1. Run v9.0 in NT8, export to C:/temp/v9_trades_v90.csv
    2. Run v9.1 in NT8, export to C:/temp/v9_trades_v91.csv
    3. python ninjatrader/compare_v9_v91.py

Or use the latest v9 output directly:
    python ninjatrader/compare_v9_v91.py --v90 C:/temp/v9_trades_latest.csv --v91 <new_file>
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['entry_dt'] = pd.to_datetime(df['entry_time'])
    df['signal_bar_et'] = df['entry_dt'] - pd.Timedelta(minutes=10)
    return df

def compare(v90_path: str, v91_path: str, baseline_path: str):
    v90 = load_trades(v90_path)
    v91 = load_trades(v91_path)
    baseline = load_trades(baseline_path)

    # Load Python export for signal-level comparison
    export = pd.read_csv(PROJECT / 'ninjatrader' / 'python_signals_v4_export.csv')
    export['bar_dt_et'] = pd.to_datetime(export['bar_time_et'])

    print("=" * 70)
    print("v9.0 vs v9.1 COMPARISON")
    print("=" * 70)

    print(f"\nTrade counts:")
    print(f"  v9.0:     {len(v90)} trades")
    print(f"  v9.1:     {len(v91)} trades")
    print(f"  Baseline: {len(baseline)} trades")
    print(f"  v9.0 gap: {len(baseline) - len(v90):+d}")
    print(f"  v9.1 gap: {len(baseline) - len(v91):+d}")

    for label, df in [("v9.0", v90), ("v9.1", v91), ("Baseline", baseline)]:
        print(f"\n  {label} breakdown:")
        print(f"    Trend: {(df['signal_type']=='trend').sum()}, MSS: {(df['signal_type']=='mss').sum()}")
        print(f"    Long: {(df['direction']==1).sum()}, Short: {(df['direction']==-1).sum()}")

    # Match each to export
    for label, df in [("v9.0", v90), ("v9.1", v91)]:
        matched = 0
        for _, row in df.iterrows():
            sig_t = row['signal_bar_et']
            m = export[(export['bar_dt_et'] - sig_t).abs() < pd.Timedelta(minutes=2)]
            if len(m) > 0:
                matched += 1
        print(f"\n  {label} matched to Python export: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")

    # Exact time match with baseline
    for label, df in [("v9.0", v90), ("v9.1", v91)]:
        exact = len(set(df['entry_time'].values) & set(baseline['entry_time'].values))
        only = len(set(df['entry_time'].values) - set(baseline['entry_time'].values))
        missing = len(set(baseline['entry_time'].values) - set(df['entry_time'].values))
        print(f"\n  {label} vs baseline: exact={exact}, phantom={only}, missing={missing}")

    # R-multiple comparison
    for label, df in [("v9.0", v90), ("v9.1", v91)]:
        total_r = df['r_multiple'].sum()
        print(f"\n  {label} total R: {total_r:.1f}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--v90', default='C:/temp/v9_trades_latest.csv')
    parser.add_argument('--v91', default='C:/temp/v9_trades_v91.csv')
    parser.add_argument('--baseline', default='C:/temp/csv_baseline.csv')
    args = parser.parse_args()

    compare(args.v90, args.v91, args.baseline)
