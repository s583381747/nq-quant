"""Compare NT trade management vs Python engine using SAME signals."""
import sys, numpy as np, pandas as pd, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

nq = pd.read_parquet("data/NQ_5m_10yr.parquet")
sig3 = pd.read_parquet("data/cache_signals_10yr_v3.parquet")
bias = pd.read_parquet("data/cache_bias_10yr_v2.parquet")
regime = pd.read_parquet("data/cache_regime_10yr_v2.parquet")
with open("config/params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# Trend-only
ss = sig3.copy()
mm = ss["signal"].astype(bool) & (ss["signal_type"] == "mss")
ss.loc[ss.index[mm], ["signal", "signal_dir"]] = [False, 0]

# Python engine
class Dummy:
    def predict(self, d): return np.ones(d.num_row(), dtype=np.float32)
dummy_X = pd.DataFrame({"d": np.zeros(len(nq))}, index=nq.index)
from backtest.engine import run_backtest
py = run_backtest(nq, ss, bias, regime["regime"], Dummy(), dummy_X, params, threshold=0.0)

print(f"=== PYTHON ENGINE (Trend-only) ===")
print(f"Trades: {len(py)}, R={py['r_multiple'].sum():.1f}, WR={100*(py['r_multiple']>0).mean():.1f}%")
print(py.groupby("exit_reason")["r_multiple"].agg(["count", "sum"]).to_string())
print(f"\nAvg TP dist (longs): {(py[py['direction']==1]['tp1_price'] - py[py['direction']==1]['entry_price']).mean():.1f} pts")
print(f"Avg stop dist: {(py['entry_price'] - py['stop_price']).abs().mean():.1f} pts")
print(f"Direction: {py['direction'].value_counts().to_dict()}")
print(f"Grade: {py['grade'].value_counts().to_dict()}")
