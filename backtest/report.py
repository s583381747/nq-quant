"""
backtest/report.py -- Generate backtest reports (console + HTML).

Metrics:
  - Total trades, win rate, profit factor, avg R
  - Max drawdown (equity curve)
  - Monthly breakdown
  - Prop firm safety check
  - Grade / signal type / direction breakdown

References: CLAUDE.md sections 10, 11, 12
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "params.yaml"
_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _load_params(config_path: Path = _CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_report(
    trades: pd.DataFrame,
    params: dict | None = None,
    save_html: bool = True,
) -> str:
    """Generate backtest report (console output + optional HTML).

    Parameters
    ----------
    trades : pd.DataFrame
        Trade log from engine.run_backtest.
    params : dict
        From params.yaml.
    save_html : bool
        If True, save HTML report to backtest/results/.

    Returns
    -------
    str
        Console-formatted report text.
    """
    if params is None:
        params = _load_params()

    if trades.empty:
        return "NO TRADES -- backtest produced 0 trades."

    # ---- Basic stats ----
    n_trades = len(trades)
    winners = trades[trades["pnl_dollars"] > 0]
    losers = trades[trades["pnl_dollars"] <= 0]
    n_win = len(winners)
    n_loss = len(losers)
    win_rate = n_win / n_trades if n_trades > 0 else 0.0

    total_pnl = trades["pnl_dollars"].sum()
    avg_pnl = trades["pnl_dollars"].mean()
    avg_r = trades["r_multiple"].mean()
    median_r = trades["r_multiple"].median()

    gross_profit = winners["pnl_dollars"].sum() if n_win > 0 else 0.0
    gross_loss = abs(losers["pnl_dollars"].sum()) if n_loss > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_winner = winners["pnl_dollars"].mean() if n_win > 0 else 0.0
    avg_loser = losers["pnl_dollars"].mean() if n_loss > 0 else 0.0
    largest_win = winners["pnl_dollars"].max() if n_win > 0 else 0.0
    largest_loss = losers["pnl_dollars"].min() if n_loss > 0 else 0.0

    # ---- Equity curve + drawdown ----
    equity = trades["pnl_dollars"].cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = drawdown.min()
    max_dd_pct = max_dd / running_max.replace(0, np.nan).max() if running_max.max() > 0 else 0.0

    # ---- Monthly breakdown ----
    trades_copy = trades.copy()
    if trades_copy["entry_time"].dt.tz is not None:
        trades_copy["month"] = trades_copy["entry_time"].dt.tz_convert("US/Eastern").dt.to_period("M")
    else:
        trades_copy["month"] = trades_copy["entry_time"].dt.to_period("M")

    monthly = trades_copy.groupby("month").agg(
        n_trades=("pnl_dollars", "count"),
        pnl=("pnl_dollars", "sum"),
        avg_r=("r_multiple", "mean"),
        win_rate=("pnl_dollars", lambda x: (x > 0).mean()),
    )

    # ---- Exit reason breakdown ----
    exit_reasons = trades["exit_reason"].value_counts()

    # ---- Signal type breakdown ----
    by_type = trades.groupby("signal_type").agg(
        n=("pnl_dollars", "count"),
        win_rate=("pnl_dollars", lambda x: (x > 0).mean()),
        avg_r=("r_multiple", "mean"),
        total_pnl=("pnl_dollars", "sum"),
    )

    # ---- Direction breakdown ----
    by_dir = trades.groupby("direction").agg(
        n=("pnl_dollars", "count"),
        win_rate=("pnl_dollars", lambda x: (x > 0).mean()),
        avg_r=("r_multiple", "mean"),
        total_pnl=("pnl_dollars", "sum"),
    )

    # ---- Grade breakdown ----
    by_grade = trades.groupby("grade").agg(
        n=("pnl_dollars", "count"),
        win_rate=("pnl_dollars", lambda x: (x > 0).mean()),
        avg_r=("r_multiple", "mean"),
        total_pnl=("pnl_dollars", "sum"),
    )

    # ---- Prop firm safety check ----
    point_value = params["position"]["instrument_point_value"]
    safety_buffer = params["backtest"]["prop_firm_safety_buffer"]
    # Typical prop firm: $2200 daily loss limit for 50K account
    # Use max drawdown in dollars vs a reference limit
    firm_dd_limit = 2200.0  # conservative estimate
    safety_pct = 1.0 - abs(max_dd) / firm_dd_limit if firm_dd_limit > 0 else 1.0
    safety_ok = safety_pct >= safety_buffer

    # ---- Consecutive loss analysis ----
    pnl_series = trades["pnl_dollars"].values
    max_consec_loss = 0
    current_consec = 0
    for p in pnl_series:
        if p <= 0:
            current_consec += 1
            max_consec_loss = max(max_consec_loss, current_consec)
        else:
            current_consec = 0

    max_consec_win = 0
    current_consec = 0
    for p in pnl_series:
        if p > 0:
            current_consec += 1
            max_consec_win = max(max_consec_win, current_consec)
        else:
            current_consec = 0

    # ---- Build console report ----
    lines = []
    lines.append("=" * 70)
    lines.append("BACKTEST REPORT -- NQ Quant V2")
    lines.append("=" * 70)
    lines.append("")

    if len(trades) > 0:
        lines.append(f"Period: {trades['entry_time'].min()} to {trades['exit_time'].max()}")
    lines.append(f"Total trades:       {n_trades:>8}")
    lines.append(f"Winners:            {n_win:>8}  ({100*win_rate:.1f}%)")
    lines.append(f"Losers:             {n_loss:>8}  ({100*(1-win_rate):.1f}%)")
    lines.append("")
    lines.append(f"Total PnL:          ${total_pnl:>12,.2f}")
    lines.append(f"Avg PnL/trade:      ${avg_pnl:>12,.2f}")
    lines.append(f"Avg R:              {avg_r:>12.3f}")
    lines.append(f"Median R:           {median_r:>12.3f}")
    lines.append(f"Profit Factor:      {profit_factor:>12.2f}")
    lines.append("")
    lines.append(f"Avg Winner:         ${avg_winner:>12,.2f}")
    lines.append(f"Avg Loser:          ${avg_loser:>12,.2f}")
    lines.append(f"Largest Win:        ${largest_win:>12,.2f}")
    lines.append(f"Largest Loss:       ${largest_loss:>12,.2f}")
    lines.append("")
    lines.append(f"Max Drawdown:       ${max_dd:>12,.2f}")
    lines.append(f"Max Consec Wins:    {max_consec_win:>12}")
    lines.append(f"Max Consec Losses:  {max_consec_loss:>12}")

    # Prop firm check
    lines.append("")
    lines.append("--- Prop Firm Safety ---")
    lines.append(f"Max DD:             ${abs(max_dd):>12,.2f}")
    lines.append(f"Firm DD Limit:      ${firm_dd_limit:>12,.2f}")
    lines.append(f"Safety Buffer:      {100*safety_pct:>11.1f}%  (need >{100*safety_buffer:.0f}%)")
    lines.append(f"Status:             {'PASS' if safety_ok else 'FAIL':>12}")

    # Monthly
    lines.append("")
    lines.append("--- Monthly Breakdown ---")
    lines.append(f"{'Month':>10} {'Trades':>7} {'PnL':>12} {'Avg R':>8} {'Win%':>7}")
    lines.append("-" * 50)
    for period, row in monthly.iterrows():
        lines.append(
            f"{str(period):>10} {int(row['n_trades']):>7} "
            f"${row['pnl']:>11,.2f} {row['avg_r']:>8.3f} {100*row['win_rate']:>6.1f}%"
        )

    # Exit reasons
    lines.append("")
    lines.append("--- Exit Reasons ---")
    for reason, count in exit_reasons.items():
        pnl_for = trades[trades["exit_reason"] == reason]["pnl_dollars"].sum()
        lines.append(f"  {reason:<15} {count:>5}  ${pnl_for:>12,.2f}")

    # Signal type
    lines.append("")
    lines.append("--- By Signal Type ---")
    lines.append(f"{'Type':>10} {'N':>6} {'Win%':>7} {'Avg R':>8} {'PnL':>12}")
    for stype, row in by_type.iterrows():
        lines.append(
            f"{stype:>10} {int(row['n']):>6} {100*row['win_rate']:>6.1f}% "
            f"{row['avg_r']:>8.3f} ${row['total_pnl']:>11,.2f}"
        )

    # Direction
    lines.append("")
    lines.append("--- By Direction ---")
    lines.append(f"{'Dir':>10} {'N':>6} {'Win%':>7} {'Avg R':>8} {'PnL':>12}")
    for d, row in by_dir.iterrows():
        d_str = "LONG" if d == 1 else "SHORT"
        lines.append(
            f"{d_str:>10} {int(row['n']):>6} {100*row['win_rate']:>6.1f}% "
            f"{row['avg_r']:>8.3f} ${row['total_pnl']:>11,.2f}"
        )

    # Grade
    lines.append("")
    lines.append("--- By Grade ---")
    lines.append(f"{'Grade':>10} {'N':>6} {'Win%':>7} {'Avg R':>8} {'PnL':>12}")
    for g, row in by_grade.iterrows():
        lines.append(
            f"{g:>10} {int(row['n']):>6} {100*row['win_rate']:>6.1f}% "
            f"{row['avg_r']:>8.3f} ${row['total_pnl']:>11,.2f}"
        )

    # Equity curve stats
    lines.append("")
    lines.append("--- Equity Curve ---")
    if len(equity) > 0:
        lines.append(f"Final equity:  ${equity.iloc[-1]:>12,.2f}")
        lines.append(f"Peak equity:   ${equity.max():>12,.2f}")
        lines.append(f"Max drawdown:  ${max_dd:>12,.2f}")

    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    # ---- Save HTML ----
    if save_html:
        html = _build_html_report(
            report_text, trades, equity, drawdown, monthly,
            by_type, by_dir, by_grade, exit_reasons, params,
        )
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = _RESULTS_DIR / f"backtest_v2_{timestamp}.html"
        html_path.write_text(html, encoding="utf-8")
        logger.info("HTML report saved to %s", html_path)
        lines.append(f"\nHTML report: {html_path}")
        report_text = "\n".join(lines)

    return report_text


def _build_html_report(
    text_report: str,
    trades: pd.DataFrame,
    equity: pd.Series,
    drawdown: pd.Series,
    monthly: pd.DataFrame,
    by_type: pd.DataFrame,
    by_dir: pd.DataFrame,
    by_grade: pd.DataFrame,
    exit_reasons: pd.Series,
    params: dict,
) -> str:
    """Build an HTML report with embedded tables and SVG equity curve."""

    # Simple SVG equity curve
    eq_vals = equity.values
    n = len(eq_vals)
    if n > 1:
        width = 800
        height = 300
        x_scale = width / (n - 1)
        y_min = min(eq_vals.min(), 0)
        y_max = max(eq_vals.max(), 100)
        y_range = y_max - y_min if y_max != y_min else 1
        y_scale = (height - 40) / y_range

        points = []
        for i, v in enumerate(eq_vals):
            x = i * x_scale
            y = height - 20 - (v - y_min) * y_scale
            points.append(f"{x:.1f},{y:.1f}")
        polyline = " ".join(points)

        # Zero line
        zero_y = height - 20 - (0 - y_min) * y_scale

        svg = f"""
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
          <rect width="{width}" height="{height}" fill="#1a1a2e"/>
          <line x1="0" y1="{zero_y:.1f}" x2="{width}" y2="{zero_y:.1f}"
                stroke="#555" stroke-dasharray="4"/>
          <polyline points="{polyline}"
                    fill="none" stroke="#00d4aa" stroke-width="2"/>
          <text x="10" y="20" fill="#ccc" font-size="14">Equity Curve ($)</text>
          <text x="10" y="{height-5}" fill="#888" font-size="11">
            Trades: {n} | Final: ${eq_vals[-1]:,.0f} | Max DD: ${drawdown.min():,.0f}
          </text>
        </svg>
        """
    else:
        svg = "<p>Not enough trades for equity curve.</p>"

    # Monthly table rows
    monthly_rows = ""
    for period, row in monthly.iterrows():
        color = "#00d4aa" if row["pnl"] > 0 else "#ff4757"
        monthly_rows += f"""
        <tr>
          <td>{period}</td>
          <td>{int(row['n_trades'])}</td>
          <td style="color:{color}">${row['pnl']:,.2f}</td>
          <td>{row['avg_r']:.3f}</td>
          <td>{100*row['win_rate']:.1f}%</td>
        </tr>"""

    # Trade log rows (last 50)
    trade_rows = ""
    display_trades = trades.tail(50)
    for _, t in display_trades.iterrows():
        color = "#00d4aa" if t["pnl_dollars"] > 0 else "#ff4757"
        dir_str = "LONG" if t["direction"] == 1 else "SHORT"
        trade_rows += f"""
        <tr>
          <td>{t['entry_time']}</td>
          <td>{dir_str}</td>
          <td>{t['signal_type']}</td>
          <td>{t['grade']}</td>
          <td>{t['entry_price']:.2f}</td>
          <td>{t['stop_price']:.2f}</td>
          <td>{t['tp1_price']:.2f}</td>
          <td style="color:{color}">${t['pnl_dollars']:,.2f}</td>
          <td>{t['r_multiple']:.2f}R</td>
          <td>{t['exit_reason']}</td>
          <td>{t['model_prob']:.2f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NQ Quant V2 Backtest Report</title>
<style>
  body {{ background: #0f0f23; color: #ccc; font-family: 'Courier New', monospace; padding: 20px; }}
  h1, h2, h3 {{ color: #00d4aa; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #333; padding: 6px 10px; text-align: right; }}
  th {{ background: #1a1a2e; color: #00d4aa; }}
  tr:hover {{ background: #1a1a2e; }}
  .section {{ margin: 20px 0; padding: 15px; background: #16162a; border-radius: 8px; }}
  pre {{ background: #1a1a2e; padding: 15px; border-radius: 5px; overflow-x: auto; }}
</style>
</head>
<body>
<h1>NQ Quant V2 -- Backtest Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="section">
<h2>Equity Curve</h2>
{svg}
</div>

<div class="section">
<h2>Summary</h2>
<pre>{text_report}</pre>
</div>

<div class="section">
<h2>Monthly Breakdown</h2>
<table>
<tr><th>Month</th><th>Trades</th><th>PnL</th><th>Avg R</th><th>Win Rate</th></tr>
{monthly_rows}
</table>
</div>

<div class="section">
<h2>Trade Log (last 50)</h2>
<table>
<tr>
  <th>Entry Time</th><th>Dir</th><th>Type</th><th>Grade</th>
  <th>Entry</th><th>Stop</th><th>TP1</th>
  <th>PnL</th><th>R</th><th>Exit</th><th>Prob</th>
</tr>
{trade_rows}
</table>
</div>

</body>
</html>"""

    return html
