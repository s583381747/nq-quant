"""
viz/chart.py — Interactive HTML feature-overlay charts for visual verification.

Generates plotly-based candlestick charts with overlays for:
  - FVG rectangles (bullish = green, bearish = red)
  - Displacement candle markers (triangles)
  - Swing high/low markers (diamonds)
  - Session levels (asia, london, overnight H/L, NY open)
  - Fluency score (subplot)
  - Volume (subplot)

All display times are US/Eastern (ET).  Internal data is UTC.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "params.yaml"
_DATA_PATH = _PROJECT_ROOT / "data" / "NQ_1min.parquet"
_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_params(path: Path = _CONFIG_PATH) -> dict:
    """Load params.yaml."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def _compute_all_features(df_slice: pd.DataFrame, params: dict) -> dict[str, Any]:
    """Compute all features needed for chart overlays.

    Parameters
    ----------
    df_slice : pd.DataFrame
        1-min OHLCV slice (UTC index).  Must include enough pre-history
        for indicators (ATR warmup, etc.).
    params : dict
        Parsed params.yaml.

    Returns
    -------
    dict with keys: fvg_df, fvg_records, displacement, fluency, swing, sessions
    """
    import sys
    sys.path.insert(0, str(_PROJECT_ROOT))

    from features.fvg import detect_fvg, track_fvg_state
    from features.displacement import detect_displacement, compute_fluency
    from features.swing import compute_swing_levels
    from features.sessions import compute_session_levels, label_sessions

    logger.info("Computing FVGs...")
    fvg_df = detect_fvg(df_slice)
    fvg_records = track_fvg_state(df_slice, fvg_df)

    logger.info("Computing displacement...")
    displacement = detect_displacement(df_slice, params)

    logger.info("Computing fluency...")
    fluency = compute_fluency(df_slice, params)

    logger.info("Computing swing levels...")
    swing_params = {
        "left_bars": params["swing"]["left_bars"],
        "right_bars": params["swing"]["right_bars"],
    }
    swing = compute_swing_levels(df_slice, swing_params)

    logger.info("Computing session levels...")
    sessions = compute_session_levels(df_slice, params)

    logger.info("Labeling sessions...")
    session_labels = label_sessions(df_slice, params)

    return {
        "fvg_df": fvg_df,
        "fvg_records": fvg_records,
        "displacement": displacement,
        "fluency": fluency,
        "swing": swing,
        "sessions": sessions,
        "session_labels": session_labels,
    }


# ---------------------------------------------------------------------------
# Chart building
# ---------------------------------------------------------------------------

def _to_et(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert UTC index to US/Eastern for display."""
    return idx.tz_convert("US/Eastern")


def _build_figure(
    df_view: pd.DataFrame,
    features: dict[str, Any],
    title: str,
) -> go.Figure:
    """Build the full plotly Figure with all overlays.

    Parameters
    ----------
    df_view : pd.DataFrame
        The visible slice of 1-min data (UTC index).
    features : dict
        Pre-computed features (keys: fvg_records, displacement, fluency, swing, sessions, session_labels).
    title : str
        Chart title.

    Returns
    -------
    go.Figure
    """
    et_index = _to_et(df_view.index)
    # Use timezone-naive strings for plotly x-axis (plotly handles tz poorly)
    x_vals = et_index.tz_localize(None)

    # --- Create subplots: main (row 1), fluency (row 2), volume (row 3) ---
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.65, 0.15, 0.20],
        subplot_titles=[title, "Fluency Score", "Volume"],
    )

    # =====================================================================
    # ROW 1: Candlestick + overlays
    # =====================================================================

    # 1a. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=x_vals,
            open=df_view["open"],
            high=df_view["high"],
            low=df_view["low"],
            close=df_view["close"],
            name="NQ",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
        ),
        row=1,
        col=1,
    )

    # 1b. FVG rectangles
    _add_fvg_rectangles(fig, df_view, features, x_vals)

    # 1c. Displacement markers
    _add_displacement_markers(fig, df_view, features, x_vals)

    # 1d. Swing points
    _add_swing_markers(fig, df_view, features, x_vals)

    # 1e. Session levels
    _add_session_levels(fig, df_view, features, x_vals)

    # =====================================================================
    # ROW 2: Fluency score
    # =====================================================================
    fluency = features.get("fluency")
    if fluency is not None:
        flu_view = fluency.reindex(df_view.index)
        params = _load_params()
        threshold = params["fluency"]["threshold"]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=flu_view,
                mode="lines",
                name="Fluency",
                line=dict(color="#7e57c2", width=1.2),
            ),
            row=2,
            col=1,
        )
        # Threshold line
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"threshold={threshold}",
            row=2,
            col=1,
        )

    # =====================================================================
    # ROW 3: Volume bars
    # =====================================================================
    colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df_view["close"], df_view["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=df_view["volume"],
            marker_color=colors,
            name="Volume",
            opacity=0.7,
        ),
        row=3,
        col=1,
    )

    # =====================================================================
    # Layout
    # =====================================================================
    fig.update_layout(
        height=900,
        width=1600,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        margin=dict(l=60, r=30, t=80, b=30),
    )

    # Disable rangeslider on all x-axes
    fig.update_xaxes(rangeslider_visible=False)

    # Y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_yaxes(title_text="Vol", row=3, col=1)

    return fig


def _add_fvg_rectangles(
    fig: go.Figure,
    df_view: pd.DataFrame,
    features: dict,
    x_vals: pd.DatetimeIndex,
) -> None:
    """Add FVG rectangles as shaded regions on the candlestick chart."""
    fvg_records = features.get("fvg_records", [])
    if not fvg_records:
        return

    view_start = df_view.index.min()
    view_end = df_view.index.max()

    n_bull = 0
    n_bear = 0

    for rec in fvg_records:
        fvg_time = rec["time"]
        # Skip FVGs that ended before the view or started after it
        if fvg_time > view_end:
            continue
        invalidated_at = rec.get("invalidated_at")
        if invalidated_at is not None and invalidated_at < view_start:
            continue

        top = rec["top"]
        bottom = rec["bottom"]
        direction = rec["direction"]
        status = rec["status"]
        is_ifvg = rec.get("is_ifvg", False)

        # Determine FVG display start/end within the view
        display_start = max(fvg_time, view_start)
        display_end = view_end
        if invalidated_at is not None and invalidated_at <= view_end:
            display_end = invalidated_at

        # Convert to ET naive for plotly
        fvg_start_et = pd.Timestamp(display_start).tz_convert("US/Eastern").tz_localize(None)
        fvg_end_et = pd.Timestamp(display_end).tz_convert("US/Eastern").tz_localize(None)

        # Color and opacity based on direction and status
        if is_ifvg:
            color = "rgba(255, 193, 7, 0.15)"  # amber for IFVG
            border_color = "rgba(255, 193, 7, 0.5)"
        elif direction == "bull":
            if status == "invalidated":
                color = "rgba(38, 166, 154, 0.08)"
                border_color = "rgba(38, 166, 154, 0.2)"
            elif status == "tested_rejected":
                color = "rgba(38, 166, 154, 0.2)"
                border_color = "rgba(38, 166, 154, 0.6)"
            else:
                color = "rgba(38, 166, 154, 0.15)"
                border_color = "rgba(38, 166, 154, 0.5)"
            n_bull += 1
        else:  # bear
            if status == "invalidated":
                color = "rgba(239, 83, 80, 0.08)"
                border_color = "rgba(239, 83, 80, 0.2)"
            elif status == "tested_rejected":
                color = "rgba(239, 83, 80, 0.2)"
                border_color = "rgba(239, 83, 80, 0.6)"
            else:
                color = "rgba(239, 83, 80, 0.15)"
                border_color = "rgba(239, 83, 80, 0.5)"
            n_bear += 1

        fig.add_shape(
            type="rect",
            x0=fvg_start_et,
            x1=fvg_end_et,
            y0=bottom,
            y1=top,
            fillcolor=color,
            line=dict(color=border_color, width=1),
            row=1,
            col=1,
        )

    logger.info("FVG rectangles drawn: %d bull, %d bear", n_bull, n_bear)


def _add_displacement_markers(
    fig: go.Figure,
    df_view: pd.DataFrame,
    features: dict,
    x_vals: pd.DatetimeIndex,
) -> None:
    """Add triangle markers for displacement candles."""
    displacement = features.get("displacement")
    if displacement is None:
        return

    disp_view = displacement.reindex(df_view.index).fillna(False)
    disp_mask = disp_view.astype(bool)

    if disp_mask.sum() == 0:
        return

    disp_bars = df_view[disp_mask]
    disp_x = x_vals[disp_mask]

    # Bullish displacement: close > open
    bull_mask = disp_bars["close"] > disp_bars["open"]
    bear_mask = ~bull_mask

    offset = (df_view["high"].max() - df_view["low"].min()) * 0.012

    # Bullish: triangle-up below the low
    if bull_mask.any():
        bull_bars = disp_bars[bull_mask]
        bull_x_vals = disp_x[bull_mask]
        fig.add_trace(
            go.Scatter(
                x=bull_x_vals,
                y=bull_bars["low"] - offset,
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=8,
                    color="#00e676",
                    line=dict(width=0.5, color="white"),
                ),
                name="Disp Bull",
                hovertext=[
                    f"Bull Disp @ {x}<br>O:{r['open']:.2f} H:{r['high']:.2f} "
                    f"L:{r['low']:.2f} C:{r['close']:.2f}"
                    for x, (_, r) in zip(bull_x_vals, bull_bars.iterrows())
                ],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    # Bearish: triangle-down above the high
    if bear_mask.any():
        bear_bars = disp_bars[bear_mask]
        bear_x_vals = disp_x[bear_mask]
        fig.add_trace(
            go.Scatter(
                x=bear_x_vals,
                y=bear_bars["high"] + offset,
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=8,
                    color="#ff1744",
                    line=dict(width=0.5, color="white"),
                ),
                name="Disp Bear",
                hovertext=[
                    f"Bear Disp @ {x}<br>O:{r['open']:.2f} H:{r['high']:.2f} "
                    f"L:{r['low']:.2f} C:{r['close']:.2f}"
                    for x, (_, r) in zip(bear_x_vals, bear_bars.iterrows())
                ],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    logger.info(
        "Displacement markers: %d bull, %d bear",
        int(bull_mask.sum()),
        int(bear_mask.sum()),
    )


def _add_swing_markers(
    fig: go.Figure,
    df_view: pd.DataFrame,
    features: dict,
    x_vals: pd.DatetimeIndex,
) -> None:
    """Add diamond markers at swing highs and lows."""
    swing = features.get("swing")
    if swing is None:
        return

    swing_view = swing.reindex(df_view.index)
    offset = (df_view["high"].max() - df_view["low"].min()) * 0.015

    # Swing highs
    sh_mask = swing_view["swing_high"].fillna(False).astype(bool)
    if sh_mask.any():
        sh_bars = df_view[sh_mask]
        sh_x = x_vals[sh_mask]
        fig.add_trace(
            go.Scatter(
                x=sh_x,
                y=sh_bars["high"] + offset,
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=7,
                    color="#ffab40",
                    line=dict(width=0.5, color="white"),
                ),
                name="Swing High",
                hovertext=[
                    f"Swing High: {r['high']:.2f}" for _, r in sh_bars.iterrows()
                ],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    # Swing lows
    sl_mask = swing_view["swing_low"].fillna(False).astype(bool)
    if sl_mask.any():
        sl_bars = df_view[sl_mask]
        sl_x = x_vals[sl_mask]
        fig.add_trace(
            go.Scatter(
                x=sl_x,
                y=sl_bars["low"] - offset,
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=7,
                    color="#42a5f5",
                    line=dict(width=0.5, color="white"),
                ),
                name="Swing Low",
                hovertext=[
                    f"Swing Low: {r['low']:.2f}" for _, r in sl_bars.iterrows()
                ],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    logger.info("Swing markers: %d highs, %d lows", int(sh_mask.sum()), int(sl_mask.sum()))


def _add_session_levels(
    fig: go.Figure,
    df_view: pd.DataFrame,
    features: dict,
    x_vals: pd.DatetimeIndex,
) -> None:
    """Add horizontal lines for session levels (asia, london, overnight, NY open)."""
    sessions = features.get("sessions")
    if sessions is None:
        return

    sess_view = sessions.reindex(df_view.index)

    level_config = [
        ("asia_high", "#42a5f5", "dash", "Asia H"),
        ("asia_low", "#42a5f5", "dash", "Asia L"),
        ("london_high", "#ff9800", "dash", "London H"),
        ("london_low", "#ff9800", "dash", "London L"),
        ("overnight_high", "#9e9e9e", "dot", "O/N H"),
        ("overnight_low", "#9e9e9e", "dot", "O/N L"),
        ("ny_open", "#ffffff", "solid", "NY Open"),
    ]

    x_start = x_vals[0]
    x_end = x_vals[-1]

    for col, color, dash, label in level_config:
        if col not in sess_view.columns:
            continue
        vals = sess_view[col].dropna()
        if vals.empty:
            continue

        # Get distinct level values that appear in this view
        # Draw a line for each unique value, starting from where it first appears
        unique_vals = vals.drop_duplicates(keep="first")
        seen_prices = set()

        for ts, val in unique_vals.items():
            if val in seen_prices:
                continue
            seen_prices.add(val)

            seg_start_et = pd.Timestamp(ts).tz_convert("US/Eastern").tz_localize(None)
            fig.add_shape(
                type="line",
                x0=seg_start_et,
                x1=x_end,
                y0=val,
                y1=val,
                line=dict(color=color, width=1, dash=dash),
                row=1,
                col=1,
            )

        # Label the most recent value
        last_val = vals.iloc[-1]
        fig.add_annotation(
            x=x_end,
            y=last_val,
            text=f" {label}: {last_val:.2f}",
            showarrow=False,
            xanchor="left",
            font=dict(color=color, size=9),
            row=1,
            col=1,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_features(
    df_1m: pd.DataFrame,
    start: str,
    end: str,
    features: dict[str, Any] | None = None,
    out_path: str | Path | None = None,
) -> str:
    """Generate an interactive HTML chart with feature overlays.

    Parameters
    ----------
    df_1m : pd.DataFrame
        1-minute OHLCV DataFrame with UTC DatetimeIndex.
    start, end : str
        Date/time strings to slice the visible range (UTC).
        Examples: '2025-03-19 13:00', '2025-03-19 20:00'
    features : dict, optional
        Pre-computed features dict (from _compute_all_features).
        If None, features are computed on-the-fly from the data.
    out_path : str or Path, optional
        Where to save the HTML file.  If None, saves to
        viz/output/chart_{start}_{end}.html.

    Returns
    -------
    str
        Absolute path to the saved HTML file.
    """
    params = _load_params()

    # Parse start/end as UTC
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")

    # We need extra pre-history for indicator warmup (ATR needs ~14 bars minimum,
    # swing needs left_bars, fluency needs window bars).  Grab 500 bars before.
    warmup_start = ts_start - pd.Timedelta(minutes=500)
    df_extended = df_1m.loc[warmup_start:ts_end].copy()

    if len(df_extended) == 0:
        logger.warning("No data in range %s to %s", start, end)
        return ""

    logger.info(
        "plot_features: visible range %s to %s (%d bars), extended for warmup: %d bars",
        ts_start, ts_end, len(df_1m.loc[ts_start:ts_end]), len(df_extended),
    )

    # Compute features if not provided
    if features is None:
        features = _compute_all_features(df_extended, params)

    # Slice to visible range for the chart
    df_view = df_1m.loc[ts_start:ts_end]

    if len(df_view) == 0:
        logger.warning("No visible bars in range %s to %s", start, end)
        return ""

    # Build title
    et_start = pd.Timestamp(ts_start).tz_convert("US/Eastern")
    et_end = pd.Timestamp(ts_end).tz_convert("US/Eastern")
    title = f"NQ 1min | {et_start.strftime('%Y-%m-%d %H:%M')} - {et_end.strftime('%H:%M')} ET"

    fig = _build_figure(df_view, features, title)

    # Save
    if out_path is None:
        safe_start = start.replace(" ", "_").replace(":", "-")
        safe_end = end.replace(" ", "_").replace(":", "-")
        out_path = _OUTPUT_DIR / f"chart_{safe_start}__{safe_end}.html"
    else:
        out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True)
    logger.info("Chart saved to %s", out_path)

    return str(out_path.resolve())


def plot_day(
    date_str: str,
    df_1m: pd.DataFrame | None = None,
    out_path: str | Path | None = None,
) -> str:
    """Plot a full trading day (6 PM previous day ET to 4 PM ET).

    Parameters
    ----------
    date_str : str
        Date string, e.g. '2025-03-19'.  This is the NY session date.
    df_1m : pd.DataFrame, optional
        1-min OHLCV data.  If None, loads from data/NQ_1min.parquet.
    out_path : str or Path, optional
        Output path.  Defaults to viz/output/day_{date_str}.html.

    Returns
    -------
    str
        Absolute path to the saved HTML file.
    """
    if df_1m is None:
        logger.info("Loading 1-min data from %s", _DATA_PATH)
        df_1m = pd.read_parquet(_DATA_PATH)

    # Trading day: from 6 PM ET the previous day to 4 PM ET on the target day
    target_date = pd.Timestamp(date_str)
    prev_date = target_date - pd.Timedelta(days=1)

    # Convert ET boundaries to UTC
    et_start = pd.Timestamp(
        f"{prev_date.strftime('%Y-%m-%d')} 18:00:00",
        tz="US/Eastern",
    )
    et_end = pd.Timestamp(
        f"{target_date.strftime('%Y-%m-%d')} 16:00:00",
        tz="US/Eastern",
    )

    utc_start = et_start.tz_convert("UTC")
    utc_end = et_end.tz_convert("UTC")

    logger.info(
        "plot_day: %s => UTC range %s to %s",
        date_str, utc_start, utc_end,
    )

    if out_path is None:
        out_path = _OUTPUT_DIR / f"day_{date_str}.html"

    return plot_features(
        df_1m,
        start=str(utc_start),
        end=str(utc_end),
        out_path=out_path,
    )


# ---------------------------------------------------------------------------
# __main__ — generate sample charts
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )

    logger.info("Loading 1-min data...")
    df = pd.read_parquet(_DATA_PATH)
    logger.info("Loaded %d bars, %s to %s", len(df), df.index.min(), df.index.max())

    # Chart 1: 2025-03-19 (Wednesday NY session)
    logger.info("=" * 60)
    logger.info("Generating chart 1: 2025-03-19 (NY session)")
    path1 = plot_day("2025-03-19", df_1m=df)
    logger.info("Chart 1 saved: %s", path1)

    # Chart 2: 2025-04-07 (post Liberation Day tariff shock)
    logger.info("=" * 60)
    logger.info("Generating chart 2: 2025-04-07 (tariff shock)")
    path2 = plot_day("2025-04-07", df_1m=df)
    logger.info("Chart 2 saved: %s", path2)

    # Chart 3: A choppy day — pick a recent Monday (typically less directional)
    # 2025-03-24 is a Monday
    logger.info("=" * 60)
    logger.info("Generating chart 3: 2025-03-24 (Monday, potential chop)")
    path3 = plot_day("2025-03-24", df_1m=df)
    logger.info("Chart 3 saved: %s", path3)

    logger.info("=" * 60)
    logger.info("All charts generated:")
    logger.info("  1. %s", path1)
    logger.info("  2. %s", path2)
    logger.info("  3. %s", path3)
