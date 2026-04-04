"""
ninjatrader/bar_by_bar_engine.py — Pure bar-by-bar engine for NQ signal detection + trade management.

Processes 5m bars ONE AT A TIME (like NinjaTrader's OnBarUpdate), maintaining all
state explicitly. Computes signals from raw OHLCV — no pre-computed caches.

CRITICAL DESIGN PRINCIPLE: Every variable is maintained as explicit state. No
vectorized lookbacks into the future. Every calculation at bar i uses only data
from bars 0..i.

Reference implementations:
  - features/entry_signals.py — signal detection logic
  - features/fvg.py — FVG detection
  - features/displacement.py — displacement and fluency
  - features/bias.py — HTF bias computation
  - backtest/engine.py — trade management
  - ninjatrader/validate_nt_logic.py — bar-by-bar validator (uses caches)

Usage:
  python ninjatrader/bar_by_bar_engine.py
"""

from __future__ import annotations

import logging
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class _LiveFVG:
    """Mutable record for a single FVG being tracked during signal scan."""
    idx: int              # bar index where FVG became visible (after c3 closes)
    direction: str        # 'bull' or 'bear'
    top: float
    bottom: float
    size: float
    candle2_open: float = 0.0  # open of the displacement candle (model stop)
    status: str = "untested"   # untested | tested_rejected | invalidated
    is_ifvg: bool = False
    ifvg_direction: str = ""   # 'bull' or 'bear' for IFVGs
    swept_liquidity: bool = False
    sweep_score: int = 0
    invalidated_at_idx: int = -1
    last_signal_idx: int = -999


@dataclass
class _HTF_FVG:
    """Record for an HTF FVG (1H or 4H) used for bias computation."""
    idx: int
    direction: str        # 'bull' or 'bear'
    top: float
    bottom: float
    size: float
    status: str = "untested"


@dataclass
class _SwingPoint:
    """A confirmed swing high or low."""
    idx: int
    price: float
    kind: str  # 'high' or 'low'


# ===========================================================================
# BarByBarEngine
# ===========================================================================

class BarByBarEngine:
    """Processes one 5m bar at a time. Maintains all state internally."""

    def __init__(self, params: dict):
        """Initialize with params from params.yaml."""
        self._params = params
        self._bar_idx = -1  # incremented to 0 on first bar

        # ---- Extract param sections ----
        disp_cfg = params["displacement"]
        flu_cfg = params["fluency"]
        swing_cfg = params["swing"]
        pos_cfg = params["position"]
        risk_cfg = params["risk"]
        bt_cfg = params["backtest"]
        grade_cfg = params["grading"]
        trim_cfg = params["trim"]
        trail_cfg = params["trail"]
        entry_cfg = params.get("entry", {})

        # Displacement params
        self._disp_atr_mult: float = disp_cfg["atr_mult"]
        self._disp_body_ratio: float = disp_cfg["body_ratio"]
        self._disp_engulf_min: int = disp_cfg.get("engulf_min_candles", 1)

        # Fluency params
        self._flu_window: int = flu_cfg["window"]
        self._flu_threshold: float = flu_cfg["threshold"]
        self._flu_w1: float = flu_cfg["w_directional"]
        self._flu_w2: float = flu_cfg["w_body_ratio"]
        self._flu_w3: float = flu_cfg["w_bar_size"]

        # Swing params
        self._swing_left: int = swing_cfg["left_bars"]
        self._swing_right: int = swing_cfg["right_bars"]

        # Entry signal params
        self._min_fvg_atr_mult: float = entry_cfg.get("min_fvg_atr_mult", 0.3)
        self._rejection_body_ratio: float = entry_cfg.get("rejection_body_ratio", 0.50)
        self._signal_cooldown: int = entry_cfg.get("signal_cooldown_bars", 6)
        self._require_displacement: bool = entry_cfg.get("require_displacement", False)
        self._sweep_lookback: int = entry_cfg.get("sweep_lookback", 20)

        # Stop loss params
        self._small_candle_atr_mult: float = params["stop_loss"]["small_candle_atr_mult"]

        # Position sizing
        self._normal_r: float = pos_cfg["normal_r"]
        self._reduced_r: float = pos_cfg["reduced_r"]
        self._point_value: float = pos_cfg["point_value"]

        # Risk management
        self._daily_max_loss_r: float = risk_cfg["daily_max_loss_r"]
        self._max_consec_losses: int = risk_cfg["max_consecutive_losses"]

        # Backtest params
        self._commission_per_side: float = bt_cfg["commission_per_side_micro"]
        self._slippage_ticks: int = bt_cfg["slippage_normal_ticks"]
        self._slippage_points: float = self._slippage_ticks * 0.25

        # Grading
        self._a_plus_mult: float = grade_cfg["a_plus_size_mult"]
        self._b_plus_mult: float = grade_cfg["b_plus_size_mult"]
        self._c_skip: bool = grade_cfg["c_skip"]

        # Trim / trail
        self._trim_pct: float = trim_cfg["pct"]
        self._be_after_trim: bool = trim_cfg["be_after_trim"]
        self._nth_swing: int = trail_cfg["use_nth_swing"]

        # PA quality threshold
        self._pa_threshold: float = params.get("pa_quality", {}).get("alt_dir_threshold", 0.334)

        # Min stop ATR mult
        self._min_stop_atr_mult: float = params.get("regime", {}).get("min_stop_atr_mult", 0.5)

        # Session filter
        sf = params.get("session_filter", {})
        self._sf_enabled: bool = sf.get("enabled", False)
        self._skip_london: bool = sf.get("skip_london", False)
        self._skip_asia: bool = sf.get("skip_asia", True)
        self._ny_direction: int = sf.get("ny_direction", 0)
        self._london_direction: int = sf.get("london_direction", 0)

        # Session rules
        sr = params.get("session_rules", {})
        self._session_rules_enabled: bool = sr.get("enabled", False)
        self._ny_tp_mult: float = sr.get("ny_tp_multiplier", 1.0)

        # Session regime
        sreg = params.get("session_regime", {})
        self._session_regime_enabled: bool = sreg.get("enabled", False)
        self._sr_am_end: float = sreg.get("am_end", 12.0)
        self._sr_lunch_start: float = sreg.get("lunch_start", 12.0)
        self._sr_lunch_end: float = sreg.get("lunch_end", 13.5)
        self._sr_pm_start: float = sreg.get("pm_start", 13.5)
        self._sr_am_mult: float = sreg.get("am_mult", 1.0)
        self._sr_lunch_mult: float = sreg.get("lunch_mult", 0.5)
        self._sr_pm_mult: float = sreg.get("pm_mult", 0.75)

        # Direction management
        dm = params.get("direction_mgmt", {})
        self._dir_mgmt_enabled: bool = dm.get("enabled", False)
        self._long_tp_mult: float = dm.get("long_tp_mult", 2.0)
        self._short_tp_mult: float = dm.get("short_tp_mult", 1.25)
        self._long_trim_pct: float = dm.get("long_trim_pct", self._trim_pct)
        self._short_trim_pct: float = dm.get("short_trim_pct", 1.0)

        # Dual mode
        dual = params.get("dual_mode", {})
        self._dual_mode_enabled: bool = dual.get("enabled", False)
        self._long_sq_threshold: float = dual.get("long_sq_threshold", 0.68)
        self._short_sq_threshold: float = dual.get("short_sq_threshold", 0.82)
        self._short_rr: float = dual.get("short_rr", 0.625)
        self._dual_short_trim_pct: float = dual.get("short_trim_pct", 1.0)

        # Signal quality
        sq = params.get("signal_quality", {})
        self._sq_enabled: bool = sq.get("enabled", False)
        self._sq_threshold: float = sq.get("threshold", 0.66)
        self._sq_w_size: float = sq.get("w_size", 0.30)
        self._sq_w_disp: float = sq.get("w_disp", 0.30)
        self._sq_w_flu: float = sq.get("w_flu", 0.20)
        self._sq_w_pa: float = sq.get("w_pa", 0.20)

        # Bias relaxation
        br = params.get("bias_relaxation", {})
        self._bias_relax_enabled: bool = br.get("enabled", False)

        # SMT config
        smt = params.get("smt", {})
        self._smt_enabled: bool = smt.get("enabled", False)
        self._smt_sweep_lookback: int = smt.get("sweep_lookback", 15)
        self._smt_time_tolerance: int = smt.get("time_tolerance", 1)
        self._smt_require_for_mss: bool = smt.get("require_for_mss", True)
        self._smt_bypass_session: bool = smt.get("bypass_session_filter", False)

        # MSS management
        mss = params.get("mss_management", {})
        self._mss_mgmt_enabled: bool = mss.get("enabled", False)
        self._mss_long_tp_mult: float = mss.get("long_tp_mult", 2.0)
        self._mss_short_rr: float = mss.get("short_rr", 0.50)
        self._mss_short_trim_pct: float = mss.get("short_trim_pct", 1.0)
        self._mss_long_trim_pct: float = mss.get("long_trim_pct", self._trim_pct)

        # Signal filter
        sig_f = params.get("signal_filter", {})
        self._allow_trend: bool = sig_f.get("allow_trend", True)
        self._allow_mss: bool = sig_f.get("allow_mss", True)

        # News config
        news = params.get("news", {})
        self._news_blackout_before: int = news.get("blackout_minutes_before", 60)
        self._news_cooldown_after: int = news.get("cooldown_minutes_after", 5)

        # ---- ATR state (Wilder smoothing, period=14) ----
        self._atr_period: int = 14
        self._atr_buffer: list[float] = []
        self._atr: float = 0.0
        self._atr_ready: bool = False
        self._prev_close: float = 0.0
        self._first_bar: bool = True

        # ---- Candle buffer (for FVG detection, displacement, fluency) ----
        # Keep at least max(3, flu_window, disp_engulf+1) bars
        buf_size = max(20, self._flu_window + 5, self._disp_engulf_min + 5)
        self._candle_buffer: deque[dict] = deque(maxlen=buf_size)

        # ---- Swing detection state ----
        swing_buf_size = self._swing_left + self._swing_right + 1
        self._high_buffer: deque[tuple[int, float]] = deque(maxlen=swing_buf_size)
        self._low_buffer: deque[tuple[int, float]] = deque(maxlen=swing_buf_size)
        self._swing_highs: list[_SwingPoint] = []  # all confirmed swings
        self._swing_lows: list[_SwingPoint] = []
        self._swing_high_price: float = float('nan')  # most recent confirmed (ffilled)
        self._swing_low_price: float = float('nan')

        # HTF swing detection (left=10, right=3 for significant levels)
        self._htf_swing_left: int = 10
        self._htf_swing_right: int = 3
        htf_buf_size = self._htf_swing_left + self._htf_swing_right + 1
        self._htf_high_buffer: deque[tuple[int, float]] = deque(maxlen=htf_buf_size)
        self._htf_low_buffer: deque[tuple[int, float]] = deque(maxlen=htf_buf_size)
        self._htf_swing_high_price: float = float('nan')
        self._htf_swing_low_price: float = float('nan')

        # ---- FVG state ----
        self._active_fvgs: list[_LiveFVG] = []
        self._active_ifvgs: list[_LiveFVG] = []

        # ---- Session state ----
        # Completed sub-session highs/lows (5 sub-sessions)
        self._session_names = ["asia", "london", "ny_am", "ny_lunch", "ny_pm"]
        self._session_bounds = [
            (18.0, 3.0, True),    # asia wraps midnight
            (3.0, 9.5, False),    # london
            (9.5, 11.0, False),   # ny_am
            (11.0, 13.0, False),  # ny_lunch
            (13.0, 16.0, False),  # ny_pm
        ]
        self._completed_highs: list[float] = [float('nan')] * 5
        self._completed_lows: list[float] = [float('nan')] * 5
        self._running_highs: list[float] = [float('nan')] * 5
        self._running_lows: list[float] = [float('nan')] * 5
        self._in_session: list[bool] = [False] * 5

        # Overnight state
        self._overnight_high: float = float('nan')
        self._overnight_low: float = float('nan')
        self._overnight_running_h: float = float('nan')
        self._overnight_running_l: float = float('nan')
        self._in_overnight: bool = False

        # ORM state (9:30-10:00 ET)
        self._orm_high: float = float('nan')
        self._orm_low: float = float('nan')
        self._orm_running_h: float = float('nan')
        self._orm_running_l: float = float('nan')
        self._in_orm: bool = False
        self._orm_bias: float = 0.0
        self._orm_bias_locked: bool = False
        self._ny_open_price: float = float('nan')
        self._ny_open_locked: bool = False

        # ---- Sweep tracking (for MSS) ----
        self._swept_low_buffer: deque[tuple[int, bool]] = deque(maxlen=100)
        self._swept_high_buffer: deque[tuple[int, bool]] = deque(maxlen=100)

        # ---- HTF bias state ----
        self._htf_1h_candle_buffer: deque[dict] = deque(maxlen=5)
        self._htf_4h_candle_buffer: deque[dict] = deque(maxlen=5)
        self._htf_1h_fvgs: list[_HTF_FVG] = []
        self._htf_4h_fvgs: list[_HTF_FVG] = []
        self._htf_bias_1h: float = 0.0
        self._htf_bias_4h: float = 0.0
        self._htf_pda_count: int = 0

        # ---- SMT state ----
        self._es_swing_highs: list[_SwingPoint] = []
        self._es_swing_lows: list[_SwingPoint] = []
        self._es_high_buffer: deque[tuple[int, float]] = deque(maxlen=swing_buf_size)
        self._es_low_buffer: deque[tuple[int, float]] = deque(maxlen=swing_buf_size)
        self._es_swing_high_price: float = float('nan')
        self._es_swing_low_price: float = float('nan')
        self._smt_bull: bool = False
        self._smt_bear: bool = False

        # ---- Daily state ----
        self._current_date = None
        self._daily_pnl_r: float = 0.0
        self._consecutive_losses: int = 0
        self._day_stopped: bool = False

        # ---- Position state ----
        self._in_position: bool = False
        self._pos_direction: int = 0
        self._pos_entry_idx: int = 0
        self._pos_entry_price: float = 0.0
        self._pos_stop: float = 0.0
        self._pos_tp1: float = 0.0
        self._pos_contracts: int = 0
        self._pos_remaining_contracts: int = 0
        self._pos_trimmed: bool = False
        self._pos_be_stop: float = 0.0
        self._pos_trail_stop: float = 0.0
        self._pos_signal_type: str = ""
        self._pos_grade: str = ""
        self._pos_trim_pct: float = self._trim_pct
        self._pos_bias_dir: float = 0.0
        self._pos_regime: float = 0.0
        self._pos_entry_time: Any = None  # timestamp of entry bar

        # ---- Bias state ----
        self._bias_direction: float = 0.0
        self._bias_confidence: float = 0.0
        self._regime: float = 0.0

        # ---- Pending entry (signal detected on previous bar, enter on this bar) ----
        self._pending_entry: dict | None = None

        # ---- Cache mode: use pre-computed signals instead of detecting from scratch ----
        self._use_cache: bool = False
        self._cache_signals: dict = {}  # bar_idx -> signal dict
        self._cache_bias: dict = {}  # bar_idx -> (bias_dir, bias_conf, regime)
        self._cache_fluency: dict = {}  # bar_idx -> fluency_value
        self._cache_pa_alt: dict = {}  # bar_idx -> pa_alt_dir_ratio
        self._cache_atr: dict = {}  # bar_idx -> atr_value
        self._cache_sq: dict = {}  # bar_idx -> signal_quality_value

        # ---- News blackout times (set externally) ----
        self._news_blackout_times: list[tuple[datetime, datetime]] = []

        # ---- Complete trade log ----
        self._completed_trades: list[dict] = []

        # ---- Diagnostic counters for filter funnel ----
        self._diag: dict[str, int] = {
            "raw_trend_signals": 0,
            "raw_mss_signals": 0,
            "mss_blocked_smt_gate": 0,
            "mss_blocked_overnight": 0,
            "blocked_by_news": 0,
            "blocked_by_orm": 0,
            "blocked_by_day_stopped": 0,
            "blocked_by_in_position": 0,
            "signals_entering_filters": 0,
            "blocked_by_bias": 0,
            "blocked_by_pa_quality": 0,
            "blocked_by_session": 0,
            "blocked_by_invalid_prices": 0,
            "blocked_by_stop_target_logic": 0,
            "blocked_by_min_stop": 0,
            "blocked_by_sq": 0,
            "blocked_by_news_filter": 0,
            "blocked_by_daily_limits": 0,
            "blocked_by_position": 0,
            "blocked_by_lunch_deadzone": 0,
            "blocked_by_grade_c_skip": 0,
            "signals_passed_all_filters": 0,
            "entries_executed": 0,
        }

    # =======================================================================
    # 1. ATR (Wilder smoothing)
    # =======================================================================

    def _update_atr(self, high: float, low: float, close: float) -> None:
        """Wilder-smoothed ATR(14). Updates self._atr."""
        if self._first_bar:
            self._prev_close = close
            self._first_bar = False
            tr = high - low
        else:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
            self._prev_close = close

        if not self._atr_ready:
            self._atr_buffer.append(tr)
            if len(self._atr_buffer) == self._atr_period:
                self._atr = sum(self._atr_buffer) / self._atr_period
                self._atr_ready = True
        else:
            self._atr = (self._atr * (self._atr_period - 1) + tr) / self._atr_period

    def _get_atr(self) -> float:
        """Return current ATR or 0.0 if not ready."""
        return self._atr if self._atr_ready else 0.0

    # =======================================================================
    # 2. Swing Detection (fractal)
    # =======================================================================

    def _update_swings(self, high: float, low: float, bar_idx: int) -> None:
        """Fractal swing detection. left=3, right=1 (default).

        Adds new (bar_idx, price) to the buffers. When buffer is full, checks
        if the bar at position [left] is a swing.

        A swing high at position P is confirmed when we have right_bars bars
        after P. So it's naturally delayed.
        """
        self._high_buffer.append((bar_idx, high))
        self._low_buffer.append((bar_idx, low))

        buf_needed = self._swing_left + self._swing_right + 1
        if len(self._high_buffer) >= buf_needed:
            self._check_swing_at(
                self._high_buffer, self._low_buffer,
                self._swing_left, self._swing_right,
                self._swing_highs, self._swing_lows,
            )

        # Update ffilled prices
        if self._swing_highs:
            self._swing_high_price = self._swing_highs[-1].price
        if self._swing_lows:
            self._swing_low_price = self._swing_lows[-1].price

    def _check_swing_at(
        self,
        high_buf: deque,
        low_buf: deque,
        left: int,
        right: int,
        swing_highs: list[_SwingPoint],
        swing_lows: list[_SwingPoint],
    ) -> None:
        """Check if the bar at position [left] in the buffer is a swing."""
        buf_needed = left + right + 1
        if len(high_buf) < buf_needed:
            return

        # Convert to list for indexed access
        hb = list(high_buf)
        lb = list(low_buf)

        # The candidate is at position [left] from the end
        # When deque has exactly buf_needed items, candidate is at index left
        candidate_pos = len(hb) - 1 - right  # position of candidate
        cand_idx, cand_high = hb[candidate_pos]
        _, cand_low = lb[candidate_pos]

        # Check swing high
        is_swing_high = True
        for j in range(candidate_pos - left, candidate_pos):
            if hb[j][1] >= cand_high:
                is_swing_high = False
                break
        if is_swing_high:
            for j in range(candidate_pos + 1, candidate_pos + right + 1):
                if j < len(hb) and hb[j][1] >= cand_high:
                    is_swing_high = False
                    break

        # Check swing low
        is_swing_low = True
        for j in range(candidate_pos - left, candidate_pos):
            if lb[j][1] <= cand_low:
                is_swing_low = False
                break
        if is_swing_low:
            for j in range(candidate_pos + 1, candidate_pos + right + 1):
                if j < len(lb) and lb[j][1] <= cand_low:
                    is_swing_low = False
                    break

        if is_swing_high:
            # Avoid duplicates
            if not swing_highs or swing_highs[-1].idx != cand_idx:
                swing_highs.append(_SwingPoint(idx=cand_idx, price=cand_high, kind='high'))

        if is_swing_low:
            if not swing_lows or swing_lows[-1].idx != cand_idx:
                swing_lows.append(_SwingPoint(idx=cand_idx, price=cand_low, kind='low'))

    def _update_htf_swings(self, high: float, low: float, bar_idx: int) -> None:
        """HTF-like swing on 5m data: left=10, right=3 for significant levels."""
        self._htf_high_buffer.append((bar_idx, high))
        self._htf_low_buffer.append((bar_idx, low))

        buf_needed = self._htf_swing_left + self._htf_swing_right + 1
        if len(self._htf_high_buffer) >= buf_needed:
            # Check swing at position [left] from the newer end
            hb = list(self._htf_high_buffer)
            lb = list(self._htf_low_buffer)
            candidate_pos = len(hb) - 1 - self._htf_swing_right

            cand_idx, cand_high = hb[candidate_pos]
            _, cand_low = lb[candidate_pos]

            # Swing high check
            is_sh = True
            for j in range(candidate_pos - self._htf_swing_left, candidate_pos):
                if j >= 0 and hb[j][1] >= cand_high:
                    is_sh = False
                    break
            if is_sh:
                for j in range(candidate_pos + 1, candidate_pos + self._htf_swing_right + 1):
                    if j < len(hb) and hb[j][1] >= cand_high:
                        is_sh = False
                        break
            if is_sh:
                self._htf_swing_high_price = cand_high

            # Swing low check
            is_sl = True
            for j in range(candidate_pos - self._htf_swing_left, candidate_pos):
                if j >= 0 and lb[j][1] <= cand_low:
                    is_sl = False
                    break
            if is_sl:
                for j in range(candidate_pos + 1, candidate_pos + self._htf_swing_right + 1):
                    if j < len(lb) and lb[j][1] <= cand_low:
                        is_sl = False
                        break
            if is_sl:
                self._htf_swing_low_price = cand_low

    def _find_nth_swing_price(self, direction: int, n: int) -> float:
        """Find the nth most recent swing low (long) or swing high (short).

        For trailing stops:
          direction=1 (long) → nth swing low price
          direction=-1 (short) → nth swing high price
        """
        if direction == 1:
            swings = self._swing_lows
        else:
            swings = self._swing_highs

        if len(swings) < n:
            return float('nan')

        # nth from the end (1-indexed)
        return swings[-n].price

    # =======================================================================
    # 3. FVG Detection
    # =======================================================================

    def _detect_fvg(self) -> None:
        """Check if the last 3 completed bars form an FVG.

        The FVG is known NOW (at bar_idx), meaning the 3 candles are:
          c1 = candle_buffer[-3], c2 = candle_buffer[-2], c3 = candle_buffer[-1]
        The FVG is anchored to c2 but only visible after c3 closes.
        """
        if len(self._candle_buffer) < 3:
            return

        buf = list(self._candle_buffer)
        c1 = buf[-3]
        c2 = buf[-2]
        c3 = buf[-1]

        # Skip if any candle is on a rollover date
        if c1.get('is_roll_date', False) or c2.get('is_roll_date', False) or c3.get('is_roll_date', False):
            return

        bar_idx = self._bar_idx

        # Compute sweep score from c2 (displacement candle)
        swept = False
        sweep_score = 0
        c2_high = c2['high']
        c2_low = c2['low']

        # Session levels (completed sub-session H/L)
        for i_s in range(5):
            ch = self._completed_highs[i_s]
            cl = self._completed_lows[i_s]
            if not math.isnan(ch) and c2_high > ch:
                sweep_score += 2
            if not math.isnan(cl) and c2_low < cl:
                sweep_score += 2

        # HTF swing levels
        if not math.isnan(self._htf_swing_high_price) and c2_high > self._htf_swing_high_price:
            sweep_score += 2
        if not math.isnan(self._htf_swing_low_price) and c2_low < self._htf_swing_low_price:
            sweep_score += 2

        swept = sweep_score >= 2

        c2_open_val = c2['open']

        # Bullish FVG: c1.high < c3.low
        if c1['high'] < c3['low']:
            gap_top = c3['low']
            gap_bot = c1['high']
            fvg_size = gap_top - gap_bot
            rec = _LiveFVG(
                idx=bar_idx,
                direction="bull",
                top=gap_top,
                bottom=gap_bot,
                size=fvg_size,
                candle2_open=c2_open_val,
                swept_liquidity=swept,
                sweep_score=sweep_score,
            )
            self._active_fvgs.append(rec)

        # Bearish FVG: c1.low > c3.high
        if c1['low'] > c3['high']:
            gap_top = c1['low']
            gap_bot = c3['high']
            fvg_size = gap_top - gap_bot
            rec = _LiveFVG(
                idx=bar_idx,
                direction="bear",
                top=gap_top,
                bottom=gap_bot,
                size=fvg_size,
                candle2_open=c2_open_val,
                swept_liquidity=swept,
                sweep_score=sweep_score,
            )
            self._active_fvgs.append(rec)

    # =======================================================================
    # 4. FVG State Machine
    # =======================================================================

    def _update_fvg_status(
        self, rec: _LiveFVG, bar_high: float, bar_low: float, bar_close: float
    ) -> str:
        """Determine new status for a _LiveFVG given one bar's OHLC."""
        if rec.status == "invalidated":
            return "invalidated"

        top = rec.top
        bottom = rec.bottom
        direction = rec.ifvg_direction if rec.is_ifvg else rec.direction

        if direction == "bull":
            entered = bar_low <= top and bar_high >= bottom
            if entered:
                if bar_close < bottom:
                    return "invalidated"
                if bar_close >= top:
                    return "tested_rejected" if rec.status == "untested" else rec.status
                return "tested_rejected" if rec.status == "untested" else rec.status
            if bar_close < bottom:
                return "invalidated"
            return rec.status
        else:  # bear
            entered = bar_high >= bottom and bar_low <= top
            if entered:
                if bar_close > top:
                    return "invalidated"
                if bar_close <= bottom:
                    return "tested_rejected" if rec.status == "untested" else rec.status
                return "tested_rejected" if rec.status == "untested" else rec.status
            if bar_close > top:
                return "invalidated"
            return rec.status

    def _update_all_fvg_states(self, high: float, low: float, close: float, bar_idx: int) -> None:
        """Update all active FVGs and IFVGs. Called AFTER signal detection."""
        # Update regular FVGs
        surviving: list[_LiveFVG] = []
        new_ifvgs: list[_LiveFVG] = []

        for rec in self._active_fvgs:
            old_status = rec.status
            new_status = self._update_fvg_status(rec, high, low, close)

            if new_status == "invalidated" and old_status != "invalidated":
                rec.status = "invalidated"
                rec.invalidated_at_idx = bar_idx
                if not rec.is_ifvg:
                    ifvg_dir = "bear" if rec.direction == "bull" else "bull"
                    # IFVG stop = FVG boundary on correct side
                    ifvg_stop = rec.bottom if ifvg_dir == "bull" else rec.top
                    ifvg = _LiveFVG(
                        idx=bar_idx,
                        direction=ifvg_dir,
                        top=rec.top,
                        bottom=rec.bottom,
                        size=rec.size,
                        candle2_open=ifvg_stop,
                        status="untested",
                        is_ifvg=True,
                        ifvg_direction=ifvg_dir,
                    )
                    new_ifvgs.append(ifvg)
                    self._active_ifvgs.append(ifvg)
            else:
                rec.status = new_status
                surviving.append(rec)

        self._active_fvgs = surviving + new_ifvgs

        # Update IFVGs
        surviving_ifvgs: list[_LiveFVG] = []
        for ifvg in self._active_ifvgs:
            if ifvg.idx == bar_idx:
                surviving_ifvgs.append(ifvg)
                continue  # just born this bar, don't test yet
            if ifvg.status == "invalidated":
                continue
            new_status = self._update_fvg_status(ifvg, high, low, close)
            ifvg.status = new_status
            if new_status != "invalidated":
                surviving_ifvgs.append(ifvg)
        self._active_ifvgs = surviving_ifvgs

        # Prune FVGs older than 500 bars
        cutoff = bar_idx - 500
        self._active_fvgs = [r for r in self._active_fvgs if r.idx > cutoff]
        self._active_ifvgs = [r for r in self._active_ifvgs if r.idx > cutoff]

    # =======================================================================
    # 5. Displacement Detection
    # =======================================================================

    def _check_displacement(self, open_: float, high: float, low: float, close: float, atr: float) -> bool:
        """Check if current bar shows displacement."""
        if atr <= 0:
            return False

        body = abs(close - open_)
        bar_range = high - low
        if bar_range <= 0:
            return False

        # Criterion 1: body > atr_mult * ATR
        if body <= self._disp_atr_mult * atr:
            return False

        # Criterion 2: body/range >= body_ratio
        if body / bar_range < self._disp_body_ratio:
            return False

        # Criterion 3: engulfs at least N prior candles
        if len(self._candle_buffer) < self._disp_engulf_min + 1:
            return False

        candle_top = max(open_, close)
        candle_bot = min(open_, close)
        buf = list(self._candle_buffer)
        engulf_count = 0
        for k in range(1, self._disp_engulf_min + 1):
            if len(buf) - 1 - k < 0:
                break
            prior = buf[-(1 + k)]
            if candle_top >= prior['high'] and candle_bot <= prior['low']:
                engulf_count += 1

        return engulf_count >= self._disp_engulf_min

    # =======================================================================
    # 6. Fluency Scoring
    # =======================================================================

    def _compute_fluency(self) -> float:
        """Rolling fluency over last N bars from candle buffer."""
        window = self._flu_window
        if len(self._candle_buffer) < window:
            return float('nan')

        buf = list(self._candle_buffer)
        recent = buf[-window:]
        atr = self._get_atr()

        bull_count = 0
        bear_count = 0
        body_ratio_sum = 0.0
        bar_size_sum = 0.0

        for bar in recent:
            o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']
            direction = 1 if c > o else (-1 if c < o else 0)
            if direction == 1:
                bull_count += 1
            elif direction == -1:
                bear_count += 1

            bar_range = h - l
            body = abs(c - o)

            if bar_range > 0:
                body_ratio_sum += body / bar_range
            # else: body_ratio = 0.0

            if atr > 0:
                ratio = min(bar_range / atr, 2.0)
                bar_size_sum += ratio

        directional_ratio = max(bull_count, bear_count) / window
        avg_body_ratio = body_ratio_sum / window
        avg_bar_size = min(bar_size_sum / window, 1.0) if atr > 0 else 0.0

        fluency = (self._flu_w1 * directional_ratio +
                   self._flu_w2 * avg_body_ratio +
                   self._flu_w3 * avg_bar_size)
        return max(0.0, min(1.0, fluency))

    # =======================================================================
    # 7. Session Detection
    # =======================================================================

    def _update_session(self, time_et: datetime, high: float, low: float, open_: float) -> str:
        """Track session state and liquidity levels.

        Returns current session name: 'asia', 'london', 'ny'
        """
        h_frac = time_et.hour + time_et.minute / 60.0

        # Update sub-sessions
        for i_s, (start, end, wraps) in enumerate(self._session_bounds):
            if wraps:
                now_in = h_frac >= start or h_frac < end
            else:
                now_in = start <= h_frac < end

            if now_in:
                if not self._in_session[i_s]:
                    # Session just started
                    self._running_highs[i_s] = high
                    self._running_lows[i_s] = low
                    self._in_session[i_s] = True
                else:
                    self._running_highs[i_s] = max(self._running_highs[i_s], high)
                    self._running_lows[i_s] = min(self._running_lows[i_s], low)
            else:
                if self._in_session[i_s]:
                    # Session just ended
                    self._completed_highs[i_s] = self._running_highs[i_s]
                    self._completed_lows[i_s] = self._running_lows[i_s]
                    self._in_session[i_s] = False

        # Overnight (Asia + London combined = 18:00-09:30 ET)
        in_overnight = h_frac >= 18.0 or h_frac < 9.5
        if in_overnight:
            if not self._in_overnight:
                self._overnight_running_h = high
                self._overnight_running_l = low
                self._in_overnight = True
            else:
                self._overnight_running_h = max(self._overnight_running_h, high)
                self._overnight_running_l = min(self._overnight_running_l, low)
        else:
            if self._in_overnight:
                self._overnight_high = self._overnight_running_h
                self._overnight_low = self._overnight_running_l
                self._in_overnight = False

        # NY open price (first bar at 9:30 ET)
        if not self._ny_open_locked and 9.5 <= h_frac < 9.5 + 5.0/60.0:
            self._ny_open_price = open_
            self._ny_open_locked = True

        # ORM (9:30-10:00 ET)
        in_orm = 9.5 <= h_frac < 10.0
        if in_orm:
            if not self._in_orm:
                self._orm_running_h = high
                self._orm_running_l = low
                self._in_orm = True
            else:
                self._orm_running_h = max(self._orm_running_h, high)
                self._orm_running_l = min(self._orm_running_l, low)
        else:
            if self._in_orm:
                self._orm_high = self._orm_running_h
                self._orm_low = self._orm_running_l
                self._in_orm = False

        # Lock ORM bias at 10:00 ET
        if not self._orm_bias_locked and h_frac >= 10.0 and not math.isnan(self._overnight_high):
            if not math.isnan(self._orm_high) and not math.isnan(self._overnight_high):
                if self._orm_high > self._overnight_high:
                    self._orm_bias = 1.0  # broke above overnight
                elif self._orm_low < self._overnight_low:
                    self._orm_bias = -1.0  # broke below overnight
                else:
                    self._orm_bias = 0.0
                self._orm_bias_locked = True

        # Reset daily session state at 18:00 ET (new futures session)
        if 18.0 <= h_frac < 18.0 + 5.0/60.0:
            # Check if we need to reset (only on first bar of new session)
            if self._ny_open_locked:  # Was locked from previous day
                self._ny_open_locked = False
                self._ny_open_price = float('nan')
                self._orm_bias_locked = False
                self._orm_bias = 0.0
                self._orm_high = float('nan')
                self._orm_low = float('nan')

        # Determine current macro session
        if h_frac >= 18.0 or h_frac < 3.0:
            return "asia"
        elif 3.0 <= h_frac < 9.5:
            return "london"
        else:
            return "ny"

    # =======================================================================
    # 8. Trend Signal Detection
    # =======================================================================

    def _check_trend_signal(self, bar_idx: int, close: float, high: float,
                             low: float, open_: float, atr: float) -> dict | None:
        """Check for Trend signal: FVG test + rejection."""
        if atr <= 0:
            return None

        bar_range = high - low
        if bar_range <= 0:
            return None

        body = abs(close - open_)
        body_ratio = body / bar_range

        # Pre-check: rejection candle body/range
        if body_ratio < self._rejection_body_ratio:
            return None

        # Fluency check
        fluency = self._compute_fluency()
        if math.isnan(fluency) or fluency < self._flu_threshold:
            return None

        # Displacement check
        is_displaced = self._check_displacement(open_, high, low, close, atr)

        if self._require_displacement and not is_displaced:
            return None

        best_signal_dir = 0
        best_fvg: _LiveFVG | None = None
        best_score = -1.0

        for rec in self._active_fvgs:
            if rec.status == "invalidated":
                continue
            if rec.is_ifvg:
                continue
            # No signal on FVG creation bar
            if rec.idx >= bar_idx:
                continue
            # FVG size filter
            if atr > 0 and rec.size < self._min_fvg_atr_mult * atr:
                continue
            # Cooldown
            if (bar_idx - rec.last_signal_idx) < self._signal_cooldown:
                continue

            if rec.direction == "bull":
                entered = low <= rec.top and high >= rec.bottom
                rejected = close > rec.top
                if entered and rejected:
                    score = rec.size + (100.0 if is_displaced else 0.0) + (200.0 if rec.swept_liquidity else 0.0)
                    if score > best_score:
                        best_score = score
                        best_signal_dir = 1
                        best_fvg = rec

            elif rec.direction == "bear":
                entered = high >= rec.bottom and low <= rec.top
                rejected = close < rec.bottom
                if entered and rejected:
                    score = rec.size + (100.0 if is_displaced else 0.0) + (200.0 if rec.swept_liquidity else 0.0)
                    if score > best_score:
                        best_score = score
                        best_signal_dir = -1
                        best_fvg = rec

        if best_fvg is not None:
            best_fvg.last_signal_idx = bar_idx
            return {
                'signal_type': 'trend',
                'direction': best_signal_dir,
                'fvg_top': best_fvg.top,
                'fvg_bottom': best_fvg.bottom,
                'fvg_size': best_fvg.size,
                'fvg_c2_open': best_fvg.candle2_open,
                'swept': best_fvg.swept_liquidity,
                'sweep_score': best_fvg.sweep_score,
                'is_displaced': is_displaced,
                'fluency': fluency,
            }
        return None

    # =======================================================================
    # 9. MSS Signal Detection
    # =======================================================================

    def _check_mss_signal(self, bar_idx: int, close: float, high: float,
                           low: float, open_: float, atr: float) -> dict | None:
        """Check for MSS signal: IFVG retest + respect."""
        if atr <= 0:
            return None

        bar_range = high - low
        if bar_range <= 0:
            return None

        body = abs(close - open_)
        body_ratio = body / bar_range

        if body_ratio < self._rejection_body_ratio:
            return None

        fluency = self._compute_fluency()
        if math.isnan(fluency) or fluency < self._flu_threshold:
            return None

        is_displaced = self._check_displacement(open_, high, low, close, atr)

        best_signal_dir = 0
        best_ifvg: _LiveFVG | None = None
        best_score = -1.0

        for ifvg in self._active_ifvgs:
            if ifvg.status == "invalidated":
                continue
            if ifvg.idx >= bar_idx:
                continue
            if atr > 0 and ifvg.size < self._min_fvg_atr_mult * atr:
                continue
            if (bar_idx - ifvg.last_signal_idx) < self._signal_cooldown:
                continue

            if ifvg.ifvg_direction == "bull":
                # Check: recent swing low sweep before IFVG formed
                had_sweep = self._had_sweep_low_before(ifvg.idx)
                if not had_sweep:
                    continue
                entered = low <= ifvg.top and high >= ifvg.bottom
                respected = close > ifvg.top
                if entered and respected:
                    score = ifvg.size + (100.0 if is_displaced else 0.0)
                    if score > best_score:
                        best_score = score
                        best_signal_dir = 1
                        best_ifvg = ifvg

            elif ifvg.ifvg_direction == "bear":
                had_sweep = self._had_sweep_high_before(ifvg.idx)
                if not had_sweep:
                    continue
                entered = high >= ifvg.bottom and low <= ifvg.top
                respected = close < ifvg.bottom
                if entered and respected:
                    score = ifvg.size + (100.0 if is_displaced else 0.0)
                    if score > best_score:
                        best_score = score
                        best_signal_dir = -1
                        best_ifvg = ifvg

        if best_ifvg is not None:
            best_ifvg.last_signal_idx = bar_idx
            return {
                'signal_type': 'mss',
                'direction': best_signal_dir,
                'fvg_top': best_ifvg.top,
                'fvg_bottom': best_ifvg.bottom,
                'fvg_size': best_ifvg.size,
                'fvg_c2_open': best_ifvg.candle2_open,
                'swept': True,
                'sweep_score': 4,
                'is_displaced': is_displaced,
                'fluency': fluency,
            }
        return None

    def _had_sweep_low_before(self, ifvg_idx: int) -> bool:
        """Check if a swing low was swept in the lookback window before IFVG birth."""
        start = max(0, ifvg_idx - self._sweep_lookback)
        for idx, swept in self._swept_low_buffer:
            if start <= idx <= ifvg_idx and swept:
                return True
        return False

    def _had_sweep_high_before(self, ifvg_idx: int) -> bool:
        """Check if a swing high was swept in the lookback window before IFVG birth."""
        start = max(0, ifvg_idx - self._sweep_lookback)
        for idx, swept in self._swept_high_buffer:
            if start <= idx <= ifvg_idx and swept:
                return True
        return False

    def _update_sweep_tracking(self, high: float, low: float, bar_idx: int) -> None:
        """Track whether current bar sweeps a swing level."""
        swept_low = False
        swept_high = False
        if not math.isnan(self._swing_low_price) and low < self._swing_low_price:
            swept_low = True
        if not math.isnan(self._swing_high_price) and high > self._swing_high_price:
            swept_high = True
        self._swept_low_buffer.append((bar_idx, swept_low))
        self._swept_high_buffer.append((bar_idx, swept_high))

    # =======================================================================
    # 10. Signal Quality
    # =======================================================================

    def _compute_signal_quality(self, signal: dict, entry_price: float,
                                 stop: float, atr: float,
                                 close: float, open_: float,
                                 high: float, low: float) -> float:
        """Compute composite signal quality score."""
        if atr <= 0:
            return 0.5

        # Size score: entry-stop distance / (ATR * 1.5)
        gap = abs(entry_price - stop)
        size_sc = min(1.0, gap / (atr * 1.5)) if atr > 0 else 0.5

        # Displacement score: signal candle body/range
        bar_range = high - low
        body = abs(close - open_)
        disp_sc = body / bar_range if bar_range > 0 else 0.0

        # Fluency score
        flu_val = signal.get('fluency', 0.5)
        flu_sc = min(1.0, max(0.0, flu_val)) if not math.isnan(flu_val) else 0.5

        # PA cleanliness: 1 - alternating direction ratio
        pa_sc = self._compute_pa_score()

        return (self._sq_w_size * size_sc +
                self._sq_w_disp * disp_sc +
                self._sq_w_flu * flu_sc +
                self._sq_w_pa * pa_sc)

    def _compute_pa_score(self) -> float:
        """Compute PA cleanliness score from last 6 bars."""
        window = 6
        if len(self._candle_buffer) < window:
            return 0.5

        buf = list(self._candle_buffer)
        recent = buf[-window:]
        dirs = []
        for bar in recent:
            d = 1 if bar['close'] > bar['open'] else (-1 if bar['close'] < bar['open'] else 0)
            dirs.append(d)

        alternations = 0
        for k in range(1, len(dirs)):
            if dirs[k] != dirs[k - 1]:
                alternations += 1

        alt_ratio = alternations / (window - 1)
        return 1.0 - alt_ratio

    def _compute_alt_dir_ratio(self) -> float:
        """Compute alternating direction ratio for PA quality filter."""
        window = 6
        if len(self._candle_buffer) < window:
            return 0.0

        buf = list(self._candle_buffer)
        recent = buf[-window:]
        dirs = []
        for bar in recent:
            d = 1 if bar['close'] > bar['open'] else (-1 if bar['close'] < bar['open'] else 0)
            dirs.append(d)

        alternations = 0
        for k in range(1, len(dirs)):
            if dirs[k] != dirs[k - 1]:
                alternations += 1

        return alternations / (window - 1)

    # =======================================================================
    # 11. HTF Bias (1H/4H)
    # =======================================================================

    def on_htf_bar(self, timeframe: str, bar: dict) -> None:
        """Called when a 1H or 4H bar completes.

        Detects HTF FVGs, updates HTF FVG states, computes draw direction.
        """
        if timeframe == '1H':
            cbuf = self._htf_1h_candle_buffer
            fvgs = self._htf_1h_fvgs
        elif timeframe == '4H':
            cbuf = self._htf_4h_candle_buffer
            fvgs = self._htf_4h_fvgs
        else:
            return

        cbuf.append(bar)

        # Detect HTF FVGs
        if len(cbuf) >= 3:
            buf = list(cbuf)
            c1, c2, c3 = buf[-3], buf[-2], buf[-1]

            # Bullish FVG
            if c1['high'] < c3['low']:
                gap_top = c3['low']
                gap_bot = c1['high']
                fvgs.append(_HTF_FVG(
                    idx=self._bar_idx, direction="bull",
                    top=gap_top, bottom=gap_bot,
                    size=gap_top - gap_bot,
                ))

            # Bearish FVG
            if c1['low'] > c3['high']:
                gap_top = c1['low']
                gap_bot = c3['high']
                fvgs.append(_HTF_FVG(
                    idx=self._bar_idx, direction="bear",
                    top=gap_top, bottom=gap_bot,
                    size=gap_top - gap_bot,
                ))

        # Update HTF FVG states
        close = bar['close']
        high = bar['high']
        low = bar['low']

        surviving: list[_HTF_FVG] = []
        for rec in fvgs:
            old_status = rec.status
            # Simplified update (same logic as LTF)
            if rec.direction == "bull":
                if close < rec.bottom:
                    rec.status = "invalidated"
                elif low <= rec.top:
                    if rec.status == "untested":
                        rec.status = "tested_rejected"
            else:  # bear
                if close > rec.top:
                    rec.status = "invalidated"
                elif high >= rec.bottom:
                    if rec.status == "untested":
                        rec.status = "tested_rejected"

            if rec.status != "invalidated":
                surviving.append(rec)

        if timeframe == '1H':
            self._htf_1h_fvgs = surviving
        else:
            self._htf_4h_fvgs = surviving

        # Prune old HTF FVGs (100 bars on HTF = ~400-1600 5m bars)
        max_age = 100
        cutoff = self._bar_idx - max_age * (12 if timeframe == '1H' else 48)
        if timeframe == '1H':
            self._htf_1h_fvgs = [r for r in self._htf_1h_fvgs if r.idx > cutoff]
        else:
            self._htf_4h_fvgs = [r for r in self._htf_4h_fvgs if r.idx > cutoff]

        # Compute draw direction
        self._recompute_htf_bias(close)

    def _recompute_htf_bias(self, current_close: float) -> None:
        """Recompute HTF bias from active FVGs."""
        # 4H draw
        bull_above_4h = sum(1 for r in self._htf_4h_fvgs
                            if r.direction == "bull" and (r.top + r.bottom) / 2 > current_close)
        bear_below_4h = sum(1 for r in self._htf_4h_fvgs
                            if r.direction == "bear" and (r.top + r.bottom) / 2 < current_close)

        if bull_above_4h > 0 and bear_below_4h == 0:
            bias_4h = 1.0
        elif bear_below_4h > 0 and bull_above_4h == 0:
            bias_4h = -1.0
        elif bull_above_4h > 0 and bear_below_4h > 0:
            net = bull_above_4h - bear_below_4h
            bias_4h = 0.5 if net > 0 else (-0.5 if net < 0 else 0.0)
        else:
            bias_4h = 0.0

        # 1H draw
        bull_above_1h = sum(1 for r in self._htf_1h_fvgs
                            if r.direction == "bull" and (r.top + r.bottom) / 2 > current_close)
        bear_below_1h = sum(1 for r in self._htf_1h_fvgs
                            if r.direction == "bear" and (r.top + r.bottom) / 2 < current_close)

        if bull_above_1h > 0 and bear_below_1h == 0:
            bias_1h = 1.0
        elif bear_below_1h > 0 and bull_above_1h == 0:
            bias_1h = -1.0
        elif bull_above_1h > 0 and bear_below_1h > 0:
            net = bull_above_1h - bear_below_1h
            bias_1h = 0.5 if net > 0 else (-0.5 if net < 0 else 0.0)
        else:
            bias_1h = 0.0

        # Weighted composite
        self._htf_bias_4h = bias_4h
        self._htf_bias_1h = bias_1h
        self._htf_pda_count = (len(self._htf_4h_fvgs) + len(self._htf_1h_fvgs))

    # =======================================================================
    # 12. Composite Bias
    # =======================================================================

    def _get_composite_bias(self, close: float) -> tuple[float, float, float]:
        """Combine HTF bias + overnight bias + ORM bias.

        Returns: (bias_direction, bias_confidence, regime)
        """
        htf_bias = 0.6 * self._htf_bias_4h + 0.4 * self._htf_bias_1h

        # Overnight bias: position of NY open in overnight range
        overnight_bias = 0.0
        if (not math.isnan(self._ny_open_price) and
            not math.isnan(self._overnight_high) and
            not math.isnan(self._overnight_low)):
            on_range = self._overnight_high - self._overnight_low
            if on_range > 0:
                position = (self._ny_open_price - self._overnight_low) / on_range
                if position > 0.6:
                    overnight_bias = 1.0
                elif position < 0.4:
                    overnight_bias = -1.0

        orm_bias = self._orm_bias

        composite = 0.4 * htf_bias + 0.3 * overnight_bias + 0.3 * orm_bias
        bias_dir = 0.0
        if composite > 0.2:
            bias_dir = 1.0
        elif composite < -0.2:
            bias_dir = -1.0

        confidence = abs(composite)

        # Regime
        if self._htf_pda_count > 0 and abs(htf_bias) > 0.2:
            regime = 1.0
        elif self._htf_pda_count > 0:
            regime = 0.5
        else:
            regime = 0.0

        return bias_dir, confidence, regime

    # =======================================================================
    # 13. Grading
    # =======================================================================

    def _compute_grade(self, signal_dir: int, bias_dir: float, regime: float) -> str:
        """Grade the setup: A+ / B+ / C."""
        if math.isnan(regime):
            return "C"
        if regime == 0.0:
            return "C"

        aligned = (signal_dir == int(bias_dir) and bias_dir != 0)
        full_regime = regime >= 1.0

        if aligned and full_regime:
            return "A+"
        elif aligned or full_regime:
            return "B+"
        else:
            return "C"

    # =======================================================================
    # 14. Filters
    # =======================================================================

    def _passes_filters(self, signal: dict, atr: float, session: str,
                         bar_time_et: datetime, close: float,
                         high: float, low: float, open_: float) -> bool:
        """Apply all entry filters. Returns True if signal passes."""
        direction = signal['direction']
        sig_type = signal['signal_type']

        # Get bias
        bias_dir, bias_conf, regime = self._get_composite_bias(close)
        self._bias_direction = bias_dir
        self._bias_confidence = bias_conf
        self._regime = regime

        # 1. Bias opposing block (exception: MSS+SMT)
        bias_opposing = (direction == -int(bias_dir) and bias_dir != 0)
        if bias_opposing:
            is_mss_smt = (self._smt_enabled and
                          (self._smt_bull if direction == 1 else self._smt_bear) and
                          sig_type == "mss")
            if is_mss_smt:
                pass  # exempt
            elif self._bias_relax_enabled and direction == -1:
                pass  # opposing shorts relaxation
            else:
                self._diag["blocked_by_bias"] += 1
                return False

        # 2. PA quality filter
        alt_dir = self._compute_alt_dir_ratio()
        if alt_dir >= self._pa_threshold:
            self._diag["blocked_by_pa_quality"] += 1
            return False

        # 3. Session filter
        et_frac = bar_time_et.hour + bar_time_et.minute / 60.0

        is_mss_smt = (sig_type == "mss" and
                      self._smt_enabled and
                      (self._smt_bull if direction == 1 else self._smt_bear))
        mss_bypass = is_mss_smt and self._smt_bypass_session

        if self._sf_enabled and not mss_bypass:
            if 9.5 <= et_frac < 16.0:
                if self._ny_direction != 0 and direction != self._ny_direction:
                    self._diag["blocked_by_session"] += 1
                    return False
            elif 3.0 <= et_frac < 9.5:
                if self._skip_london:
                    self._diag["blocked_by_session"] += 1
                    return False
                if self._london_direction != 0 and direction != self._london_direction:
                    self._diag["blocked_by_session"] += 1
                    return False
            else:
                if self._skip_asia:
                    self._diag["blocked_by_session"] += 1
                    return False
        elif not self._sf_enabled and not mss_bypass:
            if not (3.0 <= et_frac < 16.0):
                self._diag["blocked_by_session"] += 1
                return False

        # 4. Compute entry price and stop
        entry_p = signal.get('entry_price', close)
        c2_open = signal.get('fvg_c2_open', float('nan'))

        # Model stop
        if not math.isnan(c2_open) and c2_open > 0:
            primary_stop = c2_open
        else:
            if direction == 1:
                primary_stop = min(open_, low)
            else:
                primary_stop = max(open_, high)

        bar_range = high - low
        small_candle_mult = self._small_candle_atr_mult
        if atr > 0 and bar_range < small_candle_mult * atr:
            if direction == 1:
                fallback = self._swing_low_price if not math.isnan(self._swing_low_price) else primary_stop
                stop = min(primary_stop, fallback)
            else:
                fallback = self._swing_high_price if not math.isnan(self._swing_high_price) else primary_stop
                stop = max(primary_stop, fallback)
        else:
            stop = primary_stop

        # IRL target
        risk = abs(close - stop)
        if direction == 1:
            target = self._swing_high_price if not math.isnan(self._swing_high_price) else float('nan')
            if not math.isnan(target) and target <= close:
                target = close + risk * 2.0 if risk > 0 else float('nan')
        else:
            target = self._swing_low_price if not math.isnan(self._swing_low_price) else float('nan')
            if not math.isnan(target) and target >= close:
                target = close - risk * 2.0 if risk > 0 else float('nan')

        tp1 = target

        if math.isnan(entry_p) or math.isnan(stop) or math.isnan(tp1):
            self._diag["blocked_by_invalid_prices"] += 1
            return False

        # Validate stop/target
        if direction == 1 and (stop >= entry_p or tp1 <= entry_p):
            self._diag["blocked_by_stop_target_logic"] += 1
            return False
        if direction == -1 and (stop <= entry_p or tp1 >= entry_p):
            self._diag["blocked_by_stop_target_logic"] += 1
            return False

        # 5. Min stop distance
        stop_dist = abs(entry_p - stop)
        min_stop = self._min_stop_atr_mult * atr if atr > 0 else 10.0
        if stop_dist < min_stop:
            self._diag["blocked_by_min_stop"] += 1
            return False

        # 6. Signal quality threshold
        if self._sq_enabled:
            sq = self._compute_signal_quality(signal, entry_p, stop, atr, close, open_, high, low)
            eff_threshold = self._sq_threshold
            if self._dual_mode_enabled and direction == -1:
                eff_threshold = self._short_sq_threshold
            if sq < eff_threshold:
                self._diag["blocked_by_sq"] += 1
                return False

        # 7. News blackout
        if self._is_in_news_blackout(bar_time_et):
            self._diag["blocked_by_news_filter"] += 1
            return False

        # 8. Daily limits
        if self._day_stopped:
            self._diag["blocked_by_daily_limits"] += 1
            return False

        # 9. One position at a time
        if self._in_position:
            self._diag["blocked_by_position"] += 1
            return False

        # 10. Lunch dead zone (session_regime with mult=0)
        if self._session_regime_enabled:
            if self._sr_lunch_start <= et_frac < self._sr_lunch_end and self._sr_lunch_mult == 0.0:
                self._diag["blocked_by_lunch_deadzone"] += 1
                return False

        # Store computed values for entry
        signal['_stop'] = stop
        signal['_tp1'] = tp1
        signal['_entry_price'] = entry_p
        signal['_bias_dir'] = bias_dir
        signal['_bias_conf'] = bias_conf
        signal['_regime'] = regime

        return True

    def _passes_filters_cached(self, signal: dict, atr: float, session: str,
                                bar_time_et: datetime, close: float,
                                high: float, low: float, open_: float,
                                bar_idx: int) -> bool:
        """Apply all entry filters using cache-provided bias/entry/stop/target.

        This matches validate_nt_logic.py filter chain exactly, using cached
        bias/regime and signal-provided entry_price/model_stop/irl_target.
        """
        direction = signal['direction']
        sig_type = signal['signal_type']

        entry_p = signal['_cache_entry_price']
        stop = signal['_cache_model_stop']
        tp1 = signal['_cache_irl_target']
        has_smt = signal.get('has_smt', False)

        # Get bias from cache
        cache_bias = self._cache_bias.get(bar_idx, (0.0, 0.0, 0.0))
        bias_dir = cache_bias[0]
        bias_conf = cache_bias[1]
        regime = cache_bias[2]

        self._bias_direction = bias_dir
        self._bias_confidence = bias_conf
        self._regime = regime

        # Validate basic price structure
        if math.isnan(entry_p) or math.isnan(stop) or math.isnan(tp1):
            self._diag["blocked_by_invalid_prices"] += 1
            return False

        if direction == 1 and (stop >= entry_p or tp1 <= entry_p):
            self._diag["blocked_by_stop_target_logic"] += 1
            return False
        if direction == -1 and (stop <= entry_p or tp1 >= entry_p):
            self._diag["blocked_by_stop_target_logic"] += 1
            return False

        # Signal type filter
        if not self._allow_mss and sig_type == "mss":
            return False
        if not self._allow_trend and sig_type == "trend":
            return False

        # 1. Bias opposing block (exception: MSS+SMT)
        # Match reference: direction == -np.sign(bias_dir) and bias_dir != 0
        bias_sign = 1.0 if bias_dir > 0 else (-1.0 if bias_dir < 0 else 0.0)
        bias_opposing = (direction == -bias_sign and bias_dir != 0)
        if bias_opposing:
            is_mss_smt = (self._smt_enabled and has_smt and sig_type == "mss")
            if is_mss_smt:
                pass  # exempt
            elif self._bias_relax_enabled and direction == -1:
                pass  # opposing shorts relaxation
            else:
                self._diag["blocked_by_bias"] += 1
                return False

        # 2. PA quality filter
        alt_dir = self._compute_alt_dir_ratio()
        if alt_dir >= self._pa_threshold:
            self._diag["blocked_by_pa_quality"] += 1
            return False

        # 3. Session filter
        et_frac = bar_time_et.hour + bar_time_et.minute / 60.0

        is_mss_smt = (sig_type == "mss" and has_smt and self._smt_enabled)
        mss_bypass = is_mss_smt and self._smt_bypass_session

        if self._sf_enabled and not mss_bypass:
            if 9.5 <= et_frac < 16.0:
                if self._ny_direction != 0 and direction != self._ny_direction:
                    self._diag["blocked_by_session"] += 1
                    return False
            elif 3.0 <= et_frac < 9.5:
                if self._skip_london:
                    self._diag["blocked_by_session"] += 1
                    return False
                if self._london_direction != 0 and direction != self._london_direction:
                    self._diag["blocked_by_session"] += 1
                    return False
            else:
                if self._skip_asia:
                    self._diag["blocked_by_session"] += 1
                    return False
        elif not self._sf_enabled and not mss_bypass:
            if not (3.0 <= et_frac < 16.0):
                self._diag["blocked_by_session"] += 1
                return False

        # 4. Min stop distance (use cached ATR for exact match)
        stop_dist = abs(entry_p - stop)
        cached_atr_val = self._cache_atr.get(bar_idx, None)
        effective_atr = cached_atr_val if cached_atr_val is not None and not math.isnan(cached_atr_val) else (atr if atr > 0 else 10.0)
        min_stop = self._min_stop_atr_mult * effective_atr if effective_atr > 0 else 10.0
        if stop_dist < min_stop:
            self._diag["blocked_by_min_stop"] += 1
            return False

        # 5. Signal quality threshold
        if self._sq_enabled:
            # Use cached SQ if available, otherwise compute
            cached_sq = self._cache_sq.get(bar_idx, None)
            if cached_sq is not None and not math.isnan(cached_sq):
                sq = cached_sq
            else:
                # Compute SQ same as validate_nt_logic.py using cached arrays
                cached_atr = self._cache_atr.get(bar_idx, None)
                a = cached_atr if cached_atr is not None and not math.isnan(cached_atr) else (atr if atr > 0 else 10.0)
                gap = abs(entry_p - stop)
                size_sc = min(1.0, gap / (a * 1.5)) if a > 0 else 0.5
                body = abs(close - open_)
                bar_range = high - low
                disp_sc = body / bar_range if bar_range > 0 else 0.0
                cached_flu = self._cache_fluency.get(bar_idx, None)
                flu_val = cached_flu if cached_flu is not None else self._compute_fluency()
                flu_sc = min(1.0, max(0.0, flu_val)) if not math.isnan(flu_val) else 0.5
                # PA from cache
                window = 6
                buf = list(self._candle_buffer)
                if len(buf) >= window:
                    recent = buf[-window:]
                    dirs = []
                    for bar in recent:
                        d = 1 if bar['close'] > bar['open'] else (-1 if bar['close'] < bar['open'] else 0)
                        dirs.append(d)
                    alternations = sum(1 for k in range(1, len(dirs)) if dirs[k] != dirs[k-1])
                    pa_sc = 1.0 - alternations / (window - 1)
                else:
                    pa_sc = 0.5
                sq = (self._sq_w_size * size_sc + self._sq_w_disp * disp_sc +
                      self._sq_w_flu * flu_sc + self._sq_w_pa * pa_sc)

            eff_threshold = self._sq_threshold
            if self._dual_mode_enabled and direction == -1:
                eff_threshold = self._short_sq_threshold
            if sq < eff_threshold:
                self._diag["blocked_by_sq"] += 1
                return False

        # 6. News blackout
        if self._is_in_news_blackout(bar_time_et):
            self._diag["blocked_by_news_filter"] += 1
            return False

        # 7. Daily limits
        if self._day_stopped:
            self._diag["blocked_by_daily_limits"] += 1
            return False

        # 8. One position at a time
        if self._in_position:
            self._diag["blocked_by_position"] += 1
            return False

        # 9. Lunch dead zone
        if self._session_regime_enabled:
            if self._sr_lunch_start <= et_frac < self._sr_lunch_end and self._sr_lunch_mult == 0.0:
                self._diag["blocked_by_lunch_deadzone"] += 1
                return False

        # Store computed values for entry
        signal['_stop'] = stop
        signal['_tp1'] = tp1
        signal['_entry_price'] = entry_p
        signal['_bias_dir'] = bias_dir
        signal['_bias_conf'] = bias_conf
        signal['_regime'] = regime

        return True

    def _is_in_news_blackout(self, time_et: datetime) -> bool:
        """Check if current time is in a news blackout window."""
        for start, end in self._news_blackout_times:
            if start <= time_et <= end:
                return True
        return False

    # =======================================================================
    # 15. Trade Management
    # =======================================================================

    def _manage_position(self, bar: dict) -> dict | None:
        """Manage open position. Returns trade dict if closed, else None.

        Matches validate_nt_logic.py exit logic exactly:
          - Early cut PA: bars_in_trade 2-4, avg_wick > 0.65 + favorable < 0.5 + no_progress
          - Stop/TP/trail: standard Lanto trade management
          - R calculation: same formula as engine.py
        """
        if not self._in_position:
            return None

        i = self._bar_idx
        high = bar['high']
        low = bar['low']
        close = bar['close']
        open_ = bar['open']
        atr = self._get_atr()

        exited = False
        exit_reason = ""
        exit_price = 0.0
        exit_contracts = self._pos_remaining_contracts

        # Early cut PA check (bars 2-4 after entry)
        bars_in_trade = i - self._pos_entry_idx
        if not self._pos_trimmed and 2 <= bars_in_trade <= 4:
            # Compute PA quality from entry bar to current bar (inclusive)
            # Use candle buffer to get bars in range [pos_entry_idx, i]
            pa_bars = []
            buf = list(self._candle_buffer)
            for b in buf:
                b_idx = b.get('_bar_idx', -1)
                if b_idx >= self._pos_entry_idx and b_idx <= i:
                    pa_bars.append(b)

            if len(pa_bars) >= 2:
                wick_sum = 0.0
                favorable_count = 0
                total_count = len(pa_bars)
                for pb in pa_bars:
                    pb_range = pb['high'] - pb['low']
                    pb_body = abs(pb['close'] - pb['open'])
                    safe_r = pb_range if pb_range > 0 else 1.0
                    wick_sum += 1.0 - (pb_body / safe_r)
                    pb_dir = 1 if pb['close'] > pb['open'] else (-1 if pb['close'] < pb['open'] else 0)
                    if pb_dir == self._pos_direction:
                        favorable_count += 1

                avg_wick = wick_sum / total_count
                favorable = favorable_count / total_count

                if self._pos_direction == 1:
                    disp = close - self._pos_entry_price
                else:
                    disp = self._pos_entry_price - close

                cur_atr = atr if atr > 0 else 30.0
                no_progress = disp < cur_atr * 0.3
                bad_pa = avg_wick > 0.65 and favorable < 0.5

                if bad_pa and no_progress and bars_in_trade >= 3:
                    # Reference: exit_price = o[i+1] if i+1 < n else c[i]
                    # We don't have next bar yet. Store as pending early cut.
                    # The exit will be processed on the next bar.
                    # Skip stop/TP checks on this bar (matches reference: exited=True)
                    self._pending_early_cut = True
                    return None  # Don't process stop/TP; exit deferred to next bar open

        # LONG position management
        if not exited and self._pos_direction == 1:
            eff_stop = self._pos_trail_stop if self._pos_trimmed and self._pos_trail_stop > 0 else self._pos_stop
            if self._pos_trimmed and self._be_after_trim and self._pos_be_stop > 0:
                eff_stop = max(eff_stop, self._pos_be_stop)

            if low <= eff_stop:
                exit_price = eff_stop - self._slippage_points
                if self._pos_trimmed and eff_stop >= self._pos_entry_price:
                    exit_reason = "be_sweep"
                else:
                    exit_reason = "stop"
                exited = True

            elif not self._pos_trimmed and high >= self._pos_tp1:
                tc = max(1, int(self._pos_contracts * self._pos_trim_pct))
                self._pos_remaining_contracts = self._pos_contracts - tc
                self._pos_trimmed = True
                self._pos_be_stop = self._pos_entry_price

                if self._pos_remaining_contracts > 0:
                    self._pos_trail_stop = self._find_nth_swing_price(1, self._nth_swing)
                    if math.isnan(self._pos_trail_stop) or self._pos_trail_stop <= 0:
                        self._pos_trail_stop = self._pos_be_stop

                if self._pos_remaining_contracts <= 0:
                    exit_price = self._pos_tp1
                    exit_reason = "tp1"
                    exit_contracts = self._pos_contracts
                    exited = True

            if self._pos_trimmed and not exited:
                nt = self._find_nth_swing_price(1, self._nth_swing)
                if not math.isnan(nt) and nt > self._pos_trail_stop:
                    self._pos_trail_stop = nt

        # SHORT position management
        elif not exited and self._pos_direction == -1:
            eff_stop = self._pos_trail_stop if self._pos_trimmed and self._pos_trail_stop > 0 else self._pos_stop
            if self._pos_trimmed and self._be_after_trim and self._pos_be_stop > 0:
                eff_stop = min(eff_stop, self._pos_be_stop)

            if high >= eff_stop:
                exit_price = eff_stop + self._slippage_points
                if self._pos_trimmed and eff_stop <= self._pos_entry_price:
                    exit_reason = "be_sweep"
                else:
                    exit_reason = "stop"
                exited = True

            elif not self._pos_trimmed and low <= self._pos_tp1:
                tc = max(1, int(self._pos_contracts * self._pos_trim_pct))
                self._pos_remaining_contracts = self._pos_contracts - tc
                self._pos_trimmed = True
                self._pos_be_stop = self._pos_entry_price

                if self._pos_remaining_contracts > 0:
                    self._pos_trail_stop = self._find_nth_swing_price(-1, self._nth_swing)
                    if math.isnan(self._pos_trail_stop) or self._pos_trail_stop <= 0:
                        self._pos_trail_stop = self._pos_be_stop

                if self._pos_remaining_contracts <= 0:
                    exit_price = self._pos_tp1
                    exit_reason = "tp1"
                    exit_contracts = self._pos_contracts
                    exited = True

            if self._pos_trimmed and not exited:
                nt = self._find_nth_swing_price(-1, self._nth_swing)
                if not math.isnan(nt) and nt < self._pos_trail_stop:
                    self._pos_trail_stop = nt

        if exited:
            return self._close_position(exit_price, exit_reason, exit_contracts, bar)

        return None

    def _close_position(self, exit_price: float, exit_reason: str,
                         exit_contracts: int, bar: dict) -> dict:
        """Close position and compute PnL."""
        if self._pos_direction == 1:
            pnl_pts = exit_price - self._pos_entry_price
        else:
            pnl_pts = self._pos_entry_price - exit_price

        if self._pos_trimmed and exit_reason != "tp1":
            trim_pnl_total = (self._pos_tp1 - self._pos_entry_price) if self._pos_direction == 1 \
                else (self._pos_entry_price - self._pos_tp1)
            trim_c = self._pos_contracts - exit_contracts
            total_pnl = (trim_pnl_total * self._point_value * trim_c +
                         pnl_pts * self._point_value * exit_contracts)
            total_comm = self._commission_per_side * 2 * self._pos_contracts
            total_pnl -= total_comm
        else:
            total_pnl = pnl_pts * self._point_value * exit_contracts
            total_comm = self._commission_per_side * 2 * exit_contracts
            total_pnl -= total_comm

        stop_dist = abs(self._pos_entry_price - self._pos_stop)
        total_risk = stop_dist * self._point_value * self._pos_contracts
        r_mult = total_pnl / total_risk if total_risk > 0 else 0.0

        trade = {
            'entry_time': self._pos_entry_time if self._pos_entry_time is not None else bar['time'],
            'exit_time': bar['time'],
            'r': r_mult,
            'reason': exit_reason,
            'dir': self._pos_direction,
            'type': self._pos_signal_type,
            'trimmed': self._pos_trimmed,
            'grade': self._pos_grade,
            'entry_price': self._pos_entry_price,
            'exit_price': exit_price,
            'stop_price': self._pos_stop,
            'tp1_price': self._pos_tp1,
            'contracts': self._pos_contracts,
        }

        # Update daily state
        self._daily_pnl_r += r_mult

        if exit_reason == "be_sweep" and self._pos_trimmed:
            pass  # Not a loss
        elif r_mult < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        if self._consecutive_losses >= self._max_consec_losses:
            self._day_stopped = True
        if self._daily_pnl_r <= -self._daily_max_loss_r:
            self._day_stopped = True

        self._in_position = False
        self._completed_trades.append(trade)

        return trade

    def _enter_position(self, signal: dict, bar: dict, next_bar_open: float | None = None) -> None:
        """Enter a new position based on the signal."""
        direction = signal['direction']
        entry_p = signal['_entry_price']
        stop = signal['_stop']
        tp1 = signal['_tp1']
        bias_dir = signal['_bias_dir']
        regime = signal['_regime']
        sig_type = signal['signal_type']
        is_mss = sig_type == 'mss'

        bar_time_et = bar['time_et']
        et_frac = bar_time_et.hour + bar_time_et.minute / 60.0
        atr = self._get_atr()

        # Grade
        ba = 1.0 if (direction == int(bias_dir) and bias_dir != 0) else 0.0
        grade = self._compute_grade(direction, bias_dir, regime)

        if grade == "C" and self._c_skip:
            return

        # Position sizing
        dow = bar_time_et.weekday()
        is_reduced = (dow in (0, 4)) or (regime < 1.0)
        base_r = self._reduced_r if is_reduced else self._normal_r

        if grade == "A+":
            r_amount = base_r * self._a_plus_mult
        elif grade == "B+":
            r_amount = base_r * self._b_plus_mult
        else:
            r_amount = base_r * 0.5

        # Session regime sizing
        if self._session_regime_enabled:
            if et_frac < self._sr_am_end:
                sr_mult = self._sr_am_mult
            elif self._sr_lunch_start <= et_frac < self._sr_lunch_end:
                sr_mult = self._sr_lunch_mult
            elif et_frac >= self._sr_pm_start:
                sr_mult = self._sr_pm_mult
            else:
                sr_mult = 1.0
            r_amount *= sr_mult
            if r_amount <= 0:
                return

        # Apply slippage to entry
        if direction == 1:
            actual_entry = entry_p + self._slippage_points
        else:
            actual_entry = entry_p - self._slippage_points

        stop_dist = abs(actual_entry - stop)
        if stop_dist < 1.0:
            return

        contracts = self._compute_contracts(r_amount, stop_dist)
        if contracts <= 0:
            return

        # TP adjustments
        if self._session_rules_enabled:
            if self._dir_mgmt_enabled:
                actual_tp_mult = self._long_tp_mult if direction == 1 else self._short_tp_mult
            else:
                actual_tp_mult = self._ny_tp_mult

            if self._mss_mgmt_enabled and is_mss and direction == 1:
                actual_tp_mult = self._mss_long_tp_mult

            if 9.5 <= et_frac < 16.0:
                tp_distance = (tp1 - actual_entry) if direction == 1 else (actual_entry - tp1)
                tp1 = (actual_entry + tp_distance * actual_tp_mult) if direction == 1 else (actual_entry - tp_distance * actual_tp_mult)

        # Dual mode short TP override
        if self._dual_mode_enabled and direction == -1:
            short_rr = self._short_rr
            if self._mss_mgmt_enabled and is_mss:
                short_rr = self._mss_short_rr
            tp1 = actual_entry - stop_dist * short_rr

        # Trim percentage
        if self._mss_mgmt_enabled and is_mss:
            trim_pct = self._mss_short_trim_pct if direction == -1 else self._mss_long_trim_pct
        elif self._dual_mode_enabled and direction == -1:
            trim_pct = self._dual_short_trim_pct
        elif self._dir_mgmt_enabled:
            trim_pct = self._long_trim_pct if direction == 1 else self._short_trim_pct
        else:
            trim_pct = self._trim_pct

        # Set position state
        self._in_position = True
        self._pos_direction = direction
        self._pos_entry_idx = self._bar_idx + 1  # entry on next bar
        self._pos_entry_price = actual_entry
        self._pos_entry_time = bar.get('time', bar.get('_entry_time', None))
        self._pos_stop = stop
        self._pos_tp1 = tp1
        self._pos_contracts = contracts
        self._pos_remaining_contracts = contracts
        self._pos_trimmed = False
        self._pos_be_stop = 0.0
        self._pos_trail_stop = 0.0
        self._pos_signal_type = sig_type
        self._pos_grade = grade
        self._pos_trim_pct = trim_pct
        self._pos_bias_dir = bias_dir
        self._pos_regime = regime

    def _compute_contracts(self, r_amount: float, stop_dist: float) -> int:
        """Compute number of contracts."""
        if stop_dist <= 0 or self._point_value <= 0:
            return 0
        return max(1, int(r_amount / (stop_dist * self._point_value)))

    # =======================================================================
    # 16. SMT Divergence
    # =======================================================================

    def on_es_bar(self, bar: dict) -> None:
        """Process ES 5m bar for SMT divergence."""
        bar_idx = self._bar_idx  # Use same bar index as NQ

        self._es_high_buffer.append((bar_idx, bar['high']))
        self._es_low_buffer.append((bar_idx, bar['low']))

        buf_needed = self._swing_left + self._swing_right + 1
        if len(self._es_high_buffer) >= buf_needed:
            self._check_swing_at(
                self._es_high_buffer, self._es_low_buffer,
                self._swing_left, self._swing_right,
                self._es_swing_highs, self._es_swing_lows,
            )

        if self._es_swing_highs:
            self._es_swing_high_price = self._es_swing_highs[-1].price
        if self._es_swing_lows:
            self._es_swing_low_price = self._es_swing_lows[-1].price

        # Detect SMT divergence using rolling sweep comparison
        lookback = self._smt_sweep_lookback

        # NQ swept low but ES didn't → bullish SMT
        nq_swept_low = False
        es_swept_low = False

        if not math.isnan(self._swing_low_price):
            # Check NQ sweeps over lookback
            for idx, swept in list(self._swept_low_buffer)[-lookback:]:
                if swept:
                    nq_swept_low = True
                    break

        if not math.isnan(self._es_swing_low_price):
            # Check ES: did it sweep its swing low?
            for k in range(max(0, len(self._es_low_buffer) - lookback), len(self._es_low_buffer)):
                es_idx, es_low = list(self._es_low_buffer)[k]
                if es_low < self._es_swing_low_price:
                    es_swept_low = True
                    break

        self._smt_bull = nq_swept_low and not es_swept_low

        # NQ swept high but ES didn't → bearish SMT
        nq_swept_high = False
        es_swept_high = False

        if not math.isnan(self._swing_high_price):
            for idx, swept in list(self._swept_high_buffer)[-lookback:]:
                if swept:
                    nq_swept_high = True
                    break

        if not math.isnan(self._es_swing_high_price):
            for k in range(max(0, len(self._es_high_buffer) - lookback), len(self._es_high_buffer)):
                es_idx, es_high = list(self._es_high_buffer)[k]
                if es_high > self._es_swing_high_price:
                    es_swept_high = True
                    break

        self._smt_bear = nq_swept_high and not es_swept_high

    # =======================================================================
    # Main on_bar loop
    # =======================================================================

    _pending_early_cut: bool = False

    def on_bar(self, bar: dict) -> dict | None:
        """Process one 5m bar. Returns trade dict if a trade closed, else None.

        Timing matches validate_nt_logic.py exactly:
          - Bar i: exit management, then signal detection, then FVG state update
          - Signal on bar i -> entry at bar i+1 (pos_entry_idx = i+1)
          - Exit management starts at bar i+1 (pos_entry_idx)

        bar = {
            'time': datetime (UTC),
            'time_et': datetime (Eastern, naive),
            'open': float, 'high': float, 'low': float, 'close': float,
            'volume': float,
            'is_roll_date': bool,
        }
        """
        self._bar_idx += 1
        bar_idx = self._bar_idx
        bar['_bar_idx'] = bar_idx

        time_et = bar['time_et']
        open_ = bar['open']
        high = bar['high']
        low = bar['low']
        close = bar['close']

        # ---- New day reset ----
        bar_date = self._get_session_date(time_et)
        if bar_date != self._current_date:
            self._current_date = bar_date
            self._daily_pnl_r = 0.0
            self._consecutive_losses = 0
            self._day_stopped = False

        # ---- Handle pending early cut from PREVIOUS bar ----
        # Early cut PA decided on bar i-1, exits at bar i's open
        if self._pending_early_cut and self._in_position:
            self._pending_early_cut = False
            exit_price = open_
            trade_result = self._close_position(exit_price, "early_cut_pa",
                                                 self._pos_remaining_contracts, bar)
            # Return immediately — no new signals on an early cut bar
            # (The reference code uses `continue` for early cuts which skips entry)
            return trade_result
        self._pending_early_cut = False

        # ---- Process pending entry from PREVIOUS bar ----
        # Signal fired on bar i-1, entry happens at bar i's open
        # Must happen BEFORE exit management (matches validate_nt_logic.py flow:
        # entry sets pos_entry_idx = i+1, then at bar i+1 the exit loop runs)
        if self._pending_entry is not None and not self._in_position:
            signal = self._pending_entry
            self._pending_entry = None
            signal['_entry_price'] = open_
            direction = signal['direction']
            stop = signal['_stop']
            tp1 = signal['_tp1']
            if not (direction == 1 and (stop >= open_ or tp1 <= open_)):
                if not (direction == -1 and (stop <= open_ or tp1 >= open_)):
                    bar['_entry_time'] = bar['time']
                    self._enter_position(signal, bar)
                    if self._in_position:
                        self._pos_entry_time = bar['time']
                        self._diag["entries_executed"] += 1
        elif self._pending_entry is not None:
            self._pending_entry = None

        # ---- Update core indicators ----
        self._update_atr(high, low, close)
        atr = self._get_atr()

        # Add to candle buffer (BEFORE position management, so PA check has current bar)
        self._candle_buffer.append(bar)

        # Update session tracking
        session = self._update_session(time_et, high, low, open_)

        # Update swings (BEFORE position management for trail stop updates)
        self._update_swings(high, low, bar_idx)
        self._update_htf_swings(high, low, bar_idx)

        # Update sweep tracking
        self._update_sweep_tracking(high, low, bar_idx)

        # ---- Manage existing position (EXIT) ----
        trade_result = None
        if self._in_position:
            trade_result = self._manage_position(bar)

        # ---- Detect FVGs ----
        self._detect_fvg()

        # ---- Detect signals (ENTRY) ----
        signal = None

        # News blackout check — blocks new entries, not position management
        if not self._in_position and self._is_in_news_blackout(time_et):
            self._diag["blocked_by_news"] += 1
            self._update_all_fvg_states(high, low, close, bar_idx)
            return trade_result

        # ORM no-trade window (9:30-10:00 ET observation only)
        et_h = time_et.hour
        et_m = time_et.minute
        if not self._in_position and not self._day_stopped:
            if (et_h == 9 and et_m >= 30) or (et_h == 10 and et_m == 0):
                self._diag["blocked_by_orm"] += 1
                self._update_all_fvg_states(high, low, close, bar_idx)
                return trade_result

        # Check for signals
        if not self._in_position and not self._day_stopped:
            if self._use_cache:
                # ---- Cache mode: use pre-computed signals ----
                cached = self._cache_signals.get(bar_idx)
                if cached is not None:
                    sig_type = cached['signal_type']
                    direction = cached['direction']
                    if sig_type == 'trend':
                        self._diag["raw_trend_signals"] += 1
                    else:
                        self._diag["raw_mss_signals"] += 1
                    signal = {
                        'signal_type': sig_type,
                        'direction': direction,
                        'fvg_top': cached.get('fvg_top', 0.0),
                        'fvg_bottom': cached.get('fvg_bottom', 0.0),
                        'fvg_size': cached.get('fvg_size', 0.0),
                        'fvg_c2_open': cached.get('model_stop', 0.0),
                        'swept': cached.get('swept', False),
                        'sweep_score': cached.get('sweep_score', 0),
                        'is_displaced': False,
                        'fluency': 0.5,
                        'has_smt': cached.get('has_smt', False),
                        '_cache_entry_price': cached.get('entry_price', close),
                        '_cache_model_stop': cached.get('model_stop', 0.0),
                        '_cache_irl_target': cached.get('irl_target', 0.0),
                    }
            else:
                # ---- From-scratch signal detection ----
                # Try Trend signal first (priority)
                if self._allow_trend:
                    signal = self._check_trend_signal(bar_idx, close, high, low, open_, atr)
                    if signal is not None:
                        self._diag["raw_trend_signals"] += 1

                # Try MSS if no Trend signal
                if signal is None and self._allow_mss:
                    mss_signal = self._check_mss_signal(bar_idx, close, high, low, open_, atr)
                    if mss_signal is not None:
                        self._diag["raw_mss_signals"] += 1
                        # SMT gate for MSS
                        if self._smt_require_for_mss and self._smt_enabled:
                            mss_dir = mss_signal['direction']
                            if mss_dir == 1 and self._smt_bull:
                                signal = mss_signal
                            elif mss_dir == -1 and self._smt_bear:
                                signal = mss_signal
                            else:
                                self._diag["mss_blocked_smt_gate"] += 1
                        elif not self._smt_require_for_mss:
                            signal = mss_signal

                    # Kill MSS in overnight (16:00-03:00 ET)
                    if signal is not None and signal['signal_type'] == 'mss':
                        et_frac = time_et.hour + time_et.minute / 60.0
                        if et_frac >= 16.0 or et_frac < 3.0:
                            self._diag["mss_blocked_overnight"] += 1
                            signal = None

            # Apply filters and queue entry for next bar
            if signal is not None:
                self._diag["signals_entering_filters"] += 1
                signal['_entry_price'] = close  # placeholder; replaced by next bar open

                if self._use_cache:
                    # In cache mode, use _passes_filters_cached which uses cache-provided
                    # entry/stop/target and bias from cache
                    if self._passes_filters_cached(signal, atr, session, time_et,
                                                     close, high, low, open_, bar_idx):
                        self._diag["signals_passed_all_filters"] += 1
                        self._pending_entry = signal
                else:
                    if self._passes_filters(signal, atr, session, time_et, close, high, low, open_):
                        self._diag["signals_passed_all_filters"] += 1
                        self._pending_entry = signal
        elif self._in_position:
            # Count would-be signals blocked by position
            pass  # Position blocking counted implicitly
        elif self._day_stopped:
            self._diag["blocked_by_day_stopped"] += 1

        # ---- Update FVG states (AFTER signal detection — Fix #7 from audit) ----
        self._update_all_fvg_states(high, low, close, bar_idx)

        return trade_result

    def _get_session_date(self, time_et: datetime) -> object:
        """Get session date (bars from 18:00+ belong to next day's session)."""
        if time_et.hour >= 18:
            return (time_et + timedelta(days=1)).date()
        return time_et.date()

    def set_news_times(self, news_times: list[tuple[datetime, datetime]]) -> None:
        """Set news blackout windows. Each tuple is (start_et, end_et)."""
        self._news_blackout_times = news_times

    def print_filter_funnel(self) -> None:
        """Print diagnostic filter funnel."""
        d = self._diag
        print()
        print("=" * 70)
        print("FILTER FUNNEL DIAGNOSTICS")
        print("=" * 70)
        print(f"  Raw trend signals detected:     {d['raw_trend_signals']}")
        print(f"  Raw MSS signals detected:       {d['raw_mss_signals']}")
        raw_total = d['raw_trend_signals'] + d['raw_mss_signals']
        print(f"  --- Total raw signals:          {raw_total}")
        print()
        print(f"  MSS blocked by SMT gate:        {d['mss_blocked_smt_gate']}")
        print(f"  MSS blocked by overnight kill:  {d['mss_blocked_overnight']}")
        print()
        print(f"  Blocked by news (pre-signal):   {d['blocked_by_news']}")
        print(f"  Blocked by ORM (9:30-10:00):    {d['blocked_by_orm']}")
        print(f"  Blocked by day_stopped:         {d['blocked_by_day_stopped']}")
        print()
        print(f"  Signals entering filter chain:  {d['signals_entering_filters']}")
        print(f"    Blocked by bias:              {d['blocked_by_bias']}")
        print(f"    Blocked by PA quality:        {d['blocked_by_pa_quality']}")
        print(f"    Blocked by session:           {d['blocked_by_session']}")
        print(f"    Blocked by invalid prices:    {d['blocked_by_invalid_prices']}")
        print(f"    Blocked by stop/target logic: {d['blocked_by_stop_target_logic']}")
        print(f"    Blocked by min_stop:          {d['blocked_by_min_stop']}")
        print(f"    Blocked by signal quality:    {d['blocked_by_sq']}")
        print(f"    Blocked by news filter:       {d['blocked_by_news_filter']}")
        print(f"    Blocked by daily limits:      {d['blocked_by_daily_limits']}")
        print(f"    Blocked by position:          {d['blocked_by_position']}")
        print(f"    Blocked by lunch dead zone:   {d['blocked_by_lunch_deadzone']}")
        print(f"    Blocked by grade C skip:      {d['blocked_by_grade_c_skip']}")
        print()
        print(f"  Signals passed all filters:     {d['signals_passed_all_filters']}")
        print(f"  Entries actually executed:       {d['entries_executed']}")
        print(f"  Completed trades:               {len(self._completed_trades)}")
        print("=" * 70)

    def force_close(self, bar: dict) -> dict | None:
        """Force-close any open position at end of data."""
        if not self._in_position:
            return None
        return self._close_position(bar['close'], "eod_close",
                                     self._pos_remaining_contracts, bar)


# ===========================================================================
# Main loop
# ===========================================================================

if __name__ == "__main__":
    import time as _time

    import numpy as np
    import pandas as pd
    import pytz
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )

    sys.path.insert(0, str(PROJECT))

    # ---- Load params ----
    with open(PROJECT / "config" / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # ---- Load raw data ----
    print("Loading data...")
    nq_5m = pd.read_parquet(PROJECT / "data" / "NQ_5m_10yr.parquet")

    # Load HTF data
    htf_1h_path = PROJECT / "data" / "NQ_1H_10yr.parquet"
    htf_4h_path = PROJECT / "data" / "NQ_4H_10yr.parquet"
    nq_1h = pd.read_parquet(htf_1h_path) if htf_1h_path.exists() else None
    nq_4h = pd.read_parquet(htf_4h_path) if htf_4h_path.exists() else None

    # Load ES for SMT
    es_5m_path = PROJECT / "data" / "ES_5m_10yr.parquet"
    es_5m = pd.read_parquet(es_5m_path) if es_5m_path.exists() else None

    et_tz = pytz.timezone('US/Eastern')

    # ---- Build news blackout times ----
    news_times = []
    news_path = PROJECT / "config" / "news_calendar.csv"
    if news_path.exists():
        from features.news_filter import build_news_blackout_mask
        news_blackout = build_news_blackout_mask(
            nq_5m.index,
            str(news_path),
            params["news"]["blackout_minutes_before"],
            params["news"]["cooldown_minutes_after"],
        )
        # Convert to ET datetime tuples
        # For the bar-by-bar engine, we'll use the mask directly
        news_blackout_arr = news_blackout.values
    else:
        news_blackout_arr = None

    # ---- Build HTF bar boundaries ----
    # For each 5m bar, check if a 1H or 4H bar just completed
    htf_1h_set = set()
    htf_4h_set = set()
    htf_1h_bars = {}
    htf_4h_bars = {}

    if nq_1h is not None:
        for k in range(1, len(nq_1h)):
            # The 1H bar at index k-1 completed when bar k opens
            # Map the 1H bar completion to the nearest 5m bar
            ts = nq_1h.index[k]
            htf_1h_set.add(ts)
            htf_1h_bars[ts] = {
                'time': nq_1h.index[k - 1],
                'open': nq_1h['open'].iat[k - 1],
                'high': nq_1h['high'].iat[k - 1],
                'low': nq_1h['low'].iat[k - 1],
                'close': nq_1h['close'].iat[k - 1],
            }

    if nq_4h is not None:
        for k in range(1, len(nq_4h)):
            ts = nq_4h.index[k]
            htf_4h_set.add(ts)
            htf_4h_bars[ts] = {
                'time': nq_4h.index[k - 1],
                'open': nq_4h['open'].iat[k - 1],
                'high': nq_4h['high'].iat[k - 1],
                'low': nq_4h['low'].iat[k - 1],
                'close': nq_4h['close'].iat[k - 1],
            }

    # ---- Select test window ----
    # Full 10-year run for convergence testing
    full_run = "--full" in sys.argv
    if full_run:
        start = str(nq_5m.index[0].date())
        end = str(nq_5m.index[-1].date() + pd.Timedelta(days=1))
        full_range = nq_5m
        warmup_start = 0
        test_start_idx_override = 0  # no warmup separation, count all trades
        print(f"FULL 10-YEAR RUN: {len(full_range)} bars, {full_range.index[0]} to {full_range.index[-1]}")
    else:
        start = '2025-01-01'
        end = '2025-01-31'
        mask = (nq_5m.index >= start) & (nq_5m.index < end)
        test_nq = nq_5m[mask]
        print(f"Test window: {len(test_nq)} bars, {test_nq.index[0]} to {test_nq.index[-1]}")

        # Need substantial warmup bars (for ATR, swings, FVGs, bias to stabilize)
        warmup_bars = 5000  # ~3+ months of 5m bars
        start_idx = nq_5m.index.get_loc(test_nq.index[0])
        warmup_start = max(0, start_idx - warmup_bars)
        # End at test window end (not end of entire dataset)
        end_idx = nq_5m.index.get_loc(test_nq.index[-1]) + 2  # +2 for next bar (early cut)
        end_idx = min(end_idx, len(nq_5m))
        full_range = nq_5m.iloc[warmup_start:end_idx]
        test_start_idx_override = None
        print(f"Including {start_idx - warmup_start} warmup bars, total range: {len(full_range)} bars")

    # ES alignment — build a lookup dict for timestamps in our range
    es_aligned = {}
    if es_5m is not None:
        es_range = es_5m.reindex(full_range.index)
        es_valid = es_range.dropna(subset=['open'])
        for ts in es_valid.index:
            es_aligned[ts] = {
                'open': float(es_valid.loc[ts, 'open']),
                'high': float(es_valid.loc[ts, 'high']),
                'low': float(es_valid.loc[ts, 'low']),
                'close': float(es_valid.loc[ts, 'close']),
            }
        print(f"ES bars aligned: {len(es_aligned)} / {len(full_range)}")

    # ---- Initialize engine ----
    engine = BarByBarEngine(params)

    # ---- Cache mode: load pre-computed signals + bias ----
    use_cache = "--use-cache" in sys.argv
    if use_cache:
        print("Loading pre-computed signal caches for convergence mode...")
        sig3_cache = pd.read_parquet(PROJECT / "data" / "cache_signals_10yr_v3.parquet")
        bias_cache = pd.read_parquet(PROJECT / "data" / "cache_bias_10yr_v2.parquet")
        regime_cache = pd.read_parquet(PROJECT / "data" / "cache_regime_10yr_v2.parquet")

        # Compute SMT + merge signals (same as validate_nt_logic.py)
        from features.smt import compute_smt
        smt_data = compute_smt(nq_5m, es_5m, {'swing': {'left_bars': 3, 'right_bars': 1},
                                                 'smt': {'sweep_lookback': 15, 'time_tolerance': 1}})

        ss = sig3_cache.copy()
        mm = ss['signal'].astype(bool) & (ss['signal_type'] == 'mss')
        mi = ss.index[mm]
        ss.loc[mi, 'signal'] = False; ss.loc[mi, 'signal_dir'] = 0; ss['has_smt'] = False
        c_idx = mi.intersection(smt_data.index)
        if len(c_idx) > 0:
            md = sig3_cache.loc[c_idx, 'signal_dir'].values
            ok = ((md == 1) & smt_data.loc[c_idx, 'smt_bull'].values.astype(bool)) | \
                 ((md == -1) & smt_data.loc[c_idx, 'smt_bear'].values.astype(bool))
            g = c_idx[ok]
            ss.loc[g, 'signal'] = sig3_cache.loc[g, 'signal']
            ss.loc[g, 'signal_dir'] = sig3_cache.loc[g, 'signal_dir']
            ss.loc[g, 'has_smt'] = True
        # Kill MSS in overnight (16:00-03:00 ET)
        rem = ss['signal'].astype(bool) & (ss['signal_type'] == 'mss')
        mi2 = ss.index[rem]
        if len(mi2) > 0:
            et_mi2 = mi2.tz_convert('US/Eastern')
            ef_mi2 = et_mi2.hour + et_mi2.minute / 60.0
            kill = (ef_mi2 >= 16.0) | (ef_mi2 < 3.0)
            if kill.any():
                ss.loc[mi2[kill], ['signal', 'signal_dir']] = [False, 0]

        sig_mask = ss['signal'].values.astype(bool)
        print(f"Cache: {sig_mask.sum()} signals after SMT gate + overnight kill")

        # Build signal dict: bar_index -> signal data
        cache_signals = {}
        signal_indices = np.where(sig_mask)[0]
        for idx in signal_indices:
            cache_signals[idx] = {
                'signal_type': str(ss['signal_type'].iat[idx]),
                'direction': int(ss['signal_dir'].iat[idx]),
                'entry_price': float(ss['entry_price'].iat[idx]),
                'model_stop': float(ss['model_stop'].iat[idx]),
                'irl_target': float(ss['irl_target'].iat[idx]),
                'has_smt': bool(ss['has_smt'].iat[idx]),
                'swept': bool(ss['swept_liquidity'].iat[idx]) if 'swept_liquidity' in ss.columns else False,
                'sweep_score': int(ss['sweep_score'].iat[idx]) if 'sweep_score' in ss.columns else 0,
            }

        # Build bias dict: bar_index -> (bias_dir, bias_conf, regime)
        cache_bias = {}
        bias_dir_arr = bias_cache["bias_direction"].values.astype(float)
        bias_conf_arr = bias_cache["bias_confidence"].values.astype(float)
        regime_arr = regime_cache["regime"].values.astype(float)
        for idx in range(len(nq_5m)):
            cache_bias[idx] = (float(bias_dir_arr[idx]), float(bias_conf_arr[idx]),
                                float(regime_arr[idx]))

        # Pre-compute SQ values for signal bars (same as validate_nt_logic.py)
        from features.displacement import compute_atr as _compute_atr, compute_fluency as _compute_fluency
        from features.pa_quality import compute_alternating_dir_ratio as _compute_pa_alt
        _atr_arr = _compute_atr(nq_5m).values
        _fluency_arr = _compute_fluency(nq_5m, params).values
        _o = nq_5m["open"].values
        _h = nq_5m["high"].values
        _l = nq_5m["low"].values
        _c = nq_5m["close"].values

        cache_sq = {}
        _sq_params = params.get("signal_quality", {})
        for idx in signal_indices:
            a = _atr_arr[idx] if not np.isnan(_atr_arr[idx]) else 10.0
            gap = abs(float(ss['entry_price'].iat[idx]) - float(ss['model_stop'].iat[idx]))
            size_sc = min(1.0, gap / (a * 1.5)) if a > 0 else 0.5
            body = abs(_c[idx] - _o[idx])
            rng = _h[idx] - _l[idx]
            disp_sc = body / rng if rng > 0 else 0.0
            flu_val = _fluency_arr[idx]
            flu_sc = min(1.0, max(0.0, flu_val)) if not np.isnan(flu_val) else 0.5
            window = 6
            if idx >= window:
                dirs = np.sign(_c[idx-window:idx] - _o[idx-window:idx])
                alt = np.sum(dirs[1:] != dirs[:-1]) / (window - 1)
                pa_sc = 1.0 - alt
            else:
                pa_sc = 0.5
            cache_sq[idx] = (_sq_params.get("w_size",0.3)*size_sc +
                              _sq_params.get("w_disp",0.3)*disp_sc +
                              _sq_params.get("w_flu",0.2)*flu_sc +
                              _sq_params.get("w_pa",0.2)*pa_sc)

        # Also cache ATR for min_stop filter
        cache_atr = {}
        for idx in range(len(nq_5m)):
            cache_atr[idx] = float(_atr_arr[idx])

        engine._use_cache = True
        engine._cache_signals = cache_signals
        engine._cache_bias = cache_bias
        engine._cache_sq = cache_sq
        engine._cache_atr = cache_atr
        print(f"Cache loaded: {len(cache_signals)} signals, {len(cache_bias)} bias entries, {len(cache_sq)} SQ values")

    # Set up news blackout: convert mask to ET datetime windows
    # Simpler approach: set a flag per bar using the pre-computed mask
    # We'll pass the original news blackout arr for the full dataset
    # and check by index position

    # ---- Run bar by bar ----
    print("Running bar-by-bar engine...")
    t0 = _time.perf_counter()

    trades = []
    if test_start_idx_override is not None:
        test_start_idx = test_start_idx_override
    else:
        test_start_idx = start_idx - warmup_start  # index in full_range where test begins

    for i in range(len(full_range)):
        ts = full_range.index[i]

        time_et = ts.tz_convert(et_tz).replace(tzinfo=None)

        bar = {
            'time': ts,
            'time_et': time_et,
            'open': float(full_range['open'].iat[i]),
            'high': float(full_range['high'].iat[i]),
            'low': float(full_range['low'].iat[i]),
            'close': float(full_range['close'].iat[i]),
            'volume': float(full_range['volume'].iat[i]) if 'volume' in full_range.columns else 0.0,
            'is_roll_date': bool(full_range['is_roll_date'].iat[i]) if 'is_roll_date' in full_range.columns else False,
        }

        # Handle news blackout via the mask
        if news_blackout_arr is not None:
            global_idx = warmup_start + i
            if global_idx < len(news_blackout_arr):
                engine._news_blackout_times = []
                if news_blackout_arr[global_idx]:
                    # Set a blackout window around current time
                    engine._news_blackout_times = [(time_et - timedelta(minutes=1),
                                                     time_et + timedelta(minutes=1))]

        # Check if HTF bars completed at this timestamp
        if ts in htf_1h_set:
            engine.on_htf_bar('1H', htf_1h_bars[ts])
        if ts in htf_4h_set:
            engine.on_htf_bar('4H', htf_4h_bars[ts])

        # Process ES bar for SMT
        if ts in es_aligned:
            engine.on_es_bar(es_aligned[ts])

        # Process the NQ bar
        result = engine.on_bar(bar)

        # Only collect trades from test window
        if i >= test_start_idx:
            if result is not None:
                trades.append(result)

    # Force close any open position
    last_bar = {
        'time': full_range.index[-1],
        'time_et': full_range.index[-1].tz_convert(et_tz).replace(tzinfo=None),
        'open': float(full_range['open'].iat[-1]),
        'high': float(full_range['high'].iat[-1]),
        'low': float(full_range['low'].iat[-1]),
        'close': float(full_range['close'].iat[-1]),
        'volume': 0.0,
        'is_roll_date': False,
    }
    final_trade = engine.force_close(last_bar)
    if final_trade is not None:
        trades.append(final_trade)

    elapsed = _time.perf_counter() - t0

    # ---- Print filter funnel ----
    engine.print_filter_funnel()

    # ---- Results ----
    print()
    print("=" * 70)
    print("BAR-BY-BAR ENGINE RESULTS")
    print("=" * 70)
    print(f"Test period: {start} to {end}")
    print(f"Processing time: {elapsed:.1f}s")
    print(f"Total trades: {len(trades)}")
    print(f"Target: 534 trades (reference)")

    if trades:
        import pandas as pd
        trade_df = pd.DataFrame(trades)
        total_r = trade_df['r'].sum()
        win_rate = 100.0 * (trade_df['r'] > 0).mean()

        print(f"Total R: {total_r:.2f}")
        print(f"Win rate: {win_rate:.1f}%")
        print()
        print("By type:")
        for st in trade_df['type'].unique():
            m = trade_df['type'] == st
            print(f"  {st}: {m.sum()} trades, R={trade_df.loc[m, 'r'].sum():.2f}")
        print()
        print("By exit reason:")
        for er in trade_df['reason'].unique():
            m = trade_df['reason'] == er
            print(f"  {er}: {m.sum()} trades, R={trade_df.loc[m, 'r'].sum():.2f}")
        print()
        print("By direction:")
        long_m = trade_df['dir'] == 1
        short_m = trade_df['dir'] == -1
        print(f"  Long:  {long_m.sum()} trades, R={trade_df.loc[long_m, 'r'].sum():.2f}")
        print(f"  Short: {short_m.sum()} trades, R={trade_df.loc[short_m, 'r'].sum():.2f}")
    else:
        print("No trades generated.")

    # ---- Individual trade details ----
    if trades:
        print()
        print("Individual trades:")
        print(f"{'#':>3} {'Entry Time':>22} {'Dir':>5} {'Type':>6} {'Entry':>10} {'Stop':>10} "
              f"{'TP1':>10} {'Exit':>10} {'R':>8} {'Reason':>14} {'Grade':>4}")
        print("-" * 115)
        for j, t in enumerate(trades):
            entry_str = str(t.get('entry_time', ''))[:22]
            print(f"{j:3d} {entry_str:>22} {t['dir']:>5} {t['type']:>6} "
                  f"{t['entry_price']:>10.2f} {t['stop_price']:>10.2f} "
                  f"{t['tp1_price']:>10.2f} {t['exit_price']:>10.2f} "
                  f"{t['r']:>8.3f} {t['reason']:>14} {t.get('grade',''):>4}")

    # ---- Compare with reference (validate_nt_logic.py output) ----
    print()
    print("=" * 70)
    print("COMPARISON WITH REFERENCE")
    print("=" * 70)

    ref_path = PROJECT / "ninjatrader" / "python_trades_545.csv"
    if ref_path.exists():
        ref_df = pd.read_csv(ref_path, parse_dates=['entry_time', 'exit_time'])

        # Filter reference to test period
        ref_test = ref_df[
            (ref_df['entry_time'] >= start) &
            (ref_df['entry_time'] < end)
        ]

        print(f"Reference (validate_nt_logic.py): {len(ref_test)} trades in {start} to {end}")
        print(f"Bar-by-bar engine:                {len(trades)} trades")

        if len(ref_test) > 0:
            ref_r = ref_test['r'].sum()
            print(f"Reference total R: {ref_r:.2f}")
            if trades:
                print(f"Bar-by-bar total R: {total_r:.2f}")
                print(f"R difference: {total_r - ref_r:.2f}")

            print()
            print("Reference trades:")
            for _, row in ref_test.iterrows():
                print(f"  {row['entry_time']} dir={row['dir']} type={row['type']} "
                      f"entry={row['entry_price']:.2f} stop={row['stop_price']:.2f} "
                      f"tp1={row['tp1_price']:.2f} R={row['r']:.3f} reason={row['reason']}")
    else:
        print("Reference file not found (ninjatrader/python_trades_545.csv)")
        print("Run ninjatrader/validate_nt_logic.py first to generate reference.")

    # ---- Export bar-by-bar trades ----
    if trades and full_run:
        trade_df.to_csv(PROJECT / "ninjatrader" / "bar_by_bar_trades.csv", index=False)
        print(f"\nExported {len(trades)} trades to ninjatrader/bar_by_bar_trades.csv")

    # ---- Detailed comparison in full+cache mode ----
    if full_run and use_cache and ref_path.exists() and trades:
        print()
        print("=" * 70)
        print("DETAILED DIVERGENCE ANALYSIS")
        print("=" * 70)
        ref_df = pd.read_csv(ref_path, parse_dates=['entry_time', 'exit_time'])
        bb_df = trade_df.copy()

        # Match trades by entry price + direction
        ref_entries = set()
        for _, row in ref_df.iterrows():
            key = (round(row['entry_price'], 2), int(row['dir']), str(row['type']))
            ref_entries.add(key)

        bb_entries = set()
        for _, row in bb_df.iterrows():
            key = (round(row['entry_price'], 2), int(row['dir']), str(row['type']))
            bb_entries.add(key)

        only_ref = ref_entries - bb_entries
        only_bb = bb_entries - ref_entries

        print(f"\nTrades in reference but NOT in bar-by-bar: {len(only_ref)}")
        if len(only_ref) <= 50:
            for key in sorted(only_ref):
                print(f"  entry={key[0]:.2f}, dir={key[1]}, type={key[2]}")
        else:
            print(f"  (showing first 50)")
            for key in sorted(only_ref)[:50]:
                print(f"  entry={key[0]:.2f}, dir={key[1]}, type={key[2]}")

        print(f"\nTrades in bar-by-bar but NOT in reference: {len(only_bb)}")
        if len(only_bb) <= 50:
            for key in sorted(only_bb):
                print(f"  entry={key[0]:.2f}, dir={key[1]}, type={key[2]}")
        else:
            print(f"  (showing first 50)")
            for key in sorted(only_bb)[:50]:
                print(f"  entry={key[0]:.2f}, dir={key[1]}, type={key[2]}")

        # Check R-multiple differences for matching trades
        matched = ref_entries & bb_entries
        if matched:
            r_diffs = []
            for key in matched:
                ref_r = ref_df[(ref_df['entry_price'].round(2) == key[0]) &
                                (ref_df['dir'] == key[1]) &
                                (ref_df['type'] == key[2])]['r'].values
                bb_r = bb_df[(bb_df['entry_price'].round(2) == key[0]) &
                              (bb_df['dir'] == key[1]) &
                              (bb_df['type'] == key[2])]['r'].values
                if len(ref_r) > 0 and len(bb_r) > 0:
                    r_diffs.append(abs(ref_r[0] - bb_r[0]))
            if r_diffs:
                print(f"\nMatched trades: {len(matched)}")
                print(f"  R-multiple diffs: max={max(r_diffs):.4f}, mean={sum(r_diffs)/len(r_diffs):.4f}")
                print(f"  Trades with R diff > 0.01: {sum(1 for d in r_diffs if d > 0.01)}")

    print()
    print("NOTE: Signal mismatch expected -- this engine computes signals from raw")
    print("OHLCV, while the reference uses pre-computed vectorized cache over 10yr.")
    print("The bar-by-bar engine is the REFERENCE for C# NinjaTrader porting.")
    print("=" * 70)
