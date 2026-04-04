"""
tests/test_mtf_fvg.py -- Tests for features/mtf_fvg.py

Covers:
  1. HTF FVG detection correctness (bullish + bearish on synthetic data)
  2. FVG state machine transitions (untested -> tested_rejected -> invalidated -> IFVG)
  3. shift(1) no-lookahead guarantee
  4. 5m alignment correctness
  5. Combined feature computation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.fvg import detect_fvg, FVGRecord, _update_fvg_status
from features.mtf_fvg import (
    _compute_htf_fvg_per_bar,
    _align_htf_to_5m,
    _pick_nearest,
    compute_htf_fvg_features,
)

# Quality filter params that pass everything through (for testing detection
# logic without quality filtering interfering on synthetic data).
_NO_QUALITY_FILTER = dict(
    min_fvg_size_atr=0.0,
    min_displacement_body_ratio=0.0,
    max_fvg_age_bars=999999,
)


# ---------------------------------------------------------------------------
# Helpers: build synthetic OHLCV DataFrames
# ---------------------------------------------------------------------------

def _make_ohlcv(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    freq: str = "1h",
    start: str = "2024-01-01 00:00",
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with is_roll_date=False."""
    n = len(opens)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000] * n,
            "is_roll_date": [False] * n,
        },
        index=idx,
    )


def _make_bullish_fvg_ohlcv(freq: str = "1h") -> pd.DataFrame:
    """3-candle pattern that creates a bullish FVG.

    Candle 1: H=100, candle 3: L=105 -> gap [100, 105].
    Then 7 more bars that stay above the gap.
    """
    #           open   high   low    close
    candles = [
        (95,   100,   93,    98),    # c1: high=100
        (99,   110,   98,    109),   # c2: displacement candle
        (108,  112,   105,   111),   # c3: low=105 -> gap: [100, 105]
        # Post-FVG bars (price stays above gap)
        (111,  115,   110,   114),
        (114,  118,   113,   117),
        (117,  120,   116,   119),
        (119,  122,   118,   121),
        (121,  124,   120,   123),
        (123,  126,   122,   125),
        (125,  128,   124,   127),
    ]
    opens = [c[0] for c in candles]
    highs = [c[1] for c in candles]
    lows = [c[2] for c in candles]
    closes = [c[3] for c in candles]
    return _make_ohlcv(opens, highs, lows, closes, freq=freq)


def _make_bearish_fvg_ohlcv(freq: str = "1h") -> pd.DataFrame:
    """3-candle pattern that creates a bearish FVG.

    Candle 1: L=110, candle 3: H=105 -> gap: [105, 110].
    Then bars that stay below the gap.
    """
    candles = [
        (112,  115,   110,   113),   # c1: low=110
        (112,  113,   98,    99),    # c2: displacement candle down
        (100,  105,   96,    97),    # c3: high=105 -> gap: [105, 110]
        # Post-FVG bars (price stays below gap)
        (97,   102,   95,    96),
        (96,   100,   94,    95),
        (95,   99,    93,    94),
        (94,   98,    92,    93),
        (93,   97,    91,    92),
        (92,   96,    90,    91),
        (91,   95,    89,    90),
    ]
    opens = [c[0] for c in candles]
    highs = [c[1] for c in candles]
    lows = [c[2] for c in candles]
    closes = [c[3] for c in candles]
    return _make_ohlcv(opens, highs, lows, closes, freq=freq)


# ---------------------------------------------------------------------------
# 1. Test HTF FVG detection
# ---------------------------------------------------------------------------

class TestHTFFVGDetection:
    """Test _compute_htf_fvg_per_bar on synthetic data."""

    def test_bullish_fvg_detected(self):
        """A clear bullish FVG pattern should be detected."""
        df = _make_bullish_fvg_ohlcv()
        feats = _compute_htf_fvg_per_bar(df, **_NO_QUALITY_FILTER)

        # The FVG at candle 2 (index 1) is visible at candle 3 (index 2)
        # due to detect_fvg's internal shift(1).
        # After that, price stays above the gap -> bull_active should be True.
        assert feats["htf_bull_active"].any(), "Bullish FVG should be detected"

        # From bar 2 onward (when the FVG becomes visible), bull_active should be True
        # since price stays above the gap
        later_bars = feats.iloc[2:]
        assert later_bars["htf_bull_active"].all(), (
            "Bullish FVG should remain active when price stays above"
        )

    def test_bearish_fvg_detected(self):
        """A clear bearish FVG pattern should be detected."""
        df = _make_bearish_fvg_ohlcv()
        feats = _compute_htf_fvg_per_bar(df, **_NO_QUALITY_FILTER)

        assert feats["htf_bear_active"].any(), "Bearish FVG should be detected"

        later_bars = feats.iloc[2:]
        assert later_bars["htf_bear_active"].all(), (
            "Bearish FVG should remain active when price stays below"
        )

    def test_no_fvg_in_flat_market(self):
        """A flat market with no gaps should produce no FVGs."""
        # All candles overlap -> no gap
        n = 20
        opens = [100.0 + i * 0.1 for i in range(n)]
        highs = [o + 2.0 for o in opens]
        lows = [o - 2.0 for o in opens]
        closes = [o + 0.5 for o in opens]
        df = _make_ohlcv(opens, highs, lows, closes)
        feats = _compute_htf_fvg_per_bar(df)

        assert not feats["htf_bull_active"].any()
        assert not feats["htf_bear_active"].any()

    def test_bias_sign(self):
        """Bias should be +1 when only bull FVGs, -1 when only bear FVGs."""
        df_bull = _make_bullish_fvg_ohlcv()
        feats_bull = _compute_htf_fvg_per_bar(df_bull, **_NO_QUALITY_FILTER)
        # Once the bullish FVG is active, bias should be +1
        active_bars = feats_bull[feats_bull["htf_bull_active"]]
        assert (active_bars["htf_bias"] == 1).all(), "Bias should be +1 with bull FVGs"

        df_bear = _make_bearish_fvg_ohlcv()
        feats_bear = _compute_htf_fvg_per_bar(df_bear, **_NO_QUALITY_FILTER)
        active_bars = feats_bear[feats_bear["htf_bear_active"]]
        assert (active_bars["htf_bias"] == -1).all(), "Bias should be -1 with bear FVGs"


# ---------------------------------------------------------------------------
# 2. Test FVG state machine transitions
# ---------------------------------------------------------------------------

class TestFVGStateMachine:
    """Test that state machine transitions work through _compute_htf_fvg_per_bar."""

    def test_invalidation_removes_fvg(self):
        """When price closes through a bullish FVG, it should be invalidated."""
        # Build a bullish FVG then crash through it
        candles = [
            (95,   100,   93,    98),    # c1: high=100
            (99,   110,   98,    109),   # c2: displacement
            (108,  112,   105,   111),   # c3: gap [100, 105] becomes visible
            (110,  113,   109,   112),   # price above gap: still active
            (112,  113,   90,    88),    # close=88 < gap_bottom=100 -> invalidated
            (88,   90,    85,    86),    # after invalidation
            (86,   88,    83,    84),
            (84,   86,    81,    82),
            (82,   84,    79,    80),
            (80,   82,    77,    78),
        ]
        opens = [c[0] for c in candles]
        highs = [c[1] for c in candles]
        lows = [c[2] for c in candles]
        closes = [c[3] for c in candles]
        df = _make_ohlcv(opens, highs, lows, closes)
        feats = _compute_htf_fvg_per_bar(df, **_NO_QUALITY_FILTER)

        # Bar 2: FVG becomes visible, bar 3: still active
        assert feats.iloc[3]["htf_bull_active"] is np.True_

        # Bar 4: crash through -> invalidated. From bar 5+, the original
        # bullish FVG is gone. An IFVG (bearish) may be spawned.
        # The bullish FVG count should drop.
        assert feats.iloc[5]["htf_num_bull"] == 0, (
            "Bullish FVG count should be 0 after invalidation"
        )

    def test_ifvg_spawned_on_invalidation(self):
        """When a bullish FVG is invalidated, a bearish IFVG should appear."""
        candles = [
            (95,   100,   93,    98),    # c1
            (99,   110,   98,    109),   # c2
            (108,  112,   105,   111),   # c3: bullish gap [100, 105]
            (110,  113,   109,   112),   # active
            (112,  113,   90,    88),    # invalidated + IFVG spawned
            (88,   95,    85,    86),    # bear IFVG should be active
            (86,   94,    83,    84),
            (84,   92,    81,    82),
            (82,   90,    79,    80),
            (80,   88,    77,    78),
        ]
        opens = [c[0] for c in candles]
        highs = [c[1] for c in candles]
        lows = [c[2] for c in candles]
        closes = [c[3] for c in candles]
        df = _make_ohlcv(opens, highs, lows, closes)
        feats = _compute_htf_fvg_per_bar(df, **_NO_QUALITY_FILTER)

        # After invalidation (bar 4), an IFVG (bearish) should be active
        # The IFVG zone is [100, 105] but now acts as bearish resistance.
        # Price at bars 5-9 is below 100, so the IFVG sits above price.
        # If price stays below it (close < bottom=100), the IFVG is untested.
        # bear_active should be True from bar 4 onward (or possibly bar 5,
        # depending on whether IFVG is spawned before or after bar processing).
        bear_after = feats.iloc[4:]["htf_bear_active"]
        assert bear_after.any(), "Bearish IFVG should be active after invalidation"


# ---------------------------------------------------------------------------
# 3. Test shift(1) no-lookahead guarantee
# ---------------------------------------------------------------------------

class TestNoLookahead:
    """Verify that the shift(1) + ffill projection prevents future data leakage."""

    def test_htf_features_delayed_by_one_bar(self):
        """HTF features on the first visible 5m bar should reflect the
        *previous* HTF bar's features, not the current one."""
        # Create 1H data with a distinguishable feature change
        df_1h = _make_bullish_fvg_ohlcv(freq="1h")
        feats_1h = _compute_htf_fvg_per_bar(df_1h, **_NO_QUALITY_FILTER)

        # Create a 5m index that spans the same period
        start = df_1h.index[0]
        end = df_1h.index[-1] + pd.Timedelta(hours=1)
        idx_5m = pd.date_range(start, end, freq="5min", tz="UTC")
        df_5m = pd.DataFrame(index=idx_5m)

        aligned = _align_htf_to_5m(feats_1h, df_5m, "1H")

        # The 1H FVG becomes visible at bar index 2 (02:00 UTC).
        # After shift(1), this feature only appears at bar index 3 (03:00 UTC).
        # So 5m bars at 02:00-02:55 should NOT see this FVG;
        # 5m bars at 03:00+ should see it.

        # Bars within the 02:00 hour (before shift takes effect)
        hour_2_bars = aligned.loc[
            (aligned.index >= df_1h.index[2])
            & (aligned.index < df_1h.index[2] + pd.Timedelta(hours=1))
        ]
        # The bull_active at hour 2 in the raw features should NOT be visible
        # in the aligned data for hour 2's 5m bars
        raw_at_h2 = feats_1h.iloc[2]["htf_bull_active"]
        raw_at_h1 = feats_1h.iloc[1]["htf_bull_active"]

        if not hour_2_bars.empty:
            # The aligned value during hour 2 should be the value from hour 1
            # (due to shift(1)), not hour 2
            aligned_val = hour_2_bars.iloc[0]["1H_htf_bull_active"]
            if not pd.isna(aligned_val):
                assert aligned_val == raw_at_h1, (
                    f"5m bars in hour 2 should see hour 1's value "
                    f"({raw_at_h1}), got {aligned_val}"
                )

    def test_5m_bars_before_first_htf_close_are_nan(self):
        """5m bars before the first HTF bar closes should have NaN features."""
        df_1h = _make_bullish_fvg_ohlcv(freq="1h")
        feats_1h = _compute_htf_fvg_per_bar(df_1h, **_NO_QUALITY_FILTER)

        start = df_1h.index[0]
        end = df_1h.index[-1] + pd.Timedelta(hours=1)
        idx_5m = pd.date_range(start, end, freq="5min", tz="UTC")
        df_5m = pd.DataFrame(index=idx_5m)

        aligned = _align_htf_to_5m(feats_1h, df_5m, "1H")

        # First 1H bar is at index[0]. After shift(1), nothing is available
        # until index[1]. So 5m bars at index[0]'s hour should all be NaN
        # for numeric columns.
        first_hour_bars = aligned.loc[
            aligned.index < df_1h.index[1]
        ]
        if not first_hour_bars.empty:
            for col in ["1H_htf_nearest_bull_dist", "1H_htf_nearest_bear_dist"]:
                assert first_hour_bars[col].isna().all(), (
                    f"Column {col} should be NaN before first HTF close"
                )


# ---------------------------------------------------------------------------
# 4. Test alignment correctness
# ---------------------------------------------------------------------------

class TestAlignment:
    """Test _align_htf_to_5m and _pick_nearest."""

    def test_ffill_works(self):
        """Features should forward-fill between HTF bar boundaries."""
        df_1h = _make_bullish_fvg_ohlcv(freq="1h")
        feats_1h = _compute_htf_fvg_per_bar(df_1h, **_NO_QUALITY_FILTER)

        start = df_1h.index[0]
        end = df_1h.index[-1] + pd.Timedelta(hours=1)
        idx_5m = pd.date_range(start, end, freq="5min", tz="UTC")
        df_5m = pd.DataFrame(index=idx_5m)

        aligned = _align_htf_to_5m(feats_1h, df_5m, "1H")

        # Between two consecutive 1H boundaries, all 5m bars should have
        # the same value (ffill).
        # Pick bars between hour 3 and hour 4 (after shift, they see hour 2's values)
        h3 = df_1h.index[3]
        h4 = df_1h.index[4] if len(df_1h.index) > 4 else h3 + pd.Timedelta(hours=1)
        between = aligned.loc[(aligned.index >= h3) & (aligned.index < h4)]

        if len(between) > 1:
            col = "1H_htf_bias"
            vals = between[col].dropna().unique()
            assert len(vals) <= 1, (
                f"All 5m bars between {h3} and {h4} should have same "
                f"ffilled value for {col}, got {vals}"
            )

    def test_pick_nearest_both_valid(self):
        """_pick_nearest should pick the closer of two valid distances."""
        a = pd.Series([10.0, 5.0, np.nan, 3.0])
        b = pd.Series([2.0, 8.0, 4.0, np.nan])
        result = _pick_nearest(a, b)
        expected = pd.Series([2.0, 5.0, 4.0, 3.0])
        pd.testing.assert_series_equal(result, expected)

    def test_pick_nearest_nan_handling(self):
        """_pick_nearest should handle NaN correctly."""
        a = pd.Series([np.nan, np.nan])
        b = pd.Series([np.nan, 5.0])
        result = _pick_nearest(a, b)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 5.0


# ---------------------------------------------------------------------------
# 5. Test compute_htf_fvg_features (combined)
# ---------------------------------------------------------------------------

class TestComputeHTFFVGFeatures:
    """Test the main public API."""

    def test_output_columns(self):
        """compute_htf_fvg_features should produce the expected columns."""
        df_5m = _make_bullish_fvg_ohlcv(freq="5min")
        df_1h = _make_bullish_fvg_ohlcv(freq="1h")
        df_4h = _make_bullish_fvg_ohlcv(freq="4h")

        result = compute_htf_fvg_features(
            df_5m, df_1h, df_4h, params={"mtf_fvg": _NO_QUALITY_FILTER}
        )

        expected_cols = {
            "htf_fvg_bullish_active",
            "htf_fvg_bearish_active",
            "htf_fvg_nearest_bull_dist",
            "htf_fvg_nearest_bear_dist",
            "htf_bias",
        }
        assert expected_cols.issubset(set(result.columns)), (
            f"Missing columns: {expected_cols - set(result.columns)}"
        )

    def test_output_index_matches_5m(self):
        """Output should have the same index as the 5m input."""
        df_5m = _make_bullish_fvg_ohlcv(freq="5min")
        df_1h = _make_bullish_fvg_ohlcv(freq="1h")
        df_4h = _make_bullish_fvg_ohlcv(freq="4h")

        result = compute_htf_fvg_features(
            df_5m, df_1h, df_4h, params={"mtf_fvg": _NO_QUALITY_FILTER}
        )
        pd.testing.assert_index_equal(result.index, df_5m.index)

    def test_combined_bullish_active_is_or_of_timeframes(self):
        """htf_fvg_bullish_active should be True if either 1H or 4H has bull FVGs."""
        df_1h = _make_bullish_fvg_ohlcv(freq="1h")

        # Build 5m data spanning the full 1H range plus extra, so that
        # after shift(1) the 1H FVG features propagate into the 5m bars.
        # The 1H FVG becomes visible at bar 2 (02:00), and after shift(1)
        # it appears at bar 3 (03:00).  So 5m bars must extend past 03:00.
        start_5m = df_1h.index[0]
        end_5m = df_1h.index[-1] + pd.Timedelta(hours=2)
        idx_5m = pd.date_range(start_5m, end_5m, freq="5min", tz="UTC")
        # Build dummy 5m OHLCV (price above the FVG zone)
        n_5m = len(idx_5m)
        df_5m = pd.DataFrame(
            {
                "open": [115.0] * n_5m,
                "high": [120.0] * n_5m,
                "low": [110.0] * n_5m,
                "close": [117.0] * n_5m,
                "volume": [1000] * n_5m,
                "is_roll_date": [False] * n_5m,
            },
            index=idx_5m,
        )

        # Use flat market for 4H (no FVGs)
        n = 10
        opens = [100.0 + i * 0.1 for i in range(n)]
        highs = [o + 2.0 for o in opens]
        lows = [o - 2.0 for o in opens]
        closes = [o + 0.5 for o in opens]
        df_4h = _make_ohlcv(opens, highs, lows, closes, freq="4h")

        # Disable quality filter for synthetic data test
        params_no_filter = {"mtf_fvg": _NO_QUALITY_FILTER}
        result = compute_htf_fvg_features(df_5m, df_1h, df_4h, params=params_no_filter)

        # 1H has bullish FVGs, 4H does not.  Combined should still show active.
        # After shift(1), 5m bars from 03:00 onward should see the 1H bull FVG.
        assert result["htf_fvg_bullish_active"].any(), (
            "Combined bullish_active should be True if 1H has bull FVGs"
        )


# ---------------------------------------------------------------------------
# 6. Integration test on real data (skipped by default, run with -m integration)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 6. Test quality filter (DECISION-007)
# ---------------------------------------------------------------------------

class TestQualityFilter:
    """Test that the DECISION-007 quality filter rejects small/weak FVGs."""

    def test_quality_filter_rejects_small_fvgs(self):
        """With strict quality filter, the small synthetic FVG should be rejected."""
        df = _make_bullish_fvg_ohlcv()
        # The synthetic FVG has gap size ~5 points; ATR is ~10-15 points.
        # Setting min_fvg_size_atr=2.0 should reject it.
        feats_strict = _compute_htf_fvg_per_bar(
            df, min_fvg_size_atr=2.0, min_displacement_body_ratio=0.0,
            max_fvg_age_bars=999999,
        )
        assert not feats_strict["htf_bull_active"].any(), (
            "Strict size filter should reject small synthetic FVG"
        )

    def test_quality_filter_passes_with_zero_threshold(self):
        """With min_fvg_size_atr=0, all FVGs should pass."""
        df = _make_bullish_fvg_ohlcv()
        feats = _compute_htf_fvg_per_bar(
            df, min_fvg_size_atr=0.0, min_displacement_body_ratio=0.0,
            max_fvg_age_bars=999999,
        )
        assert feats["htf_bull_active"].any(), (
            "Zero threshold should pass all FVGs"
        )

    def test_body_ratio_filter(self):
        """FVG created by a wicky candle should be rejected by body ratio filter."""
        # Create a bullish FVG where candle-2 is very wicky (small body, large range)
        candles = [
            (95,   100,   93,    98),    # c1: high=100
            (99,   115,   92,    100),   # c2: wicky displacement (body=1, range=23, ratio=0.04)
            (108,  112,   105,   111),   # c3: low=105 -> gap [100, 105]
            (111,  115,   110,   114),
            (114,  118,   113,   117),
            (117,  120,   116,   119),
            (119,  122,   118,   121),
            (121,  124,   120,   123),
            (123,  126,   122,   125),
            (125,  128,   124,   127),
        ]
        opens = [c[0] for c in candles]
        highs = [c[1] for c in candles]
        lows = [c[2] for c in candles]
        closes = [c[3] for c in candles]
        df = _make_ohlcv(opens, highs, lows, closes)

        # With strict body ratio filter, wicky candle-2 should be rejected
        feats_strict = _compute_htf_fvg_per_bar(
            df, min_fvg_size_atr=0.0, min_displacement_body_ratio=0.5,
            max_fvg_age_bars=999999,
        )
        assert not feats_strict["htf_bull_active"].any(), (
            "Wicky candle-2 should be rejected by body ratio filter"
        )

        # With no body ratio filter, same data should pass
        feats_permissive = _compute_htf_fvg_per_bar(
            df, min_fvg_size_atr=0.0, min_displacement_body_ratio=0.0,
            max_fvg_age_bars=999999,
        )
        assert feats_permissive["htf_bull_active"].any(), (
            "Zero body ratio threshold should pass all FVGs"
        )

    def test_age_expiry(self):
        """FVGs should expire after max_fvg_age_bars."""
        df = _make_bullish_fvg_ohlcv()
        # FVG born at bar 2 (visible at bar 2 after shift).
        # With max_age=3, it should expire at bar 5.
        feats = _compute_htf_fvg_per_bar(
            df, min_fvg_size_atr=0.0, min_displacement_body_ratio=0.0,
            max_fvg_age_bars=3,
        )
        # Bar 2 + 3 = bar 5: FVG should still be active at bar 4 but gone by bar 6
        if len(feats) > 6:
            assert not feats.iloc[6]["htf_bull_active"], (
                "FVG should be expired after max_fvg_age_bars"
            )


@pytest.mark.integration
class TestIntegrationRealData:
    """Integration tests on real parquet data.  Run with: pytest -m integration"""

    def test_sanity_check_on_real_data(self):
        """Real data should produce reasonable feature distributions."""
        df_5m = pd.read_parquet("data/NQ_5m_10yr.parquet")
        df_1h = pd.read_parquet("data/NQ_1H_10yr.parquet")
        df_4h = pd.read_parquet("data/NQ_4H_10yr.parquet")

        # Use last 20k bars for speed
        df_5m_slice = df_5m.iloc[-20_000:]
        start_dt = df_5m_slice.index[0] - pd.Timedelta(days=60)
        df_1h_slice = df_1h.loc[df_1h.index >= start_dt]
        df_4h_slice = df_4h.loc[df_4h.index >= start_dt]

        result = compute_htf_fvg_features(df_5m_slice, df_1h_slice, df_4h_slice)

        assert result.shape[0] == len(df_5m_slice)

        # Sanity: should have some but not all bars with HTF FVGs
        bull_pct = result["htf_fvg_bullish_active"].mean()
        bear_pct = result["htf_fvg_bearish_active"].mean()
        assert 0.05 < bull_pct < 0.95, f"Bull active {bull_pct:.2%} outside [5%, 95%]"
        assert 0.05 < bear_pct < 0.95, f"Bear active {bear_pct:.2%} outside [5%, 95%]"

        # Bias should not be all one direction
        bias_vals = result["htf_bias"].dropna().unique()
        assert len(bias_vals) >= 2, "Bias should have at least 2 distinct values"
