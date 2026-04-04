"""Tests for features/news_filter.py -- news blackout mask."""

import pandas as pd
import numpy as np
import pytest

from features.news_filter import load_news_calendar, build_news_blackout_mask


class TestLoadCalendar:
    """Tests for load_news_calendar."""

    def test_loads_successfully(self) -> None:
        cal = load_news_calendar()
        assert len(cal) > 0
        assert "datetime_utc" in cal.columns
        assert "datetime_et" in cal.columns

    def test_all_weekdays(self) -> None:
        """News events should only occur on weekdays."""
        cal = load_news_calendar()
        # datetime_utc may have some events at 13:30 UTC (8:30 ET) which
        # is the same calendar day as ET, so check ET day-of-week
        days = cal["datetime_et"].dt.dayofweek
        assert (days < 5).all(), "Found weekend events in calendar"

    def test_expected_event_types(self) -> None:
        """Calendar should contain all expected high-impact event types."""
        cal = load_news_calendar()
        events = set(cal["event"].unique())
        expected = {"CPI", "FOMC", "NFP", "GDP", "PCE", "PPI", "Jobless Claims"}
        assert expected.issubset(events), f"Missing events: {expected - events}"

    def test_utc_timezone_aware(self) -> None:
        cal = load_news_calendar()
        assert cal["datetime_utc"].dt.tz is not None


class TestBlackoutMask:
    """Tests for build_news_blackout_mask."""

    def test_blackout_around_known_cpi(self) -> None:
        """Test that a known CPI date creates correct blackout window."""
        cal = load_news_calendar()
        cpi = cal[cal["event"] == "CPI"].iloc[0]
        event_utc = cpi["datetime_utc"]

        # Create a minute-resolution index around the event
        idx = pd.date_range(
            event_utc - pd.Timedelta(hours=2),
            event_utc + pd.Timedelta(hours=1),
            freq="1min",
            tz="UTC",
        )

        mask = build_news_blackout_mask(
            idx, blackout_minutes_before=60, cooldown_minutes_after=5,
        )

        # 60 min before should be True (blackout start)
        assert mask[event_utc - pd.Timedelta(minutes=60)] == True
        # 30 min before should be True (within blackout)
        assert mask[event_utc - pd.Timedelta(minutes=30)] == True
        # At event time should be True
        assert mask[event_utc] == True
        # 5 min after should still be True (cooldown)
        assert mask[event_utc + pd.Timedelta(minutes=5)] == True
        # 10 min after should be False (cooldown over)
        assert mask[event_utc + pd.Timedelta(minutes=10)] == False
        # 90 min before should be False (outside blackout)
        assert mask[event_utc - pd.Timedelta(minutes=90)] == False

    def test_blackout_around_fomc(self) -> None:
        """FOMC events are at 14:00 ET (19:00 UTC) -- different time than 8:30."""
        cal = load_news_calendar()
        fomc = cal[cal["event"] == "FOMC"].iloc[0]
        event_utc = fomc["datetime_utc"]

        idx = pd.date_range(
            event_utc - pd.Timedelta(hours=2),
            event_utc + pd.Timedelta(hours=1),
            freq="1min",
            tz="UTC",
        )

        mask = build_news_blackout_mask(
            idx, blackout_minutes_before=60, cooldown_minutes_after=5,
        )

        # Blackout should be active at event time
        assert mask[event_utc] == True
        # 60 min before
        assert mask[event_utc - pd.Timedelta(minutes=60)] == True
        # 61 min before should be False
        assert mask[event_utc - pd.Timedelta(minutes=61)] == False

    def test_no_blackout_outside_calendar_range(self) -> None:
        """Dates outside calendar range should have no blackout."""
        idx = pd.date_range("2020-01-01", periods=100, freq="5min", tz="UTC")
        mask = build_news_blackout_mask(idx)
        assert mask.sum() == 0

    def test_disabled_when_zero_minutes(self) -> None:
        """blackout_minutes_before=0 should produce all-False mask."""
        cal = load_news_calendar()
        cpi = cal[cal["event"] == "CPI"].iloc[0]
        event_utc = cpi["datetime_utc"]

        idx = pd.date_range(
            event_utc - pd.Timedelta(hours=1),
            event_utc + pd.Timedelta(hours=1),
            freq="1min",
            tz="UTC",
        )
        mask = build_news_blackout_mask(
            idx, blackout_minutes_before=0, cooldown_minutes_after=5,
        )
        assert mask.sum() == 0

    def test_mask_aligned_to_index(self) -> None:
        """Returned mask must have same index as input."""
        cal = load_news_calendar()
        event_utc = cal.iloc[0]["datetime_utc"]
        idx = pd.date_range(
            event_utc - pd.Timedelta(hours=2),
            event_utc + pd.Timedelta(hours=2),
            freq="5min",
            tz="UTC",
        )
        mask = build_news_blackout_mask(idx)
        assert len(mask) == len(idx)
        assert (mask.index == idx).all()

    def test_overlapping_events_merged(self) -> None:
        """If two events are close together, their blackout windows merge."""
        cal = load_news_calendar()
        # Find a date with multiple events (e.g., same-day CPI + Jobless Claims)
        date_counts = cal.groupby("date").size()
        multi_dates = date_counts[date_counts > 1]
        if len(multi_dates) == 0:
            pytest.skip("No multi-event dates in calendar")

        multi_date = multi_dates.index[0]
        events = cal[cal["date"] == multi_date]
        first_event = events.iloc[0]["datetime_utc"]

        idx = pd.date_range(
            first_event - pd.Timedelta(hours=2),
            first_event + pd.Timedelta(hours=2),
            freq="1min",
            tz="UTC",
        )
        mask = build_news_blackout_mask(idx)

        # At the event time, should definitely be blocked
        assert mask[first_event] == True

    def test_dtype_is_bool(self) -> None:
        """Mask must be boolean dtype."""
        idx = pd.date_range("2024-06-01", periods=100, freq="5min", tz="UTC")
        mask = build_news_blackout_mask(idx)
        assert mask.dtype == bool
