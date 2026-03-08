"""Tests for tqqq.trading_calendar module."""

from datetime import date

import pytest

from tqqq.trading_calendar import is_trading_day, next_trading_day


class TestIsTradingDay:
    """Tests for is_trading_day function."""

    def test_regular_weekday_is_trading_day(self):
        """A normal Monday should be a trading day."""
        # 2025-01-06 is a Monday (not a holiday)
        assert is_trading_day(date(2025, 1, 6)) is True

    def test_saturday_is_not_trading_day(self):
        """Saturdays are not trading days."""
        # 2025-01-04 is a Saturday
        assert is_trading_day(date(2025, 1, 4)) is False

    def test_sunday_is_not_trading_day(self):
        """Sundays are not trading days."""
        # 2025-01-05 is a Sunday
        assert is_trading_day(date(2025, 1, 5)) is False

    def test_mlk_day_is_not_trading_day(self):
        """MLK Day (3rd Monday of January) is a market holiday."""
        # 2025-01-20 is MLK Day
        assert is_trading_day(date(2025, 1, 20)) is False

    def test_christmas_is_not_trading_day(self):
        """Christmas Day is a market holiday."""
        # 2025-12-25 is Christmas (Thursday)
        assert is_trading_day(date(2025, 12, 25)) is False

    def test_presidents_day_is_not_trading_day(self):
        """Presidents' Day is a market holiday."""
        # 2025-02-17 is Presidents' Day
        assert is_trading_day(date(2025, 2, 17)) is False

    def test_day_after_thanksgiving_is_trading_day(self):
        """Day after Thanksgiving (Black Friday) is a trading day (early close)."""
        # 2025-11-28 is Black Friday — market is open (early close)
        assert is_trading_day(date(2025, 11, 28)) is True

    def test_thanksgiving_is_not_trading_day(self):
        """Thanksgiving is a market holiday."""
        # 2025-11-27 is Thanksgiving
        assert is_trading_day(date(2025, 11, 27)) is False


class TestNextTradingDay:
    """Tests for next_trading_day function."""

    def test_next_day_after_regular_weekday(self):
        """Next trading day after Monday is Tuesday."""
        # 2025-01-06 (Mon) -> 2025-01-07 (Tue)
        assert next_trading_day(date(2025, 1, 6)) == date(2025, 1, 7)

    def test_next_day_after_friday(self):
        """Next trading day after Friday is Monday."""
        # 2025-01-03 (Fri) -> 2025-01-06 (Mon)
        assert next_trading_day(date(2025, 1, 3)) == date(2025, 1, 6)

    def test_next_day_after_holiday(self):
        """Next trading day after a holiday skips to the next open day."""
        # 2025-01-19 (Sun before MLK) -> 2025-01-21 (Tue, since Mon is MLK)
        assert next_trading_day(date(2025, 1, 19)) == date(2025, 1, 21)

    def test_next_day_skips_weekend_and_holiday(self):
        """Next trading day correctly skips consecutive non-trading days."""
        # 2025-01-17 (Fri) -> 2025-01-21 (Tue, Mon is MLK Day)
        assert next_trading_day(date(2025, 1, 17)) == date(2025, 1, 21)
