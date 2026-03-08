"""Integration tests using actual historical TQQQ data."""

import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqqq.config import DB_PATH, MA_SHORT, MA_LONG, TICKER
from tqqq.database import (
    get_connection,
    load_prices,
    get_price_count,
    get_date_range,
    get_new_signals,
    save_signals,
)
from tqqq.signals import detect_crossovers, get_current_status
from tqqq.notifications import format_signal_message, trigger_all_notifications


# Skip integration tests if database doesn't exist or is empty
def has_historical_data():
    """Check if we have historical data to test with."""
    if not DB_PATH.exists():
        return False
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tqqq_prices WHERE ticker = ?", (TICKER,))
    count = cursor.fetchone()[0]
    conn.close()
    return count >= 30  # Need at least 30 days for meaningful tests


requires_historical_data = pytest.mark.skipif(
    not has_historical_data(),
    reason="Requires historical TQQQ data in database"
)


@requires_historical_data
class TestDatabaseIntegration:
    """Integration tests for database operations with real data."""

    def test_database_has_sufficient_data(self):
        """Verify database has enough data for MA calculations."""
        conn = get_connection()
        count = get_price_count(conn, TICKER)
        conn.close()

        assert count >= MA_LONG, f"Need at least {MA_LONG} days of data"

    def test_database_date_range_is_reasonable(self):
        """Verify date range spans a reasonable period."""
        conn = get_connection()
        min_date, max_date = get_date_range(conn, TICKER)
        conn.close()

        assert min_date is not None
        assert max_date is not None
        assert min_date < max_date

    def test_load_prices_returns_valid_data(self):
        """Verify loaded prices have valid structure and values."""
        conn = get_connection()
        df = load_prices(conn, TICKER)
        conn.close()

        # Check structure
        assert "date" in df.columns
        assert "close" in df.columns

        # Check data types
        assert df["close"].dtype in ["float64", "int64"]

        # Check values are reasonable for TQQQ (typically $5-$200 range)
        assert df["close"].min() > 0
        assert df["close"].max() < 500  # Sanity check

    def test_prices_are_ordered_by_date(self):
        """Verify prices are in chronological order."""
        conn = get_connection()
        df = load_prices(conn, TICKER)
        conn.close()

        dates = df["date"].tolist()
        assert dates == sorted(dates)

    def test_no_duplicate_dates(self):
        """Verify no duplicate dates in price data."""
        conn = get_connection()
        df = load_prices(conn, TICKER)
        conn.close()

        assert df["date"].is_unique


@requires_historical_data
class TestSignalDetectionIntegration:
    """Integration tests for crossover signal detection with real data."""

    def test_detect_crossovers_returns_signals(self):
        """Verify crossover detection finds signals in historical data."""
        conn = get_connection()
        with patch("tqqq.signals.MA_GAP_THRESHOLD", 0):
            signals = detect_crossovers(conn, TICKER)
        conn.close()

        # Should have found some signals in a year of data
        assert len(signals) > 0

    def test_signals_have_valid_structure(self):
        """Verify detected signals have correct structure."""
        conn = get_connection()
        with patch("tqqq.signals.MA_GAP_THRESHOLD", 0):
            signals = detect_crossovers(conn, TICKER)
        conn.close()

        for signal in signals:
            assert "ticker" in signal
            assert "date" in signal
            assert "signal_type" in signal
            assert "close_price" in signal
            assert "ma5" in signal
            assert "ma30" in signal

            # Validate signal type
            assert signal["signal_type"] in ["GOLDEN_CROSS", "DEAD_CROSS"]

            # Validate date format (YYYY-MM-DD)
            assert len(signal["date"]) == 10
            assert signal["date"][4] == "-"
            assert signal["date"][7] == "-"

    def test_signals_have_valid_price_values(self):
        """Verify signal price values are reasonable."""
        conn = get_connection()
        with patch("tqqq.signals.MA_GAP_THRESHOLD", 0):
            signals = detect_crossovers(conn, TICKER)
        conn.close()

        for signal in signals:
            assert signal["close_price"] > 0
            assert signal["ma5"] > 0
            assert signal["ma30"] > 0
            assert signal["close_price"] < 500  # Sanity check

    def test_golden_cross_ma5_above_ma30(self):
        """Verify golden cross signals have MA5 > MA30."""
        conn = get_connection()
        with patch("tqqq.signals.MA_GAP_THRESHOLD", 0):
            signals = detect_crossovers(conn, TICKER)
        conn.close()

        golden_crosses = [s for s in signals if s["signal_type"] == "GOLDEN_CROSS"]

        for signal in golden_crosses:
            assert signal["ma5"] > signal["ma30"], \
                f"Golden cross on {signal['date']} has MA5 <= MA30"

    def test_dead_cross_ma5_below_ma30(self):
        """Verify dead cross signals have MA5 < MA30."""
        conn = get_connection()
        with patch("tqqq.signals.MA_GAP_THRESHOLD", 0):
            signals = detect_crossovers(conn, TICKER)
        conn.close()

        dead_crosses = [s for s in signals if s["signal_type"] == "DEAD_CROSS"]

        for signal in dead_crosses:
            assert signal["ma5"] < signal["ma30"], \
                f"Dead cross on {signal['date']} has MA5 >= MA30"

    def test_signals_alternate_between_types(self):
        """Verify signals generally alternate (can't have two golden in a row)."""
        conn = get_connection()
        with patch("tqqq.signals.MA_GAP_THRESHOLD", 0):
            signals = detect_crossovers(conn, TICKER)
        conn.close()

        # Sort by date
        sorted_signals = sorted(signals, key=lambda x: x["date"])

        # Check for consecutive same-type signals (shouldn't happen in theory)
        for i in range(1, len(sorted_signals)):
            current = sorted_signals[i]["signal_type"]
            previous = sorted_signals[i - 1]["signal_type"]
            # They should alternate
            assert current != previous, \
                f"Consecutive {current} signals on {sorted_signals[i-1]['date']} and {sorted_signals[i]['date']}"

    def test_signals_can_be_sorted_by_date(self):
        """Verify signals can be sorted chronologically."""
        conn = get_connection()
        with patch("tqqq.signals.MA_GAP_THRESHOLD", 0):
            signals = detect_crossovers(conn, TICKER)
        conn.close()

        # Signals may come grouped by type, but should be sortable
        sorted_signals = sorted(signals, key=lambda x: x["date"])
        dates = [s["date"] for s in sorted_signals]

        # Verify dates are valid and sortable
        assert dates == sorted(dates)
        assert len(dates) == len(signals)

    def test_filtered_signals_are_subset_of_unfiltered(self):
        """Verify that MA gap filter only removes signals, never adds them."""
        conn = get_connection()
        with patch("tqqq.signals.MA_GAP_THRESHOLD", 0):
            all_signals = detect_crossovers(conn, TICKER)
        filtered_signals = detect_crossovers(conn, TICKER)
        conn.close()

        filtered_dates = {(s["date"], s["signal_type"]) for s in filtered_signals}
        all_dates = {(s["date"], s["signal_type"]) for s in all_signals}

        assert filtered_dates.issubset(all_dates)
        assert len(filtered_signals) <= len(all_signals)


@requires_historical_data
class TestCurrentStatusIntegration:
    """Integration tests for current market status with real data."""

    def test_get_current_status_returns_valid_status(self):
        """Verify current status is calculated correctly."""
        conn = get_connection()
        status = get_current_status(conn, TICKER)
        conn.close()

        assert status["status"] in ["BULLISH", "BEARISH"]

    def test_current_status_has_all_fields(self):
        """Verify current status contains all required fields."""
        conn = get_connection()
        status = get_current_status(conn, TICKER)
        conn.close()

        assert "ticker" in status
        assert "date" in status
        assert "status" in status
        assert "close" in status
        assert "ma_short" in status
        assert "ma_long" in status
        assert "gap" in status

    def test_current_status_values_are_consistent(self):
        """Verify status is consistent with MA values."""
        conn = get_connection()
        status = get_current_status(conn, TICKER)
        conn.close()

        if status["status"] == "BULLISH":
            assert status["ma_short"] > status["ma_long"]
            assert status["gap"] > 0
        else:
            assert status["ma_short"] < status["ma_long"]
            assert status["gap"] < 0

    def test_gap_calculation_is_correct(self):
        """Verify gap is calculated as MA_SHORT - MA_LONG."""
        conn = get_connection()
        status = get_current_status(conn, TICKER)
        conn.close()

        expected_gap = status["ma_short"] - status["ma_long"]
        assert abs(status["gap"] - expected_gap) < 0.01


@requires_historical_data
class TestNotificationIntegration:
    """Integration tests for notification formatting with real signals."""

    def test_format_real_signals(self):
        """Verify notification formatting works with real signals."""
        conn = get_connection()
        with patch("tqqq.signals.MA_GAP_THRESHOLD", 0):
            signals = detect_crossovers(conn, TICKER)
        conn.close()

        for signal in signals[:5]:  # Test first 5 signals
            emoji, signal_name, message = format_signal_message(signal)

            assert emoji in ["🟢", "🔴"]
            assert signal["date"] in message
            assert "$" in message  # Should have dollar signs for prices

    def test_trigger_notifications_with_real_signal(self):
        """Verify notification triggering works with real signals."""
        conn = get_connection()
        with patch("tqqq.signals.MA_GAP_THRESHOLD", 0):
            signals = detect_crossovers(conn, TICKER)
        conn.close()

        if signals:
            signal = signals[0]

            # Mock all notification methods to avoid side effects
            with patch("tqqq.notifications.log_to_console") as mock_console:
                with patch("tqqq.notifications.log_to_file") as mock_file:
                    with patch("tqqq.notifications.send_macos_notification") as mock_macos:
                        with patch("tqqq.notifications.WEBHOOK_URL", ""):
                            with patch("tqqq.notifications.EMAIL_ENABLED", False):
                                trigger_all_notifications(signal, "2025-01-15 18:00:00")

                                mock_console.assert_called_once()
                                mock_file.assert_called_once()
                                mock_macos.assert_called_once()


@requires_historical_data
class TestEndToEndIntegration:
    """End-to-end integration tests simulating real usage."""

    def test_full_signal_detection_flow(self):
        """Test complete flow: load data -> detect signals -> format notifications."""
        conn = get_connection()

        # Step 1: Load and verify data
        df = load_prices(conn, TICKER)
        assert len(df) >= MA_LONG

        # Step 2: Detect signals (without filter to ensure we get some)
        with patch("tqqq.signals.MA_GAP_THRESHOLD", 0):
            signals = detect_crossovers(conn, TICKER)
        assert len(signals) > 0

        # Step 3: Get current status
        status = get_current_status(conn, TICKER)
        assert status["status"] in ["BULLISH", "BEARISH"]

        # Step 4: Format most recent signal for notification
        most_recent = sorted(signals, key=lambda x: x["date"])[-1]
        emoji, signal_name, message = format_signal_message(most_recent)

        assert emoji in ["🟢", "🔴"]
        assert most_recent["date"] in message

        conn.close()

    def test_new_signal_detection_after_saving(self):
        """Test that saved signals are not detected as new."""
        conn = get_connection()

        # Get all signals
        with patch("tqqq.signals.MA_GAP_THRESHOLD", 0):
            all_signals = detect_crossovers(conn, TICKER)

        # Check which would be "new"
        new_signals = get_new_signals(conn, TICKER, all_signals)

        # All new signals should be in the original list
        for new_sig in new_signals:
            matching = [s for s in all_signals
                       if s["date"] == new_sig["date"]
                       and s["signal_type"] == new_sig["signal_type"]]
            assert len(matching) == 1

        conn.close()


@requires_historical_data
class TestTradingSimulation:
    """Trading simulation tests using real historical data."""

    def test_trading_simulation_from_2020(self):
        """Simulate trading strategy: buy at golden cross, sell at dead cross."""
        INITIAL_CAPITAL = 10000.00
        START_DATE = "2020-01-01"

        conn = get_connection()
        df = load_prices(conn, TICKER)
        conn.close()

        # Calculate moving averages
        df["MA_SHORT"] = df["close"].rolling(window=MA_SHORT).mean()
        df["MA_LONG"] = df["close"].rolling(window=MA_LONG).mean()
        df = df.dropna()

        # Filter from start date
        df = df[df["date"] >= START_DATE].copy()

        if len(df) == 0:
            pytest.skip(f"No data available from {START_DATE}")

        # Detect crossovers
        df["short_above"] = df["MA_SHORT"] > df["MA_LONG"]
        df["prev_short_above"] = df["short_above"].shift(1)

        # Trading state
        cash = INITIAL_CAPITAL
        shares = 0.0
        position = "CASH"
        trades = []

        for _, row in df.iterrows():
            # Golden Cross - BUY signal
            if row["short_above"] == True and row["prev_short_above"] == False:
                if position == "CASH" and cash > 0:
                    shares = cash / row["close"]
                    trades.append({"action": "BUY", "price": row["close"], "value": cash})
                    cash = 0
                    position = "HOLDING"

            # Dead Cross - SELL signal
            elif row["short_above"] == False and row["prev_short_above"] == True:
                if position == "HOLDING" and shares > 0:
                    sell_value = shares * row["close"]
                    trades.append({"action": "SELL", "price": row["close"], "value": sell_value})
                    cash = sell_value
                    shares = 0
                    position = "CASH"

        # Calculate final portfolio value
        last_price = df.iloc[-1]["close"]
        final_value = (shares * last_price) if position == "HOLDING" else cash

        assert final_value > 0, "Final portfolio value should be positive"
        assert len(trades) >= 2, "Should have at least one buy and one sell"

    def test_buy_and_hold_simulation(self):
        """Simulate buy-and-hold strategy for comparison."""
        INITIAL_CAPITAL = 10000.00
        START_DATE = "2020-01-01"

        conn = get_connection()
        df = load_prices(conn, TICKER)
        conn.close()

        df = df[df["date"] >= START_DATE].copy()

        if len(df) == 0:
            pytest.skip(f"No data available from {START_DATE}")

        first_price = df.iloc[0]["close"]
        last_price = df.iloc[-1]["close"]

        shares = INITIAL_CAPITAL / first_price
        final_value = shares * last_price

        assert final_value > 0, "Final portfolio value should be positive"
        assert shares > 0, "Should have purchased shares"


@requires_historical_data
class TestDataQualityIntegration:
    """Tests for data quality and consistency."""

    def test_no_missing_trading_days(self):
        """Check for unusual gaps in trading days."""
        conn = get_connection()
        df = load_prices(conn, TICKER)
        conn.close()

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Calculate gaps between consecutive days
        df["gap"] = df["date"].diff()

        # Most gaps should be 1-3 days (weekends, holidays)
        # Gaps > 5 days are suspicious
        max_gap = df["gap"].max()
        assert max_gap.days <= 10, f"Suspicious gap of {max_gap.days} days found"

    def test_prices_are_positive(self):
        """Verify all prices are positive."""
        conn = get_connection()
        df = load_prices(conn, TICKER)
        conn.close()

        assert (df["close"] > 0).all(), "Found non-positive price values"

    def test_no_extreme_daily_changes(self):
        """Check for unrealistic daily price changes."""
        conn = get_connection()
        df = load_prices(conn, TICKER)
        conn.close()

        df = df.sort_values("date")
        df["pct_change"] = df["close"].pct_change().abs()

        # TQQQ is 3x leveraged, so 30% daily moves are possible but rare
        # 50%+ would be extremely unusual
        max_change = df["pct_change"].max()
        assert max_change < 0.5, f"Suspicious daily change of {max_change:.1%}"
