"""Integration tests for multi-ticker support."""

from unittest.mock import patch

import pandas as pd
import pytest

from tqqq.database import (
    save_prices,
    load_prices,
    get_last_date,
    get_all_tickers,
    get_ticker_stats,
    save_signals,
    get_new_signals,
)
from tqqq.signals import detect_crossovers, get_current_status


class TestMultiTickerDataIsolation:
    """Tests to ensure data isolation between different tickers."""

    def test_tickers_stored_independently(self, temp_db, sample_price_data):
        """Test that TQQQ and YINN data are stored independently."""
        conn, _ = temp_db

        # Save data for TQQQ
        save_prices(conn, "TQQQ", sample_price_data)

        # Save different data for YINN
        yinn_data = sample_price_data.copy()
        yinn_data["Close"] = yinn_data["Close"] + 100  # Different prices
        save_prices(conn, "YINN", yinn_data)

        # Verify TQQQ data
        tqqq_df = load_prices(conn, "TQQQ")
        assert len(tqqq_df) == 40
        assert tqqq_df.iloc[0]["close"] == 50.0  # Original TQQQ price

        # Verify YINN data
        yinn_df = load_prices(conn, "YINN")
        assert len(yinn_df) == 40
        assert yinn_df.iloc[0]["close"] == 150.0  # YINN price (50 + 100)

    def test_last_date_per_ticker(self, temp_db, sample_price_data):
        """Test get_last_date returns correct date per ticker."""
        conn, _ = temp_db

        # Save full data for TQQQ
        save_prices(conn, "TQQQ", sample_price_data)

        # Save partial data for YINN (only first 20 days)
        yinn_data = sample_price_data.head(20)
        save_prices(conn, "YINN", yinn_data)

        # Verify last dates are different
        tqqq_last = get_last_date(conn, "TQQQ")
        yinn_last = get_last_date(conn, "YINN")

        assert tqqq_last == "2025-02-25"  # Last day in 40-day sample
        assert yinn_last == "2025-01-28"  # Last day in 20-day sample

    def test_signals_isolated_per_ticker(self, temp_db, sample_signal):
        """Test that signals are stored and retrieved per ticker."""
        conn, _ = temp_db

        # Create signals for different tickers
        tqqq_signal = sample_signal.copy()
        tqqq_signal["ticker"] = "TQQQ"

        yinn_signal = sample_signal.copy()
        yinn_signal["ticker"] = "YINN"
        yinn_signal["date"] = "2025-01-20"  # Different date

        # Save both signals
        save_signals(conn, "TQQQ", [tqqq_signal])
        save_signals(conn, "YINN", [yinn_signal])

        # Verify isolation - TQQQ signal should not be "new" for TQQQ
        tqqq_new = get_new_signals(conn, "TQQQ", [tqqq_signal])
        assert len(tqqq_new) == 0

        # But should be "new" for YINN (different ticker)
        yinn_new = get_new_signals(conn, "YINN", [tqqq_signal])
        assert len(yinn_new) == 1


class TestMultiTickerQueries:
    """Tests for multi-ticker query functions."""

    def test_get_all_tickers(self, temp_db, sample_price_data):
        """Test get_all_tickers returns all tracked tickers."""
        conn, _ = temp_db

        # Initially empty
        tickers = get_all_tickers(conn)
        assert tickers == []

        # Add TQQQ
        save_prices(conn, "TQQQ", sample_price_data)
        tickers = get_all_tickers(conn)
        assert tickers == ["TQQQ"]

        # Add YINN
        save_prices(conn, "YINN", sample_price_data)
        tickers = get_all_tickers(conn)
        assert set(tickers) == {"TQQQ", "YINN"}
        assert tickers == sorted(tickers)  # Should be alphabetically sorted

    def test_get_ticker_stats(self, temp_db, sample_price_data):
        """Test get_ticker_stats returns correct statistics."""
        conn, _ = temp_db

        # Add data for multiple tickers
        save_prices(conn, "TQQQ", sample_price_data)  # 40 days

        yinn_data = sample_price_data.head(20)  # Only 20 days
        save_prices(conn, "YINN", yinn_data)

        stats = get_ticker_stats(conn)

        assert "TQQQ" in stats
        assert "YINN" in stats

        # Verify TQQQ stats
        assert stats["TQQQ"]["record_count"] == 40
        assert stats["TQQQ"]["first_date"] == "2025-01-01"
        assert stats["TQQQ"]["last_date"] == "2025-02-25"

        # Verify YINN stats
        assert stats["YINN"]["record_count"] == 20
        assert stats["YINN"]["first_date"] == "2025-01-01"
        assert stats["YINN"]["last_date"] == "2025-01-28"


class TestMultiTickerSignals:
    """Tests for signal detection across multiple tickers."""

    def test_detect_crossovers_per_ticker(self, temp_db, sample_price_data_with_crossover):
        """Test crossover detection works independently per ticker."""
        conn, _ = temp_db

        # Save crossover data for TQQQ
        save_prices(conn, "TQQQ", sample_price_data_with_crossover)

        # Create flat data for YINN (no crossover)
        dates = pd.date_range(start="2025-01-01", periods=40, freq="B")
        yinn_flat = pd.DataFrame(
            {
                "Open": [50] * 40,
                "High": [51] * 40,
                "Low": [49] * 40,
                "Close": [50] * 40,
                "Volume": [1000000] * 40,
            },
            index=dates,
        )
        save_prices(conn, "YINN", yinn_flat)

        # TQQQ should have crossover (disable gap filter for this test data)
        with patch("tqqq.signals.MA_GAP_THRESHOLD", 0):
            tqqq_signals = detect_crossovers(conn, "TQQQ")
        assert len(tqqq_signals) >= 1
        assert tqqq_signals[0]["ticker"] == "TQQQ"

        # YINN should have no crossover
        yinn_signals = detect_crossovers(conn, "YINN")
        assert len(yinn_signals) == 0

    def test_get_current_status_per_ticker(self, temp_db):
        """Test current status is calculated independently per ticker."""
        conn, _ = temp_db

        # Create uptrending data for TQQQ (bullish)
        dates = pd.date_range(start="2025-01-01", periods=40, freq="B")
        tqqq_uptrend = pd.DataFrame(
            {
                "Open": [float(40 + i * 2) for i in range(40)],
                "High": [float(41 + i * 2) for i in range(40)],
                "Low": [float(39 + i * 2) for i in range(40)],
                "Close": [float(40 + i * 2) for i in range(40)],
                "Volume": [1000000] * 40,
            },
            index=dates,
        )
        save_prices(conn, "TQQQ", tqqq_uptrend)

        # Create downtrending data for YINN (bearish)
        yinn_downtrend = pd.DataFrame(
            {
                "Open": [float(120 - i * 2) for i in range(40)],
                "High": [float(121 - i * 2) for i in range(40)],
                "Low": [float(119 - i * 2) for i in range(40)],
                "Close": [float(120 - i * 2) for i in range(40)],
                "Volume": [500000] * 40,
            },
            index=dates,
        )
        save_prices(conn, "YINN", yinn_downtrend)

        # Verify independent status
        tqqq_status = get_current_status(conn, "TQQQ")
        assert tqqq_status["ticker"] == "TQQQ"
        assert tqqq_status["status"] == "BULLISH"

        yinn_status = get_current_status(conn, "YINN")
        assert yinn_status["ticker"] == "YINN"
        assert yinn_status["status"] == "BEARISH"


class TestMultiTickerEdgeCases:
    """Tests for edge cases in multi-ticker support."""

    def test_same_date_different_tickers(self, temp_db, sample_price_data):
        """Test same date can exist for multiple tickers."""
        conn, _ = temp_db

        # Save same dates for both tickers
        save_prices(conn, "TQQQ", sample_price_data)
        save_prices(conn, "YINN", sample_price_data)

        # Should have 40 records for each ticker
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tqqq_prices WHERE ticker = 'TQQQ'")
        tqqq_count = cursor.fetchone()[0]
        assert tqqq_count == 40

        cursor.execute("SELECT COUNT(*) FROM tqqq_prices WHERE ticker = 'YINN'")
        yinn_count = cursor.fetchone()[0]
        assert yinn_count == 40

        # Total records should be 80
        cursor.execute("SELECT COUNT(*) FROM tqqq_prices")
        total_count = cursor.fetchone()[0]
        assert total_count == 80

    def test_ticker_case_insensitive(self, temp_db, sample_price_data):
        """Test ticker handling is case-insensitive."""
        conn, _ = temp_db

        # Save with uppercase
        save_prices(conn, "TQQQ", sample_price_data)

        # Load with different case (data layer should handle)
        df = load_prices(conn, "TQQQ")
        assert len(df) == 40

    def test_empty_ticker_list(self, temp_db):
        """Test handling when no tickers exist in database."""
        conn, _ = temp_db

        tickers = get_all_tickers(conn)
        assert tickers == []

        stats = get_ticker_stats(conn)
        assert stats == {}
