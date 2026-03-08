"""Crossover signal detection."""

import sqlite3
from typing import List, Dict

import pandas as pd

from .config import MA_SHORT, MA_LONG, MA_GAP_THRESHOLD
from .database import load_prices


def detect_crossovers(conn: sqlite3.Connection, ticker: str) -> List[Dict]:
    """Detect golden cross and dead cross signals from price data for a specific ticker.

    Golden Cross: Short MA crosses above Long MA (bullish)
    Dead Cross: Short MA crosses below Long MA (bearish)

    Signals are filtered by MA gap threshold to avoid whipsaws in choppy
    ("monkey") markets. A crossover is only reported when the gap between
    MA_SHORT and MA_LONG is at least MA_GAP_THRESHOLD percent of the price.

    The filter is position-aware: after skipping a weak crossover, subsequent
    crossovers of the same type are suppressed until a valid opposite signal
    occurs. This ensures signals always alternate (buy, sell, buy, sell...).

    Args:
        conn: Database connection.
        ticker: Stock ticker symbol.

    Returns:
        List of signal dictionaries with ticker, date, type, and price info.
    """
    df = load_prices(conn, ticker)

    if len(df) < MA_LONG:
        return []

    # Calculate moving averages
    df["MA_SHORT"] = df["close"].rolling(window=MA_SHORT).mean()
    df["MA_LONG"] = df["close"].rolling(window=MA_LONG).mean()
    df = df.dropna()

    # Detect crossovers
    df["short_above"] = df["MA_SHORT"] > df["MA_LONG"]
    df["prev_short_above"] = df["short_above"].shift(1)

    # MA gap as percentage of price (monkey market filter)
    df["ma_gap_pct"] = ((df["MA_SHORT"] - df["MA_LONG"]) / df["close"]).abs() * 100

    signals = []
    last_signal_type = None  # Track last emitted signal to enforce alternation

    # Process all crossovers in chronological order
    crossovers = df[
        ((df["short_above"] == True) & (df["prev_short_above"] == False))
        | ((df["short_above"] == False) & (df["prev_short_above"] == True))
    ]

    for _, row in crossovers.iterrows():
        is_golden = row["short_above"]
        signal_type = "GOLDEN_CROSS" if is_golden else "DEAD_CROSS"

        # Skip if gap is too small (monkey market filter)
        if MA_GAP_THRESHOLD > 0 and row["ma_gap_pct"] < MA_GAP_THRESHOLD:
            continue

        # Skip if same signal type as last emitted (enforce alternation)
        if signal_type == last_signal_type:
            continue

        signals.append({
            "ticker": ticker,
            "date": row["date"].strftime("%Y-%m-%d"),
            "signal_type": signal_type,
            "close_price": row["close"],
            "ma5": row["MA_SHORT"],
            "ma30": row["MA_LONG"]
        })
        last_signal_type = signal_type

    return signals


def get_current_status(conn: sqlite3.Connection, ticker: str) -> Dict:
    """Get current MA status and values for a specific ticker.

    Args:
        conn: Database connection.
        ticker: Stock ticker symbol.

    Returns:
        Dictionary with ticker, current status, MA values, and gap.
    """
    df = load_prices(conn, ticker)

    if len(df) < MA_LONG:
        return {"ticker": ticker, "status": "INSUFFICIENT_DATA"}

    df["MA_SHORT"] = df["close"].rolling(window=MA_SHORT).mean()
    df["MA_LONG"] = df["close"].rolling(window=MA_LONG).mean()

    last = df.iloc[-1]
    is_bullish = last["MA_SHORT"] > last["MA_LONG"]

    return {
        "ticker": ticker,
        "date": last["date"].strftime("%Y-%m-%d"),
        "status": "BULLISH" if is_bullish else "BEARISH",
        "close": last["close"],
        "ma_short": last["MA_SHORT"],
        "ma_long": last["MA_LONG"],
        "gap": last["MA_SHORT"] - last["MA_LONG"]
    }
