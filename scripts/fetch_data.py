#!/usr/bin/env python3
"""Main script to fetch stock data and check for crossover signals."""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqqq.config import HISTORY_DAYS, TICKERS
from tqqq.trading_calendar import is_trading_day
from tqqq.database import (
    get_connection,
    get_last_date,
    save_prices,
    get_price_count,
    get_date_range,
    get_new_signals,
    save_signals,
)
from tqqq.fetcher import fetch_prices, fetch_all_tickers_parallel
from tqqq.signals import detect_crossovers
from tqqq.notifications import trigger_all_notifications
from tqqq.fear_greed import fetch_fear_greed


def main():
    parser = argparse.ArgumentParser(description="Fetch stock data for configured tickers")
    parser.add_argument(
        "--full",
        action="store_true",
        help=f"Fetch full {HISTORY_DAYS} days (default: incremental update)",
    )
    parser.add_argument(
        "--ticker",
        help="Process specific ticker only (default: all configured tickers)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Fetch multiple tickers in parallel (faster)",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip notifications (useful when initializing new ticker history)",
    )
    parser.add_argument(
        "--skip-calendar-check",
        action="store_true",
        help="Skip NYSE trading day check (run even on holidays)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if today is a trading day
    if not args.skip_calendar_check and not args.full:
        today = datetime.now().date()
        if not is_trading_day(today):
            print(f"[{timestamp}] Today ({today}) is not a NYSE trading day. Skipping.")
            print(f"[{timestamp}] Use --skip-calendar-check to override.")
            return

    # Determine which tickers to process
    tickers_to_process = [args.ticker.upper()] if args.ticker else TICKERS

    print(f"[{timestamp}] Starting data fetch for {len(tickers_to_process)} ticker(s): {', '.join(tickers_to_process)}")

    conn = get_connection()

    # Fetch data for all tickers
    all_data = {}

    if args.parallel and len(tickers_to_process) > 1:
        # Parallel fetching for multiple tickers
        print(f"[{timestamp}] Using parallel fetching...")

        if args.full:
            all_data = fetch_all_tickers_parallel(tickers_to_process, days=HISTORY_DAYS)
        else:
            # For incremental, we need to determine start dates per ticker first
            for ticker in tickers_to_process:
                last_date = get_last_date(conn, ticker)
                if last_date:
                    start_date = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
                    all_data[ticker] = fetch_prices(ticker, start_date=start_date)
                else:
                    # No data exists, fetch full history
                    if ticker not in all_data:
                        all_data[ticker] = fetch_prices(ticker, days=HISTORY_DAYS)
    else:
        # Sequential processing
        for ticker in tickers_to_process:
            print(f"[{timestamp}] Fetching {ticker}...")

            if args.full:
                all_data[ticker] = fetch_prices(ticker, days=HISTORY_DAYS)
            else:
                last_date = get_last_date(conn, ticker)
                if last_date:
                    start_date = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
                    print(f"[{timestamp}] Incremental update from {last_date}...")
                    all_data[ticker] = fetch_prices(ticker, start_date=start_date)
                else:
                    print(f"[{timestamp}] No existing data, fetching full {HISTORY_DAYS} days...")
                    all_data[ticker] = fetch_prices(ticker, days=HISTORY_DAYS)

    # Process each ticker's data
    for ticker, df in all_data.items():
        print(f"\n{'='*60}\nProcessing {ticker}\n{'='*60}")
        print(f"[{timestamp}] Fetched {len(df)} days of data")

        # Save to database
        if len(df) > 0:
            rows = save_prices(conn, ticker, df)
            print(f"[{timestamp}] Saved {rows} rows to database")
        else:
            print(f"[{timestamp}] No new data to save")

        # Show stats
        count = get_price_count(conn, ticker)
        min_date, max_date = get_date_range(conn, ticker)
        print(f"[{timestamp}] Total rows for {ticker}: {count}")
        if min_date and max_date:
            print(f"[{timestamp}] Date range: {min_date} to {max_date}")

        # Detect and process crossover signals
        print(f"[{timestamp}] Checking for crossover signals...")
        all_signals = detect_crossovers(conn, ticker)
        new_signals = get_new_signals(conn, ticker, all_signals)

        if new_signals:
            print(f"[{timestamp}] Found {len(new_signals)} new crossover signal(s)!")
            save_signals(conn, ticker, new_signals)

            if args.no_notify:
                print(f"[{timestamp}] Skipping notifications (--no-notify flag set)")
            else:
                for signal in new_signals:
                    trigger_all_notifications(signal, timestamp)
        else:
            print(f"[{timestamp}] No new crossover signals")

    # Fear & Greed Index (once for all tickers)
    print(f"\n{'='*60}")
    fg = fetch_fear_greed()
    if fg:
        print(f"[{timestamp}] Fear & Greed Index: {fg['score']} ({fg['rating']})")

    conn.close()
    print(f"[{timestamp}] Done!")


if __name__ == "__main__":
    main()
