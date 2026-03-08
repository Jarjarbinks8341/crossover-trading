#!/usr/bin/env python3
"""Compare MACD vs MA5/MA30 crossover strategies via simulation."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from tqqq.config import MA_SHORT, MA_LONG, TICKER
from tqqq.database import get_connection, load_prices


def add_ma_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA5/MA30 crossover signals."""
    df["MA5"] = df["close"].rolling(window=MA_SHORT).mean()
    df["MA30"] = df["close"].rolling(window=MA_LONG).mean()
    df["ma_bullish"] = df["MA5"] > df["MA30"]
    df["ma_prev_bullish"] = df["ma_bullish"].shift(1)
    return df


def add_macd_signals(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """Add MACD signals (EMA12/EMA26 with 9-period signal line)."""
    df["EMA_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["EMA_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = df["EMA_fast"] - df["EMA_slow"]
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    df["macd_bullish"] = df["MACD"] > df["MACD_signal"]
    df["macd_prev_bullish"] = df["macd_bullish"].shift(1)
    return df


def simulate_trades(df: pd.DataFrame, bullish_col: str, prev_bullish_col: str):
    """Simulate buy/sell trades and calculate returns.

    Strategy: buy on bullish crossover, sell on bearish crossover.
    Returns list of trades and summary stats.
    """
    trades = []
    position = None  # None = no position, dict = open position

    for _, row in df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")

        # Buy signal
        if row[bullish_col] and not row[prev_bullish_col]:
            if position is None:
                position = {"buy_date": date_str, "buy_price": row["close"]}

        # Sell signal
        elif not row[bullish_col] and row[prev_bullish_col]:
            if position is not None:
                ret = (row["close"] - position["buy_price"]) / position["buy_price"] * 100
                trades.append({
                    "buy_date": position["buy_date"],
                    "sell_date": date_str,
                    "buy_price": position["buy_price"],
                    "sell_price": row["close"],
                    "return_pct": ret,
                })
                position = None

    # Handle open position at end of period
    if position is not None:
        last = df.iloc[-1]
        ret = (last["close"] - position["buy_price"]) / position["buy_price"] * 100
        trades.append({
            "buy_date": position["buy_date"],
            "sell_date": last["date"].strftime("%Y-%m-%d") + " (open)",
            "buy_price": position["buy_price"],
            "sell_price": last["close"],
            "return_pct": ret,
        })

    return trades


def print_signals(df: pd.DataFrame, strategy_name: str, bullish_col: str, prev_bullish_col: str):
    """Print all crossover signals."""
    signals = []
    for _, row in df.iterrows():
        if row[bullish_col] and not row[prev_bullish_col]:
            signals.append((row["date"].strftime("%Y-%m-%d"), "BUY (bullish crossover)", row["close"]))
        elif not row[bullish_col] and row[prev_bullish_col]:
            signals.append((row["date"].strftime("%Y-%m-%d"), "SELL (bearish crossover)", row["close"]))

    print(f"\n  {strategy_name} Signals:")
    if not signals:
        print("    No signals in this period.")
    for date, sig_type, price in signals:
        marker = "🟢" if "BUY" in sig_type else "🔴"
        print(f"    {marker} {date}  {sig_type:<30s} ${price:.2f}")


def print_trades(trades, strategy_name):
    """Print trade summary."""
    print(f"\n  {strategy_name} Trades:")
    if not trades:
        print("    No trades.")
        return

    print(f"    {'Buy Date':<14s} {'Sell Date':<20s} {'Buy':>8s} {'Sell':>8s} {'Return':>8s}")
    print(f"    {'-'*62}")
    for t in trades:
        ret_str = f"{t['return_pct']:+.2f}%"
        print(f"    {t['buy_date']:<14s} {t['sell_date']:<20s} ${t['buy_price']:>7.2f} ${t['sell_price']:>7.2f} {ret_str:>8s}")

    total_return = sum(t["return_pct"] for t in trades)
    wins = sum(1 for t in trades if t["return_pct"] > 0)
    losses = sum(1 for t in trades if t["return_pct"] <= 0)
    print(f"    {'-'*62}")
    print(f"    Total trades: {len(trades)}  |  Wins: {wins}  |  Losses: {losses}  |  Sum of returns: {total_return:+.2f}%")

    if wins + losses > 0:
        print(f"    Win rate: {wins/(wins+losses)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Compare MACD vs MA5/MA30 crossover strategies")
    parser.add_argument("--start", default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-03-08", help="End date (YYYY-MM-DD)")
    parser.add_argument("--ticker", default=TICKER, help=f"Ticker (default: {TICKER})")
    parser.add_argument("--macd-fast", type=int, default=12, help="MACD fast EMA period (default: 12)")
    parser.add_argument("--macd-slow", type=int, default=26, help="MACD slow EMA period (default: 26)")
    parser.add_argument("--macd-signal", type=int, default=9, help="MACD signal period (default: 9)")
    args = parser.parse_args()

    ticker = args.ticker.upper()

    conn = get_connection()
    df = load_prices(conn, ticker)
    conn.close()

    if len(df) == 0:
        print(f"No data for {ticker}")
        return

    # Calculate indicators on full dataset (need history for MAs/EMAs)
    df = add_ma_signals(df)
    df = add_macd_signals(df, fast=args.macd_fast, slow=args.macd_slow, signal=args.macd_signal)
    df = df.dropna(subset=["MA30"])  # MA30 needs most history

    # Filter to date range for display and trading
    mask = (df["date"] >= args.start) & (df["date"] <= args.end)
    data = df[mask].copy()

    if len(data) == 0:
        print(f"No data for {ticker} in range {args.start} to {args.end}")
        return

    first_date = data.iloc[0]["date"].strftime("%Y-%m-%d")
    last_date = data.iloc[-1]["date"].strftime("%Y-%m-%d")
    first_close = data.iloc[0]["close"]
    last_close = data.iloc[-1]["close"]
    buy_hold_return = (last_close - first_close) / first_close * 100

    print("=" * 70)
    print(f"  {ticker} STRATEGY COMPARISON: {args.start} to {args.end}")
    print(f"  MACD Settings: EMA({args.macd_fast}/{args.macd_slow}), Signal({args.macd_signal})")
    print("=" * 70)

    # --- Signals ---
    print_signals(data, "MA5/MA30", "ma_bullish", "ma_prev_bullish")
    print_signals(data, f"MACD({args.macd_fast}/{args.macd_slow}/{args.macd_signal})", "macd_bullish", "macd_prev_bullish")

    # --- Trades ---
    ma_trades = simulate_trades(data, "ma_bullish", "ma_prev_bullish")
    macd_trades = simulate_trades(data, "macd_bullish", "macd_prev_bullish")

    print("\n" + "=" * 70)
    print("  TRADE RESULTS")
    print("=" * 70)

    print_trades(ma_trades, "MA5/MA30")
    print_trades(macd_trades, f"MACD({args.macd_fast}/{args.macd_slow}/{args.macd_signal})")

    # --- Buy & Hold comparison ---
    print("\n" + "=" * 70)
    print("  BUY & HOLD COMPARISON")
    print("=" * 70)
    print(f"    Period: {first_date} to {last_date}")
    print(f"    Start price: ${first_close:.2f}  |  End price: ${last_close:.2f}")
    print(f"    Buy & Hold return: {buy_hold_return:+.2f}%")

    ma_total = sum(t["return_pct"] for t in ma_trades) if ma_trades else 0
    macd_total = sum(t["return_pct"] for t in macd_trades) if macd_trades else 0
    print(f"    MA5/MA30 return:   {ma_total:+.2f}%")
    print(f"    MACD return:       {macd_total:+.2f}%")
    print()


if __name__ == "__main__":
    main()
