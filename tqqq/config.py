"""Configuration settings for TQQQ trading bot."""

import os
from pathlib import Path

# Base paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Database
DB_PATH = DATA_DIR / "trading_data.db"

# Logs
EVENTS_LOG_PATH = LOGS_DIR / "crossover_events.log"
CRON_LOG_PATH = LOGS_DIR / "cron.log"

# Moving average settings
MA_SHORT = 5
MA_LONG = 30

# Monkey market filter: ignore crossovers when |MA_SHORT - MA_LONG| / price < threshold
# Set to 0 to disable the filter
MA_GAP_THRESHOLD = float(os.environ.get("TQQQ_MA_GAP_THRESHOLD", "1.0"))

# Webhook (Slack/Discord)
WEBHOOK_URL = os.environ.get("TQQQ_WEBHOOK_URL", "")

# Email configuration (Gmail)
EMAIL_ENABLED = os.environ.get("TQQQ_EMAIL_ENABLED", "false").lower() == "true"
EMAIL_SENDER = os.environ.get("TQQQ_EMAIL_SENDER", "")
EMAIL_PASSWORD = os.environ.get("TQQQ_EMAIL_PASSWORD", "")
# Support comma-separated list; fall back to singular TQQQ_EMAIL_RECIPIENT
_recipients_raw = os.environ.get(
    "TQQQ_EMAIL_RECIPIENTS",
    os.environ.get("TQQQ_EMAIL_RECIPIENT", ""),
)
EMAIL_RECIPIENTS = [r.strip() for r in _recipients_raw.split(",") if r.strip()]
# Keep backward compat for any code referencing the singular form
EMAIL_RECIPIENT = EMAIL_RECIPIENTS[0] if EMAIL_RECIPIENTS else ""
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Stock settings
# Support multiple tickers via environment variable (comma-separated)
TICKERS = os.environ.get("TQQQ_TICKERS", "TQQQ").split(",")
TICKERS = [t.strip().upper() for t in TICKERS if t.strip()]

# Keep for backward compatibility
TICKER = TICKERS[0] if TICKERS else "TQQQ"
HISTORY_DAYS = 365
