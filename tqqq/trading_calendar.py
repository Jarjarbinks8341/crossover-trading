"""NYSE trading calendar utilities.

Determines NYSE trading days based on federal holidays and market-specific
closures. No external dependencies required.
"""

from datetime import date, timedelta


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Get the nth occurrence of a weekday in a month.

    Args:
        year: Year.
        month: Month (1-12).
        weekday: Day of week (0=Monday, 6=Sunday).
        n: Which occurrence (1=first, -1=last).
    """
    if n > 0:
        first = date(year, month, 1)
        # Days until the target weekday
        days_ahead = (weekday - first.weekday()) % 7
        first_occurrence = first + timedelta(days=days_ahead)
        return first_occurrence + timedelta(weeks=n - 1)
    else:
        # Last occurrence
        if month == 12:
            last = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last = date(year, month + 1, 1) - timedelta(days=1)
        days_back = (last.weekday() - weekday) % 7
        return last - timedelta(days=days_back)


def _observed_holiday(d: date) -> date:
    """Adjust holiday to observed date (Fri if Sat, Mon if Sun)."""
    if d.weekday() == 5:  # Saturday -> Friday
        return d - timedelta(days=1)
    elif d.weekday() == 6:  # Sunday -> Monday
        return d + timedelta(days=1)
    return d


def get_nyse_holidays(year: int) -> set:
    """Get all NYSE holiday dates for a given year.

    NYSE holidays:
    - New Year's Day (Jan 1)
    - Martin Luther King Jr. Day (3rd Monday in January)
    - Presidents' Day (3rd Monday in February)
    - Good Friday (Friday before Easter)
    - Memorial Day (last Monday in May)
    - Juneteenth (June 19, observed since 2022)
    - Independence Day (July 4)
    - Labor Day (1st Monday in September)
    - Thanksgiving Day (4th Thursday in November)
    - Christmas Day (December 25)
    """
    holidays = set()

    # New Year's Day
    holidays.add(_observed_holiday(date(year, 1, 1)))

    # MLK Day: 3rd Monday in January
    holidays.add(_nth_weekday(year, 1, 0, 3))  # Monday=0

    # Presidents' Day: 3rd Monday in February
    holidays.add(_nth_weekday(year, 2, 0, 3))

    # Good Friday: 2 days before Easter Sunday
    holidays.add(_easter(year) - timedelta(days=2))

    # Memorial Day: Last Monday in May
    holidays.add(_nth_weekday(year, 5, 0, -1))

    # Juneteenth (observed since 2022)
    if year >= 2022:
        holidays.add(_observed_holiday(date(year, 6, 19)))

    # Independence Day
    holidays.add(_observed_holiday(date(year, 7, 4)))

    # Labor Day: 1st Monday in September
    holidays.add(_nth_weekday(year, 9, 0, 1))

    # Thanksgiving: 4th Thursday in November
    holidays.add(_nth_weekday(year, 11, 3, 4))  # Thursday=3

    # Christmas
    holidays.add(_observed_holiday(date(year, 12, 25)))

    return holidays


def _easter(year: int) -> date:
    """Calculate Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def is_trading_day(check_date: date = None) -> bool:
    """Check if a given date is a NYSE trading day.

    Args:
        check_date: Date to check. Defaults to today.

    Returns:
        True if the date is a trading day (market open), False otherwise.
    """
    if check_date is None:
        check_date = date.today()

    # Weekends
    if check_date.weekday() >= 5:
        return False

    # Holidays
    holidays = get_nyse_holidays(check_date.year)
    if check_date in holidays:
        return False

    return True


def next_trading_day(after_date: date = None) -> date:
    """Get the next trading day after the given date.

    Args:
        after_date: Starting date. Defaults to today.

    Returns:
        The next NYSE trading day.
    """
    if after_date is None:
        after_date = date.today()

    candidate = after_date + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate
