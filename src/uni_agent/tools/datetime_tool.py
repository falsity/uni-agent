"""Tool to get current date and time."""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool


# Default timezone for display (can be overridden via env if needed)
_DEFAULT_TZ = "Asia/Shanghai"


@tool
def get_current_datetime(timezone: str = _DEFAULT_TZ) -> str:
    """Get the current date and time.

    Use this when the user asks about today's date, current time, what day it is,
    or when you need to reference the current moment (e.g. for greetings, deadlines).

    Args:
        timezone: IANA timezone name (e.g. Asia/Shanghai, America/New_York). Defaults to Asia/Shanghai.

    Returns:
        Current date and time in human-readable format, with weekday.
    """
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo(_DEFAULT_TZ)
    now = datetime.now(tz)
    # e.g. 2025-02-13 16:30:00 CST 星期四
    weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')} {now.tzname()} {weekday_cn}"
