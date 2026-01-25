from datetime import datetime
import re


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")


def detect_language(text: str) -> str:
    """
    Detect the primary language of the text.
    Returns 'zh' for Chinese, 'en' for English, or 'auto' if uncertain.
    """
    if not text:
        return "auto"
    
    # Check for Chinese characters (CJK unified ideographs)
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    has_chinese = bool(chinese_pattern.search(text))
    
    # Count Chinese vs English characters
    chinese_chars = len(chinese_pattern.findall(text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    # If text contains Chinese characters, likely Chinese
    if has_chinese:
        return "zh"
    # If mostly English characters, likely English
    elif english_chars > chinese_chars * 2:
        return "en"
    # Default to auto (let AI decide)
    else:
        return "auto"
