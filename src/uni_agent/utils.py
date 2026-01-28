from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)


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


def get_state_from_checkpoint(checkpointer, config: dict):
    """Get state from checkpoint by config.

    Args:
        checkpointer: PostgresSaver instance
        config: Config dict with thread_id in configurable.thread_id

    Returns:
        dict: State dictionary with keys like 'messages', 'mcp_job_results', 'job_brief', etc.
              Returns None if checkpoint not found or error occurs.
    """
    try:
        thread_id = (config.get("configurable") or {}).get("thread_id", "1")
        checkpoint = checkpointer.get(config)
        if checkpoint:
            state = checkpoint.get("channel_values", {})
            logger.info(
                f"get_state_from_checkpoint: Retrieved state from thread_id={thread_id}, state keys={list(state.keys())}"
            )
            return state
        else:
            logger.warning(
                f"get_state_from_checkpoint: No checkpoint found for thread_id={thread_id}"
            )
            return None
    except Exception as e:
        logger.error(
            f"get_state_from_checkpoint: Error retrieving checkpoint - {str(e)}",
            exc_info=True,
        )
        return None
