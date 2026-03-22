"""
Shared utilities: date, language, checkpoint, and node helpers (message/tool).
Node helpers are used by chat_agent and job_agent without importing agent.
"""

import logging
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

logger = logging.getLogger(__name__)


# ---------- Message / tool helpers (for nodes) ----------
def truncate_messages_list(msgs: list, max_tokens: int = 12000) -> list:
    """Truncate messages to recent max_tokens (approximate). Used by chat/job nodes."""
    if not msgs:
        return msgs
    return trim_messages(
        msgs,
        max_tokens=max_tokens,
        token_counter=count_tokens_approximately,
        strategy="last",
        include_system=True,
        start_on="human",
        end_on=("human", "tool"),
    )


def parse_tool_call(tool_call: Any) -> tuple:
    """Return (tool_name, tool_args, tool_id) from a tool_call dict or object."""
    if isinstance(tool_call, dict):
        return (
            tool_call.get("name"),
            tool_call.get("args", {}),
            tool_call.get("id"),
        )
    return (
        getattr(tool_call, "name", None),
        getattr(tool_call, "args", {}),
        getattr(tool_call, "id", None),
    )


def has_tool_calls(msg: Any) -> bool:
    """True if message has non-empty tool_calls (e.g. AIMessage)."""
    return hasattr(msg, "tool_calls") and msg.tool_calls and len(msg.tool_calls) > 0


def tool_result_to_content(result: Any, max_chars: int | None = None) -> str:
    """Normalize tool result to string; optionally truncate."""
    content = result if isinstance(result, str) else str(result)
    if max_chars and len(content) > max_chars:
        content = content[:max_chars] + "\n\n[truncated for context limit]"
    return content


def config_user_id(config: dict | None) -> str:
    """Get user_id from config.configurable; fallback to thread_id."""
    if config is None:
        return ""
    cfg = config.get("configurable", {}) if isinstance(config, dict) else {}
    return (cfg.get("user_id") or cfg.get("thread_id") or "").strip()


def message_content_to_str(content: Any) -> str:
    """Normalize message content to string (str or list of parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            else:
                parts.append(str(part))
        return " ".join(parts).strip()
    return str(content).strip()


def last_user_content(messages: list) -> str:
    """Return content of last HumanMessage in messages."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            raw = getattr(m, "content", "") or ""
            return message_content_to_str(raw)
    return ""


def last_assistant_content(messages: list) -> str:
    """Return content of last AIMessage (for supervisor: detect clarification Q&A)."""
    from langchain_core.messages import AIMessage

    for m in reversed(messages):
        if isinstance(m, AIMessage):
            raw = getattr(m, "content", "") or ""
            return message_content_to_str(raw)
    return ""


# ---------- Date / language / checkpoint ----------


def get_today_str() -> str:
    """Get current date in a human-readable format (portable: no %-d)."""
    dt = datetime.now()
    return dt.strftime("%a %b ") + str(dt.day) + ", " + dt.strftime("%Y")


def detect_language(text: str) -> str:
    """
    Detect the primary language of the text.
    Returns 'zh' for Chinese, 'en' for English, or 'auto' if uncertain.
    """
    if not text:
        return "auto"

    # Check for Chinese characters (CJK unified ideographs)
    chinese_pattern = re.compile(r"[\u4e00-\u9fff]+")
    has_chinese = bool(chinese_pattern.search(text))

    # Count Chinese vs English characters
    chinese_chars = len(chinese_pattern.findall(text))
    english_chars = len(re.findall(r"[a-zA-Z]", text))

    # If text contains Chinese characters, likely Chinese
    if has_chinese:
        return "zh"
    # If mostly English characters, likely English
    elif english_chars > chinese_chars * 2:
        return "en"
    # Default to auto (let AI decide)
    else:
        return "auto"


def get_state_from_checkpoint(checkpointer, config: dict) -> dict:
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
