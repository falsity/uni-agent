"""
Cross-session user memory via Mem0: persistent storage and retrieval-augmented context.
"""

from uni_agent.memory.mem0_adapter import (
    add_messages_to_mem0,
    get_mem0_context_for_query,
    is_mem0_available,
)

__all__ = [
    "is_mem0_available",
    "get_mem0_context_for_query",
    "add_messages_to_mem0",
]
