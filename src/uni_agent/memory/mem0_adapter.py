"""
Mem0 adapter: external persistent memory and retrieval for chat across sessions.

- get_mem0_context_for_query: semantic search by user_id + query, returns context string for RAG.
- add_messages_to_mem0: persist recent conversation turn so future sessions can remember.
"""

import logging
from typing import Any, List

from uni_agent.config import MEM0_DISABLED, MEM0_SEARCH_LIMIT, get_mem0_config

logger = logging.getLogger(__name__)

# Lazy singleton Memory instance (None when disabled or import error)
_memory_client: Any = None
_memory_available: bool | None = None

# One-time patch applied so vLLM (Qwen3-Embedding) does not receive dimensions param
_mem0_openai_patched = False


def _patch_mem0_openai_embedder_no_dims() -> None:
    """Do not pass dimensions to embedding API so vLLM uses model native dim (e.g. 1024 for 0.6B)."""
    global _mem0_openai_patched
    if _mem0_openai_patched:
        return
    try:
        from mem0.embeddings import openai as _mem0_openai

        _orig_embed = _mem0_openai.OpenAIEmbedding.embed

        def _embed_no_dims(self: Any, text: Any, memory_action: Any = None) -> Any:
            text = text.replace("\n", " ")
            return (
                self.client.embeddings.create(input=[text], model=self.config.model)
                .data[0]
                .embedding
            )

        _mem0_openai.OpenAIEmbedding.embed = _embed_no_dims
        _mem0_openai_patched = True
        logger.debug("Mem0: patched OpenAI embedder to omit dimensions param.")
    except Exception as e:  # pylint: disable=broad-except
        logger.debug("Mem0: could not patch OpenAI embedder: %s", e)


def _get_memory_client() -> Any:
    """Lazy init Mem0 Memory; returns None if disabled or mem0ai not available."""
    global _memory_client, _memory_available
    if _memory_available is False:
        return None
    if _memory_client is not None:
        return _memory_client
    if MEM0_DISABLED:
        _memory_available = False
        return None
    try:
        _patch_mem0_openai_embedder_no_dims()
        from mem0 import Memory

        config = get_mem0_config()
        _memory_client = Memory.from_config(config)
        _memory_available = True
        logger.info("Mem0 memory client initialized (cross-session memory enabled).")
        return _memory_client
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Mem0 not available: %s. Cross-session memory disabled.", e)
        _memory_available = False
        return None


def is_mem0_available() -> bool:
    """Return True if Mem0 is enabled and ready."""
    client = _get_memory_client()
    return client is not None


def get_mem0_context_for_query(user_id: str, query: str, limit: int | None = None) -> str:
    """
    Retrieve relevant memories for the user and query (semantic search).
    Returns a single string for injection into system/context; empty if no memories or disabled.
    """
    if not user_id or not (query or "").strip():
        return ""
    client = _get_memory_client()
    if not client:
        return ""
    k = limit if limit is not None else MEM0_SEARCH_LIMIT
    try:
        result = client.search(query=query.strip(), user_id=user_id, limit=k)
        # Mem0 returns dict with "results" list; each item has "memory" text
        if not result or not isinstance(result, dict):
            return ""
        results_list = result.get("results") or result
        if isinstance(results_list, dict):
            results_list = results_list.get("results", [])
        if not isinstance(results_list, list):
            return ""
        texts = []
        for item in results_list:
            if isinstance(item, dict) and item.get("memory"):
                texts.append(str(item["memory"]).strip())
            elif hasattr(item, "memory") and getattr(item, "memory"):
                texts.append(str(getattr(item, "memory")).strip())
        if not texts:
            return ""
        return "\n".join(texts)
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Mem0 search failed: %s", e)
        return ""


def _messages_to_mem0_format(messages: List[Any]) -> List[dict]:
    """Convert LangChain-style messages to Mem0 format [{role, content}]."""
    from langchain_core.messages import AIMessage, HumanMessage

    out = []
    for m in messages:
        if isinstance(m, HumanMessage):
            content = getattr(m, "content", "") or ""
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in content
                )
            out.append({"role": "user", "content": str(content).strip()})
        elif isinstance(m, AIMessage):
            content = getattr(m, "content", "") or ""
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in content
                )
            out.append({"role": "assistant", "content": str(content).strip()})
    return out


def add_messages_to_mem0(user_id: str, messages: List[Any], metadata: dict | None = None) -> None:
    """
    Add a conversation turn to Mem0 so it can extract and store facts (cross-session).
    messages: list of LangChain messages (HumanMessage, AIMessage) for this turn.
    """
    if not user_id:
        return
    client = _get_memory_client()
    if not client:
        return
    mem0_messages = _messages_to_mem0_format(messages)
    if len(mem0_messages) < 2:
        return
    try:
        client.add(
            mem0_messages,
            user_id=user_id,
            metadata=metadata or {},
            infer=True,
        )
        logger.debug("Mem0: added %d messages for user_id=%s", len(mem0_messages), user_id)
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Mem0 add failed: %s", e)
