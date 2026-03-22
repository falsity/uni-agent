"""
Mem0 adapter: cross-session user memory (persistent store + RAG).

Follows official usage: add(messages, user_id=...), search(query, user_id=..., limit=...).
- get_mem0_context_for_query: search by user_id + query, returns context string for RAG.
- add_messages_to_mem0: add last turn [user, assistant] so Mem0 can infer and store facts.
"""

import logging
from typing import Any, List

from langchain_core.messages import SystemMessage
from mem0 import Memory

from uni_agent.config import (
    EMBED_BASE_URL,
    MEM0_DISABLED,
    MEM0_SEARCH_LIMIT,
    get_mem0_config,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Lazy singleton Memory instance (None when disabled or import error)
_memory_client: Any = None
_memory_available: bool | None = None

# One-time patch for vLLM: no response_format on fact-extraction
_mem0_llm_patched = False
# One-time patch for fixed-dimension embedders (e.g. Qwen3-Embedding): do not pass dimensions to API
_mem0_embedder_patched = False


def _patch_mem0_embedder_no_dims(memory_client: Memory) -> None:
    """Do not pass dimensions to embedder API; required for models that do not support matryoshka (e.g. Qwen3-Embedding-0.6B)."""
    global _mem0_embedder_patched
    if _mem0_embedder_patched:
        return
    emb = getattr(memory_client, "embedding_model", None)
    if not emb or not getattr(emb, "client", None) or not hasattr(emb.client, "embeddings"):
        return
    try:
        _original_embed = emb.embed

        def _embed_no_dims(text, memory_action=None):
            text = text.replace("\n", " ")
            # Call API without dimensions so fixed-dimension models (Qwen3-Embedding) do not get BadRequestError
            resp = emb.client.embeddings.create(input=[text], model=emb.config.model)
            return resp.data[0].embedding

        emb.embed = _embed_no_dims
        _mem0_embedder_patched = True
        logger.debug("Mem0: patched embedder to not pass dimensions (fixed-dimension model).")
    except Exception as e:  # pylint: disable=broad-except
        logger.debug("Mem0: could not patch embedder for fixed dims: %s", e)


def _patch_mem0_llm_for_vllm(memory_client: Memory) -> None:
    """Disable response_format for fact-extraction so vLLM returns proper JSON."""
    global _mem0_llm_patched
    if _mem0_llm_patched or not getattr(memory_client, "llm", None):
        return
    try:
        _llm = memory_client.llm
        _original = _llm.generate_response

        def _is_fact_extraction_call(messages):
            return (
                isinstance(messages, (list, tuple))
                and len(messages) >= 2
                and isinstance(messages[0], dict)
                and messages[0].get("role") == "system"
                and isinstance(messages[1], dict)
                and messages[1].get("role") == "user"
                and (messages[1].get("content") or "").strip().startswith("Input:")
            )

        def _patched_generate_response(messages, response_format=None, **kwargs):
            if _is_fact_extraction_call(messages):
                response_format = None
            return _original(messages=messages, response_format=response_format, **kwargs)

        _llm.generate_response = _patched_generate_response
        _mem0_llm_patched = True
    except Exception as e:  # pylint: disable=broad-except
        logger.debug("Mem0: could not patch LLM for vLLM: %s", e)


def _get_memory_client() -> Memory:
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
        config = get_mem0_config()
        _memory_client = Memory.from_config(config)
        _memory_available = True
        _patch_mem0_llm_for_vllm(_memory_client)
        _patch_mem0_embedder_no_dims(_memory_client)
        logger.info("Mem0 memory client initialized (cross-session memory enabled).")
        return _memory_client
    except Exception:  # pylint: disable=broad-except
        logger.exception(
            "Mem0 not available, cross-session memory disabled | MEM0_DISABLED=%s",
            MEM0_DISABLED,
        )
        _memory_available = False
        return None


def is_mem0_available() -> bool:
    """Return True if Mem0 is enabled and ready."""
    client = _get_memory_client()
    return client is not None

def get_messages_from_mem0(messages: list, user_id: str, limit: int | None = None) -> list:
    """Return [system_message] + messages with Mem0 context injected; or messages unchanged if no user_id/client/query."""
    if not messages or not user_id:
        return messages
    # Normalize to str: messages[-1].content can be a list (multimodal), mem0.search expects query to be str
    last_content = _message_content(messages[-1])
    if not last_content:
        return messages
    client = _get_memory_client()
    if not client:
        return messages
    k = limit if limit is not None else MEM0_SEARCH_LIMIT
    try:
        memories = client.search(query=last_content, user_id=user_id, limit=k)
        memory_list = (memories or {}).get("results") or []
    except Exception:  # pylint: disable=broad-except
        query_repr = repr(last_content) if not isinstance(last_content, str) else last_content
        if len(query_repr) > 500:
            query_repr = query_repr[:500] + "..."
        logger.exception(
            "Mem0 search failed in get_messages_from_mem0 | query=%s user_id=%s limit=%s embed_base_url=%s",
            query_repr,
            user_id,
            k,
            EMBED_BASE_URL,
        )
        return messages
    context = "Relevant information from previous conversations:\n"
    for memory in memory_list:
        if isinstance(memory, dict) and memory.get("memory"):
            context += f"- {memory['memory']}\n"
    system_message = SystemMessage(
        content="""You are a helpful customer support assistant. Use the provided context to personalize your responses and remember user preferences and past interactions.
{context}""".format(
            context=context
        )
    )
    return [system_message] + messages

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
    query_str = query.strip()
    try:
        result = client.search(query=query_str, user_id=user_id, limit=k)
        # Official format: {"results": [{"memory": "..."}, ...]}
        results = (result or {}).get("results") or []
        if not isinstance(results, list):
            return ""
        texts = [
            str(item["memory"]).strip()
            for item in results
            if isinstance(item, dict) and item.get("memory")
        ]
        return "\n".join(texts) if texts else ""
    except Exception:  # pylint: disable=broad-except
        query_repr = query_str if len(query_str) <= 500 else query_str[:500] + "..."
        logger.exception(
            "Mem0 search failed in get_mem0_context_for_query | query=%s user_id=%s limit=%s embed_base_url=%s",
            query_repr,
            user_id,
            k,
            EMBED_BASE_URL,
        )
        return ""


def _message_content(msg: Any) -> str:
    """Extract plain text from a message (supports list content e.g. multimodal)."""
    content = getattr(msg, "content", "") or ""
    if isinstance(content, list):
        content = " ".join(
            p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in content
        )
    return str(content).strip()


def _messages_to_mem0_format(messages: List[Any]) -> List[dict]:
    """Convert LangChain messages to Mem0 format: [{"role": "user"|"assistant", "content": "..."}]."""
    from langchain_core.messages import AIMessage, HumanMessage

    role_map = {HumanMessage: "user", AIMessage: "assistant"}
    out = []
    for m in messages:
        if type(m) in role_map:
            out.append({"role": role_map[type(m)], "content": _message_content(m)})
    return out


def add_messages_to_mem0(
    user_id: str, messages: List[Any], metadata: dict | None = None
) -> dict | None:
    """
    Add a conversation turn to Mem0 so it can extract and store facts (cross-session).
    messages: list of LangChain messages (HumanMessage, AIMessage) for this turn.
    Returns mem0 add result dict (e.g. {"results": [...]}) on success, None on skip/failure.
    """
    if not user_id:
        return None
    client = _get_memory_client()
    if not client:
        return None
    mem0_messages = _messages_to_mem0_format(messages)
    if len(mem0_messages) < 2:
        logger.debug("Mem0: skip add (turn has %d messages, need 2)", len(mem0_messages))
        return None
    try:
        # Same pattern as reference: add([user, assistant], user_id=...); infer=True extracts facts
        result = client.add(
            mem0_messages, user_id=user_id, metadata=metadata or {}, infer=True
        )
        results_list = result.get("results") if isinstance(result, dict) else []
        if isinstance(results_list, list):
            if results_list:
                logger.info(
                    "Mem0: memory saved for user_id=%s, %d memories added",
                    user_id,
                    len(results_list),
                )
            else:
                # mem0 catches fact-extraction errors (e.g. KeyError 'facts') and continues with empty results
                logger.warning(
                    "Mem0 add returned 0 memories (fact extraction may have failed; check mem0/mem0_adapter logs for LLM response)",
                )
        return result
    except Exception:  # pylint: disable=broad-except
        logger.exception(
            "Mem0 add failed (error saving memory) | user_id=%s messages_len=%s",
            user_id,
            len(messages),
        )
        return None
