"""
Chat sub-agent: general Q&A, greetings, off-topic; can use tools.
Uses short-term memory (recent messages) and Mem0 for cross-session user memory (persistent storage + retrieval-augmented context).
Self-contained: config, nodes, graph construction. Exports build_chat_agent(store, checkpointer).
When used as a subgraph, parent graph's checkpointer persists state; when standalone, pass checkpointer.
"""

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

from uni_agent.config import (
    CHAT_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SAFE_MESSAGE_TOKENS,
    TOOL_RESULT_MAX_CHARS,
)
from uni_agent.memory.mem0_adapter import (
    add_messages_to_mem0,
    get_mem0_context_for_query,
    get_messages_from_mem0,
    is_mem0_available,
)
from uni_agent.prompts import llm_call_prompt
from uni_agent.state import JobState
from uni_agent.tools.datetime_tool import get_current_datetime
from uni_agent.tools.tavily_search import tavily_search
from uni_agent.tools.think_tool import job_search_think_tool
from uni_agent.utils import (
    config_user_id,
    has_tool_calls,
    last_user_content,
    parse_tool_call,
    tool_result_to_content,
    truncate_messages_list,
)

logger = logging.getLogger(__name__)

# ===== CONFIGURATION =====
_tools = [tavily_search, job_search_think_tool, get_current_datetime]
_tool_map = {t.name: t for t in _tools}


def _get_model():
    """Lazy initialization of LLM to avoid import-time connection."""
    if not hasattr(_get_model, "_model"):
        _get_model._model = ChatOpenAI(
            model=OPENAI_MODEL,
            base_url=CHAT_BASE_URL,
            api_key=OPENAI_API_KEY,
        )
    return _get_model._model


def _build_rag_context(store: BaseStore | None, user_id: str | None, query: str) -> str:
    """Build RAG context from Mem0 (cross-session user memory) only."""
    if not user_id:
        return ""
    if not is_mem0_available():
        return ""
    mem0_ctx = get_mem0_context_for_query(user_id, query or "用户偏好 工作偏好")
    if not mem0_ctx.strip():
        return ""
    return "[Mem0 cross-session user memory]\n" + mem0_ctx.strip()


# ===== NODES =====
def _llm_call(
    state: JobState, config: RunnableConfig, *, store: BaseStore | None = None
) -> dict:
    """LLM with tools; injects Mem0 RAG context via get_messages_from_mem0; fallback without memory on error."""
    messages = truncate_messages_list(state.get("messages", []), max_tokens=SAFE_MESSAGE_TOKENS)
    user_id = config_user_id(config)

    full_messages = get_messages_from_mem0(messages, user_id)
    model_with_tools = _get_model().bind_tools(_tools)
    response = model_with_tools.invoke(full_messages, config=config)
    return {"messages": [response]}


def _tool_node(
    state: JobState, config: RunnableConfig, *, store: BaseStore | None = None
) -> dict:
    """Execute tool calls from last AIMessage; return ToolMessages."""
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    if not last_msg or not has_tool_calls(last_msg):
        return {"messages": []}
    tool_messages = []
    for tool_call in last_msg.tool_calls:
        tool_name, tool_args, tool_id = parse_tool_call(tool_call)
        if not tool_name:
            continue
        if tool_name in _tool_map:
            try:
                result = _tool_map[tool_name].invoke(tool_args)
                content = tool_result_to_content(result, TOOL_RESULT_MAX_CHARS)
                tool_messages.append(
                    ToolMessage(
                        content=content, tool_call_id=tool_id, name=tool_name
                    )
                )
            except Exception as e:  # pylint: disable=broad-except
                import traceback
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: {e}\n{traceback.format_exc()}",
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )
        else:
            tool_messages.append(
                ToolMessage(
                    content=f"Tool {tool_name} not found. Available: {list(_tool_map.keys())}",
                    tool_call_id=tool_id,
                    name=tool_name,
                )
            )
    return {"messages": tool_messages}


def _mem0_persist_node(state: JobState, config: RunnableConfig) -> dict:
    """
    Persist last user+assistant turn to Mem0 for cross-session memory.
    Runs when LLM finishes without tool calls; Mem0 infers facts from the exchange.
    Same pattern as reference: build [user, assistant] interaction and mem0.add(..., user_id).
    """
    user_id = config_user_id(config)
    if not user_id or not is_mem0_available():
        return {}
    messages = state.get("messages", []) or []
    # Collect last HumanMessage and the AIMessage that follows (this turn)
    turn: list = []
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            if not turn:
                turn.append(m)
            else:
                break
        elif isinstance(m, HumanMessage) and len(turn) == 1:
            turn.append(m)
            break
    if len(turn) != 2:
        return {}
    # Order: human first, then assistant (same as reference interaction format)
    turn = [turn[1], turn[0]]
    try:
        result = add_messages_to_mem0(user_id, turn)
        if result is not None and isinstance(result.get("results"), list):
            logger.debug(
                "Memory saved: %d memories added",
                len(result["results"]),
            )
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Error saving memory: %s", e)
    return {}


def _route_after_llm(state: JobState) -> str:
    """Route to tool_node if last message has tool_calls, else to mem0_persist then END."""
    messages = state.get("messages", [])
    if not messages or not has_tool_calls(messages[-1]):
        return "mem0_persist"
    return "tool_node"


# ===== GRAPH CONSTRUCTION =====
def build_chat_agent(
    store: BaseStore | None = None,
    checkpointer: "BaseCheckpointSaver | None" = None,
) -> CompiledStateGraph:
    """
    Build and compile the chat subgraph with optional store and checkpointer.

    - store: optional; passed to compiled graph for use by parent or future features.
    - checkpointer: for persisting conversation by thread_id. Omit when chat is used as
      a subgraph (parent graph's checkpointer handles state); pass when running standalone.
    - Mem0: cross-session user memory (retrieve before LLM, persist after each turn).
    """
    builder = StateGraph(JobState)
    builder.add_node("llm_call", _llm_call)
    builder.add_node("tool_node", _tool_node)
    builder.add_node("mem0_persist", _mem0_persist_node)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call",
        _route_after_llm,
        {"tool_node": "tool_node", "mem0_persist": "mem0_persist"},
    )
    builder.add_edge("tool_node", "llm_call")
    builder.add_edge("mem0_persist", END)
    return builder.compile(store=store, checkpointer=checkpointer)


def _get_chat_agent():
    """Lazy-loaded default chat_agent instance for backward compatibility."""
    if not hasattr(_get_chat_agent, "_chat_agent"):
        _get_chat_agent._chat_agent = build_chat_agent(store=None)
    return _get_chat_agent._chat_agent


# Default no-store, no-checkpointer graph for backward compat (e.g. tests)
# Note: Changed from module-level instance to lazy-loaded to avoid import-time side effects
chat_agent: CompiledStateGraph = None  # type: ignore
