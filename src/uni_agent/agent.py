"""
Main agent: supervisor + chat/job sub-agents. Builds graph with store/checkpointer.
Entry points: agent(), chat_agent(), job_agent() for langgraph.json.
"""

import logging
import os
import sys
import re
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore

from uni_agent.config import (
    CHAT_BASE_URL,
    DB_URI,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    get_store_index_config,
)
from uni_agent.graphs.chat_agent import build_chat_agent
from uni_agent.graphs import build_job_agent
from uni_agent.prompts import supervisor_prompt, supervisor_prompt_no_results
from uni_agent.state import JobState, SupervisorDecision
from uni_agent.store.store_adapter import get_job_count
from uni_agent.utils import config_user_id, last_assistant_content, last_user_content

# ============ Logging ============
BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "agent.log")
LOG_RETENTION_DAYS = 7

file_handler = TimedRotatingFileHandler(
    LOG_FILE,
    when="midnight",
    interval=1,
    backupCount=LOG_RETENTION_DAYS - 1,
    encoding="utf-8",
)
file_handler.suffix = "%Y-%m-%d"
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
    )
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s",
    handlers=[file_handler, logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.info(
    "Logging initialized. Log file: %s (daily rotate at midnight, keep %s days)",
    LOG_FILE,
    LOG_RETENTION_DAYS,
)

# ============ Supervisor ============


def _get_supervisor_model():
    """Lazy initialization of supervisor model."""
    if not hasattr(_get_supervisor_model, "_model"):
        _get_supervisor_model._model = ChatOpenAI(
            model=OPENAI_MODEL,
            base_url=CHAT_BASE_URL,
            api_key=OPENAI_API_KEY,
        )
    return _get_supervisor_model._model


# Simple patterns that should always route to chat_agent (no LLM needed)
_CHAT_PATTERNS = (
    "你好", "hello", "hi", "hey", "您好", "嗨", "good morning", "good afternoon",
    "谢谢", "thanks", "thank you", "拜拜", "再见", "bye", "好的", "知道了",
    "几点", "今天", "明天", "日期", "time", "date", "weather",
)


def _is_simple_chat(text: str) -> bool:
    """Lightweight check if message is a simple greeting/chitchat that doesn't need job routing."""
    if not text:
        return False
    text_lower = text.lower().strip()
    # Very short messages that are greetings
    if len(text_lower) <= 15 and any(p in text_lower for p in ("hello", "hi", "hey", "你好", "嗨", "您好")):
        return True
    # Simple acknowledgments
    if text_lower in ("ok", "okay", "好的", "好", "👍", "👌"):
        return True
    return False


model = None  # placeholder, will use _get_supervisor_model() below


def strip_junk(msg):
    text = msg.content if hasattr(msg, "content") else str(msg)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group() if match else text

supervisor_parser = PydanticOutputParser(pydantic_object=SupervisorDecision)
robust_supervisor_parser = RunnableLambda(strip_junk) | supervisor_parser


def _supervisor_next_node(
    has_results: bool,
    last_user: str,
    last_assistant: str,
    config: RunnableConfig,
) -> str:
    """Return next_agent from supervisor decision (classify_job_detail | optimize_recommendations | llm_call)."""
    model = _get_supervisor_model()
    if not has_results:
        prompt_content = supervisor_prompt_no_results.format(
            last_user_message=last_user or "(none)",
            last_assistant_message=last_assistant or "(none)",
            format_instructions=supervisor_parser.get_format_instructions(),
        )
        output = model.invoke([HumanMessage(content=prompt_content)], config=config)
        out = robust_supervisor_parser.invoke(output)
        return "classify_job_detail" if out.route == "job_search" else "llm_call"
    prompt_content = supervisor_prompt.format(
        last_user_message=last_user or "(none)",
        last_assistant_message=last_assistant or "(none)",
        format_instructions=supervisor_parser.get_format_instructions(),
    )
    output = model.invoke([HumanMessage(content=prompt_content)], config=config)
    out = robust_supervisor_parser.invoke(output)
    return {
        "job_search": "classify_job_detail",
        "optimize": "optimize_recommendations",
    }.get(out.route, "llm_call")


def _is_clarification_reply(last_assistant: str, last_user: str) -> bool:
    """
    True if last assistant asked something and user gave a short reply (continue job flow).
    Language-agnostic: uses length + question mark detection instead of Chinese keywords.
    """
    if not (last_assistant and last_user):
        return False
    a, u = last_assistant.strip(), last_user.strip()
    # Check if assistant asked a question (contains ? or question-like patterns)
    has_question = "?" in a or any(q in a.lower() for q in ("please ", "could you", "can you", "what ", "which ", "how "))
    # Short user reply suggests clarification response
    is_short_reply = len(u) <= 200
    return has_question and is_short_reply


def supervisor(
    state: JobState, config: RunnableConfig, *, store: BaseStore | None = None
) -> dict:
    """Route to chat_agent or job_agent. When no results and last turn was clarification Q&A,
    go to job_agent so user's reply (e.g. in chat) continues the flow without mis-route.
    Lightweight fallback: simple greetings/chitchat routes directly to chat_agent without LLM call.
    """
    messages = state.get("messages", [])
    last_user = last_user_content(messages)[:800]
    last_assistant = last_assistant_content(messages)[:500]

    # Lightweight fallback: simple chat patterns don't need LLM routing
    if _is_simple_chat(last_user):
        return {"next_agent": "llm_call"}

    has_results = state.get("has_job_results") or bool(
        (state.get("mcp_job_results") or "").strip()
    )
    user_id = config_user_id(config)
    if not has_results and user_id and store:
        has_results = get_job_count(store, user_id) > 0
    if not has_results and _is_clarification_reply(last_assistant, last_user):
        return {"next_agent": "classify_job_detail"}
    next_node = _supervisor_next_node(has_results, last_user, last_assistant, config)
    return {"next_agent": next_node}


# ============ Graph build ============
def build_agent(store: BaseStore, checkpointer: PostgresSaver) -> CompiledStateGraph:
    """
    Build main graph: supervisor -> chat_agent | job_agent (both to END).
    Chat and job subgraphs are self-contained modules; job subgraph needs store.
    """
    job_graph = build_job_agent(store)
    chat_graph = build_chat_agent(store)
    builder = StateGraph(JobState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("chat_agent", chat_graph)
    builder.add_node("job_agent", job_graph)

    builder.add_edge(START, "supervisor")

    def _route(state: JobState) -> str:
        next_agent = state.get("next_agent") or "job_agent"
        if next_agent == "llm_call":
            return "chat_agent"
        return "job_agent"

    builder.add_conditional_edges(
        "supervisor",
        _route,
        {"chat_agent": "chat_agent", "job_agent": "job_agent"},
    )
    builder.add_edge("chat_agent", END)
    builder.add_edge("job_agent", END)
    return builder.compile(checkpointer=checkpointer, store=store)


# ============ Entry points (langgraph.json) ============


def _build_store_and_checkpointer():
    """
    Factory: create and set up PostgresStore + PostgresSaver.
    Returns (store, checkpointer, store_ref, checkpointer_ref) where refs are
    the context managers that need to be kept alive by the caller.
    """
    store_ref = PostgresStore.from_conn_string(DB_URI, index=get_store_index_config())
    store = store_ref.__enter__()
    checkpointer_ref = PostgresSaver.from_conn_string(DB_URI)
    checkpointer = checkpointer_ref.__enter__()
    store.setup()
    checkpointer.setup()
    return store, checkpointer, store_ref, checkpointer_ref


def _attach_refs(graph, store_ref, checkpointer_ref):
    """Attach store/checkpointer refs to graph for lifecycle management."""
    graph._store_ref = store_ref
    graph._checkpointer_ref = checkpointer_ref
    return graph


def agent(config: RunnableConfig | None = None) -> CompiledStateGraph:
    """Return compiled main graph; store/checkpointer created here for persistence."""
    store, checkpointer, store_ref, checkpointer_ref = _build_store_and_checkpointer()
    graph = build_agent(store, checkpointer)
    return _attach_refs(graph, store_ref, checkpointer_ref)


def chat_agent(config: RunnableConfig | None = None) -> CompiledStateGraph:
    """Standalone chat subgraph with store (RAG) and checkpointer (thread persistence)."""
    store, checkpointer, store_ref, checkpointer_ref = _build_store_and_checkpointer()
    graph = build_chat_agent(store=store, checkpointer=checkpointer)
    return _attach_refs(graph, store_ref, checkpointer_ref)


def job_agent(config: RunnableConfig | None = None) -> CompiledStateGraph:
    """Standalone job subgraph with store and checkpointer (thread persistence)."""
    store, checkpointer, store_ref, checkpointer_ref = _build_store_and_checkpointer()
    graph = build_job_agent(store, checkpointer=checkpointer)
    return _attach_refs(graph, store_ref, checkpointer_ref)
