"""
Main agent: supervisor + chat/job sub-agents. Builds graph with store/checkpointer.
Entry points: agent(), chat_agent(), job_agent() for langgraph.json.
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
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
from uni_agent.graphs import build_job_agent, chat_agent as chat_agent_graph
from uni_agent.prompts import supervisor_prompt, supervisor_prompt_no_results
from uni_agent.state import JobState, SupervisorDecision
from uni_agent.store.store_adapter import get_job_count
from uni_agent.utils import config_user_id, last_user_content

# ============ Logging ============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
model = ChatOpenAI(
    model=OPENAI_MODEL,
    base_url=CHAT_BASE_URL,
    api_key=OPENAI_API_KEY,
)
supervisor_parser = PydanticOutputParser(pydantic_object=SupervisorDecision)


def _supervisor_next_node(
    has_results: bool, last_user: str, config: RunnableConfig
) -> str:
    """Return next_agent from supervisor decision (classify_job_detail | optimize_recommendations | llm_call)."""
    if not has_results:
        prompt_content = supervisor_prompt_no_results.format(
            last_user_message=last_user or "(none)",
            format_instructions=supervisor_parser.get_format_instructions(),
        )
        output = model.invoke([HumanMessage(content=prompt_content)], config=config)
        out = supervisor_parser.invoke(output)
        return "classify_job_detail" if out.route == "job_search" else "llm_call"
    prompt_content = supervisor_prompt.format(
        last_user_message=last_user or "(none)",
        format_instructions=supervisor_parser.get_format_instructions(),
    )
    output = model.invoke([HumanMessage(content=prompt_content)], config=config)
    out = supervisor_parser.invoke(output)
    return {
        "job_search": "classify_job_detail",
        "optimize": "optimize_recommendations",
    }.get(out.route, "llm_call")


def supervisor(
    state: JobState, config: RunnableConfig, *, store: BaseStore | None = None
) -> dict:
    """Route to chat_agent or job_agent based on LLM decision."""
    has_results = state.get("has_job_results") or bool(
        (state.get("mcp_job_results") or "").strip()
    )
    user_id = config_user_id(config)
    if not has_results and user_id and store:
        has_results = get_job_count(store, user_id) > 0
    last_user = last_user_content(state.get("messages", []))[:800]
    next_node = _supervisor_next_node(has_results, last_user, config)
    return {"next_agent": next_node}


# ============ Graph build ============
def build_agent(store: BaseStore, checkpointer: PostgresSaver) -> CompiledStateGraph:
    """
    Build main graph: supervisor -> chat_agent | job_agent (both to END).
    Chat and job subgraphs are self-contained modules; job subgraph needs store.
    """
    job_graph = build_job_agent(store)
    builder = StateGraph(JobState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("chat_agent", chat_agent_graph)
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
def agent(config: RunnableConfig | None = None) -> CompiledStateGraph:
    """Return compiled main graph; store/checkpointer created here for persistence."""
    store_ref = PostgresStore.from_conn_string(DB_URI, index=get_store_index_config())
    store = store_ref.__enter__()
    checkpointer_ref = PostgresSaver.from_conn_string(DB_URI)
    checkpointer = checkpointer_ref.__enter__()
    store.setup()
    checkpointer.setup()
    graph = build_agent(store, checkpointer)
    graph._store_ref = store_ref
    graph._checkpointer_ref = checkpointer_ref
    return graph


def chat_agent(config: RunnableConfig | None = None) -> CompiledStateGraph:
    """Standalone chat subgraph (no supervisor)."""
    return chat_agent_graph


def job_agent(config: RunnableConfig | None = None) -> CompiledStateGraph:
    """Standalone job subgraph; creates store and builds job graph."""
    store_ref = PostgresStore.from_conn_string(DB_URI, index=get_store_index_config())
    store = store_ref.__enter__()
    store.setup()
    graph = build_job_agent(store)
    graph._store_ref = store_ref
    return graph


# ============ Demos ============
def run_demo() -> None:
    """Run three-step demo with context manager for store/checkpointer."""
    store_index = get_store_index_config()
    with (
        PostgresStore.from_conn_string(DB_URI, index=store_index) as store,
        PostgresSaver.from_conn_string(DB_URI) as checkpointer,
    ):
        store.setup()
        checkpointer.setup()
        graph = build_agent(store, checkpointer)
        run_config: RunnableConfig = {
            "configurable": {
                "thread_id": datetime.now().strftime("%Y%m%d%H%M%S"),
                "user_id": "1",
            }
        }
        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="我是一名有3年经验的软件工程师，我想找一份工作，我擅长golang和python, 请帮我搜索一下agent开发相关职位"
                    )
                ]
            },
            config=run_config,
        )
        logger.info("First invoke done. Result keys: %s", list(result.keys()))
        result = graph.invoke(
            {
                "messages": result.get("messages", [])
                + [HumanMessage(content="我希望寻找北京的agent 开发相关工作, 其他条件不限")]
            },
            config=run_config,
        )
        logger.info("Second invoke done. Result keys: %s", list(result.keys()))
        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content="按薪资从高到低重新排一下，只保留25k以上的")
                ]
            },
            config=run_config,
        )
        logger.info("Third invoke done. Result keys: %s", list(result.keys()))


async def stream_messages_demo() -> None:
    """Stream LLM tokens for typewriter effect (graph.astream stream_mode='messages')."""
    store_index = get_store_index_config()
    with (
        PostgresStore.from_conn_string(DB_URI, index=store_index) as store,
        PostgresSaver.from_conn_string(DB_URI) as checkpointer,
    ):
        store.setup()
        checkpointer.setup()
        graph = build_agent(store, checkpointer)
        run_config: RunnableConfig = {
            "configurable": {
                "thread_id": datetime.now().strftime("%Y%m%d%H%M%S"),
                "user_id": "1",
            }
        }
        inputs = {
            "messages": [HumanMessage(content="你好，请简单介绍一下你自己")]
        }
        print("Streaming tokens (typewriter style): ", end="", flush=True)
        async for msg_chunk, metadata in graph.astream(
            inputs, config=run_config, stream_mode="messages"
        ):
            if getattr(msg_chunk, "content", None):
                print(msg_chunk.content, end="", flush=True)
        print("\nDone.")


if __name__ == "__main__":
    run_demo()
