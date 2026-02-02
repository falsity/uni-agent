import asyncio
import traceback
from datetime import datetime
import logging
import os
import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore
from langgraph.types import Command
from uni_agent.prompts import (
    clarify_job_detail_prompt,
    optimize_recommendations_prompt,
    optimize_retrieval_prompt,
    search_jobs_agent_prompt_with_mcp,
    supervisor_prompt,
)
from uni_agent.state import (
    ClarifyJobDetail,
    JobState,
    OptimizeRetrievalParams,
    SupervisorDecision,
)
from uni_agent.store.job_store import (
    _extract_json_from_mcp_content_parts,
    parse_mcp_result_to_jobs,
)
from uni_agent.store.store_adapter import (
    get_job_count,
    get_jobs_by_search,
    get_jobs_for_prompt,
    save_job_results_async,
    save_user_preference_sync,
)
from uni_agent.tools.mcp_jobs import get_mcp_tools
from uni_agent.tools.tavily_search import tavily_search
from uni_agent.tools.think_tool import job_search_think_tool

# Configure logger to include full path and line number, and write to logs/agent_YYYYMMDD_HHMMSS.log
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(
    LOG_DIR, f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
logger.info("Logging initialized. Log file: %s", LOG_FILE)

# --- Config (deployable via env) ---
CHAT_BASE_URL = os.environ.get(
    "OPENAI_API_BASE",
    "http://192.168.0.201:9000/v1",
)
EMBED_BASE_URL = os.environ.get(
    "EMBED_BASE_URL",
    "http://192.168.0.201:9001/v1",
)
DB_URI = os.environ.get(
    "POSTGRES_URI",
    "postgresql://postgres:password@192.168.0.201:15433/postgres?sslmode=disable",
)
TIMEOUT_SECONDS = int(os.environ.get("AGENT_TIMEOUT_SECONDS", "180"))
STORE_EMBEDDING_DIMS = 1024  # Qwen3-Embedding-0.6B
_JOB_PROMPT_MAX_CHARS = 28_000

# LLM: use OPENAI_API_KEY and OPENAI_MODEL in production; default for local Qwen
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-8B")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")

tools = [tavily_search, job_search_think_tool]
model = ChatOpenAI(
    model=OPENAI_MODEL,
    base_url=CHAT_BASE_URL,
    api_key=OPENAI_API_KEY,
)
llm_with_tools = model.bind_tools(tools)


# --- Helpers ---
def _truncate_messages_for_prompt(msgs, max_per_msg=600, max_msgs=14):
    """Build a short summary of recent messages to stay within context limit."""
    out = []
    for m in list(msgs)[-max_msgs:]:
        if isinstance(m, ToolMessage):
            out.append("[ToolMessage: MCP result stored in mcp_job_results]")
        else:
            raw = str(getattr(m, "content", "") or "")
            s = raw[:max_per_msg] + ("..." if len(raw) > max_per_msg else "")
            out.append(f"{type(m).__name__}: {s}")
    return "\n---\n".join(out)


def _truncate_messages_list(msgs, max_tokens=15000):
    """
    Truncate messages list to keep only recent messages, avoiding context overflow.
    Uses LangChain's trim_messages utility which intelligently trims by token count.
    """
    if not msgs:
        return msgs

    # Use trim_messages with token-based trimming
    trimmed = trim_messages(
        msgs,
        max_tokens=max_tokens,
        token_counter="approximate",
        strategy="last",
        include_system=True,
        start_on="human",
    )
    return trimmed


def classify_job_detail(
    state: JobState, config: RunnableConfig, *, store: BaseStore | None = None
) -> Command:
    """Classify job detail and determine if clarification is needed."""
    messages = state.get("messages")
    truncated = _truncate_messages_for_prompt(messages)
    structured_output_model = model.with_structured_output(ClarifyJobDetail)
    response = structured_output_model.invoke(
        [HumanMessage(content=clarify_job_detail_prompt.format(messages=truncated))]
    )
    # Store the response in state for conditional routing
    if response.need_clarify:
        return Command(
            goto=END, update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Full update of user job preference in store (verification = requirement summary)
        user_id = _config_user_id(config)
        if user_id and store is not None and (response.verification or "").strip():
            save_user_preference_sync(store, user_id, response.verification.strip())
        return Command(
            goto="mcp_jobs_tool_call",
            update={"messages": [AIMessage(content=response.verification)]},
        )


def _config_user_id(config: RunnableConfig | None) -> str:
    """Get user_id from config.configurable; fallback to thread_id for single-user."""
    if config is None:
        return ""
    cfg = config.get("configurable", {}) if isinstance(config, dict) else {}
    return (cfg.get("user_id") or cfg.get("thread_id") or "").strip()


def _message_content_to_str(content) -> str:
    """Normalize message content to string; content may be str or list (multimodal)."""
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


def _last_user_content(messages: list) -> str:
    """Return content of last HumanMessage in messages."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            raw = getattr(m, "content", "") or ""
            return _message_content_to_str(raw)
    return ""


def _get_retrieval_params_agentic(last_user: str, job_brief: str) -> OptimizeRetrievalParams:
    """
    Agentic RAG: use LLM to parse user message into retrieval params (salary filter, sort, semantic query, limit).
    Replaces hardcoded regex with intent-based retrieval.
    """
    structured = model.with_structured_output(OptimizeRetrievalParams)
    prompt = optimize_retrieval_prompt.format(
        last_user_message=last_user or "(none)",
        job_brief=job_brief or "(none)",
    )
    out = structured.invoke([HumanMessage(content=prompt)])
    return out if isinstance(out, OptimizeRetrievalParams) else OptimizeRetrievalParams()


def _get_optimize_job_text(state, store, user_id: str, messages: list) -> str:
    """Build job text for optimize_recommendations using agentic RAG (LLM-parsed retrieval params)."""
    last_user = _last_user_content(messages)
    job_brief = state.get("job_brief") or ""
    params = _get_retrieval_params_agentic(last_user or "", job_brief)
    salary_min = params.salary_min_k
    salary_max = params.salary_max_k
    limit = params.limit or 60
    jobs = []
    if user_id and store:
        if params.semantic_query and (params.semantic_query or "").strip():
            jobs = get_jobs_by_search(
                store, user_id, query=params.semantic_query.strip(), limit=limit,
                salary_min_filter=salary_min, salary_max_filter=salary_max,
            )
        if not jobs:
            jobs = get_jobs_for_prompt(
                store, user_id, limit=limit,
                salary_min_filter=salary_min, salary_max_filter=salary_max,
            )
    if params.sort_by_salary_desc and jobs:
        def _salary_key(j):
            smax = j.get("salary_max")
            smin = j.get("salary_min")
            if smax is not None:
                return (0, -(smax if isinstance(smax, (int, float)) else 0))
            if smin is not None:
                return (1, -(smin if isinstance(smin, (int, float)) else 0))
            return (2, 0)
        jobs = sorted(jobs, key=_salary_key)
    if jobs:
        return _format_jobs_for_prompt(jobs, max_chars=_JOB_PROMPT_MAX_CHARS)
    mcp_raw = state.get("mcp_job_results") or ""
    if mcp_raw and "[Job results stored in store]" not in mcp_raw:
        return mcp_raw[:_JOB_PROMPT_MAX_CHARS] + (
            "\n\n[truncated]" if len(mcp_raw) > _JOB_PROMPT_MAX_CHARS else ""
        )
    return "(none stored). Use any job info in the conversation."


def _format_jobs_for_prompt(jobs: list[dict], max_chars: int = 28000) -> str:
    """Format job list as text for LLM prompt; truncate if over max_chars."""
    lines = []
    for i, j in enumerate(jobs, 1):
        line = f"[{i}] {j.get('title') or 'N/A'} | {j.get('company') or 'N/A'}"
        if j.get("location"):
            line += f" | {j['location']}"
        if j.get("salary_min") or j.get("salary_max"):
            line += f" | {j.get('salary_min') or '?'}-{j.get('salary_max') or '?'}"
        if j.get("description_snippet"):
            line += f"\n  {j['description_snippet'][:500]}"
        if j.get("link"):
            line += f"\n  link: {j['link']}"
        lines.append(line)
    text = "\n\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[truncated for context limit]"
    return text


def _parse_tool_call(tool_call):
    """Parse tool_call (dict or object) to (tool_name, tool_args, tool_id)."""
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


def _build_mcp_system_messages(job_brief: str, messages: list, all_messages: list):
    """Build [SystemMessage] + messages + all_messages for MCP tool loop."""
    system = SystemMessage(
        content=search_jobs_agent_prompt_with_mcp + f"\n\nThe job brief is: {job_brief}"
    )
    return [system] + messages + all_messages


MCP_PLACEHOLDER = "[MCP job result stored]"


def _normalize_mcp_tool_result(tool_result) -> str:
    """
    Extract inner text from MCP tool result so parse_mcp_result_to_jobs gets valid JSON.

    MCP may return: (1) [{"type": "text", "text": "..."}] with inner JSON;
    (2) zhipin format dict with "items" / "config" / "timestamp";
    (3) string (repr or JSON). We normalize to a string the parser can handle.
    """
    if isinstance(tool_result, str):
        inner = _extract_json_from_mcp_content_parts(tool_result)
        if inner is not None:
            return inner
        return tool_result
    if isinstance(tool_result, list):
        parts = []
        for item in tool_result:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(parts)
    if isinstance(tool_result, dict):
        if "text" in tool_result:
            return str(tool_result["text"])
        # Zhipin / crawler format: {"items": [...], "config": ..., "timestamp": ...} -> pass as JSON string
        if "items" in tool_result or "config" in tool_result:
            import json
            return json.dumps(tool_result, ensure_ascii=False)
    return str(tool_result)


def _mcp_raw_and_messages_with_placeholder(
    all_messages: list, job_count: int | None = None
):
    """
    Extract raw tool content and replace ToolMessage content with short placeholder.
    If job_count is provided, placeholder includes it so conversation reflects MCP result size.
    """
    tool_parts = [
        str(m.content) for m in all_messages if isinstance(m, ToolMessage)
    ]
    mcp_raw = "\n\n---\n".join(tool_parts) if tool_parts else ""

    placeholder = MCP_PLACEHOLDER
    if job_count is not None and job_count >= 0:
        placeholder = f"[MCP job result stored: {job_count} job(s)]"

    def replace_tool(msg):
        if isinstance(msg, ToolMessage):
            return ToolMessage(
                content=placeholder,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            )
        return msg

    new_messages = [replace_tool(m) for m in all_messages]
    return mcp_raw, new_messages


# --- Nodes ---
async def _mcp_jobs_tool_call_async(
    state: JobState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
) -> dict:
    """Call MCP jobs tools to search for jobs (async). store injected by framework via sync wrapper."""
    # Get tools from MCP client using helper function
    mcp_tools = await get_mcp_tools()

    # Combine MCP tools with think tool
    tools = mcp_tools + [job_search_think_tool]
    # Create a tool map for easy lookup
    tool_map = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools)

    # Get messages from state
    messages = state.get("messages", [])
    # Truncate messages to avoid context overflow (keep ~15000 tokens)
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=128,
        start_on="human",
        end_on=("human", "tool"),
    )

    # Get job brief from state or extract from messages
    job_brief = state.get("job_brief") or ""
    if not job_brief:
        # Try to extract from messages
        for msg in reversed(messages):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                job_brief = msg.content
                break

    # Invoke model with tools
    response = await model_with_tools.ainvoke(
        _build_mcp_system_messages(job_brief, messages, [])
    )
    all_messages = [response]

    # Check if there are tool calls to execute
    max_iterations = 10  # Prevent infinite loops
    iteration = 0
    # Check if response has tool_calls attribute and it's not empty
    has_tool_calls = (
        hasattr(response, "tool_calls")
        and response.tool_calls
        and len(response.tool_calls) > 0
    )
    while has_tool_calls and iteration < max_iterations:
        iteration += 1
        # Execute all tool calls
        tool_messages = []
        for tool_call in response.tool_calls:
            tool_name, tool_args, tool_id = _parse_tool_call(tool_call)
            if not tool_name:
                continue

            if tool_name in tool_map:
                try:
                    # Execute the tool
                    tool = tool_map[tool_name]
                    # Check if tool has ainvoke method (async) or invoke method (sync)
                    if hasattr(tool, "ainvoke"):
                        tool_result = await tool.ainvoke(tool_args)
                    elif hasattr(tool, "invoke"):
                        tool_result = tool.invoke(tool_args)
                    else:
                        # Try calling directly if it's a callable
                        if asyncio.iscoroutinefunction(tool):
                            tool_result = await tool(tool_args)
                        else:
                            tool_result = tool(tool_args)

                    # Create ToolMessage with result (normalize MCP content so parser gets JSON)
                    tool_messages.append(
                        ToolMessage(
                            content=_normalize_mcp_tool_result(tool_result),
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )
                except Exception as e:
                    error_trace = traceback.format_exc()
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error executing tool {tool_name}: {str(e)}\n{error_trace}",
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )
            else:
                # Tool not found
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool {tool_name} not found. Available tools: {list(tool_map.keys())}",
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )

        # Add tool messages to the conversation
        all_messages.extend(tool_messages)

        # Invoke model again with tool results
        # Truncate all_messages if it gets too long (keep last 30 messages from tool loop)
        if len(all_messages) > 30:
            all_messages = all_messages[-30:]

        response = await model_with_tools.ainvoke(
            _build_mcp_system_messages(job_brief, messages, all_messages)
        )
        all_messages.append(response)
        # Check if there are more tool calls
        has_tool_calls = (
            hasattr(response, "tool_calls")
            and response.tool_calls
            and len(response.tool_calls) > 0
        )

    mcp_raw, _ = _mcp_raw_and_messages_with_placeholder(all_messages)
    user_id = _config_user_id(config)
    jobs = parse_mcp_result_to_jobs(mcp_raw)
    logger.info("Parsed %d job(s) from MCP result (mcp_raw len=%d)", len(jobs), len(mcp_raw or ""))
    # Placeholder with job count so ToolMessage reflects MCP result size
    _, new_messages = _mcp_raw_and_messages_with_placeholder(all_messages, job_count=len(jobs))
    if user_id and jobs and store is not None:
        await save_job_results_async(store, user_id, jobs)
    return {
        "messages": new_messages,
        "has_job_results": bool(jobs),
        "mcp_job_results": "[Job results stored in store]" if jobs else "",
        "jobs_stored_count": len(jobs),
    }


def mcp_jobs_tool_call(
    state: JobState, config: RunnableConfig, *, store: BaseStore | None = None
) -> dict:
    """Synchronous wrapper: run async implementation via asyncio.run(); store injected by framework."""
    return asyncio.run(_mcp_jobs_tool_call_async(state, config, store=store))


def supervisor(
    state: JobState, config: RunnableConfig, *, store: BaseStore | None = None
) -> dict:
    """
    Supervisor: route to job_search (classify->mcp) or optimize_recommendations.
    - If no job results (has_job_results False and store empty): go to classify_job_detail.
    - If has results: decide by last user message (new search vs optimize).
    """
    has_results = state.get("has_job_results") or bool((state.get("mcp_job_results") or "").strip())
    if not has_results:
        user_id = _config_user_id(config)
        if user_id and store and get_job_count(store, user_id) > 0:
            has_results = True
    if not has_results:
        return {"next_agent": "classify_job_detail"}

    messages = state.get("messages", [])
    last_user = _last_user_content(messages)[:800]

    # Use LLM to determine if user has a clear new search request
    structured = model.with_structured_output(SupervisorDecision)
    out = structured.invoke(
        [
            HumanMessage(
                content=supervisor_prompt.format(
                    last_user_message=last_user or "(none)"
                )
            )
        ]
    )

    # Route based on LLM decision
    next_node = (
        "classify_job_detail"
        if out.route == "job_search"
        else "optimize_recommendations"
    )
    return {"next_agent": next_node}


def optimize_recommendations(
    state: JobState, config: RunnableConfig, *, store: BaseStore | None = None
) -> dict:
    """
    Optimize agent: answer user questions and optimize the display of job results.
    Uses semantic search (RAG) or full list fallback; injects job text into prompt.
    """
    messages = _truncate_messages_list(
        state.get("messages", []), max_tokens=20000
    )
    job_brief = state.get("job_brief") or ""
    user_id = _config_user_id(config)
    job_text = _get_optimize_job_text(state, store, user_id, messages)

    system = optimize_recommendations_prompt
    if job_brief:
        system += f"\n\nJob brief for context: {job_brief}"
    system += "\n\n---\nJob results (use as source for job listings):\n" + job_text

    response = model.invoke([SystemMessage(content=system)] + messages)
    return {"messages": [response]}


# --- Store / embedding ---
def _get_store_index_config():
    """
    Return PostgresStore index config for job vector search.
    Uses EMBED_BASE_URL (port 9001); chat uses CHAT_BASE_URL (port 9000).
    """
    from langchain_openai import OpenAIEmbeddings

    embed = OpenAIEmbeddings(
        model=os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        base_url=EMBED_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
    )
    return {
        "dims": STORE_EMBEDDING_DIMS,
        "embed": embed,
        "fields": ["title", "description_snippet", "company", "location"],
    }


# --- Graph ---

def build_agent(store: BaseStore, checkpointer: PostgresSaver) -> CompiledStateGraph:
    """Build and compile the graph. Caller owns store/checkpointer (e.g. use `with`)."""
    builder = StateGraph(JobState)
    builder.add_node("classify_job_detail", classify_job_detail)
    builder.add_node("mcp_jobs_tool_call", mcp_jobs_tool_call)
    builder.add_node("supervisor", supervisor)
    builder.add_node("optimize_recommendations", optimize_recommendations)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda s: s.get("next_agent") or "classify_job_detail",
        {"classify_job_detail": "classify_job_detail", "optimize_recommendations": "optimize_recommendations"},
    )
    builder.add_edge("mcp_jobs_tool_call", END)
    builder.add_edge("optimize_recommendations", END)
    return builder.compile(checkpointer=checkpointer, store=store)


_agent_instance: CompiledStateGraph | None = None
_store_ref = None
_checkpointer_ref = None


def agent(config: RunnableConfig | None = None) -> CompiledStateGraph:
    """Return cached graph for langgraph dev / tests. Creates store/checkpointer on first call (lazy init)."""
    global _agent_instance, _store_ref, _checkpointer_ref
    if _agent_instance is not None:
        return _agent_instance
    _store_ref = PostgresStore.from_conn_string(DB_URI, index=_get_store_index_config())
    store = _store_ref.__enter__()
    _checkpointer_ref = PostgresSaver.from_conn_string(DB_URI)
    checkpointer = _checkpointer_ref.__enter__()
    store.setup()
    checkpointer.setup()
    _agent_instance = build_agent(store, checkpointer)
    return _agent_instance


def run_demo() -> None:
    """Run the three-step demo inside a `with` block (reference pattern)."""
    store_index = _get_store_index_config()
    with (
        PostgresStore.from_conn_string(DB_URI, index=store_index) as store,
        PostgresSaver.from_conn_string(DB_URI) as checkpointer,
    ):
        store.setup()
        checkpointer.setup()
        graph = build_agent(store, checkpointer)
        config: RunnableConfig = {
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
            config=config,
        )
        logger.info("First invoke done. Result keys: %s", list(result.keys()))
        result = graph.invoke(
            {
                "messages": result.get("messages", [])
                + [HumanMessage(content="我希望寻找北京的agent 开发相关工作, 其他条件不限")]
            },
            config=config,
        )
        logger.info("Second invoke done. Result keys: %s", list(result.keys()))
        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content="按薪资从高到低重新排一下，只保留25k以上的")
                ]
            },
            config=config,
        )
        logger.info("Third invoke done. Result keys: %s", list(result.keys()))
