"""
Job sub-agent: classify intent, MCP job search, optimize/filter results.
Self-contained: config, nodes, helpers, graph. Exports build_job_agent(store, checkpointer).
When used as subgraph, parent's checkpointer persists state; when standalone, pass checkpointer.
"""

import asyncio
import json
import logging
import traceback
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

from uni_agent.config import (
    CHAT_BASE_URL,
    JOB_PROMPT_MAX_CHARS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SAFE_MESSAGE_TOKENS,
)
from uni_agent.prompts import (
    clarify_job_detail_prompt,
    optimize_recommendations_prompt,
    optimize_retrieval_prompt,
    search_jobs_agent_prompt_with_mcp,
)
from uni_agent.state import (
    ClarifyJobDetail,
    JobState,
    OptimizeRetrievalParams,
)
from uni_agent.store.job_store import (
    _extract_json_from_mcp_content_parts,
    parse_mcp_result_to_jobs,
)
from uni_agent.store.store_adapter import (
    get_jobs_by_search,
    get_jobs_for_prompt,
    save_job_results_async,
)
from uni_agent.tools.mcp_jobs import get_mcp_tools
from uni_agent.tools.think_tool import job_search_think_tool
from uni_agent.utils import (
    config_user_id,
    has_tool_calls,
    last_user_content,
    parse_tool_call,
    truncate_messages_list,
)

logger = logging.getLogger(__name__)

# ===== CONFIGURATION =====
_model = ChatOpenAI(
    model=OPENAI_MODEL,
    base_url=CHAT_BASE_URL,
    api_key=OPENAI_API_KEY,
)
_clarify_parser = PydanticOutputParser(pydantic_object=ClarifyJobDetail)
_retrieval_parser = PydanticOutputParser(pydantic_object=OptimizeRetrievalParams)


# ===== HELPERS: messages / prompt =====
def _truncate_messages_for_prompt(msgs, max_per_msg=600, max_msgs=14):
    """Build a short summary of recent messages for context limit."""
    out = []
    for m in list(msgs)[-max_msgs:]:
        if isinstance(m, ToolMessage):
            out.append("[ToolMessage: MCP result stored in mcp_job_results]")
        else:
            raw = str(getattr(m, "content", "") or "")
            s = raw[:max_per_msg] + ("..." if len(raw) > max_per_msg else "")
            out.append(f"{type(m).__name__}: {s}")
    return "\n---\n".join(out)


def _build_mcp_system_messages(job_brief: str, messages: list, all_messages: list):
    """Build [SystemMessage] + messages + all_messages for MCP tool loop."""
    system = SystemMessage(
        content=search_jobs_agent_prompt_with_mcp + f"\n\nThe job brief is: {job_brief}"
    )
    return [system] + messages + all_messages


MCP_PLACEHOLDER = "[MCP job result stored]"


def _normalize_mcp_tool_result(tool_result) -> str:
    """Extract inner text from MCP tool result so parser gets valid JSON."""
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
        if "items" in tool_result or "config" in tool_result:
            return json.dumps(tool_result, ensure_ascii=False)
    return str(tool_result)


def _mcp_raw_and_messages_with_placeholder(
    all_messages: list, job_count: int | None = None
):
    """Extract raw tool content and replace ToolMessage content with short placeholder."""
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


# ===== HELPERS: optimize / RAG =====
def _get_retrieval_params_agentic(
    last_user: str, job_brief: str, config: RunnableConfig | None = None
) -> OptimizeRetrievalParams:
    """Use LLM to parse user message into retrieval params."""
    prompt_content = optimize_retrieval_prompt.format(
        last_user_message=last_user or "(none)",
        job_brief=job_brief or "(none)",
        format_instructions=_retrieval_parser.get_format_instructions(),
    )
    output = _model.invoke(
        [HumanMessage(content=prompt_content)], config=config or {}
    )
    try:
        return _retrieval_parser.invoke(output)
    except Exception:  # pylint: disable=broad-except
        return OptimizeRetrievalParams()


def _job_salary_sort_key(j: dict):
    """Sort key for jobs by salary (prefer max, then min)."""
    smax, smin = j.get("salary_max"), j.get("salary_min")
    if smax is not None:
        return (0, -(smax if isinstance(smax, (int, float)) else 0))
    if smin is not None:
        return (1, -(smin if isinstance(smin, (int, float)) else 0))
    return (2, 0)


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


def _get_optimize_job_text(
    state, store, user_id: str, messages: list, config: RunnableConfig | None = None
) -> str:
    """Build job text for optimize_recommendations using agentic RAG."""
    last_user = last_user_content(messages)
    job_brief = state.get("job_brief") or ""
    params = _get_retrieval_params_agentic(last_user or "", job_brief, config=config)
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
        jobs = sorted(jobs, key=_job_salary_sort_key)
    if jobs:
        return _format_jobs_for_prompt(jobs, max_chars=JOB_PROMPT_MAX_CHARS)
    mcp_raw = state.get("mcp_job_results") or ""
    if mcp_raw and "[Job results stored in store]" not in mcp_raw:
        return mcp_raw[:JOB_PROMPT_MAX_CHARS] + (
            "\n\n[truncated]" if len(mcp_raw) > JOB_PROMPT_MAX_CHARS else ""
        )
    return "(none stored). Use any job info in the conversation."


# ===== NODES =====
def classify_job_detail(
    state: JobState, config: RunnableConfig, *, store: BaseStore | None = None
) -> dict:
    """Classify job detail and set classify_goto for routing."""
    messages = state.get("messages", [])
    truncated = _truncate_messages_for_prompt(messages)
    prompt_content = clarify_job_detail_prompt.format(
        messages=truncated,
        format_instructions=_clarify_parser.get_format_instructions(),
    )
    output = _model.invoke([HumanMessage(content=prompt_content)], config=config)
    response = _clarify_parser.invoke(output)
    if response.need_clarify:
        return {
            "messages": [AIMessage(content=response.question)],
            "classify_goto": "__end__",
        }
    return {
        "messages": [AIMessage(content=response.verification)],
        "classify_goto": "mcp_jobs_tool_call",
    }


async def _mcp_jobs_tool_call_async(
    state: JobState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
) -> dict:
    """Call MCP jobs tools (async). Store injected by framework."""
    mcp_tools = await get_mcp_tools()
    tools = mcp_tools + [job_search_think_tool]
    tool_map = {t.name: t for t in tools}
    model_with_tools = _model.bind_tools(tools)

    messages = state.get("messages", [])
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=128,
        start_on="human",
        end_on=("human", "tool"),
    )
    job_brief = state.get("job_brief") or ""
    if not job_brief:
        for msg in reversed(messages):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                job_brief = msg.content
                break

    response = await model_with_tools.ainvoke(
        _build_mcp_system_messages(job_brief, messages, []), config=config
    )
    all_messages = [response]
    max_iterations = 10
    iteration = 0
    while has_tool_calls(response) and iteration < max_iterations:
        iteration += 1
        tool_messages = []
        for tool_call in response.tool_calls:
            tool_name, tool_args, tool_id = parse_tool_call(tool_call)
            if not tool_name:
                continue
            if tool_name in tool_map:
                try:
                    tool = tool_map[tool_name]
                    if hasattr(tool, "ainvoke"):
                        tool_result = await tool.ainvoke(tool_args)
                    elif hasattr(tool, "invoke"):
                        tool_result = tool.invoke(tool_args)
                    else:
                        if asyncio.iscoroutinefunction(tool):
                            tool_result = await tool(tool_args)
                        else:
                            tool_result = tool(tool_args)
                    tool_messages.append(
                        ToolMessage(
                            content=_normalize_mcp_tool_result(tool_result),
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )
                except Exception as e:  # pylint: disable=broad-except
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error executing tool {tool_name}: {str(e)}\n{traceback.format_exc()}",
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )
            else:
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool {tool_name} not found. Available tools: {list(tool_map.keys())}",
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )
        all_messages.extend(tool_messages)
        if len(all_messages) > 30:
            all_messages = all_messages[-30:]
        response = await model_with_tools.ainvoke(
            _build_mcp_system_messages(job_brief, messages, all_messages),
            config=config,
        )
        all_messages.append(response)

    mcp_raw, _ = _mcp_raw_and_messages_with_placeholder(all_messages)
    user_id = config_user_id(config)
    jobs = parse_mcp_result_to_jobs(mcp_raw)
    logger.info(
        "Parsed %d job(s) from MCP result (mcp_raw len=%d)",
        len(jobs), len(mcp_raw or ""),
    )
    _, new_messages = _mcp_raw_and_messages_with_placeholder(
        all_messages, job_count=len(jobs)
    )
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
    """Synchronous wrapper for MCP jobs tool call."""
    return asyncio.run(_mcp_jobs_tool_call_async(state, config, store=store))


def optimize_recommendations(
    state: JobState, config: RunnableConfig, *, store: BaseStore | None = None
) -> dict:
    """Optimize/filter job results and answer user questions using RAG."""
    messages = truncate_messages_list(
        state.get("messages", []), max_tokens=SAFE_MESSAGE_TOKENS
    )
    job_brief = state.get("job_brief") or ""
    user_id = config_user_id(config)
    job_text = _get_optimize_job_text(state, store, user_id, messages, config=config)
    system = optimize_recommendations_prompt
    if job_brief:
        system += f"\n\nJob brief for context: {job_brief}"
    system += "\n\n---\nJob results (use as source for job listings):\n" + job_text
    response = _model.invoke(
        [SystemMessage(content=system)] + messages, config=config
    )
    return {"messages": [response]}


# ===== GRAPH =====
def build_job_agent(
    store: BaseStore | None = None,
    checkpointer: "BaseCheckpointSaver | None" = None,
) -> CompiledStateGraph:
    """
    Build and compile the job subgraph: entry -> classify | optimize;
    classify -> mcp_jobs_tool_call | END (when need_clarify, END so user can reply in next turn;
    supervisor will route the reply back to job_agent).
    """
    builder = StateGraph(JobState)
    builder.add_node("classify_job_detail", classify_job_detail)
    builder.add_node("mcp_jobs_tool_call", mcp_jobs_tool_call)
    builder.add_node("optimize_recommendations", optimize_recommendations)

    def _entry(state: JobState) -> str:
        next_agent = state.get("next_agent") or "classify_job_detail"
        if next_agent == "optimize_recommendations":
            return "optimize_recommendations"
        return "classify_job_detail"

    builder.add_conditional_edges(
        START,
        _entry,
        {
            "classify_job_detail": "classify_job_detail",
            "optimize_recommendations": "optimize_recommendations",
        },
    )
    builder.add_conditional_edges(
        "classify_job_detail",
        lambda s: s.get("classify_goto") or "__end__",
        {"mcp_jobs_tool_call": "mcp_jobs_tool_call", "__end__": END},
    )
    builder.add_edge("mcp_jobs_tool_call", END)
    builder.add_edge("optimize_recommendations", END)
    return builder.compile(store=store, checkpointer=checkpointer)


# Default no-store graph for backward compat (e.g. tests)
job_agent: CompiledStateGraph = build_job_agent(store=None)
