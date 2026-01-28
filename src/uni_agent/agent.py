import asyncio
from datetime import datetime
import logging
import os
import signal
import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import trim_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from uni_agent.prompts import (
    clarify_job_detail_prompt,
    optimize_recommendations_prompt,
    search_jobs_agent_prompt_with_mcp,
    supervisor_prompt,
)
from uni_agent.state import ClarifyJobDetail, JobState, SupervisorDecision
from uni_agent.tools.mcp_jobs import get_mcp_tools
from uni_agent.tools.tavily_search import tavily_search
from uni_agent.tools.think_tool import job_search_think_tool

# Configure logger to include full path and line number, and write to logs/agent_YYYYMMDD_HHMMSS.log
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

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

tools = [tavily_search, job_search_think_tool]
model = ChatOpenAI(
    model="Qwen/Qwen3-8B", base_url="http://192.168.0.201:9000/v1", api_key="dummy"
)
llm_with_tools = model.bind_tools(tools)


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
    
    try:
        # Use trim_messages with token-based trimming
        # strategy='last': keep most recent messages
        # include_system=True: preserve SystemMessage if present
        # start_on='human': ensure valid chat history starts with HumanMessage
        trimmed = trim_messages(
            msgs,
            max_tokens=max_tokens,
            token_counter="approximate",  # Use approximate token counting
            strategy="last",
            include_system=True,
            start_on="human",
        )
        return trimmed
    except Exception:
        # Fallback: if trim_messages fails, use simple truncation
        # Keep last 20 messages, preserving SystemMessage
        if len(msgs) <= 20:
            return msgs
        system_msg = None
        if isinstance(msgs[0], SystemMessage):
            system_msg = msgs[0]
            msgs = msgs[1:]
        truncated = msgs[-19:] if len(msgs) > 19 else msgs
        return [system_msg] + truncated if system_msg else truncated


def classify_job_detail(state: JobState) -> Command:
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
        return Command(
            goto="mcp_jobs_tool_call",
            update={"messages": [AIMessage(content=response.verification)]},
        )


async def _mcp_jobs_tool_call_async(state: JobState) -> JobState:
    """Call MCP jobs tools to search for jobs (async implementation)."""
    try:
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

        # Prepare messages with system message
        messages_with_system = [
            SystemMessage(
                content=search_jobs_agent_prompt_with_mcp
                + f"\n\nThe job brief is: {job_brief}"
            )
        ] + messages

        # Invoke model with tools
        response = await model_with_tools.ainvoke(messages_with_system)
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
                # Handle both dict and object formats
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id")
                else:
                    # Assume it's an object with attributes
                    tool_name = getattr(tool_call, "name", None)
                    tool_args = getattr(tool_call, "args", {})
                    tool_id = getattr(tool_call, "id", None)

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

                        # Create ToolMessage with result
                        tool_messages.append(
                            ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_id,
                                name=tool_name,
                            )
                        )
                    except Exception as e:
                        # If tool execution fails, add error message
                        import traceback

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
            
            messages_with_system = (
                [
                    SystemMessage(
                        content=search_jobs_agent_prompt_with_mcp
                        + f"\n\nThe job brief is: {job_brief}"
                    )
                ]
                + messages
                + all_messages
            )
            response = await model_with_tools.ainvoke(messages_with_system)
            all_messages.append(response)
            # Check if there are more tool calls
            has_tool_calls = (
                hasattr(response, "tool_calls")
                and response.tool_calls
                and len(response.tool_calls) > 0
            )

        # Save raw MCP content to state; replace ToolMessage content with short placeholder so it is not in general context
        tool_parts = [str(m.content) for m in all_messages if isinstance(m, ToolMessage)]
        mcp_raw = "\n\n---\n".join(tool_parts) if tool_parts else ""
        _MCP_PLACEHOLDER = "[MCP job result stored]"

        def _replace_tool(msg):
            if isinstance(msg, ToolMessage):
                return ToolMessage(
                    content=_MCP_PLACEHOLDER,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
            return msg

        new_messages = [_replace_tool(m) for m in all_messages]
        return {"messages": new_messages, "mcp_job_results": mcp_raw}
    except Exception as e:
        # Return error message if something goes wrong
        import traceback

        error_msg = f"Error in job search: {str(e)}\n{traceback.format_exc()}"
        return {"messages": [AIMessage(content=error_msg)]}


def mcp_jobs_tool_call(state: JobState) -> JobState:
    """Synchronous wrapper: run async implementation via asyncio.run() for use with agent.invoke()."""
    return asyncio.run(_mcp_jobs_tool_call_async(state))


def supervisor(state: JobState) -> dict:
    """
    Supervisor: route to job_search (classify->mcp) or optimize_recommendations.
    - If mcp_job_results is empty: go to classify_job_detail to start a new search.
    - If mcp_job_results is not empty: decide by last user message:
      - If user has a clear new search request: go to classify_job_detail (new search).
      - Otherwise: go to optimize_recommendations (refine existing results).
    """
    mcp_job_results = state.get("mcp_job_results") or ""
    
    # If mcp_job_results is empty, must go to classify_job_detail to start search
    if not mcp_job_results.strip():
        return {"next_agent": "classify_job_detail"}
    
    # If mcp_job_results exists, check if user has a clear new search request
    messages = state.get("messages", [])
    last_user = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user = str(getattr(m, "content", "") or "")[:800]
            break
    
    # Use LLM to determine if user has a clear new search request
    structured = model.with_structured_output(SupervisorDecision)
    out = structured.invoke(
        [
            HumanMessage(
                content=supervisor_prompt.format(last_user_message=last_user or "(none)")
            )
        ]
    )
    
    # Route based on LLM decision
    next_node = (
        "classify_job_detail" if out.route == "job_search" else "optimize_recommendations"
    )
    return {"next_agent": next_node}


# Max chars of mcp_job_results to put into LLM context; avoid exceeding model context
_MCP_RAW_MAX_CHARS = 28_000


def optimize_recommendations(state: JobState) -> dict:
    """
    Optimize agent: answer user questions and optimize the display of job results.
    Uses mcp_job_results from state (raw MCP data) only here; does NOT call MCP or any tools.
    General context keeps ToolMessage as placeholders; full raw is injected only when needed.
    """
    try:
        messages = state.get("messages", [])
        # Truncate messages to avoid context overflow (keep ~20000 tokens for optimize)
        messages = _truncate_messages_list(messages, max_tokens=20000)
        
        job_brief = state.get("job_brief") or ""
        mcp_raw = state.get("mcp_job_results") or ""
        if len(mcp_raw) > _MCP_RAW_MAX_CHARS:
            mcp_raw = mcp_raw[:_MCP_RAW_MAX_CHARS] + "\n\n[truncated for context limit]"

        system = (
            optimize_recommendations_prompt
            + (f"\n\nJob brief for context: {job_brief}" if job_brief else "")
        )
        if mcp_raw:
            system += "\n\n---\nRaw MCP job results (use as source for job listings):\n" + mcp_raw
        else:
            system += "\n\n---\nRaw MCP job results: (none stored). Use any job info in the conversation."

        response = model.invoke([SystemMessage(content=system)] + messages)
        return {"messages": [response]}
    except Exception as e:
        import traceback
        return {"messages": [AIMessage(content=f"Error: {e}\n{traceback.format_exc()}")]}


DB_URI = "postgresql://postgres:password@192.168.0.201:15433/postgres?sslmode=disable"

TIMEOUT_SECONDS = 120  # Increased timeout for tool execution and result processing


def _build_agent(checkpointer):
    """Build and compile the multi-agent graph: supervisor -> [job_search | optimize]."""
    builder = StateGraph(JobState)
    # Sub-agents (unchanged structure for job_search: classify -> mcp_jobs)
    builder.add_node("classify_job_detail", classify_job_detail)
    builder.add_node("mcp_jobs_tool_call", mcp_jobs_tool_call)
    builder.add_node("supervisor", supervisor)
    builder.add_node("optimize_recommendations", optimize_recommendations)
    # Multi-agent routing: START -> supervisor -> classify | optimize
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda s: s.get("next_agent") or "classify_job_detail",
        {"classify_job_detail": "classify_job_detail", "optimize_recommendations": "optimize_recommendations"},
    )
    builder.add_edge("mcp_jobs_tool_call", END)
    builder.add_edge("optimize_recommendations", END)
    return builder.compile(checkpointer=checkpointer)


def timeout_handler(signum, frame):
    """Handle timeout signal."""
    logger.error(
        "Program execution exceeded %s seconds timeout. Exiting...",
        TIMEOUT_SECONDS,
    )
    os._exit(1)


def run(agent):
    """Run agent: PostgresSaver only implements sync get_tuple, so use invoke not ainvoke."""
    config = {"configurable": {"thread_id": "1"}}
    logger.info("Starting first agent invocation")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="我是一名有3年经验的软件工程师，我想找一份工作，我擅长golang和python, 请帮我搜索一下agent开发相关职位"
                )
            ]
        },
        config=config,
    )
    logger.info("First agent invocation finished. Result: %s", result)

    result = agent.invoke(
        {
            "messages": result.get("messages", [])
            + [
                HumanMessage(content="我希望寻找北京的agent 开发相关工作, 其他条件不限")
            ],
            "job_brief": result.get("job_brief"),
            "supervisor_messages": result.get("supervisor_messages", []),
        },
        config=config,
    )
    logger.info("Second agent invocation finished. Result: %s", result)

    # Third invoke: after job results, LLM dialogue to optimize recommendations (supervisor -> optimize)
    result = agent.invoke(
        {
            "messages": result.get("messages", [])
            + [HumanMessage(content="按薪资从高到低重新排一下，只保留25k以上的")],
            "job_brief": result.get("job_brief"),
            "supervisor_messages": result.get("supervisor_messages", []),
        },
        config=config,
    )
    logger.info("Third agent invocation finished. Result: %s", result)


def main():
    """Main function with 30 second timeout."""
    logger.info("Agent main starting")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        agent = _build_agent(checkpointer)
        run(agent)
    signal.alarm(0)


if __name__ == "__main__":
    main()
