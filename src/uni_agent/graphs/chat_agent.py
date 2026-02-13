"""
Chat sub-agent: general Q&A, greetings, off-topic; can use tools.
Self-contained: config, nodes, graph construction. Exports compiled graph.
"""

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_openai import ChatOpenAI

from uni_agent.config import (
    CHAT_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SAFE_MESSAGE_TOKENS,
    TOOL_RESULT_MAX_CHARS,
)
from uni_agent.prompts import llm_call_prompt
from uni_agent.state import JobState
from uni_agent.tools.datetime_tool import get_current_datetime
from uni_agent.tools.tavily_search import tavily_search
from uni_agent.tools.think_tool import job_search_think_tool
from uni_agent.utils import (
    has_tool_calls,
    parse_tool_call,
    tool_result_to_content,
    truncate_messages_list,
)

# ===== CONFIGURATION =====
_tools = [tavily_search, job_search_think_tool, get_current_datetime]
_model = ChatOpenAI(
    model=OPENAI_MODEL,
    base_url=CHAT_BASE_URL,
    api_key=OPENAI_API_KEY,
)
_model_with_tools = _model.bind_tools(_tools)
_tool_map = {t.name: t for t in _tools}


# ===== NODES =====
def _llm_call(
    state: JobState, config: RunnableConfig, *, store=None  # noqa: ARG001
) -> dict:
    """LLM with tools; may return tool_calls."""
    messages = truncate_messages_list(
        state.get("messages", []), max_tokens=SAFE_MESSAGE_TOKENS
    )
    response = _model_with_tools.invoke(
        [SystemMessage(content=llm_call_prompt)] + messages, config=config
    )
    return {"messages": [response]}


def _tool_node(
    state: JobState, config: RunnableConfig, *, store=None  # noqa: ARG001
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


def _route_after_llm(state: JobState) -> str:
    """Route to tool_node if last message has tool_calls, else END."""
    messages = state.get("messages", [])
    if not messages or not has_tool_calls(messages[-1]):
        return "__end__"
    return "tool_node"


# ===== GRAPH CONSTRUCTION =====
_builder = StateGraph(JobState)
_builder.add_node("llm_call", _llm_call)
_builder.add_node("tool_node", _tool_node)
_builder.add_edge(START, "llm_call")
_builder.add_conditional_edges(
    "llm_call",
    _route_after_llm,
    {"tool_node": "tool_node", "__end__": END},
)
_builder.add_edge("tool_node", "llm_call")

chat_agent: CompiledStateGraph = _builder.compile()
