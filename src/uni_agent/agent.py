import asyncio
import os
import signal
import sys
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from uni_agent.prompts import (
    clarify_job_detail_prompt,
    search_jobs_agent_prompt_with_mcp,
)
from uni_agent.state import ClarifyJobDetail, JobState
from uni_agent.tools.mcp_jobs import get_mcp_tools
from uni_agent.tools.tavily_search import tavily_search
from uni_agent.tools.think_tool import job_search_think_tool

tools = [tavily_search, job_search_think_tool]
model = ChatOpenAI(
    model="Qwen/Qwen3-8B", base_url="http://192.168.0.201:9000/v1", api_key="dummy"
)
llm_with_tools = model.bind_tools(tools)


def classify_job_detail(state: JobState) -> Command:
    """Classify job detail and determine if clarification is needed."""
    messages = state.get("messages")
    structured_output_model = model.with_structured_output(ClarifyJobDetail)
    response = structured_output_model.invoke(
        [
            HumanMessage(
                content=clarify_job_detail_prompt.format(messages=messages)
            )
        ]
    )
    # Store the response in state for conditional routing
    if response.need_clarify:
        return Command(
            goto = END,
            update = {"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto = "mcp_jobs_tool_call",
            update = {"messages": [AIMessage(content=response.verification)]}
        )


async def mcp_jobs_tool_call(state: JobState):
    """Call MCP jobs tools to search for jobs."""
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
        
        # Get job brief from state or extract from messages
        job_brief = state.get("job_brief") or ""
        if not job_brief:
            # Try to extract from messages
            for msg in reversed(messages):
                if hasattr(msg, 'content') and isinstance(msg.content, str):
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
        has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls and len(response.tool_calls) > 0
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
                                name=tool_name
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
                                name=tool_name
                            )
                        )
                else:
                    # Tool not found
                    tool_messages.append(
                        ToolMessage(
                            content=f"Tool {tool_name} not found. Available tools: {list(tool_map.keys())}",
                            tool_call_id=tool_id,
                            name=tool_name
                        )
                    )
            
            # Add tool messages to the conversation
            all_messages.extend(tool_messages)
            
            # Invoke model again with tool results
            messages_with_system = [
                SystemMessage(
                    content=search_jobs_agent_prompt_with_mcp
                    + f"\n\nThe job brief is: {job_brief}"
                )
            ] + messages + all_messages
            response = await model_with_tools.ainvoke(messages_with_system)
            all_messages.append(response)
            # Check if there are more tool calls
            has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls and len(response.tool_calls) > 0
        
        return {"messages": all_messages}
    except Exception as e:
        # Return error message if something goes wrong
        import traceback
        error_msg = f"Error in job search: {str(e)}\n{traceback.format_exc()}"
        return {"messages": [AIMessage(content=error_msg)]}


agent_builder = StateGraph(JobState)
agent_builder.add_node("classify_job_detail", classify_job_detail)
agent_builder.add_node("mcp_jobs_tool_call", mcp_jobs_tool_call)
agent_builder.add_edge(START, "classify_job_detail")
agent_builder.add_edge("mcp_jobs_tool_call", END)

agent = agent_builder.compile()

TIMEOUT_SECONDS = 120  # Increased timeout for tool execution and result processing


def timeout_handler(signum, frame):
    """Handle timeout signal."""
    print(f"\nError: Program execution exceeded {TIMEOUT_SECONDS} seconds timeout. Exiting...", file=sys.stderr)
    os._exit(1)


async def main_async():
    """Async main function."""
    try:
        # First invocation with initial user message
        result = await agent.ainvoke({"messages": [HumanMessage(content="我是一名有3年经验的软件工程师，我想找一份工作，我擅长golang和python, 请帮我搜索一下agent开发相关职位")]})
        print(result)
        
        # Add user's follow-up message after AI response
        # Use the previous result as base state and add new message
        result = await agent.ainvoke({
            "messages": result.get("messages", []) + [HumanMessage(content="我希望寻找北京的agent 开发相关工作, 其他条件不限")],
            "job_brief": result.get("job_brief"),
            "supervisor_messages": result.get("supervisor_messages", [])
        })
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main function with 30 second timeout."""
    # Set up signal handler for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    
    try:
        asyncio.run(main_async())
        # Cancel the alarm if we complete successfully
        signal.alarm(0)
    except KeyboardInterrupt:
        print("\nError: Program interrupted by user. Exiting...", file=sys.stderr)
        signal.alarm(0)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        signal.alarm(0)
        sys.exit(1)

async def test_mcp_jobs_tool_call():
    """Test function for mcp_jobs_tool_call to ensure it works correctly."""
    print("Testing mcp_jobs_tool_call...")
    try:
        # Create a test state using dict (JobState is a TypedDict)
        test_state = {
            "messages": [HumanMessage(content="I'm looking for a software engineer job")],
            "job_brief": "Software engineer position",
            "supervisor_messages": []
        }
        
        # Call the function
        result = await mcp_jobs_tool_call(test_state)
        
        print("✓ mcp_jobs_tool_call completed successfully")
        print(f"Result type: {type(result)}")
        if "messages" in result:
            print(f"Messages count: {len(result['messages'])}")
            if result['messages']:
                print(f"First message type: {type(result['messages'][0])}")
        return True
    except Exception as e:
        print(f"✗ Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import asyncio
    
    # Run test first
    # print("Running test...")
    # test_result = asyncio.run(test_mcp_jobs_tool_call())
    
    # if test_result:
    #     print("\nTest passed! Running main...")
    #     main()
    # else:
    #     print("\nTest failed! Please check the error above.")
    #     sys.exit(1)
    main()